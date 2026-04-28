"""vLLM-backed model wrapper that exposes the subset of HookedRootModule the
activations store needs.

vLLM is a much faster inference engine than HF transformers for activation
extraction (the vllm-lens project reports ~8x over HF for batched extraction),
which is the dominant cost during SAE training when the LLM and SAE live on
different GPUs and the LLM is large.

This module is a thin shim that lets a user pass ``model_class_name="VLLMLens"``
to the runner and have activations come out of vLLM via the
`vllm-lens <https://github.com/UKGovernmentBEIS/vllm-lens>`_ plugin.

Limitations:
    Only the activation-extraction path (``run_with_cache``) is supported.
    Eval components that need ``run_with_hooks`` (CE-loss / KL with SAE
    replacement) are not — vLLM's engine doesn't expose that interception
    point. The runner detects this via ``supports_forward = False`` on the
    proxy and skips those eval components.
"""

import os
from types import SimpleNamespace
from typing import Any

import torch
from transformer_lens.hook_points import HookedRootModule


def _layer_from_hook_name(hook_name: str) -> int:
    """Pull the integer layer index out of a TransformerLens hook name.

    vLLM's activation-extraction API takes integer layer indices and only
    exposes the post-residual full-layer output, so we restrict to
    ``blocks.{i}.hook_resid_post`` to make the contract unambiguous.
    """
    parts = hook_name.split(".")
    if len(parts) == 3 and parts[0] == "blocks" and parts[2] == "hook_resid_post":
        try:
            return int(parts[1])
        except ValueError:
            pass
    raise ValueError(
        f"VLLMLens only supports TransformerLens-style residual-stream hooks "
        f"of the form 'blocks.{{i}}.hook_resid_post'; got hook_name={hook_name!r}. "
        "vllm-lens captures the post-residual full-layer output, which is what "
        "hook_resid_post represents."
    )


class VLLMLensProxy(HookedRootModule):
    """vLLM-backed proxy exposing ``run_with_cache`` for SAELens.

    Forward, ``run_with_hooks``, and loss-style calls are intentionally
    unimplemented: vLLM's engine produces activations via a request-output
    metadata channel rather than as a ``nn.Module.forward`` we can hook into,
    so the replacement-hook eval path can't be supported. ``supports_forward``
    is set to ``False`` so the runner can skip CE-loss / KL evals.
    """

    supports_forward: bool = False

    def __init__(
        self,
        model_name: str,
        target_device: str | torch.device = "cpu",
        vllm_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        # vLLM's EngineCore is a subprocess that re-initializes CUDA. With the
        # default fork start-method, that fails if the parent has already
        # touched CUDA (common in notebooks / when SAELens runs alongside
        # torch CUDA ops). Force spawn so the child gets a fresh CUDA state.
        # Must be set before `import vllm`.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Lazy import: vLLM is a heavy optional dep we don't want loaded on
        # the default HF / HookedTransformer paths.
        from vllm import LLM  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        vllm_kwargs = dict(vllm_kwargs or {})
        vllm_kwargs.setdefault("dtype", "bfloat16")

        self._llm = LLM(model=model_name, **vllm_kwargs)
        self._target_device = torch.device(target_device)
        self._model_name = model_name
        self._tokenizer = self._llm.get_tokenizer()
        # _get_input_token_device falls back to model.cfg.device when the
        # model has no W_E / get_input_embeddings. vLLM accepts tokens as
        # python lists, so cpu is the right answer.
        self.cfg = SimpleNamespace(device=torch.device("cpu"))
        self.setup()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_name(self) -> str:
        return self._model_name

    def to(self, *_: Any, **__: Any) -> "VLLMLensProxy":
        # vLLM manages its own device placement. Make ``.to(device)`` a no-op
        # so SAELens's blanket ``self.model.to(device)`` calls don't error.
        return self

    def parameters(self, recurse: bool = True):  # noqa: ARG002
        # No torch parameters live on this wrapper; weights live inside vLLM's
        # engine. Return an empty iterator so consumers walking parameters
        # don't trip.
        return iter([])

    @torch.no_grad()
    def run_with_cache(
        self,
        tokens: torch.Tensor,
        names_filter: list[str] | None = None,
        stop_at_layer: int | None = None,  # noqa: ARG002 — vLLM does the full forward
        prepend_bos: bool = False,  # noqa: ARG002 — caller manages BOS
        **_: Any,
    ) -> tuple[None, dict[str, torch.Tensor]]:
        if names_filter is None or len(names_filter) != 1:
            raise NotImplementedError(
                "VLLMLens supports extracting one hook at a time; pass exactly "
                "one hook name in `names_filter`."
            )
        from vllm import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            SamplingParams,
        )

        hook_name = names_filter[0]
        layer = _layer_from_hook_name(hook_name)

        if tokens.dim() != 2:
            raise ValueError(
                f"Expected tokens of shape (batch, ctx); got {tuple(tokens.shape)}"
            )
        # vLLM's offline LLM API takes prompt_token_ids as list[list[int]].
        prompt_token_ids = tokens.to(torch.int64).tolist()

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            extra_args={"output_residual_stream": [layer]},
        )
        outputs = self._llm.generate(
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        per_prompt = []
        for out in outputs:
            acts = getattr(out, "activations", None)
            if acts is None or "residual_stream" not in acts:
                raise RuntimeError(
                    "vLLM RequestOutput has no `activations` field. Check "
                    "that vllm-lens is installed and its plugin is active."
                )
            stream = acts["residual_stream"]
            # vllm-lens packs activations as (num_layers_extracted, seq_len,
            # d_model). We requested one layer, so collapse that axis to get
            # (seq_len, d_model).
            if stream.dim() == 3:
                if stream.shape[0] != 1:
                    raise RuntimeError(
                        "Expected one extracted layer per request, got "
                        f"residual_stream of shape {tuple(stream.shape)}"
                    )
                stream = stream.squeeze(0)
            per_prompt.append(stream)

        stacked = torch.stack(per_prompt, dim=0).to(self._target_device)
        return None, {hook_name: stacked}

    def forward(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError(
            "VLLMLens does not support direct forward / loss computation. "
            "The runner will skip CE-loss / KL eval automatically; sparsity "
            "and variance evals still work because they use run_with_cache."
        )

    def run_with_hooks(self, *_: Any, **__: Any) -> Any:
        raise NotImplementedError(
            "VLLMLens does not support run_with_hooks (used by the SAE "
            "replacement-hook eval path). Sparsity / variance eval is fine; "
            "only CE-loss and KL eval are unavailable."
        )
