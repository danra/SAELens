import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sae_lens.vllm_lens_loader import VLLMLensProxy, _layer_from_hook_name


def test_layer_from_hook_name_extracts_from_resid_post():
    assert _layer_from_hook_name("blocks.0.hook_resid_post") == 0
    assert _layer_from_hook_name("blocks.30.hook_resid_post") == 30


def test_layer_from_hook_name_rejects_non_resid_post_tlens_hooks():
    # vllm-lens only captures post-residual full-layer output; other TLens
    # hooks (attn.hook_q, hook_resid_pre, hook_mlp_out, etc.) aren't available.
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("blocks.17.attn.hook_q")
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("blocks.5.hook_resid_pre")
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("blocks.5.hook_mlp_out")


def test_layer_from_hook_name_rejects_hf_style():
    # HF-style hook names are no longer accepted — caller must translate to
    # TLens form so the contract (post-residual output) is explicit.
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("model.language_model.layers.30")
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("model.layers.5")


def test_layer_from_hook_name_raises_on_unrecognized():
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("transformer.h.0.mlp")
    with pytest.raises(ValueError, match="hook_resid_post"):
        _layer_from_hook_name("model.embed_tokens")


def _install_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    activations_per_request: list[torch.Tensor] | None = None,
):
    """Install a minimal fake `vllm` module so VLLMLensProxy can be exercised
    without the real package. Returns the FakeLLM mock so tests can inspect
    captured calls.
    """
    fake_llm_instance = MagicMock(name="LLM_instance")
    fake_llm_instance.get_tokenizer.return_value = MagicMock(name="tokenizer")

    def fake_generate(
        prompt_token_ids: list[list[int]],
        sampling_params: object,  # noqa: ARG001
        use_tqdm: bool,  # noqa: ARG001
    ) -> list[SimpleNamespace]:
        # Return one RequestOutput-shaped object per prompt with the canned
        # activation tensor (in (num_layers=1, seq_len, d_model) shape).
        outs: list[SimpleNamespace] = []
        for _ in prompt_token_ids:
            o = SimpleNamespace()
            if activations_per_request is None:
                o.activations = {"residual_stream": torch.zeros(1, 4, 8)}
            else:
                o.activations = {"residual_stream": activations_per_request.pop(0)}
            outs.append(o)
        return outs

    fake_llm_instance.generate.side_effect = fake_generate

    fake_llm_class = MagicMock(name="LLM_class", return_value=fake_llm_instance)
    fake_sampling_params = MagicMock(name="SamplingParams_class")

    fake_module = ModuleType("vllm")
    fake_module.LLM = fake_llm_class  # type: ignore[attr-defined]
    fake_module.SamplingParams = fake_sampling_params  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_module)
    return fake_llm_instance, fake_llm_class, fake_sampling_params


def test_vllm_lens_proxy_advertises_no_forward_support(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy", target_device="cpu")
    assert proxy.supports_forward is False


def test_vllm_lens_proxy_run_with_cache_returns_stacked_activations(
    monkeypatch: pytest.MonkeyPatch,
):
    # Two prompts × 4 tokens × 8 dim — distinct tensors so we can verify
    # ordering survives the round trip.
    a = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
    b = torch.arange(32, 64, dtype=torch.float32).reshape(1, 4, 8)
    _, llm_cls, _ = _install_fake_vllm(monkeypatch, activations_per_request=[a, b])

    proxy = VLLMLensProxy(model_name="dummy", target_device="cpu")
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    out, cache = proxy.run_with_cache(
        tokens, names_filter=["blocks.30.hook_resid_post"]
    )

    assert out is None
    acts = cache["blocks.30.hook_resid_post"]
    # Squeezed layer dim, stacked over batch → (batch, ctx, d_model).
    assert acts.shape == (2, 4, 8)
    torch.testing.assert_close(acts[0], a.squeeze(0))
    torch.testing.assert_close(acts[1], b.squeeze(0))
    llm_cls.assert_called_once_with(model="dummy", dtype="bfloat16")


def test_vllm_lens_proxy_run_with_cache_passes_layer_to_extra_args(
    monkeypatch: pytest.MonkeyPatch,
):
    _, _, sp_cls = _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    proxy.run_with_cache(
        torch.tensor([[1, 2]]),
        names_filter=["blocks.30.hook_resid_post"],
    )
    sp_cls.assert_called_once()
    kwargs = sp_cls.call_args.kwargs
    assert kwargs["max_tokens"] == 1
    assert kwargs["extra_args"] == {"output_residual_stream": [30]}


def test_vllm_lens_proxy_run_with_cache_rejects_multi_hook(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    with pytest.raises(NotImplementedError, match="one hook at a time"):
        proxy.run_with_cache(
            torch.tensor([[1, 2]]),
            names_filter=["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
        )


def test_vllm_lens_proxy_run_with_cache_rejects_unbatched_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    with pytest.raises(ValueError, match=r"shape \(batch, ctx\)"):
        proxy.run_with_cache(
            torch.tensor([1, 2, 3]),  # 1D → invalid
            names_filter=["blocks.0.hook_resid_post"],
        )


def test_vllm_lens_proxy_raises_when_activations_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_llm, _, _ = _install_fake_vllm(monkeypatch)

    def no_activations(
        prompt_token_ids: list[list[int]],
        sampling_params: object,  # noqa: ARG001
        use_tqdm: bool,  # noqa: ARG001
    ) -> list[SimpleNamespace]:
        return [SimpleNamespace() for _ in prompt_token_ids]

    fake_llm.generate.side_effect = no_activations

    proxy = VLLMLensProxy(model_name="dummy")
    with pytest.raises(RuntimeError, match="vllm-lens"):
        proxy.run_with_cache(
            torch.tensor([[1, 2]]),
            names_filter=["blocks.0.hook_resid_post"],
        )


def test_vllm_lens_proxy_forward_and_run_with_hooks_raise(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    with pytest.raises(NotImplementedError):
        proxy.forward(torch.tensor([[1]]))
    with pytest.raises(NotImplementedError):
        proxy.run_with_hooks(torch.tensor([[1]]))


def test_vllm_lens_proxy_to_is_noop(monkeypatch: pytest.MonkeyPatch):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    # .to() must not error and must return self so chained calls work.
    assert proxy.to("cuda:0") is proxy


def test_vllm_lens_proxy_cfg_device_is_cpu(monkeypatch: pytest.MonkeyPatch):
    # _get_input_token_device falls back to model.cfg.device. vLLM consumes
    # tokens as python lists, so cpu is the right answer.
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy")
    assert proxy.cfg.device == torch.device("cpu")


def test_vllm_lens_proxy_target_device_moves_activations(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_fake_vllm(monkeypatch)
    proxy = VLLMLensProxy(model_name="dummy", target_device="cpu")
    _, cache = proxy.run_with_cache(
        torch.tensor([[1, 2]]),
        names_filter=["blocks.0.hook_resid_post"],
    )
    assert cache["blocks.0.hook_resid_post"].device == torch.device("cpu")
