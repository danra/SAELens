"""Validate that VLLMLens residual stream matches TLens / HF on a tiny model.

Run on a box with a GPU, with vllm + vllm-lens installed:

    python -m scripts.validate_vllm_lens_equivalence

This is a standalone smoke check, not a pytest test, because vLLM ships only
as a CUDA wheel and is too heavy for CPU CI. Use it as a one-off sanity check
before kicking off a long training run with ``model_class_name="VLLMLens"``.
"""

from typing import cast

import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from sae_lens.vllm_lens_loader import VLLMLensProxy

MODEL_NAME = "EleutherAI/pythia-14m"
LAYER = 3
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
# Equal-length sequences of valid Pythia token ids; both reference impls see
# identical inputs so any mismatch is real numerical divergence, not BOS/pad.
TOKENS = torch.tensor(
    [
        [10, 200, 30, 400, 50, 600, 70, 800],
        [11, 201, 31, 401, 51, 601, 71, 801],
    ],
    dtype=torch.long,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This script requires a CUDA GPU (vLLM is GPU-only).")

    tokens = TOKENS.to("cuda")

    print(f"[vLLM] loading {MODEL_NAME} via VLLMLensProxy...")
    proxy = VLLMLensProxy(
        model_name=MODEL_NAME,
        target_device="cuda",
        # Leave room for HF + TLens on the same GPU.
        vllm_kwargs={"dtype": "float32", "gpu_memory_utilization": 0.4},
    )
    _, vllm_cache = proxy.run_with_cache(tokens, names_filter=[HOOK_NAME])
    vllm_acts = vllm_cache[HOOK_NAME]
    print(f"[vLLM] residual stream shape={tuple(vllm_acts.shape)}")

    print(f"[TLens] loading {MODEL_NAME}...")
    tl_model = HookedTransformer.from_pretrained_no_processing(
        MODEL_NAME, device="cuda"
    )
    tl_model.to(torch.float32)
    _, tl_cache = tl_model.run_with_cache(tokens, names_filter=[HOOK_NAME])
    tl_acts = tl_cache[HOOK_NAME]

    diff = (vllm_acts - tl_acts).abs()
    print(
        f"[TLens vs vLLM] max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e}"
    )
    torch.testing.assert_close(vllm_acts, tl_acts, atol=1e-3, rtol=1e-3)
    print("[TLens vs vLLM] OK (atol=1e-3, rtol=1e-3)")
    del tl_model
    torch.cuda.empty_cache()

    print(f"[HF] loading {MODEL_NAME}...")
    hf_model = cast(
        torch.nn.Module,
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32),
    )
    hf_model.to("cuda")
    hf_model.eval()
    with torch.no_grad():
        hf_out = hf_model(tokens, output_hidden_states=True)
    # hidden_states[i+1] is the post-residual output of layer i.
    hf_acts = hf_out.hidden_states[LAYER + 1]
    diff = (vllm_acts - hf_acts).abs()
    print(
        f"[HF vs vLLM] max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e}"
    )
    torch.testing.assert_close(vllm_acts, hf_acts, atol=1e-3, rtol=1e-3)
    print("[HF vs vLLM] OK (atol=1e-3, rtol=1e-3)")


if __name__ == "__main__":
    main()
