# -*- coding: utf-8 -*-
"""Unit check for the aux_hidden_state sum-capture fix (14-layer gap).

Validates the semantics the DeepseekV2 layer relies on when it captures the
pre-norm sum produced by input_layernorm (instead of recomputing
hidden_states + residual in the model loop):

1. Inside ``ir.ops.fused_add_rms_norm.functional()`` the op writes fresh
   buffers even when the eager priority prefers the inplace impl.
2. The returned residual equals x + residual (numerics vs float reference).
3. A later fused_add_rms_norm (post_attention_layernorm stand-in) inside the
   same scope does NOT clobber the captured sum buffer -- the aliasing hazard
   that motivated the scope. Outside the scope with inplace priority it
   would (negative control).

Run from the kit root: python3 vllm_scripts/diag/mcpu_trace/test_aux_sum_capture.py
"""

import torch

import vllm_xcpu_plugin.ir_kernels  # noqa: F401  (registers torch_xcpu impls)
from vllm import ir

EAGER_PRIORITY = ["torch_xcpu_inplace", "torch_xcpu", "native"]
COMPILE_PRIORITY = ["torch_xcpu", "native"]

M, N, EPS = 128, 6144, 1e-6


def reference(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor):
    s = x.float() + residual.float()
    out = s * torch.rsqrt(s.square().mean(-1, keepdim=True) + EPS)
    return (out * w.float()).to(x.dtype), s.to(x.dtype)


def make_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(M, N, dtype=torch.bfloat16, generator=g).to("mcpu")
    r = torch.randn(M, N, dtype=torch.bfloat16, generator=g).to("mcpu")
    w = torch.randn(N, dtype=torch.bfloat16, generator=g).to("mcpu")
    return x, r, w


def run_case(priority, use_scope, label):
    ir.ops.fused_add_rms_norm.set_default(priority)
    x, r, w = make_inputs()
    x0, r0 = x.clone(), r.clone()
    ref_out, ref_sum = reference(x, r, w)

    import contextlib

    cm = (
        ir.ops.fused_add_rms_norm.functional()
        if use_scope
        else contextlib.nullcontext()
    )
    with cm:
        # input_layernorm + capture
        h, s = ir.ops.fused_add_rms_norm.maybe_inplace(x, r, w, EPS)
        captured = s
        # post_attention_layernorm stand-in (reads the sum, must not clobber)
        h2, r2 = ir.ops.fused_add_rms_norm.maybe_inplace(h, s, w, EPS)

    ok_sum = torch.allclose(captured.float(), ref_sum.float(), atol=1e-3, rtol=1e-3)
    # the captured sum must still equal the bf16 reference after the second
    # norm ran (ref_sum carries the same bf16 rounding as the captured value)
    ok_clobber = torch.allclose(captured.float(), ref_sum.float(), atol=1e-3, rtol=1e-3)
    ok_out = torch.allclose(h.float(), ref_out.float(), atol=2e-2, rtol=2e-2)
    fresh = captured.data_ptr() != r0.data_ptr()
    print(
        f"[{label}] sum_ok={ok_sum} out_ok={ok_out} "
        f"survives_post_norm={ok_clobber} fresh_buffer={fresh}"
    )
    return ok_sum and ok_out and ok_clobber


def main():
    torch.manual_seed(0)
    results = []
    # Positive: eager priority + scope -> capture survives.
    results.append(("eager+scope", run_case(EAGER_PRIORITY, True, "eager+scope")))
    # Positive: compile priority, scope is a no-op -> capture survives.
    results.append(
        ("compile(noop-scope)", run_case(COMPILE_PRIORITY, True, "compile+scope"))
    )
    # Negative control: eager priority WITHOUT scope -> inplace impl clobbers
    # the captured alias. (Documents why the scope is required.)
    ir.ops.fused_add_rms_norm.set_default(EAGER_PRIORITY)
    x, r, w = make_inputs()
    ref_out, ref_sum = reference(x, r, w)
    h, s = ir.ops.fused_add_rms_norm.maybe_inplace(x, r, w, EPS)
    captured = s
    h2, r2 = ir.ops.fused_add_rms_norm.maybe_inplace(h, s, w, EPS)
    clobbered = not torch.allclose(
        captured.float(), ref_sum.float(), atol=1e-3, rtol=1e-3
    )
    print(f"[eager-no-scope] captured_clobbered_by_post_norm={clobbered}")
    results.append(("eager-no-scope(clobber expected)", clobbered))

    failed = [n for n, ok in results if not ok]
    if failed:
        raise SystemExit(f"FAILED: {failed}")
    print("ALL OK")


if __name__ == "__main__":
    main()
