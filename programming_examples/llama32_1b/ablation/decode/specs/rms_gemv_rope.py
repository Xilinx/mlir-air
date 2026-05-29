"""Concrete KernelGroupSpec for the decode rms_gemv_rope kernel-group.

Mirrors the production stitch-spec in
multi_launch_builder/rms_gemv_rope_multi.py (the 6-launch decode pipeline:
RMSNorm + Q/K/V GEMV + RoPE Q + RoPE K).

Slot conventions for standalones:
  - rmsnorm:  (x_in[emb], norm_w[emb], out[emb])           weight=1, out=2
  - gemv:     (W[out, in], x[in], y[out])                  weight=0, out=2
              (matvec convention — W is at slot 0, NOT slot 1 like prefill GEMM.)
  - rope:     (in_flat[N], lut[head_dim], out_flat[N])     weight=1 (LUT), out=2

Production decode shapes (single token):
  emb_dim=2048, kv_dim=512, n_heads=32, n_kv_heads=8, head_dim=64.
  q_total = n_heads * head_dim = 2048 (= emb_dim by construction)
  k_total = n_kv_heads * head_dim = 512 (= kv_dim by construction)
"""

from standalone_builders.rms_gemv_rope import STANDALONES as _PLAN0_STANDALONES
from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec

# Plan 0's STANDALONES is a list of (name, build_fn, build_kwargs) tuples.
# Convert to a name→(build_fn, build_kwargs) lookup for SubLaunchSpec construction.
_BUILDERS = {name: (build_fn, kwargs) for name, build_fn, kwargs in _PLAN0_STANDALONES}


def _b(name):
    """Helper: extract (build_fn, build_kwargs) for a sub-launch by name."""
    return _BUILDERS[name]


SPEC = KernelGroupSpec(
    name="rms_gemv_rope",
    sub_launches=(
        # idx=0: RMSNorm — slot 0=x_in, slot 1=norm_w (weight), slot 2=normed (out)
        SubLaunchSpec("rmsnorm", _b("rmsnorm")[0], _b("rmsnorm")[1], 1, 2),
        # idx=1: Q GEMV — slot 0=W (wq), slot 1=x (normed), slot 2=y (q)
        SubLaunchSpec("q_gemv", _b("q_gemv")[0], _b("q_gemv")[1], 0, 2),
        # idx=2: K GEMV — slot 0=W (wk), slot 1=x, slot 2=y (k)
        SubLaunchSpec("k_gemv", _b("k_gemv")[0], _b("k_gemv")[1], 0, 2),
        # idx=3: V GEMV — slot 0=W (wv), slot 1=x, slot 2=y (v)
        SubLaunchSpec("v_gemv", _b("v_gemv")[0], _b("v_gemv")[1], 0, 2),
        # idx=4: RoPE Q — slot 0=in (q), slot 1=lut_q (weight), slot 2=out (q_roped)
        SubLaunchSpec("rope_q", _b("rope_q")[0], _b("rope_q")[1], 1, 2),
        # idx=5: RoPE K — slot 0=in (k), slot 1=lut_k, slot 2=out (k_roped)
        SubLaunchSpec("rope_k", _b("rope_k")[0], _b("rope_k")[1], 1, 2),
    ),
    merged_arg_signature=(
        "x_in",  # 0  activation input
        "norm_w",  # 1  weight (static)
        "normed",  # 2  intermediate
        "wq",  # 3  weight (static)
        "q",  # 4  intermediate
        "wk",  # 5  weight (static)
        "k",  # 6  intermediate
        "wv",  # 7  weight (static)
        "v",  # 8  intermediate
        "lut_q",  # 9  weight (static)
        "lut_k",  # 10 weight (static)
        "q_roped",  # 11 intermediate (also output for validation)
        "k_roped",  # 12 intermediate (also output for validation)
    ),
    weight_slots=frozenset({1, 3, 5, 7, 9, 10}),
    intermediate_slots=frozenset({2, 4, 6, 8, 11, 12}),
    output_slots_for_validation=(2, 4, 6, 8, 11, 12),
    baton_links=(
        # rmsnorm.normed (slot 2) -> q/k/v_gemv.x (slot 1 — matvec convention!)
        BatonLink(
            producer_idx=0, producer_out_slot=2, consumer_idx=1, consumer_in_slot=1
        ),
        BatonLink(
            producer_idx=0, producer_out_slot=2, consumer_idx=2, consumer_in_slot=1
        ),
        BatonLink(
            producer_idx=0, producer_out_slot=2, consumer_idx=3, consumer_in_slot=1
        ),
        # q_gemv.q (slot 2) -> rope_q.in (slot 0)
        BatonLink(
            producer_idx=1, producer_out_slot=2, consumer_idx=4, consumer_in_slot=0
        ),
        # k_gemv.k (slot 2) -> rope_k.in (slot 0)
        BatonLink(
            producer_idx=2, producer_out_slot=2, consumer_idx=5, consumer_in_slot=0
        ),
    ),
)
