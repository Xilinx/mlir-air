"""Concrete KernelGroupSpec for the prefill rms_gemms_rope kernel-group.

Mirrors the production stitch-spec in
multi_launch_builder/rms_gemms_rope_multi.py:467-474 (which lists the
arg mappings for the 6 sub-launches in the merged ELF).

Slot conventions for standalones:
  - rmsnorm:  (x_in[seq, emb], norm_w[emb], out[seq, emb])     output at slot 2
  - gemm:     (a[seq, K], b[K, N], c[seq, N])                  output at slot 2
              (kernel_builder/gemm_builder.py:107 signature is (m, k, n, ...) —
               no positional M arg; weight at slot 1, output at slot 2.)
  - rope_2d:  (in_2d[rows, cols], lut_1d[N], out_2d[rows, cols]) output at slot 2
"""

from ml_dtypes import bfloat16

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec


def _build_rmsnorm_standalone():
    """Wrap weighted_rms_norm in air.launch+segment for solo invocation."""
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from kernel_builder.stitching import _wrap_ir_in_launch
    from air.ir import Module

    bare = str(build_rms(2048, 2048, bfloat16, 16, herd_x=8))
    wrapped_text = _wrap_ir_in_launch(bare)
    return Module.parse(wrapped_text)


def _build_gemm_standalone(k, n):
    """Production prefill GEMM: (seq=2048, k, n) with the production tile config.

    _build_gemm_module signature: (m, k, n, tile_m, tile_k_l2, tile_k_l1, tile_n,
    herd_m, herd_n).  Slots in standalone: 0=A (activation), 1=B (weight), 2=C (output).
    """
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        k,
        n,
        tile_m=64,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=128,
        herd_m=8,
        herd_n=4,
    )


def _build_rope_2d_standalone(outer_rows, outer_cols):
    from multi_launch_builder.rms_gemms_rope_multi import _build_rope_2d

    return _build_rope_2d(outer_rows, outer_cols, 64, bfloat16, herd_x=8)


SPEC = KernelGroupSpec(
    name="rms_gemms_rope",
    sub_launches=(
        SubLaunchSpec("rmsnorm", _build_rmsnorm_standalone, {}, 1, 2),
        SubLaunchSpec("q_gemm", _build_gemm_standalone, {"k": 2048, "n": 2048}, 1, 2),
        SubLaunchSpec("k_gemm", _build_gemm_standalone, {"k": 2048, "n": 512}, 1, 2),
        SubLaunchSpec("v_gemm", _build_gemm_standalone, {"k": 2048, "n": 512}, 1, 2),
        SubLaunchSpec(
            "rope_q",
            _build_rope_2d_standalone,
            {"outer_rows": 2048, "outer_cols": 2048},
            1,
            2,
        ),
        SubLaunchSpec(
            "rope_k",
            _build_rope_2d_standalone,
            {"outer_rows": 2048, "outer_cols": 512},
            1,
            2,
        ),
    ),
    merged_arg_signature=(
        "x_in",
        "norm_w",
        "normed",
        "wq",
        "q",
        "wk",
        "k",
        "wv",
        "v",
        "lut_q",
        "lut_k",
        "q_roped",
        "k_roped",
    ),
    weight_slots=frozenset({1, 3, 5, 7, 9, 10}),
    intermediate_slots=frozenset({2, 4, 6, 8, 11, 12}),
    output_slots_for_validation=(2, 4, 6, 8, 11, 12),
    baton_links=(
        BatonLink(
            producer_idx=0,
            producer_out_slot=2,
            consumer_idx=1,
            consumer_in_slot=0,
        ),  # rmsnorm.normed -> q_gemm.x
        BatonLink(
            producer_idx=0,
            producer_out_slot=2,
            consumer_idx=2,
            consumer_in_slot=0,
        ),  # rmsnorm.normed -> k_gemm.x
        BatonLink(
            producer_idx=0,
            producer_out_slot=2,
            consumer_idx=3,
            consumer_in_slot=0,
        ),  # rmsnorm.normed -> v_gemm.x
        BatonLink(
            producer_idx=1,
            producer_out_slot=2,
            consumer_idx=4,
            consumer_in_slot=0,
        ),  # q_gemm.q -> rope_q.in
        BatonLink(
            producer_idx=2,
            producer_out_slot=2,
            consumer_idx=5,
            consumer_in_slot=0,
        ),  # k_gemm.k -> rope_k.in
    ),
)
