"""Concrete KernelGroupSpec for the decode o_gemv_ffn kernel-group.

Mirrors the production stitch-spec in
multi_launch_builder/o_gemv_ffn_multi.py:308-482 (the 8-launch decode pipeline:
O GEMV + Add + RMSNorm + Gate GEMV + Up GEMV + SwiGLU + Down GEMV + Add).

15 merged-func args (slots 0-14); weights at {0,5,7,9,12};
intermediates at {2,4,6,8,10,11,13,14}.

Slot conventions for standalones (CRITICAL — different from prefill GEMM):
  - gemv:     (W[out, in], x[in], y[out])         weight=0, out=2 (matvec convention)
  - add_2d:   (a[N,d], b[N,d], out[N,d])           no weight, out=2
                (called as N=emb_dim//8, d=emb_dim, herd_x=8)
  - rms_1d:   (x[emb], norm_w[emb], out[emb])      weight=1, out=2
  - swiglu:   (gate[hidden], up[hidden], out[hidden])  no weight, out=2

Production decode shapes (single token):
  emb_dim=2048, hidden_dim=8192, head_dim=64.
  K=2048 GEMVs (O, Gate, Up): tile_m=8, m_input=4, herd_m=8
  K=8192 Down GEMV:            tile_m=2, m_input=1, herd_m=8

Note on Down GEMV "renaming":
  The PRODUCTION MERGED ELF renames Down GEMV's @matvec to
  @dg_matvec_vectorized_bf16_bf16 + link_with="mv_k8192.o" because two GEMVs
  with different signatures can't coexist in one ELF with the same C symbol.
  STANDALONE down_gemv has no such conflict — it's its own ELF — so it uses
  the standard @matvec_vectorized_bf16_bf16 + mv.o (compiled with default
  tile_m). The MLIR loop structure uses tile_m=2, m_input=1 from build_gemv.
"""

from ml_dtypes import bfloat16

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec

# ---------------------------------------------------------------------------
# Sub-launch standalone builders
# ---------------------------------------------------------------------------


def _build_o_gemv_standalone():
    """O GEMV: (wo[2048,2048], attn_out[2048], proj[2048])."""
    from matvec import build_module as build_gemv

    return build_gemv(2048, 2048, 8, 4, 8, bfloat16, bfloat16)


def _build_add1_standalone():
    """Residual add #1 (post-attn): (proj[2048], x_residual[2048], res1[2048]).

    eltwise_add.build_module(M, N, ...) accepts 2D shape (M, N). Production
    calls it with M=emb_dim, N=emb_dim//8, herd=[8,1] — so the 1D activation
    is reshaped/tiled across M=emb_dim rows of N=emb_dim//8 cols.

    Wraps via _wrap_ir_in_launch (eltwise_add emits a bare herd).
    """
    from eltwise_add.eltwise_add import build_module as build_add
    from kernel_builder.stitching import _wrap_ir_in_launch
    from air.ir import Module

    bare = str(build_add(2048, 2048 // 8, bfloat16, vector_size=16, herd_x=8, herd_y=1))
    return Module.parse(_wrap_ir_in_launch(bare))


def _build_rmsnorm_standalone():
    """1D RMSNorm: (res1[2048], ffn_norm_w[2048], normed2[2048]).

    Imports _build_rms_1d_ir from o_gemv_ffn_multi (returns MLIR text)
    and parses to a Module. This is the SAME 1D RMSNorm wrapper used by
    the production merged ELF, so byte-equality is guaranteed.
    """
    from multi_launch_builder.o_gemv_ffn_multi import _build_rms_1d_ir
    from air.ir import Module

    return Module.parse(_build_rms_1d_ir(2048, vector_size=16))


def _build_gate_or_up_gemv_standalone():
    """Gate or Up GEMV: (w[8192,2048], normed2[2048], out[8192])."""
    from matvec import build_module as build_gemv

    return build_gemv(8192, 2048, 8, 4, 8, bfloat16, bfloat16)


def _build_swiglu_standalone():
    """SwiGLU: (gate[8192], up[8192], swiglu[8192]).

    Uses kernel_builder.ffn_swiglu.silu_and_mul.build_module (1D variant).
    Wraps via _wrap_ir_in_launch (silu emits a bare herd).
    """
    from kernel_builder.ffn_swiglu.silu_and_mul import build_module as build_silu
    from kernel_builder.stitching import _wrap_ir_in_launch
    from air.ir import Module

    bare = str(build_silu(8192, 8192 // 8, bfloat16, herd_x=8, herd_y=1))
    return Module.parse(_wrap_ir_in_launch(bare))


def _build_down_gemv_standalone():
    """Down GEMV: (wdown[2048,8192], swiglu[8192], down[2048]).

    Smaller tiles: tile_m=2, m_input=1 (production uses these for K=8192).
    As a STANDALONE, uses the default mv.o — no rename needed (only the
    merged ELF needs the rename to avoid C-symbol collision with K=2048
    GEMVs).
    """
    from matvec import build_module as build_gemv

    return build_gemv(2048, 8192, 2, 1, 8, bfloat16, bfloat16)


def _build_add2_standalone():
    """Residual add #2 (post-FFN): (down[2048], res1[2048], output[2048]).

    Same builder as _build_add1_standalone — production uses the SAME
    config (M=emb_dim, N=emb_dim//8, herd=[8,1]) for both residual adds.
    """
    return _build_add1_standalone()


# ---------------------------------------------------------------------------
# KernelGroupSpec
# ---------------------------------------------------------------------------

SPEC = KernelGroupSpec(
    name="o_gemv_ffn",
    sub_launches=(
        # idx=0: O GEMV — slot 0=W (wo), slot 1=x (attn_out), slot 2=y (proj)
        SubLaunchSpec("o_gemv", _build_o_gemv_standalone, {}, 0, 2),
        # idx=1: Add (post-attn residual) — no weight, slot 0=A, 1=B, 2=res1
        SubLaunchSpec("add_attn_residual", _build_add1_standalone, {}, None, 2),
        # idx=2: FFN RMSNorm — slot 0=x (res1), 1=norm_w, 2=normed2
        SubLaunchSpec("ffn_rmsnorm", _build_rmsnorm_standalone, {}, 1, 2),
        # idx=3: Gate GEMV — slot 0=W (wgate), 1=x (normed2), 2=y (gate)
        SubLaunchSpec("gate_gemv", _build_gate_or_up_gemv_standalone, {}, 0, 2),
        # idx=4: Up GEMV — slot 0=W (wup), 1=x (normed2), 2=y (up)
        SubLaunchSpec("up_gemv", _build_gate_or_up_gemv_standalone, {}, 0, 2),
        # idx=5: SwiGLU — no weight, slot 0=gate, 1=up, 2=swiglu
        SubLaunchSpec("swiglu", _build_swiglu_standalone, {}, None, 2),
        # idx=6: Down GEMV — slot 0=W (wdown), 1=x (swiglu), 2=y (down)
        SubLaunchSpec("down_gemv_k8192", _build_down_gemv_standalone, {}, 0, 2),
        # idx=7: Add (FFN residual) — no weight, slot 0=A (down), 1=B (res1), 2=output
        SubLaunchSpec("add_ffn_residual", _build_add2_standalone, {}, None, 2),
    ),
    merged_arg_signature=(
        "wo",  # 0  weight (static)
        "attn_out",  # 1  activation input
        "proj",  # 2  intermediate
        "x_residual",  # 3  activation input
        "res1",  # 4  intermediate (shared: add1 out + add2 B)
        "ffn_norm_w",  # 5  weight (static)
        "normed2",  # 6  intermediate
        "wgate",  # 7  weight (static)
        "gate",  # 8  intermediate
        "wup",  # 9  weight (static)
        "up",  # 10 intermediate
        "swiglu",  # 11 intermediate
        "wdown",  # 12 weight (static)
        "down",  # 13 intermediate
        "output",  # 14 intermediate (final output)
    ),
    weight_slots=frozenset({0, 5, 7, 9, 12}),
    intermediate_slots=frozenset({2, 4, 6, 8, 10, 11, 13, 14}),
    output_slots_for_validation=(14,),
    baton_links=(
        # Stitch arg_map verified against o_gemv_ffn_multi.py lines 394-403:
        #   L1 {0:0,1:1,2:2}  L2 {0:2,1:3,2:4}   L3 {0:4,1:5,2:6}
        #   L4 {0:7,1:6,2:8}  L5 {0:9,1:6,2:10}  L6 {0:8,1:10,2:11}
        #   L7 {0:12,1:11,2:13}  L8 {0:13,1:4,2:14}
        BatonLink(0, 2, 1, 0),  # o_gemv.proj -> add_attn.A
        BatonLink(1, 2, 2, 0),  # add_attn.res1 -> ffn_rmsnorm.x
        BatonLink(2, 2, 3, 1),  # ffn_rmsnorm.normed2 -> gate_gemv.x (slot 1!)
        BatonLink(2, 2, 4, 1),  # ffn_rmsnorm.normed2 -> up_gemv.x (slot 1!)
        BatonLink(3, 2, 5, 0),  # gate_gemv.gate -> swiglu.gate
        BatonLink(4, 2, 5, 1),  # up_gemv.up -> swiglu.up
        BatonLink(5, 2, 6, 1),  # swiglu -> down_gemv.x (slot 1!)
        BatonLink(6, 2, 7, 0),  # down_gemv.down -> add_ffn.A
        BatonLink(1, 2, 7, 1),  # add_attn.res1 -> add_ffn.B (residual-of-residual)
    ),
)
