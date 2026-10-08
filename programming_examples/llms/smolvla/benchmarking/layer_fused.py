# SPDX-License-Identifier: MIT
"""One whole SmolVLA backbone layer as a single multi-launch ELF.

Stitches the three per-layer ELFs -- rms_gemms_rope_fused_qkv (4 launches),
masked FlashAttention (1 launch) and o_ffn_fused_gu (6 launches) -- into one
func, so a layer is one XRT dispatch instead of three. Each sub-module's
launches keep their own shim-DMA tiling through the per-launch
`air.shim_dma_tile_sizes` attribute: the global runtime_loop_tiling_sizes
cannot serve all three (the FA head axis only tolerates a factor of 1).

Combined args:
  %arg0  x           (seq, emb)          layer input, also the O residual
  %arg1  attn_norm   (emb,)
  %arg2  normed      (seq, emb)
  %arg3  w_qkv       (emb, emb+2kv)
  %arg4  qkv         (seq, emb+2kv)      FA reads V from its last kv columns
  %arg5  rope_lut_q
  %arg6  q_roped     (seq, emb)
  %arg7  rope_lut_k
  %arg8  k_roped     (seq, kv)
  %arg9  mask        (seq, seq)          additive bf16
  %arg10 attn_out    (seq, emb)
  %arg11 wo          (emb, emb)
  %arg12 proj        (seq, emb)
  %arg13 res1        (seq, emb)
  %arg14 ffn_norm    (emb,)
  %arg15 normed2     (seq, emb)
  %arg16 w_gateup    (emb, 2*hidden)     SwiGLU-interleaved
  %arg17 swiglu      (seq, hidden)
  %arg18 w_down      (hidden, emb)
  %arg19 down        (seq, emb)
  %arg20 output      (seq*emb,)
"""
import re
import sys
import types
from pathlib import Path

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[3])
]

from air_examples.llms.shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

# Sub-module operand -> combined arg. o_ffn's operand 8 (the wide gate|up
# buffer) is dead under the SwiGLU epilogue and is dropped.
_RGR_MAP = {i: i for i in range(9)}
_FA_MAP = {0: 6, 1: 8, 2: 4, 3: 9, 4: 10}
_OFFN_MAP = {
    0: 10,
    1: 11,
    2: 12,
    3: 0,
    4: 13,
    5: 14,
    6: 15,
    7: 16,
    9: 17,
    10: 18,
    11: 19,
    12: 20,
}

# gemm_engine O+FFN (one launch; operands attn, wo, x, res1, wgu, sw, wdn, out). proj,
# normed2 and down (12, 15, 19) are never materialised and ffn_norm (14) is folded into
# w_gateup, which is permuted (gemm_engine.permute_gate_up); out is 2-D.
OFFN_ENGINE_ORDER = ["attn", "wo", "x", "res1", "wgu", "sw", "wdn", "out"]
_OFFN_ENGINE_MAP = {0: 10, 1: 11, 2: 0, 3: 13, 4: 16, 5: 17, 6: 18, 7: 20}
_OFFN_ENGINE_UNUSED = {12, 14, 15, 19}

LAYER_STATIC = {1, 3, 5, 7, 9, 11, 14, 16, 18}
LAYER_INTERMEDIATE = {2, 4, 6, 8, 10, 12, 13, 15, 17, 19, 20}
LAYER_OUT = 20


def _signature_types(ir):
    sig = re.search(r"func\.func @\w+\(([^)]*)\)", ir).group(1)
    return [a.split(":", 1)[1].strip() for a in sig.split(",") if a.strip()]


def _privates(ir):
    return set(re.findall(r"func\.func private (@\w+)", ir))


def build_layer_module(rgr_ir, fa_ir, offn_ir, tilings, offn_engine=False):
    """rgr_ir / fa_ir / offn_ir: sub-module texts (FA built with attn_mask=True
    and v_cols = the qkv width). tilings: {"rgr", "fa", "offn"} -> the shim-DMA
    tile sizes each sub-module's launches were tuned with. offn_engine: offn_ir
    is the one-launch gemm_engine O+FFN."""
    offn_map = _OFFN_ENGINE_MAP if offn_engine else _OFFN_MAP
    types = [None] * 21
    for ir, amap in ((rgr_ir, _RGR_MAP), (fa_ir, _FA_MAP), (offn_ir, offn_map)):
        for op_idx, t in enumerate(_signature_types(ir)):
            if op_idx in amap:
                c = amap[op_idx]
                assert types[c] in (None, t), f"arg{c}: {types[c]} vs {t}"
                types[c] = t
    unused = _OFFN_ENGINE_UNUSED if offn_engine else set()
    for c in unused:
        types[c] = types[1] if c == 14 else types[0]
    assert None not in types, types
    base_args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(types)]

    parts = (
        ("rg", rgr_ir, _RGR_MAP, "rgr"),
        ("at", fa_ir, _FA_MAP, "fa"),
        ("of", offn_ir, offn_map, "offn"),
    )
    slices = [
        KernelSlice(ir, p, amap, extern_syms=_privates(ir)) for p, ir, amap, _ in parts
    ]
    module = stitch_elf(
        "layer",
        base_args,
        slices,
        debug_dump_path="/tmp/layer_fused_parse_error.mlir",
        allow_unreferenced_args=unused,
    )

    from air.ir import DenseI64ArrayAttr

    per_launch = [
        tilings[key] for _, ir, _, key in parts for _ in range(ir.count("air.launch "))
    ]
    func = next(
        op
        for op in module.body.operations
        if op.operation.name == "func.func"
        and op.attributes["sym_name"].value == "layer"
    )
    launches = [
        op
        for op in func.regions[0].blocks[0].operations
        if op.operation.name == "air.launch"
    ]
    assert len(launches) == len(per_launch), (len(launches), len(per_launch))
    with module.context:
        for op, ts in zip(launches, per_launch):
            if ts:
                op.attributes["air.shim_dma_tile_sizes"] = DenseI64ArrayAttr.get(ts)
    print(
        f"  Layer module: {len(launches)} launches, {len(str(module).splitlines())} lines, parsed OK"
    )
    return module


# N layers in one func. Layer args shared by every layer (luts, mask,
# intermediates) appear once; the per-layer weights repeat. Layer i reads
# X[i % 2] and writes X[(i + 1) % 2] through a flat view (o_ffn's output is 1-D).
MULTI_SHARED = (2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 19)
MULTI_PER_LAYER = (1, 3, 11, 14, 16, 18)


def multi_layer_arg(layer, layer_arg):
    """Combined arg index of `layer`'s single-layer arg `layer_arg` (not 0 / 20)."""
    if layer_arg in MULTI_SHARED:
        return 2 + MULTI_SHARED.index(layer_arg)
    return (
        2
        + len(MULTI_SHARED)
        + layer * len(MULTI_PER_LAYER)
        + MULTI_PER_LAYER.index(layer_arg)
    )


# Three-launch layer: gemm_engine rms+QKV+RoPE (q/k head dims pair-interleaved,
# gemm_engine.qkv_col_perm) -> FlashAttention reading Q|K|V out of the one wide
# buffer (fused_qkv) -> gemm_engine O+FFN.
ENG_LAYER_ARGS = [
    "x",
    "wqkv",
    "rope",
    "qkv",
    "mask",
    "attn",
    "wo",
    "res1",
    "wgu",
    "sw",
    "wdn",
    "out",
]
QKV_ENGINE_ORDER = ["x", "wqkv", "rope", "qkv"]
_QKV_ENG_MAP = {0: 0, 1: 1, 2: 2, 3: 3}
_FA_FUSED_MAP = {0: 3, 1: 4, 2: 5}
_OFFN_ENG_MAP2 = {0: 5, 1: 6, 2: 0, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11}
ENG_LAYER_STATIC = {1, 2, 4, 6, 8, 10}
ENG_LAYER_INTERMEDIATE = {3, 5, 7, 9, 11}
ENG_LAYER_OUT = 11
ENG_LAYER_QKV = 3


def build_engine_layer_module(qkv_ir, fa_ir, offn_ir, fa_tiling):
    """qkv_ir / offn_ir: gemm_engine modules (no shim tiling); fa_ir: FA built
    with fused_qkv=True, attn_mask=True."""
    parts = (
        ("qe", qkv_ir, _QKV_ENG_MAP, []),
        ("at", fa_ir, _FA_FUSED_MAP, fa_tiling),
        ("of", offn_ir, _OFFN_ENG_MAP2, []),
    )
    types = [None] * len(ENG_LAYER_ARGS)
    for _, ir, amap, _ in parts:
        for op_idx, t in enumerate(_signature_types(ir)):
            c = amap[op_idx]
            assert types[c] in (None, t), f"arg{c}: {types[c]} vs {t}"
            types[c] = t
    assert None not in types, types
    slices = [
        KernelSlice(ir, p, amap, extern_syms=_privates(ir)) for p, ir, amap, _ in parts
    ]
    module = stitch_elf(
        "layer",
        [FuncArg(f"%arg{i}", t) for i, t in enumerate(types)],
        slices,
        debug_dump_path="/tmp/layer_eng_parse_error.mlir",
    )

    from air.ir import DenseI64ArrayAttr

    per_launch = [ts for _, ir, _, ts in parts for _ in range(ir.count("air.launch "))]
    func = next(
        op
        for op in module.body.operations
        if op.operation.name == "func.func"
        and op.attributes["sym_name"].value == "layer"
    )
    launches = [
        op
        for op in func.regions[0].blocks[0].operations
        if op.operation.name == "air.launch"
    ]
    assert len(launches) == len(per_launch), (len(launches), len(per_launch))
    with module.context:
        for op, ts in zip(launches, per_launch):
            if ts:
                op.attributes["air.shim_dma_tile_sizes"] = DenseI64ArrayAttr.get(ts)
    print(
        f"  Engine layer module: {len(launches)} launches, {len(str(module).splitlines())} lines, parsed OK"
    )
    return module


def build_multi_layer_module(rgr_ir, fa_ir, offn_ir, tilings, n_layers):
    single = build_layer_module(rgr_ir, fa_ir, offn_ir, tilings)
    types = _signature_types(str(single))
    n_args = 2 + len(MULTI_SHARED) + n_layers * len(MULTI_PER_LAYER)
    comb = [types[0], types[0]] + [None] * (n_args - 2)
    for i in range(n_layers):
        for a in MULTI_SHARED + MULTI_PER_LAYER:
            comb[multi_layer_arg(i, a)] = types[a]
    base_args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(comb)]

    flat = types[20]
    seq_emb = re.search(r"memref<(\d+)x(\d+)xbf16>", types[0]).groups()
    prelude = "\n".join(
        f"    %x{j}_flat = memref.reinterpret_cast %arg{j} to offset: [0], "
        f"sizes: [{int(seq_emb[0]) * int(seq_emb[1])}], strides: [1] : {types[0]} to {flat}"
        for j in (0, 1)
    )

    parts = (
        ("rg", rgr_ir, _RGR_MAP, "rgr"),
        ("at", fa_ir, _FA_MAP, "fa"),
        ("of", offn_ir, _OFFN_MAP, "offn"),
    )
    slices, per_launch = [], []
    for i in range(n_layers):
        for p, ir, amap, key in parts:
            m, aliases = {}, {}
            for op_idx, layer_arg in amap.items():
                if layer_arg == 0:
                    m[op_idx] = i % 2
                elif layer_arg == 20:
                    aliases[op_idx] = f"%x{(i + 1) % 2}_flat"
                else:
                    m[op_idx] = multi_layer_arg(i, layer_arg)
            slices.append(
                KernelSlice(
                    ir, f"{p}{i}", m, arg_aliases=aliases, extern_syms=_privates(ir)
                )
            )
            per_launch += [tilings[key]] * ir.count("air.launch ")
    module = stitch_elf(
        "layers",
        base_args,
        slices,
        prelude=prelude,
        debug_dump_path="/tmp/layers_fused_parse_error.mlir",
    )

    from air.ir import DenseI64ArrayAttr

    func = next(
        op
        for op in module.body.operations
        if op.operation.name == "func.func"
        and op.attributes["sym_name"].value == "layers"
    )
    launches = [
        op
        for op in func.regions[0].blocks[0].operations
        if op.operation.name == "air.launch"
    ]
    assert len(launches) == len(per_launch), (len(launches), len(per_launch))
    with module.context:
        for op, ts in zip(launches, per_launch):
            op.attributes["air.shim_dma_tile_sizes"] = DenseI64ArrayAttr.get(ts)
    print(
        f"  {n_layers}-layer module: {len(launches)} launches, {n_args} args, parsed OK"
    )
    return module
