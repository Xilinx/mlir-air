# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""External C++ kernel compilation utilities.

Compiles all external .o files from source to avoid relying on stale
pre-compiled artifacts. Each function checks if the .o exists and skips
recompilation if so (delete the .o to force recompile).

Compiled .o files are placed in CWD (build_peano/) where aiecc finds them
via its link_with search path.
"""

import os
import shutil
import subprocess
from pathlib import Path


def _get_peano_clang():
    """Find the Peano clang++ compiler."""
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        return os.path.join(peano_dir, "bin", "clang++")
    raise RuntimeError("PEANO_INSTALL_DIR not set")


def _get_aie_include_dir():
    """Find the AIE API include directory (for aie_api/aie.hpp)."""
    # Primary: locate via aie-opt on PATH. Matches the convention used by
    # every other Makefile in this repo (AIEOPT_DIR = $(dir $(which aie-opt))/..)
    # and works for both local source builds and CI's mlir_aie wheel install.
    aie_opt = shutil.which("aie-opt")
    if aie_opt:
        p = Path(aie_opt).resolve().parent.parent / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Explicit override: MLIR_AIE_INSTALL_DIR env var (useful in git worktrees
    # where the local-dev relative path below resolves to the worktree root
    # rather than the main repo root).
    mlir_aie_dir = os.environ.get("MLIR_AIE_INSTALL_DIR", "")
    if mlir_aie_dir:
        p = Path(mlir_aie_dir) / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Fallback: explicit local dev install path.
    p = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / "my_install"
        / "mlir-aie"
        / "install"
        / "include"
    )
    if (p / "aie_api" / "aie.hpp").exists():
        return str(p)
    raise RuntimeError(
        "Cannot find aie_api/aie.hpp include directory "
        "(no aie-opt on PATH, no MLIR_AIE_INSTALL_DIR, no my_install/mlir-aie/install)"
    )


_PEANO_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-DNDEBUG",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-empty-body",
]


def _compile_kernel(src_path, output_name, extra_flags=None, force=False):
    """Compile a C++ kernel to .o using Peano clang++.

    Args:
        src_path: Path to the .cc source file
        output_name: Name of the output .o file (placed in CWD)
        extra_flags: Additional compiler flags (e.g., -D defines)
        force: If True, recompile even if .o exists
    """
    if not force and Path(output_name).exists():
        return

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Kernel source not found: {src}")

    clang = _get_peano_clang()
    include_dir = _get_aie_include_dir()

    cmd = [clang] + _PEANO_FLAGS + [f"-I{include_dir}"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd.extend(["-c", str(src), "-o", output_name])

    print(f"  Compiling {output_name} from {src.name}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Filter warnings, only show errors
        errors = [l for l in result.stderr.split("\n") if "error" in l.lower()]
        raise RuntimeError(f"Failed to compile {output_name}: {' '.join(errors[:3])}")


# ---------------------------------------------------------------------------
# Individual kernel compilation functions
# ---------------------------------------------------------------------------

_PROJ_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
)  # programming_examples/


def compile_silu_and_mul():
    """Compile silu_and_mul.o from programming_examples/silu_and_mul/silu_and_mul.cc."""
    src = _PROJ_ROOT / "silu_and_mul" / "silu_and_mul.cc"
    include_dir = _get_aie_include_dir()
    utils_header = Path(include_dir) / "aie_kernels" / "aie_kernel_utils.h"
    extra = []
    if utils_header.exists():
        extra = [f"-include", str(utils_header)]
    _compile_kernel(src, "silu_and_mul.o", extra_flags=extra)


def compile_gemm_mm(
    tile_m=64, tile_n=128, tile_k_l1=32, sym_suffix="", out_name="mm.o"
):
    """Compile mm.o from matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc.

    The hand-tuned Peano -O2 vectorized GEMM microkernel (external path), ~1.5-1.65x
    faster than direct-codegen on large shapes (kernel_registry/details/GEMM_bf16_in_fp32_out.md).
    DIM_M/DIM_N/DIM_K are baked in at compile time and MUST match the tile_m/tile_n/
    tile_k_l1 passed to the GEMM module builder. Exposes op_has_no_registered_library_name
    (f32-C matmul), zero_f32_mn, f32_to_bf16_mn.

    sym_suffix / out_name: to link TWO mm.o variants (e.g. tile_m=32 drain +
    tile_m=64 fused-cast) into ONE ELF, the symbols must not collide. Pass
    sym_suffix="_m64" (-> @op_has_no_registered_library_name_m64 etc.) and a
    distinct out_name="mm_m64.o". Default empty suffix / "mm.o" keeps the original
    names for single-variant ELFs (back-compat).
    """
    src = _PROJ_ROOT / "matrix_multiplication" / "bf16_in_fp32_out" / "mm_aie2p.cc"
    extra = [
        "-DBIT_WIDTH=8",
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DDIM_N_DIV_4={tile_n // 4}",
        f"-DDIM_M_DIV_4={tile_m // 4}",
        f"-DDIM_N_DIV_8={tile_n // 8}",
        f"-DDIM_M_DIV_8={tile_m // 8}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    if sym_suffix:
        extra.append(f"-DSYM_SUFFIX={sym_suffix}")
    _compile_kernel(src, out_name, extra_flags=extra, force=True)


def compile_rope():
    """Compile rope.o from programming_examples/rope_halfsplit/rope_halfsplit.cc.

    Uses rope_halfsplit.cc (half-split rotation matching HuggingFace Llama)
    instead of upstream rope.cc (interleaved rotation). Same function name
    (@rope) and signature, so no MLIR changes needed. The kernel lives in the
    standalone rope_halfsplit registry example; llama links the same source.
    """
    src = _PROJ_ROOT / "rope_halfsplit" / "rope_halfsplit.cc"
    _compile_kernel(src, "rope.o")


def compile_attn_npu2(head_dim=64, lkp=None, lqp_tile=None, force=False):
    """Compile attn_npu2.o (FlashAttention kernel) from source.

    The attn_npu2.cc defines are PER-TILE, not per-launch (see the canonical
    Makefile): ``lqp`` = tile_size_q (= lqp_launch / num_q_tiles), ``lkp`` =
    K/V chunk size per tile, ``dk``/``dv`` = the K/V dimension TILE (= lkp),
    and ``dk_full``/``dv_full`` = the full head_dim. The matmul microkernels
    are instantiated with these tile shapes, so they MUST match the L1 buffer
    shapes the Python builder emits or the kernel hangs (ERT_CMD_STATE_TIMEOUT).

    head_dim=64 (llama32_1b seq-first): lkp == head_dim, so the legacy
    "everything = head_dim" defaults are correct.

    head_dim=128 (head-first path): the kernel tiles dk/dv into dv_chunks=2
    slices of lkp=64, and tile_size_q=64 (lqp_launch=256 / num_q_tiles=4). So
    pass lkp=64, lqp_tile=64; dk_full/dv_full stay at head_dim (128).

    Args:
        head_dim: full head dimension (-> dk_full / dv_full).
        lkp: K/V chunk size per tile (= dk/dv tile). Defaults to head_dim
            (legacy hd==lkp behavior).
        lqp_tile: Q tile size (tile_size_q). Defaults to lkp.
        force: recompile even if attn_npu2.o exists (needed when the same CWD
            previously built a different-shaped .o, e.g. hd=64 then hd=128).
    """
    if lkp is None:
        lkp = head_dim
    if lqp_tile is None:
        lqp_tile = lkp
    src = _PROJ_ROOT / "flash_attention" / "kernel_fusion_based" / "attn_npu2.cc"
    _compile_kernel(
        src,
        "attn_npu2.o",
        extra_flags=[
            "-DBIT_WIDTH=8",
            f"-Dlqp={lqp_tile}",
            f"-Dlkp={lkp}",
            f"-Ddk={lkp}",
            f"-Ddk_full={head_dim}",
            f"-Ddv={lkp}",
            f"-Ddv_full={head_dim}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DROUND_CONV_EVEN",
        ],
        force=force,
    )
    # Also create attn.o copy (some link_with attributes use "attn.o").
    # Refresh whenever attn_npu2.o exists so a force-rebuild (different tile
    # shape) doesn't leave a stale attn.o behind.
    if Path("attn_npu2.o").exists():
        shutil.copy2("attn_npu2.o", "attn.o")


def compile_mv(tile_m=8):
    """Compile mv.o (standard GEMV kernel) from source."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(src, "mv.o", extra_flags=[f"-DDIM_M_OUTPUT={tile_m}"])


def compile_mv_int4_bf16(m_tile=8, k_chunk=2048, gs=128):
    """Compile mv_int4_bf16.o (int4-AWQ GEMV micro-kernel) from source.

    Produces `mv_int4_bf16_gemv.o` (config-tagged) and stages it as the
    canonical `mv_int4_bf16.o` (the name link_with attributes expect).
    The int4 GEMM prefill compiles the same .cc with DIM_M=16 to a
    different config-tagged name (`mv_int4_bf16_matmul.o`), so the two
    variants don't clobber each other in CWD across sessions; the
    last-staged canonical .o is whichever variant the current compile
    needs.
    """
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "int4_awq" / "mv_int4_bf16.cc"
    _compile_kernel(
        src,
        "mv_int4_bf16_gemv.o",
        extra_flags=[
            f"-DDIM_M={m_tile}",
            f"-DDIM_K={k_chunk}",
            f"-DDIM_GS={gs}",
        ],
    )
    shutil.copy2("mv_int4_bf16_gemv.o", "mv_int4_bf16.o")


def compile_mv_bf16():
    """Compile mv_bf16.o for the 2-tile matvec+add primitive used by
    o_gemv_ffn stages 1 and 3."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16_cascade" / "mv_bf16.cc"
    _compile_kernel(src, "mv_bf16.o")


def compile_attn_decode_npu2(head_dim=64):
    """Compile attn_decode_npu2.o (RoPE helpers for the fused decode kernel)."""
    src = _PROJ_ROOT / "attention_decode" / "attn_decode_npu2.cc"
    _compile_kernel(
        src,
        "attn_decode_npu2.o",
        extra_flags=[
            f"-DDIM_N={head_dim}",
            f"-DHEAD_SIZE={head_dim}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
    )


def compile_all_external_kernels(head_dim=64, quant="bf16"):
    """Compile all external C++ kernels from source.

    Call this before kernel compilation to ensure all .o files are fresh.
    Each kernel is only compiled if its .o doesn't already exist.
    Delete build_peano/*.o to force recompilation.

    Args:
        head_dim: attention head dimension (RoPE / attn kernel macros).
        quant: "bf16" (default) or "awq". When "awq" the int4-AWQ GEMV
            micro-kernel (`mv_int4_bf16.o`) is also built so the int4
            decode ELFs can link it. bf16 GEMV objects are still built
            so mixed paths (e.g. bf16 prefill + int4 decode) keep working.
    """
    compile_silu_and_mul()
    compile_rope()
    compile_attn_npu2(head_dim=head_dim)
    compile_attn_decode_npu2(head_dim=head_dim)
    compile_mv()
    compile_mv_bf16()
    if quant == "awq":
        compile_mv_int4_bf16()
