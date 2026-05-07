"""External C++ kernel compilation utilities.

Compiles all external .o files from source to avoid relying on stale
pre-compiled artifacts. Each function checks if the .o exists and skips
recompilation if so (delete the .o to force recompile).

Compiled .o files are placed in CWD (build_peano/) where aiecc finds them
via its link_with search path.
"""

import os
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
    # Try mlir-aie install path
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent
        / "my_install"
        / "mlir-aie"
        / "install"
        / "include",
    ]
    for p in candidates:
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    # Fallback: search from PEANO_INSTALL_DIR
    peano_dir = os.environ.get("PEANO_INSTALL_DIR", "")
    if peano_dir:
        p = Path(peano_dir).parent.parent / "include"
        if (p / "aie_api" / "aie.hpp").exists():
            return str(p)
    raise RuntimeError("Cannot find aie_api/aie.hpp include directory")


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

_PROJ_ROOT = Path(__file__).resolve().parent.parent.parent  # programming_examples/


def compile_silu_and_mul():
    """Compile silu_and_mul.o from kernel_builder/ffn_swiglu/silu_and_mul.cc."""
    src = (
        _PROJ_ROOT / "llama32_1b" / "kernel_builder" / "ffn_swiglu" / "silu_and_mul.cc"
    )
    include_dir = _get_aie_include_dir()
    utils_header = Path(include_dir) / "aie_kernels" / "aie_kernel_utils.h"
    extra = []
    if utils_header.exists():
        extra = [f"-include", str(utils_header)]
    _compile_kernel(src, "silu_and_mul.o", extra_flags=extra)


def compile_rope():
    """Compile rope.o from our half-split RoPE kernel.

    Uses rope_halfsplit.cc (half-split rotation matching HuggingFace Llama)
    instead of upstream rope.cc (interleaved rotation). Same function name
    (@rope) and signature, so no MLIR changes needed.
    """
    src = Path(__file__).resolve().parent / "rope_halfsplit.cc"
    _compile_kernel(src, "rope.o")


def compile_attn_npu2(head_dim=64):
    """Compile attn_npu2.o (FlashAttention kernel) from source."""
    src = _PROJ_ROOT / "flash_attention" / "kernel_fusion_based" / "attn_npu2.cc"
    _compile_kernel(
        src,
        "attn_npu2.o",
        extra_flags=[
            "-DBIT_WIDTH=8",
            f"-Dlqp={head_dim}",
            f"-Dlkp={head_dim}",
            f"-Ddk={head_dim}",
            f"-Ddk_full={head_dim}",
            f"-Ddv={head_dim}",
            f"-Ddv_full={head_dim}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DROUND_CONV_EVEN",
        ],
    )
    # Also create attn.o symlink/copy (some link_with attributes use "attn.o")
    if not Path("attn.o").exists() and Path("attn_npu2.o").exists():
        import shutil

        shutil.copy2("attn_npu2.o", "attn.o")


def compile_mv_k8192():
    """Compile mv_k8192.o with renamed GEMV symbols for K=8192 decode merge."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(
        src,
        "mv_k8192.o",
        extra_flags=[
            "-DDIM_M_OUTPUT=2",
            "-Dmatvec_vectorized_bf16_bf16=dg_matvec_vectorized_bf16_bf16",
            "-Dlinalg_fill_bf16=dg_linalg_fill_bf16",
        ],
    )


def compile_mv(tile_m=8):
    """Compile mv.o (standard GEMV kernel) from source."""
    src = _PROJ_ROOT / "matrix_vector_multiplication" / "bf16" / "mv.cc"
    _compile_kernel(src, "mv.o", extra_flags=[f"-DDIM_M_OUTPUT={tile_m}"])


def compile_all_external_kernels(head_dim=64):
    """Compile all external C++ kernels from source.

    Call this before kernel compilation to ensure all .o files are fresh.
    Each kernel is only compiled if its .o doesn't already exist.
    Delete build_peano/*.o to force recompilation.
    """
    compile_silu_and_mul()
    compile_rope()
    compile_attn_npu2(head_dim=head_dim)
    compile_mv()
    compile_mv_k8192()
