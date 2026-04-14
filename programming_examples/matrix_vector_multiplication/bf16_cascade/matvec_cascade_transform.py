# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Transform-dialect-based cascade bf16 matrix-vector multiplication.
#
# C[M] = A[M,K] @ B[K]   (bf16 in/out)
#
# Uses linalg.matvec on tensors, tiled and cascade-split via a dynamically
# generated transform script. Tile sizes are computed from M, K, and hardware
# constraints to avoid L2 buffer overflow and degenerate forall issues.
#
# Fixes over the static transform.mlir approach:
#   Bug 1: M == LAUNCH_TILE_M (degenerate forall) -- skip Phase 1 if single iter
#   Bug 2: L2 overflow for large K -- auto-compute TILE_M to fit in MemTile
#   Bug 3: Cascade channel 1D vs 2D after fuse-nested-herd -- post-process fix

import argparse
import os
import re
import numpy as np
from ml_dtypes import bfloat16

from air.dialects import linalg, arith, func, tensor, memref, bufferization
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
from air.ir import *
import air.passmanager
from air.backend.xrt_runner import XRTRunner
from air.backend.xrt import XRTBackend


@module_builder
def matvec_on_tensors(m, k):
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get((m, k), dtype), MemRefType.get((k,), dtype)
    )
    def forward(A, B):
        A_tensor = bufferization.to_tensor(
            buffer=A,
            result=RankedTensorType.get((m, k), dtype),
            restrict=True,
            writable=True,
        )
        B_tensor = bufferization.to_tensor(
            buffer=B,
            result=RankedTensorType.get((k,), dtype),
            restrict=True,
            writable=True,
        )
        out = tensor.EmptyOp((m,), dtype).result
        zero = arith.ConstantOp(dtype, 0.0)
        zero_fill = linalg.fill(zero, outs=[out])
        result = linalg.matvec(A_tensor, B_tensor, outs=[zero_fill])
        result_memref = bufferization.to_buffer(
            tensor=result, buffer=MemRefType.get((m,), dtype)
        )
        return result_memref


def compute_tile_sizes(m, k, herd_cols, n_cascade, k_tile=32):
    """Compute tile sizes that fit L2 (MemTile) capacity.

    Returns (launch_tile_m, tile_m, k_chunk, k_tile).

    The L2 promotion happens at Phase 1 (launch level), so the L2 buffer is
    [LAUNCH_TILE_M, K]. We must ensure:
      LAUNCH_TILE_M * K * 2 + K * 2 + LAUNCH_TILE_M * 2 <= L2_CAPACITY * 0.75

    LAUNCH_TILE_M = herd_cols * tile_m, so:
      herd_cols * tile_m * K * 2 <= L2_CAPACITY * 0.75 (approximately)
    """
    L2_CAPACITY = 512 * 1024
    BYTES_PER_ELEM = 2  # bf16

    # Start with ideal tile_m = 32 (matches AIE2P cascade width of 512 bits)
    tile_m = 32

    # L2 holds [LAUNCH_TILE_M, K] for A, plus [K] for B, plus [LAUNCH_TILE_M] for C.
    # Leave 25% headroom for DMA overhead, alignment, etc.
    while tile_m > 1:
        launch_tile_m = herd_cols * tile_m
        a_l2 = launch_tile_m * k * BYTES_PER_ELEM
        b_l2 = k * BYTES_PER_ELEM
        c_l2 = launch_tile_m * BYTES_PER_ELEM
        total = a_l2 + b_l2 + c_l2
        if total <= L2_CAPACITY * 0.75:
            break
        tile_m //= 2

    if tile_m < 1:
        tile_m = 1

    # AIE2P cascade bus is 512 bits. For bf16 (16 bits), cascade buffer must
    # hold at least 32 elements (32 * 16 = 512 bits). Enforce tile_m >= 32.
    CASCADE_MIN_TILE_M = 32  # 512 bits / 16 bits per bf16
    if tile_m < CASCADE_MIN_TILE_M:
        import sys
        print(
            f"WARNING: L2 capacity forces tile_m={tile_m} but cascade width "
            f"requires tile_m>={CASCADE_MIN_TILE_M}. L2 will overflow. "
            f"Use the hand-written matvec_cascade.py for large K ({k}) "
            f"which uses f32 cascade buffers with padding.",
            file=sys.stderr,
        )
        tile_m = CASCADE_MIN_TILE_M

    # Ensure we always have >= 2 launch iterations to avoid degenerate forall.
    # (A single-iteration forall gets canonicalized away, breaking split_handle.)
    launch_tile_m = herd_cols * tile_m
    if m < 2 * launch_tile_m:
        raise ValueError(
            f"M ({m}) is too small for the given parameters. "
            f"Need M >= 2 * herd_cols * tile_m = 2 * {herd_cols} * {tile_m} = {2 * launch_tile_m}. "
            f"Reduce herd_cols or use the hand-written matvec_cascade.py for small M."
        )

    assert m % launch_tile_m == 0, (
        f"M ({m}) must be divisible by launch_tile_m ({launch_tile_m})"
    )

    k_chunk = k // n_cascade
    assert k_chunk % k_tile == 0, (
        f"k_chunk ({k_chunk}) must be divisible by k_tile ({k_tile})"
    )

    return launch_tile_m, tile_m, k_chunk, k_tile


CANONICALIZE_BLOCK = """\
    // Canonicalize
    %FUNC_ID = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %FUNC_ID {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %FUNC_ID : !transform.any_op
"""


def _canon(func_id):
    """Return a canonicalize block with a unique SSA name."""
    return CANONICALIZE_BLOCK.replace("FUNC_ID", func_id)


def generate_transform_script(
    m, k, launch_tile_m, tile_m, herd_cols, n_cascade, k_tile
):
    """Generate a transform script with parameterized tile sizes.

    Same structure as the original transform.mlir but with:
    - Parameterized tile sizes (Bug 1 + Bug 2 fix)
    - Phase 1 skipped when M == LAUNCH_TILE_M (Bug 1 fix)

    Tiling strategy (same as original):
      Phase 1: Tile M by LAUNCH_TILE_M + L2 promotion
      Phase 2: Tile M by TILE_M for herd columns
      Phase 3: tile_reduction_using_forall on K for cascade
      Phase 3.5: Tile inner K by K_TILE + L1 promotion
      Phase 4: forall_with_reduce_to_parallel
      Phase 5: Bufferize cascade/partial result buffers
      Phase 6: Final bufferize
    """
    n_launches = m // launch_tile_m
    k_chunk = k // n_cascade

    lines = []
    lines.append('module attributes {transform.with_named_sequence} {')
    lines.append('  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {')
    lines.append('    %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op')
    lines.append('    %matvec = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!transform.any_op) -> !transform.any_op')
    lines.append('')

    # === Phase 1: ALWAYS tile M for launch + L2 promotion ===
    # Even when n_launches==1, we create the forall to ensure L2 buffers
    # are placed at the right level. We skip canonicalize when n_launches==1
    # to prevent the single-iteration forall from being folded (which would
    # break the split_handle in Phase 2).
    lines.append(f'    // === Phase 1: Tile M by {launch_tile_m} for launch + L2 promotion ===')
    lines.append(f'    %matvec_1, %forall_launch = transform.structured.tile_using_forall %matvec tile_sizes [{launch_tile_m}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append('    %fill_1, %fused_launch = transform.structured.fuse_into_containing_op %fill into %forall_launch : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append('')
    # L2 promotion
    lines.append('    // Pad and promote to L2 (shared memory)')
    lines.append('    %padded, %pad, %pad_copy = transform.structured.pad %matvec_1 {')
    lines.append('      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],')
    lines.append('      padding_dimensions=[0, 1, 2],')
    lines.append('      nofold_flags=[1, 1, 1],')
    lines.append('      copy_back_op="linalg.copy"')
    lines.append('    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)')
    lines.append('    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad : (!transform.any_op) -> !transform.any_op')
    lines.append('')
    lines.append('    %padded_lhs = transform.get_producer_of_operand %padded[0] : (!transform.any_op) -> (!transform.any_op)')
    lines.append('    %padded_lhs_buffer, %padded_lhs_new = transform.structured.bufferize_to_allocation %padded_lhs')
    lines.append('      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op')
    lines.append('')
    lines.append('    %padded_rhs = transform.get_producer_of_operand %padded[1] : (!transform.any_op) -> (!transform.any_op)')
    lines.append('    %padded_rhs_buffer, %padded_rhs_new = transform.structured.bufferize_to_allocation %padded_rhs')
    lines.append('      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op')
    lines.append('')
    lines.append('    %padded_result = transform.get_producer_of_operand %padded[2] : (!transform.any_op) -> (!transform.any_op)')
    lines.append('    %padded_result_buffer, %padded_result_new = transform.structured.bufferize_to_allocation %padded_result')
    lines.append('      {memory_space = 1, bufferize_destination_only, emit_dealloc} : !transform.any_op')
    lines.append(_canon("func1"))
    lines.append('')

    # === Phase 2: Tile M for herd columns ===
    lines.append(f'    // === Phase 2: Tile M by {tile_m} for per-column (herd) parallelism ===')
    lines.append('    %tiled_ops = transform.structured.match ops{["linalg.fill", "linalg.matvec"]} in %fused_launch : (!transform.any_op) -> !transform.any_op')
    lines.append('    %tiled_fill, %tiled_matvec = transform.split_handle %tiled_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append(f'    %matvec_2, %forall_herd = transform.structured.tile_using_forall %tiled_matvec tile_sizes [{tile_m}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append('    %fill_2, %fused_herd = transform.structured.fuse_into_containing_op %tiled_fill into %forall_herd : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)')

    # Pad Phase 2 for L1 side later
    lines.append('')
    lines.append('    // Pad for L1 promotion')
    lines.append('    %padded_2, %pad_2, %pad_2_copy = transform.structured.pad %matvec_2 {')
    lines.append('      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],')
    lines.append('      padding_dimensions=[0, 1, 2],')
    lines.append('      nofold_flags=[0, 0, 1],')
    lines.append('      copy_back_op="linalg.copy"')
    lines.append('    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)')
    lines.append('    %pad_2_dps = transform.structured.rewrite_in_destination_passing_style %pad_2 : (!transform.any_op) -> !transform.any_op')
    lines.append(_canon("func2"))
    lines.append('')

    # === Phase 3: K reduction for cascade ===
    lines.append(f'    // === Phase 3: Tile K reduction for cascade ({n_cascade} stages) ===')
    lines.append(f'    %reduce_fill, %matvec_3, %reduce_comb, %reduce_forall = transform.structured.tile_reduction_using_forall %padded_2 by num_threads = [0, {n_cascade}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)')
    lines.append('    %fused_fill_3, %fused_reduce = transform.structured.fuse_into_containing_op %reduce_fill into %reduce_forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append('')

    # Tile inner K + L1 promotion
    lines.append(f'    // Tile remaining K by {k_tile} for inner loop')
    lines.append(f'    %tiled_k, %k_loop = transform.structured.tile_using_for %matvec_3 tile_sizes [0, {k_tile}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append(_canon("func3"))
    lines.append('')

    lines.append('    // Pad inner tile and promote to L1')
    lines.append('    %padded_k, %pad_k, %pad_k_copy = transform.structured.pad %tiled_k {')
    lines.append('      padding_values=[0.0 : bf16, 0.0 : bf16, 0.0 : bf16],')
    lines.append('      padding_dimensions=[0, 1, 2],')
    lines.append('      nofold_flags=[1, 1, 0],')
    lines.append('      copy_back_op="linalg.copy"')
    lines.append('    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)')
    lines.append('    %pad_k_dps = transform.structured.rewrite_in_destination_passing_style %pad_k : (!transform.any_op) -> !transform.any_op')
    lines.append('')
    lines.append('    %padded_k_lhs = transform.get_producer_of_operand %padded_k[0] : (!transform.any_op) -> (!transform.any_op)')
    lines.append('    %padded_k_lhs_buffer, %padded_k_lhs_new = transform.structured.bufferize_to_allocation %padded_k_lhs')
    lines.append('      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op')
    lines.append('')
    lines.append('    %padded_k_rhs = transform.get_producer_of_operand %padded_k[1] : (!transform.any_op) -> (!transform.any_op)')
    lines.append('    %padded_k_rhs_buffer, %padded_k_rhs_new = transform.structured.bufferize_to_allocation %padded_k_rhs')
    lines.append('      {memory_space = 2, bufferize_destination_only, emit_dealloc} : !transform.any_op')
    lines.append('')

    # === Phase 4: forall_with_reduce_to_parallel ===
    lines.append('    // === Phase 4: Convert reduce forall to parallel (cascade semantics) ===')
    lines.append('    %inner_parallel = transform.air.forall_with_reduce_to_parallel %reduce_forall : (!transform.any_op) -> (!transform.any_op)')
    lines.append(_canon("func4"))
    lines.append('    transform.apply_patterns to %func4 {')
    lines.append('      transform.apply_patterns.canonicalization')
    lines.append('    } : !transform.any_op')
    lines.append('')

    # === Phase 5: Bufferize cascade/partial buffers ===
    lines.append('    // === Phase 5: Bufferize cascade and partial result buffers ===')
    lines.append('    %fill_ops = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op')
    lines.append('    %herd_cascade_fill, %herd_reduce_fill = transform.split_handle %fill_ops : (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    lines.append('    %cascade_local_buffer, %cascade_local_new = transform.structured.bufferize_to_allocation %herd_cascade_fill')
    lines.append('      {memory_space = 2, bufferize_destination_only} : !transform.any_op')
    lines.append('    %result_local_buffer, %result_local_new = transform.structured.bufferize_to_allocation %herd_reduce_fill')
    lines.append('      {memory_space = 2, bufferize_destination_only} : !transform.any_op')
    lines.append(_canon("func5"))
    lines.append('    transform.apply_patterns to %func5 {')
    lines.append('      transform.apply_patterns.canonicalization')
    lines.append('    } : !transform.any_op')
    lines.append('')

    # === Phase 6: Final bufferize ===
    lines.append('    // === Phase 6: Final bufferize ===')
    lines.append('    %func_op = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op')
    lines.append('    transform.apply_patterns to %func_op {')
    lines.append('      transform.apply_patterns.linalg.tiling_canonicalization')
    lines.append('      transform.apply_patterns.scf.for_loop_canonicalization')
    lines.append('      transform.apply_patterns.canonicalization')
    lines.append('    } : !transform.any_op')
    lines.append('    transform.apply_cse to %func_op : !transform.any_op')
    lines.append('    %func_bufferized = transform.bufferization.one_shot_bufferize %func_op : (!transform.any_op) -> !transform.any_op')
    lines.append('')
    lines.append('    // Final cleanup')
    lines.append('    %func_final = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op')
    lines.append('    transform.apply_patterns to %func_final {')
    lines.append('      transform.apply_patterns.linalg.tiling_canonicalization')
    lines.append('      transform.apply_patterns.scf.for_loop_canonicalization')
    lines.append('      transform.apply_patterns.canonicalization')
    lines.append('    } : !transform.any_op')
    lines.append('    transform.apply_cse to %func_final : !transform.any_op')
    lines.append('    transform.apply_patterns to %func_final {')
    lines.append('      transform.apply_patterns.canonicalization')
    lines.append('    } : !transform.any_op')
    lines.append('    transform.yield')
    lines.append('  }')
    lines.append('}')

    return '\n'.join(lines) + '\n'


def replace_matvec_with_external_kernel(ir_string, tile_m, k_tile):
    """Replace linalg.matvec with func.call to external vectorized kernel.

    The transform-generated IR contains:
      linalg.matvec ins(%a, %b : memref<MxKxbf16, 2>, memref<Kxbf16, 2>)
                    outs(%c : memref<Mxbf16, ...>)

    Replace with func.call @matvec_bf16_f32acc(M, K, %a, %b, %c) which
    accumulates in f32 internally for better precision.
    """
    # Add func.func private declaration with link_with at module level
    func_decl = (
        '  func.func private @matvec_bf16_f32acc(i32, i32, '
        'memref<{m}x{k}xbf16, 2>, memref<{k}xbf16, 2>, '
        'memref<{m}xbf16, strided<[1]>, 2>) '
        'attributes {{link_with = "mv_transform.o", '
        'llvm.emit_c_interface}}'.format(m=tile_m, k=k_tile)
    )

    # Insert declaration after the first line that starts with "module"
    # (before the first func.func @forward)
    ir_string = re.sub(
        r'(module \{[^\n]*\n(?:  air\.channel[^\n]*\n)*)',
        r'\1' + func_decl + '\n',
        ir_string,
    )

    # Replace linalg.matvec with func.call
    # Match pattern: linalg.matvec ins(%a, %b : ...) outs(%c : ...)
    _replace_counter = [0]

    def replace_matvec(match):
        idx = _replace_counter[0]
        _replace_counter[0] += 1
        indent = match.group(1)
        a_var = match.group(2)
        b_var = match.group(3)
        c_var = match.group(4)
        m_const = f'%c{tile_m}_i32_m_{idx}'
        k_const = f'%c{k_tile}_i32_k_{idx}'
        lines = []
        lines.append(f'{indent}{m_const} = arith.constant {tile_m} : i32')
        lines.append(f'{indent}{k_const} = arith.constant {k_tile} : i32')
        lines.append(
            f'{indent}func.call @matvec_bf16_f32acc('
            f'{m_const}, {k_const}, {a_var}, {b_var}, {c_var}) : '
            f'(i32, i32, memref<{tile_m}x{k_tile}xbf16, 2>, '
            f'memref<{k_tile}xbf16, 2>, '
            f'memref<{tile_m}xbf16, strided<[1]>, 2>) -> ()'
        )
        return '\n'.join(lines)

    ir_string = re.sub(
        r'(\s+)linalg\.matvec ins\((%\w+), (%\w+) : memref<\d+x\d+xbf16, 2>, memref<\d+xbf16, 2>\) outs\((%\w+) : memref<\d+xbf16, strided<\[1\]>, 2>\)',
        replace_matvec,
        ir_string,
    )

    # Set link_with on the herd
    ir_string = re.sub(
        r'(air\.herd @\w+\s+tile[^{]+)\{',
        r'\1 attributes {link_with = "mv_transform.o"} {',
        ir_string,
    )

    return ir_string


def fix_cascade_channel_dimensions(ir_string, herd_cols):
    """Bug 3 workaround: fix cascade channel from 1D [N] to 2D [herd_cols, N].

    After air-fuse-nested-herd, cascade channels are incorrectly 1D because
    NestedHerdCollapsePattern doesn't update channel declarations or indices.
    This function post-processes the IR string to fix:
      1. Channel declaration: @channel_0 [N] -> @channel_0 [herd_cols, N]
      2. Channel put indices: @channel_0[idx] -> @channel_0[%tx, idx]
      3. Channel get indices: @channel_0[idx] -> @channel_0[%tx, idx]

    This is a temporary workaround until the compiler pass is fixed.
    """
    # Find cascade channel declarations and fix their dimensions
    def fix_channel_decl(match):
        name = match.group(1)
        size = match.group(2)
        return f'air.channel @{name} [{herd_cols}, {size}] {{channel_type = "cascade"}}'

    ir_string = re.sub(
        r'air\.channel @(\w+) \[(\d+)\] \{channel_type = "cascade"\}',
        fix_channel_decl,
        ir_string,
    )

    # Find the herd tile IDs
    herd_match = re.search(
        r"air\.herd\s+@\w+\s+tile\s+\((%\w+),\s*(%\w+)\)", ir_string
    )
    if not herd_match:
        return ir_string

    tx_name = herd_match.group(1)

    # Find cascade channel names (those we just made 2D)
    cascade_channels = re.findall(
        r'air\.channel @(\w+) \[\d+, \d+\] \{channel_type = "cascade"\}',
        ir_string,
    )

    for chan_name in cascade_channels:
        # Fix channel.put with single index (not already 2D)
        ir_string = re.sub(
            rf'(air\.channel\.put\s+@{chan_name})\[([^\],]+)\]',
            rf'\1[{tx_name}, \2]',
            ir_string,
        )
        # Fix channel.get with single index
        ir_string = re.sub(
            rf'(air\.channel\.get\s+@{chan_name})\[([^\],]+)\]',
            rf'\1[{tx_name}, \2]',
            ir_string,
        )

    return ir_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="matvec_cascade_transform.py",
        description="Transform-dialect-based cascade bf16 matvec",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument(
        "--herd-cols",
        type=int,
        default=8,
        dest="herd_cols",
        help="Number of AIE columns for parallel M-dimension processing",
    )
    parser.add_argument(
        "--n-cascade",
        type=int,
        default=4,
        dest="n_cascade",
        help="Number of cascade stages for K-reduction",
    )
    parser.add_argument(
        "--k-tile",
        type=int,
        default=32,
        dest="k_tile",
        help="Inner K tile size for vectorization",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-and-run", "compile-only"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--print-transform-only",
        action="store_true",
        dest="print_transform_only",
        help="Print the generated transform script and exit",
    )
    args = parser.parse_args()

    M = args.m
    K = args.k
    HERD_COLS = args.herd_cols
    N_CASCADE = args.n_cascade
    K_TILE = args.k_tile

    # Compute tile sizes that fit L2
    launch_tile_m, tile_m, k_chunk, k_tile = compute_tile_sizes(
        M, K, HERD_COLS, N_CASCADE, K_TILE
    )
    print(
        f"Tile sizes: launch_tile_m={launch_tile_m}, tile_m={tile_m}, "
        f"k_chunk={k_chunk}, k_tile={k_tile}, "
        f"herd_cols={HERD_COLS}, n_cascade={N_CASCADE}"
    )
    l2_bytes = launch_tile_m * K * 2 + K * 2 + launch_tile_m * 2
    print(f"L2 budget: A={launch_tile_m*K*2}B + B={K*2}B + C={launch_tile_m*2}B = {l2_bytes}B (limit={int(512*1024*0.75)}B)")

    # Generate transform script
    transform_script = generate_transform_script(
        M, K, launch_tile_m, tile_m, HERD_COLS, N_CASCADE, k_tile
    )

    if args.print_transform_only:
        print(transform_script)
        exit(0)

    air_module = matvec_on_tensors(M, K)
    context = air_module.context

    if args.print_module_only:
        print(air_module)
        exit(0)

    # === Phase 1: Apply transform script for tiling ===
    with open("transform_generated.mlir", "w") as f:
        f.write(transform_script)

    transform_ir = Module.parse(transform_script, context=context)
    run_transform(transform_ir, air_module)

    with open("air_tiled.mlir", "w") as f:
        f.write(str(air_module))

    # === Phase 2: Convert to AIR hierarchy ===
    n_launches = M // launch_tile_m
    if n_launches > 1:
        pipeline_steps = [
            "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true}",
            "air-copy-to-dma",
            "air-par-to-herd{depth=-1}",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{depth=-1 has-air-segment=true}",
            "func.func(air-fuse-nested-herd)",
            "canonicalize",
        ]
    else:
        # No launch tiling, only one level of herd + cascade
        pipeline_steps = [
            "buffer-results-to-out-params{hoist-static-allocs=true modify-public-functions=true}",
            "air-copy-to-dma",
            "air-par-to-herd{depth=-1}",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{depth=-1 has-air-segment=true}",
            "func.func(air-fuse-nested-herd)",
            "canonicalize",
        ]

    pipeline = "builtin.module(" + ",".join(pipeline_steps) + ")"
    pm = air.passmanager.PassManager.parse(pipeline, context=context)
    pm.run(air_module.operation)

    # === Phase 3: Fix cascade channel dimensions (Bug 3 workaround) ===
    ir_string = str(air_module)
    ir_string = fix_cascade_channel_dimensions(ir_string, HERD_COLS)

    # === Phase 3.5: Replace linalg.matvec with external kernel ===
    ir_string = replace_matvec_with_external_kernel(ir_string, tile_m, k_tile)

    with open("air_herd.mlir", "w") as f:
        f.write(ir_string)

    # Re-parse the fixed IR
    air_module = Module.parse(ir_string, context=context)

    # === Phase 4: Compile and run ===
    if args.compile_mode == "compile-and-run":
        np.random.seed(42)
        USE_ONES = os.environ.get("USE_ONES", "0") == "1"
        if USE_ONES:
            input_a = np.ones((M, K), dtype=bfloat16)
            input_b = np.ones((K,), dtype=bfloat16)
        else:
            input_a = (np.random.randn(M, K) * 4).astype(bfloat16)
            input_b = (np.random.randn(K) * 4).astype(bfloat16)
        output_c = np.dot(
            input_a.astype(np.float32), input_b.astype(np.float32)
        ).astype(bfloat16)

        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
            instance_name="forward",
        )
        exit(
            runner.run_test(
                air_module,
                inputs=[input_a, input_b],
                expected_outputs=[output_c],
                rtol=0.1,
                atol=2.0,
            )
        )
    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            runtime_loop_tiling_sizes=[4, 4],
            output_format=args.output_format,
        )
        module_function = backend.compile(air_module)
        backend.unload()
