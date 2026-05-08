# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Text-based MLIR stitching utilities for multi-launch kernel assembly.

Provides functions to extract, rename, and combine MLIR text fragments from
individually-built kernel modules into a single multi-launch module. These
utilities handle SSA value renaming, affine map prefixing, func-arg fixup,
and air.launch/segment wrapping for bare herds.
"""

import re


def _extract_between_func_and_return(mlir_text):
    """Extract func body (between func signature and return)."""
    lines = mlir_text.split("\n")
    body_start = body_end = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            body_start = i + 1
    # Match the trailing `return` even if a future MLIR printer attaches
    # operands, attributes, location, or trailing comments
    # (e.g. `return`, `return %0`, `return loc(...)`, `return // comment`).
    return_re = re.compile(r"^\s*return(\s|$|//|loc\()")
    for i in range(len(lines) - 1, body_start, -1):
        if return_re.match(lines[i]):
            body_end = i
            break
    return "\n".join(lines[body_start:body_end])


def _extract_affine_maps(mlir_text):
    return [l for l in mlir_text.split("\n") if l.startswith("#map")]


def _extract_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


_DEFAULT_EXTERN_FUNCS = {
    "@silu_and_mul_bf16",
    "@zero_vectorized_bf16",
    "@matmul_bf16",
}


def _rename_all(text, prefix):
    """Rename SSA values, affine maps, and symbols with a unique prefix.

    External kernel symbols (those in `_DEFAULT_EXTERN_FUNCS`) are preserved
    so they can be linked across stitched modules.
    """
    return _rename_all_with_externs(text, prefix, _DEFAULT_EXTERN_FUNCS)


def _fix_launch_func_args(text, prefix, arg_map):
    """Fix func-arg references in launch's args() clause after _rename_all."""
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    return text


def _wrap_ir_in_launch(mlir_text):
    """Wrap a module whose func body contains bare herds in air.launch+segment.

    Transforms:
        func.func @name(%arg0: T0, %arg1: T1, %arg2: T2) {
            <body with bare air.herd>
            return
        }
    Into:
        func.func @name(%arg0: T0, %arg1: T1, %arg2: T2) {
            air.launch () in () args(%argL0=%arg0, ...) : T0, ... {
                air.segment @name_seg args(%argS0=%argL0, ...) : T0, ... {
                    <body with herd refs remapped to %argS0, ...>
                }
            }
            return
        }

    Both launch AND segment wrappers are needed. Without air.segment, the
    lowered IR uses airrt.herd_load instead of airrt.segment_load. The
    airrt-to-npu pass's identifyLaunchRegions only looks for segment_load ops
    to associate launch regions with aie.device ops. A bare herd_load region
    gets silently dropped when other segment-based launches exist in the same
    function, causing "failed to legalize airrt.dma_memcpy_nd".
    """
    lines = mlir_text.split("\n")

    # Find the public func signature
    func_line_idx = None
    for i, line in enumerate(lines):
        if "func.func @" in line and "private" not in line:
            func_line_idx = i
            break
    if func_line_idx is None:
        return mlir_text

    func_line = lines[func_line_idx]

    # Check if body already has air.launch (skip wrapping)
    body_text = "\n".join(lines[func_line_idx + 1 :])
    if "air.launch" in body_text:
        return mlir_text

    # Extract the func name for generating segment name
    func_name_match = re.search(r"func\.func @(\w+)", func_line)
    func_name = func_name_match.group(1) if func_name_match else "wrapped"

    # Parse func args: extract (%arg0: type0, %arg1: type1, ...)
    sig_match = re.search(r"func\.func @\w+\(([^)]*)\)", func_line)
    if not sig_match:
        return mlir_text

    args_str = sig_match.group(1)
    func_args = []
    for arg in args_str.split(","):
        arg = arg.strip()
        if not arg:
            continue
        parts = arg.split(":")
        name = parts[0].strip()
        typ = ":".join(parts[1:]).strip()
        func_args.append((name, typ))

    n_args = len(func_args)

    # Find the body (between func line and return)
    body_start = func_line_idx + 1
    body_end = None
    for i in range(len(lines) - 1, body_start, -1):
        if lines[i].strip() == "return":
            body_end = i
            break

    body_lines = lines[body_start:body_end]
    body_text = "\n".join(body_lines)

    # Find the max %argN index used in the body to avoid conflicts.
    existing_args = [int(m) for m in re.findall(r"%arg(\d+)", body_text)]
    max_existing = max(existing_args) if existing_args else n_args - 1
    launch_arg_start = max_existing + 1
    segment_arg_start = launch_arg_start + n_args

    # Build launch args clause
    launch_args = ", ".join(
        f"%arg{launch_arg_start + i}={func_args[i][0]}" for i in range(n_args)
    )
    launch_types = ", ".join(func_args[i][1] for i in range(n_args))

    # Build segment args clause (segment args reference launch args)
    segment_args = ", ".join(
        f"%arg{segment_arg_start + i}=%arg{launch_arg_start + i}" for i in range(n_args)
    )
    segment_types = launch_types

    # In the body, remap func arg references to segment arg references.
    for i in range(n_args - 1, -1, -1):
        old_name = func_args[i][0]
        new_name = f"%arg{segment_arg_start + i}"
        body_text = re.sub(re.escape(old_name) + r"(?!\w)", new_name, body_text)

    # Reconstruct the module with both launch and segment wrappers
    new_lines = lines[:body_start]
    new_lines.append(f"    air.launch () in () args({launch_args}) : {launch_types} {{")
    new_lines.append(
        f"      air.segment @{func_name}_seg args({segment_args})"
        f" : {segment_types} {{"
    )
    for line in body_text.split("\n"):
        new_lines.append("    " + line)
    new_lines.append("      }")
    new_lines.append("    }")
    new_lines.extend(lines[body_end:])

    return "\n".join(new_lines)


def _rename_all_with_externs(text, prefix, extern_funcs):
    """Like _rename_all but with a configurable extern_funcs set."""
    # Affine maps (longest first)
    for name in sorted(set(re.findall(r"#map\d*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"#{prefix}_{name[1:]}", text)

    # SSA word values
    for name in sorted(set(re.findall(r"%[a-zA-Z_]\w*", text)), key=len, reverse=True):
        text = re.sub(re.escape(name) + r"(?!\w)", f"%{prefix}_{name[1:]}", text)

    # SSA numbered values — re.sub with `(?!\d)` boundary so `%10` cannot
    # substring-match `%100`. Longest-first ordering is no longer required for
    # correctness but kept for determinism.
    for name in sorted(
        set(re.findall(r"%\d+", text)), key=lambda x: int(x[1:]), reverse=True
    ):
        text = re.sub(re.escape(name) + r"(?!\d)", f"%{prefix}_n{name[1:]}", text)

    # Symbol names but NOT extern functions
    for name in sorted(set(re.findall(r"@[\w]+", text)), key=len, reverse=True):
        if name not in extern_funcs:
            text = text.replace(name, f"@{prefix}_{name[1:]}")

    return text
