# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Text-based MLIR stitching utilities for multi-launch kernel assembly.

Provides functions to extract, rename, and combine MLIR text fragments from
individually-built kernel modules into a single multi-launch module. These
utilities handle SSA value renaming, affine map prefixing, func-arg fixup,
and air.launch/segment wrapping for bare herds.

The low-level primitives (`_extract_*`, `_rename_*`, `_fix_launch_func_args`,
`_wrap_ir_in_launch`) do the mechanical text surgery. `stitch_elf()` is the
declarative orchestration layer on top: given the combined func signature and a
list of `KernelSlice`s (each = one sub-kernel IR + its prefix + arg wiring), it
runs the extract/rename/fixup loop, assembles the module, and parses it. It
folds the three historically error-prone concerns into explicit, validated
inputs: arg-number wiring (`KernelSlice.arg_map`), extern de-dup (per-slice
`extern_syms`), and registry-driven tail scratch args (`alloc_gemm_scratch`).
"""

from dataclasses import dataclass, field

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
    """Top-level affine attribute decls: `#map...` and `#set...` lines."""
    return [
        l for l in mlir_text.split("\n") if l.startswith("#map") or l.startswith("#set")
    ]


def _extract_private_funcs(mlir_text):
    return [l for l in mlir_text.split("\n") if "func.func private" in l]


def _extract_channel_decls(mlir_text):
    """Extract module-level `air.channel @name ...` declaration lines."""
    return [l for l in mlir_text.split("\n") if re.match(r"\s*air\.channel @", l)]


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


def _fix_launch_func_args(text, prefix, arg_map, arg_aliases=None):
    """Fix func-arg references in launch's args() clause after _rename_all.

    arg_map: {orig_idx: combined_idx} — map per-launch %{prefix}_argN to outer
        %argM of the combined func.
    arg_aliases: {orig_idx: "%some_ssa_name"} — map per-launch %{prefix}_argN
        to an arbitrary SSA value defined in the combined func body (e.g. a
        subview/cast result emitted at the top of the func). Use to alias
        multiple launches onto a shared sub-region of a packed buffer without
        burning an extra func arg.
    """
    for orig_idx, combined_idx in arg_map.items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        new_ref = f"%arg{combined_idx}"
        text = text.replace(f"={old_ref},", f"={new_ref},")
        text = text.replace(f"={old_ref})", f"={new_ref})")
    for orig_idx, ssa_name in (arg_aliases or {}).items():
        old_ref = f"%{prefix}_arg{orig_idx}"
        text = text.replace(f"={old_ref},", f"={ssa_name},")
        text = text.replace(f"={old_ref})", f"={ssa_name})")
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
    # Affine attribute symbols: `#map...` and `#set...` (longest first).
    affine_names = set(re.findall(r"#map\d*", text)) | set(re.findall(r"#set\d*", text))
    for name in sorted(affine_names, key=len, reverse=True):
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


# ---------------------------------------------------------------------------
# Declarative ELF assembly: stitch_elf()
# ---------------------------------------------------------------------------


@dataclass
class FuncArg:
    """One argument of the combined func signature.

    `name` is the MLIR SSA name (e.g. "%arg0"); `type` is the memref type
    string (e.g. "memref<2048x2048xbf16>"). Emitted verbatim as
    "{name}: {type}".
    """

    name: str
    type: str


@dataclass
class KernelSlice:
    """One sub-kernel to splice into the combined func.

    ir: the sub-kernel module text. May itself contain multiple air.launch ops
        (e.g. a fused-cast GEMM = GEMM launch + cast launch); the whole func
        body is copied, so launch count is transparent to stitching.
    prefix: unique SSA/affine/symbol prefix (e.g. "q") to avoid collisions.
    arg_map: {launch-operand-idx: combined-func-arg-idx}. The data-flow wiring
        — which combined arg feeds each operand of this slice's launch(es).
    arg_aliases: {launch-operand-idx: "%ssa_name"} routes an operand onto an
        SSA value defined in `prelude` (e.g. a subview) instead of a func arg.
    extern_syms: symbols this slice references that must NOT be prefix-renamed
        (external .o kernel entry points), so they link across slices.
    private_from: collect this slice's `func.func private` decls into the
        module header.
    """

    ir: str
    prefix: str
    arg_map: dict
    arg_aliases: dict = field(default_factory=dict)
    extern_syms: set = field(default_factory=set)
    private_from: bool = True


def alloc_gemm_scratch(specs_in_order, base_arg_count):
    """Registry-driven f32 C-scratch allocation for fused-cast GEMMs.

    `specs_in_order`: list of (spec, out_rows, out_cols) in the SAME order the
    GEMM slices appear, where `spec` is a gemm_method_spec dict (has
    `needs_f32_scratch`). For each fused-cast GEMM, allocate one tail func arg
    (index counts up from `base_arg_count`); drain GEMMs get none.

    Returns (scratch_args, scratch_for):
      scratch_args: list[FuncArg] to append after the base signature args.
      scratch_for:  list[int|None] parallel to specs_in_order — the allocated
                    combined-arg index for that GEMM, or None if drain.

    This is the single owner of scratch-arg numbering: callers thread
    scratch_for[i] into the GEMM slice's arg_map instead of hand-writing tail
    indices. Making the GQA(1 scratch)->MHA(3 scratch) transition correct by
    construction — the bug class this whole helper exists to kill.
    """
    scratch_args, scratch_for = [], []
    nxt = base_arg_count
    for spec, out_rows, out_cols in specs_in_order:
        if spec["needs_f32_scratch"]:
            scratch_args.append(
                FuncArg(f"%arg{nxt}", f"memref<{out_rows}x{out_cols}xf32>")
            )
            scratch_for.append(nxt)
            nxt += 1
        else:
            scratch_for.append(None)
    return scratch_args, scratch_for


def stitch_elf(
    func_name,
    base_args,
    slices,
    *,
    scratch_args=(),
    prelude="",
    extra_externs=frozenset(),
    debug_dump_path=None,
):
    """Assemble sub-kernel IRs into one multi-launch module and parse it.

    func_name: combined func symbol WITHOUT the leading "@" (e.g.
        "rms_gemms_rope" -> emits `func.func @rms_gemms_rope`).
    base_args: list[FuncArg], the always-present signature args (indices
        0..len(base_args)-1).
    slices: ordered list[KernelSlice]; bodies are spliced in this order.
    scratch_args: list[FuncArg] appended after base_args (registry-driven tail,
        typically from alloc_gemm_scratch). Their indices continue from
        len(base_args).
    prelude: SSA injected at the top of the combined func body (before any
        slice), e.g. a subview/cast that `arg_aliases` route onto.
    extra_externs: externs to preserve beyond the union of slice.extern_syms.
    debug_dump_path: if parsing fails, write the combined IR here before
        re-raising.

    Returns the parsed `air.ir.Module`.
    """
    from air.ir import Context, Module

    all_args = list(base_args) + list(scratch_args)
    n_args = len(all_args)

    # --- Validation (hard errors — silent arg drift is the bug class we kill) ---
    referenced = set()
    for sl in slices:
        operands = set(sl.arg_map) | set(sl.arg_aliases)
        overlap = set(sl.arg_map) & set(sl.arg_aliases)
        if overlap:
            raise ValueError(
                f"stitch_elf: slice '{sl.prefix}' operand(s) {sorted(overlap)} "
                f"appear in BOTH arg_map and arg_aliases"
            )
        for operand_idx, combined_idx in sl.arg_map.items():
            if not (0 <= combined_idx < n_args):
                raise ValueError(
                    f"stitch_elf: slice '{sl.prefix}' arg_map maps operand "
                    f"{operand_idx} -> combined arg {combined_idx}, out of range "
                    f"[0,{n_args})"
                )
            referenced.add(combined_idx)
    unreferenced = set(range(n_args)) - referenced
    if unreferenced:
        raise ValueError(
            f"stitch_elf: combined func arg(s) {sorted(unreferenced)} are never "
            f"referenced by any slice's arg_map (dangling signature args)"
        )

    # --- Mechanical stitch (extract / rename / fixup) ---
    externs = set(extra_externs)
    for sl in slices:
        externs |= set(sl.extern_syms)

    bodies, maps_all, all_privates = [], [], set()
    for sl in slices:
        body = _extract_between_func_and_return(sl.ir)
        maps = _extract_affine_maps(sl.ir)
        body = _rename_all_with_externs(body, sl.prefix, externs)
        maps = [_rename_all_with_externs(m, sl.prefix, externs) for m in maps]
        body = _fix_launch_func_args(
            body, sl.prefix, sl.arg_map, arg_aliases=sl.arg_aliases
        )
        bodies.append(body)
        maps_all.extend(maps)
        if sl.private_from:
            for p in _extract_private_funcs(sl.ir):
                all_privates.add(p.strip())

    privates_str = "\n  ".join(sorted(all_privates))
    sig = ",\n    ".join(f"{a.name}: {a.type}" for a in all_args)
    prelude_str = (prelude + "\n") if prelude else ""
    bodies_str = "\n".join(bodies)

    combined = "\n".join(maps_all) + f"""
module {{
  {privates_str}
  func.func @{func_name}(
    {sig}
  ) {{
{prelude_str}{bodies_str}
    return
  }}
}}
"""

    with Context() as ctx:
        try:
            return Module.parse(combined, ctx)
        except Exception:
            if debug_dump_path:
                with open(debug_dump_path, "w") as f:
                    f.write(combined)
                print(f"  PARSE ERROR: dumped to {debug_dump_path}")
            raise
