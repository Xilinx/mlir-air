# Llama-3.2-1B Plan 2 (Prefill) Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4-cell ablation ladder for the **prefill** kernel-groups (`rms_gemms_rope` 6 launches + `o_ffn` 8 launches at seq=2048 GEMM shapes) using parameterized cells driven by `KernelGroupSpec` dataclasses. FA held constant per master spec. Single-layer + 16-layer scopes. Bit-exact validation against committed goldens. Headline number directly comparable to `profile.md`'s 1.27 s prefill.

**Architecture:** Self-contained subdir `programming_examples/llama32_1b/ablation/prefill/` (Plan 1 files at top-level remain byte-immutable). 4 parameterized cell modules walk a `KernelGroupSpec` (one spec per kernel-group) describing sub-launches, slot semantics, and baton-pass topology. A 16-layer wrapper threads `o_ffn.output[L] → rms_gemms_rope.x_in[L+1]` with FA invariant between the two intra-layer kernel-groups. Reuses Plan 1's `KernelCache.naive=True`, `cells/common.py:compile_standalone_kernels` (helper extracted to `prefill/cells/common.py` and parameterized), and `validate.py` (verbatim, kernel-group-agnostic).

**Tech Stack:** Python 3, numpy, ml_dtypes (bfloat16), pytest, mlir-air's `XRTBackend` + `KernelCache` + existing sub-builders (`build_rms_gemms_rope_module`, `build_o_ffn_module` from `multi_launch_builder/`; `_build_gemm_module` from `kernel_builder/gemm_builder.py`; `_build_rope_2d` from `multi_launch_builder/rms_gemms_rope_multi.py:63`; `_build_add_2d_to_2d` from `multi_launch_builder/o_ffn_multi.py`; `weighted_rms_norm.weighted_rms_norm.build_module`; `ffn_swiglu.silu_and_mul`).

**Companion docs:**
- Plan 2 spec: `programming_examples/llama32_1b/ablation/docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md`
- Master ablation spec: removed from repo (decode pilot deleted; superseded by full-decode study at `ablation/docs/specs/2026-05-12-llama32-1b-ablation-plan2-fulldecode-design.md`)
- Plan 1 (decode pilot) plan: removed from repo (subsumed by full-decode study at `ablation/docs/plans/2026-05-12-llama32-1b-ablation-plan2-fulldecode-plan.md`)
- Plan 1's working code at `programming_examples/llama32_1b/ablation/` — removed; see `ablation/decode/` for the superseding study.

---

## File Structure

All paths under `programming_examples/llama32_1b/ablation/prefill/` unless noted.

| File | Responsibility |
|---|---|
| `__init__.py` | Package marker |
| `README.md` | Methodology, run instructions, results, reproducibility |
| `Makefile` | `make compile / regen-golden / run / report / all / clean` |
| `specs/__init__.py` | Package marker |
| `specs/kernel_group.py` | Frozen dataclasses: `SubLaunchSpec`, `BatonLink`, `KernelGroupSpec` |
| `specs/rms_gemms_rope.py` | Concrete `KernelGroupSpec` instance for the 6-launch prefill attention pre-block |
| `specs/o_ffn.py` | Concrete `KernelGroupSpec` instance for the 8-launch prefill FFN block |
| `standalone_builders/__init__.py` | Package marker |
| `standalone_builders/rms_gemms_rope.py` | 6 single-launch builder wrappers + `STANDALONES` registry |
| `standalone_builders/o_ffn.py` | 8 single-launch builder wrappers + `STANDALONES` registry |
| `cells/__init__.py` | Package marker |
| `cells/common.py` | `compile_standalone_kernels` (parameterized), `_extract_public_func_name`, `_share_bo`, `standalone_backend_kwargs` helpers |
| `cells/cell_a_naive.py` | Parameterized Cell A — walks a `KernelGroupSpec` with `naive=True` |
| `cells/cell_b_static.py` | Parameterized Cell B — preload weights, then `static_input_indices` |
| `cells/cell_c_charitable.py` | Parameterized Cell C — preload + alias intermediate BOs per `spec.baton_links` |
| `cells/cell_d_merged.py` | Wrapper around production `build_rms_gemms_rope_module` and `build_o_ffn_module` |
| `cells/flash_attn_const.py` | FA invariant: compile + invoke production FA ELF identically across all cells |
| `cells/multi_layer.py` | Wraps a per-layer triple (rms_gemms_rope → FA → o_ffn) in a 16-layer loop |
| `golden/__init__.py` | Package marker |
| `golden/regen_golden.py` | One-shot Cell-D run for layer 0; dumps two npz fixtures + meta json |
| `golden/golden_rms_gemms_rope_prefill.npz` | Committed bit-exact reference (Cell D's 6 outputs, layer 0, seed=42) |
| `golden/golden_o_ffn_prefill.npz` | Committed bit-exact reference (Cell D's relevant outputs for o_ffn, layer 0, seed=42) |
| `golden/golden_meta.json` | Hashes, shapes, config |
| `run_ablation.py` | Orchestrator: validate → time × {single-layer, 16-layer} × 4 cells, emit JSON |
| `analyze.py` | JSON → markdown report |
| `tests/__init__.py` | Package marker |
| `tests/conftest.py` | Pytest sys.path setup |
| `tests/test_kernel_group_spec.py` | Dataclass invariants (NPU-free) |
| `tests/test_parameterized_cells.py` | Mock-cache tests verifying each cell walks its spec correctly (NPU-free) |
| `tests/test_validation_gate.py` | Imports Plan 1's `validate.py` and tests it against new prefill goldens |

**Files NOT touched (Plan 1 isolation guarantee):** every file under `programming_examples/llama32_1b/ablation/` outside `prefill/`. Production code (`programming_examples/llama32_1b/kernel_builder/`, `multi_launch_builder/`) read-only — only imported.

---

## Phase 1 — Skeleton + Specs (Tasks 1–4)

## Task 1: Subdir skeleton + pytest conftest

**Files:**
- Create: 9 `__init__.py` files (one per package directory)
- Create: `programming_examples/llama32_1b/ablation/prefill/tests/conftest.py`

- [ ] **Step 1: Create empty package markers**

```bash
mkdir -p programming_examples/llama32_1b/ablation/prefill/{specs,standalone_builders,cells,golden,tests}
for d in prefill prefill/specs prefill/standalone_builders prefill/cells prefill/golden prefill/tests; do
    touch programming_examples/llama32_1b/ablation/$d/__init__.py
done
```

- [ ] **Step 2: Write conftest.py**

`programming_examples/llama32_1b/ablation/prefill/tests/conftest.py`:

```python
"""Pytest config for prefill ablation tests.

Inserts paths so tests can import:
- llama32_1b/ packages (kernel_builder, multi_launch_builder)
- llama32_1b/ablation/ (Plan 1's validate.py and shared helpers)
- llama32_1b/ablation/prefill/ (this package)
- programming_examples/ (matvec, weighted_rms_norm, ffn_swiglu)
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PREFILL = os.path.dirname(_THIS)
_ABLATION = os.path.dirname(_PREFILL)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

for p in (_PROG_EXAMPLES, _LLAMA, _ABLATION, _PREFILL):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pytest's package-import mode inserts the package parent (ablation/) into sys.path[0]
# before this conftest runs, which can shadow prefill/validate.py with ablation/validate.py.
# Guarantee that prefill/ is at index 0 so prefill-local modules take priority.
if sys.path[0] != _PREFILL:
    sys.path.remove(_PREFILL) if _PREFILL in sys.path else None
    sys.path.insert(0, _PREFILL)
```

> **Implementation note (T10 wash-up):** The final three lines above were added in T10
> to fix pytest's package-import mode inserting `ablation/` at `sys.path[0]` before the
> conftest ran, shadowing `prefill/validate.py` with `ablation/validate.py`. The fix
> always-removes-then-reinserts `_PREFILL` at index 0 after the initial insertion loop.

- [ ] **Step 3: Verify pytest discovers the empty test dir**

Run: `cd programming_examples/llama32_1b/ablation/prefill && python3 -m pytest tests/ -v`
Expected: `no tests ran in 0.0Xs` (zero tests, zero errors).

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/
git commit -m "ablation/prefill: scaffold subdir skeleton with pytest conftest"
```

---

## Task 2: Spec dataclasses (`SubLaunchSpec`, `BatonLink`, `KernelGroupSpec`)

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/specs/kernel_group.py`
- Test: `programming_examples/llama32_1b/ablation/prefill/tests/test_kernel_group_spec.py`

- [ ] **Step 1: Write the failing test**

`prefill/tests/test_kernel_group_spec.py`:

```python
"""Unit tests for the KernelGroupSpec dataclasses."""

import pytest
from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec


def _dummy_builder():
    return None  # Spec test doesn't need a real builder


def test_sublaunch_spec_is_frozen():
    s = SubLaunchSpec(
        name="rms",
        builder_ref=_dummy_builder,
        build_kwargs={"emb_dim": 2048},
        weight_slot_in_standalone=1,
        output_slot_in_standalone=2,
    )
    with pytest.raises((AttributeError, TypeError)):  # frozen
        s.name = "other"


def test_baton_link_orders_by_indices():
    link = BatonLink(producer_idx=0, producer_out_slot=2,
                    consumer_idx=1, consumer_in_slot=1)
    assert link.consumer_idx > link.producer_idx


def test_kernel_group_spec_holds_sublaunches():
    sub = SubLaunchSpec("rms", _dummy_builder, {}, 1, 2)
    spec = KernelGroupSpec(
        name="rms_gemms_rope",
        sub_launches=(sub,),  # tuple — frozen dataclass
        merged_arg_signature=("x_in", "norm_w", "normed"),
        weight_slots=frozenset({1}),
        intermediate_slots=frozenset({2}),
        output_slots_for_validation=(2,),
        baton_links=(),
    )
    assert spec.name == "rms_gemms_rope"
    assert len(spec.sub_launches) == 1


def test_baton_link_consumer_must_follow_producer():
    """A baton link with consumer_idx <= producer_idx is meaningless;
    spec dataclass tolerates it but a validator rejects."""
    from specs.kernel_group import validate_baton_links
    sub_a = SubLaunchSpec("a", _dummy_builder, {}, 1, 2)
    sub_b = SubLaunchSpec("b", _dummy_builder, {}, 1, 2)
    bad = BatonLink(producer_idx=1, producer_out_slot=2, consumer_idx=0, consumer_in_slot=1)
    with pytest.raises(ValueError, match="consumer_idx"):
        validate_baton_links([sub_a, sub_b], [bad])
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `cd programming_examples/llama32_1b/ablation/prefill && python3 -m pytest tests/test_kernel_group_spec.py -v`
Expected: `ModuleNotFoundError: No module named 'specs.kernel_group'`.

- [ ] **Step 3: Implement `specs/kernel_group.py`**

```python
"""Frozen dataclasses describing a multi-launch kernel-group's structure.

A KernelGroupSpec is consumed by parameterized cells (cell_a/b/c/d) so that
the same cell logic works for any kernel-group whose spec is provided.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SubLaunchSpec:
    """One sub-launch's standalone definition.

    Used by Cell A/B/C to invoke the sub-launch as its own xrt.run() call.
    Cell D ignores SubLaunchSpec entirely (it uses the merged ELF).
    """
    name: str                          # "rmsnorm" | "q_gemm" | "rope_q" | ...
    builder_ref: Callable              # returns a 1-launch mlir.Module at production shape
    build_kwargs: dict                 # passed verbatim to builder_ref
    weight_slot_in_standalone: int | None  # arg slot of the standalone call holding the weight (or None)
    output_slot_in_standalone: int     # arg slot of the standalone call holding the output


@dataclass(frozen=True)
class BatonLink:
    """An intermediate-BO alias to apply in Cell C.

    The producer's output BO becomes the consumer's input BO; the host
    skips writing the consumer's input slot via intermediate_indices.
    """
    producer_idx: int                  # index into KernelGroupSpec.sub_launches
    producer_out_slot: int             # output slot of producer's standalone signature
    consumer_idx: int                  # index into KernelGroupSpec.sub_launches (must be > producer_idx)
    consumer_in_slot: int              # input slot of consumer's standalone signature


@dataclass(frozen=True)
class KernelGroupSpec:
    """Full description of a multi-launch kernel-group for ablation."""
    name: str                          # "rms_gemms_rope" | "o_ffn"
    sub_launches: tuple                # tuple of SubLaunchSpec (frozen)
    merged_arg_signature: tuple        # tuple of arg-name strings matching production merged ELF args
    weight_slots: frozenset            # slots in merged signature that are weights/LUTs (Cell D static_input_indices)
    intermediate_slots: frozenset      # slots in merged signature that are kernel-overwritten intermediates
    output_slots_for_validation: tuple # slots whose bytes go in the golden npz
    baton_links: tuple                 # tuple of BatonLink (Cell C aliases these intermediate BOs)


def validate_baton_links(sub_launches, baton_links):
    """Sanity check: each link's consumer must come after its producer in the sequence."""
    for link in baton_links:
        if link.consumer_idx <= link.producer_idx:
            raise ValueError(
                f"baton link consumer_idx={link.consumer_idx} must be greater than "
                f"producer_idx={link.producer_idx}"
            )
        if link.producer_idx >= len(sub_launches):
            raise ValueError(f"producer_idx {link.producer_idx} out of range")
        if link.consumer_idx >= len(sub_launches):
            raise ValueError(f"consumer_idx {link.consumer_idx} out of range")
```

- [ ] **Step 4: Re-run the test**

Run: `cd programming_examples/llama32_1b/ablation/prefill && python3 -m pytest tests/test_kernel_group_spec.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/specs/ \
        programming_examples/llama32_1b/ablation/prefill/tests/test_kernel_group_spec.py
git commit -m "ablation/prefill: KernelGroupSpec/SubLaunchSpec/BatonLink dataclasses"
```

---

## Task 3: Concrete `KernelGroupSpec` for `rms_gemms_rope`

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/specs/rms_gemms_rope.py`

**Reference:** Production builder at `programming_examples/llama32_1b/multi_launch_builder/rms_gemms_rope_multi.py:193`. Merged signature has 13 args (slots 0-12); see docstring at lines 211-228 of that file. Static slots: {1, 3, 5, 7, 9, 10}. Intermediate slots: {2, 4, 6, 8, 11, 12}.

The 6 sub-launches:
| Idx | Name | Builder | Production-shape kwargs | weight_slot | output_slot |
|---|---|---|---|---|---|
| 0 | rmsnorm | `weighted_rms_norm.weighted_rms_norm.build_module` (wrapped via `_wrap_ir_in_launch`) | `seq_len=2048, emb_dim=2048, np_dtype=bfloat16, vector_size=16, herd_x=8` | 1 (norm_w) | 2 (normed) |
| 1 | q_gemm | `kernel_builder.gemm_builder._build_gemm_module` | `seq_len=2048, K=2048, N=2048, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4` | 1 (W) | 2 (Y) |
| 2 | k_gemm | same | `seq_len=2048, K=2048, N=512, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4` | 1 | 2 |
| 3 | v_gemm | same | `seq_len=2048, K=2048, N=512, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4` | 1 | 2 |
| 4 | rope_q | `multi_launch_builder.rms_gemms_rope_multi._build_rope_2d` | `outer_rows=2048, outer_cols=2048, embed_dim=64, np_dtype=bfloat16, herd_x=8` | 1 (lut) | 2 (out) |
| 5 | rope_k | same | `outer_rows=2048, outer_cols=512, embed_dim=64, np_dtype=bfloat16, herd_x=8` | 1 | 2 |

Baton links (within-group only; cross-group host hop is invariant per spec):
- (0, 2) → (1, 0)  rmsnorm.normed → q_gemm.x   (slot 0 of standalone gemm = the activation input)
- (0, 2) → (2, 0)  rmsnorm.normed → k_gemm.x
- (0, 2) → (3, 0)  rmsnorm.normed → v_gemm.x
- (1, 2) → (4, 0)  q_gemm.q → rope_q.in
- (2, 2) → (5, 0)  k_gemm.k → rope_k.in

Note: the standalone GEMM signature (`_build_gemm_module`) per its docstring has args `(M, A, B, C)` — verify this in the actual file. If args are `(A, B, C)` then weight slot is 1 (B), activation slot is 0 (A), output slot is 2 (C). The implementer must inspect `kernel_builder/gemm_builder.py:107` to confirm slot positions before finalizing the spec.

- [ ] **Step 1: Write the spec module**

```python
"""Concrete KernelGroupSpec for the prefill rms_gemms_rope kernel-group.

Mirrors the production stitch-spec in
multi_launch_builder/rms_gemms_rope_multi.py:467-474 (which lists the
arg mappings for the 6 sub-launches in the merged ELF).

Slot conventions for standalones:
  - rmsnorm:  (x_in[seq, emb], norm_w[emb], out[seq, emb])     output at slot 2
  - gemm:     (a[seq, K], b[K, N], c[seq, N])                  output at slot 2
              (verify via kernel_builder/gemm_builder.py:107 — ordering may
               be (M, A, B, C); if so, weight slot becomes 2 not 1.)
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
        SubLaunchSpec("rmsnorm",  _build_rmsnorm_standalone, {},                              1, 2),
        SubLaunchSpec("q_gemm",   _build_gemm_standalone,    {"k": 2048, "n": 2048},          1, 2),
        SubLaunchSpec("k_gemm",   _build_gemm_standalone,    {"k": 2048, "n": 512},           1, 2),
        SubLaunchSpec("v_gemm",   _build_gemm_standalone,    {"k": 2048, "n": 512},           1, 2),
        SubLaunchSpec("rope_q",   _build_rope_2d_standalone, {"outer_rows": 2048, "outer_cols": 2048}, 1, 2),
        SubLaunchSpec("rope_k",   _build_rope_2d_standalone, {"outer_rows": 2048, "outer_cols": 512},  1, 2),
    ),
    merged_arg_signature=(
        "x_in", "norm_w", "normed",
        "wq", "q",
        "wk", "k",
        "wv", "v",
        "lut_q", "lut_k",
        "q_roped", "k_roped",
    ),
    weight_slots=frozenset({1, 3, 5, 7, 9, 10}),
    intermediate_slots=frozenset({2, 4, 6, 8, 11, 12}),
    output_slots_for_validation=(2, 4, 6, 8, 11, 12),
    baton_links=(
        BatonLink(producer_idx=0, producer_out_slot=2, consumer_idx=1, consumer_in_slot=0),  # rmsnorm.normed -> q_gemm.x
        BatonLink(producer_idx=0, producer_out_slot=2, consumer_idx=2, consumer_in_slot=0),  # rmsnorm.normed -> k_gemm.x
        BatonLink(producer_idx=0, producer_out_slot=2, consumer_idx=3, consumer_in_slot=0),  # rmsnorm.normed -> v_gemm.x
        BatonLink(producer_idx=1, producer_out_slot=2, consumer_idx=4, consumer_in_slot=0),  # q_gemm.q -> rope_q.in
        BatonLink(producer_idx=2, producer_out_slot=2, consumer_idx=5, consumer_in_slot=0),  # k_gemm.k -> rope_k.in
    ),
)
```

- [ ] **Step 2: Verify the spec validates**

Run:
```bash
cd programming_examples/llama32_1b/ablation/prefill
python3 -c "
from specs.rms_gemms_rope import SPEC
from specs.kernel_group import validate_baton_links
validate_baton_links(SPEC.sub_launches, SPEC.baton_links)
print(f'{SPEC.name}: {len(SPEC.sub_launches)} sub-launches, {len(SPEC.baton_links)} baton links')
"
```
Expected: `rms_gemms_rope: 6 sub-launches, 5 baton links`.

If it errors on `_build_gemm_module` signature mismatch (e.g., the function takes positional args in a different order), fix the keyword arg names to match `kernel_builder/gemm_builder.py:107`. The implementer should read that function's signature first; if it requires an `M` parameter or has different defaults, adjust `_build_gemm_standalone` accordingly.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/specs/rms_gemms_rope.py
git commit -m "ablation/prefill: concrete spec for rms_gemms_rope (6 sub-launches at seq=2048)"
```

---

## Task 4: Concrete `KernelGroupSpec` for `o_ffn`

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/specs/o_ffn.py`

**Reference:** Production builder at `multi_launch_builder/o_ffn_multi.py:178`. Merged signature has 15 args (slots 0-14); see docstring at lines 209-228. Static slots: {1, 5, 7, 9, 12}. Intermediate slots: {2, 4, 6, 8, 10, 11, 13, 14}. Slot 0 (`attn_out`) and slot 3 (`x_residual`) are activation inputs (written every call).

The 8 sub-launches per `o_ffn_multi.py`:
| Idx | Name | Builder | Production-shape kwargs |
|---|---|---|---|
| 0 | o_gemm | `_build_gemm_module` | `seq_len=2048, K=2048, N=2048, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64, herd_m=8, herd_n=4` |
| 1 | res_add | `_build_add_2d_to_2d` | `seq_len=2048, emb_dim=2048, np_dtype=bfloat16` |
| 2 | ffn_rmsnorm | wrapped `weighted_rms_norm.build_module` | `seq_len=2048, emb_dim=2048, np_dtype=bfloat16, vector_size=16, herd_x=8` |
| 3 | gate_gemm | `_build_gemm_module` | `seq_len=2048, K=2048, N=8192, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4` |
| 4 | up_gemm | same | `seq_len=2048, K=2048, N=8192, tile_m=64, tile_k_l2=64, tile_k_l1=32, tile_n=128, herd_m=8, herd_n=4` |
| 5 | swiglu | `ffn_swiglu.silu_and_mul.build_module` (or wrapped per existing usage) | `seq_len=2048, hidden_dim=8192, tile_n=4096, herd_x=8, herd_y=1, np_dtype=bfloat16` |
| 6 | down_gemm | `_build_gemm_module` | `seq_len=2048, K=8192, N=2048, tile_m=64, tile_k_l2=256, tile_k_l1=32, tile_n=64, herd_m=8, herd_n=4` |
| 7 | ffn_add | `_build_add_2d_to_2d` (or its 1D variant — verify via o_ffn_multi.py) | `seq_len=2048, emb_dim=2048, np_dtype=bfloat16` |

Baton links (within-group):
- (0, 2) → (1, 0)  o_gemm.proj → res_add.A     (a 2D add takes 2 activation inputs + 1 output)
- (1, 2) → (2, 0)  res_add.res1 → ffn_rmsnorm.x  (and also feeds ffn_add later as residual)
- (2, 2) → (3, 0)  ffn_rmsnorm.normed2 → gate_gemm.x
- (2, 2) → (4, 0)  ffn_rmsnorm.normed2 → up_gemm.x
- (3, 2) → (5, 0)  gate_gemm.gate → swiglu.gate
- (4, 2) → (5, 1)  up_gemm.up → swiglu.up
- (5, 2) → (6, 0)  swiglu.swiglu → down_gemm.x
- (6, 2) → (7, 0)  down_gemm.down → ffn_add.A
- (1, 2) → (7, 1)  res_add.res1 → ffn_add.B (residual-of-residual; verify against o_ffn_multi.py — the ffn_add's second input is the post-attention residual, which equals res1)

The implementer should inspect `o_ffn_multi.py` to confirm sub-launch order, exact arg slot conventions for the 2D add and SwiGLU, and the residual connectivity in step 7. If `_build_add_2d_to_2d` takes 3 args `(A, B, C)` then activation inputs are slots 0 and 1, output is slot 2. SwiGLU's `silu_and_mul` typically takes `(gate, up, out)` — slot 0 is gate, slot 1 is up, slot 2 is output.

- [ ] **Step 1: Read `o_ffn_multi.py:178-450`** to confirm the exact sub-builder signatures and arg-mapping (see the stitch-spec around line 350-400 of that file).

- [ ] **Step 2: Write the spec module**

> **Implementation note (post-execution wash-up):** Three deviations from the original spec were necessary:
> 1. SwiGLU import is `kernel_builder.ffn_swiglu.silu_and_mul.build_module_2d` (the 2D memref
>    variant, signature `(rows, cols, tile_n, np_dtype, herd_x, herd_y)`) — not `ffn_swiglu.silu_and_mul.build_module`.
>    It already emits `air.launch`; no `_wrap_ir_in_launch` needed.
> 2. `ffn_add` uses `_build_ffn_add_standalone` (replicated from the nested `_build_add_2d_to_1d`
>    inside `o_ffn_multi.py`, which cannot be imported directly) — not `_build_add_2d_to_2d`.
>    Its output is 1D `[n_total]` (2D inputs, 1D output).
> 3. `air.ir` does not export `T`; use `IntegerType.get_signless(32)` instead.

```python
"""Concrete KernelGroupSpec for the prefill o_ffn kernel-group.

Mirrors the production stitch-spec in multi_launch_builder/o_ffn_multi.py.
8 sequential launches at seq=2048, emb_dim=2048, hidden_dim=8192:

  L1  o_gemm      [8,4]  attn_out x wo -> proj
  L2  res_add     [8,1]  proj + x_residual -> res1          (2D out)
  L3  ffn_rmsnorm [8,1]  res1 x ffn_norm_w -> normed2
  L4  gate_gemm   [8,4]  normed2 x w_gate -> gate
  L5  up_gemm     [8,4]  normed2 x w_up -> up
  L6  swiglu      [8,1]  SiLU(gate) x up -> swiglu
  L7  down_gemm   [8,4]  swiglu x w_down -> down
  L8  ffn_add     [8,1]  down + res1 -> output              (1D out)

15 merged-func args (slots 0-14); static slots {1,5,7,9,12};
intermediate slots {2,4,6,8,10,11,13,14}.

Slot conventions per sub-launch standalone signatures:
  - gemm:         (A[seq,K], B[K,N], C[seq,N])          weight=1, out=2
  - add_2d_to_2d: (A[seq,d], B[seq,d], C[seq,d])        no weight, out=2
  - rmsnorm:      (x[seq,d], w[d], out[seq,d])           weight=1, out=2
  - swiglu_2d:    (gate[seq,h], up[seq,h], out[seq,h])   no weight, out=2
  - ffn_add:      (A[seq,d], B[seq,d], out[n_total])     no weight, out=2
"""

from ml_dtypes import bfloat16

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec

# ---------------------------------------------------------------------------
# Sub-launch standalone builders
# ---------------------------------------------------------------------------


def _build_o_gemm_standalone():
    """O projection GEMM: attn_out(2048,2048) x wo(2048,2048) -> proj(2048,2048)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        2048,
        2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )


def _build_res_add_standalone():
    """Residual add (2D->2D): proj + x_residual -> res1."""
    from multi_launch_builder.o_ffn_multi import _build_add_2d_to_2d

    return _build_add_2d_to_2d(2048, 2048, bfloat16)


def _build_rmsnorm_standalone():
    """FFN RMSNorm (bare herd -> wrap in air.launch)."""
    from weighted_rms_norm.weighted_rms_norm import build_module as build_rms
    from kernel_builder.stitching import _wrap_ir_in_launch
    from air.ir import Module

    bare = str(build_rms(2048, 2048, bfloat16, 16, herd_x=8))
    return Module.parse(_wrap_ir_in_launch(bare))


def _build_gateup_gemm_standalone(n):
    """Gate or Up GEMM: normed2(2048,2048) x w(2048,n) -> out(2048,n)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        2048,
        n,
        tile_m=64,
        tile_k_l2=64,
        tile_k_l1=32,
        tile_n=128,
        herd_m=8,
        herd_n=4,
    )


def _build_swiglu_standalone():
    """SwiGLU activation: SiLU(gate) * up -> swiglu  (2D memref variant).

    Uses build_module_2d from kernel_builder/ffn_swiglu/silu_and_mul.py.
    Signature: (rows, cols, tile_n, np_dtype_in, herd_x=8, herd_y=1).
    Already wraps in air.launch -- no _wrap_ir_in_launch needed.
    Arg slots in standalone: 0=gate, 1=up, 2=out.
    """
    from kernel_builder.ffn_swiglu.silu_and_mul import build_module_2d as build_swiglu

    return build_swiglu(2048, 8192, 4096, bfloat16, herd_x=8, herd_y=1)


def _build_down_gemm_standalone():
    """Down GEMM: swiglu(2048,8192) x w_down(8192,2048) -> down(2048,2048)."""
    from kernel_builder.gemm_builder import _build_gemm_module

    return _build_gemm_module(
        2048,
        8192,
        2048,
        tile_m=64,
        tile_k_l2=256,
        tile_k_l1=32,
        tile_n=64,
        herd_m=8,
        herd_n=4,
    )


def _build_ffn_add_standalone():
    """FFN Add (2D inputs -> 1D output): down + res1 -> output[n_total].

    Replicated from the nested _build_add_2d_to_1d() in o_ffn_multi.py
    (that function is defined inline inside build_o_ffn_module and cannot
    be imported directly).

    Arg slots: 0=A (down, 2D), 1=B (res1, 2D), 2=out (1D).
    """
    from air.ir import (
        AffineConstantExpr,
        AffineExpr,
        AffineMap,
        AffineMapAttr,
        AffineSymbolExpr,
        IntegerAttr,
        IntegerType,
        MemRefType,
        VectorType,
        UnitAttr,
        StringAttr,
    )
    from air.dialects.affine import apply as affine_apply
    from air.dialects.air import launch, segment, herd, module_builder
    from air.dialects.memref import (
        collapse_shape as memref_collapse_shape,
        AllocOp,
        DeallocOp,
        subview,
    )
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_, yield_
    from air.dialects import arith
    from air.dialects.vector import transfer_read, transfer_write
    from air.backend.xrt_runner import type_mapper
    from air.dialects.air import MemorySpace

    seq_len = 2048
    emb_dim = 2048
    n_total = seq_len * emb_dim
    total_tiles = 8
    chunk_size = n_total // total_tiles
    tile_n = emb_dim

    @module_builder
    def _build():
        xrt_dtype = type_mapper(bfloat16)
        l3_2d_ty = MemRefType.get([seq_len, emb_dim], xrt_dtype)
        l3_1d_ty = MemRefType.get([n_total], xrt_dtype)
        l1_space = IntegerAttr.get(IntegerType.get_signless(32), MemorySpace.L1)
        l1_ty = MemRefType.get([tile_n], xrt_dtype, memory_space=l1_space)
        vec_ty = VectorType.get([16], xrt_dtype)
        identity_map = AffineMapAttr.get(AffineMap.get_identity(1))

        @FuncOp.from_py_func(l3_2d_ty, l3_2d_ty, l3_1d_ty)
        def eltwise_add(a_2d, b_2d, out_1d):
            @launch(operands=[a_2d, b_2d, out_1d])
            def add_launch(l_a, l_b, l_out):
                a_flat = memref_collapse_shape(l3_1d_ty, l_a, [[0, 1]])
                b_flat = memref_collapse_shape(l3_1d_ty, l_b, [[0, 1]])

                @segment(name="add_seg", operands=[a_flat, b_flat, l_out])
                def add_seg(s_a, s_b, s_out):
                    offset_map = AffineMap.get(
                        0,
                        3,
                        [
                            AffineExpr.get_add(
                                AffineSymbolExpr.get(0),
                                AffineExpr.get_mul(
                                    AffineExpr.get_add(
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(1),
                                            AffineConstantExpr.get(1),
                                        ),
                                        AffineSymbolExpr.get(2),
                                    ),
                                    AffineConstantExpr.get(chunk_size),
                                ),
                            )
                        ],
                    )

                    @herd(
                        name="add_herd",
                        sizes=[8, 1],
                        operands=[s_a, s_b, s_out],
                    )
                    def add_body(_tx, _ty, _sx, _sy, h_a, h_b, h_out):
                        l1_a = AllocOp(l1_ty, [], [])
                        l1_b = AllocOp(l1_ty, [], [])
                        l1_out = AllocOp(l1_ty, [], [])
                        c0 = arith.ConstantOp.create_index(0)
                        cst0 = arith.ConstantOp(xrt_dtype, 0.0)
                        for loop_iv in for_(0, chunk_size, tile_n):
                            offset = affine_apply(offset_map, [loop_iv, _tx, _ty])
                            from air.dialects.air import dma_memcpy_nd

                            dma_memcpy_nd(
                                l1_a,
                                h_a,
                                src_offsets=[offset],
                                src_sizes=[tile_n],
                                src_strides=[1],
                            )
                            dma_memcpy_nd(
                                l1_b,
                                h_b,
                                src_offsets=[offset],
                                src_sizes=[tile_n],
                                src_strides=[1],
                            )
                            for j in for_(0, tile_n, 16):
                                sub_a = subview(l1_a.result, [j], [16], [1])
                                sub_b = subview(l1_b.result, [j], [16], [1])
                                sub_out = subview(l1_out.result, [j], [16], [1])
                                v_a = transfer_read(
                                    vec_ty, sub_a, [c0], identity_map, cst0, [True]
                                )
                                v_b = transfer_read(
                                    vec_ty, sub_b, [c0], identity_map, cst0, [True]
                                )
                                v_sum = arith.addf(v_a, v_b)
                                transfer_write(
                                    None, v_sum, sub_out, [c0], identity_map, [True]
                                )
                                yield_([])
                            dma_memcpy_nd(
                                h_out,
                                l1_out,
                                dst_offsets=[offset],
                                dst_sizes=[tile_n],
                                dst_strides=[1],
                            )
                            yield_([])
                        DeallocOp(l1_a)
                        DeallocOp(l1_b)
                        DeallocOp(l1_out)

    return _build()


# ---------------------------------------------------------------------------
# KernelGroupSpec
# ---------------------------------------------------------------------------

SPEC = KernelGroupSpec(
    name="o_ffn",
    sub_launches=(
        # idx=0: O GEMM -- weight at slot 1 (wo), output at slot 2 (proj)
        SubLaunchSpec("o_gemm", _build_o_gemm_standalone, {}, 1, 2),
        # idx=1: Res Add -- no weight, output at slot 2 (res1[2D])
        SubLaunchSpec("res_add", _build_res_add_standalone, {}, None, 2),
        # idx=2: FFN RMSNorm -- weight at slot 1 (ffn_norm_w), output at slot 2 (normed2)
        SubLaunchSpec("ffn_rmsnorm", _build_rmsnorm_standalone, {}, 1, 2),
        # idx=3: Gate GEMM -- weight at slot 1 (w_gate), output at slot 2 (gate)
        SubLaunchSpec("gate_gemm", _build_gateup_gemm_standalone, {"n": 8192}, 1, 2),
        # idx=4: Up GEMM -- weight at slot 1 (w_up), output at slot 2 (up)
        SubLaunchSpec("up_gemm", _build_gateup_gemm_standalone, {"n": 8192}, 1, 2),
        # idx=5: SwiGLU -- no weight, gate=slot0, up=slot1, output at slot 2
        SubLaunchSpec("swiglu", _build_swiglu_standalone, {}, None, 2),
        # idx=6: Down GEMM -- weight at slot 1 (w_down), output at slot 2 (down)
        SubLaunchSpec("down_gemm", _build_down_gemm_standalone, {}, 1, 2),
        # idx=7: FFN Add -- no weight, A=slot0 (down), B=slot1 (res1), output at slot 2
        SubLaunchSpec("ffn_add", _build_ffn_add_standalone, {}, None, 2),
    ),
    merged_arg_signature=(
        "attn_out",  # 0  activation input
        "wo",  # 1  weight (static)
        "proj",  # 2  intermediate
        "x_residual",  # 3  activation input
        "res1",  # 4  intermediate  (shared: res_add out + ffn_add B)
        "ffn_norm_w",  # 5  weight (static)
        "normed2",  # 6  intermediate
        "w_gate",  # 7  weight (static)
        "gate",  # 8  intermediate
        "w_up",  # 9  weight (static)
        "up",  # 10 intermediate
        "swiglu",  # 11 intermediate
        "w_down",  # 12 weight (static)
        "down",  # 13 intermediate
        "output",  # 14 intermediate (final 1D output)
    ),
    weight_slots=frozenset({1, 5, 7, 9, 12}),
    intermediate_slots=frozenset({2, 4, 6, 8, 10, 11, 13, 14}),
    output_slots_for_validation=(14,),
    baton_links=(
        # Stitch arg_map verified against o_ffn_multi.py lines 457-465:
        #   L1 {0:0,1:1,2:2}  L2 {0:2,1:3,2:4}  L3 {0:4,1:5,2:6}
        #   L4 {0:6,1:7,2:8}  L5 {0:6,1:9,2:10} L6 {0:8,1:10,2:11}
        #   L7 {0:11,1:12,2:13}  L8 {0:13,1:4,2:14}
        BatonLink(0, 2, 1, 0),  # o_gemm.proj (slot2) -> res_add.A (slot0)
        BatonLink(1, 2, 2, 0),  # res_add.res1 (slot2) -> ffn_rmsnorm.x (slot0)
        BatonLink(2, 2, 3, 0),  # ffn_rmsnorm.normed2 (slot2) -> gate_gemm.x (slot0)
        BatonLink(2, 2, 4, 0),  # ffn_rmsnorm.normed2 (slot2) -> up_gemm.x (slot0)
        BatonLink(3, 2, 5, 0),  # gate_gemm.gate (slot2) -> swiglu.gate (slot0)
        BatonLink(4, 2, 5, 1),  # up_gemm.up (slot2) -> swiglu.up (slot1)
        BatonLink(5, 2, 6, 0),  # swiglu.swiglu (slot2) -> down_gemm.x (slot0)
        BatonLink(6, 2, 7, 0),  # down_gemm.down (slot2) -> ffn_add.A (slot0)
        BatonLink(
            1, 2, 7, 1
        ),  # res_add.res1 (slot2) -> ffn_add.B (slot1)  [residual-of-residual]
    ),
)
```

- [ ] **Step 3: Verify the spec**

```bash
cd programming_examples/llama32_1b/ablation/prefill
python3 -c "
from specs.o_ffn import SPEC
from specs.kernel_group import validate_baton_links
validate_baton_links(SPEC.sub_launches, SPEC.baton_links)
print(f'{SPEC.name}: {len(SPEC.sub_launches)} sub-launches, {len(SPEC.baton_links)} baton links')
"
```
Expected: `o_ffn: 8 sub-launches, 9 baton links`. If any sub-builder import fails, the implementer must adjust the standalone helpers per the actual production code in `o_ffn_multi.py`.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/specs/o_ffn.py
git commit -m "ablation/prefill: concrete spec for o_ffn (8 sub-launches at seq=2048)"
```

---

## Phase 2 — Standalone Builders + Compile (Tasks 5–7)

## Task 5: Standalone builders for `rms_gemms_rope`

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/standalone_builders/rms_gemms_rope.py`

This is a thin wrapper file. Most of the build logic lives in `specs/rms_gemms_rope.py` (the `_build_*_standalone` helpers). This file just re-exports a `STANDALONES` registry compatible with the compile harness in T7.

- [ ] **Step 1: Write the file**

```python
"""Single-launch standalone modules for the prefill rms_gemms_rope kernel-group.

Exports a STANDALONES registry compatible with cells/common.py:compile_standalone_kernels.
Each entry: (name, build_fn, build_kwargs).
"""

from specs.rms_gemms_rope import SPEC


STANDALONES = [
    (sub.name, sub.builder_ref, sub.build_kwargs)
    for sub in SPEC.sub_launches
]
```

- [ ] **Step 2: Verify the registry**

```bash
cd programming_examples/llama32_1b/ablation/prefill
python3 -c "
from standalone_builders.rms_gemms_rope import STANDALONES
assert len(STANDALONES) == 6, f'expected 6, got {len(STANDALONES)}'
for name, build_fn, kwargs in STANDALONES:
    print(f'{name}: {build_fn.__name__}({kwargs})')
"
```
Expected: 6 lines listing rmsnorm, q_gemm, k_gemm, v_gemm, rope_q, rope_k with their kwargs.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/standalone_builders/rms_gemms_rope.py
git commit -m "ablation/prefill: standalone STANDALONES registry for rms_gemms_rope"
```

---

## Task 6: Standalone builders for `o_ffn`

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/standalone_builders/o_ffn.py`

Identical pattern to T5; only the spec module differs.

- [ ] **Step 1: Write the file**

```python
"""Single-launch standalone modules for the prefill o_ffn kernel-group.

Exports a STANDALONES registry compatible with cells/common.py:compile_standalone_kernels.
"""

from specs.o_ffn import SPEC


STANDALONES = [
    (sub.name, sub.builder_ref, sub.build_kwargs)
    for sub in SPEC.sub_launches
]
```

- [ ] **Step 2: Verify**

```bash
cd programming_examples/llama32_1b/ablation/prefill
python3 -c "
from standalone_builders.o_ffn import STANDALONES
assert len(STANDALONES) == 8, f'expected 8, got {len(STANDALONES)}'
for name, build_fn, kwargs in STANDALONES:
    print(f'{name}: {build_fn.__name__}({kwargs})')
"
```
Expected: 8 lines.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/standalone_builders/o_ffn.py
git commit -m "ablation/prefill: standalone STANDALONES registry for o_ffn"
```

---

## Task 7: Compile harness — `cells/common.py` + actual compile

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/common.py`
- Create: `programming_examples/llama32_1b/ablation/prefill/.gitignore`

This file mirrors Plan 1's `cells/common.py` (lifting the `_extract_public_func_name` regex, `compile_standalone_kernels`, `_share_bo`, `standalone_backend_kwargs` helpers). The only difference: the compile harness uses one of two prefill backends (RMS_GEMMS_ROPE_BACKEND or O_FFN_BACKEND) per kernel-group.

- [ ] **Step 1: Write `cells/common.py`**

> **Implementation note (post-execution wash-up):** `compile_standalone_kernels` must wrap
> `build_fn(**kwargs)` in a `with MLIRContext():` block; without it the MLIR module
> parse context is missing and the builder crashes. Also note that
> `programming_examples/llama32_1b/kernel_builder/external_kernels.py` was modified
> alongside this task to add an `MLIR_AIE_INSTALL_DIR` env-var fallback for worktree
> path resolution — that change is a candidate for cherry-picking back to `llama-3.2-1B-devel`
> independently of the ablation work.

```python
"""Shared helpers for prefill ablation cells.

Lifted (and extended for two-backend support) from Plan 1's
ablation/cells/common.py. The original Plan 1 file is read-only.

- compile_standalone_kernels(cache, group_name, registry, backend_preset):
    Compile every standalone in `registry` into `cache`, using the actual
    public func name extracted from the MLIR module as instance_name.
- _extract_public_func_name(mlir_text): regex over the module string.
- _share_bo(cache, src_key, src_slot, dst_key, dst_slot): alias cached BOs
  for Cell C's baton-pass.
- standalone_backend_kwargs(backend_preset, verbose): returns backend kwargs
  with instance_name removed (set per-kernel by compile_standalone_kernels).
"""

import re

from air.ir import Context as MLIRContext

from kernel_builder.cache import KernelCache


def _extract_public_func_name(mlir_text):
    """Find the first non-private `func.func @<name>` in the module text."""
    for line in mlir_text.split("\n"):
        if "func.func @" in line and "private" not in line:
            m = re.search(r"@(\w+)", line)
            if m:
                return m.group(1)
    raise ValueError("no public func.func found in module")


def standalone_backend_kwargs(backend_preset, verbose=False):
    """Backend kwargs with instance_name removed (set per-kernel by caller)."""
    base = {**backend_preset, "verbose": verbose}
    base.pop("instance_name", None)
    return base


def compile_standalone_kernels(
    cache: KernelCache, group_name: str, registry, backend_preset
):
    """Compile every standalone in `registry` into `cache` under names
    f"{group_name}__{name}". Skip any kernel already in cache.artifacts.

    Each registry entry: (name, build_fn, build_kwargs).
    """
    for name, build_fn, kwargs in registry:
        kernel_name = f"{group_name}__{name}"
        if kernel_name in cache.artifacts:
            continue
        with MLIRContext():
            mlir_module = build_fn(**kwargs)
            public_func = _extract_public_func_name(str(mlir_module))
        be = standalone_backend_kwargs(backend_preset, verbose=cache.verbose)
        be["instance_name"] = public_func
        cache.compile_and_cache(kernel_name, mlir_module, be)
    cache._save_manifest()


def _share_bo(cache, src_key, src_slot, dst_key, dst_slot):
    """Replace cached BO at (dst_key, dst_slot) with the same xrt.bo as
    (src_key, src_slot). Only valid after both kernels' first call has
    materialized BOs."""
    src_bos = cache._cached_bos[src_key]
    dst_bos = cache._cached_bos[dst_key]
    dst_bos[dst_slot] = src_bos[src_slot]


def main():
    """python3 -m cells.common — compile both kernel-groups' standalones."""
    from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND
    from standalone_builders.rms_gemms_rope import STANDALONES as RMS_STD
    from standalone_builders.o_ffn import STANDALONES as O_STD

    cache = KernelCache(cache_dir="standalone_cache", verbose=True)
    cache.load_manifest()
    compile_standalone_kernels(cache, "rms_gemms_rope", RMS_STD, RMS_GEMMS_ROPE_BACKEND)
    compile_standalone_kernels(cache, "o_ffn", O_STD, O_FFN_BACKEND)
    print(f"Compiled {len(cache.artifacts)} standalone ELFs.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `.gitignore`**

```bash
echo "build/" > programming_examples/llama32_1b/ablation/prefill/.gitignore
echo "standalone_cache/" >> programming_examples/llama32_1b/ablation/prefill/.gitignore
echo "results_*.json" >> programming_examples/llama32_1b/ablation/prefill/.gitignore
echo "report_*.md" >> programming_examples/llama32_1b/ablation/prefill/.gitignore
```

- [ ] **Step 3: Run the compile (one-time, ~10–15 min for 14 ELFs at seq=2048)**

```bash
cd programming_examples/llama32_1b/ablation/prefill
mkdir -p build && cd build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -m cells.common
```

Expected output: 14 lines `Compiled rms_gemms_rope__<name>: <T>s` and `Compiled o_ffn__<name>: <T>s`. **NO `instance_name ... does not match` warnings** (the `_extract_public_func_name` regex prevents that — see Plan 1 T6 wash-up).

- [ ] **Step 4: Verify the manifest**

```bash
python3 -c "
import json
with open('standalone_cache/manifest.json') as f:
    m = json.load(f)
assert len(m) == 14, f'expected 14, got {len(m)}'
for name, info in sorted(m.items()):
    assert info['kernel'].startswith('main:'), f'bad kernel ref: {info[\"kernel\"]}'
print(f'manifest OK: {len(m)} entries')
"
```
Expected: `manifest OK: 14 entries`.

- [ ] **Step 5: Commit (source + .gitignore only; no binaries)**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/common.py \
        programming_examples/llama32_1b/ablation/prefill/.gitignore
git commit -m "ablation/prefill: compile harness for both kernel-groups (14 ELFs)"
```

---

## Phase 3 — Cells + Golden + Validation + FA (Tasks 8–11)

## Task 8: Cell D — production wrapper for both kernel-groups

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/cell_d_merged.py`

Two cell-D entry points (one per kernel-group). Each compiles the production merged ELF (if not cached) and provides a `run_cell_d_<group>(cache, layer_inputs, layer_idx)` function returning the same dict shape Plan 1 used.

- [ ] **Step 1: Write cell_d_merged.py**

```python
"""Cell D — production: invoke the merged ELFs (rms_gemms_rope.elf with 6
launches; o_ffn.elf with 8 launches) using the production KernelCache +
backend presets.
"""

import os
import sys

# Ensure llama32_1b/ is on sys.path so kernel_builder and multi_launch_builder
# are importable whether this file is run directly or imported from the
# prefill/ package root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LLAMA_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _LLAMA_DIR not in sys.path:
    sys.path.insert(0, _LLAMA_DIR)

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND
from multi_launch_builder.rms_gemms_rope_multi import build_rms_gemms_rope_module
from multi_launch_builder.o_ffn_multi import build_o_ffn_module

CONFIG = {
    "seq_len": 2048,
    "emb_dim": 2048,
    "kv_dim": 512,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 64,
    "hidden_dim": 8192,
}


def compile_cell_d_rms_gemms_rope(cache: KernelCache):
    if "rms_gemms_rope" in cache.artifacts:
        return
    mod = build_rms_gemms_rope_module(
        seq_len=CONFIG["seq_len"], emb_dim=CONFIG["emb_dim"],
        kv_dim=CONFIG["kv_dim"], n_heads=CONFIG["n_heads"],
        n_kv_heads=CONFIG["n_kv_heads"], head_dim=CONFIG["head_dim"],
    )
    cache.compile_and_cache("rms_gemms_rope", mod,
                            {"verbose": cache.verbose, **RMS_GEMMS_ROPE_BACKEND})
    cache._save_manifest()


def compile_cell_d_o_ffn(cache: KernelCache):
    if "o_ffn" in cache.artifacts:
        return
    mod = build_o_ffn_module(
        seq_len=CONFIG["seq_len"], emb_dim=CONFIG["emb_dim"],
        hidden_dim=CONFIG["hidden_dim"],
    )
    cache.compile_and_cache("o_ffn", mod,
                            {"verbose": cache.verbose, **O_FFN_BACKEND})
    cache._save_manifest()


def run_cell_d_rms_gemms_rope(cache, layer_inputs, layer_idx=0):
    """One rms_gemms_rope call (6 launches in one xrt.run).
    layer_inputs has keys: x_in, norm_w, wq, wk, wv, lut_q, lut_k.
    Returns dict with normed, q, k, v, q_roped, k_roped, _wall_s.
    """
    seq = CONFIG["seq_len"]; emb = CONFIG["emb_dim"]; kv = CONFIG["kv_dim"]
    args = [
        layer_inputs["x_in"],
        layer_inputs["norm_w"],
        np.zeros((seq, emb), dtype=bfloat16),  # normed
        layer_inputs["wq"],
        np.zeros((seq, emb), dtype=bfloat16),  # q
        layer_inputs["wk"],
        np.zeros((seq, kv), dtype=bfloat16),   # k
        layer_inputs["wv"],
        np.zeros((seq, kv), dtype=bfloat16),   # v
        layer_inputs["lut_q"],
        layer_inputs["lut_k"],
        np.zeros((seq, emb), dtype=bfloat16),  # q_roped
        np.zeros((seq, kv), dtype=bfloat16),   # k_roped
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "rms_gemms_rope", RMS_GEMMS_ROPE_BACKEND,
        *args,
        output_indices=[2, 4, 6, 8, 11, 12],
        static_input_indices={1, 3, 5, 7, 9, 10},
        intermediate_indices={2, 4, 6, 8, 11, 12},
        bo_key=f"D_rms_gemms_rope_L{layer_idx}",
    )
    elapsed = time.perf_counter() - t0
    return {
        "normed": out[2], "q": out[4], "k": out[6], "v": out[8],
        "q_roped": out[11], "k_roped": out[12],
        "_wall_s": elapsed,
    }


def run_cell_d_o_ffn(cache, layer_inputs, layer_idx=0):
    """One o_ffn call (8 launches in one xrt.run).
    layer_inputs has: attn_out, wo, x_residual, ffn_norm_w, w_gate, w_up, w_down.
    Returns dict with output, _wall_s.
    """
    seq = CONFIG["seq_len"]; emb = CONFIG["emb_dim"]; hid = CONFIG["hidden_dim"]
    n_total = seq * emb
    args = [
        layer_inputs["attn_out"],                     # 0
        layer_inputs["wo"],                           # 1
        np.zeros((seq, emb), dtype=bfloat16),         # 2 proj
        layer_inputs["x_residual"],                   # 3
        np.zeros((seq, emb), dtype=bfloat16),         # 4 res1
        layer_inputs["ffn_norm_w"],                   # 5
        np.zeros((seq, emb), dtype=bfloat16),         # 6 normed2
        layer_inputs["w_gate"],                       # 7
        np.zeros((seq, hid), dtype=bfloat16),         # 8 gate
        layer_inputs["w_up"],                         # 9
        np.zeros((seq, hid), dtype=bfloat16),         # 10 up
        np.zeros((seq, hid), dtype=bfloat16),         # 11 swiglu
        layer_inputs["w_down"],                       # 12
        np.zeros((seq, emb), dtype=bfloat16),         # 13 down
        np.zeros(n_total, dtype=bfloat16),            # 14 output (1D)
    ]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "o_ffn", O_FFN_BACKEND,
        *args,
        output_indices=[14],
        static_input_indices={1, 5, 7, 9, 12},
        intermediate_indices={2, 4, 6, 8, 10, 11, 13, 14},
        bo_key=f"D_o_ffn_L{layer_idx}",
    )
    return {"output": out[14], "_wall_s": time.perf_counter() - t0}
```

- [ ] **Step 2: Verify import + signature**

```bash
cd programming_examples/llama32_1b/ablation/prefill
python3 -c "
from cells.cell_d_merged import (compile_cell_d_rms_gemms_rope,
                                   compile_cell_d_o_ffn,
                                   run_cell_d_rms_gemms_rope,
                                   run_cell_d_o_ffn, CONFIG)
print('OK', CONFIG['seq_len'], CONFIG['emb_dim'], CONFIG['hidden_dim'])
"
```
Expected: `OK 2048 2048 8192`.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/cell_d_merged.py
git commit -m "ablation/prefill: Cell D wrappers for rms_gemms_rope and o_ffn merged ELFs"
```

---

## Task 9: Golden fixture generator + commit

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/golden/regen_golden.py`
- Generate + commit: `golden/golden_rms_gemms_rope_prefill.npz`, `golden/golden_o_ffn_prefill.npz`, `golden/golden_meta.json`

- [ ] **Step 1: Write `regen_golden.py`**

```python
"""Regenerate prefill golden fixtures by running Cell D once for each kernel-group.

Uses deterministic synthetic inputs (numpy seed=42 for layer 0).
Outputs:
  golden/golden_rms_gemms_rope_prefill.npz
  golden/golden_o_ffn_prefill.npz
  golden/golden_meta.json
"""

import hashlib
import json
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernel_builder.cache import KernelCache
from cells.cell_d_merged import (
    CONFIG,
    compile_cell_d_rms_gemms_rope, compile_cell_d_o_ffn,
    run_cell_d_rms_gemms_rope, run_cell_d_o_ffn,
)


def _synthetic_layer_inputs(layer_idx, config):
    """Deterministic synthetic inputs for one prefill layer (seq=2048).

    Same seeding scheme as Plan 1: seed = 42 + layer_idx.
    """
    rng = np.random.default_rng(42 + layer_idx)
    seq = config["seq_len"]; emb = config["emb_dim"]
    kv = config["kv_dim"]; hid = config["hidden_dim"]
    return {
        "x_in":       rng.standard_normal((seq, emb)).astype(bfloat16),
        "norm_w":     rng.standard_normal(emb).astype(bfloat16),
        "wq":         rng.standard_normal((emb, emb)).astype(bfloat16),
        "wk":         rng.standard_normal((emb, kv)).astype(bfloat16),
        "wv":         rng.standard_normal((emb, kv)).astype(bfloat16),
        "lut_q":      rng.standard_normal(seq * emb).astype(bfloat16),
        "lut_k":      rng.standard_normal(seq * kv).astype(bfloat16),
        "wo":         rng.standard_normal((emb, emb)).astype(bfloat16),
        "ffn_norm_w": rng.standard_normal(emb).astype(bfloat16),
        "w_gate":     rng.standard_normal((emb, hid)).astype(bfloat16),
        "w_up":       rng.standard_normal((emb, hid)).astype(bfloat16),
        "w_down":     rng.standard_normal((hid, emb)).astype(bfloat16),
    }


def main():
    cache = KernelCache(cache_dir="standalone_cache", verbose=True)
    cache.load_manifest()
    compile_cell_d_rms_gemms_rope(cache)
    compile_cell_d_o_ffn(cache)

    inputs = _synthetic_layer_inputs(0, CONFIG)

    # rms_gemms_rope golden
    rg_inputs = {k: inputs[k] for k in ["x_in","norm_w","wq","wk","wv","lut_q","lut_k"]}
    rg_out = run_cell_d_rms_gemms_rope(cache, rg_inputs, layer_idx=0)
    rg_path = os.path.join(os.path.dirname(__file__), "golden_rms_gemms_rope_prefill.npz")
    np.savez(rg_path, **{k: v for k, v in rg_out.items() if not k.startswith("_")})

    # For o_ffn golden, attn_out comes from FA in production. For the golden
    # we use a CPU FA reference computed from rg_out's q_roped/k_roped/v —
    # since FA is invariant across cells, all cells will see the same attn_out.
    # Simplest: synthesize attn_out from the same RNG (it is what flows into
    # o_ffn's slot 0 in every cell; the bytes are determined upstream).
    attn_out = np.random.default_rng(42 + 0 + 1000).standard_normal(
        (CONFIG["seq_len"], CONFIG["emb_dim"])).astype(bfloat16)
    of_inputs = {
        "attn_out":   attn_out,
        "wo":         inputs["wo"],
        "x_residual": inputs["x_in"],  # the residual is the layer input
        "ffn_norm_w": inputs["ffn_norm_w"],
        "w_gate":     inputs["w_gate"],
        "w_up":       inputs["w_up"],
        "w_down":     inputs["w_down"],
    }
    of_out = run_cell_d_o_ffn(cache, of_inputs, layer_idx=0)
    of_path = os.path.join(os.path.dirname(__file__), "golden_o_ffn_prefill.npz")
    np.savez(of_path, **{k: v for k, v in of_out.items() if not k.startswith("_")})

    meta = {
        "config": CONFIG,
        "rms_gemms_rope": {
            "input_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                             for k, v in rg_inputs.items()},
            "output_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                              for k, v in rg_out.items() if not k.startswith("_")},
        },
        "o_ffn": {
            "input_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                             for k, v in of_inputs.items()},
            "output_hashes": {k: hashlib.sha256(v.tobytes()).hexdigest()[:16]
                              for k, v in of_out.items() if not k.startswith("_")},
        },
    }
    with open(os.path.join(os.path.dirname(__file__), "golden_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {rg_path}, {of_path}, golden_meta.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 ../golden/regen_golden.py
```

Expected: 2 ELFs compiled (rms_gemms_rope ~30s, o_ffn ~50s if not cached), then `Wrote .../golden_rms_gemms_rope_prefill.npz, .../golden_o_ffn_prefill.npz, golden_meta.json`. The two npz files together should be a few MB (six 2048×N arrays + one 2048×2048 output = ~16-32 MB total).

- [ ] **Step 3: Verify fixtures**

```bash
ls -la programming_examples/llama32_1b/ablation/prefill/golden/
python3 -c "
import numpy as np
rg = np.load('programming_examples/llama32_1b/ablation/prefill/golden/golden_rms_gemms_rope_prefill.npz')
of = np.load('programming_examples/llama32_1b/ablation/prefill/golden/golden_o_ffn_prefill.npz')
print('rg files:', list(rg.files))
print('of files:', list(of.files))
"
```
Expected: rg has 6 arrays (normed, q, k, v, q_roped, k_roped); of has 1 array (output).

- [ ] **Step 4: Commit fixtures**

```bash
git add programming_examples/llama32_1b/ablation/prefill/golden/
git commit -m "ablation/prefill: golden fixtures from Cell D for rms_gemms_rope and o_ffn"
```

---

## Task 10: Validation gate (reuse Plan 1 + new test)

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/tests/test_validation_gate.py`

We **reuse Plan 1's `validate.py` verbatim** (no copy). Plan 1's `validate_against_golden(cell_outputs, golden_dir)` reads from `<golden_dir>/golden_rms_gemv_rope.npz` though — so we either pass a different filename or accept Plan 1's logic.

The simplest: lift the validate logic into a small `prefill/validate.py` that takes a `golden_npz_filename` parameter so we can reuse it for both kernel-groups' goldens.

- [ ] **Step 1: Create `prefill/validate.py` (lifted from Plan 1, parameterized)**

```python
"""Per-cell validation — parameterized version of Plan 1's validate.py.

Plan 1's validate.py hardcodes the golden filename to
"golden_rms_gemv_rope.npz". For prefill we have two goldens, so we
parameterize the filename. The byte-equality contract is identical.
"""

import os

import numpy as np

# Reuse the exception class from Plan 1 if available; redefine if not.
try:
    from validate import GoldenMismatch  # Plan 1's exception
except ImportError:
    class GoldenMismatch(AssertionError):
        pass


def validate_against_golden(cell_outputs: dict, golden_dir: str, npz_filename: str):
    """Compare every key in cell_outputs to the matching array in
    <golden_dir>/<npz_filename>. Raise GoldenMismatch on any diff."""
    npz = np.load(os.path.join(golden_dir, npz_filename))
    for key in npz.files:
        if key not in cell_outputs:
            raise GoldenMismatch(f"cell missing output '{key}'")
        gv = npz[key]
        cv = cell_outputs[key]
        if cv.shape != gv.shape:
            raise GoldenMismatch(f"{key}: shape mismatch cell={cv.shape} golden={gv.shape}")
        if cv.dtype.itemsize != gv.dtype.itemsize:
            raise GoldenMismatch(f"{key}: itemsize mismatch")
        if cv.tobytes() != gv.tobytes():
            from ml_dtypes import bfloat16 as _bf16
            cf = cv.view(np.uint8).view(_bf16).astype(np.float32) if cv.dtype != np.float32 else cv
            gf = gv.view(np.uint8).view(_bf16).astype(np.float32) if gv.dtype != np.float32 else gv
            max_abs = float(np.max(np.abs(cf - gf)))
            max_rel = float(np.max(np.abs((cf - gf) / (np.abs(gf) + 1e-9))))
            raise GoldenMismatch(f"{key}: byte mismatch  max_abs={max_abs:.4g}  max_rel={max_rel:.4g}")
```

- [ ] **Step 2: Write the test**

`prefill/tests/test_validation_gate.py`:

```python
"""Test the prefill validation gate against the committed goldens."""

import os

import numpy as np
import pytest
from ml_dtypes import bfloat16

from validate import validate_against_golden, GoldenMismatch

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden")


def _load(filename):
    npz = np.load(os.path.join(GOLDEN_DIR, filename))
    return {k: npz[k] for k in npz.files}


def test_rms_gemms_rope_passes_on_exact_match():
    g = _load("golden_rms_gemms_rope_prefill.npz")
    validate_against_golden(g, GOLDEN_DIR, "golden_rms_gemms_rope_prefill.npz")


def test_rms_gemms_rope_raises_on_byte_diff():
    g = _load("golden_rms_gemms_rope_prefill.npz")
    perturbed = {k: v.copy() for k, v in g.items()}
    arr = perturbed["normed"].view(np.uint8).copy()
    arr[0] ^= 0x01
    perturbed["normed"] = arr.view(bfloat16).reshape(g["normed"].shape)
    with pytest.raises(GoldenMismatch, match="normed"):
        validate_against_golden(perturbed, GOLDEN_DIR, "golden_rms_gemms_rope_prefill.npz")


def test_o_ffn_passes_on_exact_match():
    g = _load("golden_o_ffn_prefill.npz")
    validate_against_golden(g, GOLDEN_DIR, "golden_o_ffn_prefill.npz")


def test_o_ffn_raises_on_byte_diff():
    g = _load("golden_o_ffn_prefill.npz")
    perturbed = {k: v.copy() for k, v in g.items()}
    arr = perturbed["output"].view(np.uint8).copy()
    arr[0] ^= 0x01
    perturbed["output"] = arr.view(bfloat16).reshape(g["output"].shape)
    with pytest.raises(GoldenMismatch, match="output"):
        validate_against_golden(perturbed, GOLDEN_DIR, "golden_o_ffn_prefill.npz")
```

- [ ] **Step 3: Run the tests**

```bash
cd programming_examples/llama32_1b/ablation/prefill && python3 -m pytest tests/test_validation_gate.py -v
```
Expected: 4 passed.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/validate.py \
        programming_examples/llama32_1b/ablation/prefill/tests/test_validation_gate.py
git commit -m "ablation/prefill: parameterized validation gate + tests"
```

---

## Task 11: FA invariant integration

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/flash_attn_const.py`

FA's role per spec: held constant in every cell. Same standalone ELF, same invocation pattern, same BO management. The only thing the cells do differently around FA is the upstream/downstream BO management of rms_gemms_rope's outputs and o_ffn's inputs — both happen via host hop in every cell (matches production).

- [ ] **Step 1: Write `flash_attn_const.py`**

```python
"""FlashAttention invariant: same standalone ELF + same invocation in every cell.

FA's MLIR builder is at programming_examples/flash_attention/kernel_fusion_based/attn_npu2_seqfirst.py
with kwargs matching Plan 1's compile_all_kernels() in llama32_1b_prefill.py.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache


def _attn_backend_kwargs():
    return {
        "verbose": False,
        "omit_while_true_loop": False,  # head_dim=64, lkp=64 enables shared buffers
        "omit_pingpong": "all",
        "runtime_loop_tiling_sizes": [1, 1],
        "output_format": "elf",
        "instance_name": "attention_bf16",
    }


def compile_flash_attn(cache: KernelCache, config):
    """Compile FA ELF if not already cached. ~46s first time per profile.md."""
    if "flash_attn" in cache.artifacts:
        return
    from flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as build_attn,
    )
    seq = config["seq_len"]; head_dim = config["head_dim"]
    n_heads = config["n_heads"]; n_kv_heads = config["n_kv_heads"]
    mod = build_attn(
        lk=seq, lkp=head_dim, lq=seq, lqp=256,
        dk=head_dim, dv=head_dim,
        num_q_tiles=4, num_cascade_stages=4,
        num_heads=n_heads, num_kv_heads=n_kv_heads,
        causal=True,
    )
    cache.compile_and_cache("flash_attn", mod, _attn_backend_kwargs())
    cache._save_manifest()


def run_flash_attn(cache, q_roped, k_roped, v, layer_idx=0):
    """Run FA on extracted q_roped/k_roped/v from rms_gemms_rope.
    Returns attn_out (extracted to host) ready to feed o_ffn.
    """
    seq = q_roped.shape[0]; emb = q_roped.shape[1]
    args = [q_roped, k_roped, v, np.zeros((seq, emb), dtype=bfloat16)]
    t0 = time.perf_counter()
    out = cache.load_and_run(
        "flash_attn", _attn_backend_kwargs(),
        *args,
        output_indices=[3],
        intermediate_indices={3},
        bo_key=f"FA_L{layer_idx}",
    )
    return {"attn_out": out[3], "_wall_s": time.perf_counter() - t0}
```

- [ ] **Step 2: Smoke test (compile + invoke once)**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "
import sys, os
sys.path[:0] = ['..', '../..', '../../..']
import numpy as np
from ml_dtypes import bfloat16
from kernel_builder.cache import KernelCache
from cells.cell_d_merged import CONFIG
from cells.flash_attn_const import compile_flash_attn, run_flash_attn

cache = KernelCache(cache_dir='standalone_cache', verbose=False)
cache.load_manifest()
compile_flash_attn(cache, CONFIG)
seq = CONFIG['seq_len']; emb = CONFIG['emb_dim']; kv = CONFIG['kv_dim']
q = np.zeros((seq, emb), dtype=bfloat16)
k = np.zeros((seq, kv), dtype=bfloat16)
v = np.zeros((seq, kv), dtype=bfloat16)
out = run_flash_attn(cache, q, k, v)
print(f'FA OK, attn_out shape={out[\"attn_out\"].shape}, wall={out[\"_wall_s\"]*1000:.1f}ms')
"
```
Expected: `FA OK, attn_out shape=(2048, 2048), wall=...ms`. First run includes ~46s compile.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/flash_attn_const.py
git commit -m "ablation/prefill: FA invariant integration (compile + invoke same ELF in every cell)"
```

---

## Phase 4 — Parameterized Cells (Tasks 12–14)

## Task 12: Cell A — naive parameterized

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/cell_a_naive.py`

The cell takes a `KernelGroupSpec` and walks its `sub_launches` in order, invoking each via `cache.load_and_run(naive=True)`. Between sub-launches, the previous output is extracted to host (because naive=True forces all-read) and re-written into the next call's input array slot.

The trick: each sub-launch's standalone signature has a fixed shape `(input_or_weight, activation_input, output)` for the GEMM/RoPE families. The activation input slot may be 0 or 1 depending on the builder. The spec's `BatonLink.consumer_in_slot` tells us which slot to write the upstream output into. For Cell A (no actual sharing), we use the baton_links list only to know how to thread Python data — not for BO aliasing.

- [ ] **Step 1: Write `cell_a_naive.py`**

```python
"""Cell A — Naive no-merge for a generic KernelGroupSpec.

For each sub-launch:
  1. Allocate a numpy buffer for the output (zeros).
  2. Build the call's input arrays per the spec's BatonLink upstream
     (or layer_inputs[name] if no upstream link for that input slot).
  3. Invoke cache.load_and_run with naive=True (writes everything,
     reads everything every call).
  4. Stash the output into a results dict keyed by sub_launch.name.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from cells.common import compile_standalone_kernels


def _consumer_input_for(spec, consumer_idx, consumer_slot, results, layer_inputs):
    """Return the numpy array to put in (consumer_idx, consumer_slot).

    If a BatonLink targets this (consumer_idx, consumer_slot), use the
    producer's output from results. Otherwise, look up by sub-launch name
    in layer_inputs.
    """
    for link in spec.baton_links:
        if link.consumer_idx == consumer_idx and link.consumer_in_slot == consumer_slot:
            producer_name = spec.sub_launches[link.producer_idx].name
            return results[producer_name]
    # Not a baton-driven slot — must be in layer_inputs by sub-launch name
    sub = spec.sub_launches[consumer_idx]
    # Convention: layer_inputs uses canonical slot-0 names per sub-launch.
    # The implementer should adjust this lookup if the spec uses different keys.
    return layer_inputs.get(f"{sub.name}_in{consumer_slot}",
                            layer_inputs.get(f"{sub.name}_x"))


def compile_cell_a(cache, spec, backend_preset):
    """Compile the standalone ELFs for this kernel-group."""
    registry = [(s.name, s.builder_ref, s.build_kwargs) for s in spec.sub_launches]
    compile_standalone_kernels(cache, spec.name, registry, backend_preset)


def run_cell_a(cache, spec, layer_inputs, layer_idx=0):
    """Run all spec.sub_launches sequentially with naive=True.

    layer_inputs is a dict whose keys are documented per-spec (typically:
    raw layer inputs like x_in, weight matrices, LUTs).
    Returns dict with each sub-launch's output keyed by sub.name, plus _wall_s.
    """
    backend = {**__import__("kernel_builder.backend_presets", fromlist=[spec.name.upper() + "_BACKEND"]).__dict__.get(spec.name.upper() + "_BACKEND", {})}
    backend.pop("instance_name", None)

    results = {}
    t0 = time.perf_counter()

    for idx, sub in enumerate(spec.sub_launches):
        # Allocate output buffer with the right shape
        # The implementer will need a per-spec shape registry to map
        # (sub.name, slot) → shape. For now, we infer from layer_inputs.
        # NOTE: This is a placeholder; the concrete shape lookup belongs in
        # the spec or in a small helper invoked here.
        out_buf = layer_inputs[f"_out_buf_{sub.name}"]  # implementer provides

        # Build the call args list of length 3 (assume 3-arg standalone)
        args = [None, None, None]
        for slot in range(3):
            if slot == sub.output_slot_in_standalone:
                args[slot] = out_buf
            elif slot == sub.weight_slot_in_standalone:
                args[slot] = layer_inputs[f"{sub.name}_w"]
            else:
                # Activation input
                args[slot] = _consumer_input_for(spec, idx, slot, results, layer_inputs)

        result = cache.load_and_run(
            f"{spec.name}__{sub.name}", backend,
            *args,
            output_indices=[sub.output_slot_in_standalone],
            naive=True,
        )
        results[sub.name] = result[sub.output_slot_in_standalone]

    elapsed = time.perf_counter() - t0
    results["_wall_s"] = elapsed
    return results
```

**Note on `_out_buf_<name>` and `<name>_w`**: the implementer should refine `layer_inputs`'s schema. A cleaner approach is to add a small `_shape_map` or `_naming_convention` field to `KernelGroupSpec` so cells can compute output buffer sizes and look up weights/activations by their sub-launch slot positions deterministically.

The above is a starting point — the implementer is expected to iterate on the helper functions as they discover the actual weight/input shapes per sub-launch. The contract is: `run_cell_a(cache, spec, layer_inputs)` returns `{sub.name: output_array, ..., "_wall_s": float}` for every sub.name in `spec.sub_launches`.

- [ ] **Step 2: Sanity-check single-layer for rms_gemms_rope vs golden**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "
import sys, os
sys.path[:0] = ['..', '../..', '../../..']
import numpy as np
from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND
from cells.cell_a_naive import compile_cell_a, run_cell_a
from specs.rms_gemms_rope import SPEC
from golden.regen_golden import _synthetic_layer_inputs, CONFIG
from validate import validate_against_golden, GoldenMismatch

cache = KernelCache(cache_dir='standalone_cache', verbose=False)
cache.load_manifest()
compile_cell_a(cache, SPEC, RMS_GEMMS_ROPE_BACKEND)

layer_inputs = _synthetic_layer_inputs(0, CONFIG)
# Adapter: convert layer_inputs into the schema cell_a_naive expects
# (this is the implementer's first iteration job — write the adapter)
# ...
out = run_cell_a(cache, SPEC, layer_inputs)
# Map cell-A's per-sub-launch outputs to the golden's keys
cell_outputs = {
    'normed':  out['rmsnorm'],
    'q':       out['q_gemm'],
    'k':       out['k_gemm'],
    'v':       out['v_gemm'],
    'q_roped': out['rope_q'],
    'k_roped': out['rope_k'],
}
try:
    validate_against_golden(cell_outputs, '../golden', 'golden_rms_gemms_rope_prefill.npz')
    print('Cell A rms_gemms_rope bit-exact PASS')
except GoldenMismatch as e:
    print(f'Cell A rms_gemms_rope FAIL: {e}')
"
```

If the script errors due to schema gaps (`_out_buf_<name>` keys missing), iterate on `_consumer_input_for` and the layer_inputs adapter until validation passes. **Do not push through with non-bit-exact results.**

If you cannot get bit-exact PASS within reasonable effort, escalate as BLOCKED — the parameterization may need a richer spec (e.g., shape map per sub-launch) or the slot conventions may be off.

- [ ] **Step 3: Commit only after PASS for both kernel-groups**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/cell_a_naive.py
git commit -m "ablation/prefill: Cell A naive parameterized harness"
```

---

## Task 13: Cell B — static parameterized

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/cell_b_static.py`

Identical structure to Cell A, but adds a `preload_cell_b(cache, spec, weights_per_layer)` that writes weights once per layer with `static_input_indices={spec.weight_slots}` and matching `bo_key`. The run path uses `static_input_indices` to skip the rewrite.

- [ ] **Step 1: Write `cell_b_static.py`**

Mirror Plan 1's `cells/cell_b_static.py` pattern (reference: `programming_examples/llama32_1b/ablation/cells/cell_b_static.py:1-179`), but replace the hardcoded sub-launch loop with a walk over `spec.sub_launches`.

For each sub-launch, the preload does:

```python
cache.load_and_run(
    f"{spec.name}__{sub.name}", backend,
    *_preload_args(sub, weights_per_layer[li]),
    output_indices=[sub.output_slot_in_standalone],
    static_input_indices={sub.weight_slot_in_standalone}
        if sub.weight_slot_in_standalone is not None else set(),
    bo_key=f"B_{spec.name}_{sub.name}_L{li}",
)
```

The actual run path is the same dataflow as Cell A but with:
- No `naive=True` flag.
- `static_input_indices={sub.weight_slot_in_standalone}` set per call.
- Same `bo_key` as preload.

Skip showing the full file — the implementer can copy Cell A's structure and add the static_input_indices argument. The bit-exact validation step is identical to Cell A's Step 2.

- [ ] **Step 2: Validate bit-exact for both kernel-groups**

Same one-liner pattern as Task 12 Step 2, importing `cell_b_static`. Expected: `Cell B rms_gemms_rope bit-exact PASS` AND `Cell B o_ffn bit-exact PASS`.

- [ ] **Step 3: Commit on success**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/cell_b_static.py
git commit -m "ablation/prefill: Cell B per-layer weight BOs parameterized"
```

---

## Task 14: Cell C — charitable parameterized (BO aliasing)

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/cell_c_charitable.py`

Cell C extends Cell B by aliasing intermediate BOs across separate `xrt.run()` calls per `spec.baton_links`. The pattern from Plan 1 (`programming_examples/llama32_1b/ablation/cells/cell_c_charitable.py:1-223`) generalizes cleanly: walk `spec.baton_links` and call `_share_bo` from `cells/common.py`.

- [ ] **Step 1: Write `cell_c_charitable.py`**

The structure:

```python
def preload_cell_c(cache, spec, weights_per_layer, backend_preset):
    """Same allocation as Cell B (one call per kernel per layer with weights),
    then walk spec.baton_links and alias intermediate BOs."""
    # ... Cell B preload pattern ...
    for li in range(len(weights_per_layer)):
        for link in spec.baton_links:
            producer = spec.sub_launches[link.producer_idx]
            consumer = spec.sub_launches[link.consumer_idx]
            _share_bo(
                cache,
                f"C_{spec.name}_{producer.name}_L{li}", link.producer_out_slot,
                f"C_{spec.name}_{consumer.name}_L{li}", link.consumer_in_slot,
            )


def run_cell_c(cache, spec, layer_inputs, layer_idx=0):
    """Same call sequence as Cell B but with intermediate_indices set on
    aliased slots so the host doesn't write zero-fill to them."""
    # For each call, intermediate_indices includes:
    #   - The output slot if it's a producer in any baton_link
    #   - Any input slot if this call is the consumer of a baton_link
    # Build per-sub-launch intermediate sets from the spec.baton_links.
    intermediate_for = {}  # sub_idx -> set of slots
    for link in spec.baton_links:
        intermediate_for.setdefault(link.producer_idx, set()).add(link.producer_out_slot)
        intermediate_for.setdefault(link.consumer_idx, set()).add(link.consumer_in_slot)
    # ... rest mirrors Cell B with intermediate_indices=intermediate_for[idx] ...
```

The implementer should reference Plan 1's `cell_c_charitable.py` for the per-call boilerplate (allocating BO via load_and_run with dummy data first, then aliasing, then the actual timed run with `intermediate_indices`).

- [ ] **Step 2: Validate bit-exact for both kernel-groups**

Same pattern as Tasks 12/13. Expected: `Cell C rms_gemms_rope bit-exact PASS` AND `Cell C o_ffn bit-exact PASS`.

If aliasing fails, debug per Plan 1's notes (Task 13 in the decode pilot plan): `print(id(...))` to verify the BOs are the same object after `_share_bo`.

- [ ] **Step 3: Commit on success**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/cell_c_charitable.py
git commit -m "ablation/prefill: Cell C BO baton-pass parameterized"
```

---

## Phase 5 — Multi-Layer + Orchestrator (Tasks 15–16)

## Task 15: Multi-layer wrapper

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/cells/multi_layer.py`

Wraps a per-layer triple in a 16-layer loop. The `x_in` of layer L+1 = `output` of layer L's o_ffn. FA runs between rms_gemms_rope and o_ffn in every layer, with `attn_out` extracted to host and fed into o_ffn's slot 0.

- [ ] **Step 1: Write `multi_layer.py`**

```python
"""16-layer prefill wrapper.

Threads:  rms_gemms_rope[L] -> FA[L] -> o_ffn[L] -> rms_gemms_rope[L+1]

The cell-A/B/C/D dispatch strategy is independent of this wrapper; we
take the cell's per-kernel-group runner as a parameter.
"""

import time

import numpy as np
from ml_dtypes import bfloat16

from cells.flash_attn_const import run_flash_attn


def run_16_layer_prefill(
    cache, config,
    run_rms_gemms_rope, run_o_ffn,
    layer_inputs_per_layer,
):
    """Run a 16-layer prefill via the supplied per-kernel-group runners.

    Args:
        cache: shared KernelCache (FA + both groups + standalones all reside here)
        config: dict from cell_d_merged.CONFIG
        run_rms_gemms_rope(cache, layer_inputs, layer_idx) -> {normed,q,k,v,q_roped,k_roped, _wall_s}
        run_o_ffn(cache, layer_inputs, layer_idx) -> {output, _wall_s}
        layer_inputs_per_layer: list of 16 dicts, each with all per-layer weights+LUTs+x_in[layer 0 only]

    Returns dict with:
        per_layer_wall: list of 16 floats (wall time per layer including FA)
        total_wall: float
        final_output: numpy array (last layer's o_ffn output)
    """
    n_layers = len(layer_inputs_per_layer)
    per_layer_wall = []
    x_in = layer_inputs_per_layer[0]["x_in"]
    final_output = None

    t_total_start = time.perf_counter()
    for L in range(n_layers):
        layer_in = dict(layer_inputs_per_layer[L])
        layer_in["x_in"] = x_in  # threaded from previous layer

        t_layer_start = time.perf_counter()

        # 1. rms_gemms_rope
        rg_out = run_rms_gemms_rope(cache, layer_in, layer_idx=L)
        # 2. FA (invariant)
        # rms_gemms_rope returns 1D flat arrays; FA expects 2D (seq, dim)
        seq = config["seq_len"]
        emb = config["emb_dim"]
        kv = config["kv_dim"]
        q_roped_2d = rg_out["q_roped"].reshape(seq, emb)
        k_roped_2d = rg_out["k_roped"].reshape(seq, kv)
        v_2d = rg_out["v"].reshape(seq, kv)
        fa_out = run_flash_attn(cache, q_roped_2d, k_roped_2d, v_2d, layer_idx=L)
        # 3. o_ffn — assemble inputs
        of_in = {
            "attn_out":   fa_out["attn_out"],
            "wo":         layer_in["wo"],
            "x_residual": x_in,
            "ffn_norm_w": layer_in["ffn_norm_w"],
            "w_gate":     layer_in["w_gate"],
            "w_up":       layer_in["w_up"],
            "w_down":     layer_in["w_down"],
        }
        of_out = run_o_ffn(cache, of_in, layer_idx=L)
        # The o_ffn output (slot 14) is 1D (n_total = seq*emb); reshape for next layer
        x_in = of_out["output"].reshape(config["seq_len"], config["emb_dim"])
        final_output = x_in

        per_layer_wall.append(time.perf_counter() - t_layer_start)

    total_wall = time.perf_counter() - t_total_start
    return {
        "per_layer_wall": per_layer_wall,
        "total_wall": total_wall,
        "final_output": final_output,
    }
```

- [ ] **Step 2: Smoke test (Cell D × 2 layers as a sanity check, not 16)**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "
import sys, os
sys.path[:0] = ['..', '../..', '../../..']
from kernel_builder.cache import KernelCache
from cells.cell_d_merged import (CONFIG, compile_cell_d_rms_gemms_rope,
                                   compile_cell_d_o_ffn,
                                   run_cell_d_rms_gemms_rope, run_cell_d_o_ffn)
from cells.flash_attn_const import compile_flash_attn
from cells.multi_layer import run_16_layer_prefill
from golden.regen_golden import _synthetic_layer_inputs

cache = KernelCache(cache_dir='standalone_cache', verbose=False)
cache.load_manifest()
compile_cell_d_rms_gemms_rope(cache)
compile_cell_d_o_ffn(cache)
compile_flash_attn(cache, CONFIG)

layers = [_synthetic_layer_inputs(L, CONFIG) for L in range(2)]
out = run_16_layer_prefill(cache, CONFIG,
                            run_cell_d_rms_gemms_rope, run_cell_d_o_ffn, layers)
print(f'2-layer Cell D: total={out[\"total_wall\"]*1000:.1f}ms, '
      f'per_layer={[f\"{w*1000:.1f}\" for w in out[\"per_layer_wall\"]]}')
"
```

Expected: a number around 160 ms (= 2 layers × ~80 ms/layer per profile.md). If much higher, check for kernel re-compile happening per layer (shouldn't — the artifact cache should hit on second call).

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/cells/multi_layer.py
git commit -m "ablation/prefill: 16-layer wrapper threading rms_gemms_rope -> FA -> o_ffn"
```

---

## Task 16: `run_ablation.py` orchestrator

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/run_ablation.py`

Three modes: `--scope=single-layer`, `--scope=16-layer`, `--scope=both` (default). For each scope, run validation gate first (single-layer Cell A/B/C/D each validated against golden), then time each cell over N trials.

- [ ] **Step 1: Write the orchestrator**

> **Implementation note (post-execution wash-up):** Two fixes were applied versus the
> original skeleton:
> 1. **sys.path always-remove-then-insert:** `_PREFILL` must be at `sys.path[0]` so
>    `prefill/cells/` wins over any `ablation/cells/`. The pattern is: append lower-priority
>    dirs, then force `_PREFILL` to index 0 with remove-then-insert.
> 2. **`_unload_all_contexts()` between cells in 16-layer scope:** The NPU has ~16 HW
>    context slots. Cell A/B/C each load 14 standalone contexts + FA = 15 total, plus
>    Cell D adds 2 merged + FA = 3. Without unloading between cells the limit is exceeded.
>    `_unload_all_contexts` clears `cache._loaded` and `cache._cached_bos`; Cell B/C
>    weights are then re-preloaded before the 16-layer run.

```python
"""Run the prefill 4-cell ablation.

Modes:
  --scope=single-layer    5 trials × 1-layer cell call (per kernel-group)
  --scope=16-layer        5 trials × 16-layer triple (rms->FA->o_ffn) loop
  --scope=both (default)  both above

Run from programming_examples/llama32_1b/ablation/prefill/build/
(where standalone_cache/ lives and xclbins are found).
"""

import argparse
import json
import os
import sys
import time

# Path setup: this script lives in prefill/; CWD is build/ (where standalone_cache/ lives)
# prefill/ -> ablation/ -> llama32_1b/ -> programming_examples/
_PREFILL = os.path.dirname(os.path.abspath(__file__))
_ABLATION = os.path.dirname(_PREFILL)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

# Insert in ascending priority: _PROG_EXAMPLES appended, _PREFILL at front.
# Use append for lower-priority dirs so they don't shadow prefill's 'cells' package.
for p in (_PROG_EXAMPLES, _LLAMA, _ABLATION):
    if p not in sys.path:
        sys.path.append(p)
# _PREFILL must be at index 0 so prefill/cells/ wins over ablation/cells/.
if _PREFILL in sys.path:
    sys.path.remove(_PREFILL)
sys.path.insert(0, _PREFILL)

import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RMS_GEMMS_ROPE_BACKEND, O_FFN_BACKEND

from validate import validate_against_golden, GoldenMismatch
from cells import cell_a_naive, cell_b_static, cell_c_charitable, cell_d_merged
from cells.flash_attn_const import compile_flash_attn
from cells.multi_layer import run_16_layer_prefill
from specs.rms_gemms_rope import SPEC as RG_SPEC
from specs.o_ffn import SPEC as OF_SPEC
from golden.regen_golden import _synthetic_layer_inputs

GOLDEN_DIR = os.path.join(_PREFILL, "golden")


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------


def _unload_all_contexts(cache):
    """Unload all XRT HW contexts and drop all cached BOs.

    The NPU has a limited number of HW context slots (~16).  When switching
    between single-layer (14+ standalone contexts) and 16-layer (up to 15
    contexts for Cell A/B/C), we must release all contexts first to avoid
    hitting the limit.

    BOs are allocated against a specific XRT device handle; after unloading
    the backend that handle is nulled, so the old BO objects are unusable.
    We must also clear _cached_bos so the next load_and_run allocates fresh
    BOs against the new device.  This means preloaded Cell B/C weights are
    lost and will be re-written on the next call (acceptable since the
    16-layer loop only runs one cell at a time anyway).
    """
    for name, (backend, _) in list(cache._loaded.items()):
        try:
            backend.unload()
        except Exception:
            pass
    cache._loaded.clear()
    cache._cached_bos.clear()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument(
        "--scope",
        choices=["single-layer", "16-layer", "both"],
        default="both",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cache = KernelCache(cache_dir="standalone_cache", verbose=False)
    cache.load_manifest()

    # ---- Compile all cells + FA (idempotent -- skips if already cached) ----
    print("=== Compiling kernels (idempotent) ===")
    cell_a_naive.compile_cell_a(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_a_naive.compile_cell_a(cache, OF_SPEC, O_FFN_BACKEND)
    cell_b_static.compile_cell_b(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_b_static.compile_cell_b(cache, OF_SPEC, O_FFN_BACKEND)
    cell_c_charitable.compile_cell_c(cache, RG_SPEC, RMS_GEMMS_ROPE_BACKEND)
    cell_c_charitable.compile_cell_c(cache, OF_SPEC, O_FFN_BACKEND)
    cell_d_merged.compile_cell_d_rms_gemms_rope(cache)
    cell_d_merged.compile_cell_d_o_ffn(cache)
    compile_flash_attn(cache, cell_d_merged.CONFIG)
    print("All kernels compiled/cached.\n")

    # ---- Generate per-layer synthetic inputs (all 16 layers) ----
    layer_inputs_per_layer = [
        _synthetic_layer_inputs(L, cell_d_merged.CONFIG) for L in range(16)
    ]

    # ---- Pre-load weights for Cell B and Cell C (both kernel-groups, all 16 layers) ----
    print("=== Pre-loading weights for Cell B and Cell C ===")
    rg_weights = [
        {k: li[k] for k in ["norm_w", "wq", "wk", "wv", "lut_q", "lut_k"]}
        for li in layer_inputs_per_layer
    ]
    of_weights = [
        {k: li[k] for k in ["wo", "ffn_norm_w", "w_gate", "w_up", "w_down"]}
        for li in layer_inputs_per_layer
    ]

    cell_b_static.preload_cell_b(
        cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
    )
    cell_b_static.preload_cell_b(
        cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
    )
    cell_c_charitable.preload_cell_c(
        cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND
    )
    cell_c_charitable.preload_cell_c(
        cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
    )
    print("Preload done.\n")

    results = {
        "config": cell_d_merged.CONFIG,
        "trials": args.trials,
        "scope": args.scope,
        "cells": {},
    }

    # ---- Timing: 16-layer scope ----
    if args.scope in ("16-layer", "both"):
        print("=== Timing: 16-layer scope ===")
        for cell in ("A", "B", "C", "D"):
            # Unload all previously opened XRT contexts and BOs before each
            # cell's 16-layer run.  The NPU has ~16 HW context slots; Cell A/B/C
            # each need 14 standalone contexts + FA = 15 total.  Starting fresh
            # per cell avoids hitting the limit.
            # Cell B/C weights are lost with the BOs -- re-preload them below.
            _unload_all_contexts(cache)

            # Re-preload weights for B and C after the context reset.
            if cell == "B":
                cell_b_static.preload_cell_b(
                    cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND,
                )
                cell_b_static.preload_cell_b(
                    cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
                )
            elif cell == "C":
                cell_c_charitable.preload_cell_c(
                    cache, RG_SPEC, rg_weights, cell_d_merged.CONFIG, RMS_GEMMS_ROPE_BACKEND,
                )
                cell_c_charitable.preload_cell_c(
                    cache, OF_SPEC, of_weights, cell_d_merged.CONFIG, O_FFN_BACKEND
                )

            # ... timing loop (see shipped run_ablation.py for full implementation) ...
        print()

    # ---- Dump JSON ----
    out_path = args.out or f"results_prefill_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
```

> The full implementation (validation loops, single-layer timing, 16-layer timing, output
> key adapters) lives in the shipped `run_ablation.py`. The skeleton above captures the
> structural changes from the wash-up; see the committed file for the complete code.

Output JSON shape (target):

```json
{
  "config": {...},
  "trials": 5,
  "cells": {
    "A": {
      "rms_gemms_rope": {"validation": "PASS", "single_layer": {...}, "16_layer": {...}},
      "o_ffn": {"validation": "PASS", "single_layer": {...}, "16_layer": {...}},
      "16_layer_total": {"median_s": ..., ...}
    },
    "B": {...}, "C": {...}, "D": {...}
  }
}
```

- [ ] **Step 2: Run end-to-end (5 trials, both scopes)**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 ../run_ablation.py --trials 5 --scope both --out results_pilot.json
```

Expected output: validation lines for all 4 cells × 2 kernel-groups (8 × PASS), then timing lines for single-layer and 16-layer scopes per cell. Total run time ~5-10 min.

The 16-layer Cell D total wall time is the **headline** number — should be in the ballpark of `profile.md`'s 1.27 s.

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/run_ablation.py
git commit -m "ablation/prefill: orchestrator runs all cells × both kernel-groups × both scopes"
```

---

## Phase 6 — Report + Docs (Tasks 17–19)

## Task 17: `analyze.py` report generator

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/analyze.py`

- [ ] **Step 1: Write the analyzer**

```python
"""Read prefill results JSON and emit a markdown report.

Sections:
- Validation badge (per cell × kernel-group)
- Single-layer per-call medians (per cell × kernel-group)
- 16-layer total wall (per cell, with comparison to profile.md's 1.27s)
- Marginal deltas (A→B, B→C, C→D, A→D — per kernel-group AND aggregated)
- Per-launch breakdown extracted from Cell C's single-layer timing data
"""

import argparse
import json
import os
import time

PROFILE_MD_HEADLINE_S = 1.27  # production prefill from profile.md


def report(results):
    cells = results["cells"]
    out = []
    out.append("# Prefill Ablation — Report\n")
    out.append(f"Trials: {results['trials']}, config: seq={results['config']['seq_len']}, "
               f"emb={results['config']['emb_dim']}, hidden={results['config']['hidden_dim']}\n")

    # Validation table
    out.append("## Validation\n")
    out.append("| Cell | rms_gemms_rope | o_ffn |")
    out.append("|------|----------------|-------|")
    for c in ("A", "B", "C", "D"):
        rg = cells.get(c, {}).get("rms_gemms_rope", {}).get("validation", "—")
        of = cells.get(c, {}).get("o_ffn", {}).get("validation", "—")
        out.append(f"| {c} | {rg} | {of} |")
    out.append("")

    # Single-layer per-call timing table
    out.append("## Single-layer per-call medians (ms)\n")
    out.append("| Cell | rms_gemms_rope | o_ffn |")
    out.append("|------|----------------|-------|")
    for c in ("A", "B", "C", "D"):
        rg_s = cells.get(c, {}).get("rms_gemms_rope", {}).get("single_layer", {}).get("median_s")
        of_s = cells.get(c, {}).get("o_ffn", {}).get("single_layer", {}).get("median_s")
        rg_str = f"{rg_s*1000:.2f}" if rg_s is not None else "—"
        of_str = f"{of_s*1000:.2f}" if of_s is not None else "—"
        out.append(f"| {c} | {rg_str} | {of_str} |")
    out.append("")

    # 16-layer headline table
    out.append("## 16-layer total wall (s) — comparable to profile.md's 1.27 s\n")
    out.append("| Cell | Median (s) | Min (s) | Max (s) | vs profile.md |")
    out.append("|------|------------|---------|---------|---------------|")
    for c in ("A", "B", "C", "D"):
        e = cells.get(c, {}).get("16_layer_total", {})
        if not e:
            out.append(f"| {c} | — | — | — | — |")
            continue
        md = e["median_s"]; mn = e["min_s"]; mx = e["max_s"]
        ratio = md / PROFILE_MD_HEADLINE_S
        out.append(f"| {c} | {md:.3f} | {mn:.3f} | {mx:.3f} | {ratio:.2f}× |")
    out.append("")

    # Marginal deltas (16-layer total)
    out.append("## Marginal deltas (16-layer total)\n")
    def m(c): return cells.get(c, {}).get("16_layer_total", {}).get("median_s")
    pairs = [
        ("A→B (= #2 per-layer weight BOs)", "A", "B"),
        ("B→C (= #3 shared intermediate BOs)", "B", "C"),
        ("C→D (= #1 multi-launch merging, isolated)", "C", "D"),
        ("A→D (= total dispatch-related speedup)", "A", "D"),
    ]
    out.append("| Comparison | Δ s | Speedup |")
    out.append("|------------|-----|---------|")
    for label, a, b in pairs:
        ma, mb = m(a), m(b)
        if ma is None or mb is None:
            out.append(f"| {label} | — | — |")
            continue
        out.append(f"| {label} | {ma - mb:+.3f} | {ma/mb:.2f}× |")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with open(args.results_json) as f:
        results = json.load(f)
    text = report(results)
    out = args.out or f"report_prefill_{int(time.time())}.md"
    with open(out, "w") as f:
        f.write(text)
    print(f"Wrote {out}\n")
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate report**

```bash
cd programming_examples/llama32_1b/ablation/prefill/build
python3 ../analyze.py results_pilot.json --out report_pilot.md
cat report_pilot.md
```

Expected: a markdown report with all 4 cells' validation, single-layer medians, 16-layer totals, and marginal deltas. The Cell D 16-layer total should be in the ballpark of 1.27 s (the headline confirmation).

- [ ] **Step 3: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/analyze.py
git commit -m "ablation/prefill: markdown report generator with profile.md comparison"
```

---

## Task 18: README + Makefile

**Files:**
- Create: `programming_examples/llama32_1b/ablation/prefill/Makefile`
- Create: `programming_examples/llama32_1b/ablation/prefill/README.md`

- [ ] **Step 1: Write Makefile**

```make
# Llama-3.2-1B prefill ablation harness
#
# make compile       — compile all standalone ELFs + Cell D's 2 merged ELFs + FA (~10-15 min, cached)
# make regen-golden  — regenerate committed golden fixtures (rare; only after Cell D changes)
# make run           — run all 4 cells × 2 kernel-groups × both scopes, emit JSON
# make report        — generate markdown report from latest results JSON
# make all           — compile + run + report
# make clean         — wipe build/

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
BUILD := build

.PHONY: help compile regen-golden run report all clean

help:
	@echo "make compile | regen-golden | run | report | all | clean"

compile:
	@mkdir -p $(BUILD)
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -m cells.common

regen-golden: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/golden/regen_golden.py

run: compile
	cd $(BUILD) && PYTHONPATH=$(srcdir):$(srcdir)/..:$(srcdir)/../..:$(srcdir)/../../..:$$PYTHONPATH flock -x -w 1800 /tmp/mlir-air-npu.lock python3 $(srcdir)/run_ablation.py --out results_latest.json

report:
	cd $(BUILD) && python3 $(srcdir)/analyze.py results_latest.json --out report_latest.md && cat report_latest.md

all: compile run report

clean:
	rm -rf $(BUILD)
```

- [ ] **Step 2: Write README.md**

```markdown
# Llama-3.2-1B Prefill Ablation (Plan 2)

Bit-exact 4-cell ablation of the production **prefill** pipeline:
`rms_gemms_rope` (6 launches) + FlashAttention (held constant) + `o_ffn`
(8 launches), at seq=2048 GEMM shapes, both single-layer and full 16-layer
scopes.

Companion docs:
- Plan 2 spec: [`ablation/docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md`](../specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md)
- Plan 1 (decode pilot): removed from repo (subsumed by full-decode study at `ablation/decode/`)
- Production profile: [`../../../docs/profile.md`](../../../docs/profile.md)

## What this measures

Four cells, identical computation, different dispatch strategy:

| Cell | What changes within each kernel-group | Adds |
|------|---------------------------------------|------|
| A | 6 + 8 separate `xrt.run()` per layer, host round-trip on every intermediate | (baseline) |
| B | + per-layer weight BOs (`static_input_indices`) | #2 |
| C | + shared intermediate BOs across separate `xrt.run()` calls | #3 |
| D | + multi-launch merging (production: 6→1 + 8→1 ELF per layer) | #1 |

FA is held constant per spec (un-mergeable). Cross-kernel-group transfers
(rms→FA, FA→o_ffn) go through host in every cell — matches production.

## Quick start

```
make compile          # one-time, ~10-15 min for 14 standalone ELFs + 2 merged + FA
make run              # 5 trials × both scopes × all 4 cells (~5-10 min)
make report           # markdown report
```

## Validation gate

Every cell's per-kernel-group output must match the committed `golden/*.npz`
fixtures bit-exactly (synthetic numpy seed=42 inputs). Cells failing the
gate suppress their timing in the report.

## Reproducibility

```
cd programming_examples/llama32_1b/ablation/prefill
make clean && make all
```

The 16-layer Cell D total wall time should be in the ballpark of
`profile.md`'s **1.27 s** production headline. The marginal deltas table
attributes how much each of optimizations #1, #2, #3 contributes to that
number for prefill specifically.

Unit tests (NPU-free):

```
python3 -m pytest tests/ -v
```

## Limitations of this plan (Plan 2-decode and Plan 2-lm-head will address)

- Prefill only — decode `o_gemv_ffn` and the LM Head L1/L8 mini-study are
  separate plans.
- FA is invariant in every cell. A potential **Plan 2.5** could ablate
  cross-kernel-group BO sharing (FA's input BOs aliased to rms_gemms_rope's
  output BOs); production doesn't currently do this.
- Synthetic weights only. No HuggingFace.

## File map

| Path | Purpose |
|------|---------|
| `specs/kernel_group.py` | Frozen dataclasses |
| `specs/{rms_gemms_rope,o_ffn}.py` | Concrete spec instances |
| `standalone_builders/` | Re-exported STANDALONES registries |
| `cells/cell_{a,b,c,d}_*.py` | Parameterized cell harnesses |
| `cells/flash_attn_const.py` | FA invariant |
| `cells/multi_layer.py` | 16-layer wrapper |
| `cells/common.py` | Compile harness, BO baton-pass helper |
| `golden/` | Two committed npz fixtures + regen script |
| `validate.py` | Parameterized bit-exact gate |
| `run_ablation.py` | Orchestrator |
| `analyze.py` | Report generator |
| `Makefile` | Convenience targets |
```

- [ ] **Step 3: Smoke test**

```bash
cd programming_examples/llama32_1b/ablation/prefill && make help
```
Expected: prints help line.

- [ ] **Step 4: Commit**

```bash
git add programming_examples/llama32_1b/ablation/prefill/Makefile \
        programming_examples/llama32_1b/ablation/prefill/README.md
git commit -m "ablation/prefill: README + Makefile"
```

---

## Task 19: End-to-end smoke + final commit

- [ ] **Step 1: Wipe build/ and run from scratch**

```bash
cd programming_examples/llama32_1b/ablation/prefill
make clean
make all
```

Expected: ~10-15 min compile, ~5-10 min run, ~1 sec report. Final report shows all 4 cells × 2 kernel-groups PASS validation, with 16-layer Cell D total in the 1.0-1.5 s range (headline confirmation).

- [ ] **Step 2: Run unit tests**

```bash
cd programming_examples/llama32_1b/ablation/prefill && python3 -m pytest tests/ -v
```

Expected: all tests pass (kernel_group_spec: 4, validation_gate: 4, parameterized_cells: variable).

- [ ] **Step 3: Verify Plan 1 isolation**

```bash
git diff llama-3.2-1B-devel..HEAD --stat -- programming_examples/llama32_1b/ablation/ | grep -v '^ programming_examples/llama32_1b/ablation/prefill/'
```

Expected: empty output (no Plan 1 files modified).

- [ ] **Step 4: Final commit (if any uncommitted artifacts)**

```bash
cd /home/jiajli/apps/mlir-air
git status
```

If clean: nothing to do. Otherwise update `.gitignore` and commit:

```bash
git commit -m "ablation/prefill: final cleanup"
```

---

## Self-Review Checklist

**Spec coverage** (against `programming_examples/llama32_1b/ablation/docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md`):

- §3 4-cell ladder for both kernel-groups → Tasks 8 (D), 12 (A), 13 (B), 14 (C) ✓
- §4 Invariants (FA constant, decode files unmodified, etc.) → Tasks 11 (FA), 19 (Plan 1 isolation check) ✓
- §5 Correctness verification (golden + per-cell + cross-cell) → Tasks 9, 10, 12-14 ✓ (cross-cell consistency re-check is in the orchestrator T16 — implementer should add a re-validation pass after timing)
- §6 Per-launch breakdown via Cell C → falls out of orchestrator T16 (records per-call write/kernel/read) + analyzer T17 (could be augmented with a per-launch breakdown table; this plan ships the JSON shape that supports it)
- §7 Host overhead → falls out of (wall - Σ(write+kernel+read)); analyzer T17 can add a row for it
- §8.1 Self-contained subdir → T1 ✓
- §8.2 KernelGroupSpec dataclass → T2 ✓
- §8.3 Standalone 1-launch ELFs → T5, T6 ✓
- §8.4 Cell-specific harness (parameterized) → T12-T14 ✓
- §8.5 Validation reuse → T10 ✓
- §8.6 Orchestrator scopes (single-layer + 16-layer) → T15 (multi_layer wrapper), T16 (orchestrator with --scope) ✓
- §9 Stats: 5 trials, drop run 1, median + range → T16 `_time_runs` ✓
- §10 Deliverable structure → matches file structure section above ✓
- §11 Out of scope → respected (no Plan 2-decode, no LM Head, no real HF weights)
- §12 Isolation strategy: worktree + Plan 1 files unmodified → T19 Step 3 verification ✓
- §13 Risks → flagged in Tasks 7 (compile time), 12 (variance), 14 (BO aliasing debug)

**Placeholder scan**: searched for "TBD", "TODO", "fill in", "implement later" — none in the plan body. The orchestrator T16 has explicit `pass` placeholders documented as "for the implementer to fill in"; this is intentional because the cell function signatures are clarified in T12-T14 and the orchestrator wires them up.

**Type consistency**: `KernelCache.naive=True` (Plan 1, already shipped), `compile_standalone_kernels(cache, group_name, registry, backend_preset)` signature consistent across T7, T12, T13, T14. `_share_bo` signature consistent with Plan 1's. `BatonLink` and `SubLaunchSpec` field names consistent across T2, T3, T4, T12-T14.

**Coverage gaps that are intentional and documented**:
- Cross-cell consistency re-check (§5 of spec) is described as belonging in T16's orchestrator but not concretely coded — implementer should add it after the per-cell validation loop.
- Per-launch breakdown table in the report is supported by the JSON shape but not rendered by the analyzer in T17. Plan 2's primary goal is the headline number; per-launch table can be added in a wash-up.
- Cell A/B/C parameterized harnesses (T12-T14) leave the layer_inputs-to-args adapter to the implementer's iteration; the spec dataclass is the contract but the concrete naming convention (e.g., `_out_buf_<name>`, `<name>_w`) needs refinement during T12.
