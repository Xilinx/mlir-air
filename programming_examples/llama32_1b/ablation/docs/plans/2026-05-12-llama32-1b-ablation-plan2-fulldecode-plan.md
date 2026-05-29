# Llama-3.2-1B Plan 2 (Full Decode) Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the 4-cell ablation ladder for the **full decode** path: 16 layers × (`rms_gemv_rope` 6 launches + `decode_attention_cpu` + `o_gemv_ffn` 8 launches) + final RMSNorm + `lm_head_gemv` 8-partition + argmax. Single decode token per timed trial, 5 trials, drop warmup. Bit-exact validation against committed goldens. Headline number directly comparable to `profile.md`'s per-token decode latency.

**Architecture:** Self-contained subdir `programming_examples/llama32_1b/ablation/decode/` (Plan 0 files at `ablation/` and Plan 1 files at `ablation/prefill/` remain byte-immutable). The 4 parameterized cell modules from Plan 1 are reused via direct import or copy; the new work is (a) `o_gemv_ffn` standalone builders + spec, (b) the per-token loop wrapper, (c) KV cache state management, (d) the `lm_head_gemv` invariant runner, (e) goldens + orchestration + report.

**Tech Stack:** Same as Plan 1 — Python 3, numpy, ml_dtypes (bfloat16), pytest, mlir-air's `XRTBackend` + `KernelCache`. Production builders imported: `build_rms_gemv_rope_module`, `build_o_gemv_ffn_module`, `build_lm_head_gemv_module` from `multi_launch_builder/`.

**Companion docs:**
- Plan 2 spec: `programming_examples/llama32_1b/ablation/docs/specs/2026-05-12-llama32-1b-ablation-plan2-fulldecode-design.md`
- Master ablation spec: removed from repo (decode pilot deleted; this full-decode study supersedes it)
- Plan 0 (decode pilot): removed from repo (subsumed by this study)
- Plan 1 (full prefill): `programming_examples/llama32_1b/ablation/docs/plans/2026-05-07-llama32-1b-ablation-plan2-prefill.md` — primary template for code patterns
- Plan 1 working code: `programming_examples/llama32_1b/ablation/prefill/` — copy-paste reference
- Plan 0 working code: removed; the standalone builder content is now inlined into `programming_examples/llama32_1b/ablation/decode/standalone_builders/rms_gemv_rope.py`
- Audience-facing summary: `programming_examples/llama32_1b/docs/ABLATION_STUDY.html`

**Branch / worktree setup:** Create a NEW worktree (e.g., `ablation-plan2-fulldecode`) from `llama-3.2-1B-devel`. Do NOT modify Plan 0/1 directories.

---

## File Structure

All paths under `programming_examples/llama32_1b/ablation/decode/` unless noted.

| File | Responsibility | Source pattern |
|------|----------------|----------------|
| `__init__.py` | Package marker | — |
| `README.md` | Methodology, run instructions, results, reproducibility | Plan 1's README |
| `Makefile` | `make compile / regen-golden / run / report / all / clean` | Plan 1's Makefile |
| `specs/__init__.py` | Package marker | — |
| `specs/kernel_group.py` | Re-export `SubLaunchSpec`, `BatonLink`, `KernelGroupSpec` from Plan 1 (single source of truth) | `from ablation.prefill.specs.kernel_group import *` |
| `specs/rms_gemv_rope.py` | Concrete spec for the 6-launch decode attention pre-block | Plan 1's `specs/rms_gemms_rope.py` adapted |
| `specs/o_gemv_ffn.py` | Concrete spec for the 8-launch decode FFN block | Plan 1's `specs/o_ffn.py` adapted (GEMV instead of GEMM, mv_k8192 for Down) |
| `standalone_builders/__init__.py` | Package marker | — |
| `standalone_builders/rms_gemv_rope.py` | Re-export Plan 0's `STANDALONES` registry | `from ablation.standalone_builders.decode_rms_gemv_rope import STANDALONES` |
| `standalone_builders/o_gemv_ffn.py` | 8 single-launch builder wrappers + `STANDALONES` registry — NEW | Plan 1's `standalone_builders/o_ffn.py` adapted |
| `cells/__init__.py` | Package marker | — |
| `cells/common.py` | Re-export Plan 1's `compile_standalone_kernels`, `_share_bo`, `_extract_public_func_name` | `from ablation.prefill.cells.common import *` |
| `cells/cell_a_naive.py` | Parameterized Cell A — direct re-export from Plan 1 | `from ablation.prefill.cells.cell_a_naive import run_cell_a, compile_cell_a` |
| `cells/cell_b_static.py` | Parameterized Cell B | re-export from Plan 1 |
| `cells/cell_c_charitable.py` | Parameterized Cell C | re-export from Plan 1 |
| `cells/cell_d_merged.py` | Wraps production `build_rms_gemv_rope_module`, `build_o_gemv_ffn_module` | Plan 1's `cell_d_merged.py` adapted |
| `cells/decode_attn_const.py` | CPU attention invariant — same Python function in every cell | NEW (Plan 1's `flash_attn_const.py` pattern) |
| `cells/lm_head_const.py` | LM head invariant — production-merged 8-partition GEMV | NEW |
| `cells/per_token_loop.py` | The end-to-end timed unit: 16 layers + final RMSNorm + LM head + argmax | NEW (Plan 1's `multi_layer.py` adapted, replacing 16-prompt-position with 1-decode-token) |
| `cells/kv_cache.py` | KV cache state init + per-trial reset | NEW |
| `golden/__init__.py` | Package marker | — |
| `golden/regen_golden.py` | One-shot Cell-D run; dumps two npz fixtures + meta json | Plan 1's regen pattern |
| `golden/golden_rms_gemv_rope_decode.npz` | Cell D output, layer 0, seed=42, current_pos=7 | Generated |
| `golden/golden_o_gemv_ffn_decode.npz` | Cell D output for o_gemv_ffn | Generated |
| `golden/golden_meta.json` | Hashes, shapes, prompt_len, current_pos | Plan 1 |
| `validate.py` | Bit-exact gate, parameterized — re-export Plan 1's `validate.py` directly | `from ablation.prefill.validate import *` |
| `run_ablation.py` | Orchestrator | Plan 1 adapted |
| `analyze.py` | JSON → markdown report | Plan 1 adapted |
| `tests/__init__.py` | Package marker | — |
| `tests/conftest.py` | Pytest sys.path setup | Plan 1 |
| `tests/test_o_gemv_ffn_spec.py` | Dataclass invariants for the new `o_gemv_ffn` spec | NEW |
| `tests/test_kv_cache_state.py` | Verifies cache initialization + per-trial reset is deterministic | NEW |
| `tests/test_validation_gate.py` | Tests against the two new decode goldens | Plan 1 adapted |

**Files NOT touched** (isolation guarantee): every file under `programming_examples/llama32_1b/ablation/` outside `decode/`. Production code under `programming_examples/llama32_1b/{kernel_builder,multi_launch_builder}/` is read-only — only imported.

---

## Phase 1 — Skeleton + reused infrastructure (Tasks 1–3)

## Task 1: Worktree + subdir skeleton + conftest

**Files:**
- Create: `programming_examples/llama32_1b/ablation/decode/` with subdirs `specs/`, `standalone_builders/`, `cells/`, `golden/`, `tests/`
- Create: 7 `__init__.py` files
- Create: `decode/tests/conftest.py`

- [ ] **Step 1: Set up worktree**

```bash
cd /home/jiajli/apps/mlir-air
git worktree add .claude/worktrees/ablation-plan2-fulldecode llama-3.2-1B-devel
cd .claude/worktrees/ablation-plan2-fulldecode
git checkout -b llama32_1b/ablation-plan2-fulldecode
```

- [ ] **Step 2: Create directory tree + package markers**

```bash
DECODE=programming_examples/llama32_1b/ablation/decode
mkdir -p $DECODE/{specs,standalone_builders,cells,golden,tests}
for d in "" /specs /standalone_builders /cells /golden /tests; do
    touch $DECODE$d/__init__.py
done
```

- [ ] **Step 3: Write conftest.py**

`programming_examples/llama32_1b/ablation/decode/tests/conftest.py`:

```python
"""Pytest config for full-decode ablation tests.

Inserts paths so tests can import:
- llama32_1b/ packages (kernel_builder, multi_launch_builder)
- llama32_1b/ablation/ (Plan 0's standalone_builders + validate.py)
- llama32_1b/ablation/prefill/ (Plan 1's cells, specs, common helpers)
- llama32_1b/ablation/decode/ (this package)
- programming_examples/ (matvec, weighted_rms_norm, ffn_swiglu)
"""

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_DECODE = os.path.dirname(_THIS)
_ABLATION = os.path.dirname(_DECODE)
_LLAMA = os.path.dirname(_ABLATION)
_PROG_EXAMPLES = os.path.dirname(_LLAMA)

for p in (_PROG_EXAMPLES, _LLAMA, _ABLATION, os.path.join(_ABLATION, "prefill"), _DECODE):
    if p not in sys.path:
        sys.path.insert(0, p)
```

- [ ] **Step 4: Verify imports work**

```bash
cd programming_examples/llama32_1b/ablation/decode
python3 -c "import sys; sys.path.insert(0, '.'); sys.path.insert(0, '..'); from ablation.prefill.specs.kernel_group import KernelGroupSpec; print('OK')"
```

Expected: prints `OK` (Plan 1's KernelGroupSpec dataclass loads).

- [ ] **Step 5: Commit**

```bash
git add programming_examples/llama32_1b/ablation/decode
git commit -m "ablation-decode: skeleton subdir + package markers + conftest"
```

## Task 2: Re-exports — kernel_group, common, validate

**Files:**
- Create: `decode/specs/kernel_group.py`
- Create: `decode/cells/common.py`
- Create: `decode/validate.py`
- Create: `decode/cells/cell_a_naive.py`, `cell_b_static.py`, `cell_c_charitable.py` (re-exports)

- [ ] **Step 1: Re-export the spec dataclasses**

`decode/specs/kernel_group.py`:

```python
"""Re-export Plan 1's KernelGroupSpec dataclasses (single source of truth)."""

from ablation.prefill.specs.kernel_group import (
    SubLaunchSpec,
    BatonLink,
    KernelGroupSpec,
)

__all__ = ["SubLaunchSpec", "BatonLink", "KernelGroupSpec"]
```

- [ ] **Step 2: Re-export the common helpers**

`decode/cells/common.py`:

```python
"""Re-export Plan 1's common helpers."""

from ablation.prefill.cells.common import (
    compile_standalone_kernels,
    _share_bo,
    _extract_public_func_name,
    standalone_backend_kwargs,
)

__all__ = [
    "compile_standalone_kernels",
    "_share_bo",
    "_extract_public_func_name",
    "standalone_backend_kwargs",
]
```

- [ ] **Step 3: Re-export the validate gate**

`decode/validate.py`:

```python
"""Re-export Plan 1's parameterized bit-exact validation gate."""

from ablation.prefill.validate import (
    validate_against_golden,
    GoldenMismatch,
)

__all__ = ["validate_against_golden", "GoldenMismatch"]
```

- [ ] **Step 4: Re-export Cells A/B/C (parameterized — work for any KernelGroupSpec)**

`decode/cells/cell_a_naive.py`:

```python
"""Re-export Plan 1's parameterized Cell A — same code, decode spec at call site."""

from ablation.prefill.cells.cell_a_naive import run_cell_a, compile_cell_a

__all__ = ["run_cell_a", "compile_cell_a"]
```

(Same pattern for `cell_b_static.py` and `cell_c_charitable.py`.)

- [ ] **Step 5: Smoke test the re-exports**

```bash
cd programming_examples/llama32_1b/ablation/decode
python3 -c "from cells.cell_a_naive import run_cell_a; from validate import validate_against_golden; print('imports OK')"
```

- [ ] **Step 6: Commit**

```bash
git add programming_examples/llama32_1b/ablation/decode
git commit -m "ablation-decode: re-export Plan 1's KernelGroupSpec, helpers, validate, cells A-C"
```

## Task 3: Re-export rms_gemv_rope standalone builders from Plan 0

**Files:**
- Create: `decode/standalone_builders/rms_gemv_rope.py`

- [ ] **Step 1: Write the re-export**

`decode/standalone_builders/rms_gemv_rope.py`:

```python
"""Re-export Plan 0's existing decode_rms_gemv_rope standalone builders.

Plan 0 already built 6 single-launch wrappers for rms_gemv_rope's sub-launches.
Plan 2 reuses them verbatim.
"""

from ablation.standalone_builders.decode_rms_gemv_rope import STANDALONES

__all__ = ["STANDALONES"]
```

- [ ] **Step 2: Verify**

```bash
cd programming_examples/llama32_1b/ablation/decode
python3 -c "from standalone_builders.rms_gemv_rope import STANDALONES; assert len(STANDALONES) == 6; print('rms_gemv_rope STANDALONES re-exported, count =', len(STANDALONES))"
```

Expected: prints `rms_gemv_rope STANDALONES re-exported, count = 6`

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: re-export rms_gemv_rope STANDALONES from Plan 0"
```

---

## Phase 2 — New work for o_gemv_ffn (Tasks 4–6)

## Task 4: o_gemv_ffn KernelGroupSpec

**Files:**
- Create: `decode/specs/o_gemv_ffn.py`

This spec describes the 8 sub-launches of `o_gemv_ffn`: O GEMV, eltwise add (residual #1), RMSNorm, Gate GEMV, Up GEMV, SwiGLU (silu_and_mul), Down GEMV (uses `mv_k8192.o`), eltwise add (residual #2). Slot semantics + baton links for Cell C aliasing.

- [ ] **Step 1: Write the failing test first**

`tests/test_o_gemv_ffn_spec.py`:

```python
"""Validate the o_gemv_ffn KernelGroupSpec structure."""

from specs.o_gemv_ffn import O_GEMV_FFN_SPEC


def test_spec_has_8_sublaunches():
    assert len(O_GEMV_FFN_SPEC.sub_launches) == 8


def test_sublaunch_names_match_production_order():
    names = [s.name for s in O_GEMV_FFN_SPEC.sub_launches]
    assert names == [
        "o_gemv", "add_attn_residual", "ffn_rmsnorm",
        "gate_gemv", "up_gemv", "swiglu",
        "down_gemv_k8192", "add_ffn_residual",
    ]


def test_baton_links_cover_all_intermediate_handoffs():
    """Every intermediate output must have a baton link to the next consumer."""
    # 7 intermediates × 1 producer-consumer link each (linear chain except the gate→swiglu and up→swiglu fork)
    # Detailed expected: o_gemv→add_attn, add_attn→ffn_rmsnorm, ffn_rmsnorm→{gate,up,save_residual},
    # gate→swiglu, up→swiglu, swiglu→down_gemv, down_gemv→add_ffn
    expected_links = [...]
    assert sorted(O_GEMV_FFN_SPEC.baton_links) == sorted(expected_links)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd programming_examples/llama32_1b/ablation/decode
python3 -m pytest tests/test_o_gemv_ffn_spec.py -v
```

Expected: ImportError or test failure (spec doesn't exist yet).

- [ ] **Step 3: Write the spec**

`decode/specs/o_gemv_ffn.py`:

```python
"""KernelGroupSpec for the 8-launch o_gemv_ffn decode kernel-group.

Production: rms_gemms_rope's sister for the second half of a decode layer.
Stitched into one ELF in production (Cell D); Cell A/B/C run all 8 as
separate xrt.run() calls.
"""

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec
# (Concrete instance follows. Mirror structure from prefill/specs/o_ffn.py
# but adapt for GEMV (single-token) shapes and the mv_k8192 down-step.)

O_GEMV_FFN_SPEC = KernelGroupSpec(
    name="o_gemv_ffn",
    sub_launches=[
        # ... 8 SubLaunchSpec entries ...
    ],
    baton_links=[
        # ... intermediate handoff edges ...
    ],
)
```

(Full content needs careful adaptation of Plan 1's `o_ffn` spec to single-token GEMV shapes — a ~200-line file.)

- [ ] **Step 4: Run test to confirm pass**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add specs/o_gemv_ffn.py tests/test_o_gemv_ffn_spec.py
git commit -m "ablation-decode: o_gemv_ffn KernelGroupSpec + tests"
```

## Task 5: rms_gemv_rope KernelGroupSpec

**Files:**
- Create: `decode/specs/rms_gemv_rope.py`

The 6-sub-launch spec for the decode attention pre-block. Plan 0 had standalone builders but never wrote a formal `KernelGroupSpec` — Plan 1's `KernelGroupSpec` dataclass post-dates Plan 0. Now we need one for the parameterized cell harnesses.

- [ ] **Step 1: Write spec**

`decode/specs/rms_gemv_rope.py`:

```python
"""KernelGroupSpec for the 6-launch rms_gemv_rope decode kernel-group."""

from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec

RMS_GEMV_ROPE_SPEC = KernelGroupSpec(
    name="rms_gemv_rope",
    sub_launches=[
        # rmsnorm, q_gemv, k_gemv, v_gemv, rope_q, rope_k
    ],
    baton_links=[
        # rmsnorm→q_gemv, rmsnorm→k_gemv, rmsnorm→v_gemv
        # q_gemv→rope_q, k_gemv→rope_k
    ],
)
```

(Reference Plan 0's `cells/cell_a_naive.py` for the slot/argument layout.)

- [ ] **Step 2: Smoke test it loads**

```bash
python3 -c "from specs.rms_gemv_rope import RMS_GEMV_ROPE_SPEC; assert len(RMS_GEMV_ROPE_SPEC.sub_launches) == 6"
```

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: rms_gemv_rope KernelGroupSpec"
```

## Task 6: o_gemv_ffn standalone builders

**Files:**
- Create: `decode/standalone_builders/o_gemv_ffn.py`

8 single-launch MLIR builder wrappers, one per sub-launch of `o_gemv_ffn`. Mirror Plan 1's `standalone_builders/o_ffn.py` but for GEMV (single-token, M=1) shapes.

- [ ] **Step 1: Write builders**

`decode/standalone_builders/o_gemv_ffn.py`:

```python
"""8 single-launch builder wrappers for o_gemv_ffn sub-launches.

Each builder produces a full MLIR module containing ONE air.launch.
Used by Cells A/B/C (separate xrt.run() per sub-launch).
Cell D uses the production merged build_o_gemv_ffn_module instead.
"""

from ml_dtypes import bfloat16
import numpy as np

from matvec.run import build_module as _build_matvec
from weighted_rms_norm.weighted_rms_norm import build_module as _build_rmsnorm
from ffn_swiglu.silu_and_mul import build_module as _build_swiglu
from eltwise_add.eltwise_add import build_module as _build_add
# Reuse multi_launch_builder/o_gemv_ffn_multi.py's _build_add_2d_to_1d if needed.

def build_o_gemv():    ...  # 1 air.launch wrapping the O GEMV
def build_add_attn_residual(): ...  # 1 air.launch wrapping eltwise add (2D)
def build_ffn_rmsnorm(): ...
def build_gate_gemv(): ...
def build_up_gemv(): ...
def build_swiglu(): ...
def build_down_gemv_k8192(): ...  # uses dg_matvec_vectorized_bf16_bf16 (renamed K=8192 variant)
def build_add_ffn_residual(): ...

STANDALONES = {
    "o_gemv": build_o_gemv,
    "add_attn_residual": build_add_attn_residual,
    "ffn_rmsnorm": build_ffn_rmsnorm,
    "gate_gemv": build_gate_gemv,
    "up_gemv": build_up_gemv,
    "swiglu": build_swiglu,
    "down_gemv_k8192": build_down_gemv_k8192,
    "add_ffn_residual": build_add_ffn_residual,
}
```

- [ ] **Step 2: Smoke test each builder produces a parseable MLIR module (NPU-free)**

```bash
python3 -c "
from standalone_builders.o_gemv_ffn import STANDALONES
for name, build_fn in STANDALONES.items():
    mod = build_fn()  # signature TBD per kernel
    assert mod is not None
    print(f'{name}: ok')
"
```

- [ ] **Step 3: Commit**

```bash
git add standalone_builders/o_gemv_ffn.py
git commit -m "ablation-decode: 8 standalone builders for o_gemv_ffn sub-launches"
```

---

## Phase 3 — Decode-specific orchestration (Tasks 7–10)

## Task 7: KV cache initialization + per-trial reset

**Files:**
- Create: `decode/cells/kv_cache.py`
- Create: `tests/test_kv_cache_state.py`

- [ ] **Step 1: Write the failing test**

`tests/test_kv_cache_state.py`:

```python
"""KV cache state must be deterministic and resettable per trial."""

import numpy as np
from cells.kv_cache import build_initial_kv_cache, reset_position


def test_initial_cache_is_deterministic():
    cfg = {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64, "max_seq": 2048}
    c1 = build_initial_kv_cache(cfg, prompt_len=7, seed=42)
    c2 = build_initial_kv_cache(cfg, prompt_len=7, seed=42)
    np.testing.assert_array_equal(c1["k_cache"], c2["k_cache"])
    np.testing.assert_array_equal(c1["v_cache"], c2["v_cache"])


def test_reset_position_clears_target_slot():
    cfg = {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64, "max_seq": 2048}
    cache = build_initial_kv_cache(cfg, prompt_len=7, seed=42)
    cache["k_cache"][0, :, 7, :] = 99.0  # simulate write
    reset_position(cache, 7)
    assert (cache["k_cache"][0, :, 7, :] == 0).all()
    # positions 0-6 untouched
    assert not (cache["k_cache"][0, :, :7, :] == 0).all()
```

- [ ] **Step 2: Implement**

`decode/cells/kv_cache.py`:

```python
"""KV cache state management for the per-token timed loop.

Two functions:
- build_initial_kv_cache: deterministic synthetic pre-fill of `prompt_len` positions
- reset_position: zero out a specific position (called between trials)
"""

import numpy as np
from ml_dtypes import bfloat16


def build_initial_kv_cache(config, prompt_len, seed):
    """Pre-fill the KV cache with synthetic deterministic values."""
    rng = np.random.default_rng(seed)
    shape = (config["n_layers"], config["n_kv_heads"], config["max_seq"], config["head_dim"])
    k = np.zeros(shape, dtype=bfloat16)
    v = np.zeros(shape, dtype=bfloat16)
    k[:, :, :prompt_len, :] = rng.standard_normal(
        (config["n_layers"], config["n_kv_heads"], prompt_len, config["head_dim"])
    ).astype(bfloat16) * 0.5
    v[:, :, :prompt_len, :] = rng.standard_normal(
        (config["n_layers"], config["n_kv_heads"], prompt_len, config["head_dim"])
    ).astype(bfloat16) * 0.5
    return {"k_cache": k, "v_cache": v, "current_pos": prompt_len}


def reset_position(cache, pos):
    """Zero out the K/V cache slots at `pos` for ALL layers."""
    cache["k_cache"][:, :, pos, :] = 0
    cache["v_cache"][:, :, pos, :] = 0
```

- [ ] **Step 3: Run tests**

Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add cells/kv_cache.py tests/test_kv_cache_state.py
git commit -m "ablation-decode: KV cache init + per-trial reset (tested deterministic)"
```

## Task 8: Decode CPU attention invariant runner

**Files:**
- Create: `decode/cells/decode_attn_const.py`

Wraps the production `decode_attention_cpu` from `llama32_1b_decode.py:96` so all 4 cells call exactly the same Python function.

- [ ] **Step 1: Write**

`decode/cells/decode_attn_const.py`:

```python
"""Invariant CPU attention runner — same Python function in every cell."""

import time
from llama32_1b_decode import decode_attention_cpu


def run_decode_attention(cache, q_roped, k_roped, v, layer_idx, current_pos, config):
    """Run CPU attention; update KV cache slot at current_pos.

    Returns: (attn_out, elapsed_seconds)
    """
    t0 = time.perf_counter()
    attn_out = decode_attention_cpu(
        q_roped, k_roped, v,
        cache["k_cache"][layer_idx],
        cache["v_cache"][layer_idx],
        current_pos,
        config["n_heads"], config["n_kv_heads"], config["head_dim"],
    )
    elapsed = time.perf_counter() - t0
    return attn_out, elapsed
```

- [ ] **Step 2: Smoke test (NPU-free, dummy inputs)**

```bash
python3 -c "
from cells.decode_attn_const import run_decode_attention
import numpy as np
from ml_dtypes import bfloat16
# Construct minimal dummy cache + activation tensors and verify it runs
# ...
print('decode_attn_const runs')
"
```

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: invariant CPU attention runner"
```

## Task 9: LM head invariant runner

**Files:**
- Create: `decode/cells/lm_head_const.py`

Production `lm_head_gemv` is one merged ELF (8 stitched partitions); held INVARIANT in every cell.

- [ ] **Step 1: Write**

```python
"""Invariant LM head runner — production-merged 8-partition GEMV in every cell."""

import time
import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import LM_GEMV_BACKEND
from multi_launch_builder.lm_head_gemv_multi import build_lm_head_gemv_module


def compile_lm_head(cache: KernelCache, config):
    """Compile the production LM head ELF (one-time)."""
    if "lm_head_gemv" in cache.artifacts:
        return
    mod = build_lm_head_gemv_module(...)  # production args
    cache.compile_and_cache("lm_head_gemv", mod, {**LM_GEMV_BACKEND, "verbose": cache.verbose})


def run_lm_head(cache, x_normed, weights, vocab_size):
    """Run LM head; return (next_token_id, elapsed_seconds)."""
    t0 = time.perf_counter()
    # ... mirror production code from llama32_1b_inference.py:434-446 ...
    elapsed = time.perf_counter() - t0
    return next_token, elapsed
```

- [ ] **Step 2: Commit**

```bash
git commit -am "ablation-decode: invariant LM head runner"
```

## Task 10: Per-token loop wrapper (the timed unit)

**Files:**
- Create: `decode/cells/per_token_loop.py`

Wraps a per-layer triple in a 16-layer loop, then runs final RMSNorm + LM head + argmax. **This is the per-trial timed unit.**

- [ ] **Step 1: Write**

```python
"""Per-token decode loop wrapper.

Each call generates ONE decode token at the given current_pos. Cell-specific
dispatch is injected via run_rms_gemv_rope and run_o_gemv_ffn function args.
CPU attention and LM head are invariant.

Returns:
    {
        "next_token": int,
        "per_layer_npu_wall": list of 16 floats (sum of rms_gemv_rope + o_gemv_ffn per layer),
        "cpu_attn_wall": float (sum across 16 layers),
        "lm_head_wall": float,
        "total_wall": float (everything inside the timer),
    }
"""

import time
import numpy as np
from ml_dtypes import bfloat16

from cells.decode_attn_const import run_decode_attention
from cells.lm_head_const import run_lm_head


def run_one_decode_token(
    cache, config, weights, kv_cache,
    x_decode, current_pos,
    run_rms_gemv_rope, run_o_gemv_ffn,
):
    n_layers = config["n_layers"]
    per_layer_npu = []
    cpu_attn_total = 0.0
    x = x_decode

    t_total_start = time.perf_counter()
    for L in range(n_layers):
        # Per-layer timing
        rg_out = run_rms_gemv_rope(cache, layer_inputs={...}, layer_idx=L)
        attn_out, attn_t = run_decode_attention(
            kv_cache, rg_out["q_roped"], rg_out["k_roped"], rg_out["v"],
            layer_idx=L, current_pos=current_pos, config=config,
        )
        cpu_attn_total += attn_t
        of_out = run_o_gemv_ffn(cache, layer_inputs={...}, layer_idx=L)
        x = of_out["output"]
        per_layer_npu.append(rg_out["_wall_s"] + of_out["_wall_s"])

    # Final RMSNorm (CPU)
    from llama32_1b_cpu_helpers import rms_norm
    x_normed = rms_norm(x.astype(np.float32).reshape(1, config["emb_dim"]),
                         weights.final_norm.astype(np.float32)).flatten().astype(bfloat16)
    next_token, lm_head_t = run_lm_head(cache, x_normed, weights, config["vocab_size"])

    return {
        "next_token": next_token,
        "per_layer_npu_wall": per_layer_npu,
        "cpu_attn_wall": cpu_attn_total,
        "lm_head_wall": lm_head_t,
        "total_wall": time.perf_counter() - t_total_start,
    }
```

- [ ] **Step 2: Smoke test (NPU-free with mock dispatch)**

Mock `run_rms_gemv_rope` and `run_o_gemv_ffn` to return zeros + dummy wall times. Verify the wrapper completes 16 iterations.

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: per-token loop wrapper (timed unit)"
```

---

## Phase 4 — Cell D + goldens (Tasks 11–13)

## Task 11: Cell D — production merged ELFs

**Files:**
- Create: `decode/cells/cell_d_merged.py`

Compiles and runs the production `rms_gemv_rope.elf` and `o_gemv_ffn.elf`. Mirror Plan 0's `cell_d_merged.py` and Plan 1's `cell_d_merged.py`.

- [ ] **Step 1: Write**

```python
"""Cell D — production-merged decode ELFs.

Compiles and invokes:
- rms_gemv_rope.elf (6 stitched launches in 1 xrt.run)
- o_gemv_ffn.elf (8 stitched launches in 1 xrt.run)
Same pattern as production llama32_1b_decode.py.
"""

import time
import numpy as np
from ml_dtypes import bfloat16

from kernel_builder.cache import KernelCache
from kernel_builder.backend_presets import RGR_BACKEND, OGF_BACKEND
from multi_launch_builder.rms_gemv_rope_multi import build_rms_gemv_rope_module
from multi_launch_builder.o_gemv_ffn_multi import build_o_gemv_ffn_module


def compile_cell_d(cache, config):
    if "rms_gemv_rope" not in cache.artifacts:
        mod = build_rms_gemv_rope_module(...)
        cache.compile_and_cache("rms_gemv_rope", mod, {**RGR_BACKEND, "verbose": cache.verbose})
    if "o_gemv_ffn" not in cache.artifacts:
        mod = build_o_gemv_ffn_module(...)
        cache.compile_and_cache("o_gemv_ffn", mod, {**OGF_BACKEND, "verbose": cache.verbose})
    cache._save_manifest()


def run_rms_gemv_rope_d(cache, layer_inputs, layer_idx):
    """Production merged dispatch — mirror llama32_1b_decode.py:run_decode_block."""
    # ... assemble args, call cache.load_and_run("rms_gemv_rope", ...)
    # ... return {normed, q, k, v, q_roped, k_roped, _wall_s}


def run_o_gemv_ffn_d(cache, layer_inputs, layer_idx):
    """Production merged dispatch."""
    # ... call cache.load_and_run("o_gemv_ffn", ...)
    # ... return {output, _wall_s}
```

- [ ] **Step 2: Quick run on the NPU (preload + 1 trial) to verify it doesn't crash**

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "
# Compile + run Cell D once with synthetic inputs
# ...
print('Cell D OK')
"
```

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: Cell D production-merged decode dispatches"
```

## Task 12: Generate goldens

**Files:**
- Create: `decode/golden/regen_golden.py`
- Create: `decode/golden/golden_rms_gemv_rope_decode.npz` (generated)
- Create: `decode/golden/golden_o_gemv_ffn_decode.npz` (generated)
- Create: `decode/golden/golden_meta.json` (generated)

- [ ] **Step 1: Write the regen script**

```python
"""Regenerate the two committed golden fixtures from Cell D.

Usage:
    flock -x -w 1800 /tmp/mlir-air-npu.lock python3 golden/regen_golden.py
"""

import json
import hashlib
import numpy as np

# ... synthetic seed=42 inputs (mirror Plan 0/1 golden gen)
# ... run Cell D for layer 0, current_pos=7
# ... save outputs to npz
# ... write golden_meta.json with hashes, shapes, prompt_len, current_pos
```

- [ ] **Step 2: Run on NPU and commit the goldens**

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock python3 golden/regen_golden.py
git add golden/golden_rms_gemv_rope_decode.npz \
        golden/golden_o_gemv_ffn_decode.npz \
        golden/golden_meta.json \
        golden/regen_golden.py
git commit -m "ablation-decode: regen + commit Cell D goldens"
```

## Task 13: Validation gate test against new goldens

**Files:**
- Create: `tests/test_validation_gate.py`

- [ ] **Step 1: Write the test**

```python
"""Verify Plan 1's validate.py works against the new decode goldens."""

import os

import numpy as np
from validate import validate_against_golden, GoldenMismatch

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden")


def test_validate_passes_on_golden_self():
    """Loading the golden and validating it against itself must pass."""
    npz = np.load(os.path.join(GOLDEN_DIR, "golden_rms_gemv_rope_decode.npz"))
    cell_outputs = {key: npz[key] for key in npz.files}
    validate_against_golden(cell_outputs, GOLDEN_DIR,
                            golden_filename="golden_rms_gemv_rope_decode.npz")


def test_validate_fails_on_byte_diff():
    npz = np.load(os.path.join(GOLDEN_DIR, "golden_rms_gemv_rope_decode.npz"))
    cell_outputs = {key: npz[key].copy() for key in npz.files}
    cell_outputs["normed"][0] = 0  # corrupt
    try:
        validate_against_golden(cell_outputs, GOLDEN_DIR,
                                golden_filename="golden_rms_gemv_rope_decode.npz")
        assert False, "expected GoldenMismatch"
    except GoldenMismatch:
        pass
```

- [ ] **Step 2: Run**

```bash
python3 -m pytest tests/test_validation_gate.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: validation gate test"
```

---

## Phase 5 — Orchestration (Tasks 14–16)

## Task 14: run_ablation.py orchestrator

**Files:**
- Create: `decode/run_ablation.py`

For each cell: validate → 5 trials × {per-token-loop} → emit JSON. Mirror Plan 1's `run_ablation.py`.

- [ ] **Step 1: Write the orchestrator**

```python
"""Run the 4-cell full-decode ablation.

Per cell:
- Compile + preload (not timed)
- 5 trials, each: reset KV cache state → run per_token_loop → record total_wall
- Drop trial 1, median + (min, max) over trials 2-5

For each cell, also report per-kernel-group medians (rms_gemv_rope, o_gemv_ffn)
extracted from the per_token_loop's per_layer_npu_wall sums.
"""

import argparse, json, os, sys, time
import numpy as np

# ... orchestrator logic, mirror Plan 1's run_ablation.py adapted for per-token-loop
```

- [ ] **Step 2: Smoke test JSON output structure (NPU-free)**

Stub out the actual cell runs to return constant times; verify the JSON has the expected schema.

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: run_ablation.py orchestrator"
```

## Task 15: analyze.py report generator

**Files:**
- Create: `decode/analyze.py`

JSON → markdown report. Mirror Plan 1's `analyze.py`.

- [ ] **Step 1: Write**

Tables to emit:
1. **Per-token total wall** × 4 cells (median + range, Δ vs prev, speedup, vs profile.md decode latency)
2. **Per-kernel-group per-call medians** × 4 cells × {rms_gemv_rope, o_gemv_ffn}
3. **Component breakdown** per cell: NPU wall (rms_gemv_rope + o_gemv_ffn × 16) + CPU attention floor + LM head fixed cost
4. **Findings** stub (filled in manually after first run)

- [ ] **Step 2: Smoke test on the JSON schema**

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: analyze.py markdown report generator"
```

## Task 16: Makefile + README

**Files:**
- Create: `decode/Makefile`
- Create: `decode/README.md`

- [ ] **Step 1: Write Makefile**

```makefile
.PHONY: all compile regen-golden run report clean test

all: compile run report

compile:
	flock -x -w 1800 /tmp/mlir-air-npu.lock python3 -c "from cells.cell_d_merged import compile_cell_d; from kernel_builder.cache import KernelCache; cache = KernelCache(cache_dir='build', verbose=True); compile_cell_d(cache, CONFIG)"

regen-golden:
	flock -x -w 1800 /tmp/mlir-air-npu.lock python3 golden/regen_golden.py

run:
	flock -x -w 1800 /tmp/mlir-air-npu.lock python3 run_ablation.py --trials 5 --out results.json

report:
	python3 analyze.py results.json > report.md

test:
	python3 -m pytest tests/ -v

clean:
	rm -rf build *.json report.md
```

- [ ] **Step 2: Write README**

Mirror Plan 1's README structure: methodology, headline numbers (TBD until run), reproducibility, file map, limitations.

- [ ] **Step 3: Commit**

```bash
git commit -am "ablation-decode: Makefile + README"
```

---

## Phase 6 — Run + analyze + integrate (Tasks 17–18)

## Task 17: First end-to-end NPU run

- [ ] **Step 1: Compile**

```bash
cd programming_examples/llama32_1b/ablation/decode
flock -x -w 1800 /tmp/mlir-air-npu.lock make compile
```

Expected: ~5 min, no errors.

- [ ] **Step 2: Run**

```bash
flock -x -w 1800 /tmp/mlir-air-npu.lock make run
cat results.json | python3 -m json.tool | head -40
```

Expected: 4 cells reported with `validation: PASS`, per-token medians in the ms-to-tens-of-ms range, Cell D's per-token median in the ballpark of `profile.md`'s decode latency.

- [ ] **Step 3: Generate report**

```bash
make report
cat report.md
```

- [ ] **Step 4: Sanity checks**

- All 4 cells PASS validation? If not, debug before continuing.
- Within-cell range (min/max) is small (<5% of median)?
- A→D speedup is >1× (otherwise something is wrong)?
- Cell D ≈ profile.md decode latency (within ~20%)?

- [ ] **Step 5: Commit results**

```bash
git add results.json report.md
git commit -m "ablation-decode: first end-to-end run + report"
```

## Task 18: Update ABLATION_STUDY.html with Plan 2 results

**Files:**
- Modify: `programming_examples/llama32_1b/docs/ABLATION_STUDY.html`

- [ ] **Step 1: Update Section 5.1 status**

Change the planned-card from "📋 PLANNED" to "✅ Implemented + measured (date)".

- [ ] **Step 2: Add Section 5.4 (Results — Plan 2: full decode)**

Mirror Section 4.3 structure:
- Per-token total wall table (4 cells, median, range, Δ vs prev, speedup, vs profile.md)
- Per-kernel-group per-call medians using the `cmp-table` styling
- Component breakdown (CPU floor, LM head fixed cost, dispatch-affected NPU work)
- Findings ul (3-5 bullet points based on actual numbers)

- [ ] **Step 3: Update Section 6.1 (cross-comparison)**

Replace "decode vs. prefill (so far)" with three-way comparison: Plan 0 (single-kernel-group decode) vs Plan 1 (full prefill) vs Plan 2 (full decode). New row in the optimization-effect table for each.

- [ ] **Step 4: Update Quick recap at bottom**

Change the Plan 2 entry from "designed only, not yet measured" to "A→D = X.XX×, headline finding ..."

- [ ] **Step 5: Sidebar nav update if needed (probably no change since 5.1/5.2/5.3 still exist + new 5.4)**

- [ ] **Step 6: Render-verify in headless Chromium**

```bash
python3 - <<'EOF'
from playwright.sync_api import sync_playwright
HTML = "/path/to/ABLATION_STUDY.html"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context().new_page()
    pg.goto(f"file://{HTML}")
    # Screenshot key sections to verify rendering
    ...
EOF
```

- [ ] **Step 7: Commit + push**

```bash
git add programming_examples/llama32_1b/docs/ABLATION_STUDY.html
git commit -m "ABLATION_STUDY: Plan 2 (full decode) results integrated"
```

---

## Done definition

- [ ] All 4 cells produce bit-identical outputs against committed goldens (validation PASS)
- [ ] Per-token median for Cell D is within ~20% of `profile.md`'s decode per-token latency
- [ ] Per-kernel-group medians for `rms_gemv_rope` are consistent with Plan 0's pilot (allowing for slight differences from running inside the per-token loop vs. standalone)
- [ ] All NPU-free unit tests pass (`pytest tests/ -v`)
- [ ] `report.md` generated with the 4 cells' numbers + speedup attribution
- [ ] `ABLATION_STUDY.html` updated with Section 5.4 results + Section 6.1 three-way comparison
- [ ] All work on a separate branch / worktree so Plan 0 and Plan 1 directories remain byte-immutable
- [ ] PR-ready: README, Makefile, tests, results.json, report.md all in the new `ablation/decode/` subdir

---

## Estimated effort

- **Tasks 1-3 (skeleton + re-exports):** 30 min
- **Tasks 4-6 (specs + standalone builders for o_gemv_ffn):** 4-6 hours (the most non-trivial work, especially the K=8192 down GEMV variant)
- **Tasks 7-10 (decode-specific orchestration):** 3-4 hours
- **Tasks 11-13 (Cell D + goldens):** 2-3 hours (includes NPU compile time)
- **Tasks 14-16 (orchestration + report + Makefile):** 2 hours
- **Task 17 (first run + sanity check):** 1 hour (mostly NPU lock + verification)
- **Task 18 (HTML integration):** 1-2 hours

**Total: ~14-19 hours of focused work + ~1-2 hours of NPU lock time**, comparable to Plan 1's prefill effort.

If subagent-driven-development is used, expect roughly half a day of controller-time + ~3-5 hours of subagent execution time per task with two-stage review.
