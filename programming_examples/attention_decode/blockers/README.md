# attention_decode kernel-inline blockers

Standalone minimal e2e tests for the three remaining extern kernels in
`attn_decode_npu2.cc` that resisted inline-MLIR conversion.

Each test isolates the suspected blocker so it can be debugged
independently of the surrounding example, filed as a repro upstream, or
re-evaluated as the toolchain evolves.

| File | Blocker | Status |
|---|---|---|
| `blocker_rope_v2.py` | `arith.subf(mulf, mulf)` + `vector.fma(_, _, mulf)` in same body | **PASSES standalone** — failure in attn_decode is contextual |
| `blocker_rope_v3.py` | Same with hoisted vectors + 5 rows of subview reads/writes | **PASSES standalone** |
| `blocker_sin_mask.py` | Vector mask construction needed for sin/cos polynomial | **FAILS at compile** (concrete blockers) |
| `blocker_attn1_perf.py` | 4-acc unroll vs naive single-chain inline form | Both compile to **identical ELFs** — perf needs hardware timing |
| `sanity_copy.py` | 1-in 1-out vector copy (harness sanity check) | PASSES |
| `sanity_2in.py` | 2-in 1-out vector mul (harness sanity check) | PASSES |
| `sanity_3in.py` | 3 distinct L3 input memrefs (hits the 2-S2MM/tile DMA channel limit) | FAILS — keep inputs ≤ 2 or pack |

## How to run

Each test takes a `--variant`/`--step` flag. From this directory:

```bash
source ../../utils/env_setup.sh ...   # standard mlir-air env
mkdir -p tmp_run && cd tmp_run

# Confirm the rope arithmetic pattern is OK in isolation:
python3 ../blocker_rope_v2.py --variant subf_fma     # PASSES
python3 ../blocker_rope_v3.py --variant subf_fma     # PASSES

# Reproduce the sinf/cosf blockers:
python3 ../blocker_sin_mask.py --step fptosi   # FAIL: G_SITOFP <16 x s32> -> <16 x s16>
python3 ../blocker_sin_mask.py --step andi     # FAIL: aievec.band illegal (no AIEVecToLLVM lowering)

# attn_1 4-acc inline (both compile, perf needs separate timing harness):
python3 ../blocker_attn1_perf.py --variant unroll4 --n-tokens 64
python3 ../blocker_attn1_perf.py --variant naive   --n-tokens 64
```

## Findings per blocker

### 1. `shuffle_apply_rope_bf16_64` — contextual failure, not a clean MLIR bug

The arithmetic pattern (`x1*c - x2*s` + `x1*s + x2*c`) **passes correctness in
both standalone tests** (`v2` packed-input, `v3` multi-row hoisted-vector form).
The failure observed when inlining into `attn_decode_npu2.py` (all-zero output)
must be triggered by additional surrounding state we have not yet isolated:

- 8 herd cores running concurrently (`sizes=[NKV=8, 1]`)
- `c_data` aliased through vecmat → rope → copy_scale_q → KV writeback
- Larger ELF / register pressure when combined with vecmat + softmax + attn

**Next debug step:** add ops to v3 incrementally (more rows, dummy compute,
sized arrays) until the standalone test starts failing too. Then upstream the
minimal repro.

### 2. `sinf_bf16_32_16` / `cosf_bf16_32_16` — concrete MLIR/Peano blockers

Two distinct hard blockers, both demonstrated:

**Blocker A — `aievec.band` has no LLVM lowering.**
`arith.andi` on `vector<16xi32>` lowers to `aievec.band` via
`ComputeBandAndBorOpPattern` (mlir-aie `VectorToAIEVecConversions.cpp:3957`),
but `AIEVecToLLVM/` has **no pattern for `aievec.band`** — only the chess C++
emitter (`TranslateAIEVecToCpp.cpp:1763`) handles it. So bitwise vector ops
(needed for the per-lane quadrant mask) cannot reach an ELF via Peano.

```
air_project/npu.air.mlir:49:16: error: failed to legalize operation
  'aievec.band' that was explicitly marked illegal:
  %25 = "aievec.band"(%24, %7) : (vector<16xi32>, vector<16xi32>) -> vector<16xi32>
```

**Blocker B — `arith.sitofp` from i32 vector to bf16 vector not legalized by Peano.**
Even when avoiding the bf16-direct fptosi (`bf16 → extf → f32 → fptosi → i32`
works), the return trip `i32 → sitofp → bf16` (or `f32 → bf16` after sitofp)
hits the GISel legalizer:

```
LLVM ERROR: unable to legalize instruction:
  %16:_(<16 x s16>) = G_SITOFP %15:_(<16 x s32>) (in function: core_0_2)
```

**Workaround paths (all costly):**

- Restructure the polynomial to avoid integer masks (use `math.floor` +
  `arith.cmpf` on bf16/f32 instead of bit-twiddling). Algorithmic rewrite.
- Add `aievec.band` → `llvm.and` lowering and i32→bf16 sitofp legalization
  to mlir-aie / llvm-aie source. Upstream contribution.
- Stay all-f32 throughout the kernel (extf at entry, truncf at the very end).
  Doesn't fix the bitwise-op blocker.

### 3. `attn_1_group` / `attn_2_group` — perf risk only

Both inline-MLIR forms (`naive` single-chain and `unroll4` 4-acc-chain)
**compile cleanly to correct results**, and produce **identical ELF .text**
size. That implies either Peano auto-pipelines the naive form to match
unroll4, or both forms compile to the same conservative sequential code.

**Next debug step:** wire this test into the `profile-decode` style timing
harness so we can A/B the cycles per token. Until then, keeping the extern
is the safe choice — the C++ form is known to deliver 553 µs vs 1170 µs (2x)
for this kernel in the full attn_decode pipeline.

## Why these tests live in the example tree

These are debug/diagnostic scripts, not production examples — they are
deliberately not registered in `programming_examples/generate_readme.py`.
Their purpose is to make each failure mode reproducible in isolation so
that:

- A minimal repro can be filed against mlir-aie / llvm-aie when an
  upstream fix is needed.
- We can re-run them after toolchain bumps to detect when a blocker is
  resolved.
- Future contributors can see exactly what was tried and what failed
  before adding new workarounds.
