# DFlash on NPU2 — feasibility

Written 2026-08-21, updated 2026-08-22, against mlir-air `32ef2677`. Excluded
from the published site (`exclude_docs` in `mkdocs.yml`).

Every number below is tagged **[measured]** (taken from a run or read out of the
tree) or **[estimated]** (derived from those). No untagged numbers.

---

## State of play

**DFlash is still worth building. The block size is 4-8, not 16, and the
speedup is ~1.6-2.8x, not 4.9x.**

**Build state, in one line: the batch-8 engine RUNS on device and every gate on
it is green.** A batch-8 llama-3.2-1b dispatch completes all 16 decode layers,
all eight tokens' K and V land, and **every token agrees with a batch-1 dispatch
at its own position** — not just token 0 **[measured]**:

    token 0 (L 1)  6.72e-03      token 4 (L 5)  7.87e-03
    token 1 (L 2)  7.59e-03      token 5 (L 6)  7.89e-03
    token 2 (L 3)  7.78e-03      token 6 (L 7)  7.88e-03
    token 3 (L 4)  7.94e-03      token 7 (L 8)  8.03e-03

The error grows smoothly with the context each token attends over, which is the
shape it should have and is the evidence that token `t` really is getting `t+1`
positions and not somebody else's. The row-map probe — which feeds row `t` the
constant `(t+1)/8` and reads the ratios back out of the KV cache — says every
row of the mmul saw its own token **[measured]**:

    role 1:  1.000  2.000  3.000  4.000  5.000  6.000  7.000  8.000
    role 0:  1.000  2.000  3.000  4.000  4.995  6.000  7.000  8.000

The last fault was **not** a permutation and not a descriptor. mlir-aie aligns a
compute-tile buffer to the tile's load/store bus width — 256 bits, **32 bytes**,
on AIE2p — and `aie::mmul<8,8,8>`'s C tile is 64 floats that Peano moves in
512-bit chunks, which need 64. One odd-sized buffer (the 528-bf16 shared egress
block, 16.5 × 64) pushed the proj **lead** tiles' second accumulator to `…820`,
and a misaligned 512-bit access on AIE2 does not fault — it masks the address.
The whole accumulator landed 8 floats low with its tail never written. See "Next
step"; `l1_align.py` is the gate.

What remains is the host driver at batch > 1, and the block-size decision below.

Section 5's roofline counted projection weight traffic and left attention out.
Attention is the one term that does not amortize over a batch — every query
re-reads the whole KV cache — so putting it back moves batch 16 from 1.04x the
memory floor to **3.58x** (section 5e), and sweeping the block size against
`max(compute, memory)` puts the best **measured** block at **8**, at 1.65x, with
block 16 at **1.06x** (section 5f). That conclusion survives every variant tried,
including feeding the model the undercounted attention figure that started the
investigation.

And it is an **attention** problem: at block 8 the verify pass is 45.5 ms of
projection against 100.0 ms of attention **[measured]**. Making the projections
faster cannot move this much. Making attention amortize over the batch is now
**measured and bounded**: at most **1.55x** on the attention term, **2.08x**
end to end against the 1.65x as built, and it leaves the block size at 8
(section 5g). Worth doing, and not a prerequisite for anything.

The recommendation to build it stands. What changes is the block size, the
headline, and where the optimization effort belongs.

Steps 1, 3, 4 and most of 5 of the plan in section 6 are **built and on disk**,
step 1's matmul is **numerically validated on device**, and step 2's proj kernel
is **built and gated against the GEMV it replaces**. Eight things worth knowing
before writing any more code:

- **Attention does not amortize, and it is 71% of batch-16 compute**
  **[measured]**. Every query re-reads the whole KV cache, so 16 tokens is 16x
  the attention calls where it is 1x the weight traffic. This is the finding
  that moves the answer, and it also says where the optimization effort belongs:
  attention runs at ~7 MAC/cycle against the matmul's 38.8 (section 5e).
- **Only 35% of attention is hoistable, and the transpose is not the part**
  **[measured]**. Deleting the `aie::transpose` makes llama's kernel *slower* —
  it is free, hidden in issue slots. The hoistable work is the K/V **tile
  loads**; the floor is the softmax `update` (55% of `attn_qk_blk`) and the y
  rescale pass, both per token by construction. Ceiling 1.55x on qwen3-4b, an
  upper bound and a loose one (section 5g).
- **The matmul was wrong, and is now right.** The numeric gate found two faults
  on its first run that every static check had passed — the operands were being
  fed to `aie::mmul` in the wrong roles, and `sizeof(q4k_block_t)` is not the
  size of a packed block. Both give plausible wrong answers, not crashes. Fixed;
  bit-exact now; bundle counts unchanged (section 5d).
- **The batched projection costs 1.4x the GEMV's error** and moves the
  projection output by 1.7% rms **[measured]**, both kernels run in one launch
  off the same weights. Real, small, and not yet checked against a model's
  output (section 5d).
- The new matmul needs a **16 KB unpacked-weight buffer** the current GEMV does
  not have — but it also **drops the reduce cache entirely**, since that exists
  only for the `+min` factorisation the batched path does not use (section 5b,
  section 6 step 2).
- The **X feed overtakes the weight feed at batch 10** — so at block 8 it is a
  non-issue, where at 16 it was the tightest number in the step (section 5c).
- ~~Verify needs an intra-block causal mask that does not exist.~~ **It already
  exists**, and `attn_qk_blk`'s tail mask *is* a per-query triangular mask when
  `L` varies per query. But `L` is **one RTP for the whole dispatch**, so "verify
  and draft differ by a scalar" holds only while attention is called per token.
  Batch attention into the mmul's R dimension and verify needs a per-query mask
  in the kernel; draft does not, because its `L` is uniform (section 6, step 2,
  "What the shipping q4nx models already answer"). Guarded by
  `batch_attn_mask.py --check`.
- The hidden-state taps step 3 needs were **already crossing the shim every
  layer**, just overwritten in place. Step 3 turned out to be an offset change
  and is now built behind `DECODE_HIDDEN_TAPS=1` (section 6).

Build step 2 on the **plain 32x256 kernel**. Both folds that would make it
faster are parked behind a measurement the static-bundle method cannot make
(section 5), and at block 8 the plain form has L1 to spare.

~~Before building more of step 2, settle the block size.~~ **Settled: build for
block 8** (section 5f). It is the best measured point, and it is also the
kernel's best shape by a wide margin — 71.4 MAC/cycle against batch 4's 20.6 and
batch 16's 55.7, because `aie::mmul<4,8,8>` is a poor fit and 1x4 blocking at
`rowA = 1` is a good one. Batch 16's egress and L1 analysis (sections 5b, 5c)
hold at 8 with room to spare.

| artifact | what it is |
|---|---|
| `kernels/q4k_mm.h` | batched q4k matmul: unpack a block, then `aie::mmul` |
| `kernels/q4k_mm_bench.cc` | one symbol per measurement point |
| `kernels/q4k_mm_gate.cc` | the same kernel behind a device-runnable entry point |
| `bench_q4k_mm.py` | cycles/block sweep + roofline (the step-1 cost gate) |
| `q4k_mm_gate.py` | **the numeric gate** — bit-exact vs numpy, on device |
| `mmul_probe.py` | what `aie::mmul` means by an A/B/C tile, measured |
| `kernels/proj_qmm.cc` | `+DPROJ_MM_BATCH`: batched zero / acc / flush entry points |
| `kernels/proj_qmm_gate.cc` | both projection paths in one launch |
| `proj_qmm_gate.py` | **batched projection vs the GEMV**, on device |
| `batch_attn_mask.py` | the causal mask, and a `--check` that guards it |
| `bench_attn.py` | attention's static cost and the corrected roofline |
| `bench_attn_batch.py` | **what a batch can hoist**, piece by piece |
| `check_kernels_inert.py` | **the inertness gate** — every shipping kernel vs `HEAD` |
| `xfeed_bd.py` | the X feed's tile-blocking BD, checked against `pack_A` |
| `egress_bd.py` | the egress gathers, both levels, checked against batch 1 |
| `kvappend_bd.py` | the KV append BD + the end-of-window overrun guard |
| `batch_path_check.py` | **the whole path composed** — token t's row vs batch 1's |
| `batch_equiv.py` | **the dataflow gate** — batch-B token t vs batch-1 at position P+t, stage by stage |
| `batch_row_probe.py` | **the row-map gate** — does row t of the projection get TOKEN t? The one question the others are blind to |
| `dflash_blocksize.py` | **the block-size answer** — passes priced max(compute, memory) |
| `kernels/q4k_mm.h` | also `q4k_mmul_small` — batch 4/8, 1x4 at `rowA = 1` |
| `models/qwen3-4b.h` | kernel-side model header for the DFlash target |
| `batch_l1_budget.py` | per-model L1 budget at batch > 1 |
| `batch_wire.py` | egress descriptors + X-feed bandwidth at batch > 1 |
| `dflash_traffic.py` | DFlash weight traffic per iteration + resident DDR |
| `fused_decode.py` | `qwen3-4b` + `qwen3-4b-draft` `_MODELS` entries; `DECODE_HIDDEN_TAPS` |
| `llms/qwen3_4b_q4nx/` | q4nx weights loader + requant for the DFlash target |

All under `programming_examples/` — everything but the last in `fused_decode/`,
plus a new `llms/qwen3_4b_q4nx/`. Outside that, only a one-line `exclude_docs`
entry in `mkdocs.yml`.

Five existing files are modified, and **every one is a proven no-op on the
shipping path** **[measured]**:

| file | change | proof it is inert |
|---|---|---|
| `fused_decode.py` | `DECODE_HIDDEN_TAPS`, `qwen3-4b` entries | emitted IR byte-identical to `HEAD` at `DECODE_HIDDEN_TAPS=0` |
| `kernels/proj_qmm.cc` | batched entry points behind `PROJ_MM_BATCH` | `.o` disassembly identical to `HEAD` |
| `kernels/attn_qk.cc`, `attn_kv.cc` | `ATTN_Q_LOOP`, plus the section-5g decomposition knobs | `.o` disassembly identical to `HEAD` |
| `kernels/aie_kernel_utils.h` | defines `ATTN_Q_LOOP` and the decomposition knobs | every kernel that includes it re-checked |
| `models/all_models.h` | `QWEN3_4B` id + include | additive; no existing model's expansion changes |

The disassembly check is the one that matters and it is worth re-running after
any kernel edit. It is `check_kernels_inert.py` — a script now, not a sketch,
because it has to compile each kernel at *its own* `-O` level (`rope`
miscompiles at `-O1`, the attention pair deadlocks at `-O2`) and has to ignore
the object path, which `cmp` cannot: `cmp` reports a difference on unmodified
sources, and that false positive is what got the check skipped by hand before.

### Resuming

Environment (this box: Ryzen AI 7 PRO 350 / Krackan, NPU2, native Windows):

```bash
source ~/air_env.sh            # PEANO_INSTALL_DIR, PYTHONPATH, air/aie/XRT on PATH
```

Reproduce every measurement, in rough cost order:

```bash
cd programming_examples/fused_decode
python3 dflash_traffic.py                              # seconds; traffic + DDR
python3 dflash_traffic.py --draft-head-frac 0.25       # prices a trimmed draft head
python3 batch_l1_budget.py --model qwen3-4b --batch 16 -v \
        --scratch-rows 32 --scratch-cols 256      # the new kernel's weight tile
python3 batch_wire.py --model qwen3-4b --batch 16 -v
python3 q4k_mm_gate.py --mode exact                    # ~2 min ON DEVICE; the
python3 q4k_mm_gate.py --mode random                   # ~2 min; numeric gate
python3 q4k_mm_gate.py --mode exact --batch 8          # ~2 min; the small-batch
python3 q4k_mm_gate.py --mode exact --batch 4          # ~2 min; kernel
python3 mmul_probe.py                                  # ~2 min; mmul tile order
python3 proj_qmm_gate.py --nblk 1 --batch 8            # ~2 min; batched vs
python3 proj_qmm_gate.py --nblk 2 --batch 8            # ~2 min; GEMV, at the
python3 proj_qmm_gate.py --nblk 1 --batch 16           # ~2 min; block size
python3 batch_attn_mask.py --check --cost              # seconds; the causal mask
python3 bench_attn.py                                  # ~4 min; attention roofline
python3 bench_attn.py --model QWEN3_4B --layers 36     # ~4 min
python3 bench_attn_batch.py                            # ~8 min; what a batch
python3 bench_attn_batch.py --model QWEN3_4B --layers 36  # ~8 min; can hoist
python3 check_kernels_inert.py                         # ~1 min; THE gate
python3 xfeed_bd.py                                    # seconds; the X-feed BD
python3 egress_bd.py                                   # seconds; egress BDs
python3 kvappend_bd.py --overrun                       # seconds; KV append BD
python3 batch_path_check.py                            # seconds; THE composition gate
python3 l1_align.py                                    # seconds, needs a build;
                                                       # 64-byte L1 alignment
python3 dflash_blocksize.py                            # seconds; the block size
python3 dflash_blocksize.py --attn-hoistable 1503      # with a perfect hoist
python3 dflash_blocksize.py --overlap                  # the optimistic bound
python3 dflash_blocksize.py --target llama-3.2-1b --draft llama-3.2-1b         --vocab-chunk-i2 18 --attn-cycles 2368         # the model's cross-check
python3 bench_q4k_mm.py --batches 16                   # ~3 min; mmul cycles only
python3 bench_q4k_mm.py --batches 4,8,16,32 --noperm   # ~9 min; the batch sweep
python3 bench_q4k_mm.py --kcol 512 --batches 16        # ~22 min  } pre-layout-fix
python3 bench_q4k_mm.py --mrows 64 --batches 16        # ~35 min  } baselines
python3 bench_q4k_mm.py --batches 16 --chunks 2        # FAILS, on purpose --
                                                       # see the gotchas below
cd ../llms/bench && python3 decode_geometry.py --check  # builder regression gate
cd ../qwen3_4b_q4nx && python3 qwen3_4b_q4nx_requant.py --check   # packer vs builder
```

The first three need the NPU; the rest are static and run anywhere. **The gates are
`q4k_mm_gate.py --mode exact`, `proj_qmm_gate.py`, `batch_attn_mask.py --check`,
`check_kernels_inert.py`, the three `*_bd.py` checkers, `batch_path_check.py`,
`l1_align.py`, `batch_row_probe.py` and the `DECODE_HIDDEN_TAPS` no-op diff
below** — the rest are measurements.

The batched on-device pair, which needs the NPU and two templates each:

```bash
RMS_CHUNK_PROBE=1 UNI_WAVE_HI=1 ./build_template.sh 8 1   # -> rename to x2_*
python3 batch_row_probe.py --batch 8 --L 1 --prefix x2    # does row t get token t?
UNI_WAVE_HI=1 ./build_template.sh 1 1                     # -> w4_b1_L1
UNI_WAVE_HI=1 ./build_template.sh 8 1                     # -> w4_b8_L1
python3 batch_equiv.py --prefix w4 --batch 8 --L 1 --tokens 0
for L in 2 3 4 5 6 7 8; do UNI_WAVE_HI=1 ./build_template.sh 1 $L; done  # -> w4_b1_L*
python3 batch_equiv.py --prefix w4 --batch 8 --L 1 --tokens all   # each token at its own position
```

**Both halves of a `batch_equiv` pair must carry the same `UNI_WAVE_HI`** — see
the warning under the stage table.

**`--noperm` is the one that gives totals.** The default build reports the
multiply as cycles and the unpack as a rolled static size, because the correct
unpack cannot be fully unrolled (see the gotchas). `--noperm` swaps in the
wrong-but-unrollable unpack, which is the only way to get an exact
total / MAC-per-cycle / roofline — and those totals are a lower bound.

### Next step

~~**Batched attention.**~~ **Measured and bounded — section 5g.** The hoistable
share is 35% on qwen3-4b, the ceiling is 1.55x on the attention term and 2.08x
end to end, and the block size stays at 8 either way. It is worth building and
it blocks nothing, so it is no longer the next thing.

Two corrections it leaves behind, both worth carrying forward: the
`aie::transpose` this document twice named as the lever is **free** — the
hoistable work is the K/V tile loads — and the real floor is the softmax
`update` plus the y rescale pass, neither of which any batch arrangement can
touch.

**Step 2's builder wiring, at block 8.** Where it stands:

| | |
|---|---|
| proj kernels | **done**, gated on device at batch 8 and 16 (section 5d) |
| proj cores in the builder | **done**, both `_core_blk` and `_core_blk_np` |
| `DECODE_BATCH` + L1 sizing rules | **done**, strict no-op at 1 |
| X-feed BD / egress BD / KV-append BD | **derived and checked** |
| whole path composed, in numpy | **done**, `batch_path_check.py` |
| X feed, egress, KV append, QKV transposer, rope | **done** |
| rms row loop (and why it is not a row loop) | **done** |
| glu row loop | **done** |
| attention token loop + per-token L | **done** |
| a batch-8 template that COMPILES | **done** |
| a batch-8 template that RUNS | **done**, all 16 layers, dispatch completes |
| the answers | **done** — every gate green, see below |
| host driver at batch > 1 | **not done** |

#### Where the layer is right and where it is not [measured]

`DECODE_ACC_STOP` sends an INTERMEDIATE residual out on `layerOut` instead of
the layer output, by dropping an add and keeping the get — so every channel
stays balanced and the shim task is the same task in the same place. That, plus
reading the DDR KV cache back (which both builds already write and the gate
already reads), splits the layer into stages that can each be compared against
a batch-1 dispatch:

| stage | what it covers | batch 8 token 0 vs batch 1 |
|---|---|---|
| `ACC_STOP=1` | the layer-out drain and the in-place residual buffer | **0.00 — bit-exact** |
| rope's K and V | rms core, X feed, batched mmul, both gathers, id-demux, QKV transpose, rope | **1.2% rms** |
| `ACC_STOP=2` | + attention, the o gather, the o-projection, residual1 | **2.0% rms** |
| full layer | + ph2 norm, gate-up, GLU, down, residual2 | **0.67% rms** |

**Every row of that table dispatches B copies of ONE token**, so it says
nothing about which token a projection row read — that is `batch_row_probe.py`'s
job, below. What the table establishes is that the DATAFLOW is right end to end,
which is exactly the question it was built to answer. The floor is the kernel
swap and nothing else: `proj_qmm_gate.py --nblk 4 --batch 8` puts the GEMV and
the batched mmul **1.65% rms** apart on device off the same weights, and says
the batched one is the closer of the two to exact fp32 (bias 0.03% against the
GEMV's 0.73%). The layer output lands *below* that floor, because the residual
stream dominates it.

**Compare only builds with the same `UNI_WAVE_HI`.** It is not a numerics knob
and it changes the answer by 7x: the same source at `UNI_WAVE_HI=1` reads
**0.67%** and unset reads **4.88%** (9,352-byte instruction stream against
149,392). A stale template from an earlier build is the easy way to compare two
different designs and read the difference as a regression — check
`ls -l <prefix>_b8_L*.insts.bin` before believing an A/B. This cost an afternoon
after the alignment fix landed, when the fix looked like a 7x regression and was
not: at matched wave count, before and after are both 0.67%, and after is
strictly better (token 7 goes from 5.98e-3 off token 0 to **bit-identical**).

**The fault, as first seen.** The KV readback found it rather than the layer
output. Every token in this gate gets the same X and the same rope LUT, and
neither K nor V depends on position beyond the LUT, so all 8 tokens' K and all
8 tokens' V have to be bit-identical to each other — they come out of ONE mmul
over one A operand. All 8 K are. **Token 7's V is not:** 93% rms against token
0, with 32 of its 256 elements per group still at the cache's initial zero, in
runs at `[24..31]`, `[88..95]`, `[152..159]`, `[216..223]` — 8 missing every 64.

Read that signature as a coordinate. 64 is `PAIR_PAY`, one emitter's per-token
block, laid `[lead's 32 rows | partner's 32]`; 24..31 is the lead's last 8 rows.
`proj_qmm_mm_flush_row` writes those from `y_acc[(j*RA+z)*64 + rr*8]`, which at
batch 8 (`RA` 1, `z` 0, `rr` 7) is **row 7 of each of the four `aie::mmul<8,8,8>`
C tiles**, and 24..31 is `j = 3` — `y_acc[248:256]`.

Two probes, both behind `-DPROJ_FLUSH_PROBE` so the shipping kernel stays inert,
narrow it a long way:

| probe | result | what it rules out |
|---|---|---|
| `=1` run the flush's tokens BACKWARDS | the hole does not move | not a write-order race — the egress is not reading before the last store lands |
| `=2` store a marker instead of the last vector | the marker reaches the KV cache, at exactly 24..31 of token 7's **K** | the WRITE path works. `y_acc[248:256]` really was zero |
| `=3` label every element with `±(t*32 + p)` — the token, the position, the role in the sign — and read the labels out of the KV cache | **every label is exactly right, for all 8 tokens.** Token 7's V region reads `[+224..+255 \| -224..-255]` per emitter, which is `[role 0 \| role 1]` of token 7 and nothing else | the whole descriptor chain: the flush's addressing, both egress gathers, the id-demux, the QKV L2 transpose, rope's per-token slice and the KV append. None of them mixes a token or drops a byte |

The labels are the strongest single result here: **the data path is correct and
the accumulator's contents are not.** V is copied through rope unrotated, so its
labels survive to DDR; K's do not, which is why the probe is read on V.

And the marker sharpens the shape. On the K round, only 24..31 changed — so
token 7's K was otherwise CORRECT, which rules out the X feed and the token
index on their own (the same `rr = 7` reads row 7 of every tile in every round).
On the V round, token 7 was already wrong from element 0, with the zeros on top.
So the V round's accumulator is wrong in all four tiles for row 7, and zero in
the fourth. Also ruled out: the append descriptor (`kvappend_bd.py`, and the
region maths checks out), rope's DMA (single-buffered K and V BDs in one MM2S
chain, correct lock protocol), the emitter's egress BD (`offset = 14 len = 514`
on a 528 buffer), and `ypair_l1`'s size (emitted as 528, correctly batched).

#### The fault: role 0 reads `(X[0] + X[t]) / 2`

The probe that names it is one stage further up, and it is the one that should
have been written first. `RMS_CHUNK_PROBE=1` makes the rms core stop normalising
and feed row `t` the **constant** `(t+1)/8`. The projection is linear, so every
output row comes out proportional to whatever X its row of the mmul saw, and the
ratio of row `t`'s output to row 0's IS that token index — read straight off the
KV cache, since V passes through rope unrotated. `batch_row_probe.py` is that
check, and it fails on its first run **[measured]**:

    role 1:  1.000  2.000  3.000  4.000  5.000  6.000  7.000  8.000   exact
    role 0:  1.000  1.500  2.000  2.498  3.000  3.500  4.000  0.000

128 of 128 output rows agree on each ratio, so it is a clean structural map and
not noise. Role 0 is the LEAD of every cascade pair, so this is half the output
rows of every projection in the model.

**Why nothing else could see it.** `batch_equiv.py` dispatches B copies of ONE
token, because that is what makes token `t` comparable to a batch-1 run at
position `P+t` — and identical rows hide it. `batch_path_check.py` models ONE
core; `proj_qmm_gate.py` runs ONE core. Three gates, one blind spot, and the
engine passed all three.

`(1 + t/2)/8` reads like `(X[0] + X[t]) / 2`, and that reading is **wrong** —
it cost most of a day of looking for a token permutation in the X feed, the
descriptors and the L2 transpose, none of which had one. Two things kill it.
First, a DMA does not average. Second, the ratio is uniform to bf16 on all 128
elements, and a half-pitch *read* would mix two tokens within each 8-element
run and show two distinct ratios, not one.

#### What it actually was: a 32-byte misalignment [measured]

Two more probes settle it, both behind `-D` guards so the shipping kernels stay
inert (`check_kernels_inert.py`):

| probe | what it does | result |
|---|---|---|
| `PROJ_FLUSH_PROBE=4` | skip the de-tiling and ship `y_acc` RAW, in `PROJ_MM_BATCH` contiguous `RB`-float chunks, read with SCALAR loads | role 1 is a textbook `mmul<8,8,8>` C — 4 tiles of 64, token `rr` at `rr*8`, scaling 1..8. Role 0 fits `observed[m] == correct[m+8]` on all 32 runs, with `y_acc[248:256]` zero |
| `PROJ_MM_PROBE=1` | skip the multiply and ship the A OPERAND as delivered | **both roles byte-identical and correct** — `1..8` repeating every 64, zero within-run spread |

So the X feed, the flush's store position, both gathers, the L2 transpose and
the KV append are all correct on both cores, and role 0's accumulator is the
correct accumulator **shifted left by exactly one 8-float vector**. The emitted
`aie.air.mlir` says the two cores are otherwise identical — same `inX` and
`wL2ToL1` BDs, same loop bounds, same buffers; the only differences are the
flush's role constant and the lead's 2-element packet header store.

The cause is in the linker script, and `l1_align.py` reads it straight off:

    LEAD    tile 0,2   buf41 (yacc, _e=1)  @ 0x7D820   <-- 0x20 past a 64B line
    PARTNER tile 0,3   buf57 (yacc, _e=1)  @ 0x63800

mlir-aie aligns a compute-tile buffer to the tile's LOAD/STORE BUS width and
packs the rest end to end (`AIEAssignBuffers.cpp`; `aligned` defaults true, the
width is `getComputeTileLoadStoreBusWidth`). On AIE2p that width is 256 bits —
**32 bytes**. `aie::mmul<8,8,8>`'s C tile is `size_C` = 64 floats = 256 bytes and
Peano moves it in 512-bit chunks, which need 64. A misaligned 512-bit access on
AIE2 does not fault; it masks the low address bits, so the whole accumulator
lands 32 bytes low and its last 8 floats are never written.

Four things follow from that, and all four are what was measured:

- **`ypair_mm_l1` is the only odd-sized buffer on a proj core** — `16 + 2*32*8`
  = 528 bf16 = 1056 bytes = 16.5 × 64 — so it misaligns whatever is packed next.
- **Only LEAD tiles host it**, so only they misplaced an accumulator. That is
  the entire role 0 / role 1 asymmetry; nothing about the cascade pairing or the
  X feed was ever involved.
- **Only the `_e=1` round** uses the misplaced accumulator (`_e=0` gets the
  other, still aligned). The QKV phase's 6 rounds put K on round 4 and V on
  round 5, which is exactly why K read correct and V did not.
- **`y_acc[248:256]` is never written** — the loose end from the section above,
  same cause, no second fault.

And the ratios stop being mysterious: normalising against `observed[0]`, which
holds `correct[8]` rather than `correct[0]`, turns a clean `t+1` into `(t+2)/2`.

**The fix** is one line: round the shared egress buffer up to a multiple of 64
bytes (528 → 544). It is inert on the wire — the egress BD is still
`offset = 14 len = 514` and the instruction stream is byte-identical — it only
moves `buf41` to `0x7D840`. `batch_row_probe.py` then passes, token 7's V stops
being 93% wrong, and token 7's layer output goes from 5.98e-3 off token 0 to
bit-identical.

**The lesson worth keeping** is that the alignment a kernel needs is the
*caller's* job here, and nothing in the toolchain says so: the allocator does
what it documents, the kernel assumes what its intrinsics need, and the two
numbers differ by a factor of two. `l1_align.py` is the check that closes it,
and it checks the emitted ADDRESSES rather than restating the rule.

#### A gate that measures floating point instead of the engine

Worth reading before trusting any number above, and the reason this took as
long as it did. `batch_equiv.py` had two ill-conditioned fills, one after the
other, and each produced a confident wrong answer.

**`min` drawn independently of `scale`.** In a real q4k block `min` is the block
minimum, so `w = q*scale + min` is CENTRED on zero; drawn independently it is a
small perturbation on a mean of `7.5*scale`, and every dot product against a
zero-mean activation becomes a sum of 2048 terms of magnitude ~55 cancelling to
a result of magnitude ~2. The 1.65% kernel difference then landed as **20%** in
K and V, compounded through each projection, and reached the layer output as
**772x** — as a near-constant additive offset on V, a clean 1.5x on the
o-projection, and an unbounded blow-up at the end. It looked exactly like a
batching fault.

**A gate-up output outside silu's LUT.** `getActivationBf16` is a 64-bin linear
approximation over about `[-8, 8)` with a truncating out-of-range policy. With
the weight scale as first written, the gate-up output had an rms of 3.7 and
roughly 5% of 8192 elements per token fell outside — so which bin a value landed
in flipped on a fraction of a percent of input, and the layer output was
CHAOTIC: 101% apart between two builds whose gate-up outputs agreed to 2%.
Deleting the silu collapsed that to 9.6%, which is how it was found.

**The instrument that settles it is a SWEEP, not a number** [measured]:

| weight scale | layer output | rope K |
|---|---|---|
| 1.0 | 49% | 1.1% |
| 0.5 | 14% | 1.1% |
| 0.25 | 3.6% | 1.1% |
| 0.1 | **0.67%** | 1.2% |

Read the columns against each other. K comes off the projection and nothing
else, and it does not move — 1.1% is the kernel swap. The layer output moves by
two orders of magnitude over the same sweep, because silu sits between them. A
wiring fault would move BOTH columns and would not care about the scale. That
is what turned "the MLP half is wrong" into "the MLP half is fine".

The lesson generalises past this file: a synthetic fill that makes the
arithmetic ill-conditioned turns every gate downstream of it into a measurement
of cancellation, and a single number cannot tell you which you are looking at.
`batch_equiv.py` now reports a difference as a SHAPE — offset or scale,
permuted or not, which tokens, how it moves with the conditioning — before it
reports it as a number.

#### Seven deadlocks, and what they have in common

None of them was visible in the AIR, all of them were visible in the emitted
AIE dialect, and none is the kind of bug an element count can find. They are
worth reading as a set, because the next one will look like them.

| what | why |
|---|---|
| token-major egress gathers | a packet's 2-word header rides ONCE at the front; a BD walks its source linearly, so no descriptor lands the header at 0 and token t at `HDR + t*stride`. Two gets do it arithmetically and eat the memtile's ping-pong ring. **The transpose moved to the consumers**; the gathers are now the batch-1 descriptor, B times longer |
| outY on the rms core's S2MM0 | the batched rms body aliases one staging buffer as both the outY destination and the @xnorm source, so the allocator's packet-flow reuse folded outY onto the port that already had rmsX/rmsW/rmsW2 — as a SECOND BD chain. A channel has one. The first also stopped cycling |
| the QKV transposer on col-3 MM2S 0 | a documented route deadlock in this builder; the KV puts already carry the floor, the transposer did not |
| rope blocked on the q broadcast | rope must finish all B tokens before the KV readback can start, the CUs wait on the readback, the q memtile waits on the CUs, rope waits on the q memtile. Got 4 of 8 tokens through |
| per-token KV append gets | each is a separate shim task and the fused launch paces a `preserve_shim_dma_order` channel at depth 2. Got 6 of 8 tokens through |
| **a per-token q get on the attention CU** | the q memtile fans to the four CUs as a DAISY CHAIN — CU c+1's transfer starts when CU c's finishes. At batch 1 each link is one 512-element landing that completes on arrival; taking q a token at a time makes CU 0's link an 8-token transfer gated on CU 0 running the whole block, and CU 0 cannot, because the KV re-block memtile hands both CUs of a column their block together and CU 1 is waiting for a q it will not get until CU 0 finishes. **Take all B rows in ONE get before the token loop** — the shape the q memtile itself already uses one hop up |
| **the vocab waves, at batch > 1** | `LM_HEAD` is refused when batched and the rms core's batched body has no vocab arm — not an empty arm, the DECODE body emitted unconditionally, because the vocab `@xnorm` put is a memtile-shaped descriptor whose 512-element wrap does not fit a compute tile's 8-bit wrap field. Left at `UNI_WAVES` this deadlocks at the FIRST vocab wave and nowhere earlier: all 16 decode layers run, every layer's output lands, every layer's KV appends, and then the rms core starts a decode pass into a chip that has taken its vocab arm and gone idle. Clamp `UNI_WAVE_HI` to `UNI_DEC` |

And one that is not a deadlock but caused one: writing the q buffering as
`for t: get slice` then `for t: fan slice` gets rebuilt by
`air-ping-pong-transform` into a 2-deep ring of slices — **the interleaved form
again**. One get and one put per CU, with the batch as a BD dimension, gives the
transform nothing to rewrite.

**Both of the last two look like a batching fault and neither is about the
batch.** The q fan is a batch-1 idiom that only has a cycle when a CU has more
than one thing to wait for; the vocab wave is a mode the batched build was never
going to run and forgot to stop driving. The common shape is still the one the
shipping q4nx decode answers by construction: *nothing on the critical path
waits for a consumer more than one hop downstream.*

#### Two facts about the shim, measured while bisecting

Both cost a build to learn and neither is written down anywhere else.

- **`air.preserve_shim_dma_order` is a GLOBAL order, not a per-channel one.**
  Moving the layer-output drain to the front of the runtime sequence -- to make
  it report progress before the KV readback rather than after -- starved the
  whole sequence: even the KV append, which normally completes, wrote nothing.
  So a diagnostic drain cannot be hoisted past unrelated traffic, and the
  ordering between two channels' shim tasks is real.
- **A drain placed between the append and the readback does not route.**
  `aie.packet_flow` source (2,2) DMA1 to destination (1,1) DMA2: the pathfinder
  gives up. The floorplan has no slack for an extra shim endpoint there.

Together those close off the obvious way to bisect a batched hang by phase. The
signal that does work is the buffer readback on timeout, which is why
`batch_equiv.py --smoke` does it.

#### The tools this needed

All were written mid-hunt and all found a real fault immediately.

- **`l1_align.py`** — reads the buffer addresses aiecc actually assigned out of
  `air_project/ldScripts_*.ld.script` and fails on any compute-tile buffer that
  is not 64-byte aligned. Checks the emitted addresses rather than restating the
  rule, so it stays true if the allocator changes. Costs one build.
- **`batch_row_probe.py`** — asks, on device, whether row `t` of the batched
  projection got token `t`. The other gates structurally cannot: `batch_equiv`
  dispatches B copies of one token, `batch_path_check` and `proj_qmm_gate` model
  a single core.
- **`check_channel_balance.py`** — how each SIDE of each channel scaled from
  batch 1. Ratios, not totals: counting elements absolutely needs a model of
  `scf.parallel` fans, `index_switch` arms and herd multiplicity, and getting
  one wrong made the first version call the SHIPPING batch-1 design broken. The
  same modelling error on both sides of a division cancels.
- **`check_dma_alloc.py`** — which tiles moved their DMA channels, and how the
  BD chains and lock counts changed. Found the two-chains-on-one-channel fault
  in seconds. Needs two AIE dumps, so it costs two builds (~9 min).

What none of them can see is ORDER, and four of the seven faults above are
ordering.
For those the tool is the emitted `air_project/aie.air.mlir` and a hypothesis.

#### The order to do the rest in

1. ~~**The device equivalence gate, FIRST.**~~ **Written — `batch_equiv.py`.**

   **Its first premise was wrong, and the correction is the point.** It asserted
   that B IDENTICAL tokens give B IDENTICAL rows. That is false, and for the
   reason the batch exists: a block occupies B CONSECUTIVE positions, so token t
   attends to t more keys than token 0 and rotates by a different RoPE angle.
   An "all rows equal" gate would have PASSED on an engine that gave every token
   position P's context — exactly the silent failure `batch_attn_mask.py` was
   written to warn about. The property that is actually true, and that DFlash
   rests on, is

       one batch-B dispatch at position P
         ==
       B batch-1 dispatches at positions P .. P+B-1, same X each time

   `--tokens 0` needs one batch-1 template and already covers the whole batched
   data path; `--tokens all` needs one per position (a non-DYNSEQ template bakes
   L) and is what proves token t gets a DIFFERENT and correct answer rather than
   a copy of token 0's. **Both now pass** at batch 8 on llama-3.2-1b — see the
   per-token table in "State of play". The eight batch-1 references cost eight
   builds; `DECODE_DYNSEQ=1` would remove that.

   **Two other things it taught.** Random bytes make random bf16 SCALES: the
   first run returned 0x7F81 — one NaN — in every element, and a gate whose
   output is constant passes on anything. It now builds REAL q4k blocks and
   refuses to compare a flat output. And a hung dispatch is not information-free:
   `--smoke` reads the buffers back on timeout and says which regions moved,
   which is the only progress signal this engine gives.

   **Writing it before the wiring already paid.** It found that the host-facing
   L3 buffers never scaled — `x_l3`, the rope-LUT slab in `rms_l3`, and `y_l3` —
   and that `decode_geometry.py` restates those sizes rather than reading the
   memref shapes, so it reported batch 8 and batch 1 as the same dispatch. Both
   fixed. The scaling is now visible and is the thesis of this whole document in
   one table **[measured]**:

   | BO | batch 1 | batch 8 | |
   |---|---|---|---|
   | X | 2560 | 20480 | x8 — B token embeddings |
   | **weights** | **154419200** | **154419200** | **x1 — the entire point** |
   | rms (+ B rope LUTs) | 200704 | 297472 | x1.48 |
   | Y | 162304 | 223232 | x1.38 |
   | KV cache | 150994944 | 152174592 | x1.01 — the last token's position |

   Read the weight row against the X row. That ratio is what the whole
   speculative-decoding argument rests on, and it is now a property of the
   emitted design rather than a claim.
2. ~~**rms row loop.**~~ **Done — and it is NOT a row loop.** Everything the rms
   core does is per row, but it has to hold all B rows of TWO things at once:
   the raw batch (the residual stream) and the normalized batch (what the
   projection re-reads REFEED[p] times). On qwen3-4b that is 2 x 40 KB against a
   54 KB budget, and neither can be dropped. So the normalized batch is **never
   materialized**: `rms_scale_row_aie` keeps one float per row and
   `rms_chunk_aie` regenerates whichever @xnorm chunk is being sent, for all B
   rows, into a staging buffer one chunk wide. The big buffer stays raw and
   accumulates in place — x, then h, then the layer output, one buffer, three
   roles. `residual_acc_row_aie` adds a projection round in where it lands,
   which also removes the K-wide landing buffer on the way in.

   54304 B of 55296 at batch 8 **[measured]**, so `BATCH_MAX_RMS` is exactly 8 —
   computed from the live set, not asserted. Gemma's sandwich norm is refused
   rather than half-wired: normalizing the SUBLAYER OUTPUT needs the whole
   projection row resident, which is the second buffer this design exists to
   avoid.

   The cost is that a chunk is recomputed once per re-broadcast round rather
   than once per token. Whether that lands on the critical path is unmeasured;
   the alternative is a resident X buffer in L2.
3. ~~**glu row loop.**~~ **Done.** Same shape, ten minutes once step 2 was.
4. ~~**Attention token loop + per-token `L`.**~~ **Done.** The mask needed no
   kernel change, as `batch_attn_mask.py` predicted: `attn_qk_blk`'s tail mask
   IS a per-query causal mask when L is a per-query value, so token t runs with
   `L + t` — the loop's own induction variable, not a second RTP. The block
   COUNT stays uniform at `ceil((L+B-1)/16)` because the shim's push and the
   core's consume have to agree; blocks past a token's own L hit the kernels'
   existing `rem <= 0` early return. `ATTN_L_BLK` is where that lives, and it
   also makes `kvappend_bd`'s overrun guard hold by construction.
5. ~~**Find the last deadlock.**~~ **Done — there were two.** The q fan and the
   vocab waves; both are in the deadlock table above, with the mechanism. The
   method that found them is the one that found the other five: build,
   `check_dma_alloc.py` against a batch-1 dump, and when that is clean, reason
   about ORDER in the emitted AIE.

   **Reading the SHIPPING q4/q4nx decode first is what closed it**, and the
   question the doc pointed at — *what keeps a producer from being blocked by a
   consumer three hops downstream?* — named the q-fan fault directly. Every one
   of the seven faults was a place where the batched wiring invented a dataflow
   the batch-1 engine does not use. Still worth reading, for the numeric work
   that is left:

   | where | what it answers |
   |---|---|
   | `llms/*/q4nx_decode_*.py`, `llama32_1b_q4nx_inference.py` | how the driver sequences a dispatch, and what it does between tokens — the closest thing to a multi-token cadence that already runs |
   | `decode_staircase.py`, `decode_insts_gen.py` | how L varies per dispatch without a rebuild. The batched build currently bakes one L; the staircase is how the shipping models avoid that |
   | `decode_dynseq.py` + `DECODE_DYNSEQ=1` | the runtime-L form. It exists, it works, and it would remove the per-position template pair `batch_equiv.py --tokens all` needs |
   | the `refeed()` sites and their `air.refeed_count` in the AIE dump | the ONE re-broadcast idiom this engine is built around. The batched rms core stepped outside it (a real production loop, not a collapsed re-broadcast) and that is the largest un-audited difference left |
   | `llms/shared/builders/*_multi.py` | the multi-launch block builders — a different answer to "more than one thing per call" than the one being built here |

   The specific question to take to them: **what does the shipping engine do that
   keeps a producer from being blocked by a consumer three hops downstream?**
   Five of the seven faults were exactly that, and the batch-1 design never hits
   it because one token never has to wait for a second.
6. ~~**The MLP half.**~~ **Not a fault — see the sweep above.** The layer output
   is 0.67% from batch 1 once the gate-up stops being driven outside silu's LUT.
   The three probes it took are still in the tree and worth knowing about:
   `GLU_ROW_PROBE=1` swaps the halves, `=2` is `up - gate` (antisymmetric and
   silu-free, and the one that proved the plumbing), `=3` is `up` alone. All
   behind `-DGLU_ROW_PROBE`, so `check_kernels_inert.py` stays green.
7. ~~**Role 0 reads `(X[0] + X[t]) / 2`.**~~ **Fixed — and it was never a token
   permutation.** The proj lead tiles' `_e=1` accumulator was 32 bytes off a
   64-byte line, because the one odd-sized buffer on a proj core (the 528-bf16
   shared egress block) is packed just before it and mlir-aie only promises
   32-byte alignment. Rounding that buffer to 544 fixes it; `batch_row_probe.py`
   passes and `l1_align.py` is the standing check. Full account above.

   Worth keeping from how it was found: the shape of a wrong answer is a weak
   signal. `(1 + t/2)/8` looked exactly like an average of two rows, which is
   not a thing a DMA can do, and a day went into the X feed and the descriptors
   before the raw-accumulator dump (`PROJ_FLUSH_PROBE=4`) and the A-operand dump
   (`PROJ_MM_PROBE=1`) said in one run each that both were fine. **Dump the
   operand and dump the accumulator before theorising about either.**
8. **The batched lm head.** The prerequisite for the host driver — a verify pass
   needs B logits — and the thing that blocks it is **not** what the refusal in
   `fused_decode.py` used to say. See the section below; the refusal has been
   corrected to name the real constraint.
9. **Host driver.** B embeddings in, B logits out, **B rope LUTs** (per
   position — the builder now feeds B of them, one put per token), and
   `check_bounds` on the KV append before the dispatch.

Keep running the batch-1 no-op diff on **both** models after every step. It has
already caught a leaked constant that folded away on qwen3-4b and did not on
llama, so one model is not enough.

#### The batched lm head: what actually blocks it [measured]

The refusal used to read "the lm-head herd runs its OWN `_gemv`/`_emit`". **That
is wrong** — there is one proj herd family (`proj_blk0`/`proj_blk1`), both arms
go through it by RTP, `_proj = _mm if BATCH > 1 else _gemv` is arm-independent,
and `_emit` already ships `HDR + PAIR_PAY*BATCH`. **The vocab projection is
already batched.** So are the group and main gathers (`GRP_ROWS_B`,
`MAIN_ROWS_B`). Writing `_rms_lm_head_batched` and scaling the Y region is
straightforward and it compiles.

What blocks it is the **rms core's port budget**, and the rule is exact:

> A compute tile has 2 MM2S and 2 S2MM. Each PORT gets ONE `aie.dma_start`
> chain, so both RTP arms must present the **same buffer on the same port**, and
> no buffer may serve more than two ports.

Batch 1 satisfies this for free — every buffer on tile (2,2) is `K` bf16, so the
four flows (rmsX/rmsW/rmsW2 in, outY in, xnorm out, layerOut out) fold onto four
chains with one BD each, shared by both arms. Batched they cannot: the buffers
are 8x and the flows want different shapes. Four arrangements were built and all
four hang **identically** — wave 0, one layer's KV written, layer output never
lands, no message — which is why this needs the emitted `aie.mem` block rather
than a device symptom to tell apart:

| arrangement | emitted | |
|---|---|---|
| layerOut from `xb` (decode) and `stg` (vocab) | **two `dma_start(MM2S,0)`**, the second with `repeat_count` | only the first chain cycles |
| `stg` on three ports (xnorm out, outY in, layerOut out) | one chain per port | still hangs |
| outY into `lo` (vocab) and `stg` (decode) | **two `dma_start(S2MM,1)`** | as above |
| `lo` = outY in + layerOut out, both arms; `stg` = xnorm only | one chain per port, **topologically identical to batch 1** | still hangs |

The last one is the interesting failure: the DMA topology matches the working
build exactly and it still hangs, so the port rule is necessary and not
sufficient. The core diff against a known-good dump shows a missing
`use_lock(..., Release)` after the ph2 re-broadcast, so something in the
dependency analysis is also unhappy — that is where the next attempt should
start, with `check_dma_alloc.py` on the two dumps rather than by inspection.

**And the space does not fit either.** One-buffer-per-flow at batch 8 on
llama-3.2-1b:

    xb   BATCH*K      32768        stg  BATCH*2*COL_BLOCK   8192
    lo   BATCH*PAYLOAD 8192        w    K                   4096
    w2   K             4096        scl  BATCH f32             32
                                        -------------------------
                                        57380 + stack 10240 = 67620

against 65536. It clears only with the stack at ~8 KB, which is a global setting
and would silently shrink the inlined attention cores' frames too.

**So this is a design decision, not a patch**, and the options are:

| | |
|---|---|
| relay the logits off a **different tile** | the glu core is idle in vocab mode and has free ports; needs a new outY destination ordinal and a new shim task |
| drain the logits from the **memtile**, via `HOST_DRAIN` | the relay already exists for other dests and is already batched (`relay_l2` is `PAYLOAD_B`); needs the vocab arm to stamp a host-drained dest instead of `RMS_DEST`, and the rms core then leaves the logit path entirely, which also frees `lo` |
| keep it on the rms core and buy the space | re-get `rmsX` per block so `xb` can hold logits — but the x-sends and the drain must stay INTERLEAVED (all 36 sends before the first drain backs the egress up and deadlocks; gemma gets away with it only because its `_voc_blks_2k` is 1), and the extra shim tasks then risk the global `preserve_shim_dma_order` pacing |

The middle one looks right: it removes a relay hop from the critical path, the
machinery is already batched, and it leaves the proven decode arm untouched.

**One design decision worth knowing about**, because it is not obvious and it
shaped everything downstream. The projection emits **(round, token)**: round r
is a 32-row band of the output for *all* B tokens, because that is what a
batched mmul computes in one go. rope wants **(token, round)** — one token's
whole `M`-wide row — and cannot be given a strided landing for it, because a
`[B][M]` L1 buffer is 96 KB against a 54 KB budget. Two ways out:

| | cost |
|---|---|
| **transpose in L2** (taken) | one `[B][M]` memtile buffer, 96 KB of 512 KB; rope, attention, the KV append and everything downstream stay per-token, looped B times |
| slice rope by head so it consumes (round, token) directly | a kernel change to `pseduo_rope` plus a rewrite of the attention feed, for a phase that is ~3% of the pass |

The L2 transpose costs nothing that matters — attention does not amortize anyway
(section 5e), so looping the post-projection phases per token is what the cost
model already assumes. It does re-introduce a memtile on the QKV path, which is
what deadlocked the fused vocab build once, so it is **arm-guarded**: in vocab
mode dest 0 never flows, and a memtile that stalls waiting for it is precisely
that failure. An idle compute-tile S2MM is harmless; an idle memtile is not.

The two descriptors are the parts that were easy to get silently wrong, and
they are now numbers rather than intentions:

- `Xt` at block 8 is `sizes=[32,8,8] strides=[8,512,1] offsets=[chunk*32,0,0]`,
  verified elementwise against `pack_A` itself rather than against a
  restatement of the derivation. Two traps closed: the token stride is
  `X_CHUNKS*COL_BLOCK`, not `KCOL`, and AIR's offsets follow
  **`memref.subview`** — the address is `base + Σ offsets[d]*strides[d]`, so a
  flat chunk offset would be multiplied by `strides[0]` and read the wrong
  activations while transferring exactly the right *number* of them.
- The egress needs **two** descriptors, group and main. See section 6 step 2.
- The flush has to **de-tile**: `aie::mmul` leaves the accumulator in C tile
  order, so one token's 32 rows are four 8-float runs 64 floats apart.
  `proj_qmm_mm_flush_row` does it; the egress BDs agree with it.

**What is left is one job, not five.** The feed and the drain both terminate at
`rms` / `rope` / `glu`, and `PAYLOAD` is a structural constant of the refeed
path as well as the drain — so the X memtile, the egress widening, the row
tiling at 4 and the attention query tile all have to land together or the
engine is inconsistent in between. Nothing smaller than that is testable.

**The gate it needs is half built.** Everything else has been gated by
"byte-identical at batch 1", which by construction says nothing about batch 8.

The *software* half now exists — `batch_path_check.py`. Each descriptor checker
passes on its own, and that is not the same as the path working, because the
pieces meet at conventions no single checker sees both sides of: `tok_stride`
across the flush and the gather, A-tile order across the X feed and `pack_A`,
C-tile order across the mmul and the de-tiling. Get one wrong and **both**
checkers still pass — each side is self-consistent, and they are consistent
with different layouts. So this walks a token block through every stage in
order, using the same functions and descriptors the engine will use, and
asserts token t's assembled row equals the row batch 1 produces. Both pairing
regimes, batch 8 and 16.

**It has been seen to fail**, which is the only reason to trust a checker that
passed first time. Flush writing role-major instead of token-major: caught.
De-tiling dropping the `RA` factor: caught — *at batch 16 and not at batch 8*,
because `RA` is 1 there, so a batch-8-only run does not exercise the tiling.

The *device* half still has to be built, and still comes before the wiring: an
**equivalence run** — dispatch at `DECODE_BATCH=B` with all B tokens identical,
assert every token's output equals the batch-1 output. No reference model
needed, and it covers what numpy cannot: DMA, locks, cascade, backpressure.

**Do not build for block 16.** It is 1.06x. See section 5f before re-litigating.

Run `q4k_mm_gate.py --mode exact` after **any** change to `q4k_mm.h`, and
`check_kernels_inert.py` after any change to a production kernel. The first has
already caught two faults that compiled, benchmarked identically, and were wrong
(section 5d); the second is what proves the five modified production files are
inert, and it is the reason the section-5g knobs could be put *inside* shipping
kernels at all.

```bash
cd programming_examples/fused_decode
python3 check_kernels_inert.py        # exit 0 = every shipping kernel unchanged
python3 check_kernels_inert.py -v     # ... and where the first difference is
```

The `DECODE_HIDDEN_TAPS` no-op gate, which is the one that matters when touching
the builder — it must print nothing:

```bash
cd programming_examples/fused_decode
E="DECODE_MODEL=llama-3.2-1b VOCAB_CHUNK_I2=18 LM_HEAD=0 NLAYERS=1 \
   DECODE_GOLDEN=1 UNIFIED=1 DECODE_GOLDEN_L=2048 W_DUAL_CHAN=1 \
   FUSED_DECODE_EMIT_ONLY=1"
# The reference copy has to live HERE, not in /tmp: it imports proj_qmm_pack
# and reads models/ by relative path.
git show HEAD:programming_examples/fused_decode/fused_decode.py > _fd_head.py
env $E python3 _fd_head.py                            > /tmp/a.mlir
env $E DECODE_HIDDEN_TAPS=0 python3 fused_decode.py   > /tmp/b.mlir
diff /tmp/a.mlir /tmp/b.mlir && rm _fd_head.py
```

### Gotchas that cost time before

- **A buffer's SIZE decides its neighbour's ALIGNMENT.** mlir-aie packs
  compute-tile buffers end to end and only aligns them to the tile load/store
  bus width — 32 bytes on AIE2p — while `aie::mmul`'s C tile is moved in 512-bit
  chunks that need 64. So one odd-sized buffer silently misaligns the next one,
  and a misaligned 512-bit access on AIE2 **does not fault, it shifts**: the
  data lands 32 bytes low and the tail is never written. Keep L1 buffer sizes a
  multiple of 64 bytes and run `l1_align.py`. This cost the most time of
  anything in this document, because the symptom (`(1 + t/2)/8`) reads like a
  token permutation and sends you into the DMA descriptors, which were fine.
- **Dump the operand and dump the accumulator before theorising about either.**
  `PROJ_MM_PROBE=1` ships the A operand instead of multiplying and
  `PROJ_FLUSH_PROBE=4` ships `y_acc` raw with scalar loads. One run each said
  "the feed is perfect, the accumulator is shifted by one vector" — which is
  most of the answer, and neither took longer than a build.
- **Two templates are only comparable if their `UNI_WAVE_HI` matches.** It is
  not a numerics knob but it moves `batch_equiv`'s answer 7x. Check
  `ls -l <prefix>_b8_L*.insts.bin`: 9,352 bytes is `UNI_WAVE_HI=1`, 149,392 is
  unset. A stale template from a previous session is the easy way to A/B two
  different designs and call the difference a regression.
- **`sizeof(q4k_block_t)` is 9216 and a packed block is 5120** **[measured]**.
  `uint4` is byte-addressed, so `uint4 qs[8192]` reserves double. Never write
  `A + b` on a `q4k_block_t *`; step blocks on the `bf16` side with
  `Q4K_BLOCK_BF16`. Every production call site already does, which is why this
  survived unnoticed.
- **A static check cannot see an operand-role swap.** The weights-as-B rework in
  section 5 changed the unpack, the layouts and the header, and left
  `q4k_mm_block` calling `q4k_mmul(W, B, C)`. It compiled, it benchmarked
  identically — bundle counts depend on the template arguments, not on which
  pointer feeds which slot — and it was wrong. Only the device gate caught it.
  Re-run `q4k_mm_gate.py` after touching `q4k_mm.h`, always.
- **Do not model AIE bf16 with `astype(bfloat16)`.** The core rounds toward
  −∞ and `aie::mmul` multiplies in bfp16, so a round-to-nearest fp32 reference
  disagrees with correct hardware — by 43% of unpacked weights, and by enough
  in a dot product to read as a 12% error. `q4k_mm_gate.py` carries both models
  (`bf16_rd`, `bfp16_ebs8`); reuse them rather than re-deriving them.
- **Random test data is not representative data.** Drawing `scale` and `min`
  independently gives dequantized weights a large positive mean, and with a
  biased rounding that turns into an 11% bias that real q4k weights do not have.
  Quantize an actual matrix with the min/max rule instead (section 5d).
- **`make compile-decode` cannot run on this box.** `preflight-peano` requires
  llvm-aie `21.0.0.2026080601+f4a72c27`; that build exists only as a manylinux
  wheel and the newest Windows wheel is `2026080301`, already installed. The
  decode templates in `fused_decode/` were built by bypassing only that prereq:
  `make -o preflight-peano _compile_decode_build`. The `llvm-link < 23` gate is
  separate, still enforced, and satisfied by the LLVM 22 shim in
  `~/air-win-build/llvm-link-shim` (must be on PATH).
- **Full unroll does not scale.** `bench_q4k_mm.py` straight-lines every loop so
  static bundles equal cycles. That is ~3 min at kcol 256, ~22 min at kcol 512,
  ~35 min at `--mrows 64`, and did not finish in 40 min for the native-bf16
  variant. Larger shapes need a rolled build with trip-count weighting instead.
- **Count the whole core, not the kernel's buffers.** Section 5 first priced the
  kcol-512 fold at "48 KB of 64, feasible but tight" by adding up only the two
  buffers the kernel itself declares. The proj core also carries `rcache`,
  `wblk`, `yacc` and `ypair`, which is another 17.5 KB at batch 16, and 10 KB of
  that 64 is stack. It does not fit. `batch_l1_budget.py --scratch-rows/--scratch-cols`
  exists so this gets counted against the real core.
- **Diffing two builds does not work here.** DecodeInstsGen recovers slopes by
  diffing runtime instruction streams; in object code the trip count sits in a
  register, so both builds emit an identical body and the difference is zero.
- **A fully-inlined multi-chunk body will not compile.** Two `[[clang::always_inline]]`
  copies of an unrolled matmul need a frame past AIE2's 16-bit load/store
  displacement field: `immediate operand value -33152 is out of range
  [-32768, -64]`, from the AIE2 assembly printer. It fails at kcol 128 as well
  as 256, so it is frame size rather than code size and a smaller shape does not
  help. This is what blocks the `--chunks` experiment (section 5).
- **Clang outlines the unrolled bodies, and that changes what you measure.** At
  full unroll each `q4k_mmul` body is ~1800 bundles, past clang's inlining size
  heuristic, so it emits **one** shared `.text.<mangled>` section and calls it.
  That is why `bench_q4k_mm.py` counts per section rather than per symbol (the
  `extern "C"` wrappers are bare tail-jumps of 2 bundles) — but it also means a
  function that calls two bodies costs exactly 2x, with no cross-body
  scheduling, no matter what the source looks like. Measuring anything about
  *adjacency* needs `[[clang::always_inline]]` at the call site; without it you
  are measuring the outliner. `q4k_mm_chunked` does this deliberately.
- **`clang-format` is not installed**, so the three new C++ files are unverified
  against that CI gate. `black` is, and passes.

---

## Answer

DFlash keeps prefill and decode. It changes decode only.

- **Prefill: unchanged.** Same feed-forward prefill we run today, plus one
  addition — save the hidden states at layers 1, 9, 17, 25, 33.
- **Decode: replaced** by a draft/verify loop. Both steps are superkernel calls
  at a batch instead of batch 1.

One new kernel to build: the superkernel's projection matmul, batched. It exists
and is validated (section 5d).

**The batch is 4-8, not 16, and the speedup is ~1.6-2.8x, not 4.9x** (section
5f). Batch 16 was chosen from a roofline that counted projection weight traffic
and left attention out; attention is the one term that does not amortize over a
batch, and putting it back moves both numbers. Everything else in this document
still holds — it is the block size and the headline that change.

---

## 1. What DFlash is

A speculative decoding method. A small model drafts a block of tokens, the real
model checks them all in one pass, and you keep the correct prefix. Output is
identical to normal decoding (lossless) because the real model decides.

The drafter is a diffusion model, which matters for one reason only: it emits
all 16 tokens in a **single** forward pass instead of one at a time.

Checkpoint named in the request: `z-lab/Qwen3-4B-DFlash-b16`. Its config
**[measured]**:

```json
"block_size": 16,                  "num_hidden_layers": 5,
"hidden_size": 2560,               "intermediate_size": 9728,
"num_attention_heads": 32,         "num_key_value_heads": 8,
"head_dim": 128,                   "rope_theta": 1000000,
"vocab_size": 151936,              "tie_word_embeddings": true,
"dtype": "bfloat16",
"dflash_config": { "mask_token_id": 151669,
                   "target_layer_ids": [1, 9, 17, 25, 33] }
```

`b16` is the **block size**, not bfloat16 (the HF model card gets this wrong).

Two facts that shape everything:

1. **The drafter is 5 Qwen3-4B layers.** Same hidden size, same intermediate
   size, same head counts, same rope theta. Nothing new to design.
2. **The drafter ships in bf16**, not 4-bit. `"dtype": "bfloat16"`, no
   quantization config.

Context fusion is one linear plus a norm **[measured]**, from the model code:

```python
self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size)   # 12800 -> 2560
target_hidden = self.hidden_norm(self.fc(target_hidden))
```

Embedding and LM head are the target's (tied).

---

## 2. What changes

**Prefill.** Unchanged. It already runs the whole prompt through the
feed-forward path. The only addition is dumping hidden states at 5 layers,
which prefill already exposes per layer.

**Decode.** Today:

```
1 superkernel call (36 layers, batch 1)  ->  1 token
```

With DFlash:

```
loop:
  draft   : superkernel call,  5 layers, batch 16  -> 16 guesses
  verify  : superkernel call, 36 layers, batch 16  -> check all 16 at once
  accept  : keep correct prefix (~6), discard the rest
```

2 calls -> ~6 tokens instead of 1 call -> 1 token.

Draft and verify are the same engine at different depths (5 vs 36 layers). One
kernel, built once, used twice.

---

## 3. Why the superkernel and not prefill

Prefill also does batched matmul, so it is a fair question. Three reasons it
loses:

| | superkernel | prefill |
|---|---|---|
| weight format | q4nx, **0.625 B/param** [measured] | bf16, 2.0 B/param — 3.2x more bytes |
| KV cache | reads + appends it | cannot read one at all |
| NPU calls, 36 layers | **1** [measured] | **252** (7 ELFs/layer x 36) [measured] |

The call count decides it. At 50-200 us per NPU call, 252 calls is **13-50 ms**
of pure overhead **[estimated]** against a 54.7 ms verify pass **[measured, see
section 5]**. Prefill gets away with it at batch 2048 because the overhead
spreads over 2048 rows; at batch 16 it spreads over 16.

---

## 4. The one kernel to build

The superkernel's projection matmul does one token at a time. It needs to do 16.

**Today** (`kernels/q4_k.h`, `_qmm_q4k_bf16<M=32, N=256>`) — the activation is a
vector, not a tile. `M`=32 is weight rows, `N`=256 is the contraction:

```cpp
aie::vector<bf16, 32> b_col = aie::load_v<32>(it_B);        // 32 activations
aie::vector<uint4, pr*8> a_cc_0 = aie::load_v<pr*8>(qs_ptr);
aie::vector<float, pr*8> a_cc_f32_0 = aie::to_float(a_cc_0, 0);   // unpack 4-bit
```

One multiply per unpacked weight. Measured rate: 512 MACs per 140 bundles =
**3.7 MAC/cycle/core** **[measured]**.

**Prefill** (`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc`) uses the
native AIE matmul:

```cpp
constexpr int r = 8, s = 8, t = 8;
using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;   // 2x2 register-blocked
```

Measured rate: 9797 GFLOP/s across 32 cores = **98 MAC/cycle/core**
**[measured]**.

**26x gap.** So do not widen the existing loop — that keeps the slow form.
Build: unpack the 32x256 weight block once into bf16, then `aie::mmul` it
against a 256x16 activation tile. The 4-bit unpacking cost then spreads over 16
rows instead of being paid per row.

### Does it fit on the tile?

Yes. Projection core L1 today **[measured]**, from `fused_decode.py:1038`:

| buffer | batch 1 | batch 16 |
|---|---|---|
| `xblk_l1` `[256]` bf16 | 512 B | 8 KB |
| `wblk_l1` `[2560]` bf16 | 5 KB | 5 KB (unchanged) |
| `yacc_l1` `[32]` f32 | 128 B | 2 KB |
| `rcache_l1` | ~0.5 KB | ~10 KB |
| `ypair_l1` `[80]` bf16 | 160 B | 2.0 KB |
| **total** | **~6 KB** | **25.5 KB of 64 KB** [measured, section 5b] |
| + unpacked weight tile (new) | — | **+16 KB → 41.5 KB** |

The activation was never held whole on the tile — it streams in 256-element
chunks. The K-wide buffers (`rms_l1[K]`, `qkv_l1[M]`, `ropeq_l1[DQ_PADDED]`) are
on the rms/rope/glu tiles, not here.

The last row is the one thing this kernel adds that the GEMV does not need: the
unpacked weight tile `aie::mmul` reads from. It still fits — 41.5 KB against a
54 KB budget — but see section 5b, because it is what caps the batch.

### The rest of the superkernel

- **Attention**: `aq_l1 + ao_l1` at batch 16 is 64 KB, and the whole CU is 86 KB
  **[measured]** — over the tile budget. Tile the queries 8 at a time and keep
  the KV block loaded. The KV block is shared by every query in the tile, so it
  is read once — this is a win, not just a cost. Per-model figures in section 5b.
- **Output packing**: widen the packet rather than repeating it, and give the
  group gather one extra dimension so the emitter-major blocks land token-major.
  Every descriptor stays legal to batch 511. Worked through in section 5c.
- **rms/rope/glu tiles**: loop over the 16 rows instead of holding them.
- **`rcache`**: becomes per-row.
- **Attention mask**: verify needs an intra-block causal mask that does not
  exist today, and draft needs none at all. Section 6, step 2.
- Unchanged: weight streaming, region-major KV, layer fusion, `DecodeInstsGen`
  instruction patching.

---

## 5. Is it worth it — memory bound or compute bound?

Today **[measured]**: prefill is compute bound, decode is memory bound.

Arithmetic intensity at batch M on q4nx weights is `3.2 x M` FLOP/byte (from
0.625 B/param). So:

| | batch | FLOP/byte | bound |
|---|---|---|---|
| prefill | 2048 | ~2048 | compute |
| decode | 1 | 3.2 | memory |
| **DFlash draft/verify** | **16** | **51.2** | **at the crossover** |

### Measured: the crossover is at batch 14.8, just below the block size

The kernel is built (`kernels/q4k_mm.h`) and swept (`bench_q4k_mm.py`). Cycles
per 32x256 weight block on one core, fully unrolled so static bundle count
equals dynamic cycles **[measured]**:

| batch | unpack | mmul | total | MAC/cycle | cycles/token |
|---|---|---|---|---|---|
| 1 (today's GEMV) | — | — | 2240 | 3.7 | 2240 |
| 16 | 1409 | 1965 | 3374 | 38.8 | **211** |
| 32 | 1409 | 3533 | 4942 | 53.0 | 154 |

`cycles/block = 1806 + 98.0 x batch` from the two points; the unpack-only build
measures 1409 directly, well under that intercept because the multiply itself
carries a fixed ~400-cycle term at these shapes rather than being proportional
to the batch.

**Batch 16 costs 10.6x fewer cycles per token than the batch-1 GEMV** (211 vs
2240), because the unpack is paid once for 16 tokens instead of once per token.
It is still 41.8% of the block cost at batch 16.

Against the memory side, using llama-3.2-1B q4nx (9440 blocks/core/token, 1.57
GHz, 19.6 ms/token measured on a Krackan box):

| | compute | memory | bound |
|---|---|---|---|
| batch 1 (today) | 13.5 ms | 19.6 ms | memory |
| **batch 16** | **20.3 ms** | 19.6 ms | **compute** |
| batch 32 | 29.7 ms | 19.6 ms | compute |

**Crossover at batch 14.8**, and DFlash's block size is 16 — so batch 16 sits
just *outside* it, compute bound by about 3.5%.

> **This corrects an earlier figure in this document.** The first sweep put the
> crossover at 16.7 with batch 16 inside it. Two later fixes to the kernel moved
> it, both described under "the correct layout is not free" below: the multiply
> costs 10% more at batch 16 in the operand order the layout derivation forced,
> and the unpack costs more again. The numbers above are the corrected ones and
> are still a **lower bound** — they use the cheap wrong-layout unpack, which is
> the only one that can be fully unrolled.

Being 3.5% over the memory floor is not a problem for DFlash — the traffic-based
speedup in section 5's result table is optimistic by about that much, and 4.87x
does not become 4.7x in any way that changes a decision. What it does remove is
the margin: there is no longer slack to absorb DMA stalls, so the open question
about static counts under real memory pressure now matters more, not less.

Two consequences:

- **Block size 16 is right for this hardware, and there is no headroom above
  it.** A b32 checkpoint would be compute bound (29.7 vs 19.6 ms) and would not
  return proportionally more.
- DFlash's own advice to use `block_size <= 5` for quantized targets does not
  apply here. That is a GPU kernel property; on NPU2 the measurement says 16.

Caveats on the measurement. Bundle counts are issue slots: they assume no DMA
stall, so this bounds the compute side and nothing else. The 19.6 ms memory
figure is measured wall time at batch 1, which also contains whatever stalls
exist, so using it as a pure memory floor is approximate. The kernel itself is
now numerically validated on device (section 5d).

One number to note: the batched kernel reaches **38.8 MAC/cycle/core at batch
16, not the prefill's 98**. K=256 per block is short and the 2x2 blocking has
little to reuse across it. The crossover lands as high as 14.8 only because the
unpack term is large enough to dominate.

### The correct layout is not free

The first version of this kernel unpacked weights into whatever order fell out
of the packed nibbles and multiplied them, which measures the right *number* of
operations but not the right kernel. Deriving the real layout — from q4_k.h's
packed order on one side and `q4k_mmul`'s pointer walk on the other — changed
three things and cost about 6%.

**A real bug, first.** The scale/min group was indexed by the global unpack step
rather than the step within the row half. Any block with `MROWS > 16` therefore
read past the end of its scale array on the second half: at 32x256, step 32 asks
for group 8 of 8. Fixing it moves the unrolled unpack 1397 → **1409**, which is
the honest cost of the index arithmetic being right.

**The weights had to become the B operand.** `aie::mmul<r,s,t>` takes B as
`[s][t]` row-major — `[contraction][output]` — and that is exactly the order a
128-nibble chunk already has. Taking the weights as A instead needs an 8x16
transpose per chunk, and **`aie::transpose` cannot do it**: the 16-bit
specialization on AIE2 covers 32, 16 and 8 elements, and a 128-lane bf16 vector
fails to instantiate. So the kernel computes `Yt = Xt * Wt` and the split into
two output tiles is two `aie::filter_even/odd` calls with a chunk size of 8.

That swap is what costs the 10% **[measured]**:

| mmul cycles | old (weights as A) | new (weights as B) | 2x2 shape old → new |
|---|---|---|---|
| batch 16 | 1787 | **1965** (+10.0%) | 4x2 → 2x4 |
| batch 32 | 3533 | 3533 (unchanged) | 4x4 → 4x4 |

Batch 32 being *identical* is the check on the explanation: at batch 32 the
register blocking is 4x4 either way, so there is nothing to lose. At batch 16 it
flips from 4x2 to 2x4 and that asymmetry is the whole difference.

**Two side effects, one good and one bad.** The good one: `Yt` is **token-major**,
which is the order the egress consumer wants — it removes the 2D group gather
section 5c worked out. The bad one: `Xt` now has to arrive tile-blocked the way
`mm_aie2p`'s A operand is, tile `(z,i)` at `(i*rowA + z)*64`, which a plain
`[BATCH][KCOL]` buffer is not. That is a strided memtile BD rather than compute,
and the memtile has the dimensions spare (section 5c), but it is not nothing.

**The 10% cannot be recovered on the host, and the reason is the scale
broadcast.** The obvious idea is that all of this is permutation, permutation is
free at requant time, and `pack_q4k_cascade` already permutes heavily — so pack
in mmul-A order for batched builds and keep the cheaper 4x2 blocking. It does
not work, for a reason independent of how the host packs:

`q4k_unpack_step` applies scales by replicating a 16-lane scale vector across
the 128-lane chunk, so lane `l` gets `s16[l % 16]`. That is only correct if the
chunk's row index *is* `l % 16` — column-major within 16 rows, which is exactly
mmul's **B** order. In A order a contiguous 128 elements are two row-major 8x8
tiles, so `row(l) = 8*(l/64) + (l%64)/8`, and lanes 0 and 16 have the same
`l % 16` but different rows. `s16[0]` would have to hold two different scales at
once. No packing order fixes that, because it is a property of the broadcast,
not of the data **[measured — enumerated over all 128 lanes]**.

Storing scales pre-expanded instead would take a 32x256 block from 5120 to
36864 bytes, 7.2x, on a decode that is memory bound. That is not a trade worth
discussing.

So **B is forced, not preferred**, and the 10% is the price of the kernel being
correct rather than an artifact to optimise away.

Nor can re-blocking recover it. The 2x2 register blocking is a choice — at batch
16 it runs one z iteration and two j iterations, and covering all four B tiles at
once would cut loads per mac from 1 to 0.75. Built and measured **[measured]**:

| | 2x2 | 2x4 | |
|---|---|---|---|
| batch 16 | 1965 | 2035 | +3.6% |
| batch 32 | 3533 | 3786 | +7.2% |

Eight accumulators cost more than the loads they save, at both batches.
`q4k_mmul_2x4` is kept in the header so the idea is not re-tried.

One more thing that did not work, worth recording because it is the obvious way
to write this: making the blocking a template parameter with `MMUL C[RB][CB]`
and unrolled index loops. The accumulators go to the stack rather than
registers and the frame overflows the same displacement field as everywhere else
in this document — `immediate operand value -52032 is out of range`. The
accumulators have to be named locals, so each blocking has to be spelled out.

The correct unpack also costs more than the wrong one, and by how much is only
loosely known: it **cannot be fully unrolled** — that crashes the Peano backend
(`Register not in mBMs`, AIE2P assembly printer) — while the rolled form, which
is what a real build uses, compiles fine. Rolled static size is **87 bundles
against 68** for the contiguous store. That is not a cycle count and 64
iterations do not multiply it, so the only honest statement is that the totals
above are a lower bound. `bench_q4k_mm.py --noperm` restores the wrong-but-
unrollable unpack, which is how the exact numbers above are still obtainable.

### Folding two blocks per call buys margin, but not as built

Running the same sweep with two q4k blocks folded into one `q4k_mmul` call
(`--kcol 512`), normalised back to a 32x256 block **[measured]**:

**These fold numbers predate the layout fix** and are quoted against the
pre-fix baseline (mmul 1787 at batch 16). The *relative* gains should carry —
folding changes the contraction, the layout fix changed the operand roles, and
they act on different things — but nothing here has been re-measured since, so
treat the absolute columns as historical.

| per 32x256 block, batch 16 | kcol 256 | kcol 512 | |
|---|---|---|---|
| unpack | 1397 | 1389 | unchanged, as expected |
| mmul | 1787 | 1547 | **-13.4%** |
| total | 3184 | 2936 | **-7.8%** |
| MAC/cycle | 41.2 | 44.6 | +8.3% |

The unpack term is identical, which is the cross-check that the normalisation is
right — folding cannot change the per-weight unpack cost. The gain is entirely
in the multiply, from a longer contraction giving the 2x2 blocking more to reuse.

Effect on the roofline, on the pre-fix baseline: compute at batch 16 dropped
from 19.1 to **17.6 ms** against the 19.6 ms memory floor, moving that
crossover from 16.7 to about **19.3** **[estimated from the single batch-16
fold-2 point]**. Applying the same -7.8% to the corrected 20.3 ms gives 18.7 ms,
which would put batch 16 back inside the memory floor — so the fold is now the
thing that would restore the margin the layout fix removed, rather than a
nice-to-have. It still does not fit L1.

**As written, it does not fit.** The kernel's own two buffers are 48 KB of 64 —
a 32 KB unpacked tile and a 16 KB activation tile — but the proj core is not
empty around them. Counting the whole core it is **57.5 KB against a 54 KB
budget** **[measured]**, and that is still with `xblk` at its 256-column size;
at 512 columns it is 65.5 KB. The ceiling is batch 9, under the 16 DFlash needs.
The `rcache` (9.5 KB at batch 16) and `wblk` (5 KB) are what a kernel-only count
leaves out.

There may be a way to keep the gain inside the budget, and it turns on a
question that is **not yet settled: where the 13.4% actually comes from.** Two
candidates, with opposite consequences:

- **Accumulator traffic.** `q4k_mmul` does `load_v(pC)` on entry and
  `store_v(pC)` on exit, so two kcol-256 calls pay that twice where one kcol-512
  call pays it once. If this is the cause, the fix is to hoist the accumulator
  and the gain survives at a 16 KB scratch.
- **Scheduling window.** The bench builds are fully unrolled, so there is no
  loop overhead to amortise at all; a longer straight-line body simply gives the
  VLIW scheduler more independent work to pack. If this is the cause, then two
  adjacent kcol-256 bodies in one function already get it, and no restructuring
  is needed — just don't put a function boundary between them.

`q4k_mm_chunked` in `q4k_mm.h` is the discriminating experiment: N contraction
chunks through **one** KCOL-wide scratch, accumulated into one C, with
`[[clang::always_inline]]` at the call sites so the two bodies land adjacent
instead of being outlined and shared. `bench_q4k_mm.py --chunks 2` runs it.

**It does not build, and the reason is a hard backend limit** **[measured]**:

```
fatal error: error in backend: immediate operand value -33152 is out of range
                               [-32768, -64]
Running pass 'AIE2 Assembly Printer' on function
                               q4k_mm_chunked<32, 128, 16, 2>
```

Two inlined fully-unrolled bodies need a frame past what AIE2's 16-bit
load/store displacement field can address. It fails even at kcol **128**, where
the two chunks together are half the code of the kcol-256 single body that
compiles fine — so this is frame size, not code size, and shrinking the shape
does not get around it. Same family as the Peano 9-bit immediate that forced
`tile_n=16` on `llama32_1b_int4` (section 6, step 1).

Rolling the loop instead would compile, but then the trip count lives in a
register and both variants emit an identical body — the failure mode already
documented for build-diffing. **So the static-bundle method cannot answer this
question at all**, and the mechanism stays open until the kernel is wired in and
timed on device. Both folds stay parked behind it.

Worth keeping for the real kernel regardless: a fully-inlined multi-chunk body
overflows the AIE2 frame-offset immediate. Anything that tries to inline several
unrolled matmul bodies into one core function will hit this.

Guessing this mechanism instead of measuring it is how the BFP16 flag was got
backwards earlier (section 5), so it is left open here rather than asserted.

Note also that the fully-unrolled bench build at kcol 512 took ~22 minutes to
compile; that is a property of the measurement method, not of the kernel.

## 5b. Does batch 16 fit in L1? (measured)

The compute side says batch 16 is affordable. The other half of step 2 is
whether the tiles have room for 16 rows of activations. `batch_l1_budget.py`
reads the real buffer shapes out of `fused_decode.py` per model and reports it.
At batch 16, against 64 KB minus the model's `DECODE_STACK` **[measured]**:

| model | proj core | max batch | attention CU | max query tile | qkv_l1 max batch |
|---|---|---|---|---|---|
| **qwen3-4b** (DFlash target) | 25.5 KB FITS | 38 | 86.0 KB OVER | **8** | **4** |
| llama-3.2-1b | 25.0 KB FITS | 39 | 46.0 KB **FITS** | 19 | 9 |
| llama-3.2-3b | 25.0 KB FITS | 39 | 86.0 KB OVER | 8 | 5 |
| gemma3-4b | 26.0 KB FITS | 37 | 84.0 KB OVER | 8 | 6 |
| phi4-mini | 25.0 KB FITS | 39 | 86.0 KB OVER | 8 | 5 |
| qwen2.5-7b | 34.5 KB FITS | 26 | 78.0 KB OVER | 10 | 6 |
| qwen3-8b | 29.0 KB FITS | 35 | 86.0 KB OVER | 9 | 4 |
| llama-3.1-8b | 31.0 KB FITS | 32 | 86.0 KB OVER | 9 | 4 |

Three findings:

- **The proj cores need no L1 work for the activations.** They fit batch 16 on
  every model with room for 26-39. That is the tile doing the weight-streaming
  work, and it confirms the premise of section 4 from the other direction: the
  activation buffers there are a 256-element chunk and a 32-element accumulator,
  not anything K-wide.

  The new kernel does add one buffer, though, and the table above does not
  include it: `q4k_unpack_block` has to materialise the unpacked weight tile
  before `aie::mmul` can read it, and today's GEMV never does. At the 32x256
  shape section 5 benchmarks that is **16 KB**, taking qwen3-4b's proj core from
  25.5 to **41.5 KB** and its ceiling from batch 38 to **25** **[measured]**.
  Batch 16 still fits with 12 KB to spare, so this changes no verdict — but it is
  the single largest buffer on the core once it exists, and it, not the batch,
  is what sets the ceiling.

  It is also tunable rather than fixed. The mmul contracts over the columns, so
  the tile can be chunked and accumulated at the same total unpack cost: 32x128
  is 33.5 KB (ceiling 31) and 32x64 is 29.5 KB (ceiling 35) **[measured]**.
- **Attention needs a query tile of 8** on every head_dim>=128 model; only
  llama-3.2-1b (head_dim 64) fits a full 16. The cost is small: `ak`/`av` hold
  the KV *block*, shared by every query in the tile and unscaled, so tiling by 8
  means reading that block from L2 twice instead of once. DDR traffic is
  unchanged, which is the part that matters.
- **rms/rope/glu need row tiling**, at 4-9 rows depending on model, with
  `qkv_l1` the tightest. These do elementwise and reduction work on activations,
  so tiling costs loop overhead and L2 traffic rather than DDR.

## 5c. Does the wire survive batch 16? (measured)

The remaining piece of step 2 is what the batch does to the data moving in and
out of the proj cores. `batch_wire.py` reports it. Sizes are **[measured]** from
the builder; the descriptor limits are read out of mlir-aie's AIE2p target model
(`AIETargetModel.h`).

**Out.** There are two ways to carry a block of tokens through the egress:
**widen** the packet B times, or **repeat** it B times per round. Widen is the
one to build — it leaves `N_ROUNDS`, the BD count and the host instruction
stream untouched, where repeat multiplies all three, and the builder already
names shim BD exhaustion as a live constraint on round count.

Every descriptor on that path is legal at batch 16, on all eight models
**[measured]**:

| descriptor | tile | batch 1 | batch 16 | limit |
|---|---|---|---|---|
| `outA` put | core | 514 elem | 514 elem | 16383 words |
| `outA` get | memtile | 130 elem, 1D | 514 elem, **2D** | 4 dims, wrap 1024 |
| `toMain` put | memtile | 130 elem | 2050 elem | 131071 words |
| `outY` put | memtile | 514 elem | 8194 elem | 131071 words |

The path is identical on every model, because `PAYLOAD = N_PAIRS * PAIR_ROWS *
ROW_BLOCK` is `NCX*NCY*ROW_BLOCK = 512` whichever side of the paired/non-paired
split a model falls on. Headroom is to **batch 511**, bound by the `outY` BD
length. Egress L2 peaks at 16 KB of 512 KB on the busiest memtile column.

The one thing that actually changes is the `outA` gather, and it changes by one
dimension. An emitter sends its B blocks back to back, but the consumer wants
token-major, so the group memtile has to land them strided: `sizes=[16, 32]`,
`strides=[128, 1]`. Both wraps are under the memtile's 1024 and the step is
under 2^17, so it is a legal 2D memtile BD — 4 dims are available and 2 are
used. The routing header also stops mattering: 0.39% of the packet at batch 1,
0.024% at batch 16.

**In, and this is the one that bites.** The `_gemv` inner loop pairs one X chunk
with one weight block, and only X scales with the batch **[measured]**:

| per inner-loop step | batch 1 | batch 16 |
|---|---|---|
| W chunk (a packed q4k block) | 5120 B | 5120 B |
| X chunk (`COL_BLOCK` x batch) | 512 B | 8192 B |
| X / W | 0.10 | **1.60** |

**X overtakes W at batch 10** — `BLOCK_BF16`/`COL_BLOCK` = 2560/256, exactly.
Above that the proj core's input port carries more activation than weight, which
inverts the assumption the whole engine was built on. Per layer per core it is
6.31 MB of X against 3.94 MB of W at batch 16, up from 0.39 MB against 3.94 MB.

It is affordable, but not by much. Combining the byte count with the measured
cycles/block line from section 5 (`1438 + 109.1*b`), the X broadcast needs
**2.57 B/cycle** at batch 16 and 3.32 at batch 32. Both are under a 32-bit
stream, and because bytes and cycles are both linear in the batch the demand
approaches an asymptote of `COL_BLOCK*2 / 109.1` = **4.69 B/cycle** rather than
growing without bound — so the broadcast saturates at **batch 76** and never
before. That last figure divides a measured slope by an assumed stream width;
the stream width is a flag on the tool, not something measured here.

The fix, if it is ever needed, is **not** the fold-2 variant from section 5.
That folds two blocks along the *contraction*, and two column blocks need two
different X chunks, so W and X both double and the ratio does not move at all —
`--kcol 512` buys roofline margin and nothing here (and, per section 5, does not
fit L1 in the form it was measured).

Folding along the *rows* does move it, and it is now measured rather than
extrapolated: `bench_q4k_mm.py --mrows 64 --batches 16` gives 2753 unpack + 3233
mmul = **5986 cycles** per 64x256 block **[measured]**. Normalised back to a
32x256 block that is:

| per 32x256 block, batch 16 | MROWS 32 | MROWS 64 | |
|---|---|---|---|
| unpack | 1397 | 1376 | -1.5%, i.e. unchanged — the cross-check |
| mmul | 1787 | 1616 | **-9.5%** |
| total | 3184 | 2993 | **-6.0%** |
| MAC/cycle | 41.2 | 43.8 | +6.3% |

As with the contraction fold, **these predate the layout fix** (section 5) and
are quoted against the pre-fix 1787 baseline. Note also that MROWS 64 at batch
16 gives a 2x8 register blocking, which is *more* asymmetric than the 2x4 the
fix landed on — so re-measuring it after the fix is worth doing before relying
on the -9.5%.

So row folding buys about as much compute as the contraction fold did (-6.0%
against -7.8%) *and* halves the X feed. Crossover goes from batch 10 to 20,
demand at batch 16 from 2.57 to **1.37 B/cycle**, and the asymptote from 4.69 to
**2.63** — under the stream width, so the broadcast stops being a limit at any
batch.

Worth noting the two cycle lines are not related by a factor of two. Fitting the
measured points gives `1438 + 109.1*b` at MROWS 32 and `2876 + 194.4*b` at
MROWS 64: the intercept doubles exactly, as unpack must, but the slope comes in
at 194.4 rather than 218.2 because the longer row block gives the 2x2 register
blocking more to reuse. `batch_wire.py` takes the line as `--cyc0/--cyc1` rather
than deriving one from the other.

It is not free in L1, and the interaction with the unpack scratch above is the
thing to watch. `MROWS=64` at the full 256-column contraction wants a 32 KB
scratch, which puts the proj core at 57.5 KB and **over** the 54 KB budget
**[measured]**. Chunk the contraction to 64 columns and the same fold costs
33.5 KB with a ceiling of batch 31 — comfortable on capacity.

But chunking the contraction shortens what each `q4k_mmul` call accumulates
over, and section 5 measured that exact effect in the other direction: going
from 256 to 512 columns cut the multiply 13.4%. Chunking to 64 may give that
back. Whether it does is the same question section 5 leaves open for
`q4k_mm_chunked` — and that experiment does not build, so it stays open in both
directions until there is on-device timing. Both folds are parked behind the
same measurement.

### The bf16 drafter is not cheap

The drafter is 13% of the target's parameters but ships in bf16, which is 3.2x
denser than q4nx. Bodies only (the LM head is counted separately in the table
above, where it turns out to matter more than this does):

| | params | format | body traffic |
|---|---|---|---|
| drafter | 0.54 B | bf16 as shipped | **1.009 GB** |
| drafter | 0.54 B | if quantized to q4nx | 0.315 GB |
| target | ~4.0 B | q4nx | 2.271 GB |

All three **[measured]** from the builder via `dflash_traffic.py`. As shipped the
drafter body is 44% of the target body. Quantizing it to q4nx is also required
to run it on the superkernel at all (same approach `qwen25_7b_q4nx` uses), since
that engine only takes q4nx weights.

Note the asymmetry: quantizing the **drafter** cannot break correctness — the
target still decides — it only lowers the acceptance rate. Quantizing the
**target** changes what "correct" means and also lowers acceptance, because the
drafter was trained against the bf16 target.

### End result, from builder geometry

Now that `fused_decode.py` has `qwen3-4b` and `qwen3-4b-draft` entries, the
weight traffic comes off the builder instead of being estimated. `dflash_traffic.py`
reports it. Bytes are **[measured]** from the builder; the 46 GB/s and tau=6 are
inputs.

| | GB | ms @ 46 GB/s |
|---|---|---|
| target body, 36 layers | 2.271 | 49.4 |
| LM head | 0.246 | 5.3 |
| **verify pass** | **2.517** | **54.7** |
| draft body, 5 layers (q4nx) | 0.315 | 6.9 |
| `fc`, 5 taps → 1 (q4nx) | 0.020 | 0.4 |
| LM head (tied) | 0.246 | 5.3 |
| **draft pass** | **0.582** | **12.6** |
| **per iteration** | **3.098** | **67.4** |

| at tau=6 | ms/token | tok/s | vs today |
|---|---|---|---|
| Qwen3-4B q4nx decode today | 54.7 | 18.3 | 1x |
| DFlash, bf16 drafter (as shipped) | 13.9 | 71.9 | **3.94x** |
| DFlash, drafter quantized to q4nx | 11.2 | 89.1 | **4.87x** |

Three things this surfaced that parameter counting did not:

- **The LM head is 42% of the q4nx draft pass** (0.246 GB against a 0.315 GB
  body). A 0.5 B drafter and a 389 M-parameter head are comparable objects, and
  the draft pays for the head just as the verify does. It is 19% of the draft
  pass when the drafter body is bf16.
- **The `fc` linear is not a rounding error, though it nearly is.** DFlash's
  context fusion is `Linear(5*2560, 2560)` = 32.8 M parameters, a **third of a
  whole Qwen3-4B layer**, and it is easy to omit from a layer-count estimate. At
  q4nx it is 6.5% of the drafter body and moves the headline from 4.91x (before
  it was counted) to 4.87x; in bf16 it would be 21% of a q4nx body. It also
  **decomposes for free** — `fc(concat(h1..h5)) == sum_i W_i @ h_i` — so it is
  five accumulating hidden→hidden projections at `I2=5, J2=5`, a shape qwen3-4b
  already has.

  Both forms are legal, though: the undecomposed 12800-wide contraction gives
  `J2 = 12800/512 = 25`, an integer, so the proj phases *can* express it. The
  reason to prefer the decomposition is L1, not legality —
  `RCACHE_LEN = 2*max(J2P)*8`, so a `J2=25` phase raises the per-core reduce
  cache from 304 to 400 elements, **+3.0 KB at batch 16** **[measured]**, on a
  core already at 41.5 KB of 54. Reusing an existing `J2` costs nothing.
- **Break-even is at tau = 1.23** (1.52 with the bf16 drafter). The loop pays
  for itself if barely more than one token per block survives verification. That
  is the strongest robustness result here: the open question about acceptance
  rate on a quantized target scales the *size* of the win but has to be
  catastrophic to eliminate it. At tau=3 -- half the paper's rate -- q4nx still
  gives 2.44x.

KV traffic is excluded throughout, which understates DFlash: it is unchanged per
call and a batched call reads it once for the whole block, so the advantage grows
with context.

### What it costs to keep resident

Traffic is one thing; DFlash also runs **two models at once**, and both keep a
KV cache. At context 2048, from the builder **[measured]**:

| | q4nx drafter | bf16 drafter |
|---|---|---|
| target weights | 2400 MB | 2400 MB |
| drafter weights + `fc` | 320 MB | 965 MB |
| target KV | 288 MB | 288 MB |
| drafter KV | 40 MB | 40 MB |
| **total** | **3048 MB** | **3693 MB** |
| added over a plain decode | +13% | +37% |

This is a second and independent reason to quantize the drafter, beyond the
throughput one: on a laptop where the NPU shares system memory, 645 MB is a
real difference, and it buys the 4.87x instead of 3.94x at the same time. The
drafter's own KV cache is 40 MB and easy to forget — it is a full-depth cache
over the accepted prefix, just five layers deep instead of thirty-six.

---

## 5d. Is the kernel correct? (measured, on device)

Everything above section 5 is a cost model. None of it says the kernel computes
the right answer, and until now nothing did — the kernel uses AIE intrinsics, so
it will not run on the host, and its layout was *derived* from two independent
sources rather than checked against one. `q4k_mm_gate.py` closes that: one core,
one herd, weights and activations in, `Yt` out, compared against numpy.

**The comparison is `==`, in every mode.** That is worth stating plainly because
it was not obviously achievable. A layout bug does not perturb an answer, it
permutes it, so a tolerance wide enough to pass is wide enough to hide one. It
passes bit-exactly at batch 16 and 32, one and two blocks, several seeds
**[measured]**.

**It found two faults on the first run, and neither was visible to any static
check.** Both produce a plausible wrong answer rather than a crash:

- **The operands were in the wrong roles.** Section 5's layout fix moved the
  weights from `aie::mmul`'s A operand to its B operand — the unpack was
  rewritten, the header was rewritten, and `q4k_mm_block` went on calling
  `q4k_mmul(W, B, C)` with the weights first. The layout half of the swap
  landed; the call sites did not.
- **`sizeof(q4k_block_t)` is 9216, not 5120** **[measured]**. `uint4` is
  byte-addressed — `load_v<N>` reads N nibbles but pointer arithmetic counts
  bytes — so `uint4 qs[8192]` reserves twice the space it uses. It has never
  mattered, because every production call site casts a `bf16*` at the boundary
  and steps blocks on the `bf16` side. It bites the moment new code writes
  `A + b`, which lands 4096 bytes into the next block's nibbles and reads them
  as scales. `q4k_mm.h` now carries `Q4K_BLOCK_BF16` and a static_assert that
  fires if the struct is ever made exact.

Getting to `==` required measuring two properties of the arithmetic that a
datasheet would not have given up, both now modelled in the gate:

| property | what it is | how it was found |
|---|---|---|
| `aie::mmul` multiplies in **bfp16** | groups of 8 share an exponent taken from the group max, 7 significant bits each | fitted until it reproduced 512/512 elements bit-for-bit |
| every bf16 rounding on the core goes **toward −∞** | not to nearest; floor, not toward-zero | round-to-nearest matches 4890/8192 unpacked weights, floor matches 8192/8192 |

The rounding mode is the part with consequences. Rounding down costs half an ulp
*every time, in the same direction*, so its contribution to a dot product is
linear in K and proportional to the **mean** of the operands rather than their
magnitude. That makes the accuracy answer depend entirely on the data:

| weights | K=256 | K=512 |
|---|---|---|
| independently drawn scale and min (large positive mean) | −7.7% bias | −11.2% bias |
| real q4k min/max codec (centred) | 1.3% rms, no bias | 1.3% rms, no bias |

**Same kernel, same build, only the test data changed.** The first row is what a
plausible-looking random codec produces and it is not representative; q4k puts
`min` at the group minimum, so dequantized weights are centred wherever the
weights are, and the bias cancels. **1.3% rms, flat in contraction depth, is
the number to carry forward** — the batched path costs about that much accuracy
against an exact fp32 matmul of the same inputs.

The instrument that settled the layout question is kept as `mmul_probe.py`: one
`aie::mmul` per probe with operands chosen so the answer names its own layout
(`A = 1..64, B = I` gives back A in A's order). It confirms plain row-major
`A[r][s]`, `B[s][t]`, `C[r][t]` **[measured]**, which is what let the fault be
localised to the call site instead of the API. Reach for it before re-deriving
anything about tile order.

### And does it agree with the GEMV it replaces?

That is a separate question, and `q4k_mm_gate.py` cannot answer it: the two
kernels do genuinely different arithmetic for the same result. The GEMV never
builds `W` — it factors the `+min` term out as `min[r,g] * (sum of x over group
g)`, which is what `b_col_reduce_add` and the whole `rc`/`fill` cache exist for.
The batched path materializes `w = q*scale + min` elementwise, because
`aie::mmul` needs a real B operand, and then multiplies in bfp16. Same maths on
paper; different roundings, in different places, in different precisions.

`proj_qmm_gate.py` runs **both, in one launch, off the same L1 weights and the
same activations**, so a difference is attributable to the kernels and nothing
else. Measured **[measured]**:

| | batch 16, K=256 | batch 16, K=512 | **batch 8, K=256** | **batch 8, K=512** |
|---|---|---|---|---|
| GEMV vs exact fp32 | 0.917% rms | 1.005% rms | 0.907% rms | 0.951% rms |
| batched vs exact fp32 | 1.274% rms | 1.456% rms | 1.279% rms | 1.401% rms |
| GEMV vs batched | 1.599% rms | 1.711% rms | 1.570% rms | 1.664% rms |
| ratio | 1.39x | 1.45x | **1.41x** | **1.47x** |

**Batch 8 was added after the block size moved, and it had to be: the gate only
ever ran at 16.** It refused to run below 16 on a stale guard — `q4k_mmul`'s
`BATCH % 16 == 0` assert, which `q4k_mmul_any` had already made obsolete — so
the block size the analysis recommends was the one block size the projection
was never checked at. It passes, and at the same error as 16.

That check also found the real limit, which is not in `q4k_mm.h` at all:
**`proj_qmm_mm_flush_row` de-tiles for `aie::mmul<8,8,8>` and breaks at batch
4.** It reads C tile `(z, j)` at `(j*RA + z)*64` with `RA = BATCH/8` — correct
at 8, 16 and 32, where it coincides with `q4k_mmul_small`'s layout at `RA == 1`.
At batch 4 the tile is `aie::mmul<4,8,8>`, `size_C` is 32 not 64, and `RA`
integer-divides to **zero**, so every `j` reads tile 0. It would have compiled
and returned a plausible wrong answer. Now a `static_assert` and a gate refusal;
`q4k_mm.h` itself is bit-exact at batch 4 and `q4k_mm_gate.py --batch 4` covers
it. **If the block size ever moves to 4, that flush needs a `size_C=32`
variant.**

**The batched path costs about 1.4x the GEMV's error, and that ratio is roughly
flat in contraction depth and in the batch.** Both are near-unbiased and both are around 1%. That
is the price of the swap, stated as a number rather than assumed either way —
and the framing that matters is the third row: swapping moves the projection
output by ~1.7% rms, which is a real change, not a rounding difference. Whether
1.4x matters is a question for `llms/verify/` on a real model, not for a kernel
test.

Worth noting what the same gate says about the *incumbent*: the shipping GEMV is
itself ~1% off exact fp32. Neither path is the truth.

What the gates do **not** cover: one core, one row-block, no cascade, no egress,
no DMA pressure. They settle the arithmetic and the layout of the swap. The
engine around them is the rest of step 2.

---

## 5e. Attention does not amortize, and it changes the answer (measured)

Section 5's roofline counts **projection weight blocks and nothing else**. That
is only sound if attention is small on the compute side, and it is not — for a
reason that is specific to batching and invisible at batch 1:

| | at batch 16 |
|---|---|
| projections | 16 tokens share one weight block. Compute scales, DDR traffic does not. *This is the entire point of batching.* |
| attention | every query re-reads the whole KV cache. 16 queries is 16x the `attn_qk`/`attn_kv` calls. **Nothing is shared.** |

One term amortizes and the other does not, so their ratio at batch 16 is 16x
what it is at batch 1. A term that is 19% of batch-1 compute is not 19% of
batch-16 compute. `bench_attn.py` measures it, by the same bundle-counting
method as section 5.

**First, a measurement trap that had to be cleared.** Attention's contraction
loops run `colQ = DH/8` times with the trip count in a register, so a rolled
build reports one *loop body* — and reports the **same number for DH=64 and
DH=128**, which is how you can tell it is not measuring a call. `attn_kv_blk` is
320 bundles rolled and **1500 unrolled** **[measured]**, a 4.7x undercount. The
kernels now carry `ATTN_BENCH_UNROLL` (bench-only, engine builds rolled) and the
tool prints both columns.

Per 16-key block, per attention CU, unrolled **[measured]**:

| | rolled | unrolled | |
|---|---|---|---|
| `attn_qk_blk` | 749 | 868 | llama-3.2-1b, DH=64 |
| `attn_kv_blk` | 320 | 1500 | |
| per block | 1069 | **2368** | 2.2x |
| `attn_qk_blk` | 749 | 1343 | qwen3-4b, DH=128 |
| `attn_kv_blk` | 320 | 2917 | |

That is ~7 MAC/cycle/core — against the 98 the prefill matmul reaches and the
38.8 the batched q4k kernel reaches. Attention is the slowest thing on the chip
per MAC, and batching multiplies how much of it there is.

**The roofline with the term put back**, llama-3.2-1b at P=2048, 1.57 GHz
**[measured]**:

| batch | proj | attn | serial | overlap | memory |
|---|---|---|---|---|---|
| 1 | 13.47 | 3.11 | 16.58 | 13.47 | 19.60 |
| 16 | 20.29 | **49.81** | **70.10** | 49.81 | 19.60 |
| 32 | 29.71 | 99.62 | 129.33 | 99.62 | 19.60 |

`serial` is the phase structure as built — qkv proj → attention → o proj → glu,
each a barrier, so the terms add. `overlap` is a bound nothing achieves, with
attention hiding entirely behind the projections. **At batch 16 both exceed the
memory floor.**

| | section 5 | with attention |
|---|---|---|
| crossover (largest batch still memory bound) | 14.8 | **2.4** |
| batch 16 vs the memory floor | 1.04x | **3.58x** |
| attention's share of batch-16 compute | not counted | **71%** |

The batch-1 row is the model's one check against reality, and it passes: 16.6 ms
of compute against a **measured** 19.6 ms wall time, memory bound with 15%
slack, which is what the decode is known to be.

### What this does and does not overturn

It does **not** say batching is pointless. Batch 16 still delivers 16 tokens in
70 ms against 314 ms one at a time — **4.5x** **[estimated]**. It says the win is
bounded by *compute*, not by the weight stream, so **`dflash_traffic.py`'s
traffic-only model overstates the headline**, and by much more than the 3.5%
section 5 allowed for. The 4.87x in section 5 should be read as an upper bound
until the verify pass is re-priced against `max(compute, memory)` rather than
memory alone.

It also moves where the effort belongs. Two things follow directly:

- **Batch 16 is probably not the right batch.** The crossover is 2.4 and the
  returns above it are sublinear in a way the traffic model hides. DFlash's
  `block_size` is a free parameter; the paper's own advice for quantized targets
  is `block_size <= 5`, which section 5 dismissed on the strength of the
  projections-only roofline. That dismissal no longer stands.
- **Attention is the thing to optimize, not the projections.** At 7 MAC/cycle it
  is 13x off the mmul the same chip runs elsewhere, and the batched case has an
  obvious lever the batch-1 case does not: 16 queries share one K block, so the
  `aie::transpose` per B tile and the K load can be hoisted across the batch by
  putting tokens in the mmul's R dimension. Unmeasured, but it is where the 71%
  is.

---

## 5f. So what block size? (measured inputs, modelled composition)

Section 5e says the batch-16 roofline was wrong. It does not by itself say what
the right batch is, because that is not a compute question alone — a bigger block
only pays if the drafter's extra tokens are *accepted*. `dflash_blocksize.py`
composes the two: each pass priced as `max(memory, compute)`, acceptance modelled
the standard speculative-decoding way, block size swept.

Acceptance is an **input**, not a prediction, exactly as `tau` is in
`dflash_traffic.py`. Modelled as a per-token probability `alpha` with
`E[accepted] = (1 - alpha^(b+1)) / (1 - alpha)`, and calibrated so that
`b = 16` reproduces the `tau = 6` the rest of this document assumes — so the
sweep is comparable to the earlier analysis rather than independently
pessimistic.

**First, the model is validated where a measurement exists.** Pointed at
llama-3.2-1b it derives **9440 projection blocks per core per token** — the same
number section 5 quotes from an independent count — and predicts a 16.8 ms floor
against a **measured** 19.6 ms wall time. 14% under, which is what an
issue-slot model with no DMA stalls should be **[measured]**.

### The kernel's cost is not linear in the batch

The first version of this sweep priced the projection with the fitted line
`1806 + 98*b`. That fit is wrong below batch 16, and not slightly — **the kernel
changes shape**. `q4k_mmul`'s 2x2 blocking needs `rowA = BATCH/8` to be even, so
it stops at 16; below that `q4k_mmul_small` runs `rowA = 1` with a 1x4 blocking,
and batch 4 drops to `aie::mmul<4,8,8>`, whose emulated path grows A to 64 lanes
and splits the accumulator in two. Measured, one build, per 32x256 block
**[measured]**:

| batch | mmul | +unpack | cycles/token | MAC/cycle (mmul only) |
|---|---|---|---|---|
| 4 | 1589 | 2998 | 749.5 | 20.6 |
| **8** | **918** | 2327 | 290.9 | **71.4** |
| 16 | 2354 | 3763 | 235.2 | 55.7 |
| 32 | 3533 | 4942 | 154.4 | 74.2 |

**Batch 8 does twice batch 4's work in 58% of the cycles.** `aie::mmul<4,8,8>`
is a bad shape and 1x4 at `rowA = 1` is a good one — 71.4 MAC/cycle is the best
figure this kernel has produced at any batch, against the 38.8 that batch 16 was
originally reported at.

One caveat on that table: batch 16 reads 2354 here against **1965** measured
before `q4k_mmul_any` existed. The dispatcher is `always_inline` and compiles
away, but its presence still moved clang's inlining decisions. Within a single
build the four rows are consistently inlined and comparable; across builds they
are not. This is the same outlining hazard the gotchas already list, and it is
why `bench_q4k_mm.py` now looks for the body in three places and says which one
it found.

### The sweep, from measurements rather than a fit

qwen3-4b target, qwen3-4b-draft drafter, 46 GB/s, 1.57 GHz, compute serial. Only
the batches that have been measured are swept, because interpolating across a
kernel-shape change is what produced the wrong answer the first time:

| block | verify ms | draft ms | iter ms | tau | ms/token | speedup | bound |
|---|---|---|---|---|---|---|---|
| 1 | 56.3 | 12.6 | 69.0 | 1.84 | 37.4 | 1.50x | compute |
| 4 | 108.7 | 20.5 | 129.2 | 3.65 | 35.4 | 1.59x | compute |
| **8** | 145.6 | 24.4 | 170.0 | 4.99 | 34.1 | **1.65x** | compute |
| 16 | 273.7 | 44.8 | 318.5 | 6.00 | 53.1 | **1.06x** | compute |
| 32 | 496.8 | 77.9 | 574.7 | 6.32 | 90.9 | 0.62x | compute |

**Block 8, at 1.65x.** Block 16's verify pass is 5.0x its memory floor — that is
the work the traffic-only model does not charge for, and it is how 4.87x and
1.06x come out of the same weights.

Under the overlap bound (attention hidden entirely behind the projections,
unattainable) the optimum moves to **4 at 2.77x**. So the honest range is
**block 4-8 and 1.6-2.8x**, and every variant tried lands inside it.

### It is an attention problem, not a projection problem

At the winning block size the verify pass splits **45.5 ms projection against
100.0 ms attention** **[measured]** — attention is **69%** of it. The block size
is being set almost entirely by a kernel running at ~7 MAC/cycle while the
matmul beside it runs at 71.

That reframes the remaining work. Making the projections faster cannot move this
much; making attention amortize over the batch could move some of it. ~~16 queries
share one K block, so the per-tile `aie::transpose` and the K load can be hoisted
by putting tokens in the mmul's R dimension.~~ **Now measured — section 5g. The
transpose is free; the tile loads are the hoistable part; the ceiling is 1.55x on
qwen3-4b, and it does not change the block size.**

### What is modelled and what is measured

Measured: the projection line, the GEMV, the attention cost, the bandwidth, all
the sizes off the builder, and the 9440/19.6 ms cross-check. Modelled: the
composition — that passes are `max(memory, compute)`, that compute is
`proj + attn` serial, and the acceptance curve. `alpha` is an input.

The one number that could still move this a lot is **acceptance on a quantized
target**, which remains unmeasured (section 7) — but it moves the *height* of
the curve, not where its maximum is, because acceptance is monotone in the block
size while the cost turns over.

---

## 5g. How much of attention can a batch actually amortize? (measured)

Section 5e ended by naming a lever: 16 queries share one K block, so put tokens
in the mmul's R dimension and the per-tile `aie::transpose` and the K load hoist
out of the per-token path. That was **reasoning, not measurement** — and it is
the kind of reasoning this document has already been wrong with twice. The
question it skips is *how much of attention is hoistable at all*, and that has a
measurable answer, so measure it before building anything.

**Method.** Split each call in two. Work that exists once per *(token, key
block)* — the online-softmax `update`, the y accumulator traffic, the MACs — a
batch can never remove. Work that depends only on the *key block* — the K and V
tile loads, the transposes — b tokens pay once. Then whatever the kernel looks
like, it is bounded by

```
per-token cost at batch b  =  PER_TOKEN + PER_BLOCK / b
ceiling                    =  (PER_TOKEN + PER_BLOCK) / PER_TOKEN
```

`bench_attn_batch.py` prices each piece by `#ifdef`-ing it out and taking the
bundle delta, at bench_attn.py's flags and **unrolled**. The knobs live in
`aie_kernel_utils.h`; `check_kernels_inert.py` proves they changed no shipping
code.

Per 16-key block, unrolled **[measured]**:

| piece | llama-3.2-1b | qwen3-4b | amortizes? |
|---|---|---|---|
| K tile loads (`attn_qk`) | 240 | 615 | over a batch |
| K transposes (`attn_qk`) | **−22** | 25 | over a batch |
| softmax `update` (`attn_qk`) | 478 | 528 | **no — per token** |
| V tile loads (`attn_kv`) | 395 | 863 | over a batch |
| y rescale pass (`attn_kv`) | 575 | 1160 | **no — per token** |
| `calculate_l` (`attn_kv`) | 21 | 37 | **no — per token** |
| **per KV block (hoistable)** | **635 (27%)** | **1503 (35%)** | |
| **per token (the floor)** | **1733 (73%)** | **2757 (65%)** | |

**The transpose is free.** Removing it made llama's kernel 22 bundles *slower* —
scheduling noise around zero. It hides in issue slots next to the MACs. So the
lever section 5e named was pointed at the one piece that costs nothing; the
hoistable work is the **tile loads**.

**The softmax is the floor and it is large.** `update` alone is 55% of
`attn_qk_blk` on llama. It runs once per *(token, head, key block)* — an
`exp`, a running max, a rescale — and no arrangement of the mmul shares it
across tokens, because each token has its own softmax state.

**The ceiling, and what it is worth:**

| | llama-3.2-1b | qwen3-4b |
|---|---|---|
| block 8 | 1.31x | **1.45x** |
| block 16 | 1.34x | 1.49x |
| b → ∞ | 1.37x | **1.55x** |

qwen3-4b does better because DH=128 doubles `colQ`, which doubles the tile loads
while leaving `update` where it is. **Every one of these is an upper bound**, and
loosely so: 64 vector loads cannot really cost 615 bundles, so what the knob
deletes is also the address arithmetic and the load-use chain pacing the
schedule. A real batched kernel lands below the table.

**End to end it is worth about a quarter, and it does not change the block
size** (`dflash_blocksize.py --attn-hoistable 1503`):

| | block 4 | block 8 | block 16 |
|---|---|---|---|
| as built | 1.59x | **1.65x** | 1.06x |
| with a perfect hoist | 1.80x | **2.08x** | 1.39x |

So: worth doing, bounded at +26% on the headline, and **not a prerequisite for
anything**. Block 8 is the optimum either way, which means step 2's builder
wiring is not blocked on it.

**One asymmetry the bundle counts do not show.** Batching attention means one
call spanning several tokens, and verify's tokens have *different* causal
lengths — `L_eff(t) = P+t+1` — where `L` is a single RTP per dispatch. So the
lever is clean on the **draft** pass, whose `L` is uniform, and on **verify** it
additionally requires a per-query mask inside the kernel. Verify is 36 layers
against draft's 5, so the half that is hard is the half that pays. Costed
separately, not as one 1.55x (section 6, step 2).

### The bigger attention lever is not batching

The two largest per-token pieces are the softmax `update` and the **y rescale
pass**, and the second one is not irreducible arithmetic — it is an L1 round
trip. `attn_fv` loads the whole y accumulator, multiplies by the flash
correction, stores it, and then the mac pass loads it *again* and stores it
*again*. On qwen3-4b that pass is 1160 bundles, **27% of all attention**, and it
is paid at batch 1 too — it would speed up the shipping decode, not just DFlash.

It exists because the 16-key block loop lives in the AIR herd, so y cannot stay
in registers across blocks. Two ways at it, neither tried: fold the correction
into the mac pass's existing `yprev` load (the two passes were split to dodge a
Peano spill defect — see the comment in `attn_fv`), or handle several key blocks
per call so the round trip is paid once per 4 blocks instead of once per block.

Not attempted here: it is a restructure of the two kernels carrying the most
Peano-workaround comments in the tree, and it is orthogonal to DFlash.

---

## 6. Build order

1. ~~**Benchmark the new matmul first.**~~ **DONE — gate passed.** Built as
   `kernels/q4k_mm.h` + `kernels/q4k_mm_bench.cc`, swept by `bench_q4k_mm.py`.
   Result in section 5: 10.6x fewer cycles per token at batch 16, crossover at
   batch 14.8, so batch 16 is compute bound by about 3.5% — enough to build on,
   with no margin left over. Reproduce with:

   ```
   python3 programming_examples/fused_decode/bench_q4k_mm.py           # mmul only
   python3 programming_examples/fused_decode/bench_q4k_mm.py --noperm  # + intercept
   ```

   The gate was worth having. This repo built a batched 4-bit matmul once
   before, for `llms/llama32_1b_int4`, and it came out **8x slower than bf16**
   (698 vs 84 ms/layer) **[measured]**, blocked by the memtile budget capping
   `herd_m=2` at K=8192 and a Peano 9-bit immediate forcing `tile_n=16`. Same
   hardware. Going through `aie::mmul` on an unpacked block avoids both.

   **The numeric gate is now built and passing** — `q4k_mm_gate.py`, section 5d.
   It found two real faults on the first run. Bundle counts are unchanged by the
   fixes **[measured]**, so every number above still stands.

   Still open from this step: 38.8 MAC/cycle is well under prefill's 98.

2. **Batch the rest of the superkernel** — attention query tiling, output
   packing, rms/rope/glu row loop, `rcache`. **The kernel half is built and
   gated; the builder wiring is not.**

   **Build for block 8, not 16** (section 5f). That is not a small edit to the
   plan — `q4k_mmul`'s 2x2 blocking `static_assert`s `BATCH % 16 == 0`, because
   `rowA = BATCH/8` has to be even, so **the batch the analysis recommends is one
   the kernel refused to compile**. `q4k_mmul_small` now covers it: `rowA` is 1,
   so the blocking moves entirely into the weight rows (1x4, four accumulators,
   same register pressure). Gated bit-exactly at batch 4 and 8 **[measured]** —
   the layout derivation generalised without change.

   It is also *faster*, which was not the reason for building it: 918 mmul
   bundles at batch 8 against 2354 at batch 16, i.e. **71.4 MAC/cycle, the best
   this kernel has managed at any batch** **[measured]**. Batch 4 is the bad
   shape, not batch 8 — `aie::mmul<4,8,8>`'s emulated path grows A to 64 lanes
   and splits the accumulator, and costs 1589 bundles for a quarter of the work.

   Everything sizing-related gets easier: at block 8 the activation tile is half
   what sections 5b and 5c priced, the X feed sits below the crossover at batch
   10, and the egress had headroom to batch 511 anyway.

   **`DECODE_BATCH` exists in the builder now**, and with it the sizing rules
   the rest of this step has to respect. It is a strict no-op at 1 — emitted IR
   byte-identical to `HEAD` on both `llama-3.2-1b` and `qwen3-4b` **[measured]**
   — and it carries three things worth having before any wiring lands:

   - The **L1 ceilings as hard failures**, not as guidance. `DECODE_BATCH=26`
     exits naming the buffer and the tool that priced it. Overshooting L1 does
     not produce a slow build, it produces an aiecc failure naming nothing.
   - `ROW_TILE`, **derived** rather than hardcoded, from the widest of `qkv_l1`
     / `ropeq_l1` / `rms_l1` against the tile budget. At block 8 on qwen3-4b it
     comes out **4** — which independently reproduces `batch_l1_budget.py`'s
     measured `qkv_l1` limit, from a different calculation in a different file.
   - `ATTN_QTILE`, capped at 8 because the attention CU measures 51.0 KB of a
     54 KB budget there. It fits with **nothing spare**.

   What the build still has to do:

   - ~~**Proj cores: one new buffer.**~~ **Kernel DONE and gated** —
     `proj_qmm_mm_zero` / `_acc` / `_flush_row` in `proj_qmm.cc`, behind
     `-DPROJ_MM_BATCH`, so a build that does not ask for it is unchanged.
     Same three-entry-point split as the GEMV, for the same alloc-sinking
     reason. Compared against the GEMV on device in section 5d at batch 8 and
     16. **Builder wiring DONE**: both proj cores — `_core_blk` (paired,
     `PAIR_ROWS==2`, llama) and `_core_blk_np` (non-paired, `PAIR_ROWS==1`,
     gemma and **qwen3-4b, the DFlash target**) — select the batched kernels at
     `DECODE_BATCH>1`. At batch 8 on qwen3-4b the emitted IR contains no
     batch-1 projection call at all **[measured]**, and at batch 1 it is
     byte-identical to `HEAD` on both models.

     Wiring only one of the two cores would have looked right and done
     nothing: the paired core is the one the code reads as "the" proj core, and
     it is the one the DFlash target does *not* use.

     Two things the analysis had not surfaced, both found while writing it:
     **the reduce cache goes away entirely** (`rc`, `fill`, `proj_qmm_rc_arm`
     are all machinery for the `+min` factorisation the batched path does not
     use), and **the flush has to de-tile** — `aie::mmul` leaves the
     accumulator in C tile order, so one token's 32 rows are four 8-float runs
     64 floats apart, where the GEMV's accumulator was already contiguous.
     Net L1: 608 bytes of `rc` traded for a 16 KB scratch. That scratch is
     independent of the batch — it holds one unpacked *weight* block — so it
     sets the ceiling at 25 regardless, and block 8 clears it with room —
     31.3 KB of the 54 KB budget **[measured]**.
   - **Egress: widen, do not repeat.** One packet per round, B times longer.
     `N_ROUNDS`, BD count and instruction stream all stay put. Legal on all
     eight models, headroom to batch 511.

     **The descriptors are derived and checked** — `egress_bd.py`. And there
     are **two** of them, not one: the `outA` gather into the *group* memtile
     *and* the `toMain` gather into the *main* memtile. Fixing only the first
     leaves each group's slab token-major internally while the groups stay laid
     end to end, so a token's `PAYLOAD` row is still in `N_GRP` pieces — and it
     would look right in any single-group test. On qwen3-4b at block 8:

     | level | offsets | sizes | strides |
     |---|---|---|---|
     | `outA` → group, emitter k | `[0, HDR+k*32]` | `[8, 32]` | `[128, 1]` |
     | `toMain` → main, group g | `[0, HDR+g*128]` | `[8, 128]` | `[512, 1]` |

     Checked three ways: exactly-once coverage of the payload, **token t's row
     byte-identical to the row batch 1 produces** (the invariant that lets
     everything downstream ignore the batch), and batch 1 collapsing to today's
     1-D descriptors — which it does, reproducing the builder's existing `66 /
     64 / 1` and `258 / 256 / 1` exactly. That collapse is the check that the
     derivation matches the code rather than only itself.
   - ~~**X feed: watch it.**~~ **Not a concern at block 8.** The activation
     chunk only overtakes the weight chunk at batch 10; at 8 it is **0.80** of
     the weight chunk and needs **1.77 of 4.0 B/cycle** per broadcast stream,
     where at 16 it was 1.60 **[measured]**. The tightest number in the step
     stopped being tight when the block size did. If it does bite, `MROWS=64`
     removes it outright — not `--kcol 512`, which does not touch this ratio.
   - **Attention: query tile of 8**, on qwen3-4b and every other head_dim>=128
     model. Costs a second read of the KV block from L2; no extra DDR traffic.
   - ~~**Attention: a new mask.**~~ **Already there** — see below. It is an RTP
     value, not a kernel change — *provided attention stays per-token*. See
     "What the shipping q4nx models already answer" below for the case where it
     does not.
   - **Attention: it is now the dominant cost.** Section 5e — 71% of batch-16
     compute, and the reason the block size had to be settled before building
     more. Batching it is worth at most 1.45x at block 8 and blocks nothing
     (section 5g), so it is not a dependency of this step.
   - **rms/rope/glu: row tiling at 4**, set by `qkv_l1` on qwen3-4b. This is a
     TILE, not the block size: `qkv_l1` wants 96 KB for 8 tokens against a
     54 KB budget, so block 8 runs these phases as two sub-tiles of 4.
     `ropeq_l1` (64 KB, fits 6) and `rms_l1` (40 KB, fits 10) also exceed a
     whole block of 8 and ride the same tiling **[measured,
     `batch_l1_budget.py --model qwen3-4b --batch 8 -v`]**.
   - **Attention query tile 8 is exactly at the ceiling** — 51.0 KB of 54 KB,
     max query tile 8 **[measured]**. It fits, with nothing spare, so any
     future per-CU L1 addition on the attention path has to be paid for.
   - **rope needs B cos/sin LUTs, not one.** See below.
   - **The KV append gains a dimension.** See below.

   ### What the shipping q4nx models already answer

   Every q4nx/q4 model in `llms/` drives this same superkernel, so its decode
   wiring is the reference for what batching has to change. Reading it moved
   three items and added two.

   **Cheaper than the list assumed: rms / rope / glu need no kernel change at
   all.** Every one of these leaves is strictly per-row — `rms_norm_aie(y, x, w)`
   normalises one `MODEL_DIM` row, `pseduo_glu<L>` takes gate at `x + L/2` of a
   single row, `rope_compute` ropes one token's qkv. So "row tiling at 4" is
   `B` calls at a row stride, not a restructure. The tile exists only because
   `qkv_l1` cannot hold 8 rows.

   **Missing: rope is per-POSITION, and a block spans B positions.** The driver
   patches a 64-word cos/sin LUT into the RMS BO every single token
   (`r_bo.write(lut, _rms_lut_off*2)` in `llama32_1b_q4nx_inference.py`,
   position `p`). A block of B tokens sits at positions `P..P+B-1`, so it needs
   **B LUT slabs**, with rope called at `rope_w + t*ROPE_W_LEN`. The L1 side is
   already priced — `batch_l1_budget.py` scales `ropelut_l1` with the batch —
   but the host upload and the builder's rope feed are not, and nothing else in
   this document mentions it.

   **Missing: the KV append gains a dimension.** Today it is one slot,
   `sizes=[NGRP, REGION_W] strides=[REGION_STRIDE, 1]` at `(L-1)*REGION_W`. B
   tokens append B consecutive slots, so it becomes 3-D. Now derived and
   checked like the other two — `kvappend_bd.py`; on qwen3-4b at block 8,
   `offsets=[p,0,0] sizes=[8,2,512] strides=[512,1048576,1]`.

   **It carries a hazard the other two do not**, and it is the reason this one
   needed writing rather than reasoning. `p*REGION_W` is a valid slot only
   while `p < ATTN_MAXL`, and position `ATTN_MAXL` of region g **is** position 0
   of region g+1. At batch 1 the cache can overrun by one and the driver's
   window bookkeeping prevents it; at batch B it can overrun by B, and the
   overrun does not fault — it writes into the next group's live KV, which a
   real attention CU is reading. Wrong logits, no error. Measured on
   llama-3.2-1b at block 8 with 3 slots left: **2560 of 4096 elements land in
   the next group's region** **[measured]**. `check_bounds()` refuses it, and
   the builder needs the same guard.

   Writing the checker also settled a convention the builder relies on twice
   and documents nowhere: **AIR left-pads a short `offsets` list with zeros**
   (`air::canonicalizeWrapAndStrideList`, `mlir/lib/Util/Util.cpp`), so a
   rank-deficient list is **right**-aligned and a single offset lands on the
   stride-1 dimension as a flat element offset. That is how the existing
   one-offset-against-two-sizes KV append is correct. Read it as left-aligned
   and a flat offset silently picks up the outermost stride — here, a factor of
   `REGION_STRIDE`.

   **Incomplete: "the mask is just an RTP scalar" holds only while attention is
   per-token.** `L_c` is ONE value for the whole dispatch — a DYNSEQ RTP patched
   into the instruction stream. Verify needs `L_eff(t) = P+t+1`, which is B
   different values inside one dispatch. Two cases, and they differ a lot:

   | | how L varies | cost |
   |---|---|---|
   | attention called per token | `L_c + t`, an `arith.addi` on the loop IV | ~free, as claimed |
   | attention batched into the mmul's R dimension (section 5g) | one call spans B different L | **per-query mask, in the kernel** |

   So section 5g's lever is clean for **draft** (`L_eff = P+16` for every token,
   uniform) and not for **verify** — and verify is the expensive pass, 36 layers
   against 5. The pass where batching attention is hardest is the pass where it
   would pay most. That does not change 5g's conclusion (bounded at 1.45x,
   blocks nothing) but it does mean the two halves of the lever should be costed
   separately.

   One piece of luck inside that: `L` currently serves two purposes — the
   core-side loop bound via `_core_rounds(Lh)` and the mask inside
   `attn_qk_blk` — and at batch B they must diverge, uniform `ceil((P+B)/16)`
   for the loop against per-token `L` for the mask. **The kernel already
   tolerates exactly that**: `attn_qk_blk` returns early on `rem <= 0`, so
   handing it a loop bound larger than its own `L` is the built-in behaviour,
   not a change.

   **Reusable rather than rebuilt.** `DecodeInstsGen` specialises one xclbin to
   any L by patching only the L-dependent words, locating them by diffing two
   same-`ATTN_MAXL` builds and interpolating a slope — verified byte-exact
   against native per-L builds. The batched build's KV-append offset and RTP-L
   are the same kind of word and get the same treatment. `decode_geometry.py`
   reads BO sizes off the builder, so it will track the batched X and Y sizes
   for free. `respace_kv` is precedent for host-side KV relayout, which is what
   DFlash's rollback on rejected tokens is.

   **No prior art for multiple tokens per dispatch.** The `gemms` / `gemv` split
   in `llms/shared/builders/` is prefill against decode in the *multi-launch*
   engine, not this one; nothing has ever pushed more than one token through the
   superkernel. Which is also why no equivalence harness exists to borrow — the
   batch-equivalence gate has to be built.

   ### The mask turned out to already exist

   **This was the item flagged as most likely to be missed, and it needs no
   kernel change at all.** `attn_qk_blk` carries exactly the right mask, for an
   unrelated reason:

   ```cpp
   int rem = L - blk * 16;
   if (rem <= 0) return;
   rem = (rem < 16) ? rem : 16;
   aie::mask<16> mask = aie::le(idx, rem);     // idx = 1..16
   ```

   That is a **tail** mask — it trims the ragged last KV block when the context
   is not a multiple of 16 — and with `idx` starting at 1 it keeps global keys
   `0..L-1`. `L` is already "number of cached positions this token attends to"
   (the builder's own wording; `DECODE_GOLDEN_L=1` is position 0 attending to
   itself). So a per-query triangular mask is a per-query **value of L**:

   | pass | `L_eff(t)` | effect |
   |---|---|---|
   | verify (target) | `P + t + 1` | token t sees `0..P+t` — causal |
   | draft (DFlash) | `P + 16` | every token sees the block — bidirectional |

   One engine, two passes, an RTP scalar apart. `_core_rounds(Lh)` already
   derives the block count from the same scalar, so the loop bound follows for
   free, and DFlash needs `DECODE_DYNSEQ` for KV rollback anyway.

   `batch_attn_mask.py --check` models the kernel's arithmetic and asserts the
   key set is exactly causal, across prefixes on both sides of a 16-boundary and
   every t **[measured]**. It is a guard on an index convention rather than a
   device run — the claim is read off three lines — but it is what fails if
   someone makes `idx` 0-based or redefines what `L` counts.

   What the mask does **not** make free is the attention itself: 16 queries is
   16x the calls, and section 5e is what that costs. Push a uniform
   `ceil((P+16)/16)` blocks for all 16 tokens rather than a per-token count —
   the same shim BD for every token, at most one wasted block each, and at
   P=2048 exactly zero **[measured]**.

3. ~~**Hidden state taps** at layers 1, 9, 17, 25, 33.~~ **DONE — built, behind
   `DECODE_HIDDEN_TAPS=1`.** It turned out to be an offset change, not new
   machinery: the taps already crossed the shim on every layer, they were just
   overwritten in place.

   Every layer ends with a `layerOut` drain of the rms residual2 (`h + down`),
   which is the layer's hidden state, written back into the `X` BO so the next
   layer reads it. The drain is exactly `K` elements on all eight models
   **[measured]** — `LAYER_RNDS * PAYLOAD == K`, e.g. qwen3-4b 5 x 512 = 2560:

   ```python
   ChannelPut("rmsX", X, offsets=[0], sizes=[K], strides=[1])           # read
   ChannelGet("layerOut", X, offsets=[0], sizes=[LAYER_RNDS*PAYLOAD])   # write
   ```

   Both offsets were the literal `0` — the chaining ABI is in-place, so layer
   L+1 overwrote layer L's hidden state before anyone could read it. The change
   gives layer `iv` the read slot `iv` and the write slot `iv+1`, so the chain is
   unbroken and the history survives. Emitted IR at `DECODE_HIDDEN_TAPS=1`
   **[measured]**:

   ```mlir
   %7  = arith.muli %arg15, %c2048              // read  slot iv
   air.channel.put @rmsX[] (%arg10[%7]    [2048] [1]) : memref<34816xbf16>
   %81 = arith.muli %arg15, %c2048
   %82 = arith.addi %81,    %c2048              // write slot iv+1
   air.channel.get @layerOut[%c0] (%arg10[%82] [2048] [1]) : memref<34816xbf16>
   ```

   The LM head reads the last slot, a compile-time `UNI_DEC*K` = 32768, since its
   own wave index is past `UNI_DEC`. At the last layer, `iv=15`, the write lands
   at 32768+2048 = 34816 — exactly the buffer end, in bounds with nothing spare.

   The cost is DDR footprint and nothing else: the `X` BO goes from `K` to
   `(UNI_DEC+1)*K`, which is **185 KB** for qwen3-4b **[measured]** (68-296 KB
   across the eight models). No extra shim traffic, no extra BDs, same drain
   count — the bytes were already being written, just to the same address every
   time.

   Verified **[measured]**:

   - **Byte-identical IR at `DECODE_HIDDEN_TAPS=0`** against `git HEAD`, on
     llama-3.2-1b, gemma3-4b and qwen2.5-7b, via the `FUSED_DECODE_EMIT_ONLY`
     hook the file provides for exactly this. The default path is untouched.
   - Builds with taps on for llama-3.2-1b, qwen3-4b, gemma3-4b and qwen2.5-7b,
     and in the `LM_HEAD=1` and `DECODE_DYNSEQ=1` configurations — the latter
     mattering because DFlash needs dynseq anyway for KV rollback (step 6).
   - `decode_geometry.py --check` still passes.

   Not verified: this is an IR-level check, not a run. The write/read pair relies
   on `air.preserve_shim_dma_order` for chaining, and the dependency is still
   real at distinct offsets in the same BO, so the ordering should still carry —
   but that wants a device run to confirm, and the host side has to allocate the
   larger `X` BO and read the tap slots back.

   As a side effect this also gives the llms per-layer cosine lens the data it
   currently has no source for on fused decodes.

4. **Qwen3-4B q4nx.** `_MODELS` entry **DONE** — added to `fused_decode.py` and
   it passes every builder assert. Derivation, all forced rather than chosen:

   - `PAIR_ROWS=1`, because the paired egress needs each phase output divisible
     by `ROW_BLOCK*NCX*NCY*PAIR_ROWS` = 1024 and the o/down phases emit K=2560,
     giving 2.5. Non-paired (512) is exact. Same reason qwen2.5-7b is non-paired.
   - `I2P = [6144, 2560, 19456, 2560]/512 = [12, 5, 38, 5]`
   - `J2P = [2560, 4096, 2560, 9728]/512 = [5, 8, 5, 19]` — note J2P[1] is
     **DQ**/512, not K/512, because the decoupled q dim makes the o-proj contract
     4096 -> 2560.
   - `VOCAB_CHUNK_I2=30`, `UNI_LM=10`: padded vocab 153600 -> 4800 rowblocks, so
     `UNI_LM*VOCAB_I2 = 300`; K/PAYLOAD = 5 must divide `VOCAB_RNDS`, so VOCAB_I2
     is a multiple of 5; 30 is the largest under the tested envelope.

   Cross-check: the builder reports `W_LAYER = 31,539,200`, and an independent
   parameter count of one Qwen3-4B layer (100.93 M params x 2560/8192) gives the
   same number. That is the check that the phase geometry is right rather than
   merely self-consistent; `GLU_OUT` also comes out at 9728, the model's true
   intermediate size.

   **The q4nx weight loader is now written too**, as
   `llms/qwen3_4b_q4nx/{qwen3_4b_q4nx_weights,qwen3_4b_q4nx_requant}.py`.

   It was a smaller job than it looked. `qwen3_8b_q4nx_requant.py` already does
   Qwen3 and is written against the builder rather than the model — it takes
   `fd` as an argument and reads `GROUP`, `NCX`, `NCY`, `NPH`, the phase
   indices, `GLU_CHUNK`, `W_LAYER`, `UNI_LM` and `pack_q4k_cascade` out of it,
   and keys the dual-channel layout off `fd.W_DUAL_CHAN` so the pack cannot
   disagree with the xclbin. Pointing it at the `qwen3-4b` entry supplies all of
   that. The weights module likewise **re-parameterizes** the 8B one rather than
   copying 421 lines: the codec, header parsing, accessors and reference forward
   are all written against `_PROJ`, so a subclass overriding `_PROJ` re-targets
   them.

   **One structural delta, and it is the opposite of 8B: Qwen3-4B ties its
   embeddings** where Qwen3-8B does not — `llms/qwen3_4b/qwen3_4b_weights.py`'s
   `LlamaConfig` is the repo's own Qwen3-4B and every dimension here came from
   it. So the head is the bf16 embedding matrix, the llama path, not the
   `lm_head.weight` branch 8B takes — that tensor is not in the bundle at all,
   so inheriting 8B's accessor would raise on a missing key. Everything else
   (two norms per layer, per-head qk-norm riding in the rope slab, no qkv bias)
   is the same. Only `D` (2560 vs 4096) and `INTER` (9728 vs 12288) change.

   **Gated without the model bundle** — which matters, because the bundle is a
   gated download and a packer that disagrees with the xclbin about phase row
   counts produces a cache that loads fine and decodes garbage. Every projection
   shape is checked against the builder's own phase geometry **[measured]**:

   ```
   $ python3 qwen3_4b_q4nx_requant.py --check
     OK  hidden K                  2560    2560      OK  phase 0 rows   6144  6144
     OK  vocab                   151936  151936      OK  phase 1 rows   2560  2560
     OK  layers UNI_DEC              36      36      OK  phase 2 rows  19456 19456
     OK  qkv out (phase 0 M)       6144    6144      OK  phase 3 rows   2560  2560
     OK  o-proj contract DQ        4096    4096
     OK  GLU_OUT vs INTER          9728    9728
     OK  head_dim DH                128     128
   SELF-CHECK PASS
   ```

   That the phase rows agree is an independent confirmation of the `_MODELS`
   entry as well: `I2P` was derived from the model dims, and this derives the
   model dims from a different source and gets the same answer.

   Still to do: quantizing the drafter, and running the requant itself — it
   needs the `model.q4nx` bundle, which is not available on this box
   (huggingface.co fails certificate verification here).

   A consequence worth noting for DFlash specifically: with a tied target head
   and a drafter that also ties to it, there is exactly **one** head matrix in
   the whole system, read once per draft pass and once per verify pass.

5. **Drafter instance.** `UNI_DEC=5` on the same engine — the `_MODELS` entry is
   **DONE** — plus mask-token embedding, the `fc` linear, `hidden_norm`, and a
   5-layer KV cache that rolls back on rejection alongside the target's.

   Three of those four are cheaper than they look:

   - **`fc`**: expressible as an extra proj phase at `I2=5, J2=5`, five
     accumulating passes. Both forms legal; decompose for the L1 reason in
     section 5. The accumulate-across-passes mechanism already exists — it is
     what the down phase's refeed does.
   - **`hidden_norm`**: one more RMSNorm on a tile that already runs two (four
     on Gemma), so the rms core has the shape for it.
   - **mask-token embedding**: a host-side gather. The engine takes `X` as an
     input already; filling a block's 16 slots with the mask embedding costs
     nothing on device.

   The KV cache is the one with real cost: 5 layers x `ATTN_MAXL*KVSZ_TOK` =
   **40 MB** at context 2048 **[measured]**, a full-depth cache over the accepted
   prefix. It rolls back the same way the target's does — see step 6.

6. **Host loop and gate.** Draft/verify/accept/bonus. DFlash is lossless, so the
   gate is stricter than the usual top-5 check: the speculative run must
   reproduce the target's greedy token stream **exactly**. Any difference is a
   bug.

   **KV rollback is free, on one condition:** build with `DECODE_DYNSEQ=1`. That
   variant already exists and already ships — five models have a
   `compile-decode-dynseq` make target for it. It takes the context length as a
   dispatch-time scalar, so the KV append lands at a runtime slot
   `(L-1)*REGION_W` rather than a baked address. Rejecting `16-tau` tokens is
   then just passing a smaller `L` on the next call: the stale entries past `L`
   are never read, and the existing tail mask already handles the ragged final
   block. No cache surgery, no compaction pass. Without dynseq the slot is a
   compile-time constant and none of this works, so it is a hard prerequisite
   rather than a preference.

   The batched append itself is one extra dimension and lands exactly on the
   limit **[measured]**: today it is a 2D shim BD, `sizes=[NGRP, REGION_W] =
   [2, 512]`; at batch 16 it becomes `[2, 16, 512]`, which is **3 of the 3
   dimensions a shim BD has** on AIE2p. It fits, with nothing left over — worth
   knowing before anything else tries to claim a dimension on that transfer.

---

## 7. Open questions

- ~~**Is the batched matmul numerically correct?**~~ **Answered: yes, bit-exactly
  on device** (section 5d). It was not, when the question was written — the
  operands were in the wrong roles.
- ~~**Does it agree with the GEMV it replaces?**~~ **Answered: to 1.7% rms, at
  1.4x the GEMV's own error** (section 5d), measured with both kernels in one
  launch off the same weights. Not a rounding difference; a real change to the
  projection output.
- **Does 1.3% rms per projection matter end to end?** Unmeasured. It is a
  per-matmul figure on one 32x256 block pair, not a per-token one, and the
  thing that decides it is the `llms/verify/` top-k gate on a real model — which
  needs step 2 built first. Worth knowing that the error is unbiased and flat in
  K, so it should not compound across the 36 layers the way a biased one would.
- **What block size is actually optimal?** Now the first question, not a
  settled one. Section 5e puts the memory/compute crossover at 2.4 and section 5
  put it at 14.8; the truth for a DFlash *iteration* needs
  `dflash_traffic.py` re-priced against `max(compute, memory)` per pass instead
  of memory alone, which is not built. Until then read 4.87x as an upper bound.
- ~~**Can batched attention be made to amortize?**~~ **Answered: partly, and by
  less than the 71% suggests.** 35% of attention is per-KV-block and hoistable
  on qwen3-4b; the rest is softmax and y accumulator traffic that is per token
  by construction. Ceiling 1.55x on the term, 2.08x end to end, block size
  unchanged (section 5g). The `aie::transpose` named here previously is free.
- **Can the y rescale round trip be removed?** Untried, and it is now the
  largest single piece of attention on qwen3-4b — 1160 bundles, 27% of the
  whole, paid at batch 1 as well as batched, so it would speed up the shipping
  decode too. Fold it into the mac pass's existing `yprev` load, or handle
  several key blocks per call. Both run into the Peano spill defect the two
  passes were split to dodge (section 5g).
- **Acceptance rate on a quantized target.** Still unmeasured, and still the
  input every headline number depends on. But it is now **bounded**: break-even
  is at tau = 1.23, and half the paper's rate still gives 2.44x (section 5).
  It scales the win; it does not decide feasibility.
- **Does a Qwen3-8B DFlash drafter exist?** No longer blocking — `qwen3-4b` and
  `qwen3-4b-draft` builder entries exist and validate, so the 4B path is open.
  Not found in z-lab's public listing, which shows Qwen3.5 and Qwen3.8 variants.
- **Is quantizing the drafter worth it?** Yes on both axes, against an unknown
  acceptance cost: 4.87x against 3.94x (~24% throughput) **and** 645 MB less
  resident DDR, 3048 against 3693 MB (section 5). It is required anyway to run
  the drafter on the superkernel at all, since that engine only takes q4nx.
- ~~**Is the LM head worth trimming for the draft pass?**~~ **Answered: no.**
  It is 42% of the q4nx draft pass, but the draft pass is only 19% of the
  iteration, so the whole head is 8% of the total and trimming it cannot buy
  more than that. Measured with `dflash_traffic.py --draft-head-frac`
  **[measured]**: cutting the draft head to a quarter of the vocabulary moves
  4.87x to 5.18x, and deleting nine tenths of it gets 5.25x.

  | draft head | speedup at tau=6 | break-even tau |
  |---|---|---|
  | full | 4.87x | 1.23 |
  | 50% | 5.07x | 1.18 |
  | 25% | 5.18x | 1.16 |
  | 10% | 5.25x | 1.14 |

  A 6% gain is not worth restricting what the drafter can propose, since that
  costs acceptance rate directly and tau is the term with real leverage. Build
  the full tied head.
- ~~**Can `aie::mmul` hold 98 MAC/cycle/core on freshly unpacked weights?**~~
  **Answered: no — 38.8 at batch 16** **[measured]**. (The BFP16 numbers just
  below were taken before the layout fix and are quoted against the pre-fix
  1787-cycle multiply; the conclusion is a 3.9x ratio, which the fix does not
  touch.)

  I guessed the gap was the BFP16 emulation converting both operands per `mac`
  with only colB=2 reuse. **That guess was wrong.** Compiling the same kernel
  without `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` **[measured]**:

  | `q4k_mmul<32,256,16>` | bundles | MAC/cycle |
  |---|---|---|
  | with BFP16 emulation | 1787 | 41.2 |
  | native bf16 mmul | **11088** | **10.5** |

  BFP16 emulation is not the cost, it is a **3.9x speedup** on the multiply, and
  removing it drops the crossover from 16.7 to **2.7** — far below block 16,
  which would sink the whole approach. It is already on by default in
  `PEANO_KBASE`; the finding is that it must stay on. (Same arithmetic as
  section 5, using the one batch-16 point and a proportional slope; that method
  reproduces 16.7 exactly for the BFP16-on case, which is what licenses it here.)

  Prefill's 98 MAC/cycle is measured with the same flag, so the remaining gap is
  shape, not codec: K=256 per call is short and colB=2 gives the 2x2 blocking
  little to reuse. Both levers tested so far are reuse levers and both moved it
  the right way (folding +8%, batch 16 -> 32 +29%), which is consistent.
- **Does the static bundle count survive real DMA pressure?** The sweep counts
  issue slots with no stalls, and after the layout fix batch 16 is already
  **over** the memory floor (20.3 vs 19.6 ms), with no slack at all to absorb
  stalls. This was the question with a 0.5 ms cushion; now it has none. Needs an
  on-device timing run, not a static count.
- **Numeric validation of the batched kernel.** The layout is now *derived*
  rather than arbitrary — see "the correct layout is not free" in section 5 —
  and fixing it caught a real out-of-bounds scale index on the way. What is
  still missing is a gate: nothing has been compared against `_qmm_q4k_bf16` on
  the same weights. The kernels use AIE intrinsics and will not run on the host,
  so this needs a device run.
- **Where the fold gain comes from — accumulator traffic or scheduling window.**
  It decides whether either fold is usable inside the L1 budget, and the
  discriminating experiment (`--chunks 2`) does not compile: two inlined
  unrolled bodies overflow the AIE2 frame-offset immediate, and the rolled
  alternative makes both variants emit identical code. **The static-bundle
  method cannot answer this**; it needs on-device timing, which is the same
  thing the DMA-pressure question needs. Until then both folds are parked and
  the plain 32x256 kernel is what step 2 should be built on — it is the one that
  fits L1 with 12 KB spare and is already measured.

---

## References

- [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) · [z-lab/dflash](https://github.com/z-lab/dflash) · [Qwen3-4B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16)

In tree: [`fused_decode/`](../programming_examples/fused_decode/) (superkernel,
`kernels/q4_k.h`, `_MODELS`),
[`matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc`](../programming_examples/matrix_multiplication/bf16_in_fp32_out/mm_aie2p.cc)
(the matmul to copy),
[`llms/qwen3_4b/ARCHITECTURE.md`](../programming_examples/llms/qwen3_4b/ARCHITECTURE.md)
(target geometry, 7 ELFs/layer),
[`llms/qwen3_8b_q4nx/README.md`](../programming_examples/llms/qwen3_8b_q4nx/README.md)
(measured prefill and decode),
[`llms/llama32_1b_int4/README.md`](../programming_examples/llms/llama32_1b_int4/README.md)
(the batched 4-bit precedent).
