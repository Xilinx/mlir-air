# qwen3-4b q4nx — decode, and the DFlash speculative loop

Fused decode for Qwen3-4B on NPU2, plus the DFlash draft/verify loop built on
top of it. `docs/DFlashFeasibility.md` is the running record of what was tried
and measured; **this file is the operating manual** — what to build, what to
run, and which results you are allowed to believe.

---

## 1. The one thing to internalise

**Nothing here is a pass/fail button.** The gates differ enormously in how much
they prove and in how reproducible they are, and the two properties are
inversely related:

| instrument | what it proves | reproducible? |
|---|---|---|
| `fused_decode/dump_layer_output.py` | two builds produce the **same layer output, bit for bit**, on the same seed | **yes** — two dispatches, no prefill, ~30 s. Start here |
| `fused_decode/dflash_build_diff.py` | two builds are **bit-identical** on synthetic weights — **decode layers and KV only** | **yes**, at L161, and only because it now retries until two dispatches agree |
| `fused_decode/batch_equiv.py` | the batched dataflow matches batch 1, token 0 | yes, but token 0 only |
| `fused_decode/dflash_template_time.py`, `dispatch_time.py` | **how fast** a template dispatches (median of N, synthetic weights) | **within a run, yes; across runs, ±1.8 ms** — see the warning below |
| `dflash_verify_gate.py` | real model, real weights, **logits** | yes, but only because it retries until two dispatches agree — see §4 |
| `dflash_loop.py`, `dflash_acceptance_device.py` | end-to-end speculative decode | inherits §4, and **does not retry** — one dispatch per verify block. The **speedup** reproduces; the **mean accepted** is a noisy sample, ±0.22 |

**DUMP AND DIFF BEFORE YOU RUN ACCEPTANCE.** For a change that is supposed to
be bit-exact, "do these two builds agree" is both a stronger question than "is
this one right" and vastly cheaper to ask. It cost a near-miss to learn: the
weight-ring change was written up as *"numerically wrong, item 2 closes at
zero"* on the strength of one acceptance run reading 2.938 against a 3.792
control — three standard errors, and junk. That run's pre-pass row read 455 ms
against 45.70, a thrashing host, and the whole thing was noise.
`dump_layer_output.py` settled it in two dispatches: **0 of 757,760 elements
differed.** A 12-minute end-to-end gate should never be the first thing you
reach for, and its anomalies mean *distrust the run*, not *ignore that column*.

**AND `dispatch_time.py`'S OWN BANDS UNDERSTATE ITS NOISE.** A single run
reports p10-p90 within a few tenths of a millisecond, which reads as high
precision and is not: that band is the spread *within* one run, and it does not
contain the spread *across* runs. Three paired replicates of one unchanged
build at L161 read **85.923, 84.514, 86.303** — a 1.8 ms spread, with every
individual band far tighter than that. `Q4K_MM_UNROLL=2` was nearly written up
as a 1.4 ms win on the strength of one such run before the replicates killed
it. **Below about 2 ms, run it three times or do not claim it.** Above that the
method is fine — the 22.9 ms the unpack FMA is worth has bands 18 ms apart.

If you change the compiler or the builder, start with `dflash_build_diff.py`.
It is the only tool here that gives the same answer twice — but it earns that
the same way the verify gate does. A single dispatch is not trustworthy even
here: three consecutive runs on a **bit-identical** pair gave IDENTICAL,
IDENTICAL, then 972/2560 at rms rel 5.9e-03, an order of magnitude larger than
the effects worth hunting and in the direction that reads as "your change broke
it". Each half now dispatches until two in a row agree (`--tries`, default 6).
The filter does not hide real differences: a genuinely wrong build reproduces
its wrong answer exactly (`RF_CYCLE_S=25`, same 12967568/12976128 every run).

**It cannot see the LM head.** Even with the vocab waves in the frozen stream
(`DECODE_NO_LM_WAVES=0`), the logits region of the Y BO comes back **written and
entirely zero** — measured by prefilling Y with a sentinel: all 1228800 elements
touched, none nonzero, while X and the KV cache in the same dispatch are normal.
Two zero regions compare bit-identical, so the tool used to print ten reassuring
"wave w: 122880/122880 bit-identical" lines that meant nothing. It now says so
instead. For anything about the vocab arm, use `dflash_verify_gate.py`.

---

## 2. Build

Everything is built by `programming_examples/fused_decode/build_template.sh`,
which **always writes `decode_b<B>_L<N>.{xclbin,insts.bin}`** into
`fused_decode/`. Every other artifact family is that file renamed, or the same
file under a `DECODE_PROBE` / `DECODE_HIDDEN_TAPS` / `DECODE_DYNSEQ` prefix.
Copy or rename it yourself; nothing does it for you.

```bash
source ~/air_env.sh                       # every shell
cd programming_examples/fused_decode

# batch-8 target template with the LM head (what the loop and the gate bind)
env -u DECODE_EXTRA_WAVES DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=30 \
    W_DUAL_CHAN=1 DECODE_NO_LM_WAVES=0 DECODE_STACK=6080 \
    ./build_template.sh 8 16
cp decode_b8_L16.* ../llms/qwen3_4b_q4nx/

# batch-1 pair, ALWAYS after any batched build (see the trap below)
./build_template.sh 1 16
```

Rules that are not optional:

- **`DECODE_STACK=6080` at batch 8.** The default leaves the rms core 55280 B of
  L1 against the 59424 B a batch-8 residual + staging + norm weights need. The
  builder refuses to import rather than build something that fits by truncation,
  and the driver must be told the same number (`--stack`).
- **`VOCAB_CHUNK_I2=30` for this model.** It has to match `UNI_LM` in
  `fused_decode.py`'s model table; the divisibility constraints are this model's
  own geometry, not a number that transfers.
- **Rebuild the batch-1 template after every batched build.** `build_template.sh`
  rebuilds `proj_qmm.o` with `-DPROJ_MM_BATCH=<batch>`; a batch-1 design linked
  against batch-8 kernels is not a build error, it is a wrong answer.
- **Build the xclbin at the FULL wave table.** `UNI_WAVE_LO`/`UNI_WAVE_HI` narrow
  the *instruction stream* against a fixed xclbin. They must never be allowed to
  change the *design* — a knob that did (dropping the LM head arm from a
  derivation when `DECODE_NO_LM_WAVES=1`) hung the decode-only build on device.
- **Which L?** `attn_maxl_of(L, B) = 16*ceil((L+B-1)/16)`, so at batch 8, L15 and
  L16 both live in the ATTN_MAXL=32 window and the driver needs *both* to
  calibrate its L-slope. `rm` the artifact before rebuilding: a failed build
  leaves the old one and you will measure it.
- **`VOCAB_CHUNK_I2=50` (with `UNI_LM=6`) is what lets mode 3 carry the LM
  head.** The vocab chunking sets `VOCAB_RNDS`, which has to share a re-feed
  cycle with the decode arm's `XN_REFEED + REFEED[GATEUP_PHASE]` = 50. At 30 the
  two share only a cycle of 10, which forces the decode arm into three and seven
  fills per phase and is wrong; at 50 both walk `[12, 38]` and the decode arm
  keeps ONE fill per phase, which is bit-exact. Three things must agree or the
  result is wrong with no error: the templates, the requant cache
  (`_ensure_requant_cache` keys on the chunking; build the v50 one with
  `dflash_vocab_repack.py --to 50 --to-uni-lm 6`, a pure re-ordering, no
  requantization), and the driver env. **The batch-1 path does not build at 50**
  -- it times out -- so batch 1 stays at 30; the two caches hold byte-identical
  decode-layer weights, so a gate run can legitimately give each half its own
  matched pair (`--b1-env` / `--bB-env`).
- **`RMS_MEMTILE_REFEED=3` carries the LM head at `VOCAB_CHUNK_I2=50`.** (It was
  decode-only at 30, and the builder still enforces that: `VOCAB_RNDS=30`
  cannot walk the decode arm's cycle. At 50 `VOCAB_RNDS` IS 50, the arm walks
  `[12, 38]` exactly once, and `DECODE_NO_LM_WAVES=1` is not needed.) It is
  bit-identical to shipping over 36 layers and 26.8 ms faster, but only at ONE
  SWEEP PER PHASE. A smaller re-feed cycle is numerically wrong. Narrowed to
  **ph2 with more than one fill**
  (`RF_CYCLE_S=25` puts ph0 at one fill and ph2 at three, and is already the
  full error). **Root-caused and FIXED in the compiler**: a fill boundary stalled
  the rms core mid-row and the core overwrote the row still in flight, because
  a core-side put acquired its buffer AFTER writing it — so every fill after the
  first in a phase lost its row 0, which is why the signal was always token 0.
  With that fixed and `VOCAB_CHUNK_I2=50`, mode 3 is bit-identical to shipping
  on the decode layers, the KV and the logits, and 31.9 ms faster.
  `RF_CYCLE_S=` / `RF_CYCLE=` force other cycles if you are working on it. See
  `docs/BZeroPlan.md` item 1 and `docs/DFlashFeasibility.md` §3.20.

---

## 3. Test, in order

```bash
cd programming_examples/fused_decode

# 1. static, seconds, no device -- run after ANY fused_decode.py or kernels/ edit
python check_kernels_inert.py          # shipping kernels emit identical code
python check_channel_balance.py        # producers and consumers scale together
python check_batch1_noop.py            # batch 1 still inert, vs HEAD, BOTH models
python check_memtile_chans.py air_project/npu.air.mlir   # DMA chains + re-feeds
#    check_batch1_noop.py builds from `git show HEAD:fused_decode.py` and from
#    the worktree and diffs the emitted MLIR, so unlike step 2 it needs no
#    locally-captured baseline and works in a fresh clone.

# 2. is my change inert on the shipping path? (emitted MLIR, byte-compare)
#    _base_b1.mlir and _base_b8.mlir were both captured at DECODE_GOLDEN_L=128.
#    At any other L the KV-region sizes and offsets differ and the compare fails
#    for a reason that has nothing to do with your change.
#    THEY ARE UNTRACKED, so a fresh clone does not have them: capture them once
#    from a known-good tree with the same command below, or use step 1's
#    check_batch1_noop.py, which needs no baseline.
#    UNSET EVERY OPT-IN KNOB. Each one that exists must appear here or the
#    compare silently tests whatever your shell happens to be carrying.
for B in 1 8; do
  env -u DECODE_EXTRA_WAVES -u RMS_MEMTILE_REFEED -u RF_CYCLE_S -u RF_CYCLE \
      -u PROJ_WS_NO_SINK -u PROJ_PP_ONLY -u PROJ_RING_DEPTH -u Q4K_UNPACK_UNROLL \
      -u UNI_LM -u UNI_WAVE_LO -u UNI_WAVE_HI \
      FUSED_DECODE_EMIT_ONLY=1 VOCAB_CHUNK_I2=30 LM_HEAD=0 NLAYERS=1 \
      DECODE_GOLDEN=1 UNIFIED=1 DECODE_NO_LM_WAVES=1 DECODE_MODEL=qwen3-4b \
      DECODE_BATCH=$B DECODE_GOLDEN_L=128 W_DUAL_CHAN=1 DECODE_STACK=6080 \
      python fused_decode.py > _chk_b$B.mlir 2>/dev/null
  cmp -s _chk_b$B.mlir _base_b$B.mlir && echo "B=$B ok" || echo "B=$B DIFFERS"
done

# 2b. DOES THE FOLDED (DFlash) PATH STILL BUILD? ~30 s, no device.
#     Step 2 above unsets DECODE_EXTRA_WAVES, so it proves NOTHING about the
#     path the DFlash target templates take. That hole is not hypothetical: an
#     unconditional _RF_PH0_EX cycle walk once broke every folded template at
#     every mode and survived a day in the tree with step 2 green, because no
#     gate anywhere in this repo had ever built a wave table. Run this after any
#     fused_decode.py edit, and compare against HEAD's builder rather than a
#     checked-in baseline -- the folded emit has no _base_*.mlir.
SPECS=$(python -c "
import sys,json; sys.path.insert(0,'../llms/qwen3_4b_q4nx')
import dflash_prepass_waves as P
w,_=P.wave_specs(P._load_draft_fd()); print(json.dumps([s.as_config() for s in w]))" \
  2>/dev/null | tail -1)
git show HEAD:programming_examples/fused_decode/fused_decode.py > _head_fd.py
for f in _head_fd.py fused_decode.py; do
  env -u RMS_MEMTILE_REFEED DECODE_MODEL=qwen3-4b DECODE_BATCH=8 VOCAB_CHUNK_I2=30 \
      W_DUAL_CHAN=1 DECODE_STACK=6080 DECODE_NO_LM_WAVES=0 DECODE_HIDDEN_TAPS=1 \
      DECODE_MASK_BIDIR=0 UNI_WAVE_LO=0 UNI_WAVE_HI=71 DECODE_EXTRA_WAVES="$SPECS" \
      FUSED_DECODE_EMIT_ONLY=1 python $f > _fold_$f.mlir || echo "$f FAILED TO EMIT"
done
cmp -s _fold__head_fd.py.mlir _fold_fused_decode.py.mlir \
  && echo "folded ok" || echo "folded DIFFERS from HEAD"

# 3. deterministic device check -- two builds, bit-exact
DECODE_STACK=6080 W_DUAL_CHAN=1 DECODE_NO_LM_WAVES=1 \
  python dflash_build_diff.py --a <ref-prefix> --b <new-prefix> --L 161

# 4. dataflow gate (synthetic weights -- NOT correctness, its header says so)
DECODE_STACK=6080 W_DUAL_CHAN=1 DECODE_NO_LM_WAVES=1 \
  python batch_equiv.py --model qwen3-4b --vocab-chunk-i2 30 --batch 8 --L 161 --tokens 0

# 5. real model, real weights (retries internally until reproducible, §4)
cd ../llms/qwen3_4b_q4nx && python dflash_verify_gate.py
```

### The VOCAB_CHUNK_I2=50 configuration, end to end

The one that lets `RMS_MEMTILE_REFEED=3` carry the LM head. Batch 8 goes to 50;
batch 1 stays at 30 because it does not build at 50.

```bash
cd programming_examples/fused_decode
V="DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=50 UNI_LM=6 W_DUAL_CHAN=1    DECODE_STACK=6080 DECODE_NO_LM_WAVES=0"
for L in 161 162; do
  env -u DECODE_EXTRA_WAVES $V RMS_MEMTILE_REFEED=3 ./build_template.sh 8 $L
  mv decode_b8_L$L.xclbin   ../llms/qwen3_4b_q4nx/_v50/decode_b8_L$L.xclbin
  mv decode_b8_L$L.insts.bin ../llms/qwen3_4b_q4nx/_v50/decode_b8_L$L.insts.bin
done
./build_template.sh 1 16                       # the usual batch-1 rebuild

cd ../llms/qwen3_4b_q4nx
python dflash_vocab_repack.py --to 50 --to-uni-lm 6      # ~1 min, no requant

# decode arm bit-exact, LM head in the stream (needs a v50 mode-0 control)
cd ../../fused_decode
DECODE_STACK=6080 W_DUAL_CHAN=1 DECODE_NO_LM_WAVES=0 UNI_LM=6   python dflash_build_diff.py --a _v50m0 --b _v50m3 --L 161 --vocab-chunk-i2 50

# the gate, each half on its own matched (template, cache) pair
cd ../llms/qwen3_4b_q4nx
env -u UNI_LM -u VOCAB_CHUNK_I2 python dflash_verify_gate.py   --prompt-len 150 --tag v50m3   --bB-env "Q4NX_QWEN3_4B_DECODE_DIR=$PWD/_v50,VOCAB_CHUNK_I2=50,UNI_LM=6"
```

Expected: the diff is IDENTICAL and the gate is **PASS 8/8**, with the logits
bit-identical to the shipping build (1215488/1215488). Before the compiler's
core-side lock ordering was fixed this was FAIL(1) with only token 0 wrong; if
you see that again, you are on an `air-opt` that predates
`core_put_acquire_before_write.mlir`. And v30 mode 0 against v50 mode 0 gives
1215488/1215488 bit-identical logits, which is how you know the repack
re-ordered and did not corrupt.
And v30 mode 0 against v50 mode 0 gives **1215488/1215488 bit-identical logits**,
which is how you know the repack re-ordered and did not corrupt.

`check_memtile_chans.py` also verifies that every alloc-carried
`air.refeed_count` in `placed.air.mlir` still appears on a buffer in
`npu.air.mlir`. A re-feed that is silently dropped in lowering leaves an arm with
no replay and hangs on device with nothing wrong in the IR; this is the only
thing that catches it.

---

## 3b. DFlash end to end: run it, then verify it

This is the whole recipe for the **optimized block 8** — the configuration
`dflash-feasibility` delivers. A batch-8 verify dispatch at L161 runs **85.9 ms
against shipping's 147.7**, i.e. `b` is 8.83 ms/token lower, and every claim
below is gated.

### The environment variables that define it

```bash
RMS_MEMTILE_REFEED=3      # the memtile replays X instead of the core regenerating it
PROJ_WS_NO_SINK=1         # keep the 16 KB unpack scratch out of the j loop ...
PROJ_PP_ONLY=w            # ... so the WEIGHT input gets a 2-deep ring
Q4K_UNPACK_FMA=1          # w = q*scale+min in ONE accumulator pass ...
Q4K_UNPACK_UNROLL=8       # ... which frees the registers that make unrolling pay
VOCAB_CHUNK_I2=50 UNI_LM=6   # what lets mode 3 carry the LM head
```

What each is worth at L161, `dispatch_time.py` median of 25, each against the
row above it:

| | B8 ms | delta | output |
|---|---|---|---|
| shipping | 147.742 | — | — |
| `RMS_MEMTILE_REFEED=3` | 116.394 | −31.3 | bit-identical |
| `+ PROJ_WS_NO_SINK PROJ_PP_ONLY=w` | 108.840 | −7.6 | bit-identical |
| `+ Q4K_UNPACK_UNROLL=8` | 105.543 | −3.3 | bit-identical |
| `+ Q4K_UNPACK_FMA=1` | **85.923** | −19.6 | **1 ulp of bf16** |

Every optimization is **opt-in and default off**. Everything measured is
qwen3-4b batch 8; llama and gemma share the `_mm` path and have not been gated.
`PROJ_PP_ONLY=x` also works and is bit-exact but loses by 1.3 ms; ringing BOTH
inputs deadlocks (see `docs/BZeroPlan.md`).

**Only the last row changes the answer**, and it changes it toward the exact
result: `Q4K_UNPACK_FMA` removes one bf16 rounding (`aie::mul` returns an
accumulator that the result type rounds down and `aie::add` lifts straight back
up). So the verify recipe below has one step that must NOT read IDENTICAL —
see level 1.

### Build

```bash
source ~/air_env.sh
cd programming_examples/fused_decode
V="DECODE_MODEL=qwen3-4b VOCAB_CHUNK_I2=50 UNI_LM=6 W_DUAL_CHAN=1 \
   DECODE_STACK=6080 DECODE_NO_LM_WAVES=0 PYTHON=python"
OPT="RMS_MEMTILE_REFEED=3 PROJ_WS_NO_SINK=1 PROJ_PP_ONLY=w      Q4K_UNPACK_FMA=1 Q4K_UNPACK_UNROLL=8"

# (a) the LOOP's target pair -- taps ABI, two adjacent L for the driver's slope
for L in 512 511; do
  env -u DECODE_EXTRA_WAVES $V $OPT DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0 \
      ./build_template.sh 8 $L
  mv -f taps_b8_L$L.{xclbin,insts.bin} ../llms/qwen3_4b_q4nx/_v50m3wf/
done

# (b) the GATE's pair -- `decode` ABI, L161/L162
for L in 161 162; do
  env -u DECODE_EXTRA_WAVES $V $OPT ./build_template.sh 8 $L
  mv -f decode_b8_L$L.{xclbin,insts.bin} ../llms/qwen3_4b_q4nx/_v50wf/
done

./build_template.sh 1 16      # ALWAYS: batch-1 kernels, after any batched build
```

The two families are **not** interchangeable: `DECODE_HIDDEN_TAPS=1` widens the
BO ABI, so the gate cannot bind a taps template and the loop cannot bind a
`decode` one. `_build_fma.sh` in `fused_decode/` is both loops, already
written; `_build_w.sh` and `_build_v50w.sh` build the pre-FMA control pair.

### Run

```bash
cd ../llms/qwen3_4b_q4nx
env -u Q4NX_QWEN3_4B_DECODE_NPZ Q4NX_QWEN3_4B_DECODE_DIR=$PWD/_v50m3wf \
    VOCAB_CHUNK_I2=50 UNI_LM=6 python dflash_acceptance_device.py \
    --prompts prompts_gsm8k.json --n 6 --n-tokens 32 --prepass cpu
```

`--prepass cpu` runs the pre-pass in numpy off the same q4k bytes. It is the
CONTROL, not a shipping mode: the folded (`waves`) pre-pass does not compose
with mode 3 yet. **Read the `target` row of the step breakdown, not the
speedup** — the host pre-pass drifts by 10x with machine load while the target
moves <1%.

### Verify — four levels, cheapest first

```bash
cd ../../fused_decode

# 1. BYTE LEVEL, ~30 s per build. Same seed => same input; only the xclbin
#    differs. Run each in its OWN process (one device context per process).
env $V $OPT DECODE_HIDDEN_TAPS=1 DECODE_MASK_BIDIR=0 \
    python dump_layer_output.py --prefix taps --batch 8 --L 511 --out opt.npy
#    ... swap in the control template, then:
python dump_layer_output.py --diff ctrl.npy opt.npy
#
#    EXPECT IDENTICAL for every flag EXCEPT Q4K_UNPACK_FMA. With the FMA in,
#    expect ~20% of elements differing by 1 in the bf16 bit pattern and NOTHING
#    ELSE -- no index shift, no zeroed tail, the same zero count on both sides.
#    To keep an EXACT gate, build the candidate with Q4K_UNPACK_UNROLL=8 but
#    WITHOUT the FMA: the unroll is semantics-preserving, so that pair must be
#    bit-identical, and it is the control proving the whole difference belongs
#    to the FMA's one removed rounding.

# 2. DEVICE BUILD DIFF, decode layers + KV, at L161 where builds are bit-exact.
#    ONLY MEANINGFUL FOR BIT-EXACT CHANGES, so this is the gate for mode 3 and
#    the ring, NOT for the FMA -- it reports element counts, and the FMA moves
#    ~20% of them by 1 on purpose. Use level 1 and level 3 for that.
for T in 161 162; do
  cp -f ../llms/qwen3_4b_q4nx/_v50/decode_b8_L$T.{xclbin,insts.bin}  . # control
  cp -f ../llms/qwen3_4b_q4nx/_v50w/decode_b8_L$T.{xclbin,insts.bin} . # under test
done   # rename to _v50_b8_L<T>.* and _v50w_b8_L<T>.* -- --a/--b are PREFIXES
env DECODE_STACK=6080 W_DUAL_CHAN=1 DECODE_NO_LM_WAVES=0 UNI_LM=6 \
  python dflash_build_diff.py --a _v50 --b _v50w --L 161 --vocab-chunk-i2 50

# 3. THE REAL GATE: real model, real weights, LOGITS. This is the one, and it
#    is the gate the FMA has to pass. Run it against the CONTROL too -- "corr
#    went up" means nothing without the number it went up from.
cd ../llms/qwen3_4b_q4nx
env -u UNI_LM -u VOCAB_CHUNK_I2 python dflash_verify_gate.py \
  --prompt-len 150 --tag v50wf \
  --bB-env "Q4NX_QWEN3_4B_DECODE_DIR=$PWD/_v50wf,VOCAB_CHUNK_I2=50,UNI_LM=6"

# 4. the loop's own correctness control: the `accepted` row
```

Expected, and what these actually returned:

| | expected | measured |
|---|---|---|
| 1. `dump_layer_output.py --diff`, unroll-only | IDENTICAL | **0 of 757,760 differ** |
| 1. `dump_layer_output.py --diff`, with the FMA | 1 ulp, nothing else | **79.7% equal, 11.8% differ by exactly 1** |
| 2. `dflash_build_diff.py` (mode 3 + ring) | IDENTICAL | **8/8 tokens + 12,976,128/12,976,128 KV** |
| 3. `dflash_verify_gate.py` | PASS | **PASS, argmax = at all 8 positions** |
| 4. `accepted` | within 0.57 of 3.792 | **3.667 ± 0.322** |

Level 3 was also run against the pre-FMA control for a like-for-like read, and
the FMA build comes out **slightly better**: mean `corr` 0.9742 vs 0.9720, mean
`rel` 0.189 vs 0.206. Token 7's low `corr` of ~0.86 appears in BOTH and is
pre-existing. Removing a rounding should improve things, and it does — but the
point of running the control is that "it improved" only means something beside
the number it improved on.

Two things about level 3 that look like failures and are not. Both halves exit
`3221225477` **after writing their result** — the known teardown fault, judge on
the file. And the per-token `rel` runs 1e-1 to 3e-1 with `corr` ~0.99: that is
the KERNEL, not the batching — batch-1 runs the v1 GEMV and batch-8 the q4k
mmul, and the gate's own header prices that gap with the argmax unchanged.

### What is NOT available here, so do not reach for it

`batch_equiv.py --tokens 0` cannot gate this configuration: the batch-1
reference at `DECODE_HIDDEN_TAPS=1` with `VOCAB_CHUNK_I2=50` times out on its
own, with **and** without mode 3. That is the reference side being broken, not
your change. `dump_layer_output.py` exists because of it.

**The batch-1 timeout is broader than that, and it costs a number.** A plain
`decode`-prefix batch-1 template at **L161** also times out — at
`VOCAB_CHUNK_I2=30`, with `DECODE_NO_LM_WAVES=0` and with `=1`, and at
`--wait 600000`. So the batch-1 anchor `B1` in `docs/BZeroPlan.md` cannot be
re-measured at the length everything else is measured at, and every absolute
`b = (B8 - B1)/7` there carries an unverified constant. Differences between
builds do not. Do not spend dispatches re-deriving this; it needs the batch-1
timeout fixed first, which is its own problem.

---

## 4. The verify gate: a single dispatch is not trustworthy

`dflash_verify_gate.py` runs the shipping driver on the real model and compares
logits. It is the right gate. A **single batch-8 dispatch on this machine is
not reproducible**, so the gate no longer uses one: it dispatches until two in a
row come back bit-identical (`--tries`, default 6), and writes no result at all
if that never happens.

Measured inside one process and one decoder, re-seeding the KV cache before each
dispatch so the input state is identical:

```
dispatch 0   differs from 1
dispatch 1   BIT-IDENTICAL to 2
dispatch 3   differs from 1 and 2 by rms rel 2.4e-01
```

A second run of the same probe (`_drv_determinism.py`) had all five agree — so
it is intermittent, and the first dispatch after construction is the usual
offender. Before the filter, three runs of the gate on a mode-0 pair that is
bit-identical to the shipping build gave argmax **5/8, 8/8, 8/8**; a correct
build was mistaken for a regression and it cost a session. With the filter,
repeat runs agree to the last decimal and report PASS.

Not root-caused. The shape fits WDDM paging the 2.1 GiB host-only weight BO —
the same pager behind the `0xc01e0200` that kills `make verify` — where a page
that is not resident when the engine reads it corrupts a region silently. If you
see `never reproduced in N dispatches`, that is the device, not your build; try
a reboot.

**Use `--prompt-len 150`.** `PARIS_PROMPT` is 5 tokens, so `max_L = P+B+1 = 14`
picks the ATTN_MAXL=32 window (L15/L16) — and at L15 two builds of the SAME
design differ by rms rel 2.5e-03 to 9.6e-03, fourteen times what the gate is
trying to detect. Every verdict there is build noise. `--prompt-len 150` puts
`max_L` at 159, binds the L161/L162 pair where builds are bit-reproducible, and
makes the gate measure the build. The token content does not matter — this
checks that one batch-8 dispatch equals eight batch-1 steps of the same tokens.

`corr` and `rel` are printed, not enforced (`--strict` if you want them). They
are a property of the projection kernel, not a target: a known-good pair sits at
corr 0.936–0.989 and rel 0.10–0.37, and the old hard thresholds reported FAIL(6)
on it at argmax 8/8.

---

## 5. Traps, each one paid for

- **One device context per process.** Two xclbins registered back to back, or a
  batch-1 and a batch-8 `FusedDecoder` alive at once, and the second dispatch
  returns `ERT_CMD_STATE_TIMEOUT` with a partly-written X — indistinguishable
  from a hang in the build under test, and not one. `dflash_verify_gate.py` and
  `dflash_build_diff.py` both fork a subprocess per half because of this.
- **A half that writes its result and then segfaults has SUCCEEDED.** Dropping a
  `FusedDecoder` takes its BOs and the XRT device down in whatever order the
  collector picks. Judge on the output file, not the exit code.
- **Build-to-build variation at small L.** Two builds of the *same* mode-0
  design at L15 differ by rms rel 2.5e-03 to 9.6e-03 at the layer output. At
  L161 they are bit-identical. Never compare two builds at small L.
- **`python3` is not on PATH** (Windows app-execution alias). Use `python`. The
  `llms` Makefiles hardcode `python3`, so invoke their runners directly.
- **`make verify` currently dies** with a WDDM paging error
  (`RuntimeError 0xc01e0200`) in `decode_staircase.make_insts_states`. Use
  `dflash_verify_gate.py`.
- **`batch_equiv.py` is not correctness.** Synthetic weights, and `--tokens 0`
  unless you build a batch-1 template per position. Its own header says so.
- **A stale template of the wrong model dispatches and returns plausible,
  meaningless numbers.** `build_template.sh` substitutes the model's own
  `MODEL_TYPE` for exactly this reason; do not bypass it.
- **`build_template.sh` skips the Peano pin preflight.** Its templates are good
  for dataflow work. For numerics claims, note that the batch-1 design it builds
  was checked bit-identical to the Makefile-built pair — but that check is worth
  repeating if the sandbox's Peano moves.

---

## 6. Where things live

- `qwen3_4b_q4nx_inference.py` — the shipping driver (`FusedDecoder`). Do not
  modify it to run an experiment; subclass it, as `hidden_taps_device.py` does.
- `dflash_loop.py` — the speculative loop. Needs the target and draft template
  families at two adjacent L.
- `dflash_acceptance_device.py` — the priced sweep.
- `hidden_taps_verify.py` — per-layer hidden states vs an HF bf16 reference.
  The only correctness instrument here that needs no LM head, and it runs its
  two halves as subprocesses (torch and XRT in one process segfaults).
- `qwen3_4b_q4nx_requant.py`, `qwen3_4b_draft_requant.py` — weight packing.
- `fused_decode/dump_layer_output.py` — dump one batch-8 dispatch's layer
  output and diff two of them. The cheapest correctness instrument here and the
  one to reach for first; see §1.
- Everything else `dflash_*` is bring-up history kept because
  `docs/DFlashFeasibility.md` cites it. `dflash_fc_wave_gate.py`,
  `dflash_phase2_bf16_reference.py` and `dflash_phase2_sweep.py` are referenced
  by nothing at all, including the doc.
- `_`-prefixed files are scratch, and **untracked by convention, not by
  `.gitignore`** — nothing ignores them, so `git add -A` would sweep hundreds of
  them in. Add named files only.
