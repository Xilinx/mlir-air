# Qwen3-8B B16 decode megakernel

`BLOCK_SIZE=16` processes an anchor and fifteen proposals, matching the official
drafter's training block size. B8 remains the default. Both configurations share
checkpoints, packed weights, NPU prefill, context preparation and host runtime.
The eight-row context-preparation kernel handles a sixteen-row update in two
dispatches.

The 4096-wide target has 36 layers; the drafter has five. Both use the
[shared four-owner B16 mapping](../../common/docs/b16-ddr-replay.md), with
128-KiB normalized scratch, a 2×2 projection schedule and two attention waves.
That document describes the AIR source and compiler fixes required for direct
generation, including their trigger conditions and regression tests.

```sh
make compile BLOCK_SIZE=16
make run BLOCK_SIZE=16
make chat BLOCK_SIZE=16
make verify BLOCK_SIZE=16
make profile BLOCK_SIZE=16
make compile-baseline BLOCK_SIZE=16
make full20 BLOCK_SIZE=16
make full80 BLOCK_SIZE=16  # optional
```

The default B16 artifact directory is `build_b16/`; B8 uses `build/`. Both roles
must use the same block size. Artifacts include the block-specific ABI and
compiler identity; use a fresh directory after either changes.

The `run_npu2_*_b16.lit` entries follow the existing compile, verify, profile and
Full20 nightly filters. Profile logs/JSON and verification identities are
separate from B8. Full20 uses four frozen cases per dataset; Full80 remains
opt-in. Decode timing includes initial and incremental NPU context preparation,
drafting, verification and host acceptance. Loading, prefill, reset, warmup and
the first prefill token are excluded. The native B1 comparator is the existing
fixed-C4096 xclbin/template runner. See [validation](validation.md) for the
shared checks and [compatibility](compatibility.md) for build requirements.
