# Vera — Round 2 Review

## Triage Validation

I concur with the triage. My Critical finding (C1 — zero test coverage) was upheld, as were the other reviewers' findings.

### On the test strategy going forward

When the implementation is fixed, the test suite should cover:

1. **Unit test in LIT** (`opt_shim_dma_bds.mlir` or a new file):
   - Input: `scf.for` loop containing `air.segment { air.herd { channel ops + compute } }`
   - Trip count >= 2
   - CHECK lines verify: correct number of unrolled segments, channel ops present in herd bodies, compute ops absent
   - CHECK lines verify: IV substitution in channel offsets/strides per iteration

2. **Regression guard**: If the clone-then-strip approach is adopted, add a `// CHECK-NOT` for compute ops inside unrolled herd bodies to ensure the stripping actually occurred.

3. **Round-trip test**: Run the full pass pipeline (`air-opt-shim-dma-bds` + downstream passes) on the test input to verify BD folding still works correctly.

4. **Warning tests**: For `scf.parallel`/`scf.if` inside herds, add `// expected-warning` tests.

## Verdict

**BLOCK MERGE** — concur with triage. The implementation is untested and likely unbuildable.
