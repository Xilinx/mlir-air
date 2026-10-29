# Reusable evidence, lessons and package maintenance

## Local kernel records

The legacy registry lookup remains a measured-configuration seed API. For verified selection use `programming_examples/kernel_registry/evidence.py`; it is opt-in and accepts local JSON records. No legacy NUMFAIL or timing-only row is converted into PASS.

A record has `schema_version: 1`, `id`, `kind` (primitive or megakernel), `key`, `abi`, `output_shape`, `validation`, optional `timing`, and `files`. Megakernels also include a `graph` describing fused operations and their connections. The exact key contains operation, shape, input_dtype, output_dtype, compute, precision, layout, target, toolchain, timing_boundary, rtol and atol. Use precise toolchain/target identities and workload dimensions; incompatible contracts do not match.

`files` binds source, artifact, reference, validation report and optional timing report to relative paths and SHA256 hashes beneath an explicit evidence root. Source can be an archive of the dependency closure. The saved validation report must equal `validation` and bind source/artifact/reference hashes in its `bindings` object. PASS requires finite full-output comparison, zero mismatches, compared_elements equal to the output shape size, and the exact key's tolerances. Timing binds the same identities and contains positive finite `samples_seconds` and the exact `boundary`.

Validate or query local records:

```bash
python programming_examples/kernel_registry/evidence.py /path/proposal/records.json
python programming_examples/kernel_registry/evidence.py /path/proposal/records.json --query /path/proposal/key.json
```

`RECORDS_CONSISTENT` checks metadata/file consistency, including retained failures. `VERIFIED_MATCH` selects the lowest mean latency among exact evidenced PASS records; otherwise the query returns `NO_VERIFIED_MATCH`. UNKNOWN and missing timing do not enter this ranking. Hash checks bind the supplied evidence but do not authenticate the reported experiment: admission still requires normal code/evidence review. The executable synthetic fixtures in `agentic/tests/test_evidence.py` illustrate the schema without claiming a device measurement.

Propose records in a task-local directory, then submit reviewed source/configuration/evidence changes through the repository's normal contribution process. Preserve failed attempts and previous measurements. Keep shared assets frozen during controlled comparisons.

## Lessons

Use one short local proposal per reusable lesson: trigger; observed evidence and source version; interpretation; coupled remedy; verification scope; counterexample or remaining unknown. Link raw logs and the accepted candidate. Review the lesson before adding it to the relevant skill/reference. Avoid turning a single model's workaround into a hardware rule.

## Package updates

Edit canonical files under agentic. Run `tools/sync_package.py` to regenerate client registrations and MANIFEST.json. Run the host tests and check generated-file consistency. CI protects links, metadata, relocation, task/native-context behavior and evidence handling; model CI protects device correctness and profiling. A guidance-only change does not require replaying all model configurations.

Old phase-0 through phase-7 skills map to build-cpu-reference, verify-kernel, optional integration diagnosis, verify-model, npu-optimization, finalize-deployment and evaluate-deployment. Debug/optimization names are retained. The four megakernel skills add change, resource, correctness and performance diagnosis. Independent evaluation is optional and explicitly labeled.
