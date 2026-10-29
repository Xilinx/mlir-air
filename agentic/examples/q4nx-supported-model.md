# Run the public workflow on Phi-2

This example exercises the maintained affine-Q4 implementation at both native configurations. It is a reproduction/optimization task on an existing model; a new HF model requires the agent to implement its model-specific adapter first.

From the AIR root, source the toolchain environment documented by your checkout, and prepare the task:

```bash
python agentic/tools/task.py microsoft/phi-2 --out /tmp/phi2-task/task.json
python agentic/tools/air_source_profile.py
```

Install the model's runtime dependencies and create the pinned reference environments
once (or reuse compatible existing interpreters):

```bash
python -m pip install -r programming_examples/llms/phi2_2_7b_q4nx/requirements.txt
python programming_examples/llms/shared/setup_reference_envs.py /tmp/phi2-task/reference-envs
export AIR_AFFINE_Q4_REFERENCE_PYTHON=/tmp/phi2-task/reference-envs/current/bin/python
```

This device example follows AIR's Linux runtime flow. The harness host checks and
client registrations also run on Windows.

Choose a fresh writable build directory. The following command runs the existing CI integration for P512: pinned weights and reference preparation, dependency-aware compilation, the full model gate, reset checks and the prescribed profiling batch. It acquires the device lock itself. A gate failure stops profiling.

```bash
cd programming_examples/llms
python -m shared.deployment_ci profile --model phi2_2_7b_q4nx --prefill 512 --work-dir /tmp/phi2-task/p512
python -m shared.deployment_ci profile --model phi2_2_7b_q4nx --prefill 1024 --work-dir /tmp/phi2-task/p1024
```

If HF access is unavailable, provide `--checkpoint /path/to/pinned/HF/snapshot`; optionally provide `--q4-dir /path/to/canonical-q4`. The shared preparation code checks the pinned identities. Reference environments follow [shared deployment documentation](../../programming_examples/llms/shared/AFFINE_Q4_DEPLOYMENTS.md). P2048/C4096 and P4096/C8192 are excluded by Phi-2's native context.

Inspect the selected work directory's `status.json`, logs, verification captures and profile records. A successful task contains the two model gates and valid timing samples; it does not require a new optimization or a faster result to count as a completed reproduction. TTFT includes embedding through first-logit checks/greedy selection. Physical KV handoff follows the first token and is counted inside decode/request time.

For a prefill-only follow-up, reuse the accepted decode/head, fixed reference and unchanged dependencies. Rebuild affected GEMMs with BFP16 and run the task's affected gate and timing batch. If the user chooses first-token-only timing, label that protocol separately from complete-request profiling.
