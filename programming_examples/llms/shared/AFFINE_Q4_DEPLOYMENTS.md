# Affine-Q4 deployment commands

These commands apply to Granite-3.3-2B, SmolLM3-3B, OLMo2-1B, OpenELM-1.1B, Phi-2 and Mistral-7B-Instruct-v0.3. Their model directories specify native P/C configurations and architecture details.

## Build and run

Set up the current [AIR Linux toolchain](../../../docs/buildingRyzenLin.md), including XRT for execution. Use the Peano version pinned by this checkout. The inline decode-attention build also needs an `llvm-link` older than LLVM 23, as checked by the shared decode Makefile.

From any of the six model directories:

```bash
python3 -m pip install -r requirements.txt
make weights CHECKPOINT=/path/to/pinned/hf
make compile P=512
make run P=512 N_TOKENS=32
```

`make compile` builds the shared fused prefill once (into `BUILD_DIR/fused_prefill`, reused while its sources and toolchain are unchanged), the P/C configuration's decode and, for OpenELM and Phi-2, the decode's LM head. That prefill takes any input up to the model's longest native input (4096 tokens, 2048 for OLMo2, OpenELM and Phi-2); `INPUT_IDS` may hold fewer than P tokens, as long as the decode capacity C leaves room for the outputs.

Without `CHECKPOINT`, weight preparation downloads the pinned HF revision. `Q4_DIR=/path/to/canonical-q4` reuses already-prepared logical Q4 tensors. Packing preserves their codes, scale and minimum values. Builds and packed weights go into `BUILD_DIR` (default `build/`); a separate directory may be selected. `make compile-kernels` compiles only the C++/LLVM kernels. `compile-decode`, `compile-prefill` (the fused prefill) and `compile-head` (the fused prefill and, for OpenELM and Phi-2, the decode's LM head) build individual components.

`make compile-all` builds the decode of every native P/C configuration; the fused prefill and the LM head are shared.

## Correctness and timing

Generate an original-checkpoint CPU BF16 reference in an environment with `requirements-reference.txt` installed:

```bash
make reference P=512 CHECKPOINT=/path/to/pinned/hf REFERENCE_DIR=/path/to/reference
make verify P=512 REFERENCE_DIR=/path/to/reference
make profile P=512
```

For OpenELM, use a separate reference environment with Transformers 4.39.3 for its original HF Python implementation. A local OpenELM checkpoint must include the pinned repository's `configuration_openelm.py` and `modeling_openelm.py`; the default HF download includes them. Other models use Transformers 5.3.0. Reference generation loads original checkpoint weights and generates 32 greedy outputs, ignoring EOS. Verification requires complete finite full-vocabulary logits for every output and applies the shared mutual-top-5 check at the first divergent choice. It is a fixed-input model gate.

For the models other than OpenELM this fixed-input gate is `make verify-native`. Their `make verify`, `make accuracy` and `make tasks` run AIR's shared [verify subsystem](../verify/README.md) through `verify_adapter.py` (`affine_verify.py`): the top-5 gate on its prompts, teacher-forced KL divergence on wikitext-2 against the BF16 checkpoint or the dequantized affine-Q4 weights (`ACC_REF=bf16|dequant`, `ACC_PATH=decode|prefill`), and GSM8K / MMLU scores. They run the session of the configuration whose KV capacity covers 2048 tokens, so build it first:

```bash
make weights
make compile P=1024
make verify
make accuracy ACC_REF=dequant ACC_PATH=prefill
make tasks TASKS=gsm8k TASKS_LIMIT=100
```

OpenELM's checkpoint ships neither a tokenizer nor an HF implementation the
verify/ tools can load, so its `make verify` remains the fixed-input gate.

`make compile-fused-prefill` and `make profile-fused-prefill FUSED_LENGTHS=512,1024,2048` build and time the fused prefill alone on synthetic inputs (`bench/sweep_prefill.py --fused`).

For the optional A/B/A request-isolation check, run the common CLI from `programming_examples/llms/` (substitute the model and build paths):

```bash
python3 -m phi2_2_7b_q4nx verify --prefill 512 \
  --build-dir /path/to/build --q4-dir /path/to/canonical-q4 \
  --reference-dir /path/to/reference --check-reset
```

This requires bitwise-identical logits for two A requests separated by a changed-input B request. It adds a state check to the same numerical gate.

Profiling warms up with complete fresh 128-output requests for at least five seconds, then measures at least three requests totaling at least ten seconds. It reports median TTFT, `P / median TTFT`, pooled decode throughput `(127 × request_count) / summed_decode_seconds`, and mean complete-request time. TTFT ends after first-logit readback, finite checks and greedy selection. Physical KV overwrite/handoff follows that boundary and is included in decode and complete-request time. Weights and compiled artifacts remain loaded between fresh requests.

`make verify-all` and `make profile-all` visit the native configurations sequentially (`verify-all` runs the fixed-input gate). NPU commands acquire `/tmp/mlir-air-npu.lock`. `INPUT_IDS=/path/to/ids.json` selects a JSON list containing exactly P valid token IDs; reference and NPU execution must use the same list. No chat template, BOS insertion or early EOS changes the fixed workload.

Raw captures, request samples, selected artifacts and execution placement are written under `BUILD_DIR/runs/`. Host work includes embedding lookup, table preparation, layout/byte transfers, dispatch and sampling; the implementations retaining final single-row prefill normalization on the host report it in the session metadata. Build/setup time is outside request timing.


## Continuous integration

The six examples follow the Llama and fused-decode lit/Makefile conventions.
Per-PR CI runs `shared/run_host_checks.lit` (codec, model-gate and CI failure-path
checks). The existing LLM compile suite builds P512 without weights. The nightly
verify and profile suites discover **19 separate native-configuration tests per
suite** and run them serially. Unsupported native contexts have no test entry.
Each test has the nightly suite's 1,800-second timeout. Every model adds
`run_npu2_prefill_sweep.lit`, and all but OpenELM add `run_npu2_verify.lit`,
`run_npu2_accuracy.lit` and `run_npu2_tasks.lit`; they share one `affine-build`
directory per model.

Seed the pinned revisions in `llms/hf_models.txt` with the existing Download LLM
Weights workflow. Tests require the corresponding `hfweights_*` cache feature;
the driver resolves the exact revision and checks checkpoint hashes. These six
checkpoints are public and do not require an HF token. Nightly uses offline mode.

Set up the two reference interpreters after the general runtime dependencies:

```bash
python3 ../shared/setup_reference_envs.py /path/to/reference-envs
export AIR_AFFINE_Q4_REFERENCE_PYTHON=/path/to/reference-envs/current/bin/python
export AIR_OPENELM_REFERENCE_PYTHON=/path/to/reference-envs/openelm/bin/python
make ci-verify P=512 CI_DIR=/path/to/ci-work
make ci-profile P=512 CI_DIR=/path/to/ci-work
```

Each interpreter has its own pinned Transformers/Tokenizers/Hub packages and
shares the invoking environment's Torch installation. Reference generation
checks Transformers 5.3.0 for five models and 4.39.3 for OpenELM. The nightly
workflow provisions and forwards these paths; the general runtime stays intact.

`ci-profile` is self-contained: prepare weights, compile, generate the CPU BF16
reference, pass the 32-output gate, then profile. P512 also passes the production
A/B/A bitwise reset check. CPU reference and NPU processes run sequentially.
The normal dependency-aware compiler cache is used; packed weights are reused
under a key of the tracked sources, or, with `CI_BUILD_DIR`, from that build
directory, which the nightly shares with the model's other tests
(`CI_BUILD_DIR=affine-build`). `make weights` reuses packed weights only when
the same packing code made them from the same logical Q4 tensors
(`weights/packing.json`), and packs them again otherwise. Use one CI work directory per model. Failed stages
stop the test and remove any stale performance record for that configuration.

The nightly dashboard gives every configuration its own model ID and P/C label.
These are **fixed full-input requests**, distinct from the short chat prompts
used by older profiles. JSON includes raw warmup/measurement samples, input
hashes, gate hashes, source identity and timing boundaries. Unique evidence
folders retain stage logs, full-logit captures and status even on failure; both
complete and partial nightly artifact uploads include them. Build and reference
time are outside the reported request latency.
