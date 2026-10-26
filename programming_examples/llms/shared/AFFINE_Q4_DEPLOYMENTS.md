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

Without `CHECKPOINT`, weight preparation downloads the pinned HF revision. `Q4_DIR=/path/to/canonical-q4` reuses already-prepared logical Q4 tensors. Packing preserves their codes, scale and minimum values. Builds and packed weights go into `BUILD_DIR` (default `build/`); a separate directory may be selected. `make compile-kernels` compiles only the C++/LLVM kernels. `compile-decode`, `compile-prefill` and `compile-head` build individual components. Compile decode kernels before prefill variants that reuse their normalization object.

`make compile-all` builds all native P/C configurations. Each point has separate prefill and decode artifacts; the full head is shared. The prefill build manifest records `gemm_bfp16=true`, and runtime rejects prefill artifacts lacking that setting. BFP16 GEMM selection does not change the model's attention precision or its head arithmetic.

## Correctness and timing

Generate an original-checkpoint CPU BF16 reference in an environment with `requirements-reference.txt` installed:

```bash
make reference P=512 CHECKPOINT=/path/to/pinned/hf REFERENCE_DIR=/path/to/reference
make verify P=512 REFERENCE_DIR=/path/to/reference
make profile P=512
```

For OpenELM, use a separate reference environment with Transformers 4.39.3 for its original HF Python implementation. A local OpenELM checkpoint must include the pinned repository's `configuration_openelm.py` and `modeling_openelm.py`; the default HF download includes them. Other models use Transformers 5.3.0. Reference generation loads original checkpoint weights and generates 32 greedy outputs, ignoring EOS. Verification requires complete finite full-vocabulary logits for every output and applies the shared mutual-top-5 check at the first divergent choice. It is a fixed-input model gate.

Profiling warms up with complete fresh 128-output requests for at least five seconds, then measures at least three requests totaling at least ten seconds. It reports median TTFT, `P / median TTFT`, pooled decode throughput `(127 × request_count) / summed_decode_seconds`, and mean complete-request time. TTFT ends after first-logit readback, finite checks and greedy selection. Physical KV overwrite/handoff follows that boundary and is included in decode and complete-request time. Weights and compiled artifacts remain loaded between fresh requests.

`make verify-all` and `make profile-all` visit the native configurations sequentially. NPU commands acquire `/tmp/mlir-air-npu.lock`. `INPUT_IDS=/path/to/ids.json` selects a JSON list containing exactly P valid token IDs; reference and NPU execution must use the same list. No chat template, BOS insertion or early EOS changes the fixed workload.

Raw captures, request samples, selected artifacts and execution placement are written under `BUILD_DIR/runs/`. Host work includes embedding lookup, table preparation, layout/byte transfers, dispatch and sampling; the implementations retaining final single-row prefill normalization on the host report it in the session metadata. Build/setup time is outside request timing.
