# Tests and nightly integration

The Llama example follows the existing DFlash Make/LIT interface. The generic
LLM suite filters discover the nested model directory; no model-specific
nightly job is necessary.

| Entry | What runs | Standard suite |
|---|---|---|
| `run_npu2_compile.lit` | Weight-free target, drafter, context prepass and NPU prefill compilation | `check-programming-examples-llms-compile` |
| `run_npu2_verify.lit` | Six-prefix state checks, full DFlash oracle test and HF BF16 token comparison | `check-programming-examples-llms-verify` |
| `run_npu2_profile.lit` | Fixed prompt, warm NPU prefill and complete decode, cap256 | `check-programming-examples-llms-profile` |
| `run_npu2_full20.lit` | Paired B1/DFlash, four requests per dataset, cap2048 | `check-programming-examples-llms-full20` |
| `run_npu2_full80.lit` | Optional sixteen requests per dataset | `check-programming-examples-llms-full80` |

Full80 requires the `dflash_full80` feature and is excluded from the ordinary
nightly schedule. The shared host acceptance/streaming tests remain in the
Qwen3-4B entry and run through the generic host suite; they are not duplicated
for each model. The download workflow seeds the three pinned repositories in
`llms/hf_models.txt` and verifies that the required files are available offline.
LIT checks their corresponding `hfweights_*` features before device tests.

The profile uses `Introduce me what is GPU`, greedy decoding and EOS stopping.
It emits `llama31_8b_dflash_b8.perf.json`, identified by its `PERF_MODEL` tag.
The benchmark uses the shared frozen cohort, including first-turn MT-Bench.
Both B1 and DFlash receive the same saved NPU prefill; their generation is
independent and can diverge because their arithmetic differs. The README
defines the decode timer and pooled-throughput calculation.

## Correctness coverage

`make verify-state` checks target and drafter at KV prefixes 0, 7, 16, 64,
1024 and 4088. It checks repeated logits/hidden outputs, full KV hashes,
causal or bidirectional block visibility, untouched prefix/tail regions and
drafter KV injection. `verify_loop.py` then runs NPU prefill, generates 32
fixed-length output tokens, and compares two DFlash executions with a
sequential oracle using the same B8 arithmetic. Acceptance decisions and
committed target KV must match exactly on the checked rounds.

`make verify-model` captures two standard instruct prompts, 32 tokens each,
using NPU prefill plus DFlash and a separately run HF BF16 reference. EOS does
not shorten these traces. It uses AIR's existing mutual top-5 comparator: at
the first differing chosen token, each choice must belong to the other's
top-5. The report records trace lengths and the visited prefix. That comparator
does not verify later tokens after an accepted divergence; the separate
same-arithmetic loop check supplies the DFlash acceptance/state evidence.

The production-linked attention and BF16/BFP prepass diagnostic scripts under
`tests/` provide operator-level numerical and visibility checks used during
qualification. A compilation, host test or fluent response alone is not a
complete model-verification result. Device commands serialize on
`/tmp/mlir-air-npu.lock`, distinct from the prefill helper's inner lock.
