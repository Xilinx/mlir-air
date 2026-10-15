# Frozen benchmark cohort

`cases.json` contains 16 examples from each dataset below, selected independently
with Hugging Face `Dataset.shuffle(seed=0)` and the first 16 rows. It preserves
the readable messages, exact Qwen chat template, token IDs, source row indices,
and prompt/prefill checksums. No benchmark questions were generated for this
delivery. MT-Bench uses its first turn only.

| Dataset and attribution | Split/configuration | Dataset-card license |
|---|---|---|
| [GSM8K, OpenAI](https://huggingface.co/datasets/openai/gsm8k) | `main`, test | MIT |
| [MATH-500, HuggingFaceH4; subset of the MATH benchmark](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | test | Not declared in the card metadata; see the source dataset |
| [HumanEval, OpenAI](https://huggingface.co/datasets/openai/openai_humaneval) | test | MIT |
| [MBPP, Google Research](https://huggingface.co/datasets/google-research-datasets/mbpp) | `sanitized`, test | CC BY 4.0 |
| [MT-Bench prompts, HuggingFaceH4 / FastChat](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts) | train, first turn | Apache 2.0 |

The license column records dataset-card metadata checked September 15, 2026;
dataset content retains its upstream terms. The source links identify the
creators and provide the original datasets. The benchmark adds task instructions
and serializes the messages through the thinking-disabled Qwen3 chat template;
the original questions are preserved in `messages[].content`.

This is a fixed coverage study, not the entire datasets or the full DFlash paper
protocol. The paper-era launcher used 128/128/164/128/80 examples for these five
datasets and evaluated local MT-Bench conversation turns. We use 16 per dataset
and first turns so the same workload can be compared across machines.

The release archive referenced by `download.json` contains:

- `inputs/cases.json` and 80 frozen CPU-prefilled `.npz` files, copied unchanged
  from the measured September 14 study (K/V, hidden taps, first output token).
- `reference/baseline.jsonl` and `reference/spec.jsonl`, including all original
  output token arrays, acceptance traces, and component timings.
- `reference/summary.json`, the original aggregate values.
- `bundle.json`, checksums for the extracted files, and this attribution file.

The archive SHA-256 is pinned in `download.json`; each prefill checksum is also
tracked in `cases.json`. `make benchmark-data` verifies both. The original
source manifest hash is retained as provenance; machine-local paths were
removed from the portable manifest. Output tokens and prefill arrays were not
changed. Use the local [validation guide](../docs/validation.md) for commands and timing
boundaries. The archive's reference logs describe its original September 14
implementation; this example measures the newer selected B8 mapping.
