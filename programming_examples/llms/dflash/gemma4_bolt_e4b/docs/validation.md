# Validation

`make verify` runs these checks; each writes a JSON record under
`build/verification/`.

| Check | Compares | Gate |
|---|---|---|
| `tests/check_target.py` | Two eight-token blocks (empty cache, then over the first block's K/V) against `reference_b8.py`'s float32 forward of the same weights | Every token's logit cosine at least 0.99 |
| `tests/check_long_context.py` | Sixteen tokens after contexts of 64 to 4064 against the NPU prefill, teacher forced | Each token's pick within 3 logits of the prefill's best, logit cosine at least 0.98 |
| `tests/check_attention.py` | The attention graph alone, full and sliding window, random inputs against float32 | Every output within 0.1 + 1.6% |
| `verify_loop.py` | The shared loop check: prefill repeats and prefixes, DFlash against the same decoder one token at a time, committed K/V | Exact |
| `tests/check_model.py` | Two prompts x 32 tokens of the full DFlash path against the float32 reference over the NPU's tokens | `llms/verify`'s mutual top-5 |

The long-context gate is looser than a near-tie rule: at a few thousand
tokens, both the decoder and the prefill move logits by about a logit from the
float32 reference and miss different near ties, so the two can disagree by
about two. A broken sliding window misses by ten or more and drops the
cosine well below the gate.

DFlash output is exactly the eight-token decoder's own output one token at a
time (`verify_loop.py`). The single-token B1 decoder uses other arithmetic, so
its greedy output can differ from DFlash's at near ties.
