# Llama-3.2-1B Plan 2 (Full Decode) Ablation

Bit-exact 4-cell ablation of the production **decode** pipeline:
`rms_gemv_rope` (6 sub-launches) + `decode_attention_cpu` (invariant) +
`o_gemv_ffn` (8 sub-launches) per layer × 16 layers + final RMSNorm +
`lm_head_gemv` (invariant) + argmax.

Per-trial timed unit: **one decode token** at fixed `current_pos = 7`
(after a 7-token synthetic pre-fill of the KV cache). 5 trials, drop trial 1
as warmup, median + (min, max) over remaining 4.

Companion docs:
- Spec: [`../docs/specs/2026-05-12-llama32-1b-ablation-plan2-fulldecode-design.md`](../docs/specs/2026-05-12-llama32-1b-ablation-plan2-fulldecode-design.md)
- Plan: [`../docs/plans/2026-05-12-llama32-1b-ablation-plan2-fulldecode-plan.md`](../docs/plans/2026-05-12-llama32-1b-ablation-plan2-fulldecode-plan.md)
- Sister study (prefill): [`../prefill/README.md`](../prefill/README.md)
- Audience-facing summary: [`../../docs/ABLATION_STUDY.html`](../../docs/ABLATION_STUDY.html)

## What this measures

Four cells, identical computation, different dispatch strategy. CPU attention
and LM head are held INVARIANT across all 4 cells.

| Cell | What changes within each kernel-group | Adds |
|------|---------------------------------------|------|
| A | 6+8 separate `xrt.run()` per layer, host round-trip on every intermediate | (baseline) |
| B | + per-layer weight BOs (`static_input_indices`) | #2 |
| C | + shared intermediate BOs across separate `xrt.run()` calls (within each group) | #3 |
| D | + multi-launch merging (production: 6→1 + 8→1 ELF per layer) | #1 |

NPU calls per token (16 layers + LM head):
- Cell A/B/C: **(6 + 8) × 16 + 1 = 225 dispatches** (LM head invariant-merged)
- Cell D: **(1 + 1) × 16 + 1 = 33 dispatches**

## Quick start

```
make compile     # one-time, ~5-10 min for all 4 cells' ELFs + LM head
make run         # 4 cells × 5 trials (~2-3 min, NPU-locked)
make report      # markdown report
```

## Validation gate

Every cell must produce **bit-identical** output bytes vs. committed Cell D
goldens for both kernel-groups (`golden_rms_gemv_rope_decode.npz`,
`golden_o_gemv_ffn_decode.npz`). Cells failing the gate suppress their timing.

## Reproducibility

```
cd programming_examples/llama32_1b/ablation/decode
make clean
make all
```

NPU-free unit tests (smoke test the harness scaffolding):

```
make test
```

Expected: **8 passed** (4 KV-cache state tests + 4 validation-gate tests).

## File map

| Path | Purpose |
|------|---------|
| `specs/kernel_group.py` | Re-export prefill study's frozen dataclasses |
| `specs/rms_gemv_rope.py` | Concrete spec for the 6-launch decode attention pre-block |
| `specs/o_gemv_ffn.py` | Concrete spec for the 8-launch decode FFN block |
| `standalone_builders/rms_gemv_rope.py` | 6 single-launch builders + STANDALONES registry |
| `standalone_builders/o_gemv_ffn.py` | 8-element STANDALONES registry derived from spec |
| `cells/kernel_group.py` (re-export) + `cells/common.py` (re-export) | Shared infrastructure |
| `cells/cell_a_naive.py` | Cell A — copy of Plan 1 with decode-spec branches added |
| `cells/cell_b_static.py` | Cell B — same |
| `cells/cell_c_charitable.py` | Cell C — same |
| `cells/cell_d_merged.py` | Cell D — production-merged decode dispatches |
| `cells/decode_attn_const.py` | Invariant CPU attention runner |
| `cells/lm_head_const.py` | Invariant 8-partition LM head runner |
| `cells/per_token_loop.py` | The end-to-end timed unit |
| `cells/kv_cache.py` | Deterministic KV-cache init + per-trial reset |
| `golden/regen_golden.py` | Cell-D one-shot to regenerate goldens |
| `golden/golden_*.npz` | Two committed bf16 goldens + meta json |
| `validate.py` | Bit-exact gate (re-export of Plan 1's parameterized validator) |
| `run_ablation.py` | Orchestrator — compile, preload, validate, time × 4 cells |
| `analyze.py` | JSON → markdown report |
| `Makefile` | Convenience targets |
| `tests/` | NPU-free unit tests |

## Limitations

- Single token at fixed position. By design (see spec §5): keeps `decode_attention_cpu`
  CPU work constant across trials, isolates dispatch overhead. Position-dependent
  multi-token decode is out of scope.
- Synthetic seed=42 weights only. No HuggingFace.
- LM head held INVARIANT across cells. A potential follow-up could ablate it.
- NPU FlashAttention decode path NOT measured. Production uses CPU attention at head_dim=64.
