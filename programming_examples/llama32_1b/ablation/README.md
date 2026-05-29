# Llama-3.2-1B NPU2 Ablation Study

4-cell controlled measurement of how each dispatch optimization (multi-launch
ELF stitching, per-layer weight BOs, shared intermediate BOs) contributes to
the production runtime.

Two sister studies:

| Subdir | Scope | Cell D headline |
|---|---|---|
| [`decode/`](decode/) | Full per-token loop: 16 × (rms_gemv_rope + decode_attention_cpu + o_gemv_ffn) + LM head + argmax | 90.65 ms/token; A→D = **2.83×** |
| [`prefill/`](prefill/) | Full 16-layer prefill: 16 × (rms_gemms_rope + FA + o_ffn) | 1.13 s/pass; A→D = **1.56×** |

Both studies use the same 4-cell ladder (A naive → B + per-layer weight BOs
→ C + shared intermediate BOs → D production-merged), bit-exact validation
against committed Cell D goldens, and the NPU exclusive-lock timing
protocol.

**Audience-facing walkthrough**: [`../docs/ABLATION_STUDY.html`](../docs/ABLATION_STUDY.html)
— headline numbers, methodology, cross-comparison.

**Reproducibility** (each subdir is self-contained):

```sh
cd decode/    && make all     # ~10 min, NPU-locked
cd prefill/   && make all     # ~15 min, NPU-locked
```

## Companion docs (in repo)

- [`../docs/IMPLEMENTATION_GUIDE.html`](../docs/IMPLEMENTATION_GUIDE.html) — production codebase walkthrough; B3-B7 describes the four gaps that the cells ablate
- [`../docs/profile.md`](../docs/profile.md) — production runtime numbers reproduced by Cell D
- `docs/specs/2026-05-07-llama32-1b-ablation-plan2-prefill-design.md` — prefill spec
- `docs/specs/2026-05-12-llama32-1b-ablation-plan2-fulldecode-design.md` — decode spec
- `docs/plans/...` — corresponding step-by-step implementation plans
