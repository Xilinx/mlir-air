# Flash Attention Precision Analysis: Matching IRON

## Goal

Match IRON MHA's test configuration:
- Input range: [0, 4), seed=42, torch.rand
- Per-element tolerance: `diff < max(atol=0.15, rtol=0.04 * (|a|+|b|))`
- Error threshold: **0.5% of elements allowed to fail**
- Golden reference: PyTorch `scaled_dot_product_attention` on bf16

## Current Status

| Aspect | IRON | mlir-air | Match? |
|--------|------|----------|--------|
| Golden ref | PyTorch SDPA bf16 | PyTorch SDPA bf16 | **Yes** |
| Tolerance formula | `diff < max(atol, rtol*(|a|+|b|))` | Same | **Yes** |
| Error threshold | 0.5% elements | 0.5% elements | **Yes** |
| atol / rtol | 0.15 / 0.04 | 0.15 / 0.04 | **Yes** |
| Rounding mode | conv_even | conv_even | **Yes** |
| val_range | [0, 4) | [0, 2.5) | **No** |
| Softmax architecture | Fused partial_softmax | Multi-function | **No** |

## Precision vs Input Range

| val_range | Error rate | Passes 0.5%? | Notes |
|-----------|-----------|:---:|-------|
| 2.0 | 0% | Yes | All configurations pass |
| 2.5 | 0.006-0.025% | Yes | Default. 390/1.57M errors for NH=12 |
| 3.0 | 5.2% | No | Errors non-uniform: 0-38% per Q row |
| 4.0 | 63% | No | Errors uniform: 26-91% per Q row |

## Root Cause of val_range > 2.5 Failure

### Error spatial distribution (val_range=4, NH=2, LQ=2048)

- **Per-head**: uniform (62-64% each) — not a head-routing issue
- **Per-Q-row**: **non-uniform (26-91%)** — correlates with softmax sharpness
- **Per-DV-column**: mildly non-uniform (57-71%) — not a dominant factor

The Q-row variation indicates the errors correlate with attention score
magnitude. Rows where softmax is more peaked (a few K elements dominate)
have worse precision because bf16 intermediate truncation amplifies the
wrong max/exp selection.

### Architectural difference from IRON

IRON's `partial_softmax_alias_bf16` (softmax.cc) keeps softmax
computation in accfloat throughout:

```cpp
// IRON: multiply-then-subtract in accfloat
scaled_accum = aie::mul(input_bf16, log2e_vec);     // accfloat
exp_in_accum = aie::sub(scaled_accum, max_val_vec);  // accfloat - bf16 → accfloat
exp_val = aie::exp2<bfloat16>(exp_in_accum.to_vector<float>());
```

Our kernel uses separate extern "C" functions for each step (max, exp,
sum), storing results as bf16 in L1 between calls:

```
max_g_bf16  →  bf16 u buffer
exp_g_minus_u  →  bf16 G buffer (overwritten in-place)
sum_g  →  bf16 s buffer
```

Each bf16 store-load cycle introduces rounding. The `getExpBf16Precise`
template was added to compute `log2e*score - log2e*max` in accfloat
(avoiding the bf16 subtraction precision loss), but this had **no
measurable impact** — the dominant precision loss comes from the
inter-function bf16 round-trips, not the subtraction itself.

## Fixes Applied

1. **conv_even rounding** (`attn.cc`) — all 19 extern "C" functions
2. **f32 reduce_add in sum_g** (`attn.cc`) — `accfloat → float` before reduce
3. **accfloat exp subtraction** (`attn.cc`) — `getExpBf16Precise` template
4. **0.5% error threshold** (`xrt_runner.py`) — `error_threshold` parameter
5. **PyTorch SDPA golden** (`attn.py`) — matches IRON's reference.py pattern

## Path to val_range=4

To match IRON's val_range=4 with < 0.5% errors, the kernel needs a
fused softmax that keeps all intermediates in accfloat within a single
function call (eliminating bf16 round-trips between separate max/exp/sum
functions). This is a significant kernel restructure.

## IRON Reference Coordinates

| File | Line | Content |
|------|------|---------|
| `IRON/aie_kernels/aie2p/mha.cc:12` | Define | `ROUNDING_MODE conv_even` |
| `IRON/aie_kernels/aie2p/softmax.cc:107` | Accfloat | `accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, scaled_accum` |
| `IRON/aie_kernels/aie2p/softmax.cc:145` | Key diff | `exp_in_accum = aie::sub(scaled_accum, max_val_vec)` (accfloat sub) |
| `IRON/iron/operators/mha/test.py:66` | Tolerance | `rel_tol=4.0e-2, abs_tol=1.5e-1` |
| `IRON/iron/operators/mha/test.py:69` | Threshold | `error_threshold = 0.005` |
| `IRON/iron/operators/mha/reference.py:56` | Range | `val_range = 4` |
| `IRON/iron/operators/mha/reference.py:67` | Golden | `torch.nn.functional.scaled_dot_product_attention` |
| `IRON/iron/operators/mha/test.py:16` | Shape | `(16384, 64, 1, 8)` |
