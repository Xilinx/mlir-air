"""Numerical comparators for end-to-end verify.

All metrics are pure numpy. Inputs may be bfloat16 or float32; we cast to
float32 internally.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


def per_position_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per position (per row).

    Reshape the inputs to (n_positions, feature_dim) by treating axis 0 as
    the position axis and flattening all remaining axes. Returns a 1D array
    of length n_positions, with NaN-safe handling: positions where either
    side has zero norm return 0.0 (not NaN).
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    n_pos = a.shape[0]
    a2 = a.reshape(n_pos, -1)
    b2 = b.reshape(n_pos, -1)
    dot = np.sum(a2 * b2, axis=1)
    na = np.linalg.norm(a2, axis=1)
    nb = np.linalg.norm(b2, axis=1)
    denom = na * nb
    out = np.zeros(n_pos, dtype=np.float32)
    mask = denom > 0
    out[mask] = dot[mask] / denom[mask]
    return out


def aggregate(cosines: np.ndarray) -> dict:
    """Aggregate per-position cosines into {min, p5, median, mean}."""
    arr = np.asarray(cosines, dtype=np.float32)
    return {
        "min": float(arr.min()),
        "p5": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
    }


def error_metrics(a: np.ndarray, b: np.ndarray) -> dict:
    """Element-wise abs/rel error stats — diagnostic complement to cosine.

    cosine is direction-only and ignores magnitude (e.g. b = 2*a -> cos = 1).
    abs/rel error catches the magnitude-side errors cosine misses.
    """
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    diff = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-6)
    rel = diff / denom
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
    }


@dataclass
class ComparisonRecord:
    """One per-layer probe result. Pure observation — diagnosis does not gate
    on these (`make verify` is the gate). Threshold + status fields used to
    live here and were retired with the threshold-based diagnosis design."""

    name: str
    pair: str  # "npu_vs_hf"
    layer: Optional[int]
    cosine: dict  # {min, p5, median, mean}
    errors: dict  # {max_abs, mean_abs, max_rel, mean_rel}

    def to_dict(self) -> dict:
        return asdict(self)


def compare_pair(
    name: str, npu: np.ndarray, hf: np.ndarray, layer: int | None
) -> ComparisonRecord:
    """Compute per-position cosine + element-wise error for one NPU vs HF
    layer probe. No threshold, no pass/fail — diagnosis is informational."""
    cos = per_position_cosine(npu, hf)
    return ComparisonRecord(
        name=name,
        pair="npu_vs_hf",
        layer=layer,
        cosine=aggregate(cos),
        errors=error_metrics(npu, hf),
    )


# ---------------------------------------------------------------------------
# Token-level top-k set inclusion check (the model-level correctness gate)
# ---------------------------------------------------------------------------
#
# Mirrors the logic of vLLM's tests/models/utils.py::check_logprobs_close.
# At each generation step:
#   - If both runners chose the same token, skip (no check needed).
#   - Otherwise: the first divergence is the only step we check. Each side's
#     chosen token must appear in the OTHER side's top-k. If either fails,
#     status is FAIL with a human-readable reason. If both succeed, status
#     is OK — divergence is informational drift within the top-k band.
# After the first divergence we stop (vLLM does the same: once divergent, the
# downstream tokens are no longer apples-to-apples since each side is feeding
# its own chosen token into the next step).
#
# This is the discrete-judgment escape from continuous-metric ULP wars: bf16
# noise can flip top-1 even between two implementations that are mathematically
# equivalent, but it almost never displaces a token out of the top-5.


def topk_token_ids(z: np.ndarray, k: int = 5) -> list[int]:
    """Return the top-k token IDs from a 1D logit vector, highest first.

    Tie-breaking matches numpy.argmax: when two logits are exactly equal
    (which happens routinely with bf16 inputs cast to F32, since adjacent
    bf16 values land at the same F32 representation), the smaller token
    ID wins. Without this, topk_token_ids[0] could disagree with
    np.argmax(z) on the SAME array.
    """
    z = np.asarray(z)
    if z.ndim != 1:
        raise ValueError(f"expected 1D logit vector, got shape {z.shape}")
    if k > z.shape[0]:
        raise ValueError(f"k={k} > vocab_size={z.shape[0]}")
    idx = np.argpartition(-z, k - 1)[:k]
    # lexsort: last key is primary. Primary = -z[idx] (largest z first);
    # secondary = idx (smaller token-ID first as tiebreaker).
    order = np.lexsort((idx, -z[idx]))
    idx = idx[order]
    return idx.tolist()


@dataclass
class TopKCheckRecord:
    """Result of a single top-k token-level inclusion check on one prompt."""

    prompt_idx: int
    prompt_text: str  # may be truncated for the report
    n_steps: int
    k: int
    divergence_step: Optional[int]
    test_chosen_at_div: Optional[int]
    ref_chosen_at_div: Optional[int]
    test_topk_at_div: Optional[list[int]]
    ref_topk_at_div: Optional[list[int]]
    status: str  # "OK" | "FAIL"
    fail_reason: Optional[str]
    # 1-based rank of each side's chosen token within the OTHER side's top-k.
    # None when the chosen token is not present (FAIL on that direction) or
    # when there is no divergence at all.
    test_chosen_rank_in_ref: Optional[int] = None
    ref_chosen_rank_in_test: Optional[int] = None
    # Decoded human-readable rendering (orchestrator populates via tokenizer).
    test_chosen_text_at_div: Optional[str] = None
    ref_chosen_text_at_div: Optional[str] = None
    agreed_prefix_text: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def compute_topk_set_check(
    test_chosen: list[int],
    test_topk: list[list[int]],
    ref_chosen: list[int],
    ref_topk: list[list[int]],
    k: int = 5,
    prompt_idx: int = 0,
    prompt_text: str = "",
) -> TopKCheckRecord:
    """Top-k token-level inclusion check on one prompt's generation sequence.

    Walk in lockstep. On the first chosen-token mismatch, both sides' chosen
    tokens must appear in the OTHER side's top-k; otherwise FAIL. Stop after
    the first divergence (mirrors vLLM's check_logprobs_close). All-match
    returns OK with divergence_step=None.
    """
    n = min(len(test_chosen), len(ref_chosen), len(test_topk), len(ref_topk))
    for i in range(n):
        if test_chosen[i] == ref_chosen[i]:
            continue
        ref_top = list(ref_topk[i][:k])
        test_top = list(test_topk[i][:k])
        try:
            test_rank: Optional[int] = ref_top.index(test_chosen[i]) + 1
        except ValueError:
            test_rank = None
        try:
            ref_rank: Optional[int] = test_top.index(ref_chosen[i]) + 1
        except ValueError:
            ref_rank = None
        test_in_ref = test_rank is not None
        ref_in_test = ref_rank is not None
        if test_in_ref and ref_in_test:
            status, reason = "OK", None
        else:
            parts = []
            if not test_in_ref:
                parts.append(
                    f"test chose {test_chosen[i]} but it is not in ref top-{k} "
                    f"({ref_top})"
                )
            if not ref_in_test:
                parts.append(
                    f"ref chose {ref_chosen[i]} but it is not in test top-{k} "
                    f"({test_top})"
                )
            status, reason = "FAIL", "; ".join(parts)
        return TopKCheckRecord(
            prompt_idx=prompt_idx,
            prompt_text=prompt_text,
            n_steps=n,
            k=k,
            divergence_step=i,
            test_chosen_at_div=int(test_chosen[i]),
            ref_chosen_at_div=int(ref_chosen[i]),
            test_topk_at_div=[int(t) for t in test_top],
            ref_topk_at_div=[int(t) for t in ref_top],
            status=status,
            fail_reason=reason,
            test_chosen_rank_in_ref=test_rank,
            ref_chosen_rank_in_test=ref_rank,
        )
    return TopKCheckRecord(
        prompt_idx=prompt_idx,
        prompt_text=prompt_text,
        n_steps=n,
        k=k,
        divergence_step=None,
        test_chosen_at_div=None,
        ref_chosen_at_div=None,
        test_topk_at_div=None,
        ref_topk_at_div=None,
        status="OK",
        fail_reason=None,
    )
