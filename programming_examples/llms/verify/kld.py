# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Teacher-forced PPL / KL-divergence statistics against a saved reference.

A port of llama.cpp's `llama-perplexity --kl-divergence` math
(tools/perplexity/perplexity.cpp), so the numbers mean what they mean there:

  * The reference ("base") model's log-probs are saved once per scored token,
    quantized to uint16 over [max-16, max] of the logits. The file layout is
    byte-compatible with llama.cpp's `--kl-divergence-base` file.
  * The model under test is then scored token by token against that file:
    NLL of the true next token for both sides, KL(base || test), the change in
    the true token's probability (dp), and whether both pick the same top-1.
  * Uncertainties use llama.cpp's formulas: standard error of the mean, the
    NLL covariance for ln(PPL ratio), and a binomial error for same-top-1.

Base file layout (all little-endian):
    b"_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk,
    int32 tokens[n_chunk * n_ctx],
    then per chunk, n_ctx - 1 - first records of nv uint16 each (first is
    n_ctx // 2 as in llama.cpp unless a caller scores a shorter tail), where
    nv = 2 * ((n_vocab + 1) // 2) + 4. The first 4 uint16 of a record hold two
    float32s (scale, min_log_prob); the rest are the quantized logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

MAGIC = b"_logits_"
_LOG_PROB_FLOOR = -16.0  # log-probs at or below this are left out of the KLD sum
_ROW_BLOCK = 128


def record_width(n_vocab: int) -> int:
    return 2 * ((n_vocab + 1) // 2) + 4


def scored_per_chunk(n_ctx: int, first: int | None = None) -> int:
    """Tokens scored per chunk: positions first .. n_ctx-2 predict the next
    token, first defaulting to n_ctx // 2."""
    return n_ctx - 1 - (n_ctx // 2 if first is None else first)


def quantize_log_probs(logits: np.ndarray) -> np.ndarray:
    """(n, n_vocab) float logits -> (n, nv) uint16 base records."""
    logits = np.asarray(logits, dtype=np.float32)
    n, n_vocab = logits.shape
    max_logit = logits.max(axis=1)
    min_logit = np.maximum(logits.min(axis=1), max_logit - 16.0)
    sum_exp = np.exp((logits - max_logit[:, None]).astype(np.float64)).sum(axis=1)
    log_sum_exp = np.log(sum_exp).astype(np.float32)
    min_log_prob = (min_logit - max_logit - log_sum_exp).astype(np.float32)
    scale = ((max_logit - min_logit) / np.float32(65535.0)).astype(np.float32)

    out = np.zeros((n, record_width(n_vocab)), dtype=np.uint16)
    header = np.stack([scale, min_log_prob], axis=1).astype(np.float32)
    out[:, :4] = header.view(np.uint16)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_scale = np.where(scale > 0, np.float32(1.0) / scale, np.float32(0.0))
    q = np.rint(inv_scale[:, None] * (logits - min_logit[:, None]))
    q = np.where(logits > min_logit[:, None], q, 0.0)
    out[:, 4 : 4 + n_vocab] = np.clip(q, 0, 65535).astype(np.uint16)
    return out


def dequantize_log_probs(records: np.ndarray, n_vocab: int) -> np.ndarray:
    """(n, nv) uint16 base records -> (n, n_vocab) float32 log-probs."""
    header = np.ascontiguousarray(records[:, :4]).view(np.float32)
    scale, min_log_prob = header[:, 0], header[:, 1]
    q = records[:, 4 : 4 + n_vocab].astype(np.float32)
    return scale[:, None] * q + min_log_prob[:, None]


class BaseLogitsWriter:
    """Streams a base file chunk by chunk."""

    def __init__(
        self,
        path: Path,
        n_ctx: int,
        n_vocab: int,
        tokens: np.ndarray,
        first: int | None = None,
    ):
        tokens = np.asarray(tokens, dtype=np.int32)
        self.per = scored_per_chunk(n_ctx, first)
        if tokens.size % n_ctx:
            raise ValueError("tokens must be a whole number of chunks")
        self.n_ctx, self.n_vocab = n_ctx, n_vocab
        self.n_chunk = tokens.size // n_ctx
        self._written = 0
        self._f = open(path, "wb")
        self._f.write(MAGIC)
        np.array([n_ctx, n_vocab, self.n_chunk], dtype="<i4").tofile(self._f)
        tokens.astype("<i4").tofile(self._f)

    def add_chunk(self, scored_logits: np.ndarray):
        if scored_logits.shape != (self.per, self.n_vocab):
            raise ValueError(f"unexpected logits shape {scored_logits.shape}")
        quantize_log_probs(scored_logits).astype("<u2").tofile(self._f)
        self._written += 1

    def close(self):
        self._f.close()
        if self._written != self.n_chunk:
            raise RuntimeError(f"wrote {self._written} of {self.n_chunk} chunks")


class BaseLogitsReader:
    """Memory-maps a base file; `chunk(i)` returns that chunk's (n, nv) records.
    `first` must match the writer's; the file does not record it."""

    def __init__(self, path: Path, first: int | None = None):
        with open(path, "rb") as f:
            if f.read(8) != MAGIC:
                raise ValueError(f"{path} is not a base logits file")
            self.n_ctx, self.n_vocab, self.n_chunk = (
                int(v) for v in np.fromfile(f, dtype="<i4", count=3)
            )
        tok_off = 8 + 12
        self.tokens = np.memmap(
            path,
            dtype="<i4",
            mode="r",
            offset=tok_off,
            shape=(self.n_chunk * self.n_ctx,),
        )
        self._nv = record_width(self.n_vocab)
        self.first = self.n_ctx // 2 if first is None else first
        self._per = scored_per_chunk(self.n_ctx, self.first)
        self._records = np.memmap(
            path,
            dtype="<u2",
            mode="r",
            offset=tok_off + 4 * self.tokens.size,
            shape=(self.n_chunk * self._per, self._nv),
        )

    def chunk_tokens(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[i * self.n_ctx : (i + 1) * self.n_ctx])

    def chunk(self, i: int) -> np.ndarray:
        return np.asarray(self._records[i * self._per : (i + 1) * self._per])


@dataclass
class KldStats:
    """Running sums, as llama.cpp's kl_divergence_result."""

    sum_nll: float = 0.0
    sum_nll2: float = 0.0
    sum_nll_base: float = 0.0
    sum_nll_base2: float = 0.0
    sum_nll_nll_base: float = 0.0
    sum_kld: float = 0.0
    sum_kld2: float = 0.0
    sum_p_diff: float = 0.0
    sum_p_diff2: float = 0.0
    sum_p_diff4: float = 0.0
    n_same_top: int = 0
    count: int = 0
    kld_values: list = field(default_factory=list)
    p_diff_values: list = field(default_factory=list)

    def add(
        self,
        logits: np.ndarray,
        base_records: np.ndarray,
        next_tokens: np.ndarray,
        normalized: bool = False,
    ):
        """Score rows of test `logits` (n, n_vocab) against base records (n, nv);
        `next_tokens[j]` is the true token row j predicts. `normalized`: the
        rows are already log-probs, so they are used as they are."""
        n = logits.shape[0]
        if base_records.shape[0] != n or next_tokens.shape[0] != n:
            raise ValueError("logits, base records and tokens disagree on length")
        # Row blocks bound the float64 temporaries.
        for lo in range(0, n, _ROW_BLOCK):
            hi = min(lo + _ROW_BLOCK, n)
            self._add_block(
                logits[lo:hi], base_records[lo:hi], next_tokens[lo:hi], normalized
            )

    def _add_block(self, logits, base_records, next_tokens, normalized):
        logits = np.asarray(logits, dtype=np.float32)
        n, n_vocab = logits.shape
        rows = np.arange(n)
        log_q = logits.astype(np.float64) if normalized else _log_softmax(logits)
        log_p = dequantize_log_probs(base_records, n_vocab).astype(np.float64)

        nll = -log_q[rows, next_tokens]
        nll_base = -log_p[rows, next_tokens]
        p = np.where(log_p > _LOG_PROB_FLOOR, np.exp(log_p), 0.0)
        kld = (p * (log_p - log_q)).sum(axis=1)
        p_diff = np.exp(-nll) - np.exp(-nll_base)

        self.sum_nll += nll.sum()
        self.sum_nll2 += (nll * nll).sum()
        self.sum_nll_base += nll_base.sum()
        self.sum_nll_base2 += (nll_base * nll_base).sum()
        self.sum_nll_nll_base += (nll * nll_base).sum()
        self.sum_kld += kld.sum()
        self.sum_kld2 += (kld * kld).sum()
        self.sum_p_diff += p_diff.sum()
        self.sum_p_diff2 += (p_diff**2).sum()
        self.sum_p_diff4 += (p_diff**4).sum()
        self.n_same_top += int((logits.argmax(axis=1) == log_p.argmax(axis=1)).sum())
        self.count += n
        self.kld_values.extend(kld.tolist())
        self.p_diff_values.extend(p_diff.tolist())

    def merge(self, other: "KldStats"):
        """Fold another accumulator's tokens into this one."""
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            setattr(self, f.name, a + b)

    def summary(self, min_tokens: int = 100) -> dict:
        """The statistics llama-perplexity prints at the end of a KLD run.
        Fewer than `min_tokens` scored tokens is an error, as there."""
        n = self.count
        if n < max(min_tokens, 2):
            raise ValueError(f"only {n} scored tokens; need at least {min_tokens}")
        ln_ppl, ln_ppl_se = _mean_se(self.sum_nll, self.sum_nll2, n)
        ln_base, ln_base_se = _mean_se(self.sum_nll_base, self.sum_nll_base2, n)
        cov = (
            self.sum_nll_nll_base / n - (self.sum_nll / n) * (self.sum_nll_base / n)
        ) / (n - 1)
        ppl, ppl_base = math.exp(ln_ppl), math.exp(ln_base)
        ln_ratio = ln_ppl - ln_base
        ln_ratio_se = math.sqrt(max(ln_ppl_se**2 + ln_base_se**2 - 2 * cov, 0.0))
        ppl_se, ppl_base_se = ppl * ln_ppl_se, ppl_base * ln_base_se
        ppl_diff_se = math.sqrt(
            max(ppl_se**2 + ppl_base_se**2 - 2 * ppl * ppl_base * cov, 0.0)
        )
        kld_mean, kld_se = _mean_se(self.sum_kld, self.sum_kld2, n)
        dp_mean, dp_se = _mean_se(self.sum_p_diff, self.sum_p_diff2, n)
        dp_mse, dp_mse_se = _mean_se(self.sum_p_diff2, self.sum_p_diff4, n)
        rms_dp = math.sqrt(dp_mse)
        same_top = self.n_same_top / n
        kv = np.sort(np.asarray(self.kld_values))
        dv = np.sort(np.asarray(self.p_diff_values))
        denom = ln_ppl_se * ln_base_se
        return {
            "n_tokens": n,
            "ppl": ppl,
            "ppl_se": ppl_se,
            "ppl_base": ppl_base,
            "ppl_base_se": ppl_base_se,
            "ln_ppl_cor": cov / denom if denom > 0 else 0.0,
            "ln_ppl_ratio": ln_ratio,
            "ln_ppl_ratio_se": ln_ratio_se,
            "ppl_ratio": math.exp(ln_ratio),
            "ppl_ratio_se": math.exp(ln_ratio) * ln_ratio_se,
            "ppl_diff": ppl - ppl_base,
            "ppl_diff_se": ppl_diff_se,
            "kld_mean": kld_mean,
            "kld_se": kld_se,
            "kld_percentiles": {
                k: float(np.percentile(kv, q)) for k, q in _PERCENTILES_KLD
            },
            "dp_mean": dp_mean,
            "dp_se": dp_se,
            "dp_percentiles": {
                k: float(np.percentile(dv, q)) for k, q in _PERCENTILES_DP
            },
            "rms_dp": rms_dp,
            "rms_dp_se": 0.5 / rms_dp * dp_mse_se if rms_dp > 0 else 0.0,
            "same_top": same_top,
            "same_top_se": math.sqrt(same_top * (1 - same_top) / (n - 1)),
        }


_PERCENTILES_KLD = [
    ("max", 100),
    ("p99.9", 99.9),
    ("p99", 99),
    ("p95", 95),
    ("p90", 90),
    ("median", 50),
    ("p10", 10),
    ("p5", 5),
    ("p1", 1),
    ("p0.1", 0.1),
    ("min", 0),
]
_PERCENTILES_DP = [
    ("max", 100),
    ("p99.9", 99.9),
    ("p99", 99),
    ("p95", 95),
    ("p90", 90),
    ("p75", 75),
    ("median", 50),
    ("p25", 25),
    ("p10", 10),
    ("p5", 5),
    ("p1", 1),
    ("p0.1", 0.1),
    ("min", 0),
]


def format_summary(s: dict) -> str:
    """Render a summary the way llama-perplexity prints it."""
    lines = [
        "====== Perplexity statistics ======",
        f"Mean PPL(Q)                   : {s['ppl']:10.6f} ± {s['ppl_se']:10.6f}",
        f"Mean PPL(base)                : {s['ppl_base']:10.6f} ± {s['ppl_base_se']:10.6f}",
        f"Cor(ln(PPL(Q)), ln(PPL(base))): {100 * s['ln_ppl_cor']:6.2f}%",
        f"Mean ln(PPL(Q)/PPL(base))     : {s['ln_ppl_ratio']:10.6f} ± {s['ln_ppl_ratio_se']:10.6f}",
        f"Mean PPL(Q)/PPL(base)         : {s['ppl_ratio']:10.6f} ± {s['ppl_ratio_se']:10.6f}",
        f"Mean PPL(Q)-PPL(base)         : {s['ppl_diff']:10.6f} ± {s['ppl_diff_se']:10.6f}",
        "",
        "====== KL divergence statistics ======",
        f"Mean    KLD: {s['kld_mean']:10.6f} ± {s['kld_se']:10.6f}",
    ]
    lines += [f"{k:<8}KLD: {v:10.6f}" for k, v in s["kld_percentiles"].items()]
    lines += [
        "",
        "====== Token probability statistics ======",
        f"Mean    Δp: {100 * s['dp_mean']:6.3f} ± {100 * s['dp_se']:5.3f} %",
    ]
    lines += [f"{k:<8}Δp: {100 * v:6.3f}%" for k, v in s["dp_percentiles"].items()]
    lines += [
        f"RMS Δp    : {100 * s['rms_dp']:6.3f} ± {100 * s['rms_dp_se']:5.3f} %",
        f"Same top p: {100 * s['same_top']:6.3f} ± {100 * s['same_top_se']:5.3f} %",
    ]
    return "\n".join(lines)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def _mean_se(total: float, total2: float, n: int) -> tuple[float, float]:
    f = total / n
    var = total2 / n - f * f
    return f, (math.sqrt(var / (n - 1)) if var > 0 and n > 10 else 0.0)


def compare_base_files(
    test_path: Path, base_path: Path, first: int | None = None
) -> dict:
    """Score one base file against another over the same tokens. The test
    side's stored log-probs stand in for its logits, so this measures, for
    example, a quantized reference against the bf16 one without an NPU."""
    test = BaseLogitsReader(test_path, first)
    base = BaseLogitsReader(base_path, first)
    if (test.n_ctx, test.n_vocab) != (base.n_ctx, base.n_vocab):
        raise ValueError("base files differ in n_ctx or n_vocab")
    n_chunk = min(test.n_chunk, base.n_chunk)
    if test.first != base.first:
        raise ValueError("base files score different positions")
    first = base.first
    stats = KldStats()
    for ci in range(n_chunk):
        toks = base.chunk_tokens(ci)
        if not np.array_equal(toks, test.chunk_tokens(ci)):
            raise ValueError(f"chunk {ci} holds different tokens in the two files")
        # The stored values are already normalized log-probs. Renormalizing
        # them would count every floor-clamped tail entry at e^-16, enough to
        # make two identical files differ.
        log_probs = dequantize_log_probs(test.chunk(ci), test.n_vocab)
        stats.add(
            log_probs,
            base.chunk(ci),
            toks[first + 1 :].astype(np.int64),
            normalized=True,
        )
    return stats.summary()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        sys.exit("usage: kld.py TEST.kld BASE.kld")
    print(format_summary(compare_base_files(Path(sys.argv[1]), Path(sys.argv[2]))))
