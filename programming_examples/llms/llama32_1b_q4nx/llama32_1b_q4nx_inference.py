#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""llama32_1b_q4nx full inference — the reference-faithful two-path generation.

Reproduces the reference's Llama-3.2-1B architecture: a batched-parallel prefill (M=P GEMM,
`llama32_1b_q4nx_prefill.py`) fills a shared per-layer KV cache, then a fused per-token decode
(the fused decode, one dispatch = 16 layers + lm_head) reads+appends that cache to
generate autoregressively. Embedding (1-row) and argmax run on the host between tokens, as
in the reference (`Embedding::forward` + host argmax).

The decode is ONE xclbin built at ATTN_MAXL=2048 with a compile-time 128-block attention loop
that skips fully-masked far blocks — exactly the reference's single MAX_L cache. Each token is specialized
on the host by `DecodeInstsGen`, the AIR analog of the reference's per-token `gen_layer_seq`/`rtp_write`:
an RTP-L + KV-append-offset insts patch serves every L in [1, 2048] from the single xclbin — no
per-window xclbin, no window switching, no per-window weight replication.

Prefill and decode use the NPU sequentially (prefill runs in a worker subprocess that
releases the device before decode sets up) to avoid AIE column contention between the two
resident contexts — faithful to the reference's separate prefill/decode xclbins.

Single weight source: the model.q4nx bundle (Q4NX_MODEL_SOURCE, default
FastFlowLM/Llama-3.2-1B-NPU2). The prefill loads it directly; the decode's
q4k-cascade requant cache + embed/norm golden are derived from the same bundle on
first use (cached under ~/.cache/q4nx). Q4NX_DECODE_WEIGHTS_NPZ / Q4NX_GOLDEN_DIR
may override those if pre-supplied.

Run:
  python3 llama32_1b_q4nx_inference.py                 # Paris gate (default prompt)
  python3 llama32_1b_q4nx_inference.py --n-tokens 9 --prompt "The capital of France is"
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PE = _HERE.parent.parent  # programming_examples
_DEC = _PE / "fused_decode"  # standalone fused superkernel decode example
# Tokenizer: a local Llama-3.2-1B tokenizer dir if present, else fall back to the
# HF checkpoint (base/instruct share one tokenizer) so `make run`/`profile` work
# without a hand-set Q4NX_TOKENIZER_DIR — same source the other llms/ examples use.
_LOCAL_TOKENIZER = os.environ.get(
    "Q4NX_TOKENIZER_DIR", os.path.expanduser("~/q4nx_data/tokenizer/Llama-3.2-1B")
)
_TOKENIZER = (
    _LOCAL_TOKENIZER
    if os.path.isdir(_LOCAL_TOKENIZER)
    else "meta-llama/Llama-3.2-1B-Instruct"
)
PARIS_PROMPT = [128000, 791, 6864, 315, 9822, 374]  # BOS + "The capital of France is"


def _llama3_rope(
    n_pos,
    dim,
    theta=500000.0,
    factor=32.0,
    low_freq_factor=1.0,
    high_freq_factor=4.0,
    old_ctx=8192.0,
):
    """RoPE cos/sin [n_pos, dim] for Llama-3.2 (rope_theta=5e5 + llama3 freq scaling);
    reproduces the HF rotary_emb golden (rope_cos32.f32.bin) to ~1e-6."""
    import numpy as np

    inv = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    low_wl = old_ctx / low_freq_factor
    high_wl = old_ctx / high_freq_factor
    wl = 2 * np.pi / inv
    new = inv.copy()
    for i in range(len(inv)):
        if wl[i] > low_wl:
            new[i] = inv[i] / factor
        elif wl[i] < high_wl:
            new[i] = inv[i]
        else:
            s = (old_ctx / wl[i] - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new[i] = (1 - s) * inv[i] / factor + s * inv[i]
    fr = np.arange(n_pos)[:, None] * new[None, :]
    emb = np.concatenate([fr, fr], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


_Q4NX_CACHE = os.path.expanduser("~/.cache/q4nx")


def _ensure_paris_golden():
    """Return a golden dir with embed_tokens/final_norm f32. Honors Q4NX_GOLDEN_DIR if
    it already has them; otherwise generates them from the single model.q4nx source
    (embed/norm bf16 -> f32; lm_head is tied to embed)."""
    import numpy as np

    gd = os.environ.get("Q4NX_GOLDEN_DIR")
    if gd and os.path.exists(os.path.join(gd, "weights", "embed_tokens.f32.bin")):
        return gd
    from llama32_1b_q4nx_prefill import MODEL_DEFAULT
    from llama32_1b_q4nx_weights import Q4nxModel

    out = os.path.join(_Q4NX_CACHE, "golden")
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "weights", "embed_tokens.f32.bin")):
        qm = Q4nxModel(os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT))
        embed = qm.bf16("model.embed_tokens.weight").astype(np.float32)
        embed.tofile(os.path.join(out, "weights", "embed_tokens.f32.bin"))
        qm.bf16("model.norm.weight").astype(np.float32).tofile(
            os.path.join(out, "weights", "final_norm.f32.bin")
        )
    os.environ["Q4NX_GOLDEN_DIR"] = out
    return out


def _ensure_requant_cache(fd):
    """Return the decode q4k-cascade cache path. Honors Q4NX_DECODE_WEIGHTS_NPZ if it
    exists; otherwise builds it from the single model.q4nx source (one-time ~pack)."""
    rc = os.environ.get("Q4NX_DECODE_WEIGHTS_NPZ")
    if rc and os.path.exists(rc):
        return rc
    from llama32_1b_q4nx_prefill import MODEL_DEFAULT
    import q4nx_requant

    rc = rc or os.path.join(_Q4NX_CACHE, "requant.npz")
    if not os.path.exists(rc):
        q4nx_requant.build_requant_cache(
            os.environ.get("Q4NX_MODEL_SOURCE", MODEL_DEFAULT), fd, rc
        )
    os.environ["Q4NX_DECODE_WEIGHTS_NPZ"] = rc
    return rc


# ------------------------------------------------------------------ prefill worker
def _prefill_worker(prompt, out_path, seq_len):
    """Runs in a subprocess: batched prefill -> per-layer roped-K/raw-V + first token."""
    sys.path.insert(0, str(_HERE))
    import numpy as np
    from llama32_1b_q4nx_prefill import LlamaQ4nxPrefill

    m = LlamaQ4nxPrefill(seq_len=seq_len, n_layers=16)
    m.load_weights()
    logits = np.asarray(m.prefill(prompt), np.float32)
    first = int(logits.argmax())
    K = np.stack(
        [np.asarray(m.kv_view(l)[0], np.float32) for l in range(16)]
    )  # [16,P,512]
    V = np.stack([np.asarray(m.kv_view(l)[1], np.float32) for l in range(16)])
    np.savez(out_path, k=K, v=V, first=first, prompt=np.array(prompt))
    print(
        f"[prefill] ctx={m.current_context_length} first_token={first} K{K.shape}",
        flush=True,
    )


def run_prefill(prompt, seq_len, kv_path):
    """Spawn the prefill worker (clean NPU release before decode)."""
    print(
        f"[inference] prefill (seq_len={seq_len}, prompt_len={len(prompt)})...",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_prefill-worker",
        "--_kv-path",
        str(kv_path),
        "--seq-len",
        str(seq_len),
        "--prompt-ids",
        ",".join(map(str, prompt)),
    ]
    subprocess.run(cmd, check=True)


def _compile_only(seq_len=2048):
    """Build/cache the prefill ELFs and exit (no weights, no NPU dispatch).

    This is the weight-free CI smoke signal (`make compile`). Constructing
    LlamaQ4nxPrefill compiles the prefill engines; the fused decode templates
    are a separate ~15 min build (`make compile-decode`)."""
    sys.path.insert(0, str(_HERE))
    from llama32_1b_q4nx_prefill import LlamaQ4nxPrefill

    print(
        f"[inference] compile-only: building prefill ELFs (seq_len={seq_len})...",
        flush=True,
    )
    LlamaQ4nxPrefill(seq_len=seq_len, n_layers=16)  # constructs -> compiles engines
    print("Compilation passed.", flush=True)


# ------------------------------------------------------------------ fused decoder
def _pick_decode_gen(dec_dir, max_L=None):
    """Return the decode insts generator: the compile-time decode_L<M> template
    (DecodeInstsGen, ATTN_MAXL=2048, masked-block skip) that serves every L in
    [1, 2048]. This is the only decode path (the runtime-L rt128 path was removed)."""
    sys.path.insert(0, str(dec_dir))
    from decode_insts_gen import DecodeInstsGen

    return DecodeInstsGen(str(dec_dir), max_L=max_L)


class FusedDecoder:
    """One-xclbin fused decode -- the faithful the reference analog (one MAX_L=2048 region-major KV cache;
    the per-token sequence specialized on the host by an RTP-L + KV-append-offset insts patch).

    A SINGLE decode xclbin built at ATTN_MAXL=2048 serves EVERY L in [1, 2048]: the core runs a
    compile-time 128-block attention loop and skips fully-masked far blocks (rem = L - 16*blk <= 0),
    so one build covers all L. `DecodeInstsGen` patches the L-dependent insts words per token. One
    xclbin, one BO set; the weight + KV BOs are uploaded once and the kernel appends each new
    token's K/V in place."""

    def __init__(self, max_L=None):
        HF = (
            _ensure_paris_golden()
        )  # embed/norm from model.q4nx (single source) if not provided
        import importlib.util
        import numpy as np
        from ml_dtypes import bfloat16
        import pyxrt as xrt

        sys.path.insert(0, str(_DEC))
        self.np = np
        self.bf16 = bfloat16
        self.xrt = xrt
        # Decode generator: DecodeInstsGen (decode_L<M>) -- ONE compile-time MAX_L=2048 build; the
        # kernel skips masked blocks and single-buffers the block loop, so it serves every L in
        # [1, 2048] via an RTP-L + append insts patch (the reference one-MAX_L design).
        self.gen = _pick_decode_gen(_DEC, max_L)
        self.ATTN_MAXL = self.gen.attn_maxl
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL

        # decode-module constants at DECODE_GOLDEN_L=ATTN_MAXL -- the LREG/ny geometry must
        # match the xclbin the DecodeInstsGen base was compiled at. The decode is always
        # REGION-MAJOR (the reference quadrants K03|K47|V03|V47 + fire-and-free readback, ~50 tok/s @2K);
        # seed_kv lays the seeded prefill K/V out region-major to match the decode module.
        # UNI_DEC/UNI_LM are fixed constants in fused_decode.py (Llama-3.2-1B: 16/9).
        os.environ.update(
            UNIFIED="1",
            VOCAB_CHUNK_I2="14",
            LM_HEAD="0",
            NLAYERS="1",
            DECODE_GOLDEN="1",  # boolean flag: enable post-attn-RMS decode path
            DECODE_GOLDEN_L=str(self.ATTN_MAXL),
        )
        spec = importlib.util.spec_from_file_location(
            "fu", str(_DEC / "fused_decode.py")
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
        _ensure_requant_cache(
            fd
        )  # build decode weights from model.q4nx if not provided
        self.UNI_LM = fd.UNI_LM
        self.K = fd.K
        RMS_LAYER = fd.RMS_LAYER
        self.KVSZ_TOK = fd.KVSZ_TOK
        self.DK_TOT_A = fd.DK_TOT_A
        self.NGRP = fd.NGRP
        self.REGION_W = fd.REGION_W
        self.REGION_STRIDE = fd.REGION_STRIDE
        self.VP = fd.VOCAB_SIZE_PADDED
        self.VPF = fd.VOCAB_SIZE_PADDED_FULL
        self.VOCAB_SIZE = fd.VOCAB_SIZE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.DH = 64
        self.ny = self.decode_y + self.UNI_LM * self.VP
        self._RMS_SIZE = 16 * RMS_LAYER + 64 + self.K
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK
        _kind = type(self.gen).__name__
        _mech = "compile-time 128-block loop, masked-block skip (RTP-L + append patch)"
        print(
            f"[decode] ONE xclbin ({_kind}): ATTN_MAXL={self.ATTN_MAXL}, serves "
            f"L in [1,{self.maxL}] -- {_mech}",
            flush=True,
        )

        # decode weights (q4k-cascade) + rope (all positions) + host embed
        _z = np.load(os.environ["Q4NX_DECODE_WEIGHTS_NPZ"])
        W = _z["W"].view(bfloat16)
        self.Wv16 = W.view(np.int16) if W.dtype != np.int16 else W
        RMS_in = list(_z["RMS_in"].view(bfloat16))
        RMS_post = list(_z["RMS_post"].view(bfloat16))
        self.final_norm = np.asarray(
            np.fromfile(f"{HF}/weights/final_norm.f32.bin", np.float32), bfloat16
        )
        self.embed = np.memmap(
            f"{HF}/weights/embed_tokens.f32.bin", np.float32, "r"
        ).reshape(self.VOCAB_SIZE, self.K)
        self.rope_cos, self.rope_sin = _llama3_rope(self.ATTN_MAXL, self.DH)
        self.rms_slabs = np.concatenate(
            [np.concatenate([RMS_in[k], RMS_post[k]]) for k in range(16)]
        )

        # ONE xclbin + ONE self-contained BO set (weight BO uploaded once)
        self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        xb = xrt.xclbin(self.gen.xclbin)
        self.dev.register_xclbin(xb)
        self._ctx = xrt.hw_context(self.dev, xb.get_uuid())
        self._xb = xb
        xk = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
        self.kern = xrt.kernel(self._ctx, xk.get_name())
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, 16 * self.LREG * 2, HO, g(7))
        self.ib = xrt.bo(self.dev, self.gen.base.nbytes, xrt.bo.cacheable, g(1))
        self.w_bo.write(self.Wv16, 0)
        self.w_bo.sync(TO)
        # Per-layer KV cache, flat [LREG], laid out region-major (the reference quadrants):
        # [K_g0 | K_g1 | V_g0 | V_g1], each region ATTN_MAXL*REGION_W = REGION_STRIDE.
        self.KV = np.zeros((16, self.LREG), dtype=bfloat16)

    def seed_kv(self, fk, fv, P):
        """Place the prefill K/V (fk/fv: [16,P,DK_TOT_A]) into the device KV cache in the layout
        the loaded xclbin expects, then prefetch it to the device (before the timed decode loop).
        """
        np = self.np
        RW, RS, NG = self.REGION_W, self.REGION_STRIDE, self.NGRP
        self.KV[:] = 0
        # Region-major (the reference layout): scatter each group's K (resp V) into its contiguous region.
        # fk[Lyr] is [P, DK_TOT_A] = [g0 K(RW) | g1 K(RW) | ...]; region g slot pos = g*RS+pos*RW.
        for Lyr in range(16):
            for g in range(NG):
                self.KV[Lyr, g * RS : g * RS + P * RW].reshape(P, RW)[:] = fk[
                    Lyr, :P, g * RW : (g + 1) * RW
                ].astype(self.bf16)
                self.KV[Lyr, (NG + g) * RS : (NG + g) * RS + P * RW].reshape(P, RW)[
                    :
                ] = fv[Lyr, :P, g * RW : (g + 1) * RW].astype(self.bf16)
        # Prefetch to the device HERE (before the decode loop is timed) so the host->device copy
        # is not charged to the first token; the BO stays resident for this turn's steps (the
        # kernel appends new tokens in-place). Region-major seeded slots are scattered across the
        # 4 regions, so upload the whole per-layer slab (one-time, off the timer).
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        for Lyr in range(16):
            boff = Lyr * self.LREG * 2
            self.kvc.write(self.KV[Lyr].view(np.int16), boff)
            self.kvc.sync(TO, self.LREG * 2, boff)
        self._kv_dirty = False  # already uploaded; dispatch skips it

    def dispatch(self, tok, p):
        """One decode step at L=p+1 on the single xclbin: patch insts for L (RTP-L + KV-append
        offset), pack the full KV, dispatch (16 layers + lm_head), append the new token's KV at
        slot p, return logits."""
        np = self.np
        xrt = self.xrt
        import os as _os, time as _t

        _prof = _os.environ.get("DECODE_PROF") == "1"
        _tk = _t.perf_counter
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        L = p + 1
        _a = _tk()
        # Cache the insts BO: the per-L patch touches only a handful of L-dependent words
        # (RTP-L + append/readback offsets, located by diffing two builds). Write the full
        # 780KB stream ONCE, then each token overwrite only the changed [lo:hi] range and
        # sync just that slice -- avoids re-writing/re-syncing the whole cacheable BO.
        if not hasattr(self, "_ld"):
            i1 = self.gen.insts_for_L(1)
            i2 = self.gen.insts_for_L(2)
            ld = np.where(i1 != i2)[0]
            self._ld = ld
            self._ld_lo = int(ld.min())
            self._ld_hi = int(ld.max()) + 1
            self._ld_base = i1[ld].astype(np.int64)
            self._ld_slope = i2[ld].astype(np.int64) - i1[ld].astype(np.int64)
            self._insts_buf = i1.astype(np.uint32).copy()
            self._insts_size = int(i1.size)
            self.ib.write(self._insts_buf, 0)
            self.ib.sync(TO)  # base written once
        self._insts_buf[self._ld] = (self._ld_base + (L - 1) * self._ld_slope).astype(
            np.uint32
        )
        _lo, _hi = self._ld_lo, self._ld_hi
        self.ib.write(self._insts_buf[_lo:_hi], _lo * 4)
        self.ib.sync(TO, (_hi - _lo) * 4, _lo * 4)
        insts_size = self._insts_size
        _t_insts = _tk() - _a
        _a = _tk()
        # KV cache is uploaded ONCE (seeded prefill positions) then left device-resident:
        # the decode kernel appends each new token's K/V in-place at slot L-1, so the 16x
        # 2048x KVSZ_TOK (~67MB) buffer must NOT be re-packed/re-synced every token (that
        # host copy dominated the per-token time, capping the chatbot at ~24 tok/s vs the
        # ~43 tok/s device rate). the reference likewise appends incrementally.
        if getattr(self, "_kv_dirty", True):
            packed = np.ascontiguousarray(self.KV).reshape(
                -1
            )  # [16*ATTN_MAXL*KVSZ_TOK]
            self.kvc.write(packed.view(np.int16), 0)
            self.kvc.sync(TO)
            self._kv_dirty = False
        _t_kv = _tk() - _a
        _a = _tk()
        x0 = np.asarray(self.embed[tok], self.bf16)
        # RMS BO: rms_slabs + final_norm are constant; only the 64-word rope LUT changes per
        # position. Write the full RMS stream ONCE, then patch just the LUT slice each token.
        if not hasattr(self, "_rms_lut_off"):
            self._rms_lut_off = int(
                self.rms_slabs.size
            )  # LUT sits right after rms_slabs
            _rmsbuf = np.concatenate(
                [self.rms_slabs, np.zeros(64, self.bf16), self.final_norm]
            )
            self.r_bo.write(_rmsbuf.view(np.int16), 0)
            self.r_bo.sync(TO)
        lut = np.empty(64, dtype=self.bf16)
        lut[:32] = self.rope_cos[p][:32].astype(self.bf16)
        lut[32:] = self.rope_sin[p][:32].astype(self.bf16)
        self.r_bo.write(lut.view(np.int16), self._rms_lut_off * 2)
        self.r_bo.sync(TO, 64 * 2, self._rms_lut_off * 2)
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(TO)
        # y BO: the kernel overwrites the vocab output region every dispatch, so the old
        # per-token full-y zeroing (ny int16 write + sync) is unnecessary.
        _t_io = _tk() - _a
        _a = _tk()
        st = self.kern(
            3, self.ib, insts_size, self.x_bo, self.w_bo, self.r_bo, self.y_bo, self.kvc
        ).wait(60000)
        _t_dev = _tk() - _a
        _a = _tk()
        # only the vocab logits (UNI_LM*VP at decode_y) are needed -- sync+read+convert just
        # that region, not the whole y BO.
        _voc_n = self.UNI_LM * self.VP
        self.y_bo.sync(FROM, _voc_n * 2, self.decode_y * 2)
        yv = (
            self.y_bo.read(_voc_n * 2, self.decode_y * 2)
            .view(self.bf16)
            .astype(np.float32)
        )
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        # yv is already the vocab region (read at decode_y); take the real vocab length.
        logits = yv[: self.VOCAB_SIZE]
        _t_read = _tk() - _a
        if _prof:
            print(
                f"[prof] p={p} L={L} insts={_t_insts*1e3:.1f} kv={_t_kv*1e3:.1f} "
                f"io={_t_io*1e3:.1f} DEV={_t_dev*1e3:.1f} read={_t_read*1e3:.1f} ms",
                flush=True,
            )
        return logits[: self.VOCAB_SIZE]


# ------------------------------------------------------------------ orchestration
EOS_IDS = (128001, 128009)  # <|end_of_text|>, <|eot_id|>


class Sampler:
    """Port of the reference's sampler (DECODE_DLL/common/modules/sampler.cpp): sliding-window
    repetition + frequency penalties, temperature, top-k, top-p, then sample. the reference's llm
    hosts use temperature=0.7, top_k=5, top_p=0.9 (not greedy) -- that stochasticity is
    what keeps it out of the greedy repeat-loop."""

    def __init__(
        self,
        np,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        rep_penalty=1.0,
        freq_penalty=1.0,
        rep_penalty_window=1024,
        freq_penalty_window=1024,
        seed=0,
    ):
        self.np = np
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_penalty = rep_penalty
        self.freq_penalty = freq_penalty
        self.rep_penalty_window = rep_penalty_window
        self.freq_penalty_window = freq_penalty_window
        self.rng = np.random.default_rng(seed)
        self.last_pos = {}  # token_id -> last position seen
        self.counts = {}  # token_id -> count in freq window
        self.hist = []  # freq window ring
        self.total = 0

    def sample(self, logits):
        np = self.np
        logits = logits.astype(np.float32).copy()
        # (A) repetition penalty (sign-based), sliding window
        if self.rep_penalty != 1.0 and self.rep_penalty_window > 0:
            for tid, lp in self.last_pos.items():
                if self.total - lp < self.rep_penalty_window:
                    v = logits[tid]
                    logits[tid] = (
                        v * self.rep_penalty if v < 0 else v / self.rep_penalty
                    )
        # (B) frequency penalty
        if self.freq_penalty != 1.0 and self.freq_penalty_window > 0:
            for tid, c in self.counts.items():
                logits[tid] -= (self.freq_penalty - 1.0) * (
                    c / self.freq_penalty_window
                )
        # (C) temperature
        logits = (logits - logits.max()) / self.temperature
        # (D) top-k
        k = min(self.top_k, logits.size)
        idx = np.argpartition(logits, -k)[-k:]
        idx = idx[np.argsort(logits[idx])[::-1]]
        probs = np.exp(logits[idx])
        probs /= probs.sum()
        # (E) top-p cutoff
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, self.top_p)) + 1
        idx = idx[:cut]
        probs = probs[:cut]
        probs /= probs.sum()
        # (F) sample
        tok = int(self.rng.choice(idx, p=probs))
        # (G) bookkeeping
        if self.freq_penalty_window > 0:
            self.hist.append(tok)
            self.counts[tok] = self.counts.get(tok, 0) + 1
            if len(self.hist) > self.freq_penalty_window:
                old = self.hist.pop(0)
                self.counts[old] -= 1
        self.last_pos[tok] = self.total
        self.total += 1
        return tok


def generate(
    prompt,
    n_tokens,
    seq_len,
    kv_path,
    temperature=0.7,
    top_k=5,
    top_p=0.9,
    rep_penalty=1.0,
    stop_on_eos=True,
    seed=0,
    greedy=False,
    profile=False,
):
    import numpy as np, time

    t_ttft0 = time.perf_counter()
    run_prefill(prompt, seq_len, kv_path)
    pf = np.load(kv_path)
    fk = pf["k"].astype(np.float32)
    fv = pf["v"].astype(np.float32)
    first = int(pf["first"])
    P = fk.shape[1]
    ttft = time.perf_counter() - t_ttft0
    print(f"[inference] prefill first token = {first} (Paris=12366)", flush=True)
    # Canonical driver line consumed by bench/extract_perf.py (shared with the
    # other llms/ examples). The prefill worker includes weight load + LM head.
    print(f"Time to first token (TTFT): {ttft:.2f}s", flush=True)

    # ONE decode xclbin serves L in [1, ATTN_MAXL]; the decoder picks the template that covers
    # the requested reach (rt<M> for short, compile-time L<M> up to 2048). Cap at its ATTN_MAXL.
    dec = FusedDecoder(P + n_tokens)
    attn_maxl = dec.ATTN_MAXL
    n_eff = min(n_tokens, attn_maxl - P)
    if n_eff <= 0:
        print(
            f"[inference] prompt P={P} >= decode ATTN_MAXL={attn_maxl}; "
            f"build a larger decode template; abort"
        )
        return [first]
    if n_eff < n_tokens:
        print(
            f"[inference] cap: ATTN_MAXL={attn_maxl}, P={P} -> at most {n_eff} decode tokens",
            flush=True,
        )
    dec.seed_kv(fk, fv, P)
    sampler = Sampler(
        np,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        rep_penalty=rep_penalty,
        seed=seed,
    )
    tokens = list(prompt) + [first]
    gen_ids = [first]
    import os as _os

    _dbg_tok = _os.environ.get("DECODE_VERBOSE") == "1"
    t_decode0 = time.perf_counter()
    for p in range(P, P + n_eff):
        logits = dec.dispatch(tokens[p], p)
        pred = int(logits.argmax()) if greedy else sampler.sample(logits)
        if stop_on_eos and pred in EOS_IDS:
            print(f"[inference] pos{p:2d} L={p+1} -> EOS ({pred}), stop")
            break
        gen_ids.append(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
        # per-token diagnostic line is off by default (the f-string + write lands inside the
        # decode timing); the full generated TEXT still prints after the loop. Set
        # DECODE_VERBOSE=1 to stream per-token.
        if _dbg_tok:
            print(f"[inference] pos{p:2d} L={p+1} -> {pred}")
    t_decode = time.perf_counter() - t_decode0
    n_gen = len(gen_ids) - 1  # exclude the prefill-provided first token
    if n_gen > 0:
        tps = n_gen / t_decode
        # Canonical driver lines consumed by bench/extract_perf.py: the
        # parenthesized "(Y.YY tok/s)" is the throughput the perf harness parses,
        # and "prompt_len=P" is the KV-cache depth the decode ran against.
        print(
            f"Generated {n_gen} tokens in {t_decode:.2f}s ({tps:.2f} tok/s)",
            flush=True,
        )
        print(f"Inference: prompt_len={P}, n_tokens={n_gen}", flush=True)
        if profile:
            print(
                f"[profile] decode {t_decode / n_gen * 1000:.1f} ms/token, "
                f"prefill TTFT {ttft:.2f}s, P={P}",
                flush=True,
            )
    return gen_ids


def _template_ids(tk, messages, add_generation_prompt=True):
    """apply_chat_template -> flat list[int] (robust to dict/BatchEncoding/batch-dim)."""
    ids = tk.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


class _StreamState:
    """Tracks how many characters of the running decoded text have already been emitted.
    BPE tokens can decode to '' alone but combine into characters with later tokens, so the
    safe streaming pattern is to decode the full id list each step and emit only the new
    suffix (matches the sibling llama32_1b streamer)."""

    def __init__(self):
        self.printed_len = 0


def _delta_text(tk, ids, state):
    """New text since the last call (advancing state); skips special tokens (EOS/eot)."""
    decoded = tk.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


class Session:
    """Resident the reference-faithful pipeline: prefill (LlamaQ4nxPrefill) and decode (FusedDecoder) are BOTH
    loaded once at init and kept resident (proven to coexist on the NPU), mirroring the sibling
    llms' build_session preload. Per-turn TTFT is then prefill COMPUTE only -- the ~1.8 GB weight
    load happens once at startup, not on every turn."""

    def __init__(self, seq_len=2048):
        import numpy as np, time

        sys.path.insert(0, str(_HERE))
        sys.path.insert(0, str(_DEC))
        from llama32_1b_q4nx_prefill import LlamaQ4nxPrefill

        self.np = np
        t0 = time.perf_counter()
        print("[session] loading prefill weights (once)...", flush=True)
        self.prefiller = LlamaQ4nxPrefill(seq_len=seq_len, n_layers=16)
        self.prefiller.load_weights()
        print(
            f"[session] prefill resident ({time.perf_counter() - t0:.2f}s); building decode...",
            flush=True,
        )
        self.dec = (
            FusedDecoder()
        )  # largest available decode context, resident (weights + xclbin)
        self.attn_maxl = self.dec.ATTN_MAXL
        # Warmup: one throwaway prefill so the FIRST user turn is warm (~1.07s) instead of the
        # ~1.2s cold start. run_turn clear_context()s the KV before every real turn, so this
        # leaves NO conversation state -- it only warms the resident kernels / BOs / dispatch path.
        t_w = time.perf_counter()
        self.prefiller.clear_context()
        self.prefiller.prefill(
            [128000]
        )  # 1-token dummy, padded to seq_len -> warms the full path
        self.prefiller.clear_context()
        print(
            f"[session] ready: prefill+decode resident (ATTN_MAXL={self.attn_maxl}, "
            f"weights preloaded, warmup {time.perf_counter() - t_w:.2f}s).",
            flush=True,
        )

    def run_turn(
        self,
        ids,
        n_tokens=None,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        rep_penalty=1.0,
        greedy=False,
        stop_on_eos=True,
        seed=0,
        on_token=None,
    ):
        np = self.np
        import time

        # --- prefill: compute only, weights already resident (fresh full-context prefill) ---
        t_ttft0 = time.perf_counter()
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(list(ids)), np.float32)
        first = int(logits.argmax())
        K = np.stack(
            [np.asarray(self.prefiller.kv_view(l)[0], np.float32) for l in range(16)]
        )
        V = np.stack(
            [np.asarray(self.prefiller.kv_view(l)[1], np.float32) for l in range(16)]
        )
        P = K.shape[1]
        ttft = time.perf_counter() - t_ttft0
        print(f"[inference] prefill first token = {first}", flush=True)
        print(f"Time to first token (TTFT): {ttft:.2f}s", flush=True)
        n_eff = min(n_tokens or self.attn_maxl, self.attn_maxl - P)
        if n_eff <= 0:
            print(
                f"[inference] prompt P={P} >= ATTN_MAXL={self.attn_maxl}; abort",
                flush=True,
            )
            return [first]
        # --- decode: reset KV, seed from this turn's prefill, generate ---
        self.dec.KV[:] = 0
        self.dec.seed_kv(K, V, P)
        sampler = Sampler(
            np,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            seed=seed,
        )
        tokens = list(ids) + [first]
        gen_ids = [first]
        if on_token:
            on_token(gen_ids)  # emit the prefill's first token immediately
        t_decode0 = time.perf_counter()
        for p in range(P, P + n_eff):
            lg = self.dec.dispatch(tokens[p], p)
            pred = int(lg.argmax()) if greedy else sampler.sample(lg)
            if stop_on_eos and pred in EOS_IDS:
                if not on_token:
                    print(
                        f"[inference] pos{p:2d} L={p+1} -> EOS ({pred}), stop",
                        flush=True,
                    )
                break
            gen_ids.append(pred)
            if p + 1 >= len(tokens):
                tokens.append(pred)
            if on_token:
                on_token(gen_ids)  # stream the new token
        t_decode = time.perf_counter() - t_decode0
        if on_token:
            sys.stdout.write("\n")
            sys.stdout.flush()  # end the streamed line
        n_gen = len(gen_ids) - 1
        if n_gen > 0:
            print(
                f"Generated {n_gen} tokens in {t_decode:.2f}s "
                f"({n_gen / t_decode:.2f} tok/s)",
                flush=True,
            )
            print(f"Inference: prompt_len={P}, n_tokens={n_gen}", flush=True)
        return gen_ids


def interactive_chat(
    system=None,
    seq_len=2048,
    temperature=0.7,
    top_k=5,
    top_p=0.9,
    rep_penalty=1.0,
    greedy=False,
    seed=0,
):
    """the reference-faithful multi-turn REPL (single_turn_conversation applied per turn with an
    accumulating message history): each turn re-applies the chat template to the full
    conversation, prefills it, and decodes until EOS. Prefill+decode are PRELOADED once via
    Session and reused across turns (resident, no per-turn weight reload). `/clear` resets the
    conversation (the reference clear_context), `/exit` quits. Context is capped by the decode ATTN_MAXL.
    """
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(_TOKENIZER)
    sess = Session(seq_len=seq_len)  # preload prefill + decode ONCE
    attn_maxl = sess.attn_maxl

    def _fresh():
        return [{"role": "system", "content": system}] if system else []

    messages = _fresh()
    print(
        f"[chat] the reference-faithful interactive chat (ATTN_MAXL={attn_maxl}, weights preloaded). "
        f"Commands: /clear (reset), /exit (quit).",
        flush=True,
    )
    turn = 0
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/clear":
            messages = _fresh()
            print("[chat] context cleared.", flush=True)
            continue
        messages.append({"role": "user", "content": user})
        ids = _template_ids(tk, messages)
        if len(ids) >= attn_maxl:
            print(
                f"[chat] conversation is {len(ids)} tokens >= ATTN_MAXL={attn_maxl}; "
                f"use /clear (or build a larger decode_rt<M> template).",
                flush=True,
            )
            messages.pop()
            continue
        # stream the reply token-by-token (BPE-safe delta), prefixed with "Assistant: "
        stream_state = _StreamState()
        started = [False]

        def _on_tok(gen_so_far):
            if not started[0]:
                sys.stdout.write("\nAssistant: ")
                started[0] = True
            d = _delta_text(tk, gen_so_far, stream_state)
            if d:
                sys.stdout.write(d)
                sys.stdout.flush()

        gen_ids = sess.run_turn(
            ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            greedy=greedy,
            seed=seed + turn,
            on_token=_on_tok,
        )
        answer = tk.decode(
            [g for g in gen_ids if g not in EOS_IDS], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": answer})
        turn += 1


def _detok(ids):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_TOKENIZER).decode(ids)
    except Exception as e:
        return f"(no detok: {e}) ids={ids}"


def _chat_encode(text, system=None):
    """the reference tokenize_messages equivalent: wrap the prompt as a chat turn and apply the
    model's chat template with the assistant generation prompt (see the reference
    apps/qwen35/host.cpp single_turn_conversation)."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(_TOKENIZER)
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}
    ]
    return _template_ids(tk, msgs)


def main():
    ap = argparse.ArgumentParser(
        description="llama32_1b_q4nx full inference (prefill+decode)"
    )
    ap.add_argument(
        "--prompt", type=str, default=None, help="prompt text (chat-templated)"
    )
    ap.add_argument(
        "--prompt-ids", type=str, default=None, help="comma-separated token ids (raw)"
    )
    ap.add_argument(
        "--raw", action="store_true", help="encode --prompt raw (no chat template)"
    )
    ap.add_argument("--system", type=str, default=None, help="optional system prompt")
    ap.add_argument(
        "--n-tokens", type=int, default=9, help="tokens to generate (<= ATTN_MAXL-P)"
    )
    # Canonical driver contract (shared with the other llms/ examples): the
    # Makefile run/profile/chat targets pass these through.
    ap.add_argument(
        "--model",
        type=str,
        default="instruct",
        help="model variant: 'instruct' (chat template) or 'base' (raw encode). "
        "Also selects the HF bf16 reference in verify_adapter.py.",
    )
    ap.add_argument(
        "--run-only",
        action="store_true",
        help="run inference (kernels are compiled on demand / from cache). "
        "Accepted for parity with the other examples; running is the default.",
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch). "
        "The fused decode templates are a separate build: `make compile-decode`.",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="print the decode-throughput / TTFT profiling summary (machine-readable "
        "lines consumed by bench/extract_perf.py).",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="the reference default 0.7"
    )
    ap.add_argument("--top-k", type=int, default=5, help="the reference default 5")
    ap.add_argument(
        "--top-p", type=float, default=0.9, help="the reference default 0.9"
    )
    ap.add_argument(
        "--rep-penalty",
        type=float,
        default=1.0,
        help="the reference sampler rep penalty (1.0=off)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sampler seed (the reference re-seeds by time)",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="greedy argmax instead of the reference sampling",
    )
    ap.add_argument("--no-eos-stop", action="store_true", help="do not stop at EOS")
    ap.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="prefill padded length (256/512/1024/2048)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="multi-turn chat REPL (the reference-faithful)",
    )
    ap.add_argument("--_prefill-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--_kv-path", type=str, default="/tmp/e2e_prefill.npz", help=argparse.SUPPRESS
    )
    args = ap.parse_args()

    if args._prefill_worker:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
        _prefill_worker(prompt, args._kv_path, args.seq_len)
        return

    if args.compile_only:
        _compile_only(args.seq_len)
        return

    if args.interactive:
        interactive_chat(
            system=args.system,
            seq_len=args.seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            greedy=args.greedy,
            seed=args.seed,
        )
        return

    # 'base' variant (or --raw) encodes without the chat template.
    raw_encode = args.raw or args.model == "base"
    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
    elif args.prompt:
        if raw_encode:
            from transformers import AutoTokenizer

            prompt = AutoTokenizer.from_pretrained(_TOKENIZER).encode(args.prompt)
        else:
            prompt = _chat_encode(
                args.prompt, system=args.system
            )  # the reference chat turn
    else:
        prompt = PARIS_PROMPT
    print(f"[inference] prompt = {len(prompt)} tokens", flush=True)

    gen_ids = generate(
        prompt,
        args.n_tokens,
        args.seq_len,
        args._kv_path,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        stop_on_eos=not args.no_eos_stop,
        seed=args.seed,
        greedy=args.greedy,
        profile=args.profile,
    )
    print("=" * 60)
    print(f"[inference] gen ids: {gen_ids}")
    print(f"[inference] TEXT: {_detok(gen_ids)!r}")
    if prompt == PARIS_PROMPT:
        print("*** PARIS ***" if gen_ids and gen_ids[0] == 12366 else "*** MISS ***")


if __name__ == "__main__":
    main()
