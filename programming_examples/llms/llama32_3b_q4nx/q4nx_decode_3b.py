# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Reusable NPU2 fused-decode driver for Llama-3.2-3B Q4NX: the FLM-faithful on-device
# decode (proj -> rope -> flash-attn with on-device KV -> o -> FFN x28 -> lm_head). Loads
# the 3B fused_decode template + q4k-cascade requant weights + embed/final-norm from the
# model.q4nx bundle, and exposes dispatch(tok, pos) -> logits. One ATTN_MAXL=2048 template
# serves every context length via an RTP-L + KV-append insts patch (no CPU attention).
# Shared by the Paris bring-up and the generate loop.
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_LLMS = _HERE.parent
_PROG = _LLMS.parent
_DEC = _PROG / "fused_decode"
_1B = _LLMS / "llama32_1b_q4nx"
for _p in (str(_PROG), str(_LLMS), str(_DEC), str(_1B), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def llama3_rope(
    n_pos,
    dim,
    theta=500000.0,
    factor=32.0,
    low_freq_factor=1.0,
    high_freq_factor=4.0,
    old_ctx=8192.0,
):
    """RoPE cos/sin [n_pos, dim] for Llama-3.2 (rope_theta=5e5 + llama3 freq scaling)."""
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


# fused_decode model key + vocab chunking. VOCAB_CHUNK_I2 must satisfy
# (K/PAYLOAD) | VOCAB_I2*PAIR_ROWS -- 6 | 18 here -- or the vocab wave deadlocks;
# it also has to match the model entry's UNI_LM (see fused_decode.py).
DECODE_MODEL = "llama-3.2-3b"
VOCAB_CHUNK_I2 = "9"


def load_fd(model_type="LLAMA_3_2_3B"):
    """Load the fused_decode module for the 3B geometry (unified decode+lm_head,
    decode path enabled). `model_type` is the C++ -DMODEL_TYPE name; the Python
    generator selects the same model with DECODE_MODEL."""
    os.environ.update(
        DECODE_MODEL=DECODE_MODEL,
        VOCAB_CHUNK_I2=VOCAB_CHUNK_I2,
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L="2048",
    )
    spec = importlib.util.spec_from_file_location("fu3b", str(_DEC / "fused_decode.py"))
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def ensure_requant_cache(model_source, fd, n_layers, wcache_dir=None, verbose=True):
    """Return the q4k-cascade requant cache path, building it ONCE if missing. Keyed by
    (n_layers, VOCAB_I2) -- the lm_head pack depends on VOCAB_I2, so a stale chunk size must
    not be silently reused. A pre-versioning legacy file (VOCAB_I2-agnostic) is adopted once
    under the versioned name."""
    from q4nx_requant import build_requant_cache

    wc = Path(wcache_dir) if wcache_dir else (_HERE / ".decode_wcache")
    wc.mkdir(parents=True, exist_ok=True)
    cache = wc / f"q4nx_3b_decode_L{n_layers}_v{fd.VOCAB_I2}.npz"
    legacy = wc / f"q4nx_3b_decode_L{n_layers}.npz"
    if cache.exists():
        return str(cache)
    if legacy.exists():
        os.replace(str(legacy), str(cache))  # one-time migration; never re-requant
        return str(cache)
    if verbose:
        print(
            f"[q4nx-3b] requantizing {n_layers} decode layers + lm_head "
            f"(VOCAB_I2={fd.VOCAB_I2}); cached ONCE to {cache}",
            flush=True,
        )
    build_requant_cache(model_source, fd, str(cache), n_layers=n_layers)
    return str(cache)


EOS_IDS = (128001, 128009)  # <|end_of_text|>, <|eot_id|>


def generate(dec, prompt_ids, n_tokens, greedy=True, sampler=None, stop_on_eos=True):
    """Option (c) autoregressive generation: decode the prompt token-by-token through the
    on-device fused decode (warming the on-device KV cache), then continue. dispatch(tok, p)
    returns the logits predicting the token at position p+1. No CPU attention, no prefill.
    """
    import time

    tokens = list(prompt_ids)
    P = len(tokens)
    if P >= dec.ATTN_MAXL:
        raise RuntimeError(
            f"prompt length {P} >= decode ATTN_MAXL {dec.ATTN_MAXL}; build a larger template"
        )
    cap = min(P + n_tokens, dec.ATTN_MAXL)

    t0 = time.perf_counter()
    logits = None
    for p in range(P):  # consume the prompt; last logits predicts the first new token
        logits = dec.dispatch(tokens[p], p)
    t_prompt = time.perf_counter() - t0

    gen = []
    t1 = time.perf_counter()
    for step in range(n_tokens):
        pred = int(logits.argmax()) if greedy else sampler.sample(logits)
        gen.append(pred)
        if stop_on_eos and pred in EOS_IDS:
            break
        p = P + step
        if step == n_tokens - 1 or p >= cap:
            break
        tokens.append(pred)
        logits = dec.dispatch(pred, p)  # feed pred at position p -> logits for p+1
    t_gen = time.perf_counter() - t1
    if gen:
        print(
            f"[q4nx-3b] prompt {P} tok in {t_prompt:.2f}s; decode {len(gen)} tok in "
            f"{t_gen:.2f}s ({len(gen) / max(t_gen, 1e-9):.1f} tok/s)",
            flush=True,
        )
    return gen


class FusedDecode3B:
    """One-xclbin fused decode. dispatch(tok, pos) writes the token embedding + per-position
    rope LUT, runs the 28-layer decode + lm_head on the AIE array (appending this token's
    roped-K/raw-V into the on-device KV cache at slot `pos`), and returns the vocab logits.
    The KV cache is device-resident and grown in place -- no CPU attention, no re-upload.
    """

    def __init__(self, model_source, templates, model_type="LLAMA_3_2_3B", max_L=None):
        import pyxrt as xrt

        self.xrt = xrt
        fd = load_fd(model_type)
        self.fd = fd
        self.N_LAYERS = fd.UNI_DEC
        self.DH = fd.DH
        self.K = fd.K
        self.VOCAB_SIZE = fd.VOCAB_SIZE

        # decode weights (q4k-cascade requant) + embed/final-norm from the same model.q4nx
        from llama32_1b_q4nx_weights import Q4nxModel

        cache = ensure_requant_cache(model_source, fd, self.N_LAYERS)
        z = np.load(cache)
        W = z["W"].view(bfloat16)
        Wv16 = W.view(np.int16)
        RMS_in = list(z["RMS_in"].view(bfloat16))
        RMS_post = list(z["RMS_post"].view(bfloat16))
        qm = Q4nxModel(model_source)
        embed, final_norm, _ = qm.embed_norm_lmhead()
        self.embed = np.asarray(embed, np.float32).reshape(self.VOCAB_SIZE, self.K)
        final_norm = np.asarray(final_norm, bfloat16)

        from decode_insts_gen import DecodeInstsGen

        self.gen = DecodeInstsGen(str(templates), max_L=max_L)
        self.ATTN_MAXL = self.gen.attn_maxl
        self.rope_cos, self.rope_sin = llama3_rope(self.ATTN_MAXL, self.DH)

        RMS_LAYER = fd.RMS_LAYER
        KVSZ_TOK = fd.KVSZ_TOK
        self.VP = fd.VOCAB_SIZE_PADDED
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + fd.UNI_LM * self.VP
        self.LREG = self.ATTN_MAXL * KVSZ_TOK
        RMS_SIZE = self.N_LAYERS * RMS_LAYER + self.DH + self.K
        rms_slabs = np.concatenate(
            [np.concatenate([RMS_in[k], RMS_post[k]]) for k in range(self.N_LAYERS)]
        )

        # XRT: one xclbin + one resident BO set (weights uploaded once)
        self.dev = xrt.device(0)
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        xb = xrt.xclbin(self.gen.xclbin)
        self.dev.register_xclbin(xb)
        ctx = xrt.hw_context(self.dev, xb.get_uuid())
        xk = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
        self.kern = xrt.kernel(ctx, xk.get_name())
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.N_LAYERS * self.LREG * 2, HO, g(7))
        self.ib = xrt.bo(self.dev, self.gen.base.nbytes, xrt.bo.cacheable, g(1))

        self.w_bo.write(Wv16, 0)
        self.w_bo.sync(self.TO)
        # KV cache seeded empty; the kernel appends each token in-place at slot `pos`.
        self.reset_kv()
        # RMS BO: [rms_slabs | DH-wide LUT slot | final_norm]; LUT patched per position.
        rmsbuf = np.concatenate([rms_slabs, np.zeros(self.DH, bfloat16), final_norm])
        self.r_bo.write(rmsbuf.view(np.int16), 0)
        self.r_bo.sync(self.TO)
        self.rms_lut_off = int(rms_slabs.size)

        # insts base + the L-dependent word slope (RTP-L + KV-append offsets), from two builds.
        i1 = self.gen.insts_for_L(1)
        i2 = self.gen.insts_for_L(2)
        self.ld = np.where(i1 != i2)[0]
        self.ld_base = i1[self.ld].astype(np.int64)
        self.ld_slope = i2[self.ld].astype(np.int64) - i1[self.ld].astype(np.int64)
        self.insts_buf = i1.astype(np.uint32).copy()
        self.insts_size = int(i1.size)
        self.ib.write(self.insts_buf, 0)
        self.ib.sync(self.TO)

    def reset_kv(self):
        """Zero the device-resident KV cache (start a fresh sequence)."""
        KV = np.zeros(self.N_LAYERS * self.LREG, dtype=bfloat16)
        self.kvc.write(KV.view(np.int16), 0)
        self.kvc.sync(self.TO)

    def seed_kv(self, k_layers, v_layers):
        """Seed the device KV cache from a prefill (the warm-start handoff).

        `k_layers[L]` / `v_layers[L]` are [ctx, n_kv_heads*DH] roped-K / raw-V as
        produced by llama32_3b_q4nx_prefill (head-major rows). The device cache is
        REGION-major: per layer, NGRP K regions then NGRP V regions, each
        ATTN_MAXL*REGION_W, with position p at p*REGION_W. Within a region the
        heads served by that column group sit contiguously, so a head-major row
        splits into NGRP contiguous REGION_W slices -- region gi takes columns
        [gi*REGION_W, (gi+1)*REGION_W). Returns the seeded context length.
        """
        fd = self.fd
        ctx = int(np.asarray(k_layers[0]).shape[0])
        if ctx > self.ATTN_MAXL:
            raise RuntimeError(
                f"prefill ctx {ctx} exceeds decode ATTN_MAXL {self.ATTN_MAXL}"
            )
        region_stride = self.ATTN_MAXL * fd.REGION_W
        buf = np.zeros(self.N_LAYERS * self.LREG, dtype=bfloat16)
        for L in range(self.N_LAYERS):
            kL = np.asarray(k_layers[L], bfloat16).reshape(ctx, -1)
            vL = np.asarray(v_layers[L], bfloat16).reshape(ctx, -1)
            base = L * self.LREG
            for gi in range(fd.NGRP):
                cols = slice(gi * fd.REGION_W, (gi + 1) * fd.REGION_W)
                kreg = base + gi * region_stride
                vreg = base + (fd.NGRP + gi) * region_stride
                buf[kreg : kreg + ctx * fd.REGION_W] = kL[:, cols].reshape(-1)
                buf[vreg : vreg + ctx * fd.REGION_W] = vL[:, cols].reshape(-1)
        self.kvc.write(buf.view(np.int16), 0)
        self.kvc.sync(self.TO)
        return ctx

    def dispatch(self, tok, p):
        """One decode step at context length L=p+1: patch insts for L, feed embed[tok] +
        the position-p rope LUT, run the fused decode, return the vocab logits."""
        L = p + 1
        self.insts_buf[self.ld] = (self.ld_base + (L - 1) * self.ld_slope).astype(
            np.uint32
        )
        self.ib.write(self.insts_buf, 0)
        self.ib.sync(self.TO)
        x0 = np.asarray(self.embed[tok], bfloat16)
        h = self.DH // 2
        lut = np.empty(self.DH, dtype=bfloat16)
        lut[:h] = self.rope_cos[p][:h].astype(bfloat16)
        lut[h:] = self.rope_sin[p][:h].astype(bfloat16)
        self.r_bo.write(lut.view(np.int16), self.rms_lut_off * 2)
        self.r_bo.sync(self.TO, self.DH * 2, self.rms_lut_off * 2)
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(self.TO)
        st = self.kern(
            3,
            self.ib,
            self.insts_size,
            self.x_bo,
            self.w_bo,
            self.r_bo,
            self.y_bo,
            self.kvc,
        ).wait(60000)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        voc_n = self.fd.UNI_LM * self.VP
        self.y_bo.sync(self.FROM, voc_n * 2, self.decode_y * 2)
        # Zero-copy view into the BO (the shared infra's readback idiom, same
        # as the 1B sibling): bo.read() returns a buffer whose stride metadata
        # is pyxrt-build dependent, and .view() on it raises on some runners.
        yv = np.frombuffer(
            self.y_bo.map(), dtype=bfloat16, count=voc_n, offset=self.decode_y * 2
        ).astype(np.float32)
        return yv[: self.VOCAB_SIZE]
