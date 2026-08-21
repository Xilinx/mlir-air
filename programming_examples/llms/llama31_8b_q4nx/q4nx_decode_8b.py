# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Reusable NPU2 fused-decode driver for Llama-3.1-8B Q4NX: the FLM-faithful on-device
# decode (proj -> rope -> flash-attn with on-device KV -> o -> FFN x32 -> lm_head). Loads
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


import decode_staircase as stair  # noqa: E402  (needs the sys.path above)


def llama3_rope(
    n_pos,
    dim,
    theta=500000.0,
    factor=8.0,
    low_freq_factor=1.0,
    high_freq_factor=4.0,
    old_ctx=8192.0,
):
    """RoPE cos/sin [n_pos, dim] for Llama-3.1 (rope_theta=5e5 + llama3 freq scaling).

    factor=8.0 is Llama-3.1's (config rope_scaling); Llama-3.2 uses 32.0. This is
    also the curve behind FastFlowLM's `llama_3b_8b_rope` table -- factor 8.0
    reproduces it to 4e-05 (the header's print precision), 32.0 is off by 75%."""
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
# (K/PAYLOAD) | VOCAB_I2*PAIR_ROWS -- 8 | 32 here -- or the vocab wave deadlocks;
# it also has to match the model entry's UNI_LM (see fused_decode.py).
DECODE_MODEL = "llama-3.1-8b"
VOCAB_CHUNK_I2 = "16"
# Decode layers per weight BO. A shim BD's byte offset is a uint32, so ONE buffer
# is only addressable over 4 GiB; the 32 layers + lm-head are 4.375 GiB here and
# wrapped (every logit came back NaN). 8 gives four 0.9 GiB groups, comfortably
# under the line, plus the lm-head on its own. Must match what the templates were
# built with (the Makefile's DECODE_WGROUP).
DECODE_WGROUP = 8
# Core stack. At K=4096 the rms core's seven K-wide L1 activation buffers leave
# under 8 KiB, so the 10240 default does not fit and buffer allocation fails at
# build time. 8064 keeps a measured >2x margin over the deepest decode frame
# (2112 B, proj_qmm_pass256 / attn_kv_fin). Near-exact fit: another K-sized
# buffer on the rms core needs a real change here, not another trim.
DECODE_STACK = 8064


def load_fd(model_type="LLAMA_3_1_8B"):
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
        DECODE_WGROUP=str(DECODE_WGROUP),
        DECODE_STACK=str(DECODE_STACK),
    )
    spec = importlib.util.spec_from_file_location("fu8b", str(_DEC / "fused_decode.py"))
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    return fd


def ensure_requant_cache(model_source, fd, n_layers, wcache_dir=None, verbose=True):
    """Return the q4k-cascade requant cache path, building it ONCE if missing. Keyed by
    (n_layers, VOCAB_I2, bundle fingerprint): the lm_head pack depends on VOCAB_I2, so a
    stale chunk size must not be silently reused, and the fingerprint ties the cache to the
    exact model.q4nx it was requantized from. Without it a warm cache survives a Hub
    re-export -- the machine keeps the old weights while a fresh one builds from the new
    bundle, same code, different numbers.

    Legacy unfingerprinted files are NOT adopted: they cannot be attributed to a bundle, and
    the pre-2026-08-04 Q4NX-encoded weights they were built from reconstruct measurably worse
    than the current ones. They are left in place, and the new cache is built alongside.
    """
    from q4nx_requant import build_requant_cache
    from llama32_1b_q4nx_weights import Q4nxModel

    wc = Path(wcache_dir) if wcache_dir else (_HERE / ".decode_wcache")
    wc.mkdir(parents=True, exist_ok=True)
    fp = Q4nxModel(model_source).fingerprint()
    # W_DUAL_CHAN reorders the cascade (the two-MM2S weight feed splits it by
    # cascade pair into [low-row half | high-row half]), so it is part of the key:
    # a warm single-channel cache would feed the dual-channel xclbin the wrong
    # blocks.
    w2 = "_w2ch" if getattr(fd, "W_DUAL_CHAN", 0) else ""
    cache = wc / f"q4nx_8b_decode_L{n_layers}_v{fd.VOCAB_I2}{w2}_{fp}.npz"
    if cache.exists():
        return str(cache)
    if verbose:
        print(
            f"[q4nx-8b] requantizing {n_layers} decode layers + lm_head "
            f"(VOCAB_I2={fd.VOCAB_I2}); cached ONCE to {cache}",
            flush=True,
        )
    # tie_lm_head=False: Llama-3.1-8B sets tie_word_embeddings=false and the bundle
    # ships a real quantized lm_head. Defaulting to the tied path here would pack the
    # EMBEDDING as the LM head -- a silently wrong model, not a build error.
    build_requant_cache(
        model_source, fd, str(cache), n_layers=n_layers, tie_lm_head=False
    )
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
            f"[q4nx-8b] prompt {P} tok in {t_prompt:.2f}s; decode {len(gen)} tok in "
            f"{t_gen:.2f}s ({len(gen) / max(t_gen, 1e-9):.1f} tok/s)",
            flush=True,
        )
    return gen


class FusedDecode8B:
    """One-xclbin fused decode. dispatch(tok, pos) writes the token embedding + per-position
    rope LUT, runs the 32-layer decode + lm_head on the AIE array (appending this token's
    roped-K/raw-V into the on-device KV cache at slot `pos`), and returns the vocab logits.
    The KV cache is device-resident and grown in place -- no CPU attention, no re-upload.
    """

    def __init__(
        self,
        model_source,
        templates,
        model_type="LLAMA_3_1_8B",
        max_L=None,
        staircase=False,
    ):
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
        # Only the input embedding + final norm are read here; the LM head lives in
        # the requant cache above (untied, from the bundle's own lm_head tensor), so
        # embed_norm_lmhead()'s tied third return is deliberately not used.
        embed = qm.bf16("model.embed_tokens.weight")
        final_norm = qm.bf16("model.norm.weight")
        # Kept in bf16, not promoted to f32 as the 1B/3B-lineage drivers do: the
        # only use is a per-token row lookup that is cast back to bf16 for the
        # device input, so f32 would just double a 1 GiB table for nothing. The
        # qwen3_8b sibling does the same.
        self.embed = np.asarray(embed, bfloat16).reshape(self.VOCAB_SIZE, self.K)
        final_norm = np.asarray(final_norm, bfloat16)

        import decode_dynseq as _dyn
        from decode_dynseq import pick_insts_gen

        self._dyn = _dyn

        self.gen = pick_insts_gen(str(templates), max_L=max_L)
        # Staircase: hold every calibrated ATTN_MAXL window and run on the smallest one
        # covering the current L. The compiled KV readback streams ATTN_MAXL positions
        # whatever L is, so a smaller window moves proportionally fewer DDR bytes/token.
        # Off by default -- one window, byte-identical to the single-template path.
        self.windows = stair.resolve_windows(self.gen, staircase)
        # BOs and the rope table are sized for the largest window and shared by all.
        self.ATTN_MAXL = max(self.windows)
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

        # XRT: one kernel per window + ONE resident BO set (weights uploaded once).
        # Host-only BOs are host memory registered with the device, not bound to a
        # hw_context slot, so the same set is valid on every window's kernel -- which is
        # what makes a window switch cheap (no 2 GB weight re-upload).
        self.dev = xrt.device(0)
        self.TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self._kern = stair.open_windows(self.dev, xrt, self.gen, self.windows)
        self.cur_maxl = self.ATTN_MAXL
        self.kern = self._kern[self.cur_maxl][1]
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.r_bo = xrt.bo(self.dev, RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.N_LAYERS * self.LREG * 2, HO, g(7))
        # DECODE_WGROUP: the engine splits the weights over ceil(UNI_DEC/G) layer
        # buffers plus a dedicated lm-head buffer, because one BO is addressable
        # only over a 4 GiB span. Group 0 stays on the original arg (g(4)); the
        # rest are appended after kvc, so every pre-existing binding is unchanged.
        # Read off the engine, not the environment, so the host slicing cannot
        # disagree with what the template was built for.
        _G, self._wsplit, _ng = fd.W_GROUP, fd.W_SPLIT, fd.N_WGRP
        if self._wsplit:
            _WL = fd.W_LAYER
            parts = [
                W[gi * _G * _WL : min((gi + 1) * _G, self.N_LAYERS) * _WL]
                for gi in range(_ng)
            ]
            parts.append(W[self.N_LAYERS * _WL :])  # lm-head slabs
            assert sum(p.size for p in parts) == W.size, "weight split lost data"
            self.w_bos = [
                xrt.bo(self.dev, p.size * 2, HO, g(4 if i == 0 else 7 + i))
                for i, p in enumerate(parts)
            ]
            for bo, p in zip(self.w_bos, parts):
                bo.write(np.ascontiguousarray(p).view(np.int16), 0)
                bo.sync(self.TO)
            print(
                f"[decode] weight split G={_G}: "
                + ", ".join(f"{p.size*2/2**30:.2f}GiB" for p in parts),
                flush=True,
            )
            self.w_bo = self.w_bos[0]
        else:
            self.w_bos = None
            self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
            self.w_bo.write(Wv16, 0)
            self.w_bo.sync(self.TO)
        # KV cache seeded empty; the kernel appends each token in-place at slot `pos`.
        self.reset_kv()
        # RMS BO: [rms_slabs | DH-wide LUT slot | final_norm]; LUT patched per position.
        rmsbuf = np.concatenate([rms_slabs, np.zeros(self.DH, bfloat16), final_norm])
        self.r_bo.write(rmsbuf.view(np.int16), 0)
        self.r_bo.sync(self.TO)
        self.rms_lut_off = int(rms_slabs.size)

        # Per window: insts base + the L-dependent word slope (RTP-L + KV-append
        # offsets, from that window's two builds) and its own instruction BO.
        self._ist = stair.make_insts_states(self.gen, xrt, self.dev, g(1), self.windows)
        self._geom = stair.KVGeometry(
            self.N_LAYERS, KVSZ_TOK, fd.REGION_W, fd.NGRP, self.LREG
        )
        self._use_window(self.cur_maxl)

    def _use_window(self, m):
        """Point the active kernel / insts state at window `m` (no KV movement)."""
        self._st = self._ist[m]
        self.cur_maxl = m
        self.kern = self._kern[m][1]
        self.ib = self._st["ib"]

    def _respace_kv(self, old_maxl, new_maxl, live):
        """Move the `live` filled positions into window `new_maxl`'s KV layout."""
        stair.respace_kv(self.kvc, self._geom, old_maxl, new_maxl, live, self.xrt)

    def reset_kv(self):
        """Zero the device-resident KV cache (start a fresh sequence)."""
        KV = np.zeros(self.N_LAYERS * self.LREG, dtype=bfloat16)
        self.kvc.write(KV.view(np.int16), 0)
        self.kvc.sync(self.TO)
        # A fresh sequence starts at L=1, so drop back to the cheapest window.
        if getattr(self, "_ist", None):
            self._use_window(self.windows[0])

    def seed_kv(self, k_layers, v_layers):
        """Seed the device KV cache from a prefill (the warm-start handoff).

        `k_layers[L]` / `v_layers[L]` are [ctx, n_kv_heads*DH] roped-K / raw-V as
        produced by llama31_8b_q4nx_prefill (head-major rows). The device cache is
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
        # Seed straight into the layout of the window the first decode step will use
        # (L = ctx+1), so no re-space is needed on the very first dispatch.
        if len(self.windows) > 1:
            self._use_window(self.gen.window_for_L(ctx + 1))
        region_stride = self.cur_maxl * fd.REGION_W
        lreg = self._geom.lreg(self.cur_maxl)
        buf = np.zeros(self.N_LAYERS * self.LREG, dtype=bfloat16)
        for L in range(self.N_LAYERS):
            kL = np.asarray(k_layers[L], bfloat16).reshape(ctx, -1)
            vL = np.asarray(v_layers[L], bfloat16).reshape(ctx, -1)
            base = L * lreg
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
        if len(self.windows) > 1:
            w = self.gen.window_for_L(L)
            if w != self.cur_maxl:
                # `p` positions are live; this token's K/V is appended by the dispatch.
                self._respace_kv(self.cur_maxl, w, p)
                self._use_window(w)
        self.insts_size = stair.patch_insts(self._st, L, self.xrt, self.TO)
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
            *(self.w_bos[1:] if self._wsplit else ()),
            *self._dyn.dispatch_args(self.gen, p + 1),
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
