# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Qwen2.5-7B fused-decode requant cache (q4k-cascade weights .npz).
# Mirrors qwen3_8b_q4nx_requant.py; Qwen2.5-7B deltas:
#   - 28 layers, TWO RMSNorm weights/layer (input, post_attention) -- the standard
#     pre-norm pair.
#   - the LM head is UNTIED, so it gets its own vocab weight slab.
#   - the weight source is normally the bf16 HF checkpoint quantized on load (no
#     FastFlowLM Qwen2.5-7B bundle exists); open_weight_source hides which.
#   - the q/k/v projection biases are NOT packed here; they ride in the
#     per-position rope_w slab the driver rewrites each step (see
#     qwen25_7b_q4nx_inference), the same way Qwen3-8B carries its qk-norm.
import os

import numpy as np
from ml_dtypes import bfloat16


def _requant_q4k(Wm, group):
    """Per-32-column-group min/max 4-bit re-quant of a matrix [M,K] -> (q, sc, mn).

    The Q4NX affine codec: w ~= sc*q + mn, q in [0,15], one (sc, mn) per group of
    `group` elements along the reduction axis. This is the ONLY quantizer in the
    example -- the prefill weights and the decode cascade cache both go through
    it, so the two see bit-identical values."""
    M, Kc = Wm.shape
    Wg = Wm.reshape(M, Kc // group, group)
    mn = Wg.min(2)
    mx = Wg.max(2)
    sc = (mx - mn) / 15.0
    sc = np.where(sc <= 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.round((Wg - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8)
    return q.reshape(M, Kc), sc, mn.astype(np.float32)


def _dequant_q4k(q, sc, mn, group):
    """Inverse of _requant_q4k: (q[M,K] uint8, sc[M,K/g], mn[M,K/g]) -> f32 [M,K]."""
    M, Kc = q.shape
    return (
        q.reshape(M, Kc // group, group).astype(np.float32) * sc[..., None]
        + mn[..., None]
    ).reshape(M, Kc)


def quantize_dequantize_q4nx(W, group=None):
    """Round a full-precision [M,K] matrix onto the Q4NX grid.

    Deliberately quantize-then-dequantize rather than using the bf16 weights
    directly: the device computes on the 4-bit grid, so the prefill's host-side
    bf16 copy has to sit on that same grid or prefill and decode would disagree
    (and the prefill would look better than the hardware it models)."""
    from proj_qmm_pack import GROUP as _G

    g = group or _G
    q, sc, mn = _requant_q4k(np.asarray(W, np.float32), g)
    return _dequant_q4k(q, sc, mn, g)


def _interleave512(up_t, gate_t, glu_chunk):
    """Interleave up/gate in GLU_CHUNK-row halves (the decode's GLU stream order)."""
    n = up_t[0].shape[0] // glu_chunk

    def il(a, b):
        return np.concatenate(
            [
                (a if h == 0 else b)[s * glu_chunk : (s + 1) * glu_chunk]
                for s in range(n)
                for h in (0, 1)
            ]
        )

    return tuple(il(up_t[i], gate_t[i]) for i in range(3))


def build_requant_cache(model, fd, cache_path, verbose=True):
    """Re-quantize + cascade-pack the Qwen2.5-7B weights into the decode .npz.

    `fd` = the loaded fused_decode module (DECODE_MODEL=qwen2.5-7b) supplying the
    cascade geometry and phase indices."""
    from qwen25_7b_q4nx_weights import open_weight_source, D, VOCAB

    qm_model = open_weight_source(model)
    G, NCX, NCY, NPH = fd.GROUP, fd.NCX, fd.NCY, fd.NPH
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    GLU_CHUNK, W_LAYER = fd.GLU_CHUNK, fd.W_LAYER
    VP, VPF, UNI_LM = fd.VOCAB_SIZE_PADDED, fd.VOCAB_SIZE_PADDED_FULL, fd.UNI_LM
    n_layers = fd.UNI_DEC
    # Dual-MM2S weight feed: the decode splits each column's slab across the
    # column's two shim channels by cascade pair, which needs the cascade laid out
    # as [low-row half | high-row half]. Keyed off the SAME flag the decode was
    # built with (fd.W_DUAL_CHAN) so the pack can never disagree with the xclbin.
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    # The reference per-block Python packer costs ~65 s per projection (~2 h for
    # this model); the vectorized one is bit-identical and ~1000x faster.
    from q4_0_codec import pack_q4k_cascade_fast

    def _pack(q, sc, mn):
        return pack_q4k_cascade_fast(q, sc, NCX, NCY, dual_chan=DUAL, mins=mn)

    PROJ = qm_model._PROJ  # {nm: (suffix, out, in)}

    def _dq(k, nm):
        """Full-precision [out, K] for one projection, whichever source backs it."""
        return qm_model.proj_fp(k, nm)

    W_all, RMS_in, RMS_post = [], [], []
    for k in range(n_layers):
        R = {nm: _dq(k, nm) for nm in PROJ}
        qm = [None] * NPH
        qm[0] = _requant_q4k(np.concatenate([R["q"], R["k"], R["v"]], 0), G)
        qm[OP] = _requant_q4k(R["o"], G)
        qm[GP] = _interleave512(
            _requant_q4k(R["up"], G), _requant_q4k(R["gate"], G), GLU_CHUNK
        )
        qm[DP] = _requant_q4k(R["down"], G)
        W_all.append(np.concatenate([_pack(*qm[p]) for p in range(NPH)]))
        r_in, r_post = qm_model.layer_rms(k)
        RMS_in.append(np.asarray(r_in, bfloat16))
        RMS_post.append(np.asarray(r_post, bfloat16))
        assert W_all[-1].size == W_LAYER, (W_all[-1].size, W_LAYER)
        if verbose:
            print(f"[qwen2.5-7b requant] layer {k} requantized", flush=True)

    # LM head: UNTIED (Qwen2.5-7B sets tie_word_embeddings=false), padded to VPF rows.
    lm = qm_model.lm_head_fp()
    lm_pad = np.zeros((VPF, D), np.float32)
    lm_pad[:VOCAB] = lm[:VOCAB]
    lq, ls, lm_ = _requant_q4k(lm_pad, G)
    Wv = [
        _pack(
            lq[w * VP : (w + 1) * VP],
            ls[w * VP : (w + 1) * VP],
            lm_[w * VP : (w + 1) * VP],
        )
        for w in range(UNI_LM)
    ]
    W = np.concatenate([np.asarray(w) for w in W_all] + Wv)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(
        cache_path,
        W=np.asarray(W).view(np.int16),
        RMS_in=np.stack(RMS_in).view(np.int16),
        RMS_post=np.stack(RMS_post).view(np.int16),
    )
    if verbose:
        print(f"[qwen2.5-7b requant] wrote {cache_path}", flush=True)
    return cache_path
