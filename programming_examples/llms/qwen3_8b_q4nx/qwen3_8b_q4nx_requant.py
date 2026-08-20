# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Qwen3-8B fused-decode requant cache (q4k-cascade weights .npz) from the
# model.q4nx bundle. Mirrors llama32_1b_q4nx/q4nx_requant.py; Qwen3-8B deltas:
#   - 36 layers, TWO RMSNorm weights/layer (input, post_attention) -- the standard
#     pre-norm pair, not Gemma's 4-norm sandwich.
#   - the LM head is UNTIED: use the bundle's separate Q4NX lm_head.weight, like
#     gemma3_4b_q4nx does, not llama's tied embed matrix.
#   - per-head qk-norm weights are NOT packed here; they ride in the per-position
#     rope_w slab the driver rewrites each step (see qwen3_8b_q4nx_inference).
import os

import numpy as np
from ml_dtypes import bfloat16


def _requant_q4k(Wm, group):
    """Per-32-column-group min/max 4-bit re-quant of a dequantized matrix [M,K]."""
    M, Kc = Wm.shape
    Wg = Wm.reshape(M, Kc // group, group)
    mn = Wg.min(2)
    mx = Wg.max(2)
    sc = (mx - mn) / 15.0
    sc = np.where(sc <= 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.round((Wg - mn[..., None]) / sc[..., None]), 0, 15).astype(np.uint8)
    return q.reshape(M, Kc), sc, mn.astype(np.float32)


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
    """Re-quantize + cascade-pack the Qwen3-8B model.q4nx weights into the decode .npz.

    `fd` = the loaded fused_decode module (DECODE_MODEL=qwen3-8b) supplying the
    cascade geometry + pack_q4k_cascade + phase indices."""
    from qwen3_8b_q4nx_weights import Q4nxModel, D, VOCAB

    qm_model = Q4nxModel(model)
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
    PROJ = qm_model._PROJ  # {nm: (suffix, out, in)}

    def _dq(k, nm):
        t, M, Kc = PROJ[nm]
        return qm_model.dequant(f"model.layers.{k}.{t}.weight", M, Kc)

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
        W_all.append(
            np.concatenate(
                [
                    fd.pack_q4k_cascade(
                        *qm[p], NCX, NCY, iter_major=True, dual_chan=DUAL
                    )
                    for p in range(NPH)
                ]
            )
        )
        r_in, r_post = qm_model.layer_rms(k)
        RMS_in.append(np.asarray(r_in, bfloat16))
        RMS_post.append(np.asarray(r_post, bfloat16))
        assert W_all[-1].size == W_LAYER, (W_all[-1].size, W_LAYER)
        if verbose:
            print(f"[qwen3-8b requant] layer {k} requantized", flush=True)

    # LM head: UNTIED -- the bundle's own Q4NX lm_head.weight, padded to VPF rows.
    lm = qm_model.dequant("lm_head.weight", VOCAB, D)
    lm_pad = np.zeros((VPF, D), np.float32)
    lm_pad[:VOCAB] = lm[:VOCAB]
    lq, ls, lm_ = _requant_q4k(lm_pad, G)
    Wv = [
        fd.pack_q4k_cascade(
            lq[w * VP : (w + 1) * VP],
            ls[w * VP : (w + 1) * VP],
            lm_[w * VP : (w + 1) * VP],
            NCX,
            NCY,
            iter_major=True,
            dual_chan=DUAL,
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
        print(f"[qwen3-8b requant] wrote {cache_path}", flush=True)
    return cache_path
