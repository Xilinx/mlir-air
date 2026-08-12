# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the fused-decode requant cache (q4k-cascade weights .npz) from the HF
# model.q4nx bundle. The decode kernels consume a re-quantized, cascade-packed
# weight stream (per-block min/max q4 + pack_q4k_cascade) that differs from the
# prefill's GEMM input-B layout; this reproduces it from the same model.q4nx the
# prefill loads, so nothing external is needed. Cached to an .npz (keys W /
# RMS_in / RMS_post) since the pack is deterministic + xclbin-independent.
import os
import numpy as np
from ml_dtypes import bfloat16

_PROJ = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "up": "mlp.up_proj",
    "gate": "mlp.gate_proj",
    "down": "mlp.down_proj",
}


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


def build_requant_cache(model, fd, cache_path, n_layers=16, verbose=True):
    """Re-quantize + cascade-pack the model.q4nx weights into the decode's .npz.

    `model` = HF repo id / local dir/file for model.q4nx; `fd` = the loaded
    fused_decode module (supplies the cascade geometry + pack_q4k_cascade)."""
    from llama32_1b_q4nx_weights import Q4nxModel

    qm_model = Q4nxModel(model)
    G, NCX, NCY, NPH, K = fd.GROUP, fd.NCX, fd.NCY, fd.NPH, fd.K
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    GLU_CHUNK, W_LAYER = fd.GLU_CHUNK, fd.W_LAYER
    # Dual-MM2S weight feed: the decode splits each column's slab across the
    # column's two shim channels by cascade pair, which needs the cascade laid out
    # as [low-row half | high-row half]. Keyed off the SAME flag the decode was
    # built with (fd.W_DUAL_CHAN) so the pack can never disagree with the xclbin.
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    VP, VPF, VOCAB, UNI_LM = (
        fd.VOCAB_SIZE_PADDED,
        fd.VOCAB_SIZE_PADDED_FULL,
        fd.VOCAB_SIZE,
        fd.UNI_LM,
    )

    # Logical (out, K) per projection. An I8-packed bundle header carries only
    # the block count, so the caller has to supply this (Q4nxModel docstring).
    # Take it from `fd`, not from Q4nxModel's 1B default table: this function
    # requantizes whichever model `fd` was configured for, and a mismatched
    # table would reshape a 3B tensor as a 1B one.
    dims = {
        "q": (fd.DQ, fd.K),
        "k": (fd.DK, fd.K),
        "v": (fd.DV, fd.K),
        "o": (fd.K, fd.DQ),
        "up": (fd.GLU_OUT, fd.K),
        "gate": (fd.GLU_OUT, fd.K),
        "down": (fd.K, fd.GLU_OUT),
    }

    W_all, RMS_in, RMS_post = [], [], []
    for k in range(n_layers):
        R = {
            nm: qm_model.dequant(f"model.layers.{k}.{t}.weight", *dims[nm])
            for nm, t in _PROJ.items()
        }
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
        rin, rpo = qm_model.layer_rms(k)
        RMS_in.append(np.asarray(rin, bfloat16))
        RMS_post.append(np.asarray(rpo, bfloat16))
        assert W_all[-1].size == W_LAYER, (W_all[-1].size, W_LAYER)
        if verbose:
            print(f"[q4nx_requant] layer {k} requantized", flush=True)
    # lm_head is tied to embed (config tie_word_embeddings=true): use the
    # full-precision embed matrix, not the bundle's separate Q4NX lm_head tensor.
    lm = qm_model.bf16("model.embed_tokens.weight")
    lm_pad = np.zeros((VPF, K), np.float32)
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
        print(f"[q4nx_requant] wrote {cache_path}", flush=True)
    return cache_path
