# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Qwen2.5-3B fused-decode weight cache (q4k-cascade .npz).
#
# Unlike the llama / Qwen2.5-7B siblings this does NOT re-quantize: FastFlowLM's
# bundle is already Q4_0 in the same 32x256 block geometry the cascade wants, so
# the codes and scales are carried through untouched and only re-ordered into the
# device's stream order. A dequant/requant round trip would quantize twice and
# lose accuracy for nothing. (An HF checkpoint source has no codes to carry, so
# it is quantized once, on load, by the weight module.)
#
# Weight stream per layer = 4 phases, each cascade-packed iteration-major:
#   ph0 QKV      [q(2048) | k(256) | v(256)]  x K=2048
#   ph1 o        [2048]                       x K=2048
#   ph2 gate/up  up|gate interleaved in PAYLOAD-row chunks
#   ph3 down     [2048]                       x K=11264
# The GLU axis is padded 11008 -> 11264 in the bundle itself (exact zeros).
import os

import numpy as np
from ml_dtypes import bfloat16


def _interleave_chunks(a, b, chunk):
    """[a0|b0|a1|b1|...] in `chunk`-row slices (the GLU stream order)."""
    n = a.shape[0] // chunk
    return np.concatenate(
        [
            (a if h == 0 else b)[s * chunk : (s + 1) * chunk]
            for s in range(n)
            for h in (0, 1)
        ]
    )


def _interleave_q4_0(up, gate, chunk):
    """Interleave two (q, scale) pairs row-chunk-wise."""
    return tuple(_interleave_chunks(up[i], gate[i], chunk) for i in range(2))


def build_requant_cache(model, fd, cache_path, verbose=True):
    """Cascade-pack the Qwen2.5-3B Q4_0 weights into the decode .npz.

    `fd` = the loaded fused_decode module (DECODE_MODEL=qwen2.5-3b) supplying the
    cascade geometry and phase indices."""
    from qwen25_3b_q4_weights import open_weight_source, D, VOCAB

    qm_model = open_weight_source(model)
    NCX, NCY, NPH = fd.NCX, fd.NCY, fd.NPH
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
    # this model); the vectorized one is bit-identical and ~1000x faster. `mins`
    # is omitted: Q4_0 has none, and the packer writes the zeros the block layout
    # still reserves for them.
    from q4_0_codec import pack_q4k_cascade_fast

    def _pack(q, sc):
        return pack_q4k_cascade_fast(q, sc, NCX, NCY, dual_chan=DUAL)

    W_all, RMS_in, RMS_post = [], [], []
    for k in range(n_layers):
        R = {nm: qm_model.proj_raw(k, nm) for nm in qm_model._PROJ}
        ph = [None] * NPH
        ph[0] = tuple(
            np.concatenate([R["q"][i], R["k"][i], R["v"][i]], 0) for i in range(2)
        )
        ph[OP] = R["o"]
        ph[GP] = _interleave_q4_0(R["up"], R["gate"], GLU_CHUNK)
        ph[DP] = R["down"]
        W_all.append(np.concatenate([_pack(*ph[p]) for p in range(NPH)]))
        r_in, r_post = qm_model.layer_rms(k)
        RMS_in.append(np.asarray(r_in, bfloat16))
        RMS_post.append(np.asarray(r_post, bfloat16))
        assert W_all[-1].size == W_LAYER, (W_all[-1].size, W_LAYER)
        if verbose:
            print(f"[qwen2.5-3b pack] layer {k} packed", flush=True)

    # LM head, padded to VPF rows. The pad rows are zero weights: code 0 with a
    # scale of 1 (a 0 scale would be an undequantizable block).
    lq, ls = qm_model.lm_head_raw()
    q_pad = np.zeros((VPF, D), np.uint8)
    s_pad = np.ones((VPF, ls.shape[1]), np.float32)
    q_pad[:VOCAB] = lq[:VOCAB]
    s_pad[:VOCAB] = ls[:VOCAB]
    Wv = [
        _pack(q_pad[w * VP : (w + 1) * VP], s_pad[w * VP : (w + 1) * VP])
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
        print(f"[qwen2.5-3b pack] wrote {cache_path}", flush=True)
    return cache_path
