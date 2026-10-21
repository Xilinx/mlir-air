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
import mmap
import tempfile
import zipfile

import numpy as np
from ml_dtypes import bfloat16


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


def _cat(*mats):
    """Stack (q, scale, min) matrices by rows."""
    return tuple(np.concatenate(p, 0) for p in zip(*mats))


def _pad_rows(t, n):
    """Zero-pad a (q, scale, min) matrix to n rows."""
    return tuple(
        np.concatenate([a, np.zeros((n - a.shape[0],) + a.shape[1:], a.dtype)])
        for a in t
    )


def build_requant_cache(model, fd, cache_path, verbose=True):
    """Cascade-pack the Qwen3-8B model.q4nx weights into the decode .npz.

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

    def _q4(k, nm):
        t, M, Kc = PROJ[nm]
        return qm_model.q4nx(f"model.layers.{k}.{t}.weight", M, Kc)

    # Stream the same NPY members as np.savez without retaining and concatenating
    # the whole model's packed arrays. Publish only a complete ZIP64 archive.
    RMS_in, RMS_post = [], []
    total_words = n_layers * W_LAYER + UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16
    directory = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as staging:
        temporary_path = os.path.join(staging, "weights.npz")
        with open(temporary_path, "w+b") as tmp, zipfile.ZipFile(
            tmp, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            with archive.open("W.npy", "w", force_zip64=True) as output:
                np.lib.format.write_array_header_1_0(
                    output,
                    dict(
                        descr=np.dtype(np.int16).str,
                        fortran_order=False,
                        shape=(total_words,),
                    ),
                )
                words = 0

                def write_weights(packed):
                    nonlocal words
                    data = np.ascontiguousarray(packed).view(np.int16)
                    output.write(memoryview(data).cast("B"))
                    words += data.size

                for k in range(n_layers):
                    R = {nm: _q4(k, nm) for nm in PROJ}
                    qm = [None] * NPH
                    qm[0] = _cat(R["q"], R["k"], R["v"])
                    qm[OP] = R["o"]
                    qm[GP] = _interleave512(R["up"], R["gate"], GLU_CHUNK)
                    qm[DP] = R["down"]
                    start = words
                    for p in range(NPH):
                        write_weights(
                            fd.pack_q4k_cascade(
                                *qm[p], NCX, NCY, iter_major=True, dual_chan=DUAL
                            )
                        )
                    assert words - start == W_LAYER, (words - start, W_LAYER)
                    r_in, r_post = qm_model.layer_rms(k)
                    RMS_in.append(np.asarray(r_in, bfloat16))
                    RMS_post.append(np.asarray(r_post, bfloat16))
                    del qm, R
                    # All source views for this layer are dead; release resident
                    # file pages as well as Python arrays on memory-limited hosts.
                    qm_model._mm._mmap.madvise(mmap.MADV_DONTNEED)
                    tmp.flush()
                    os.posix_fadvise(tmp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    if verbose:
                        print(f"[qwen3-8b requant] layer {k} packed", flush=True)

                # UNTIED LM head, padded to VPF rows, in the original wave order.
                lq, ls, lm_ = _pad_rows(qm_model.q4nx("lm_head.weight", VOCAB, D), VPF)
                for w in range(UNI_LM):
                    write_weights(
                        fd.pack_q4k_cascade(
                            lq[w * VP : (w + 1) * VP],
                            ls[w * VP : (w + 1) * VP],
                            lm_[w * VP : (w + 1) * VP],
                            NCX,
                            NCY,
                            iter_major=True,
                            dual_chan=DUAL,
                        )
                    )
                assert words == total_words, (words, total_words)
            for name, values in (("RMS_in", RMS_in), ("RMS_post", RMS_post)):
                with archive.open(name + ".npy", "w") as output:
                    np.lib.format.write_array(output, np.stack(values).view(np.int16))
        os.replace(temporary_path, cache_path)
    if verbose:
        print(f"[qwen3-8b requant] wrote {cache_path}", flush=True)
    return cache_path
