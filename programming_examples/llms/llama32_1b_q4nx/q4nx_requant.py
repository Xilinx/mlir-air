# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the fused-decode weight cache (q4k-cascade weights .npz) from the HF
# model.q4nx bundle. The decode kernels consume a cascade-packed weight stream
# (pack_q4k_cascade) that differs from the prefill's GEMM input-B layout; this
# reproduces it from the same model.q4nx the prefill loads, so nothing external
# is needed. Bundle matrices keep their stored q4 scale/min; only a tied LM head,
# which is bf16, is quantized here. Cached to an .npz (keys W / RMS_in /
# RMS_post) since the pack is deterministic + xclbin-independent.
import os
import mmap
import tempfile
import zipfile
import numpy as np
from ml_dtypes import bfloat16

import sys  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

# llms/shared is imported as the air_examples package, not through sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]
from air_examples.llms.shared.infra import host_weights  # noqa: E402

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
    return host_weights.requant_q4k(Wm, group)


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


def build_requant_cache(
    model, fd, cache_path, n_layers=16, verbose=True, tie_lm_head=True
):
    """Cascade-pack the model.q4nx weights into the decode's .npz.

    `model` = HF repo id / local dir/file for model.q4nx; `fd` = the loaded
    fused_decode module (supplies the cascade geometry + pack_q4k_cascade).

    `tie_lm_head` follows the model's config tie_word_embeddings. It is TRUE for
    Llama-3.2-1B/3B and Gemma3, whose LM head IS the embedding matrix, and FALSE
    for Llama-3.1-8B, which ships a real quantized lm_head -- there, taking the
    embedding would be a silently wrong LM head rather than a build error."""
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
    # packs whichever model `fd` was configured for, and a mismatched
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

    # Keep the same NPY members and word order as np.savez, but write each
    # projection immediately. Retaining all layers and their concatenation
    # needs two full packed-model copies during cold 8B cache preparation.
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
                    R = {
                        nm: qm_model.q4nx(f"model.layers.{k}.{t}.weight", *dims[nm])
                        for nm, t in _PROJ.items()
                    }
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
                    rin, rpo = qm_model.layer_rms(k)
                    RMS_in.append(np.asarray(rin, bfloat16))
                    RMS_post.append(np.asarray(rpo, bfloat16))
                    del qm, R
                    qm_model._mm._mmap.madvise(mmap.MADV_DONTNEED)
                    tmp.flush()
                    os.posix_fadvise(tmp.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                    if verbose:
                        print(f"[q4nx_requant] layer {k} packed", flush=True)

                if tie_lm_head:
                    # Tied: quantize the BF16 embedding, as in the old packer.
                    lm_pad = np.zeros((VPF, K), np.float32)
                    lm_pad[:VOCAB] = qm_model.bf16("model.embed_tokens.weight")[:VOCAB]
                    lq, ls, lm_ = _requant_q4k(lm_pad, G)
                    del lm_pad
                else:
                    # Untied: preserve the bundle's real quantized LM head.
                    lq, ls, lm_ = _pad_rows(
                        qm_model.q4nx("lm_head.weight", VOCAB, K), VPF
                    )
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
        print(f"[q4nx_requant] wrote {cache_path}", flush=True)
    return cache_path


def reference_weights(model, n_layers, tie_lm_head=True, path="decode"):
    """(HF parameter name, fn(shape)) for each weight build_requant_cache packs,
    giving the float values the decode computes with: the projections as the
    bundle stores them, and the LM head as packed above. The input for the
    accuracy reference (verify/dequant_reference.py); it has to follow the
    packer, so it lives next to it. The input embedding is the bundle's; for
    path="prefill" a tied head stays that bf16 embedding, which is what the
    fused prefill computes with."""
    from llama32_1b_q4nx_weights import GROUP, Q4nxModel

    qm = Q4nxModel(model)
    weights = [
        (
            f"model.layers.{k}.{t}.weight",
            lambda shape, n=f"model.layers.{k}.{t}.weight": qm.dequant(n, *shape),
        )
        for k in range(n_layers)
        for t in _PROJ.values()
    ]
    # The NPU looks tokens up in the bundle's embedding, which differs from the
    # checkpoint's. On the prefill path a tied head is that embedding too.
    weights.append(
        (
            "model.embed_tokens.weight",
            lambda shape: qm.bf16("model.embed_tokens.weight").reshape(shape),
        )
    )
    if tie_lm_head:
        if path == "decode":
            head = lambda shape: host_weights.requant_q4k_dequant(  # noqa: E731
                qm.bf16("model.embed_tokens.weight"), GROUP
            )
            weights.append(("lm_head.weight", head))
    else:
        head = lambda shape: qm.dequant("lm_head.weight", *shape)  # noqa: E731
        weights.append(("lm_head.weight", head))
    return weights
