# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Qwen3-4B fused-decode requant cache (q4k-cascade weights .npz) from
# the model.q4nx bundle. This is the DFlash target model -- see
# docs/DFlashFeasibility.md.
#
# It is qwen3_8b_q4nx_requant.py with one structural change and a pile of
# dimensions that come off the builder rather than being restated:
#
#   - 36 layers, TWO RMSNorm weights/layer (input, post_attention) -- the
#     standard pre-norm pair, as 8B. Not Gemma's 4-norm sandwich.
#   - per-head qk-norm weights are NOT packed here; they ride in the
#     per-position rope_w slab the driver rewrites each step, as 8B.
#   - THE LM HEAD IS TIED, which 8B's is not. Qwen3-4B sets
#     tie_word_embeddings=true (see llms/qwen3_4b/qwen3_4b_inference.py, which
#     states it), so the head is the bf16 embedding matrix and NOT a separate
#     Q4NX lm_head tensor. That is the llama path, and taking 8B's branch here
#     would read a tensor the bundle does not have.
#
# Everything dimensional -- GROUP, NCX/NCY, the phase indices, GLU_CHUNK,
# W_LAYER, UNI_LM, pack_q4k_cascade -- is read out of the passed-in
# fused_decode module, so pointing this at DECODE_MODEL=qwen3-4b is what
# supplies the geometry. Nothing here restates a shape.
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
    """Re-quantize + cascade-pack the Qwen3-4B model.q4nx weights into the .npz.

    `fd` = the loaded fused_decode module (DECODE_MODEL=qwen3-4b) supplying the
    cascade geometry + pack_q4k_cascade + phase indices."""
    from qwen3_4b_q4nx_weights import Q4nxModel, D, VOCAB

    qm_model = Q4nxModel(model)
    G, NCX, NCY, NPH = fd.GROUP, fd.NCX, fd.NCY, fd.NPH
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    GLU_CHUNK, W_LAYER = fd.GLU_CHUNK, fd.W_LAYER
    VP, VPF, UNI_LM = fd.VOCAB_SIZE_PADDED, fd.VOCAB_SIZE_PADDED_FULL, fd.UNI_LM
    n_layers = fd.UNI_DEC
    # The builder and this packer must agree on the model, or the cache is
    # silently wrong in a way only a garbage decode reveals.
    assert fd.K == D, (fd.K, D)
    assert fd.VOCAB_SIZE == VOCAB, (fd.VOCAB_SIZE, VOCAB)
    # Dual-MM2S weight feed: the decode splits each column's slab across the
    # column's two shim channels by cascade pair, which needs the cascade laid
    # out as [low-row half | high-row half]. Keyed off the SAME flag the decode
    # was built with (fd.W_DUAL_CHAN) so the pack cannot disagree with the
    # xclbin.
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
            print(f"[qwen3-4b requant] layer {k} requantized", flush=True)

    # LM head: TIED (config tie_word_embeddings=true) -- the full-precision
    # embedding matrix is the head. This is the one place 4B and 8B differ; 8B
    # reads its own Q4NX lm_head.weight, which this bundle does not carry.
    lm = qm_model.bf16("model.embed_tokens.weight")
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
        print(f"[qwen3-4b requant] wrote {cache_path}", flush=True)
    return cache_path


def self_check(verbose=True):
    """Assert the loader's dims agree with the builder's phase geometry.

    Runs WITHOUT the model bundle, which matters: the bundle is a gated download
    and this is the part that can go silently wrong without one. A packer that
    disagrees with the xclbin about how many rows a phase emits produces a cache
    that loads fine and decodes garbage, so the shapes are worth gating even
    though the weights are not available to test against.

    Mirrors llms/bench/decode_geometry.py --check in intent."""
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    fd_dir = here.parents[1] / "fused_decode"
    os.environ.update(
        DECODE_MODEL="qwen3-4b",
        VOCAB_CHUNK_I2="30",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        UNIFIED="1",
        DECODE_GOLDEN_L="2048",
        W_DUAL_CHAN="1",
    )
    for p in (str(fd_dir), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("_fd_q34", fd_dir / "fused_decode.py")
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)
    import qwen3_4b_q4nx_weights as w

    checks = [
        ("hidden K", fd.K, w.D),
        ("vocab", fd.VOCAB_SIZE, w.VOCAB),
        ("layers UNI_DEC", fd.UNI_DEC, w.NUM_LAYERS),
        ("qkv out (phase 0 M)", fd.M, w.DQ + w.DK + w.DV),
        ("o-proj contract DQ", fd.DQ, w.DQ),
        ("GLU_OUT vs INTER", fd.GLU_OUT, w.INTER),
        ("head_dim DH", fd.DH, w.DH),
    ]
    # Rows each phase emits, from the builder's own I2P geometry, against what
    # the projections above actually produce.
    rows = [w.DQ + w.DK + w.DV, w.D, 2 * w.INTER, w.D]
    for p, r in enumerate(rows):
        emitted = fd.I2P[p] * fd.PAIR_ROWS * fd.NCX * fd.NCY * fd.ROW_BLOCK
        checks.append((f"phase {p} rows", emitted, r))

    bad = 0
    for name, a, b in checks:
        if a != b:
            bad += 1
        if verbose:
            print(f"  {'OK ' if a == b else 'MISMATCH'} {name:22s} {a:7d} {b:7d}")
    print("SELF-CHECK PASS" if not bad else f"SELF-CHECK FAIL ({bad})")
    return 0 if not bad else 1


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="assert loader dims against the builder; needs no model bundle",
    )
    a = ap.parse_args()
    if not a.check:
        ap.error("nothing to do without --check; import build_requant_cache instead")
    sys.exit(self_check())
