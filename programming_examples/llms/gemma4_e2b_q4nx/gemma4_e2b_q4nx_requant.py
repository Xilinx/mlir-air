# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the Gemma4-E2B fused-decode weight cache from the model.q4nx bundle.
# Mirrors gemma3_4b_q4nx_requant.py; the Gemma4 deltas are all consequences of
# ONE design decision in fused_decode_ple.py -- every layer is built at the
# WIDEST layer's geometry (M=5120, INTER=12288) and the narrow layers are padded
# up to it, so the device sees a single uniform shape.
#
# Padding the FFN is trivial (zero rows; gelu(0)*0 contributes nothing, and the
# down-proj columns that read them are zero too). Padding ATTENTION is not, and
# the two traps below were both found by measurement, not inspection:
#
#   1. RoPE. kernels/rope.cc pairs dim i with dim i + DH/2, with DH the BUILD's
#      head dim (512). A sliding layer's real head is 256 wide, so its correct
#      pairing is (i, i+128). Padding it as [real(256) | zeros(256)] makes the
#      kernel pair real dims against zeros -- verified WRONG. Padding it as
#      [real_lo(128) | 0(128) | real_hi(128) | 0(128)] makes the kernel's
#      (i, i+256) land exactly on the real (i, i+128) pairs -- verified an exact
#      match. That interleave is what _pad_head does.
#
#   2. qk-norm. RMSNorm over the padded 512 head sees half the mean square of
#      the true 256 head, so it scales by sqrt(2). Folding 1/sqrt(2) into the
#      norm weight cancels it exactly (measured max rel err 4e-7).
#
# Everything the padding touches -- q, k, v, the o-proj's contraction columns,
# and the q/k norm weights -- has to use the SAME interleave, or the heads
# silently mismatch and the model degrades without failing.
import os

import numpy as np
from ml_dtypes import bfloat16

import gemma4_e2b_q4nx_weights as gw


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


def _head_perm(dh_real, dh_build):
    """Row indices of a padded head that carry real data, in real order.

    [real_lo | zeros | real_hi | zeros] -- see trap 1 in the module docstring.
    Returns an index array of length dh_real into a dh_build-wide head.
    """
    h = dh_real // 2
    return np.concatenate([np.arange(h), np.arange(dh_build // 2, dh_build // 2 + h)])


def _pad_head(rows, n_heads, dh_real, dh_build, axis=0):
    """Scatter per-head rows (or columns) into the padded head layout."""
    if dh_real == dh_build:
        return rows
    src = np.moveaxis(rows, axis, 0)
    out = np.zeros((n_heads * dh_build,) + src.shape[1:], src.dtype)
    p = _head_perm(dh_real, dh_build)
    for h in range(n_heads):
        out[h * dh_build + p] = src[h * dh_real : (h + 1) * dh_real]
    return np.moveaxis(out, 0, axis)


def _pad_rows(A, n):
    """Zero-pad a [M,K] matrix up to n rows."""
    if A.shape[0] == n:
        return A
    out = np.zeros((n, A.shape[1]), A.dtype)
    out[: A.shape[0]] = A
    return out


def _pad_cols(A, n):
    """Zero-pad a [M,K] matrix up to n columns."""
    if A.shape[1] == n:
        return A
    out = np.zeros((A.shape[0], n), A.dtype)
    out[:, : A.shape[1]] = A
    return out


def layer_rope_w(pos, L, fd, rope_freqs):
    """One layer's rope_w slab: [cos/sin(DH) | q_norm(DH) | k_norm(DH)].

    The sliding/full split lives ENTIRELY here, in the data: different theta,
    and the "proportional" divisor table on the full layers. The builder needs
    no per-layer attention variant for it (ROPE_W_PER_LAYER is already true,
    since this model has qk-norm).
    """
    DH = fd.DH_A
    cos, sin, dh = gw.rope_lut(pos, L, rope_freqs=rope_freqs)
    half = DH // 2
    c = np.ones(half, np.float32)
    s = np.zeros(half, np.float32)
    # A padded head rotates its real pairs at (i, i+DH/2); the padded lanes get
    # cos=1/sin=0, which is the identity on the zeros sitting there.
    c[: dh // 2] = cos
    s[: dh // 2] = sin
    return np.concatenate([c, s]).astype(np.float32), dh


def build_requant_cache(model, fd, cache_path, layers=None, verbose=True):
    """Re-quantize + cascade-pack model.q4nx into the decode .npz.

    `fd` is the loaded fused_decode_ple module (DECODE_MODEL=gemma4-e2b), which
    supplies the cascade geometry and the PLE slab layout, so the two cannot
    drift. `layers` selects which model layers become device slabs, in order --
    it exists so a 1-layer build can be pointed at a FULL-attention layer
    (4, 9, ...) instead of layer 0, which is sliding. Default is 0..UNI_DEC-1.
    """
    qm = gw.Q4nxModel(model)
    G, NCX, NCY, NPH = fd.GROUP, fd.NCX, fd.NCY, fd.NPH
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    GLU_CHUNK, W_LAYER = fd.GLU_CHUNK, fd.W_LAYER
    DH, K = fd.DH_A, fd.K
    NQ, NKV = fd.NUM_Q_HEADS, fd.NUM_KV_HEADS
    INTER = fd.MODEL["INTER_NARROW"] * 2  # the padded (widest) FFN
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    layers = list(range(fd.UNI_DEC)) if layers is None else list(layers)

    W_all, PLE_all = [], []
    RMS = {n: [] for n in ("in", "post_attn", "pre_ffn", "post_ffn", "post_ple")}
    ROPE_QK = []

    for slab, L in enumerate(layers):
        w = qm.layer_weights(L)
        nm = qm.layer_norms(L)
        dh = gw.head_dim(L)

        # --- attention, padded into the build's head geometry ---
        q = _pad_head(w["q"], NQ, dh, DH)
        if "k" in w:
            k, v = _pad_head(w["k"], NKV, dh, DH), _pad_head(w["v"], NKV, dh, DH)
        else:
            # Layers >= FIRST_KV_SHARED carry no k/v at all: they reuse a lower
            # layer's cache. Their QKV phase still has to be the same shape, so
            # the k/v rows are zero and the device simply recomputes nothing
            # useful into a slot it will not read.
            k = np.zeros((NKV * DH, K), np.float32)
            v = np.zeros((NKV * DH, K), np.float32)
        o = _pad_head(w["o"], NQ, dh, DH, axis=1)

        # --- FFN, zero-padded up to the widest layer ---
        up, gate = _pad_rows(w["up"], INTER), _pad_rows(w["gate"], INTER)
        down = _pad_cols(w["down"], INTER)

        qmats = [None] * NPH
        qmats[0] = _requant_q4k(np.concatenate([q, k, v], 0), G)
        qmats[OP] = _requant_q4k(o, G)
        qmats[GP] = _interleave512(
            _requant_q4k(up, G), _requant_q4k(gate, G), GLU_CHUNK
        )
        qmats[DP] = _requant_q4k(down, G)
        packed = np.concatenate(
            [
                fd.pack_q4k_cascade(
                    *qmats[p], NCX, NCY, iter_major=True, dual_chan=DUAL
                )
                for p in range(NPH)
            ]
        )
        assert packed.size == W_LAYER, (packed.size, W_LAYER)
        W_all.append(packed)

        for key, src in (
            ("in", "input"),
            ("post_attn", "post_attn"),
            ("pre_ffn", "pre_ffn"),
            ("post_ffn", "post_ffn"),
            ("post_ple", "post_ple"),
        ):
            RMS[key].append(np.asarray(nm[src], bfloat16))

        # qk-norm, with the 1/sqrt(2) that cancels a padded head's RMSNorm.
        fix = 1.0 if dh == DH else float(np.sqrt(dh / DH))
        qn = np.zeros(DH, np.float32)
        kn = np.zeros(DH, np.float32)
        p = _head_perm(dh, DH)
        qn[p] = np.asarray(nm["q_norm"], np.float32) * fix
        kn[p] = np.asarray(nm["k_norm"], np.float32) * fix
        ROPE_QK.append((qn.astype(bfloat16), kn.astype(bfloat16)))

        # --- PLE slab: RAW bundle bytes, no de-tiling (see kernels/ple.cc) ---
        pw = _ple_raw(qm, L)
        sl = np.zeros(fd.PLE_LAYER, bfloat16)
        sl[: fd.PLE_EMB_OFF] = pw
        sl[fd.PLE_EMB_OFF : fd.PLE_NORMW_OFF] = _ple_token_embed(qm, L)
        sl[fd.PLE_NORMW_OFF : fd.PLE_UPNORMW_OFF] = np.asarray(
            qm.globals()["ple_proj_norm"], bfloat16
        )
        sl[fd.PLE_UPNORMW_OFF : fd.PLE_SCALE_OFF] = np.asarray(nm["post_ple"], bfloat16)
        sl[fd.PLE_SCALE_OFF] = bfloat16(nm["out_scale"])
        PLE_all.append(sl)

        if verbose:
            kind = "s" if gw.is_sliding(L) else "F"
            print(
                f"[gemma4 requant] slab {slab} <- layer {L} ({kind} dh={dh})",
                flush=True,
            )

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez(
        cache_path,
        W=np.concatenate(W_all).view(np.int16),
        PLE=np.concatenate(PLE_all).view(np.int16),
        layers=np.asarray(layers, np.int32),
        **{
            f"RMS_{n}": np.stack(RMS[n]).view(np.int16)
            for n in ("in", "post_attn", "pre_ffn", "post_ffn", "post_ple")
        },
        QNORM=np.stack([a for a, _ in ROPE_QK]).view(np.int16),
        KNORM=np.stack([b for _, b in ROPE_QK]).view(np.int16),
    )
    if verbose:
        print(f"[gemma4 requant] wrote {cache_path}", flush=True)
    return cache_path


def _ple_raw(qm, L):
    """The three PLE matrices as the bundle stores them, concatenated.

    RAW and un-transposed on purpose: the bundle's (8,1536,32) tiling IS the
    block order kernels/ple.cc consumes, so any de-tiling here would have to be
    undone. layer_ple() de-tiles for the numpy reference; this does not.
    """
    p = f"model.layers.{L}."
    return np.concatenate(
        [
            qm.bf16(p + "inp_gate.weight").ravel(),
            qm.bf16(f"model.per_layer_model_proj.weight_layer{L}").ravel(),
            qm.bf16(p + "per_layer_projection.weight").ravel(),
        ]
    ).astype(bfloat16)


def _ple_token_embed(qm, L):
    """This layer's slice of the per-layer token-embedding table.

    Filled per token by the driver; the slab carries a zero placeholder so its
    size and offsets are fixed at pack time.
    """
    return np.zeros(gw.PLI_D, bfloat16)
