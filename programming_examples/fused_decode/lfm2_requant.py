# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the LFM2-1.2B fused-decode weight cache (.npz) straight from the HF
# bf16 checkpoint (LiquidAI/LFM2-1.2B).
#
# Quantizes the full-precision checkpoint directly rather than re-quantizing a
# pre-quantized bundle -- see llms/lfm2_1_2b_q4nx/docs/Q4NX_DECODE_STATUS.md.
# The device is built -DQ4_0 (symmetric signed int4, w = q*scale, no min term),
# so `requant_q4_0` from the Qwen path is the codec, reused verbatim.
#
# LFM2 is HYBRID: 6 of its 16 layers are attention (2,5,8,10,12,14) and the
# other 10 are a gated causal depthwise conv ("ShortConv"). Both layer types
# map onto the SAME 4-phase weight stream, which is why one packer serves both:
#
#   phase   attention                        conv
#   ph0     QKV [q2048|k512|v512] = 3072     in_proj [B|C|v] = 6144
#   ph1     o_proj      2048 x 2048          out_proj    2048 x 2048
#   ph2     gate/up, up|gate interleaved     identical
#   ph3     down 8192 -> 2048                identical
#
# Only ph0's row count and the mixer between ph0 and ph1 differ. ph1-ph3 are
# byte-for-byte the same shape.
#
# Two simplifications versus the Qwen packer:
#   * INTERMEDIATE is exactly 8192 (already a multiple of 512), so there is no
#     row/column padding anywhere -- Qwen has to pad 11008 -> 11264.
#   * LFM2 projections are bias-free, so there is no BIAS array. Instead the
#     rope slab grows to carry per-head QK-norm (cos/sin 64 | q_norm 64 |
#     k_norm 64 = 192 bf16), which is what HAS_QK_NORM widens rope_w to.
import os
import sys

import numpy as np
from ml_dtypes import bfloat16

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from proj_qmm_pack import BLOCK_BF16  # noqa: E402
from qwen25_3b_requant import (  # noqa: E402
    HFModel,
    requant_q4_0,
    pack_q4k_cascade_fast,
    _interleave_chunks,
)

HF_REPO = "LiquidAI/LFM2-1.2B"

# Layer schedule. IRREGULAR (gaps of 2,2,1,1,1,0 conv layers) -- always drive
# off this list, never a modulo.
FULL_ATTN_IDXS = (2, 5, 8, 10, 12, 14)
N_LAYERS = 16
CONV_DIM = 2048
CONV_L_CACHE = 3

# HF LFM2 tensor names. These are NOT the names the pre-quantized bundle uses
# (it renames everything onto llama-style naming); reading with the wrong set
# silently finds nothing.
_ATTN_PROJ = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.out_proj",
}
_CONV_PROJ = {"in": "conv.in_proj", "out": "conv.out_proj"}
# LFM2's SwiGLU names gate/up/down as w1/w3/w2.
_FFN = {"gate": "feed_forward.w1", "up": "feed_forward.w3", "down": "feed_forward.w2"}


def is_attn_layer(k):
    return k in FULL_ATTN_IDXS


def _ffn_phases(hf, k, GLU_CHUNK):
    """ph2 (gate/up interleaved) and ph3 (down) -- identical for both layer
    types, and unpadded because INTERMEDIATE is exactly 8192."""
    gate = hf.bf16(f"model.layers.{k}.{_FFN['gate']}.weight")
    up = hf.bf16(f"model.layers.{k}.{_FFN['up']}.weight")
    down = hf.bf16(f"model.layers.{k}.{_FFN['down']}.weight")
    qu, su = requant_q4_0(up)
    qg, sg = requant_q4_0(gate)
    # Quants and scales interleave with the SAME row chunking -- both have M
    # rows, they differ only in column count (K vs K/GROUP).
    ph_gu = (
        _interleave_chunks(qu, qg, GLU_CHUNK),
        _interleave_chunks(su, sg, GLU_CHUNK),
    )
    return ph_gu, requant_q4_0(down)


def pack_lm_head(fd, hf, verbose=True):
    """Cascade-pack the LM head into `UNI_LM` vocab waves.

    LFM2 TIES the head to the embedding (HF ships no `lm_head.weight`), so the
    head IS `model.embed_tokens.weight`, quantized Q4_0 like any other
    projection -- it goes through the same device proj path, so it cannot stay
    full precision here even though the prefill loader keeps a fp copy.

    Returns a list of `UNI_LM` arrays, concatenated after the layer slabs to
    form the decode W buffer.
    """
    VP, VPF, UNI_LM = fd.VOCAB_SIZE_PADDED, fd.VOCAB_SIZE_PADDED_FULL, fd.UNI_LM
    NCX, NCY = fd.NCX, fd.NCY
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    D = fd.K

    lm = hf.bf16("model.embed_tokens.weight")
    # VOCAB_SIZE_PADDED_FULL == VOCAB_SIZE for LFM2 (65536 is already a
    # multiple of K), so this pad is a no-op -- kept so the code stays correct
    # for a vocab that is not.
    lm_pad = np.zeros((VPF, D), np.float32)
    lm_pad[: min(lm.shape[0], VPF)] = lm[:VPF]
    lq, ls = requant_q4_0(lm_pad)
    waves = [
        pack_q4k_cascade_fast(
            lq[w * VP : (w + 1) * VP],
            ls[w * VP : (w + 1) * VP],
            NCX,
            NCY,
            dual_chan=DUAL,
        )
        for w in range(UNI_LM)
    ]
    if verbose:
        print(
            f"[lfm2 requant] lm_head (TIED to embed): {UNI_LM} waves x {waves[0].size}",
            flush=True,
        )
    return waves


def build_requant_cache(
    fd, cache_path, model=HF_REPO, layer_kind="attn", with_lm_head=True, verbose=True
):
    """Re-quantize + cascade-pack one LFM2 layer TYPE into a decode .npz.

    `fd`         = the loaded fused_decode module (supplies the cascade geometry).
    `layer_kind` = "attn" (the 6 attention layers), "conv" (the 10 ShortConv
                   layers), or "all" (the WHOLE 16-layer model, in EXECUTION
                   order, for the hybrid single-binary build).

    "attn" and "conv" are the two half-builds, which use different ph0 widths
    and so need different `fd` builds. "all" targets the hybrid, whose phase
    schedule is UNIFORM: every layer gets the same W_LAYER, and an attention
    layer -- whose QKV fills only the first of the two mixer phases -- has the
    second packed as ZEROS. Padding is not free (see the model entry) but it is
    what lets the weight offset stay a plain `iv * W_LAYER`.

    Writes keys:
      W        [n, W_LAYER] int16 -- phase-major [ph][cx][...] weight stream
      IDX      [n] int32  -- the model layer index each row corresponds to
      RMS_in   [n, K] bf16 -- operator_norm
      RMS_post [n, K] bf16 -- ffn_norm
      NORM     [K]    bf16 -- final norm (model.embedding_norm)
      QNORM/KNORM [n, DH] bf16 -- per-head QK-norm     (attn only)
      CONV_W   [n, CONV_DIM, 3] bf16 -- depthwise taps (conv only)
      KIND     [n] int32  -- arm per layer, 1=conv 2=attn        ("all" only)
      ROPE_W   [n, ROPE_W_LEN] bf16 -- the per-layer mixer-weight slab, already
               in device layout: taps tap-major for a conv layer, and for an
               attention layer q/k-norm at [DH:3*DH] with [0:DH] left for the
               runtime to write this position's cos/sin.      ("all" only)
    """
    assert layer_kind in ("attn", "conv", "all"), layer_kind
    hf = HFModel(model)
    NCX, NCY = fd.NCX, fd.NCY
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    GLU_CHUNK = fd.PAYLOAD
    OP, GP, DP = fd.OPROJ_PHASE, fd.GATEUP_PHASE, fd.DOWN_PHASE
    # The mixer projection may span several phases: LFM2's conv in_proj is 3*K
    # wide and runs as two phases of 3*K/2. Splitting is a plain row split --
    # the mixer reassembles the waves in order, so [B|C|X] comes out unchanged.
    MIXP = fd.MIXER_PHASES
    NPH = fd.NPH
    W_LAYER = fd.W_LAYER

    if layer_kind == "all":
        idxs = list(range(N_LAYERS))  # EXECUTION order -- the two kinds interleave
    else:
        idxs = [
            k for k in range(N_LAYERS) if is_attn_layer(k) == (layer_kind == "attn")
        ]
    if verbose:
        print(
            f"[lfm2 requant] {layer_kind}: layers {idxs} "
            f"(W_LAYER={W_LAYER} bf16, dual_chan={int(DUAL)})",
            flush=True,
        )

    W_all, RMS_in, RMS_post = [], [], []
    QN, KN, CONV_W = [], [], []
    KIND, ROPE_W = [], []
    RW_LEN = fd.ROPE_W_LEN
    DH = fd.DH

    for k in idxs:
        p = f"model.layers.{k}"
        ph = [None] * NPH
        _attn = is_attn_layer(k) if layer_kind == "all" else (layer_kind == "attn")
        if layer_kind == "all":
            KIND.append(2 if _attn else 1)
            ROPE_W.append(np.zeros(RW_LEN, np.float32))
        if _attn:
            R = {nm: hf.bf16(f"{p}.{t}.weight") for nm, t in _ATTN_PROJ.items()}
            _qkv = np.concatenate([R["q"], R["k"], R["v"]], 0)
            # QKV fills the FIRST mixer phase; on the hybrid's uniform
            # schedule the rest is zero padding, which the mixer consumes and
            # discards. All-zero groups quantize to scale 1, so this packs
            # cleanly rather than producing an undequantizable block.
            for _i, _mp in enumerate(MIXP):
                _rows = _qkv[_i * fd.M : (_i + 1) * fd.M]
                if _rows.shape[0] < fd.M:
                    _rows = np.concatenate(
                        [_rows, np.zeros((fd.M - _rows.shape[0], fd.K), _qkv.dtype)]
                    )
                ph[_mp] = requant_q4_0(_rows)
            # NO column permutation on o-proj. `attn_out_perm` is a QWEN-ENGINE
            # concept (fused_decode_qwen); it appears nowhere in fused_decode.py
            # or its packer, which LFM2 drives. Applying it scrambles the o-proj
            # input mapping -- that cost a debugging cycle, hence this note.
            ph[OP] = requant_q4_0(R["o"])
            _qn = hf.bf16(f"{p}.self_attn.q_layernorm.weight")
            _kn = hf.bf16(f"{p}.self_attn.k_layernorm.weight")
            QN.append(_qn)
            KN.append(_kn)
            if layer_kind == "all":
                # [cos|sin] occupy [0:DH] and are POSITION-dependent, so the
                # runtime writes them; q/k-norm are weights and live here.
                ROPE_W[-1][DH : 2 * DH] = _qn.astype(np.float32)
                ROPE_W[-1][2 * DH : 3 * DH] = _kn.astype(np.float32)
        else:
            _inp = hf.bf16(f"{p}.{_CONV_PROJ['in']}.weight")
            for _i, _mp in enumerate(MIXP):
                ph[_mp] = requant_q4_0(_inp[_i * fd.M : (_i + 1) * fd.M])
            ph[OP] = requant_q4_0(hf.bf16(f"{p}.{_CONV_PROJ['out']}.weight"))
            # Depthwise taps: bf16, NOT quantized. Stored [conv_dim, 1, 3] --
            # squeeze HF's singleton Conv1d groups axis.
            _taps = hf.bf16(f"{p}.conv.conv.weight").reshape(CONV_DIM, CONV_L_CACHE)
            CONV_W.append(_taps)
            if layer_kind == "all":
                # TAP-MAJOR [w0|w1|w2]: HF ships channel-major (CONV_DIM,1,3),
                # and the kernel loads one contiguous 16-lane vector per tap.
                ROPE_W[-1][: CONV_DIM * CONV_L_CACHE] = np.ascontiguousarray(
                    _taps.T
                ).reshape(-1)

        ph[GP], ph[DP] = _ffn_phases(hf, k, GLU_CHUNK)

        w = np.concatenate(
            [
                pack_q4k_cascade_fast(*ph[i], NCX, NCY, dual_chan=DUAL)
                for i in range(NPH)
            ]
        )
        assert w.size == W_LAYER, (k, w.size, W_LAYER)
        W_all.append(w)
        RMS_in.append(hf.bf16(f"{p}.operator_norm.weight"))
        RMS_post.append(hf.bf16(f"{p}.ffn_norm.weight"))
        if verbose:
            _kn2 = "attn" if _attn else "conv"
            print(f"[lfm2 requant] layer {k:2d} [{_kn2}] packed", flush=True)

    def _b(a):
        return np.stack(a).astype(bfloat16).view(np.int16)

    # The LM head runs in the SAME dispatch as the layers, so its vocab waves
    # are concatenated after the layer slabs to form the decode W buffer. BOTH
    # builds carry it -- each is a standalone decode whose UNI_LM waves are part
    # of its own weight buffer.
    WV = pack_lm_head(fd, hf, verbose=verbose) if with_lm_head else []
    if WV:
        # Check what was PACKED, not what the builder currently streams. The
        # cache is a per-layer PREFIX by construction, so one full pack serves
        # both the 16-layer model and every short `LAYERS=N` bisect build -- the
        # consumer slices it. Asserting against fd.UNI_DEC instead made a pack
        # under DECODE_UNI_DEC=1 fail here, which broke the documented bisect
        # workflow for a cache that was in fact correct.
        total = sum(w.size for w in W_all) + sum(w.size for w in WV)
        want = len(idxs) * W_LAYER + fd.UNI_LM * WV[0].size
        assert total == want, (total, want, len(idxs))
        print(
            f"[lfm2 requant] W total = {total} bf16 " f"({len(idxs)} layers + lm_head)",
            flush=True,
        )

    out = dict(
        W=np.stack(W_all),
        WV=np.stack(WV) if WV else np.zeros((0,), np.int16),
        IDX=np.asarray(idxs, np.int32),
        RMS_in=_b(RMS_in),
        RMS_post=_b(RMS_post),
        NORM=hf.bf16("model.embedding_norm.weight").astype(bfloat16).view(np.int16),
    )
    if layer_kind == "all":
        out["KIND"] = np.asarray(KIND, np.int32)
        out["ROPE_W"] = _b(ROPE_W)
        out["QNORM"], out["KNORM"] = _b(QN), _b(KN)
        out["CONV_W"] = _b(CONV_W)
    elif layer_kind == "attn":
        out["QNORM"], out["KNORM"] = _b(QN), _b(KN)
    else:
        out["CONV_W"] = _b(CONV_W)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
    np.savez(cache_path, **out)
    if verbose:
        print(f"[lfm2 requant] wrote {cache_path}", flush=True)
    return cache_path
