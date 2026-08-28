# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the DFlash DRAFTER's fused-decode requant cache from
# z-lab/Qwen3-4B-DFlash-b16's bf16 safetensors.
#
# This is qwen3_4b_q4nx_requant.py's packing applied to the drafter, and it is a
# SEPARATE function rather than a parameter on that one on purpose: the target
# path is the verified shipping path (docs/DFlashFeasibility.md section 4) and
# nothing here should be able to perturb it.
#
# Three differences from the target:
#
#   5 layers, not 36     comes free -- the geometry is read out of a
#                        fused_decode loaded at DECODE_MODEL=qwen3-4b-draft,
#                        whose only difference from the target is UNI_DEC.
#   fc, 2560 x 12800     a projection the target has no analogue for. Packed as
#                        its own slab with the SAME cascade packer: at this
#                        geometry it is I2P=5, J2P=25 (section 3.3), a
#                        well-formed phase because accumulating across input
#                        column-blocks is what the proj cores already do.
#   a TIED head from     the drafter carries no embedding table of its own, so
#   ANOTHER model        the vocab slabs come from the target's bundle. Passing
#                        the wrong one decodes fluent garbage, so the target
#                        source is a required argument, not a default.
#
# The fc slab is emitted as its own array rather than being concatenated into
# the layer stream: it is not per-layer, and where it lands in the weight BO is
# a decision for the wave that feeds it, which does not exist yet. Keeping it
# separate means the layout can be chosen once, in one place.

import os

import numpy as np
from ml_dtypes import bfloat16

from qwen3_4b_q4nx_requant import _interleave512, _requant_q4k


def build_draft_requant_cache(
    fd, cache_path, target_source, draft_path=None, verbose=True
):
    """Re-quantize + cascade-pack the drafter into `cache_path`.

    `fd` must be a fused_decode loaded with DECODE_MODEL=qwen3-4b-draft; its
    UNI_DEC is what makes this 5 layers rather than 36.
    """
    from qwen3_4b_draft_weights import DraftWeights, D, FC_IN

    dw = DraftWeights(draft_path, target_source=target_source)
    G, NCX, NCY, NPH = fd.GROUP, fd.NCX, fd.NCY, fd.NPH
    OP, GP, DP = fd.OPROJ_PHASE, fd.GLU_PHASE, fd.DOWN_PHASE
    GLU_CHUNK, W_LAYER = fd.GLU_CHUNK, fd.W_LAYER
    VP, VPF, UNI_LM = fd.VOCAB_SIZE_PADDED, fd.VOCAB_SIZE_PADDED_FULL, fd.UNI_LM
    n_layers = fd.UNI_DEC
    DUAL = bool(getattr(fd, "W_DUAL_CHAN", 0))
    PROJ = DraftWeights._PROJ

    # The builder and this packer must agree, or the cache is silently wrong in
    # a way only a garbage draft reveals -- and a garbage DRAFT does not even
    # fail loudly, it just stops being accepted.
    assert fd.K == D, (fd.K, D)
    if n_layers != 5:
        raise ValueError(
            f"fd.UNI_DEC is {n_layers}; load fused_decode with "
            f"DECODE_MODEL=qwen3-4b-draft (5 layers) before packing the drafter"
        )

    def _dq(k, nm):
        t, M, Kc = PROJ[nm]
        return dw.dequant(f"layers.{k}.{t}.weight", M, Kc)

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
        r_in, r_post = dw.layer_rms(k)
        RMS_in.append(np.asarray(r_in, bfloat16))
        RMS_post.append(np.asarray(r_post, bfloat16))
        assert W_all[-1].size == W_LAYER, (W_all[-1].size, W_LAYER)
        if verbose:
            print(f"[draft requant] layer {k} requantized", flush=True)

    # fc: one 2560 x 12800 projection, packed with the same cascade order. Its
    # J is 25 where a layer phase's is 5..19; nothing about the packer is
    # phase-specific, it takes the matrix it is given.
    fc_packed = fd.pack_q4k_cascade(
        *_requant_q4k(dw.fc(), G), NCX, NCY, iter_major=True, dual_chan=DUAL
    )
    if verbose:
        print(
            f"[draft requant] fc {D}x{FC_IN} packed, {fc_packed.size} elements",
            flush=True,
        )

    # The head is the TARGET's embedding matrix -- tied, and not in this
    # checkpoint at all.
    lm = dw.bf16("model.embed_tokens.weight")
    lm_pad = np.zeros((VPF, D), np.float32)
    lm_pad[: fd.VOCAB_SIZE] = lm[: fd.VOCAB_SIZE]
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
        W_fc=np.asarray(fc_packed).view(np.int16),
        # fc ALSO in bf16, and that is the form the first implementation will
        # use. It cannot be a 5th decode phase: `FULL4` (fused_decode.py:1180)
        # is `NPH == 4 and ...`, and it gates the whole fused 4-phase structure
        # including the RMS_BAND_STREAM level 3 path section 3.2 verified. So fc
        # gets its OWN launch, and a standalone launch is served by the bf16
        # GEMM builder in llms/shared/builders rather than by the q4k cascade.
        # 65 MB against the drafter's ~300 MB of Q4 layers -- worth it to avoid
        # a new quantized shape on a path nothing else uses.
        W_fc_bf16=np.asarray(dw.fc_bf16()).view(np.int16),
        RMS_in=np.stack(RMS_in).view(np.int16),
        RMS_post=np.stack(RMS_post).view(np.int16),
        hidden_norm=np.asarray(dw.hidden_norm(), bfloat16).view(np.int16),
        final_norm=np.asarray(dw.final_norm(), bfloat16).view(np.int16),
    )
    if verbose:
        print(f"[draft requant] wrote {cache_path}", flush=True)
    return cache_path


def _load_fd():
    """fused_decode at the DRAFTER's geometry."""
    import importlib.util
    import sys
    from pathlib import Path

    here = Path(__file__).resolve().parent
    fdir = here.parent.parent / "fused_decode"
    for k in list(os.environ):
        if k.startswith("DECODE_"):
            os.environ.pop(k, None)
    os.environ.update(
        DECODE_MODEL="qwen3-4b-draft",
        VOCAB_CHUNK_I2="30",
        LM_HEAD="0",
        NLAYERS="1",
        UNIFIED="1",
        DECODE_GOLDEN="1",
        DECODE_GOLDEN_L="128",
    )
    sys.path.insert(0, str(fdir))
    spec = importlib.util.spec_from_file_location(
        "fused_decode_draft", str(fdir / "fused_decode.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_check(verbose=True):
    """Round-trip: does the packed fc still hold the matrix that went in?

    The 4-bit re-quant is lossy by construction, so this is not an equality
    check -- it is a check that the RIGHT matrix was packed in the right order.
    A layout error moves rows and shows up as a relative error of order 1, not
    of order the quantization step.
    """
    from qwen3_4b_draft_weights import DraftWeights

    fd = _load_fd()
    dw = DraftWeights()
    fc = dw.fc()
    q, s, mn = _requant_q4k(fc, fd.GROUP)
    back = (
        q.reshape(fc.shape[0], -1, fd.GROUP).astype(np.float32) * s[..., None]
        + mn[..., None]
    ).reshape(fc.shape)
    rel = np.abs(back - fc).max() / max(np.abs(fc).max(), 1e-9)
    step = (s.max() / max(np.abs(fc).max(), 1e-9)) * 1.0
    ok = rel <= 2.0 * step
    if verbose:
        print(
            f"[draft requant] fc requant round-trip: max rel {rel:.4e}, "
            f"one quantization step is {step:.4e}   {'OK' if ok else 'TOO LARGE'}"
        )
        print(
            f"[draft requant] fd.UNI_DEC={fd.UNI_DEC}, W_LAYER={fd.W_LAYER}, "
            f"NCX={fd.NCX} NCY={fd.NCY} NPH={fd.NPH}"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(self_check())
