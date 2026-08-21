# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Emit `bench_decode.exe` BO-geometry flags for a model at a given ATTN_MAXL.

bench_decode.cpp hard-codes llama-3.2-1B at L=2048. Every other (model, context)
cell needs its BO sizes passed in, and getting one wrong does not fail loudly --
it benches a differently shaped dispatch and reports a plausible number. So the
sizes are not restated here: this imports the production builder
(`fused_decode.py`) with the same environment the driver uses and reads them off
it, mirroring the driver's own sizing:

    x   K
    w   W.size                                    (from the requant cache)
    r   UNI_DEC*RMS_LAYER + 64 + K, LUT at UNI_DEC*RMS_LAYER
    y   (HOST_ROUNDS+LAYER_RNDS)*PAYLOAD + UNI_LM*VOCAB_SIZE_PADDED
    kv  UNI_DEC*ATTN_MAXL*KVSZ_TOK

Models whose weights exceed a shim BD's 4 GiB reach (qwen3-8b) build with
DECODE_WGROUP set and take several weight buffers instead of one; `w_parts` then
carries their element counts, in the order the kernel signature expects them.

`--check` reproduces bench_decode.cpp's built-in defaults from the same code
path, which is what licenses using this for the other models. Run it whenever
the builder's BO sizing changes.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

FUSED_DECODE = Path(__file__).resolve().parents[2] / "fused_decode"

# bench_decode.cpp's built-in defaults, i.e. llama-3.2-1B at ATTN_MAXL=2048.
# VOCAB_CHUNK_I2=18 and w_elems are that model's, from its lit/Makefile.
_CHECK_MODEL = dict(model="llama-3.2-1b", i2="18", ctx=2048, w_elems=386662400)
# The values that used to be hand-carried per model. Derivation must reproduce
# all three, which is what licenses deriving rather than restating them.
_CHECK_W_ELEMS = [
    ("llama-3.2-1b", "18", 16, 386662400),
    ("llama-3.2-3b", "9", 28, 1004666880),
    ("gemma3-4b", "5", 34, 1213644800),
]
# The split models. Their parts summing to w_elems checks the part that can
# actually be wrong: that the per-group `min(G, UNI_DEC - g*G)` covers exactly
# UNI_DEC layers, with no gap and no overlap. The head term is shared by both
# routes, so it is the grouping arithmetic that is under test here. Both cases
# are kept because they divide differently -- 36 into 9 is exact, 32 into 8 is
# exact at a different group count -- and a gap/overlap bug need not hit both.
_CHECK_SPLITS = [
    (
        "qwen3-8b",
        "8",
        36,
        dict(DECODE_STACK="6144", DECODE_WGROUP="9"),
        [542638080, 542638080, 542638080, 542638080, 199229440],
    ),
    (
        "llama-3.1-8b",
        "16",
        32,
        dict(DECODE_STACK="8064", DECODE_WGROUP="8"),
        [545259520, 545259520, 545259520, 545259520, 167772160],
    ),
    (
        "qwen2.5-7b",
        "7",
        28,
        dict(DECODE_STACK="6144", DECODE_WGROUP="7"),
        [509788160, 509788160, 509788160, 509788160, 172605440],
    ),
]
_CHECK_WANT = dict(
    k=2048,
    w_elems=386662400,
    rms_size=67648,
    ny=134144,
    kv_elems=33554432,
    decode_y=5120,
    voc_n=129024,
    rms_lut_off=65536,
)


# Builder env keys the last geometry() call set via env_extra, so the next one
# can clear them (the builder reads its knobs at import time).
_EXTRA_SET = set()


def _head_elems(fd):
    """Packed elements in the LM head slab, as the builder sizes it (`_wvoc_len`)."""
    return fd.UNI_LM * fd.VOCAB_W_BLOCKS * fd.BLOCK_BF16


def derive_w_elems(fd, n_layers):
    """Total packed decode weight elements: N layers plus the LM head.

    Both terms are read off the builder rather than recomputed from the packing
    geometry: W_LAYER already sums pack_q4k_cascade's per-matrix
    `nbi*nbj*BLOCK_BF16` over one layer's matrices, and VOCAB_W_BLOCKS is the
    head's block count. Re-deriving either from ROW_BLOCK/COL_BLOCK would
    reintroduce a floor division that silently undercounts if the blocking or the
    vocab padding ever changes.

    Derived rather than passed in: the value used to be a per-model constant with
    no source of truth in the tree -- bench_decode.cpp hard-codes llama-3.2-1B's
    as a default and every other model's lived in a scratch script. Getting it
    wrong does not fail, it sizes the weight BO wrongly and reports a plausible
    number. Reproduces all three previously hand-carried values exactly
    (see _CHECK_W_ELEMS).
    """
    return n_layers * fd.W_LAYER + _head_elems(fd)


def derive_w_parts(fd):
    """Per-buffer weight element counts, or None when the model uses one buffer.

    A shim BD's byte offset is a uint32, so one buffer only reaches 4 GiB;
    DECODE_WGROUP splits the layers over ceil(UNI_DEC/G) buffers plus a dedicated
    lm-head buffer. Read straight off the builder (`_wgrp_len` / `_wvoc_len` in
    fused_decode.py) rather than restated, because the host has to slice the
    weights exactly as the emitted signature expects: a wrong split still
    dispatches, it just feeds the later layers from the wrong base.
    """
    if not fd.W_SPLIT:
        return None
    return [
        min(fd.W_GROUP, fd.UNI_DEC - g * fd.W_GROUP) * fd.W_LAYER
        for g in range(fd.N_WGRP)
    ] + [_head_elems(fd)]


def geometry(model, vocab_chunk_i2, ctx, w_elems=None, n_layers=None, env_extra=None):
    """Import the builder at this context and read its BO sizes back out.

    w_elems is derived from the builder unless given explicitly; when both are
    present they are cross-checked, because a silent mismatch there is the one
    error in here that produces a number rather than a failure.
    """
    os.environ.update(
        DECODE_MODEL=model,
        VOCAB_CHUNK_I2=str(vocab_chunk_i2),
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        UNIFIED="1",
        DECODE_GOLDEN_L=str(ctx),
        W_DUAL_CHAN="1",
    )
    # Some models need extra builder env (qwen3-8b's DECODE_STACK/DECODE_WGROUP).
    # Drop whatever a previous call in this process set and this one does not:
    # the builder reads these at import, so a leftover DECODE_WGROUP would make
    # the next model come back split when it is not (which --check would hit).
    global _EXTRA_SET
    for k in _EXTRA_SET - set(env_extra or {}):
        os.environ.pop(k, None)
    _EXTRA_SET = set(env_extra or {})
    os.environ.update(env_extra or {})
    # fused_decode.py imports its siblings (proj_qmm_pack, ...) by bare name.
    if str(FUSED_DECODE) not in sys.path:
        sys.path.insert(0, str(FUSED_DECODE))
    spec = importlib.util.spec_from_file_location(
        "_fused_decode_geom", FUSED_DECODE / "fused_decode.py"
    )
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)

    # `is not None`, not truthiness: 0 is a wrong answer to "how many layers",
    # not an absent one, and silently falling back to a head-only weight BO
    # would size the dispatch wrongly rather than say so.
    derived = derive_w_elems(fd, n_layers) if n_layers is not None else None
    if w_elems is not None and derived is not None and w_elems != derived:
        raise SystemExit(
            f"w_elems mismatch for {model}: passed {w_elems}, builder says {derived}"
        )
    w_elems = w_elems if w_elems is not None else derived
    # Catches both "neither was given" and a degenerate 0 from either route.
    if not w_elems:
        raise SystemExit(
            "need a positive --w-elems or --n-layers to size the weight BO"
        )

    w_parts = derive_w_parts(fd)
    if w_parts and sum(w_parts) != w_elems:
        raise SystemExit(
            f"weight split for {model} sums to {sum(w_parts)}, not {w_elems}"
        )

    decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
    return dict(
        k=fd.K,
        w_elems=w_elems,
        **({"w_parts": w_parts} if w_parts else {}),
        rms_size=fd.UNI_DEC * fd.RMS_LAYER + 64 + fd.K,
        ny=decode_y + fd.UNI_LM * fd.VOCAB_SIZE_PADDED,
        kv_elems=fd.UNI_DEC * fd.ATTN_MAXL * fd.KVSZ_TOK,
        decode_y=decode_y,
        voc_n=fd.UNI_LM * fd.VOCAB_SIZE_PADDED,
        rms_lut_off=fd.UNI_DEC * fd.RMS_LAYER,
    )


def as_flags(g):
    """bench_decode.exe CLI form. ATTN_MAXL is deliberately not emitted -- it is
    an input to the sizing above, not a flag bench_decode.cpp accepts."""
    fmt = lambda v: ",".join(str(x) for x in v) if isinstance(v, list) else v
    return " ".join(f"--{k.replace('_', '-')} {fmt(v)}" for k, v in g.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="DECODE_MODEL, e.g. llama-3.2-1b")
    p.add_argument("--vocab-chunk-i2", help="VOCAB_CHUNK_I2 for this model")
    p.add_argument("--ctx", type=int, help="ATTN_MAXL / decode context")
    p.add_argument("--w-elems", type=int, help="override; normally derived")
    p.add_argument("--n-layers", type=int, help="decoder layers, to derive w_elems")
    p.add_argument("--json", action="store_true", help="emit JSON, not flags")
    p.add_argument(
        "--check",
        action="store_true",
        help="reproduce bench_decode.cpp's built-in defaults and exit",
    )
    a = p.parse_args()

    if a.check:
        g = geometry(
            _CHECK_MODEL["model"],
            _CHECK_MODEL["i2"],
            _CHECK_MODEL["ctx"],
            _CHECK_MODEL["w_elems"],
        )
        bad = {k: (v, g[k]) for k, v in _CHECK_WANT.items() if g[k] != v}
        for model, i2, nl, want in _CHECK_W_ELEMS:
            got = geometry(model, i2, 2048, n_layers=nl)["w_elems"]
            if got != want:
                bad[f"w_elems[{model}]"] = (want, got)
        for model, i2, nl, extra, want in _CHECK_SPLITS:
            got = geometry(model, i2, 2048, n_layers=nl, env_extra=extra).get("w_parts")
            if got != want:
                bad[f"w_parts[{model}]"] = (want, got)
        print("SELF-CHECK", "PASS" if not bad else f"FAIL {bad}")
        return 0 if not bad else 1

    missing = [f for f in ("model", "vocab_chunk_i2", "ctx") if getattr(a, f) is None]
    if missing:
        p.error(
            "--check, or all of: "
            + ", ".join("--" + m.replace("_", "-") for m in missing)
        )

    g = geometry(a.model, a.vocab_chunk_i2, a.ctx, a.w_elems, a.n_layers)
    print(json.dumps(g) if a.json else as_flags(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
