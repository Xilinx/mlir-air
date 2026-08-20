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


def geometry(model, vocab_chunk_i2, ctx, w_elems):
    """Import the builder at this context and read its BO sizes back out."""
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
    # fused_decode.py imports its siblings (proj_qmm_pack, ...) by bare name.
    if str(FUSED_DECODE) not in sys.path:
        sys.path.insert(0, str(FUSED_DECODE))
    spec = importlib.util.spec_from_file_location(
        "_fused_decode_geom", FUSED_DECODE / "fused_decode.py"
    )
    fd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fd)

    decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
    return dict(
        k=fd.K,
        w_elems=w_elems,
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
    return " ".join(f"--{k.replace('_', '-')} {v}" for k, v in g.items())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="DECODE_MODEL, e.g. llama-3.2-1b")
    p.add_argument("--vocab-chunk-i2", help="VOCAB_CHUNK_I2 for this model")
    p.add_argument("--ctx", type=int, help="ATTN_MAXL / decode context")
    p.add_argument("--w-elems", type=int, help="weight element count")
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
        print("SELF-CHECK", "PASS" if not bad else f"FAIL {bad}")
        return 0 if not bad else 1

    missing = [
        f
        for f in ("model", "vocab_chunk_i2", "ctx", "w_elems")
        if getattr(a, f) is None
    ]
    if missing:
        p.error(
            "--check, or all of: "
            + ", ".join("--" + m.replace("_", "-") for m in missing)
        )

    g = geometry(a.model, a.vocab_chunk_i2, a.ctx, a.w_elems)
    print(json.dumps(g) if a.json else as_flags(g))
    return 0


if __name__ == "__main__":
    sys.exit(main())
