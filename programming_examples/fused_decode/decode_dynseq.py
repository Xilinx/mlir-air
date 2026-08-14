#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-token instruction streams built at dispatch, from one DECODE_DYNSEQ build.

The staircase reaches every context length by extrapolating a template's
L-dependent words from two calibrated builds. That works because those words
happen to be affine in L, and it needs a template pair per window.

A DECODE_DYNSEQ build takes the context length as a runtime scalar instead, so
the compiler emits a C++ builder that assembles the stream from it. Calling that
builder gives the exact stream for any L from a single build -- including the
readback's ceil(L/16) block count, which is a staircase in L and not affine, so
no amount of two-point calibration reproduces it.

The surface matches DecodeInstsGen so a driver can hold either.
"""

import glob
import os

import numpy as np

_AIR_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "python",
)


def _txn_builder(header, verbose=False):
    import sys

    if _AIR_PYTHON not in sys.path:
        sys.path.insert(0, _AIR_PYTHON)
    from air.backend.txn_builder import TxnBuilder

    return TxnBuilder(header, verbose=verbose)


class DynseqInstsGen:
    """One DECODE_DYNSEQ build serving every context length.

    `exact` marks the streams as computed rather than extrapolated, which is what
    tells decode_staircase to rewrite the whole stream per token instead of a
    calibrated slice.
    """

    exact = True

    def __init__(self, artifact_dir, max_L=None, verbose=False):
        self.dir = artifact_dir
        xclbins = sorted(glob.glob(os.path.join(artifact_dir, "decode_dyn_L*.xclbin")))
        if not xclbins:
            raise FileNotFoundError(
                f"no decode_dyn_L*.xclbin in {artifact_dir}; build one with "
                f"`make compile-decode-dynseq`"
            )
        self.templates = {}
        for xb in xclbins:
            maxl = int(os.path.basename(xb)[len("decode_dyn_L") : -len(".xclbin")])
            header = xb[: -len(".xclbin")] + ".txn.h"
            if not os.path.exists(header):
                raise FileNotFoundError(
                    f"{xb} has no TXN builder next to it ({header}); it was not "
                    f"built with DECODE_DYNSEQ=1"
                )
            self.templates[maxl] = dict(xclbin=xb, header=header, builder=None)
        self.verbose = verbose
        self.select(max_L)

    def _builder(self, maxl):
        t = self.templates[maxl]
        if t["builder"] is None:
            b = _txn_builder(t["header"], self.verbose)
            names = b.function_names
            if len(names) != 1:
                raise RuntimeError(
                    f"{t['header']} declares {len(names)} builders "
                    f"({', '.join(names)}); expected exactly one runtime sequence"
                )
            t["builder"] = (b, names[0])
        return t["builder"]

    def select(self, max_L=None):
        """Pick the smallest build covering `max_L`, as DecodeInstsGen does.

        A dynseq build still has a compile-time ATTN_MAXL: it sizes the KV cache
        and the core's buffers. What it no longer fixes is how much of that cache
        a token moves.
        """
        ws = self.calibrated_windows()
        self.active_maxl = (
            next((m for m in ws if m >= int(max_L)), ws[-1]) if max_L else ws[-1]
        )
        return self.active_maxl

    @property
    def attn_maxl(self):
        return self.active_maxl

    @property
    def xclbin(self):
        return self.templates[self.active_maxl]["xclbin"]

    @property
    def base(self):
        return self.insts_for(self.active_maxl, 1)

    def insts_for_L(self, L):
        return self.insts_for(self.active_maxl, L)

    def insts_for(self, maxl, L):
        if not (1 <= L <= maxl):
            raise ValueError(f"L={L} out of range for ATTN_MAXL={maxl}")
        b, name = self._builder(maxl)
        return b(name, int(L))

    def calibrated_windows(self):
        return sorted(self.templates)

    def window_for_L(self, L):
        for m in self.calibrated_windows():
            if m >= L:
                return m
        raise KeyError(f"no build covers L={L}")

    def xclbin_for_maxl(self, m):
        return self.templates[m]["xclbin"]

    def describe(self):
        return "\n".join(
            f"dynseq build ATTN_MAXL={m}: {os.path.basename(t['xclbin'])}"
            + (" (active)" if m == self.active_maxl else "")
            for m, t in sorted(self.templates.items())
        )


def pick_insts_gen(artifact_dir, max_L=None, verbose=False):
    """The decode instruction generator a driver should use.

    DECODE_DYNSEQ=1 selects the build that takes its context length at dispatch
    (one xclbin for every L, streaming only this token's context); otherwise the
    compile-time template pair, extrapolated per token. Shared by every model so
    the choice cannot drift between them.
    """
    if os.environ.get("DECODE_DYNSEQ") == "1":
        return DynseqInstsGen(str(artifact_dir), max_L=max_L, verbose=verbose)
    import sys

    if str(artifact_dir) not in sys.path:
        sys.path.insert(0, str(artifact_dir))
    from decode_insts_gen import DecodeInstsGen

    return DecodeInstsGen(str(artifact_dir), max_L=max_L)


def dispatch_args(gen, L):
    """Trailing kernel arguments for a dispatch at context length L.

    A dynseq build's runtime sequence takes the context length as a scalar, so
    the kernel signature carries it. The value the hardware acts on is already
    assembled into the instruction stream; this keeps the arity right.
    """
    return [int(L)] if getattr(gen, "exact", False) else []
