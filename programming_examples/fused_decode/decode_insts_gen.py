#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""AIR analog of the reference's gen_layer_seq: emit the fused-decode instruction stream for a
given KV context length L by writing only the L-dependent fields (attention RTP-L words +
KV-append byte offset) into ONE decode template.

the reference allocates the KV cache once at MAX_L (a power-of-2 >= 4096) and grows the context
within it, regenerating the per-token sequence with the current L (append at (L-1)*4*DH +
rtp_write(L)); there is no re-allocation or block-count switching (llama_npu.cpp:90-96,
llama_npu_sequence.cpp:49). The AIR fused decode is the faithful analog: ONE xclbin built
at a fixed ATTN_MAXL (= the MAX_L analog, = 16*ceil(build_L/16)); the KV readback streams
ATTN_MAXL positions and the attention loop is bounded at runtime by the RTP-L value, so the
same template serves every L in [1, ATTN_MAXL]. This generator specializes that template to
each L by writing only the L-dependent words -- byte-identical to a native per-L aircc
build (the two same-ATTN_MAXL reference builds locate those words; verified byte-exact).

(An ATTN_MAXL smaller than the target generation length streams less KV per step but caps
context; choosing ATTN_MAXL >= max generation length is the single knob, exactly MAX_L.)
"""

import os
import numpy as np


def attn_maxl_of(L, batch=1):
    """ATTN_MAXL a decode build at context length L and batch B is compiled for.

    16*ceil((L + B - 1)/16), and the B is not decoration. A block of B tokens
    occupies positions L-1 .. L+B-2, so the builder sizes its block loop from
    ATTN_L_BLK = L + B - 1 (fused_decode.py) -- at L=128 batch 8 that is 144
    positions, not 128. Getting it wrong here is not an off-by-one in a loop
    bound: ATTN_MAXL is the KV REGION STRIDE, so a host that assumes 128 lays
    every group's region 16 positions short of where the device reads it, every
    layer reads the wrong keys, and the answer comes back finite, token-varying
    and wrong.
    """
    return ((L + batch - 1 + 15) // 16) * 16


class DecodeInstsGen:
    def __init__(self, artifact_dir, max_L=None, prefix="decode_L", batch=1):
        """artifact_dir holds <prefix><N>.{xclbin,insts.bin} builds. Templates are grouped by
        ATTN_MAXL; each needs two same-ATTN_MAXL builds to calibrate the L-slope. `max_L`
        (session context cap) selects the smallest ATTN_MAXL template that covers it; if
        None, the largest calibrated template is used.

        `prefix` exists so a batched build can live in the SAME directory as the
        shipping batch-1 pair. It has to: the two are different xclbins for the
        same model at the same context length, and naming them alike would make
        the scan below pick whichever was built last -- which is exactly the
        "leftover pair from an unrelated build" failure _check_declared_windows
        was written for, one directory closer."""
        self.dir = artifact_dir
        self.prefix = prefix
        # Needed BEFORE the scan: the window a template belongs to depends on the
        # batch it was built at (see attn_maxl_of).
        self.batch = int(batch)
        builds = {}
        for fn in os.listdir(artifact_dir):
            if fn.startswith(prefix) and fn.endswith(".insts.bin"):
                try:
                    L = int(fn[len(prefix) : -len(".insts.bin")])
                except ValueError:
                    continue
                if os.path.exists(os.path.join(artifact_dir, f"{prefix}{L}.xclbin")):
                    builds[L] = fn
        by_maxl = {}
        for L in builds:
            by_maxl.setdefault(attn_maxl_of(L, self.batch), []).append(L)
        # calibrate a base+slope per ATTN_MAXL from two same-ATTN_MAXL builds
        self.templates = {}
        for maxl, Ls in by_maxl.items():
            Ls = sorted(Ls)
            # drop broken/foreign builds whose insts size differs from the group's mode
            sizes = {
                L: os.path.getsize(os.path.join(artifact_dir, builds[L])) for L in Ls
            }
            good_size = max(set(sizes.values()), key=list(sizes.values()).count)
            Ls = [L for L in Ls if sizes[L] == good_size]
            base_L = Ls[0]
            base = np.fromfile(
                os.path.join(artifact_dir, builds[base_L]), dtype=np.uint32
            )
            slope = None
            for ref_L in Ls[1:]:
                ref = np.fromfile(
                    os.path.join(artifact_dir, builds[ref_L]), dtype=np.uint32
                )
                if ref.size == base.size:
                    d = ref.astype(np.int64) - base.astype(np.int64)
                    dL = ref_L - base_L
                    if (d % dL == 0).all():
                        slope = (d // dL).astype(np.int64)
                        break
            self.templates[maxl] = dict(
                base_L=base_L,
                base=base,
                slope=slope,
                Ls=Ls,
                xclbin=os.path.join(artifact_dir, f"{prefix}{base_L}.xclbin"),
            )
        self._check_declared_windows(artifact_dir)
        self.select(max_L)

    # Build stamp naming the ATTN_MAXL windows a directory is meant to hold, written by
    # `make compile-decode-windows`.
    WINDOWS_STAMP = ".decode_windows"

    @property
    def windows_stamp(self):
        """The stamp for THIS template family. Batch-1 keeps the historical
        name; a prefixed family gets its own, because the two families calibrate
        different windows and one stamp cannot describe both."""
        if self.prefix == "decode_L":
            return self.WINDOWS_STAMP
        return self.WINDOWS_STAMP + "." + self.prefix.rstrip("_L")

    def _check_declared_windows(self, artifact_dir):
        """Reject a directory whose calibrated windows differ from what the build declared.

        Templates are discovered by scanning, so a leftover pair from an unrelated build
        otherwise changes which window is selected -- silently, and the symptom is a
        plausible-looking but wrong generation rather than an error.
        """
        stamp = os.path.join(artifact_dir, self.windows_stamp)
        if not os.path.exists(stamp):
            return  # nothing declared (single-template tree) -- nothing to check
        with open(stamp) as f:
            want = sorted(int(t) for t in f.read().split())
        have = self.calibrated_windows()
        if have != want:
            raise RuntimeError(
                f"decode templates in {artifact_dir} do not match {self.windows_stamp}: "
                f"declared ATTN_MAXL {want}, found calibrated {have}. Remove strays or "
                f"re-run `make compile-decode-windows`."
            )

    def select(self, max_L=None):
        """Pick the active template: smallest calibrated ATTN_MAXL >= max_L (or the largest
        calibrated one if max_L is None)."""
        cal = {m: t for m, t in self.templates.items() if t["slope"] is not None}
        if not cal:
            raise RuntimeError(
                "no calibrated decode template (need two same-ATTN_MAXL builds)"
            )
        if max_L is None:
            maxl = max(cal)
        else:
            covering = sorted(m for m in cal if m >= max_L)
            if not covering:
                raise KeyError(
                    f"no decode template covers max_L={max_L}; largest is {max(cal)} "
                    f"(build {self.prefix}{max_L} + an adjacent same-ATTN_MAXL build)"
                )
            maxl = covering[0]
        self.active_maxl = maxl
        self._t = cal[maxl]
        return maxl

    @property
    def attn_maxl(self):
        return self.active_maxl

    @property
    def base_L(self):
        return self._t["base_L"]

    @property
    def xclbin(self):
        return self._t["xclbin"]

    @property
    def base(self):
        # active template's base insts stream (for sizing a host insts BO).
        return self._t["base"]

    def insts_for_L(self, L):
        """uint32 insts stream for context length L on the active template (1 <= L <= ATTN_MAXL)."""
        return self.insts_for(self.active_maxl, L)

    def insts_for(self, maxl, L):
        """uint32 insts stream for L on the `maxl` template, leaving the active one alone.

        A staircase driver holds every window at once and picks per token, so it needs
        the streams without the select()/active_maxl round trip.
        """
        t = self.templates[maxl]
        if t["slope"] is None:
            raise KeyError(f"template ATTN_MAXL={maxl} is not calibrated")
        if not (1 <= L <= maxl):
            raise ValueError(f"L={L} out of range for ATTN_MAXL={maxl}")
        out = t["base"].astype(np.int64)
        ld = t["slope"] != 0
        out[ld] = t["base"][ld].astype(np.int64) + (L - t["base_L"]) * t["slope"][ld]
        return out.astype(np.uint32)

    def calibrated_windows(self):
        """Calibrated ATTN_MAXL windows, ascending."""
        return sorted(m for m, t in self.templates.items() if t["slope"] is not None)

    def window_for_L(self, L):
        """Smallest calibrated window that can serve context length L.

        The compiled KV readback streams ATTN_MAXL positions regardless of L, so the
        smallest covering window is also the cheapest one to run at.
        """
        for m in self.calibrated_windows():
            if m >= L:
                return m
        raise KeyError(f"no calibrated window covers L={L}")

    def xclbin_for_maxl(self, m):
        return self.templates[m]["xclbin"]

    def windows_for_range(self, L_lo, L_hi):
        """Set of ATTN_MAXL window templates needed to cover L in [L_lo, L_hi]."""
        return sorted({attn_maxl_of(L, self.batch) for L in range(L_lo, L_hi + 1)})

    def describe(self):
        lines = []
        for m in sorted(self.templates):
            t = self.templates[m]
            nl = int((t["slope"] != 0).sum()) if t["slope"] is not None else None
            act = " (active)" if m == self.active_maxl else ""
            lines.append(
                f"template ATTN_MAXL={m}: base_L={t['base_L']} builds={t['Ls']} "
                f"L-dep_words={nl}{act}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    g = DecodeInstsGen(HERE)
    print(g.describe())
    print(f"active ATTN_MAXL={g.attn_maxl} base_L={g.base_L}")
    # self-check every calibrated template byte-exact vs its native builds
    ok = True
    for maxl in sorted(g.templates):
        if g.templates[maxl]["slope"] is None:
            continue
        g.select(max_L=maxl)
        for L in g.templates[maxl]["Ls"]:
            nat = np.fromfile(
                os.path.join(HERE, f"decode_L{L}.insts.bin"), dtype=np.uint32
            )
            gen = g.insts_for_L(L)
            same = nat.size == gen.size and bool((nat == gen).all())
            ok = ok and same
            print(f"  ATTN_MAXL={maxl} L={L:3d} byte-exact={same}")
    print("ALL BYTE-EXACT" if ok else "MISMATCH")
