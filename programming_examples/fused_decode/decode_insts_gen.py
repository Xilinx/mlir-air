#!/usr/bin/env python3
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


def attn_maxl_of(L):
    """ATTN_MAXL a decode build at context length L is compiled for (16*ceil(L/16))."""
    return ((L + 15) // 16) * 16


class DecodeInstsGen:
    def __init__(self, artifact_dir, max_L=None):
        """artifact_dir holds decode_L<N>.{xclbin,insts.bin} builds. Templates are grouped by
        ATTN_MAXL; each needs two same-ATTN_MAXL builds to calibrate the L-slope. `max_L`
        (session context cap) selects the smallest ATTN_MAXL template that covers it; if
        None, the largest calibrated template is used."""
        self.dir = artifact_dir
        builds = {}
        for fn in os.listdir(artifact_dir):
            if fn.startswith("decode_L") and fn.endswith(".insts.bin"):
                try:
                    L = int(fn[len("decode_L") : -len(".insts.bin")])
                except ValueError:
                    continue
                if os.path.exists(os.path.join(artifact_dir, f"decode_L{L}.xclbin")):
                    builds[L] = fn
        by_maxl = {}
        for L in builds:
            by_maxl.setdefault(attn_maxl_of(L), []).append(L)
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
                xclbin=os.path.join(artifact_dir, f"decode_L{base_L}.xclbin"),
            )
        self.select(max_L)

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
                    f"(build decode_L{max_L} + an adjacent same-ATTN_MAXL build)"
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
        if not (1 <= L <= self.active_maxl):
            raise ValueError(f"L={L} out of range for ATTN_MAXL={self.active_maxl}")
        t = self._t
        out = t["base"].astype(np.int64)
        ld = t["slope"] != 0
        out[ld] = t["base"][ld].astype(np.int64) + (L - t["base_L"]) * t["slope"][ld]
        return out.astype(np.uint32)

    def xclbin_for_maxl(self, m):
        return self.templates[m]["xclbin"]

    def windows_for_range(self, L_lo, L_hi):
        """Set of ATTN_MAXL window templates needed to cover L in [L_lo, L_hi]."""
        return sorted({attn_maxl_of(L) for L in range(L_lo, L_hi + 1)})

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
