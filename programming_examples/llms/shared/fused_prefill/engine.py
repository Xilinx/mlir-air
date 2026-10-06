# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host side of the fused prefill device: one hw_context, the GEMM weight
sites, attention with insts generated per chunk, and the KV records.

A model's prefill subclasses Engine and runs its layers through mm(),
ffn() and attention(); norms, rope and residuals are its own (hostops).
"""

import json
import time
from pathlib import Path

import numpy as np
import pyxrt as xrt
from ml_dtypes import bfloat16

from . import device as D
from . import hostops as H
from . import insts as I
from . import packing as P

TO, FROM = (
    xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
    xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
)


class _Run:
    """An xrt.run that can be rebuilt on a fresh context with the same args."""

    def __init__(self, ctx, args):
        self.ctx, self.args = ctx, list(args)
        self.make()

    def make(self):
        self.r = xrt.run(self.ctx.kern)
        for i, v in enumerate(self.args):
            self.r.set_arg(i, v)

    def set_arg(self, i, v):
        self.args[i] = v
        self.r.set_arg(i, v)


class Ctx:
    """One xclbin on its own hw_context, with prepared runs. close() releases
    the context (BOs survive); open() makes it again and rebinds the runs."""

    def __init__(self, dev, xclbin):
        self.dev = dev
        self.xb = xrt.xclbin(str(xclbin))
        dev.register_xclbin(self.xb)
        self.runs = []
        self.open()

    def open(self):
        self.ctx = xrt.hw_context(self.dev, self.xb.get_uuid())
        self.kern = xrt.kernel(self.ctx, "MLIR_AIE")
        for r in self.runs:
            r.make()

    def close(self):
        for r in self.runs:
            r.r = None
        self.kern = self.ctx = None

    def bo(self, nbytes, arg):
        return xrt.bo(
            self.dev,
            max(int(nbytes), 4096),
            xrt.bo.host_only,
            self.kern.group_id(3 + arg),
        )

    def insts(self, words):
        words = np.asarray(words, np.uint32)
        ib = xrt.bo(self.dev, words.nbytes, xrt.bo.cacheable, self.kern.group_id(1))
        view = np.frombuffer(ib.map(), np.uint32, count=words.size)
        view[:] = words
        ib.sync(TO)
        return ib, view

    def run(self, ib, n, bos):
        r = _Run(self, [3, ib, n, *bos])
        self.runs.append(r)
        return r


def view(bo, dtype, shape):
    return np.frombuffer(bo.map(), dtype, count=int(np.prod(shape))).reshape(shape)


class Engine:
    def __init__(self, build_dir, cfg, max_len=2048):
        self.dir = Path(build_dir)
        self.cfg = cfg
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.max_len = max_len
        self.cols = D.cols(cfg)
        # first kernel BO argument of each group: the GEMM's, then the
        # attention groups' in config order
        self.slot = {"g": 0}
        self.slot.update({a: 3 + 3 * i for i, a in enumerate(cfg.attn)})
        self.nargs = 3 + 3 * len(cfg.attn)
        H.load(str(self.dir))
        self.dev = xrt.device(0)
        self.fp = Ctx(self.dev, self.dir / f"{self.man['gemm'][0]}.xclbin")
        self.dev_t, self.by_op = 0.0, {}
        self.suspended = False
        self.dummy = [self.fp.bo(4096, i) for i in range(self.nargs)]
        self.abuf, self.sites = {}, {}

    # ---- device plumbing ----

    def _insts(self, name, group):
        raw = (self.dir / f"{name}.insts.bin").read_bytes()
        return np.frombuffer(I.strip_columns(raw, self.cols[group]), np.uint32)

    def _go(self, run, tag, out):
        # Unless out's first cache line is flushed before the dispatch, the
        # host can read that line back with the previous dispatch's bytes.
        out.sync(TO, 64, 0)
        t0 = time.perf_counter()
        run.r.start()
        st = run.r.wait()
        dt = time.perf_counter() - t0
        self.dev_t += dt
        self.by_op[tag] = self.by_op.get(tag, 0.0) + dt
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"fused prefill {tag}: {st}")

    def gemm_site(self, site, k, n, packed, wq):
        """Load one weight matrix (packed by pack_q4 if wq, else pack_bf16)."""
        fp = self.fp
        npd = P.n_pad(n)
        if k not in self.abuf:
            a = fp.bo(D.M * k * 2, 0)
            am = view(a, bfloat16, (D.HR, k // D.TK, D.TM, D.TK))
            am[:] = 0
            self.abuf[k] = dict(a=a, am=am, last=None)
        w = fp.bo(packed.nbytes, 1)
        w.write(packed.tobytes(), 0)
        w.sync(TO)
        c = fp.bo(D.M * npd * 2, 2)
        self.sites[site] = dict(
            k=k,
            n=n,
            npd=npd,
            wq=wq,
            w=w,
            c=c,
            cm=view(c, bfloat16, (D.M, npd)),
            runs={},
        )

    def _write_a(self, k, x):
        ab = self.abuf[k]
        if ab["last"] is not x:
            H.tile_a(x, ab["am"], D.TM, D.TK)
            ab["a"].sync(TO)
            ab["last"] = x

    def _gemm(self, site, act, t):
        s = self.sites[site]
        if act not in s["runs"]:
            name = f"g_{s['k']}_{s['npd']}_{act}_{s['wq']}"
            ins = self._insts(name, "g")
            ib, _ = self.fp.insts(ins)
            bos = list(self.dummy)
            bos[0:3] = [self.abuf[s["k"]]["a"], s["w"], s["c"]]
            s["runs"][act] = (self.fp.run(ib, len(ins), bos), ib, name)
        self._go(s["runs"][act][0], site.split(".")[-1], s["c"])
        s["c"].sync(FROM, t * s["npd"] * 2, 0)
        return s

    def _out(self, s, t):
        return H.bf16_to_f32(s["cm"], t, s["n"])

    def mm(self, site, x, act=0):
        t = x.shape[0]
        self._write_a(self.sites[site]["k"], x)
        return self._out(self._gemm(site, act, t), t)

    def a_from(self, k, fill):
        ab = self.abuf[k]
        fill(ab["am"])
        ab["a"].sync(TO)
        ab["last"] = object()

    def ffn(self, gate, up, down, h):
        """down(act(gate h) * up h), the activation in the gate's drain."""
        t = h.shape[0]
        self._write_a(self.sites[gate]["k"], h)
        g = self._gemm(gate, 1, t)
        u = self._gemm(up, 0, t)
        n = g["n"]
        self.a_from(n, lambda am: H.glu_tile(g["cm"], u["cm"], t, n, am, D.TM, D.TK))
        return self._out(self._gemm(down, 0, t), t)

    # ---- attention ----

    def attn_setup(self):
        fp, self.attn = self.fp, {}
        pts = {
            k: dict(zip(("heads", "nkv", "q0", "k0"), v))
            for k, v in self.man["attn_points"].items()
        }
        for op, groups in D.attn_ops(self.cfg).items():
            names = self.man["attn"][op]
            base = self._insts(names["base"], op).tobytes()
            var = {
                p: (self._insts(names[p], op).tobytes(), pts[p][p] - pts["base"][p])
                for p in ("nkv", "q0", "k0")
            }
            tmpl = I.LinearInsts(
                base, {p: pts["base"][p] for p in ("nkv", "q0", "k0")}, var
            )
            chk = pts["check"]
            if not np.array_equal(
                tmpl.at(nkv=chk["nkv"], q0=chk["q0"], k0=chk["k0"]),
                self._insts(names["check"], op),
            ):
                raise RuntimeError(
                    f"attention {op}: insts are not linear in (nkv, q0, k0)"
                )
            hg = pts["base"]["heads"] // len(groups)
            bos = list(self.dummy)
            parts = []
            for a in groups:
                dh, s = self.cfg.attn[a].dh, self.slot[a]
                q = fp.bo(hg * D.M * dh * 2, s)
                o = fp.bo(hg * D.M * dh * 2, s + 2)
                qm = view(q, bfloat16, (hg, D.M, dh))
                qm[:] = 0
                bos[s : s + 3] = [q, self.dummy[s + 1], o]
                parts.append(dict(q=q, qm=qm, o=o, om=view(o, bfloat16, (hg, D.M, dh))))
            ib, iv = fp.insts(tmpl.at(nkv=1, q0=0, k0=0))
            self.attn[op] = dict(
                tmpl=tmpl,
                groups=groups,
                parts=parts,
                ib=ib,
                iv=iv,
                run=fp.run(ib, iv.size, bos),
            )

    def kv_bo(self, op):
        """The KV records of one layer for attention op `op`: per group, per
        KV head, a block per lkp tokens."""
        parts = []
        for a in D.attn_ops(self.cfg)[op]:
            g = self.cfg.attn[a]
            nblk = g.max_blocks if g.kv_heads > 1 else -(-self.max_len // g.lkp)
            assert nblk * g.lkp >= self.max_len, (nblk, g.lkp, self.max_len)
            shape = (g.kv_heads, nblk, D.kv_rec(g))
            bo = self.fp.bo(int(np.prod(shape)) * 2, self.slot[a] + 1)
            parts.append(dict(bo=bo, m=view(bo, bfloat16, shape), a=a))
        return dict(op=op, parts=parts)

    def kv_append(self, kv, r0, k, v):
        """K/V [T, kv_heads, dh] of tokens [r0, r0+T) as KV records (r0 is a
        whole number of blocks); the op's groups take the heads in order."""
        h0 = 0
        for p in kv["parts"]:
            g = self.cfg.attn[p["a"]]
            b0 = r0 // g.lkp
            rec = D.kv_rec(g) * 2
            nblk = p["m"].shape[1]
            for h in range(g.kv_heads):
                recs = P.kv_records(g, k[:, h0 + h], v[:, h0 + h])
                p["m"][h, b0 : b0 + len(recs)] = recs
                p["bo"].sync(TO, len(recs) * rec, (h * nblk + b0) * rec)
            h0 += g.kv_heads

    def attention(self, kv, qe, r0, scale):
        """qe [T, heads, dh] (roped) of tokens [r0, r0+T), multiplied by
        scale on the way in; K/V of tokens [0, r0+T) are in kv's records."""
        at = self.attn[kv["op"]]
        g = self.cfg.attn[at["groups"][0]]
        t, h, dh = qe.shape
        hg = h // len(at["groups"])
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        window = (self.cfg.windows or {}).get(kv["op"], g.window)
        k0 = max(0, q0 - window // g.lkp) if window else 0
        rec = D.kv_rec(g) * 2
        subs = []
        for i, (a, p, kp) in enumerate(zip(at["groups"], at["parts"], kv["parts"])):
            H.q_pack(qe[:, i * hg : (i + 1) * hg], scale, p["qm"])
            p["q"].sync(TO, hg * D.M * dh * 2, 0)
            sub = xrt.bo(kp["bo"], kp["bo"].size() - k0 * rec, k0 * rec)
            at["run"].set_arg(3 + self.slot[a] + 1, sub)
            subs.append(sub)
        at["subs"] = subs
        at["iv"][:] = at["tmpl"].at(nkv=nend - k0, q0=q0, k0=k0)
        at["ib"].sync(TO)
        for p in at["parts"][:-1]:  # _go flushes the last
            p["o"].sync(TO, 64, 0)
        self._go(at["run"], f"attn {kv['op']}", at["parts"][-1]["o"])
        outs = []
        for p in at["parts"]:
            p["o"].sync(FROM)
            outs.append(H.o_unpack(p["om"], t).reshape(t, hg, dh))
        return np.concatenate(outs, axis=1).reshape(t, h * dh)

    # ---- context ----

    def contexts(self):
        return [self.fp]

    def suspend(self):
        """Release the hw_contexts; the weight BOs stay. prefill() resumes."""
        if not self.suspended:
            for c in self.contexts():
                c.close()
            self.suspended = True

    def resume(self):
        if self.suspended:
            for c in self.contexts():
                c.open()
            self.suspended = False
