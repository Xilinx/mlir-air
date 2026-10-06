# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host side of the fused prefill device: one hw_context, the GEMM weight
sites, attention with insts generated per chunk, and the KV records.

A model's prefill subclasses Engine and runs a layer at a time over the
prompt's 128-token chunks: gemm_start and attn_start queue one chunk's op,
gemm_wait and attn_wait collect it, so the host's work on one chunk overlaps
the device's on the next. Norms, rope and residuals are its own (hostops).
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
        self.chunk_bufs, self.chunk_runs, self.chunk_attn = {}, {}, {}

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

    # ---- chunk-parallel GEMMs: per chunk A and C buffers, queued dispatches

    def a_chunk(self, k, c):
        """The A buffer of chunk c for GEMMs of depth k, and its tiled view."""
        key = ("a", k, c)
        if key not in self.chunk_bufs:
            a = self.fp.bo(D.M * k * 2, 0)
            am = view(a, bfloat16, (D.HR, k // D.TK, D.TM, D.TK))
            am[:] = 0
            self.chunk_bufs[key] = (a, am)
        return self.chunk_bufs[key]

    def _chunk_run(self, site, c, act, role):
        s = self.sites[site]
        name = f"g_{s['k']}_{s['npd']}_{act}_{s['wq']}"
        key = (name, role, c)
        if key not in self.chunk_runs:
            if name not in self.chunk_bufs:
                ins = self._insts(name, "g")
                self.chunk_bufs[name] = (self.fp.insts(ins)[0], len(ins))
            ib, n = self.chunk_bufs[name]
            ck = ("c", role, s["npd"], c)
            if ck not in self.chunk_bufs:
                cb = self.fp.bo(D.M * s["npd"] * 2, 2)
                self.chunk_bufs[ck] = (cb, view(cb, bfloat16, (D.M, s["npd"])))
            bos = list(self.dummy)
            bos[0:3] = [self.a_chunk(s["k"], c)[0], s["w"], self.chunk_bufs[ck][0]]
            self.chunk_runs[key] = (self.fp.run(ib, n, bos), self.chunk_bufs[ck])
        return self.chunk_runs[key]

    def gemm_start(self, site, c, act=0, role=None):
        """Queue site's GEMM on chunk c's A buffer; gemm_wait returns its
        output, bf16 [M, n_pad]. Outputs are per (role, chunk), so a role's
        output is overwritten by its next dispatch on that chunk."""
        s = self.sites[site]
        tag = site.split(".")[-1]
        run, (cb, cm) = self._chunk_run(site, c, act, role or tag)
        run.set_arg(4, s["w"])
        cb.sync(TO, 64, 0)
        run.r.start()
        return run, cb, cm, s["npd"], site, tag

    def warm(self, site):
        """Configure the device now rather than in the first prompt: the first
        dispatch on a hw_context pays for it (~130 ms)."""
        self.gemm_wait(self.gemm_start(site, 0), 1)

    def wait(self, pend, nbytes=None):
        run, out, tag = pend[0], pend[1], pend[-1]
        t0 = time.perf_counter()
        st = run.r.wait()
        dt = time.perf_counter() - t0
        self.dev_t += dt
        self.by_op[tag] = self.by_op.get(tag, 0.0) + dt
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"fused prefill {tag}: {st}")
        if nbytes is None:
            out.sync(FROM)
        else:
            out.sync(FROM, nbytes, 0)

    def gemm_wait(self, pend, t):
        self.wait(pend, t * pend[3] * 2)
        return pend[2]

    # ---- attention ----

    def attn_setup(self):
        self.attn = {}
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
            self.attn[op] = dict(
                tmpl=tmpl, groups=groups, hg=pts["base"]["heads"] // len(groups)
            )

    def _attn_part(self, g, hg, s, bos):
        """One group's q and o buffers for hg heads; q is bfp16 for attn_bfp16."""
        n = hg * D.M * g.dh
        q = self.fp.bo(n * 9 // 8 if g.kern == "bfp16" else n * 2, s)
        o = self.fp.bo(n * 2, s + 2)
        if g.kern == "bfp16":
            qm = view(q, np.uint8, (hg, D.M * g.dh * 9 // 8))
        else:
            qm = view(q, bfloat16, (hg, D.M, g.dh))
        qm[:] = 0
        bos[s : s + 3] = [q, self.dummy[s + 1], o]
        om = view(o, bfloat16, (hg, D.M, g.dh))
        return dict(q=q, qm=qm, o=o, om=om, hg=hg)

    @staticmethod
    def _nkv(g, n):
        """attn_bfp16 reads K blocks in pairs; the extra one is masked."""
        return n + n % 2 if g.kern == "bfp16" else n

    def attn_chunk(self, op, c):
        """Chunk c's attention dispatch state for op: per group its q and o
        buffers (_attn_part), and its own insts and run."""
        key = (op, c)
        if key not in self.chunk_attn:
            at = self.attn[op]
            fp, bos, parts = self.fp, list(self.dummy), []
            for a in at["groups"]:
                parts.append(
                    self._attn_part(self.cfg.attn[a], at["hg"], self.slot[a], bos)
                )
            ib, iv = fp.insts(at["tmpl"].at(**at["tmpl"].point))
            self.chunk_attn[key] = dict(
                parts=parts, ib=ib, iv=iv, run=fp.run(ib, iv.size, bos), pt=None
            )
        return self.chunk_attn[key]

    def attn_start(self, kv, c, r0, t):
        """Queue chunk c's attention (tokens [r0, r0+t)) on its q buffers,
        which the caller has filled; attn_wait returns the o views."""
        op = kv["op"]
        at, ch = self.attn[op], self.attn_chunk(op, c)
        g = self.cfg.attn[at["groups"][0]]
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        window = (self.cfg.windows or {}).get(op, g.window)
        k0 = max(0, q0 - window // g.lkp) if window else 0
        rec = D.kv_rec(g) * 2
        for a, p, kp in zip(at["groups"], ch["parts"], kv["parts"]):
            p["q"].sync(TO)
            sk = (id(kp["bo"]), k0)
            sub = kp.setdefault("subs", {}).get(sk)
            if sub is None:
                sub = xrt.bo(kp["bo"], kp["bo"].size() - k0 * rec, k0 * rec)
                kp["subs"][sk] = sub
            ch["run"].set_arg(3 + self.slot[a] + 1, sub)
            p["o"].sync(TO, 64, 0)
        pt = (self._nkv(g, nend - k0), q0, k0)
        if ch["pt"] != pt:
            ch["iv"][:] = at["tmpl"].at(nkv=pt[0], q0=q0, k0=k0)
            ch["ib"].sync(TO)
            ch["pt"] = pt
        ch["run"].r.start()
        return ch["run"], ch, f"attn {op}"

    def attn_wait(self, pend):
        run, ch, tag = pend
        t0 = time.perf_counter()
        st = run.r.wait()
        dt = time.perf_counter() - t0
        self.dev_t += dt
        self.by_op[tag] = self.by_op.get(tag, 0.0) + dt
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"fused prefill {tag}: {st}")
        for p in ch["parts"]:
            p["o"].sync(FROM)
        return [p["om"] for p in ch["parts"]]

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
            nb = -(-k.shape[0] // g.lkp)
            for h in range(g.kv_heads):
                if g.kern == "bfp16":
                    kb = np.ascontiguousarray(k[:, h0 + h]).astype(bfloat16)
                    vb = np.ascontiguousarray(v[:, h0 + h]).astype(bfloat16)
                    H.kv_rec_bfp(kb, vb, 0, kb.shape[0], g.dh, g.lkp, p["m"][h, b0])
                else:
                    recs = P.kv_records(g, k[:, h0 + h], v[:, h0 + h])
                    p["m"][h, b0 : b0 + nb] = recs
                p["bo"].sync(TO, nb * rec, (h * nblk + b0) * rec)
            h0 += g.kv_heads

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
