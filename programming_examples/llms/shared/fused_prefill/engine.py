# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Host side of the fused prefill device: one hw_context, the GEMM weight
sites, attention with insts built per chunk by the compiler's TXN builder, and
the KV records.

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

from air.backend.txn_builder import TxnBuilder

from . import device as D
from . import hostops as H
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
        # the operand format of the device's GEMMs (packing.a_layout), and
        # whether it has ops for chunk pairs (_pair)
        self.gemm_format = self.man.get("gemm_format", "bfp16")
        self.gemm_pairs = self.man.get("gemm_pairs", True)
        self.max_len = max_len
        # first kernel BO argument of each group: the GEMM's, then the
        # attention groups' in config order
        self.slot = {"g": 0}
        self.slot.update({a: 3 + 3 * i for i, a in enumerate(cfg.attn)})
        self.nargs = 3 + 3 * len(cfg.attn)
        H.load(str(self.dir))
        self.dev = xrt.device(0)
        self.fp = Ctx(self.dev, self.dir / "fused_prefill.xclbin")
        # every op's runtime sequence of the device, as C++ that builds its insts
        self.txn = TxnBuilder(str(self.dir / "fused_prefill.txn.h"))
        self.dev_t, self.by_op = 0.0, {}
        self.suspended = False
        self.dummy = [self.fp.bo(4096, i) for i in range(self.nargs)]
        self.abuf, self.sites = {}, {}
        self.chunk_bufs, self.chunk_runs, self.chunk_attn = {}, {}, {}
        self.n_chunks = 0
        self.held = set()

    # ---- device plumbing ----

    def _insts(self, name, *args):
        """The insts of op `name`'s runtime sequence, for its run-time args."""
        return np.asarray(self.txn(f"seg_{name}", *args), np.uint32)

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

    def _a_buf(self, k, chunks):
        """A new zeroed A buffer for GEMMs of depth k on `chunks` chunks, and
        its view [HR, K steps, chunk, tile]."""
        dt, shape = P.a_layout(self.gemm_format, k, chunks)
        a = self.fp.bo(np.prod(shape) * np.dtype(dt).itemsize, 0)
        am = view(a, dt, shape)
        am[:] = 0
        return a, am

    def pack_q4(self, q, scale, mn):
        """Q4NX weights as the device's GEMM reads them (packing.pack_q4)."""
        return P.pack_q4(q, scale, mn, self.gemm_format)

    def pack_bf16(self, w):
        """bf16 weights as the device's GEMM reads them (packing.pack_bf16)."""
        return P.pack_bf16(w, self.gemm_format)

    def gemm_site(self, site, k, n, packed, wq):
        """Load one weight matrix (packed by self.pack_q4 if wq, else
        self.pack_bf16)."""
        fp = self.fp
        npd = P.n_pad(n)
        if k not in self.abuf:
            a, am = self._a_buf(k, 1)
            self.abuf[k] = dict(a=a, am=am[:, : k // D.TK, 0], last=None)
        # the device's K steps past k read zero packets
        kp = packed.shape[1] * D.k_pad(k) // k
        packed = np.pad(packed, ((0, 0), (0, kp - packed.shape[1]), (0, 0)))
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
            name = f"g_{s['k']}_{s['npd']}_{act}_{s['wq']}_0"
            ins = self._insts(name)
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

    def chunk_rows(self, n):
        """(r0, t) of the prompt's chunks. The GEMMs then run on chunks 2i and
        2i + 1 together, one weight read for both (_pair)."""
        chunks = [(r0, min(D.M, n - r0)) for r0 in range(0, n, D.M)]
        self.n_chunks = len(chunks)
        return chunks

    def _pair(self, c):
        """(pair, slot) of chunk c when it runs with a partner, else None:
        always on a device without pair ops. Two chunks run one at a time: a
        pair would keep the device idle until the host has prepared both."""
        if not self.gemm_pairs or self.n_chunks < 3 or c | 1 >= self.n_chunks:
            return None
        return c // 2, c % 2

    def a_chunk(self, k, c):
        """The A buffer of chunk c for GEMMs of depth k, and chunk c's tiled
        view into it. Paired chunks share a buffer, their tiles of each K
        step side by side."""
        pr = self._pair(c)
        key = ("a", k, pr[0] if pr else c, pr is not None)
        if key not in self.chunk_bufs:
            self.chunk_bufs[key] = self._a_buf(k, 1 + (pr is not None))
        a, am = self.chunk_bufs[key]
        return a, am[:, : k // D.TK, pr[1] if pr else 0]

    def _chunk_run(self, site, c, act, role):
        s = self.sites[site]
        pr = self._pair(c)
        p = int(pr is not None)
        name = f"g_{s['k']}_{s['npd']}_{act}_{s['wq']}_{p}"
        g = pr[0] if pr else c
        key = (name, role, g)
        if key not in self.chunk_runs:
            if name not in self.chunk_bufs:
                ins = self._insts(name)
                self.chunk_bufs[name] = (self.fp.insts(ins)[0], len(ins))
            ib, n = self.chunk_bufs[name]
            ck = ("c", role, s["npd"], g, p)
            if ck not in self.chunk_bufs:
                cb = self.fp.bo((1 + p) * D.M * s["npd"] * 2, 2)
                cm = view(cb, bfloat16, (1 + p, D.M, s["npd"]))
                self.chunk_bufs[ck] = (cb, cm)
            bos = list(self.dummy)
            bos[0:3] = [self.a_chunk(s["k"], c)[0], s["w"], self.chunk_bufs[ck][0]]
            self.chunk_runs[key] = (self.fp.run(ib, n, bos), self.chunk_bufs[ck])
        return self.chunk_runs[key]

    def gemm_start(self, site, c, act=0, role=None):
        """Queue site's GEMM on chunk c's A buffer; gemm_wait returns its
        output, bf16 [M, n_pad]. Outputs are per (role, chunk), so a role's
        output is overwritten by its next dispatch on that chunk. A paired
        chunk's dispatch starts with its partner's, so a stage starts all its
        chunks before it waits on any."""
        s = self.sites[site]
        tag = site.split(".")[-1]
        run, (cb, cm) = self._chunk_run(site, c, act, role or tag)
        pr = self._pair(c)
        slot = pr[1] if pr else 0
        if pr and slot == 0:
            self.held.add(run)
        else:
            self.held.discard(run)
            run.set_arg(4, s["w"])
            for part in range(len(cm)):
                cb.sync(TO, 64, part * cm[0].nbytes)
            run.r.start()
        return run, cb, cm[slot], s["npd"], site, slot * cm[0].nbytes, tag

    def warm(self, site):
        """Configure the device now rather than in the first prompt: the first
        dispatch on a hw_context pays for it (~130 ms)."""
        self.gemm_wait(self.gemm_start(site, 0), 1)

    def wait(self, pend, nbytes=None, offset=0):
        run, out, tag = pend[0], pend[1], pend[-1]
        assert run not in self.held, f"{tag}: its pair was never started"
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
            out.sync(FROM, nbytes, offset)

    def gemm_wait(self, pend, t):
        self.wait(pend, t * pend[3] * 2, pend[5])
        return pend[2]

    # ---- attention ----

    def attn_setup(self):
        self.attn = {}
        for op, groups in D.attn_ops(self.cfg).items():
            self.attn[op] = dict(
                seq=self.man["attn"][op],
                groups=groups,
                hg=self.man["heads"] // len(groups),
            )

    def _attn_part(self, g, hg, s, bos):
        """One group's q (bfp16) and o buffers for hg heads."""
        n = hg * D.M * g.dh
        q = self.fp.bo(n * 9 // 8, s)
        o = self.fp.bo(n * 2, s + 2)
        qm = view(q, np.uint8, (hg, D.M * g.dh * 9 // 8))
        qm[:] = 0
        bos[s : s + 3] = [q, self.dummy[s + 1], o]
        om = view(o, bfloat16, (hg, D.M, g.dh))
        return dict(q=q, qm=qm, o=o, om=om, hg=hg, g=g)

    @staticmethod
    def _nkv(g, n):
        """The cores read K blocks in pairs; the extra one is masked."""
        return n + n % 2

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
            # the stream's length does not depend on (nkv, q0, k0)
            ib, iv = fp.insts(self._insts(at["seq"], 1, 0, 0))
            self.chunk_attn[key] = dict(
                parts=parts, ib=ib, iv=iv, run=fp.run(ib, iv.size, bos), pt=None
            )
        return self.chunk_attn[key]

    def attn_start(self, kv, c, r0, t):
        """Queue chunk c's attention (tokens [r0, r0+t)) on its q buffers,
        which the caller has filled; attn_wait returns the parts."""
        op = kv["op"]
        at, ch = self.attn[op], self.attn_chunk(op, c)
        g = self.cfg.attn[at["groups"][0]]
        q0 = r0 // g.lkp
        nend = -(-(r0 + t) // g.lkp)
        window = (self.cfg.windows or {}).get(op)
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
            ch["iv"][:] = self._insts(at["seq"], *pt)
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
        return ch["parts"]

    def kv_bo(self, op, kv_heads=None):
        """The KV records of one layer for attention op `op`: per group, per
        KV head, a block per lkp tokens. Given the model's kv_heads, groups
        that take the same KV heads share their records."""
        groups = D.attn_ops(self.cfg)[op]
        parts, made = [], {}
        for i, a in enumerate(groups):
            g = self.cfg.attn[a]
            # the cores read an even number of blocks, so one past the last
            last = -(-self.max_len // g.lkp)
            nblk = g.max_blocks if g.kv_heads > 1 else last + 1
            assert nblk > last, (nblk, g.lkp, self.max_len)
            shape = (g.kv_heads, nblk, D.kv_rec(g))
            h0 = i * kv_heads // len(groups) if kv_heads else None
            key = (h0, g._replace(col=0)) if kv_heads else a
            if key not in made:
                bo = self.fp.bo(int(np.prod(shape)) * 2, self.slot[a] + 1)
                made[key] = (bo, view(bo, bfloat16, shape))
            bo, m = made[key]
            parts.append(dict(bo=bo, m=m, a=a))
        return dict(op=op, parts=parts)

    def kv_append(self, kv, r0, k, v):
        """K/V [T, kv_heads, dh] of tokens [r0, r0+T) as KV records (r0 is a
        whole number of blocks); each of the op's groups takes the KV heads of
        its q heads."""
        groups, done = len(kv["parts"]), set()
        for i, p in enumerate(kv["parts"]):
            if id(p["bo"]) in done:
                continue
            done.add(id(p["bo"]))
            h0 = i * k.shape[1] // groups
            g = self.cfg.attn[p["a"]]
            b0 = r0 // g.lkp
            rec = D.kv_rec(g) * 2
            nblk = p["m"].shape[1]
            nb = -(-k.shape[0] // g.lkp)
            for h in range(g.kv_heads):
                kb = np.ascontiguousarray(k[:, h0 + h]).astype(bfloat16)
                vb = np.ascontiguousarray(v[:, h0 + h]).astype(bfloat16)
                t, dst = kb.shape[0], p["m"][h, b0]
                H.kv_rec_bfp(kb, vb, 0, t, g.dh, g.lkp, dst)
                p["bo"].sync(TO, nb * rec, (h * nblk + b0) * rec)

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
