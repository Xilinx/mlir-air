# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3-Coder-30B-A3B prefill on the shared fused prefill device
(fused_spec.py): every projection and expert GEMM and the attention are ops
of the device; the norms, RoPE, router and GLU run on the host.

A layer is its attention over the prompt's 128-token chunks, then the MoE over
every row: the q | k | v GEMM, q / k norm and RoPE on the host, attention, the
o GEMM; the router on the host, then per expert its up | gate GEMM, the GLU on
the host and its down GEMM. The GEMMs read the bundle's Q4_K blocks: the dense
ones from this engine's copy, the experts from a WE buffer laid out as the
decode's.

prefill(ids) keeps each layer's K and V for the decode.
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import moe_block as MB  # noqa: E402
import qwen3_coder_30b_a3b_q4k_weights as W  # noqa: E402
from air_examples.llms.qwen3_coder_30b_a3b_q4k import fused_spec as S  # noqa: E402
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import hostops as H  # noqa: E402
from air_examples.llms.shared.fused_prefill import packing as P  # noqa: E402
from air_examples.llms.shared.fused_prefill import txn  # noqa: E402
from air_examples.llms.shared.fused_prefill.engine import (  # noqa: E402
    FROM,
    TO,
    Engine,
)

HOST_THREADS = 16
# dispatches of a layer's up | gate GEMMs (and of its down GEMMs)
MOE_GROUPS = 8
BB = D.Q4K_BYTES
GROUP = 4 * BB  # one column block of a 128-row group
E_GROUPS = MB.RG_GU * MB.NCB_X + MB.RG_DN * MB.NCB_H  # an expert's, in WE
GU, DN = "gu", "down"
NCB_O = S.DQ // D.Q4K_COLS  # column blocks of the o projection
# WE needs this much past its last expert: the down GEMM's padded panels, and
# the device's K rounded up past the expert width
WE_SLACK = (S.npad(S.D_MODEL) - S.D_MODEL) // 128 * MB.NCB_H * GROUP + (
    D.k_pad(S.I) - S.I
) // D.Q4K_COLS * GROUP


def row_groups(blocks):
    """A projection's row blocks [rows / 32, column blocks, BB] -> its row
    groups' column blocks [rows / 128 * column blocks, 4, BB]."""
    nb, ncb, _ = blocks.shape
    t = blocks.reshape(nb // 4, 4, ncb, BB).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t).reshape(-1, 4, BB)


def experts_we(w, layer):
    """A layer's experts as the decode's WE holds them, [E, groups, 4, BB]."""
    wgu, wdn, _ = MB.weights(w, layer)
    return np.concatenate(
        [wgu.reshape(MB.E, -1, 4, BB), wdn.reshape(MB.E, -1, 4, BB)], axis=1
    )


class Q3CPrefill(Engine):
    """`we`: (BO, byte offset of layers[0]'s experts) of a WE buffer the
    decode holds; without it the engine holds every layer's experts itself,
    and with "stream" one layer's, copied in before the layer's MoE."""

    def __init__(self, build_dir, w, layers=None, we=None):
        H.build(build_dir, HOST_THREADS)
        super().__init__(build_dir, S.CFG, S.max_len)
        self.w = w
        self.layers = list(range(W.N_LAYERS)) if layers is None else list(layers)
        inv = 1.0 / W.THETA ** (np.arange(0, W.DH, 2) / W.DH)
        ang = np.arange(S.max_len)[:, None] * inv[None]
        self.cs = np.ascontiguousarray(np.cos(ang), np.float32)
        self.sn = np.ascontiguousarray(np.sin(ang), np.float32)
        self.times = {}
        self._load(we)

    # ---- weights ----

    def _site(self, name, blocks, k, n):
        self.gemm_site(name, k, n, row_groups(blocks), 3)

    def _raw(self, L, part, ncb):
        return np.asarray(self.w.raw(W.name(L, part))).reshape(-1, ncb, BB)

    def _load(self, we):
        w = self.w
        self.p = []
        lb = MB.E * E_GROUPS * GROUP
        self.stream = we == "stream"
        if we is None or self.stream:
            import pyxrt as xrt

            held = 1 if self.stream else len(self.layers)
            bo = xrt.ext.bo(self.dev, held * lb + WE_SLACK)
            m = np.frombuffer(bo.map(), np.uint8)
            m[held * lb :] = 0
            we = (bo, 0)
        else:
            m = None
        self.we_bo, self.we_off, self.we_map = we[0], we[1], m
        self.wsub = {}
        sa = "self_attn."
        for j, L in enumerate(self.layers):
            raw = lambda part, n: self._raw(L, part, n)  # noqa: E731
            vec = lambda part: np.ascontiguousarray(  # noqa: E731
                w.vec(L, part), np.float32
            )
            n1, n2 = w.norms(L)
            qkv = np.concatenate(
                [raw(sa + f"{n}_proj.weight", MB.NCB_X) for n in "qkv"]
            )
            self._site(f"{j}.qkv", qkv, *S.SHAPES["qkv"])
            self._site(f"{j}.o", raw(sa + "o_proj.weight", NCB_O), *S.SHAPES["o"])
            self.p.append(
                dict(
                    n1=np.asarray(n1, np.float32),
                    n2=np.asarray(n2, np.float32),
                    router=torch.from_numpy(
                        np.ascontiguousarray(w.router(L).T, np.float32)
                    ),
                    qn=vec(sa + "q_norm.weight"),
                    kn=vec(sa + "k_norm.weight"),
                )
            )
            if m is not None and not self.stream:
                m[j * lb : (j + 1) * lb] = experts_we(w, L).reshape(-1)
        if m is not None and not self.stream:
            self.we_bo.sync(TO)
        self.attn_setup()
        self.kvb = {j: self.kv_bo("a") for j in range(len(self.layers))}
        self.parts = {}
        self.warm("0.qkv")

    def _wbo(self, j, e, part):
        """The sub-buffer of WE that expert e of layer j's part reads."""
        j = 0 if self.stream else j
        key = (j, e, part)
        if key not in self.wsub:
            import pyxrt as xrt

            k, n = S.SHAPES[part]
            ncb = k // D.Q4K_COLS
            # the last panel reads the device's K, rounded up
            groups = (S.npad(n) // 128 - 1) * ncb + D.k_pad(k) // D.Q4K_COLS
            off = self.we_off + (j * MB.E + e) * E_GROUPS * GROUP
            off += MB.RG_GU * MB.NCB_X * GROUP if part == DN else 0
            self.wsub[key] = xrt.bo(self.we_bo, groups * GROUP, off)
        return self.wsub[key]

    # ---- a prompt ----

    def _t(self, k, t0):
        self.times[k] = self.times.get(k, 0.0) + time.perf_counter() - t0

    def prefill(self, ids):
        """The prompt's hidden states after the last layer, float32 [n, D]."""
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        x = np.ascontiguousarray(self.w.embed(ids), np.float32)
        chunks = self.chunk_rows(n)
        self.kv = {}
        for j in range(len(self.layers)):
            pend = self._attn_all(j, x, chunks)
            self._moe(j, x, pend, chunks)
        self.n = n
        return x

    def kv_rows(self):
        """Per layer (K, V) [n, NKV * DH]: K normed and roped, V raw."""
        return [
            (np.concatenate(self.kv[j][0]), np.concatenate(self.kv[j][1]))
            for j in range(len(self.layers))
        ]

    def _attn_all(self, j, x, chunks):
        kvb = self.kvb[j]
        t0 = time.perf_counter()
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(S.D_MODEL, c)
            H.rms_tile(x[r0 : r0 + t], self.p[j]["n1"], W.EPS, a[1], D.TM, D.TK)
            a[0].sync(TO)
        self._t("host", t0)
        pq = [self.gemm_start(f"{j}.qkv", c) for c in range(len(chunks))]
        att = []
        self.kv[j] = ([], [])
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pq[c], t)
            t0 = time.perf_counter()
            self._qk_post(j, kvb, c, r0, t, qkv)
            self._t("attn_pre", t0)
            att.append(self.attn_start(kvb, c, r0, t))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            parts = self.attn_wait(att[c])
            t0 = time.perf_counter()
            a = self.a_chunk(S.DQ, c)
            h0 = 0
            for p in parts:
                self.o_read(p, t, h0, S.DQ, a[1])
                h0 += p["hg"]
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{j}.o", c))
            self._t("attn_post", t0)
        return pend

    def _qk_post(self, j, kvb, c, r0, t, qkv):
        """q (normed, roped) into chunk c's attention q buffers; K (normed,
        roped) and V into the layer's records and the kept rows."""
        DQ, DH, p = S.DQ, S.DH, self.p[j]
        rope = (self.cs[r0 : r0 + t], self.sn[r0 : r0 + t], DH)
        ch = self.attn_chunk(kvb["op"], c)
        h0 = 0
        for part in ch["parts"]:
            hg = part["hg"]
            qm = self.q_stage(part)
            H.head_post(
                qkv, t, h0 * DH, hg, DH, None, p["qn"], W.EPS, rope, 1.0, qm,
                D.M * DH, DH,
            )  # fmt: skip
            if t < D.M:
                qm[:, t:] = 0
            self.q_write(part, qm)
            h0 += hg
        nkv = S.KV_HEADS * DH
        kb = np.empty((t, nkv), bfloat16)
        vb = np.empty((t, nkv), bfloat16)
        H.head_post(
            qkv, t, DQ, S.KV_HEADS, DH, None, p["kn"], W.EPS, rope, 1.0, kb,
            DH, nkv,
        )  # fmt: skip
        H.head_post(
            qkv, t, DQ + nkv, S.KV_HEADS, DH, None, None, W.EPS, None, 1.0,
            vb, DH, nkv,
        )  # fmt: skip
        self.kv[j][0].append(kb)
        self.kv[j][1].append(vb)
        h0 = 0
        for part in kvb["parts"]:
            g = self.cfg.attn[part["a"]]
            b0, rec = r0 // g.lkp, self.kv_rec(g)
            nblk = part["m"].shape[1]
            nb = -(-t // g.lkp)
            for h in range(g.kv_heads):
                self.kv_write(kb, vb, (h0 + h) * DH, t, g, part["m"][h, b0])
                part["bo"].sync(TO, nb * rec * 2, (h * nblk + b0) * rec * 2)
            h0 += g.kv_heads

    # ---- MoE ----

    def _moe(self, j, x, pend, chunks):
        """The attention's residual add, then the MoE, in place on x."""
        for c, (r0, t) in enumerate(chunks):
            o = self.gemm_wait(pend[c], t)
            t0 = time.perf_counter()
            H.add_rms_tile(x[r0 : r0 + t], o, t, None, W.EPS, None, None, D.TM, D.TK)
            self._t("host", t0)
        t0 = time.perf_counter()
        if self.stream:
            lb = MB.E * E_GROUPS * GROUP
            self.we_map[:lb] = experts_we(self.w, self.layers[j]).reshape(-1)
            self.we_bo.sync(TO, lb, 0)
        self._t("stream", t0)
        t0 = time.perf_counter()
        p = self.p[j]
        h = H.round_bf16(H.rms(x, p["n2"], W.EPS))
        ht = torch.from_numpy(h)
        # softmax over every expert, the top k renormalized, their weights bf16
        probs = (ht @ p["router"]).softmax(-1)
        top_v, top_i = probs.topk(W.TOP_K, -1)
        top_v = (top_v / top_v.sum(-1, keepdim=True)).to(torch.bfloat16).float()
        self._t("router", t0)
        x += self._experts(j, h, top_i.numpy(), top_v.numpy())

    # A layer's experts run in MOE_GROUPS dispatches of their up | gate GEMMs
    # and as many of their down GEMMs, each dispatch the op's runtime sequence
    # once per block of an expert's rows (txn.concat), on that block's part of
    # one A, one C and the WE buffer. The host's GLU of a group's up | gate
    # overlaps the device's up | gate of the next.

    def _part(self, part):
        """part's GEMM: its op's sequence for one chunk and a pair (Copies),
        its A and C buffers by block, and two dispatch slots."""
        if part not in self.parts:
            k, n = S.SHAPES[part]
            npd = S.npad(n)
            dt, shape = P.a_layout(self.gemm_format, k, 1)
            abytes = [
                int(np.prod(shape)) * np.dtype(dt).itemsize * (1 + q) for q in (0, 1)
            ]
            self.parts[part] = dict(
                k=k,
                npd=npd,
                steps=k // D.TK,
                cp=[txn.Copies(self._insts(f"g_{k}_{npd}_0_3_{q}")) for q in (0, 1)],
                abytes=abytes,
                cbytes=[(1 + q) * D.M * npd * 2 for q in (0, 1)],
                cap=[0, 0],  # blocks of one chunk / a pair the buffers hold
                runs=[None, None],
            )
        return self.parts[part]

    def _part_bufs(self, g, nb):
        """g's A and C hold at least nb = [single, pair] blocks: pairs first,
        then single blocks, each block always the one layout so that its A's
        padding stays zero."""
        if all(nb[q] <= g["cap"][q] for q in (0, 1)):
            return
        g["cap"] = [max(nb[q], g["cap"][q]) for q in (0, 1)]
        pa, pc = g["cap"][1] * g["abytes"][1], g["cap"][1] * g["cbytes"][1]
        g["a0"], g["c0"] = [pa, 0], [pc, 0]  # first block's offset, by pair
        g["a"] = self.fp.bo(pa + g["cap"][0] * g["abytes"][0], 0)
        g["am"] = np.frombuffer(g["a"].map(), np.uint8)
        g["am"][:] = 0
        g["a"].sync(TO)
        g["c"] = self.fp.bo(pc + g["cap"][0] * g["cbytes"][0], 2)
        g["cm"] = np.frombuffer(g["c"].map(), np.uint8)

    def _a_tiles(self, g, q, i):
        """Block i (of pair-ness q)'s A tiles, one view per chunk."""
        dt, shape = P.a_layout(self.gemm_format, g["k"], 1 + q)
        o = g["a0"][q] + i * g["abytes"][q]
        v = g["am"][o : o + g["abytes"][q]].view(dt).reshape(shape)
        return [v[:, : g["steps"], hh] for hh in range(1 + q)]

    def _c_rows(self, g, q, i):
        o = g["c0"][q] + i * g["cbytes"][q]
        return g["cm"][o : o + g["cbytes"][q]].view(bfloat16).reshape(-1, g["npd"])

    def _dispatch(self, part, s, blocks, j):
        """Start part's GEMM on blocks [(expert, pair, index)] in dispatch
        slot s; returns what _collect needs."""
        g = self._part(part)
        b0 = MB.RG_GU * MB.NCB_X * GROUP if part == DN else 0
        runs = []
        for q in (1, 0):
            off = [
                (
                    g["a0"][q] + i * g["abytes"][q],
                    self.we_off
                    + ((0 if self.stream else j) * MB.E + e) * E_GROUPS * GROUP
                    + b0,
                    g["c0"][q] + i * g["cbytes"][q],
                )
                for e, p, i in blocks
                if p == q
            ]
            runs.append((g["cp"][q], np.asarray(off, np.int64).reshape(-1, 3)))
            # flush each chunk's first output line: unflushed, it can read back
            # with the previous dispatch's bytes
            for _, _, c in off:
                for hh in range(1 + q):
                    g["c"].sync(TO, 64, c + hh * D.M * g["npd"] * 2)
        nw = sum(c.body.size * len(o) for c, o in runs) + 4
        slot = g["runs"][s]
        if slot is None or slot["cap"] < nw:
            ib, view_ = self.fp.insts(np.zeros(nw, np.uint32))
            if slot is None:
                bos = list(self.dummy)
                bos[0:3] = [g["a"], self.we_bo, g["c"]]
                slot = dict(run=self.fp.run(ib, nw, bos))
                g["runs"][s] = slot
            # a larger stream: the slot's one run (Ctx keeps every run it
            # made) takes the new instruction buffer, the old one goes
            slot["run"].set_arg(1, ib)
            slot.update(ib=ib, view=view_, cap=nw)
        txn.concat(runs, slot["view"])
        slot["ib"].sync(TO, nw * 4, 0)
        run = slot["run"]
        run.set_arg(2, nw)
        run.set_arg(3, g["a"])
        run.set_arg(4, self.we_bo)
        run.set_arg(5, g["c"])
        t0 = time.perf_counter()
        run.r.start()
        return dict(run=run, t0=t0, tag=part)

    def _collect(self, d):
        r = d["run"].r.wait()
        dt = time.perf_counter() - d["t0"]
        self.dev_t += dt
        self.by_op[d["tag"]] = self.by_op.get(d["tag"], 0.0) + dt
        if "COMPLETED" not in str(r):
            raise RuntimeError(f"fused prefill {d['tag']}: {r}")

    def _experts(self, j, h, top_i, top_v):
        """The routed experts' weighted sum of h [n, D]."""
        n, M = h.shape[0], D.M
        out = np.zeros((n, S.D_MODEL), np.float32)
        zero = np.zeros(S.D_MODEL, np.float32)
        blocks = []  # (expert, rows, weights, pair)
        # the busiest experts first
        flat = top_i.ravel()
        counts = np.bincount(flat, minlength=MB.E)
        # each expert's (row, slot)s in row order, as np.nonzero(top_i == e)
        by_e = np.argsort(flat, kind="stable")
        first = np.concatenate([[0], np.cumsum(counts)])
        for e in np.argsort(-counts, kind="stable"):
            if counts[e] == 0:
                break
            f = by_e[first[e] : first[e + 1]]
            rows = f // top_i.shape[1]
            wt = top_v.ravel()[f]
            for r0 in range(0, len(rows), 2 * M):
                r = rows[r0 : r0 + 2 * M]
                blocks.append((e, r, wt[r0 : r0 + 2 * M], int(len(r) > M)))
        # groups of about equal rows, in block order (the order of the sums)
        cut = np.cumsum([len(b[1]) for b in blocks])
        ng = max(1, min(MOE_GROUPS, len(blocks)))
        edges = np.searchsorted(cut, cut[-1] * np.arange(1, ng) / ng, "right")
        groups = [
            blocks[a:b] for a, b in zip([0, *edges], [*edges, len(blocks)]) if b > a
        ]
        # each block's place in its part's buffers: pairs, then single blocks,
        # numbered in group order so that a group's are contiguous
        nb = [sum(1 - b[3] for b in blocks), sum(b[3] for b in blocks)]
        gu, dn = self._part(GU), self._part(DN)
        self._part_bufs(gu, nb)
        self._part_bufs(dn, nb)
        idx, ctr = [], [0, 0]
        for grp in groups:
            ii = []
            for b in grp:
                ii.append(ctr[b[3]])
                ctr[b[3]] += 1
            idx.append(ii)

        def _runs(grp, ii):
            """A group's blocks as _dispatch takes them, its pairs first: each
            of its two sequences is then one run of copies."""
            return sorted(
                ((b[0], b[3], i) for b, i in zip(grp, ii)), key=lambda r: -r[1]
            )

        def sync(g, buf, grp, ii, d):
            """A group's part of g's A or C, each pair-ness's in one sync."""
            for q in (0, 1):
                sel = [i for b, i in zip(grp, ii) if b[3] == q]
                if sel:
                    sz = g[f"{buf}bytes"][q]
                    o = g[f"{buf}0"][q] + min(sel) * sz
                    g[buf].sync(d, len(sel) * sz, o)

        def start_gu(gi):
            grp, ii = groups[gi], idx[gi]
            t0 = time.perf_counter()
            jobs = []
            for (e, rows, _, q), i in zip(grp, ii):
                am = self._a_tiles(gu, q, i)
                jobs += [(rows[hh * M : (hh + 1) * M], am[hh]) for hh in range(1 + q)]
            H.tile_a_rows_n(h, jobs, D.TM, D.TK)
            sync(gu, "a", grp, ii, TO)
            self._t("experts_host", t0)
            return self._dispatch(GU, gi % 2, _runs(grp, ii), j)

        def start_dn(gi):
            grp, ii = groups[gi], idx[gi]
            t0 = time.perf_counter()
            sync(gu, "c", grp, ii, FROM)
            jobs = []
            for (e, rows, _, q), i in zip(grp, ii):
                y, am, t = self._c_rows(gu, q, i), self._a_tiles(dn, q, i), len(rows)
                for hh in range(1 + q):
                    r0, r1 = hh * M, min(t, (hh + 1) * M)
                    jobs.append((y[r0:r1], r1 - r0, am[hh]))
            H.glu_tile_a_n(jobs, S.I, D.TM, D.TK)
            sync(dn, "a", grp, ii, TO)
            self._t("experts_host", t0)
            return self._dispatch(DN, gi % 2, _runs(grp, ii), j)

        def finish_dn(gi, d):
            self._collect(d)
            t0 = time.perf_counter()
            grp, ii = groups[gi], idx[gi]
            sync(dn, "c", grp, ii, FROM)
            jobs = [
                (self._c_rows(dn, q, i)[: len(rows)], rows, wt)
                for (e, rows, wt, q), i in zip(grp, ii)
            ]
            H.scatter_add_rows_n(out, jobs, zero, S.D_MODEL)
            self._t("experts_host", t0)

        # device order: gu 0, gu 1, dn 0, gu 2, dn 1, ...: the host's GLU of
        # group g runs while the device does gu g + 1
        gus = {0: start_gu(0)}
        prev = None
        for gi in range(len(groups)):
            if gi + 1 < len(groups):
                gus[gi + 1] = start_gu(gi + 1)
            self._collect(gus.pop(gi))
            d = start_dn(gi)
            if prev is not None:
                finish_dn(*prev)
            prev = (gi, d)
        finish_dn(*prev)
        return out


def main():
    """The prefill of a prompt against the CPU reference: the last position's
    logits, by cosine and next token. The experts are streamed in a layer at a
    time unless --resident."""
    import argparse
    import os
    import tempfile

    import filelock
    import reference as R

    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", help="the fused prefill build (fused_build.py)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-len", type=int, default=0, help="repeat to N tokens")
    ap.add_argument("--layers", type=int, default=W.N_LAYERS)
    ap.add_argument("--resident", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    w = W.Qwen3CoderQ4k()
    ids = w.tokenizer().encode(args.prompt).ids
    if args.prompt_len:
        ids = (ids * (args.prompt_len // len(ids) + 1))[: args.prompt_len]
    layers = list(range(args.layers))
    t0 = time.perf_counter()
    # the device is held from the first context to the last dispatch
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        pf = Q3CPrefill(args.build_dir, w, layers, None if args.resident else "stream")
        print(f"load {time.perf_counter() - t0:.1f} s")
        for _ in range(args.runs):
            pf.times, pf.by_op, pf.dev_t = {}, {}, 0.0
            t0 = time.perf_counter()
            x = pf.prefill(ids)
            dt = time.perf_counter() - t0
            ops = {k: round(v, 3) for k, v in sorted(pf.by_op.items())}
            tms = {k: round(v, 3) for k, v in sorted(pf.times.items())}
            print(f"{len(ids)} tokens: {dt:.3f} s; device {ops}; host {tms}")
        del pf
    if args.no_ref:
        return 0
    fn = torch.from_numpy(np.asarray(w.final_norm(), np.float32))
    head = torch.from_numpy(w.lm_head)
    got = R.rmsnorm(torch.from_numpy(x[-1]), fn) @ head.T
    ref = R.Reference(w, layers)
    want = R.rmsnorm(ref.forward(ids, lm=False)[-1], fn) @ head.T
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0)
    a, b = int(got.argmax()), int(want.argmax())
    print(f"{len(ids)} tokens: logits cos {cos:.5f}; next token {a} vs ref {b}")
    ok = a == b and cos > 0.98
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
