# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.8-27B prefill on the shared fused prefill device (fused_spec.py),
as Qwen3.6-35B-A3B's (qwen36_35b_a3b_q4nx/prefill.py) runs: every projection
GEMM, and the full-attention layers' attention, are ops of the device; the
rest runs on the host.

A layer is its mixer over the prompt's 128-token chunks, then the MLP. Gated
DeltaNet: the q | k | v and z GEMMs, alpha / beta on the host, the conv, the
delta rule on an FP32 state and the gated norm on the host (kernels/
q38_host.c), the out GEMM. Full attention: the q | gate and k | v GEMMs, q / k
norm and RoPE on the host, attention, the output gate on the host, the o GEMM.
MLP: the interleaved up | gate GEMM in three pieces, the GLU on the host into
the down GEMM's A, the down GEMM. The GEMMs read the bundle's Q4_K blocks from
this engine's copy.

prefill(ids) keeps what the decode takes: per attention layer K and V, per
Gated DeltaNet layer the conv history and the state.
"""

import ctypes
import subprocess
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

import mlp_block as MB  # noqa: E402
import qwen38_27b_q4nx_weights as W  # noqa: E402
from air_examples.llms.qwen38_27b_q4nx import fused_spec as S  # noqa: E402
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import hostops as H  # noqa: E402
from air_examples.llms.shared.fused_prefill.engine import TO, Engine  # noqa: E402

HOST_C = HERE / "kernels" / "q38_host.c"
HOST_DEPS = (HOST_C, HERE.parent / "qwen35_0_8b_q4nx" / "kernels" / "q35_host.c")
HOST_THREADS = 16
BB = D.Q4K_BYTES
_p, _i = ctypes.c_void_p, ctypes.c_int


def build_host(out):
    """hostops with HOST_THREADS row threads, then the model's host library."""
    flags = ["gcc", "-O3", "-march=native", "-ffp-contract=off", "-fopenmp"]
    flags += ["-fno-trapping-math", "-shared", "-fPIC", f"-DNT={HOST_THREADS}"]
    for src, so in (
        (HERE.parents[0] / "shared/fused_prefill/kernels/hostops.c", "libhostops.so"),
        (HOST_C, "libq38host.so"),
    ):
        subprocess.run(flags + [str(src), "-o", str(Path(out) / so), "-lm"], check=True)


def row_groups(blocks):
    """A projection's row blocks [rows / 32, column blocks, BB] -> its row
    groups' column blocks [rows / 128 * column blocks, 4, BB]."""
    nb, ncb, _ = blocks.shape
    t = blocks.reshape(nb // 4, 4, ncb, BB).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(t).reshape(-1, 4, BB)


class Qwen38Prefill(Engine):
    def __init__(self, build_dir, w, layers=None, threads=HOST_THREADS):
        so = Path(build_dir) / "libq38host.so"
        if not so.exists() or so.stat().st_mtime < max(
            p.stat().st_mtime for p in HOST_DEPS
        ):
            build_host(build_dir)
        super().__init__(build_dir, S.CFG, S.max_len)
        self.w = w
        self.layers = list(range(W.N_LAYERS)) if layers is None else list(layers)
        self.q = ctypes.CDLL(str(so))
        self.q.gdn_chunk_z.argtypes = [_p, _i, _p, _i, _p, _i, _i] + [_p] * 9 + [_i]
        self.q.o_gate_tile.argtypes = [_p] + [_i] * 9 + [_p, _i, _p, _i]
        self.q.glu_tile.argtypes = [_p] + [_i] * 7 + [_p, _i]
        self.nt = threads
        inv = 1.0 / W.THETA ** (np.arange(0, W.ROPE_DIM, 2) / W.ROPE_DIM)
        ang = np.arange(S.max_len)[:, None] * inv[None]
        self.cs = np.ascontiguousarray(np.cos(ang), np.float32)
        self.sn = np.ascontiguousarray(np.sin(ang), np.float32)
        self.times = {}
        self._load()

    # ---- weights ----

    def _raw(self, L, part, ncb):
        return np.asarray(self.w.raw(self.w.name(L, part))).reshape(-1, ncb, BB)

    def _load(self):
        w, ncb_x, ncb_q, ncb_i = self.w, S.D_MODEL // 256, S.DQ // 256, S.I // 256
        self.p = []
        for j, L in enumerate(self.layers):
            raw = lambda part, n: self._raw(L, part, n)  # noqa: E731
            vec = lambda part: np.ascontiguousarray(  # noqa: E731
                w.vec(L, part), np.float32
            )
            n1, n2 = w.norms(L)
            p = dict(n1=np.asarray(n1, np.float32), n2=np.asarray(n2, np.float32))
            site = lambda name, blocks: self.gemm_site(  # noqa: E731
                f"{j}.{name}", *S.SHAPES[name], row_groups(blocks), 3
            )
            if W.is_attention(L):
                sa = "self_attn."
                site("qg", raw(sa + "q_proj.weight", ncb_x))
                site(
                    "kv",
                    np.concatenate([raw(sa + f"{n}_proj.weight", ncb_x) for n in "kv"]),
                )
                site("o", raw(sa + "o_proj.weight", ncb_q))
                p.update(qn=vec(sa + "q_norm.weight"), kn=vec(sa + "k_norm.weight"))
            else:
                la = "linear_attn."
                site("qkv", raw(la + "qkv_proj.weight", ncb_x))
                site("z", raw("self_attn.gate_proj.weight", ncb_x))
                site("lin_out", raw(la + "ssm_out_proj.weight", ncb_q))
                al = vec(la + "ssm_alpha_proj.bf16.weight")  # [NVH, D]
                be = vec(la + "ssm_beta_proj.bf16.weight")
                p.update(
                    ab=torch.from_numpy(
                        np.ascontiguousarray(np.concatenate([be, al]).T)
                    ),
                    conv=vec(la + "ssm_conv1d.weight"),  # [4, QKV]
                    nega=vec(la + "ssm_a").reshape(-1),
                    dtb=vec(la + "ssm_dt.bias").reshape(-1),
                    norm=vec(la + "ssm_norm.weight").reshape(-1),
                )
            m_ = "mlp."
            gu = MB.interleave(
                raw(m_ + "up_proj.weight", ncb_x), raw(m_ + "gate_proj.weight", ncb_x)
            )
            for i, (c0, c1) in enumerate(S.GU_PIECES):
                self.gemm_site(
                    f"{j}.gu{i}",
                    *S.SHAPES[f"gu{i}"],
                    np.ascontiguousarray(gu[c0 // 128 : c1 // 128]).reshape(-1, 4, BB),
                    3,
                )
            site("down", raw(m_ + "down_proj.weight", ncb_i))
            self.p.append(p)
        self.attn_setup()
        self.kvb = {
            j: self.kv_bo("a") for j, L in enumerate(self.layers) if W.is_attention(L)
        }
        self.warm(f"0.gu0")

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
        self.kv, self.conv, self.state = {}, {}, {}
        for j, L in enumerate(self.layers):
            h = self._norm_in(j, x, chunks)
            if W.is_attention(L):
                pend = self._attn_all(j, chunks)
            else:
                pend = self._gdn_all(j, h, chunks)
            self._mlp(j, x, pend, chunks)
        self.n = n
        return x

    def _norm_in(self, j, x, chunks):
        """rms(x) * w into each chunk's A, as the device and the host take
        it: bf16 rows [n, D]."""
        t0 = time.perf_counter()
        h = H.rms(x, self.p[j]["n1"], W.EPS).astype(bfloat16)
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(S.D_MODEL, c)
            H.tile_a(h[r0 : r0 + t].astype(np.float32), a[1], D.TM, D.TK)
            a[0].sync(TO)
        self._t("host", t0)
        return h

    def _gdn_all(self, j, h, chunks):
        p, nc = self.p[j], len(chunks)
        pl = [self.gemm_start(f"{j}.qkv", c) for c in range(nc)]
        pz = [self.gemm_start(f"{j}.z", c) for c in range(nc)]
        t0 = time.perf_counter()
        # beta | alpha, float32 on bf16 rows as FLM's host takes them
        ba = (torch.from_numpy(h.astype(np.float32)) @ p["ab"]).numpy()
        ba = np.ascontiguousarray(ba.astype(bfloat16))
        nz = W.NVH * W.DV
        hist = np.zeros((W.CONV - 1, W.QKV), np.float32)
        st = np.zeros((W.NVH, W.DK, W.DV), np.float32)
        u = np.empty((D.M + 3, W.QKV), np.float32)
        y = np.empty((D.M, W.QKV), np.float32)
        out = np.empty((D.M, nz), np.float32)
        self._t("host", t0)
        pend = []
        for c, (r0, t) in enumerate(chunks):
            lin = self.gemm_wait(pl[c], t)
            z = self.gemm_wait(pz[c], t)
            t0 = time.perf_counter()
            self.q.gdn_chunk_z(
                lin.ctypes.data,
                lin.shape[1],
                z.ctypes.data,
                z.shape[1],
                ba[r0:].ctypes.data,
                ba.shape[1],
                t,
                hist.ctypes.data,
                p["conv"].ctypes.data,
                p["nega"].ctypes.data,
                p["dtb"].ctypes.data,
                p["norm"].ctypes.data,
                st.ctypes.data,
                u.ctypes.data,
                y.ctypes.data,
                out.ctypes.data,
                self.nt,
            )
            self._t("gdn", t0)
            t0 = time.perf_counter()
            a = self.a_chunk(nz, c)
            H.tile_a(out[:t], a[1], D.TM, D.TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{j}.lin_out", c))
            self._t("host", t0)
        self.conv[j], self.state[j] = hist, st
        return pend

    def _attn_all(self, j, chunks):
        DQ, nc = S.DQ, len(chunks)
        kvb = self.kvb[j]
        pq = [self.gemm_start(f"{j}.qg", c) for c in range(nc)]
        pk = [self.gemm_start(f"{j}.kv", c) for c in range(nc)]
        att, qgs = [], []
        self.kv[j] = ([], [])
        for c, (r0, t) in enumerate(chunks):
            qg = self.gemm_wait(pq[c], t)
            kv = self.gemm_wait(pk[c], t)
            t0 = time.perf_counter()
            self._qk_post(j, kvb, c, r0, t, qg, kv)
            self._t("attn_pre", t0)
            att.append(self.attn_start(kvb, c, r0, t))
            qgs.append(qg)
        pend = []
        for c, (r0, t) in enumerate(chunks):
            parts = self.attn_wait(att[c])
            t0 = time.perf_counter()
            a = self.a_chunk(DQ, c)
            tiles = H._bf16_tiles(a[1], D.TM, D.TK)
            gate = qgs[c][:, DQ:]
            h0 = 0
            for p in parts:
                om = p["om"]
                nh, m, dh = om.shape
                self.q.o_gate_tile(
                    om.ctypes.data,
                    t,
                    nh,
                    dh,
                    m,
                    p["g"].lkp,
                    h0,
                    DQ,
                    D.TM,
                    D.TK,
                    gate.ctypes.data,
                    qgs[c].shape[1],
                    tiles.ctypes.data,
                    self.nt,
                )
                h0 += nh
            H._to_bfp(tiles, t, a[1])
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{j}.o", c))
            self._t("attn_post", t0)
        return pend

    def _qk_post(self, j, kvb, c, r0, t, qg, kv):
        """q (normed, roped) into chunk c's attention q buffers; K (normed,
        roped) and V into the layer's records and the kept rows."""
        DH, p = S.DH, self.p[j]
        rope = (self.cs[r0 : r0 + t], self.sn[r0 : r0 + t], W.ROPE_DIM)
        ch = self.attn_chunk(kvb["op"], c)
        h0 = 0
        for part in ch["parts"]:
            hg = part["hg"]
            qm = self.q_stage(part)
            H.head_post(
                qg, t, h0 * DH, hg, DH, None, p["qn"], W.EPS, rope, 1.0, qm,
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
            kv, t, 0, S.KV_HEADS, DH, None, p["kn"], W.EPS, rope, 1.0, kb, DH, nkv
        )
        H.head_post(
            kv, t, nkv, S.KV_HEADS, DH, None, None, W.EPS, None, 1.0, vb, DH, nkv
        )
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

    # ---- MLP ----

    def _mlp(self, j, x, pend, chunks):
        """The mixer's residual add, then the MLP, in place on x."""
        n2, ks = self.p[j]["n2"], S.I // D.TK
        npieces = len(S.GU_PIECES)
        gus = []
        for c, (r0, t) in enumerate(chunks):
            o = self.gemm_wait(pend[c], t)
            t0 = time.perf_counter()
            a = self.a_chunk(S.D_MODEL, c)
            H.add_rms_tile(x[r0 : r0 + t], o, t, None, W.EPS, n2, a[1], D.TM, D.TK)
            a[0].sync(TO)
            self._t("host", t0)
            gus.append([self.gemm_start(f"{j}.gu{i}", c) for i in range(npieces)])
        dn = []
        for c, (r0, t) in enumerate(chunks):
            outs = [self.gemm_wait(g, t) for g in gus[c]]
            t0 = time.perf_counter()
            a = self.a_chunk(S.I, c)
            tiles = H._bf16_tiles(a[1], D.TM, D.TK)
            for (c0, c1), g in zip(S.GU_PIECES, outs):
                self.q.glu_tile(
                    g.ctypes.data,
                    g.shape[1],
                    t,
                    (c1 - c0) // 2,
                    c0 // 2 // D.TK,
                    ks,
                    D.TM,
                    D.TK,
                    tiles.ctypes.data,
                    self.nt,
                )
            H._to_bfp(tiles, t, a[1])
            a[0].sync(TO)
            self._t("glu", t0)
            dn.append(self.gemm_start(f"{j}.down", c))
        for c, (r0, t) in enumerate(chunks):
            y = self.gemm_wait(dn[c], t)
            t0 = time.perf_counter()
            H.add_rms_tile(x[r0 : r0 + t], y, t, None, W.EPS, None, None, D.TM, D.TK)
            self._t("host", t0)


def main():
    """The prefill of a prompt against the CPU reference: the last position's
    logits, by cosine and next token."""
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
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    w = W.Qwen38Q4nx()
    ids = w.tokenizer().encode(args.prompt).ids
    if args.prompt_len:
        ids = (ids * (args.prompt_len // len(ids) + 1))[: args.prompt_len]
    layers = list(range(args.layers))
    t0 = time.perf_counter()
    # the device is held from the first context to the last dispatch
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        pf = Qwen38Prefill(args.build_dir, w, layers)
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
    ref = R.Reference(w, layers)
    fn = torch.from_numpy(np.asarray(w.final_norm(), np.float32))
    got = ref.lm_head(R.rmsnorm(torch.from_numpy(x[-1]), fn))
    want = ref.lm_head(R.rmsnorm(ref.forward(ids, lm=False)[-1], fn))
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0)
    a, b = int(got.argmax()), int(want.argmax())
    print(f"{len(ids)} tokens: logits cos {cos:.5f}; next token {a} vs ref {b}")
    ok = a == b and cos > 0.98
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
