# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Qwen3.5-0.8B (qwen35_0_8b_q4nx) on the fused prefill device: every
projection GEMM, the lm_head and the softmax attention of the 6 full-attention
layers are ops of one device (the model's q35_spec); the element-wise work and
the Gated DeltaNet mixer run on the host, a chunk at a time, overlapping the
device's work on the next chunk.

Full-attention layer: qkv GEMM (q | gate | k | v, the bundle's q_proj order);
q/k RMSNorm and partial RoPE on the host into attn_bfp16's q and KV records;
the output gate o * sigmoid(gate) on the host into the o GEMM's A. Gated
DeltaNet layer: lin_in (q | k | v | z) and ba (beta | alpha, Q8 as two Q4
halves along K against [x | x]) GEMMs; conv1d, gates, the delta rule on an
FP32 state carried across chunks and the gated RMSNorm on the host
(kernels/q35_host.c in the model's directory); out_proj GEMM. MLP: one
gate | up GEMM, SiLU(gate) * up on the host, down; it runs a few chunk pairs
behind the mixer (_Ffn). The weights are FastFlowLM's Q4K / Q8 bundle
(q35_weights), Q4K on the affine grid bf16(d * sc), bf16(dmin * m).

After prefill(), decode_state() gives what the decode takes
(q35_fdg.FdgDecode.handoff, on programming_examples/fused_decode_gdn): kv
{L: (K, V) bf16 [2, n, 256]}, conv {L: [3, 6144]}, rec {L: the state as hi +
lo bf16 words}.
"""

import ctypes
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The model directory is imported as the air_examples package, not through
# sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[3])
]
from air_examples.llms.qwen35_0_8b_q4nx import q35_spec as S  # noqa: E402
from air_examples.llms.qwen35_0_8b_q4nx import q35_weights as W  # noqa: E402

from . import device as D
from . import hostops as H
from .engine import TO, Engine

HERE = Path(__file__).resolve().parent
HOST_C = HERE.parents[1] / "qwen35_0_8b_q4nx" / "kernels" / "q35_host.c"
EPS = 1e-6
LIN_H, LIN_DK, NC, NZ = 16, 128, 6144, 2048
DQ, DH, ROT = 2048, 256, 64  # 8 heads x 256; partial RoPE over 64 dims
_p, _i = ctypes.c_void_p, ctypes.c_int


HOST_THREADS = 16


def build_host(out):
    """hostops (build.py's libhostops.so) again with HOST_THREADS row threads
    (its rows are independent, so its results do not depend on the count),
    then the model's host C library (libq35host.so) into the build dir. The
    model library is built last: Qwen35Prefill rebuilds both when it is older
    than libhostops.so."""
    flags = [
        "gcc",
        "-O3",
        "-march=native",
        "-ffp-contract=off",
        "-fopenmp",
        "-shared",
        "-fPIC",
    ]
    for src, so in (
        (HERE / "kernels" / "hostops.c", "libhostops.so"),
        (HOST_C, "libq35host.so"),
    ):
        subprocess.run(
            flags
            + [f"-DNT={HOST_THREADS}", str(src), "-o", str(Path(out) / so), "-lm"],
            check=True,
        )


def _rows(parts):
    return tuple(np.concatenate(a) for a in zip(*parts))


def _zero_rows(n, k):
    return (
        np.zeros((n, k), np.uint8),
        np.zeros((n, k // 32), np.float32),
        np.zeros((n, k // 32), np.float32),
    )


def _bf(a):
    return np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)


class Qwen35Prefill(Engine):
    def __init__(
        self,
        build_dir,
        Wt,
        kinds,
        cos,
        sin,
        embed=None,
        max_len=S.max_len,
        threads=None,
    ):
        """Wt: the bundle (q35_weights.Q4nx); kinds: the layer types; cos /
        sin: the rotary table, bf16 [positions, 64] (HF's); embed: the
        embedding table (Wt.embed())."""
        self.embed = Wt.embed() if embed is None else embed
        so = Path(build_dir) / "libq35host.so"
        hops = Path(build_dir) / "libhostops.so"
        # (re)built with the C source, or after build.py's own hostops
        if not so.exists() or so.stat().st_mtime < max(
            HOST_C.stat().st_mtime, hops.stat().st_mtime
        ):
            build_host(build_dir)
        super().__init__(build_dir, S.CFG, max_len)
        self.Wt, self.kinds = Wt, list(kinds)
        self.q = ctypes.CDLL(str(so))
        self.q.gdn_chunk.argtypes = [_p, _i, _p, _i, _i] + [_p] * 9 + [_i]
        self.q.o_gate_tile.argtypes = [_p] + [_i] * 9 + [_p, _i, _p, _i]
        self.q.swiglu_tile.argtypes = [_p] + [_i] * 5 + [_p, _i]
        self.nt = threads or HOST_THREADS
        cos, sin = np.asarray(cos, np.float32), np.asarray(sin, np.float32)
        self.cs = np.ascontiguousarray(cos[:, : ROT // 2])
        self.sn = np.ascontiguousarray(sin[:, : ROT // 2])
        self.times = {}
        self._load()

    # ---- weights ----

    def _site(self, name, q4, k):
        q, sc, mn = q4
        self.gemm_site(name, k, q.shape[0], self.pack_q4(q, _bf(sc), _bf(mn)), 1)

    def _load(self):
        Wt, d, I = self.Wt, W.D, W.I
        self.w = {}
        self.gdn = {}
        self.attn_layers = [
            L for L, k in enumerate(self.kinds) if k == "full_attention"
        ]
        for L, kind in enumerate(self.kinds):
            p = f"model.layers.{L}."
            for nm in ("input_layernorm", "post_attention_layernorm"):
                self.w[f"{L}.{nm}"] = Wt.bf16(p + nm + ".weight")  # 1 + w
            if kind == "full_attention":
                s = p + "self_attn."
                # the bundle's q_proj: 8 heads' queries, then their gates
                qkv = _rows(
                    [
                        Wt.q4(s + "q_proj.weight", 2 * DQ, d),
                        Wt.q4(s + "k_proj.weight", 512, d),
                        Wt.q4(s + "v_proj.weight", 512, d),
                    ]
                )
                self._site(f"{L}.qkv", qkv, d)
                self._site(f"{L}.o", Wt.q4(s + "o_proj.weight", d, DQ), DQ)
                for nm in ("q_norm", "k_norm"):
                    self.w[f"{L}.{nm}"] = Wt.bf16(s + nm + ".weight")
            else:
                s = p + "linear_attn."
                lin = _rows(
                    [
                        Wt.q4(s + "qkv_proj.weight", NC, d),
                        Wt.q4(p + "self_attn.gate_proj.weight", NZ, d),
                    ]
                )
                self._site(f"{L}.lin_in", lin, d)
                self._site(
                    f"{L}.ba",
                    _rows(
                        [
                            Wt.as_q4(s + "ssm_beta_proj.weight", LIN_H, d),
                            Wt.as_q4(s + "ssm_alpha_proj.weight", LIN_H, d),
                        ]
                    ),
                    2 * d,
                )
                self._site(f"{L}.lo", Wt.as_q4(s + "ssm_out_proj.weight", d, NZ), NZ)
                self.gdn[L] = dict(
                    conv=np.ascontiguousarray(
                        Wt.bf16(s + "ssm_conv1d.weight"), np.float32
                    ),  # [4, NC]
                    nega=np.ascontiguousarray(Wt.bf16(s + "ssm_a"), np.float32).reshape(
                        -1
                    ),
                    dtb=np.ascontiguousarray(
                        Wt.bf16(s + "ssm_dt.bias"), np.float32
                    ).reshape(-1),
                    norm=np.ascontiguousarray(
                        Wt.bf16(s + "ssm_norm.weight"), np.float32
                    ).reshape(-1),
                )
            # gate | up as one GEMM; SiLU(gate) * up on the host
            self._site(
                f"{L}.gu",
                _rows(
                    [
                        Wt.q4(p + "mlp.gate_proj.weight", I, d),
                        Wt.q4(p + "mlp.up_proj.weight", I, d),
                    ]
                ),
                d,
            )
            self._site(f"{L}.down", Wt.q4(p + "mlp.down_proj.weight", d, I), I)
        self.w["norm"] = Wt.bf16("model.norm.weight")
        q, sc = Wt.q8("lm_head.weight", W.NV, d)
        width = S.SHAPES["head"][1]
        for i, (n0, n1) in enumerate(S.HEAD_PIECES):
            t = W.q8_as_q4(q[n0:n1], sc[n0:n1])
            if n1 - n0 < width:  # the last piece: zero rows to the built width
                t = _rows([t, _zero_rows(width - (n1 - n0), 2 * d)])
            self._site(f"head.{i}", t, 2 * d)
        self.attn_setup()
        self.kvb = {L: self.kv_bo("a") for L in self.attn_layers}
        self.warm("0.gu")

    # ---- prefill ----

    def _t(self, k, t0):
        self.times[k] = self.times.get(k, 0.0) + time.perf_counter() - t0

    def prefill(self, ids):
        """The last row's logits (float32 [vocab]); keeps the state the
        decode takes (decode_state)."""
        t0 = time.perf_counter()
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        x = np.ascontiguousarray(self.embed[np.asarray(ids)], np.float32)
        chunks = self.chunk_rows(n)
        self.conv, self.state, self._k = {}, {}, {}
        self._t("host", t0)
        for L, kind in enumerate(self.kinds):
            ffn = _Ffn(self, L, x, chunks)
            if kind == "full_attention":
                self._attn_all(L, x, chunks, ffn)
            else:
                self._gdn_all(L, x, chunks, ffn)
            ffn.drain()
        t0 = time.perf_counter()
        xn = H.rms(x[-1:], self.w["norm"], EPS)
        a = np.concatenate([xn, xn], 1)  # the Q8 head's [x | x]
        z = np.concatenate(
            [
                self.mm(f"head.{i}", a)[0, : n1 - n0]
                for i, (n0, n1) in enumerate(S.HEAD_PIECES)
            ]
        )
        self._t("head", t0)
        self.n = n
        return z

    def decode_state(self):
        """The prompt's state as the decode takes it, (kv, conv,
        rec): built when decode starts, after the first token."""
        n = self.n
        rec = {L: self.state_words(s) for L, s in self.state.items()}
        kv = {}
        for L, (ks, vs) in self._k.items():
            k = np.concatenate(ks).reshape(n, 2, DH).transpose(1, 0, 2)
            v = np.concatenate(vs).reshape(n, 2, DH).transpose(1, 0, 2)
            kv[L] = (k, v)
        return kv, self.conv, rec

    @staticmethod
    def state_words(s):
        """FP32 state [16, 128 k, 128 v] -> the scan's hi + lo bf16 words,
        [16, 2 slices, 2 * 128 * 64]: per head and slice of 64 v columns, hi
        [128][64] then lo [128][64]."""
        sv = s.reshape(LIN_H, LIN_DK, 2, 64).transpose(0, 2, 1, 3)
        hi = sv.astype(bfloat16)
        lo = (sv - hi.astype(np.float32)).astype(bfloat16)
        return np.stack([hi, lo], 2).reshape(LIN_H, 2, -1)

    def _norm_in(self, L, x, chunks, k=None):
        """rms(x) * w into each chunk's A, depth d (and a copy at depth 2d,
        [x | x], when k = 2d)."""
        TM, TK, d = D.TM, D.TK, W.D
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(d, c)
            H.rms_tile(
                x[r0 : r0 + t], self.w[f"{L}.input_layernorm"], EPS, a[1], TM, TK
            )
            a[0].sync(TO)
            yield c, a

    def _gdn_all(self, L, x, chunks, ffn):
        TM, TK, d = D.TM, D.TK, W.D
        g = self.gdn[L]
        pl, pb = [], []
        t0 = time.perf_counter()
        for c, a in self._norm_in(L, x, chunks):
            pl.append(self.gemm_start(f"{L}.lin_in", c))
            a2 = self.a_chunk(2 * d, c)
            ks = d // TK
            a2[1][:, :ks] = a[1][:, :ks]
            a2[1][:, ks : 2 * ks] = a[1][:, :ks]
            a2[0].sync(TO)
            pb.append(self.gemm_start(f"{L}.ba", c))
        hist = np.zeros((3, NC), np.float32)
        st = np.zeros((LIN_H, LIN_DK, LIN_DK), np.float32)
        u = np.empty((D.M + 3, NC), np.float32)
        y = np.empty((D.M, NC), np.float32)
        out = np.empty((D.M, NZ), np.float32)
        self._t("host", t0)
        for c, (r0, t) in enumerate(chunks):
            lin = self.gemm_wait(pl[c], t)
            ba = self.gemm_wait(pb[c], t)
            t0 = time.perf_counter()
            self.q.gdn_chunk(
                lin.ctypes.data,
                lin.shape[1],
                ba.ctypes.data,
                ba.shape[1],
                t,
                hist.ctypes.data,
                g["conv"].ctypes.data,
                g["nega"].ctypes.data,
                g["dtb"].ctypes.data,
                g["norm"].ctypes.data,
                st.ctypes.data,
                u.ctypes.data,
                y.ctypes.data,
                out.ctypes.data,
                self.nt,
            )
            self._t("gdn", t0)
            t0 = time.perf_counter()
            a = self.a_chunk(NZ, c)
            H.tile_a(out[:t], a[1], TM, TK)
            a[0].sync(TO)
            ffn.push(c, self.gemm_start(f"{L}.lo", c))
            self._t("host", t0)
        self.conv[L], self.state[L] = hist, st

    def _attn_all(self, L, x, chunks, ffn):
        TM, TK = D.TM, D.TK
        kvb = self.kvb[L]
        pend = []
        t0 = time.perf_counter()
        for c, a in self._norm_in(L, x, chunks):
            pend.append(self.gemm_start(f"{L}.qkv", c))
        self._t("host", t0)
        att, qkvs = [], []
        self._k[L] = ([], [])
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c], t)
            t0 = time.perf_counter()
            self._qkv_post(L, kvb, c, r0, t, qkv)
            self._t("attn_pre", t0)
            att.append(self.attn_start(kvb, c, r0, t))
            qkvs.append(qkv)
        for c, (r0, t) in enumerate(chunks):
            parts = self.attn_wait(att[c])
            t0 = time.perf_counter()
            a = self.a_chunk(DQ, c)
            tiles = H._bf16_tiles(a[1], TM, TK)
            gate = qkvs[c][:, DQ:]
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
                    TM,
                    TK,
                    gate.ctypes.data,
                    qkvs[c].shape[1],
                    tiles.ctypes.data,
                    self.nt,
                )
                h0 += nh
            H._to_bfp(tiles, t, a[1])
            a[0].sync(TO)
            ffn.push(c, self.gemm_start(f"{L}.o", c))
            self._t("attn_post", t0)

    def _qkv_post(self, L, kvb, c, r0, t, qkv):
        """q (normed, roped) into chunk c's attention q buffers; K (normed,
        roped) and V into the layer's records and the kept rows."""
        rope = (self.cs[r0 : r0 + t], self.sn[r0 : r0 + t], ROT)
        ch = self.attn_chunk(kvb["op"], c)
        h0 = 0
        for p in ch["parts"]:
            hg = p["hg"]
            qm = self.q_stage(p)
            H.head_post(
                qkv,
                t,
                h0 * DH,
                hg,
                DH,
                None,
                self.w[f"{L}.q_norm"],
                EPS,
                rope,
                1.0,
                qm,
                D.M * DH,
                DH,
            )
            if t < D.M:
                qm[:, t:] = 0
            self.q_write(p, qm)
            h0 += hg
        kb = np.empty((t, 2 * DH), bfloat16)
        vb = np.empty((t, 2 * DH), bfloat16)
        H.head_post(
            qkv,
            t,
            2 * DQ,
            2,
            DH,
            None,
            self.w[f"{L}.k_norm"],
            EPS,
            rope,
            1.0,
            kb,
            DH,
            2 * DH,
        )
        H.head_post(
            qkv, t, 2 * DQ + 2 * DH, 2, DH, None, None, EPS, None, 1.0, vb, DH, 2 * DH
        )
        self._k[L][0].append(kb)
        self._k[L][1].append(vb)
        h0 = 0
        for p in kvb["parts"]:
            g = self.cfg.attn[p["a"]]
            b0, rec = r0 // g.lkp, self.kv_rec(g)
            nblk = p["m"].shape[1]
            nb = -(-t // g.lkp)
            for h in range(g.kv_heads):
                self.kv_write(kb, vb, (h0 + h) * DH, t, g, p["m"][h, b0])
                p["bo"].sync(TO, nb * rec * 2, (h * nblk + b0) * rec * 2)
            h0 += g.kv_heads


class _Ffn:
    """The rest of layer L, in place on x: residual add + post-attention
    norm, gate / up, GLU, down, residual add. It runs as a pipeline over the
    chunk groups (the GEMMs' chunk pairs), its stages one, two and three
    groups behind the mixer's output projection, so the device's MLP GEMMs
    of earlier chunks overlap the host's mixer work (the Gated DeltaNet, the
    output gate) on later ones."""

    def __init__(self, e, L, x, chunks):
        self.e, self.L, self.chunks = e, L, chunks
        self.rows = [x[r0 : r0 + t] for r0, t in chunks]
        self.w = e.w[f"{L}.post_attention_layernorm"]
        self.groups, c = [], 0
        while c < len(chunks):
            n = 2 if e._pair(c) else 1
            self.groups.append(tuple(range(c, c + n)))
            c += n
        self.last = {g[-1]: i for i, g in enumerate(self.groups)}
        self.o, self.gu, self.dn = {}, {}, {}
        self.done = [0, 0, 0]  # groups through each stage

    def push(self, c, pend):
        """The mixer queued chunk c's output projection."""
        self.o[c] = pend
        if c in self.last:
            self._advance(self.last[c])

    def drain(self):
        self._advance(len(self.groups) + 2)

    def _advance(self, gi):
        for k, lag in ((0, 1), (1, 2), (2, 3)):
            while self.done[k] <= min(gi - lag, len(self.groups) - 1):
                (self._s1, self._s2, self._s3)[k](self.groups[self.done[k]])
                self.done[k] += 1

    def _s1(self, grp):
        e, TM, TK = self.e, D.TM, D.TK
        for c in grp:
            t = self.chunks[c][1]
            o = e.gemm_wait(self.o.pop(c), t)
            t0 = time.perf_counter()
            a = e.a_chunk(W.D, c)
            H.add_rms_tile(self.rows[c], o, t, None, EPS, self.w, a[1], TM, TK)
            a[0].sync(TO)
            self.gu[c] = e.gemm_start(f"{self.L}.gu", c)
            e._t("host", t0)

    def _s2(self, grp):
        e, TM, TK = self.e, D.TM, D.TK
        for c in grp:
            t = self.chunks[c][1]
            gu = e.gemm_wait(self.gu.pop(c), t)
            t0 = time.perf_counter()
            a = e.a_chunk(W.I, c)
            tiles = H._bf16_tiles(a[1], TM, TK)
            e.q.swiglu_tile(
                gu.ctypes.data, gu.shape[1], t, W.I, TM, TK, tiles.ctypes.data, e.nt
            )
            H._to_bfp(tiles, t, a[1])
            a[0].sync(TO)
            self.dn[c] = e.gemm_start(f"{self.L}.down", c)
            e._t("host", t0)

    def _s3(self, grp):
        e, TM, TK = self.e, D.TM, D.TK
        for c in grp:
            t = self.chunks[c][1]
            f = e.gemm_wait(self.dn.pop(c), t)
            t0 = time.perf_counter()
            H.add_rms_tile(self.rows[c], f, t, None, EPS, None, None, TM, TK)
            e._t("host", t0)
