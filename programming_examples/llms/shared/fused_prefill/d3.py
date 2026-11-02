# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The d3 decision models (vllm-sr/d3-edge, vllm-sr/d3-lite) on the fused
prefill device. Both are Qwen3.5-0.8B derivatives, so D3Prefill is
Qwen35Prefill (qwen35.py) with three changes:

  * full attention is noncausal: every query reads keys [0, n), so a layer's
    attention starts once every chunk's K / V records are written
    (attn_bfp16 with Attn.noncausal; the Gated DeltaNet stays causal);
  * the last layer (full attention in both models) feeds only the last row:
    its K / V cover every row, its queries, output projection and MLP only
    the last chunk / row;
  * forward() returns the final-norm hidden state of the last row, which the
    model's FP32 readout reads (d3_edge/d3_npu.py), instead of logits.

The host row ops that feed a GEMM write its bfp16 A directly
(d3_edge/kernels/d3_host.c).

Two configurations, one device each (d3_spec.py in each model directory):

  edge  vllm-sr/d3-edge, 12 layers, its BF16 weights (wq=0). The GEMM reads
        a weight as BFP16; a site held as [hi | lo] is [bfp16(W) | W -
        bfp16(W)] at depth 2k against [x | x], the cores' FP32 accumulator
        summing both, the others one BFP16 copy at depth k (WFORMS).
  lite  vllm-sr/d3-lite, 24 layers, FastFlowLM's D3-Lite-NPU2 Q4K bundle,
        read as qwen35_0_8b_q4nx reads Qwen3.5-0.8B's.
"""

import ctypes
import json
import queue
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ml_dtypes import bfloat16

# The model directories are imported as the air_examples package, not through
# sys.path.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[3])
]
from air_examples.llms.d3_edge import d3_weights as W3  # noqa: E402

from . import device as D  # noqa: E402
from . import hostops as H  # noqa: E402
from . import qwen35 as Q35  # noqa: E402
from .build import attn_points  # noqa: E402
from .engine import TO, Engine  # noqa: E402

EPS, NC, NZ, DQ, DH, ROT = Q35.EPS, Q35.NC, Q35.NZ, Q35.DQ, Q35.DH, Q35.ROT
D_MODEL, INTER = 1024, 3584
HEADS, KV_HEADS = 8, 2
LIN_HEADS = 16
MAX_LEN = 4096
_p, _i, _l = ctypes.c_void_p, ctypes.c_int, ctypes.c_long

# the model directory of each configuration
DIRS = {"edge": "d3_edge", "lite": "d3_lite_q4nx"}
HOST_C = Path(__file__).resolve().parents[2] / "d3_edge" / "kernels" / "d3_host.c"
# edge: the site kinds each weight form holds as [hi | lo]
WFORMS = {
    "hilo:gu,down": {"gu", "down"},
    "hilo": {"qkv", "o", "lin_in", "lo", "gu", "down"},
}
DEFAULT_WFORM = "hilo:gu,down"
# prompts of at least this many chunks run the DeltaNet on a second host
# thread; below it the thread's hand-offs cost more than the overlap gains
GDN_THREAD_CHUNKS = 16


def _npad(v):
    return -(-v // D.NR) * D.NR


def device_spec(model):
    """The fused prefill device of a configuration, as the attributes
    build.py reads from a spec module: CFG, ATTN_POINTS, max_len,
    gemm_shapes(). The attention is attn_bfp16 with head_dim 256, the 8
    query heads in two groups of 4 on columns 0 and 7, one KV head each,
    noncausal, KV records for MAX_LEN tokens."""
    a = D.Attn(
        col=0,
        dh=DH,
        lkp=32,
        kv_heads=1,
        # the cores read an even number of blocks, so one past the last
        max_blocks=-(-MAX_LEN // 32) + 1,
        noncausal=True,
    )
    cfg = D.Config(
        attn={"a0": a, "a1": a._replace(col=7)}, act="silu", ops={"a": ("a0", "a1")}
    )
    # site -> (depth, output width)
    shapes = {
        "qkv": (D_MODEL, 2 * DQ + 2 * KV_HEADS * DH),  # q | gate | k | v
        "o": (DQ, D_MODEL),  # also the DeltaNet out_proj (lo)
        "gu": (D_MODEL, 2 * INTER),  # MLP gate | up
        "down": (INTER, D_MODEL),
    }
    if model == "edge":
        # q | k | v | z | b | a: the BF16 beta / alpha rows ride in lin_in's pad
        shapes["lin_in"] = (D_MODEL, NC + NZ + 2 * LIN_HEADS)
        # depth 2k for every site, depth k for the sites some form holds as
        # one copy
        split = set.intersection(*WFORMS.values())
        gemms = {(2 * k, _npad(m), 0, 0) for k, m in shapes.values()}
        gemms |= {(k, _npad(m), 0, 0) for s, (k, m) in shapes.items() if s not in split}
    else:
        shapes["lin_in"] = (D_MODEL, NC + NZ)
        shapes["ba"] = (2 * D_MODEL, 2 * LIN_HEADS)  # Q8: two Q4 halves along K
        gemms = {(k, _npad(m), 0, 1) for k, m in shapes.values()}
    return SimpleNamespace(
        CFG=cfg,
        ATTN_POINTS=attn_points(HEADS),
        max_len=MAX_LEN,
        SHAPES=shapes,
        gemm_shapes=lambda: sorted(gemms),
    )


def build_device(model, argv):
    """build.py's main for the configuration's spec module; marks the build
    directory with the configuration, which D3Prefill checks."""
    from .build import main

    main(f"air_examples.llms.{DIRS[model]}.d3_spec", argv)
    (Path(argv[0]) / "d3_device.txt").write_text(model + "\n")


def build_host(out):
    """qwen35.build_host's libraries, then d3_host.c (libd3host.so), into
    the build dir."""
    Q35.build_host(out)
    subprocess.run(
        ["gcc", "-O3", "-march=native", "-ffp-contract=off", "-fopenmp"]
        + ["-shared", "-fPIC", f"-DNT={Q35.HOST_THREADS}", str(HOST_C)]
        + ["-o", str(Path(out) / "libd3host.so"), "-lm"],
        check=True,
    )


def bfp16_values(w):
    """The values the device's BFP16 operand holds for w [n, k] float32 (bf16
    values): per 8 values along k, int8 mantissas under a shared exponent
    (hostops.bfp_rows), dequantized."""
    g = H.bfp_rows(np.ascontiguousarray(w.astype(bfloat16)).reshape(-1, 8))
    e = g[:, :1].astype(np.int32) - 127 - 6
    m = g[:, 1:].view(np.int8).astype(np.float32)
    return (m * np.exp2(e.astype(np.float32))).reshape(w.shape)


def rotary_table(root, n):
    """HF's rotary table of the first n positions, bf16 values [n, 64]."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
    )

    tc = json.loads((Path(root) / "config.json").read_text())["text_config"]
    with torch.no_grad():
        rot = Qwen3_5TextRotaryEmbedding(config=Qwen3_5TextConfig(**tc))
        pos = torch.arange(n).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rot(torch.zeros(1, n, D_MODEL, dtype=torch.bfloat16), pos)
    return cos[0].float().numpy(), sin[0].float().numpy()


class D3Prefill(Q35.Qwen35Prefill):
    """The d3 forward pass to the last row's final-norm hidden state."""

    def __init__(self, build_dir, model, wform=None):
        """model: "edge" or "lite"; wform: edge's weight form (WFORMS)."""
        self.model = model
        if model == "edge":
            self.wform = wform or DEFAULT_WFORM
            if self.wform not in WFORMS:
                raise ValueError(f"wform {self.wform!r}: one of {list(WFORMS)}")
            self.hilo = WFORMS[self.wform]
        else:
            assert wform in (None, "q4"), wform
            self.wform, self.hilo = "q4", set()
        dev = (Path(build_dir) / "d3_device.txt").read_text().strip()
        assert dev == model, f"{build_dir} is the {dev} device, not {model}'s"
        spec = device_spec(model)
        self.root = W3.model_dir(model)
        tc = json.loads((self.root / "config.json").read_text())["text_config"]
        kinds = tc["layer_types"][: tc["num_hidden_layers"]]
        assert kinds[-1] == "full_attention"
        if model == "lite":
            self.Wt = W3.LiteBundle()
            self.embed = self.Wt.embed()
        else:
            self.ck = W3.EdgeCheckpoint(self.root)
            self.embed = self.ck.get("embed_tokens.weight")
        # (re)built with the C sources, or after build.py's own hostops
        so = Path(build_dir) / "libd3host.so"
        hops = Path(build_dir) / "libhostops.so"
        srcs = (HOST_C, Q35.HOST_C, Q35.HERE / "kernels" / "hostops.c", hops)
        if not so.exists() or so.stat().st_mtime < max(p.stat().st_mtime for p in srcs):
            build_host(build_dir)
        Engine.__init__(self, build_dir, spec.CFG, spec.max_len)
        self.kinds = list(kinds)
        self.q = ctypes.CDLL(str(Path(build_dir) / "libq35host.so"))
        self.q.gdn_chunk.argtypes = [_p, _i, _p, _i, _i] + [_p] * 9 + [_i]
        self.q.o_gate_tile.argtypes = [_p] + [_i] * 9 + [_p, _i, _p, _i]
        self.q.swiglu_tile.argtypes = [_p] + [_i] * 5 + [_p, _i]
        self.d3 = ctypes.CDLL(str(so))
        tail = [_i, _i, _p, _l, _l, _i]  # TM, TK, dst, dr, dk, threads
        self.d3.d3_norm_bfp.argtypes = [_p, _i, _i, _p, ctypes.c_float] + tail
        self.d3.d3_add_norm_bfp.argtypes = [_p, _p] + [_i] * 3 + [_p, ctypes.c_float]
        self.d3.d3_add_norm_bfp.argtypes += tail
        self.d3.d3_swiglu_bfp.argtypes = [_p] + [_i] * 3 + tail
        self.d3.d3_rows_bfp.argtypes = [_p, _i, _i] + tail
        self.nt = Q35.HOST_THREADS
        cos, sin = rotary_table(self.root, spec.max_len)
        self.cs = np.ascontiguousarray(cos[:, : ROT // 2])
        self.sn = np.ascontiguousarray(sin[:, : ROT // 2])
        self.times = {}
        self._load()

    # ---- weights ----

    def _site(self, name, q4, k):
        if not name.startswith("head."):  # d3 has no lm_head
            super()._site(name, q4, k)

    def _bsite(self, name, w):
        """A BF16 weight w float32 [n, k]: one BFP16 copy at depth k, or
        [bfp16(w) | w - bfp16(w)] at depth 2k against [x | x]."""
        n, k = w.shape
        if name.split(".")[-1] in self.hilo:
            hi = bfp16_values(w)
            w2 = np.concatenate([hi, w - hi], 1)
            self.gemm_site(name, 2 * k, n, self.pack_bf16(w2), 0)
            self.sites[name]["dup"] = k
        else:
            self.gemm_site(name, k, n, self.pack_bf16(w), 0)

    def _load(self):
        if self.model == "lite":
            return super()._load()
        ck = self.ck
        self.w, self.gdn = {}, {}
        self.attn_layers = [
            L for L, k in enumerate(self.kinds) if k == "full_attention"
        ]
        one = lambda nm: 1.0 + ck.f32(nm)  # noqa: E731  zero-centred RMSNorms
        for L, kind in enumerate(self.kinds):
            p = f"layers.{L}."
            for nm in ("input_layernorm", "post_attention_layernorm"):
                self.w[f"{L}.{nm}"] = one(p + nm + ".weight")
            if kind == "full_attention":
                s = p + "self_attn."
                # HF's q_proj rows are per head [q | gate]; the host code
                # takes the 8 heads' queries, then their gates
                q = ck.f32(s + "q_proj.weight").reshape(HEADS, 2, DH, D_MODEL)
                qkv = np.concatenate(
                    [
                        q[:, 0].reshape(DQ, D_MODEL),
                        q[:, 1].reshape(DQ, D_MODEL),
                        ck.f32(s + "k_proj.weight"),
                        ck.f32(s + "v_proj.weight"),
                    ]
                )
                self._bsite(f"{L}.qkv", qkv)
                self._bsite(f"{L}.o", ck.f32(s + "o_proj.weight"))
                for nm in ("q_norm", "k_norm"):
                    self.w[f"{L}.{nm}"] = one(s + nm + ".weight")
            else:
                s = p + "linear_attn."
                lin = np.concatenate(
                    [
                        ck.f32(s + "in_proj_qkv.weight"),
                        ck.f32(s + "in_proj_z.weight"),
                        ck.f32(s + "in_proj_b.weight"),
                        ck.f32(s + "in_proj_a.weight"),
                    ]
                )
                self._bsite(f"{L}.lin_in", lin)
                self._bsite(f"{L}.lo", ck.f32(s + "out_proj.weight"))
                self.gdn[L] = dict(
                    conv=np.ascontiguousarray(ck.f32(s + "conv1d.weight")[:, 0].T),
                    nega=-np.exp(ck.f32(s + "A_log")),
                    dtb=ck.f32(s + "dt_bias"),
                    norm=ck.f32(s + "norm.weight"),
                )
            gu = np.concatenate(
                [ck.f32(p + "mlp.gate_proj.weight"), ck.f32(p + "mlp.up_proj.weight")]
            )
            self._bsite(f"{L}.gu", gu)
            self._bsite(f"{L}.down", ck.f32(p + "mlp.down_proj.weight"))
        self.w["norm"] = one("norm.weight")
        self.attn_setup()
        self.kvb = {L: self.kv_bo("a") for L in self.attn_layers}
        self.warm("0.gu")

    # ---- GEMMs ----

    def gemm_start(self, site, c, act=0, role=None):
        """Engine.gemm_start; a [hi | lo] site's A is chunk c's A at depth k
        twice, [x | x], copied here."""
        k = self.sites[site].get("dup")
        if k:
            a, a2 = self.a_chunk(k, c), self.a_chunk(2 * k, c)
            ks = k // D.TK
            a2[1][:, :ks] = a[1][:, :ks]
            a2[1][:, ks : 2 * ks] = a[1][:, :ks]
            a2[0].sync(TO)
        return super().gemm_start(site, c, act, role)

    def mm_rows(self, site, x):
        """site's GEMM on rows x float32 [t, k] (t <= 128), synchronously:
        its output bf16 [128, n_pad] (rows [0, t) valid)."""
        s = self.sites[site]
        if s.get("dup"):
            x = np.concatenate([x, x], 1)
        self._write_a(s["k"], np.ascontiguousarray(x, np.float32))
        return self._gemm(site, 0, x.shape[0])["cm"]

    # ---- the forward pass ----

    def forward(self, ids):
        """The last row's final-norm hidden state, bf16 values as float32
        [d]."""
        t0 = time.perf_counter()
        self.resume()
        ids = np.asarray(ids, np.int64).reshape(-1)
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        x = np.ascontiguousarray(self.embed[ids], np.float32)
        chunks = self.chunk_rows(n)
        self._t("host", t0)
        last = len(self.kinds) - 1
        for L, kind in enumerate(self.kinds):
            if L == last:
                self._attn_last(L, x, chunks, n)
                break
            ffn = _D3Ffn(self, L, x, chunks)
            if kind == "full_attention":
                self._attn_all(L, x, chunks, ffn, n)
            else:
                self._gdn_all(L, x, chunks, ffn)
            ffn.drain()
        t0 = time.perf_counter()
        h = H.rms(x[n - 1 : n], self.w["norm"], EPS)[0]
        h = h.astype(bfloat16).astype(np.float32)
        self._t("host", t0)
        return h

    def bfp(self, am):
        """(TM, TK, A, dr, dk, threads): the tail of the d3_host.c calls
        writing the A view am (Engine.a_chunk) of a GEMM."""
        dr, dk = (v // am.shape[2] for v in am.strides[:2])
        return D.TM, D.TK, am.ctypes.data, dr, dk, self.nt

    def _norm_in(self, L, x, chunks, k=None):
        """Qwen35Prefill._norm_in, straight into bfp16 (d3_host.c)."""
        w = self.w[f"{L}.input_layernorm"]
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(D_MODEL, c)
            xr = x[r0 : r0 + t]
            self.d3.d3_norm_bfp(
                xr.ctypes.data, t, D_MODEL, w.ctypes.data, EPS, *self.bfp(a[1])
            )
            a[0].sync(TO)
            yield c, a

    def _gdn_all(self, L, x, chunks, ffn):
        """Qwen35Prefill._gdn_all. From GDN_THREAD_CHUNKS chunks on, the
        per-chunk DeltaNet (conv, gates, scan, gated norm) and the out
        projection's queueing run on a second host thread, overlapping the
        MLP's host stages (ffn) on this one. edge has no ba GEMM: b | a are
        lin_in's columns past z."""
        TK, d = D.TK, D_MODEL
        g = self.gdn[L]
        pl, pb = [], []
        t0 = time.perf_counter()
        for c, a in self._norm_in(L, x, chunks):
            pl.append(self.gemm_start(f"{L}.lin_in", c))
            if self.model == "lite":
                a2 = self.a_chunk(2 * d, c)
                ks = d // TK
                a2[1][:, :ks] = a[1][:, :ks]
                a2[1][:, ks : 2 * ks] = a[1][:, :ks]
                a2[0].sync(TO)
                pb.append(self.gemm_start(f"{L}.ba", c))
        hist = np.zeros((3, NC), np.float32)
        st = np.zeros((Q35.LIN_H, Q35.LIN_DK, Q35.LIN_DK), np.float32)
        u = np.empty((D.M + 3, NC), np.float32)
        y = np.empty((D.M, NC), np.float32)
        out = np.empty((D.M, NZ), np.float32)
        self._t("host", t0)
        threaded = len(chunks) >= GDN_THREAD_CHUNKS
        done = queue.Queue()
        push = done.put if threaded else lambda item: ffn.push(*item)

        def mixer():
            try:
                for c, (r0, t) in enumerate(chunks):
                    lin = self.gemm_wait(pl[c], t)
                    if pb:
                        ba = self.gemm_wait(pb[c], t)
                        ldb = ba.shape[1]
                    else:
                        ba, ldb = lin[:, NC + NZ :], lin.shape[1]
                    t0 = time.perf_counter()
                    self.q.gdn_chunk(
                        lin.ctypes.data,
                        lin.shape[1],
                        ba.ctypes.data,
                        ldb,
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
                    self.d3.d3_rows_bfp(out.ctypes.data, t, NZ, *self.bfp(a[1]))
                    a[0].sync(TO)
                    push((c, self.gemm_start(f"{L}.lo", c)))
                    self._t("host", t0)
            except BaseException as e:  # re-raised on the caller's thread
                if not threaded:
                    raise
                done.put(e)

        if not threaded:
            return mixer()
        th = threading.Thread(target=mixer)
        th.start()
        for _ in chunks:
            item = done.get()
            if isinstance(item, BaseException):
                th.join()
                raise item
            ffn.push(*item)
        th.join()

    def _qkv_in(self, L, x, chunks):
        pend = []
        t0 = time.perf_counter()
        for c, a in self._norm_in(L, x, chunks):
            pend.append(self.gemm_start(f"{L}.qkv", c))
        self._t("host", t0)
        return pend

    def _attn_all(self, L, x, chunks, ffn, n):
        """Every chunk's queries read the keys of the whole prompt, so the
        attention starts after the last chunk's K / V records; then each
        chunk's queries are written and its attention queued, and the output
        projections and the MLP (ffn) follow chunk by chunk."""
        kvb = self.kvb[L]
        pend = self._qkv_in(L, x, chunks)
        qkvs = []
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c], t)
            t0 = time.perf_counter()
            self._post_kv(L, kvb, r0, t, qkv)
            self._t("attn_pre", t0)
            qkvs.append(qkv)
        att = []
        for c, (r0, t) in enumerate(chunks):
            t0 = time.perf_counter()
            self._post_q(L, kvb, c, r0, t, qkvs[c])
            self._t("attn_pre", t0)
            att.append(self.attn_start(kvb, c, r0, t, n))
        for c, (r0, t) in enumerate(chunks):
            ffn.push(c, self._o_start(L, c, t, qkvs[c], self.attn_wait(att[c])))

    def _o_start(self, L, c, t, qkv, parts):
        """o * sigmoid(gate) of chunk c into the o GEMM's A; queue the GEMM."""
        t0 = time.perf_counter()
        TM, TK = D.TM, D.TK
        a = self.a_chunk(DQ, c)
        tiles = H._bf16_tiles(a[1], TM, TK)
        gate = qkv[:, DQ:]
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
                qkv.shape[1],
                tiles.ctypes.data,
                self.nt,
            )
            h0 += nh
        H._to_bfp(tiles, t, a[1])
        a[0].sync(TO)
        pend = self.gemm_start(f"{L}.o", c)
        self._t("attn_post", t0)
        return pend

    def _attn_last(self, L, x, chunks, n):
        """The last layer: K / V of every chunk, then the attention, output
        projection, residual and MLP of the last chunk's last row."""
        kvb = self.kvb[L]
        pend = self._qkv_in(L, x, chunks)
        cl = len(chunks) - 1
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c], t)
            t0 = time.perf_counter()
            self._post_kv(L, kvb, r0, t, qkv)
            self._t("attn_pre", t0)
        r0, t = chunks[cl]
        t0 = time.perf_counter()
        self._post_q(L, kvb, cl, r0, t, qkv)
        self._t("attn_pre", t0)
        parts = self.attn_wait(self.attn_start(kvb, cl, r0, t, n))
        o = self.gemm_wait(self._o_start(L, cl, t, qkv, parts), t)
        t0 = time.perf_counter()
        xl = x[n - 1 : n]
        H.add_rms_tile(xl, o[t - 1 : t], 1, None, EPS, None, None, D.TM, D.TK)
        xn = H.rms(xl, self.w[f"{L}.post_attention_layernorm"], EPS)
        self._t("host", t0)
        gu = self.mm_rows(f"{L}.gu", xn)
        t0 = time.perf_counter()
        g = gu[0, :INTER].astype(np.float32)
        up = gu[0, INTER : 2 * INTER].astype(np.float32)
        s = (g / (1.0 + np.exp(-g))).astype(bfloat16).astype(np.float32)
        hm = (s * up).astype(bfloat16).astype(np.float32)[None]
        self._t("host", t0)
        f = self.mm_rows(f"{L}.down", hm)
        H.add_rms_tile(xl, f[0:1], 1, None, EPS, None, None, D.TM, D.TK)

    def _rope(self, r0, t):
        return (self.cs[r0 : r0 + t], self.sn[r0 : r0 + t], ROT)

    def _post_q(self, L, kvb, c, r0, t, qkv):
        """Qwen35Prefill._qkv_post's q: normed, roped, into chunk c's
        attention q buffers."""
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
                self._rope(r0, t),
                1.0,
                qm,
                D.M * DH,
                DH,
            )
            if t < D.M:
                qm[:, t:] = 0
            self.q_write(p, qm)
            h0 += hg

    def _post_kv(self, L, kvb, r0, t, qkv):
        """Qwen35Prefill._qkv_post's K (normed, roped) and V into the
        layer's records, without the decode's kept rows."""
        rope = self._rope(r0, t)
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


class _D3Ffn(Q35._Ffn):
    """Qwen35Prefill's MLP pipeline, its residual + norm and SwiGLU written
    straight into the next GEMM's bfp16 A (d3_host.c)."""

    def _s1(self, grp):
        e = self.e
        for c in grp:
            t = self.chunks[c][1]
            o = e.gemm_wait(self.o.pop(c), t)
            t0 = time.perf_counter()
            a = e.a_chunk(D_MODEL, c)
            e.d3.d3_add_norm_bfp(
                self.rows[c].ctypes.data,
                o.ctypes.data,
                o.shape[1],
                t,
                D_MODEL,
                self.w.ctypes.data,
                EPS,
                *e.bfp(a[1]),
            )
            a[0].sync(TO)
            self.gu[c] = e.gemm_start(f"{self.L}.gu", c)
            e._t("host", t0)

    def _s2(self, grp):
        e = self.e
        for c in grp:
            t = self.chunks[c][1]
            gu = e.gemm_wait(self.gu.pop(c), t)
            t0 = time.perf_counter()
            a = e.a_chunk(INTER, c)
            e.d3.d3_swiglu_bfp(gu.ctypes.data, gu.shape[1], t, INTER, *e.bfp(a[1]))
            a[0].sync(TO)
            self.dn[c] = e.gemm_start(f"{self.L}.down", c)
            e._t("host", t0)
