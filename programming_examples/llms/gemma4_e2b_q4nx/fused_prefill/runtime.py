# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B prefill on the fused prefill device (shared/fused_prefill).

Same interface as Gemma4Q4nxPrefill: prefill(ids) returns the last token's
logits and kv_stack() the cache for the decoder. The prompt runs a layer at a
time in 128-token chunks on one hw_context; the LM head is the int4 GEMV on a
second one. Norms, rope, residuals and the GLU and PLE
products run on the host (hostops).
"""

import numpy as np
from ml_dtypes import bfloat16

import gemma4_e2b_q4nx_weights as gw

from ...shared.fused_prefill import device as D
from ...shared.fused_prefill import hostops as H
from ...shared.fused_prefill import lm_gemv
from ...shared.fused_prefill import packing as P
from ...shared.fused_prefill.engine import TO, FROM, Ctx, Engine, view

from .spec import CFG


def _layer_q4(qm, b, L):
    """A layer's Q4NX projections as raw (q, scale, min), [out, in]."""
    dh = gw.head_dim(L)
    dq, dkv = gw.N_Q_HEADS * dh, gw.N_KV_HEADS * dh
    inter = qm.mlp_inter(L)
    p = f"model.layers.{L}."
    d = gw.D
    w = dict(
        q=b.q4nx(p + "self_attn.q_proj.weight", dq, d),
        o=b.q4nx(p + "self_attn.o_proj.weight", d, dq),
        up=b.q4nx(p + "mlp.up_proj.weight", inter, d),
        gate=b.q4nx(p + "mlp.gate_proj.weight", inter, d),
        down=b.q4nx(p + "mlp.down_proj.weight", d, inter),
    )
    if gw.owns_kv(L):
        w["k"] = b.q4nx(p + "self_attn.k_proj.weight", dkv, d)
        w["v"] = b.q4nx(p + "self_attn.v_proj.weight", dkv, d)
    return w


class FusedPrefill(Engine):
    def __init__(self, build_dir, max_len=2048):
        super().__init__(build_dir, CFG, max_len)
        self.current_context_length = 0
        self.rope_cache = {}

    def contexts(self):
        return [self.fp, self.lm["ctx"]]

    # ---- model ----

    def load_weights(self, model=None):
        self.model = (
            gw.Q4nxModel(model) if not isinstance(model, gw.Q4nxModel) else model
        )
        qm = self.model
        self.bundle = P.Bundle(qm.path)
        for L in range(gw.NUM_LAYERS):
            for name, raw in _layer_q4(qm, self.bundle, L).items():
                self.gemm_site(
                    f"{L}.{name}", raw[0].shape[1], raw[0].shape[0], P.pack_q4(*raw), 1
                )
            pw = qm.layer_ple(L)
            for name, w in (
                ("ple_gate", pw["inp_gate"]),
                ("ple_proj", pw["per_layer_projection"]),
            ):
                self.gemm_site(f"{L}.{name}", w.shape[1], w.shape[0], P.pack_bf16(w), 0)
        mp = np.concatenate(
            [qm.layer_ple(L)["model_proj"] for L in range(gw.NUM_LAYERS)]
        )
        self.gemm_site("mp", mp.shape[1], mp.shape[0], P.pack_bf16(mp), 0)
        self.norms = [
            {
                k: (np.asarray(v, np.float32) if isinstance(v, np.ndarray) else v)
                for k, v in qm.layer_norms(L).items()
            }
            for L in range(gw.NUM_LAYERS)
        ]
        self.glob = qm.globals()
        self.freqs = qm.rope_freqs()
        # one layer of each attention class: rope depends only on the class
        self.rope_layers = [
            next(L for L in range(gw.NUM_LAYERS) if gw.is_sliding(L) == c)
            for c in (True, False)
        ]
        self.attn_setup()
        self.kvb = {
            L: self.kv_bo("s" if gw.is_sliding(L) else "f")
            for L in range(gw.NUM_LAYERS)
            if gw.owns_kv(L)
        }
        self.kv = {}
        self._lm_setup()

    def _lm_setup(self):
        ctx = Ctx(self.dev, self.dir / "lm.xclbin")
        G = lm_gemv
        w = G.pack(*self.bundle.q4nx("lm_head.weight", G.V, G.K))
        bw = ctx.bo(w.nbytes, 0)
        bw.write(w.tobytes(), 0)
        bw.sync(TO)
        bx, by = ctx.bo(G.K * 2, 1), ctx.bo(G.V * 2, 2)
        ins = np.fromfile(self.dir / "lm.insts.bin", np.uint32)
        ib, _ = ctx.insts(ins)
        self.lm = dict(
            ctx=ctx,
            keep=(bw, ib),
            bx=bx,
            by=by,
            xm=view(bx, bfloat16, (G.K,)),
            ym=view(by, bfloat16, (G.V,)),
            run=ctx.run(ib, len(ins), [bw, bx, by]),
        )

    def _lm_head(self, last):
        lm = self.lm
        np.copyto(lm["xm"], last.reshape(-1), casting="unsafe")
        lm["bx"].sync(TO)
        self._go(lm["run"], "lm", lm["by"])
        lm["by"].sync(FROM)
        return lm_gemv.unpack_y(lm["ym"].astype(np.float32))

    def _rope(self, L, r0, t):
        key = (gw.head_dim(L), r0, t)
        if key not in self.rope_cache:
            self.rope_cache[key] = self._rope_tab(L, r0, t)
        return self.rope_cache[key]

    def _rope_tab(self, L, r0, t):
        rows = [gw.rope_lut(p, L, rope_freqs=self.freqs) for p in range(r0, r0 + t)]
        h = rows[0][2] // 2
        cs = np.ascontiguousarray(np.stack([r[0] for r in rows])[:, :h], np.float32)
        sn = np.ascontiguousarray(np.stack([r[1] for r in rows])[:, :h], np.float32)
        return cs, sn, rows[0][2]

    def _kv_append(self, L, r0, k, v):
        """Tokens [r0, r0+T) of layer L (k, v [T, 1, dh]): kept for kv_stack,
        and written as the attention's KV records."""
        ks, vs = self.kv.setdefault(L, ([], []))
        ks.append(k[:, 0])
        vs.append(v[:, 0])
        self.kv_append(self.kvb[L], r0, k, v)

    def prefill(self, ids):
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        qm = self.model
        emb = qm.embed_rows("model.embed_tokens.weight", ids) * gw.EMBED_SCALE
        tbl = qm.embed_rows("model.per_layer_token_embd.weight", ids)
        tbl = tbl.reshape(n, gw.NUM_LAYERS, gw.PLI_D) * gw.PLE_EMBED_SCALE
        self.kv = {}
        x = np.array(emb, np.float32, order="C")
        chunks = [(r0, min(D.M, n - r0)) for r0 in range(0, n, D.M)]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            self._tile(x[r0 : r0 + t], gw.D, c)
            pend.append(self.gemm_start("mp", c))
        pli = np.empty((n, gw.NUM_LAYERS, gw.PLI_D), np.float32)
        for c, (r0, t) in enumerate(chunks):
            proj = self._f32(pend[c], t).reshape(t, gw.NUM_LAYERS, gw.PLI_D)
            proj = proj * gw.PLE_MODEL_PROJ_SCALE
            pli[r0 : r0 + t] = (
                gw._rmsnorm(proj, self.glob["ple_proj_norm"]) + tbl[r0 : r0 + t]
            ) * gw.PLE_INPUT_SCALE
        ropes = [
            {gw.head_dim(L): self._rope(L, r0, t) for L in self.rope_layers}
            for r0, t in chunks
        ]
        for L in range(gw.NUM_LAYERS):
            self._layer_all(L, x, pli, chunks, ropes)
        self.current_context_length = n
        last = gw._rmsnorm(x[-1:], self.glob["final_norm"])
        logits = self._lm_head(last)
        if gw.FINAL_LOGIT_SOFTCAP:
            logits = gw.FINAL_LOGIT_SOFTCAP * np.tanh(logits / gw.FINAL_LOGIT_SOFTCAP)
        return logits

    def _tile(self, x, k, c):
        """x [t, k] float32 as chunk c's A for GEMMs of depth k."""
        a = self.a_chunk(k, c)
        H.tile_a(x, a[1], D.TM, D.TK)
        a[0].sync(TO)

    def _f32(self, pend, t):
        """A queued GEMM's output rows [0, t) as float32."""
        return H.bf16_to_f32(self.gemm_wait(pend, t), t, self.sites[pend[4]]["n"])

    def _layer_all(self, L, x, pli, chunks, ropes):
        """Layer L over the prompt x [n, d], in place; each stage queues one
        dispatch per chunk, so the host's work on a chunk overlaps the
        device's on the chunks after it."""
        eps, nm = gw.RMS_EPS, self.norms[L]
        dh = gw.head_dim(L)
        dq = gw.N_Q_HEADS * dh
        own = gw.owns_kv(L)
        rows = [x[r0 : r0 + t] for r0, t in chunks]
        pend = []
        for c, (r0, t) in enumerate(chunks):
            self._tile(H.rms(rows[c], nm["input"], eps), gw.D, c)
            ps = [self.gemm_start(f"{L}.q", c)]
            if own:
                ps += [self.gemm_start(f"{L}.k", c), self.gemm_start(f"{L}.v", c)]
            pend.append(ps)
        kv = self.kvb[gw.kv_source_layer(L)]
        # the kernel divides by sqrt(head_dim); gemma4 scales by ATTN_SCALE
        scale = float(np.sqrt(dh) * gw.ATTN_SCALE)
        att = []
        for c, (r0, t) in enumerate(chunks):
            cs, sn, rot = ropes[c][dh]
            q = self._f32(pend[c][0], t).reshape(t, gw.N_Q_HEADS, dh)
            q = H.rms(q, nm["q_norm"], eps)
            H.rope(q, cs, sn, rot)
            if own:
                k = self._f32(pend[c][1], t).reshape(t, gw.N_KV_HEADS, dh)
                v = self._f32(pend[c][2], t).reshape(t, gw.N_KV_HEADS, dh)
                k = H.rms(k, nm["k_norm"], eps)
                v = H.rms(v, None, eps)
                H.rope(k, cs, sn, rot)
                self._kv_append(L, r0, k, v)
            ch = self.attn_chunk(kv["op"], c)
            hg = ch["parts"][0]["hg"]
            for i, p in enumerate(ch["parts"]):
                H.q_pack(q[:, i * hg : (i + 1) * hg], scale, p["qm"])
            att.append(self.attn_start(kv, c, r0, t))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(dq, c)
            h0 = 0
            for om in self.attn_wait(att[c]):
                H.o_tile(om, t, h0, dq, a[1], D.TM, D.TK)
                h0 += om.shape[0]
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.o", c))
        gu = []
        for c, (r0, t) in enumerate(chunks):
            rows[c][:] = H.add_rms(rows[c], self._f32(pend[c], t), nm["post_attn"], eps)
            self._tile(H.rms(rows[c], nm["pre_ffn"], eps), gw.D, c)
            gu.append(
                (self.gemm_start(f"{L}.gate", c, 1), self.gemm_start(f"{L}.up", c))
            )
        pend = []
        for c, (r0, t) in enumerate(chunks):
            g = self.gemm_wait(gu[c][0], t)
            u = self.gemm_wait(gu[c][1], t)
            n = self.sites[f"{L}.gate"]["n"]
            a = self.a_chunk(n, c)
            H.glu_tile(g, u, t, n, a[1], D.TM, D.TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.down", c))
        pg = []
        for c, (r0, t) in enumerate(chunks):
            rows[c][:] = H.add_rms(rows[c], self._f32(pend[c], t), nm["post_ffn"], eps)
            self._tile(rows[c], gw.D, c)
            pg.append(self.gemm_start(f"{L}.ple_gate", c, 1))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            g = self.gemm_wait(pg[c], t)
            n = self.sites[f"{L}.ple_gate"]["n"]
            a = self.a_chunk(n, c)
            H.mul_tile(g, np.ascontiguousarray(pli[r0 : r0 + t, L]), a[1], D.TM, D.TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{L}.ple_proj", c))
        for c, (r0, t) in enumerate(chunks):
            o3 = H.add_rms(rows[c], self._f32(pend[c], t), nm["post_ple"], eps)
            rows[c][:] = o3 * nm["out_scale"]

    def kv_stack(self):
        """Per layer (shared ones resolved) the prompt's K and V, float32
        [n, head_dim], as Gemma4Q4nxPrefill.kv_stack returns them."""
        ks, vs = [], []
        for L in range(gw.NUM_LAYERS):
            k, v = self.kv[gw.kv_source_layer(L)]
            ks.append(np.concatenate(k).astype(np.float32))
            vs.append(np.concatenate(v).astype(np.float32))
        return ks, vs

    def clear_context(self):
        self.current_context_length = 0
        self.kv = {}
