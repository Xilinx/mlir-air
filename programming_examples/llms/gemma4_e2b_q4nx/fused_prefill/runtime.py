# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Gemma4-E2B prefill on the fused prefill device.

Same interface as Gemma4Q4nxPrefill: prefill(ids) returns the last token's
logits and kv_stack() the cache for the decoder. The prompt runs in chunks of
128 tokens, each chunk through every layer, on one hw_context; the LM head is
the int4 GEMV on a second one. Norms, rope, residuals and the GLU and PLE
products run on the host (hostops).
"""

import json
import time
from pathlib import Path

import numpy as np
import pyxrt as xrt
from ml_dtypes import bfloat16

import device as D
import gemma4_e2b_q4nx_weights as gw
import hostops as H
import insts as I
import lm_gemv
import packing as P

TO, FROM = (
    xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
    xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
)
SLOT = {"g": 0, "f": 3, "s": 6}  # first kernel BO argument of each group


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


class _Ctx:
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


def _view(bo, dtype, shape):
    return np.frombuffer(bo.map(), dtype, count=int(np.prod(shape))).reshape(shape)


class FusedPrefill:
    def __init__(self, build_dir, max_len=2048):
        self.dir = Path(build_dir)
        self.man = json.loads((self.dir / "manifest.json").read_text())
        self.max_len = max_len
        H.load(str(self.dir))
        self.dev = xrt.device(0)
        self.fp = _Ctx(self.dev, self.dir / f"{self.man['gemm'][0]}.xclbin")
        self.current_context_length = 0
        self.dev_t, self.by_op = 0.0, {}
        self.rope_cache = {}
        self.suspended = False

    # ---- device plumbing ----

    def _insts(self, name, group):
        raw = (self.dir / f"{name}.insts.bin").read_bytes()
        return np.frombuffer(I.strip_columns(raw, D.COLS[group]), np.uint32)

    def _go(self, run, tag):
        t0 = time.perf_counter()
        run.r.start()
        st = run.r.wait()
        dt = time.perf_counter() - t0
        self.dev_t += dt
        self.by_op[tag] = self.by_op.get(tag, 0.0) + dt
        if "COMPLETED" not in str(st):
            raise RuntimeError(f"fused prefill {tag}: {st}")

    def _gemm_site(self, site, k, n, packed, wq):
        fp = self.fp
        npd = P.n_pad(n)
        if k not in self.abuf:
            a = fp.bo(D.M * k * 2, 0)
            am = _view(a, bfloat16, (D.HR, k // D.TK, D.TM, D.TK))
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
            cm=_view(c, bfloat16, (D.M, npd)),
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
            bos = [self.dummy[i] for i in range(9)]
            bos[0:3] = [self.abuf[s["k"]]["a"], s["w"], s["c"]]
            s["runs"][act] = (self.fp.run(ib, len(ins), bos), ib, name)
        self._go(s["runs"][act][0], site.split(".")[-1])
        s["c"].sync(FROM, t * s["npd"] * 2, 0)
        return s

    def _out(self, s, t):
        return H.bf16_to_f32(s["cm"], t, s["n"])

    def mm(self, site, x, act=0):
        t = x.shape[0]
        self._write_a(self.sites[site]["k"], x)
        return self._out(self._gemm(site, act, t), t)

    def _a_from(self, k, fill):
        ab = self.abuf[k]
        fill(ab["am"])
        ab["a"].sync(TO)
        ab["last"] = object()

    def ffn(self, L, h):
        t = h.shape[0]
        self._write_a(self.sites[f"{L}.gate"]["k"], h)
        g = self._gemm(f"{L}.gate", 1, t)
        u = self._gemm(f"{L}.up", 0, t)
        n = g["n"]
        self._a_from(n, lambda am: H.glu_tile(g["cm"], u["cm"], t, n, am, D.TM, D.TK))
        return self._out(self._gemm(f"{L}.down", 0, t), t)

    def ple(self, L, x, pli):
        t = x.shape[0]
        self._write_a(self.sites[f"{L}.ple_gate"]["k"], x)
        g = self._gemm(f"{L}.ple_gate", 1, t)
        self._a_from(g["n"], lambda am: H.mul_tile(g["cm"], pli, am, D.TM, D.TK))
        return self._out(self._gemm(f"{L}.ple_proj", 0, t), t)

    def _attn_setup(self):
        fp, self.attn = self.fp, {}
        for a, (_, dh, lkp, _) in D.ATTN.items():
            names = self.man["attn"][a]
            pts = {
                k: dict(zip(("heads", "nkv", "q0", "k0"), v))
                for k, v in self.man["attn_points"].items()
            }
            base = self._insts(names["base"], a).tobytes()
            var = {
                p: (self._insts(names[p], a).tobytes(), pts[p][p] - pts["base"][p])
                for p in ("nkv", "q0", "k0")
            }
            tmpl = I.LinearInsts(
                base, {p: pts["base"][p] for p in ("nkv", "q0", "k0")}, var
            )
            chk = pts["check"]
            if not np.array_equal(
                tmpl.at(nkv=chk["nkv"], q0=chk["q0"], k0=chk["k0"]),
                self._insts(names["check"], a),
            ):
                raise RuntimeError(
                    f"attention {a}: insts are not linear in (nkv, q0, k0)"
                )
            heads = gw.N_Q_HEADS
            q = fp.bo(heads * D.M * dh * 2, SLOT[a])
            o = fp.bo(heads * D.M * dh * 2, SLOT[a] + 2)
            qm = _view(q, bfloat16, (heads, D.M, dh))
            qm[:] = 0
            ib, iv = fp.insts(tmpl.at(nkv=1, q0=0, k0=0))
            bos = [self.dummy[i] for i in range(9)]
            bos[SLOT[a] : SLOT[a] + 3] = [q, self.dummy[SLOT[a] + 1], o]
            self.attn[a] = dict(
                tmpl=tmpl,
                q=q,
                qm=qm,
                o=o,
                om=_view(o, bfloat16, (heads, D.M, dh)),
                ib=ib,
                iv=iv,
                run=fp.run(ib, iv.size, bos),
                lkp=lkp,
                dh=dh,
            )

    def _kv_bo(self, L):
        a = "s" if gw.is_sliding(L) else "f"
        _, dh, lkp, _ = D.ATTN[a]
        nblk = -(-self.max_len // lkp)
        bo = self.fp.bo(nblk * D.kv_rec(a) * 2, SLOT[a] + 1)
        return dict(bo=bo, m=_view(bo, bfloat16, (nblk, D.kv_rec(a))), a=a)

    def attention(self, L, qe, r0):
        """qe [T, H, dh] (roped) of tokens [r0, r0+T); K/V of tokens [0, r0+T)
        are in the source layer's KV records."""
        src = gw.kv_source_layer(L)
        kv = self.kvb[src]
        at = self.attn[kv["a"]]
        t, h, dh = qe.shape
        lkp = at["lkp"]
        q0 = r0 // lkp
        nend = -(-(r0 + t) // lkp)
        k0 = max(0, q0 - D.WINDOW // lkp) if kv["a"] == "s" else 0
        np.copyto(
            at["qm"][:, :t],
            (qe * (np.sqrt(dh) * gw.ATTN_SCALE)).transpose(1, 0, 2),
            casting="unsafe",
        )
        at["qm"][:, t:] = 0
        at["q"].sync(TO, h * D.M * dh * 2, 0)
        rec = D.kv_rec(kv["a"]) * 2
        sub = xrt.bo(kv["bo"], (nend - k0) * rec, k0 * rec)
        at["run"].set_arg(3 + SLOT[kv["a"]] + 1, sub)
        at["sub"] = sub
        at["iv"][:] = at["tmpl"].at(nkv=nend - k0, q0=q0, k0=k0)
        at["ib"].sync(TO)
        self._go(at["run"], f"attn {kv['a']}")
        at["o"].sync(FROM)
        return at["om"][:, :t].transpose(1, 0, 2).reshape(t, h * dh).astype(np.float32)

    # ---- model ----

    def load_weights(self, model=None):
        self.model = (
            gw.Q4nxModel(model) if not isinstance(model, gw.Q4nxModel) else model
        )
        qm = self.model
        fp = self.fp
        self.dummy = [fp.bo(4096, i) for i in range(9)]
        self.abuf, self.sites = {}, {}
        for L in range(gw.NUM_LAYERS):
            for name, raw in P.layer_q4(qm, L).items():
                self._gemm_site(
                    f"{L}.{name}", raw[0].shape[1], raw[0].shape[0], P.pack_q4(*raw), 1
                )
            pw = qm.layer_ple(L)
            for name, w in (
                ("ple_gate", pw["inp_gate"]),
                ("ple_proj", pw["per_layer_projection"]),
            ):
                self._gemm_site(
                    f"{L}.{name}", w.shape[1], w.shape[0], P.pack_bf16(w), 0
                )
        mp = np.concatenate(
            [qm.layer_ple(L)["model_proj"] for L in range(gw.NUM_LAYERS)]
        )
        self._gemm_site("mp", mp.shape[1], mp.shape[0], P.pack_bf16(mp), 0)
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
        self._attn_setup()
        self.kvb = {L: self._kv_bo(L) for L in range(gw.NUM_LAYERS) if gw.owns_kv(L)}
        self.kv = {}
        self._lm_setup()

    def _lm_setup(self):
        ctx = _Ctx(self.dev, self.dir / "lm.xclbin")
        G = lm_gemv
        w = G.pack(*P.q4nx_raw(self.model, "lm_head.weight", G.V, G.K))
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
            xm=_view(bx, bfloat16, (G.K,)),
            ym=_view(by, bfloat16, (G.V,)),
            run=ctx.run(ib, len(ins), [bw, bx, by]),
        )

    def _lm_head(self, last):
        lm = self.lm
        np.copyto(lm["xm"], last.reshape(-1), casting="unsafe")
        lm["bx"].sync(TO)
        self._go(lm["run"], "lm")
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

    def _layer(self, L, x, pli, r0, rope):
        eps, nm = gw.RMS_EPS, self.norms[L]
        t = x.shape[0]
        dh = gw.head_dim(L)
        cs, sn, rot = rope[dh]
        x1 = H.rms(x, nm["input"], eps)
        q = H.rms(self.mm(f"{L}.q", x1).reshape(t, gw.N_Q_HEADS, dh), nm["q_norm"], eps)
        H.rope(q, cs, sn, rot)
        if gw.owns_kv(L):
            k = H.rms(
                self.mm(f"{L}.k", x1).reshape(t, gw.N_KV_HEADS, dh), nm["k_norm"], eps
            )
            v = H.rms(self.mm(f"{L}.v", x1).reshape(t, gw.N_KV_HEADS, dh), None, eps)
            H.rope(k, cs, sn, rot)
            self._kv_append(L, r0, k[:, 0], v[:, 0])
        o = self.attention(L, q, r0)
        o1 = H.add_rms(x, self.mm(f"{L}.o", o), nm["post_attn"], eps)
        h = H.rms(o1, nm["pre_ffn"], eps)
        o2 = H.add_rms(o1, self.ffn(L, h), nm["post_ffn"], eps)
        o3 = H.add_rms(o2, self.ple(L, o2, pli), nm["post_ple"], eps)
        return o3 * nm["out_scale"]

    def _kv_append(self, L, r0, k, v):
        """Tokens [r0, r0+T) of layer L: kept for kv_stack, and written as
        the attention's KV records (r0 is a whole number of blocks)."""
        ks, vs = self.kv.setdefault(L, ([], []))
        ks.append(k)
        vs.append(v)
        kv = self.kvb[L]
        lkp = D.ATTN[kv["a"]][2]
        recs = P.kv_records(kv["a"], k, v)
        b0 = r0 // lkp
        kv["m"][b0 : b0 + len(recs)] = recs
        rec = D.kv_rec(kv["a"]) * 2
        kv["bo"].sync(TO, len(recs) * rec, b0 * rec)

    def suspend(self):
        """Release both hw_contexts; the weight BOs stay. Called after
        kv_stack() so the decoder runs with only its own context open;
        prefill() resumes."""
        if not self.suspended:
            self.fp.close()
            self.lm["ctx"].close()
            self.suspended = True

    def resume(self):
        if self.suspended:
            self.fp.open()
            self.lm["ctx"].open()
            self.suspended = False

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
        for r0 in range(0, n, D.M):
            t = min(D.M, n - r0)
            x = np.ascontiguousarray(emb[r0 : r0 + t], np.float32)
            proj = (
                self.mm("mp", x).reshape(t, gw.NUM_LAYERS, gw.PLI_D)
                * gw.PLE_MODEL_PROJ_SCALE
            )
            pli = (
                gw._rmsnorm(proj, self.glob["ple_proj_norm"]) + tbl[r0 : r0 + t]
            ) * gw.PLE_INPUT_SCALE
            rope = {gw.head_dim(L): self._rope(L, r0, t) for L in self.rope_layers}
            for L in range(gw.NUM_LAYERS):
                x = self._layer(L, x, np.ascontiguousarray(pli[:, L]), r0, rope)
        self.current_context_length = n
        last = gw._rmsnorm(x[-1:], self.glob["final_norm"])
        logits = self._lm_head(last)
        if gw.FINAL_LOGIT_SOFTCAP:
            logits = gw.FINAL_LOGIT_SOFTCAP * np.tanh(logits / gw.FINAL_LOGIT_SOFTCAP)
        return logits

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
