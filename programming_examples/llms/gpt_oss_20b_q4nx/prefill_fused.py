# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPT-OSS-20B prefill on the shared fused prefill device (fused_spec.py):
one hw_context and one PDI for every op, each op a dispatch of its own insts.

A layer is the attention over the prompt's 128-token chunks, as the dense
models run it (shared/fused_prefill/dense.py), with the q / k / v and o
biases, YaRN rope and the sinks; then the MoE over every row at once. The
router runs on the host. Each expert's rows go through its gate | up GEMM, the
clamped SwiGLU on the host and its down GEMM, up to two chunks (256 rows) per
dispatch, the next expert's gate | up queued while the host works on this
one's. The experts' weights are read where the decode holds them (its WE
buffer), as MXFP4 blocks; their biases sit in those blocks and are added on
the host.
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

import gpt_oss_20b_q4nx_weights as g  # noqa: E402
from air_examples.llms.gpt_oss_20b_q4nx import fused_spec as S  # noqa: E402
from air_examples.llms.shared.fused_prefill import device as D  # noqa: E402
from air_examples.llms.shared.fused_prefill import hostops as H  # noqa: E402
from air_examples.llms.shared.fused_prefill.dense import DensePrefill  # noqa: E402
from air_examples.llms.shared.fused_prefill.engine import (  # noqa: E402
    FROM,
    TO,
    Engine,
    view,
)
from air_examples.llms.shared.fused_prefill.models import Desc  # noqa: E402

NCB = S.D_PAD // 128  # column blocks of a row group
GROUP = NCB * 4 * g.BLOCK_BYTES  # bytes of one 128-row group
# the decode's WE needs this much past its last expert: the down GEMM's last
# panel and its K steps past the group read beyond it
WE_SLACK = 2 * GROUP
GU, DN = "gate_up", "down"
DESC = Desc(
    g.FLM_REPO,
    g.D,
    g.DH,
    g.N_Q_HEADS,
    g.N_KV_HEADS,
    g.INTER,
    g.N_LAYERS,
    g.VOCAB,
    qkv_bias=True,
    window=S.WINDOW,
    global_every=2,  # the odd layers are full, the even ones sliding
    max_len=S.max_len,
)


def q4_1_raw(blocks, rows, k):
    """The bundle's Q4_1 blocks [row groups, column blocks, 4, 2560] as raw
    (q [rows, k] uint8, scale [rows, k / 32], min [rows, k / 32]):
    gpt_oss_20b_q4nx_weights.dequant_q4_1 without the multiply."""
    b = g._ungroup_rows(blocks, 4)
    nrb, ncb, _ = b.shape
    sc = g._bf16(b[..., 0:256]).reshape(nrb, ncb, 4, 32)
    mn = g._bf16(b[..., 256:512]).reshape(nrb, ncb, 4, 32)
    qb = b[..., 512:].reshape(nrb, ncb, 2, 128, 8)
    q = torch.stack((qb & 0xF, qb >> 4), dim=-1).reshape(nrb, ncb, 2, 128, 16)
    q = q.permute(0, 1, 2, 4, 3).reshape(nrb, ncb, 32, 128)
    q = q.permute(0, 2, 1, 3).reshape(nrb * 32, ncb * 128)

    def per_row(a):
        return a.permute(0, 3, 1, 2).reshape(nrb * 32, ncb * 4)

    return (
        q[:rows, :k].numpy().astype(np.uint8),
        per_row(sc)[:rows, : k // 32].numpy(),
        per_row(mn)[:rows, : k // 32].numpy(),
    )


def expert_biases(cb0):
    """(gate, up, down) biases [experts, D] of a layer's experts from their
    column block 0, [experts, 69, 4, 2560]: the bf16 in bytes 128..191 of each
    32-row block."""
    b = np.ascontiguousarray(cb0[..., 128:192]).view(bfloat16)
    b = b.reshape(cb0.shape[0], g.EXPERT_ROW_GROUPS, 128).astype(np.float32)
    gate = b[:, 0:46:2].reshape(len(b), -1)[:, : g.INTER]
    up = b[:, 1:46:2].reshape(len(b), -1)[:, : g.INTER]
    down = b[:, 46:].reshape(len(b), -1)[:, : g.D]
    return gate, up, down


class GptOssFused(DensePrefill):
    """prefill(ids) -> per layer (k [KV, n, DH], v) for the decode to seed
    from (logits are the decode's). `we` is the decode's WE buffer and the
    offset of `layers[0]`'s experts in it; without it the engine holds its own
    copy of those layers' experts, and with "stream" one layer's, copied in
    before the layer's MoE (slower; for a check without the decode)."""

    def __init__(self, build_dir, w, layers=None, we=None):
        Engine.__init__(self, build_dir, S.CFG, S.max_len)
        self.desc, self.context = DESC, None
        self.current_context_length = 0
        self.rope_cache, self.kv = {}, []
        self.w = w
        self.layers = list(range(g.N_LAYERS)) if layers is None else list(layers)
        self.eps = g.RMS_EPS
        inv, self.rope_scale = g.yarn_inv_freq()
        inv = inv.double().numpy()
        self.inv_freq = {"f": inv, "s": inv}
        self.rot = g.DH
        self._load(we)

    # ---- weights ----

    def _load(self, we):
        w, nl = self.w, len(self.layers)
        self.norms, self.bias, self.bo, self.router = [], [], [], []
        self.sink_m0, self.sink_l0 = [], []
        self.eb = []  # per layer (gate, up, down) expert biases
        lb = g.N_EXPERTS * g.EXPERT_BYTES
        self.stream = we == "stream"
        if we is None or self.stream:
            import pyxrt as xrt

            held = 1 if self.stream else nl
            bo = xrt.ext.bo(self.dev, held * lb + WE_SLACK)
            m = np.frombuffer(bo.map(), np.uint8)
            m[held * lb :] = 0
            we = (bo, 0)
        else:
            m = None
        self.we_bo, self.we_off, self.we_map = we[0], we[1], m
        self.wsub = {}
        for j, L in enumerate(self.layers):
            p = f"model.layers.{L}.self_attn."
            raw = lambda n: w.raw(p + n + ".weight").view(torch.uint8)  # noqa: E731
            qkv = [
                q4_1_raw(raw(n), r, S.D_PAD)
                for n, r in (
                    ("q_proj", g.N_Q_HEADS * g.DH),
                    ("k_proj", g.N_KV_HEADS * g.DH),
                    ("v_proj", g.N_KV_HEADS * g.DH),
                )
            ]
            qkv = [np.concatenate(a) for a in zip(*qkv)]
            self.gemm_site(f"{j}.qkv", S.D_PAD, len(qkv[0]), self.pack_q4(*qkv), 1)
            o = q4_1_raw(raw("o_proj"), g.D, g.N_Q_HEADS * g.DH)
            self.gemm_site(f"{j}.o", g.N_Q_HEADS * g.DH, g.D, self.pack_q4(*o), 1)
            vec = lambda n: w.vec(p + n).numpy()  # noqa: E731
            self.bias.append(
                np.concatenate([vec(f"{n}_proj.bias") for n in "qkv"]).astype(
                    np.float32
                )
            )
            self.bo.append(vec("o_proj.bias").astype(np.float32))
            # the sink as a score in q k units, rounded as the cores hold it,
            # and its weight against that rounding
            s = vec("sinks.weight").astype(np.float64) * np.sqrt(g.DH)
            m0 = s.astype(np.float32).astype(bfloat16)
            self.sink_m0.append(m0)
            self.sink_l0.append(
                np.exp((s - m0.astype(np.float64)) / np.sqrt(g.DH)).astype(np.float32)
            )
            n1, n2 = w.norms(L)
            self.norms.append(dict(input=n1.numpy(), post_attn=n2.numpy()))
            wr, br = w.router(L)
            self.router.append((wr.T.contiguous(), br.numpy()))
            name = f"model.layers.{L}.ffn_gate_up_down_exps.weight"
            if m is not None and not self.stream:
                m[j * lb : (j + 1) * lb] = w.raw(name).view(torch.uint8).numpy().ravel()
            cb0 = w._f.get_slice(name)[:, :, 0].view(torch.uint8).numpy()
            self.eb.append(expert_biases(cb0))
        if m is not None and not self.stream:
            self.we_bo.sync(TO)
        self.attn_setup()
        self.kvb = [self.kv_bo(self._op(j), g.N_KV_HEADS) for j in range(nl)]
        self.slots = {}
        self.warm(f"{0}.qkv")

    def _op(self, j):
        return "f" if (self.layers[j] + 1) % 2 == 0 else "s"

    def _wbo(self, j, e, part):
        """The sub-buffer of WE that expert e of layer j's gate | up (part GU)
        or down GEMM reads."""
        j = 0 if self.stream else j
        key = (j, e, part)
        if key not in self.wsub:
            import pyxrt as xrt

            k, n, _ = S.SHAPES[part]
            npd = -(-n // D.NR) * D.NR
            rows = (npd // 128 - 1) * NCB + D.k_pad(k) // D.TK
            off = self.we_off + (j * g.N_EXPERTS + e) * g.EXPERT_BYTES
            off += 46 * GROUP if part == DN else 0
            self.wsub[key] = xrt.bo(self.we_bo, rows * 2 * D.QB, off)
        return self.wsub[key]

    # ---- attention: q buffers carry each core's head's sink ----

    def _attn_part(self, g_, hg, s, bos):
        tile = g_.lkp * g_.dh * 9 // 8
        qt = D.q_tile(g_) * 2
        q = self.fp.bo(hg * D.NQT * qt, s)
        qm = view(q, np.uint8, (hg, D.NQT, qt))
        qm[:] = 0
        o = self.fp.bo(hg * D.M * g_.dh * 2, s + 2)
        bos[s : s + 3] = [q, self.dummy[s + 1], o]
        om = view(o, bfloat16, (hg, D.M, g_.dh))
        h0 = (s - 3) // 3 * hg
        return dict(q=q, qm=qm, o=o, om=om, hg=hg, g=g_, h0=h0, tile=tile)

    def q_write(self, p, q16):
        hg, tile = p["hg"], p["tile"]
        tmp = np.empty((hg, D.M * p["g"].dh * 9 // 8), np.uint8)
        H.q_bfp(q16, p["g"].lkp, tmp)
        qm = p["qm"]
        qm[:, :, :tile] = tmp.reshape(hg, D.NQT, tile)
        heads = slice(p["h0"], p["h0"] + hg)
        m0 = self.sink_m0[self._j][heads].view(np.uint8).reshape(hg, 1, 2)
        l0 = self.sink_l0[self._j][heads].view(np.uint8).reshape(hg, 1, 4)
        qm[:, :, tile : tile + 2] = m0
        qm[:, :, tile + 4 : tile + 8] = l0

    # ---- a layer ----

    def prefill(self, ids):
        """Per layer (k [KV, n, DH], v [KV, n, DH]) of the prompt, k roped."""
        self.resume()
        ids = [int(t) for t in np.asarray(ids).reshape(-1)]
        n = len(ids)
        if n > self.max_len:
            raise ValueError(f"prompt of {n} tokens exceeds max_len {self.max_len}")
        x = np.ascontiguousarray(self.w.embed(torch.as_tensor(ids)).numpy(), np.float32)
        self.kv = [([], []) for _ in self.layers]
        chunks = self.chunk_rows(n)
        for j in range(len(self.layers)):
            self._j = j
            self._moe_layer(j, x, self._attn_layer(j, x, chunks), chunks)
        self.current_context_length = n
        out = []
        for k, v in self.kv:
            k = np.concatenate(k).astype(np.float32).reshape(n, g.N_KV_HEADS, g.DH)
            v = np.concatenate(v).astype(np.float32).reshape(n, g.N_KV_HEADS, g.DH)
            out.append(
                (
                    torch.from_numpy(k.transpose(1, 0, 2).copy()),
                    torch.from_numpy(v.transpose(1, 0, 2).copy()),
                )
            )
        self.x = x
        return out

    def _attn_layer(self, j, x, chunks):
        """Layer j's attention over x [n, D], up to the queued o GEMMs."""
        dq = g.N_Q_HEADS * g.DH
        nm, kvb, op = self.norms[j], self.kvb[j], self._op(j)
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(S.D_PAD, c)
            H.rms_tile(x[r0 : r0 + t], nm["input"], self.eps, a[1], D.TM, D.TK)
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{j}.qkv", c))
        att = []
        for c, (r0, t) in enumerate(chunks):
            qkv = self.gemm_wait(pend[c], t)
            self._qkv_post(j, kvb, c, r0, t, qkv, op)
            att.append(self.attn_start(kvb, c, r0, t))
        pend = []
        for c, (r0, t) in enumerate(chunks):
            a = self.a_chunk(dq, c)
            for p in self.attn_wait(att[c]):
                self.o_read(p, t, p["h0"], dq, a[1])
            a[0].sync(TO)
            pend.append(self.gemm_start(f"{j}.o", c))
        return pend

    def _moe_layer(self, j, x, pend, chunks):
        """The o GEMMs' residual add, then the MoE, in place on x."""
        for c, (r0, t) in enumerate(chunks):
            o = self.gemm_wait(pend[c], t)
            xs = x[r0 : r0 + t]
            xs += self.bo[j]
            H.add_rms_tile(xs, o, t, None, self.eps, None, None, D.TM, D.TK)
        if self.stream:
            name = f"model.layers.{self.layers[j]}.ffn_gate_up_down_exps.weight"
            lb = g.N_EXPERTS * g.EXPERT_BYTES
            self.we_map[:lb] = self.w.raw(name).view(torch.uint8).numpy().ravel()
            self.we_bo.sync(TO, lb, 0)
        h = H.rms(x, self.norms[j]["post_attn"], self.eps)
        wr, br = self.router[j]
        # through torch: numpy's OpenBLAS leaves a spinning thread on every
        # core after a GEMM, which starves the host ops between the experts
        logits = (torch.from_numpy(h) @ wr).numpy() + br
        top_i = np.argsort(-logits, axis=1, kind="stable")[:, : g.TOP_K]
        top_v = np.take_along_axis(logits, top_i, 1)
        top_w = np.exp(top_v - top_v[:, :1])
        top_w /= top_w.sum(1, keepdims=True)
        x += self._experts(j, h, top_i, top_w)

    # ---- experts ----

    def _slot(self, part, i, pair):
        """Dispatch slot i of `part`: its A and C buffers and run."""
        key = (part, i, pair)
        if key not in self.slots:
            k, n, _ = S.SHAPES[part]
            npd = -(-n // D.NR) * D.NR
            name = f"g_{k}_{npd}_0_2_{pair}"
            ins = self._insts(name)
            ib, _ = self.fp.insts(ins)
            a, am = self._a_buf(k, 1 + pair)
            c = self.fp.bo((1 + pair) * D.M * npd * 2, 2)
            cm = view(c, bfloat16, ((1 + pair) * D.M, npd))
            bos = list(self.dummy)
            bos[0:3] = [a, self.dummy[1], c]
            run = self.fp.run(ib, len(ins), bos)
            steps = k // D.TK
            self.slots[key] = dict(
                a=a,
                am=[am[:, :steps, h] for h in range(1 + pair)],
                c=c,
                cm=cm,
                run=run,
                ib=ib,
                npd=npd,
                tag=part,
            )
        return self.slots[key]

    def _start(self, st, wbo):
        st["run"].set_arg(4, wbo)
        st["c"].sync(TO, 64, 0)
        st["t0"] = time.perf_counter()
        st["run"].r.start()

    def _wait(self, st, t):
        r = st["run"].r.wait()
        dt = time.perf_counter() - st["t0"]
        self.dev_t += dt
        self.by_op[st["tag"]] = self.by_op.get(st["tag"], 0.0) + dt
        if "COMPLETED" not in str(r):
            raise RuntimeError(f"fused prefill {st['tag']}: {r}")
        st["c"].sync(FROM, t * st["npd"] * 2, 0)
        return st["cm"]

    def _experts(self, j, h, top_i, top_w):
        """sum over the routed experts of weight * expert(h [n, D])."""
        n, M = h.shape[0], D.M
        out = np.zeros((n, g.D), np.float32)
        gb, ub, db = self.eb[j]
        counts = np.bincount(top_i.ravel(), minlength=g.N_EXPERTS)
        blocks = []
        for e in np.argsort(-counts, kind="stable"):
            if counts[e] == 0:
                break
            rows, slot = np.nonzero(top_i == e)
            wt = top_w[rows, slot]
            for r0 in range(0, len(rows), 2 * M):
                blocks.append((int(e), rows[r0 : r0 + 2 * M], wt[r0 : r0 + 2 * M]))

        def start_gu(i):
            e, rows, _ = blocks[i]
            pair = int(len(rows) > M)
            st = self._slot(GU, i % 2, pair)
            for hh in range(1 + pair):
                H.tile_a(h[rows[hh * M : (hh + 1) * M]], st["am"][hh], D.TM, D.TK)
            st["a"].sync(TO)
            self._start(st, self._wbo(j, e, GU))
            return st

        def finish_dn(i, st):
            e, rows, wt = blocks[i]
            y = self._wait(st, len(rows))
            H.scatter_add_rows(out, y, rows, wt, db[e], g.D)

        gus = {0: start_gu(0)} if blocks else {}
        prev = None
        for i, (e, rows, wt) in enumerate(blocks):
            if i + 1 < len(blocks):
                gus[i + 1] = start_gu(i + 1)
            t = len(rows)
            pair = int(t > M)
            gu = self._wait(gus.pop(i), t)
            st = self._slot(DN, i % 2, pair)
            for hh in range(1 + pair):
                r = slice(hh * M, min(t, (hh + 1) * M))
                H.oss_glu_tile(
                    gu[r],
                    gb[e],
                    ub[e],
                    r.stop - r.start,
                    g.INTER,
                    st["am"][hh],
                    D.TM,
                    D.TK,
                )
            st["a"].sync(TO)
            self._start(st, self._wbo(j, e, DN))
            if prev is not None:
                finish_dn(*prev)
            prev = (i, st)
        if prev is not None:
            finish_dn(*prev)
        return out


def main():
    """The fused prefill of a prompt against the CPU reference: the last
    position's logits, by cosine and next token. The experts are streamed in a
    layer at a time, so this runs without the decode's weights resident."""
    import argparse
    import os
    import tempfile

    import filelock
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir", help="the fused prefill build (fused_build.py)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-len", type=int, default=128, help="repeat to N tokens")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(g.flm_model_dir() / "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    ids = (ids * (args.prompt_len // len(ids) + 1))[: args.prompt_len]
    w = g.GptOssQ4nx()
    # the reference first: its expert cache is freed before the NPU loads
    ref = g.Reference(w)
    want = ref.forward(ids)
    head, final = ref._head, ref._final
    del ref
    w._expert_cache.clear()
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        pf = GptOssFused(args.build_dir, w, we="stream")
        pf.prefill(ids)
    x = torch.from_numpy(pf.x[-1])
    got = g.rmsnorm(x, final) @ head.T
    cos = torch.nn.functional.cosine_similarity(got, want, dim=0)
    a, b = int(got.argmax()), int(want.argmax())
    print(f"{len(ids)} tokens: logits cos {cos:.5f}; next token {a} vs ref {b}")
    ok = a == b and cos > 0.98
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
