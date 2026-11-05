# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS prefill, decomposed as FastFlowLM's gpt_oss_npu _prefill_with_mm
# does it: the whole prompt at once, rows padded to ROUND, per layer
#
#   rmsnorm -> q/k/v GEMMs (+bias) -> YaRN rope -> attention with sinks
#   -> o GEMM (+bias) -> residual, rmsnorm -> router GEMM, top-4
#   -> per expert: gather its rows, gate GEMM (mode 1), up GEMM (mode 2),
#      gate * up, down GEMM -> weighted combine, residual
#
# then the final norm and LM head on the last row. The steps FLM runs on the
# NPU go through `ops`; CpuOps does them in numpy at the precision the NPU
# hands back (bf16 in and out, f32 accumulation).
import argparse
import collections
import contextlib
import functools
import math
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

import gpt_oss_20b_q4nx_weights as g  # noqa: E402

ROUND = 256  # prompt rows per GEMM round (4 rows of 64)
SHORT = 128  # an expert with at most this many rows takes the m = 32 GEMM
MIN_PREFILL = 32  # shorter prompts go through the decode loop


# seconds per step of the last forward, when timing
TIMES = collections.defaultdict(float)


@contextlib.contextmanager
def timed(name):
    t0 = time.perf_counter()
    yield
    TIMES[name] += time.perf_counter() - t0


def bf16(t):
    return t.to(torch.bfloat16).float()


def pad_rows(n):
    return -(-n // ROUND) * ROUND


def _t16(a):
    """A bf16 numpy array as a torch bfloat16 tensor, no copy."""
    return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16)


class CpuOps:
    """The NPU steps, on the CPU."""

    def gemm(self, a, wb, mode=0, key=None):
        """mode(bf16(a @ w.T + b)) for (w, b) = wb(): 0 as is, 1 the gate's
        clamp and x * sigmoid(1.702 x), 2 the up's clamp and + 1."""
        w, b = wb()
        y = bf16(bf16(a) @ bf16(w).T + (0 if b is None else bf16(b)))
        if mode == 1:
            y = y.clamp(max=g.SWIGLU_LIMIT)
            y = bf16(y * torch.sigmoid(g.SWIGLU_ALPHA * y))
        elif mode == 2:
            y = bf16(y.clamp(-g.SWIGLU_LIMIT, g.SWIGLU_LIMIT) + 1)
        return y

    def layer_begin(self, w, layer):
        pass

    def experts(self, w, layer, x, tasks):
        """sum over tasks (e, rows, weights) of weights * expert e(x[rows])."""
        out = torch.zeros(x.shape[0], g.D)
        for e, rows, wt in tasks:
            out.index_add_(0, rows, wt * self.expert_ffn(w, layer, e, x[rows]))
        return out

    def expert_ffn(self, w, layer, e, xe):
        """Expert e's gate (mode 1), up (mode 2), their product, then down."""
        gate, up, down, gb, ub, db = w.expert(layer, e)
        h = bf16(
            self.gemm(xe, lambda: (gate, gb), 1) * self.gemm(xe, lambda: (up, ub), 2)
        )
        return self.gemm(h, lambda: (down, db))

    def attention(self, q, k, v, sinks, layer, n):
        """q [H, Lp, DH], k/v [KV, Lp, DH] of which the first n rows are the
        prompt; causal (sliding layers: last SLIDING_WINDOW keys), the sink
        joining each row's softmax."""
        rep = g.N_Q_HEADS // g.N_KV_HEADS
        kr, vr = k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0)
        s = (q @ kr.transpose(1, 2)) * (1.0 / math.sqrt(g.DH))
        lp = q.shape[1]
        qpos = torch.arange(lp)[:, None]
        kpos = torch.arange(lp)[None]
        keep = (kpos <= qpos) & (kpos < n)
        if g.LAYER_TYPES[layer] == "sliding":
            keep &= kpos > qpos - g.SLIDING_WINDOW
        s = s.masked_fill(~keep, float("-inf"))
        sink = sinks.reshape(-1, 1, 1).expand(-1, lp, 1)
        p = torch.softmax(torch.cat((s, sink), dim=-1), dim=-1)[..., :-1]
        return bf16(p @ vr)


class NpuOps(CpuOps):
    """The GEMMs and the expert dequant on the NPU, one ELF per shape; the rest
    on the CPU. Dense weights stay on the device; each layer's experts are
    dequantized into one buffer that their GEMMs read."""

    MAX_L = 4096

    def __init__(self, build_dir="build_prefill", experts=None):
        """`experts(layer)`, when given, is (a device buffer holding the
        layer's expert blocks, their biases [groups, 128]); otherwise the
        blocks are read from the model file and copied in."""
        import prefill_gemm as pg

        self.pg, self.dir, self.experts_src = pg, build_dir, experts
        self.elfs, self.bos = {}, {}
        self.dq, self.dq_pending = None, False
        self.attn = None

    @staticmethod
    def _up(v, q):
        return -(-v // q) * q

    def _elf(self, k, n, tm, pstride=1):
        key = (k, n, tm, pstride)
        if key not in self.elfs:
            max_m = self.MAX_L if tm == 64 else SHORT
            self.elfs[key] = self.pg.GemmElf(self.dir, k, n, max_m, tm, pstride)
        return self.elfs[key]

    def attention(self, q, k, v, sinks, layer, n):
        import prefill_attn as pa

        lp = q.shape[1]
        self._attn()
        with timed("attention"):
            self.attn.load_kv(self._bf16(k), self._bf16(v))
        window = g.SLIDING_WINDOW if g.LAYER_TYPES[layer] == "sliding" else lp
        qh = self._bf16(q.transpose(0, 1))  # [rows, HEADS, DH]
        sk = self._bf16(sinks)
        with timed("attention"):
            o = np.concatenate(
                [self.attn(qh, sk, q0, window) for q0 in range(0, lp, pa.ROUND)]
            )
        return torch.from_numpy(o.astype(np.float32)).transpose(0, 1)

    def _dequant(self):
        import prefill_dequant as pd

        if self.dq is None:
            self.dq = pd.DequantElf(self.dir, g.N_EXPERTS * g.EXPERT_ROW_GROUPS, 23)

    def _attn(self):
        import prefill_attn as pa

        if self.attn is None:
            self.attn = pa.AttnElf(self.dir, self.MAX_L // pa.ROUND)

    def _bf16(self, t):
        return t.to(torch.bfloat16).view(torch.int16).numpy().view(bfloat16)

    def layer_begin(self, w, layer):
        """Start the layer's expert dequant: it needs only the weights, so it
        runs while the host works on the attention half."""
        self._dequant()
        with timed("expert dequant"):
            if self.experts_src:
                src, bias = self.experts_src(layer)
                self.dq.start(src=src, bias=bias)
            else:
                name = f"model.layers.{layer}.ffn_gate_up_down_exps.weight"
                blocks = w.raw(name).view(torch.uint8).numpy()
                self.dq.start(blocks.reshape(-1, 23, 4, 2560))
        self.dq_pending = True

    def experts(self, w, layer, x, tasks):
        if self.dq_pending:
            with timed("expert dequant"):
                self.dq.wait()
            self.dq_pending = False
        with timed("expert gemm"):
            return self._experts(x, tasks)

    def _experts(self, x, tasks):
        """The experts in turn, the host overlapping the NPU: the next expert's
        gate is queued behind this one's down, and this one's rows are added
        in while it runs."""
        k, n = g.D_PAD, 3072
        x16 = self._bf16(x)
        out = torch.zeros(x.shape[0], g.D)

        def elfs(m):
            tm = 32 if m <= SHORT else 64
            return self._elf(k, n, tm, 2), self._elf(k, n, tm)

        def start_gate(e, rows):
            gu = elfs(len(rows))[0]
            gu.start(x16[rows.numpy()], self._panels(e, 0), 1)
            return gu

        gu = start_gate(*tasks[0][:2])
        for i, (e, rows, wt) in enumerate(tasks):
            m = len(rows)
            # gate is copied out: up's call reuses the same output buffer
            gate = _t16(gu.wait()[:m, : g.D_PAD].copy())
            gu.start(m, self._panels(e, 1), 2)
            up = _t16(gu.wait()[:m, : g.D_PAD])
            h = (gate.float() * up.float()).to(torch.bfloat16)
            dn = elfs(m)[1]
            dn.start(h.view(torch.int16).numpy().view(bfloat16), self._panels(e, 2), 0)
            if i + 1 < len(tasks):
                gu = start_gate(*tasks[i + 1][:2])
            y = _t16(dn.wait()[:m, : g.D]).float()
            out.index_add_(0, rows, wt * y)
        return out

    def _panels(self, e, which):
        """Expert e's gate (0), up (1) or down (2) panels in the dequant
        buffer: gate and up interleaved, then down."""
        base = e * g.EXPERT_ROW_GROUPS
        if which == 2:
            return self.dq.panels(base + 46, 24)
        return self.dq.panels(base + which, 47)

    def _weight(self, key, wb):
        if key not in self.bos:
            w, b = wb()
            n, k = w.shape
            mm = self._elf(self._up(k, 128), self._up(n, 1024), 64)
            bo = mm.weight_bo(self._bf16(w), None if b is None else self._bf16(b))
            self.bos[key] = (bo, mm, n)
        return self.bos[key]

    def preload(self, w, layers=range(g.N_LAYERS)):
        """What the first prompt would otherwise set up: every ELF, and the
        dense weights packed on the device (FLM does this at model load)."""
        for layer in layers:
            a = w.attn(layer)
            for n in "qkvo":
                self._weight((n, layer), lambda: (a["w" + n], a["b" + n]))
            wr, _ = w.router(layer)
            self._weight(("router", layer), lambda: (wr, None))
        k, n = g.D_PAD, 3072
        for tm in (32, 64):
            for stride in (1, 2):
                self._elf(k, n, tm, stride)
        self._dequant()
        self._attn()

    def gemm(self, a, wb, mode=0, key=None):
        bo, mm, n = self._weight(key, wb)
        with timed("dense gemm"):
            y = mm(self._bf16(a), bo, mode)
        return torch.from_numpy(y[: a.shape[0], :n].astype(np.float32))


class Prefill:
    def __init__(self, w, ops=None):
        self.w = w
        self.ops = ops or CpuOps()
        self._head = None

    def attention_block(self, layer, xn, n, kv):
        ops, lp = self.ops, xn.shape[0]
        # weights only when an op needs them: the NPU keeps its own copies
        a = functools.lru_cache(maxsize=1)(lambda: self.w.attn(layer))
        q = ops.gemm(xn, lambda: (a()["wq"], a()["bq"]), key=("q", layer))
        k = ops.gemm(xn, lambda: (a()["wk"], a()["bk"]), key=("k", layer))
        v = ops.gemm(xn, lambda: (a()["wv"], a()["bv"]), key=("v", layer))
        cos, sin = g.rope_cos_sin(range(lp))
        q = bf16(
            g.apply_rope(q.reshape(lp, g.N_Q_HEADS, g.DH).transpose(0, 1), cos, sin)
        )
        k = bf16(
            g.apply_rope(k.reshape(lp, g.N_KV_HEADS, g.DH).transpose(0, 1), cos, sin)
        )
        v = v.reshape(lp, g.N_KV_HEADS, g.DH).transpose(0, 1)
        kv.append((k[:, :n], v[:, :n]))
        sinks = self.w.vec(f"model.layers.{layer}.self_attn.sinks.weight")
        o = ops.attention(q, k, v, sinks, layer, n)
        o = o.transpose(0, 1).reshape(lp, g.N_Q_HEADS * g.DH)
        return ops.gemm(o, lambda: (a()["wo"], a()["bo"]), key=("o", layer))

    def moe_block(self, layer, xn, n):
        ops = self.ops
        wr, br = self.w.router(layer)
        logits = ops.gemm(xn, lambda: (wr, None), key=("router", layer))[:n] + br
        top_v, top_i = torch.topk(logits, g.TOP_K, dim=-1)
        top_w = torch.softmax(top_v, dim=-1)
        # experts by count, most rows first (FLM's order)
        experts = torch.bincount(top_i.flatten(), minlength=g.N_EXPERTS)
        tasks = []
        for e in torch.argsort(experts, descending=True).tolist():
            rows, slot = (top_i == e).nonzero(as_tuple=True)
            if not len(rows):
                break
            tasks.append((e, rows, top_w[rows, slot, None]))
        return ops.experts(self.w, layer, xn[:n], tasks), top_i

    @torch.no_grad()
    def forward(self, ids, collect=None, lm=True):
        """Logits of the last prompt position (None without `lm`), and each
        layer's (K, V) of the prompt."""
        n = len(ids)
        lp = pad_rows(n)
        x = torch.zeros(lp, g.D)
        x[:n] = self.w.embed(torch.as_tensor(ids))
        kv = []
        for layer in range(g.N_LAYERS):
            self.ops.layer_begin(self.w, layer)
            n1, n2 = self.w.norms(layer)
            x = x + self.attention_block(layer, bf16(g.rmsnorm(x, n1)), n, kv)
            m, top_i = self.moe_block(layer, bf16(g.rmsnorm(x, n2)), n)
            x[:n] += m
            x[n:] = 0
            if collect is not None:
                collect.append((x[:n].clone(), top_i.clone()))
        if not lm:
            return None, kv
        if self._head is None:
            self._head = self.w.lm_head()
        return g.rmsnorm(x[n - 1], self.w.final_norm()) @ self._head.T, kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--prompt-len", type=int, default=0)
    ap.add_argument("--npu", action="store_true", help="GEMMs on the NPU")
    ap.add_argument("--build-dir", default="build_prefill")
    ap.add_argument("--cpu-attn", action="store_true", help="attention on the CPU")
    ap.add_argument("--no-ref", action="store_true", help="skip the CPU reference")
    ap.add_argument(
        "--warm", action="store_true", help="time a second run, weights resident"
    )
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(g.flm_model_dir() / "tokenizer.json"))
    ids = tok.encode(args.prompt).ids
    if args.prompt_len:
        ids = (ids * (args.prompt_len // len(ids) + 1))[: args.prompt_len]
    w = g.GptOssQ4nx()
    got_c, ref_c = [], []
    if args.npu:
        import filelock

        # pyxrt directly, so the NPU lock is ours to take
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            ops = NpuOps(args.build_dir)
            if args.cpu_attn:
                ops.attention = types.MethodType(CpuOps.attention, ops)
            pf = Prefill(w, ops)
            if args.warm:
                pf.forward(ids)
            TIMES.clear()
            t0 = time.perf_counter()
            logits, _ = pf.forward(ids, collect=got_c)
            total = time.perf_counter() - t0
        print(f"NPU prefill: {total:.2f} s for {len(ids)} tokens")
        for name, t in sorted(TIMES.items(), key=lambda kv: -kv[1]):
            print(f"  {name:16s} {t:7.2f} s")
        print(f"  {'other (host)':16s} {total - sum(TIMES.values()):7.2f} s")
        dev = {"gemm": sum(e.device_s for e in ops.elfs.values())}
        dev["dequant"] = ops.dq.device_s if ops.dq else 0.0
        dev["attention"] = ops.attn.device_s if ops.attn else 0.0
        print("  device time (both runs):", {k: round(v, 2) for k, v in dev.items()})
    else:
        logits, _ = Prefill(w).forward(ids, collect=got_c)
    if args.no_ref:
        a = int(logits.argmax())
        print(f"next token {a} {tok.decode([a])!r}")
        return
    ref = g.Reference(w).forward(ids, collect=ref_c)
    for layer, ((x, ti), (rx, rti)) in enumerate(zip(got_c, ref_c)):
        cos = torch.nn.functional.cosine_similarity(x, rx, dim=-1).min()
        agree = (ti.sort(-1).values == rti.sort(-1).values).float().mean()
        print(f"layer {layer:2d}: hidden cos min {cos:.5f}, experts agree {agree:.3f}")
    cos = torch.nn.functional.cosine_similarity(logits, ref, dim=0)
    a, b = int(logits.argmax()), int(ref.argmax())
    print(f"logits cos {cos:.5f}; next token {a} {tok.decode([a])!r} vs ref {b}")
    print("PASS!" if a == b and cos > 0.99 else "failed.")


if __name__ == "__main__":
    main()
