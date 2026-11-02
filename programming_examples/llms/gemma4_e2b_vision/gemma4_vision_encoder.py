# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4 (E2B, E4B) vision tower on NPU2.
#
# The GEMMs and the attention run on the NPU. The elementwise work between them
# runs on the host in hostops.c. Rows are the processor's patch count L padded
# to L_pad; padding patches come last, so rows >= L_valid are padding.
#
# ClippableLinear is y = clip(clip(x, in) @ W, out). The host op that writes a
# GEMM's input applies the input clamp and the one that reads its output
# applies the output clamp, so the GEMMs stay plain:
#   q/k/v -> qkv_prep    o, down -> add_rms_clip    gate, up -> glu_clip
# The gate GEMM applies GELU in its drain, before the output clamp can run.
# GELU is increasing above -0.75 and within 2e-5 of 0 below _GATE_LO_MAX, so
# with lo <= _GATE_LO_MAX (asserted per layer)
#   GELU(clip(y, lo, hi)) == min(GELU(y), GELU(hi)) to within 2e-5,
# which is what glu_clip applies.
#
# The attention kernel has no key mask, and the padding patches must not be
# attended to. Each head is widened from DH to DK_AUG columns. Column DH is 1
# in Q, 0 in a real key and -_BIG in a padding key, so every padding score is
# -_BIG. K/V hold the real rows rounded up to the key block; Q keeps all L_pad
# rows and the padding queries' outputs are ignored.
import sys
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# programming_examples/ is published as the air_examples package rather than put
# on sys.path, so its directories do not shadow installed top-level modules.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[2])
]

import hostops as H  # noqa: E402
from gemma4_vision_weights import (  # noqa: E402
    ATTN_SCALE,
    D,
    DH,
    INTER,
    NUM_LAYERS,
    N_HEADS,
    RMS_EPS,
    VisionWeights,
    gelu_tanh,
    pool,
    rmsnorm,
    rope_tables,
)

# GEMM rows: drain tile_m=32 on 8 herd rows.
_ROW_ALIGN = 256
DK_AUG = 2 * DH
_LKP = 64
_BIG = 1.0e4

_GATE_LO_MAX = -4.4


def _round_up(x, m):
    return -(-x // m) * m


class Shape:
    """Row counts of one image: L patches (L_valid real), L_pad GEMM rows,
    kv_rows attention keys, n_soft output tokens."""

    def __init__(self, L, L_valid, kv_block):
        self.L, self.L_valid = L, L_valid
        self.L_pad = _round_up(L, _ROW_ALIGN)
        self.kv_rows = _round_up(L_valid, kv_block)
        assert self.kv_rows <= self.L_pad, (self.kv_rows, self.L_pad)


class CpuOps:
    """The device ops in numpy, for checking the host pipeline on its own:
    bf16 in, f32 accumulate, bf16 out, as the NPU kernels do."""

    kv_block = _LKP

    def build(self, shape, m_soft, out_dim):
        pass

    def gemm(self, name, x, w, site=None):
        y = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
        if name == "gate":
            y = gelu_tanh(y)
        return y.astype(bfloat16)

    def attention(self, q, k, v, scale):
        """q [Lq, H*dk], k [Lk, H*dk], v [Lk, H*dh], scaled by `scale`."""
        Lq, Lk = q.shape[0], k.shape[0]
        qf = np.asarray(q, np.float32).reshape(Lq, N_HEADS, -1)
        kf = np.asarray(k, np.float32).reshape(Lk, N_HEADS, -1)
        vf = np.asarray(v, np.float32).reshape(Lk, N_HEADS, -1)
        s = np.einsum("lhd,mhd->hlm", qf, kf) * scale
        s -= s.max(-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(-1, keepdims=True)
        return np.einsum("hlm,mhd->lhd", p, vf).reshape(Lq, -1).astype(bfloat16)

    def timings(self):
        return {}


# Drain GEMM recipe per (K, N): (tile_n, herd_n); tile_m=32, herd_m=8,
# tile_k_l1=32, tile_k_l2=min(256, K).
_GEMM = {
    (D, 3 * D): (64, 4),  # q|k|v
    (D, D): (64, 3),  # patch embed, o
    (D, INTER): (128, 4),  # gate (GELU drain), up
    (INTER, D): (128, 3),  # down
}
# The multimodal projection, D -> the text model's hidden size.
_PROJ = (64, 4)
_TILE_M, _HERD_M, _TILE_K_L1 = 32, 8, 32
_GEMM_OPS = ("qkv", "o", "gate", "up", "down", "proj")

# Attention schedule: q blocks of 256 rows in 4 tiles, 2 heads per unroll.
_FA_LQP, _FA_Q_TILES, _FA_HEADS_PER_UNROLL = 256, 4, 2


def _fa_stages(kv_rows):
    """Deepest cascade (<= 4) that splits the key blocks evenly."""
    blocks = kv_rows // _LKP
    return next(s for s in (4, 3, 2, 1) if blocks % s == 0)


class NpuOps:
    """The device ops as AIR ELFs, compiled per shape into `cache_dir` and
    reused across runs."""

    kv_block = _LKP

    def __init__(self, cache_dir=None, verbose=False):
        from air_examples.llms.shared.infra.cache import KernelCache

        self.dir = Path(cache_dir or H.build_dir() / "elf_cache")
        # One image shape's ELFs (the GEMMs and the attention) stay loaded; a
        # new shape evicts the old one's, which leaves the device's remaining
        # hardware contexts to the prefill and the decoder.
        self.cache = KernelCache(
            cache_dir=str(self.dir), verbose=verbose, max_contexts=len(_GEMM_OPS) + 1
        )
        self.cache.load_manifest()
        self.verbose = verbose
        self.names = {}
        self.op_t = {}

    @staticmethod
    def _gemm_backend():
        return {
            "verbose": False,
            "omit_while_true_loop": False,
            "output_format": "elf",
            "instance_name": "matmul_bf16",
            "runtime_loop_tiling_sizes": [2, 2],
        }

    @staticmethod
    def _fa_backend():
        return {
            "verbose": False,
            "omit_while_true_loop": False,
            "omit_pingpong": "all",
            "runtime_loop_tiling_sizes": [1, 1],
            "output_format": "elf",
            "instance_name": "attention_bf16",
        }

    def build(self, shape, m_soft, out_dim):
        """Compile whatever this shape needs and is not cached yet."""
        import os

        rows = shape.L_pad
        plan = {
            "qkv": (rows, D, 3 * D, False),
            "o": (rows, D, D, False),
            "gate": (rows, D, INTER, True),
            "up": (rows, D, INTER, False),
            "down": (rows, INTER, D, False),
            "proj": (m_soft, D, out_dim, False),
        }
        assert tuple(plan) == _GEMM_OPS, tuple(plan)
        for op, (m, k, n, gelu) in plan.items():
            self.names[op] = (f"gemm_{m}x{k}x{n}" + ("_gelu" if gelu else ""), m, k, n)
        stages = _fa_stages(shape.kv_rows)
        self.names["fa"] = f"fa_{rows}x{shape.kv_rows}_s{stages}"
        todo = [v[0] for v in self.names.values() if isinstance(v, tuple)]
        todo.append(self.names["fa"])
        # Another shape's buffers (its weight copies above all) are not needed
        # again until that shape returns, and then are rebuilt on first use.
        live = set(todo)
        self.cache.drop_bos(lambda key: key.split(":")[0] not in live)
        if all(t in self.cache.artifacts for t in todo):
            return

        cwd = os.getcwd()
        build = self.dir / "build"
        build.mkdir(parents=True, exist_ok=True)
        os.chdir(build)
        try:
            self._compile_gemms(plan)
            self._compile_fa(shape, stages)
        finally:
            os.chdir(cwd)
            self.cache._save_manifest()

    def _compile_gemms(self, plan):
        from air_examples.llms.shared.builders.gemm_builder import (
            _build_gemm_module,
            _spec_with_tiles,
        )
        from air_examples.llms.shared.infra.external_kernels import compile_gemm_mm

        # DIM_N is baked into mm.o, so each tile_n gets its own object, compiled
        # first so the ELF links below find it in the build dir.
        for tn in sorted({tn for tn, _ in [*_GEMM.values(), _PROJ]}):
            sfx = f"_m{_TILE_M}n{tn}"
            compile_gemm_mm(
                tile_m=_TILE_M,
                tile_n=tn,
                tile_k_l1=_TILE_K_L1,
                sym_suffix=sfx,
                out_name=f"mm{sfx}.o",
            )
        for op, (m, k, n, gelu) in plan.items():
            name = self.names[op][0]
            if name in self.cache.artifacts:
                continue
            tn, hn = _PROJ if op == "proj" else _GEMM[(k, n)]
            assert n % (tn * hn) == 0, (op, n, tn, hn)
            tk2 = min(256, k)
            sfx = f"_m{_TILE_M}n{tn}"
            spec = _spec_with_tiles(
                "drain",
                dict(tile_m=_TILE_M, tile_k_l2=tk2, tile_k_l1=_TILE_K_L1, tile_n=tn),
            )
            kw = dict(spec["build_kwargs"], sym_suffix=sfx, link_with_name=f"mm{sfx}.o")
            mod = _build_gemm_module(
                m, k, n, _TILE_M, tk2, _TILE_K_L1, tn, _HERD_M, hn,
                epilogue_gelu=gelu, **kw,
            )  # fmt: skip
            self.cache.compile_and_cache(name, str(mod), self._gemm_backend())

    def _compile_fa(self, shape, stages):
        from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
            build_module,
        )
        from air_examples.llms.shared.infra.external_kernels import compile_attn_npu2

        name = self.names["fa"]
        if name in self.cache.artifacts:
            return
        # Block-float mmul shares an exponent across 8 values along d. The mask
        # column starts a block of its own, so -_BIG shares it only with zeros.
        compile_attn_npu2(
            head_dim=DK_AUG,
            lkp=_LKP,
            lqp_tile=_FA_LQP // _FA_Q_TILES,
            dv_tile=_LKP,
            dv_full=DH,
            bfp16=True,
            force=True,
        )
        mod = build_module(
            lk=shape.kv_rows,
            lkp=_LKP,
            lq=shape.L_pad,
            lqp=_FA_LQP,
            dk=DK_AUG,
            dv=DH,
            num_q_tiles=_FA_Q_TILES,
            num_cascade_stages=stages,
            num_heads=N_HEADS,
            num_kv_heads=N_HEADS,
            causal=False,
            num_heads_per_unroll=_FA_HEADS_PER_UNROLL,
        )
        self.cache.compile_and_cache(name, str(mod), self._fa_backend())

    def _run(self, site, name, backend, args, static):
        t = time.perf_counter()
        res = self.cache.load_and_run(
            name,
            backend,
            *args,
            output_indices=[len(args) - 1],
            static_input_indices=static,
            intermediate_indices={len(args) - 1},
            bo_key=f"{name}:{site}",
        )
        tag = site.rstrip("0123456789")
        self.op_t[tag] = self.op_t.get(tag, 0.0) + time.perf_counter() - t
        return res[-1]

    def gemm(self, op, x, w, site=None):
        """`site` names a call whose weight never changes: its weight BO is
        written once and stays resident."""
        name, m, k, n = self.names[op]
        args = [
            np.ascontiguousarray(x, bfloat16).reshape(m, k),
            np.ascontiguousarray(w, bfloat16),
            np.zeros((m, n), bfloat16),
        ]
        out = self._run(
            site or op, name, self._gemm_backend(), args, {1} if site else set()
        )
        # A copy: same-shaped calls share the output BO.
        return np.array(out, bfloat16).reshape(m, n)

    def attention(self, q, k, v, scale):
        assert abs(scale * np.sqrt(DK_AUG) - 1.0) < 1e-6, scale  # baked in
        out = np.zeros((q.shape[0], N_HEADS * DH), bfloat16)
        res = self._run(
            "fa", self.names["fa"], self._fa_backend(), [q, k, v, out], set()
        )
        return np.array(res, bfloat16).reshape(q.shape[0], N_HEADS * DH)

    def timings(self):
        return dict(self.op_t)


class VisionEncoder:
    """Gemma4 vision tower: pixels -> [n_soft, out_dim] soft tokens.

    enc = VisionEncoder(ops)            # ops: NpuOps() or CpuOps()
    soft = enc.encode(pixel_values, position_ids)
    """

    def __init__(self, ops, weights=None, n_layers=NUM_LAYERS):
        self.ops = ops
        self.w = weights or VisionWeights()
        self.n_layers = n_layers
        self.host_t = 0.0
        self._prep_weights()

    # ---------------------------------------------------------------- weights
    def _prep_weights(self):
        """Per-layer GEMM weights ([in, out] bf16, q|k|v fused) and the clamp
        and norm constants the host ops take."""
        w = self.w
        self.layers = []
        for i in range(NUM_LAYERS):
            c = {p: w.clamp(i, p) for p in ("q", "k", "v", "o", "gate", "up", "down")}
            # q/k/v share one input (and gate/up another), so one input clamp
            # each must serve the group.
            assert c["q"][0] == c["k"][0] == c["v"][0], (i, c["q"][0], c["k"][0])
            assert c["gate"][0] == c["up"][0], (i, c["gate"][0], c["up"][0])
            assert c["gate"][1][0] <= _GATE_LO_MAX, (i, c["gate"][1])
            self.layers.append(
                dict(
                    qkv=np.ascontiguousarray(
                        np.concatenate([w.wT(i, p) for p in ("q", "k", "v")], axis=1)
                    ),
                    o=w.wT(i, "o"),
                    gate=w.wT(i, "gate"),
                    up=w.wT(i, "up"),
                    down=w.wT(i, "down"),
                    clamp=c,
                    qkv_bounds=np.array(
                        [b for p in ("q", "k", "v") for b in c[p][1]], np.float32
                    ),
                    gate_hi=float(gelu_tanh(np.float32(c["gate"][1][1]))),
                    norms={
                        n: w.norm(i, n)
                        for n in (
                            "input_ln",
                            "post_attn_ln",
                            "pre_ffn_ln",
                            "post_ffn_ln",
                            "q_norm",
                            "k_norm",
                        )
                    },
                )
            )
        self.patch_w = w.patch_proj_T()
        self.proj_w = w.embed_proj_T()

    # ---------------------------------------------------------------- encode
    def shape_of(self, position_ids):
        pad = (position_ids == -1).all(axis=-1)
        L, L_valid = len(pad), int((~pad).sum())
        # The kernel and host ops take padding as a tail of rows.
        assert not pad[:L_valid].any(), "padding patches must trail the image"
        return Shape(L, L_valid, self.ops.kv_block)

    def encode(self, pixel_values, position_ids, taps=None):
        sh = self.shape_of(position_ids)
        n_slots = sh.L // 9
        m_soft = _round_up(n_slots, _ROW_ALIGN)
        self.ops.build(sh, m_soft, self.w.out_dim)
        t0 = time.perf_counter()
        self._dev_t = 0.0

        h = self._patch_embed(sh, pixel_values, position_ids)
        if taps is not None:
            taps["embeds"] = h[: sh.L].copy()
        cos, sin = rope_tables(self._pad_rows(position_ids, sh.L_pad))
        bufs = self._buffers(sh)
        for i in range(self.n_layers):
            self._layer(sh, i, h, cos, sin, bufs)
            if taps is not None:
                taps[f"L{i}.out"] = h[: sh.L].copy()

        pooled = pool(h[: sh.L], position_ids)
        x = np.zeros((m_soft, D), bfloat16)
        x[: len(pooled)] = rmsnorm(pooled, None)
        out = self._gemm("proj", x, self.proj_w, "proj")[: len(pooled)]
        self.host_t = time.perf_counter() - t0 - self._dev_t
        return np.asarray(out, np.float32)

    @staticmethod
    def _pad_rows(position_ids, rows):
        p = np.full((rows, 2), -1, np.int64)
        p[: len(position_ids)] = position_ids
        return p

    def _buffers(self, sh):
        return dict(
            x=np.empty((sh.L_pad, D), bfloat16),
            q=np.empty((sh.L_pad, N_HEADS * DK_AUG), bfloat16),
            k=np.empty((sh.kv_rows, N_HEADS * DK_AUG), bfloat16),
            v=np.empty((sh.kv_rows, N_HEADS * DH), bfloat16),
            f=np.empty((sh.L_pad, INTER), bfloat16),
        )

    def _gemm(self, name, x, w, site=None):
        t = time.perf_counter()
        y = self.ops.gemm(name, x, w, site)
        self._dev_t += time.perf_counter() - t
        return y

    def _patch_embed(self, sh, pixel_values, position_ids):
        x = np.zeros((sh.L_pad, D), bfloat16)
        x[: sh.L] = 2.0 * (np.asarray(pixel_values, np.float32) - 0.5)
        h = np.asarray(self._gemm("o", x, self.patch_w, "patch"), np.float32)
        table = self.w.position_table()
        idx = np.maximum(position_ids, 0)
        h[: sh.L] += table[0][idx[:, 0]] + table[1][idx[:, 1]]
        h[sh.L_valid :] = 0.0
        return np.ascontiguousarray(h)

    def _layer(self, sh, i, h, cos, sin, b):
        P = self.layers[i]
        c, nm = P["clamp"], P["norms"]
        eps, valid = RMS_EPS, sh.L_valid

        # ---- attention
        H.rms_clip_bf16(h, valid, nm["input_ln"], eps, *c["q"][0], b["x"])
        qkv = self._gemm("qkv", b["x"], P["qkv"], f"qkv{i}")
        # The kernel scales scores by 1/sqrt(DK_AUG), the model by ATTN_SCALE.
        qscale = float(ATTN_SCALE * np.sqrt(DK_AUG))
        H.qkv_prep(
            qkv,
            valid,
            N_HEADS,
            DH,
            DK_AUG,
            nm["q_norm"],
            nm["k_norm"],
            eps,
            cos,
            sin,
            P["qkv_bounds"],
            qscale,
            _BIG,
            b["q"],
            b["k"],
            b["v"],
        )
        t = time.perf_counter()
        attn = self.ops.attention(b["q"], b["k"], b["v"], 1.0 / np.sqrt(DK_AUG))
        self._dev_t += time.perf_counter() - t
        attn = np.ascontiguousarray(attn, bfloat16)
        H.clip_bf16(attn, *c["o"][0])
        o = self._gemm("o", attn, P["o"], f"o{i}")
        H.add_rms_clip(h, o, valid, nm["post_attn_ln"], eps, *c["o"][1])

        # ---- mlp
        H.rms_clip_bf16(h, valid, nm["pre_ffn_ln"], eps, *c["gate"][0], b["x"])
        g = self._gemm("gate", b["x"], P["gate"], f"gate{i}")
        u = self._gemm("up", b["x"], P["up"], f"up{i}")
        H.glu_clip(g, u, P["gate_hi"], *c["up"][1], *c["down"][0], b["f"])
        d = self._gemm("down", b["f"], P["down"], f"down{i}")
        H.add_rms_clip(h, d, valid, nm["post_ffn_ln"], eps, *c["down"][1])
