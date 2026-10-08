# SPDX-License-Identifier: MIT
"""SmolVLA action-expert layers, attention included, as gemm_engine jobs in ONE launch.

Per layer, 10 jobs over a tile-major activation arena (gemm_engine.ArenaLayout):
  qkv    rms + QKV + RoPE (norm folded into W_qkv, 1/sqrt(head_dim) into W_q)
  s_t    t = 0..2: scores of the 5 q heads in Q column tile t against a block
         diagonal K^T (each head's keys padded to 320), + additive mask, exp
  pv_t   [P.V | P.1] of those heads, paired like SwiGLU: drain P.V / P.1
  o      attn @ Wo + x
  gu     rms + Gate|Up + SwiGLU
  dn     sw @ Wdn + res1 -> next layer's x
Softmax without the running max: fine while scores stay below ~88.

Expert shapes: M = 50 action tokens padded to 64, hidden 720 padded to 960 (15 q
heads x 64), MLP 2048 padded to 2240. K/V (5 kv heads) are inputs: LK valid keys
per layer (241 prefix keys for cross layers, 241 + 50 for self layers), so the
probe times the device work and checks it against an fp32 reference, not the
self layers' data flow (their action K/V come from this step's QKV).
CPU reference point: ~0.96 ms per layer-step (153 ms for 10 steps x 16 layers).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
from gemm_engine import (
    Job,
    arena_layout,
    build_gemm_engine,
    compile_mm_engine,
    permute_gate_up,
    qkv_col_perm,
    rope_pair_perm,
    rope_table,
    weights_layout,
)

M_REAL, M = 50, 64
E_REAL, E = 720, 960
H_REAL, H = 2048, 2240
NH, NKV, HD = 15, 5, 64
KV = NKV * HD
PREFIX = 241
KP = 320  # keys per head, padded
TILE_M, TN, TK1, HERD = 16, 80, 160, 4
L2N = TN * HERD
HPT = L2N // HD  # q heads per arena tile
SFX, OBJ = "_xeng", "mm_xengine.o"
BACKEND = {
    "verbose": False,
    "omit_while_true_loop": False,
    "output_format": "elf",
    "instance_name": "gemm_engine",
}


PARTS = ("qkv", "s", "pv", "o", "gu", "dn")


def layer_jobs(l, self_attn, parts=PARTS, ondev=False):
    """ondev: a self layer's own 50 keys come from its QKV job on the device (Job.own):
    kb/vb hold the prefix only, and each P tile gains an own-key window."""
    s, nx = str(l), str(l + 1)
    mask = "mask_self" if self_attn else "mask_cross"
    own = ondev and self_attn
    kv = dict(kv="qkv" + s, kv_off=E) if own else {}
    jobs = {
        "qkv": [
            Job("x" + s, "wqkv" + s, "qkv" + s, E, E + 2 * KV, rms=True, rope="rope")
        ],
        "s": [
            Job(
                "qkv" + s,
                f"kb{s}_{t}",
                f"p{s}_{t}",
                L2N,
                HPT * KP + (L2N if own else 0),
                residual=mask,
                exp=True,
                a_off=t * L2N,
                **(dict(own="s", head_t=t, **kv) if own else {}),
            )
            for t in range(NH // HPT)
        ],
        "pv": [
            Job(
                f"p{s}_{t}",
                f"vb{s}_{t}",
                "attn" + s,
                HPT * KP + (L2N if own else 0),
                2 * L2N,
                div=True,
                c_off=t * L2N,
                **(dict(own="pv", head_t=t, **kv) if own else {}),
            )
            for t in range(NH // HPT)
        ],
        "o": [Job("attn" + s, "wo" + s, "res1" + s, E, E, residual="x" + s)],
        "gu": [Job("res1" + s, "wgu" + s, "sw" + s, E, 2 * H, rms=True, swiglu=True)],
        "dn": [Job("sw" + s, "wdn" + s, "x" + nx, H, E, residual="res1" + s)],
    }
    return [j for p in PARTS if p in parts for j in jobs[p]]


PRE_PASSES = 4  # 64-key passes of the 241 prefix keys (256 slots)


def layer_jobs_v2(l, self_attn, parts=PARTS, prefix_passes=PRE_PASSES):
    """The layer without any host-packed K/V: every key is read as bf16 from an arena tile through
    the B channel (Job.own). The prefix K|V tiles `kvp{l}_{p}` come from the prefix engine
    (prefix_jobs_v2, the step engine's external `kv` argument); a self layer's own 50 keys are the
    qkv job's K|V tiles in the step arena (the last pass). Cross layers need only Q from the qkv job.
    """
    s, nx = str(l), str(l + 1)
    mask = "mask_self" if self_attn else "mask_cross"
    srcs = tuple((f"kvp{s}_{p}", 0) for p in range(prefix_passes)) + (
        (("qkv" + s, E),) if self_attn else ()
    )
    n_p = len(srcs)
    own = dict(own_passes=n_p, kv="kvp" + s, kv_srcs=srcs)
    jobs = {
        "qkv": [
            Job(
                "x" + s,
                ("wqkv" if self_attn else "wq") + s,
                "qkv" + s,
                E,
                E + (2 * KV if self_attn else 0),
                rms=True,
                rope="rope",
            )
        ],
        "s": [
            Job(
                "qkv" + s,
                "kbn",
                f"p{s}_{t}",
                L2N,
                n_p * L2N,
                residual=mask,
                exp=True,
                a_off=t * L2N,
                own="s",
                head_t=t,
                **own,
            )
            for t in range(NH // HPT)
        ],
        "pv": [
            Job(
                f"p{s}_{t}",
                "vbn",
                "attn" + s,
                n_p * L2N,
                2 * L2N,
                div=True,
                c_off=t * L2N,
                own="pv",
                head_t=t,
                **own,
            )
            for t in range(NH // HPT)
        ],
        "o": [Job("attn" + s, "wo" + s, "res1" + s, E, E, residual="x" + s)],
        "gu": [Job("res1" + s, "wgu" + s, "sw" + s, E, 2 * H, rms=True, swiglu=True)],
        "dn": [Job("sw" + s, "wdn" + s, "x" + nx, H, E, residual="res1" + s)],
    }
    return [j for p in PARTS if p in parts for j in jobs[p]]


def prefix_jobs_v2(self_flags, prefix_passes=PRE_PASSES):
    """The prefix engine, once per chunk: kvp{l}_{p} = [kc{l}_{p} @ WK | vc{l}_{p} @ WV] for 64 keys per pass.
    Cross layers: the expert's k/v projections (columns pair-interleaved like Q). Self layers: K is the
    backbone's K rotated by -p0 (one 320 x 320 block-diagonal matrix `wr`, built per chunk, columns
    pair-interleaved) and V is copied (identity `wid`)."""
    jobs = []
    for l, self_attn in enumerate(self_flags):
        for p in range(prefix_passes):
            jobs += [
                Job(
                    f"kc{l}_{p}",
                    "wr" if self_attn else f"wk{l}",
                    f"kvp{l}_{p}",
                    L2N,
                    L2N,
                ),
                Job(
                    f"vc{l}_{p}",
                    "wid" if self_attn else f"wv{l}",
                    f"kvp{l}_{p}",
                    L2N,
                    L2N,
                    c_off=L2N,
                ),
            ]
    return jobs


def pad(a, shape):
    out = np.zeros(shape, np.float32)
    out[tuple(slice(0, n) for n in a.shape)] = a
    return out


def valid_keys(self_attn):
    v = np.zeros(KP, bool)
    v[:PREFIX] = True
    if self_attn:
        v[PREFIX : PREFIX + M_REAL] = True
    return v


def rope_lut():
    inv = 1.0 / (10000.0 ** (np.arange(0, HD, 2) / HD))
    ang = np.outer(np.arange(M), inv)
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


def rope_ref(x, lut, heads):
    x = x.reshape(x.shape[0], heads, HD)
    c, s = lut[:, None, : HD // 2], lut[:, None, HD // 2 :]
    x1, x2 = x[..., : HD // 2], x[..., HD // 2 :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).reshape(
        x.shape[0], -1
    )


def make_layer(rng, self_attn):
    """Random layer weights (bf16-valued f32) and its K/V [KP, KV] (rows past the valid keys unused)."""
    bf = (
        lambda a: np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)
    )  # noqa: E731
    r = lambda *s: rng.standard_normal(s)  # noqa: E731
    return dict(
        anorm=bf(pad(1 + 0.1 * r(E_REAL), (E,))),
        fnorm=bf(pad(1 + 0.1 * r(E_REAL), (E,))),
        wq=bf(pad(r(E_REAL, NH * HD) / np.sqrt(E_REAL), (E, NH * HD))),
        wk=bf(pad(r(E_REAL, KV) / np.sqrt(E_REAL), (E, KV))),
        wv=bf(pad(r(E_REAL, KV) / np.sqrt(E_REAL), (E, KV))),
        wo=bf(pad(r(NH * HD, E_REAL) / np.sqrt(NH * HD), (NH * HD, E))),
        wg=bf(pad(r(E_REAL, H_REAL) / np.sqrt(E_REAL), (E, H))),
        wu=bf(pad(r(E_REAL, H_REAL) / np.sqrt(E_REAL), (E, H))),
        wd=bf(pad(r(H_REAL, E_REAL) / np.sqrt(H_REAL), (H, E))),
        k=bf(r(KP, KV)),
        v=bf(r(KP, KV)),
        self_attn=self_attn,
    )


def layer_b(w):
    """The layer's B matrices (f32, [k, n]) by job-B suffix."""
    wqkv = np.concatenate([w["wq"] / np.sqrt(HD), w["wk"], w["wv"]], axis=1)
    wqkv = (w["anorm"][:, None] * wqkv)[:, qkv_col_perm(NH, NKV, HD)]
    out = {"wqkv": wqkv}
    out.update(
        attn_b(
            w["k"], w["v"], w["valid"] if "valid" in w else valid_keys(w["self_attn"])
        )
    )
    out["wo"] = w["wo"]
    out["wgu"] = permute_gate_up(
        w["fnorm"][:, None] * w["wg"], w["fnorm"][:, None] * w["wu"], TN, L2N
    )
    out["wdn"] = w["wd"]
    return out


def attn_b(k, v, valid):
    """kb_t / vb_t (f32) from K, V [KP, KV] and the keys they hold, bool [KP]."""
    f32 = np.float32
    kp = k[:, rope_pair_perm(NKV, HD)]  # q/k dims are pair-interleaved on the device
    out = {}
    for t in range(NH // HPT):
        kb = np.zeros((L2N, HPT * KP), f32)
        vb = np.zeros((HPT * KP, L2N), f32)
        ones = np.zeros((HPT * KP, L2N), f32)
        for j in range(HPT):
            g = (t * HPT + j) // (NH // NKV)
            kb[j * HD : (j + 1) * HD, j * KP : (j + 1) * KP] = (
                kp[:, g * HD : (g + 1) * HD] * valid[:, None]
            ).T
            vb[j * KP : (j + 1) * KP, j * HD : (j + 1) * HD] = (
                v[:, g * HD : (g + 1) * HD] * valid[:, None]
            )
            ones[j * KP : (j + 1) * KP, j * HD : (j + 1) * HD] = valid[:, None]
        out[f"kb_{t}"] = kb
        out[f"vb_{t}"] = permute_gate_up(vb, ones, TN, L2N)
    return out


def engine_mask(m, n_pre=None):
    """The engine's bool mask [M, width] for lerobot's m [M_REAL, keys] (True = attend). n_pre: m's
    keys from n_pre on are the layer's own (Job(own="s")): each head's KP columns get the prefix,
    then the own-key window in Job(own="s")'s column order (key group c, head, key)."""
    mp = np.zeros((M, KP), bool)
    mp[:M_REAL, : m.shape[1]] = m
    own = None
    if n_pre is not None:
        own = np.zeros((M, M), bool)
        own[:M_REAL, :M_REAL] = mp[:M_REAL, n_pre : n_pre + M_REAL]
        own[M_REAL:] = own[M_REAL - 1]
        mp[:, n_pre:] = False
    mp[M_REAL:] = mp[M_REAL - 1]  # padded rows: any non-empty row, so P.1 > 0
    if own is None:
        return mp
    # Key 16c + key, the same for every head.
    own = np.repeat(own.reshape(M, HERD, 1, TILE_M), HPT, axis=2).reshape(M, L2N)
    return np.concatenate([np.tile(mp, (1, HPT)), own], axis=1)


def reference_layer(x, w, lut):
    """fp32 expert layer on the real rows/columns. x [M_REAL, E_REAL]."""

    def rms(a, g):
        return a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-5) * g

    qkv = (
        rms(x, w["anorm"][:E_REAL])
        @ np.concatenate([w["wq"], w["wk"], w["wv"]], axis=1)[:E_REAL]
    )
    q = rope_ref(qkv[:, : NH * HD], lut[:M_REAL], NH).reshape(M_REAL, NH, HD)
    valid = valid_keys(w["self_attn"])
    k, v = w["k"][valid].reshape(-1, NKV, HD), w["v"][valid].reshape(-1, NKV, HD)
    att = np.empty((M_REAL, NH, HD), np.float32)
    for h in range(NH):
        s = q[:, h] @ k[:, h // (NH // NKV)].T / np.sqrt(HD)
        p = np.exp(s - s.max(-1, keepdims=True))
        att[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, h // (NH // NKV)]
    res1 = att.reshape(M_REAL, -1) @ w["wo"][:, :E_REAL] + x
    n2 = rms(res1, w["fnorm"][:E_REAL])
    g, u = n2 @ w["wg"][:E_REAL, :H_REAL], n2 @ w["wu"][:E_REAL, :H_REAL]
    return (g / (1 + np.exp(-g)) * u) @ w["wd"][:H_REAL, :E_REAL] + res1


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class RealData:
    """SmolVLA's expert weights and the captured denoising calls (expert_capture.py).

    Layer l is self-attention when l % 2 == 0, as in lerobot. K/V are inputs here:
    a cross layer's are its projections of the cached prefix K/V, a self layer's
    are the prefix's plus its own 50, taken from the fp32 reference chain (on the
    device they would come from this step's QKV job).
    """

    def __init__(self):
        import expert_capture as ec

        self.ec = ec
        self.policy = ec.load_policy()
        self.layers, self.meta = ec.expert_weights(self.policy)
        self.d = dict(np.load(ec.CAPTURE))
        self.n_steps = len(self.d["x"])

    def step(self, s, n_layers, ondev=0):
        return self.step_from(
            {key: self.d[key][s] for key in ("x", "k", "v", "mask", "pos")},
            n_layers,
            ondev,
        )

    def step_from(self, r, n_layers, ondev=0):
        """(padded layer dicts, {mask name: bool [M, KP]}, x0 [M_REAL, E_REAL], fp32 chain) for one
        denoising call r (x, k, v, mask, pos as in the capture). ondev: self layers l < ondev have K/V
        the prefix only, and mask_self is [M, HPT * KP + L2N]: the prefix mask per head,
        then the own-key mask in Job(own="s")'s column order (key group c, head, key).
        """
        ec = self.ec
        chain = []
        ec.reference_expert(
            r["x"],
            self.layers[:n_layers],
            self.meta,
            r["k"],
            r["v"],
            r["mask"],
            r["pos"],
            collect=chain,
        )
        bf = (
            lambda a: np.asarray(a, np.float32).astype(bfloat16).astype(np.float32)
        )  # noqa: E731
        ws, masks = [], {}
        for l in range(n_layers):
            _, k, v, m = chain[l]
            w, sa = self.layers[l], ec.is_self(l, self.meta)
            ws.append(
                dict(
                    anorm=pad(w["anorm"], (E,)),
                    fnorm=pad(w["fnorm"], (E,)),
                    wq=pad(w["wq"], (E, NH * HD)),
                    wk=pad(w["wk"], (E, KV)),
                    wv=pad(w["wv"], (E, KV)),
                    wo=pad(w["wo"], (NH * HD, E)),
                    wg=pad(w["wg"], (E, H)),
                    wu=pad(w["wu"], (E, H)),
                    wd=pad(w["wd"], (H, E)),
                    k=bf(pad(k, (KP, KV))),
                    v=bf(pad(v, (KP, KV))),
                    valid=np.arange(KP)
                    < (len(r["k"][l]) if l < ondev and sa else len(k)),
                    self_attn=sa,
                )
            )
            masks["mask_self" if sa else "mask_cross"] = engine_mask(
                m, len(r["k"][l]) if l < ondev and sa else None
            )
        return ws, masks, r["x"], chain

    def chunk(self, expert):
        """lerobot's predict_action_chunk on the capture's batch and noise, with each denoising call's expert
        replaced by expert(r) -> final-normed output [50, 720] (r: x, k, v, mask, pos).
        """
        import torch
        from smolvla_inference import build_oracle_batch, fixed_noise

        vwe, policy = self.policy.model.vlm_with_expert, self.policy
        orig = vwe.forward
        f32 = lambda t: t.detach().float().numpy()  # noqa: E731

        def wrapped(*a, **kw):
            embs, pkv = kw.get("inputs_embeds"), kw.get("past_key_values")
            if pkv is None or embs is None or embs[0] is not None:
                return orig(*a, **kw)
            r = dict(
                x=f32(embs[1][0]),
                mask=kw["attention_mask"][0].numpy(),
                pos=kw["position_ids"][0].numpy(),
                k=np.stack(
                    [
                        f32(pkv.layers[i].keys[0].transpose(0, 1))
                        for i in range(len(pkv.layers))
                    ]
                ),
                v=np.stack(
                    [
                        f32(pkv.layers[i].values[0].transpose(0, 1))
                        for i in range(len(pkv.layers))
                    ]
                ),
            )
            out = torch.from_numpy(np.asarray(expert(r), np.float32)).to(embs[1].dtype)[
                None
            ]
            return [None, out], pkv

        vwe.forward = wrapped
        try:
            policy.reset()
            with torch.no_grad():
                return (
                    policy.predict_action_chunk(
                        build_oracle_batch(policy), noise=fixed_noise(policy)
                    )
                    .float()
                    .numpy()
                )
        finally:
            vwe.forward = orig

    def ref_layer(self, x, l, chain):
        _, k, v, m = chain[l]
        return self.ec.ref_layer(x, self.layers[l], self.meta, k, v, m)

    def final(self, x):
        return self.ec.rms(x, self.meta["norm"], self.meta["eps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--compile-only", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--run-only",
        action="store_true",
        help="reuse the cached ELF (must match this code)",
    )
    ap.add_argument(
        "--shim-at-launch",
        action="store_true",
        help="shim DMAs in the launch, not the segment",
    )
    ap.add_argument(
        "--parts", default=",".join(PARTS), help="job kinds per layer (bisecting)"
    )
    ap.add_argument(
        "--real",
        action="store_true",
        help="SmolVLA weights + captured steps (expert_capture.py)",
    )
    ap.add_argument(
        "--dump", help="save every arena tensor of each step to DUMP/step{s}.npz"
    )
    ap.add_argument(
        "--ondev",
        action="store_true",
        help="with --real: self layers' own K/V computed on the device",
    )
    ap.add_argument(
        "--ondev-layers",
        type=int,
        help="with --ondev: only the first N layers (bisecting)",
    )
    ap.add_argument(
        "--arena-pad",
        type=int,
        default=0,
        help="unused arena tiles per herd row (probing)",
    )
    ap.add_argument(
        "--closed-loop",
        action="store_true",
        help="with --real at all layers: lerobot's denoising loop driven by the NPU expert",
    )
    args = ap.parse_args()
    parts = args.parts.split(",")
    full = set(parts) == set(PARTS)
    assert args.real or not args.ondev, "--ondev checks against the real fp32 chain"
    real = RealData() if args.real else None
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )
    from reconfig_probe import ctrl_kb
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler

    compile_mm_engine(TILE_M, TN, TK1, SFX, OBJ, rms_k=E_REAL)
    bfa = lambda a: np.asarray(a, np.float32).astype(bfloat16)  # noqa: E731
    lut = bfa(rope_lut()).astype(np.float32)
    for n_layers in args.layers:
        ondev_n = (
            (n_layers if args.ondev_layers is None else args.ondev_layers)
            if args.ondev
            else 0
        )
        if real:
            ws, masks, x0_real, chain = real.step(0, n_layers, ondev_n)
        else:
            rng = np.random.default_rng(0)
            ws = [make_layer(rng, l % 2 == 1) for l in range(n_layers)]
            masks = {
                nm: np.tile(valid_keys(sa), (M, 1))
                for nm, sa in (("mask_cross", False), ("mask_self", True))
            }
        jobs = [
            j
            for l in range(n_layers)
            for j in layer_jobs(l, ws[l]["self_attn"], parts, l < ondev_n)
        ]
        lay = arena_layout(M, jobs, TILE_M, HERD, L2N, args.arena_pad)
        wbase, wrows = weights_layout(jobs, TN, L2N)
        tag = (
            ("" if full else "_" + "-".join(parts))
            + ("_sl" if args.shim_at_launch else "")
            + (f"_ondev{ondev_n if ondev_n < n_layers else ''}" if ondev_n else "")
            + (f"_pad{args.arena_pad}" if args.arena_pad else "")
        )
        cache = KernelCache(
            str(
                Path(__file__).resolve().parent
                / "build"
                / f"expert_engine_L{n_layers}{tag}"
            ),
            verbose=False,
            profiler=Profiler(enabled=True),
        )
        backend = dict(BACKEND, verbose=args.verbose)
        module = build_gemm_engine(
            M,
            jobs,
            TILE_M,
            TN,
            TK1,
            L2N,
            HERD,
            HERD,
            SFX,
            OBJ,
            arg_order=["wts", "act"],
            arena="act",
            weights="wts",
            shim_at_launch=args.shim_at_launch,
            arena_pad=args.arena_pad,
        )
        try:
            elf = cache.cache_dir / "eng.elf"
            if args.run_only and elf.exists():
                from air.backend.xrt import XRTCompileArtifact

                cache.artifacts["eng"] = XRTCompileArtifact(
                    str(elf), "main:gemm_engine", None
                )
            else:
                cache.compile_and_cache("eng", module, backend)
        except Exception as e:  # noqa: BLE001
            print(
                f"L={n_layers}: compile failed: {str(e)[-4000:] if args.verbose else str(e).splitlines()[-1]}"
            )
            continue
        print(
            f"L={n_layers}: {len(jobs)} jobs, arena {lay.n_tiles} tiles/row, control code "
            f"{ctrl_kb(cache.cache_dir):.1f} KB"
        )
        if args.compile_only:
            continue

        def pack_weights(ws):
            wts = None
            for l, w in enumerate(ws):
                for name, b in layer_b(w).items():
                    key = (
                        name.replace("_", str(l) + "_")
                        if "_" in name
                        else name + str(l)
                    )
                    if key not in wbase:
                        continue
                    packed = pack_b_bfp16ebs8(bfa(b), TN, TK1)
                    rows = packed.reshape(-1, L2N // TK1, packed.shape[-1])
                    if wts is None:
                        wts = np.zeros(
                            (wrows, L2N // TK1, packed.shape[-1]), packed.dtype
                        )
                    wts[wbase[key] : wbase[key] + len(rows)] = rows
            return wts

        def pack_act(x0, masks):
            act = lay.empty()
            if "x0" in lay.base:
                lay.pack(act, "x0", x0)
            if "rope" in lay.base:
                lay.pack(act, "rope", bfa(rope_table(lut, NH, NKV, HD, KV)))
            for nm, mk in masks.items():
                if nm in lay.base:
                    lay.pack(
                        act,
                        nm,
                        bfa(
                            np.tile(
                                np.where(mk, 0.0, -1e30),
                                (1, lay.width[nm] // mk.shape[1]),
                            )
                        ),
                    )
            if not full:  # a job subset: inputs normally produced by the missing jobs
                rng_in = np.random.default_rng(2)
                for nm in lay.base:
                    if (
                        lay.base[nm] < lay.drain_lo
                        and nm not in ("x0", "rope")
                        and nm not in masks
                    ):
                        lay.pack(
                            act,
                            nm,
                            bfa(
                                pad(
                                    rng_in.standard_normal((M_REAL, lay.width[nm])),
                                    (M, lay.width[nm]),
                                )
                            ),
                        )
            return act

        def check(got, x0, ws, step=None, chain=None):
            if args.dump:
                Path(args.dump).mkdir(parents=True, exist_ok=True)
                np.savez(
                    Path(args.dump) / f"step{step or 0}.npz",
                    **{n: lay.unpack(got, n).astype(np.float32) for n in lay.base},
                )
            f = lambda n: lay.unpack(got, n).astype(np.float32)[:M_REAL]  # noqa: E731
            per, x_ref = [], x0.astype(np.float32)[:M_REAL, :E_REAL]
            for l, w in enumerate(ws):
                xin = f("x" + str(l))[:, :E_REAL] if l else x_ref
                ref = (
                    real.ref_layer(xin, l, chain)
                    if real
                    else reference_layer(xin, w, lut)
                )
                per.append(cos(f("x" + str(l + 1))[:, :E_REAL], ref))
            if real:
                chained = chain[n_layers]
            else:
                chained = x_ref
                for w in ws:
                    chained = reference_layer(chained, w, lut)
            last = f("x" + str(n_layers))
            line = (
                f"per-layer cosine vs fp32 min {min(per):.6f} ({', '.join(f'{c:.5f}' for c in per)}); "
                f"chained {cos(last[:, :E_REAL], chained):.6f}; pad cols max |x| "
                f"{float(np.abs(last[:, E_REAL:]).max()):.3g}"
            )
            if real and n_layers == len(real.layers):
                line += f"; final-normed vs lerobot {cos(real.final(last[:, :E_REAL]), real.d['out'][step]):.6f}"
            return line

        x0 = bfa(
            pad(
                (
                    x0_real
                    if real
                    else np.random.default_rng(1).standard_normal((M_REAL, E_REAL))
                ),
                (M, E),
            )
        )
        wts, act = pack_weights(ws), pack_act(x0, masks)
        run = lambda: cache.load_and_run(
            "eng", backend, wts, act, output_indices=[1], bo_key="xe"
        )  # noqa: E731
        got = np.asarray(run()[1]).reshape(act.shape)
        if not full:
            if args.dump:
                Path(args.dump).mkdir(parents=True, exist_ok=True)
                np.savez(
                    Path(args.dump) / "step0.npz",
                    **{n: lay.unpack(got, n).astype(np.float32) for n in lay.base},
                )
            print(f"L={n_layers} parts {args.parts}: ran")
        elif real:
            print(f"L={n_layers} step 0: {check(got, x0, ws, 0, chain)}")
            for s in range(1, real.n_steps):
                ws_s, masks_s, xs, chain_s = real.step(s, n_layers, ondev_n)
                x0_s = bfa(pad(xs, (M, E)))
                got_s = np.asarray(
                    cache.load_and_run(
                        "eng",
                        backend,
                        pack_weights(ws_s),
                        pack_act(x0_s, masks_s),
                        output_indices=[1],
                        bo_key="xe",
                    )[1]
                ).reshape(act.shape)
                print(f"L={n_layers} step {s}: {check(got_s, x0_s, ws_s, s, chain_s)}")
            if args.closed_loop and n_layers == len(real.layers):

                def npu_expert(r):
                    ws_r, masks_r, xr, _ = real.step_from(r, n_layers, ondev_n)
                    g = np.asarray(
                        cache.load_and_run(
                            "eng",
                            backend,
                            pack_weights(ws_r),
                            pack_act(bfa(pad(xr, (M, E))), masks_r),
                            output_indices=[1],
                            bo_key="xe",
                        )[1]
                    ).reshape(act.shape)
                    return real.final(
                        lay.unpack(g, f"x{n_layers}").astype(np.float32)[
                            :M_REAL, :E_REAL
                        ]
                    )

                ec = real.ec
                ref = real.d["chunk"]
                for name, fn in (
                    (
                        "fp32 reference",
                        lambda r: ec.reference_expert(
                            r["x"],
                            real.layers,
                            real.meta,
                            r["k"],
                            r["v"],
                            r["mask"],
                            r["pos"],
                        ),
                    ),
                    ("NPU", npu_expert),
                ):
                    c = real.chunk(fn)
                    err = np.abs(c - ref)
                    print(
                        f"closed loop, expert = {name}: action chunk {c.shape} vs lerobot cosine "
                        f"{cos(c, ref):.6f}, max |err| {err.max():.4g}, mean |err| {err.mean():.4g} "
                        f"(|lerobot| max {np.abs(ref).max():.3g}, mean {np.abs(ref).mean():.3g})"
                    )
        else:
            print(f"L={n_layers}: {check(got, x0, ws)}")
        cache.profiler.kernel_breakdowns.clear()
        for _ in range(args.iters):
            run()
        dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["eng"])
        med = dev[len(dev) // 2] * 1e3
        print(
            f"L={n_layers}: device median {med:.0f} us = {med / n_layers:.0f} us per layer "
            f"(CPU ~960 us per layer-step)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
