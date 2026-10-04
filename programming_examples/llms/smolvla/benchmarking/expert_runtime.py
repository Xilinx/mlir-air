# SPDX-License-Identifier: MIT
"""SmolVLA's action expert on the NPU: one denoising call = one launch of the
gemm_engine ELF expert_engine_probe.py builds (all 16 layers, --ondev).

What the device reads besides x changes once per action chunk, not per call:
the cross layers' K/V are the expert's projections of the backbone's prefix
K/V, and the self layers' prefix K/V are the cached ones rotated back by the
first action position (their own 50 keys come from each call's QKV job). So
the K/V blocks and masks are packed when the prefix changes and written into
the resident weights and arena buffers; a call writes only x into the arena,
runs, and reads back only the last layer's output.
"""
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from bfp16_rows_pack import pack_rows_bfp16ebs8
import backbone_npu as bn  # noqa: F401  (sys.path setup)
import expert_capture as ec
import expert_engine_probe as xp
from gemm_engine import (
    arena_layout,
    build_gemm_engine,
    compile_mm_engine,
    rope_pair_perm,
    rope_table,
    weights_layout,
)

BO_KEY = "expert"
CACHE_DIR = Path(__file__).resolve().parent / "build"
REC = 9  # bytes per bfp16ebs8 record: shared exponent + 8 mantissas


def _bf(a):
    return np.asarray(a, np.float32).astype(bfloat16)


def _rec(k_rows, kg, n):
    """Record index, within pack_b_bfp16ebs8(B [k_rows, N], TN, TK1)'s output, of B[8kg:8kg+8, n]."""
    kbi_n, nbi_n = xp.TK1 // 8, xp.TN // 8
    nb, nbi, ni = n // xp.TN, (n % xp.TN) // 8, n % 8
    kb, kbi = kg // kbi_n, kg % kbi_n
    return (((nb * (k_rows // xp.TK1) + kb) * nbi_n + nbi) * kbi_n + kbi) * 8 + ni


def _gu_col(o):
    """permute_gate_up's column for gate column o."""
    half, cols = xp.TN // 2, xp.L2N // xp.TN
    p, c, hh, i = o // xp.L2N, (o % xp.L2N) // xp.TN, (o % xp.TN) // half, o % half
    return ((2 * p + hh) * cols + c) * xp.TN + i


def _merge(spans):
    """Sorted (offset, size) byte ranges, adjacent ones joined."""
    out = []
    for off, size in spans:
        if out and out[-1][0] + out[-1][1] == off:
            out[-1] = (out[-1][0], out[-1][1] + size)
        else:
            out.append((off, size))
    return out


class ExpertRuntime:
    def __init__(self, policy, profile=False):
        from air.backend.xrt import XRTCompileArtifact
        from matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
            pack_b_bfp16ebs8,
        )
        from shared.infra.cache import KernelCache, Profiler

        self._pack_b = pack_b_bfp16ebs8
        self.layers, self.meta = ec.expert_weights(policy)
        n = self.n_layers = len(self.layers)
        self.self_attn = [ec.is_self(l, self.meta) for l in range(n)]
        jobs = [
            j
            for l in range(n)
            for j in xp.layer_jobs(l, self.self_attn[l], xp.PARTS, True)
        ]
        self.lay = arena_layout(xp.M, jobs, xp.TILE_M, xp.HERD, xp.L2N)
        self.wbase, wrows = weights_layout(jobs, xp.TN, xp.L2N)

        compile_mm_engine(xp.TILE_M, xp.TN, xp.TK1, xp.SFX, xp.OBJ, rms_k=xp.E_REAL)
        self.cache = KernelCache(
            str(CACHE_DIR / f"expert_engine_L{n}_sl_ondev"),
            verbose=False,
            profiler=Profiler(enabled=profile),
        )
        elf = self.cache.cache_dir / "eng.elf"
        if elf.exists():
            self.cache.artifacts["eng"] = XRTCompileArtifact(
                str(elf), "main:gemm_engine", None
            )
        else:
            module = build_gemm_engine(
                xp.M,
                jobs,
                xp.TILE_M,
                xp.TN,
                xp.TK1,
                xp.L2N,
                xp.HERD,
                xp.HERD,
                xp.SFX,
                xp.OBJ,
                arg_order=["wts", "act"],
                arena="act",
                weights="wts",
                shim_at_launch=True,
            )
            self.cache.compile_and_cache("eng", module, xp.BACKEND)

        # Layer matrices: packed once. K/V blocks: per prefix (_set_prefix).
        self.wts = None
        for l, w in enumerate(self.layers):
            wl = dict(
                anorm=xp.pad(w["anorm"], (xp.E,)),
                fnorm=xp.pad(w["fnorm"], (xp.E,)),
                wq=xp.pad(w["wq"], (xp.E, xp.NH * xp.HD)),
                wk=xp.pad(w["wk"], (xp.E, xp.KV)),
                wv=xp.pad(w["wv"], (xp.E, xp.KV)),
                wo=xp.pad(w["wo"], (xp.NH * xp.HD, xp.E)),
                wg=xp.pad(w["wg"], (xp.E, xp.H)),
                wu=xp.pad(w["wu"], (xp.E, xp.H)),
                wd=xp.pad(w["wd"], (xp.H, xp.E)),
                k=np.zeros((xp.KP, xp.KV)),
                v=np.zeros((xp.KP, xp.KV)),
                valid=np.zeros(xp.KP, bool),
            )
            for name, b in xp.layer_b(wl).items():
                if not name.startswith(("kb", "vb")):
                    self._put(l, name, b, wrows)
        self._row_bytes = self.wts[0].nbytes
        self.act = self.lay.empty()
        lut = _bf(xp.rope_lut()).astype(np.float32)
        self.lay.pack(
            self.act, "rope", _bf(rope_table(lut, xp.NH, xp.NKV, xp.HD, xp.KV))
        )
        tile_bytes = self.act[0].nbytes
        span = lambda nm: (
            self.lay.base[nm] * tile_bytes,  # noqa: E731
            (self.lay.width[nm] // xp.L2N)
            * (xp.M // (xp.TILE_M * xp.HERD))
            * tile_bytes,
        )
        self._x_in, self._x_out = span("x0"), span(f"x{n}")
        # What a new prefix rewrites: the K/V blocks' weight rows and the masks' arena tiles.
        kv = sorted(
            (
                self.wbase[key] * self._row_bytes,
                (end - self.wbase[key]) * self._row_bytes,
            )
            for key, end in zip(
                sorted(self.wbase, key=self.wbase.get),
                sorted(self.wbase.values())[1:] + [wrows],
            )
            if key.startswith(("kb", "vb"))
        )
        self._prefix_spans = (
            _merge(kv),
            _merge(sorted(span(nm) for nm in ("mask_self", "mask_cross"))),
        )
        self._prefix = None
        self._kv_src = None
        self._n_pre = None
        self._written = False
        self.timings = {}

    def _put(self, l, name, b, wrows=None):
        key = name.replace("_", str(l) + "_") if "_" in name else name + str(l)
        packed = self._pack_b(_bf(b), xp.TN, xp.TK1)
        rows = packed.reshape(-1, xp.L2N // xp.TK1, packed.shape[-1])
        if self.wts is None:
            self.wts = np.zeros(
                (wrows, xp.L2N // xp.TK1, packed.shape[-1]), packed.dtype
            )
        self.wts[self.wbase[key] : self.wbase[key] + len(rows)] = rows

    def _layer_kv(self, l, k_cache, v_cache, p0):
        """Layer l's prefix K, V [n_pre, KV] (bf16-valued f32)."""
        kc, vc, n_pre = k_cache[l], v_cache[l], k_cache.shape[1]
        if self.self_attn[l]:
            k = ec.rope(
                kc.reshape(n_pre, xp.NKV, xp.HD),
                np.full(n_pre, -p0),
                self.meta["theta"],
            )
            k, v = k.reshape(n_pre, -1), vc
        else:
            w = self.layers[l]
            k, v = kc @ w["wk"], vc @ w["wv"]
        return _bf(k).astype(np.float32), _bf(v).astype(np.float32)

    def _index(self, n_pre):
        """Where _fast_kv's records go: each kv group's K and V records are packed once, per layer
        [K: NKV, 8 dim groups, n_pre keys | V: NKV, nk8 key groups, HD dims], and every q head of
        the group (its head tile t, slot j) takes a copy."""
        nk8 = self._nk8 = -(-n_pre // 8)
        nk, nv = xp.NKV * 8 * n_pre, xp.NKV * nk8 * xp.HD
        d8, key = np.ix_(np.arange(8), np.arange(n_pre))
        key8, d = np.ix_(np.arange(nk8), np.arange(xp.HD))
        src, dst = [], []
        for l in range(self.n_layers):
            for t in range(xp.NH // xp.HPT):
                kb0, vb0 = (
                    self.wbase[f"{nm}{l}_{t}"] * self._row_bytes // REC
                    for nm in ("kb", "vb")
                )
                for j in range(xp.HPT):
                    g = (t * xp.HPT + j) // (xp.NH // xp.NKV)
                    base = l * (nk + nv)
                    src += [
                        base + g * 8 * n_pre + np.arange(8 * n_pre),
                        base + nk + g * nk8 * xp.HD + np.arange(nk8 * xp.HD),
                    ]
                    dst += [
                        kb0 + _rec(xp.L2N, j * 8 + d8, j * xp.KP + key).ravel(),
                        vb0
                        + _rec(
                            xp.HPT * xp.KP,
                            j * (xp.KP // 8) + key8,
                            _gu_col(j * xp.HD + d),
                        ).ravel(),
                    ]
        self._src, self._dst = np.concatenate(src), np.concatenate(dst)

    def _fast_kv(self, kvs):
        """Pack only the K/V records into self.wts (the rest of kb_t / vb_t -- zeros past the
        prefix, the P.1 ones -- is what the full pack for this prefix length left there).
        """
        n_pre, nk8 = len(kvs[0][0]), self._nk8
        vals = []
        for k, v in kvs:
            kp = k[:, rope_pair_perm(xp.NKV, xp.HD)].reshape(n_pre, xp.NKV, 8, 8)
            vp = np.zeros((nk8 * 8, xp.KV), np.float32)
            vp[:n_pre] = v
            vals += [
                kp.transpose(1, 2, 0, 3).reshape(-1, 8),  # [NKV, 8, n_pre, 8]
                vp.reshape(nk8, 8, xp.NKV, xp.HD).transpose(2, 0, 3, 1).reshape(-1, 8),
            ]  # [NKV, nk8, HD, 8]
        # vals are already bf16-valued, so this is pack_b_bfp16ebs8(vals.T, 8, 8) without its transpose.
        packed = pack_rows_bfp16ebs8(np.concatenate(vals))
        # One 9-byte void element per record: a fancy-indexed copy 2.5x faster than 9-column uint8 rows.
        rec = lambda a: a.reshape(-1, REC).view(f"V{REC}").reshape(-1)  # noqa: E731
        rec(self.wts)[self._dst] = rec(packed)[self._src]

    def _set_prefix(self, k_cache, v_cache, mask, pos):
        """K/V blocks and masks for a prefix: k_cache/v_cache [layers, keys, KV], mask [S, keys + S]."""
        p0, n_pre = int(pos.min()), k_cache.shape[1]
        kvs = [self._layer_kv(l, k_cache, v_cache, p0) for l in range(self.n_layers)]
        if n_pre != self._n_pre:
            valid = np.arange(xp.KP) < n_pre
            for l, (k, v) in enumerate(kvs):
                for name, b in xp.attn_b(
                    xp.pad(k, (xp.KP, xp.KV)), xp.pad(v, (xp.KP, xp.KV)), valid
                ).items():
                    self._put(l, name, b)
            self._index(n_pre)
            self._n_pre = n_pre
        else:
            self._fast_kv(kvs)
        for nm, mk in (
            ("mask_self", xp.engine_mask(mask, n_pre)),
            ("mask_cross", xp.engine_mask(mask[:, :n_pre])),
        ):
            self.lay.pack(
                self.act,
                nm,
                _bf(
                    np.tile(
                        np.where(mk, 0.0, -1e30), (1, self.lay.width[nm] // mk.shape[1])
                    )
                ),
            )
        if (
            self._written
        ):  # resident buffers: load_and_run writes the weights only on its first call
            import pyxrt as xrt

            t = time.perf_counter()
            for bo, a, spans in zip(
                self.cache._cached_bos[BO_KEY], (self.wts, self.act), self._prefix_spans
            ):
                src = a.view(np.uint8).ravel()
                mv = np.frombuffer(bo.map(), np.uint8, count=src.size)
                for off, size in spans:
                    mv[off : off + size] = src[off : off + size]
                    bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, off)
            self.timings["prefix_sync_ms"] = (
                self.timings.get("prefix_sync_ms", 0.0)
                + (time.perf_counter() - t) * 1e3
            )

    def forget_prefix(self):
        """Make the next call repack its prefix, as a new observation's would be (benchmarking)."""
        self._prefix = None
        self._kv_src = None

    def __call__(self, x, kv, mask, pos, kv_src=None):
        """One denoising call: x [S, 720] suffix embeddings; returns the final-normed output [S, 720].
        kv: (k_cache, v_cache) [layers, keys, KV], or a function returning them. kv_src: the object
        they are read from (the backbone's KV cache): while the same one is passed they are taken
        as unchanged, not read and compared (it is held weakly, so a new cache never aliases it).
        """
        import weakref

        import pyxrt as xrt

        t0 = time.perf_counter()
        same_kv = (
            kv_src is not None and self._kv_src is not None and self._kv_src() is kv_src
        )
        if same_kv:
            prefix = self._prefix[:2] + (mask, pos)
        else:
            prefix = (*(kv() if callable(kv) else kv), mask, pos)
        if self._prefix is None or not all(
            np.array_equal(a, b) for a, b in zip(prefix, self._prefix)
        ):
            self._set_prefix(*prefix)
            self._prefix = tuple(np.array(a) for a in prefix)
            self.timings["prefix_ms"] = (
                self.timings.get("prefix_ms", 0.0) + (time.perf_counter() - t0) * 1e3
            )
            self.timings["prefixes"] = self.timings.get("prefixes", 0) + 1
        self._kv_src = weakref.ref(kv_src) if kv_src is not None else None
        self.lay.pack(self.act, "x0", _bf(xp.pad(x, (xp.M, xp.E))))
        t1 = time.perf_counter()
        if not self._written:
            got = self.cache.load_and_run(
                "eng",
                xp.BACKEND,
                self.wts,
                self.act,
                output_indices=[1],
                static_input_indices={0},
                bo_key=BO_KEY,
            )[1]
            got = np.asarray(got).reshape(self.act.shape)
            self._written = True
        else:
            # Only x changed: write its tiles, launch with the arena resident, sync back the output's.
            bo = self.cache._cached_bos[BO_KEY][1]
            mv = np.frombuffer(bo.map(), np.uint8, count=self.act.nbytes)
            (off, size), src = self._x_in, self.act.view(np.uint8).ravel()
            mv[off : off + size] = src[off : off + size]
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, off)
            self.cache.load_and_run(
                "eng",
                xp.BACKEND,
                self.wts,
                self.act,
                output_indices=[],
                static_input_indices={0},
                intermediate_indices={1},
                bo_key=BO_KEY,
            )
            off, size = self._x_out
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, off)
            got = mv.view(bfloat16).reshape(self.act.shape)
        t2 = time.perf_counter()
        out = self.lay.unpack(got, f"x{self.n_layers}")
        out = ec.rms(
            out.astype(np.float32)[: len(x), : xp.E_REAL],
            self.meta["norm"],
            self.meta["eps"],
        )
        self.timings["run_ms"] = self.timings.get("run_ms", 0.0) + (t2 - t1) * 1e3
        self.timings["calls"] = self.timings.get("calls", 0) + 1
        return out
