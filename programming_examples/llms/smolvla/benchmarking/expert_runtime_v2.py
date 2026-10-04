# SPDX-License-Identifier: MIT
"""SmolVLA's action expert on the NPU, no host K/V packing (layer_jobs_v2 + prefix engine).

Two ELFs share one buffer. Once per action chunk the PREFIX engine turns the backbone's K/V rows
into K|V tiles (cross layers: the expert's k/v projections, self layers: K rotated by -p0 and V
copied, as GEMMs); every denoising call is one launch of the STEP engine (all 16 layers), whose
own-key score / PV jobs read those tiles as bf16 through its third argument, the prefix engine's
arena buffer. The host only lays the backbone's K/V rows into that buffer (no packing, no
arithmetic), builds the masks and the per-chunk rotation matrix R(p0), and writes x per call.
"""
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn  # noqa: F401  (sys.path setup)
import expert_capture as ec
import expert_engine_probe as xp
from expert_runtime import _bf, _merge
from gemm_engine import (
    arena_layout,
    build_gemm_engine,
    compile_mm_engine,
    qkv_col_perm,
    rope_pair_perm,
    rope_table,
    weights_layout,
)

CACHE_DIR = Path(__file__).resolve().parent / "build"
M, M_REAL, L2N, TN, TK1, HERD, TILE_M = (
    xp.M,
    xp.M_REAL,
    xp.L2N,
    xp.TN,
    xp.TK1,
    xp.HERD,
    xp.TILE_M,
)
HPT, HD, NH, NKV, KV, E, E_REAL, PASSES = (
    xp.HPT,
    xp.HD,
    xp.NH,
    xp.NKV,
    xp.KV,
    xp.E,
    xp.E_REAL,
    xp.PRE_PASSES,
)


def engine_mask_v2(m, n_pre, self_attn):
    """Bool mask [M, passes * 320] for one layer's score tiles. m: lerobot's [M_REAL, n_pre (+ M_REAL own)].
    Pass p's tile column c*80 + j*16 + kk is key 64p + 16c + kk for every head j; the own pass (self layers,
    last) holds the layer's own keys 16c + kk."""
    keys = np.zeros((M, PASSES * 64), bool)
    keys[:M_REAL, :n_pre] = m[:, :n_pre]
    keys[M_REAL:] = keys[M_REAL - 1]  # padded rows: any non-empty row, so P.1 > 0
    cols = np.arange(L2N)
    c_, kk_ = cols // (HPT * 16), cols % 16
    tiles = [keys[:, 64 * p + 16 * c_ + kk_] for p in range(PASSES)]
    if self_attn:
        own = np.zeros((M, 64), bool)
        own[:M_REAL, :M_REAL] = m[:, n_pre : n_pre + M_REAL]
        own[M_REAL:] = own[M_REAL - 1]
        tiles.append(own[:, 16 * c_ + kk_])
    return np.concatenate(tiles, axis=1)


class ExpertRuntimeV2:
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
        jobs_pre = xp.prefix_jobs_v2(self.self_attn)
        self.lay_pre = arena_layout(M, jobs_pre, TILE_M, HERD, L2N, pad_tiles=2)
        self.wbase_pre, wrows_pre = weights_layout(jobs_pre, TN, L2N)
        jobs = [j for l in range(n) for j in xp.layer_jobs_v2(l, self.self_attn[l])]
        self.lay = arena_layout(M, jobs, TILE_M, HERD, L2N, external=self.lay_pre.base)
        self.wbase, wrows = weights_layout(jobs, TN, L2N)

        compile_mm_engine(TILE_M, TN, TK1, xp.SFX, xp.OBJ, rms_k=E_REAL)
        self.cache = KernelCache(
            str(CACHE_DIR / f"expert_v2_L{n}"),
            verbose=False,
            profiler=Profiler(enabled=profile),
        )
        self.backend = dict(xp.BACKEND)
        elf = {nm: self.cache.cache_dir / f"{nm}.elf" for nm in ("pre", "step")}
        if all(e.exists() for e in elf.values()):
            for nm in elf:
                self.cache.artifacts[nm] = XRTCompileArtifact(
                    str(elf[nm]), "main:gemm_engine", None
                )
        else:
            try:
                self.cache.compile_and_cache(
                    "pre",
                    build_gemm_engine(
                        M,
                        jobs_pre,
                        TILE_M,
                        TN,
                        TK1,
                        L2N,
                        HERD,
                        HERD,
                        xp.SFX,
                        xp.OBJ,
                        arg_order=["wts", "act"],
                        arena="act",
                        weights="wts",
                        shim_at_launch=True,
                    ),
                    self.backend,
                )
                self.cache.compile_and_cache(
                    "step",
                    build_gemm_engine(
                        M,
                        jobs,
                        TILE_M,
                        TN,
                        TK1,
                        L2N,
                        HERD,
                        HERD,
                        xp.SFX,
                        xp.OBJ,
                        arg_order=["wts", "act", "kv"],
                        arena="act",
                        weights="wts",
                        shim_at_launch=True,
                        kv_arena="kv",
                        kv_lay=self.lay_pre,
                    ),
                    self.backend,
                )
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"the expert engine ELFs are not in {self.cache.cache_dir} and building them failed "
                    f"({str(e).splitlines()[-1][:200]}). Build them once with `make compile-expert`, with a compiler "
                    "that honours air.order_drains first on PATH (and its python on PYTHONPATH) and PEANO_INSTALL_DIR "
                    'at a no-unroll Peano; see the README, "Experimental: backbone and expert on the NPU".'
                ) from e

        nbytes = self._pack_b(_bf(np.zeros((L2N, TN))), TN, TK1).shape[-1]
        self.wts = np.zeros((wrows, L2N // TK1, nbytes), np.uint8)
        self.pwts = np.zeros((wrows_pre, L2N // TK1, nbytes), np.uint8)
        pad = xp.pad
        qperm, qkvperm, kperm = (
            rope_pair_perm(NH, HD),
            qkv_col_perm(NH, NKV, HD),
            rope_pair_perm(NKV, HD),
        )
        for l, w in enumerate(self.layers):
            an, fn = pad(w["anorm"], (E,)), pad(w["fnorm"], (E,))
            wq = pad(w["wq"], (E, NH * HD)) / np.sqrt(HD)
            if self.self_attn[l]:
                b = np.concatenate(
                    [wq, pad(w["wk"], (E, KV)), pad(w["wv"], (E, KV))], axis=1
                )
                self._put_step(f"wqkv{l}", (an[:, None] * b)[:, qkvperm])
            else:
                self._put_step(f"wq{l}", (an[:, None] * wq)[:, qperm])
                self._put_pre(f"wk{l}", w["wk"][:, kperm])
                self._put_pre(f"wv{l}", w["wv"])
            self._put_step(f"wo{l}", pad(w["wo"], (NH * HD, E)))
            self._put_step(
                f"wgu{l}",
                xp.permute_gate_up(
                    fn[:, None] * pad(w["wg"], (E, xp.H)),
                    fn[:, None] * pad(w["wu"], (E, xp.H)),
                    TN,
                    L2N,
                ),
            )
            self._put_step(f"wdn{l}", pad(w["wd"], (xp.H, E)))
        self._put_pre("wid", np.eye(L2N))
        self._kperm = kperm
        self.act_pre = self.lay_pre.empty()
        self.act = self.lay.empty()
        lut = _bf(xp.rope_lut()).astype(np.float32)
        self.lay.pack(self.act, "rope", _bf(rope_table(lut, NH, NKV, HD, KV)))
        tile_bytes = self.act[0].nbytes
        span = lambda nm: (
            self.lay.base[nm] * tile_bytes,  # noqa: E731
            (self.lay.width[nm] // L2N) * (M // (TILE_M * HERD)) * tile_bytes,
        )
        self._x_in, self._x_out = span("x0"), span(f"x{n}")
        self._mask_spans = _merge(
            sorted(span(nm) for nm in ("mask_self", "mask_cross"))
        )
        pre_tile = self.act_pre[0].nbytes
        self._pre_in = (
            0,
            self.lay_pre.drain_lo * pre_tile,
        )  # every tensor no job writes: kc / vc rows
        self._row_bytes_pre = self.pwts[0].nbytes
        self._wr = (
            self.wbase_pre["wr"] * self._row_bytes_pre,
            4 * self._row_bytes_pre,
        )  # 320 x 320 = 4 rows
        self._prefix = None
        self._kv_src = None
        self._written = False
        self._shared = False
        self.timings = {}

    def _put(self, wts, wbase, key, b):
        packed = self._pack_b(_bf(b), TN, TK1)
        rows = packed.reshape(-1, L2N // TK1, packed.shape[-1])
        wts[wbase[key] : wbase[key] + len(rows)] = rows

    def _put_step(self, key, b):
        self._put(self.wts, self.wbase, key, b)

    def _put_pre(self, key, b):
        self._put(self.pwts, self.wbase_pre, key, b)

    def _write(self, key, idx, arr, spans):
        """Write byte spans of arr into the key's idx-th resident BO and sync them."""
        import pyxrt as xrt

        bo = self.cache._cached_bos[key][idx]
        src = arr.view(np.uint8).ravel()
        mv = np.frombuffer(bo.map(), np.uint8, count=src.size)
        for off, size in spans:
            mv[off : off + size] = src[off : off + size]
            bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, off)

    def _set_prefix(self, k_cache, v_cache, mask, pos):
        """kc/vc rows -> prefix buffer, R(p0) -> prefix weights, masks -> step arena; run the prefix launch."""
        p0, n_pre = int(pos.min()), k_cache.shape[1]
        assert n_pre <= PASSES * 64, n_pre
        # K_rot = k @ R, R[i, :] = RoPE(-p0) of unit vector i, columns pair-interleaved like Q
        eye = np.eye(L2N, dtype=np.float32).reshape(L2N, NKV, HD)
        R = ec.rope(eye, np.full(L2N, -p0), self.meta["theta"]).reshape(L2N, L2N)
        self._put_pre("wr", R[:, self._kperm])
        for nm, c in (("kc", k_cache), ("vc", v_cache)):
            rows = np.zeros((self.n_layers, PASSES * 64, L2N), np.float32)
            rows[:, :n_pre] = c
            rows = _bf(rows).reshape(self.n_layers, PASSES, HERD, TILE_M, L2N)
            for l in range(self.n_layers):
                for p in range(PASSES):
                    self.act_pre[self.lay_pre.base[f"{nm}{l}_{p}"]] = rows[l, p]
        for nm, sa in (("mask_self", True), ("mask_cross", False)):
            mk = engine_mask_v2(mask, n_pre, sa)
            self.lay.pack(self.act, nm, _bf(np.where(mk, 0.0, -1e30)))
        if self._written:
            self._write("pre", 0, self.pwts, [self._wr])
            self._write("pre", 1, self.act_pre, [self._pre_in])
            self._write("step", 1, self.act, self._mask_spans)
            self.cache.load_and_run(
                "pre",
                self.backend,
                self.pwts,
                self.act_pre,
                output_indices=[],
                static_input_indices={0},
                intermediate_indices={1},
                bo_key="pre",
            )
        else:
            self.cache.load_and_run(
                "pre",
                self.backend,
                self.pwts,
                self.act_pre,
                output_indices=[1],
                bo_key="pre",
                static_input_indices={0},
            )

    def forget_prefix(self):
        """Make the next call repack its prefix, as a new observation's would be (benchmarking)."""
        self._prefix = None
        self._kv_src = None

    def __call__(self, x, kv, mask, pos, kv_src=None):
        """One denoising call: x [S, 720] suffix embeddings; returns the final-normed output [S, 720].
        kv: (k_cache, v_cache) [layers, keys, KV], or a function returning them. kv_src: the object
        they are read from (the backbone's KV cache): while the same one is passed they are taken
        as unchanged (held weakly)."""
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
        self.lay.pack(self.act, "x0", _bf(xp.pad(x, (M, E))))
        t1 = time.perf_counter()
        c = self.cache
        if not self._written:
            # First call: allocate the step buffers, make the kv argument THE prefix buffer, then run.
            c.load_and_run(
                "step",
                self.backend,
                self.wts,
                self.act,
                self.act_pre,
                output_indices=[1],
                bo_key="step",
                static_input_indices={0},
            )
            c._cached_bos["step"][2] = c._cached_bos["pre"][1]
            self._written = True
        bo = c._cached_bos["step"][1]
        mv = np.frombuffer(bo.map(), np.uint8, count=self.act.nbytes)
        (off, size), src = self._x_in, self.act.view(np.uint8).ravel()
        mv[off : off + size] = src[off : off + size]
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, size, off)
        c.load_and_run(
            "step",
            self.backend,
            self.wts,
            self.act,
            self.act_pre,
            output_indices=[],
            static_input_indices={0, 2},
            intermediate_indices={1},
            bo_key="step",
        )
        off, size = self._x_out
        bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, size, off)
        got = mv.view(bfloat16).reshape(self.act.shape)
        t2 = time.perf_counter()
        out = self.lay.unpack(got, f"x{self.n_layers}")
        out = ec.rms(
            out.astype(np.float32)[: len(x), :E_REAL],
            self.meta["norm"],
            self.meta["eps"],
        )
        self.timings["run_ms"] = self.timings.get("run_ms", 0.0) + (t2 - t1) * 1e3
        self.timings["calls"] = self.timings.get("calls", 0) + 1
        return out
