# SPDX-License-Identifier: MIT
"""Expert step engine with layer_jobs_v2 (no host-packed K/V) + the prefix engine: compile, size, device time.

    python expert_v2_probe.py --layers 2 [--compile-only | --run-only]

Random activations, zero weights: it checks that the engines compile (core program fits 16 KB), how
big the control code is, and what a step and a prefix launch cost. Correctness is the real-weights
chain in expert_runtime / the probe."""
import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(Path(__file__).resolve().parents[3])
]

import backbone_npu as bn  # noqa: F401  (sys.path setup)
import expert_engine_probe as xp
from gemm_engine import (
    arena_layout,
    build_gemm_engine,
    compile_mm_engine,
    weights_layout,
)

ap = argparse.ArgumentParser()
ap.add_argument("--layers", type=int, default=2)
ap.add_argument(
    "--first",
    type=int,
    default=0,
    help="first layer of the step engine (split-mode probes)",
)
ap.add_argument(
    "--jobs",
    default="",
    help="a:b, keep only that slice of the step jobs (split-mode probes)",
)
ap.add_argument("--compile-only", action="store_true")
ap.add_argument("--run-only", action="store_true")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()
NL = args.layers
M, L2N, TN, TK1, HERD, TILE_M = xp.M, xp.L2N, xp.TN, xp.TK1, xp.HERD, xp.TILE_M

from air.backend.xrt import XRTCompileArtifact
from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
    pack_b_bfp16ebs8,
)
from reconfig_probe import ctrl_kb
from air_examples.llms.shared.infra.cache import KernelCache, Profiler

compile_mm_engine(TILE_M, TN, TK1, xp.SFX, xp.OBJ, rms_k=xp.E_REAL)
self_flags = [
    l % 2 == 0 for l in range(args.first + NL)
]  # lerobot: self-attention on the even layers (ec.is_self)
jobs_pre = xp.prefix_jobs_v2(self_flags)
lay_pre = arena_layout(M, jobs_pre, TILE_M, HERD, L2N, pad_tiles=2)
wbase_pre, wrows_pre = weights_layout(jobs_pre, TN, L2N)
jobs = [
    j
    for l in range(args.first, args.first + NL)
    for j in xp.layer_jobs_v2(l, self_flags[l])
]
if args.jobs:
    _a, _b = (int(v) for v in args.jobs.split(":"))
    jobs = jobs[_a:_b]
lay = arena_layout(M, jobs, TILE_M, HERD, L2N, external=lay_pre.base)
wbase, wrows = weights_layout(jobs, TN, L2N)
print(
    f"L={NL}: prefix {len(jobs_pre)} jobs ({lay_pre.n_tiles} tiles), step {len(jobs)} jobs ({lay.n_tiles} tiles, "
    f"{wrows} weight rows)"
)

cache = KernelCache(
    str(
        Path(__file__).resolve().parent
        / "build"
        / (
            f"expert_v2_L{NL}"
            + (f"_from{args.first}" if args.first else "")
            + (f"_j{args.jobs.replace(':', '_')}" if args.jobs else "")
        )
    ),
    verbose=False,
    profiler=Profiler(enabled=True),
)
backend = dict(xp.BACKEND, verbose=args.verbose)
elf = {nm: cache.cache_dir / f"{nm}.elf" for nm in ("pre", "step")}
t0 = time.time()
if args.run_only and all(e.exists() for e in elf.values()):
    for nm in elf:
        cache.artifacts[nm] = XRTCompileArtifact(str(elf[nm]), "main:gemm_engine", None)
else:
    for nm, mod in (
        (
            "pre",
            lambda: build_gemm_engine(
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
        ),
        (
            "step",
            lambda: build_gemm_engine(
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
                kv_lay=lay_pre,
            ),
        ),
    ):
        t = time.time()
        try:
            cache.compile_and_cache(nm, mod(), backend)
        except Exception as e:  # noqa: BLE001
            print(
                f"{nm}: compile failed: {str(e)[-3000:] if args.verbose else str(e).splitlines()[-1]}"
            )
            raise SystemExit(1)
        print(f"{nm}: compiled in {time.time() - t:.0f} s")
print(f"control code {ctrl_kb(cache.cache_dir):.1f} KB (both ELFs)")
if args.compile_only:
    raise SystemExit(0)

bf = lambda a: np.asarray(a, np.float32).astype(bfloat16)  # noqa: E731
rng = np.random.default_rng(0)
nbytes = pack_b_bfp16ebs8(bf(np.zeros((L2N, TN))), TN, TK1).shape[-1]
wts_pre = np.zeros((wrows_pre, L2N // TK1, nbytes), np.uint8)
wts = np.zeros((wrows, L2N // TK1, nbytes), np.uint8)
act_pre = lay_pre.empty()
for nm in lay_pre.base:
    if lay_pre.base[nm] < lay_pre.drain_lo:
        lay_pre.pack(act_pre, nm, bf(0.3 * rng.standard_normal((M, lay_pre.width[nm]))))
act = lay.empty()
for nm in lay.base:
    if lay.base[nm] < lay.drain_lo:
        if nm.startswith("mask"):
            lay.pack(
                act, nm, bf(np.where(rng.random((M, lay.width[nm])) < 0.75, 0.0, -1e30))
            )
        else:
            lay.pack(act, nm, bf(0.3 * rng.standard_normal((M, lay.width[nm]))))
cache.load_and_run("pre", backend, wts_pre, act_pre, output_indices=[1], bo_key="pre")
cache.load_and_run(
    "step",
    backend,
    wts,
    act,
    act_pre,
    output_indices=[1],
    bo_key="step",
    static_input_indices={0},
)
cache._cached_bos["step"][2] = cache._cached_bos["pre"][1]
got = np.asarray(
    cache.load_and_run(
        "step",
        backend,
        wts,
        act,
        act_pre,
        output_indices=[1],
        bo_key="step",
        static_input_indices={0, 2},
    )[1]
).reshape(act.shape)
if f"x{args.first + NL}" in lay.base:
    xo = lay.unpack(got, f"x{args.first + NL}").astype(np.float32)
    print(
        f"output x{args.first + NL}: finite {bool(np.isfinite(xo).all())}, mean |x| {np.abs(xo).mean():.4g}"
    )
for nm, fn in (
    (
        "prefix",
        lambda: cache.load_and_run(
            "pre",
            backend,
            wts_pre,
            act_pre,
            output_indices=[],
            bo_key="pre",
            static_input_indices={0},
            intermediate_indices={1},
        ),
    ),
    (
        "step",
        lambda: cache.load_and_run(
            "step",
            backend,
            wts,
            act,
            act_pre,
            output_indices=[1],
            bo_key="step",
            static_input_indices={0, 2},
        ),
    ),
):
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        fn()
    dev = sorted(
        e["kernel_ms"]
        for e in cache.profiler.kernel_breakdowns["pre" if nm == "prefix" else "step"]
    )
    print(
        f"{nm} launch: device median {dev[len(dev) // 2] * 1e3:.0f} us"
        + (
            f" = {dev[len(dev) // 2] * 1e3 / NL:.0f} us per layer"
            if nm == "step"
            else ""
        )
    )
