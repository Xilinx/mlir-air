# SPDX-License-Identifier: MIT
"""Several bfp16-weight GEMMs under ONE device configuration.

Each air.launch costs a reset plus a full reconfiguration (core programs,
DMA/lock/switch setup): ~63 us + 1.53 us per KB of control code
(reconfig_probe.py). The GEMM engine runs a list of GEMM jobs, one after the
other, inside one launch and one segment: the herd is the same cores with the
same program, and every job reuses the same accumulator, drain and output tile
buffers, so only loop trip counts and DRAM addresses change between jobs.

All jobs share tile_m / tile_n / tile_k_l1 / tile_k_l2 (one microkernel
object, identical memtile DMA patterns); each job has its own K and N.
Args: A_j, B_j (packed), C_j per job, in order.

--mode stitched is the baseline (one launch per job in one ELF); --mode loads
is the first per-job ops.load version, kept because it shows why the channel
version is needed.
"""

import argparse
import os
from contextlib import ExitStack, nullcontext
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

# programming_examples/ is published as the air_examples package rather than
# put on sys.path: every directory under it would otherwise become a
# top-level module name and shadow an installed package that shares it.
import types

sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(_HERE.parent.parent.parent)
]

from air import api as air
from air.api import ops
from air.api.types import bf16, f32, i32, i8

# Own-key attention (Job.own) geometry; must match mm_engine.cc's OWN_*: head dim,
# q heads per kv group, kv groups, and the V columns / rows per herd row a "pv" own
# chunk carries (two kv groups plus padding, rows past tile_m are padding).
OWN_HD, OWN_QPG, OWN_NKV = 64, 3, 5
OWN_VW, OWN_VR = 144, 25


def _order_drain():
    """Mark the air.channel.get just emitted as an ordered drain (air.order_drains).

    The engine chains jobs through one activation arena in host memory: a later job's
    input reads rows an earlier job's drain wrote. air-to-std then has to keep the
    drains in program order instead of deferring every wait to the launch end; that is
    opt-in per launch, by marking its drains."""
    if os.environ.get("ENGINE_NO_ORDER_DRAINS"):  # repro: leave the drains unmarked
        return
    from air.ir import InsertionPoint, UnitAttr

    ops = InsertionPoint.current.block.operations
    op = ops[len(ops) - 1]
    assert op.operation.name == "air.channel.get", op.operation.name
    op.operation.attributes["air.order_drains"] = UnitAttr.get()


def own_pv_col0(head_t, c, half, tile_n, l2_n):
    """First V column a "pv" own step's core column c, output half `half`, reads:
    its tile_n / 2 output dims span at most two heads, so two kv groups."""
    j = (c * tile_n + half * tile_n // 2) // OWN_HD
    g0 = (head_t * (l2_n // OWN_HD) + j) // OWN_QPG
    return min(g0 * OWN_HD, l2_n - OWN_VW)


@dataclass
class Job:
    """One engine GEMM job, C = epilogue(A @ B), over named tensors.

    A [m, k] bf16, B packed bfp16 [n / tile_n, k / tile_k_l1, bytes], C [m, n_out].
    residual: C += R ([m, n_out]) in f32 at drain; R's tiles ride the A channel as
      one extra tile_k_l2 step per output tile (needs tile_k_l2 == l2_n).
    rms: rows of A are RMS-normalised: the norm weight is folded into B on the
      host, the per-row sum of squares is accumulated from the A chunks and the
      drain scales rows by rsqrt(ss / k + eps) (kernel RMS_K must equal k).
    swiglu: B's tile_n blocks hold tile_n/2 gate then tile_n/2 up columns; two
      consecutive output tiles fill the two halves of one tile_n-wide store, so
      n_out = n / 2 (B columns permuted on the host, see permute_gate_up).
    rope: RoPE table P ([m, n] bf16, (cos, sin) per adjacent column pair, (1, 0)
      to pass a pair through), riding the A channel like a residual. The rotation
      acts on adjacent pairs, so head dims must be pair-interleaved in B (see
      interleave_rope_heads). Needs rms (the scale is applied first).
    exp: drain exp(acc + R) (attention probabilities, R the additive mask).
    div: paired like swiglu, drain value / sum (softmax normalisation of P.V).
    a_off, c_off: column offsets of A's and C's window in their tensors (arena
      only, multiples of the arena tile width).
    own: attention against this step's own keys, whose K and V are the arena tiles
      of `kv` at columns kv_off and kv_off + l2_n (all kv groups, K rope-interleaved
      like Q), read as bf16 through the B channel and converted on the cores.
      "s": the job's last output tile is the scores of Q tile head_t (A) against the
      own keys, core column c holding (head, key) for herd row c's tile_m keys;
      the other tiles use packed B as usual.
      "pv": the job's last tile_k_l2 step is P_own (s's last tile) times own V.
      Consecutive own jobs must have consecutive head_t (the cores loop over them).
    """

    a: str
    b: str
    c: str
    k: int
    n: int
    residual: str = None
    rms: bool = False
    swiglu: bool = False
    rope: str = None
    exp: bool = False
    div: bool = False
    a_off: int = 0
    c_off: int = 0
    own: str = None
    kv: str = None
    kv_off: int = 0
    head_t: int = 0
    own_passes: int = 1
    kv_srcs: tuple = ()

    @property
    def kv_names(self):
        """The tensors holding the own keys' K|V tiles: one, or one per pass of 64 keys. kv_srcs
        ((name, column offset), ...) names them one by one, each in the step arena or in the
        external kv arena (build_gemm_engine's kv_lay); then own_passes == len(kv_srcs).
        """
        if self.kv_srcs:
            return [n for n, _ in self.kv_srcs]
        if not self.kv:
            return []
        return (
            [self.kv]
            if self.own_passes == 1
            else [f"{self.kv}{p}" for p in range(self.own_passes)]
        )

    def kv_off_at(self, p):
        return self.kv_srcs[p][1] if self.kv_srcs else self.kv_off

    @property
    def n_out(self):
        return self.n // 2 if self.swiglu or self.div else self.n

    @property
    def paired(self):
        return self.swiglu or self.div

    def packed(self, l2_n):
        """(k, n) of packed B: own steps and tiles have none (tile_k_l2 == l2_n)."""
        return (
            self.k - self.own_passes * l2_n if self.own == "pv" else self.k,
            self.n - self.own_passes * l2_n if self.own == "s" else self.n,
        )


@dataclass
class ArenaLayout:
    """Tile-major activation arena: arena[t, i] is one [tile_m, l2_n] tile of
    herd row i. Tensor X [m, w] has its (li, lj) tile (rows li*l2_m + i*tile_m..,
    cols lj*l2_n..) at t = base[X] + li * (w // l2_n) + lj. Tensors no job writes
    come first, then every job's C in job order, so the produced tiles are the
    contiguous run [drain_lo, n_tiles) in production order. The herd rows of one
    tile are adjacent: a transfer across herd rows strides one tile, where a
    stride of a whole herd row's run (megabytes) hangs the shim DMA."""

    base: dict
    width: dict
    drain_lo: int
    n_tiles: int
    m: int
    tile_m: int
    herd_m: int
    l2_n: int

    def pack(self, arena, name, x):
        l2_m, cols = self.tile_m * self.herd_m, self.width[name] // self.l2_n
        x = np.asarray(x).reshape(
            self.m // l2_m, self.herd_m, self.tile_m, cols, self.l2_n
        )
        n = x.shape[0] * cols
        arena[self.base[name] : self.base[name] + n] = x.transpose(
            0, 3, 1, 2, 4
        ).reshape(n, self.herd_m, self.tile_m, self.l2_n)

    def unpack(self, arena, name):
        l2_m, cols = self.tile_m * self.herd_m, self.width[name] // self.l2_n
        n = self.m // l2_m * cols
        x = np.asarray(arena)[self.base[name] : self.base[name] + n]
        x = x.reshape(self.m // l2_m, cols, self.herd_m, self.tile_m, self.l2_n)
        return x.transpose(0, 2, 3, 1, 4).reshape(self.m, self.width[name])

    def empty(self):
        return np.zeros((self.n_tiles, self.herd_m, self.tile_m, self.l2_n), bfloat16)


def arena_layout(m, jobs, tile_m, herd_m, l2_n, pad_tiles=0, external=()):
    """Several jobs may write column windows (c_off) of one C, in column order.
    pad_tiles: unused tiles appended to every herd row (probing arena size effects)."""
    width, produced = {}, list(dict.fromkeys(j.c for j in jobs))
    for j in jobs:
        for name, lo, w in (
            (j.a, j.a_off, j.k),
            (j.residual, 0, j.n_out),
            (j.rope, 0, j.n),
            *[
                (kv, 0, j.kv_off_at(q) + 2 * l2_n)
                for q, kv in enumerate(j.kv_names)
                if kv not in external
            ],
            (j.c, j.c_off, j.n_out),
        ):
            if name:
                assert lo % l2_n == 0 and w % l2_n == 0, name
                width[name] = max(width.get(name, 0), lo + w)
    base, t, rows = {}, 0, m // (tile_m * herd_m)
    for name in [n for n in width if n not in produced] + produced:
        base[name] = t
        t += rows * (width[name] // l2_n)
    drain_lo = base[produced[0]]
    seq = [
        base[j.c] + li * (width[j.c] // l2_n) + j.c_off // l2_n + lj
        for j in jobs
        for li in range(rows)
        for lj in range(j.n_out // l2_n)
    ]
    assert seq == list(
        range(drain_lo, t)
    ), "outputs must fill the arena tail in production order"
    return ArenaLayout(base, width, drain_lo, t + pad_tiles, m, tile_m, herd_m, l2_n)


def weights_layout(jobs, tile_n, tile_k_l2):
    """Row offsets of every job's B in one weights tensor [rows, tile_k_l2 / tile_k_l1,
    tile_bytes]: a row is one L2 B step, so B (packed [n / tile_n, k / tile_k_l1, bytes])
    is rows base .. base + n / tile_n * k / tile_k_l2 in its own order."""
    base, rows = {}, 0
    for j in jobs:
        if j.b not in base:
            kp, np_ = j.packed(tile_k_l2)
            base[j.b] = rows
            rows += np_ // tile_n * (kp // tile_k_l2)
    return base, rows


def build_gemm_engine(
    m,
    jobs,
    tile_m,
    tile_n,
    tile_k_l1,
    tile_k_l2,
    herd_m,
    herd_n,
    sym_suffix,
    link_with,
    arg_order=None,
    stack_c=None,
    arena=None,
    weights=None,
    shim_at_launch=False,
    arena_pad=0,
    kv_arena=None,
    kv_lay=None,
):
    """jobs: [Job]. Args are the jobs' named tensors, in arg_order (default:
    first appearance, A, B, residual, C per job).

    Every DMA is an explicit channel shared by all jobs. With per-job ops.load /
    ops.store each job gets its own channels, and two jobs overflow the memtile's
    48 BDs; air-fuse-channels' aggressive mode, which should time-multiplex them,
    fails to verify ("operand does not dominate this use"). Row r's A tile,
    column c's B tile and row r's output tile each have their own L2 buffer, so
    a memtile carries one A, one B and one C stream, the same as one launch of
    build_gemm_bfp16. The memtile side is one flat loop over every job's K
    steps; only the core loop trip counts and the shim addresses differ by job.
    A core DMA channel cycles one BD chain, so every transfer into or out of a
    core has the same size in every job: residual tiles come in as A chunks and
    SwiGLU halves are paired into full-width output tiles.

    Every output drain is armed at launch start and a shim S2MM channel queues 4
    tasks, so more than 4 jobs with separate C tensors hang. stack_c names one
    [len(jobs) * m, n_out] tensor that takes every job's C (job i in rows i*m..),
    drained by one task per channel.

    arena names one tensor [n_tiles, herd_m, tile_m, l2_n] replacing every A,
    residual, RoPE and C tensor (see ArenaLayout): every activation transfer is
    whole tiles and all jobs' outputs, of any width, drain as one task per
    channel. B tensors stay separate.
    """
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        bfp_tile_bytes,
    )

    r, s, t = 8, 8, 8
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n
    tk2 = tile_k_l2
    k_per_l2 = tk2 // tile_k_l1
    assert m % l2_m == 0 and tk2 % tile_k_l1 == 0

    shapes = {}

    def declare(name, shape, dtype):
        assert shapes.setdefault(name, (shape, dtype)) == (shape, dtype), (
            name,
            shapes[name],
            shape,
        )

    for j in jobs:
        assert j.n % l2_n == 0 and j.k % tk2 == 0
        assert (
            not (j.residual or j.rope) or tk2 == l2_n
        ), "a residual tile is one tile_k_l2 step"
        assert not (j.residual and j.rope) and (not j.rope or (j.rms and not j.swiglu))
        assert not j.paired or (j.n // l2_n) % 2 == 0
        assert (
            not (j.swiglu and j.div)
            and not (j.div and j.rms)
            and not (j.exp and (j.paired or j.rope or j.rms))
        )
        assert not j.exp or j.residual, "exp's mask rides the residual path"
        assert arena or not (j.a_off or j.c_off)
        assert not j.own or (
            arena and weights and j.kv
        ), "own K/V ride the weights' B channel from the arena"
        assert j.own != "s" or (
            j.k == tk2 == l2_n and j.n >= j.own_passes * l2_n and j.residual and j.exp
        )
        assert j.own_passes == 1 or j.own, "own_passes: only own jobs"
        assert (
            j.own != "pv" or j.k // tk2 >= 3
        ), "own pv with fewer than 3 K steps in total gives wrong results (not understood)"
        assert j.own != "pv" or (
            j.div and j.k >= j.own_passes * tk2 and j.n == 2 * l2_n
        ), "one output tile pair"
        if not weights:
            declare(j.b, [j.n // tile_n, j.k // tile_k_l1, tile_bytes], i8)
        if arena:
            continue
        declare(j.a, [m, j.k], bf16)
        if j.residual:
            declare(j.residual, [m, j.n_out], bf16)
        if j.rope:
            declare(j.rope, [m, j.n], bf16)
        if not stack_c:
            declare(j.c, [m, j.n_out], bf16)
    if arena:
        assert not stack_c and tk2 == l2_n, "arena tiles are one tile_k_l2 step wide"
        lay = arena_layout(
            m,
            jobs,
            tile_m,
            herd_m,
            l2_n,
            arena_pad,
            external=kv_lay.base if kv_lay else (),
        )
        declare(arena, [lay.n_tiles, herd_m, tile_m, l2_n], bf16)
        if (
            kv_lay
        ):  # K|V tiles another launch (the prefix engine) produced, in its own arena layout
            assert kv_arena and kv_arena != arena
            declare(kv_arena, [kv_lay.n_tiles, herd_m, tile_m, l2_n], bf16)
    if weights:
        wbase, wrows = weights_layout(jobs, tile_n, tk2)
        declare(weights, [wrows, k_per_l2, tile_bytes], i8)
    if stack_c:
        assert len({j.n_out for j in jobs}) == 1 and not any(j.swiglu for j in jobs)
        declare(stack_c, [len(jobs) * m, jobs[0].n_out], bf16)
    arg_order = arg_order or list(shapes)
    assert sorted(arg_order) == sorted(shapes), (arg_order, list(shapes))
    T = {name: air.tensor(*shapes[name]) for name in arg_order}

    n_tiles = [(m // l2_m) * (j.n // l2_n) for j in jobs]
    a_steps = sum(
        nt * (j.k // tk2 + bool(j.residual or j.rope)) for nt, j in zip(n_tiles, jobs)
    )
    b_steps = sum(nt * (j.k // tk2) for nt, j in zip(n_tiles, jobs))
    c_tiles = sum(nt // (2 if j.paired else 1) for nt, j in zip(n_tiles, jobs))

    def ext(name, **kw):
        return air.extern(f"{name}{sym_suffix}", link_with=link_with, **kw)

    zero_acc = ext("zero_vectorized_f32_mn")
    matmul = ext("matmul_bf16_x_bfp16_packed_f32")
    drain_fn = ext("f32_to_bf16_mn")
    if any(j.residual for j in jobs):
        add_res = ext("add_residual_blocked", scalars=[i32])
    if any(j.rope for j in jobs):
        rope_fn = ext("rms_rope_blocked", scalars=[i32])
    if any(j.rms for j in jobs):
        zero_rows = ext("zero_rows")
        sumsq = ext("sumsq_rows_blocked")
        rows_rstd = ext("rows_rstd")
    if any(j.swiglu for j in jobs):
        swiglu_fn = ext(
            (
                "f32_to_bf16_rms_swiglu"
                if all(j.rms for j in jobs if j.swiglu)
                else "f32_to_bf16_swiglu_mn"
            ),
            scalars=[i32],
        )
        assert all(
            j.rms == jobs[[x.swiglu for x in jobs].index(True)].rms
            for j in jobs
            if j.swiglu
        )
    if any(j.exp for j in jobs):
        exp_fn = ext("f32_to_bf16_exp_mn")
    if any(j.div for j in jobs):
        div_fn = ext("f32_to_bf16_div_mn", scalars=[i32])
    if any(j.own for j in jobs):
        # One entry point, dispatching at run time, so own jobs share the tile code.
        own_fn = ext("matmul_engine_own", scalars=[i32] * 6)
    assert all(
        not j.rms or j.swiglu or j.rope for j in jobs
    ), "rms is only in the SwiGLU and RoPE drains"

    a_in = air.channel("EngAIn", size=[herd_m])
    b_in = air.channel("EngBIn", size=[herd_n])
    a2l1 = air.channel("EngA2L1", size=[herd_m, 1], broadcast_shape=[herd_m, herd_n])
    b2l1 = air.channel("EngB2L1", size=[1, herd_n], broadcast_shape=[herd_m, herd_n])
    c2l2 = air.channel("EngC2L2", size=[herd_m, herd_n])
    c_out = air.channel("EngCOut", size=[herd_m])

    with air.launch(name="gemm_engine") as launch:

        def act(name, li, i, col):
            """Herd row i's tile (li, col) of activation `name`."""
            if arena:
                at = lay.base[name] + li * (lay.width[name] // l2_n) + col
                return T[arena][at : at + 1, i, :, :]
            row = li * l2_m + i * tile_m
            return T[name][row : row + tile_m, col * l2_n : col * l2_n + l2_n]

        chunk_el = tile_bytes // 2  # bf16 elements per B chunk
        if any(j.own for j in jobs):
            assert tile_bytes % 4 == 0 and m == l2_m and herd_m == 2 * k_per_l2
            assert 2 * OWN_VR * OWN_VW == chunk_el and OWN_VR >= tile_m
            tile_el, arena_el = tile_m * l2_n, lay.n_tiles * herd_m * tile_m * l2_n
            arena_flat = T[arena].reshape(arena_el)
            if kv_lay:
                kv_el = kv_lay.n_tiles * herd_m * tile_m * l2_n
                kv_flat = T[kv_arena].reshape(kv_el)

        def kv_src(j, q):
            """Where pass q's K|V tiles of own job j are: (flat view, tensor, tile of its K, flat size)."""
            nm = j.kv_names[q]
            if kv_lay and nm in kv_lay.base:
                return (
                    kv_flat,
                    T[kv_arena],
                    kv_lay.base[nm] + j.kv_off_at(q) // l2_n,
                    kv_el,
                )
            return arena_flat, T[arena], lay.base[nm] + j.kv_off_at(q) // l2_n, arena_el

        def own_b(j, c, lj, kc=None, pas=0):
            """Core column c's own-step B, one put (one task, like a weights step: the
            shim zips the channels' task lists): "s", chunk kc; "pv", both chunks."""
            if j.own == "s":
                flat, _, kv_t, size = kv_src(j, lj - (j.n // l2_n - j.own_passes))
                # Herd row c's keys: its K tile row-major, padded with what follows.
                at = (kv_t * herd_m + c) * tile_el
                assert at + chunk_el <= size
                return flat[at : at + chunk_el]
            # Chunk kc = herd rows 2kc, 2kc+1 (P_own chunk kc's key groups): V rows of each,
            # padded with the rows that follow (the next herd row's), in the kv groups core
            # column c reads. The rows overlap, which no subscript spells: a raw pattern.
            from air.api._index import coerce_index
            from air.api._value import TensorSlice

            _, src, kv_t, size = kv_src(j, pas)
            col0, at = (
                own_pv_col0(j.head_t, c, lj % 2, tile_n, l2_n),
                (kv_t + 1) * herd_m * tile_el,
            )
            assert at + (herd_m - 1) * tile_el + OWN_VR * l2_n <= size
            return TensorSlice(
                src,
                [coerce_index(0), coerce_index(0), coerce_index(at + col0)],
                [herd_m, OWN_VR, OWN_VW],
                [tile_el, l2_n, 1],
                is_view=True,
            )

        drain_before = {}
        if arena:
            # Jobs read their inputs back from the host memory the drains write, so each
            # drain group is issued before its jobs' inputs, and air-to-std awaits it
            # before the next group's inputs. A group ends before a job reading its outputs.
            start, group, produced_in = lay.drain_lo, 0, set()
            rows = m // l2_m
            for idx, j in enumerate(jobs):
                # repro toggle: ENGINE_SINGLE_DRAIN=1 keeps one drain group for the whole launch
                if not idx or (
                    not os.environ.get("ENGINE_SINGLE_DRAIN")
                    and {j.a, j.residual, *j.kv_names} & produced_in
                ):
                    group, produced_in = idx, set()
                    drain_before[group] = [start, start]
                produced_in.add(j.c)
                start += rows * (j.n_out // l2_n)
                drain_before[group][1] = start

        def shim_side():
            for idx, j in enumerate(jobs):
                if idx in drain_before:
                    lo, hi = drain_before[idx]
                    for i in range(herd_m):
                        c_out.get(T[arena][lo:hi, i, :, :], indices=[i])
                        _order_drain()
                B = None if weights else T[j.b]
                # air-isolate-async-dma-loop-nests gives every put its own loop nest,
                # which would send all of a job's A steps before any residual step.
                # Unrolled tile loops keep each tile's residual put right after its K steps.
                tile_loop = range if j.residual or j.rope or j.own else air.sequential
                for li in tile_loop(m // l2_m):
                    for lj in tile_loop(j.n // l2_n):
                        # A "pv" own step is its own A task, pairing with its own B task.
                        nk_ = j.k // tk2
                        a_steps_j = (
                            [(0, nk_ - j.own_passes)]
                            + [
                                (nk_ - j.own_passes + q, nk_ - j.own_passes + q + 1)
                                for q in range(j.own_passes)
                            ]
                            if j.own == "pv"
                            else [(0, nk_)]
                        )
                        for s0, s1 in [h for h in a_steps_j if h[0] < h[1]]:
                            for k2 in air.sequential(s0, s1):
                                for i in range(herd_m):
                                    if arena:
                                        a_in.put(
                                            act(j.a, li, i, j.a_off // l2_n + k2),
                                            indices=[i],
                                        )
                                        continue
                                    row = li * l2_m + i * tile_m
                                    a_in.put(
                                        T[j.a][
                                            row : row + tile_m,
                                            k2 * tk2 : k2 * tk2 + tk2,
                                        ],
                                        indices=[i],
                                    )
                        # The shim command stream zips the channels' task lists, and a BD-reuse
                        # await on a later job's task before this job's last A task deadlocks.
                        # A residual tile is two A tasks (K steps, residual), so B is two too.
                        k_steps = j.k // tk2
                        halves = (
                            [(0, k_steps // 2), (k_steps // 2, k_steps)]
                            if j.residual or j.rope
                            else [(0, k_steps)]
                        )
                        if k_steps == 1 and (j.residual or j.rope):
                            # One K step: its two L1 chunks as the two B tasks (the memtile's
                            # single S2MM receive only counts bytes).
                            assert weights and k_per_l2 == 2
                            for kk in range(k_per_l2):
                                for c in range(herd_n):
                                    if (
                                        j.own == "s"
                                        and lj >= j.n // l2_n - j.own_passes
                                    ):
                                        b_in.put(own_b(j, c, lj, kk), indices=[c])
                                        continue
                                    wr = wbase[j.b] + lj * herd_n + c
                                    b_in.put(
                                        T[weights][wr : wr + 1, kk : kk + 1, :],
                                        indices=[c],
                                    )
                            halves = []
                        kp = j.packed(tk2)[0] // tk2
                        if j.own == "pv":
                            halves = [(0, kp)]
                        for lo, hi in [h for h in halves if h[0] < h[1]]:
                            for k2 in air.sequential(lo, hi):
                                for c in range(herd_n):
                                    kc = k2 * k_per_l2
                                    if weights:
                                        wr = wbase[j.b] + (lj * herd_n + c) * kp + k2
                                        b_in.put(
                                            T[weights][wr : wr + 1, :, :], indices=[c]
                                        )
                                    else:
                                        b_in.put(
                                            B[lj * herd_n + c, kc : kc + k_per_l2, :],
                                            indices=[c],
                                        )
                        if j.own == "pv":
                            for q in range(j.own_passes):
                                for c in range(herd_n):
                                    b_in.put(own_b(j, c, lj, pas=q), indices=[c])
                        if j.residual or j.rope:
                            for i in range(herd_m):
                                a_in.put(
                                    act(j.residual or j.rope, li, i, lj), indices=[i]
                                )
                if stack_c or arena:
                    continue
                # Right after the job's own puts: a later job that reads C then waits on
                # transfers issued before it (emitted at the end, O -> Down's residual hangs).
                C = T[j.c]
                for li in air.sequential(m // l2_m):
                    for lj in air.sequential(j.n_out // l2_n):
                        for i in range(herd_m):
                            row = li * l2_m + i * tile_m
                            c_out.get(
                                C[row : row + tile_m, lj * l2_n : lj * l2_n + l2_n],
                                indices=[i],
                            )
            if stack_c:
                S, n_out = T[stack_c], jobs[0].n_out
                for g in air.sequential(len(jobs) * m // l2_m):
                    for lj in air.sequential(n_out // l2_n):
                        for i in range(herd_m):
                            row = g * l2_m + i * tile_m
                            c_out.get(
                                S[row : row + tile_m, lj * l2_n : lj * l2_n + l2_n],
                                indices=[i],
                            )

        @launch.body
        def _():
            if shim_at_launch:
                # Written in the launch: in the segment, air-dma-to-channel hoists each shim
                # put out on its own, one greedy rewrite of the whole module per put
                # (compile time quadratic in the job count). The launch region opens lazily
                # at its first use, so open it first: the puts' loops must live inside it.
                from air.api._trace import in_launch_body

                in_launch_body(lambda: (shim_side(), segment()))
            else:
                segment()

        def segment():
            with air.segment(name="engine_seg") as seg:

                @seg.body
                def _():
                    if not shim_at_launch:
                        shim_side()

                    def l2(shape, dtype, col):
                        return air.alloc(
                            shape, dtype, scope=seg.private(), column=col, split=False
                        )

                    l2_a = [l2([tile_m, tk2], bf16, i) for i in range(herd_m)]
                    l2_b = [l2([k_per_l2, tile_bytes], i8, c) for c in range(herd_n)]
                    l2_c = [l2([tile_m, l2_n], bf16, i) for i in range(herd_m)]

                    for i in range(herd_m):
                        for _ in air.sequential(a_steps):
                            a_in.get(l2_a[i], indices=[i])
                            for kk in range(k_per_l2):
                                a2l1.put(
                                    l2_a[i][:, kk * tile_k_l1 : (kk + 1) * tile_k_l1]
                                    .reshape(1, 1, tile_m // r, r, tile_k_l1 // s, s)
                                    .transpose(0, 1, 2, 4, 3, 5),
                                    indices=[i, 0],
                                )
                    for c in range(herd_n):
                        for _ in air.sequential(b_steps):
                            b_in.get(l2_b[c], indices=[c])
                            for kk in range(k_per_l2):
                                b2l1.put(l2_b[c][kk, :], indices=[0, c])

                    RJ = os.environ.get(
                        "ENGINE_RJ", "gu,qkv,pair"
                    )  # experiment: which kinds loop

                    def run_job(j, nt, tile, emit):
                        if j.swiglu and j.rms:
                            pairs = j.n // l2_n // 2
                            if "gu" in RJ:
                                for _ in air.sequential(m // l2_m):
                                    for pr in air.sequential(pairs):
                                        for hf in air.sequential(2):
                                            tile(j, hf, stats=[pr == 0, hf == 0])
                                        emit()
                            else:
                                for _ in air.sequential(m // l2_m):
                                    tile(j, 0, stats=True)
                                    tile(j, 1)
                                    emit()
                                    for _ in air.sequential(pairs - 1):
                                        tile(j, 0)
                                        tile(j, 1)
                                        emit()
                        elif j.rms:
                            if "qkv" in RJ:
                                for _ in air.sequential(m // l2_m):
                                    for tn_ in air.sequential(j.n // l2_n):
                                        tile(j, stats=[tn_ == 0])
                                        emit()
                            else:
                                for _ in air.sequential(m // l2_m):
                                    tile(j, stats=True)
                                    emit()
                                    for _ in air.sequential(j.n // l2_n - 1):
                                        tile(j)
                                        emit()
                        elif j.paired:
                            if "pair" in RJ:
                                for _ in air.sequential(nt // 2):
                                    for hf in air.sequential(2):
                                        tile(j, hf)
                                    emit()
                            else:
                                for _ in air.sequential(nt // 2):
                                    tile(j, 0)
                                    tile(j, 1)
                                    emit()
                        else:
                            assert (
                                not j.own or m == l2_m
                            ), "own tile index = output tile index"
                            for tix in air.sequential(nt):
                                tile(j, tix=tix)
                                emit()

                    with air.herd(
                        [range(herd_m), range(herd_n)],
                        name="herd_0",
                        shape=(herd_m, herd_n),
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            acc = air.alloc(
                                [1, 1, tile_n // t, tile_m // r, r, t],
                                f32,
                                scope=h.private(),
                            )
                            drain = air.alloc(
                                [1, 1, tile_n // t, tile_m // r, r, t],
                                bf16,
                                scope=h.private(),
                            )
                            l1_a = air.alloc(
                                [1, 1, tile_m // r, tile_k_l1 // s, r, s],
                                bf16,
                                scope=h.private(),
                            )
                            l1_b = air.alloc([tile_bytes], i8, scope=h.private())
                            ss = (
                                air.alloc([tile_m], f32, scope=h.private())
                                if any(j.rms for j in jobs)
                                else None
                            )

                            head = [
                                0
                            ]  # the running own job's head_t (a core loop index)

                            def when(conds):
                                # conds: True (always), or dynamic conditions that must all hold
                                if conds is True:
                                    return nullcontext()
                                stack = ExitStack()
                                for c in conds:
                                    stack.enter_context(ops.branch(c))
                                return stack

                            def tile(j, half=None, stats=False, tix=None):
                                zero_acc(acc)
                                if stats:
                                    with when(stats):
                                        zero_rows(ss)
                                n_ch = j.k // tile_k_l1
                                for ch in air.sequential(n_ch):
                                    a2l1.get(l1_a, indices=[tx, ty])
                                    b2l1.get(l1_b, indices=[tx, ty])
                                    if stats:
                                        with when(stats):
                                            sumsq(l1_a, ss)
                                    if j.own:
                                        own = (
                                            (
                                                tix == j.n // l2_n - 1
                                                if j.own_passes == 1
                                                else tix >= j.n // l2_n - j.own_passes
                                            )
                                            if j.own == "s"
                                            else (ch >= n_ch - j.own_passes * k_per_l2)
                                        )
                                        own_fn(
                                            l1_a,
                                            l1_b,
                                            acc,
                                            1 if j.own == "s" else 2,
                                            own,
                                            ch,
                                            head[0],
                                            ty,
                                            half or 0,
                                        )
                                    else:
                                        matmul(l1_a, l1_b, acc)
                                if stats:
                                    with when(stats):
                                        rows_rstd(ss)
                                if j.residual or j.rope:
                                    # Core column c's output columns are A chunk c // 2, half c % 2.
                                    hpc = tile_k_l1 // tile_n
                                    for ch in air.sequential(k_per_l2):
                                        a2l1.get(l1_a, indices=[tx, ty])
                                        # one call site: core ty takes half ty % hpc of chunk ty // hpc
                                        with ops.branch(ty // hpc == ch):
                                            if j.rope:
                                                rope_fn(acc, ss, l1_a, ty % hpc)
                                            else:
                                                add_res(acc, l1_a, ty % hpc)
                                if j.swiglu:
                                    if j.rms:
                                        swiglu_fn(acc, ss, drain, half)
                                    else:
                                        swiglu_fn(acc, drain, half)
                                elif j.div:
                                    div_fn(acc, drain, half)
                                elif j.exp:
                                    exp_fn(acc, drain)
                                else:
                                    drain_fn(acc, drain)

                            # Cores see only shapes and epilogues: a job list repeating with period p
                            # (e.g. per layer) runs one period in a loop, as core program memory is 16 KB.
                            sig = [
                                (
                                    j.k,
                                    j.n,
                                    bool(j.residual),
                                    j.rms,
                                    j.swiglu,
                                    bool(j.rope),
                                    j.exp,
                                    j.div,
                                    j.own,
                                )
                                for j in jobs
                            ]
                            # head_t is a core loop index within a run, but must repeat with the period.
                            psig = [
                                sg + (j.head_t if j.own else 0,)
                                for sg, j in zip(sig, jobs)
                            ]
                            period = next(
                                p
                                for p in range(1, len(jobs) + 1)
                                if len(jobs) % p == 0
                                and psig == psig[:p] * (len(jobs) // p)
                            )
                            runs = (
                                []
                            )  # consecutive identical jobs of one period, run as one loop
                            for p in range(period):
                                if runs and sig[runs[-1][0]] == sig[p]:
                                    runs[-1][1] += 1
                                else:
                                    runs.append([p, 1])
                            for q in range(len(jobs)):
                                p = q % period
                                start = next(s for s, n in runs if s <= p < s + n)
                                assert (
                                    not jobs[q].own
                                    or jobs[q].head_t == jobs[start].head_t + p - start
                                ), "own jobs of one run need consecutive head_t"
                            for _ in air.sequential(len(jobs) // period):
                                for p, count in runs:
                                    for rep in air.sequential(count):
                                        head[0] = rep + jobs[p].head_t
                                        run_job(
                                            jobs[p],
                                            n_tiles[p],
                                            tile,
                                            lambda: c2l2.put(
                                                drain.transpose(0, 1, 3, 4, 2, 5),
                                                indices=[tx, ty],
                                            ),
                                        )

                    for i in range(herd_m):
                        for _ in air.sequential(c_tiles):
                            for c in air.parallel(0, herd_n):
                                c2l2.get(
                                    l2_c[i][:, c * tile_n : (c + 1) * tile_n],
                                    indices=[i, c],
                                )
                            c_out.put(l2_c[i], indices=[i])

    return launch.build(target="npu2")


def permute_gate_up(w_gate, w_up, tile_n, l2_n):
    """(K, H) gate/up -> (K, 2H) B for Job(swiglu=True): output tile pair p, half h,
    core column c holds output columns p*l2_n + c*tile_n + h*tile_n/2 + [0, tile_n/2),
    computed from B tile (2p + h) * (l2_n / tile_n) + c = [gate cols | up cols]."""
    k, hdim = w_gate.shape
    half, cols = tile_n // 2, l2_n // tile_n
    o = np.arange(hdim)
    p, c, hh, i = o // l2_n, (o % l2_n) // tile_n, (o % tile_n) // half, o % half
    nt = (2 * p + hh) * cols + c
    w = np.empty((k, 2 * hdim), dtype=w_gate.dtype)
    w[:, nt * tile_n + i] = w_gate
    w[:, nt * tile_n + half + i] = w_up
    return w


def rope_pair_perm(n_heads, head_dim):
    """Column order pair-interleaving each head: new columns 2p, 2p+1 of a head
    are its old p and p + head_dim/2 (the rotate-half partners)."""
    h = head_dim // 2
    local = np.stack([np.arange(h), np.arange(h) + h], axis=1).ravel()
    return (np.arange(n_heads)[:, None] * head_dim + local).ravel()


def qkv_col_perm(n_heads, n_kv_heads, head_dim):
    """B / output column order of the rms+QKV+RoPE job: q and k heads
    pair-interleaved, v unchanged. q.k per head is invariant (same permutation)."""
    q, kv = n_heads * head_dim, n_kv_heads * head_dim
    return np.concatenate(
        [
            rope_pair_perm(n_heads, head_dim),
            q + rope_pair_perm(n_kv_heads, head_dim),
            q + kv + np.arange(kv),
        ]
    )


def rope_table(lut, n_heads, n_kv_heads, head_dim, v_cols):
    """lut [m, head_dim] = [cos | sin] per row -> the Job(rope=) table: (cos, sin)
    per pair-interleaved q and k column pair, (1, 0) over the v columns."""
    h = head_dim // 2
    pair = np.stack([lut[:, :h], lut[:, h:]], axis=2).reshape(len(lut), head_dim)
    v = np.tile(np.array([1.0, 0.0], np.float32), (len(lut), v_cols // 2))
    return np.concatenate([np.tile(pair, (1, n_heads + n_kv_heads)), v], axis=1)


def compile_mm_engine(tile_m, tile_n, tile_k_l1, sym_suffix, out_name, rms_k=960):
    from air_examples.llms.shared.infra.external_kernels import (
        _PROJ_ROOT,
        _compile_kernel,
    )

    extra = [
        f"-I{_PROJ_ROOT / 'matrix_multiplication' / 'bf16_x_bfp16'}",
        f"-DDIM_M={tile_m}",
        f"-DDIM_N={tile_n}",
        f"-DDIM_K={tile_k_l1}",
        f"-DSYM_SUFFIX={sym_suffix}",
        f"-DRMS_K={rms_k}",
        "-Wno-macro-redefined",
        "-O3",  # -O2 (the shared default) makes the own-key attention jobs ~2x slower
    ] + os.environ.get("ENGINE_KERNEL_FLAGS", "").split()
    # Rebuilt only when stale: the kernel cache keys ELFs on the object's mtime, so an
    # unconditional rebuild recompiles every cached engine (tens of minutes for many jobs).
    # An out_name must keep one set of flags.
    srcs = [_HERE / "kernels_bfp16" / n for n in ("mm_engine.cc", "mm_bfp16.cc")]
    obj = Path(out_name)
    stale = (
        bool(os.environ.get("ENGINE_KERNEL_FLAGS"))
        or not obj.exists()
        or max(s.stat().st_mtime for s in srcs) > obj.stat().st_mtime
    )
    _compile_kernel(srcs[0], out_name, extra_flags=extra, force=stale)


def build_gemm_engine_loads(
    m, jobs, tile_m, tile_n, tile_k_l1, herd_m, herd_n, sym_suffix, link_with
):
    """First attempt, ops.load/ops.store per job: one job compiles (same time as
    one launch); two jobs overflow the memtile BDs. jobs: [(k, n, tile_k_l2)]."""
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        bfp_tile_bytes,
    )

    r, s, t = 8, 8, 8
    tile_bytes = bfp_tile_bytes(tile_n, tile_k_l1)
    l2_m, l2_n = tile_m * herd_m, tile_n * herd_n
    assert m % l2_m == 0
    tensors = []
    for k, n, tk2 in jobs:
        assert n % l2_n == 0 and k % tk2 == 0 and tk2 % tile_k_l1 == 0
        tensors.append(
            (
                air.tensor([m, k], bf16),
                air.tensor([n // tile_n, k // tile_k_l1, tile_bytes], i8),
                air.tensor([m, n], bf16),
            )
        )

    zero_acc = air.extern(f"zero_vectorized_f32_mn{sym_suffix}", link_with=link_with)
    matmul = air.extern(
        f"matmul_bf16_x_bfp16_packed_f32{sym_suffix}", link_with=link_with
    )
    drain_fn = air.extern(f"f32_to_bf16_mn{sym_suffix}", link_with=link_with)

    with air.launch(name="gemm_engine") as launch:

        @launch.body
        def _():
            with air.segment(name="engine_seg") as seg:

                @seg.body
                def _():
                    hd = (herd_m, herd_n)
                    acc = air.alloc(
                        [*hd, tile_n // t, tile_m // r, r, t], f32, scope=seg.shared()
                    )
                    drain = air.alloc(
                        [*hd, tile_n // t, tile_m // r, r, t], bf16, scope=seg.shared()
                    )
                    l2_c = air.alloc(
                        [herd_m, herd_n, tile_m, tile_n], bf16, scope=seg.private()
                    )

                    def herd():
                        return air.herd(
                            [range(hd[0]), range(hd[1])], name="herd_0", shape=hd
                        )

                    def l2_inputs(tk2):
                        return (
                            air.alloc([herd_m, tile_m, tk2], bf16, scope=seg.private()),
                            air.alloc(
                                [herd_n, tk2 // tile_k_l1, tile_bytes],
                                i8,
                                scope=seg.private(),
                            ),
                        )

                    # With one tile_k_l2 for every job, the L2 input buffers and hence every
                    # memtile DMA pattern are identical across jobs; only trip counts differ.
                    shared_l2 = (
                        l2_inputs(jobs[0][2])
                        if len({j[2] for j in jobs}) == 1
                        else None
                    )

                    for (k, n, tk2), (A, B, C) in zip(jobs, tensors):
                        k_per_l2 = tk2 // tile_k_l1
                        l2_a, l2_b = shared_l2 or l2_inputs(tk2)
                        for li in air.sequential(m // l2_m):
                            for lj in air.sequential(n // l2_n):
                                row, col = li * l2_m, lj * l2_n
                                n_outer = lj * herd_n

                                with herd() as zh:

                                    @zh.body
                                    def _(tx, ty):
                                        zero_acc(acc)

                                for k2 in air.sequential(k // tk2):
                                    k_l2_off = k2 * tk2
                                    k_chunk_off = k2 * k_per_l2
                                    ops.load(
                                        l2_a,
                                        A[
                                            row : row + l2_m, k_l2_off : k_l2_off + tk2
                                        ].reshape(herd_m, tile_m, tk2),
                                    )
                                    ops.load(
                                        l2_b,
                                        B[
                                            n_outer : n_outer + herd_n,
                                            k_chunk_off : k_chunk_off + k_per_l2,
                                            :,
                                        ],
                                    )

                                    with herd() as h:

                                        @h.body
                                        def _(tx, ty):
                                            l1_a = air.alloc(
                                                [
                                                    1,
                                                    1,
                                                    tile_m // r,
                                                    tile_k_l1 // s,
                                                    r,
                                                    s,
                                                ],
                                                bf16,
                                                scope=h.private(),
                                            )
                                            l1_b = air.alloc(
                                                [tile_bytes], i8, scope=h.private()
                                            )
                                            for j in air.sequential(k_per_l2):
                                                k1 = j * tile_k_l1
                                                ops.load(
                                                    l1_a,
                                                    l2_a[tx, :, k1 : k1 + tile_k_l1]
                                                    .reshape(
                                                        1,
                                                        1,
                                                        tile_m // r,
                                                        r,
                                                        tile_k_l1 // s,
                                                        s,
                                                    )
                                                    .transpose(0, 1, 2, 4, 3, 5),
                                                )
                                                ops.load(l1_b, l2_b[ty, j, :])
                                                matmul(l1_a, l1_b, acc)

                                with herd() as dh:

                                    @dh.body
                                    def _(tx, ty):
                                        drain_fn(acc, drain)
                                        ops.store(
                                            drain[tx, ty, :, :, :, :].transpose(
                                                0, 1, 3, 4, 2, 5
                                            ),
                                            l2_c[tx, ty, :, :],
                                        )

                                ops.store(
                                    l2_c.transpose(0, 2, 1, 3),
                                    C[row : row + l2_m, col : col + l2_n],
                                )

    return launch.build(target="npu2")


def build_stitched(
    m, jobs, tile_m, tile_n, tile_k_l1, herd_m, herd_n, sym_suffix, link_with
):
    """Baseline: the same jobs as separate launches (one configuration each) in one ELF."""
    from gemm_bfp16 import bfp16_extern_syms, bfp16_weight_type, build_gemm_bfp16
    from air_examples.llms.shared.infra.stitching import (
        FuncArg,
        KernelSlice,
        stitch_elf,
    )

    args, slices = [], []
    for i, (k, n, tk2) in enumerate(jobs):
        base = len(args)
        args += [
            FuncArg(f"%arg{base}", f"memref<{m}x{k}xbf16>"),
            FuncArg(f"%arg{base + 1}", bfp16_weight_type(k, n, tile_n, tile_k_l1)),
            FuncArg(f"%arg{base + 2}", f"memref<{m}x{n}xbf16>"),
        ]
        ir = str(
            build_gemm_bfp16(
                m,
                k,
                n,
                tile_m,
                tk2,
                tile_k_l1,
                tile_n,
                herd_m,
                herd_n,
                sym_suffix,
                link_with,
            )
        )
        slices.append(
            KernelSlice(
                ir,
                f"g{i}",
                {0: base, 1: base + 1, 2: base + 2},
                extern_syms=bfp16_extern_syms(sym_suffix),
                private_from=(i == 0),
            )
        )
    return stitch_elf("gemm_stitched", args, slices)


SHAPES = {"qkv": (960, 1600), "o": (960, 960), "gu": (960, 5120), "dn": (2560, 960)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="o,dn", help="comma list of GEMM shapes")
    ap.add_argument("--tk2", default="480,320", help="tile_k_l2 per job")
    ap.add_argument("--tk1", type=int, default=160)
    ap.add_argument("--tile-n", type=int, default=80)
    ap.add_argument(
        "--mode",
        default="engine",
        choices=["engine", "loads", "stitched", "ffn", "qkv"],
    )
    ap.add_argument("--pingpong", default="", help="omit_pingpong value")
    ap.add_argument(
        "--tiling", default="2,3", help="stitched launches' runtime_loop_tiling_sizes"
    )
    ap.add_argument(
        "--chmux",
        default="",
        help="air channel multiplexing memory spaces, e.g. L2 or L1,L2",
    )
    ap.add_argument("--debug-ir", action="store_true")
    ap.add_argument(
        "--ffn-jobs", default="o,gu,dn", help="--mode ffn: subset of the three jobs"
    )
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    if args.mode == "ffn":
        return main_ffn(args)
    if args.mode == "qkv":
        return main_qkv(args)

    from gemm_bfp16 import compile_mm_bfp16
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler

    m, tile_m, herd = 256, 32, 4
    names = args.jobs.split(",")
    jobs = [(*SHAPES[nm], int(t)) for nm, t in zip(names, args.tk2.split(","))]
    tag = (
        f"{args.mode}_{'-'.join(names)}_k{args.tk2.replace(',', '-')}x{args.tk1}_n{args.tile_n}"
        f"{'_pp' + args.pingpong if args.pingpong else ''}"
        f"{'_mux' + args.chmux.replace(',', '') if args.chmux else ''}"
    )
    cache = KernelCache(
        str(_HERE / "build" / f"gemm_engine_{tag}"),
        verbose=False,
        profiler=Profiler(enabled=True),
    )
    sfx, obj = "_eng", "mm_eng.o"
    compile_mm_bfp16(tile_m, args.tile_n, args.tk1, sfx, obj)
    if args.mode == "engine":
        assert len(set(t for *_, t in jobs)) == 1, "the engine takes one tile_k_l2"
        mod = build_gemm_engine(
            m,
            [Job(f"A{i}", f"B{i}", f"C{i}", k, n) for i, (k, n, _) in enumerate(jobs)],
            tile_m,
            args.tile_n,
            args.tk1,
            jobs[0][2],
            herd,
            herd,
            sfx,
            obj,
        )
    else:
        build = {"loads": build_gemm_engine_loads, "stitched": build_stitched}[
            args.mode
        ]
        mod = build(m, jobs, tile_m, args.tile_n, args.tk1, herd, herd, sfx, obj)
    backend = {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gemm_stitched" if args.mode == "stitched" else "gemm_engine",
        "omit_pingpong": args.pingpong,
        "channel_multiplexing": args.chmux.split(",") if args.chmux else [],
        "debug_ir": args.debug_ir,
    }
    if args.mode == "stitched":
        backend["runtime_loop_tiling_sizes"] = [int(v) for v in args.tiling.split(",")]
    cache.compile_and_cache("gemm", mod, backend)

    rng = np.random.default_rng(0)
    bufs, outs, refs = [], [], []
    for k, n, _ in jobs:
        a = (rng.standard_normal((m, k)) * 0.5).astype(bfloat16)
        w = (rng.standard_normal((k, n)) / np.sqrt(k)).astype(bfloat16)
        bufs += [
            a,
            pack_b_bfp16ebs8(w, args.tile_n, args.tk1),
            np.zeros((m, n), bfloat16),
        ]
        outs.append(len(bufs) - 1)
        refs.append(a.astype(np.float32) @ w.astype(np.float32))

    def run():
        return cache.load_and_run(
            "gemm", backend, *bufs, output_indices=outs, bo_key="g"
        )

    res = run()
    for i, (nm, (k, n, _)) in enumerate(zip(names, jobs)):
        o = np.asarray(res[outs[i]], dtype=np.float32).reshape(m, n)
        ref = refs[i]
        cos = float(o.ravel() @ ref.ravel() / (np.linalg.norm(o) * np.linalg.norm(ref)))
        np.save(_HERE / "build" / f"gemm_engine_{tag}_{nm}.npy", o)
        print(f"  {nm}: cosine {cos:.6f}")
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["gemm"])
    print(
        f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
        f"p10 {dev[len(dev) // 10] * 1e3:.0f})"
    )


def _cos(a, b):
    a, b = np.asarray(a, np.float32).ravel(), np.asarray(b, np.float32).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main_ffn(args):
    """O + residual, RMSNorm + GateUp + SwiGLU, Down + residual as three jobs of one engine launch."""
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler

    m, emb, hid, tile_m, herd = 256, 960, 2560, 32, 4
    tn, tk1 = args.tile_n, args.tk1
    l2_n = tn * herd
    all_jobs = {
        "o": Job("attn", "wo", "res1", emb, emb, residual="x"),
        "gu": Job("res1", "wgu", "sw", emb, 2 * hid, rms=True, swiglu=True),
        "dn": Job("sw", "wdn", "out", hid, emb, residual="res1"),
    }
    sel = args.ffn_jobs.split(",")
    jobs = [all_jobs[nm] for nm in sel]
    names = ["attn", "wo", "x", "res1", "wgu", "sw", "wdn", "out"]
    used = {getattr(j, f) for j in jobs for f in ("a", "b", "c", "residual")} - {None}
    order = [nm for nm in names if nm in used]
    tag = f"ffn{'' if len(sel) == 3 else '_' + '-'.join(sel)}_n{tn}_k{tk1}{'_pp' + args.pingpong if args.pingpong else ''}"
    cache = KernelCache(
        str(_HERE / "build" / f"gemm_engine_{tag}"),
        verbose=False,
        profiler=Profiler(enabled=True),
    )
    sfx, obj = "_eng", "mm_engine.o"
    compile_mm_engine(tile_m, tn, tk1, sfx, obj, rms_k=emb)
    mod = build_gemm_engine(
        m, jobs, tile_m, tn, tk1, l2_n, herd, herd, sfx, obj, arg_order=order
    )
    backend = {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gemm_engine",
        "omit_pingpong": args.pingpong,
        "debug_ir": args.debug_ir,
    }
    cache.compile_and_cache("ffn", mod, backend)

    rng = np.random.default_rng(0)
    f32 = np.float32

    def bf(x):
        return np.asarray(x, f32).astype(bfloat16)

    attn = bf(rng.standard_normal((m, emb)) * 0.5)
    x = bf(rng.standard_normal((m, emb)))
    wo = bf(rng.standard_normal((emb, emb)) / np.sqrt(emb))
    wg = bf(rng.standard_normal((emb, hid)) / np.sqrt(emb))
    wu = bf(rng.standard_normal((emb, hid)) / np.sqrt(emb))
    wd = bf(rng.standard_normal((hid, emb)) / np.sqrt(hid))
    nw = bf(1.0 + 0.1 * rng.standard_normal(emb))
    wgu = permute_gate_up(
        bf(nw.astype(f32)[:, None] * wg.astype(f32)),
        bf(nw.astype(f32)[:, None] * wu.astype(f32)),
        tn,
        l2_n,
    )
    bufs = [
        attn,
        pack_b_bfp16ebs8(wo, tn, tk1),
        x,
        np.zeros((m, emb), bfloat16),
        pack_b_bfp16ebs8(wgu, tn, tk1),
        np.zeros((m, hid), bfloat16),
        pack_b_bfp16ebs8(wd, tn, tk1),
        np.zeros((m, emb), bfloat16),
    ]
    if "o" not in sel:
        bufs[3] = bf(rng.standard_normal((m, emb)))
    if "gu" not in sel:
        bufs[5] = bf(rng.standard_normal((m, hid)) * 0.3)
    bufs = [b for nm, b in zip(names, bufs) if nm in used]
    outs = [order.index(nm) for nm in ("res1", "sw", "out") if nm in used]

    def silu(v):
        return v / (1.0 + np.exp(-v))

    def ref_res1():
        return attn.astype(f32) @ wo.astype(f32) + x.astype(f32)

    def ref_sw(r1):
        nrm = (
            r1
            / np.sqrt(np.mean(r1 * r1, axis=1, keepdims=True) + 1e-5)
            * nw.astype(f32)
        )
        return silu(nrm @ wg.astype(f32)) * (nrm @ wu.astype(f32))

    def ref_out(s, r1):
        return s @ wd.astype(f32) + r1

    def run():
        return cache.load_and_run(
            "ffn", backend, *bufs, output_indices=outs, bo_key="ffn"
        )

    res = run()
    from reconfig_probe import ctrl_kb

    print(f"  control code {ctrl_kb(cache.cache_dir):.1f} KB")
    if len(sel) < 3:
        for nm in sel:
            if nm == "o":
                print(
                    f"  res1: cosine {_cos(np.asarray(res[order.index('res1')], f32), ref_res1()):.6f}"
                )
            if nm == "dn":
                producer = {"res1": "o", "sw": "gu"}

                def data(v):
                    src = (
                        res[order.index(v)]
                        if producer[v] in sel
                        else bufs[order.index(v)]
                    )
                    return np.asarray(src, f32).reshape(m, -1)

                sw_in, r1_in = data("sw"), data("res1")
                print(
                    f"  out:  cosine {_cos(np.asarray(res[order.index('out')], f32), ref_out(sw_in, r1_in)):.6f}"
                )
    else:
        r1, sw, out = (np.asarray(res[i], dtype=f32).reshape(m, -1) for i in outs)
        R1 = ref_res1()
        SW = ref_sw(R1)
        print(f"  res1: cosine {_cos(r1, R1):.6f}")
        print(
            f"  sw:   cosine {_cos(sw, SW):.6f} (from NPU res1: {_cos(sw, ref_sw(r1)):.6f})"
        )
        print(
            f"  out:  cosine {_cos(out, ref_out(SW, R1)):.6f} (from NPU sw, res1: {_cos(out, ref_out(sw, r1)):.6f})"
        )
        for nm, v in (("res1", r1), ("sw", sw), ("out", out)):
            np.save(_HERE / "build" / f"gemm_engine_{tag}_{nm}.npy", v)
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["ffn"])
    print(
        f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
        f"p10 {dev[len(dev) // 10] * 1e3:.0f})"
    )


def main_qkv(args):
    """RMSNorm + QKV + RoPE as one engine job (q/k head dims pair-interleaved)."""
    from air_examples.matrix_multiplication.bf16_x_bfp16.matmul_bf16_x_bfp16 import (
        pack_b_bfp16ebs8,
    )
    from air_examples.llms.shared.infra.cache import KernelCache, Profiler

    m, emb, nh, nkv, hd, tile_m, herd = 256, 960, 15, 5, 64, 32, 4
    kv = nkv * hd
    n = emb + 2 * kv
    tn, tk1 = args.tile_n, args.tk1
    tag = f"qkv_n{tn}_k{tk1}"
    cache = KernelCache(
        str(_HERE / "build" / f"gemm_engine_{tag}"),
        verbose=False,
        profiler=Profiler(enabled=True),
    )
    sfx, obj = "_eng", "mm_engine.o"
    compile_mm_engine(tile_m, tn, tk1, sfx, obj, rms_k=emb)
    mod = build_gemm_engine(
        m,
        [Job("x", "wqkv", "qkv", emb, n, rms=True, rope="rope")],
        tile_m,
        tn,
        tk1,
        tn * herd,
        herd,
        herd,
        sfx,
        obj,
        arg_order=["x", "wqkv", "rope", "qkv"],
    )
    backend = {
        "verbose": False,
        "omit_while_true_loop": False,
        "output_format": "elf",
        "instance_name": "gemm_engine",
        "debug_ir": args.debug_ir,
    }
    cache.compile_and_cache("qkv", mod, backend)

    rng = np.random.default_rng(0)
    f32 = np.float32

    def bf(x):
        return np.asarray(x, f32).astype(bfloat16)

    x = bf(rng.standard_normal((m, emb)))
    w = bf(rng.standard_normal((emb, n)) / np.sqrt(emb))
    nw = bf(1.0 + 0.1 * rng.standard_normal(emb))
    pos = np.minimum(np.arange(m), 240)
    inv = 1.0 / (10000.0 ** (np.arange(0, hd, 2) / hd))
    ang = np.outer(pos, inv)
    lut = bf(np.concatenate([np.cos(ang), np.sin(ang)], axis=1))
    perm = qkv_col_perm(nh, nkv, hd)
    wp = bf(nw.astype(f32)[:, None] * w.astype(f32))[:, perm]
    bufs = [
        x,
        pack_b_bfp16ebs8(wp, tn, tk1),
        bf(rope_table(lut.astype(f32), nh, nkv, hd, kv)),
        np.zeros((m, n), bfloat16),
    ]

    def rope(a, heads):
        a = a.reshape(m, heads, hd)
        c, s_ = lut.astype(f32)[:, None, : hd // 2], lut.astype(f32)[:, None, hd // 2 :]
        a1, a2 = a[..., : hd // 2], a[..., hd // 2 :]
        return np.concatenate([a1 * c - a2 * s_, a2 * c + a1 * s_], axis=-1).reshape(
            m, -1
        )

    xf = x.astype(f32)
    ref = (
        xf / np.sqrt(np.mean(xf * xf, axis=1, keepdims=True) + 1e-5) * nw.astype(f32)
    ) @ w.astype(f32)
    ref = np.concatenate(
        [rope(ref[:, :emb], nh), rope(ref[:, emb : emb + kv], nkv), ref[:, emb + kv :]],
        axis=1,
    )
    ref = ref[:, perm]

    def run():
        return cache.load_and_run(
            "qkv", backend, *bufs, output_indices=[3], bo_key="qkv"
        )

    out = np.asarray(run()[3], f32).reshape(m, n)
    from reconfig_probe import ctrl_kb

    print(f"  control code {ctrl_kb(cache.cache_dir):.1f} KB")
    for nm, sl in (
        ("q", slice(0, emb)),
        ("k", slice(emb, emb + kv)),
        ("v", slice(emb + kv, n)),
    ):
        print(f"  {nm}: cosine {_cos(out[:, sl], ref[:, sl]):.6f}")
    cache.profiler.kernel_breakdowns.clear()
    for _ in range(args.iters):
        run()
    dev = sorted(e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["qkv"])
    print(
        f"{tag}: device median {dev[len(dev) // 2] * 1e3:.0f} us (min {dev[0] * 1e3:.0f}, "
        f"p10 {dev[len(dev) // 10] * 1e3:.0f})"
    )


if __name__ == "__main__":
    main()
