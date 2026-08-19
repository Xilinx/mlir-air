# ./python/air/api/_trace.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Trace-time state, the herd context, and the declarations that feed them.

Nothing here emits IR until a herd body is registered. A DSL program looks like::

    A = air.tensor([M, N], bf16)          # declares a func argument
    with air.launch() as launch:
        @launch.body
        def _():
            with air.herd(product(range(...), range(...))) as h:
                @h.body
                def _(tx, ty):
                    ...

Tensors declared before a ``launch`` become that launch's interface, in
declaration order; the launch itself lives in ``_compile.py``, which replays the
recorded body inside an MLIR context once the function signature is known.

Herd grids larger than the physical array are strip-mined: a logical grid of
``G`` tiles on a physical array of ``P`` cores emits ``air.herd sizes=[P...]``
wrapped in an ``scf.for`` nest of ``G // P`` iterations, with the user's ``tx``
bound to ``tx_phys * (G // P) + i`` so each core owns a contiguous block of
tiles. This is the same decomposition the hand-written eltwise kernel performs
manually with its ``chunk_size`` arithmetic.
"""

import inspect
import itertools
import re

from ._index import IndexExpr
from ._value import Buffer, Tensor, Token

__all__ = [
    "Symbol",
    "HerdContext",
    "Scope",
    "herd",
    "alloc",
    "tensor",
    "symbol",
    "wait",
    "current_herd",
]


# ---------------------------------------------------------------------------
# Trace-time state
# ---------------------------------------------------------------------------

# Tensors and symbols declared since the last launch was opened; air.launch()
# claims them as its interface.
PENDING_TENSORS = []
PENDING_SYMBOLS = []

# The herd currently being traced, and the launch currently being traced.
_CURRENT_HERD = None
_ACTIVE_TRACE = None

# The device being traced for. A herd resolves its physical shape when it is
# constructed, which happens while the launch body runs -- so the launch has to
# publish its target here first, otherwise every herd would size itself for the
# default device and then fail placement on the real one.
_CURRENT_TARGET = None

# Default physical herd shape per target, indexed by the rank of the logical
# grid. These are core *counts*, and they are bounded by shim DMA capacity, not
# by how many compute tiles exist: a herd whose cores each stream their operands
# straight from L3 needs a shim channel per core per tensor. Measured against
# this package's three-tensor elementwise kernel, npu1 routes 4 cores and npu2
# routes 8; 16 cores fails placement with "no ShimNOCTile has sufficient DMA
# capacity". A kernel with fewer L3 operands, or one that stages through L2, can
# go wider -- pass shape= to air.herd() to ask for it.
PHYSICAL_HERD = {
    "npu1": {1: (4,), 2: (1, 4)},
    "npu2": {1: (8,), 2: (2, 4)},
}
DEFAULT_TARGET = "npu2"


# L1 (tile-local) memory per compute tile. Same on AIE2 and AIE2p.
L1_BYTES = 65536


class Trace:
    """Per-build state: the tensors bound to the function being traced."""

    def __init__(self, tensors):
        self.tensors = tensors
        # Peak declared L1 bytes across the trace, used to explain an
        # out-of-memory failure from the AIE placer in DSL terms.
        self.l1_peak = 0


def set_active_trace(trace):
    """Install (or clear) the active trace; returns the previous one."""
    global _ACTIVE_TRACE
    previous, _ACTIVE_TRACE = _ACTIVE_TRACE, trace
    return previous


def set_target(target):
    """Install (or clear) the target being traced for; returns the previous one."""
    global _CURRENT_TARGET
    previous, _CURRENT_TARGET = _CURRENT_TARGET, target
    return previous


def current_target():
    return _CURRENT_TARGET or DEFAULT_TARGET


def active_trace():
    if _ACTIVE_TRACE is None:
        raise RuntimeError(
            "air.herd(...) must be used inside a launch body; did you forget the "
            "@launch.body decorator?"
        )
    return _ACTIVE_TRACE


def current_herd():
    if _CURRENT_HERD is None:
        raise RuntimeError(
            "this operation must be used inside a herd body (@herd.body)"
        )
    return _CURRENT_HERD


def infer_name(fallback, depth=2):
    """Recover ``tile_m`` from the source line ``tile_m = air.symbol()``.

    Cosmetic only: it affects how ``launch.search_space`` and error messages
    read back, never what is emitted.
    """
    try:
        frame = inspect.stack()[depth]
        line = (frame.code_context or [""])[0]
        match = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if match:
            return match.group(1)
    except Exception:
        pass
    return fallback


# ---------------------------------------------------------------------------
# Symbol
# ---------------------------------------------------------------------------


class Symbol:
    """A compile-time integer the DSL leaves open.

    v1 *resolves* symbols rather than searching over them: the value is the
    ``hint`` if given, else the first of ``choices``, else ``default``. The
    resolved binding is recorded in ``launch.search_space`` so a program can
    report what it was compiled with. Autotuning is not implemented -- a symbol
    is a named constant with a recorded provenance, nothing more.
    """

    _counter = itertools.count()

    def __init__(self, choices=None, hint=None, default=64, name=None):
        # `hint` is a single value and `choices` is the candidate list. Getting
        # these the wrong way round otherwise fails deep inside int(), with a
        # message that says nothing about which argument was wrong.
        if hint is not None and not isinstance(hint, int):
            raise TypeError(
                f"air.symbol(hint=...) takes a single integer, got "
                f"{type(hint).__name__} {hint!r}"
                + (
                    "; pass a list of candidates as choices= instead"
                    if isinstance(hint, (list, tuple, set, range))
                    else ""
                )
            )
        if choices is not None:
            if isinstance(choices, int):
                raise TypeError(
                    f"air.symbol(choices=...) takes a sequence of integers, got "
                    f"the single value {choices!r}; did you mean hint={choices!r}?"
                )
            choices = list(choices)
            bad = [c for c in choices if not isinstance(c, int)]
            if bad:
                raise TypeError(
                    f"air.symbol(choices=...) must contain integers; got {bad!r}"
                )
            if not choices:
                raise ValueError("air.symbol(choices=[]) has no candidates")
            if hint is not None and hint not in choices:
                raise ValueError(
                    f"air.symbol(hint={hint}) is not one of choices={choices}"
                )
        self.choices = choices
        self.hint = hint
        self.name = name or infer_name(f"sym{next(Symbol._counter)}", depth=3)
        if hint is not None:
            self.value = int(hint)
        elif self.choices:
            self.value = int(self.choices[0])
        else:
            self.value = int(default)
        PENDING_SYMBOLS.append(self)

    def __index__(self):
        return self.value

    def __int__(self):
        return self.value

    def __repr__(self):
        return f"air.api.symbol({self.name}={self.value})"

    # Arithmetic yields plain ints: a resolved symbol is just a constant.
    def __add__(self, o):
        return self.value + int(o)

    __radd__ = __add__

    def __sub__(self, o):
        return self.value - int(o)

    def __rsub__(self, o):
        return int(o) - self.value

    def __mul__(self, o):
        return self.value * int(o)

    __rmul__ = __mul__

    def __floordiv__(self, o):
        return self.value // int(o)

    def __rfloordiv__(self, o):
        return int(o) // self.value

    def __mod__(self, o):
        return self.value % int(o)

    def __eq__(self, o):
        return self.value == int(o)

    def __hash__(self):
        return hash(self.value)


class Scope:
    """A memory scope handle produced by ``herd.private()``."""

    __slots__ = ("kind", "owner")

    def __init__(self, kind, owner):
        self.kind = kind
        self.owner = owner

    def __repr__(self):
        return f"{type(self.owner).__name__}.{self.kind}()"


# ---------------------------------------------------------------------------
# Grid parsing
# ---------------------------------------------------------------------------


class _Dim:
    """One dimension of an iteration space: ``count`` tiles of size ``step``."""

    __slots__ = ("start", "step", "count")

    def __init__(self, start, step, count):
        self.start = start
        self.step = step
        self.count = count


def _dim_from_range(r):
    if len(r) == 0:
        raise ValueError(f"empty iteration range {r}")
    return _Dim(r.start, r.step, len(r))


def _dim_from_values(values):
    """Recover (start, step, count) from a materialised sequence of offsets."""
    if len(values) == 0:
        raise ValueError("empty iteration range")
    if len(values) == 1:
        return _Dim(values[0], 1, 1)
    step = values[1] - values[0]
    for a, b in zip(values, values[1:]):
        if b - a != step:
            raise ValueError(
                "air.api requires a uniformly strided iteration space; got a "
                f"non-constant step in {list(values[:4])}..."
            )
    return _Dim(values[0], step, len(values))


def parse_grid(iterable):
    """Parse a range / product-of-ranges into per-dimension :class:`_Dim`s."""
    if isinstance(iterable, range):
        return [_dim_from_range(iterable)]

    if isinstance(iterable, itertools.product):
        # product materialises its inputs at construction; __reduce__ hands them
        # back as tuples, giving exact starts and steps without consuming the
        # iterator or costing anything product has not already paid.
        args = iterable.__reduce__()[1]
        if not args or not all(isinstance(a, tuple) for a in args):
            raise TypeError(
                "air.api could not introspect this itertools.product; pass a "
                "product of range() objects"
            )
        return [_dim_from_values(a) for a in args]

    if isinstance(iterable, (list, tuple)) and all(
        isinstance(r, range) for r in iterable
    ):
        return [_dim_from_range(r) for r in iterable]

    raise TypeError(
        f"cannot use {type(iterable).__name__} as a herd iteration space; "
        "air.api accepts range(...) or itertools.product(range(...), ...)"
    )


def _largest_divisor_at_most(n, cap):
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def _positional_arity(fn):
    sig = inspect.signature(fn)
    return sum(
        1
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )


# ---------------------------------------------------------------------------
# Herd
# ---------------------------------------------------------------------------


class HerdContext:
    """A herd of compute cores over a (possibly strip-mined) tile grid."""

    def __init__(self, iterable, name=None, shape=None, target=None):
        self.dims = parse_grid(iterable)
        if len(self.dims) > 2:
            raise NotImplementedError(
                f"air.api supports 1-D and 2-D herd grids; got {len(self.dims)}-D"
            )
        self.name = name or "herd_0"
        self.target = target or current_target()
        self.grid = tuple(d.count for d in self.dims)
        self.tile_sizes = tuple(d.step for d in self.dims)
        self.physical = self._resolve_physical(shape)
        self.repeats = tuple(g // p for g, p in zip(self.grid, self.physical))
        self._buffers = []
        self._registered = False

    def _resolve_physical(self, shape):
        by_rank = PHYSICAL_HERD.get(self.target, PHYSICAL_HERD[DEFAULT_TARGET])
        default = by_rank.get(len(self.grid))
        if default is None:
            raise NotImplementedError(
                f"no default herd shape for a {len(self.grid)}-D grid on "
                f"{self.target}; pass shape= to air.herd()"
            )
        if shape is not None:
            shape = tuple(int(s) for s in shape)
            if len(shape) != len(self.grid):
                raise ValueError(
                    f"explicit herd shape {shape} has rank {len(shape)} but the "
                    f"iteration space is {len(self.grid)}-D"
                )
            for g, p in zip(self.grid, shape):
                if p <= 0 or g % p:
                    raise ValueError(
                        f"herd shape {shape} does not evenly divide the logical "
                        f"grid {self.grid}; each dimension must strip-mine exactly"
                    )
            return shape
        # Largest divisor of the logical grid that fits the array: strip-mining
        # is then always exact, so no tile is silently dropped.
        return tuple(
            _largest_divisor_at_most(g, cap) for g, cap in zip(self.grid, default)
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def private(self):
        return Scope("private", self)

    def shared(self):
        raise NotImplementedError(
            "shared L1 scope is not exposed by air.api yet; use herd.private()"
        )

    @property
    def body(self):
        def decorator(fn):
            if self._registered:
                raise RuntimeError("herd body registered twice")
            self._registered = True
            self._emit(fn)
            return fn

        return decorator

    def register_buffer(self, buf):
        self._buffers.append(buf)

    # -- emission -----------------------------------------------------------

    def _emit(self, fn):
        trace = active_trace()
        n_expected = len(self.grid)
        n_actual = _positional_arity(fn)
        if n_actual != n_expected:
            raise TypeError(
                f"herd body takes {n_actual} coordinate argument(s) but the "
                f"iteration space is {n_expected}-D"
            )

        from air.dialects.air import herd as herd_region
        from air.dialects.memref import DeallocOp
        from air.dialects.scf import for_ as range_, yield_

        tensors = trace.tensors
        operands = [t.value for t in tensors]
        # air.herd is always 2-D. A 1-D grid is laid out along x -- [P, 1], the
        # orientation the hand-written eltwise_add kernel uses for its 8-core
        # npu2 config. [1, P] does not place.
        two_d = len(self.physical) == 2
        sizes = list(self.physical) if two_d else [self.physical[0], 1]

        herd_self = self

        @herd_region(name=self.name, sizes=sizes, operands=operands)
        def herd_body(*args):
            global _CURRENT_HERD

            coords = list(args[:2])
            inner_tensors = args[4:]

            # Inside the herd, tensors resolve to the herd's block arguments.
            saved = [t.value for t in tensors]
            for t, v in zip(tensors, inner_tensors):
                t.value = v
            previous, _CURRENT_HERD = _CURRENT_HERD, herd_self

            phys_coords = coords if two_d else coords[:1]

            def run(strip_ivs):
                tile_ids = []
                for axis, phys in enumerate(phys_coords):
                    tile = IndexExpr.leaf(phys, f"t{axis}") * herd_self.repeats[axis]
                    if herd_self.repeats[axis] > 1:
                        tile = tile + IndexExpr.leaf(strip_ivs[axis], f"i{axis}")
                    tile_ids.append(tile)
                fn(*tile_ids)
                # Free within the same region the allocs were emitted into: with
                # strip-mining that region is the innermost scf.for body, and a
                # dealloc hoisted above it would not be dominated by its alloc.
                for buf in herd_self._buffers:
                    DeallocOp(buf.value)
                herd_self._buffers.clear()

            try:
                run_strip_mined(run, herd_self.repeats, range_, yield_)
            finally:
                herd_self._buffers.clear()
                for t, v in zip(tensors, saved):
                    t.value = v
                _CURRENT_HERD = previous


def run_strip_mined(run, repeats, range_, yield_):
    """Call ``run(ivs)`` inside an scf.for nest over the non-unit repeat factors."""
    axes = [axis for axis, r in enumerate(repeats) if r > 1]

    def rec(level, ivs):
        if level == len(axes):
            full = [None] * len(repeats)
            for axis, iv in zip(axes, ivs):
                full[axis] = iv
            run(full)
            return
        for iv in range_(0, repeats[axes[level]], 1):
            rec(level + 1, ivs + [iv])
            yield_([])

    rec(0, [])


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def herd(iterable, name=None, shape=None, target=None):
    """A herd of cores over ``iterable``, strip-mined onto the physical array."""
    return HerdContext(iterable, name=name, shape=shape, target=target)


def tensor(shape, dtype, name=None):
    """Declare a host-visible L3 array; becomes a kernel argument."""
    t = Tensor(shape, dtype, name=name or infer_name(f"t{len(PENDING_TENSORS)}"))
    PENDING_TENSORS.append(t)
    return t


def alloc(shape, dtype, scope=None, vector=None):
    """Allocate an L1 tile inside a herd body."""
    from air.ir import IntegerAttr, MemRefType
    from air.dialects.air import MemorySpace
    from air.dialects.memref import AllocOp
    from air.extras import types as T

    h = current_herd()
    if scope is None:
        raise ValueError("air.alloc requires scope=<herd>.private()")
    if not isinstance(scope, Scope) or scope.kind != "private":
        raise NotImplementedError(
            f"air.api can only allocate in a herd's private (L1) scope, got {scope!r}"
        )

    memref_ty = MemRefType.get(
        [int(s) for s in shape],
        dtype.mlir(),
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )
    # Reject what is certainly impossible before the AIE placer has to. A
    # single tile's declared buffers cannot exceed L1; note that the pipeline
    # may additionally ping-pong these, so fitting here is necessary but not
    # sufficient (see the hint attached to placement failures).
    nbytes = 1
    for extent in shape:
        nbytes *= int(extent)
    nbytes *= dtype.itemsize
    live = sum(_buffer_bytes(b) for b in h._buffers) + nbytes
    if nbytes > L1_BYTES:
        raise ValueError(
            f"air.alloc({list(shape)}, {dtype}) needs {nbytes / 1024:.1f} KB but a "
            f"compute tile has {L1_BYTES / 1024:.0f} KB of L1; use a smaller tile"
        )
    if live > L1_BYTES:
        raise ValueError(
            f"L1 budget exceeded: this herd body has {live / 1024:.1f} KB of live "
            f"buffers but a compute tile has {L1_BYTES / 1024:.0f} KB; use a "
            "smaller tile"
        )

    op = AllocOp(memref_ty, [], [])
    buf = Buffer(shape, dtype, scope=scope, vector_width=vector, value=op.result)
    h.register_buffer(buf)
    trace = active_trace()
    trace.l1_peak = max(trace.l1_peak, live)
    return buf


def _buffer_bytes(buf):
    n = buf.dtype.itemsize
    for extent in buf.shape:
        n *= extent
    return n


def symbol(choices=None, hint=None, default=64, name=None):
    """A compile-time integer, resolved now and reported in ``search_space``.

    ``name`` overrides the name inferred from the assigning source line, which
    matters when symbols are built in a loop or comprehension and there is no
    ``tile_m = air.symbol()`` line to read.
    """
    return Symbol(choices=choices, hint=hint, default=default, name=name)


def wait(*tokens):
    """Join tokens. AIR builds the real dependency graph from program order."""
    for t in tokens:
        if not isinstance(t, Token):
            raise TypeError(f"air.wait expects tokens, got {type(t).__name__}")
    return Token()
