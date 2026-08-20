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
from ._pack import PackedShape
from ._value import Buffer, Tensor, Token

__all__ = [
    "Symbol",
    "HerdContext",
    "SegmentContext",
    "Scope",
    "herd",
    "segment",
    "alloc",
    "tensor",
    "symbol",
    "wait",
    "current_herd",
    "current_segment",
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
_CURRENT_SEGMENT = None
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
# Resolved against the installed NPU rather than pinned, so one traced program
# runs on whichever generation is present. npu2 is only the answer when there is
# no device to ask (compile-only on a host without XRT).
DEFAULT_TARGET = "auto"
NO_DEVICE_TARGET = "npu2"


# L1 (tile-local) memory per compute tile. Same on AIE2 and AIE2p.
L1_BYTES = 65536

# L2 memory per *memtile*, and how many memtiles a segment can reach. There is
# one memtile per column and a segment spans several, so its budget is a
# multiple of this figure -- not this figure.
#
# Charging everything to one memtile is wrong in the harmful direction: it
# rejects designs that run. matrix_multiplication/bf16 at herd 4x4 with an f32
# output stages 608 KB across four memtiles, a configuration CI exercises on
# npu2, and a 512 KB cap refuses it.
#
# The herds do not exist when the L2 allocs run -- the allocs come first -- so
# the column span is not yet known and the check uses the whole device's L2.
# That makes it a test for "certainly impossible" rather than a placement
# prediction, which is all it was ever able to be: the AIE placer owns the real
# answer and reports it accurately.
L2_BYTES_PER_MEMTILE = 512 * 1024
DEVICE_COLUMNS = {"npu1": 4, "npu2": 8}


class Trace:
    """Per-build state: the tensors bound to the function being traced."""

    def __init__(self, tensors, module=None):
        self.tensors = tensors
        # The enclosing builtin.module, so that air.extern can place its private
        # func.func declarations at module scope from deep inside a herd body.
        self.module = module
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
    return _CURRENT_TARGET or resolve_target(DEFAULT_TARGET)


def resolve_target(target):
    """Map None/"auto" onto the NPU generation actually installed.

    The herd sizes itself from this (PHYSICAL_HERD) and the backend compiles for
    it, so both have to agree: a design sized for one generation and compiled
    for the other places wrong, and a design compiled for a generation that is
    not plugged in loads without error and computes nothing.
    """
    if target not in (None, "auto"):
        if target not in PHYSICAL_HERD:
            raise ValueError(
                f"unknown target {target!r}; expected 'auto' or one of "
                f"{sorted(PHYSICAL_HERD)}"
            )
        return target
    from air.backend.xrt import detect_target_device

    return detect_target_device(default=NO_DEVICE_TARGET)


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


def current_segment(required=True):
    if _CURRENT_SEGMENT is None and required:
        raise RuntimeError(
            "this operation must be used inside a segment body (@segment.body)"
        )
    return _CURRENT_SEGMENT


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
    """One dimension of an iteration space: ``count`` tiles of size ``step``.

    There is no ``start``: the space must begin at 0, enforced below.
    """

    __slots__ = ("step", "count")

    def __init__(self, step, count):
        self.step = step
        self.count = count


def _reject_nonzero_start(start, what):
    """A herd body gets tile *indices*, so a non-zero start cannot survive."""
    if start:
        raise NotImplementedError(
            f"air.api requires a herd iteration space starting at 0; got {what}"
            f" (start={start}). A body receives tile indices and a slice built "
            "from them offsets by index * tile_size, so a non-zero start would "
            "be dropped and the kernel would read the wrong window. Index from "
            "0 and add the start inside the body."
        )


def _dim_from_range(r):
    if len(r) == 0:
        raise ValueError(f"empty iteration range {r}")
    _reject_nonzero_start(r.start, f"range({r.start}, {r.stop}, {r.step})")
    if (r.stop - r.start) % r.step:
        raise ValueError(
            f"iteration range range({r.start}, {r.stop}, {r.step}) does not "
            f"tile its extent exactly: {len(r)} steps of {r.step} span "
            f"{len(r) * r.step}, past the extent of {r.stop - r.start}. "
            "air.api has no partial tiles, so the last tile would read and "
            f"write {len(r) * r.step - (r.stop - r.start)} elements past the "
            "end of the tensor. Use a step that divides the extent."
        )
    return _Dim(r.step, len(r))


def _dim_from_values(values):
    """Recover (step, count) from a materialised sequence of offsets.

    Only the offsets survive itertools.product, so unlike a range there is no
    extent here to check the last tile against -- that check belongs to the
    caller, which knows the tensor shape.
    """
    if len(values) == 0:
        raise ValueError("empty iteration range")
    _reject_nonzero_start(values[0], f"{list(values[:4])}...")
    if len(values) == 1:
        # itertools.product materialises its inputs, so a one-element axis
        # arrives as `(0,)` and its step is simply gone -- range(0, 32, 32) and
        # range(0, 32, 1)[:1] are indistinguishable here. Guessing 1 is what the
        # tile size then becomes, and the kernel quietly computes on 1x1 tiles.
        raise ValueError(
            "air.herd cannot recover the tile size of a single-tile axis from "
            "an itertools.product: product materialises its inputs, so "
            "range(0, N, N) reaches the DSL as just (0,) with no step. Pass the "
            "grid as a list of ranges instead -- air.herd([range(0, M, tm), "
            "range(0, N, tn)]) -- which keeps the steps, or use a 1-D "
            "air.herd(range(...)) if the other axis is trivial."
        )
    step = values[1] - values[0]
    for a, b in zip(values, values[1:]):
        if b - a != step:
            raise ValueError(
                "air.api requires a uniformly strided iteration space; got a "
                f"non-constant step in {list(values[:4])}..."
            )
    return _Dim(step, len(values))


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
# Segment
# ---------------------------------------------------------------------------


class SegmentContext:
    """A device segment: memtile (L2) scope, with herds nested inside it.

    A segment stages data between L3 and L1 so that a herd's cores read their
    operands from a memtile rather than each streaming from L3 over its own shim
    channel. That is what the hand-written matmul examples all do, and it is the
    reason a wide herd is worth having.

    ``air.launch``, ``air.segment`` and ``air.herd`` are independent ops, and a
    kernel that does not need staging keeps the plain ``func`` + ``air.herd``
    shape. A segment, though, is emitted **inside an ``air.launch``**, because
    the pipeline will not supply one: ``air-insert-launch-around-herd`` wraps a
    *bare* herd in a launch and a segment, and skips a herd that is already
    inside a segment. A segment with no launch above it compiles, and silently
    computes zeros for anything beyond a plain copy -- verified on npu1, and the
    reason every hand-written staging example in the tree nests the two.
    """

    def __init__(self, grid=None, name=None):
        # A grid here is the *launch* iteration space: one segment instance per
        # point, which is how the reference matmul tiles M and N. It cannot be
        # the herd's job, because the L2 staging buffers are re-filled per point
        # -- the outer tiling has to sit where the L3->L2 transfers are.
        self.dims = parse_grid(grid) if grid is not None else ()
        if len(self.dims) > 2:
            raise NotImplementedError(
                f"air.launch is 2-D, so a segment grid is 1-D or 2-D; got "
                f"{len(self.dims)}-D"
            )
        self.grid = tuple(d.count for d in self.dims)
        self.tile_sizes = tuple(d.step for d in self.dims)
        self.name = name or "segment_0"
        self._buffers = []
        self._registered = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def private(self):
        """L2 (memtile) scope, shared by every core in the segment."""
        return Scope("private", self)

    def shared(self):
        """L1 scope, but allocated *here* rather than in the herd body.

        A buffer in a herd body lives and dies with one entry into the herd. An
        accumulator cannot: it is zeroed once, added into across every trip of a
        reduction loop that sits at segment scope -- so the herd is entered once
        per trip -- and drained at the end. Allocating it here gives it the
        lifetime of the segment, and the nested herd receives it as an operand
        like any other segment buffer.

        It is still L1, so it is still charged against the 64 KB core budget, and
        each core addresses its own slab of it -- which is why such a buffer is
        allocated with leading herd dimensions and subscripted by tile
        coordinate.
        """
        return Scope("shared", self)

    def register_buffer(self, buf):
        self._buffers.append(buf)

    @property
    def body(self):
        def decorator(fn):
            if self._registered:
                raise RuntimeError("segment body registered twice")
            self._registered = True
            self._emit(fn)
            return fn

        return decorator

    def _emit(self, fn):
        global _CURRENT_SEGMENT

        trace = active_trace()
        n_expected = len(self.dims)
        if _positional_arity(fn) != n_expected:
            raise TypeError(
                f"segment body takes {_positional_arity(fn)} coordinate "
                f"argument(s) but the launch iteration space is "
                f"{n_expected}-D"
                + (
                    "; a segment with no grid is a single instance and its body "
                    "takes no arguments"
                    if n_expected == 0
                    else ""
                )
            )

        from air.dialects.air import launch as launch_region
        from air.dialects.air import segment as segment_region
        from air.dialects.memref import DeallocOp

        tensors = trace.tensors
        segment_self = self

        # air.launch is always 2-D; an absent or 1-D grid pads with 1. Its block
        # arguments are sizes*2 + operands, so the operands start at index 4.
        counts = list(self.grid) + [1] * (2 - len(self.grid))

        @launch_region(sizes=counts, operands=[t.value for t in tensors])
        def launch_body(*largs):
            # The launch induction variables become segment operands, ahead of
            # the tensors -- air.segment is IsolatedFromAbove, so a coordinate
            # cannot simply be referenced from inside it.
            ivs = list(largs[: len(segment_self.dims)])
            outer = ivs + list(largs[4:])

            # sizes=[] on the segment means its block arguments are exactly the
            # operands.
            @segment_region(name=segment_self.name, operands=outer)
            def segment_body(*args):
                global _CURRENT_SEGMENT

                coords = [
                    IndexExpr.leaf(v, f"s{axis}")
                    for axis, v in enumerate(args[: len(ivs)])
                ]
                bound = args[len(ivs) :]
                saved = [t.value for t in tensors]
                for t, v in zip(tensors, bound):
                    t.value = v
                previous, _CURRENT_SEGMENT = _CURRENT_SEGMENT, segment_self
                try:
                    fn(*coords)
                    # Freed here, inside the region the allocs were emitted into.
                    for buf in segment_self._buffers:
                        DeallocOp(buf.value)
                finally:
                    segment_self._buffers.clear()
                    for t, v in zip(tensors, saved):
                        t.value = v
                    _CURRENT_SEGMENT = previous


def segment(grid=None, name=None):
    """A device segment with L2 scope; nest herds inside its body."""
    return SegmentContext(grid=grid, name=name)


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
        self.target = resolve_target(target) if target else current_target()
        self.grid = tuple(d.count for d in self.dims)
        self.tile_sizes = tuple(d.step for d in self.dims)
        self.physical = self._resolve_physical(shape)
        self.repeats = tuple(g // p for g, p in zip(self.grid, self.physical))
        self._buffers = []
        self._registered = False
        # Object files required by air.extern calls in this herd's body.
        self._objects = {}
        # The core's position in the herd, bound while the body is traced.
        self._coords = []

    def require_object(self, obj, kernel):
        """Record that this herd calls ``kernel`` from ``obj``.

        aircc links one object per herd -- link_with is a single string -- so a
        body that reaches into two of them cannot be built.
        """
        if self._objects and obj not in self._objects:
            other, other_kernel = next(iter(self._objects.items()))
            raise ValueError(
                f"herd '{self.name}' calls {kernel} from {obj!r} and "
                f"{other_kernel} from {other!r}, but a herd links against a "
                "single object file. Compile both kernels into one object, or "
                "put them in separate herds."
            )
        self._objects[obj] = kernel

    def _resolve_physical(self, shape):
        by_rank = PHYSICAL_HERD[self.target]
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
            "air.api has no <herd>.shared(): a buffer allocated in a herd body "
            "is core-local, and it dies when the body ends. For a buffer the "
            "herd's cores share -- an accumulator carried across a reduction "
            "loop at segment scope, say -- use <segment>.shared(), which "
            "allocates L1 with the segment's lifetime and passes it into the "
            "herd as an operand."
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

        from ._loop import aborted_loops, enter_body, exit_body

        aborted_before = aborted_loops()

        tensors = trace.tensors
        # air.herd is IsolatedFromAbove, so an L2 buffer allocated in the
        # enclosing segment has to be passed in explicitly. Every live one is
        # passed, referenced or not -- the same policy already applied to
        # tensors, and the tracer cannot know what the body will touch until it
        # has run it.
        enclosing = current_segment(required=False)
        staged = list(enclosing._buffers) if enclosing is not None else []
        operands = [t.value for t in tensors] + [b.value for b in staged]
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
            inner = args[4:]

            # Inside the herd, tensors and staged L2 buffers resolve to the
            # herd's block arguments, in the order they were passed as operands.
            saved = [t.value for t in tensors]
            saved_staged = [b.value for b in staged]
            for t, v in zip(tensors, inner[: len(tensors)]):
                t.value = v
            for b, v in zip(staged, inner[len(tensors) :]):
                b.value = v
            previous, _CURRENT_HERD = _CURRENT_HERD, herd_self

            phys_coords = coords if two_d else coords[:1]
            # The core's own position in the herd, as opposed to the logical
            # tile id the body is handed (which folds in strip-mining). A
            # herd-shared buffer is indexed by this: it has one slab per core,
            # not one per logical tile.
            herd_self._coords = [
                IndexExpr.leaf(c, f"c{axis}") for axis, c in enumerate(phys_coords)
            ]

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

            outer_depth = enter_body()
            try:
                run_strip_mined(run, herd_self.repeats, range_, yield_)
            finally:
                exit_body(outer_depth)
                herd_self._buffers.clear()
                for t, v in zip(tensors, saved):
                    t.value = v
                for b, v in zip(staged, saved_staged):
                    b.value = v
                _CURRENT_HERD = previous

        # aircc compiles the object named here alongside the herd's cores.
        if herd_self._objects:
            from air.ir import StringAttr

            herd_body.attributes["link_with"] = StringAttr.get(
                next(iter(herd_self._objects))
            )

        if aborted_loops() != aborted_before:
            raise RuntimeError(
                "a body left an air.sequential loop early (break, return, or a "
                "swallowed exception). An air.sequential body is traced once and "
                "stands for every trip, so an early exit does not shorten the "
                "loop -- it truncates the body of all of them, and the kernel "
                "computes a partial result. Restructure the loop bounds instead."
            )


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
    """Allocate a tile: L1 in a herd body, or L2 in a segment body."""
    from air.ir import IntegerAttr, MemRefType
    from air.dialects.air import MemorySpace
    from air.dialects.memref import AllocOp
    from air.extras import types as T

    from ._loop import loop_depth

    if scope is None:
        raise ValueError(
            "air.alloc requires scope=<herd>.private() (L1) or "
            "scope=<segment>.private() (L2)"
        )
    if not isinstance(scope, Scope) or scope.kind not in ("private", "shared"):
        raise NotImplementedError(
            f"air.api can only allocate in a private or shared scope, got {scope!r}"
        )

    owner = scope.owner
    if isinstance(owner, HerdContext):
        if scope.kind != "private":
            raise NotImplementedError(
                "air.api has no <herd>.shared(); a buffer allocated in a herd "
                "body is core-local. For a buffer the herd's cores share, use "
                "<segment>.shared(), which allocates it at segment scope."
            )
        space, memory_space, capacity, where = "L1", MemorySpace.L1, L1_BYTES, "a herd"
        holder = current_herd()
    elif isinstance(owner, SegmentContext):
        # private() is the memtile; shared() is core L1 with segment lifetime.
        if scope.kind == "shared":
            space, memory_space, capacity = "L1", MemorySpace.L1, L1_BYTES
            if not isinstance(shape, PackedShape):
                # The L1 budget for a shared buffer is per *core*, so the
                # accounting has to know which leading dimensions are the herd.
                # A PackedShape records them; a plain list does not, and
                # guessing would either overcharge (rejecting a design that
                # fits) or undercharge (passing one that does not).
                raise NotImplementedError(
                    "<segment>.shared() currently requires a micro-tiled shape "
                    "from air.micro_tile(...), because the per-core L1 charge "
                    "depends on knowing which leading dimensions are the herd. "
                    f"Got a plain shape {list(shape)}. Either allocate it with "
                    "mm.c(...)/mm.a(...)/mm.b(...), or, if the buffer does not "
                    "have to outlive one entry into the herd, allocate it "
                    "per-core in the herd body with <herd>.private()."
                )
        else:
            space, memory_space, capacity = (
                "L2",
                MemorySpace.L2,
                L2_BYTES_PER_MEMTILE * DEVICE_COLUMNS[current_target()],
            )
        where = "a segment"
        holder = current_segment()
    else:
        raise NotImplementedError(
            f"air.api cannot allocate in {scope!r}; use <herd>.private() for L1 "
            "or <segment>.private() for L2"
        )
    if holder is not owner:
        raise RuntimeError(
            f"air.alloc(scope=...) names {where} that is not the one currently "
            "being traced; allocate inside the body of the scope you are asking "
            "for"
        )
    if space == "L1" and loop_depth():
        raise NotImplementedError(
            "air.alloc inside an air.sequential body is not supported: the herd "
            "frees its buffers once the body is finished, which is outside the "
            "loop, so the dealloc would not be dominated by its alloc. Hoist the "
            "allocation above the loop -- the buffer is reused across trips, "
            "which is what a loop is for."
        )
    # 0 is meaningful -- it selects the scalar path. Negative is not, and it
    # would otherwise pass a caller's own `tile % width` guard unnoticed, since
    # Python's modulo is 0 for any divisor of the tile regardless of sign.
    if vector is not None and int(vector) < 0:
        raise ValueError(
            f"air.alloc vector width must be >= 0, got {vector} "
            "(0 selects the scalar path)"
        )

    memref_ty = MemRefType.get(
        [int(s) for s in shape],
        dtype.mlir(),
        memory_space=IntegerAttr.get(T.i32(), memory_space),
    )
    # Reject what is certainly impossible before the AIE placer has to. Note
    # that for L1 the pipeline may additionally ping-pong these, so fitting here
    # is necessary but not sufficient (see the hint on placement failures).
    nbytes = 1
    for extent in shape:
        nbytes *= int(extent)
    nbytes *= dtype.itemsize
    if scope.kind == "shared":
        # A herd-shared buffer is declared once with leading herd dimensions,
        # and each core addresses exactly one slab of it. Charging the whole
        # thing against one core's 64 KB would reject configurations that fit
        # comfortably -- a 4x4 herd of 16 KB slabs is 256 KB in total and 16 KB
        # per core. The leading dimensions are the herd shape by construction:
        # that is what makes the per-core subview well defined.
        per_core = nbytes
        for extent in shape.lead:
            per_core //= int(extent)
        nbytes = per_core
    # A segment holds L2 memtile buffers and herd-shared L1 buffers at once, so
    # each budget only counts its own space.
    live = sum(_buffer_bytes(b) for b in holder._buffers if b.space == space) + nbytes
    if space == "L1":
        unit, verb = "a compute tile", "has"
    else:
        cols = DEVICE_COLUMNS[current_target()]
        unit, verb = f"this device's {cols} memtiles", "have"
    if nbytes > capacity:
        raise ValueError(
            f"air.alloc({list(shape)}, {dtype}) needs {nbytes / 1024:.1f} KB but "
            f"{unit} {verb} {capacity / 1024:.0f} KB of {space}; use a smaller "
            "tile"
        )
    if live > capacity:
        raise ValueError(
            f"{space} budget exceeded: this {where.split()[-1]} body has "
            f"{live / 1024:.1f} KB of live buffers but {unit} {verb} "
            f"{capacity / 1024:.0f} KB; use a smaller tile"
        )

    op = AllocOp(memref_ty, [], [])
    buf = Buffer(
        shape,
        dtype,
        scope=scope,
        vector_width=vector,
        value=op.result,
        space=space,
        # A PackedShape is an ordinary tuple as far as the memref is concerned --
        # the packing is not a layout, it is just this shape. Carrying the
        # descriptor onto the buffer is what lets ops.load/store derive the
        # micro-tiled access pattern without the call site restating it.
        pack=shape if isinstance(shape, PackedShape) else None,
    )
    holder.register_buffer(buf)
    if space == "L1":
        trace = active_trace()
        trace.l1_peak = max(trace.l1_peak, live)
    return buf


def _buffer_bytes(buf):
    n = buf.dtype.itemsize
    for extent in buf.shape:
        n *= extent
    # Mirror the per-core charge applied when a herd-shared buffer was
    # allocated, so the running total and the new allocation are on the same
    # footing.
    if getattr(buf.scope, "kind", None) == "shared" and buf.pack is not None:
        for extent in buf.pack.lead:
            n //= int(extent)
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
