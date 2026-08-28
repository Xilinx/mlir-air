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

from ._index import IndexExpr, Leaf
from ._value import Buffer, Tensor, Token

__all__ = [
    "Symbol",
    "HerdContext",
    "SegmentContext",
    "Scope",
    "herd",
    "segment",
    "alloc",
    "dealloc",
    "tensor",
    "symbol",
    "wait",
    "current_herd",
    "current_segment",
    "current_launch",
    "set_launch",
    "LaunchState",
    "open_launch_region",
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
_CURRENT_LAUNCH = None
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


def current_herd(required=True):
    if _CURRENT_HERD is None and required:
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


class LaunchState:
    """The air.launch region being emitted, and whether it is open yet.

    air.launch, air.segment and air.herd each own an iteration space, and each
    is written on the op that has it. A launch with a grid opens its region
    itself, before the body runs. A launch without one opens nothing until a
    segment needs somewhere to sit -- which keeps a kernel that stages nothing
    at the plain `func` + `air.herd` shape its hand-written predecessors have.
    """

    __slots__ = ("ctx", "opened", "coords", "leaves", "reentry")

    def __init__(self, ctx):
        self.ctx = ctx
        self.opened = False
        # Where to put work traced *after* the region closed, for a gridless
        # launch. Such a launch is opened lazily around the first thing that
        # needs it, so everything after that -- a second segment, a channel get
        # draining the result -- would otherwise be emitted at func scope, as a
        # sibling of the launch instead of a child.
        # (block, tensor block-argument values); None while the region is open,
        # and for a launch with a grid, which stays open for the whole body.
        self.reentry = None
        # This launch's coordinates, as expressions handed to the body, plus the
        # leaves behind them. A nested IsolatedFromAbove region rebinds each
        # leaf's .value to its own block argument.
        self.coords = []
        self.leaves = []


def set_launch(state):
    global _CURRENT_LAUNCH
    previous, _CURRENT_LAUNCH = _CURRENT_LAUNCH, state
    return previous


def current_launch():
    if _CURRENT_LAUNCH is None:
        raise RuntimeError(
            "this operation must be used inside a launch body (@launch.body)"
        )
    return _CURRENT_LAUNCH


def open_launch_region(launch, tensors, counts, body):
    """Emit air.launch with ``counts`` as its sizes and run ``body`` inside.

    Block arguments are ids + sizes + operands; air.launch is 2-D, so the
    operands start at index 4.
    """
    from air.dialects.air import launch as launch_region
    from air.ir import InsertionPoint

    closed = []

    @launch_region(sizes=counts, operands=[t.value for t in tensors])
    def launch_body(*largs):
        launch.opened = True
        launch.leaves = [
            Leaf(largs[axis], f"l{axis}") for axis in range(len(launch.ctx.grid))
        ]
        launch.coords = [IndexExpr({leaf: 1}, 0) for leaf in launch.leaves]
        saved = [t.value for t in tensors]
        for t, v in zip(tensors, largs[4:]):
            t.value = v
        # Captured here, published only once this region closes: while it is
        # open the ordinary insertion point is already right, and the block has
        # no terminator to insert ahead of yet.
        closed.append((InsertionPoint.current.block, list(largs[4:])))
        try:
            body()
        finally:
            for t, v in zip(tensors, saved):
                t.value = v

    if closed:
        launch.reentry = closed[0]


def in_launch_body(emit):
    """Run ``emit()`` inside air.launch's region, opening or re-entering it.

    Everything that has to live inside the launch goes through here: a segment,
    and an L3 channel endpoint, which needs the shim DMA allocation the launch
    brings. Three cases, and the third is the one that used to be missing:

    * the launch carries a grid, so its region is open for the whole body and
      the current insertion point is already right;
    * it is gridless and nothing has needed it yet, so open it now, lazily --
      which is what keeps a kernel that stages nothing at the plain ``func`` +
      ``air.herd`` shape its hand-written predecessors have;
    * it is gridless and was already opened *and closed* around something
      earlier, so step back into its block, ahead of the terminator, with the
      tensors rebound to its block arguments.
    """
    launch = current_launch()
    if not launch.opened:
        # One point, 2-D as air.launch requires.
        open_launch_region(launch, active_trace().tensors, [1, 1], emit)
        return
    if launch.reentry is None:
        emit()
        return

    from air.ir import InsertionPoint

    block, arg_values = launch.reentry
    terminator = block.operations[len(block.operations) - 1]
    tensors = active_trace().tensors
    saved = [t.value for t in tensors]
    for t, v in zip(tensors, arg_values):
        t.value = v
    try:
        with InsertionPoint(terminator):
            emit()
    finally:
        for t, v in zip(tensors, saved):
            t.value = v


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

    # Comparisons do NOT fold, unlike the arithmetic below. The two have
    # opposite requirements and both are honest: a symbol used as a tile size
    # has to be a Python int, because it sizes a memref; a symbol used as a
    # branch condition has to stay a value, because folding it would delete the
    # scf.if. Which is the whole point of a symbol -- the prototype defines one
    # as "known at dispatch time rather than compile time", and v1 resolving it
    # early is an implementation detail, not licence to compile the branch away.
    # `Condition.materialize()` emits arith.constant + arith.cmpi even when both
    # sides are constant, and air-to-aie's SpecializeScfIfPattern folds it once
    # the herd is unrolled -- so the branch costs nothing and still exists in
    # the IR the compiler is handed.
    def _compare(self, other, predicate, symbol):
        from ._index import coerce_index

        return coerce_index(self)._compare(other, predicate, symbol)

    def __eq__(self, o):
        return self._compare(o, "eq", "==")

    def __ne__(self, o):
        return self._compare(o, "ne", "!=")

    def __lt__(self, o):
        return self._compare(o, "slt", "<")

    def __le__(self, o):
        return self._compare(o, "sle", "<=")

    def __gt__(self, o):
        return self._compare(o, "sgt", ">")

    def __ge__(self, o):
        return self._compare(o, "sge", ">=")

    # Defining __eq__ would otherwise drop the default hash, and a Symbol is
    # kept in PENDING_SYMBOLS and looked up by identity.
    __hash__ = object.__hash__

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


def _block_position(block, op):
    for i, other in enumerate(block.operations):
        if other.operation == op:
            return i
    raise AssertionError("operation is not in the block it reports as its parent")


def _lift_into(op, ops_in_home):
    """The index in ``ops_in_home`` of ``op``'s ancestor there, or None.

    An index rather than the op itself because the caller is comparing
    positions -- it wants the latest use, and ``ops_in_home`` is in block
    order, so the larger index wins.

    Walks outward by parent rather than by block, which matters: ``.block`` on
    a top-level operation does not return None, it trips an assertion inside
    MLIR and takes the process with it. Anything that walks past the op it was
    looking for therefore has to notice by running out of parents, not by
    asking each one where it lives.
    """
    while op is not None:
        for i, other in enumerate(ops_in_home):
            if other == op:
                return i
        try:
            op = op.parent
        except (ValueError, IndexError):
            return None
    return None


def _last_use_anchor(value, ignore=None):
    """The op in ``value``'s own block after which ``value`` is no longer read.

    A use nested inside a loop keeps the buffer live until the loop as a whole
    is finished, so each use is walked out to the ancestor that sits directly
    in the block the alloc was emitted into. ``ignore`` drops one op from the
    scan, so that a dealloc can ask where the last *other* use was.

    A use that is *not* under that block has no such ancestor, and this is
    where that has to be caught. The buffer would be read where its allocation
    does not reach -- a loop or a branch arm it was declared in has closed --
    and the walk would otherwise run off the top of the IR and abort. Both
    known instances were real: an L2 buffer allocated in a staging loop and
    handed to a later herd, and an L1 buffer allocated in one arm of an
    ops.branch and read after the branch.
    """
    home = value.owner.operation.block
    ops_in_home = [o.operation for o in home.operations]
    anchor = value.owner.operation
    best = _block_position(home, anchor)
    for use in value.uses:
        op = use.owner.operation
        if ignore is not None and op == ignore:
            continue
        position = _lift_into(op, ops_in_home)
        if position is None:
            raise RuntimeError(
                f"a buffer is used outside the region it was allocated in: the "
                f"allocation sits in a {_region_name(home)} that has already "
                f"closed by the time {op.name} uses it, so it does not reach "
                "that far. Allocate it in the scope where it is read -- above "
                "the loop or the ops.branch rather than inside one."
            )
        if position > best:
            anchor, best = ops_in_home[position], position
    return home, anchor


def _region_name(block):
    """Name the construct owning ``block``, for a diagnostic."""
    owner = block.owner
    return owner.name if owner is not None else "region"


def prune_unused_operands(op):
    """Drop the kernel operands ``op``'s body turned out not to touch.

    ``air.segment`` and ``air.herd`` are IsolatedFromAbove, so anything the
    body reaches has to arrive as an operand -- and the operand list is fixed
    when the op is created, which is *before* the body runs. The tracer
    therefore cannot know what the body will touch at the one moment it has to
    decide, and over-approximates: every tensor, every enclosing coordinate,
    every L2 buffer still in scope. This is the same bind any builder of these
    ops is in, which is why ``canonicalizeHierarchyOpArgs`` exists in
    AIRDialect.cpp to clean up after it.

    Leaving the cleanup to that canonicalization is too late. It runs at PASS
    014, and ``air-dependency`` runs at 008 -- so the dead operands have
    already been read as data dependencies by the time they are removed, and
    the async edges they produced outlive them. In
    ``flash_attention/dataflow_based`` that serialises three herds the
    predecessor leaves independent; a loop body with three impure ops instead
    of one fails ``hasNImpureOps(body, 1)`` in
    ``HoistAIRHerdsToSharedRegionPattern``, so the herds never hoist out of
    the loop, never merge, and their L1 accumulators are still herd operands
    when ``air-verify-hierarchy-locality`` rejects them at 043.

    So the over-approximation is undone here, while the body is fresh and its
    uses are visible, rather than left for a pass that runs after the damage.
    The op is rebuilt because an operand list is not resizable in place; the
    traced block moves across unchanged.

    ``air.launch`` is deliberately not pruned: its operands are the kernel
    interface, and ``LaunchState.reentry`` holds its block arguments so a
    later segment can step back into the region.
    """
    from air.ir import DenseI32ArrayAttr, InsertionPoint, Operation

    op = op.operation if hasattr(op, "operation") else op
    segments = op.attributes["operandSegmentSizes"]
    n_async, n_sizes = segments[0], segments[1]
    # Block arguments are ids + sizes + operands, so the operands start after
    # twice the number of size operands.
    ctrl = 2 * n_sizes
    block = op.regions[0].blocks[0]
    dead = [
        i
        for i in range(ctrl, len(block.arguments))
        if not list(block.arguments[i].uses)
    ]
    if not dead:
        return op

    kept = [j for j in range(len(block.arguments) - ctrl) if j + ctrl not in dead]
    base = n_async + n_sizes
    operands = list(op.operands[:base]) + [op.operands[base + j] for j in kept]
    attributes = {name: op.attributes[name] for name in op.attributes}
    attributes["operandSegmentSizes"] = DenseI32ArrayAttr.get(
        [n_async, n_sizes, len(kept)]
    )
    new = Operation.create(
        op.name,
        results=[r.type for r in op.results],
        operands=operands,
        attributes=attributes,
        regions=1,
        ip=InsertionPoint(op),
        loc=op.location,
    )
    # Highest index first: erasing shifts everything after it down.
    for i in reversed(dead):
        block.erase_argument(i)
    block.append_to(new.regions[0])
    for old_result, new_result in zip(op.results, new.results):
        old_result.replace_all_uses_with(new_result)
    op.erase()
    return new


def free_buffers(buffers):
    """End the life of every buffer that ``air.dealloc`` did not already end.

    A buffer's lifetime is a logical property the tracer can see: it ends at
    the last op that reads or writes it. Emitting every dealloc at the end of
    the body instead overstates that lifetime, and the overstatement is not
    free -- it tells the compiler a value is still needed when it is not, so
    the schedule it builds has to keep the buffer available for longer than the
    program requires. In a design where each worker hands its tile to a
    neighbour, that is enough to serialise the hand-off into a cycle and hang
    the herd (``channel_examples/worker_to_worker``).

    Inference is the default rather than the only option: ``air.dealloc``
    exists for a program that wants to say where a buffer dies. This also
    checks those explicit calls, which is the one thing inference cannot get
    wrong and a hand-placed release can.

    Every anchor is computed before any dealloc is emitted, so the deallocs
    this adds cannot themselves count as uses.
    """
    from air.dialects.memref import DeallocOp
    from air.ir import InsertionPoint

    pending = []
    for buf in buffers:
        if buf.released is not None:
            _check_released(buf)
        else:
            pending.append((buf, *_last_use_anchor(buf.value)))

    for buf, home, anchor in pending:
        successor, seen = None, False
        for op in home.operations:
            if seen:
                successor = op.operation
                break
            seen = op.operation == anchor
        # No successor means the anchor is currently last in the block; the
        # region's terminator has not been appended yet, so the end is right.
        with InsertionPoint(successor) if successor else InsertionPoint(home):
            DeallocOp(buf.value)


def _check_released(buf):
    """Reject an air.dealloc that a later use contradicts."""
    release = buf.released
    home, anchor = _last_use_anchor(buf.value, ignore=release)
    if release.block != home:
        raise ValueError(
            "air.dealloc released a buffer from a different region than the "
            "one it was allocated in; release it in the body that allocated "
            "it, so the dealloc is dominated by its alloc"
        )
    if _block_position(home, anchor) > _block_position(home, release):
        raise ValueError(
            f"air.dealloc released this buffer before its last use "
            f"({anchor.name}): a released buffer cannot be read or written "
            "again. Move the air.dealloc after the last op that touches it, "
            "or drop it and let air.api place it there."
        )


def dealloc(buffer):
    """End a buffer's life here, rather than letting the tracer infer it.

    The counterpart to :func:`alloc`. Without it the tracer places the release
    after the last use it observes, which is what a program that does not care
    wants; call this when the program has something to say about the point
    itself -- and on a target with a real allocator behind ``memref.dealloc``,
    that point is a free rather than a scheduling hint.

    Using the buffer afterwards is an error, reported when the body is finished
    rather than at this call: the tracer has to see the rest of the body before
    it can know whether a later use exists.
    """
    from air.dialects.memref import DeallocOp

    if not isinstance(buffer, Buffer):
        raise TypeError(
            f"air.dealloc takes a buffer from air.alloc, got "
            f"{type(buffer).__name__}"
        )
    if buffer.released is not None:
        raise ValueError("air.dealloc: this buffer has already been released")
    buffer.released = DeallocOp(buffer.value).operation


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

    _what = "air.segment"

    def __init__(self, grid=None, name=None):
        # A grid here is this segment's *own* iteration space -- air.segment's
        # `sizes`, which the dialect prints as `unroll(...)`. air.launch,
        # air.segment and air.herd each carry one and they are not the same
        # thing: a launch point is a repetition of the whole segment, while a
        # segment point is a spatial copy of the segment body, which air-to-aie
        # lays out across columns or devices. Writing a grid here used to set
        # the *launch's* sizes, which conflated the two and made air.segment's
        # own iteration space unreachable.
        self.dims = parse_grid(grid) if grid is not None else ()
        if len(self.dims) > 2:
            raise NotImplementedError(
                f"an air.segment iteration space is 1-D or 2-D; got "
                f"{len(self.dims)}-D"
            )
        self.grid = tuple(d.count for d in self.dims)
        self.tile_sizes = tuple(d.step for d in self.dims)
        self.name = name or "segment_0"
        self._buffers = []
        self._registered = False
        # This segment's own coordinates, as Leaf objects, bound while the body
        # is traced. A nested herd needs them threaded in as operands, because
        # air.herd is IsolatedFromAbove -- the same reason the segment itself
        # takes the launch's coordinates as operands. Empty for a gridless
        # segment, which is what keeps every existing example's operand list
        # byte-identical.
        self.leaves = []
        # The segment body's own block, bound while the body is traced. It is
        # the top of the scope a nested herd is emitted into, and so the place
        # a walk up the block chain has to stop -- see _staged_in_scope.
        self._entry_block = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        # A body that was never registered is the one way to leave this block
        # having emitted nothing at all -- the `with` is pure bookkeeping and
        # every op comes from the decorator. Silently emitting nothing is the
        # worst available outcome: the enclosing ops vanish, the kernel still
        # builds, and on a small grid it still runs and still passes, so
        # neither a hardware test nor an op-count diff notices.
        #
        # Not raised while another exception is propagating: the body itself
        # failing is far more interesting than the body being absent, and
        # replacing it here would bury the real error.
        if exc_type is None and not self._registered:
            raise RuntimeError(
                f"{self._what} was opened but its body was never registered, so "
                "nothing was emitted for this scope and the ops inside it were "
                "traced into the enclosing one. The body is a function decorated "
                "inside the `with`, which means the scope needs a name to "
                f"decorate:\n\n    with {self._what}(...) as scope:\n"
                "        @scope.body\n        def _():\n            ...\n\n"
                "`as scope` is easy to leave off, and without it there is no "
                "variable to hang the decorator on."
            )
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

    def per_core(self):
        """L1 allocated here, and every core gets its own copy of the whole thing.

        The sibling of :meth:`shared`, and the distinction is what the cores
        see. A ``shared()`` buffer is one allocation that the cores divide
        between them -- it carries a leading dimension per herd axis and each
        core addresses its own slab. A ``per_core()`` buffer is not divided:
        every core gets the shape as written, privately, and no core can see
        another's.

        What the two have in common is the lifetime, which is the reason to
        allocate at segment scope at all. A buffer in a herd body dies when that
        body ends, so state that has to survive from one herd to the next cannot
        live there. flash_attention/dataflow_based carries a running maximum, a
        running sum and a running output across three separate herds this way.

        Nothing is sliced, so nothing is subscripted by tile coordinate, and the
        buffer reaches a kernel whole. It is charged against the 64 KB core
        budget at full size, because that is what each core spends on it.
        """
        return Scope("per_core", self)

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
        trace = active_trace()
        n_expected = len(self.dims)
        if _positional_arity(fn) != n_expected:
            raise TypeError(
                f"segment body takes {_positional_arity(fn)} coordinate "
                f"argument(s) but the segment iteration space is "
                f"{n_expected}-D"
                + (
                    "; a segment with no grid is a single instance and its body "
                    "takes no arguments. Outer tiling belongs on air.launch, "
                    "whose coordinates arrive in the launch body"
                    if n_expected == 0
                    else ""
                )
            )

        # A gridless launch still has to exist for a segment to sit in:
        # air-insert-launch-around-herd only wraps a *bare* herd, and skips one
        # already inside a segment, so a segment with no launch above it
        # compiles and silently computes zeros.
        in_launch_body(lambda: self._emit_segment(fn, trace))

    def _emit_segment(self, fn, trace):
        """Emit air.segment inside the already-open air.launch."""
        global _CURRENT_SEGMENT

        from air.dialects.air import segment as segment_region

        tensors = trace.tensors
        segment_self = self
        launch = current_launch()

        # air.segment is IsolatedFromAbove, so a launch coordinate used in this
        # body cannot simply be referenced -- it is threaded in as an operand
        # ahead of the tensors, and rebound to the block argument inside.
        outer_leaves = launch.leaves
        operands = [leaf.value for leaf in outer_leaves] + [t.value for t in tensors]
        sizes = list(self.grid) + [1] * (2 - len(self.grid)) if self.grid else []

        @segment_region(name=self.name, operands=operands, sizes=sizes)
        def segment_body(*args):
            global _CURRENT_SEGMENT

            # Block arguments are ids + sizes + operands, so a segment with an
            # iteration space of its own offsets the operands by 2 * its rank.
            # That rank is the *padded* one -- air.segment's sizes are 2-D like
            # air.launch's -- but the body only ever sees as many coordinates as
            # it declared, so a 1-D grid yields one. This is the same split the
            # launch makes; keeping them symmetric is the point of #1868.
            n = len(sizes)
            declared = len(segment_self.dims)
            segment_self.leaves = [
                Leaf(v, f"u{axis}") for axis, v in enumerate(args[:declared])
            ]
            coords = [IndexExpr({leaf: 1}, 0) for leaf in segment_self.leaves]
            bound = args[2 * n :]
            saved_outer = [leaf.value for leaf in outer_leaves]
            for leaf, v in zip(outer_leaves, bound[: len(outer_leaves)]):
                leaf.value = v
            saved = [t.value for t in tensors]
            for t, v in zip(tensors, bound[len(outer_leaves) :]):
                t.value = v
            previous, _CURRENT_SEGMENT = _CURRENT_SEGMENT, segment_self
            segment_self._entry_block = args[0].owner
            try:
                fn(*coords)
                free_buffers(segment_self._buffers)
            finally:
                segment_self._buffers.clear()
                for t, v in zip(tensors, saved):
                    t.value = v
                for leaf, v in zip(outer_leaves, saved_outer):
                    leaf.value = v
                _CURRENT_SEGMENT = previous

        # The body has run, so what it touched is finally knowable. Herds
        # nested inside pruned themselves as they closed, which is what makes
        # an operand only they had a candidate here too.
        prune_unused_operands(segment_body)


def segment(grid=None, name=None):
    """A device segment with L2 scope; nest herds inside its body."""
    return SegmentContext(grid=grid, name=name)


# ---------------------------------------------------------------------------
# Herd
# ---------------------------------------------------------------------------


def _needs(obj, kernel):
    """Phrase one claim on a herd's single link_with slot, for a conflict."""
    return f"calls {kernel} from {obj!r}" if kernel else f"declares link_with={obj!r}"


def _staged_in_scope(segment):
    """The segment's L2 buffers that are still live where a herd is going.

    ``air.herd`` is IsolatedFromAbove, so an L2 buffer the body reaches has to
    be passed in as an operand, and the tracer cannot know which ones the body
    touches until it has run it -- so it passes every one it can, and drops the
    unused ones afterwards (``prune_unused_operands``).

    "Can" is the point. A buffer allocated inside an ``air.sequential`` at
    segment scope dies with that loop: by the time a *later* herd is emitted,
    its ``memref.alloc`` sits in a region that has already been closed, and
    naming it as an operand is not merely wasteful but ill-formed -- it does not
    dominate the use. The staging loop in the int4 GEMV is exactly this shape:
    an L2 tile allocated per trip, filled from L3 and forwarded to L1 over a
    channel, with a herd beside it that never touches the buffer at all.

    Left unfiltered that operand reaches ``free_buffers``, which walks the
    buffer's uses out to the block its alloc lives in, finds an ``air.herd``
    that is not under that block, and walks off the top of the IR -- aborting
    the process inside MLIR rather than raising. So the filter is what the
    surrounding comment always claimed: only the ones in scope here.
    """
    if segment is None:
        return []
    from air.ir import InsertionPoint

    top = segment._entry_block
    if top is None:
        return list(segment._buffers)
    # The blocks whose values are visible here: this one and its ancestors, up
    # to the segment body. The walk stops there rather than running to the
    # module, because `block.owner.operation.block` on the top-level block
    # aborts the process instead of returning None.
    # Ancestry, not dominance -- and those are only the same thing because the
    # tracer appends. A buffer is in segment._buffers here only if its alloc has
    # already been emitted, and a herd is created at the end of its block or
    # ahead of the terminator, so anything sitting in an ancestor block also
    # precedes this point. Checked against an exact dominance test over all 72
    # converted examples: they agree on every buffer. A construct that stepped
    # back into a closed region to allocate would break that, and this check
    # would have to compare positions (Operation.is_before_in_block) rather than
    # just block membership.
    visible, block = set(), InsertionPoint.current.block
    while True:
        visible.add(block)
        if block == top:
            break
        owner = block.owner
        if owner is None:
            break
        block = owner.operation.block
    return [b for b in segment._buffers if b.value.owner.operation.block in visible]


class HerdContext:
    """A herd of compute cores over a (possibly strip-mined) tile grid."""

    _what = "air.herd"

    def __init__(self, iterable, name=None, shape=None, target=None, link_with=None):
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
        # Object files this herd links against: either called through
        # air.extern, or named by link_with= for a lowering that emits the call
        # itself (see require_object).
        self._objects = {}
        # The core's position in the herd, bound while the body is traced.
        self._coords = []
        # Reduction scratch buffers, one per (dtype, lanes), allocated at the
        # top of the body and reused by every reduction under it. See scratch().
        self._scratch = {}
        self._entry_block = None
        if link_with is not None:
            if not isinstance(link_with, str) or not link_with:
                raise TypeError(
                    "air.herd(link_with=...) takes the name of a compiled "
                    'object file, e.g. link_with="extern_func.o"'
                )
            self.require_object(link_with, None)

    def require_object(self, obj, kernel):
        """Record that this herd links against ``obj``.

        ``kernel`` names the symbol an air.extern call reaches for, or is None
        when ``air.herd(link_with=...)`` declared the dependency directly. The
        second form exists because not every reference to an object file is a
        call the DSL emits: ``ops.exp`` on bf16 becomes ``math.exp``, and it is
        the AIE lowering, several passes later, that turns that into a call to
        ``getExpBf16`` in the example's ``extern_func.o``. Nothing at trace time
        writes a func.call, so air.extern -- which exists to emit one -- cannot
        express it, but link_with is needed all the same.

        aircc links one object per herd -- link_with is a single string -- so a
        body that reaches into two of them cannot be built.
        """
        if self._objects and obj not in self._objects:
            other, other_kernel = next(iter(self._objects.items()))
            raise ValueError(
                f"herd '{self.name}' {_needs(obj, kernel)} and "
                f"{_needs(other, other_kernel)}, but a herd links against a "
                "single object file. Compile both kernels into one object, or "
                "put them in separate herds."
            )
        # An air.extern call names the symbol it wants; keep that over a bare
        # link_with= declaration of the same file, since it makes a later conflict
        # report the more specific of the two.
        if kernel is not None or obj not in self._objects:
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

    def __exit__(self, exc_type, exc_val, tb):
        # A body that was never registered is the one way to leave this block
        # having emitted nothing at all -- the `with` is pure bookkeeping and
        # every op comes from the decorator. Silently emitting nothing is the
        # worst available outcome: the enclosing ops vanish, the kernel still
        # builds, and on a small grid it still runs and still passes, so
        # neither a hardware test nor an op-count diff notices.
        #
        # Not raised while another exception is propagating: the body itself
        # failing is far more interesting than the body being absent, and
        # replacing it here would bury the real error.
        if exc_type is None and not self._registered:
            raise RuntimeError(
                f"{self._what} was opened but its body was never registered, so "
                "nothing was emitted for this scope and the ops inside it were "
                "traced into the enclosing one. The body is a function decorated "
                "inside the `with`, which means the scope needs a name to "
                f"decorate:\n\n    with {self._what}(...) as scope:\n"
                "        @scope.body\n        def _():\n            ...\n\n"
                "`as scope` is easy to leave off, and without it there is no "
                "variable to hang the decorator on."
            )
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

    def scratch(self, dtype, lanes):
        """An L1 vector the emitter accumulates a long reduction through.

        A reduction longer than one vector has to accumulate across steps, and
        the accumulator cannot be a loop-carried SSA vector: LLVM splits one
        into sub-512-bit pieces the AIE2 backend will not legalize, which is the
        failure ``ops.dot`` documents. Every hand-written kernel this models --
        rms_norm, layer_norm, weighted_rms_norm -- round-trips its partial sums
        through a small L1 buffer instead, so that is what this provides.

        Allocated **at the top of the herd body**, not where the reduction is
        written. A reduction normally sits inside a row loop, and allocating
        there would emit one alloc per trip and leave the dealloc outside the
        region that dominates it. One buffer per (dtype, lanes) serves every
        reduction in the body, which is what the predecessors do by hand.
        """
        from air.ir import InsertionPoint

        key = (dtype, int(lanes))
        if key in self._scratch:
            return self._scratch[key]
        if self._entry_block is None:
            raise RuntimeError(
                "a reduction scratch buffer was requested outside a herd body"
            )
        with InsertionPoint.at_block_begin(self._entry_block):
            buf = alloc([int(lanes)], dtype, scope=self.private(), _hoisted=True)
        # Kept out of _buffers, which is freed at the end of every strip-mined
        # run. A scratch buffer outlives those: it is allocated once above the
        # strip loop and reused by every trip, so a dealloc inside the loop
        # would free it after the first trip and leave the rest reading a dead
        # buffer. It is freed once, after the strip nest, by _free_scratch.
        self._buffers.remove(buf)
        self._scratch[key] = buf
        return buf

    def _free_scratch(self):
        """Dealloc the reduction scratch, once, after the strip-mined runs."""
        free_buffers(list(self._scratch.values()))
        self._scratch.clear()

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
        from air.dialects.scf import for_ as range_, yield_

        from ._loop import aborted_regions

        aborted_before = len(aborted_regions())

        tensors = trace.tensors
        # air.herd is IsolatedFromAbove, so an L2 buffer allocated in the
        # enclosing segment has to be passed in explicitly. Every live one is
        # passed, referenced or not -- the same policy already applied to
        # tensors, and the tracer cannot know what the body will touch until it
        # has run it.
        enclosing = current_segment(required=False)
        staged = _staged_in_scope(enclosing)
        # The herd is the first thing that knows how many of a shared buffer's
        # leading dimensions are cores, so it is where their L1 charge is gated.
        _charge_shared_l1(enclosing, len(self.grid), self.name)
        # Every coordinate in scope, threaded in for the same reason the L2
        # buffers are: air.herd is IsolatedFromAbove, so a body that offsets a
        # transfer by an enclosing coordinate cannot simply reference it.
        #
        # Both the launch's and the enclosing segment's, and in that order --
        # outermost first, matching how air.segment already receives the
        # launch's. Passing only the segment's used to be enough by accident:
        # the launch's coordinates were reachable from segment scope, so an
        # example whose outer tiling stayed there worked, and one that carried
        # a launch coordinate *into* the herd emitted an affine.apply on a value
        # defined outside the region and failed verification.
        #
        # Every live one is passed, referenced or not, which is the policy
        # already applied to tensors and to L2 buffers: the tracer cannot know
        # what the body will touch until it has run it. The ones it turns out
        # not to touch are dropped once it has -- see prune_unused_operands,
        # and note that leaving them in place is not merely untidy: they reach
        # air-dependency as data dependencies and serialise herds that have
        # nothing to do with each other.
        outer = list(current_launch().leaves)
        outer += list(enclosing.leaves) if enclosing is not None else []
        operands = (
            [leaf.value for leaf in outer]
            + [t.value for t in tensors]
            + [b.value for b in staged]
        )
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
            saved_outer = [leaf.value for leaf in outer]
            n_outer = len(outer)
            for leaf, v in zip(outer, inner[:n_outer]):
                leaf.value = v
            for t, v in zip(tensors, inner[n_outer : n_outer + len(tensors)]):
                t.value = v
            for b, v in zip(staged, inner[n_outer + len(tensors) :]):
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
            # Where a reduction's scratch accumulator is allocated, whatever
            # loop or branch the reduction itself sits in. See scratch().
            herd_self._entry_block = args[0].owner
            herd_self._scratch.clear()

            def run(strip_ivs):
                tile_ids = []
                for axis, phys in enumerate(phys_coords):
                    tile = IndexExpr.leaf(phys, f"t{axis}") * herd_self.repeats[axis]
                    if herd_self.repeats[axis] > 1:
                        tile = tile + IndexExpr.leaf(strip_ivs[axis], f"i{axis}")
                    tile_ids.append(tile)
                fn(*tile_ids)
                # Each dealloc lands in the block its alloc was emitted into,
                # which under strip-mining is the innermost scf.for body: a
                # dealloc hoisted above that would not be dominated by its
                # alloc.
                free_buffers(herd_self._buffers)
                herd_self._buffers.clear()

            try:
                run_strip_mined(run, herd_self.repeats, range_, yield_)
                # After the strip nest, not inside it: one alloc above the loop
                # needs one dealloc below it.
                herd_self._free_scratch()
            finally:
                herd_self._buffers.clear()
                herd_self._scratch.clear()
                for t, v in zip(tensors, saved):
                    t.value = v
                for b, v in zip(staged, saved_staged):
                    b.value = v
                for leaf, v in zip(outer, saved_outer):
                    leaf.value = v
                _CURRENT_HERD = previous

        # aircc compiles the object named here alongside the herd's cores.
        if herd_self._objects:
            from air.ir import StringAttr

            herd_body.attributes["link_with"] = StringAttr.get(
                next(iter(herd_self._objects))
            )

        # After link_with, because pruning rebuilds the op and carries its
        # attributes across.
        prune_unused_operands(herd_body)

        aborted = aborted_regions()[aborted_before:]
        if aborted:
            # Name the construct that was abandoned. ops.branch shares the
            # region bookkeeping with air.sequential, and reporting a truncated
            # branch as "left a loop early" sends the reader to the wrong line.
            loops = [a for a in aborted if a in ("air.sequential", "air.parallel")]
            if loops:
                raise RuntimeError(
                    f"a body left an {loops[0]} loop early (break, return, or a "
                    "swallowed exception). An air.sequential body is traced once and "
                    "stands for every trip, so an early exit does not shorten the "
                    "loop -- it truncates the body of all of them, and the kernel "
                    "computes a partial result. Restructure the loop bounds instead."
                )
            raise RuntimeError(
                "a body left an ops.branch region early (break, return, or a "
                "swallowed exception). The region is emitted either way, so the "
                "ops written after the exit are simply missing from it, and the "
                "cores that take that branch compute a partial result. Let the "
                "`with` block run to its end."
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


def herd(iterable, name=None, shape=None, target=None, link_with=None):
    """A herd of cores over ``iterable``, strip-mined onto the physical array.

    ``link_with=`` names the object file to stamp on the herd, for a lowering
    that emits its own call into it -- ``ops.exp`` and ``ops.rsqrt`` on bf16 are
    the cases in this tree. It is spelled as the attribute it sets, and as the
    raw ``@herd`` decorator already spells it. For a call the DSL emits itself,
    use air.extern, which sets link_with from the kernel's own declaration.
    """
    return HerdContext(
        iterable, name=name, shape=shape, target=target, link_with=link_with
    )


def _require_allocatable(dtype, what):
    """Refuse an element type no buffer can hold. Only i4 is such a type."""
    if getattr(dtype, "allocatable", True):
        return
    raise TypeError(
        f"{what} cannot have element type {dtype}: a DMA moves whole bytes and "
        f"the L1 budget is counted in them, so there is no buffer of half-bytes "
        f"to allocate. {dtype} names what packed *bytes* contain -- read them "
        f"as a byte buffer and reinterpret with air.api.ops.bitcast"
    )


def tensor(shape, dtype, name=None):
    """Declare a host-visible L3 array; becomes a kernel argument."""
    _require_allocatable(dtype, "air.tensor")
    t = Tensor(shape, dtype, name=name or infer_name(f"t{len(PENDING_TENSORS)}"))
    PENDING_TENSORS.append(t)
    return t


def alloc(shape, dtype, scope=None, vector=None, _hoisted=False):
    """Allocate a tile: L1 in a herd body, or L2 in a segment body."""
    _require_allocatable(dtype, "air.alloc")
    from air.ir import IntegerAttr, MemRefType
    from air.dialects.air import MemorySpace
    from air.dialects.memref import AllocOp
    from air.extras import types as T

    if scope is None:
        raise ValueError(
            "air.alloc requires scope=<herd>.private() (L1) or "
            "scope=<segment>.private() (L2)"
        )
    if not isinstance(scope, Scope) or scope.kind not in (
        "private",
        "shared",
        "per_core",
    ):
        raise NotImplementedError(
            f"air.api can only allocate in a private, shared or per_core scope, "
            f"got {scope!r}"
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
        if scope.kind in ("shared", "per_core"):
            space, memory_space, capacity = "L1", MemorySpace.L1, L1_BYTES
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
    # _hoisted: the caller has already moved the insertion point to the top of
    # the herd body, so the alloc does not land in the loop the caller is
    # standing in and the dominance argument below does not apply. Only
    # HerdContext.scratch sets it, and it is not part of the public signature.
    #
    # A loop body used to be refused here too, on the grounds that the herd
    # frees its buffers after the loop has closed. That was not so:
    # free_buffers anchors each dealloc in the block the *alloc* lives in, so a
    # buffer allocated in a loop is already released inside that loop, which is
    # both dominated and what the hand-written kernels write. The refusal cost
    # real IR -- hoisting the int4 GEMV's per-trip L1 tiles above their loop
    # gives the pipeline one buffer to rotate instead of a fresh one per trip,
    # which is the ping-pong the kernel is built around.
    #
    # A branch arm allocates on the same terms. The arm is a region like the
    # loop body is, and placement treats it the same way: the dealloc lands in
    # the arm beside its alloc, so a tile that only one kind of core needs is
    # written where it is needed rather than hoisted above the branch and paid
    # for by every core. flash_attention/dataflow_based does this twelve times,
    # once per scratch tile in each arm of its cascade-stage select.
    #
    # What used to make this unsafe was not the allocation but the diagnosis: a
    # buffer read *after* the arm closed walked off the top of the IR and
    # aborted the process. _last_use_anchor now reports that as an error naming
    # the region, so the bad case is caught and the good case is allowed.
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
    if scope.kind in ("shared", "per_core"):
        # A herd-shared buffer is declared once with one leading dimension per
        # herd axis, and each core addresses exactly one slab of it. Charging
        # the whole thing against one core's 64 KB would reject configurations
        # that fit comfortably -- a 4x4 herd of 16 KB slabs is 256 KB in total
        # and 16 KB per core. How many leading dimensions are the herd is the
        # herd's business, and no herd has been entered yet: this is segment
        # scope. So the charge is deferred to the herd -- see _charge_shared_l1.
        #
        # A per_core buffer has no such ambiguity -- every core spends the whole
        # of it -- but it is deferred alongside, because the two kinds compete
        # for the same 64 KB and only a combined total means anything.
        pass
    else:
        # A segment holds L2 memtile buffers and herd-shared L1 buffers at once,
        # so each budget only counts its own space.
        live = (
            sum(_buffer_bytes(b) for b in holder._buffers if b.space == space) + nbytes
        )
        if space == "L1" and isinstance(owner, HerdContext):
            # A core's 64 KB holds its private tiles *and* its slab of whatever
            # the enclosing segment shares across the herd. Those are one
            # budget, not two: charging them separately passes a design that
            # overflows once they are added up, which surfaces later as a
            # placement failure rather than as an air.alloc error.
            enclosing = current_segment(required=False)
            if enclosing is not None:
                nlead = len(owner.grid)
                # A shared buffer is charged by the slab this core owns; a
                # per_core buffer by the whole of it, since every core has one.
                for b in enclosing._buffers:
                    if b.space != "L1":
                        continue
                    kind = getattr(b.scope, "kind", None)
                    if kind == "shared":
                        live += _buffer_bytes(b, nlead)
                    elif kind == "per_core":
                        live += _buffer_bytes(b)
        if space == "L1":
            unit, verb = "a compute tile", "has"
        else:
            cols = DEVICE_COLUMNS[current_target()]
            unit, verb = f"this device's {cols} memtiles", "have"
        if nbytes > capacity:
            raise ValueError(
                f"air.alloc({list(shape)}, {dtype}) needs {nbytes / 1024:.1f} KB "
                f"but {unit} {verb} {capacity / 1024:.0f} KB of {space}; use a "
                "smaller tile"
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
    )
    holder.register_buffer(buf)
    if space == "L1" and scope.kind not in ("shared", "per_core"):
        trace = active_trace()
        trace.l1_peak = max(trace.l1_peak, live)
    return buf


def _buffer_bytes(buf, nlead=0):
    n = buf.dtype.itemsize
    for extent in buf.shape:
        n *= extent
    # A herd-shared buffer is charged per core: its first `nlead` dimensions are
    # the herd, and a core touches one slab.
    for extent in buf.shape[:nlead]:
        n //= int(extent)
    return n


def _charge_shared_l1(segment, nlead, herd_name):
    """Gate the L1 budget for herd-shared buffers, now that the herd is known.

    ``<segment>.shared()`` allocates at segment scope, where nothing yet says
    how many of the buffer's leading dimensions are cores rather than tile. The
    herd is what says, so the check waits until one is entered. Deferring it is
    not a loosening: a shared buffer is unusable without a herd, so every one
    of them reaches this.

    ``per_core()`` buffers are counted here too. Their own charge needs no herd
    -- a core spends the whole of one -- but they share the 64 KB with the
    shared slabs, so only the combined figure is worth checking.
    """
    if segment is None:
        return
    kinds = {}
    for b in segment._buffers:
        kind = getattr(b.scope, "kind", None)
        if b.space == "L1" and kind in ("shared", "per_core"):
            kinds.setdefault(kind, []).append(b)
    shared = kinds.get("shared", [])
    per_core = kinds.get("per_core", [])
    if not shared and not per_core:
        return
    for b in shared:
        if len(b.shape) <= nlead:
            raise ValueError(
                f"air.alloc({list(b.shape)}, {b.dtype}) is herd-shared and the "
                f"herd {herd_name!r} is {nlead}-D, so its first {nlead} "
                "dimension(s) are the cores -- leaving nothing for the tile "
                "itself. Give it one leading dimension per herd axis and at "
                "least one more."
            )
    live = sum(_buffer_bytes(b, nlead) for b in shared) + sum(
        _buffer_bytes(b) for b in per_core
    )
    if live > L1_BYTES:
        detail = ", ".join(
            f"{list(b.shape)} {b.dtype} ({_buffer_bytes(b, lead) / 1024:.1f} KB "
            "per core)"
            for group, lead in ((shared, nlead), (per_core, 0))
            for b in group
        )
        raise ValueError(
            f"L1 budget exceeded: the buffers shared across herd {herd_name!r} "
            f"come to {live / 1024:.1f} KB per core but a compute tile has "
            f"{L1_BYTES / 1024:.0f} KB -- {detail}; use a smaller tile"
        )
    trace = active_trace()
    trace.l1_peak = max(trace.l1_peak, live)


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
