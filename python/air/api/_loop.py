# ./python/air/api/_loop.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A sequential loop inside a herd body.

    for k in air.sequential(0, K, tile_k):
        air.ops.load(a_buf, A[row : row + tm, k : k + tile_k])
        air.ops.dot(a_buf, b_buf, acc=acc)

This emits one ``scf.for`` and hands the body an induction *variable*. The
distinction from a plain Python ``for k in range(...)`` is the whole point:
Python's loop runs at trace time and unrolls, emitting one straight-line copy of
the body per trip against the same L1 buffers. The AIE core that comes out has no
loop in it, so the objectFifo acquire/release pairs that the pipeline would place
inside the loop end up stranded between the unrolled copies, and the kernel
computes with stale operands. A reduction that reuses a buffer across trips --
which is to say, every reduction -- has to reach the compiler as a loop.

Bounds must be Python integers. A dynamic trip count would need the bound to be
an SSA value, and nothing in the DSL produces one; accepting an ``IndexExpr``
here would only defer the failure into the IR.
"""

from ._index import IndexExpr

__all__ = [
    "sequential",
    "parallel",
    "loop_depth",
    "aborted_regions",
    "enter_region",
    "exit_region",
]

# Names of the regions whose body exited early (break / return / an exception
# the caller swallowed). Emission still closes the region so the IR stays well
# formed, but the body is short of what was written and the herd would compute
# a partial result, so this is checked once the herd body is finished.
#
# It records the *construct*, not just a count, because ops.branch shares this
# machinery with air.sequential and the two need different advice: a truncated
# loop is fixed by restructuring the bounds, a truncated branch is not a loop
# problem at all. A single counter reported both as "left a loop early".
_ABORTED = []

# How many air.sequential bodies are currently open. air.alloc consults this: an
# allocation inside a loop is freed by the herd after the loop closes, and the
# dealloc would not be dominated by its alloc.
_DEPTH = 0


def loop_depth():
    return _DEPTH


def enter_body():
    """Start a fresh loop-depth frame, returning the one to restore.

    Depth exists to catch an ``air.alloc`` nested inside a loop *in the same
    body*, where the dealloc would be emitted after the loop and so would not be
    dominated by its alloc. A loop further out is a different matter entirely: a
    herd inside a segment-scope reduction is re-entered once per trip, and its
    body's allocs and deallocs are both inside that trip. Counting the outer loop
    would reject the shape every staged matmul in the tree uses.
    """
    global _DEPTH
    previous, _DEPTH = _DEPTH, 0
    return previous


def exit_body(previous):
    global _DEPTH
    _DEPTH = previous


def aborted_regions():
    return tuple(_ABORTED)


def enter_region():
    """Count one more open nested region (a loop body, or an ``ops.branch``)."""
    global _DEPTH
    _DEPTH += 1


def exit_region(aborted, what="air.sequential"):
    """Close it, recording ``what`` if its body did not run to the end."""
    global _DEPTH
    _DEPTH -= 1
    if aborted:
        _ABORTED.append(what)


def sequential(start, stop=None, step=None, name=None):
    """A sequential ``scf.for`` over ``[start, stop)`` in strides of ``step``.

    Yields the induction variable as an :class:`IndexExpr`, so it composes with
    tile coordinates and tensor slices exactly like a herd coordinate does.
    """
    global _ABORTED, _DEPTH

    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1

    lo, hi, st = (
        _as_bound(v, n) for v, n in ((start, "start"), (stop, "stop"), (step, "step"))
    )
    if st <= 0:
        raise ValueError(f"air.sequential needs a positive step, got {st}")
    if hi < lo:
        raise ValueError(
            f"air.sequential({lo}, {hi}) counts backwards; stop must be >= start"
        )
    if (hi - lo) % st:
        raise ValueError(
            f"air.sequential({lo}, {hi}, {st}) does not tile its extent exactly: "
            f"{(hi - lo) // st} steps of {st} span {((hi - lo) // st) * st}, past "
            f"the extent of {hi - lo}. air.api has no partial trips, so the last "
            "one would run off the end of whatever it indexes."
        )

    from air.dialects.scf import for_ as scf_for, yield_

    for iv in scf_for(lo, hi, st):
        _DEPTH += 1
        completed = False
        try:
            yield IndexExpr.leaf(iv, name or "k")
            completed = True
        finally:
            _DEPTH -= 1
            if not completed:
                _ABORTED.append("air.sequential")
            # Terminate the region either way: leaving a block without a
            # terminator turns a diagnosable "you broke out of a loop" into an
            # MLIR verifier crash somewhere else entirely.
            yield_([])


def parallel(start, stop=None, step=None, name=None):
    """An unordered ``scf.forall`` over ``[start, stop)`` in strides of ``step``.

    The counterpart of :func:`sequential`, and not interchangeable with it. Two
    things follow from the trips being unordered rather than merely unrolled,
    and both are load-bearing where staging fans a memtile buffer out to a row
    of cores:

    * **The induction variable may index a channel bundle.** A bundled put is
      one slot of a spatial fan-out, so the index has to name a destination, not
      a point in time; ``air-place-herds`` refuses a temporal one outright --
      *"channel bundle indices must not be temporal scf.for induction
      variables"*.
    * **The trips share one set of buffer descriptors.** Writing the same
      transfers out as a Python ``for`` unrolls them into that many independent
      DMAs, which is not the same design: herd_dataflow's four per-column L3
      reads fit on npu1 inside this loop and do not fit unrolled -- *"no
      ShimNOCTile has sufficient DMA capacity"*.

    Emitted as ``scf.forall``; the pipeline's ``scf-forall-to-parallel`` turns
    it into the ``scf.parallel`` that the hand-written examples spell directly.
    """
    global _ABORTED, _DEPTH

    if stop is None:
        start, stop = 0, start
    if step is None:
        step = 1

    lo, hi, st = (
        _as_bound(v, n, "parallel")
        for v, n in ((start, "start"), (stop, "stop"), (step, "step"))
    )
    if st <= 0:
        raise ValueError(f"air.parallel needs a positive step, got {st}")
    if hi < lo:
        raise ValueError(
            f"air.parallel({lo}, {hi}) counts backwards; stop must be >= start"
        )
    if (hi - lo) % st:
        raise ValueError(
            f"air.parallel({lo}, {hi}, {st}) does not tile its extent exactly: "
            f"{(hi - lo) // st} steps of {st} span {((hi - lo) // st) * st}, past "
            f"the extent of {hi - lo}. air.api has no partial trips, so the last "
            "one would run off the end of whatever it indexes."
        )

    from air.ir import InsertionPoint
    from air.dialects.scf import ForallOp, InParallelOp

    op = ForallOp(lower_bounds=[lo], upper_bounds=[hi], steps=[st])
    _DEPTH += 1
    completed = False
    try:
        with InsertionPoint(op.body):
            yield IndexExpr.leaf(op.induction_variables[0], name or "p")
            completed = True
            # scf.forall's terminator, emitted inside the body region.
            InParallelOp()
    finally:
        _DEPTH -= 1
        if not completed:
            _ABORTED.append("air.parallel")
            with InsertionPoint(op.body):
                InParallelOp()


def _as_bound(value, which, what="sequential"):
    if isinstance(value, bool) or not hasattr(value, "__index__"):
        raise TypeError(
            f"air.{what}({which}=...) takes a Python integer (or an air.symbol), "
            f"got {type(value).__name__} {value!r}. Loop bounds are resolved at "
            "trace time; a bound computed from a tile coordinate is not supported."
        )
    return int(value)
