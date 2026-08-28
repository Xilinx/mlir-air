# ./python/air/api/_cond.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``ops.branch`` -- a real branch, as against ``ops.select``.

    with ops.branch(tx == 0) as head:
        chan_in.get(recv)
    with head.otherwise():
        cascade.get(recv, indices=[tx - 1])

Read this next to :func:`air.api.ops.select`, because the DSL now has two
conditionals and picking the wrong one is the mistake worth designing against.
They are the two halves of *if-conversion*, the classical transformation that
replaces a branch with a select, and they differ on every axis that matters:

===============  ==========================  ============================
                 ``ops.select(c, a, b)``     ``ops.branch(c)``
===============  ==========================  ============================
what ``c`` is    a buffer comparison,        an index comparison,
                 ``x[:] >= y[:]``            ``tx == 0``
granularity      one decision per element    one decision per instance of
                                             the body -- per core, per trip
what runs        **both** sides, always      **one** side
what it holds    an expression               statements, including effects
                                             -- a channel put, a DMA
lowers to        ``arith.select``            ``scf.if``
it is            an expression you assign    a ``with`` block
===============  ==========================  ============================

The last two rows are the practical test. ``ops.select`` cannot express
"only this core sends on the cascade", because a channel put is not a value and
there is nothing to select between; ``ops.branch`` cannot express "clamp each
element", because a per-element decision is not a branch any core can take.
Handing either one the other's condition raises, and the message names the one
you wanted.

Why it has to be a region rather than a Python ``if``: a herd body is traced
**once**, for all of the herd's cores at once. A coordinate is an SSA value with
no value at trace time, so ``bool()`` on a comparison is not a question the
tracer can answer -- :class:`Condition` refuses it rather than picking one
branch for every core.

What it is *for* is a herd whose cores are not interchangeable: the ends of a
cascade, a corner of a stencil, the one tile that drains a reduction.
``programming_examples/cascade_reduction`` is the worked case -- four cores in a
row, of which the first reads from L3, the last writes to L3, and the middle two
only forward. Nothing about the construct is spatial, though, and it is not
named as if it were: ``air-to-aie`` folds the branch away once the herd is
unrolled and the coordinate is a literal, by exactly the same pattern it applies
to an ``scf.if`` on a dispatch-time parameter (``SpecializeScfIfPattern``, and
``SpecializeAffineIfPattern`` beside it).

The predicate is any of the six comparisons between index expressions -- a tile
coordinate most often, but equally a loop induction variable, an ``air.symbol``
or a Python integer, in any mix. There is no ``and``/``or``: conjunction is
nesting, and Python's keywords could not be overloaded to mean it anyway
(``and`` calls ``bool()``, which is exactly what has to fail here).

Both regions are created up front, so a branch with no ``otherwise`` leaves an
``scf.if`` with an else holding only its terminator. That is not a difference in
the compiled design: MLIR's ``RemoveEmptyElseBranch`` canonicalization deletes
it, and ``canonicalize`` runs ahead of everything in the AIR pipeline that reads
the region structure.
"""

# An scf.if region is counted alongside an air.sequential body: a buffer
# allocated inside one is freed by the herd after the region closes, so the
# dealloc would not be dominated by its alloc. See air.alloc's guard.
from ._loop import enter_region, exit_region

__all__ = ["Condition", "Branch", "branch"]


class Condition:
    """A comparison between two index expressions, awaiting ``ops.branch``.

    Built by ``==``/``!=``/``<``/``<=``/``>``/``>=`` on an
    :class:`~air.api._index.IndexExpr`. Nothing is emitted until ``ops.branch``
    opens a region with it, so writing the comparison next to its use and
    writing it far above have the same result.
    """

    __slots__ = ("lhs", "rhs", "predicate", "symbol")

    def __init__(self, lhs, rhs, predicate, symbol):
        self.lhs = lhs
        self.rhs = rhs
        self.predicate = predicate
        self.symbol = symbol

    def __repr__(self):
        return f"({self.lhs} {self.symbol} {self.rhs})"

    def __bool__(self):
        raise TypeError(
            f"the truth of {self!r} is not known at trace time: a herd body is "
            "traced once for the whole herd, so a comparison against a tile "
            "coordinate has no Python value. Write "
            f"`with ops.branch({self.lhs} {self.symbol} {self.rhs}):` for a "
            "region only some cores execute. (A Python `if`, `and`, `or` or "
            "`not` on this would have to pick one branch for every core.)"
        )

    def _reject_boolean(self, spelling, advice):
        raise NotImplementedError(
            f"air.api has no `{spelling}` on a condition. {advice}"
        )

    def __and__(self, other):
        self._reject_boolean(
            "&",
            "Conjunction is nesting: put the second ops.branch inside the first.",
        )

    __rand__ = __and__

    def __or__(self, other):
        self._reject_boolean(
            "|",
            "Write the two ops.branch regions one after the other, or invert the "
            "comparison and use otherwise().",
        )

    __ror__ = __or__

    def __invert__(self):
        self._reject_boolean(
            "~",
            "Use otherwise(), or write the opposite comparison (== for !=, "
            "< for >=).",
        )

    def materialize(self):
        """Emit the ``arith.cmpi``, returning its ``i1`` result."""
        from air.dialects import arith

        predicates = {
            "eq": arith.CmpIPredicate.eq,
            "ne": arith.CmpIPredicate.ne,
            "slt": arith.CmpIPredicate.slt,
            "sle": arith.CmpIPredicate.sle,
            "sgt": arith.CmpIPredicate.sgt,
            "sge": arith.CmpIPredicate.sge,
        }

        def operand(expr):
            # A constant folds to a Python int here, which cmpi cannot take.
            # Emitting arith.constant keeps a both-constant comparison as a
            # real scf.if that canonicalize then folds, rather than making the
            # branch vanish at trace time -- a design that selects between two
            # kernels with a Python-level parameter still wants the region in
            # the IR it hands the compiler.
            value = expr.materialize()
            if isinstance(value, int):
                return arith.ConstantOp.create_index(value)
            return value

        return arith.CmpIOp(
            predicates[self.predicate], operand(self.lhs), operand(self.rhs)
        )


# The branch arm currently being traced, as a list of (branch, arm) pairs --
# one per enclosing ops.branch. air.alloc stamps it on each buffer so the L1
# budget can tell "both of these exist" from "only one of these exists".
_ARM_PATH = []


def current_arm_path():
    """Where in the branch nest the tracer is, as a hashable tuple."""
    return tuple(_ARM_PATH)


class ValueCondition:
    """A comparison between values read out of buffers, awaiting ``ops.branch``.

    The sibling of :class:`Condition`, which compares *index* expressions --
    tile coordinates and loop variables, known per core before anything runs.
    This one compares data, and is legal for the same reason the index form is:
    it decides once per core, not once per element. Rank is what says which was
    meant, and it is the rule the whole DSL indexes by -- ``ctr[0] <= n`` is a
    single value and so a branch, while ``a[:] <= n`` is one bool per element
    and so ``ops.select``, which ``_why_not_a_branch`` still says.

    flash_attention/kernel_fusion_based's ``--causal-skip`` is the case. A core
    keeps its q-block index in an L1 counter tile, and blocks the mask kills
    outright have their matmul, softmax and PV skipped rather than computed and
    discarded. That cannot be ``ops.select``: what is being skipped is a
    ``func.call``, and a select evaluates both of its arms.
    """

    __slots__ = ("expr",)

    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return f"({self.expr!r})"

    def __bool__(self):
        raise TypeError(
            f"the truth of {self!r} is not known at trace time: it compares "
            "values that only exist once the kernel runs. Write "
            f"`with ops.branch(...):` for a region taken on that condition."
        )

    def materialize(self):
        """Emit the comparison, returning its ``i1`` result."""
        from ._emit import emit_scalar_value

        dtype = self.expr.args[0].element_dtype() or self.expr.args[1].element_dtype()
        if dtype is None:
            raise TypeError(
                "ops.branch was given a comparison between two scalars, which "
                "has no buffer to read and therefore no run-time value"
            )
        return emit_scalar_value(self.expr, dtype)


def _why_not_a_branch(condition):
    """Say which of the two conditionals the caller actually reached for."""
    from ._value import BufferExpr

    if isinstance(condition, BufferExpr):
        return (
            "That is an elementwise comparison on buffer *data*, which is "
            "ops.select's condition, not ops.branch's. A branch is taken once "
            "per core, so it cannot depend on what is in the buffer -- "
            "different elements would need different branches. Write "
            "`out[:] = ops.select(cond, a[:], b[:])`, which evaluates both "
            "sides and picks per element."
        )
    if isinstance(condition, bool):
        return (
            "A Python bool is not one: the region would be present or absent "
            "at trace time rather than chosen per instance, which is what a "
            "plain Python `if` around the same statements already does."
        )
    return (
        "For a per-element choice between two values, see ops.select; "
        "ops.branch decides once per core."
    )


class _Region:
    """One region of an ``scf.if``, as a context manager.

    ``terminate`` distinguishes the two: the then region is built empty and
    closes itself with an ``scf.yield``, while the else region already has one
    -- ``Branch`` terminates it as soon as the then region closes, so that an
    unused else is still a valid block -- and ``otherwise()`` inserts ahead of
    it instead.
    """

    def __init__(self, at, terminate, arm=None):
        self._at = at
        self._terminate = terminate
        # (branch, index) while this region is open, so air.alloc can tell a
        # buffer in the then arm from one in the else arm.
        self._arm = arm
        self.open = False

    def __enter__(self):
        from air.ir import InsertionPoint

        self._ip = InsertionPoint(self._at) if self._terminate else self._at
        self._ip.__enter__()
        self.open = True
        if self._arm is not None:
            _ARM_PATH.append(self._arm)
        enter_region()
        return self

    def __exit__(self, exc_type, exc, tb):
        from air.dialects.scf import yield_

        # An exception leaves the body short of the ops it was going to emit,
        # so the herd would compute something partial. Count it the way an
        # abandoned loop trip is counted, and still terminate the region: a
        # block with no terminator turns a diagnosable error into a verifier
        # crash somewhere unrelated.
        self.open = False
        if self._arm is not None:
            _ARM_PATH.pop()
        exit_region(aborted=exc_type is not None, what="ops.branch")
        if self._terminate:
            yield_([])
        self._ip.__exit__(exc_type, exc, tb)
        return False


def _as_condition(condition):
    """Promote a rank-0 data comparison to a :class:`ValueCondition`.

    Rank decides. A comparison whose operands are single elements -- ``ctr[0]``,
    not ``ctr[:]`` -- yields one bool and can open a region; anything wider
    yields one per element and is ops.select's, which the diagnostic says.
    """
    from ._emit import expr_shape
    from ._value import BufferExpr

    if isinstance(condition, BufferExpr) and condition.kind == "compare":
        if expr_shape(condition) == ():
            return ValueCondition(condition)
    return condition


class Branch:
    """The ``scf.if`` opened by :func:`branch`; ``otherwise()`` is its else."""

    def __init__(self, condition):
        condition = _as_condition(condition)
        if not isinstance(condition, (Condition, ValueCondition)):
            raise TypeError(
                "ops.branch takes a comparison between index expressions, such "
                f"as `tx == 0` or `k < n - 1`, got {condition!r} "
                f"({type(condition).__name__}). " + _why_not_a_branch(condition)
            )
        self._condition = condition
        self._op = None
        self._then = None
        self._otherwise_taken = False

    def __enter__(self):
        from air.dialects import scf

        if self._op is not None:
            raise RuntimeError(f"ops.branch{self._condition} entered twice")
        self._op = scf.IfOp(self._condition.materialize(), has_else=True)
        # The Branch itself keys the arm, not id(self): an id is only unique
        # while the object is alive, and `with ops.branch(...)` binds no name,
        # so a freed Branch's id can be handed to the next one. Two unrelated
        # branches would then share a key and _peak_bytes would take a max
        # across arms that are not alternatives, under-counting L1.
        self._then = _Region(self._op.then_block, terminate=True, arm=(self, 0))
        self._then.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        from air.ir import InsertionPoint
        from air.dialects.scf import yield_

        handled = self._then.__exit__(exc_type, exc, tb)
        # Terminate the else region now, whether or not otherwise() will claim
        # it. An else block holding nothing at all does not verify ("'scf.if'
        # op expects a non-empty block"), and there is no later hook to fix it
        # in: otherwise() is called after this returns.
        with InsertionPoint(self._op.else_block):
            self._else_terminator = yield_([])
        return handled

    def otherwise(self):
        """The else region, as a second ``with`` block."""
        from air.ir import InsertionPoint

        if self._op is None:
            raise RuntimeError(
                "otherwise() on an ops.branch whose region was never opened; the "
                "spelling is `with ops.branch(...) as branch:` followed by "
                "`with branch.otherwise():`"
            )
        if self._then.open:
            raise RuntimeError(
                f"otherwise() inside the body of ops.branch{self._condition}; "
                "close the first `with` block before opening the second"
            )
        if self._otherwise_taken:
            raise RuntimeError(
                f"ops.branch{self._condition} already has an otherwise() region"
            )
        self._otherwise_taken = True
        return _Region(
            InsertionPoint(self._else_terminator), terminate=False, arm=(self, 1)
        )


def branch(condition):
    """A region only the cores satisfying ``condition`` execute.

    Returns the :class:`Branch`, so the else region can be opened from it::

        with ops.branch(tx == 0) as head:
            ...
        with head.otherwise():
            ...
    """
    return Branch(condition)
