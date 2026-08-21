# ./python/air/api/_index.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Affine index arithmetic for tile coordinates.

Tile coordinates written in the DSL (``tx * tm + row``) must reach the IR as a
single ``affine.apply``, not as a chain of ``arith.muli``/``arith.addi``: the AIR
dependency analysis and the DMA specialisation passes read affine offsets
directly, and lose track of them when the arithmetic is scattered across
``arith`` ops. ``programming_examples/eltwise_add/eltwise_add.py`` builds that
``affine.apply`` by hand; :class:`IndexExpr` builds the same thing from ordinary
Python operators.

Expressions are kept as an *affine linear form* -- a map from leaf to integer
coefficient, plus a constant. That representation makes the two things the
tracer needs cheap and exact:

  * ``(row + tm) - row`` folds to the constant ``tm``, which is how a slice
    like ``A[row : row + tm]`` recovers its static size; and
  * an expression with no leaves left is a plain Python ``int``, which
    ``split_static_dynamic`` then keeps in the static half of a memcpy's
    access pattern instead of materialising an SSA operand.

``x // k`` and ``x % k`` for a *constant* ``k`` are affine too, but they are not
linear, so they cannot be a coefficient on a leaf. They become a
:class:`DerivedLeaf` -- an opaque term that the linear form treats exactly like
a coordinate, and that expands back into ``affine.floordiv``/``affine.mod``
only at materialisation. That keeps ``((tw + 1) % sw) * 2`` linear in the thing
it is actually linear in, and it keeps the whole expression a single
``affine.apply``, which is what ``worker_to_worker`` builds by hand.

Genuinely non-affine combinations -- leaf * leaf, and floordiv/mod by a
coordinate -- are rejected with an explicit error rather than silently
degraded.
"""

__all__ = ["IndexExpr", "Leaf", "DerivedLeaf", "coerce_index", "materialize_index"]


class Leaf:
    """A named index SSA value (a herd coordinate or a loop induction var).

    Identity is object identity: the tracer creates one ``Leaf`` per coordinate
    and reuses it, so two references to ``tx`` share a term and cancel properly
    under subtraction.
    """

    __slots__ = ("value", "name")

    def __init__(self, value, name):
        self.value = value
        self.name = name

    def leaves(self):
        return (self,)

    def _affine(self, symbol_of):
        return symbol_of(self)

    def __repr__(self):
        return self.name


class DerivedLeaf:
    """``expr // k`` or ``expr % k`` for a constant ``k``, used as a leaf.

    Neither operation is linear, so it cannot live in :class:`IndexExpr`'s
    coefficient map directly. Wrapping it makes it opaque to the linear form
    while remaining fully affine in the IR.

    Unlike :class:`Leaf`, identity here is *structural*: two separately built
    copies of ``(tw + 1) % sw`` must share a term so that they cancel under
    subtraction and materialise once.
    """

    __slots__ = ("kind", "expr", "k", "_key")

    def __init__(self, kind, expr, k):
        self.kind = kind
        self.expr = expr
        self.k = int(k)
        self._key = (kind, expr._key(), self.k)

    def leaves(self):
        return self.expr.leaves()

    def _affine(self, symbol_of):
        from air.ir import AffineExpr, AffineConstantExpr

        lhs = self.expr._affine(symbol_of)
        rhs = AffineConstantExpr.get(self.k)
        if self.kind == "mod":
            return AffineExpr.get_mod(lhs, rhs)
        return AffineExpr.get_floor_div(lhs, rhs)

    def __eq__(self, other):
        return isinstance(other, DerivedLeaf) and self._key == other._key

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        op = "mod" if self.kind == "mod" else "floordiv"
        return f"({self.expr} {op} {self.k})"


class IndexExpr:
    """An affine linear form over :class:`Leaf` values: ``sum(c_i * leaf_i) + k``."""

    __slots__ = ("terms", "const")

    def __init__(self, terms=None, const=0):
        # Drop zero coefficients so that cancellation yields a true constant.
        self.terms = {leaf: c for leaf, c in (terms or {}).items() if c != 0}
        self.const = int(const)

    # -- construction -------------------------------------------------------

    @staticmethod
    def leaf(value, name):
        leaf = Leaf(value, name)
        return IndexExpr({leaf: 1}, 0)

    @staticmethod
    def constant(k):
        return IndexExpr({}, int(k))

    # -- arithmetic ---------------------------------------------------------

    def _combine(self, other, sign):
        other = coerce_index(other)
        terms = dict(self.terms)
        for leaf, c in other.terms.items():
            terms[leaf] = terms.get(leaf, 0) + sign * c
        return IndexExpr(terms, self.const + sign * other.const)

    def __add__(self, other):
        return self._combine(other, 1)

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, -1)

    def __rsub__(self, other):
        return coerce_index(other)._combine(self, -1)

    def __mul__(self, other):
        other = coerce_index(other)
        if self.terms and other.terms:
            raise TypeError(
                f"non-affine index expression: cannot multiply {self} by {other} "
                "(at most one operand may depend on a tile coordinate)"
            )
        if other.terms:
            self, other = other, self
        k = other.const
        return IndexExpr(
            {leaf: c * k for leaf, c in self.terms.items()}, self.const * k
        )

    __rmul__ = __mul__

    def __neg__(self):
        return IndexExpr({leaf: -c for leaf, c in self.terms.items()}, -self.const)

    def _divide(self, other, kind):
        other = coerce_index(other)
        if other.terms:
            raise TypeError(
                f"non-affine index expression: cannot take {kind} of {self} by "
                f"{other} (the divisor must be a constant, not a tile "
                "coordinate)"
            )
        k = other.const
        if k <= 0:
            raise ValueError(
                f"index {kind} by {k}: the divisor must be a positive constant"
            )
        if not self.terms:
            # Python's // and % are floor-based, which is what affine.floordiv
            # and affine.mod implement, so the folded and emitted forms agree.
            return IndexExpr(
                {}, self.const // k if kind == "floordiv" else self.const % k
            )
        if k == 1:
            return IndexExpr({}, 0) if kind == "mod" else self
        return IndexExpr({DerivedLeaf(kind, self, k): 1}, 0)

    def __floordiv__(self, other):
        return self._divide(other, "floordiv")

    def __mod__(self, other):
        return self._divide(other, "mod")

    def __rfloordiv__(self, other):
        return coerce_index(other)._divide(self, "floordiv")

    def __rmod__(self, other):
        return coerce_index(other)._divide(self, "mod")

    # -- queries ------------------------------------------------------------

    def as_const(self):
        """The Python int value, or None if the expression is coordinate-dependent."""
        return self.const if not self.terms else None

    def _key(self):
        """A hashable structural key, so equal expressions share a DerivedLeaf."""
        return (frozenset(self.terms.items()), self.const)

    def leaves(self):
        """The distinct real :class:`Leaf` values this expression reads, in order."""
        seen = {}
        for term in self.terms:
            for leaf in term.leaves():
                seen.setdefault(leaf, None)
        return list(seen)

    def _affine(self, symbol_of):
        from air.ir import AffineExpr, AffineConstantExpr

        expr = None
        for term, coeff in self.terms.items():
            part = term._affine(symbol_of)
            if coeff != 1:
                part = AffineExpr.get_mul(part, AffineConstantExpr.get(coeff))
            expr = part if expr is None else AffineExpr.get_add(expr, part)
        if expr is None:
            return AffineConstantExpr.get(self.const)
        if self.const:
            expr = AffineExpr.get_add(expr, AffineConstantExpr.get(self.const))
        return expr

    def __repr__(self):
        parts = [
            (f"{c}*{leaf}" if c != 1 else str(leaf)) for leaf, c in self.terms.items()
        ]
        if self.const or not parts:
            parts.append(str(self.const))
        return " + ".join(parts)

    # -- lowering -----------------------------------------------------------

    def materialize(self):
        """Emit this expression, returning a Python int or an index-typed Value.

        A constant is returned as an ``int`` so that callers can keep it in the
        static half of an access pattern; anything else becomes a single
        ``affine.apply``.
        """
        if not self.terms:
            return self.const

        from air.ir import AffineMap, AffineSymbolExpr
        from air.dialects.affine import apply as affine_apply

        # One symbol per distinct real leaf, shared across every term -- a
        # DerivedLeaf expands in place rather than taking a symbol of its own,
        # so `(tw + 1) % sw` and `tw` reach the map as the same symbol.
        leaves = self.leaves()
        position = {leaf: i for i, leaf in enumerate(leaves)}
        expr = self._affine(lambda leaf: AffineSymbolExpr.get(position[leaf]))

        amap = AffineMap.get(0, len(leaves), [expr])
        return affine_apply(amap, [leaf.value for leaf in leaves])


def coerce_index(value):
    """Coerce an int, a Symbol, or an IndexExpr into an IndexExpr."""
    if isinstance(value, IndexExpr):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a valid index")
    if isinstance(value, int):
        return IndexExpr.constant(value)
    # Symbol and anything else exposing __index__ resolves to its current value.
    if hasattr(value, "__index__"):
        return IndexExpr.constant(int(value))
    raise TypeError(f"cannot use {value!r} ({type(value).__name__}) as an index")


def materialize_index(value):
    """Materialize an int / Symbol / IndexExpr into an int or an SSA Value."""
    return coerce_index(value).materialize()
