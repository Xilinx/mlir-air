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

Non-linear combinations (leaf * leaf, floordiv/mod by a leaf) are rejected with
an explicit error rather than silently degraded.
"""

__all__ = ["IndexExpr", "Leaf", "coerce_index", "materialize_index"]


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

    def __repr__(self):
        return self.name


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

    def __floordiv__(self, other):
        other = coerce_index(other)
        if not self.terms and not other.terms:
            return IndexExpr({}, self.const // other.const)
        raise NotImplementedError(
            "floordiv on a tile coordinate is not supported by air.api yet; "
            "restructure the index as a linear expression"
        )

    def __mod__(self, other):
        other = coerce_index(other)
        if not self.terms and not other.terms:
            return IndexExpr({}, self.const % other.const)
        raise NotImplementedError(
            "mod on a tile coordinate is not supported by air.api yet; "
            "restructure the index as a linear expression"
        )

    # -- queries ------------------------------------------------------------

    def as_const(self):
        """The Python int value, or None if the expression is coordinate-dependent."""
        return self.const if not self.terms else None

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

        from air.ir import AffineMap, AffineExpr, AffineSymbolExpr, AffineConstantExpr
        from air.dialects.affine import apply as affine_apply

        leaves = list(self.terms)
        expr = None
        for i, leaf in enumerate(leaves):
            term = AffineSymbolExpr.get(i)
            coeff = self.terms[leaf]
            if coeff != 1:
                term = AffineExpr.get_mul(term, AffineConstantExpr.get(coeff))
            expr = term if expr is None else AffineExpr.get_add(expr, term)
        if self.const:
            expr = AffineExpr.get_add(expr, AffineConstantExpr.get(self.const))

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
