# ./python/air/api/_pack.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Micro-blocked (``packed``) tile layouts for the AIE2 matmul intrinsic.

The AIE2 vector unit multiplies a fixed micro-tile -- ``4x8x4`` for bf16 -- so a
32x32 L1 tile has to be laid out as a grid of those micro-tiles rather than
row-major, and the contraction becomes a 6-D ``linalg.generic`` over
``[.., group, group, micro, micro]``. This module builds those shapes and the
DMA access patterns that produce them.

Where the packing actually lives
--------------------------------

Read the IR of the example this models
(``programming_examples/matrix_multiplication/bf16/run.py --print-module-only``)
and three things are true of every alloc in it:

* there is **no layout attribute anywhere** -- every ``memref.alloc`` is an
  ordinary contiguous memref;
* the packing is carried by the **shape**, ``memref<1x1x2x8x4x8xbf16, 2 : i32>``;
* the only strided memref type in the module is the *result* of
  ``memref.subview``, which MLIR derives rather than anyone declaring.

The transform itself is in the DMA, and specifically in the fact that its access
pattern has a **higher rank than the memref it reads**::

    air.dma_memcpy_nd (%l1_a[] [] [],
        %l2_a[%tx, 0, 0, 0, 0, %k] [1, 1, 2, 8, 4, 8] [1024, 1024, 8, 128, 32, 1])
      : (memref<1x1x2x8x4x8xbf16, 2 : i32>, memref<1x1x32x32xbf16, 1 : i32>)

Six offsets/sizes/strides against a rank-4 L2 memref. That tuple *is* the pack:
the DMA walks the flat L2 tile in micro-tile order and lands it contiguously in
L1. So packing is neither a buffer layout nor a reshape -- it is an access
pattern, and the buffer on the packed side is plain.

Why this is a derivation and not a stride tuple
-----------------------------------------------

Handing the user six raw strides would be the raw bindings with extra steps, and
a wrong stride is silently wrong output rather than an error. But the strides are
fully determined by the micro-tile and the *source buffer's own* row-major
strides, so the DSL can compute them and check the result against the
destination's shape. Writing ``s_M``/``s_K``/``s_N`` for the source strides along
its logical axes:

======  =========================  ======================  =====================================
role    pattern dims               sizes                   strides
======  =========================  ======================  =====================================
``A``   ``k_grp, m_grp, m, k``     ``K/k, M/m, m, k``      ``k*s_K, m*s_M, s_M, s_K``
``B``   ``n_grp, k_grp, k, n``     ``N/n, K/k, k, n``      ``n*s_N, k*s_K, s_K, s_N``
``C``   ``m_grp, m, n_grp, n``     ``M/m, m, N/n, n``      ``s_mgrp, s_m, s_ngrp, s_n``
======  =========================  ======================  =====================================

``A`` and ``B`` pack (flat source, packed destination); ``C`` unpacks, reading
the packed L1 buffer back in logical row-major order, so its strides come from
the packed buffer's own shape. Each row above reproduces the reference example's
literal stride tuple exactly -- ``[1024,1024,8,128,32,1]``,
``[1024,1024,4,256,32,1]`` and ``[1024,1024,16,4,128,1]`` respectively -- which is
the test that this table is right rather than merely plausible.

Offsets follow one rule: a logical offset lands on the *innermost* pattern
dimension for its axis, the one carrying that axis's base stride. That is why the
reference puts the K offset last, at stride 1, for ``A``.
"""

from ._index import coerce_index

__all__ = ["MicroTile", "PackedShape", "micro_tile"]


class PackedShape(tuple):
    """A packed tile shape, plus the descriptor that produced it.

    It *is* a tuple, so ``air.alloc`` takes it wherever a shape goes and nothing
    else has to know about packing. Carrying the descriptor along is what lets
    ``ops.load``/``ops.store`` derive the access pattern later without the call
    site repeating the micro-tile.
    """

    # No __slots__: a subtype of tuple cannot have a non-empty one.

    def __new__(cls, extents, role, micro, logical, lead):
        self = super().__new__(cls, tuple(int(e) for e in extents))
        self.role = role
        self.micro = micro
        self.logical = tuple(int(x) for x in logical)
        self.lead = tuple(int(x) for x in lead)
        return self

    def __repr__(self):
        return (
            f"PackedShape({tuple(self)}, role={self.role!r}, "
            f"micro={self.micro!r}, logical={self.logical})"
        )


def _check_divides(name, extent, unit, role, axis):
    if unit <= 0:
        raise ValueError(f"micro-tile {name} must be positive, got {unit}")
    if extent % unit:
        raise ValueError(
            f"operand {role}'s {axis} extent {extent} is not a multiple of the "
            f"micro-tile {name}={unit}, so it cannot be laid out as whole "
            f"micro-tiles. Pick a tile size divisible by {unit}, or a micro-tile "
            f"that divides {extent}."
        )
    return extent // unit


class MicroTile:
    """The AIE2 matmul intrinsic's operand shape: ``m x k`` by ``k x n``.

    ``(4, 8, 4)`` is the bf16 shape and ``(4, 8, 8)`` the int8 one; the value that
    matters is whatever the external kernel was compiled for. The reference
    example passes exactly these through ``-DDIM_M``/``-DDIM_K``/``-DDIM_N`` to
    ``mm.cc``, so a mismatch here is a mismatch with the object file.
    """

    __slots__ = ("m", "k", "n")

    def __init__(self, m, k, n):
        for name, v in (("m", m), ("k", k), ("n", n)):
            if int(v) <= 0:
                raise ValueError(f"micro-tile {name} must be positive, got {v}")
        self.m, self.k, self.n = int(m), int(k), int(n)

    def __repr__(self):
        return f"MicroTile(m={self.m}, k={self.k}, n={self.n})"

    def __eq__(self, other):
        return isinstance(other, MicroTile) and (self.m, self.k, self.n) == (
            other.m,
            other.k,
            other.n,
        )

    def __hash__(self):
        return hash((self.m, self.k, self.n))

    # -- packed shapes -----------------------------------------------------
    #
    # `lead` is the pair of outer dimensions every buffer in this layout carries.
    # For the per-core A and B tiles it is (1, 1); for a C accumulator shared
    # across the herd it is the herd shape, and the core selects its own slab
    # with a subview -- which is exactly what the reference does.

    def a(self, M, K, lead=(1, 1)):
        """``[*lead, K/k, M/m, m, k]`` -- the packed left operand."""
        M, K = int(M), int(K)
        kg = _check_divides("k", K, self.k, "A", "K")
        mg = _check_divides("m", M, self.m, "A", "M")
        return PackedShape(
            tuple(lead) + (kg, mg, self.m, self.k), "A", self, (M, K), lead
        )

    def b(self, K, N, lead=(1, 1)):
        """``[*lead, N/n, K/k, k, n]`` -- the packed right operand."""
        K, N = int(K), int(N)
        ng = _check_divides("n", N, self.n, "B", "N")
        kg = _check_divides("k", K, self.k, "B", "K")
        return PackedShape(
            tuple(lead) + (ng, kg, self.k, self.n), "B", self, (K, N), lead
        )

    def c(self, M, N, lead=(1, 1)):
        """``[*lead, N/n, M/m, m, n]`` -- the packed accumulator."""
        M, N = int(M), int(N)
        ng = _check_divides("n", N, self.n, "C", "N")
        mg = _check_divides("m", M, self.m, "C", "M")
        return PackedShape(
            tuple(lead) + (ng, mg, self.m, self.n), "C", self, (M, N), lead
        )


def micro_tile(m, k, n):
    """The AIE2 matmul intrinsic shape; ``(4, 8, 4)`` for bf16."""
    return MicroTile(m, k, n)


# ---------------------------------------------------------------------------
# Access-pattern derivation
# ---------------------------------------------------------------------------


def pack_pattern(packed, sizes, strides, offsets):
    """Derive the rank-``lead+4`` DMA pattern for one packed operand.

    The three inputs describe the region in **logical** terms -- ``lead``
    dimensions followed by the operand's two logical axes -- and the result is
    the micro-tiled pattern over the same elements.

    Which buffer they describe depends on the direction, because a pack and an
    unpack put the pattern on opposite sides:

    * ``A``/``B`` **pack**: the pattern goes on the flat source, so ``strides``
      are the *source* buffer's row-major strides and the derivation reorders
      them into micro-tile order.
    * ``C`` **unpacks**: the pattern goes on the packed buffer itself and walks
      it back in logical row-major order, so only the ``lead`` strides are taken
      from the caller -- the four inner strides come from the packed shape.
    """
    nlead = len(packed.lead)
    if len(sizes) != nlead + 2:
        raise ValueError(
            f"a packed {packed.role} operand needs a region of rank "
            f"{nlead + 2} -- {nlead} leading dimension(s) then the two logical "
            f"axes -- but got rank {len(sizes)}, {tuple(sizes)}"
        )

    lead_sizes = list(sizes[:nlead])
    lead_strides = list(strides[:nlead])
    lead_offsets = list(offsets[:nlead])
    # The two logical axes, in the order the flat side stores them: A is (M, K),
    # B is (K, N), C is (M, N).
    e0, e1 = sizes[nlead], sizes[nlead + 1]
    s0, s1 = strides[nlead], strides[nlead + 1]
    o0, o1 = offsets[nlead], offsets[nlead + 1]
    zero = coerce_index(0)
    micro = packed.micro

    if packed.role == "A":
        # (M, K) -> [k_grp, m_grp, m, k]
        m, k = micro.m, micro.k
        _expect(packed, "M", e0, m)
        _expect(packed, "K", e1, k)
        inner = (
            [e1 // k, e0 // m, m, k],
            [k * s1, m * s0, s0, s1],
            [zero, zero, o0, o1],
        )
    elif packed.role == "B":
        # (K, N) -> [n_grp, k_grp, k, n]
        k, n = micro.k, micro.n
        _expect(packed, "K", e0, k)
        _expect(packed, "N", e1, n)
        inner = (
            [e1 // n, e0 // k, k, n],
            [n * s1, k * s0, s0, s1],
            [zero, zero, o0, o1],
        )
    elif packed.role == "C":
        # (M, N) -> [m_grp, m, n_grp, n], read out of the packed buffer.
        m, n = micro.m, micro.n
        _expect(packed, "M", e0, m)
        _expect(packed, "N", e1, n)
        mg = packed[nlead + 1]
        s_n, s_m, s_mgrp, s_ngrp = 1, n, m * n, mg * m * n
        inner = (
            [e0 // m, m, e1 // n, n],
            [s_mgrp, s_m, s_ngrp, s_n],
            [zero, o0, zero, o1],
        )
    else:  # pragma: no cover -- role is set only by MicroTile
        raise AssertionError(f"unknown packed operand role {packed.role!r}")

    isizes, istrides, ioffsets = inner
    return lead_offsets + ioffsets, lead_sizes + isizes, lead_strides + istrides


def _expect(packed, axis, extent, unit):
    if extent % unit:
        raise ValueError(
            f"the {axis} extent of this region is {extent}, which is not a "
            f"multiple of the micro-tile {packed.micro!r} the packed buffer was "
            f"allocated with, so it cannot be split into whole micro-tiles"
        )
