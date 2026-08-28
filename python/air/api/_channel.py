# ./python/air/api/_channel.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Named data channels: ``air.channel`` plus ``put`` and ``get``.

A channel is a point-to-point connection between two memrefs, named by a
module-level symbol::

    stream = air.channel("ChanIn")
    ...
    stream.put(A)                    # from a segment, an L3 tensor
    ...
    stream.get(l1_tile)              # from a herd, into an L1 buffer

Why this is not just a DMA
--------------------------

Two properties make a channel a different thing from ``ops.load``/``ops.store``,
and both come straight from the hand-written examples this models.

**It crosses scopes without being an operand.** ``air.herd`` and ``air.segment``
are ``IsolatedFromAbove``, so a buffer has to be threaded in explicitly -- which
is what the DSL already does for staged L2 tiles. A channel does not: it is a
module-level ``Symbol`` referenced by name, so a herd can consume what a segment
produced with no plumbing at all. That is the whole point of the abstraction.

**Put and get sizes need not match.** ``passthrough_channel`` puts one whole
``[4096]`` tensor and the herd takes four gets of ``[1024]``;
``vector_matrix_multiplication`` puts ``[k]`` and gets ``[tile_k]`` at a time. A
channel is a *stream* -- each get takes the next chunk -- so the two sides are
validated independently and the pairing is the compiler's business. Imposing
``ops.load``'s shape-equality rule here would reject every canonical example.

Where a put or get may appear
-----------------------------

An endpoint in L3 -- a tensor or a slice of one -- has to sit inside an
``air.segment``. Measured on npu1, not assumed: the same passthrough with its
put and get hoisted to function scope, outside ``air.launch``, fails to compile
with ``'air.channel.get' op failed to link to any shim dma allocation``, while
at segment scope it passes with resource counts identical to the original's.
The check below turns that into an error naming the fix.

An L1 endpoint inside a herd body is the consumer side and is always fine.
"""

from ._value import Buffer, Tensor, TensorSlice, Token

__all__ = ["Channel", "channel"]


def _as_dims(value, what):
    if value is None:
        return None
    if isinstance(value, int):
        raise TypeError(
            f"air.channel({what}=...) takes a list of extents, e.g. {what}=[2, 3]; "
            f"got the bare integer {value}. A 1-D channel array is {what}=[{value}]."
        )
    dims = [int(v) for v in value]
    for d in dims:
        if d <= 0:
            raise ValueError(
                f"air.channel({what}=...) extents must be positive, got {dims}"
            )
    return dims


# Construction order, which is the order the declarations are emitted in.
# See Channel._declare for why it is not first-use order.
_NEXT_SEQ = [0]
# Symbol name -> construction index, for the channels declared so far in the
# current module. Keyed on the name because that is all an already-emitted
# air.channel op carries back.
_DECLARED_SEQ = {}


def _reset_declaration_order():
    """Start a fresh module. Called by the tracer when it opens one.

    Only the name->index map is cleared. The counter keeps running: channels
    are constructed *before* the trace opens, so restarting it here would
    renumber objects that already have an index and scramble the very order
    this exists to preserve. Only relative order is ever read.
    """
    _DECLARED_SEQ.clear()


class Channel:
    """A named channel; ``put`` and ``get`` move data through it."""

    __slots__ = ("name", "size", "broadcast_shape", "channel_type", "_declared", "_seq")

    def __init__(
        self, name, size=None, broadcast_shape=None, channel_type=None, **unsupported
    ):
        if not isinstance(name, str) or not name:
            raise TypeError(
                "air.channel(name) takes the channel's symbol name, e.g. "
                'air.channel("ChanIn")'
            )
        _reject_unsupported(unsupported)

        self.size = _as_dims(size, "size")
        self.broadcast_shape = _as_dims(broadcast_shape, "broadcast_shape")

        if self.broadcast_shape is not None:
            if self.size is None:
                raise ValueError(
                    "air.channel(broadcast_shape=...) also needs size=: the "
                    "broadcast is described relative to the channel's own "
                    "extents, so a 1-to-N fan-out is "
                    f"size=[1]*{len(self.broadcast_shape)}, "
                    f"broadcast_shape={self.broadcast_shape}."
                )
            if len(self.broadcast_shape) != len(self.size):
                raise ValueError(
                    f"air.channel: broadcast_shape {self.broadcast_shape} has "
                    f"rank {len(self.broadcast_shape)} but size {self.size} has "
                    f"rank {len(self.size)}; they describe the same axes"
                )
            for axis, (s, b) in enumerate(zip(self.size, self.broadcast_shape)):
                if b % s:
                    raise ValueError(
                        f"air.channel: broadcast_shape {self.broadcast_shape} is "
                        f"not a whole multiple of size {self.size} on axis "
                        f"{axis} ({b} % {s} != 0), so the fan-out is not an "
                        "integer number of destinations per source"
                    )

        self.name = name
        # Only the types this package can gate against a real design are
        # implemented: npu_cascade by matrix_vector_multiplication/bf16_cascade
        # on npu1, and npu_dma_packet by llms/.../o_gemv_ffn_int4_fused, whose
        # 16-destination res1ToCons broadcast is packet-switched because the LD
        # stream switch arbiter's 4-msel multicast limit leaves no other way to
        # reach that many cores. The rest each have their own lowering and
        # verifier rules -- see _UNSUPPORTED -- and adding one blind would be a
        # channel that compiles as something other than what was asked for.
        if channel_type is not None and channel_type not in _IMPLEMENTED_TYPES:
            raise NotImplementedError(
                f"air.channel(channel_type={channel_type!r}) is not implemented; "
                f"{_UNSUPPORTED['channel_type']}. Available: "
                f"{', '.join(repr(t) for t in sorted(_IMPLEMENTED_TYPES))}; the "
                "default (npu_dma_stream) is what you get by leaving "
                "channel_type off."
            )
        # A cascade is a point-to-point link between neighbouring cores, so
        # there is nothing for a broadcast shape to describe. That reasoning is
        # specific to cascade and does not generalise: a *packet* broadcast is
        # precisely a one-to-many fan-out, and reaches all of its destinations
        # over one flow where a circuit-switched channel needs one per
        # destination.
        if channel_type == "npu_cascade" and broadcast_shape is not None:
            raise ValueError(
                "air.channel does not take broadcast_shape= with "
                "channel_type='npu_cascade': a cascade is a point-to-point link "
                "between neighbouring cores, so there is nothing for a broadcast "
                "shape to describe. channel_type='npu_dma_packet' does take one "
                "-- a packet broadcast is a genuine one-to-many fan-out."
            )
        self.channel_type = channel_type
        self._declared = False
        self._seq = _NEXT_SEQ[0]
        _NEXT_SEQ[0] += 1

    def __repr__(self):
        extra = f", size={self.size}" if self.size is not None else ""
        if self.broadcast_shape is not None:
            extra += f", broadcast_shape={self.broadcast_shape}"
        return f"air.api.channel({self.name!r}{extra})"

    # -- emission ----------------------------------------------------------

    def _declare(self):
        """Materialise the ``air.channel`` symbol at module scope, once.

        Deferred to first use for the same reason ``air.extern`` defers its
        private ``func.func``: the module does not exist until a trace is
        active, and this way a channel declared but never used emits nothing.
        """
        if self._declared:
            return
        from air.ir import InsertionPoint
        from air.dialects.air import Channel as ChannelOp
        from air.dialects.air import channel as channel_decl

        from ._trace import active_trace

        trace = active_trace()
        # Placed by *construction* order, not by first-use order.
        #
        # Declaration is deferred to first use because the module does not
        # exist until a trace is active, and because a channel declared and
        # never used should emit nothing. But emitting them in the order they
        # happen to be used first is not merely untidy: air-to-aie walks the
        # air.channel symbols in order when it hands out shim DMA channels, so
        # on a design near the shim's limits the order decides whether an L3
        # endpoint gets an allocation at all. o_gemv_ffn_int4_fused is the
        # worked case -- use order there puts laResDebug where a column has no
        # MM2S left, and it fails to compile with "failed to get MM2S tile for
        # L3 allocation".
        #
        # Construction order is what the author wrote and what the raw-bindings
        # predecessors control directly, so that is the order to reproduce:
        # insert ahead of the first channel already emitted whose construction
        # index is greater than this one's.
        #
        # The anchor is a *channel* rather than the first non-channel, because
        # air.extern prepends its private func.func decls at block begin --
        # anchoring on "the first thing that is not a channel" would start
        # inserting above them once a kernel had been called.
        ops = list(trace.module.body.operations)
        channels = [
            (i, op) for i, op in enumerate(ops) if op.operation.name == "air.channel"
        ]
        successor = next(
            (
                op
                for _, op in channels
                if _DECLARED_SEQ.get(op.operation.attributes["sym_name"].value, -1)
                > self._seq
            ),
            None,
        )
        if successor is not None:
            ip = InsertionPoint(successor)
        elif channels:
            # After the last one; the enclosing func always follows, so there
            # is a next sibling to insert before.
            ip = InsertionPoint(ops[channels[-1][0] + 1])
        else:
            ip = InsertionPoint.at_block_begin(trace.module.body)
        with ip:
            if self.channel_type is None:
                ChannelOp(
                    self.name,
                    size=self.size,
                    broadcast_shape=self.broadcast_shape,
                )
            elif self.broadcast_shape is None:
                # The generated builder rather than the extension class: only
                # it carries channel_type.
                channel_decl(self.name, size=self.size, channel_type=self.channel_type)
            else:
                # Neither builder takes both: the generated one has no
                # broadcast_shape parameter and the extension class has no
                # channel_type. A packet broadcast needs both, so build it with
                # the extension class and stamp the type on afterwards.
                from air.ir import StringAttr

                op = ChannelOp(
                    self.name,
                    size=self.size,
                    broadcast_shape=self.broadcast_shape,
                )
                op.operation.attributes["channel_type"] = StringAttr.get(
                    self.channel_type
                )
        _DECLARED_SEQ[self.name] = self._seq
        self._declared = True

    def _indices(self, indices, direction):
        if indices is None:
            indices = ()
        if isinstance(indices, int):
            indices = (indices,)
        indices = list(indices)
        if not indices:
            return []
        if self.size is None:
            raise ValueError(
                f"air.channel {self.name!r} was declared without size=, so it is "
                f"a single channel and takes no indices; got indices={indices}. "
                f"Declare it as air.channel({self.name!r}, size=[...]) to make it "
                "an array."
            )
        if len(indices) != len(self.size):
            raise ValueError(
                f"air.channel {self.name!r} has size {self.size}, so it takes "
                f"{len(self.size)} index/indices; got {len(indices)} "
                f"({indices})"
            )
        from ._index import coerce_index

        # A herd coordinate or loop variable arrives as an IndexExpr and has to
        # be materialised into an index-typed Value; a constant folds back to a
        # Python int, which the op keeps static. Only the constant case can be
        # range-checked -- the herd shape already bounds the dynamic one.
        # The two ends of a broadcast are indexed over different grids, so the
        # bound depends on the direction. A consumer names its slot among the
        # *destinations*: with `size=[1, 1], broadcast_shape=[3, 1]`, one
        # producer feeds a 3x1 grid and the gets are [0,0], [1,0], [2,0] --
        # bounding those by size would admit only [0, 0], which is what
        # broadcast/multi_herd hit. A producer still indexes the *source*
        # bundle, which is `size`, so widening the bound for puts as well would
        # accept an out-of-range source and emit an invalid bundle index.
        bounds = (
            self.broadcast_shape
            if (direction == "get" and self.broadcast_shape)
            else self.size
        )
        out = []
        for axis, (idx, extent) in enumerate(zip(indices, bounds)):
            value = coerce_index(idx).materialize()
            if isinstance(value, int) and not 0 <= value < extent:
                raise ValueError(
                    f"air.channel {self.name!r} index {value} is out of range "
                    f"on axis {axis}: "
                    + (
                        f"broadcast_shape is {self.broadcast_shape}"
                        if bounds is self.broadcast_shape
                        else f"size is {self.size}"
                    )
                    + f", so that axis admits 0..{extent - 1}"
                )
            out.append(value)
        return out

    def _emit(self, obj, indices, dependency, direction):
        from ._trace import current_herd, current_launch, current_segment
        from ._value import Tensor, TensorSlice
        from .ops import _check_dependency, _endpoint

        _check_dependency(dependency)
        self._declare()

        # Decide placement from the type alone, and resolve the endpoint only
        # once the insertion point is final. `_endpoint` materialises a tensor
        # slice's offset arithmetic where it is called, so resolving it here
        # would emit an affine.apply outside the launch on the re-entry path
        # below, and then a second one inside -- leaving the first orphaned at
        # func scope.
        tensor = (
            obj.tensor
            if isinstance(obj, TensorSlice)
            else obj if isinstance(obj, Tensor) else None
        )
        what = "tensor slice" if isinstance(obj, TensorSlice) else "tensor"

        # Where an L3 endpoint may sit, from three measurements rather than
        # one; see the module docstring.
        if tensor is not None:
            try:
                current_launch()
            except RuntimeError:
                raise RuntimeError(
                    f"air.channel.{direction} on an L3 {what} has to be "
                    "inside an air.launch: reaching L3 needs a shim DMA "
                    "allocation, and outside a launch the compiler has none to "
                    f"link to (it fails with \"'air.channel.{direction}' op failed "
                    'to link to any shim dma allocation"). Open one with '
                    "`with air.launch(...) as launch:` and move it there."
                ) from None
            if (
                current_herd(required=False) is not None
                and current_segment(required=False) is None
            ):
                raise RuntimeError(
                    f"air.channel.{direction} on an L3 {what} inside a "
                    "herd body needs an air.segment around that herd: the shim "
                    "DMA allocation is the segment's, and without one aircc does "
                    "not diagnose it but crashes, in air-dependency, on a "
                    "dependencyGraph index assertion. Either wrap the herd in "
                    "`with air.segment(...) as seg:`, or move this "
                    f"{direction} out to launch scope, where an L3 endpoint is "
                    "fine with no segment at all."
                )

        if direction == "get" and tensor is not None:
            # Same rule ops.store applies: being written from the device is what
            # makes a tensor an output, which fixes the calling convention.
            tensor.is_output = True

        from air.dialects.air import ChannelGet, ChannelPut

        op = ChannelGet if direction == "get" else ChannelPut

        def build():
            # Resolved here, not above, so that both the endpoint's offset
            # arithmetic and the bundle indices are emitted at whichever
            # insertion point this op ends up at. Opening or re-entering the
            # launch's region also rebinds each tensor to that region's block
            # argument, so an endpoint captured outside would still name the
            # function's, and air.channel.put would fail to verify with "using
            # value defined outside the region".
            endpoint = _endpoint(obj, f"channel.{direction}", "argument")
            offsets, sizes, strides = endpoint.pattern or ([], [], [])
            return op(
                self.name,
                endpoint.value,
                offsets=offsets,
                sizes=sizes,
                strides=strides,
                indices=self._indices(indices, direction),
            )

        if tensor is None:
            return Token(build())

        # An L3 endpoint has to sit inside the launch, which is what brings the
        # shim DMA allocation. In a gridless launch that region is opened
        # lazily, so a put that runs *before* the first segment is what opens
        # it, and a get that drains the result afterwards has to step back in --
        # data_transfer_transpose/channel does both, around the segment rather
        # than within it. Without this they land at func scope and the compiler
        # reports "failed to link to any shim dma allocation".
        from ._trace import in_launch_body

        launch = current_launch()
        if launch.opened and launch.reentry is None:
            # Already inside the region; nothing will be rebound.
            return Token(build())

        # Stepping back into the launch moves the op, and with it the offset
        # arithmetic, inside an IsolatedFromAbove region. A constant travels;
        # anything derived from a coordinate or a loop variable does not,
        # because that value was created out here. Emitting it anyway produces
        # "'affine.apply' op using value defined outside the region", which
        # reads as a DSL bug rather than as the shape of the program.
        if isinstance(obj, TensorSlice) and any(o.terms for o in obj.offsets):
            raise RuntimeError(
                f"air.channel.{direction} on an L3 tensor slice whose offset is "
                "computed from a coordinate or loop variable cannot sit after "
                "an air.segment at launch scope: the op has to move inside "
                "air.launch to reach a shim DMA allocation, and the value the "
                "offset is built from was created outside it. Either move this "
                f"{direction} inside the segment, where the offset and the "
                "transfer are in the same region, or put it before the segment, "
                "which opens the launch around both."
            )

        built = []
        in_launch_body(lambda: built.append(build()))
        return Token(built[0])

    # -- public ------------------------------------------------------------

    def put(self, obj, indices=None, dependency=None, **unsupported):
        """Send a tensor, a buffer, or a region of either into the channel."""
        _reject_unsupported(unsupported)
        return self._emit(obj, indices, dependency, "put")

    def get(self, obj, indices=None, dependency=None, **unsupported):
        """Receive the channel's next chunk into a tensor, buffer, or region."""
        _reject_unsupported(unsupported)
        return self._emit(obj, indices, dependency, "get")


# Keywords the underlying ops accept and this DSL does not lower. They raise
# rather than being dropped: a channel that silently ignores `channel_type` is
# one that compiles as a stream and was asked for a cascade.
# Channel types this package implements. Each is here because a real design
# in this tree exercises it end to end; see the guard in Channel.__init__.
_IMPLEMENTED_TYPES = ("npu_cascade", "npu_dma_packet")


_UNSUPPORTED = {
    "channel_type": (
        "channel types other than the default npu_dma_stream (npu_dma_packet, "
        "npu_cascade, npu_mmio, gpu_symmetric_heap). Each has its own lowering "
        "and its own verifier rules -- mmio, for instance, requires an L3 put, "
        "an L1 get and a constant memref.get_global source"
    ),
    "dest": "the packet-demux destination index, which needs npu_dma_packet",
    "packet_ids": "explicit packet routing ids, which need npu_dma_packet",
    "buffer_resources": "the objectFifo depth knob",
    "pad_before": "padded transfers",
    "pad_after": "padded transfers",
}


def _reject_unsupported(kwargs):
    for key in kwargs:
        what = _UNSUPPORTED.get(key)
        if what is None:
            raise TypeError(f"air.channel got an unexpected keyword {key!r}")
        raise NotImplementedError(f"air.api does not implement {key}=: {what}")


def channel(name, size=None, broadcast_shape=None, channel_type=None, **unsupported):
    """Declare a named channel; ``put`` into it and ``get`` out of it."""
    return Channel(
        name,
        size=size,
        broadcast_shape=broadcast_shape,
        channel_type=channel_type,
        **unsupported,
    )
