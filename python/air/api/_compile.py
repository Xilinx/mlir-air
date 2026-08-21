# ./python/air/api/_compile.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The launch context: replays a traced body into MLIR, then compiles it.

The body is not executed when ``@launch.body`` runs -- it is recorded, and
replayed later by :meth:`LaunchContext.build` from inside an MLIR context. The
deferral matters: the function signature is built from the tensors the launch
claimed, and the body (which references those tensors) can only run once its
block arguments exist.

Compilation goes through the existing ``XRTBackend``, so an ``air.api`` kernel
takes exactly the same pipeline as a hand-written module built with
``@module_builder``.
"""

import os
import tempfile

import numpy as np

from ._trace import (
    DEFAULT_TARGET,
    PENDING_SYMBOLS,
    PENDING_TENSORS,
    LaunchState,
    Trace,
    _positional_arity,
    open_launch_region,
    parse_grid,
    resolve_target,
    set_active_trace,
    set_launch,
    set_target,
)

__all__ = ["LaunchContext", "CompiledKernel", "launch", "compile"]


class LaunchContext:
    """A traced kernel: an interface, a body, and a compiler entry point."""

    def __init__(self, grid=None, name="kernel", target=None):
        # This launch's own iteration space -- air.launch's `sizes`. One point
        # is one replay of everything inside, so outer tiling belongs here:
        # a segment's L2 staging is refilled per point. air.segment and
        # air.herd each own a separate one; see LaunchState.
        self.dims = parse_grid(grid) if grid is not None else ()
        if len(self.dims) > 2:
            raise NotImplementedError(
                f"air.launch is 1-D or 2-D; got {len(self.dims)}-D"
            )
        self.grid = tuple(d.count for d in self.dims)
        self.tile_sizes = tuple(d.step for d in self.dims)
        self.name = name
        self.target = target or DEFAULT_TARGET
        self.tensors = list(PENDING_TENSORS)
        PENDING_TENSORS.clear()
        self.symbols = list(PENDING_SYMBOLS)
        PENDING_SYMBOLS.clear()
        self._body = None
        self._module = None
        self._l1_peak = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def body(self):
        def decorator(fn):
            if self._body is not None:
                raise RuntimeError("launch body registered twice")
            self._body = fn
            return fn

        return decorator

    @property
    def search_space(self):
        """The resolved value of every symbol this launch captured."""
        return {s.name: s.value for s in self.symbols}

    @property
    def inputs(self):
        return [t for t in self.tensors if not t.is_output]

    @property
    def outputs(self):
        return [t for t in self.tensors if t.is_output]

    # -- building -----------------------------------------------------------

    def build(self, target=None):
        """Trace the body into an MLIR module (cached after the first call)."""
        if self._module is not None:
            return self._module
        if self._body is None:
            raise RuntimeError(
                "launch has no body; decorate one with @launch.body before "
                "calling mlir() or compile()"
            )
        if not self.tensors:
            raise RuntimeError(
                "launch has no interface; declare air.tensor(...) values before "
                "opening the launch"
            )

        # Resolved here rather than in __init__ so that declaring a launch does
        # not shell out to xrt-smi; `target` is a concrete generation from this
        # point on, and `self.target` is what the backend must compile for.
        self.target = resolve_target(target or self.target)

        from air.ir import Context, InsertionPoint, Location, MemRefType, Module
        from air.dialects.func import FuncOp

        with Context(), Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                arg_types = [
                    MemRefType.get(list(t.shape), t.dtype.mlir()) for t in self.tensors
                ]

                @FuncOp.from_py_func(*arg_types, name=self.name)
                def _kernel(*args):
                    for t, v in zip(self.tensors, args):
                        t.value = v
                    trace = Trace(self.tensors, module=module)
                    previous = set_active_trace(trace)
                    previous_target = set_target(self.target)
                    state = LaunchState(self)
                    previous_launch = set_launch(state)
                    try:
                        self._run_body(state)
                    finally:
                        set_launch(previous_launch)
                        self._l1_peak = trace.l1_peak
                        set_target(previous_target)
                        set_active_trace(previous)
                        for t in self.tensors:
                            t.value = None

        self._module = module
        self._check_interface()
        return module

    def _run_body(self, state):
        """Run the launch body, opening air.launch first if this launch has a grid.

        With no grid there is nothing for air.launch to say, so nothing is
        emitted until a segment needs a launch to sit inside -- which keeps a
        kernel that stages nothing at the plain `func` + `air.herd` shape its
        hand-written predecessor had.
        """
        n_expected = len(self.dims)
        if _positional_arity(self._body) != n_expected:
            raise TypeError(
                f"launch body takes {_positional_arity(self._body)} coordinate "
                f"argument(s) but the launch iteration space is {n_expected}-D"
                + (
                    "; a launch with no grid runs once and its body takes no "
                    "arguments"
                    if n_expected == 0
                    else ""
                )
            )
        if not self.dims:
            self._body()
            return
        # air.launch is always 2-D; a 1-D grid pads with 1.
        counts = list(self.grid) + [1] * (2 - len(self.grid))
        open_launch_region(
            state, self.tensors, counts, lambda: self._body(*state.coords)
        )

    def _check_interface(self):
        outputs = self.outputs
        if not outputs:
            raise RuntimeError(
                "kernel writes no output; at least one air.api.ops.store(...) "
                "into a tensor is required"
            )
        first_output = self.tensors.index(outputs[0])
        if any(not t.is_output for t in self.tensors[first_output:]):
            raise RuntimeError(
                "output tensors must be declared after all input tensors; the "
                "interface order is "
                f"{[(t.name, 'out' if t.is_output else 'in') for t in self.tensors]} "
                "and the XRT invocation passes inputs first, then outputs"
            )

    def mlir(self):
        return str(self.build())

    # -- compilation --------------------------------------------------------

    def compile(self, target=None, verbose=False, output_format="xclbin", **kwargs):
        """Compile through the XRT backend, returning a callable kernel."""
        module = self.build(target=target)

        from air.backend.xrt import XRTBackend

        backend = XRTBackend(
            verbose=verbose,
            omit_while_true_loop=False,
            output_format=output_format,
            instance_name=kwargs.pop("instance_name", self.name),
            target_device=self.target,
            runtime_loop_tiling_sizes=kwargs.pop("runtime_loop_tiling_sizes", [4, 4]),
            **kwargs,
        )
        try:
            compiled = backend.compile(module)
        except Exception as e:
            if "exceeded available memory" in str(e):
                raise _annotate_l1_failure(e, self._l1_peak) from e
            raise _annotate_placement_failure(e) from e
        return CompiledKernel(self, backend, compiled)


def _annotate_l1_failure(error, l1_peak):
    """Explain an AIE out-of-memory failure in terms of the DSL's tile size.

    The declared buffers can fit in L1 and the design still not place, because
    the pipeline ping-pongs L1 buffers -- so the figure that has to fit is
    roughly twice what air.alloc asked for. Say that, with the numbers.
    """
    from ._trace import L1_BYTES

    return type(error)(
        f"{str(error)}\n\n"
        f"air.api hint: this herd declares {l1_peak / 1024:.1f} KB of L1 buffers "
        f"per core, against {L1_BYTES / 1024:.0f} KB available. The pipeline "
        "ping-pongs L1 buffers, so budget for roughly twice the declared "
        f"figure ({2 * l1_peak / 1024:.1f} KB here). Reduce the tile size."
    )


def _annotate_placement_failure(error):
    """Turn a shim-capacity placement failure into an actionable message.

    A herd whose cores each stream operands straight from L3 needs a shim DMA
    channel per core per tensor, so widening the herd eventually fails deep in
    the AIE placer with an error that says nothing about the DSL. Point at the
    knob that fixes it.
    """
    text = str(error)
    if "ShimNOCTile" in text or "No valid placement found" in text:
        return type(error)(
            f"{text}\n\n"
            "air.api hint: this usually means the herd is too wide for the "
            "number of L3 tensors it streams -- each core needs a shim DMA "
            "channel per tensor. Pass a smaller shape= to air.herd(), e.g. "
            "shape=(1, 2)."
        )
    return error


class CompiledKernel:
    """A compiled kernel; calling it runs on the device."""

    def __init__(self, launch_ctx, backend, compiled):
        self.launch = launch_ctx
        self.backend = backend
        self.compiled = compiled
        self.search_space = launch_ctx.search_space

    def __call__(self, *args):
        import filelock

        expected = self.launch.inputs
        if len(args) != len(expected):
            raise TypeError(f"kernel takes {len(expected)} input(s), got {len(args)}")
        for arr, t in zip(args, expected):
            if tuple(arr.shape) != t.shape:
                raise ValueError(
                    f"input '{t.name}' has shape {t.shape} but got {tuple(arr.shape)}"
                )
            if arr.dtype != np.dtype(t.dtype.np_dtype):
                raise ValueError(
                    f"input '{t.name}' has dtype {np.dtype(t.dtype.np_dtype)} but "
                    f"got {arr.dtype}"
                )

        outputs = [
            np.zeros(t.shape, dtype=t.dtype.np_dtype) for t in self.launch.outputs
        ]
        with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
            fn = self.backend.load(self.compiled)
            results = fn(*args, *outputs)
        # XRT hands buffers back flat; restore the declared tensor shape so the
        # caller gets the array it asked for rather than a 1-D view of it.
        results = [
            np.asarray(r).reshape(t.shape)
            for r, t in zip(list(results)[len(args) :], self.launch.outputs)
        ]
        return results[0] if len(results) == 1 else tuple(results)

    def unload(self):
        self.backend.unload()


def launch(grid=None, name="kernel", target=None):
    """Open a launch; claims every tensor declared since the last launch.

    `grid` is the launch's own iteration space -- one point is one replay of
    everything inside, with a segment's L2 staging refilled each time, so outer
    tiling goes here. `air.segment` and `air.herd` each take their own.

    `target` names an NPU generation, or "auto"/None to use the installed one.
    """
    return LaunchContext(grid=grid, name=name, target=target)


def compile(launch_ctx, **kwargs):
    """Functional form of ``launch.compile(...)``."""
    return launch_ctx.compile(**kwargs)
