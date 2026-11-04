# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MoE half of a decode layer on FastFlowLM's GEMV array.

Three cores next to the array, as FLM places them in its GPT-OSS layer:

  router master   the router GEMV, into logits it shares with the slave
  router slave    sorts the logits and starts the top-k experts' weight
                  transfers itself: ``W[idx[i], ...]`` is an L3 slice whose
                  offset the core chose. Gate/up go first, then the softmax;
                  the down transfers wait for the GLU core. It then combines
                  the experts' down rows.
  GLU             gate/up rounds in, activated rows back to the array

Expert rank i runs on array column i, so top_k equals the array's columns.
An expert's weights are ``[row group, col block, row block, bytes]``: the
gate/up row groups (interleaved), then the down ones.

The router kernels take -DMOE_N_EXPERTS and -DMOE_TOP_K; the GLU kernel is
the model's.
"""

from dataclasses import dataclass
from pathlib import Path

from air import api as air
from air.api import bf16, f32, i32

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"


@dataclass
class MoeShape:
    n_experts: int
    top_k: int
    d_pad: int  # the input width the router and experts read
    router_blocks: int  # router GEMV blocks of n_experts x 64
    rg_gu: int  # gate/up row groups of an expert
    rg_dn: int  # down row groups of an expert
    col_block: int  # combine width: one row block of every expert

    @property
    def rg(self):
        return self.rg_gu + self.rg_dn


def router_kernels(obj="router.o"):
    """The router kernels, by name."""
    return {
        n: air.extern(n, link_with=obj, scalars=s)
        for n, s in (
            ("router_bias_init", []),
            ("router_mvm", [i32]),
            ("router_flush", []),
            ("router_sort", [i32]),
            ("router_softmax", []),
            ("moe_combine", []),
        )
    }


def _herd(name, at, params=()):
    return air.herd([range(1)], name=name, shape=(1,), at=at, params=list(params))


class Moe:
    def __init__(self, shape, array, kernels, router_col=6):
        assert shape.top_k == array.cols, "expert rank i runs on array column i"
        self.s, self.a, self.k = shape, array, kernels
        self.col = router_col

    def shared_buffers(self, seg):
        """The logits (master to slave) and the GLU's done signal (to the
        slave). Each side touches each one in a single whole-buffer access."""
        self.logits = air.alloc([self.s.n_experts], bf16, scope=seg.shared())
        self.glu_done = air.alloc([32], bf16, scope=seg.shared())

    def master(self, x_get, wr):
        """Router GEMV: ``x_get(x)`` receives the normed input, ``wr`` brings
        the bias block, then the weight blocks."""
        s, k = self.s, self.k
        with _herd("router_master", (self.col, 4)) as h:

            @h.body
            def _(tx):
                p = h.private()
                x = air.alloc([s.d_pad], bf16, scope=p)
                w = air.alloc([s.n_experts * 64], bf16, scope=p)
                acc = air.alloc([s.n_experts], f32, scope=p)
                x_get(x)
                wr.get(w)
                k["router_bias_init"](w, acc)
                for j in air.sequential(0, s.router_blocks):
                    wr.get(w)
                    k["router_mvm"](w, x, acc, j)
                k["router_flush"](acc, self.logits)

    def slave(self, W, ebase, dest, out):
        """Top-k, the experts' transfers, and the combine of the ``dest``
        rounds into ``out``. ``ebase`` offsets the expert ids into W's
        leading dimension (e.g. a layer's first expert): an int, or a run-time
        value the herd takes as a parameter."""
        s, k, a = self.s, self.k, self.a

        def fetch(rg0, rg1, idx):
            a.put_weights(lambda i, h: [W[idx[i], rg0:rg1, :, 2 * h : 2 * h + 2, :]])

        params = [] if isinstance(ebase, int) else [ebase]
        with _herd("router_slave", (self.col, 3), params=params) as h:

            @h.body
            def _(tx):
                p = h.private()
                idx = air.alloc([s.n_experts], i32, scope=p)
                topw = air.alloc([16], bf16, scope=p)
                srt = air.alloc([s.n_experts], bf16, scope=p)
                tok = air.alloc([32], bf16, scope=p)
                d = air.alloc([a.round], bf16, scope=p)
                o = air.alloc([s.col_block], bf16, scope=p)
                # a single read of each shared buffer
                srt[:] = self.logits[:]
                k["router_sort"](srt, idx, ebase)
                # FLM issues the gate/up fetches before the softmax
                fetch(0, s.rg_gu, idx)
                k["router_softmax"](srt, topw)
                # and the down fetches once the GLU core is done
                tok[:] = self.glu_done[:]
                fetch(s.rg_gu, s.rg, idx)
                for rg in air.sequential(0, s.rg_dn):
                    a.out_y.get(d, indices=[0, dest])
                    k["moe_combine"](d, topw, o)
                    out.put(o)

    def glu(self, glu_kernel, dest, out, params=(), tail=None):
        """Gate/up rounds (two per slice) through ``glu_kernel`` to ``out``.
        ``tail(u)`` runs after, on the same core (e.g. forwarding the LM
        head's rounds)."""
        a = self.a
        with _herd("glu", (self.col, 2), params=params) as h:

            @h.body
            def _(tx):
                p = h.private()
                u = air.alloc([2 * a.round], bf16, scope=p)
                hh = air.alloc([a.round], bf16, scope=p)
                if tail is not None:
                    tail_buf = air.alloc([2 * a.round], bf16, scope=p)
                for sl in air.sequential(0, self.s.rg_gu // 2):
                    a.out_y.get(u[0 : a.round], indices=[0, dest])
                    a.out_y.get(u[a.round : 2 * a.round], indices=[0, dest])
                    glu_kernel(u, hh)
                    out.put(hh)
                self.glu_done[:] = hh[0:32]
                if tail is not None:
                    tail(u, tail_buf)
