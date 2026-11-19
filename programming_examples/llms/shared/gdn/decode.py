# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The Gated DeltaNet mixer of a decode wave on FastFlowLM's Qwen3.5 tile map:
# memtile 4 staging the state, the alpha / beta core, the conv core and the
# gate core. `g` is the model's GDN layer module (its head counts, sizes and
# core positions); the models declare the channels and kernels and pass them.
from air import api as air
from air.api import bf16, f32

from ..decode.layer import herd, l2, wave_loop


def state_memtile(seg, ngw, g, chans):
    """Memtile 4: each GDN layer's state, head by head, out to the recurrence
    column's R cores and back."""
    s_l3, s_in, s_out, s_l3o = chans
    for _j in wave_loop(ngw):
        for _h in wave_loop(g.NV):
            sb = l2(seg, [g.R, g.HD * 32], 4, f32)
            s_l3.get(sb)
            for r in range(g.R):
                s_in.put(sb[r, :], indices=[r])
            air.dealloc(sb)
            ob = l2(seg, [g.R, g.HD * 32], 4, f32)
            for r in range(g.R):
                s_out.get(ob[r, :], indices=[r])
            s_l3o.put(ob)
            air.dealloc(ob)


def ab_core(pw, ngw, g, ab_sh, chans, k, router=None):
    """Alpha / beta of every layer of the wave (an attention layer's weights
    are zero and its result unused), into the shared L1 for the conv core.
    `router(p)` allocates in the core and returns its per-layer step
    `(j, x, blk)`, run after alpha / beta on the same x."""
    to_ab, ab_w = chans
    with herd("ab", g.AB_AT) as h:

        @h.body
        def _(tx):
            p = h.private()
            x = air.alloc([g.D], bf16, scope=p)
            adt = air.alloc([4 * g.NV], bf16, scope=p)
            blk = air.alloc([g.CB * g.RB], bf16, scope=p)
            ab = air.alloc([2 * g.NV], f32, scope=p)
            after = router(p) if router else None
            for j in wave_loop(pw):
                to_ab.get(x)
                ab_w.get(blk)
                adt[:] = blk[0 : 4 * g.NV]
                k.ab_zero(ab)
                for rb in air.sequential(0, 2 * g.NV // g.RB):
                    for cb in air.sequential(0, g.NCB_X):
                        ab_w.get(blk)
                        k.ab_mvm(blk, x, ab, rb, cb)
                k.ab_act(ab, adt)
                with air.ops.branch(j < ngw):
                    ab_sh[:] = ab[:]
                if after:
                    after(j, x, blk)


def conv_core(arr, ngw, g, ab_sh, chans, k, dest):
    """Per GDN layer: the qkv rounds through the causal conv (q and k then
    L2-normed), then each value head's packed q, k, v and alpha / beta to the
    recurrence column."""
    cw_in, cw_out, kvq_ch = chans
    nq = g.NK * g.HD // 512
    with herd("conv", g.CONV_AT) as h:

        @h.body
        def _(tx):
            p = h.private()
            u = air.alloc([arr.round], bf16, scope=p)
            hx_ = air.alloc([7 * g.HD], bf16, scope=p)
            ho = air.alloc([3 * g.HD], bf16, scope=p)
            cv = air.alloc([g.QKV], bf16, scope=p)
            ab = air.alloc([2 * g.NV], f32, scope=p)
            kvq = air.alloc([3 * g.SLOT], bf16, scope=p)
            for _j in wave_loop(ngw):
                for lo, hi, norm in (
                    (0, nq, 1),
                    (nq, 2 * nq, 0),
                    (2 * nq, g.RG_QKV, None),
                ):
                    for rd in air.sequential(lo, hi):
                        arr.out_y.get(u, indices=[0, dest])
                        for kk in range(4):
                            c = rd * 4 + kk
                            cw_in.get(hx_)
                            k.conv_k(hx_, u, cv, ho, c)
                            cw_out.put(ho)
                            if norm is not None:
                                k.l2n(cv, c, norm)
                ab[:] = ab_sh[:]
                for hh in air.sequential(0, g.NV):
                    k.pack(cv, ab, kvq, hh)
                    kvq_ch.put(kvq, indices=[0])


def gate_core(arr, pw, ngw, g, dq, chans, k, dest):
    """Per layer: z from the array, then the DeltaNet norm and SiLU gate (GDN
    layers) or the sigmoid gate (attention layers) on the mixer output."""
    o_in, to_asm = chans
    with herd("gate", g.GATE_AT) as h:

        @h.body
        def _(tx):
            p = h.private()
            u = air.alloc([arr.round], bf16, scope=p)
            z = air.alloc([dq], bf16, scope=p)
            wn = air.alloc([g.HD], bf16, scope=p)
            o = air.alloc([dq], bf16, scope=p)
            on = air.alloc([dq], bf16, scope=p)
            for j in wave_loop(pw):
                for jj in air.sequential(0, g.RG_Z):
                    arr.out_y.get(u, indices=[0, dest])
                    k.z_rd(z, u, jj)
                o_in.get(wn)
                o_in.get(o)
                with air.ops.branch(j < ngw):
                    k.dn(on, o, wn)
                    k.gsilu(on, z, on, dq)
                with air.ops.branch(j >= ngw):
                    k.gsig(on, z, o, dq)
                to_asm.put(on)
