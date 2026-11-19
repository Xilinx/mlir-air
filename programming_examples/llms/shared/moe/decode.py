# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The MoE half of a decode wave on FastFlowLM's Qwen3.5 tile map: its
# projections on the GEMV array, the assemble memtile, the RMS core, the router
# GEMV, router and GLU cores on column 5, and memtile 5 staging the RMS core's
# traffic. A wave is PER_WAVE layers; every core loops over them. The models
# keep their mixer cores and the kernels they declare.
#
# A shared expert adds its gate to the RMS core, its gate/up rounds to the GLU
# core and its down rounds to the RMS core.
from air import api as air
from air.api import bf16, f32, i32

from ..decode.array import Phase
from ..decode.layer import herd, l2, wave_loop


def moe_l1(seg, n_experts):
    """The router GEMV's logits, the top-k scores, and the GLU core's go for
    the router's second turn."""
    logits = air.alloc([n_experts], f32, scope=seg.shared())
    scores = air.alloc([16], f32, scope=seg.shared())
    turn_go = air.alloc([16], f32, scope=seg.shared())
    return logits, scores, turn_go


def memtile5(seg, pw, d, n_in, sides, m_in, to_rms, glu_y, xs_ch):
    """The RMS core's inputs and its side outputs: `n_in` inputs from the host,
    the MoE output, then one buffer per channel in `sides` from xs_ch."""
    # The memtiles' buffers serve every layer: allocated in the layer loop,
    # each would be doubled for ping-pong
    ins = [l2(seg, [d], 5) for _ in range(n_in)]
    moe5 = l2(seg, [d], 5)
    outs = [l2(seg, [d], 5) for _ in sides]
    for _j in wave_loop(pw):
        for b in ins:
            m_in.get(b)
            to_rms.put(b)
        glu_y.get(moe5)
        to_rms.put(moe5)
        for b in outs:
            xs_ch.get(b)
        for b, c in zip(outs, sides):
            c.put(b)


def assemble(seg, arr, pw, col, sizes, n_x, n_y, xn_ch, to_asm, broadcast):
    """The assemble memtile: xa takes the RMS core's `n_x` normed x a layer;
    the next buffers, of `sizes`, take one GLU or mixer output each, and the
    last `n_y` times. `broadcast(xa, bufs)` lists each column's inputs."""
    # One buffer for the normed x and one for the turns' GLU outputs: each is
    # broadcast before the next fills it (and a buffer filled more than once a
    # layer keeps the loop from being doubled for ping-pong, which would take
    # more locks than the memtile has)
    xa = l2(seg, [sizes[0]], col)
    bufs = [l2(seg, s if isinstance(s, list) else [s], col) for s in sizes[1:]]
    for _j in wave_loop(pw):
        for _ in range(n_x):
            xn_ch.get(xa)
        for b in bufs[:-1]:
            to_asm.get(b)
        for _t in range(n_y):
            to_asm.get(bufs[-1])
        arr.broadcast(lambda i: broadcast(xa, bufs, i))


def rms_core(arr, pw, at, d, rg_o, n_xn2, chans, k, shared=None):
    """Per layer: norm x, take o's rounds and add them, norm again, then add
    the MoE output. x carries over from layer to layer; only the wave's first
    takes it from the host. `shared` = (its down rounds, its dest) adds the
    shared expert's gate."""
    to_rms, xn_ch, xs_ch, out_dest = chans
    names = ["x", "w1", "w2"] + ["wsg"] * bool(shared) + ["xn", "xn2", "mix"]
    names += ["sdn"] * bool(shared) + ["moe", "y", "xin"]
    with herd("rms", at) as h:

        @h.body
        def _(tx):
            p = h.private()
            b = {n: air.alloc([d], bf16, scope=p) for n in names}
            u = air.alloc([arr.round], bf16, scope=p)
            if shared:
                gsc = air.alloc([16], bf16, scope=p)
            x, xn, xn2, mix, y = b["x"], b["xn"], b["xn2"], b["mix"], b["y"]
            for j in wave_loop(pw):
                ins = ("xin", "w1", "w2") + ("wsg",) * bool(shared)
                for n in ins:
                    to_rms.get(b[n])
                with air.ops.branch(j == 0):
                    x[:] = b["xin"][:]
                k.rms_norm(xn, x, b["w1"], d)
                xn_ch.put(xn)
                if shared:
                    xs_ch.put(xn)
                for jj in air.sequential(0, rg_o):
                    arr.out_y.get(u, indices=[0, out_dest])
                    k.copy_r(mix, u, jj)
                k.res_add(y, x, mix, d)
                k.rms_norm(xn2, x, b["w2"], d)
                if shared:
                    # done with xn2 before it goes
                    k.sg_dot(xn2, b["wsg"], gsc, d)
                for _t in range(n_xn2):
                    xn_ch.put(xn2)
                xs_ch.put(xn2)
                if shared:
                    for jj in air.sequential(0, shared):
                        arr.out_y.get(u, indices=[0, out_dest])
                        k.copy_r(b["sdn"], u, jj)
                    k.sg_apply(gsc, b["sdn"], d)
                    xs_ch.put(b["sdn"])
                to_rms.get(b["moe"])
                k.res_add(y, x, b["moe"], d)
                xs_ch.put(y)


def router_gemv(x, blk, lg, logits, to_ab, ab_w, k, nrb, ncb):
    """One layer's router logits: x times the router weights, block by block,
    into the shared logits."""
    to_ab.get(x)
    k.r_zero(lg)
    for rb in air.sequential(0, nrb):
        for cb in air.sequential(0, ncb):
            ab_w.get(blk)
            k.r_mvm(blk, x, lg, rb, cb)
    logits[:] = lg[:]


def router_core(arr, pw, at, ebase, lbase, n_experts, ncol, turns, wd, we, l1, k):
    """Each layer's dense weights, then its turns of experts, on the array's
    shim channels."""
    logits, scores, turn_go = l1
    with herd("router", at, params=[ebase, lbase]) as h:

        @h.body
        def _(tx):
            p = h.private()
            lg = air.alloc([n_experts], f32, scope=p)
            idx = air.alloc([16], i32, scope=p)
            sc = air.alloc([16], f32, scope=p)
            go = air.alloc([16], f32, scope=p)
            for j in wave_loop(pw):
                k.r_slot(idx, 8, lbase, j)
                arr.put_weights(
                    lambda i, hh: [wd[idx[8], i, :, 2 * hh : 2 * hh + 2, :]]
                )
                lg[:] = logits[:]
                k.r_topk(lg, idx, sc, ebase, j)
                scores[:] = sc[:]
                for t in range(turns):
                    # turn 1 waits for the GLU core to finish turn 0's gate/up
                    if t:
                        go[:] = turn_go[:]
                    arr.put_weights(
                        lambda i, hh, t=t: [
                            we[idx[ncol * t + i], :, 2 * hh : 2 * hh + 2, :]
                        ]
                    )


def glu_core(arr, pw, at, dims, rounds, l1, chans, k, glu_dest, shared=None):
    """Per layer and turn: each expert's gate/up rounds through the GLU, its
    output to the array, then the down rounds weighted into the sum. `shared` =
    its gate/up rounds adds the shared expert's GLU first and its down output
    (from to_glu) to the sum."""
    d, i_exp, ncol = dims
    slices, rg_dn, turns = rounds
    _, scores, turn_go = l1
    to_asm, glu_y, to_glu = chans
    with herd("glu", at) as h:

        @h.body
        def _(tx):
            p = h.private()
            u = air.alloc([arr.round], bf16, scope=p)
            g = air.alloc([arr.round], bf16, scope=p)
            if shared:
                hsl = air.alloc([i_exp], bf16, scope=p)
            hh = air.alloc([ncol * i_exp], bf16, scope=p)
            if shared:
                sdn = air.alloc([d], bf16, scope=p)
            acc = air.alloc([d], f32, scope=p)
            y = air.alloc([d], bf16, scope=p)
            sc = air.alloc([16], f32, scope=p)
            for _j in wave_loop(pw):
                if shared:
                    for jj in air.sequential(0, shared):
                        arr.out_y.get(u, indices=[0, glu_dest])
                        k.glu_s(hsl, u, jj)
                    to_asm.put(hsl)
                k.acc_zero(acc, d)
                sc[:] = scores[:]
                for t in range(turns):
                    for s_ in air.sequential(0, slices):
                        # every round lands in u
                        arr.out_y.get(u, indices=[0, glu_dest])
                        k.copy_up(g, u)
                        arr.out_y.get(u, indices=[0, glu_dest])
                        k.glu_e(hh, g, u, s_)
                    to_asm.put(hh)
                    if t + 1 < turns:
                        turn_go[:] = sc[:]
                    for rg in air.sequential(0, rg_dn):
                        arr.out_y.get(u, indices=[0, glu_dest])
                        k.acc_r(acc, u, sc, t, rg)
                if shared:
                    to_glu.get(sdn)
                    k.out_k(y, sdn, acc, d)
                else:
                    k.out_k(y, acc, d)
                glu_y.put(y)


def moe_phases(step, dims, rounds, dests, fresh=True, shared=None):
    """The array's MoE projections of a layer: the shared expert's gate/up and
    down when `shared` = their round counts, then each turn's experts."""
    d, i_exp = dims
    rg_gu, rg_dn, turns, ncb_x, ncb_h = rounds
    glu, rms = dests
    if shared:
        rg_sgu, rg_sdn = shared
        yield Phase(d, rg_sgu * ncb_x // 2, ncb_x, step, glu)
        yield Phase(i_exp, rg_sdn * ncb_h // 2, ncb_h, step, rms, xbuf=1)
    for _t in range(turns):
        yield Phase(d, rg_gu * ncb_x // 2, ncb_x, step, glu, fresh=fresh)
        yield Phase(i_exp, rg_dn * ncb_h // 2, ncb_h, step, glu, xbuf=1)


def layer_misc(m_in, ab_w, m, misc, lay, j, last, ab):
    """A layer's norm weights to the RMS core (x's slot too after the wave's
    first layer, which takes x from the host) and its router block from `ab`."""
    if j == 0:
        m_in.put(m[lay, misc["w1"][0] : misc[last][1]])
    else:
        m_in.put(m[lay, 0 : misc[last][1]])
    ab_w.put(m[lay, misc[ab][0] : misc["router"][1]])
