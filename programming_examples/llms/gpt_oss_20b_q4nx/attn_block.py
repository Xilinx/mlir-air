# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# GPT-OSS decode attention, laid out as FastFlowLM's layer.xclbin lays it out
# (FLM_Xclbin/GPTOSS/gpt_npu_bin/iron/{flowArch_aie,attn,aux_memtile}.py,
# FLM_DLL/detail/gpt_oss_npu/gpt_oss_npu_sequence.cpp):
#
#   RoPE core (1, 4): q/k/v bias, then RoPE (a YaRN table from the host).
#   k and v go to the DDR KV cache at this token's slot; the cache is
#   [position][4 KV heads][64] in four regions (K03, V03, K47, V47), a 128-slot
#   ring for a sliding-window layer. Attention does not care about the order of
#   its keys, so the ring is read back from slot 0, and the window is just "at
#   most 128 keys".
#   The cache comes back through memtiles 0 (K/V 0-3) and 7 (K/V 4-7), which put
#   each 16-key block into the attention kernel's layout for its unit.
#   Four attention units of 16 query heads (2 KV heads) each, a qk core and a kv
#   core: (0,2)/(0,3), (0,4)/(0,5), (7,2)/(7,3), (7,4)/(7,5). The score block
#   passes between the two through shared L1, and so do the sinks, which reach
#   the kv core (each core takes two streams). q arrives through the aux
#   memtile (column 6), which also puts o back into natural head order.
#
# The context length is fixed at build time here.
import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import filelock
import numpy as np
import torch
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kernel_tests"))

import common  # noqa: E402
import gpt_oss_20b_q4nx_weights as g  # noqa: E402
from air import api as air  # noqa: E402
from air.api import bf16, f32, i32  # noqa: E402
from air.backend.xrt import XRTBackend  # noqa: E402

DH = g.DH
DQ, DK = g.N_Q_HEADS * DH, g.N_KV_HEADS * DH
QKV = DQ + 2 * DK
ROPE_W = DH + QKV  # cos[32] sin[32], then the q|k|v bias
NCU = 4
QH = 16  # query heads per unit
SLAB = 4 * DH  # one position of one cache region: 4 KV heads
BLK = 2 * 16 * DH  # a unit's K or V block: 2 KV heads x 16 keys
S_BLK = 16 * 16 + 64  # scores, then c[16] and m[16] as floats
WINDOW = g.SLIDING_WINDOW
# FLM's unit placement: (column, qk row); the kv core sits one row up
CU_LOC = [(0, 2), (0, 4), (7, 2), (7, 4)]


def geometry(L, sliding):
    """Cache slots, this token's slot, and the keys attention reads. Attention
    reads whole 16-key blocks, so the cache has room for a whole last block;
    slots no token has written yet hold zeros."""
    slots = WINDOW if sliding else (L + 15) // 16 * 16
    pos = L - 1
    slot = pos % WINDOW if sliding else pos
    nkeys = min(L, WINDOW) if sliding else L
    return slots, slot, nkeys


def build_module(L, sliding):
    slots, slot, nkeys = geometry(L, sliding)
    nb = (nkeys + 15) // 16
    rope = air.extern("rope_compute", link_with="rope.o", scalars=[i32])
    qk = air.extern("gpt_attn_qk_blk", link_with="attn.o", scalars=[i32, i32, i32])
    kv = air.extern("gpt_attn_kv_blk", link_with="attn.o", scalars=[i32])
    fin = air.extern("gpt_attn_kv_fin", link_with="attn.o")

    QKVT = air.tensor([QKV], bf16)
    RW = air.tensor([ROPE_W + DH], bf16)  # rope table + bias, then the 64 sinks
    # regions K03, V03, K47, V47; this token's slot is written in place
    KV = air.tensor([4, slots, SLAB], bf16, inout=True)
    O = air.tensor([DQ], bf16)

    qkv_in = air.channel("QkvIn")
    rw_in = air.channel("RwIn")
    q_out = air.channel("QOut")
    # k and v share the RoPE core's second MM2S as two packet flows, as in FLM
    k_out = air.channel("KOut", channel_type="npu_dma_packet")
    v_out = air.channel("VOut", channel_type="npu_dma_packet")
    sink_in = air.channel("SinkIn", size=[2])
    q_cu = air.channel("QCu", size=[NCU])
    # one put per channel per round, so ping-pong pairs each put with its buffer
    sink_cu = air.channel("SinkCu", size=[NCU])  # to the kv cores
    k_rd = air.channel("KRd", size=[2])  # cache regions K03, K47 back in
    v_rd = air.channel("VRd", size=[2])
    k_cu = air.channel("KCu", size=[NCU])
    v_cu = air.channel("VCu", size=[NCU])
    o_cu = air.channel("OCu", size=[NCU])
    o_out = air.channel("OOut")

    with air.launch(name="attn") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    qkv_in.put(QKVT)
                    rw_in.put(RW[0:ROPE_W])
                    for half in range(2):
                        at = ROPE_W + half * 2 * QH
                        sink_in.put(RW[at : at + 2 * QH], indices=[half])
                    # this token's k and v into their slot of each region
                    for half in range(2):
                        k_out.get(KV[2 * half, slot, :])
                        v_out.get(KV[2 * half + 1, slot, :])
                    # then the cache back, from slot 0
                    for half in range(2):
                        k_rd.put(KV[2 * half, 0 : nb * 16, :], indices=[half])
                        v_rd.put(KV[2 * half + 1, 0 : nb * 16, :], indices=[half])
                    o_out.get(O)

                    # aux memtile: q and the sinks to each unit, o back in order
                    qm = air.alloc([DQ], bf16, scope=seg.private(), column=6)
                    om = air.alloc([DQ], bf16, scope=seg.private(), column=6)
                    q_out.get(qm)
                    for c in range(NCU):
                        # natural [16 heads][64] -> [half][dchunk][head][d]
                        q_cu.put(
                            qm.reshape(NCU, 2, 8, 8, 8)[
                                c : c + 1, :, :, :, :
                            ].transpose(0, 1, 3, 2, 4),
                            indices=[c],
                        )
                    for c in range(NCU):
                        o_cu.get(
                            om.reshape(NCU, 2, 8, 8, 8)[
                                c : c + 1, :, :, :, :
                            ].transpose(0, 1, 3, 2, 4),
                            indices=[c],
                        )
                    o_out.put(om)

                    # cache memtiles: a unit's 16-key block in kernel layout
                    for half, col in ((0, 0), (1, 7)):
                        km = air.alloc(
                            [16, SLAB], bf16, scope=seg.private(), column=col
                        )
                        vm = air.alloc(
                            [16, SLAB], bf16, scope=seg.private(), column=col
                        )
                        # the two units' sinks go out with their cache
                        sm = air.alloc([2 * QH], bf16, scope=seg.private(), column=col)
                        sink_in.get(sm, indices=[half])
                        for u in range(2):
                            sink_cu.put(
                                sm[u * QH : (u + 1) * QH], indices=[2 * half + u]
                            )
                        for b in air.sequential(0, nb):
                            k_rd.get(km, indices=[half])
                            v_rd.get(vm, indices=[half])
                            for u in range(2):
                                c = 2 * half + u
                                # [pos][(kvh, dc)][d] -> [(kvh, dc)][pos][d],
                                # pos = khalf*8 + key; (kvh, dc) merge as kvh
                                # steps 8 dchunks of 8
                                kh = km.reshape(16, 32, 8)[:, 16 * u : 16 * u + 16, :]
                                k_cu.put(kh.transpose(1, 0, 2), indices=[c])
                                # -> [khalf][(kvh, dc)][key][d]
                                vh = vm.reshape(2, 8, 32, 8)[
                                    :, :, 16 * u : 16 * u + 16, :
                                ]
                                v_cu.put(vh.transpose(0, 2, 1, 3), indices=[c])

                    with air.herd([range(1)], name="rope", shape=(1,), at=(1, 4)) as hr:

                        @hr.body
                        def _(tx):
                            qkv = air.alloc([QKV], bf16, scope=hr.private())
                            rw = air.alloc([ROPE_W], bf16, scope=hr.private())
                            q = air.alloc([DQ], bf16, scope=hr.private())
                            k = air.alloc([DK], bf16, scope=hr.private())
                            v = air.alloc([DK], bf16, scope=hr.private())
                            qkv_in.get(qkv)
                            rw_in.get(rw)
                            rope(q, k, v, qkv, rw, 0)
                            q_out.put(q)
                            k_out.put(k)
                            v_out.put(v)

                    def unit(c, col, row):
                        s = air.alloc([S_BLK], bf16, scope=seg.shared())
                        sink = air.alloc([QH], bf16, scope=seg.shared())

                        with air.herd(
                            [range(1)], name=f"qk{c}", shape=(1,), at=(col, row)
                        ) as hq:

                            @hq.body
                            def _(tx):
                                q = air.alloc([16 * DH + QH], bf16, scope=hq.private())
                                kb = air.alloc([BLK], bf16, scope=hq.private())
                                m = air.alloc([16], f32, scope=hq.private())
                                cc = air.alloc([16], f32, scope=hq.private())
                                q_cu.get(q[0 : 16 * DH], indices=[c])
                                # the kv core receives the sinks and shares them
                                q[16 * DH : 16 * DH + QH] = sink[:]
                                for b in air.sequential(0, nb):
                                    k_cu.get(kb, indices=[c])
                                    qk(q, kb, m, cc, s, b, nkeys, 0)

                        with air.herd(
                            [range(1)], name=f"kv{c}", shape=(1,), at=(col, row + 1)
                        ) as hk:

                            @hk.body
                            def _(tx):
                                vb = air.alloc([BLK], bf16, scope=hk.private())
                                y = air.alloc([16 * DH], f32, scope=hk.private())
                                l = air.alloc([16], f32, scope=hk.private())
                                ml = air.alloc([16], f32, scope=hk.private())
                                sk = air.alloc([QH], bf16, scope=hk.private())
                                o = air.alloc([16 * DH], bf16, scope=hk.private())
                                sink_cu.get(sk, indices=[c])
                                sink[:] = sk[:]
                                for b in air.sequential(0, nb):
                                    v_cu.get(vb, indices=[c])
                                    kv(s, vb, y, l, ml, b)
                                fin(y, l, sk, ml, o)
                                o_cu.put(o, indices=[c])

                    for c, (col, row) in enumerate(CU_LOC):
                        unit(c, col, row)

    return launch.build(target="npu2")


def bf(t):
    return t.to(torch.bfloat16).float()


def reference(qkv, rw, kv, L, sliding):
    """o for this token, and the cache after it is appended."""
    slots, slot, nkeys = geometry(L, sliding)
    bias = rw[DH:ROPE_W]
    sinks = rw[ROPE_W:]
    cos, sin = rw[None, : DH // 2], rw[None, DH // 2 : DH]
    x = bf(qkv + bias)
    q = bf(g.apply_rope(x[:DQ].reshape(-1, DH), cos, sin))
    k = bf(g.apply_rope(x[DQ : DQ + DK].reshape(-1, DH), cos, sin))
    v = x[DQ + DK :].reshape(-1, DH)
    kv = kv.clone()
    kv[0, slot], kv[2, slot] = k[:4].ravel(), k[4:].ravel()
    kv[1, slot], kv[3, slot] = v[:4].ravel(), v[4:].ravel()
    keys = torch.cat(
        (kv[0, :nkeys].reshape(nkeys, 4, DH), kv[2, :nkeys].reshape(nkeys, 4, DH)), 1
    )
    vals = torch.cat(
        (kv[1, :nkeys].reshape(nkeys, 4, DH), kv[3, :nkeys].reshape(nkeys, 4, DH)), 1
    )
    kr = keys.permute(1, 0, 2).repeat_interleave(8, 0)  # [64 heads, n, 64]
    vr = vals.permute(1, 0, 2).repeat_interleave(8, 0)
    sc = (q[:, None, :] @ kr.transpose(1, 2))[:, 0] / math.sqrt(DH)
    p = torch.softmax(torch.cat((sc, sinks[:, None]), -1), -1)[:, :-1]
    o = (p[:, None, :] @ vr)[:, 0]
    return o.ravel(), kv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--print-module-only", action="store_true")
    ap.add_argument("--L", type=int, default=37)
    ap.add_argument("--sliding", action="store_true")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-format", choices=["xclbin", "elf"], default="xclbin")
    a = ap.parse_args()
    module = build_module(a.L, a.sliding)
    if a.print_module_only:
        print(module)
        return 0
    common.compile_kernel("attn")
    common.compile_kernel("rope", shared=True)

    w = g.GptOssQ4nx()
    att = w.attn(a.layer)
    gen = torch.Generator().manual_seed(a.seed)
    slots, slot, nkeys = geometry(a.L, a.sliding)
    qkv = bf(2 * torch.randn(QKV, generator=gen))
    cos, sin = g.rope_cos_sin([a.L - 1])
    rw = torch.cat(
        (
            bf(cos[0]),
            bf(sin[0]),
            bf(torch.cat((att["bq"], att["bk"], att["bv"]))),
            bf(att["sinks"]),
        )
    )
    kv = bf(torch.randn(4, slots, SLAB, generator=gen))
    kv[:, min(a.L, slots) :] = 0  # not written yet
    want_o, want_kv = reference(qkv, rw, kv, a.L, a.sliding)

    backend = XRTBackend(
        output_format=a.output_format,
        instance_name="attn",
        stack_size=4096,
        omit_while_true_loop=False,
        use_lock_race_condition_fix_v2=True,
    )
    art = backend.compile(module)
    kvb = kv.numpy().astype(bfloat16)
    with filelock.FileLock(os.path.join(tempfile.gettempdir(), "npu.lock")):
        f = backend.load(art)
        res = f(
            qkv.numpy().astype(bfloat16),
            rw.numpy().astype(bfloat16),
            kvb,
            np.zeros(DQ, bfloat16),
        )
    got_o = torch.from_numpy(res[3].view(bfloat16).astype(np.float32))
    got_kv = torch.from_numpy(res[2].view(bfloat16).astype(np.float32)).reshape(
        4, slots, SLAB
    )
    cs = float(torch.nn.functional.cosine_similarity(got_o, want_o, dim=0))
    kv_rel = float((got_kv - want_kv).norm() / want_kv.norm())
    per_unit = [
        round(float(torch.nn.functional.cosine_similarity(u, v, dim=0)), 5)
        for u, v in zip(got_o.reshape(NCU, -1), want_o.reshape(NCU, -1))
    ]
    print(
        f"o vs reference: cos {cs:.5f} (per unit {per_unit}); "
        f"cache after append: rel err {kv_rel:.4f}"
    )
    # random q/k give scores far larger than the model's, and FLM's kernels
    # round them to bf16 before the exp LUT
    ok = cs > 0.998 and kv_rel < 1e-2
    print("PASS!" if ok else "failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
