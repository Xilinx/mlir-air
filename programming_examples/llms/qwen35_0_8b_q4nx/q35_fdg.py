# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decode on programming_examples/fused_decode_gdn: one dispatch per generated
token covers all 24 layers (18 Gated DeltaNet, 6 gated full attention) and the
lm_head.

The device layouts this file produces (the builder's, see
fused_decode_gdn/README.md):

  w      per decode wave (layer) the six phases' 32 x 256 blocks in the
         engine's cascade order, then the 18 lm_head waves. A projection block
         is the bundle's Q4K block, exact, in the engine's 5120-byte slot
         (q4k_dev); a Q8 tensor (lm_head, beta / alpha) is carried exactly as
         two affine-Q4 rows per row (q35_weights.q8_pair_rows).
  rms    per layer [input norm | post-attention norm], then per layer a
         768-word mixer slab (attention: [cos 32 | sin 32 | . | q_norm 256 |
         k_norm 256], cos / sin written per token), then the final norm.
  kvc    per layer [K positions x (head 0 | head 1) | V ...] (region-major).
  gdn_a, gdn_b  the GDN herd's chunk streams, per layer and core
         [reset | (X, 8 state units) x 4 heads | NOP]; a token reads one and
         writes the other, the next token the other way round.

Per token the host writes the embedding row and the attention layers' cos /
sin, swaps gdn_a / gdn_b, dispatches (the context length as scratchpad
parameters), reads the logits (row pairs added) and takes the argmax.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

HERE = Path(__file__).resolve().parent
EX = HERE.parents[1]
FDG = EX / "fused_decode_gdn"
FDEC = EX / "fused_decode"
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(EX)
]

from air_examples.llms.qwen35_0_8b_q4nx.q35_weights import q8_pair_rows  # noqa: E402

D, I, NV = 1024, 3584, 248320
LAYERS = 24
ATT = [L for L in range(LAYERS) if L % 4 == 3]
LIN = [L for L in range(LAYERS) if L % 4 != 3]
DH, ROPE_HALF = 256, 32
Q4K_TAG = np.array([0x51, 0x34, 0x4B, 0x21], np.uint8)  # 0x214B3451, little-endian
MAGIC = np.array([0x7AB1, 0xC0DE, 0x5EED], np.uint16)
OHIST, OTAPS, OPRM2 = 544, 1696, 3236


def builder(capacity):
    """The fused_decode_gdn builder module for the full ELF of `capacity`
    (its constants; nothing is built)."""
    os.environ.update(
        DECODE_MODEL="qwen3.5-0.8b",
        UNIFIED="1",
        LM_HEAD="0",
        NLAYERS="1",
        DECODE_GOLDEN="1",
        PROJ_RC_CACHE="1",
        DECODE_OUTPUT_FORMAT="elf",
        DECODE_GOLDEN_L=str(capacity),
        DECODE_DYNSEQ="1",
        DECODE_RT_ROUNDS="1",
    )
    for p in (str(FDEC), str(FDG)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(
        f"fdg_{capacity}", FDG / "fused_decode_gdn.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------- weight blocks


def pairs_nibbles(q):
    """Nibbles q (uint8 [..., 32 rows, 256 cols], 0..15) -> the kernels'
    column-pair layout, uint8 [..., 4096]: [row half][group][4][4 column pairs]
    [16 rows][2 columns], two nibbles a byte, low first."""
    sh = q.shape[:-2]
    n = len(sh)
    q = q.reshape(sh + (2, 16, 8, 4, 4, 2))  # [h][r][g][l4][jp][e]
    q = q.transpose(
        *range(n), n, n + 2, n + 3, n + 4, n + 1, n + 5
    )  # [h][g][l4][jp][r][e]
    return (q[..., 0] | (q[..., 1] << 4)).astype(np.uint8).reshape(sh + (4096,))


def q4k_dev(raw):
    """Bundle Q4K blocks (uint8 [..., 4736]: sc [8][32], m [8][32], nibbles
    [256 cols][16 B], d [32], dmin [32]) -> the engine's tagged Q4K slot (uint8
    [..., 5120]): sc | m | d | dmin | zeros | tag, then the nibbles in the
    column-pair layout (a permutation of the bundle's values)."""
    raw = np.asarray(raw)
    sh = raw.shape[:-1]
    out = np.zeros(sh + (5120,), np.uint8)
    out[..., 0:512] = raw[..., 0:512]
    out[..., 512:640] = raw[..., 4608:4736]
    out[..., 1020:1024] = Q4K_TAG
    # the bundle's nibbles [col][16 B] (byte k: rows 2k low, 2k + 1 high)
    nb = raw[..., 512:4608].reshape(sh + (256, 16))
    q = np.stack([nb & 15, nb >> 4], -1).reshape(sh + (256, 32))  # [col][row]
    out[..., 1024:] = pairs_nibbles(np.swapaxes(q, -1, -2))
    return out


def affine_dev(q, sc, mn):
    """Affine-Q4 rows (q uint8 [rows, K], scale and min [rows, K / 32]; rows a
    multiple of 32) -> uint8 [rows / 32, K / 256, 5120]: bf16 scale and min
    [group][row], then the nibbles in the column-pair layout."""
    rows, K = q.shape
    nbi, nbj = rows // 32, K // 256

    def by_group(x):  # [rows, K/32] -> [nbi, nbj, group, row] bf16 bytes
        x = np.asarray(x).reshape(nbi, 32, nbj, 8).transpose(0, 2, 3, 1)
        return (
            np.ascontiguousarray(x.astype(bfloat16))
            .view(np.uint8)
            .reshape(nbi, nbj, 512)
        )

    qb = (np.asarray(q, np.uint8) & 15).reshape(nbi, 32, nbj, 256).transpose(0, 2, 1, 3)
    return np.concatenate([by_group(sc), by_group(mn), pairs_nibbles(qb)], axis=2)


def q8_rows_dev(q, s, rows=None):
    """Q8 rows (int8 [n, K], scale [n, K / 32]) -> hi / lo affine row pairs
    (2n rows, zero-padded to `rows`), as engine blocks."""
    hq, hs, hm = q8_pair_rows(q, s)
    n2, K = hq.shape
    rows = rows or -(-n2 // 32) * 32
    Q = np.zeros((rows, K), np.uint8)
    S = np.zeros((rows, K // 32), np.float32)
    M = np.zeros((rows, K // 32), np.float32)
    Q[:n2], S[:n2], M[:n2] = hq, hs, hm
    out = affine_dev(Q, S, M)
    assert (
        not (out[..., 1020:1024] == Q4K_TAG).all(-1).any()
    ), "an affine block carries the Q4K tag"
    return out


def cascade(blocks, NCX, NCY, dual):
    """The engine's weight stream order for one phase's block matrix (uint8
    [nbi, nbj, 5120]): pack_q4k_cascade's iteration-major order (that of
    q4_0_codec.pack_q4k_cascade_fast, which the engine's egress assembles back
    into natural row order)."""
    nbi, nbj = blocks.shape[:2]
    nbi_pc = nbi // (NCX * NCY)
    assert nbi_pc * NCX * NCY == nbi
    steps = [(i, j) for i in range(nbi_pc) for j in range(nbj)]
    if dual:
        order = [
            (i, j, cy)
            for h in range(2)
            for (i, j) in steps
            for cy in range(h * (NCY // 2), (h + 1) * (NCY // 2))
        ]
    else:
        order = [(i, j, cy) for (i, j) in steps for cy in range(NCY)]
    rows, cols = [], []
    for cx in range(NCX):
        for i, j, cy in order:
            rows.append(i * (NCX * NCY) + cx * NCY + cy)
            cols.append(j)
    return blocks[rows, cols].reshape(-1)


def zero_q4k(nbi, nbj):
    return q4k_dev(np.zeros((nbi, nbj, 4736), np.uint8))


def layer_phases(Wt, L, fd):
    """The six phases' block matrices of layer L (uint8 [nbi, nbj, 5120])."""
    p = f"model.layers.{L}."
    blk = lambda name, rows, K: q4k_dev(Wt.blocks(name, rows, K))  # noqa: E731
    M = fd.M
    nb = M // 32
    if L in LIN:
        s_ = p + "linear_attn."
        qkv = blk(s_ + "qkv_proj.weight", 6144, D)
        z = blk(p + "self_attn.gate_proj.weight", 2048, D)
        qb, sb = Wt.q8(s_ + "ssm_beta_proj.weight", 16, D)
        qa, sa = Wt.q8(s_ + "ssm_alpha_proj.weight", 16, D)
        rows = []
        for h in range(16):
            for base in (0, 2048, 4096):
                r0 = (base + h * 128) // 32
                rows.append(qkv[r0 : r0 + 4])
            rows.append(z[h * 4 : h * 4 + 4])
            q8 = np.concatenate([qb[h : h + 1], qa[h : h + 1]])
            s8 = np.concatenate([sb[h : h + 1], sa[h : h + 1]])
            # rows [b hi, b lo, a hi, a lo, 0 x 28]: the herd reads b and a as
            # the sums of rows 512 + 513 and 514 + 515
            rows.append(q8_rows_dev(q8, s8, 32))
        mix = np.concatenate(rows + [zero_q4k(3 * nb - 16 * 17, 4)])
        o = blk(p + "linear_attn.ssm_out_proj.weight", D, 2048)
    else:
        s_ = p + "self_attn."
        qg = blk(s_ + "q_proj.weight", 4096, D)  # [q 2048 | gate 2048]
        k = blk(s_ + "k_proj.weight", 512, D)
        v = blk(s_ + "v_proj.weight", 512, D)
        mix = np.concatenate([qg[:64], k, v, qg[64:], zero_q4k(32 + nb, 4)])
        o = blk(s_ + "o_proj.weight", D, 2048)
    assert mix.shape[0] == 3 * nb
    g = blk(p + "mlp.gate_proj.weight", I, D)
    u = blk(p + "mlp.up_proj.weight", I, D)
    c = fd.GLU_CHUNK // 32  # [up | gate] in GLU_CHUNK-row slices
    gu = np.concatenate(
        [x[s * c : (s + 1) * c] for s in range(I // fd.GLU_CHUNK) for x in (u, g)]
    )
    down = blk(p + "mlp.down_proj.weight", D, I)
    return [mix[0:nb], mix[nb : 2 * nb], mix[2 * nb : 3 * nb], o, gu, down]


def pack_weights(Wt, fd, log=print):
    """The decode weight buffer (uint8): UNI_DEC layer slabs, then the lm_head
    waves."""
    NCX, NCY, dual = fd.NCX, fd.NCY, bool(fd.W_DUAL_CHAN)
    n_layer = fd.W_LAYER * 2
    VP = fd.VOCAB_SIZE_PADDED
    n_voc = fd.VOCAB_W_BLOCKS * 5120
    out = np.empty(LAYERS * n_layer + fd.UNI_LM * n_voc, np.uint8)
    t0 = time.perf_counter()
    for L in range(LAYERS):
        w = np.concatenate(
            [cascade(ph, NCX, NCY, dual) for ph in layer_phases(Wt, L, fd)]
        )
        assert w.size == n_layer, (L, w.size, n_layer)
        out[L * n_layer : (L + 1) * n_layer] = w
    o = LAYERS * n_layer
    nvw = VP // 2  # vocab rows per wave
    q, s = Wt.q8("lm_head.weight", NV, D)
    for wv in range(fd.UNI_LM):
        r0, r1 = wv * nvw, min(NV, (wv + 1) * nvw)
        if r1 > r0:
            b = q8_rows_dev(q[r0:r1], s[r0:r1], VP)
        else:
            b = affine_dev(
                np.zeros((VP, D), np.uint8),
                np.zeros((VP, D // 32), np.float32),
                np.zeros((VP, D // 32), np.float32),
            )
        out[o : o + n_voc] = cascade(b, NCX, NCY, dual)
        o += n_voc
    log(
        f"[fdg] packed {out.size / 1e6:.1f} MB of decode weights in {time.perf_counter() - t0:.0f} s"
    )
    return out


def pack_rms(Wt, fd):
    """The rms buffer (bf16): norms, the attention layers' q/k-norm slabs, the
    final norm. cos / sin of each attention slab are written per token."""
    RW = fd.ROPE_W_LEN
    r = np.zeros(LAYERS * fd.RMS_LAYER + LAYERS * RW + D, bfloat16)
    for L in range(LAYERS):
        p = f"model.layers.{L}."
        r[L * 2 * D : L * 2 * D + D] = Wt.bf16(p + "input_layernorm.weight").astype(
            bfloat16
        )
        r[L * 2 * D + D : (L + 1) * 2 * D] = Wt.bf16(
            p + "post_attention_layernorm.weight"
        ).astype(bfloat16)
    base = LAYERS * fd.RMS_LAYER
    for L in ATT:
        s = f"model.layers.{L}.self_attn."
        o = base + L * RW
        r[o + DH : o + 2 * DH] = Wt.bf16(s + "q_norm.weight").astype(bfloat16)
        r[o + 2 * DH : o + 3 * DH] = Wt.bf16(s + "k_norm.weight").astype(bfloat16)
    for L in LIN:
        # a Gated DeltaNet layer's slab: its 16 heads' -exp(A_log), then their
        # dt_bias (f32), which the mixer tile turns into the heads' gates
        s = f"model.layers.{L}.linear_attn."
        g = np.concatenate(
            [Wt.bf16(s + "ssm_a").reshape(-1), Wt.bf16(s + "ssm_dt.bias").reshape(-1)]
        )
        o = base + L * RW
        r[o : o + 64] = g.astype(np.float32).view(bfloat16)
    r[base + LAYERS * RW :] = Wt.bf16("model.norm.weight").astype(bfloat16)
    return r


def gdn_init(Wt, fd):
    """One GDN chunk buffer (bf16 words): per layer and core the magic chunks
    and the X chunks' statics (taps, -exp(A_log), dt bias, norm weight)."""
    W = fd.GDN_W
    g = np.zeros(LAYERS * fd.GDN_LAYER, bfloat16)
    u = g.view(np.uint16).reshape(LAYERS, 4, fd.GDN_CH, W)
    u[:, :, 0, W - 4 : W - 1] = MAGIC
    u[:, :, 0, W - 1] = 0xFACE
    u[:, :, -1, W - 4 : W - 1] = MAGIC
    u[:, :, -1, W - 1] = 0x0D0D
    gv = g.reshape(LAYERS, 4, fd.GDN_CH, W)
    for L in LIN:
        s = f"model.layers.{L}.linear_attn."
        conv = Wt.bf16(s + "ssm_conv1d.weight").reshape(
            4, -1
        )  # [4 taps, 6144], oldest first
        nega = Wt.bf16(s + "ssm_a").reshape(-1)
        dtb = Wt.bf16(s + "ssm_dt.bias").reshape(-1)
        nw = Wt.bf16(s + "ssm_norm.weight").reshape(-1)
        for h in range(16):
            b, j = divmod(h, 4)
            x = gv[L, b, 1 + fd.GDN_PER_HEAD * j]
            ch = np.concatenate(
                [
                    np.arange(base + h * 128, base + (h + 1) * 128)
                    for base in (0, 2048, 4096)
                ]
            )
            x[OTAPS : OTAPS + 1536] = conv[:, ch].reshape(-1).astype(bfloat16)
            x[OTAPS + 1536 : OTAPS + 1540] = np.array(
                [nega[h], dtb[h]], np.float32
            ).view(bfloat16)
            x[OTAPS + 1568 : OTAPS + 1696] = nw.astype(bfloat16)
            # the core's first X chunk carries its 4 heads' gate constants
            # (the kernel computes the 4 heads' gates at once)
            gv[L, b, 1, OPRM2 + 4 * j : OPRM2 + 4 * j + 4] = np.array(
                [nega[h], dtb[h]], np.float32
            ).view(bfloat16)
    return g


# ------------------------------------------------------------------- runtime


class FdgDecode:
    """The fused_decode_gdn decode of capacity C: device buffers, prefill
    hand-off, one dispatch per token."""

    def __init__(self, build_dir, capacity, Wt, cos, sin, cache_dir=None):
        import pyxrt as xrt

        self.xrt = xrt
        self.C = capacity
        self.fd = fd = builder(capacity)
        art = Path(build_dir) / f"fdg_{capacity}"
        sys.path.insert(0, str(FDEC))
        from decode_elf import ElfDecode

        self.dev = xrt.device(0)
        self.elf = ElfDecode(art, self.dev, xrt, fd.REGION_W)
        assert self.elf.attn_maxl >= capacity, (self.elf.attn_maxl, capacity)
        t0 = time.perf_counter()
        # the packed weights, cached per bundle (its header names every
        # tensor's shape and offsets)
        key = hashlib.sha256(json.dumps(Wt.hdr, sort_keys=True).encode()).hexdigest()[
            :12
        ]
        cache = Path(cache_dir or build_dir) / f"fdg_weights_{key}.npy"
        if cache.exists():
            w = np.load(cache, mmap_mode="r")
        else:
            w = pack_weights(Wt, fd)
            np.save(cache, w)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        bo = lambda n: xrt.ext.bo(self.dev, n)  # noqa: E731
        self.n_y = (
            fd.HOST_ROUNDS + fd.LAYER_RNDS
        ) * fd.PAYLOAD + fd.UNI_LM * fd.VOCAB_SIZE_PADDED
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.bo = dict(
            x=bo(D * 2),
            w=bo(w.size),
            y=bo(self.n_y * 2),
            kv=bo(fd.KV_BO_ELEMS[0] * 2),
            ga=bo(LAYERS * fd.GDN_LAYER * 2),
            gb=bo(LAYERS * fd.GDN_LAYER * 2),
        )
        rms = pack_rms(Wt, fd)
        self.bo["rms"] = bo(rms.size * 2)
        np.frombuffer(self.bo["w"].map(), np.uint8, count=w.size)[:] = w
        self.v = {
            k: np.frombuffer(b.map(), np.uint16, count=b.size() // 2).view(bfloat16)
            for k, b in self.bo.items()
            if k != "w"
        }
        self.v["rms"][:] = rms
        g = gdn_init(Wt, fd)
        self.v["ga"][:] = g
        self.v["gb"][:] = g
        self.v["kv"][:] = 0
        for k in self.bo:
            self.bo[k].sync(TO)
        self.rope_base = LAYERS * fd.RMS_LAYER
        self.load_s = time.perf_counter() - t0
        self.cos = np.asarray(cos)[:, :ROPE_HALF].astype(bfloat16)
        self.sin = np.asarray(sin)[:, :ROPE_HALF].astype(bfloat16)
        self.elf.bind([self.bo[k] for k in ("x", "w", "rms", "y", "kv", "ga", "gb")])
        self.par = 0
        self.kv_hi = 0  # the device has written cache rows [0, kv_hi) since load
        self.times = {}

    def handoff(self, kv, conv, state_raw):
        """After prefill (Qwen35Prefill.decode_state()): kv {L: (K, V)
        [2, C, 256]}, conv {L: [3, 6144]} the last three tokens' raw q | k | v,
        state_raw {L: [16 heads * 2 slices][>= 2 * 128 * 64] bf16 (hi, lo)}.
        Writes and syncs only what the decode reads: the attention layers'
        first n positions, zeros over the rows earlier requests left behind
        (the attention reads the cache in 16-position blocks and its result
        changes with what lies past the context, so the tail must be zero as
        it is at load), and the Gated DeltaNet layers' history rows and state
        units."""
        fd, xrt = self.fd, self.xrt
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        RS = fd.ATTN_MAXL * fd.REGION_W
        assert RS * 2 == fd.KV_LAYER
        K_ = self.v["kv"][: LAYERS * fd.KV_LAYER].reshape(
            LAYERS, 2, fd.ATTN_MAXL, 2, DH
        )
        n_kv = 0
        for L in ATT:
            k, v = kv[L]
            n = min(k.shape[1], fd.ATTN_MAXL)
            K_[L, 0, :n] = np.asarray(k[:, :n]).transpose(1, 0, 2).astype(bfloat16)
            K_[L, 1, :n] = np.asarray(v[:, :n]).transpose(1, 0, 2).astype(bfloat16)
            hi = max(n, self.kv_hi)
            K_[L, :, n:hi] = 0
            for r in range(2):  # the K and V regions' first hi rows
                self.bo["kv"].sync(
                    TO, hi * fd.REGION_W * 2, (L * fd.KV_LAYER + r * RS) * 2
                )
            n_kv = n
        self.kv_hi = n_kv  # the prompt's rows; each step adds its own
        self.par = 0
        ph, W = fd.GDN_PER_HEAD, fd.GDN_W
        gv = self.v["ga"].reshape(LAYERS, 4, fd.GDN_CH, W)
        for L in LIN:
            st = (
                np.asarray(state_raw[L])
                .view(bfloat16)
                .reshape(16, 2, -1)[:, :, : 2 * 128 * 64]
            )
            # units [hi | lo][128 k][16 v] -> the herd's [hi | lo][64 k pairs][16 v][2]
            st = st.reshape(16, 2, 2, 128, 4, 16).transpose(0, 1, 4, 2, 3, 5)
            st = (
                st.reshape(16, 8, 2, 64, 2, 16)
                .transpose(0, 1, 2, 3, 5, 4)
                .reshape(4, 4, 8, 4096)
            )
            cv = (
                np.asarray(conv[L]).astype(bfloat16).reshape(3, 3, 16, 128)
            )  # [t][q|k|v][h][128]
            hist = cv.transpose(2, 0, 1, 3).reshape(
                4, 4, 3 * 384
            )  # [core][head][t][q k v]
            for j in range(4):  # (views of the mapped buffer, no reshape copies)
                gv[L, :, 1 + ph * j, OHIST : OHIST + 3 * 384] = hist[:, j]
                gv[L, :, 2 + ph * j : 10 + ph * j] = st[:, j]
            # the 4 cores' chunks 1 .. 4 ph (contiguous per core)
            for b in range(4):
                o = (L * fd.GDN_LAYER + b * fd.GDN_CORE + W) * 2
                self.bo["ga"].sync(TO, 4 * ph * W * 2, o)

    def step(self, emb_row, pos):
        """One token at position pos -> logits float32 [NV]."""
        xrt, fd = self.xrt, self.fd
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        t0 = time.perf_counter()
        self.v["x"][:] = emb_row
        self.bo["x"].sync(TO)
        lut = np.concatenate([self.cos[pos], self.sin[pos]])
        rms = self.v["rms"]
        for L in ATT:
            o = self.rope_base + L * fd.ROPE_W_LEN
            rms[o : o + 2 * ROPE_HALF] = lut
        self.bo["rms"].sync(TO, LAYERS * fd.ROPE_W_LEN * 2, self.rope_base * 2)
        gi, go = ("ga", "gb") if self.par == 0 else ("gb", "ga")
        self.elf.run.set_arg(5, self.bo[gi])
        self.elf.run.set_arg(6, self.bo[go])
        t1 = time.perf_counter()
        st = self.elf.dispatch(pos + 1, np)
        t2 = time.perf_counter()
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"fused_decode_gdn dispatch at position {pos}: {st}")
        self.par ^= 1
        self.kv_hi = max(self.kv_hi, pos + 1)
        self.bo["y"].sync(FROM, 2 * NV * 2, self.decode_y * 2)
        y = self.v["y"][self.decode_y : self.decode_y + 2 * NV].astype(np.float32)
        z = y[0::2] + y[1::2]
        t3 = time.perf_counter()
        for k, v in (
            ("dec_host_in", t1 - t0),
            ("dec_run", t2 - t1),
            ("dec_logits", t3 - t2),
        ):
            self.times[k] = self.times.get(k, 0) + v
        return z
