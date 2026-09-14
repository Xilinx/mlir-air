#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""gemma4_e2b_q4nx full inference -- NPU prefill + fused per-layer-embedding decode.

Reproduces FastFlowLM's Gemma4-E2B (text) end to end on the NPU:

  prefill: gemma4_e2b_q4nx_prefill.Gemma4Q4nxPrefill -- 22 ELFs, the two attention
    classes (sliding head_dim 256 / full head_dim 512), the dual-width GELU GLU,
    the per-layer-embedding tail and the 4-bit LM head. Produces the per-layer
    roped-K / normed-V KV seed and the greedy first token.
  decode: the fused_decode_ple engine -- all 35 decoder layers plus the LM head in
    ONE dispatch, appending each new token's K/V in place.

`--numpy-prefill` swaps the NPU prefill for gemma4_e2b_q4nx_weights.forward_prompt,
the CPU oracle the prefill is gated against. Minutes, not seconds; it is a
debugging fallback, not a mode to benchmark.

TWO THINGS ARE SPECIFIC TO THIS MODEL and have no analogue in the sibling drivers:

  Mixed head dims in one uniform cache. Every KV row on device is REGION_W=1024
  wide = N_ATTN_CU(2) * DH_A(512), and the single MQA head is replicated into both
  halves. A sliding layer's head is only 256 wide and does NOT sit contiguously in
  its 512-wide slot: it is scattered as [real_lo | zeros | real_hi | zeros] so the
  rope kernel's fixed (i, i+256) pairing lands on the real (i, i+128) pairs.
  seed_kv() is the only place that knows this, and getting it wrong does not
  raise -- it seeds plausible garbage.

  The PLE branch. A third sub-layer after the FFN whose input comes from the token
  EMBEDDING, not from the layer's hidden state. It needs two extra BOs (pw, px)
  that the non-PLE engine has no arguments for, and the per-token embedding slice
  is patched into each layer's PLE slab every step.

Run:
  python3 gemma4_e2b_q4nx_inference.py                 # Paris gate (greedy)
  python3 gemma4_e2b_q4nx_inference.py --prompt "The capital of France is" --n-tokens 12
  python3 gemma4_e2b_q4nx_inference.py --numpy-prefill # CPU-oracle prefill instead

Needs a FULL-DEPTH decode template pair in this directory
(decode_L<N>.{xclbin,insts.bin} at UNI_DEC=35): `make compile-decode-full`. The
per-layer gate's template is a ONE-layer build and will not serve a token.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PE = _HERE.parent.parent  # programming_examples
_DEC = _PE / "fused_decode_ple"  # the PLE fork of the decode engine
_FD = _PE / "fused_decode"  # decode_insts_gen / decode_staircase live here
sys.path.insert(0, str(_HERE))

# HF repo id of the self-contained model.q4nx bundle, or a local dir/file
# (gemma4_e2b_q4nx_weights.resolve_q4nx_model resolves all three).
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma4-E2B-IT-NPU2")
# <bos> + "The capital of France is". The Gemma4 tokenizer does not add <bos>, so
# it is prepended by hand -- exactly as gemma4_e2b_q4nx_prefill._main and
# run_reference.py do, or the two paths score different prompts.
PARIS_PROMPT = [2, 818, 5279, 529, 7001, 563]
PARIS_FIRST = 9079  # ' Paris'
# The greedy continuation, CONFIRMED AGAINST THE CPU ORACLE rather than merely
# recorded from a device run: gemma4_e2b_q4nx_weights.forward_prompt, re-run on
# the growing prompt, emits exactly [9079, 236761, 106] -- " Paris." then
# <end_of_turn>.
#
# Two tokens is a short sequence, which is why PARIS_STOP is part of the gate:
# the run must also STOP where the oracle stops, so all three of the model's
# decisions are covered. The failure this exists to catch is a correct first
# token -- which comes from the PREFILL -- followed by plausible garbage from the
# decode. Seeding the KV cache without the padded-head interleave produced
# exactly that: ' Parisнии est le Humदा is a het de'.
PARIS_GREEDY = [9079, 236761]
PARIS_STOP = 106  # <end_of_turn>; the run must also STOP where the oracle stops
EOS_IDS = (1, 106)  # <eos>, <end_of_turn>

# Where decode_L<N>.{xclbin,insts.bin} live. `make compile-decode-full` writes the
# pair here; the sweep writes its own pairs here too, which is fine -- the
# generator picks the smallest calibrated window covering the session's reach.
_DECODE_DIR = Path(os.environ.get("Q4NX_GEMMA4_DECODE_DIR", str(_HERE)))
# The packed decode weights. Repo-local and keyed on the geometry that shaped
# them, because a cache packed for a different layer count or vocab chunking is
# not detectably wrong at dispatch time -- it just feeds the kernels the wrong
# blocks. ~1.8 GB, and packing it from the bundle takes the better part of an hour.
_WCACHE_DIR = Path(
    os.environ.get("Q4NX_GEMMA4_WCACHE_DIR", str(_HERE / ".decode_wcache"))
)
VOCAB_CHUNK_I2 = "27"


def _load_builder(uni_dec, attn_maxl, kv_src):
    """Import fused_decode_ple with this build's geometry in the environment.

    Delegated to validate_layer_npu rather than restated: the layer gate and the
    token driver MUST configure the builder identically or they are scoring and
    running two different designs.
    """
    import validate_layer_npu as vln

    return vln._load_fd(uni_dec, attn_maxl, kv_src=kv_src)


def _wcache_path(uni_dec, fingerprint):
    """Cache path, keyed on the geometry AND the bundle it was packed from.

    The fingerprint is load-bearing, not decoration: without it, pointing
    Q4NX_MODEL_SOURCE at a different bundle -- or FLM re-exporting one at the
    same source -- silently reuses the old packed weights. Nothing downstream
    catches that; the layer-count and LM-head checks below still pass and the
    dispatch completes, running a different model than the caller asked for.
    """
    return _WCACHE_DIR / f"decode_uni{uni_dec}_v{VOCAB_CHUNK_I2}_{fingerprint}.npz"


def _ensure_wcache(model, fd, uni_dec, qm, verbose=True):
    """Return the packed decode weight cache, building it once if absent.

    pack_vocab=True is what separates this from the layer gate's cache: the gate
    never reads logits and deliberately fills the vocab region with a nonzero
    pattern, so its cache cannot produce a token.
    """
    import gemma4_e2b_q4nx_requant as rq
    import numpy as np

    path = _wcache_path(uni_dec, qm.fingerprint())
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(
                f"[inference] packing decode weights -> {path} "
                f"(one-time, {uni_dec} layers + LM head; this takes ~an hour)",
                flush=True,
            )
        rq.build_requant_cache(
            model, fd, str(path), layers=list(range(uni_dec)), pack_vocab=True
        )
    z = np.load(str(path))
    if not int(z["has_vocab"]):
        raise RuntimeError(
            f"{path} was packed WITHOUT the LM head (pack_vocab=False), so it "
            f"cannot produce a token. Delete it and re-run."
        )
    if list(z["layers"]) != list(range(uni_dec)):
        raise RuntimeError(
            f"{path} holds layers {list(z['layers'])}, not 0..{uni_dec - 1}. "
            f"Delete it and re-run."
        )
    return z


class FusedDecoder:
    """One-xclbin Gemma4-E2B fused decode, all 35 layers plus the LM head per dispatch.

    A single template built at ATTN_MAXL serves every L in [1, ATTN_MAXL]: the
    L-dependent instruction words (attention bound + KV-append offset) are patched
    per token by DecodeInstsGen. Weights, PLE slabs and the KV cache are uploaded
    once; the kernel appends each new token's K/V in place.
    """

    def __init__(self, model=MODEL_DEFAULT, max_L=None, verbose=True):
        import numpy as np
        from ml_dtypes import bfloat16
        import pyxrt as xrt
        import gemma4_e2b_q4nx_weights as gw

        self.np, self.bf16, self.xrt, self.gw = np, bfloat16, xrt, gw

        sys.path.insert(0, str(_FD))
        from decode_insts_gen import DecodeInstsGen
        import decode_staircase as stair

        self._stair = stair
        self.gen = DecodeInstsGen(str(_DECODE_DIR), max_L)
        self.windows = stair.resolve_windows(self.gen)
        self.ATTN_MAXL = max(self.windows)
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL

        # Every model layer is a device slab, in order. A KV-shared layer (>=15)
        # projects no K/V of its own and reads the slab holding the layer
        # kv_source_layer() names; with one slab per layer those indices coincide,
        # but they are resolved rather than assumed so the map stays honest if the
        # slab set ever stops being the whole model.
        self.UNI = gw.NUM_LAYERS
        Ls = list(range(self.UNI))
        kv_src = [Ls.index(gw.kv_source_layer(L)) for L in Ls]
        fd = _load_builder(self.UNI, self.ATTN_MAXL, kv_src)
        # The builder is configured from the environment; the xclbin on disk was
        # built from some OTHER invocation of it. If the two disagree the BO
        # layout is wrong and nothing says so -- the dispatch completes.
        if fd.UNI_DEC != self.UNI or fd.ATTN_MAXL != self.ATTN_MAXL:
            raise RuntimeError(
                f"decode template in {_DECODE_DIR} is ATTN_MAXL={self.ATTN_MAXL} "
                f"but the builder reports UNI_DEC={fd.UNI_DEC} "
                f"ATTN_MAXL={fd.ATTN_MAXL} (want UNI_DEC={self.UNI}). The layer "
                f"gate builds a ONE-layer template; run `make compile-decode-full`."
            )
        if not getattr(fd, "PLE", False):
            raise RuntimeError("builder came back without the PLE branch")

        self.K, self.DH_A = fd.K, fd.DH_A
        self.UNI_LM = fd.UNI_LM
        self.REGION_W, self.NGRP = fd.REGION_W, fd.NGRP
        self.KV_LAYER = fd.KV_LAYER
        self.PLE_LAYER = fd.PLE_LAYER
        self.PLE_EMB_OFF, self.PLE_NORMW_OFF = fd.PLE_EMB_OFF, fd.PLE_NORMW_OFF
        self.VOCAB_SIZE, self.VP = fd.VOCAB_SIZE, fd.VOCAB_SIZE_PADDED
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + self.UNI_LM * self.VP
        self.n_w = (
            self.UNI * fd.W_TOTAL_BLOCKS + self.UNI_LM * fd.VOCAB_W_BLOCKS
        ) * fd.BLOCK_BF16
        # RMS BO: [UNI per-layer 5-norm slabs | UNI per-layer rope_w slabs | final_norm]
        self._rope_base = self.UNI * fd.RMS_LAYER
        self._RMS_SIZE = self._rope_base + self.UNI * fd.ROPE_W_LEN + self.K

        self.qm = gw.Q4nxModel(model)
        self.rope_freqs = self.qm.rope_freqs()
        self.final_norm = np.asarray(self.qm.globals()["final_norm"], bfloat16)

        z = _ensure_wcache(model, fd, self.UNI, self.qm, verbose)
        W = z["W"]
        if W.size != self.n_w:
            raise RuntimeError(
                f"weight cache holds {W.size} elements, the build wants {self.n_w}"
            )
        self.QNORM = [z["QNORM"][i].view(bfloat16) for i in range(self.UNI)]
        self.KNORM = [z["KNORM"][i].view(bfloat16) for i in range(self.UNI)]
        self.rms_slabs = np.concatenate(
            [
                z[f"RMS_{n}"][i].view(bfloat16)
                for i in range(self.UNI)
                for n in ("in", "post_attn", "pre_ffn", "post_ffn", "post_ple")
            ]
        )
        assert self.rms_slabs.size == self._rope_base, (
            self.rms_slabs.size,
            self._rope_base,
        )
        self.ple = z["PLE"].view(bfloat16).copy().reshape(self.UNI, self.PLE_LAYER)

        print(
            f"[decode] gemma4-e2b ONE xclbin: ATTN_MAXL={self.ATTN_MAXL}, serves "
            f"L in [1,{self.maxL}]; {self.UNI} layers + PLE + lm-head/dispatch",
            flush=True,
        )

        self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._kern = stair.open_windows(self.dev, xrt, self.gen, self.windows)
        self.kern = self._kern[self.ATTN_MAXL][1]
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.w_bo = xrt.bo(self.dev, self.n_w * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.UNI * self.KV_LAYER * 2, HO, g(7))
        self.pw_bo = xrt.bo(self.dev, self.UNI * self.PLE_LAYER * 2, HO, g(8))
        self.px_bo = xrt.bo(self.dev, self.K * 2, HO, g(9))
        self._ist = stair.make_insts_states(self.gen, xrt, self.dev, g(1), self.windows)
        self._st = self._ist[self.ATTN_MAXL]
        self.ib = self._st["ib"]

        self.w_bo.write(W if W.dtype == np.int16 else W.view(np.int16), 0)
        self.w_bo.sync(TO)
        # RMS: norms and final_norm are constants, the rope region is rewritten per
        # token. Written whole once so the per-token patch is only the rope slice.
        rbuf = np.concatenate(
            [
                self.rms_slabs,
                np.zeros(self._RMS_SIZE - self._rope_base - self.K, bfloat16),
                self.final_norm,
            ]
        )
        assert rbuf.size == self._RMS_SIZE, (rbuf.size, self._RMS_SIZE)
        self.r_bo.write(rbuf.view(np.int16), 0)
        self.r_bo.sync(TO)
        self.pw_bo.write(self.ple.reshape(-1).view(np.int16), 0)
        self.pw_bo.sync(TO)
        self.KV = np.zeros((self.UNI, self.KV_LAYER), dtype=bfloat16)
        self._partial_sync = True

    # ---- host -> device helpers ----

    def _patch(self, bo, arr, elem_off):
        """Write `arr` at `elem_off` and sync just that slice.

        Falls back to a whole-BO sync the first time a ranged sync is refused:
        pyxrt's offset sync is not available on every XRT build, and the PLE and
        rope patches are small slices of an 80 MB and a 500 KB buffer.
        """
        np = self.np
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        bo.write(arr.view(np.int16), elem_off * 2)
        if self._partial_sync:
            try:
                bo.sync(TO, arr.size * 2, elem_off * 2)
                return
            except Exception:
                self._partial_sync = False
        bo.sync(TO)

    def seed_kv(self, ks, vs, P):
        """Place the prefill's per-layer K/V into the device cache.

        `ks`/`vs` are PER-LAYER LISTS of (P, head_dim), not a stacked array: head_dim
        is 256 on the 28 sliding layers and 512 on the 7 full ones, so they do not
        stack. Each device row is REGION_W wide and holds the single MQA head twice,
        once per attention CU.

        A 256-wide head does NOT sit contiguously in its 512-wide slot. Every layer
        is built at the widest layer's geometry and the narrow heads are scattered
        as [real_lo | zeros | real_hi | zeros] so that the rope kernel's fixed
        (i, i+DH_A/2) pairing lands on the real (i, i+128) pairs -- see
        gemma4_e2b_q4nx_requant's module docstring, where that interleave is
        recorded as measured-correct and the contiguous layout as measured-wrong.
        Seeding contiguously does not fail: it produces a correct first token
        (which comes from the prefill) followed by fluent garbage.
        """
        np = self.np
        from gemma4_e2b_q4nx_requant import _head_perm

        RW, RS = self.REGION_W, self.ATTN_MAXL * self.REGION_W
        if P > self.ATTN_MAXL:
            raise ValueError(f"prompt of {P} exceeds ATTN_MAXL={self.ATTN_MAXL}")
        self.KV[:] = 0
        for L in range(self.UNI):
            # _head_perm is the identity at dh == DH_A, so the full layers take
            # the same path rather than a special case.
            perm = _head_perm(self.gw.head_dim(L), self.DH_A)
            for reg, src in ((0, ks[L]), (1, vs[L])):
                src = np.asarray(src, np.float32).reshape(P, -1)
                dh = src.shape[1]
                if dh != self.gw.head_dim(L):
                    raise ValueError(
                        f"layer {L} KV is {dh} wide, the model says "
                        f"{self.gw.head_dim(L)}"
                    )
                rows = self.KV[L, reg * RS : reg * RS + P * RW].reshape(P, RW)
                rows[:, perm] = src.astype(self.bf16)
                rows[:, self.DH_A + perm] = src.astype(self.bf16)
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.kvc.write(np.ascontiguousarray(self.KV).reshape(-1).view(np.int16), 0)
        self.kvc.sync(TO)

    def _rope_slab(self, p):
        """The UNI per-layer rope_w slabs for position p: [cos/sin LUT | q_norm | k_norm].

        Per LAYER, not shared: the sliding layers rotate 256 dims at theta 1e4 and
        the full ones 128 of 512 at theta 1e6 (partial rotary, folded into the
        bundle's rope_freqs divisor). The LUT is DH_A wide with cos in the low half
        and sin in the high half, so a 256-wide layer fills a quarter of each.
        """
        np = self.np
        out = []
        for L in range(self.UNI):
            cos, sin, dh = self.gw.rope_lut(p, L, rope_freqs=self.rope_freqs)
            lut = np.zeros(self.DH_A, np.float32)
            lut[: dh // 2] = cos
            lut[self.DH_A // 2 : self.DH_A // 2 + dh // 2] = sin
            out += [lut.astype(self.bf16), self.QNORM[L], self.KNORM[L]]
        return np.concatenate(out)

    def _ple_embed(self, tok):
        """This token's per-layer embedding slice, [UNI, PLI_D].

        validate_layer_npu.ple_embed for every layer at once -- the bundle's table
        is already scaled (PLE_EMBED_SCALE == 1), and reading it once per token
        rather than once per layer is the only difference.
        """
        np = self.np
        gw = self.gw
        tbl = self.qm.embed_rows("model.per_layer_token_embd.weight", [tok])
        tbl = tbl.reshape(gw.NUM_LAYERS, gw.PLI_D) * gw.PLE_EMBED_SCALE
        return np.asarray(tbl[: self.UNI], self.bf16)

    def dispatch(self, tok, p):
        """One decode step at L=p+1. Returns softcapped logits [VOCAB_SIZE].

        Appends this token's K/V at slot p on-device, and writes the final hidden
        state back into x_bo (which is why x_bo is both an input and an output).
        """
        np, xrt = self.np, self.xrt
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        L = p + 1
        if not (1 <= L <= self.maxL):
            raise ValueError(f"position {p} is outside [0,{self.maxL - 1}]")
        insts_size = self._stair.patch_insts(self._st, L, xrt, TO)

        # NO embed re-scale: the q4nx bundle's embed_tokens is already multiplied
        # by sqrt(hidden_size), so EMBED_SCALE is 1.
        x0 = np.asarray(
            self.qm.embed_rows("model.embed_tokens.weight", [tok])[0], self.bf16
        )
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(TO)
        # px is the SAME embedding: the PLE branch's model_proj reads the token
        # embedding, never the running hidden state. Feeding it x would be a
        # plausible-looking model that is not this one.
        self.px_bo.write(x0.view(np.int16), 0)
        self.px_bo.sync(TO)

        emb = self._ple_embed(tok)
        for i in range(self.UNI):
            self._patch(self.pw_bo, emb[i], i * self.PLE_LAYER + self.PLE_EMB_OFF)
        self._patch(self.r_bo, self._rope_slab(p), self._rope_base)

        st = self.kern(
            3,
            self.ib,
            insts_size,
            self.x_bo,
            self.w_bo,
            self.r_bo,
            self.y_bo,
            self.kvc,
            self.pw_bo,
            self.px_bo,
        ).wait(60000)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        _voc_n = self.UNI_LM * self.VP
        self.y_bo.sync(FROM, _voc_n * 2, self.decode_y * 2)
        # Zero-copy view into the BO: bo.read() returns a buffer whose stride
        # metadata is pyxrt-build dependent and .view() on it raises on some runners.
        yv = np.frombuffer(
            self.y_bo.map(), dtype=self.bf16, count=_voc_n, offset=self.decode_y * 2
        ).astype(np.float32)[: self.VOCAB_SIZE]
        cap = self.gw.FINAL_LOGIT_SOFTCAP
        # Monotonic, so it cannot move the argmax -- but it IS the model's output,
        # and anything scoring logits against the CPU reference needs it applied.
        return cap * np.tanh(yv / cap) if cap else yv

    # Kernels, insts states and BOs are all created against self.dev, but self.dev
    # is assigned first and CPython clears an instance __dict__ in insertion order,
    # so the device would be released before the objects that depend on it. Linux
    # XRT tolerates that; Windows XRT faults with an access violation while the
    # decoder is collected. `ib` and `_st` alias into `_ist`, so they have to be
    # dropped ahead of it or they keep that state (and its cacheable BO) alive past
    # the device.
    _XRT_RELEASE_ORDER = (
        "ib",
        "_st",
        "_ist",
        "px_bo",
        "pw_bo",
        "kvc",
        "y_bo",
        "r_bo",
        "w_bo",
        "x_bo",
        "kern",
        "_kern",
        "dev",
    )

    def close(self):
        """Release the XRT objects in reverse dependency order."""
        for name in self._XRT_RELEASE_ORDER:
            if name in self.__dict__:
                self.__dict__[name] = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _prefill_npu(prompt, model, seq_len=None):
    """Batched AIR prefill on the NPU -> (ks, vs, first_token, ttft_s).

    ks/vs are the per-layer lists FusedDecoder.seed_kv consumes. The prefill runs
    at a fixed padded seq_len -- every ELF is built for it -- so its cost is
    ~constant in prompt length.

    ttft_s times the prefill dispatch only. Building the engine and load_weights
    (host dequant of the bundle plus the one-time write of every weight into its
    resident BO) are model-load costs paid once per process, reported separately.
    """
    from gemma4_e2b_q4nx_prefill import Gemma4Q4nxPrefill

    seq_len = seq_len or int(os.environ.get("Q4NX_SEQ_LEN", "2048"))
    t_load = time.perf_counter()
    pf = Gemma4Q4nxPrefill(
        seq_len=seq_len, cache_dir=os.environ.get("Q4NX_CACHE_DIR") or None
    )
    pf.load_weights(model=model)
    print(
        f"[inference] model load (dequant + resident BOs): "
        f"{time.perf_counter() - t_load:.1f}s",
        flush=True,
    )
    t0 = time.perf_counter()
    logits = pf.prefill(prompt)
    ttft = time.perf_counter() - t0
    ks, vs = pf.kv_stack()
    return ks, vs, int(logits.argmax()), ttft


def _prefill_numpy(prompt, model):
    """The CPU oracle prefill -> the same tuple, for debugging the device path."""
    import numpy as np
    import gemma4_e2b_q4nx_weights as gw

    t0 = time.perf_counter()
    qm = gw.Q4nxModel(model)
    logits, kv = gw.forward_prompt(qm, prompt)
    # forward_prompt keys its cache by SOURCE layer and keeps the kv-head axis;
    # seed_kv wants one (P, head_dim) entry per layer, MQA head squeezed off.
    ks, vs = [], []
    for L in range(gw.NUM_LAYERS):
        ke, v = kv[gw.kv_source_layer(L)]
        ks.append(np.asarray(ke[:, 0], np.float32))
        vs.append(np.asarray(v[:, 0], np.float32))
    return ks, vs, int(np.asarray(logits).argmax()), time.perf_counter() - t0


def generate(
    prompt, n_tokens, model=MODEL_DEFAULT, numpy_prefill=False, ignore_eos=False
):
    src = "numpy reference" if numpy_prefill else "AIR NPU"
    print(
        f"[inference] {src} prefill (KV seed + first token), "
        f"prompt_len={len(prompt)}...",
        flush=True,
    )
    if numpy_prefill:
        ks, vs, first, ttft = _prefill_numpy(prompt, model)
    else:
        ks, vs, first, ttft = _prefill_npu(prompt, model)
    P = ks[0].shape[0]
    print(
        f"[inference] prefill first token = {first} (Paris={PARIS_FIRST})", flush=True
    )
    # Machine-readable line for bench/extract_perf.py (nightly LLM dashboard).
    print(f"Time to first token (TTFT): {ttft:.3f}s", flush=True)

    dec = FusedDecoder(model=model, max_L=P + n_tokens)
    n_eff = min(n_tokens, dec.maxL - P)
    if n_eff <= 0:
        print(f"[inference] P={P} >= ATTN_MAXL={dec.ATTN_MAXL}; abort", flush=True)
        return [first], None
    dec.seed_kv(ks, vs, P)
    tokens = list(prompt) + [first]
    gen_ids = [first]
    stop = None
    t_dec0 = time.perf_counter()
    for p in range(P, P + n_eff):
        pred = int(dec.dispatch(tokens[p], p).argmax())
        if pred in EOS_IDS and not ignore_eos:
            print(f"[inference] pos{p} L={p + 1} -> EOS ({pred}), stop", flush=True)
            stop = pred
            break
        gen_ids.append(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
    t_dec = time.perf_counter() - t_dec0
    n_gen = len(gen_ids) - 1
    if n_gen > 0:
        print(
            f"[inference] decode: {n_gen} tokens in {t_dec:.2f}s "
            f"{n_gen / t_dec:.2f} tok/s ({t_dec / n_gen * 1000:.1f} ms/token)",
            flush=True,
        )
        # Machine-readable lines for bench/extract_perf.py (nightly LLM dashboard).
        print(
            f"[inference] Inference: prompt_len={len(prompt)}, n_tokens={n_gen}",
            flush=True,
        )
        print(f"Tokens/second: {n_gen / t_dec:.2f}", flush=True)
    dec.close()
    return gen_ids, stop


def _first_diff(got, want):
    """Index of the first differing token, or the length of the shorter run."""
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return i
    return min(len(got), len(want))


def _paris_verdict(gen_ids, stop):
    """(lines, ok) for the Paris gate.

    The misses are kept apart because they point at different halves: the first
    token comes out of the PREFILL, the continuation and the stop out of the
    DECODE loop.
    """
    if not gen_ids or gen_ids[0] != PARIS_FIRST:
        return (
            [f"*** MISS *** first token {gen_ids[:1]}, expected [{PARIS_FIRST}]"],
            False,
        )
    n = min(len(gen_ids), len(PARIS_GREEDY))
    if gen_ids[:n] != PARIS_GREEDY[:n]:
        return (
            [
                f"*** MISS *** decode drifted at token "
                f"{_first_diff(gen_ids, PARIS_GREEDY)}",
                f"    expected {PARIS_GREEDY[:n]}",
                f"    got      {gen_ids[:n]}",
            ],
            False,
        )
    # Only meaningful once the whole recorded run has been generated: a shorter
    # --n-tokens legitimately stops early with stop=None.
    if len(gen_ids) >= len(PARIS_GREEDY) and stop != PARIS_STOP:
        return (
            [
                f"*** MISS *** run ended on {stop}, expected to stop on "
                f"{PARIS_STOP} (<end_of_turn>) after {len(PARIS_GREEDY)} tokens",
            ],
            False,
        )
    return ["*** PARIS ***"], True


def _detok(ids, model=MODEL_DEFAULT):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model).decode(ids)
    except Exception as e:
        return f"(no detok: {e}) ids={ids}"


def main():
    ap = argparse.ArgumentParser(description="gemma4_e2b_q4nx full inference")
    ap.add_argument("--prompt", type=str, default=None, help="prompt text")
    ap.add_argument("--prompt-ids", type=str, default=None, help="comma-separated ids")
    ap.add_argument("--n-tokens", type=int, default=9, help="tokens to generate")
    ap.add_argument(
        "--model", type=str, default=MODEL_DEFAULT, help="model.q4nx dir/path"
    )
    ap.add_argument(
        "--numpy-prefill",
        action="store_true",
        help="seed the KV cache with the CPU reference forward instead of the AIR "
        "NPU prefill (the oracle it is gated against; minutes, not seconds)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero unless the Paris continuation matches",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help="keep generating past <eos>/<end_of_turn>. FOR THROUGHPUT ONLY: this "
        "model answers the Paris prompt in two tokens and then stops, so a tok/s "
        "figure taken from a run that honours EOS is dominated by per-call setup "
        "rather than by decode. The text past the stop is not meaningful.",
    )
    args = ap.parse_args()

    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
    elif args.prompt:
        from transformers import AutoTokenizer

        # <bos> by hand: this tokenizer does not add it.
        prompt = [2] + AutoTokenizer.from_pretrained(args.model).encode(args.prompt)
    else:
        prompt = PARIS_PROMPT
    print(f"[inference] prompt = {len(prompt)} tokens: {prompt}", flush=True)

    gen_ids, stop = generate(
        prompt,
        args.n_tokens,
        model=args.model,
        numpy_prefill=args.numpy_prefill,
        ignore_eos=args.ignore_eos,
    )
    print("=" * 60)
    print(f"[inference] gen ids: {gen_ids}")
    print(f"[inference] TEXT: {_detok(gen_ids, args.model)!r}")
    ok = True
    # --ignore-eos deliberately runs past the stop, so the recorded continuation
    # no longer describes the run and the verdict would be a false MISS.
    if prompt == PARIS_PROMPT and not args.ignore_eos:
        lines, ok = _paris_verdict(gen_ids, stop)
        for line in lines:
            print(line)
    return 0 if (ok or not args.gate) else 1


if __name__ == "__main__":
    sys.exit(main())
