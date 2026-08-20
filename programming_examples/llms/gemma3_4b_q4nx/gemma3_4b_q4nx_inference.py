#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""gemma3_4b_q4nx full inference -- FLM-faithful Gemma3-4B prefill + decode.

Reproduces FastFlowLM's Gemma3-4B mechanism end to end on the NPU:

  prefill: the batched AIR prefill (gemma3_4b_q4nx_prefill.GemmaQ4nxPrefill) -- 8 ELFs
    per layer-pass with resident weight BOs, alternating sliding-window / global
    flash attention, GELU-tanh GLU, on-device tied LM head. Produces the per-layer
    roped-K / raw-V KV seed and the greedy first token.
  decode: the shared fused_decode engine (DECODE_MODEL=gemma3-4b) -- 34 decoder
    layers + tied lm-head in ONE dispatch, the 4-norm Gemma sandwich, dual-theta
    RoPE with per-head qk-norm, appending each new token's K/V in place.

`--numpy-prefill` swaps the NPU prefill for the numpy reference forward
(gemma3_4b_q4nx_weights.forward_prompt) -- the oracle the prefill is checked against,
kept as a fallback and for debugging (it takes tens of seconds).

The decode is ONE xclbin built at ATTN_MAXL (=16*ceil(L/16)); DecodeInstsGen patches the
L-dependent insts words (attention RTP-L + KV-append offset) per token. The RMS BO holds
UNI_DEC per-layer norm slabs [input|post_attn|pre_ffn|post_ffn], then UNI_DEC per-layer
rope_w slabs [cos/sin | q_norm | k_norm] (dual-theta, rewritten per position), then the
final norm.

Sliding-window (1024) is a no-op below position 1024 (a window that spans the whole context
equals full causal), so for the Paris gate it is not yet wired; dual-theta + per-layer
qk-norm ARE wired (they affect every position).

Run:
  python3 gemma3_4b_q4nx_inference.py                 # Paris gate (greedy, PARIS_GREEDY)
  python3 gemma3_4b_q4nx_inference.py --prompt "The capital of France is" --n-tokens 12
  python3 gemma3_4b_q4nx_inference.py --numpy-prefill # numpy-oracle prefill instead
"""

import os
import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PE = _HERE.parent.parent  # programming_examples
_DEC = _PE / "fused_decode"  # shared fused superkernel decode engine
sys.path.insert(0, str(_HERE))

# HF repo id of the self-contained model.q4nx bundle, or a local dir/file
# (gemma3_4b_q4nx_weights resolves all three).
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Gemma3-4B-NPU2")
# <bos> + "The capital of France is"  (Gemma3 tokenizer); numpy ref -> 9079 " Paris".
PARIS_PROMPT = [2, 818, 5279, 529, 7001, 563]
PARIS_FIRST = 9079
# Greedy continuation at the gate shape (`make verify` runs --n-tokens 9). The gate
# checks this, not just PARIS_FIRST: a decode that starts right and then drifts is a
# real failure mode here -- a stale decode template yields a correct first token and
# fluent garbage after it, which a first-token gate passes. Greedy, so deterministic;
# re-record from `make run` if a deliberate numerics change moves it.
PARIS_GREEDY = [9079, 236761, 108, 50429, 563, 496, 4185, 3988, 573, 1610]
EOS_IDS = (1, 106)  # <eos>, <end_of_turn>
_Q4NX_CACHE = os.path.expanduser("~/.cache/q4nx_gemma")


def _ensure_requant_cache(fd, model):
    """Return the decode q4k-cascade cache path, building it from model.q4nx on first
    use (one-time ~pack of 34 layers + lm-head). Honors Q4NX_GEMMA_DECODE_NPZ."""
    rc = os.environ.get("Q4NX_GEMMA_DECODE_NPZ")
    if rc and os.path.exists(rc):
        return rc
    import gemma3_4b_q4nx_requant as rq

    # W_DUAL_CHAN reorders the cascade (the two-MM2S weight feed splits it by
    # cascade pair into [low-row half | high-row half]), so it gets its own cache
    # entry -- a warm single-channel cache would feed the dual-channel xclbin the
    # wrong blocks.
    _w2 = "_w2ch" if getattr(fd, "W_DUAL_CHAN", 0) else ""
    rc = rc or os.path.join(_Q4NX_CACHE, f"requant{_w2}.npz")
    if not os.path.exists(rc):
        rq.build_requant_cache(model, fd, rc)
    os.environ["Q4NX_GEMMA_DECODE_NPZ"] = rc
    return rc


# Gemma decode xclbins (decode_L<N>) live in a gemma-specific artifact dir so they don't
# collide with the Llama builds in fused_decode/. DecodeInstsGen (the L->insts affine
# specializer) is imported from fused_decode/ but scans this dir for the templates.
_DECODE_DIR = Path(os.environ.get("Q4NX_GEMMA_DECODE_DIR", str(_HERE)))

# decode_staircase lives in fused_decode/, which joins sys.path in _pick_decode_gen;
# imported lazily and cached for the methods that need it.
_stair = None


def _load_stair():
    global _stair
    if _stair is None:
        import decode_staircase

        _stair = decode_staircase
    return _stair


_dyn = None  # set by _pick_decode_gen, which joins fused_decode/ to sys.path


def _pick_decode_gen(dec_dir, max_L=None):
    sys.path.insert(0, str(_DEC))
    global _dyn
    import decode_dynseq as _dyn
    from decode_dynseq import pick_insts_gen

    return pick_insts_gen(str(dec_dir), max_L=max_L)


class FusedDecoder:
    """One-xclbin Gemma3-4B fused decode. A SINGLE decode xclbin built at ATTN_MAXL serves
    every L in [1, ATTN_MAXL] via a per-token RTP-L + KV-append insts patch (DecodeInstsGen).
    The weight + KV BOs are uploaded once; the kernel appends each new token's K/V in place.
    """

    def __init__(self, model=MODEL_DEFAULT, max_L=None, staircase=False):
        import importlib.util
        import numpy as np
        from ml_dtypes import bfloat16
        import pyxrt as xrt
        import gemma3_4b_q4nx_weights as gw

        self.np = np
        self.bf16 = bfloat16
        self.xrt = xrt
        self.gw = gw

        self.gen = _pick_decode_gen(_DECODE_DIR, max_L)
        _load_stair()
        # Staircase: hold every calibrated ATTN_MAXL window and dispatch each token on the
        # smallest one covering L (the readback streams ATTN_MAXL positions regardless).
        # Off by default -- one window, identical to the single-template path.
        self.windows = _stair.resolve_windows(self.gen, staircase)
        self.ATTN_MAXL = max(self.windows)
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL

        # DECODE_MODEL / geometry env must be set BEFORE importing fused_decode.py (its
        # module-level constants read them). Matches the Llama driver's env block.
        os.environ.update(
            DECODE_MODEL="gemma3-4b",
            UNIFIED="1",
            VOCAB_CHUNK_I2="5",
            LM_HEAD="0",
            NLAYERS="1",
            DECODE_GOLDEN="1",
            DECODE_GOLDEN_L=str(self.ATTN_MAXL),
        )
        spec = importlib.util.spec_from_file_location(
            "fu_gemma", str(_DEC / "fused_decode.py")
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
        _ensure_requant_cache(fd, model)

        self.UNI_DEC = fd.UNI_DEC
        self.UNI_LM = fd.UNI_LM
        self.K = fd.K
        self.DH = fd.DH
        self.RMS_LAYER = fd.RMS_LAYER
        self.ROPE_W_LEN = fd.ROPE_W_LEN
        self.KVSZ_TOK = fd.KVSZ_TOK
        self.DK_TOT_A = fd.DK_TOT_A
        self.NGRP = fd.NGRP
        self.REGION_W = fd.REGION_W
        self.REGION_STRIDE = fd.REGION_STRIDE
        self.VP = fd.VOCAB_SIZE_PADDED
        self.VOCAB_SIZE = fd.VOCAB_SIZE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + self.UNI_LM * self.VP
        # RMS BO: [UNI_DEC per-layer norm slabs | UNI_DEC per-layer rope_w slabs | final_norm]
        self._rope_base = self.UNI_DEC * self.RMS_LAYER
        self._final_off = self._rope_base + self.UNI_DEC * self.ROPE_W_LEN
        self._RMS_SIZE = self._final_off + self.K
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK

        # host weights: embed (x0 gather), final_norm (RMS BO), per-layer qk-norm (rope_w).
        self.qm = gw.Q4nxModel(model)
        embed, final_norm, _lm = self.qm.embed_norm_lmhead()
        self.embed = np.asarray(embed, bfloat16).reshape(-1, self.K)
        self.final_norm = np.asarray(final_norm, bfloat16)
        self.qk = [
            self.qm.layer_qk_norm(L) for L in range(self.UNI_DEC)
        ]  # (qn,kn) [DH]

        # decode weights (q4k-cascade) + 4 RMS norm stacks from the requant cache.
        _z = np.load(os.environ["Q4NX_GEMMA_DECODE_NPZ"])
        W = _z["W"].view(bfloat16)
        self.Wv16 = W.view(np.int16) if W.dtype != np.int16 else W
        _rms = {
            n: list(_z[n].view(bfloat16))
            for n in ("RMS_in", "RMS_post_attn", "RMS_pre_ffn", "RMS_post_ffn")
        }
        self.rms_slabs = np.concatenate(
            [
                np.concatenate(
                    [
                        _rms["RMS_in"][k],
                        _rms["RMS_post_attn"][k],
                        _rms["RMS_pre_ffn"][k],
                        _rms["RMS_post_ffn"][k],
                    ]
                )
                for k in range(self.UNI_DEC)
            ]
        )
        assert self.rms_slabs.size == self._rope_base, (
            self.rms_slabs.size,
            self._rope_base,
        )

        print(
            f"[decode] gemma3-4b ONE xclbin: ATTN_MAXL={self.ATTN_MAXL}, serves "
            f"L in [1,{self.maxL}]; {self.UNI_DEC} layers + lm-head/dispatch",
            flush=True,
        )

        # ONE xclbin + ONE self-contained BO set (weight BO uploaded once)
        self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._kern = _stair.open_windows(self.dev, xrt, self.gen, self.windows)
        self.cur_maxl = self.ATTN_MAXL
        self.kern = self._kern[self.cur_maxl][1]
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.UNI_DEC * self.LREG * 2, HO, g(7))
        self._ist = _stair.make_insts_states(
            self.gen, xrt, self.dev, g(1), self.windows
        )
        self._geom = _stair.KVGeometry(
            self.UNI_DEC, self.KVSZ_TOK, self.REGION_W, self.NGRP, self.LREG
        )
        self._use_window(self.cur_maxl)
        self.w_bo.write(self.Wv16, 0)
        self.w_bo.sync(TO)
        self.KV = np.zeros((self.UNI_DEC, self.LREG), dtype=bfloat16)

    def _use_window(self, m):
        """Point the active kernel / insts state at window `m` (no KV movement)."""
        self._st = self._ist[m]
        self.cur_maxl = m
        self.kern = self._kern[m][1]
        self.ib = self._st["ib"]

    def seed_kv(self, fk, fv, P):
        """Place the numpy-prefill K/V (fk/fv: [UNI_DEC,P,DK_TOT_A]) into the device KV cache
        region-major, then prefetch to the device (before the timed decode loop)."""
        np = self.np
        if len(self.windows) > 1:
            self._use_window(self.gen.window_for_L(P + 1))
        RW, NG = self.REGION_W, self.NGRP
        RS = self.cur_maxl * RW
        self.KV[:] = 0
        for Lyr in range(self.UNI_DEC):
            for gi in range(NG):
                self.KV[Lyr, gi * RS : gi * RS + P * RW].reshape(P, RW)[:] = fk[
                    Lyr, :P, gi * RW : (gi + 1) * RW
                ].astype(self.bf16)
                self.KV[Lyr, (NG + gi) * RS : (NG + gi) * RS + P * RW].reshape(P, RW)[
                    :
                ] = fv[Lyr, :P, gi * RW : (gi + 1) * RW].astype(self.bf16)
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        _lreg = self._geom.lreg(self.cur_maxl)
        for Lyr in range(self.UNI_DEC):
            boff = Lyr * _lreg * 2
            self.kvc.write(self.KV[Lyr, :_lreg].view(np.int16), boff)
            self.kvc.sync(TO, _lreg * 2, boff)
        self._kv_dirty = False

    def _rope_slab(self, p):
        """Build the UNI_DEC per-layer rope_w slabs for position p (dual-theta cos/sin +
        per-layer q/k-norm), concatenated: [layer0 (ROPE_W_LEN) | layer1 | ...]."""
        np = self.np
        return np.concatenate(
            [
                self.gw.rope_w_layer(p, L, self.qk[L][0], self.qk[L][1])
                for L in range(self.UNI_DEC)
            ]
        ).astype(self.bf16)

    def dispatch(self, tok, p):
        """One decode step at L=p+1: patch insts for L, write x0/rope, dispatch (34 layers +
        lm-head), return logits. Appends the new token's K/V at slot p on-device."""
        np = self.np
        xrt = self.xrt
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        L = p + 1
        # insts: write full stream once, then patch only the L-dependent [lo:hi] slice.
        if len(self.windows) > 1:
            _w = self.gen.window_for_L(L)
            if _w != self.cur_maxl:
                # `p` positions are live; this token's K/V is appended by the dispatch.
                _stair.respace_kv(self.kvc, self._geom, self.cur_maxl, _w, p, xrt)
                self._use_window(_w)
        insts_size = _stair.patch_insts(self._st, L, xrt, TO)
        # KV uploaded once (seed), then device-resident (kernel appends in place).
        if getattr(self, "_kv_dirty", True):
            packed = np.ascontiguousarray(self.KV).reshape(-1)
            self.kvc.write(packed.view(np.int16), 0)
            self.kvc.sync(TO)
            self._kv_dirty = False
        x0 = np.asarray(
            self.embed[tok], self.bf16
        )  # NO embed re-scale (bundle pre-scaled)
        # RMS BO: norm slabs + final_norm constant; write once, then patch the per-layer
        # rope region each token (positions change cos/sin).
        if not hasattr(self, "_rms_init"):
            _rmsbuf = np.concatenate(
                [
                    self.rms_slabs,
                    np.zeros(self.UNI_DEC * self.ROPE_W_LEN, self.bf16),
                    self.final_norm,
                ]
            )
            assert _rmsbuf.size == self._RMS_SIZE, (_rmsbuf.size, self._RMS_SIZE)
            self.r_bo.write(_rmsbuf.view(np.int16), 0)
            self.r_bo.sync(TO)
            self._rms_init = True
        rope = self._rope_slab(p)
        self.r_bo.write(rope.view(np.int16), self._rope_base * 2)
        self.r_bo.sync(TO, rope.size * 2, self._rope_base * 2)
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(TO)
        st = self.kern(
            3,
            self.ib,
            insts_size,
            self.x_bo,
            self.w_bo,
            self.r_bo,
            self.y_bo,
            self.kvc,
            *_dyn.dispatch_args(self.gen, L),
        ).wait(60000)
        _voc_n = self.UNI_LM * self.VP
        self.y_bo.sync(FROM, _voc_n * 2, self.decode_y * 2)
        # Zero-copy view into the BO (the shared infra's readback idiom): bo.read()
        # returns a buffer whose stride metadata is pyxrt-build dependent, and
        # .view() on it raises on some runners.
        yv = np.frombuffer(
            self.y_bo.map(), dtype=self.bf16, count=_voc_n, offset=self.decode_y * 2
        ).astype(np.float32)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        return yv[: self.VOCAB_SIZE]

    # Kernels, insts states and BOs are all created against self.dev, but
    # self.dev is assigned first and CPython clears an instance __dict__ in
    # insertion order, so the device is released before the objects that
    # depend on it. Linux XRT tolerates that; Windows XRT faults with an
    # access violation while the decoder is collected. `ib` and `_st` alias
    # into `_ist`, so they have to be dropped ahead of it or they keep that
    # state (and its cacheable BO) alive past the device.
    _XRT_RELEASE_ORDER = (
        "ib",
        "_st",
        "_ist",
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
    """Batched AIR prefill on the NPU -> (Kc, Vc, first_token, ttft_s).

    Kc/Vc are [NUM_LAYERS, P, DK] roped-K / raw-V, the layout FusedDecoder.seed_kv
    consumes. The prefill runs at a fixed padded seq_len (the GEMM registry carries
    the Gemma shapes at M=2048), so its cost is ~constant in prompt length.

    ttft_s times the prefill dispatch only. Building the engine (ELF cache) and
    load_weights (Q4NX host dequant of ~6 GB + the one-time write of every weight
    into its resident BO) are model-load costs paid once per process, and are
    reported separately."""
    import os
    import time

    from gemma3_4b_q4nx_prefill import GemmaQ4nxPrefill

    seq_len = seq_len or int(os.environ.get("Q4NX_SEQ_LEN", "2048"))
    t_load = time.perf_counter()
    pf = GemmaQ4nxPrefill(
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
    Kc, Vc = pf.kv_stack()
    return Kc, Vc, int(logits.argmax()), ttft


def generate(prompt, n_tokens, model=MODEL_DEFAULT, greedy=True, numpy_prefill=False):
    import numpy as np, time
    import gemma3_4b_q4nx_weights as gw

    src = "numpy reference" if numpy_prefill else "AIR NPU"
    print(
        f"[inference] {src} prefill (KV seed + first token), "
        f"prompt_len={len(prompt)}...",
        flush=True,
    )
    if numpy_prefill:
        t0 = time.perf_counter()
        qm = gw.Q4nxModel(model)
        Kc, Vc, logits = gw.forward_prompt(qm, prompt)
        first = int(logits[-1].argmax())
        ttft = time.perf_counter() - t0  # the numpy path fuses load and compute
    else:
        Kc, Vc, first, ttft = _prefill_npu(prompt, model)
    P = Kc.shape[1]
    print(f"[inference] prefill first token = {first} (Paris=9079)", flush=True)
    # Machine-readable line for bench/extract_perf.py (nightly LLM dashboard).
    print(f"Time to first token (TTFT): {ttft:.3f}s", flush=True)

    dec = FusedDecoder(
        model=model,
        max_L=P + n_tokens,
        staircase=os.environ.get("DECODE_STAIRCASE") == "1",
    )
    n_eff = min(n_tokens, dec.ATTN_MAXL - P)
    if n_eff <= 0:
        print(f"[inference] P={P} >= ATTN_MAXL={dec.ATTN_MAXL}; abort")
        return [first]
    dec.seed_kv(Kc, Vc, P)
    tokens = list(prompt) + [first]
    gen_ids = [first]
    t_dec0 = time.perf_counter()
    for p in range(P, P + n_eff):
        lg = dec.dispatch(tokens[p], p)
        pred = int(lg.argmax())
        if pred in EOS_IDS:
            print(f"[inference] pos{p} L={p+1} -> EOS ({pred}), stop", flush=True)
            break
        gen_ids.append(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
    t_dec = time.perf_counter() - t_dec0
    n_gen = len(gen_ids) - 1
    if n_gen > 0:
        print(
            f"[inference] decode: {n_gen} tokens in {t_dec:.2f}s "
            f"{n_gen/t_dec:.2f} tok/s ({t_dec/n_gen*1000:.1f} ms/token)",
            flush=True,
        )
        # Machine-readable lines for bench/extract_perf.py (nightly LLM dashboard).
        print(
            f"[inference] Inference: prompt_len={len(prompt)}, n_tokens={n_gen}",
            flush=True,
        )
        print(f"Tokens/second: {n_gen/t_dec:.2f}", flush=True)
    return gen_ids


def _first_diff(got, want):
    """Index of the first differing token, or the length of the shorter run."""
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            return i
    return min(len(got), len(want))


def _paris_verdict(gen_ids):
    """Lines to print for the Paris gate. The two misses are separated because they
    point at different halves: the first token comes out of the prefill, the rest
    out of the decode loop."""
    n = min(len(gen_ids), len(PARIS_GREEDY))
    if not gen_ids or gen_ids[0] != PARIS_FIRST:
        return [f"*** MISS *** first token {gen_ids[:1]}, expected [{PARIS_FIRST}]"]
    if gen_ids[:n] != PARIS_GREEDY[:n]:
        return [
            f"*** MISS *** decode drifted at token {_first_diff(gen_ids, PARIS_GREEDY)}",
            f"    expected {PARIS_GREEDY[:n]}",
            f"    got      {gen_ids[:n]}",
        ]
    return ["*** PARIS ***"]


def _detok(ids, model=MODEL_DEFAULT):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model).decode(ids)
    except Exception as e:
        return f"(no detok: {e}) ids={ids}"


def main():
    ap = argparse.ArgumentParser(description="gemma3_4b_q4nx full inference (decode)")
    ap.add_argument("--prompt", type=str, default=None, help="prompt text")
    ap.add_argument("--prompt-ids", type=str, default=None, help="comma-separated ids")
    ap.add_argument("--n-tokens", type=int, default=9, help="tokens to generate")
    ap.add_argument(
        "--model", type=str, default=MODEL_DEFAULT, help="model.q4nx dir/path"
    )
    ap.add_argument(
        "--numpy-prefill",
        action="store_true",
        help="seed the KV cache with the numpy reference forward instead of the "
        "AIR NPU prefill (the oracle it is checked against; tens of seconds)",
    )
    args = ap.parse_args()

    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
    elif args.prompt:
        from transformers import AutoTokenizer

        prompt = AutoTokenizer.from_pretrained(args.model).encode(args.prompt)
    else:
        prompt = PARIS_PROMPT
    print(f"[inference] prompt = {len(prompt)} tokens: {prompt}", flush=True)

    gen_ids = generate(
        prompt, args.n_tokens, model=args.model, numpy_prefill=args.numpy_prefill
    )
    print("=" * 60)
    print(f"[inference] gen ids: {gen_ids}")
    print(f"[inference] TEXT: {_detok(gen_ids, args.model)!r}")
    if prompt == PARIS_PROMPT:
        for line in _paris_verdict(gen_ids):
            print(line)


if __name__ == "__main__":
    main()
