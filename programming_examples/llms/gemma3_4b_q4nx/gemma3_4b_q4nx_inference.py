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
  python3 gemma3_4b_q4nx_inference.py                 # Paris gate (greedy, first token 9079)
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


def _pick_decode_gen(dec_dir, max_L=None):
    sys.path.insert(0, str(_DEC))
    from decode_insts_gen import DecodeInstsGen

    return DecodeInstsGen(str(dec_dir), max_L=max_L)


class FusedDecoder:
    """One-xclbin Gemma3-4B fused decode. A SINGLE decode xclbin built at ATTN_MAXL serves
    every L in [1, ATTN_MAXL] via a per-token RTP-L + KV-append insts patch (DecodeInstsGen).
    The weight + KV BOs are uploaded once; the kernel appends each new token's K/V in place.
    """

    def __init__(self, model=MODEL_DEFAULT, max_L=None):
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
        self.ATTN_MAXL = self.gen.attn_maxl
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
        xb = xrt.xclbin(self.gen.xclbin)
        self.dev.register_xclbin(xb)
        self._ctx = xrt.hw_context(self.dev, xb.get_uuid())
        self._xb = xb
        xk = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
        self.kern = xrt.kernel(self._ctx, xk.get_name())
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.UNI_DEC * self.LREG * 2, HO, g(7))
        self.ib = xrt.bo(self.dev, self.gen.base.nbytes, xrt.bo.cacheable, g(1))
        self.w_bo.write(self.Wv16, 0)
        self.w_bo.sync(TO)
        self.KV = np.zeros((self.UNI_DEC, self.LREG), dtype=bfloat16)

    def seed_kv(self, fk, fv, P):
        """Place the numpy-prefill K/V (fk/fv: [UNI_DEC,P,DK_TOT_A]) into the device KV cache
        region-major, then prefetch to the device (before the timed decode loop)."""
        np = self.np
        RW, RS, NG = self.REGION_W, self.REGION_STRIDE, self.NGRP
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
        for Lyr in range(self.UNI_DEC):
            boff = Lyr * self.LREG * 2
            self.kvc.write(self.KV[Lyr].view(np.int16), boff)
            self.kvc.sync(TO, self.LREG * 2, boff)
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
        if not hasattr(self, "_ld"):
            i1 = self.gen.insts_for_L(1)
            i2 = self.gen.insts_for_L(2)
            ld = np.where(i1 != i2)[0]
            self._ld = ld
            self._ld_lo = int(ld.min())
            self._ld_hi = int(ld.max()) + 1
            self._ld_base = i1[ld].astype(np.int64)
            self._ld_slope = i2[ld].astype(np.int64) - i1[ld].astype(np.int64)
            self._insts_buf = i1.astype(np.uint32).copy()
            self._insts_size = int(i1.size)
            self.ib.write(self._insts_buf, 0)
            self.ib.sync(TO)
        self._insts_buf[self._ld] = (self._ld_base + (L - 1) * self._ld_slope).astype(
            np.uint32
        )
        _lo, _hi = self._ld_lo, self._ld_hi
        self.ib.write(self._insts_buf[_lo:_hi], _lo * 4)
        self.ib.sync(TO, (_hi - _lo) * 4, _lo * 4)
        insts_size = self._insts_size
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
            3, self.ib, insts_size, self.x_bo, self.w_bo, self.r_bo, self.y_bo, self.kvc
        ).wait(60000)
        _voc_n = self.UNI_LM * self.VP
        self.y_bo.sync(FROM, _voc_n * 2, self.decode_y * 2)
        yv = (
            self.y_bo.read(_voc_n * 2, self.decode_y * 2)
            .view(self.bf16)
            .astype(np.float32)
        )
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        return yv[: self.VOCAB_SIZE]


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

    dec = FusedDecoder(model=model, max_L=P + n_tokens)
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
        print(
            "*** PARIS ***" if gen_ids and gen_ids[0] == PARIS_FIRST else "*** MISS ***"
        )


if __name__ == "__main__":
    main()
