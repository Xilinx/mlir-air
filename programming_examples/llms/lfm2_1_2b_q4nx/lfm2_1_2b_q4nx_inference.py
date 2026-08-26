# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""lfm2_1_2b_q4nx full inference -- batched NPU prefill + fused NPU decode.

Two paths over one model, as every other llms/ example does: a batched prefill
(`lfm2_1_2b_q4nx_prefill.py`) fills the device-resident per-layer state, then a
fused per-token decode (one dispatch = 16 layers + LM head) reads and extends
it. Embedding, sampling, the chat template and streaming stay on the host.

## What is LFM2-specific

LFM2 is a HYBRID: 6 of its 16 layers are attention and the other 10 are
Lfm2ShortConv, a gated causal depthwise convolution. So the state the prefill
hands to the decode is TWO regions in one buffer, not one:

    arg4 = [ 16 KV-cache slabs | 16 ShortConv-state slabs ]

  * attention layers seed a region-major K/V slab, exactly as a pure
    transformer does;
  * ShortConv layers seed a carried state -- the last `conv_L_cache - 1` rows of
    the PRE-convolution gated signal. Causality is a left PAD, not a mask, so
    that state IS the pad the next token's convolution consumes.

Every layer gets a slot in BOTH regions even though it only ever uses one. That
keeps the decode's per-layer address a plain `iv * SLAB` and costs a few hundred
KB; the alternative is an irregular per-layer offset table threaded through
every DMA in the launch.

Usage:
  python3 lfm2_1_2b_q4nx_inference.py                 # Paris gate (default prompt)
  python3 lfm2_1_2b_q4nx_inference.py --n-tokens 64 --prompt "..."
  python3 lfm2_1_2b_q4nx_inference.py --interactive   # chat REPL
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PE = _HERE.parent.parent  # programming_examples
_DEC = _PE / "fused_decode"  # standalone fused superkernel decode example

# decode_staircase lives in fused_decode/, which only joins sys.path when a decoder is
# constructed, so it is imported lazily and cached here for the methods that need it.
_stair = None


def _staircase_on():
    """Multi-window decode (smallest ATTN_MAXL covering L). Env so every entry point --
    CLI, verify adapter, lit -- opts in the same way."""
    return os.environ.get("DECODE_STAIRCASE") == "1"


def _load_stair():
    global _stair
    if _stair is None:
        import decode_staircase

        _stair = decode_staircase
    return _stair


# Tokenizer: LFM2 publishes ONE checkpoint, so the tokenizer, the NPU weight
# source and the bf16 verify reference are all the same repo.
_TOKENIZER = os.environ.get("LFM2_MODEL_SOURCE") or "LiquidAI/LFM2-1.2B"
# Tokenized lazily rather than hardcoded as ids: LFM2's vocab is its own, and a
# stale id list is a silent wrong-prompt bug.
PARIS_TEXT = "The capital of France is"


def _lfm2_rope(n_pos, dim, theta=1000000.0):
    """RoPE cos/sin [n_pos, dim] for LFM2.

    Plain RoPE at rope_theta = 1e6. NO llama3 frequency scaling: LFM2's config
    carries no `rope_scaling`, so the piecewise wavelength rescale the Llama-3.2
    driver applies would be wrong here -- and wrong in a way that only shows up
    as degraded long-context quality, never as an error.
    """
    import numpy as np

    inv = 1.0 / (theta ** (np.arange(0, dim, 2) / dim))
    fr = np.arange(n_pos)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], axis=1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


_LFM2_CACHE = os.path.expanduser("~/.cache/lfm2_1_2b_q4nx")


def _ensure_paris_golden():
    """Directory holding the host-side embed table and final norm as f32.

    The decode's host half needs the embedding (to look up each new token) and
    the final RMSNorm weight. Both are bf16 in the checkpoint and are NOT
    quantized, so they are dumped straight through rather than round-tripped.
    """
    import numpy as np

    gd = os.environ.get("LFM2_GOLDEN_DIR")
    if gd and os.path.exists(os.path.join(gd, "weights", "embed_tokens.f32.bin")):
        return gd
    from lfm2_1_2b_q4nx_prefill import MODEL_DEFAULT
    from lfm2_1_2b_q4nx_weights import Lfm2Q4Model

    out = os.path.join(_LFM2_CACHE, "golden")
    os.makedirs(os.path.join(out, "weights"), exist_ok=True)
    if not os.path.exists(os.path.join(out, "weights", "embed_tokens.f32.bin")):
        qm = Lfm2Q4Model(os.environ.get("LFM2_MODEL_SOURCE", MODEL_DEFAULT))
        qm.bf("model.embed_tokens.weight").astype(np.float32).tofile(
            os.path.join(out, "weights", "embed_tokens.f32.bin")
        )
        # LFM2 names the final norm `model.embedding_norm`, not `model.norm`.
        qm.bf("model.embedding_norm.weight").astype(np.float32).tofile(
            os.path.join(out, "weights", "final_norm.f32.bin")
        )
    os.environ["LFM2_GOLDEN_DIR"] = out
    return out


def _ensure_requant_cache(fd):
    """Path to the decode's packed Q4_0 weight cache, building it if absent.

    Keyed on the layouts that change the packing, so a warm cache cannot be fed
    to a build that wants a different one -- each of these would otherwise be a
    silent wrong-weights run rather than an error:
      * W_DUAL_CHAN reorders the cascade into per-channel halves;
      * VOCAB_CHUNK_I2 sets how the lm-head rows are split across vocab waves.
    """
    rc = os.environ.get("LFM2_DECODE_WEIGHTS_NPZ")
    if rc and os.path.exists(rc):
        return rc
    import lfm2_requant

    src = os.environ.get("LFM2_MODEL_SOURCE") or "LiquidAI/LFM2-1.2B"
    if not rc:
        _w2 = "_w2ch" if getattr(fd, "W_DUAL_CHAN", 0) else ""
        _v = f"_v{getattr(fd, 'VOCAB_I2', 0)}"
        _tag = re.sub(r"[^A-Za-z0-9]+", "_", src).strip("_")
        rc = os.path.join(_LFM2_CACHE, f"requant_{_tag}{_w2}{_v}.npz")
    if not os.path.exists(rc):
        os.makedirs(os.path.dirname(rc), exist_ok=True)
        # layer_kind="all": ONE uniform per-layer slab covering both layer
        # types, which is what makes the decode's weight offset a plain
        # iv * W_LAYER.
        lfm2_requant.build_requant_cache(fd, rc, model=src, layer_kind="all")
    os.environ["LFM2_DECODE_WEIGHTS_NPZ"] = rc
    return rc


# ------------------------------------------------------------------ prefill worker
def _prefill_worker(prompt, out_path, seq_len, warm_ttft=False):
    """Runs in a subprocess: batched prefill -> per-layer roped-K/raw-V + first token."""
    sys.path.insert(0, str(_HERE))
    import numpy as np
    from lfm2_1_2b_q4nx_prefill import Lfm2Q4nxPrefill

    m = Lfm2Q4nxPrefill(seq_len=seq_len)
    m.compile()
    m.load_weights()
    logits = np.asarray(m.prefill(prompt), np.float32)
    first = int(logits.argmax())
    cfg = m.config
    P = m.get_current_context_length()
    HALO = cfg.conv_L_cache - 1
    # Both regions, one per layer, zero where the layer is of the other kind.
    # Dense arrays rather than a dict: they cross a process boundary as an npz,
    # and the decoder indexes them by MODEL layer.
    K = np.zeros((cfg.n_layers, P, cfg.kv_dim), np.float32)
    V = np.zeros((cfg.n_layers, P, cfg.kv_dim), np.float32)
    S = np.zeros((cfg.n_layers, HALO, cfg.conv_dim), np.float32)
    for l in range(cfg.n_layers):
        if cfg.is_attn_layer(l):
            K[l] = np.asarray(m.get_k_cache(l), np.float32)
            V[l] = np.asarray(m.get_v_cache(l), np.float32)
        else:
            S[l] = np.asarray(m.get_conv_state(l), np.float32)
    np.savez(out_path, k=K, v=V, s=S, first=first, prompt=np.array(prompt))
    print(
        f"[prefill] ctx={P} first_token={first} K{K.shape} conv_state{S.shape}",
        flush=True,
    )
    if warm_ttft:
        # Steady-state TTFT: repeat the prefill with weights already resident in
        # their BOs, so the number reflects prefill latency rather than the
        # one-time host weight load (~90% of the cold figure). Same methodology
        # as `make profile-prefill`; done after the KV is saved so it cannot
        # perturb the decode seed. Handed back via a sidecar so the parent can
        # subtract it from its own wall clock and keep the cold number honest.
        import time

        m.clear_context()
        t0 = time.perf_counter()
        m.prefill(prompt)
        Path(str(out_path) + ".warm").write_text(f"{time.perf_counter() - t0:.6f}")


def run_prefill(prompt, seq_len, kv_path, warm_ttft=False):
    """Spawn the prefill worker (clean NPU release before decode)."""
    # Note: avoid the token "prompt_len=" here so it doesn't shadow the canonical
    # "Inference: prompt_len=<seq_len>" line that bench/extract_perf.py parses (it
    # takes the first match).
    print(
        f"[inference] prefill (seq_len={seq_len}, prompt={len(prompt)} tok)...",
        flush=True,
    )
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_prefill-worker",
        "--_kv-path",
        str(kv_path),
        "--seq-len",
        str(seq_len),
        "--prompt-ids",
        ",".join(map(str, prompt)),
    ]
    if warm_ttft:
        cmd.append("--_warm-ttft")
    subprocess.run(cmd, check=True)


def _compile_only(seq_len=2048):
    """Build/cache the prefill ELFs and exit (no weights, no NPU dispatch).

    This is the weight-free CI smoke signal (`make compile`). Constructing
    Lfm2Q4nxPrefill compiles the prefill engines; the fused decode templates
    are a separate ~15 min build (`make compile-decode`)."""
    sys.path.insert(0, str(_HERE))
    from lfm2_1_2b_q4nx_prefill import Lfm2Q4nxPrefill

    print(
        f"[inference] compile-only: building prefill ELFs (seq_len={seq_len})...",
        flush=True,
    )
    Lfm2Q4nxPrefill(seq_len=seq_len).compile()
    print("Compilation passed.", flush=True)


# ------------------------------------------------------------------ fused decoder
def _pick_decode_gen(dec_dir, max_L=None):
    """Return the decode insts generator.

    Default: the compile-time decode_L<M> template (DecodeInstsGen, ATTN_MAXL=2048,
    masked-block skip) serving every L in [1, 2048] by extrapolating its
    L-dependent words.

    DECODE_DYNSEQ=1: a build that takes the context length as a runtime scalar, so
    the stream is assembled per token from the compiler-emitted TXN builder. The
    readback then moves this token's context instead of the padded ATTN_MAXL --
    what the staircase approximates with a template per window, exactly and from a
    single build."""
    sys.path.insert(0, str(dec_dir))
    from decode_dynseq import pick_insts_gen

    return pick_insts_gen(dec_dir, max_L=max_L)


class FusedDecoder:
    """One-xclbin fused decode -- the faithful the reference analog (one MAX_L=2048 region-major KV cache;
    the per-token sequence specialized on the host by an RTP-L + KV-append-offset insts patch).

    A SINGLE decode xclbin built at ATTN_MAXL=2048 serves EVERY L in [1, 2048]: the core runs a
    compile-time 128-block attention loop and skips fully-masked far blocks (rem = L - 16*blk <= 0),
    so one build covers all L. `DecodeInstsGen` patches the L-dependent insts words per token. One
    xclbin, one BO set; the weight + KV BOs are uploaded once and the kernel appends each new
    token's K/V in place."""

    def __init__(self, max_L=None, staircase=False):
        HF = (
            _ensure_paris_golden()
        )  # embed/norm from model.q4nx (single source) if not provided
        import importlib.util
        import numpy as np
        from ml_dtypes import bfloat16
        import pyxrt as xrt

        sys.path.insert(0, str(_DEC))
        _load_stair()
        self.np = np
        self.bf16 = bfloat16
        self.xrt = xrt
        # Decode generator: DecodeInstsGen (decode_L<M>) -- ONE compile-time MAX_L=2048 build; the
        # kernel skips masked blocks and single-buffers the block loop, so it serves every L in
        # [1, 2048] via an RTP-L + append insts patch (the reference one-MAX_L design).
        # The decode templates live in THIS example's directory (the Makefile
        # writes decode_L<ATTN_MAXL>.* here), unlike the llama example whose
        # decode is built inside fused_decode/ itself. The BUILDER still comes
        # from fused_decode/.
        self.gen = _pick_decode_gen(_HERE, max_L)
        # Staircase: hold every calibrated ATTN_MAXL window and dispatch each token on the
        # smallest one covering L (the readback streams ATTN_MAXL positions regardless).
        # Off by default -- one window, identical to the single-template path.
        self.windows = _stair.resolve_windows(self.gen, staircase)
        self.ATTN_MAXL = max(self.windows)
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL

        # decode-module constants at DECODE_GOLDEN_L=ATTN_MAXL -- the LREG/ny geometry must
        # match the xclbin the DecodeInstsGen base was compiled at. The decode is always
        # REGION-MAJOR (the reference quadrants K03|K47|V03|V47 + fire-and-free readback, ~50 tok/s @2K);
        # seed_state lays the seeded prefill K/V out region-major to match the decode module.
        # UNI_DEC/UNI_LM are fixed constants in fused_decode.py (LFM2-1.2B: 16/4).
        os.environ.update(
            # The builder is SHARED by every fused-decode model, so the model
            # name has to be set explicitly here: unset it defaults to
            # llama-3.2-1b, whose vocab does not divide by LFM2's chunking and
            # which therefore fails an assert rather than silently mis-sizing.
            DECODE_MODEL="lfm2-1.2b",
            # in_proj arrives as CONV_WAVES landings of the attention ph0 width;
            # must match what the templates and shortconv.o were built with.
            CONV_WAVES=os.environ.get("CONV_WAVES", "2"),
            UNIFIED="1",
            # Must match the value the decode templates were BUILT with; honour an
            # explicit override so the lm-head wave count can be varied. It is
            # PAIRED with UNI_LM -- their product is fixed by the vocab size -- so
            # overriding this one alone trips fused_decode.py's
            # `UNI_LM == N_VOCAB_CHUNKS` assert. That is deliberate: the assert names
            # both values, and a silent mismatch would sweep the wrong vocab length.
            VOCAB_CHUNK_I2=os.environ.get("VOCAB_CHUNK_I2", "16"),
            LM_HEAD="0",
            NLAYERS="1",
            DECODE_GOLDEN="1",  # boolean flag: enable post-attn-RMS decode path
            DECODE_GOLDEN_L=str(self.ATTN_MAXL),
        )
        spec = importlib.util.spec_from_file_location(
            "fu", str(_DEC / "fused_decode.py")
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
        _ensure_requant_cache(
            fd
        )  # build decode weights from model.q4nx if not provided
        self.UNI_LM = fd.UNI_LM
        self.K = fd.K
        RMS_LAYER = fd.RMS_LAYER
        self.KVSZ_TOK = fd.KVSZ_TOK
        self.DK_TOT_A = fd.DK_TOT_A
        self.NGRP = fd.NGRP
        self.REGION_W = fd.REGION_W
        self.REGION_STRIDE = fd.REGION_STRIDE
        self.VP = fd.VOCAB_SIZE_PADDED
        self.VPF = fd.VOCAB_SIZE_PADDED_FULL
        self.VOCAB_SIZE = fd.VOCAB_SIZE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.DH = 64
        self.ny = self.decode_y + self.UNI_LM * self.VP
        # LFM2's rms/rope BO is NOT llama's [16 norm slabs | one 64-wide rope
        # LUT | final norm]. Every layer carries its own ROPE_W_LEN slab, and
        # what lives in it depends on the layer type:
        #   attention -> [cos(DH/2) | sin(DH/2) | q_norm(DH) | k_norm(DH)]
        #   ShortConv -> the depthwise taps, TAP-MAJOR [w0 | w1 | w2]
        # so it is 3*conv_dim wide, not 64, and only the cos/sin of the
        # ATTENTION layers is position-dependent.
        self.ROPE_W_LEN = fd.ROPE_W_LEN
        self.ATTN_IDXS = tuple(fd.ATTN_LAYERS)
        self._RMS_SIZE = 16 * RMS_LAYER + 16 * self.ROPE_W_LEN + self.K
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK
        # LFM2 arg4 = [N_LAYERS KV slabs | N_LAYERS ShortConv-state slabs].
        # CONV_ST_LAYER is [BX(t-2) | BX(t-1)] per layer; every layer gets a slot
        # in BOTH regions even though it uses exactly one, which is what keeps
        # the decode's per-layer address a plain iv * SLAB.
        self.N_LAYERS = fd.UNI_DEC
        self.CONV_ST_LAYER = 2 * fd.CONV_DIM
        self.CONV_ST_BASE = self.N_LAYERS * self.LREG
        self.NKV_TOTAL = self.CONV_ST_BASE + self.N_LAYERS * self.CONV_ST_LAYER
        _kind = type(self.gen).__name__
        _mech = "compile-time 128-block loop, masked-block skip (RTP-L + append patch)"
        print(
            f"[decode] ONE xclbin ({_kind}): ATTN_MAXL={self.ATTN_MAXL}, serves "
            f"L in [1,{self.maxL}] -- {_mech}",
            flush=True,
        )

        # decode weights (q4k-cascade) + rope (all positions) + host embed
        _z = np.load(os.environ["LFM2_DECODE_WEIGHTS_NPZ"])
        # LFM2's packer stores the per-layer slabs and the LM-head vocab waves
        # as SEPARATE arrays, so the decode's weight BO is their concatenation.
        # Uploading _z["W"] alone leaves the 4 vocab waves unwritten: the first
        # token still looks right because it comes from the PREFILL, and only
        # the decoded continuation is wrong -- so this fails as garbage text
        # rather than as an error. Assert the total against the compiled
        # signature below so it cannot silently drift again.
        W = np.concatenate(
            [_z["W"].reshape(-1).view(bfloat16), _z["WV"].reshape(-1).view(bfloat16)]
        )
        self.Wv16 = W.view(np.int16)
        RMS_in = list(_z["RMS_in"].view(bfloat16))
        RMS_post = list(_z["RMS_post"].view(bfloat16))
        self.final_norm = np.asarray(
            np.fromfile(f"{HF}/weights/final_norm.f32.bin", np.float32), bfloat16
        )
        self.embed = np.memmap(
            f"{HF}/weights/embed_tokens.f32.bin", np.float32, "r"
        ).reshape(self.VOCAB_SIZE, self.K)
        self.rope_cos, self.rope_sin = _lfm2_rope(self.ATTN_MAXL, self.DH)
        self.rms_slabs = np.concatenate(
            [np.concatenate([RMS_in[k], RMS_post[k]]) for k in range(16)]
        )
        # Per-layer rope_w, straight from the packer: QK-norm weights on the
        # attention layers, tap-major conv taps on the ShortConv ones. cos/sin
        # occupy [0:DH] of an attention layer's slab and are written per token.
        self.rope_w = _z["ROPE_W"].view(bfloat16).reshape(16, self.ROPE_W_LEN).copy()
        self._rope_base = int(self.rms_slabs.size)

        # ONE xclbin + ONE self-contained BO set (weight BO uploaded once)
        self.dev = xrt.device(0)
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._kern = _stair.open_windows(self.dev, xrt, self.gen, self.windows)
        self.cur_maxl = self.ATTN_MAXL
        self.kern = self._kern[self.cur_maxl][1]
        g = self.kern.group_id
        HO = xrt.bo.host_only
        self.x_bo = xrt.bo(self.dev, self.K * 2, HO, g(3))
        # Check the weight vector against the TEMPLATE'S OWN compiled signature,
        # not against the builder constants loaded here -- those two can
        # disagree, and that disagreement is the failure this catches.
        #
        # `make compile-decode LAYERS=N` writes a short template under the same
        # decode_L<ATTN_MAXL>.* name a full build uses, and DecodeInstsGen picks
        # a template by CONTEXT REACH, not by layer count. So a leftover 1-layer
        # bisect build gets selected for a short prompt and runs one layer of a
        # sixteen-layer model. The first token still looks right (it comes from
        # the PREFILL), and only the continuation is wrong -- it presents as
        # garbage text, not as an error.
        _sig = self._template_n_w()
        if _sig is not None and W.size != _sig:
            raise RuntimeError(
                f"decode template {self.gen.xclbin} streams {_sig} bf16 of "
                f"weights but the packed cache has {W.size} "
                f"({fd.UNI_DEC} layers x {fd.W_LAYER} + lm head). This is what a "
                f"leftover `LAYERS=N` bisect build looks like -- run "
                f"`make clean && make compile-decode` to rebuild the full model."
            )
        self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.NKV_TOTAL * 2, HO, g(7))
        self._ist = _stair.make_insts_states(
            self.gen, xrt, self.dev, g(1), self.windows
        )
        self._geom = _stair.KVGeometry(
            16, self.KVSZ_TOK, self.REGION_W, self.NGRP, self.LREG
        )
        self._use_window(self.cur_maxl)
        self.w_bo.write(self.Wv16, 0)
        self.w_bo.sync(TO)
        # Host mirror of arg4, flat. Per layer the KV slab is region-major
        # ([K_g0 | K_g1 | V_g0 | V_g1], each region ATTN_MAXL*REGION_W); the
        # ShortConv states follow all of them.
        self.KVC = np.zeros(self.NKV_TOTAL, dtype=bfloat16)
        self.KV = self.KVC[: 16 * self.LREG].reshape(16, self.LREG)
        self.CST = self.KVC[self.CONV_ST_BASE :].reshape(
            self.N_LAYERS, self.CONV_ST_LAYER
        )

    def _template_n_w(self):
        """Weight-BO length from the selected template's compiled signature.

        None when the build predates the Makefile emitting decode_L*.air.mlir,
        in which case the caller skips the check rather than inventing one.
        """
        import re as _re

        xb = str(getattr(self.gen, "xclbin", "") or "")
        mlir = xb[: -len(".xclbin")] + ".air.mlir" if xb.endswith(".xclbin") else ""
        if not mlir or not os.path.exists(mlir):
            return None
        for line in open(mlir):
            if "func.func @q4nx_decode" in line:
                sizes = [int(n) for n in _re.findall(r"memref<(\d+)xbf16>", line)]
                return sizes[1] if len(sizes) > 1 else None
        return None

    def _use_window(self, m):
        """Point the active kernel / insts state at window `m` (no KV movement)."""
        self._st = self._ist[m]
        self.cur_maxl = m
        self.kern = self._kern[m][1]
        self.ib = self._st["ib"]

    def seed_state(self, fk, fv, fs, P):
        """Seed BOTH prefill-fed regions of arg4 and prefetch them.

        LFM2 hands the decode two different things, so this is not just a KV
        seed:

          fk/fv [N_LAYERS, P, kv_dim]  attention layers' K/V, rows for conv
                                       layers are zero and never read;
          fs    [N_LAYERS, HALO, conv_dim]
                                       ShortConv layers' carried state, rows for
                                       attention layers likewise unread.

        Uploaded HERE, before the decode loop is timed, so the host->device copy
        is not charged to the first token. The BO then stays resident for the
        turn: the kernel appends each new token's K/V and rewrites each conv
        state in place.
        """
        np = self.np
        if len(self.windows) > 1:
            self._use_window(self.gen.window_for_L(P + 1))
        RW, NG = self.REGION_W, self.NGRP
        RS = self.cur_maxl * RW
        self.KVC[:] = 0
        # Region-major: scatter each group's K (resp V) into its contiguous
        # region. fk[Lyr] is [P, kv_dim] = [g0 K(RW) | g1 K(RW) | ...]; region g
        # slot pos sits at g*RS + pos*RW.
        for Lyr in range(self.N_LAYERS):
            for g in range(NG):
                self.KV[Lyr, g * RS : g * RS + P * RW].reshape(P, RW)[:] = fk[
                    Lyr, :P, g * RW : (g + 1) * RW
                ].astype(self.bf16)
                self.KV[Lyr, (NG + g) * RS : (NG + g) * RS + P * RW].reshape(P, RW)[
                    :
                ] = fv[Lyr, :P, g * RW : (g + 1) * RW].astype(self.bf16)
        # ShortConv state, [BX(t-2) | BX(t-1)] per layer -- the same oldest-first
        # order the kernel reads and rewrites.
        if fs is not None:
            self.CST[:, :] = np.asarray(fs, self.bf16).reshape(self.N_LAYERS, -1)
        # One upload of the whole buffer: the seeded slots are scattered across
        # the four KV regions and then the state region, so a per-region sync
        # would be many small transfers for the same bytes.
        TO = self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.kvc.write(np.ascontiguousarray(self.KVC).view(np.int16), 0)
        self.kvc.sync(TO)
        self._kv_dirty = False  # already uploaded; dispatch skips it

    # Kept under the pure-transformer name the shared driver contract uses, so
    # a caller written against another model still works; it just cannot seed
    # the ShortConv half.
    def seed_kv(self, fk, fv, P):
        return self.seed_state(fk, fv, None, P)

    def dispatch(self, tok, p):
        """One decode step at L=p+1 on the single xclbin: patch insts for L (RTP-L + KV-append
        offset), pack the full KV, dispatch (16 layers + lm_head), append the new token's KV at
        slot p, return logits."""
        np = self.np
        xrt = self.xrt
        import os as _os, time as _t

        _prof = _os.environ.get("DECODE_PROF") == "1"
        _tk = _t.perf_counter
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        L = p + 1
        _a = _tk()
        # Cache the insts BO: the per-L patch touches only a handful of L-dependent words
        # (RTP-L + append/readback offsets, located by diffing two builds). Write the full
        # 780KB stream ONCE, then each token overwrite only the changed [lo:hi] range and
        # sync just that slice -- avoids re-writing/re-syncing the whole cacheable BO.
        if len(self.windows) > 1:
            _w = self.gen.window_for_L(L)
            if _w != self.cur_maxl:
                # `p` positions are live; this token's K/V is appended by the dispatch.
                _stair.respace_kv(self.kvc, self._geom, self.cur_maxl, _w, p, xrt)
                self._use_window(_w)
        insts_size = _stair.patch_insts(self._st, L, xrt, TO)
        _t_insts = _tk() - _a
        _a = _tk()
        # KV cache is uploaded ONCE (seeded prefill positions) then left device-resident:
        # the decode kernel appends each new token's K/V in-place at slot L-1, so the 16x
        # 2048x KVSZ_TOK (~67MB) buffer must NOT be re-packed/re-synced every token (that
        # host copy dominated the per-token time, capping the chatbot at ~24 tok/s vs the
        # ~43 tok/s device rate). the reference likewise appends incrementally.
        if getattr(self, "_kv_dirty", True):
            packed = np.ascontiguousarray(self.KVC).reshape(-1)
            self.kvc.write(packed.view(np.int16), 0)
            self.kvc.sync(TO)
            self._kv_dirty = False
        _t_kv = _tk() - _a
        _a = _tk()
        x0 = np.asarray(self.embed[tok], self.bf16)
        # RMS BO: the norm slabs, the QK-norm weights and the conv taps are
        # constant, so the whole stream is written ONCE. Per token only the
        # cos/sin of each ATTENTION layer's rope_w slab changes, and each is a
        # DH-wide patch -- the conv layers' slabs (their taps) are never
        # touched.
        if not hasattr(self, "_rms_written"):
            _rmsbuf = np.concatenate(
                [self.rms_slabs, self.rope_w.reshape(-1), self.final_norm]
            )
            self.r_bo.write(np.ascontiguousarray(_rmsbuf).view(np.int16), 0)
            self.r_bo.sync(TO)
            self._rms_written = True
        _half = self.DH // 2
        lut = np.empty(self.DH, dtype=self.bf16)
        lut[:_half] = self.rope_cos[p][:_half].astype(self.bf16)
        lut[_half:] = self.rope_sin[p][:_half].astype(self.bf16)
        _lut16 = lut.view(np.int16)
        for _li in self.ATTN_IDXS:
            _off = self._rope_base + _li * self.ROPE_W_LEN
            self.r_bo.write(_lut16, _off * 2)
            self.r_bo.sync(TO, self.DH * 2, _off * 2)
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(TO)
        # y BO: the kernel overwrites the vocab output region every dispatch, so the old
        # per-token full-y zeroing (ny int16 write + sync) is unnecessary.
        _t_io = _tk() - _a
        _a = _tk()
        # A dynseq build's runtime sequence takes the context length as a trailing
        # scalar, so the kernel signature carries it. The value the hardware acts on
        # is already assembled into the stream above; this keeps the arity right.
        import decode_dynseq as _dyn

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
        _t_dev = _tk() - _a
        _a = _tk()
        # only the vocab logits (UNI_LM*VP at decode_y) are needed -- sync+read+convert just
        # that region, not the whole y BO.
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
        # yv is already the vocab region (read at decode_y); take the real vocab length.
        logits = yv[: self.VOCAB_SIZE]
        _t_read = _tk() - _a
        if _prof:
            print(
                f"[prof] p={p} L={L} insts={_t_insts*1e3:.1f} kv={_t_kv*1e3:.1f} "
                f"io={_t_io*1e3:.1f} DEV={_t_dev*1e3:.1f} read={_t_read*1e3:.1f} ms",
                flush=True,
            )
        return logits[: self.VOCAB_SIZE]

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


# ------------------------------------------------------------------ orchestration
EOS_IDS = (128001, 128009)  # <|end_of_text|>, <|eot_id|>


class Sampler:
    """Port of the reference's sampler (DECODE_DLL/common/modules/sampler.cpp): sliding-window
    repetition + frequency penalties, temperature, top-k, top-p, then sample. the reference's llm
    hosts use temperature=0.7, top_k=5, top_p=0.9 (not greedy) -- that stochasticity is
    what keeps it out of the greedy repeat-loop."""

    def __init__(
        self,
        np,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        rep_penalty=1.0,
        freq_penalty=1.0,
        rep_penalty_window=1024,
        freq_penalty_window=1024,
        seed=0,
    ):
        self.np = np
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.rep_penalty = rep_penalty
        self.freq_penalty = freq_penalty
        self.rep_penalty_window = rep_penalty_window
        self.freq_penalty_window = freq_penalty_window
        self.rng = np.random.default_rng(seed)
        self.last_pos = {}  # token_id -> last position seen
        self.counts = {}  # token_id -> count in freq window
        self.hist = []  # freq window ring
        self.total = 0

    def sample(self, logits):
        np = self.np
        logits = logits.astype(np.float32).copy()
        # (A) repetition penalty (sign-based), sliding window
        if self.rep_penalty != 1.0 and self.rep_penalty_window > 0:
            for tid, lp in self.last_pos.items():
                if self.total - lp < self.rep_penalty_window:
                    v = logits[tid]
                    logits[tid] = (
                        v * self.rep_penalty if v < 0 else v / self.rep_penalty
                    )
        # (B) frequency penalty
        if self.freq_penalty != 1.0 and self.freq_penalty_window > 0:
            for tid, c in self.counts.items():
                logits[tid] -= (self.freq_penalty - 1.0) * (
                    c / self.freq_penalty_window
                )
        # (C) temperature
        logits = (logits - logits.max()) / self.temperature
        # (D) top-k
        k = min(self.top_k, logits.size)
        idx = np.argpartition(logits, -k)[-k:]
        idx = idx[np.argsort(logits[idx])[::-1]]
        probs = np.exp(logits[idx])
        probs /= probs.sum()
        # (E) top-p cutoff
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, self.top_p)) + 1
        idx = idx[:cut]
        probs = probs[:cut]
        probs /= probs.sum()
        # (F) sample
        tok = int(self.rng.choice(idx, p=probs))
        # (G) bookkeeping
        if self.freq_penalty_window > 0:
            self.hist.append(tok)
            self.counts[tok] = self.counts.get(tok, 0) + 1
            if len(self.hist) > self.freq_penalty_window:
                old = self.hist.pop(0)
                self.counts[old] -= 1
        self.last_pos[tok] = self.total
        self.total += 1
        return tok


def generate(
    prompt,
    n_tokens,
    seq_len,
    kv_path,
    temperature=0.7,
    top_k=5,
    top_p=0.9,
    rep_penalty=1.0,
    stop_on_eos=True,
    seed=0,
    greedy=False,
    profile=False,
):
    import numpy as np, time

    t_ttft0 = time.perf_counter()
    warm_path = Path(str(kv_path) + ".warm")
    warm_path.unlink(missing_ok=True)  # never report a stale run's number
    run_prefill(prompt, seq_len, kv_path, warm_ttft=profile)
    warm_ttft = float(warm_path.read_text()) if warm_path.exists() else None
    pf = np.load(kv_path)
    fk = pf["k"].astype(np.float32)
    fv = pf["v"].astype(np.float32)
    fs = pf["s"].astype(np.float32) if "s" in pf.files else None
    first = int(pf["first"])
    P = fk.shape[1]
    ttft = time.perf_counter() - t_ttft0 - (warm_ttft or 0.0)
    print(f"[inference] prefill first token = {first}", flush=True)
    # Canonical driver lines consumed by bench/extract_perf.py (shared with the
    # other llms/ examples). The cold number is the whole first-query pipeline
    # (worker spawn + host weight load + prefill + KV handoff), which the
    # one-time 1.8GB weight load dominates; the warm number is the steady-state
    # prefill latency the perf dashboard tracks.
    print(f"Time to first token (TTFT): {ttft:.2f}s", flush=True)
    if warm_ttft is not None:
        print(f"Warm time to first token (TTFT): {warm_ttft:.3f}s", flush=True)

    # ONE decode xclbin serves L in [1, ATTN_MAXL]; the decoder picks the template that covers
    # the requested reach (rt<M> for short, compile-time L<M> up to 2048). Cap at its ATTN_MAXL.
    dec = FusedDecoder(P + n_tokens, staircase=_staircase_on())
    attn_maxl = dec.ATTN_MAXL
    n_eff = min(n_tokens, attn_maxl - P)
    if n_eff <= 0:
        print(
            f"[inference] prompt P={P} >= decode ATTN_MAXL={attn_maxl}; "
            f"build a larger decode template; abort"
        )
        return [first]
    if n_eff < n_tokens:
        print(
            f"[inference] cap: ATTN_MAXL={attn_maxl}, P={P} -> at most {n_eff} decode tokens",
            flush=True,
        )
    dec.seed_state(fk, fv, fs, P)
    sampler = Sampler(
        np,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        rep_penalty=rep_penalty,
        seed=seed,
    )
    tokens = list(prompt) + [first]
    gen_ids = [first]
    import os as _os

    _dbg_tok = _os.environ.get("DECODE_VERBOSE") == "1"
    t_decode0 = time.perf_counter()
    for p in range(P, P + n_eff):
        logits = dec.dispatch(tokens[p], p)
        pred = int(logits.argmax()) if greedy else sampler.sample(logits)
        if stop_on_eos and pred in EOS_IDS:
            print(f"[inference] pos{p:2d} L={p+1} -> EOS ({pred}), stop")
            break
        gen_ids.append(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
        # per-token diagnostic line is off by default (the f-string + write lands inside the
        # decode timing); the full generated TEXT still prints after the loop. Set
        # DECODE_VERBOSE=1 to stream per-token.
        if _dbg_tok:
            print(f"[inference] pos{p:2d} L={p+1} -> {pred}")
    t_decode = time.perf_counter() - t_decode0
    n_gen = len(gen_ids) - 1  # exclude the prefill-provided first token
    if n_gen > 0:
        tps = n_gen / t_decode
        # Canonical driver lines consumed by bench/extract_perf.py: the
        # parenthesized "(Y.YY tok/s)" is the throughput the perf harness parses.
        # "prompt_len" reports the NOMINAL prefill/decode context window (seq_len,
        # == the decoder's ATTN_MAXL), matching the sibling drivers which print the
        # padded seq_len (2048) — so the nightly's context_len column stays
        # consistent. (Both here and the siblings decode from the real prompt
        # position onward, so per-token tok/s is measured at the same real depth.)
        print(
            f"Generated {n_gen} tokens in {t_decode:.2f}s ({tps:.2f} tok/s)",
            flush=True,
        )
        print(f"Inference: prompt_len={seq_len}, n_tokens={n_gen}", flush=True)
        if profile:
            print(
                f"[profile] decode {t_decode / n_gen * 1000:.1f} ms/token, "
                f"prefill TTFT {ttft:.2f}s, P={P}",
                flush=True,
            )
    return gen_ids


def _template_ids(tk, messages, add_generation_prompt=True):
    """apply_chat_template -> flat list[int] (robust to dict/BatchEncoding/batch-dim)."""
    ids = tk.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


class _StreamState:
    """Tracks how many characters of the running decoded text have already been emitted.
    BPE tokens can decode to '' alone but combine into characters with later tokens, so the
    safe streaming pattern is to decode the full id list each step and emit only the new
    suffix."""

    def __init__(self):
        self.printed_len = 0


def _delta_text(tk, ids, state):
    """New text since the last call (advancing state); skips special tokens (EOS/eot)."""
    decoded = tk.decode(ids, skip_special_tokens=True)
    delta = decoded[state.printed_len :]
    state.printed_len = len(decoded)
    return delta


class Session:
    """Resident the reference-faithful pipeline: prefill (Lfm2Q4nxPrefill) and decode (FusedDecoder) are BOTH
    loaded once at init and kept resident (proven to coexist on the NPU), mirroring the sibling
    llms' build_session preload. Per-turn TTFT is then prefill COMPUTE only -- the ~1.8 GB weight
    load happens once at startup, not on every turn."""

    def __init__(self, seq_len=2048):
        import numpy as np, time

        self.seq_len = seq_len  # nominal prefill/decode context window (reported)
        sys.path.insert(0, str(_HERE))
        sys.path.insert(0, str(_DEC))
        from lfm2_1_2b_q4nx_prefill import Lfm2Q4nxPrefill

        self.np = np
        t0 = time.perf_counter()
        print("[session] loading prefill weights (once)...", flush=True)
        self.prefiller = Lfm2Q4nxPrefill(seq_len=seq_len)
        self.prefiller.load_weights()
        print(
            f"[session] prefill resident ({time.perf_counter() - t0:.2f}s); building decode...",
            flush=True,
        )
        self.dec = FusedDecoder(
            staircase=_staircase_on()
        )  # largest available decode context, resident (weights + xclbin)
        self.attn_maxl = self.dec.ATTN_MAXL
        # Warmup: one throwaway prefill so the FIRST user turn is warm (~1.07s) instead of the
        # ~1.2s cold start. run_turn clear_context()s the KV before every real turn, so this
        # leaves NO conversation state -- it only warms the resident kernels / BOs / dispatch path.
        t_w = time.perf_counter()
        self.prefiller.clear_context()
        self.prefiller.prefill(
            [128000]
        )  # 1-token dummy, padded to seq_len -> warms the full path
        self.prefiller.clear_context()
        _win = (
            f", staircase windows={self.dec.windows}"
            if len(self.dec.windows) > 1
            else ""
        )
        print(
            f"[session] ready: prefill+decode resident (ATTN_MAXL={self.attn_maxl}{_win}, "
            f"weights preloaded, warmup {time.perf_counter() - t_w:.2f}s).",
            flush=True,
        )

    def run_turn(
        self,
        ids,
        n_tokens=None,
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        rep_penalty=1.0,
        greedy=False,
        stop_on_eos=True,
        seed=0,
        on_token=None,
    ):
        np = self.np
        import time

        # --- prefill: compute only, weights already resident (fresh full-context prefill) ---
        t_ttft0 = time.perf_counter()
        self.prefiller.clear_context()
        logits = np.asarray(self.prefiller.prefill(list(ids)), np.float32)
        first = int(logits.argmax())
        cfg = self.prefiller.config
        P = self.prefiller.get_current_context_length()
        HALO = cfg.conv_L_cache - 1
        K = np.zeros((cfg.n_layers, P, cfg.kv_dim), np.float32)
        V = np.zeros((cfg.n_layers, P, cfg.kv_dim), np.float32)
        S = np.zeros((cfg.n_layers, HALO, cfg.conv_dim), np.float32)
        for _l in range(cfg.n_layers):
            if cfg.is_attn_layer(_l):
                K[_l] = np.asarray(self.prefiller.get_k_cache(_l), np.float32)
                V[_l] = np.asarray(self.prefiller.get_v_cache(_l), np.float32)
            else:
                S[_l] = np.asarray(self.prefiller.get_conv_state(_l), np.float32)
        ttft = time.perf_counter() - t_ttft0
        print(f"[inference] prefill first token = {first}", flush=True)
        print(f"Time to first token (TTFT): {ttft:.2f}s", flush=True)
        n_eff = min(n_tokens or self.attn_maxl, self.attn_maxl - P)
        if n_eff <= 0:
            print(
                f"[inference] prompt P={P} >= ATTN_MAXL={self.attn_maxl}; abort",
                flush=True,
            )
            return [first]
        # --- decode: reset KV, seed from this turn's prefill, generate ---
        self.dec.KVC[:] = 0
        self.dec.seed_state(K, V, S, P)
        sampler = Sampler(
            np,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            seed=seed,
        )
        tokens = list(ids) + [first]
        gen_ids = [first]
        if on_token:
            on_token(gen_ids)  # emit the prefill's first token immediately
        t_decode0 = time.perf_counter()
        for p in range(P, P + n_eff):
            lg = self.dec.dispatch(tokens[p], p)
            pred = int(lg.argmax()) if greedy else sampler.sample(lg)
            if stop_on_eos and pred in EOS_IDS:
                if not on_token:
                    print(
                        f"[inference] pos{p:2d} L={p+1} -> EOS ({pred}), stop",
                        flush=True,
                    )
                break
            gen_ids.append(pred)
            if p + 1 >= len(tokens):
                tokens.append(pred)
            if on_token:
                on_token(gen_ids)  # stream the new token
        t_decode = time.perf_counter() - t_decode0
        if on_token:
            sys.stdout.write("\n")
            sys.stdout.flush()  # end the streamed line
        n_gen = len(gen_ids) - 1
        if n_gen > 0:
            print(
                f"Generated {n_gen} tokens in {t_decode:.2f}s "
                f"({n_gen / t_decode:.2f} tok/s)",
                flush=True,
            )
            print(f"Inference: prompt_len={self.seq_len}, n_tokens={n_gen}", flush=True)
        return gen_ids


def interactive_chat(
    system=None,
    seq_len=2048,
    temperature=0.7,
    top_k=5,
    top_p=0.9,
    rep_penalty=1.0,
    greedy=False,
    seed=0,
):
    """the reference-faithful multi-turn REPL (single_turn_conversation applied per turn with an
    accumulating message history): each turn re-applies the chat template to the full
    conversation, prefills it, and decodes until EOS. Prefill+decode are PRELOADED once via
    Session and reused across turns (resident, no per-turn weight reload). `/clear` resets the
    conversation (the reference clear_context), `/exit` quits. Context is capped by the decode ATTN_MAXL.
    """
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(_TOKENIZER)
    sess = Session(seq_len=seq_len)  # preload prefill + decode ONCE
    attn_maxl = sess.attn_maxl

    def _fresh():
        return [{"role": "system", "content": system}] if system else []

    messages = _fresh()
    print(
        f"[chat] the reference-faithful interactive chat (ATTN_MAXL={attn_maxl}, weights preloaded). "
        f"Commands: /clear (reset), /exit (quit).",
        flush=True,
    )
    turn = 0
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/clear":
            messages = _fresh()
            print("[chat] context cleared.", flush=True)
            continue
        messages.append({"role": "user", "content": user})
        ids = _template_ids(tk, messages)
        if len(ids) >= attn_maxl:
            print(
                f"[chat] conversation is {len(ids)} tokens >= ATTN_MAXL={attn_maxl}; "
                f"use /clear (or build a larger decode_rt<M> template).",
                flush=True,
            )
            messages.pop()
            continue
        # stream the reply token-by-token (BPE-safe delta), prefixed with "Assistant: "
        stream_state = _StreamState()
        started = [False]

        def _on_tok(gen_so_far):
            if not started[0]:
                sys.stdout.write("\nAssistant: ")
                started[0] = True
            d = _delta_text(tk, gen_so_far, stream_state)
            if d:
                sys.stdout.write(d)
                sys.stdout.flush()

        if len(sess.dec.windows) > 1:
            print(
                f"[chat] context {len(ids)} tok -> KV window {sess.dec.gen.window_for_L(len(ids) + 1)}",
                flush=True,
            )
        gen_ids = sess.run_turn(
            ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rep_penalty=rep_penalty,
            greedy=greedy,
            seed=seed + turn,
            on_token=_on_tok,
        )
        answer = tk.decode(
            [g for g in gen_ids if g not in EOS_IDS], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": answer})
        turn += 1


def _detok(ids):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_TOKENIZER).decode(ids)
    except Exception as e:
        return f"(no detok: {e}) ids={ids}"


def _chat_encode(text, system=None):
    """the reference tokenize_messages equivalent: wrap the prompt as a chat turn and apply the
    model's chat template with the assistant generation prompt (see the reference
    apps/qwen35/host.cpp single_turn_conversation)."""
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(_TOKENIZER)
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}
    ]
    return _template_ids(tk, msgs)


def main():
    ap = argparse.ArgumentParser(
        description="lfm2_1_2b_q4nx full inference (prefill+decode)"
    )
    ap.add_argument(
        "--prompt", type=str, default=None, help="prompt text (chat-templated)"
    )
    ap.add_argument(
        "--prompt-ids", type=str, default=None, help="comma-separated token ids (raw)"
    )
    ap.add_argument(
        "--raw", action="store_true", help="encode --prompt raw (no chat template)"
    )
    ap.add_argument("--system", type=str, default=None, help="optional system prompt")
    ap.add_argument(
        "--n-tokens", type=int, default=9, help="tokens to generate (<= ATTN_MAXL-P)"
    )
    # Canonical driver contract (shared with the other llms/ examples): the
    # Makefile run/profile/chat targets pass these through.
    ap.add_argument(
        "--model",
        type=str,
        default="instruct",
        help="model variant: 'instruct' (chat template) or 'base' (raw encode). "
        "Also selects the HF bf16 reference in verify_adapter.py.",
    )
    ap.add_argument(
        "--run-only",
        action="store_true",
        help="run inference (kernels are compiled on demand / from cache). "
        "Accepted for parity with the other examples; running is the default.",
    )
    ap.add_argument(
        "--compile-only",
        action="store_true",
        help="build/cache the prefill ELFs and exit (no weights, no NPU dispatch). "
        "The fused decode templates are a separate build: `make compile-decode`.",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="print the decode-throughput / TTFT profiling summary (machine-readable "
        "lines consumed by bench/extract_perf.py).",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.7, help="the reference default 0.7"
    )
    ap.add_argument("--top-k", type=int, default=5, help="the reference default 5")
    ap.add_argument(
        "--top-p", type=float, default=0.9, help="the reference default 0.9"
    )
    ap.add_argument(
        "--rep-penalty",
        type=float,
        default=1.0,
        help="the reference sampler rep penalty (1.0=off)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="sampler seed (the reference re-seeds by time)",
    )
    ap.add_argument(
        "--greedy",
        action="store_true",
        help="greedy argmax instead of the reference sampling",
    )
    ap.add_argument("--no-eos-stop", action="store_true", help="do not stop at EOS")
    ap.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="prefill padded length (256/512/1024/2048)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="multi-turn chat REPL (the reference-faithful)",
    )
    ap.add_argument("--_prefill-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_warm-ttft", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument(
        "--_kv-path", type=str, default="/tmp/e2e_prefill.npz", help=argparse.SUPPRESS
    )
    args = ap.parse_args()

    if args._prefill_worker:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
        _prefill_worker(prompt, args._kv_path, args.seq_len, warm_ttft=args._warm_ttft)
        return

    if args.compile_only:
        _compile_only(args.seq_len)
        return

    if args.interactive:
        interactive_chat(
            system=args.system,
            seq_len=args.seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            rep_penalty=args.rep_penalty,
            greedy=args.greedy,
            seed=args.seed,
        )
        return

    # 'base' variant (or --raw) encodes without the chat template.
    raw_encode = args.raw or args.model == "base"
    _is_paris_gate = False
    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
    elif args.prompt:
        if raw_encode:
            from transformers import AutoTokenizer

            prompt = AutoTokenizer.from_pretrained(_TOKENIZER).encode(args.prompt)
        else:
            prompt = _chat_encode(
                args.prompt, system=args.system
            )  # the reference chat turn
    else:
        # Tokenized here, not a hardcoded id list: LFM2 has its own vocab.
        from transformers import AutoTokenizer

        prompt = AutoTokenizer.from_pretrained(_TOKENIZER).encode(PARIS_TEXT)
        _is_paris_gate = True
    print(f"[inference] prompt = {len(prompt)} tokens", flush=True)

    gen_ids = generate(
        prompt,
        args.n_tokens,
        args.seq_len,
        args._kv_path,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        rep_penalty=args.rep_penalty,
        stop_on_eos=not args.no_eos_stop,
        seed=args.seed,
        greedy=args.greedy,
        profile=args.profile,
    )
    print("=" * 60)
    print(f"[inference] gen ids: {gen_ids}")
    print(f"[inference] TEXT: {_detok(gen_ids)!r}")
    if _is_paris_gate:
        # Decode the id rather than comparing against a hardcoded one: LFM2 has
        # its own vocab, and a stale id would make this gate pass or fail for
        # the wrong reason.
        _first = _detok([gen_ids[0]]).strip().lower() if gen_ids else ""
        print("*** PARIS ***" if _first == "paris" else f"*** MISS ({_first!r}) ***")


if __name__ == "__main__":
    main()
