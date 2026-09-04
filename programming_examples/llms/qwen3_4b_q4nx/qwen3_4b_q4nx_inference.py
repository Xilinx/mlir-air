#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""qwen3_4b_q4nx full inference -- FLM-faithful Qwen3-4B prefill + decode.

Qwen3-4B is the DFlash TARGET model (docs/DFlashFeasibility.md). This is
qwen3_8b_q4nx_inference.py at Qwen3-4B's dimensions and weight source; the
mechanism is identical.

Reproduces FastFlowLM's Qwen3-4B mechanism end to end on the NPU:

  prefill: the batched AIR prefill (qwen3_4b_q4nx_prefill.Qwen3Q4nxPrefill) -- a
    thin driver over the qwen3_4b builders (rms+QKV+qk-norm+RoPE, head-first
    flash attention, O+residual+RMSNorm, gate/up/SwiGLU, down+add) with resident
    weight BOs, plus an on-device 10-partition LM head GEMV. Produces the
    per-layer roped-K / raw-V KV seed and the greedy first token.
  decode: the shared fused_decode engine (DECODE_MODEL=qwen3-4b) -- 36 decoder
    layers + the TIED lm-head in ONE dispatch, standard 2-norm pre-norm, SiLU
    GLU, single-theta RoPE with per-head qk-norm, appending each new token's K/V
    in place.

`--numpy-prefill` swaps the NPU prefill for the numpy reference forward
(qwen3_8b_q4nx_weights.forward_prompt, which is dimension-generic and reused
here unmodified -- see qwen3_4b_q4nx_weights.py) -- the oracle the prefill is
checked against, kept as a fallback and for debugging (it takes tens of
seconds).

Unlike qwen3_8b_q4nx, the total decode Q4NX weight is ~2.1 GiB (36 layers of
K=2560 vs 8B's K=4096), well under the 4 GiB one-BO shim-BD-offset limit --
so there is no DECODE_WGROUP weight split here.

The decode is ONE xclbin built at ATTN_MAXL (=16*ceil(L/16)); DecodeInstsGen patches
the L-dependent insts words (attention RTP-L + KV-append offset) per token. The RMS BO
holds UNI_DEC per-layer norm slabs [input|post_attn], then UNI_DEC per-layer rope_w
slabs [cos/sin | q_norm | k_norm] (rewritten per position), then the final norm.

Run:
  python3 qwen3_4b_q4nx_inference.py                 # Paris gate (greedy, PARIS_GREEDY)
  python3 qwen3_4b_q4nx_inference.py --prompt "The capital of France is" --n-tokens 12
  python3 qwen3_4b_q4nx_inference.py --numpy-prefill # numpy-oracle prefill instead
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
# (qwen3_4b_q4nx_weights resolves all three).
MODEL_DEFAULT = os.environ.get("Q4NX_MODEL_SOURCE", "FastFlowLM/Qwen3-4B-NPU2")
# "The capital of France is" (Qwen3 tokenizer, no BOS); prefill -> 12095 " Paris"
# (same tokenizer/prompt as qwen3_8b_q4nx -- Qwen3 shares one tokenizer family).
PARIS_PROMPT = [785, 6722, 315, 9625, 374]
PARIS_FIRST = 12095
# Greedy continuation at the gate shape (`make run` uses --n-tokens 9).
# Recorded on real NPU2 hardware with real weights, decoding to " Paris. The
# capital of Germany is Berlin. The" -- fluent, sensible continuation.
# Recorded only AFTER fixing the models/qwen3-4b.h GLU_SLICE bug (1024,
# inherited unmodified from qwen3-8b's header; should have been 512, matching
# this model's own GLU_PKTS parity -- see docs/DFlashFeasibility.md section 8).
# Before that fix the FFN/down-projection's contribution was silently zero for
# every layer and decode produced garbage from the first generated token.
PARIS_GREEDY = [12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576]
EOS_IDS = (151643, 151645)  # <|endoftext|>, <|im_end|>
# No weight split needed (see module docstring): total decode weight is
# ~2.1 GiB, under the 4 GiB one-BO limit that forces qwen3_8b_q4nx's
# DECODE_WGROUP=9. Kept as an override point (0 = disabled) for parity with
# the other Q4NX drivers, not because 4B is expected to need it.
DECODE_WGROUP = int(os.environ.get("Q4NX_DECODE_WGROUP", "0"))
_Q4NX_CACHE = os.path.expanduser("~/.cache/q4nx_qwen3_4b")


def _ensure_requant_cache(fd, model):
    """Return the decode q4k-cascade cache path, building it from model.q4nx on first
    use (one-time ~pack of 36 layers + tied lm-head). Honors Q4NX_QWEN3_4B_DECODE_NPZ.
    """
    rc = os.environ.get("Q4NX_QWEN3_4B_DECODE_NPZ")
    if rc and os.path.exists(rc):
        return rc
    import qwen3_4b_q4nx_requant as rq

    # W_DUAL_CHAN reorders the cascade (the two-MM2S weight feed splits it by
    # cascade pair into [low-row half | high-row half]), so it gets its own cache
    # entry -- a warm single-channel cache would feed the dual-channel xclbin the
    # wrong blocks.
    _w2 = "_w2ch" if getattr(fd, "W_DUAL_CHAN", 0) else ""
    # ... and so does the vocab chunking, for exactly the same reason. The
    # lm-head slab is packed ONE WAVE AT A TIME (qwen3_4b_q4nx_requant.py) and
    # pack_q4k_cascade's outermost loop is the column, so the wave boundary
    # falls INSIDE the cx dimension: 10 waves of VOCAB_CHUNK_I2=30 and 6 of 50
    # cover the same rows in a different order. Without this in the key a v50
    # xclbin warms on a v30 cache and gets the wrong vocab blocks -- no error,
    # just wrong logits.
    # Keyed off the model table's own default (UNI_LM there implies the chunk),
    # so the existing warm cache keeps its name and only a non-default chunking
    # gets a suffix.
    _dflt = fd.VOCAB_FULL_ROWBLKS // (
        fd.MODEL["UNI_LM"] * fd.NCX * fd.NCY * fd.PAIR_ROWS
    )
    _vc = "" if fd.VOCAB_I2 == _dflt else f"_v{fd.VOCAB_I2}"
    rc = rc or os.path.join(_Q4NX_CACHE, f"requant{_w2}{_vc}.npz")
    if not os.path.exists(rc):
        rq.build_requant_cache(model, fd, rc)
    os.environ["Q4NX_QWEN3_4B_DECODE_NPZ"] = rc
    return rc


# Qwen3-4B decode xclbins (decode_L<N>) live in this example's own artifact dir so they
# don't collide with the Llama/Qwen3-8B builds elsewhere. DecodeInstsGen (the L->insts
# affine specializer) is imported from fused_decode/ but scans this dir for templates.
_DECODE_DIR = Path(os.environ.get("Q4NX_QWEN3_4B_DECODE_DIR", str(_HERE)))

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


def _pick_decode_gen(dec_dir, max_L=None, batch=1, prefix=None):
    sys.path.insert(0, str(_DEC))
    global _dyn
    import decode_dynseq as _dyn
    from decode_dynseq import pick_insts_gen

    # A batched build is a DIFFERENT template family (`decode_b<B>_L<N>`) in the
    # same directory, and `batch` also moves which ATTN_MAXL window an L falls
    # in. Passing one without the other picks up the batch-1 templates and
    # dispatches them with batched BOs.
    return pick_insts_gen(
        str(dec_dir),
        max_L=max_L,
        batch=batch,
        prefix=prefix or ("decode_L" if batch == 1 else f"decode_b{batch}_L"),
    )


class FusedDecoder:
    """One-xclbin Qwen3-4B fused decode. A SINGLE decode xclbin built at ATTN_MAXL serves
    every L in [1, ATTN_MAXL] via a per-token RTP-L + KV-append insts patch (DecodeInstsGen).
    The weight + KV BOs are uploaded once; the kernel appends each new token's K/V in place.
    """

    def __init__(
        self,
        model=MODEL_DEFAULT,
        max_L=None,
        staircase=False,
        batch=1,
        template_prefix=None,
        env_extra=None,
        decode_model="qwen3-4b",
        weights=None,
        npz=None,
        artifact_dir=None,
        extra_weights=None,
    ):
        """`batch` > 1 dispatches a block of B CONSECUTIVE positions of one
        sequence -- what DECODE_BATCH means in this engine, and what a
        speculative verify pass needs. It requires templates built at that
        batch (`decode_b<B>_L<N>`), and every batch-sensitive size below comes
        off the builder rather than being restated, so a host that disagrees
        with the build is not expressible. At batch 1 every term is unchanged.

        `env_extra` reaches the builder's import-time environment, which is the
        only way to select a build variant (DECODE_MASK_MODE_RTP for the
        drafter's bidirectional pass, RMS_BAND_STREAM, ...). It must match what
        the template was BUILT with.

        `decode_model` / `weights` / `npz` drive the DFlash DRAFTER through this
        same class: `qwen3-4b-draft` is qwen3-4b's per-layer geometry with
        UNI_DEC=5 (fused_decode.py:427), so nothing below is target-specific
        once the weight source and the requant cache are arguments. `weights` is
        any object with `embed_norm_lmhead()` and `layer_qk_norm(L)`
        (`qwen3_4b_draft_weights.DraftWeights` is one); `npz` is that model's
        own requant cache, and passing the target's would decode fluent
        garbage. `artifact_dir` is where the templates live, so the drafter's
        can sit beside the target's without either scan picking up the other.

        `extra_weights` is the weight BO for a build whose `env_extra` carries
        DECODE_EXTRA_WAVES -- extra launch iterations of the projection engine
        running someone else's matrices (the DFlash pre-pass's fc and context
        K/V; see llms/qwen3_4b_q4nx/dflash_prepass_waves.py). It is a separate
        buffer appended after every existing binding position, so a template
        without extra waves is bound exactly as before.
        """
        import importlib.util
        import numpy as np
        from ml_dtypes import bfloat16
        import pyxrt as xrt
        import qwen3_4b_q4nx_weights as gw

        self.np = np
        self.bf16 = bfloat16
        self.xrt = xrt
        self.gw = gw
        self.batch = int(batch)

        self.decode_model = decode_model
        # Where the templates were found, kept so anything that ships a SECOND
        # instruction stream against this same xclbin (see `dispatch_insts`)
        # looks for it beside them rather than guessing.
        self.artifact_dir = artifact_dir or _DECODE_DIR
        self.gen = _pick_decode_gen(
            artifact_dir or _DECODE_DIR,
            max_L,
            batch=self.batch,
            prefix=template_prefix,
        )
        _load_stair()
        # Staircase: hold every calibrated ATTN_MAXL window and dispatch each token on the
        # smallest one covering L (the readback streams ATTN_MAXL positions regardless).
        # Off by default -- one window, identical to the single-template path.
        self.windows = _stair.resolve_windows(self.gen, staircase)
        self.ATTN_MAXL = max(self.windows)
        self.maxL = min(int(max_L), self.ATTN_MAXL) if max_L else self.ATTN_MAXL

        # DECODE_MODEL / geometry env must be set BEFORE importing fused_decode.py (its
        # module-level constants read them). Matches the other q4nx drivers' env block.
        # VOCAB_CHUNK_I2=30 is qwen3-4b's own value (build_template.sh), not 8B's 8 --
        # the vocab-chunk divisibility constraints depend on this model's own NCX/NCY/
        # PAIR_ROWS geometry, not a number that transfers between models.
        #
        # OVERRIDABLE, because the vocab chunking is not free at batch 8: it
        # sets VOCAB_RNDS, which has to share a re-feed cycle with the decode
        # arm's XN_REFEED + REFEED[GATEUP_PHASE] (see docs/BZeroPlan.md item 1).
        # 30 is the shipping value; 50 is the one that lets RMS_MEMTILE_REFEED=3
        # carry the LM head. The TEMPLATES and the requant cache must agree with
        # whatever is set here -- _ensure_requant_cache keys the cache on it, and
        # a template built at the other chunking has a different wave count.
        _env = dict(
            DECODE_MODEL=decode_model,
            UNIFIED="1",
            VOCAB_CHUNK_I2=os.environ.get("VOCAB_CHUNK_I2", "30"),
            LM_HEAD="0",
            NLAYERS="1",
            DECODE_GOLDEN="1",
            DECODE_GOLDEN_L=str(self.ATTN_MAXL),
        )
        if os.environ.get("UNI_LM"):
            # Paired with VOCAB_CHUNK_I2 above: the two must satisfy
            # UNI_LM * VOCAB_CHUNK_I2 == the padded vocab's row-block count, and
            # fused_decode.py asserts it rather than deriving one from the other.
            _env["UNI_LM"] = os.environ["UNI_LM"]
        if DECODE_WGROUP:
            _env["DECODE_WGROUP"] = str(DECODE_WGROUP)
        if self.batch != 1:
            _env["DECODE_BATCH"] = str(self.batch)
        _env.update(env_extra or {})
        os.environ.update(_env)
        spec = importlib.util.spec_from_file_location(
            "fu_qwen3_4b", str(_DEC / "fused_decode.py")
        )
        fd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fd)
        # The drafter brings its own cache (qwen3_4b_draft_requant.py); only the
        # target's is built on demand here.
        self._npz = npz or _ensure_requant_cache(fd, model)

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
        # Batch-sensitive sizing, mirroring llms/bench/decode_geometry.py (which
        # self-checks against bench_decode.cpp's built-in defaults). B tokens
        # means B embeddings in, B of every drained PAYLOAD row, B rope/qk-norm
        # slabs -- one per POSITION -- and B logit rows out. The KV cache does
        # NOT scale: it is indexed by position, and the block's B tokens occupy
        # B positions of the window that already exists.
        _b = self.batch
        assert getattr(fd, "BATCH", 1) == _b, (
            f"template built at DECODE_BATCH={getattr(fd, 'BATCH', 1)}, driver "
            f"asked for {_b}"
        )
        self.X_SLOTS = fd.X_SLOTS  # >1 only with DECODE_HIDDEN_TAPS
        self.nx = self.X_SLOTS * self.K * _b
        # Extra waves, all zero on a build that has none. X_SLOTS above already
        # covers their output slots: the builder grows the X buffer to whatever
        # the wave list reaches, which is why nothing here restates it.
        self.N_EXTRA = getattr(fd, "N_EXTRA", 0)
        self.EXTRA_W_ELEMS = getattr(fd, "EXTRA_W_ELEMS", 0)
        self.EXTRA_OUT_SLOT = list(getattr(fd, "EXTRA_OUT_SLOT", []))
        self.RMS_ONES_OFF = getattr(fd, "RMS_ONES_OFF", None)
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD * _b
        self.decode_y += getattr(fd, "PROBE_TOTAL", 0)
        self.ny = (
            self.decode_y + self.UNI_LM * self.VP * _b + getattr(fd, "RMS_SCRATCH", 0)
        )
        # RMS BO: [UNI_DEC per-layer norm slabs | B*UNI_DEC rope_w slabs | final_norm]
        # ( | RMS_TAIL_SLACK | K ones, when the build has extra waves)
        self._rope_base = self.UNI_DEC * self.RMS_LAYER
        self._final_off = self._rope_base + _b * self.UNI_DEC * self.ROPE_W_LEN
        self._RMS_SIZE = self._final_off + self.K + getattr(fd, "RMS_TAIL_SLACK", 0)
        # The builder puts the extra waves' ones run at RMS_ONES_OFF, which is
        # exactly the end of everything above -- so the arithmetic here, which
        # is a SECOND COPY of the builder's, has to be checked against it and
        # then extended. Not doing that is not a crash: numpy drops a write that
        # starts at the end of an array, the ones stay zero, every extra wave
        # multiplies its X by zero and returns the residual it was added into
        # unchanged. Measured as target_hidden at cos 0.008 with the dispatch
        # reporting COMPLETED, and fused_decode.py says why the symbol is at
        # module scope at all: "a second copy of this arithmetic on that side is
        # exactly how a silently-wrong buffer happens".
        if self.RMS_ONES_OFF is not None:
            assert self.RMS_ONES_OFF == self._RMS_SIZE, (
                f"host RMS layout disagrees with the builder: ones at "
                f"{self.RMS_ONES_OFF}, host tail at {self._RMS_SIZE}"
            )
            if self.N_EXTRA:
                self._RMS_SIZE = self.RMS_ONES_OFF + self.K
        self.LREG = self.ATTN_MAXL * self.KVSZ_TOK

        # host weights: embed (x0 gather), final_norm (RMS BO), per-layer qk-norm (rope_w).
        self.qm = weights if weights is not None else gw.Q4nxModel(model)
        embed, final_norm, _lm = self.qm.embed_norm_lmhead()  # _lm == embed (TIED)
        self.embed = np.asarray(embed, bfloat16).reshape(-1, self.K)
        self.final_norm = np.asarray(final_norm, bfloat16)
        self.qk = [
            self.qm.layer_qk_norm(L) for L in range(self.UNI_DEC)
        ]  # (qn,kn) [DH]

        # decode weights (q4k-cascade) + the 2 RMS norm stacks from the requant cache.
        _z = np.load(self._npz)
        W = _z["W"].view(bfloat16)
        self.Wv16 = W.view(np.int16) if W.dtype != np.int16 else W
        _rms = {n: list(_z[n].view(bfloat16)) for n in ("RMS_in", "RMS_post")}
        self.rms_slabs = np.concatenate(
            [
                np.concatenate([_rms["RMS_in"][k], _rms["RMS_post"][k]])
                for k in range(self.UNI_DEC)
            ]
        )
        assert self.rms_slabs.size == self._rope_base, (
            self.rms_slabs.size,
            self._rope_base,
        )

        print(
            f"[decode] {decode_model} ONE xclbin: ATTN_MAXL={self.ATTN_MAXL}, "
            f"serves L in [1,{self.maxL}]; {self.UNI_DEC} layers + "
            f"lm-head/dispatch" + (f", batch {_b}" if _b != 1 else ""),
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
        self.x_bo = xrt.bo(self.dev, self.nx * 2, HO, g(3))
        # DECODE_WGROUP: only relevant if explicitly forced on (see module
        # docstring -- 4B's total weight is under the 4 GiB one-BO limit).
        # Taken from the engine, not the environment, so the host slicing cannot
        # disagree with what the template was built for.
        _G, self._wsplit, _ng = fd.W_GROUP, fd.W_SPLIT, fd.N_WGRP
        if self._wsplit:
            _WL = fd.W_LAYER
            _parts = [
                W[gi * _G * _WL : min((gi + 1) * _G, self.UNI_DEC) * _WL]
                for gi in range(_ng)
            ]
            _parts.append(W[self.UNI_DEC * _WL :])  # lm-head slabs
            assert sum(p.size for p in _parts) == W.size, "weight split lost data"
            # group 0 -> g(4); groups 1.. and lm-head -> g(8), g(9), ...
            self.w_bos = [
                xrt.bo(self.dev, p.size * 2, HO, g(4 if i == 0 else 7 + i))
                for i, p in enumerate(_parts)
            ]
            for bo, p in zip(self.w_bos, _parts):
                bo.write(np.ascontiguousarray(p).view(np.int16), 0)
                bo.sync(TO)
            print(
                f"[decode] weight split G={_G}: "
                + ", ".join(f"{p.size*2/2**30:.2f}GiB" for p in _parts),
                flush=True,
            )
            self.w_bo = self.w_bos[0]
        else:
            self.w_bos = None
            self.w_bo = xrt.bo(self.dev, W.size * 2, HO, g(4))
        self.r_bo = xrt.bo(self.dev, self._RMS_SIZE * 2, HO, g(5))
        self.y_bo = xrt.bo(self.dev, self.ny * 2, HO, g(6))
        self.kvc = xrt.bo(self.dev, self.UNI_DEC * self.LREG * 2, HO, g(7))
        # The extra waves' weights, at the binding position the builder appends
        # them to: after W_SPLIT's groups, which is g(8) on every model whose
        # weights fit one BO. Uploaded once, like every other weight here.
        self.e_bo = None
        if self.N_EXTRA:
            if extra_weights is None:
                raise ValueError(
                    f"the template has {self.N_EXTRA} extra waves and "
                    f"{self.EXTRA_W_ELEMS} elements of weights for them, but no "
                    f"extra_weights was passed; the dispatch would read zeros"
                )
            # REINTERPRET, never convert. These are packed q4k blocks, and a
            # requant cache stores them as int16 because they are bytes and not
            # numbers -- `np.asarray(x, bfloat16)` on one turns the bit pattern
            # 15632 into the value 15632.0 and every wave then streams garbage
            # that still dispatches COMPLETED. Same rule as `self.Wv16` above.
            _e = np.asarray(extra_weights).reshape(-1)
            if _e.dtype == np.int16:
                _e = _e.view(bfloat16)
            elif _e.dtype != bfloat16:
                raise TypeError(
                    f"extra_weights is {_e.dtype}; packed weights must arrive as "
                    f"bfloat16 or as the int16 bit pattern of one"
                )
            _e = np.ascontiguousarray(_e)
            if _e.size != self.EXTRA_W_ELEMS:
                raise ValueError(
                    f"extra_weights is {_e.size} elements, the template's wave "
                    f"table says {self.EXTRA_W_ELEMS}"
                )
            _ng = len(self.w_bos) - 1 if self._wsplit else 0
            # The template on disk is what decides whether this argument exists,
            # and the ENVIRONMENT is what said there are extra waves -- so the
            # two can disagree, and when they do XRT raises "invalid vector
            # subscript" from inside group_id, which names neither. Say it here.
            try:
                _gid = g(8 + _ng)
            except Exception:
                raise RuntimeError(
                    f"this build declares {self.N_EXTRA} extra waves but the "
                    f"template {self.gen.templates[self.cur_maxl]['xclbin']} has "
                    f"no extra-weight argument -- it was built without "
                    f"DECODE_EXTRA_WAVES. Rebuild that window's templates, or "
                    f"point --max-L at one that was."
                ) from None
            self.e_bo = xrt.bo(self.dev, _e.size * 2, HO, _gid)
            self.e_bo.write(_e.view(np.int16), 0)
            self.e_bo.sync(TO)
        self._check_arity(g)
        self._ist = _stair.make_insts_states(
            self.gen, xrt, self.dev, g(1), self.windows
        )
        self._geom = _stair.KVGeometry(
            self.UNI_DEC, self.KVSZ_TOK, self.REGION_W, self.NGRP, self.LREG
        )
        self._use_window(self.cur_maxl)
        if not self._wsplit:
            self.w_bo.write(self.Wv16, 0)
            self.w_bo.sync(TO)
        self.KV = np.zeros((self.UNI_DEC, self.LREG), dtype=bfloat16)

    def _use_window(self, m):
        """Point the active kernel / insts state at window `m` (no KV movement)."""
        self._st = self._ist[m]
        self.cur_maxl = m
        self.kern = self._kern[m][1]
        self.ib = self._st["ib"]

    def _check_arity(self, g):
        """Does this host bind every buffer the TEMPLATE declares?

        The environment decides what this driver builds; the file on disk
        decides what the kernel takes. When they disagree the dispatch does not
        fail -- it binds the buffers it has, in order, and the argument nobody
        filled reads whatever is there. Measured: a taps template carrying the
        DFlash pre-pass's waves, opened by a caller that did not know about
        them, gave 0/8 batch-vs-batch-1 agreement and NaNs at every context
        length. Nothing in that output points here.

        `group_id` is the only thing that knows the real arity, so count up
        until it refuses. Only an UNDER-bind is an error: over-binding would
        already have raised where the buffer was allocated.
        """
        want = 5 + (len(self.w_bos) - 1 if self._wsplit else 0)
        want += 1 if self.e_bo is not None else 0
        have = 0
        while have < 64:
            try:
                g(3 + have)
            except Exception:
                break
            have += 1
        if have > want:
            raise RuntimeError(
                f"the template takes {have} buffers and this driver binds "
                f"{want}: argument {3 + want} would go unfilled and the "
                f"dispatch would read whatever is at it. If this is a target "
                f"built with DFlash pre-pass waves, the caller needs "
                f"dflash_prepass_waves.taps_decoder_args()."
            )

    def _init_rms(self):
        """RMS BO: norm slabs + final_norm constant. Written once.

        The per-layer rope region on top of it is per TOKEN and stays in
        `dispatch`; this is only the part that never changes. It is its own
        method because `dispatch_insts` needs it too and may run FIRST -- block
        0 of a DFlash loop dispatches the pre-pass's fc stream before the target
        has decoded anything, and that stream reads the norm weight below.
        """
        if hasattr(self, "_rms_init"):
            return
        np = self.np
        _rmsbuf = np.zeros(self._RMS_SIZE, self.bf16)
        _rmsbuf[: self.rms_slabs.size] = self.rms_slabs
        _rmsbuf[self._final_off : self._final_off + self.K] = self.final_norm
        # ONES for the extra waves' norm weight. An extra wave's X reaches the
        # projection through the rms core -- the only producer on @xnorm with a
        # route to DDR -- and feeding w == 1 leaves `rms_chunk` as the strided
        # gather it already is, with only a per-row scale on top that the host
        # divides back out. See dflash_prepass_waves.assemble.
        if self.RMS_ONES_OFF is not None and self.N_EXTRA:
            _rmsbuf[self.RMS_ONES_OFF : self.RMS_ONES_OFF + self.K] = self.bf16(1.0)
        self.r_bo.write(_rmsbuf.view(np.int16), 0)
        self.r_bo.sync(self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._rms_init = True

    def dispatch_insts(self, insts, timeout=60000):
        """Run a DIFFERENT instruction stream against this same device program.

        UNI_WAVE_LO/HI are build-time and restrict only the fused launch loop --
        "keeps ABI/CDO fixed", fused_decode.py says where it reads them -- so a
        build at a narrower wave range produces the SAME xclbin configuration
        and a shorter insts.bin. Dispatching one here costs no PDI switch, which
        is what lets the DFlash pre-pass's two halves sit at two different
        points in a speculative block instead of in a third program of their own.

        No L patching: a stream of extra waves alone has no decode wave in it,
        so there is nothing in it that depends on the context length. Handing an
        L-dependent stream to this would silently dispatch it at whatever L it
        was compiled for.
        """
        np, xrt = self.np, self.xrt
        self._init_rms()
        insts = np.ascontiguousarray(np.asarray(insts, np.uint8))
        key = insts.nbytes
        cache = self.__dict__.setdefault("_alt_ib", {})
        if key not in cache:
            ib = xrt.bo(self.dev, insts.nbytes, xrt.bo.cacheable, self.kern.group_id(1))
            ib.write(insts, 0)
            ib.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            cache[key] = ib
        _kargs = (
            3,
            cache[key],
            insts.size,
            self.x_bo,
            self.w_bo,
            self.r_bo,
            self.y_bo,
            self.kvc,
            *(self.w_bos[1:] if self._wsplit else ()),
            *((self.e_bo,) if self.e_bo is not None else ()),
        )
        # Same lost-submission recovery as `dispatch` -- see `_wait_dispatch`.
        # Never observed on an alternate stream, but it is the same command path
        # and the same driver, and losing one here kills a run just as dead.
        _poll = int(os.environ.get("DFLASH_DISPATCH_POLL", "1"))
        _retries = int(os.environ.get("DFLASH_DISPATCH_RETRY", "2"))
        st = self._wait_dispatch(self.kern(*_kargs), timeout, _poll, "alternate stream")
        for _i in range(_retries):
            if str(st).endswith("COMPLETED"):
                break
            st = self._wait_dispatch(
                self.kern(*_kargs), timeout, _poll, "alternate stream"
            )
            print(f"[retry] alternate stream attempt {_i + 1}: {st}", flush=True)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"alternate-stream dispatch state={st}")
        return st

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
        """Build the UNI_DEC per-layer rope_w slabs for position p (single-theta cos/sin +
        per-layer q/k-norm), concatenated: [layer0 (ROPE_W_LEN) | layer1 | ...]."""
        np = self.np
        return np.concatenate(
            [
                self.gw.rope_w_layer(p, L, self.qk[L][0], self.qk[L][1])
                for L in range(self.UNI_DEC)
            ]
        ).astype(self.bf16)

    def _rope_block(self, positions):
        """The rope region for a block of B positions -- LAYER-major, then token.

        Layer L's block is at `rms_lut_off + L*B*ROPE_W_LEN`, and token t sits at
        `+ t*ROPE_W_LEN` inside it (fused_decode.py: `_rope_off` indexes the
        per-wave slab by `BATCH*ROPE_W_LEN`, and the per-token puts step by
        ROPE_W_LEN within it). That is the TRANSPOSE of B copies of
        `_rope_slab`, which is layer-major for ONE position -- and the
        difference is not visible in the region size, only in the answer: token
        0 comes back correct (its slab happens to land first either way) and
        every later token gets another layer's rotation. Measured, on the
        Paris prompt: [13, 1096, 279, 315, 279, 30, 279, 30] against a batch-1
        reference of [13, 576, 6722, 315, 9856, 374, 19846, 13].
        """
        np = self.np
        assert len(positions) == self.batch, (len(positions), self.batch)
        return np.concatenate(
            [
                self.gw.rope_w_layer(p, L, self.qk[L][0], self.qk[L][1])
                for L in range(self.UNI_DEC)
                for p in positions
            ]
        ).astype(self.bf16)

    @staticmethod
    def _wait_dispatch(run, timeout_ms, poll_ms, what):
        """Wait for one dispatch, watching `run.state()` rather than blocking in `wait`.

        A DECODE DISPATCH ON THIS MACHINE IS INTERMITTENTLY NEVER SUBMITTED.
        Eleven of ~5600 batch-8 target dispatches -- 0.2%, and six of them in one
        sweep -- did this, every one with the identical trajectory:

            [('ERT_CMD_STATE_NEW', 0.0), ('ERT_CMD_STATE_TIMEOUT', 7.006)]

        NEW at t=0 and NEW until the driver's own ~7 s command timeout fires. It
        never reaches QUEUED, SUBMITTED or RUNNING, so it never touches the
        device: the KV cache the dispatch would have appended to is untouched,
        and re-issuing the identical command is exact rather than merely likely.
        That is not an inference -- `--exactness` reproduces the non-speculative
        stream 192/192 on every prompt that took a re-issue.

        Two consequences shape this function:

        - SPIN ON state(), DO NOT CHUNK wait(). `wait(250)` does not come back in
          250 ms on a lost command; it blocks the driver's full ~7 s. So a
          chunked wait can neither sample the trajectory nor detect the fault
          sooner than one long wait, and `wait(60000)` costs a full minute per
          fault. Spinning on the non-blocking `state()` costs nothing measurable
          -- 103.97 ms per verify block against 103.06 without it.
        - NEVER wait() ON A LOST COMMAND. `state()` already returned the driver's
          verdict; calling `wait` after it re-pays the same 7 s.

        pyxrt exposes no `abort`, so the lost command cannot be cancelled. It
        does not need to be: the driver has already given up on it.
        """
        if not poll_ms:
            return run.wait(timeout_ms)
        import time as _time

        t0 = _time.time()
        traj, done = [], False
        while (_time.time() - t0) * 1e3 < timeout_ms:
            s = str(run.state()).rsplit(".", 1)[-1]
            if not traj or traj[-1][0] != s:
                traj.append((s, round(_time.time() - t0, 3)))
            if s == "ERT_CMD_STATE_COMPLETED":
                done = True
                break
            if s in ("ERT_CMD_STATE_TIMEOUT", "ERT_CMD_STATE_ERROR"):
                break  # the driver has abandoned it; waiting cannot help
            _time.sleep(0.001)
        if done:
            return run.wait(timeout_ms)  # returns at once; lets XRT finalize
        print(f"[poll] {what} lost after {_time.time() - t0:.3f}s, {traj}", flush=True)
        return run.state()

    def dispatch(self, tok, p):
        """One decode step at L=p+1: patch insts for L, write x0/rope, dispatch (36 layers +
        lm-head), return logits. Appends the new token's K/V at slot p on-device.

        At `batch` > 1, `tok` is a sequence of B token ids and `p` the position
        of the FIRST of them; the return is [B, VOCAB_SIZE]. The B tokens are B
        consecutive positions of one sequence, and the RTP-L the insts carry is
        token 0's context length -- the engine derives token t's from it.
        """
        np = self.np
        xrt = self.xrt
        TO = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        FROM = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        B = self.batch
        toks = np.atleast_1d(np.asarray(tok)).reshape(-1)
        assert toks.size == B, (toks.size, B)
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
        # X is B embeddings, token-major. Qwen3 has no embedding scale.
        x0 = np.ascontiguousarray(np.asarray(self.embed[toks], self.bf16).reshape(-1))
        self._init_rms()
        rope = (
            self._rope_block([p + t for t in range(B)]) if B > 1 else self._rope_slab(p)
        )
        self.r_bo.write(rope.view(np.int16), self._rope_base * 2)
        self.r_bo.sync(TO, rope.size * 2, self._rope_base * 2)
        self.x_bo.write(x0.view(np.int16), 0)
        self.x_bo.sync(TO)
        _kargs = (
            3,
            self.ib,
            insts_size,
            self.x_bo,
            self.w_bo,
            self.r_bo,
            self.y_bo,
            self.kvc,
            *(self.w_bos[1:] if self._wsplit else ()),
            *((self.e_bo,) if self.e_bo is not None else ()),
            *_dyn.dispatch_args(self.gen, L),
        )
        # RE-ISSUE A DISPATCH THE DRIVER NEVER SUBMITTED. See `_wait_dispatch`
        # for the measurement; the short version is that ~0.2% of batch-8
        # dispatches sit in ERT_CMD_STATE_NEW until a ~7 s driver timeout, never
        # reach the device, and complete on the first re-issue in ~2.1 s. Before
        # this, one such dispatch killed the whole run with
        # `decode dispatch pos<N> state=ERT_CMD_STATE_TIMEOUT`.
        #
        # The three env vars only turn parts of it OFF, for measuring it:
        #   DFLASH_DISPATCH_POLL=0    block in wait() instead of spinning, which
        #                             costs the driver's full timeout per fault
        #   DFLASH_DISPATCH_RETRY=0   do not re-issue -- fail as it used to
        #   DFLASH_DISPATCH_TIMEOUT   ms before a dispatch is called lost
        _to = int(os.environ.get("DFLASH_DISPATCH_TIMEOUT", "60000"))
        _poll = int(os.environ.get("DFLASH_DISPATCH_POLL", "1"))
        _retries = int(os.environ.get("DFLASH_DISPATCH_RETRY", "2"))
        st = self._wait_dispatch(self.kern(*_kargs), _to, _poll, f"pos{p}")
        if _retries and not str(st).endswith("COMPLETED"):
            import time as _time

            for _i in range(_retries):
                _t0 = _time.time()
                _st2 = self._wait_dispatch(self.kern(*_kargs), _to, _poll, f"pos{p}")
                print(
                    f"[retry] pos{p} attempt {_i + 1}: {st} -> {_st2} in "
                    f"{_time.time() - _t0:.3f}s",
                    flush=True,
                )
                st = _st2
                if str(st).endswith("COMPLETED"):
                    break
        _voc_n = self.UNI_LM * self.VP * B
        self.y_bo.sync(FROM, _voc_n * 2, self.decode_y * 2)
        # Zero-copy view into the BO (the shared infra's readback idiom): bo.read()
        # returns a buffer whose stride metadata is pyxrt-build dependent, and
        # .view() on it raises on some runners.
        yv = np.frombuffer(
            self.y_bo.map(), dtype=self.bf16, count=_voc_n, offset=self.decode_y * 2
        ).astype(np.float32)
        if not str(st).endswith("COMPLETED"):
            raise RuntimeError(f"decode dispatch pos{p} state={st}")
        if B == 1:
            return yv[: self.VOCAB_SIZE]
        # WAVE-major, then token-major inside a wave: token t's chunk w is at
        # w*B*VP + t*VP. UNI_LM is 1 for every shipping qwen3-4b build, but
        # slicing as if it were flat would silently interleave if it were not.
        return (
            yv.reshape(self.UNI_LM, B, self.VP)
            .transpose(1, 0, 2)
            .reshape(B, self.UNI_LM * self.VP)[:, : self.VOCAB_SIZE]
        )


def _build_prefiller(model, seq_len=None):
    """Construct the AIR prefill engine and load its resident weight BOs.

    This is the model-load cost -- the ELF cache plus the Q4NX host dequant of
    ~2.5 GB and the one-time write of every weight into its BO. Split out from the
    per-turn prefill so an interactive session pays it once."""
    import os
    import time

    from qwen3_4b_q4nx_prefill import Qwen3Q4nxPrefill

    seq_len = seq_len or int(os.environ.get("Q4NX_SEQ_LEN", "2048"))
    t_load = time.perf_counter()
    pf = Qwen3Q4nxPrefill(
        seq_len=seq_len, cache_dir=os.environ.get("Q4NX_CACHE_DIR") or None
    )
    pf.load_weights(model=model)
    print(
        f"[inference] model load (dequant + resident BOs): "
        f"{time.perf_counter() - t_load:.1f}s",
        flush=True,
    )
    return pf


def _prefill_turn(pf, prompt):
    """One batched AIR prefill dispatch -> (Kc, Vc, first_token, ttft_s).

    Kc/Vc are [NUM_LAYERS, P, DK] roped-K / raw-V, the layout FusedDecoder.seed_kv
    consumes. The prefill runs at a fixed padded seq_len (the GEMM registry carries
    the Qwen3-4B shapes at M=2048), so its cost is ~constant in prompt length.
    ttft_s times this dispatch only; the model load is reported separately."""
    import time

    t0 = time.perf_counter()
    logits = pf.prefill(prompt)
    ttft = time.perf_counter() - t0
    Kc, Vc = pf.kv_stack()
    return Kc, Vc, int(logits.argmax()), ttft


def _prefill_npu(prompt, model, seq_len=None):
    """Load the prefill engine and run one prompt through it (the one-shot path)."""
    return _prefill_turn(_build_prefiller(model, seq_len), prompt)


def _decode_loop(dec, prompt, first, n_tokens, Kc, Vc, on_token=None, stop_on_eos=True):
    """Seed the KV cache and run the fused per-token decode -> (gen_ids, seconds).

    gen_ids[0] is `first` (the prefill's own token), so n_generated is len-1.
    `on_token` is called with each newly decoded id, for streaming."""
    import time

    P = Kc.shape[1]
    dec.seed_kv(Kc, Vc, P)
    tokens = list(prompt) + [first]
    gen_ids = [first]
    n_eff = min(n_tokens, dec.ATTN_MAXL - P)
    t_dec0 = time.perf_counter()
    for p in range(P, P + n_eff):
        lg = dec.dispatch(tokens[p], p)
        pred = int(lg.argmax())
        if pred in EOS_IDS and stop_on_eos:
            print(f"[inference] pos{p} L={p+1} -> EOS ({pred}), stop", flush=True)
            break
        gen_ids.append(pred)
        if on_token:
            on_token(pred)
        if p + 1 >= len(tokens):
            tokens.append(pred)
    return gen_ids, time.perf_counter() - t_dec0


def generate(
    prompt,
    n_tokens,
    model=MODEL_DEFAULT,
    greedy=True,
    numpy_prefill=False,
    stop_on_eos=True,
):
    import numpy as np, time
    import qwen3_4b_q4nx_weights as gw

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
    print(
        f"[inference] prefill first token = {first} (Paris={PARIS_FIRST})", flush=True
    )
    # Machine-readable line for bench/extract_perf.py (nightly LLM dashboard).
    print(f"Time to first token (TTFT): {ttft:.3f}s", flush=True)

    dec = FusedDecoder(
        model=model,
        max_L=P + n_tokens,
        staircase=os.environ.get("DECODE_STAIRCASE") == "1",
    )
    if P >= dec.ATTN_MAXL:
        print(f"[inference] P={P} >= ATTN_MAXL={dec.ATTN_MAXL}; abort")
        return [first]
    gen_ids, t_dec = _decode_loop(
        dec, prompt, first, n_tokens, Kc, Vc, stop_on_eos=stop_on_eos
    )
    # Printed here, flushed, before anything else (detok, device teardown) can
    # crash and lose it -- the ids are the result that matters.
    print(f"[inference] gen ids (raw): {gen_ids}", flush=True)
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
    if not gen_ids or gen_ids[0] != PARIS_FIRST:
        return [f"*** MISS *** first token {gen_ids[:1]}, expected [{PARIS_FIRST}]"]
    if PARIS_GREEDY is None:
        return [
            "*** PARIS (first token only) ***",
            f"    record the continuation: PARIS_GREEDY = {gen_ids}",
        ]
    n = min(len(gen_ids), len(PARIS_GREEDY))
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


def _format_prompt(tokenizer, text):
    """Chat-template one turn -> a flat list of ids.

    Two template kwargs, each needed and each optional depending on the tokenizer:
    `enable_thinking=False` because Qwen3 is a hybrid reasoning model and its
    template otherwise opens a <think> block that eats a short generation budget,
    and `user_system_prompt` because the FastFlowLM bundle's template references
    that variable unguarded and raises without it. Degrades to raw encoding."""
    msg = [{"role": "user", "content": text}]
    out = None
    for kw in ({"enable_thinking": False, "user_system_prompt": ""}, {}):
        try:
            out = tokenizer.apply_chat_template(
                msg, tokenize=True, add_generation_prompt=True, **kw
            )
            break
        except Exception:
            continue
    if out is None:
        out = tokenizer(text)["input_ids"]
    # tokenize=True returns a BatchEncoding on current transformers and a plain
    # list on older ones; either may be batched.
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def repl(model=MODEL_DEFAULT, n_tokens=64):
    """Interactive chat. The prefill engine and the decode templates are built once
    and held across turns, so only the first turn pays the model load."""
    import os

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    pf = _build_prefiller(model)
    dec = FusedDecoder(model=model, staircase=os.environ.get("DECODE_STAIRCASE") == "1")
    print(
        f"\nInteractive chat (Ctrl-D or 'exit' to quit). "
        f"{n_tokens} tokens/turn, context <= {dec.ATTN_MAXL}."
    )
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line in ("exit", "quit"):
            break
        ids = _format_prompt(tokenizer, line)
        if len(ids) >= dec.ATTN_MAXL:
            print(f"[prompt is {len(ids)} tokens, limit {dec.ATTN_MAXL}]")
            continue
        # prefill() APPENDS to the prefiller's KV (the causal_lm contract the verify
        # runner needs), so without this turn 2 reports a context of turn1+turn2 for
        # a turn-2-length prompt. Turns are independent: the decode's cache is seeded
        # from this turn's prefill alone, so carrying history would mean re-prefilling
        # the whole conversation as one prompt. Same single-turn shape as the llama
        # examples' REPL.
        pf.clear_context()
        Kc, Vc, first, ttft = _prefill_turn(pf, ids)
        out = [first]
        print(tokenizer.decode(out), end="", flush=True)

        def _emit(tok, _out=out):
            prev = tokenizer.decode(_out)
            _out.append(tok)
            print(tokenizer.decode(_out)[len(prev) :], end="", flush=True)

        gen_ids, t_dec = _decode_loop(dec, ids, first, n_tokens, Kc, Vc, on_token=_emit)
        n_gen = len(gen_ids) - 1
        rate = f"{n_gen / t_dec:.2f} tok/s" if n_gen and t_dec else "-"
        print(f"\n[{len(ids)} prompt tok, TTFT {ttft:.2f}s | {n_gen} gen, {rate}]")


def main():
    ap = argparse.ArgumentParser(description="qwen3_4b_q4nx full inference (decode)")
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
    ap.add_argument(
        "--no-eos-stop",
        action="store_true",
        help="keep generating past EOS, so the decode rate is measured over the "
        "full --n-tokens (what `make profile` wants)",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        help="chat REPL: build the engine once and hold it across turns",
    )
    args = ap.parse_args()

    if args.interactive:
        if args.numpy_prefill:
            ap.error("--interactive cannot be combined with --numpy-prefill")
        repl(model=args.model, n_tokens=args.n_tokens)
        return

    if args.prompt_ids:
        prompt = [int(x) for x in args.prompt_ids.split(",")]
    elif args.prompt:
        from transformers import AutoTokenizer

        # Chat-templated, same as the REPL: raw-encoding an instruction gives a
        # continuation rather than an answer. --prompt-ids stays literal.
        prompt = _format_prompt(AutoTokenizer.from_pretrained(args.model), args.prompt)
    else:
        prompt = PARIS_PROMPT
    print(f"[inference] prompt = {len(prompt)} tokens: {prompt}", flush=True)

    gen_ids = generate(
        prompt,
        args.n_tokens,
        model=args.model,
        numpy_prefill=args.numpy_prefill,
        stop_on_eos=not args.no_eos_stop,
    )
    print("=" * 60)
    print(f"[inference] gen ids: {gen_ids}")
    print(f"[inference] TEXT: {_detok(gen_ids, args.model)!r}")
    if prompt == PARIS_PROMPT:
        for line in _paris_verdict(gen_ids):
            print(line)


if __name__ == "__main__":
    main()
