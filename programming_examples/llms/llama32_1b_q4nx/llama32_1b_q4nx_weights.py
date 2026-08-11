# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Q4NX weight loader + dequant for the Llama-3.2-1B Q4NX example.
#
# Reads the per-layer Q4NX weight bins (L{k}_proj_w.bin / L{k}_rms_w.bin) and
# reconstructs the full-precision bf16 weight matrices from the Q4NX blocks
# (reorder with vertical_blocks=2; tensor order q|k|v|o | up/gate interleaved in
# 512-row phases | down). Q4NX = per-block 4-bit affine quant: w = q*scale + min,
# 32x256 blocks, bf16 scale/min per 32-column group.
#
# Layout facts:
#   proj_w = concat[ q(2048x2048) | k(512x2048) | v(512x2048) | o(2048x2048)
#                    | up/gate interleaved per 512-row phase (16384x2048)
#                    | down(2048x8192) ], each Q4NX-reordered (pairs of row-
#                    blocks interleaved column-by-column).
#   rms_w  = [ input_layernorm(2048) | post_attention_layernorm(2048) | 0-pad ]
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

# The Q4NX block packer/dequant reference lives in the standalone fused_decode
# example (this dir already depends on it for the decode path).
_FUSED_DECODE = str(Path(__file__).resolve().parents[2] / "fused_decode")
if _FUSED_DECODE not in sys.path:
    sys.path.insert(0, _FUSED_DECODE)
from proj_qmm_pack import (  # noqa: E402
    ROW_BLOCK,
    COL_BLOCK,
    GROUP,
    N_GROUPS,
    ROW_GROUPS,
    PARALLEL,
    BLOCK_BF16,
)

# Llama-3.2-1B dims.
D = 2048  # model dim
DQ = 2048  # q proj out (32 heads * 64)
DK = 512  # k proj out (8 kv heads * 64)
DV = 512
DH = 64  # head dim
N_Q_HEADS = 32
N_KV_HEADS = 8
Q_PER_KV = N_Q_HEADS // N_KV_HEADS  # 4
INTER = 8192  # mlp intermediate


def _bf(a):
    return a.astype(bfloat16).astype(np.float32)


# ---------------------------------------------------------------------------
# model.q4nx loader (HF-hosted, self-contained)
#
# The Q4NX model ships as a single safetensors file `model.q4nx` on the Hub
# (repo FastFlowLM/Llama-3.2-1B-NPU2) with standard HF tensor names: per-layer
# {q,k,v,o,gate,up,down}_proj (Q4NX), input/post_attention_layernorm + norm +
# embed_tokens (bf16), and lm_head. The quantized tensors are per-block 4-bit
# affine, 32x256 blocks with a bf16 scale/offset per 32-col group, in plain
# (row-block, col-block) order, under one of two headers/dequants -- see
# Q4nxModel. This loader mmaps the file, parses the safetensors header, and
# vectorized-dequantizes each tensor.
_G = ROW_BLOCK // PARALLEL  # row groups per block (2)
_EVEN = np.array(
    [g * PARALLEL + 2 * k for g in range(_G) for k in range(PARALLEL // 2)]
)
_ODD = _EVEN + 1


# FastFlowLM re-exported every NPU2 bundle to the I8 packed encoding on
# 2026-08-04 (all four repos within 20 minutes of each other). Those revisions
# are the ones to use: decoded with the additive dequant below they reproduce
# the reference bf16 weights to cosine ~0.997 with matching standard deviation,
# whereas the older Q4NX-encoded revisions reconstruct to only 0.87 - 0.99 with
# a standard deviation off by up to 13x. So nothing is pinned: take whatever the
# repo currently serves, and let the codec be detected from the header.
#
# Old revisions remain reachable, so pinning one is still possible for a
# bisect -- Llama-3.2-1B-NPU2 was d0c7f84a and Llama-3.2-3B-NPU2 was 790271a8
# before the re-export -- but they are the worse weights, not a safe harbour.
_PINNED_Q4NX_REVISION = {}


def resolve_q4nx_model(model):
    """Resolve `model` to a local model.q4nx path. `model` may be an HF repo id
    (contains '/'), a directory containing model.q4nx, or a direct file path.

    Repo ids download their current revision; `_PINNED_Q4NX_REVISION` can pin
    one for a bisect (empty by default, see above)."""
    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        p = os.path.join(model, "model.q4nx")
        if os.path.isfile(p):
            return p
    # treat as an HF repo id
    from huggingface_hub import hf_hub_download

    rev = _PINNED_Q4NX_REVISION.get(model)
    try:
        return hf_hub_download(model, "model.q4nx", revision=rev)
    except Exception:
        # Offline with a cache that predates the current revision: an offline
        # runner pinned to a snapshot keeps working rather than failing to
        # resolve. Both encodings decode correctly (the older one just
        # reconstructs less faithfully), so an older snapshot is usable.
        if rev is not None:
            raise
        cached = _any_cached_snapshot(model)
        if cached is None:
            raise
        return cached


def _any_cached_snapshot(model):
    """Path to the most recently fetched cached model.q4nx for `model`."""
    import glob

    from huggingface_hub.constants import HF_HUB_CACHE

    hits = glob.glob(
        os.path.join(
            HF_HUB_CACHE,
            "models--" + model.replace("/", "--"),
            "snapshots",
            "*",
            "model.q4nx",
        )
    )
    return max(hits, key=os.path.getmtime) if hits else None


class Q4nxModel:
    """mmap + parse a model.q4nx safetensors file; vectorized Q4NX dequant.

    HEADER CONVENTIONS. FastFlowLM ships quantized projections under two
    different safetensors headers, and they are not distinguishable by shape
    alone -- only by `dtype`:

      dtype "Q4NX": shape is the LOGICAL [M, K] (e.g. [3072, 3072]). The block
                    count is derived as (M/ROW_BLOCK)*(K/COL_BLOCK).
      dtype "I8":   shape is the PACKED [n_blocks, bytes_per_block] (e.g.
                    [1152, 5120]). The logical [M, K] is NOT recoverable from
                    the header -- n_blocks is a single product of the two --
                    so the caller must supply it.

    The two are not interchangeable encodings of the same bytes: the dequant
    differs as well (see `dequant`), and the I8 re-export is the more faithful
    of the two. Every FastFlowLM NPU2 repo currently serves I8; the Q4NX form
    survives only in pre-2026-08-04 snapshots. Reading an I8 header as logical
    yields a wrong block count and a reshape error, so `dequant` dispatches on
    `dtype` and callers that may meet an I8 bundle must pass `M`/`K`.
    """

    def __init__(self, model):
        import json

        path = resolve_q4nx_model(model)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        hlen = int(np.frombuffer(self._mm[:8].tobytes(), dtype="<u8")[0])
        self._hdr_bytes = self._mm[8 : 8 + hlen].tobytes()
        self._hdr = json.loads(self._hdr_bytes)
        self._base = 8 + hlen

    def fingerprint(self):
        """Short digest of the safetensors header. It changes whenever the Hub
        re-exports the bundle -- encoding, block layout, or tensor set -- so a
        derived cache keyed on it cannot silently outlive the weights it was
        built from (the 2026-08-04 Q4NX -> I8 re-export is exactly that case).
        Hashes the header only, never the multi-GB payload."""
        import hashlib

        return hashlib.sha256(self._hdr_bytes).hexdigest()[:8]

    def _raw_i16(self, name):
        o0, o1 = self._hdr[name]["data_offsets"]
        return np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=np.int16
        )

    def has(self, name):
        return name in self._hdr

    def bf16(self, name):
        """A BF16 tensor as float32, in its declared shape."""
        o0, o1 = self._hdr[name]["data_offsets"]
        v = np.frombuffer(
            self._mm[self._base + o0 : self._base + o1].tobytes(), dtype=bfloat16
        )
        return v.astype(np.float32).reshape(self._hdr[name]["shape"])

    def dims(self, name):
        """Logical (M, K) of a quantized tensor, or None for an I8 header."""
        e = self._hdr[name]
        return tuple(e["shape"]) if e.get("dtype") == "Q4NX" else None

    def dequant(self, name, M=None, K=None):
        """A quantized tensor dequantized to float32 [M, K].

        `M`/`K` are required for an I8-header bundle and cross-checked against
        the header for a Q4NX one (see the class docstring), which also selects
        the dequant: Q4NX subtracts, I8 adds.
        """
        hdr_dims = self.dims(name)
        if hdr_dims is None:
            if M is None or K is None:
                raise ValueError(
                    f"{name}: I8-packed header {self._hdr[name]['shape']} does not "
                    "carry the logical shape; pass M/K from the model config"
                )
        else:
            if M is not None and (M, K) != hdr_dims:
                raise ValueError(f"{name}: caller says {(M, K)}, header {hdr_dims}")
            M, K = hdr_dims
        nbi, nbj = M // ROW_BLOCK, K // COL_BLOCK
        nb = nbi * nbj
        if hdr_dims is None and self._hdr[name]["shape"][0] != nb:
            # The only cross-check available for an I8 header: caller dims that
            # disagree with the block count would otherwise reshape-fail deep
            # inside the unpack, or silently succeed on a transposed pair.
            raise ValueError(
                f"{name}: header holds {self._hdr[name]['shape'][0]} blocks but "
                f"(M/{ROW_BLOCK})*(K/{COL_BLOCK}) = {nb} for [M, K] = {[M, K]}"
            )
        i16 = self._raw_i16(name).reshape(nb, BLOCK_BF16)
        sc = (
            i16[:, 0:256]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nb, N_GROUPS, ROW_BLOCK)
        )
        mn = (
            i16[:, 256:512]
            .copy()
            .view(bfloat16)
            .astype(np.float32)
            .reshape(nb, N_GROUPS, ROW_BLOCK)
        )
        qb = (
            i16[:, 512:BLOCK_BF16]
            .copy()
            .view(np.uint8)
            .reshape(nb, _G, COL_BLOCK, PARALLEL // 2)
        )
        lo = (qb & 0xF).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        hi = (qb >> 4).transpose(0, 1, 3, 2).reshape(nb, ROW_BLOCK // 2, COL_BLOCK)
        q = np.zeros((nb, ROW_BLOCK, COL_BLOCK), np.float32)
        q[:, _EVEN, :] = lo
        q[:, _ODD, :] = hi
        scale = np.repeat(sc.transpose(0, 2, 1), GROUP, axis=2)
        offset = np.repeat(mn.transpose(0, 2, 1), GROUP, axis=2)
        # The two codecs also differ in the dequant itself, not just the header:
        # Q4NX subtracts an unscaled zero point, I8 (GGUF Q4_1) adds an already
        # scaled offset (offset/scale ~ -7.4), matching the AIR kernel q4_k.h
        # (c += (q*b)*scale; c += min*sum(b)). Using the wrong one yields weights
        # of plausible magnitude that decode to garbage, so it must follow dtype.
        w = scale * (q - offset) if hdr_dims is not None else scale * q + offset
        return (
            w.reshape(nbi, nbj, ROW_BLOCK, COL_BLOCK)
            .transpose(0, 2, 1, 3)
            .reshape(M, K)
        )

    _PROJ = {
        "q": "self_attn.q_proj",
        "k": "self_attn.k_proj",
        "v": "self_attn.v_proj",
        "o": "self_attn.o_proj",
        "up": "mlp.up_proj",
        "gate": "mlp.gate_proj",
        "down": "mlp.down_proj",
    }

    # Logical (out, K) per projection, needed for an I8-packed header whose
    # shape is [n_blocks, bytes_per_block]. These are the Llama-3.2-1B dims; a
    # caller with different ones passes its own to layer_weights().
    _PROJ_DIMS = {
        "q": (DQ, D),
        "k": (DK, D),
        "v": (DV, D),
        "o": (D, DQ),
        "up": (INTER, D),
        "gate": (INTER, D),
        "down": (D, INTER),
    }

    def layer_weights(self, k, dims=None):
        """Dequantized bf16 GEMM input-B [K, out] per projection for layer k.

        `dims` maps a projection key to its logical (out, K), and is required
        for an I8-header bundle whose dims differ from Llama-3.2-1B's (see the
        class docstring); it defaults to _PROJ_DIMS.
        """
        dims = dims or self._PROJ_DIMS
        out = {}
        for nm, t in self._PROJ.items():
            M, K = dims.get(nm, (None, None))
            wt = self.dequant(f"model.layers.{k}.{t}.weight", M, K)  # [out, K]
            out[nm] = np.ascontiguousarray(wt.T, dtype=bfloat16)  # [K, out]
        return out

    def layer_rms(self, k):
        """(input_layernorm, post_attention_layernorm) float32 for layer k."""
        return (
            self.bf16(f"model.layers.{k}.input_layernorm.weight"),
            self.bf16(f"model.layers.{k}.post_attention_layernorm.weight"),
        )

    def embed_norm_lmhead(self):
        """(embed_tokens [VOCAB,D], final_norm [D], lm_head [VOCAB,D]) float32.

        Llama-3.2-1B ties the LM head to the embedding (config
        tie_word_embeddings=true), so the LM head IS the full-precision embed
        matrix. The bundle also carries a separate Q4NX-quantized lm_head tensor;
        the tied fp embed is used instead (lossless, matches HF)."""
        embed = self.bf16("model.embed_tokens.weight")
        norm = self.bf16("model.norm.weight")
        return embed, norm, embed
