# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Weight reader for the DFlash DRAFTER (z-lab/Qwen3-4B-DFlash-b16), presenting
# the same surface qwen3_4b_q4nx_requant.py already consumes for the target.
#
# The drafter is NOT distributed as a model.q4nx bundle -- it ships one bf16
# `model.safetensors` -- so there is nothing to dequantize, only to read. Its
# per-layer shapes are the target's exactly (verified against the header):
#
#     q 4096x2560   k,v 1024x2560   o 2560x4096
#     gate,up 9728x2560   down 2560x9728
#
# which is why fused_decode.py's `qwen3-4b-draft` entry is the target's geometry
# with UNI_DEC=5 and nothing else changed.
#
# TWO THINGS THE TARGET'S READER HAS NO ANALOGUE FOR:
#
#   fc.weight  [2560, 12800]   fuses the 5 tapped target hidden states. A phase
#                              of its own (I2P=5, J2P=25 at this geometry) --
#                              see docs/DFlashFeasibility.md section 3.3.
#   hidden_norm.weight [2560]  the RMSNorm applied to fc's output.
#
# AND ONE THING IT DOES NOT CARRY AT ALL: an embedding table. The drafter's head
# is tied to the TARGET's, so `bf16("model.embed_tokens.weight")` delegates to
# the target bundle. A drafter packed against its own (nonexistent) embedding
# would fail at load; packed against a DIFFERENT model's would decode garbage,
# so the delegation is explicit rather than a fallback.
#
# Key naming: the checkpoint uses `layers.N.…` where the target bundle uses
# `model.layers.N.…`. `dequant`/`bf16` accept either so the shared requant code
# needs no special-casing.

import glob
import json
import os
import struct

import numpy as np
from ml_dtypes import bfloat16

DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"
D = 2560
NUM_LAYERS = 5
NTAP = 5
FC_IN = NTAP * D  # 12800


def _default_snapshot():
    pat = os.path.expanduser(
        "~/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/"
        "snapshots/*/model.safetensors"
    )
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(
            f"no local snapshot of {DRAFT_ID}. Fetch it first:\n"
            f"    huggingface-cli download {DRAFT_ID}"
        )
    return hits[-1]


class DraftWeights:
    """The drafter's bf16 safetensors, read the way the requant packer expects.

    Mirrors the accessor surface of qwen3_4b_q4nx_weights.Q4nxModel -- `_PROJ`,
    `dequant`, `layer_rms`, `layer_qk_norm`, `bf16` -- so the packing code is
    shared rather than forked. `dequant` is a plain read here: the source is
    already full precision, and the 4-bit re-quantization happens downstream in
    _requant_q4k, identically to the target path.
    """

    _PROJ = {
        "q": ("self_attn.q_proj", 4096, D),
        "k": ("self_attn.k_proj", 1024, D),
        "v": ("self_attn.v_proj", 1024, D),
        "o": ("self_attn.o_proj", D, 4096),
        "up": ("mlp.up_proj", 9728, D),
        "gate": ("mlp.gate_proj", 9728, D),
        "down": ("mlp.down_proj", D, 9728),
    }

    def __init__(self, path=None, target_source=None):
        self.path = path or _default_snapshot()
        self.target_source = target_source
        with open(self.path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self._hdr = json.loads(f.read(n))
            self._base = 8 + n
        self._hdr.pop("__metadata__", None)
        self._target = None

    # -- raw access ---------------------------------------------------------
    def _entry(self, name):
        if name in self._hdr:
            return name
        # accept both `model.layers.N.…` and `layers.N.…`
        alt = name[len("model.") :] if name.startswith("model.") else "model." + name
        if alt in self._hdr:
            return alt
        raise KeyError(f"{name!r} not in {os.path.basename(self.path)}")

    def bf16(self, name):
        """[shape] bfloat16 view. Delegates the tied head to the TARGET."""
        if "embed_tokens" in name:
            return self._target_reader().bf16(name)
        e = self._hdr[self._entry(name)]
        if e["dtype"] != "BF16":
            raise TypeError(f"{name} is {e['dtype']}, expected BF16")
        lo, hi = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._base + lo)
            buf = f.read(hi - lo)
        return np.frombuffer(buf, dtype=bfloat16).reshape(e["shape"])

    def _target_reader(self):
        if self._target is None:
            if self.target_source is None:
                raise RuntimeError(
                    "the drafter's LM head is TIED to the target's embedding "
                    "table and this checkpoint carries none. Construct "
                    "DraftWeights(target_source=<the target's model.q4nx>)."
                )
            from qwen3_4b_q4nx_weights import Q4nxModel

            self._target = Q4nxModel(self.target_source)
        return self._target

    # -- the surface the requant packer uses --------------------------------
    def dequant(self, name, M, K):
        w = np.asarray(self.bf16(name), dtype=np.float32)
        if w.shape != (M, K):
            raise ValueError(f"{name}: got {w.shape}, expected {(M, K)}")
        return w

    def layer_rms(self, k):
        return (
            np.asarray(self.bf16(f"layers.{k}.input_layernorm.weight"), np.float32),
            np.asarray(
                self.bf16(f"layers.{k}.post_attention_layernorm.weight"), np.float32
            ),
        )

    def layer_qk_norm(self, k):
        return (
            np.asarray(self.bf16(f"layers.{k}.self_attn.q_norm.weight"), np.float32),
            np.asarray(self.bf16(f"layers.{k}.self_attn.k_norm.weight"), np.float32),
        )

    # -- drafter-only tensors ------------------------------------------------
    def fc(self):
        """[2560, 12800]: the tap-fusion projection, as one matrix.

        Fed to the engine as a single J=25 phase rather than five accumulated
        2560->2560 ones -- accumulating across input column-blocks is what the
        projection cores already do. dflash_draft_decomp.py checks the two are
        the same computation.
        """
        return np.asarray(self.fc_bf16(), np.float32)

    def fc_bf16(self):
        w = self.bf16("fc.weight")
        if w.shape != (D, FC_IN):
            raise ValueError(f"fc.weight: got {w.shape}, expected {(D, FC_IN)}")
        return w

    def hidden_norm(self):
        return np.asarray(self.bf16("hidden_norm.weight"), np.float32)

    def final_norm(self):
        return np.asarray(self.bf16("norm.weight"), np.float32)

    def embed_norm_lmhead(self):
        """(embed_in [VOCAB,D] bf16, final_norm [D] f32, lm_head [VOCAB,D]).

        The surface `FusedDecoder` gathers x0 and the final norm from, so a
        drafter decode can be driven by the same class as the target's.

        TIED TO ANOTHER MODEL. `embed_in` is the TARGET's table -- the drafter
        checkpoint carries none, which is also why `target_source` is a required
        argument rather than a default. The final norm is the drafter's OWN
        (`norm.weight`, not the target's `model.norm.weight`); taking the
        target's there decodes fluently and wrongly.
        """
        embed_in = self.bf16("model.embed_tokens.weight")
        return embed_in, self.final_norm(), embed_in


def self_check(verbose=True):
    """Every tensor the packer will ask for is present and the right shape."""
    w = DraftWeights()
    bad = 0
    for k in range(NUM_LAYERS):
        for nm, (suffix, M, K) in DraftWeights._PROJ.items():
            try:
                w.dequant(f"layers.{k}.{suffix}.weight", M, K)
            except Exception as e:
                bad += 1
                print(f"  layer {k} {nm}: {e}")
        w.layer_rms(k)
        w.layer_qk_norm(k)
    fc, hn, fn = w.fc(), w.hidden_norm(), w.final_norm()
    if verbose:
        print(f"[draft weights] {os.path.basename(os.path.dirname(w.path))}")
        print(
            f"  {NUM_LAYERS} layers x {len(DraftWeights._PROJ)} projections: "
            f"{'OK' if not bad else str(bad) + ' MISSING'}"
        )
        print(f"  fc {fc.shape}, hidden_norm {hn.shape}, norm {fn.shape}")
        print(
            f"  no embed_tokens (head tied to the target) -- "
            f"{'embed_tokens' not in ' '.join(w._hdr)}"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    import sys

    sys.exit(self_check())
