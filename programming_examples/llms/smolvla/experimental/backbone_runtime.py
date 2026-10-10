# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""SmolLM2 backbone prefix fill on NPU2, for the SmolVLA pipeline (experimental).

`BackboneRuntime.fill` stands in for lerobot's `vlm_with_expert.forward` fill
call (past_key_values=None, no expert input). lerobot's `sample_actions` keeps
only the KV cache from that call, and the action expert reads each layer's
post-RoPE K and V from it. So the fill runs the 16 whole-layer ELFs
(backbone_npu.py --fused-layer --offn-engine, the best configuration) and
returns a DynamicCache of the NPU's K and V; the final hidden state and norm are
never computed.

The attention mask and RoPE tables are static per-layer BOs. They depend only on
the prefix structure (camera count, prompt padding), so they are rewritten in
place when that changes, not per call.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import backbone_npu as bn

CACHE_DIR = (
    Path(__file__).resolve().parent
    / "build"
    / "backbone_npu_cache_fgu128bstsw_fqkv80bst_rgrt2x5_offnt2x6_bqkv80x480x160"
    "_bo80x480x160_bgu128x480x160x8_bdn80x256x128_faOs_his5_qb_offneng"
)
GU_TILE_N = 128
_MASK_ARG, _ROPE_Q_ARG, _ROPE_K_ARG = 9, 5, 7


class BackboneRuntime:
    def __init__(self, policy, profile=False):
        from air_examples.llms.shared.infra.cache import KernelCache, Profiler

        bn._BFP16.update({k: bn._BFP16_TILES[k] for k in ("qkv", "o", "gu", "dn")})
        bn._TILING.update(rgr=[2, 5], offn=[2, 6])
        bn._ENGINE["on"] = True
        self.cfg = bn.BACKBONE_CONFIG
        self.cache = KernelCache(
            str(CACHE_DIR), verbose=False, profiler=Profiler(enabled=profile)
        )
        if not (self.cache.load_manifest() and "layer" in self.cache.artifacts):
            bn.compile_backbone_kernels(
                self.cache,
                self.cfg,
                bn.SEQ_PAD,
                fused_gu=True,
                gu_tile_n=GU_TILE_N,
                fused_qkv=True,
                qkv_tile_n=80,
                gu_bstationary=True,
                qkv_bstationary=True,
                gu_swiglu=True,
                npu_attn=True,
                fused_layer=True,
                fa_opt="-Os",
                fa_his=5,
                fa_qb=True,
                offn_engine=True,
            )
            self.cache._save_manifest()
        self.weights = bn.extract_backbone_weights(policy)
        tm = policy.model.vlm_with_expert.get_vlm_model().text_model
        self.kv_dtype = tm.layers[0].self_attn.k_proj.weight.dtype
        self._sig = None
        self._rope = None

    def _set_prefix(self, mask_bool, position_ids):
        """Mask + RoPE LUT for this prefix; rewrites the static BOs if they changed."""
        sig = (mask_bool.shape[0], mask_bool.tobytes(), position_ids.tobytes())
        if sig == self._sig:
            return
        seq, pad = mask_bool.shape[0], bn.SEQ_PAD
        assert seq <= pad, f"prefix {seq} > {pad} rows the layer ELF was built for"
        mask = bn.additive_attn_mask(bn.pad_mask(mask_bool, pad))
        lut = bn.pad_seq(bn.build_rope_lut_gathered(position_ids, self.cfg), pad)
        lut[seq:] = lut[seq - 1]
        bn._NPU_ATTN["mask"] = mask
        self._rope = lut
        new = {
            _MASK_ARG: mask,
            _ROPE_Q_ARG: np.repeat(lut, self.cfg.n_heads, axis=0).flatten(),
            _ROPE_K_ARG: np.repeat(lut, self.cfg.n_kv_heads, axis=0).flatten(),
        }
        self._rewrite_static(new)
        self._sig = sig

    def _rewrite_static(self, new):
        import pyxrt as xrt

        arg_cache = getattr(self.cache, "_arg_cache_layer_fused", {})
        for i in range(self.cfg.n_layers):
            key = f"layer_L{i}"
            if key in arg_cache:
                for idx, a in new.items():
                    arg_cache[key][idx] = a
            bos = self.cache._cached_bos.get(key)
            if bos is None:
                continue
            for idx, a in new.items():
                src = np.frombuffer(
                    a.view(np.int16) if a.dtype == bfloat16 else a, dtype=np.uint8
                )
                dst = np.frombuffer(bos[idx].map(), dtype=np.uint8, count=len(src))
                np.copyto(dst, src, casting="no")
                bos[idx].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def fill(self, prefix_embs, attention_mask, position_ids):
        """prefix_embs (1, L, 960), attention_mask (1, L, L) bool, position_ids (1, L)
        -> DynamicCache with every layer's post-RoPE K and V, (1, 5, L, 64)."""
        import torch
        from transformers.cache_utils import DynamicCache

        assert prefix_embs.shape[0] == 1, "batch 1 only"
        seq = prefix_embs.shape[1]
        cfg = self.cfg
        self._set_prefix(
            attention_mask[0, :seq, :seq].numpy().astype(bool),
            position_ids[0, :seq].numpy(),
        )
        x = bn.pad_seq(
            prefix_embs[0].to(torch.float32).numpy().astype(bfloat16), bn.SEQ_PAD
        )
        emb, kv, hd, nkv = (
            cfg.emb_dim,
            cfg.n_kv_heads * cfg.head_dim,
            cfg.head_dim,
            cfg.n_kv_heads,
        )
        cache = DynamicCache()

        def heads(a):  # (seq, kv) bf16 -> (1, nkv, seq, hd) torch
            t = torch.from_numpy(np.array(a[:seq]).view(np.int16)).view(torch.bfloat16)
            return t.reshape(1, seq, nkv, hd).transpose(1, 2).to(self.kv_dtype)

        for i in range(cfg.n_layers):
            x, qkv, k = bn.run_layer_fused(
                x,
                self.weights.layers[i],
                self._rope,
                cfg,
                self.cache,
                layer_idx=i,
                gu_swiglu_half=GU_TILE_N // 2,
                with_kv=True,
            )
            cache.update(heads(k), heads(qkv[:, emb + kv :]), i)
        return cache
