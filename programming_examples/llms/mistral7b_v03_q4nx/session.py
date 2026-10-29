# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite production session.
"""Production NPU session. No CPU transformer or replay path.

Reuses the AIR Llama BF16 prefill blocks and custom Q4NX spatial decode.
All generated artifacts and prepared arrays are resolved relative to candidate_root.
"""

import json
import os
from pathlib import Path
import time

import numpy as np
from ml_dtypes import bfloat16

from ..shared.deployment import require_bfp16
from .common import Config, RUN, REVISION, save_json, file_hash
from .build import decode_module
from ..shared.deployment_runtime import read_bf16, load_cache


def rope_lut(capacity=1024):
    inv = 1.0 / (np.float32(1e6) ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(bfloat16)


class FusedDecode:
    def __init__(self, artifacts, weights, lut, capacity=1024):
        import pyxrt as xrt
        from air_examples.fused_decode.decode_insts_gen import DecodeInstsGen
        from air_examples.fused_decode.decode_staircase import (
            open_windows,
            make_insts_states,
        )

        self.xrt = xrt
        self.fd = fd = decode_module()
        assert fd.UNI_DEC == 32 and fd.W_SPLIT and fd.N_WGRP == 4
        self.capacity = capacity
        self.gen = DecodeInstsGen(str(artifacts), max_L=capacity)
        if self.gen.attn_maxl != capacity:
            raise ValueError("decode template capacity does not match the KV layout")
        self.dev = xrt.device(0)
        self.contexts = open_windows(self.dev, xrt, self.gen, [capacity])
        self.kernel = self.contexts[capacity][1]
        self.insts = make_insts_states(
            self.gen, xrt, self.dev, self.kernel.group_id(1), [capacity]
        )[capacity]
        self.w = np.load(weights / "decode_weights.npy", mmap_mode="r")
        self.norms = read_bf16(weights / "norms.npy").reshape(-1)
        self.final_norm = read_bf16(weights / "final_norm.npy")
        self.embed = read_bf16(weights / "embedding.npy")
        self.lut = lut
        self.to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + fd.UNI_LM * fd.VOCAB_SIZE_PADDED
        lengths = [
            4096,
            8 * fd.W_LAYER,
            32 * fd.RMS_LAYER + 128 + 4096,
            self.ny,
            32 * capacity * fd.KVSZ_TOK,
        ]
        lengths += [8 * fd.W_LAYER] * 3 + [fd.UNI_LM * fd.VOCAB_W_BLOCKS * 2560]
        self.bos = [
            xrt.bo(self.dev, int(n) * 2, xrt.bo.host_only, self.kernel.group_id(3 + i))
            for i, n in enumerate(lengths)
        ]
        # Additional groups and head follow the original five-argument ABI.
        for group, index in enumerate((1, 5, 6, 7, 8)):
            lo = group * 8 * fd.W_LAYER
            hi = (group + 1) * 8 * fd.W_LAYER if group < 4 else self.w.size
            slab = self.w[lo:hi]
            assert slab.nbytes < 2**32 and slab.nbytes == lengths[index] * 2
            self.bos[index].write(slab.view(np.int16), 0)
            self.bos[index].sync(self.to)
        rms = np.concatenate([self.norms, np.zeros(128, bfloat16), self.final_norm])
        self.bos[2].write(rms.view(np.int16), 0)
        self.bos[2].sync(self.to)
        self.submissions = 0

    def seed_kv(self, keys, values, length):
        fd = self.fd
        rw, ng = fd.REGION_W, fd.NGRP
        stride = self.capacity * rw
        kv = np.zeros((32, self.capacity * fd.KVSZ_TOK), bfloat16)
        for layer in range(32):
            for group in range(ng):
                kv[layer, group * stride : group * stride + length * rw] = keys[
                    layer, :length, group * rw : (group + 1) * rw
                ].ravel()
                kv[
                    layer, (ng + group) * stride : (ng + group) * stride + length * rw
                ] = values[layer, :length, group * rw : (group + 1) * rw].ravel()
        self.bos[4].write(kv.view(np.int16).ravel(), 0)
        self.bos[4].sync(self.to)

    def step(self, token, pos):
        from air_examples.fused_decode.decode_staircase import patch_insts

        size = patch_insts(self.insts, pos + 1, self.xrt, self.to)
        x = np.ascontiguousarray(self.embed[token])
        self.bos[0].write(x.view(np.int16), 0)
        self.bos[0].sync(self.to)
        layer_lut = np.ascontiguousarray(self.lut[pos])
        self.bos[2].write(layer_lut.view(np.int16).ravel(), self.norms.size * 2)
        self.bos[2].sync(self.to, layer_lut.nbytes, self.norms.size * 2)
        state = self.kernel(3, self.insts["ib"], size, *self.bos).wait(60000)
        self.submissions += 1
        if not str(state).endswith("COMPLETED"):
            raise RuntimeError(f"fused decode position {pos}: {state}")
        n = self.fd.VOCAB_SIZE_PADDED_FULL
        self.bos[3].sync(self.fr, n * 2, self.decode_y * 2)
        z = np.frombuffer(
            self.bos[3].map(), dtype=bfloat16, count=n, offset=self.decode_y * 2
        ).astype(np.float32)
        return z[:32768]

    def close(self):
        self.bos = []
        self.insts = None
        self.kernel = None
        self.contexts = None
        self.dev = None


class MistralSession:
    def __init__(self, task, runtime):
        start = time.perf_counter()
        self.root = Path(runtime["candidate_root"]).resolve()
        self.work = Path(runtime["work_dir"]).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        selected = runtime["configuration"]
        expected = next(p for p in task["prompts"] if p["id"] == selected["id"])
        for key in ("tokens", "prefill_build_length", "capacity", "ids_sha256"):
            if selected[key] != expected[key]:
                raise ValueError(f"configuration {key} differs from fixed task")
        self.record_state = bool(runtime.get("record_state", False))
        build = self.root / runtime.get("build_dir", f"{RUN}/build-{selected['id']}")
        head_build = self.root / runtime.get("head_dir", f"{RUN}/build-shared/head")
        weights_dir = self.root / runtime.get("weights_dir", f"{RUN}/weights")
        manifest = json.loads((weights_dir / "manifest.json").read_text())
        if manifest["logical_manifest_sha256"] != task["assets"]["q4_manifest_sha256"]:
            raise ValueError("packed weights do not bind the fixed logical weights")
        if task["model"]["revision"] != REVISION:
            raise ValueError("unsupported checkpoint revision")
        os.environ["AIR_TARGET_DEVICE"] = "npu2"
        from .recipes import install

        install()
        self.config = Config()
        self.prefill_length = int(selected["prefill_build_length"])
        self.capacity = int(selected["capacity"])
        if (self.prefill_length, self.capacity) not in (
            (512, 1024),
            (1024, 2048),
            (2048, 4096),
            (4096, 8192),
        ):
            raise ValueError("unsupported prefill/capacity configuration")
        compiled_prefill = json.loads((build / "prefill/build.json").read_text())
        require_bfp16(compiled_prefill)
        self.config.gemm_bfp16 = bool(compiled_prefill.get("gemm_bfp16", False))
        if compiled_prefill["sequence_length"] != self.prefill_length:
            raise ValueError("prefill artifact shape does not match the session")
        self.prefill_rms_impl = compiled_prefill.get("rms_implementation", "seed")
        self.resident_prefill = bool(runtime.get("resident_prefill", True))
        self.cache = load_cache(build / "prefill/cache", resident=self.resident_prefill)
        self.head_cache = load_cache(head_build / "cache")
        self.artifact_identity = {
            f"{mode}/{p.name}": {"path": str(p.resolve()), "sha256": file_hash(p)}
            for mode, paths in (
                ("prefill", (build / "prefill/cache").glob("*.elf")),
                ("head", (head_build / "cache").glob("*.elf")),
                ("decode", (build / "decode").glob("decode_L*.bin")),
                ("decode", (build / "decode").glob("decode_L*.xclbin")),
            )
            for p in paths
            if "-before-" not in p.name
        }
        self.packing_manifest_sha256 = file_hash(weights_dir / "manifest.json")
        self.lut = rope_lut(self.capacity)
        self.decode = FusedDecode(
            build / "decode", weights_dir, self.lut, self.capacity
        )
        self.embed = self.decode.embed
        self.final_norm = self.decode.final_norm
        from llama32_1b_weights import LayerWeights, LlamaWeights
        from llama32_1b_prefill import preload_prefill_weights

        norms = read_bf16(weights_dir / "norms.npy")
        fields = {
            "q": "wq",
            "k": "wk",
            "v": "wv",
            "o": "wo",
            "gate": "w_gate",
            "up": "w_up",
            "down": "w_down",
        }
        layers = [
            LayerWeights(
                attn_norm=norms[i, 0],
                ffn_norm=norms[i, 1],
                **{
                    field: read_bf16(weights_dir / f"layer{i}_{key}.npy")
                    for key, field in fields.items()
                },
            )
            for i in range(32)
        ]
        self.weights = LlamaWeights(None, layers, self.final_norm, None)
        p = self.prefill_length
        self.layer_luts = np.tile(self.lut[None, :p, :], (32, 1, 1))
        preload_prefill_weights(
            self.weights, self.config, self.cache, p, self.layer_luts
        )
        if self.resident_prefill:
            # KernelCache writes even intermediate slots on their FIRST use.
            # Prime the attention BO set now, before a real request produces QKV.
            # No request reset or actual prompt inference occurs during setup.
            self.cache.load_and_run(
                "flash_attn",
                {"output_format": "elf", "instance_name": "attention_bf16"},
                np.zeros((p, 4096), bfloat16),
                np.zeros((p, 1024), bfloat16),
                np.zeros((p, 1024), bfloat16),
                np.zeros((p, 4096), bfloat16),
            )
            rms = self.cache._cached_bos["rms_gemms_rope_L0"]
            attn = self.cache._cached_bos["flash_attn"]
            ffn = self.cache._cached_bos["o_ffn_L0"]
            assert rms[11] is attn[0] and rms[12] is attn[1] and rms[8] is attn[2]
            assert attn[3] is ffn[0] and rms[0] is ffn[3] and ffn[3] is ffn[14]
        head_weights = read_bf16(weights_dir / "head_bf16.npy")
        self.head_parts = []
        for part in range(2):
            lo, hi = part * 16384, min((part + 1) * 16384, 32768)
            arr = np.zeros((16384, 4096), bfloat16)
            if hi > lo:
                arr[: hi - lo] = head_weights[lo:hi]
            self.head_parts.append(arr)
        self.head_submissions = 0
        self.head(np.zeros(4096, bfloat16))
        self.position = None
        self.request_number = 0
        self.prefill_submissions = 0
        self.closed = False
        self.loading_seconds = time.perf_counter() - start
        save_json(self.work / "session.json", self.describe_execution())

    def head(self, hidden):
        from ..shared.infra.backend_presets import LM_GEMV_BACKEND

        args = [np.ascontiguousarray(hidden, dtype=bfloat16)]
        for part in self.head_parts:
            args.extend([part, np.zeros(16384, bfloat16)])
        out = self.head_cache.load_and_run(
            "lm_head_gemv",
            LM_GEMV_BACKEND,
            *args,
            output_indices=list(range(2, 5, 2)),
            static_input_indices=set(range(1, 4, 2)),
            intermediate_indices=set(range(2, 5, 2)),
        )
        self.head_submissions += 1
        return np.concatenate([out[i] for i in range(2, 5, 2)]).astype(np.float32)[
            :32768
        ]

    def prefill(self, prompt_ids):
        from llama32_1b_prefill import run_transformer_block

        if self.closed:
            raise RuntimeError("session is closed")
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if (
            ids.ndim != 1
            or len(ids) != self.prefill_length
            or np.any((ids < 0) | (ids >= 32768))
        ):
            raise ValueError("invalid prompt IDs")
        self.position = None
        self.request_number += 1
        x = np.zeros((self.prefill_length, 4096), bfloat16)
        x[: len(ids)] = self.embed[ids]
        keys = np.zeros((32, len(ids), 1024), bfloat16)
        values = np.zeros_like(keys)
        for layer in range(32):
            x, inter = run_transformer_block(
                x,
                self.weights.layers[layer],
                self.layer_luts[layer],
                self.config,
                self.cache,
                layer,
                cpu_attn=False,
            )
            keys[layer] = inter["k_roped"][: len(ids)]
            values[layer] = inter["v"][: len(ids)]
            self.prefill_submissions += 3
        # Stage request KV; physical overwrite/handoff follows first-token delivery.
        self._pending_kv = (keys, values, len(ids))
        if self.record_state:
            self.last_prefill_keys, self.last_prefill_values = keys, values
        xf = x[len(ids) - 1].astype(np.float32)
        normalized = (xf / np.sqrt(np.mean(xf * xf) + 1e-5)).astype(bfloat16)
        hidden = (
            normalized.astype(np.float32) * self.final_norm.astype(np.float32)
        ).astype(bfloat16)
        logits = self.head(hidden)
        self.position = len(ids)
        return logits

    def finish_prefill(self):
        """Complete the physical KV reset/handoff after the first-token boundary."""
        pending = getattr(self, "_pending_kv", None)
        if pending is not None:
            self.decode.seed_kv(*pending)
            self._pending_kv = None

    def decode_step(self, input_token, current_pos):
        self.finish_prefill()
        if self.closed or self.position is None or current_pos != self.position:
            raise ValueError("decode position does not match active request")
        if not 0 <= input_token < 32768 or not 0 <= current_pos < self.capacity:
            raise ValueError("decode token/position out of bounds")
        logits = self.decode.step(input_token, current_pos)
        self.position += 1
        return logits

    def synchronize(self):
        # Both paths synchronously wait and read back before returning logits.
        pass

    def describe_execution(self):
        return {
            "model": "mistral7b-v03",
            "blocks": 32,
            "prefill_build_length": self.prefill_length,
            "kv_capacity": self.capacity,
            "prefill_native_submissions_per_layer": 3,
            "prefill_rms_implementation": self.prefill_rms_impl,
            "prefill_gemm_bfp16": self.config.gemm_bfp16,
            "resident_prefill_handoffs": self.resident_prefill,
            "artifact_identity": self.artifact_identity,
            "packing_manifest_sha256": self.packing_manifest_sha256,
            "prefill_head_native_submissions": 1,
            "decode_native_submissions_per_token": 1,
            "decode_body": "32 device layer waves followed by two full-vocabulary head waves",
            "observed_requests": self.request_number,
            "observed_prefill_block_submissions": self.prefill_submissions,
            "observed_prefill_head_submissions_including_setup": self.head_submissions,
            "observed_decode_submissions": self.decode.submissions,
            "cpu": [
                "embedding lookup",
                "prefill final single-row RMSNorm",
                "KV handoff",
                "dispatch",
                "sampling",
            ],
            "loading_seconds": self.loading_seconds,
            "rope_theta": 1000000.0,
            "rope_scaling": None,
            "sliding_window": None,
            "decode_weight_groups": 4,
            "decode_head_waves": 2,
            "rope_layers": [i for i in range(32)],
            "ffn_logical_width": 14336,
            "ffn_padded_width": 14336,
        }

    def close(self):
        if not self.closed:
            save_json(self.work / "session_close.json", self.describe_execution())
            self.cache.close()
            self.head_cache.close()
            self.decode.close()
            self.closed = True


def create_session(*, task: dict, runtime: dict):
    return MistralSession(task, runtime)
