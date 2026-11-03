# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# BO/template handling adapted from the declared Granite and SmolLM3 examples.
"""OLMo2 production adapter shared by generation and fixed evaluation.

Prefill runs on the shared fused prefill (shared/fused_prefill/affine.py) for
any input up to its built length; decode is the custom Q4NX spatial decode.
"""

import json
import os
from pathlib import Path
import time
import numpy as np
from ml_dtypes import bfloat16
from .common import REVISION, save_json, file_hash
from .build import decode_module
from ..shared.deployment_runtime import FusedPrefillSession, read_bf16


def rope_lut(capacity):
    inv = 1.0 / (np.float32(500000.0) ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(bfloat16)


MODEL = "olmo2_1b_q4nx"


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
        assert fd.UNI_DEC == 16 and not fd.W_SPLIT
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
        norms = read_bf16(weights / "norms.npy")
        slots = np.zeros((16, 4, 2048), bfloat16)
        slots[:, 1], slots[:, 3] = norms[:, 2], norms[:, 3]
        self.norms = slots.reshape(-1)
        self.rope_slabs = np.zeros((16, 4224), bfloat16)
        self.rope_slabs[:, 128:2176] = norms[:, 0]
        self.rope_slabs[:, 2176:] = norms[:, 1]
        self.final_norm = read_bf16(weights / "final_norm.npy")
        self.embed = read_bf16(weights / "embedding.npy")
        self.lut = lut
        self.to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + fd.UNI_LM * fd.VOCAB_SIZE_PADDED
        lengths = [
            2048,
            self.w.size,
            16 * fd.RMS_LAYER + 16 * fd.ROPE_W_LEN + 2048,
            self.ny,
            16 * capacity * fd.KVSZ_TOK,
        ]
        self.bos = [
            xrt.bo(self.dev, int(n) * 2, xrt.bo.host_only, self.kernel.group_id(3 + i))
            for i, n in enumerate(lengths)
        ]
        self.bos[1].write(self.w.view(np.int16), 0)
        self.bos[1].sync(self.to)
        rms = np.concatenate([self.norms, self.rope_slabs.ravel(), self.final_norm])
        self.bos[2].write(rms.view(np.int16), 0)
        self.bos[2].sync(self.to)
        self.submissions = 0

    def seed_kv(self, keys, values, length):
        fd = self.fd
        rw, ng = fd.REGION_W, fd.NGRP
        stride = self.capacity * rw
        kv = np.zeros((16, self.capacity * fd.KVSZ_TOK), bfloat16)
        for layer in range(16):
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
        self.rope_slabs[:, :128] = self.lut[pos]
        layer_lut = self.rope_slabs
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
        return z[:100352]

    def close(self):
        self.bos = []
        self.insts = None
        self.kernel = None
        self.contexts = None
        self.dev = None


class OLMo2Session(FusedPrefillSession):
    vocab = 100352

    def __init__(self, task, runtime):
        start = time.perf_counter()
        self.root = Path(runtime["candidate_root"]).resolve()
        self.work = Path(runtime["work_dir"]).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        if task["model"]["revision"] != REVISION:
            raise ValueError("unsupported checkpoint revision")
        build = Path(runtime["build_dir"])
        fused = Path(runtime["fused_prefill_dir"])
        weights = Path(runtime["weights_dir"])
        manifest = json.loads((weights / "manifest.json").read_text())
        if manifest["logical_manifest_sha256"] != task["assets"]["q4_manifest_sha256"]:
            raise ValueError("packed weights do not bind the fixed logical weights")
        os.environ["AIR_TARGET_DEVICE"] = "npu2"
        self.record_state = bool(runtime.get("record_state", False))
        self.prefill_length = int(runtime["prefill_length"])
        self.capacity = int(runtime["kv_capacity"])
        self.lut = rope_lut(self.capacity)
        self.decode = FusedDecode(build / "decode", weights, self.lut, self.capacity)
        self.request_number = 0
        self.closed = False
        self.open_prefill(MODEL, fused, task["assets"]["q4_root"])
        if self.prefill_length > self.max_prompt:
            raise ValueError("the fused prefill build is shorter than the input")
        self.artifact_identity = self.prefill_identity(fused)
        self.artifact_identity.update(
            {
                f"decode/{p.name}": {"path": str(p.resolve()), "sha256": file_hash(p)}
                for pattern in ("decode_L*.bin", "decode_L*.xclbin")
                for p in (build / "decode").glob(pattern)
            }
        )
        self.packing_manifest_sha256 = file_hash(weights / "manifest.json")
        self.loading_seconds = time.perf_counter() - start
        save_json(self.work / "session.json", self.describe_execution())

    def decode_step(self, input_token, current_pos):
        self.finish_prefill()
        if self.closed or self.position is None or current_pos != self.position:
            raise ValueError("decode position does not match active request")
        if not 0 <= input_token < self.vocab or not 0 <= current_pos < self.capacity:
            raise ValueError("decode token/position out of bounds")
        logits = self.decode.step(input_token, current_pos)
        self.position += 1
        return logits

    def synchronize(self):
        # Every submit waits and reads back before returning logits.
        pass

    def describe_execution(self):
        return {
            "model": "OLMo-2-0425-1B-Instruct",
            "revision": REVISION,
            "blocks": 16,
            "prefill": "shared fused prefill, 128-row chunks, one hardware context",
            "prefill_max_input": self.max_prompt,
            "native_input_length": self.prefill_length,
            "kv_capacity": self.capacity,
            "prefill_gemm": "BFP16 (int4 weights dequantized on the cores)",
            "prefill_attention": "fused prefill attn_bfp16",
            "artifact_identity": self.artifact_identity,
            "packing_manifest_sha256": self.packing_manifest_sha256,
            "decode_native_submissions_per_token": 1,
            "decode_body": "16 device layer waves followed by seven independent-head waves",
            "observed_requests": self.request_number,
            "observed_prefills": self.prefill_dispatches,
            "observed_decode_submissions": self.decode.submissions,
            "cpu": [
                "embedding lookup",
                "prefill post-sublayer RMSNorms, q/k RMSNorm over all heads, RoPE, residual adds and SwiGLU",
                "prefill q / KV / A layouts",
                "KV handoff",
                "dispatch",
                "sampling",
            ],
            "qk_norm_domain": 2048,
            "post_norm_before_residual": True,
            "loading_seconds": self.loading_seconds,
        }

    def close(self):
        if not self.closed:
            self.prefiller.suspend()
            self.prefiller = None
            self.decode.close()
            self.closed = True


def create_session(*, task: dict, runtime: dict):
    return OLMo2Session(task, runtime)
