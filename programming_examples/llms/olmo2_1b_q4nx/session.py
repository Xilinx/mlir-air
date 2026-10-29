# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# BO/template handling adapted from the declared Granite and SmolLM3 examples.
"""OLMo2 production adapter shared by generation and fixed evaluation."""

import json
from pathlib import Path
import time
import numpy as np
from ml_dtypes import bfloat16
from ..shared.deployment import require_bfp16
from .common import RUN, REVISION, save_json, file_hash
from .build import decode_module
from ..shared.deployment_runtime import read_bf16, load_cache


def rope_lut(capacity):
    inv = 1.0 / (np.float32(500000.0) ** (np.arange(0, 128, 2, dtype=np.float32) / 128))
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


class OLMo2Session:
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
        if task["model"]["revision"] != REVISION:
            raise ValueError("unsupported checkpoint")
        self.p, self.capacity = selected["prefill_build_length"], selected["capacity"]
        if (self.p, self.capacity) not in ((512, 1024), (1024, 2048), (2048, 4096)):
            raise ValueError("unsupported native-context configuration")
        base = self.root / RUN
        build = base / selected["id"]
        prefill_build = self.root / runtime.get(
            "prefill_build_root", str(Path(RUN) / selected["id"] / "prefill")
        )
        self.prefill_build = prefill_build
        weights = base / "weights"
        manifest = json.loads((weights / "manifest.json").read_text())
        if manifest["logical_manifest_sha256"] != task["assets"]["q4_manifest_sha256"]:
            raise ValueError("logical Q4 identity differs")
        prefill_meta = json.loads((prefill_build / "build.json").read_text())
        require_bfp16(prefill_meta)
        self.prefill_gemm_bfp16 = prefill_meta["gemm_bfp16"]
        if prefill_meta["sequence_length"] != self.p:
            raise ValueError("compiled prefill length differs")
        self.cache = load_cache(prefill_build / "cache")
        self.head_cache = load_cache(base / "shared/head/cache")
        self.lut = rope_lut(self.capacity)
        self.decode = FusedDecode(build / "decode", weights, self.lut, self.capacity)
        norms = read_bf16(weights / "norms.npy")
        self.weights = [
            dict(
                {
                    k: read_bf16(weights / f"layer{l}_{k}.npy")
                    for k in ("q", "k", "v", "o", "gate", "up", "down")
                },
                norms=norms[l],
            )
            for l in range(16)
        ]
        from .prefill import Prefill

        self.prefiller = Prefill(self.cache, self.weights, self.p, self.lut)
        head = read_bf16(weights / "head_bf16.npy")
        self.head_parts = []
        for i in range(8):
            arr = np.zeros((16384, 2048), bfloat16)
            count = max(0, min(16384, 100352 - i * 16384))
            arr[:count] = head[i * 16384 : i * 16384 + count]
            self.head_parts.append(arr)
        self.head_submissions = 0
        self.position = None
        self.closed = False
        self.request_number = 0
        self.artifacts = {
            str(f.resolve()): file_hash(f)
            for path, pattern in (
                (prefill_build / "cache", "*.elf"),
                (build / "decode", "decode_L*.xclbin"),
                (build / "decode", "decode_L*.bin"),
                (base / "shared/head/cache", "*.elf"),
            )
            for f in path.glob(pattern)
        }
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
            output_indices=list(range(2, 17, 2)),
            static_input_indices=set(range(1, 16, 2)),
            intermediate_indices=set(range(2, 17, 2)),
        )
        self.head_submissions += 1
        return np.concatenate([out[i] for i in range(2, 17, 2)]).astype(np.float32)[
            :100352
        ]

    def prefill(self, prompt_ids):
        if self.closed:
            raise RuntimeError("closed session")
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != self.p or np.any((ids < 0) | (ids >= 100352)):
            raise ValueError("invalid prompt IDs")
        self.position = None
        self.request_number += 1
        x = np.ascontiguousarray(self.decode.embed[ids])
        x, keys, values = self.prefiller.run(x)
        # Stage all KV slots; finish_prefill overwrites the full cache and tails.
        self._pending_kv = (keys, values, self.p)
        xf = x[-1].astype(np.float32)
        hidden = (
            xf
            / np.sqrt(np.mean(xf * xf) + 1e-6)
            * self.decode.final_norm.astype(np.float32)
        ).astype(bfloat16)
        logits = self.head(hidden)
        self.position = self.p
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
        if not 0 <= input_token < 100352 or not current_pos < min(self.capacity, 4096):
            raise ValueError("token or native absolute position out of bounds")
        logits = self.decode.step(input_token, current_pos)
        self.position += 1
        return logits

    def synchronize(self):
        # Every submit waits and every output is read back before return.
        pass

    def describe_execution(self):
        return dict(
            model="OLMo-2-0425-1B-Instruct",
            revision=REVISION,
            blocks=16,
            prefill_build_length=self.p,
            kv_capacity=self.capacity,
            decode_native_submissions_per_token=1,
            decode_body="16 device layer waves followed by seven independent-head waves",
            prefill_native_submissions_per_layer=17,
            prefill_head_native_submissions=1,
            cpu=[
                "embedding lookup",
                "single-row prefill final norm",
                "KV reset/handoff",
                "dispatch/transfers/readback",
                "sampling/interface checks in caller",
            ],
            qk_norm_domain=2048,
            post_norm_before_residual=True,
            rope_theta=500000,
            rope_scaling=None,
            prefill_gemm="logical Q4 reconstruction to BF16, "
            + ("BFP16" if self.prefill_gemm_bfp16 else "native BF16")
            + " multiply, FP32 accumulation",
            prefill_build_root=str(self.prefill_build.resolve()),
            decode_projection="affine Q4NX with BFP16 emulation and FP32 accumulation",
            decode_attention="BF16 vector multiplication, FP32 accumulation, BF16 softmax scores",
            loading_seconds=self.loading_seconds,
            artifacts=self.artifacts,
            request_count=self.request_number,
            decode_submissions=self.decode.submissions,
        )

    def close(self):
        if not self.closed:
            save_json(self.work / "session-final.json", self.describe_execution())
            self.cache.close()
            self.head_cache.close()
            self.decode.close()
            self.closed = True


def create_session(*, task: dict, runtime: dict):
    return OLMo2Session(task, runtime)
