# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# BO/template handling adapted from the declared Granite and SmolLM3 examples.
"""Phi-2 production adapter shared by generation and fixed evaluation."""

import json
import os
from pathlib import Path
import time
import numpy as np
from ml_dtypes import bfloat16
from air_examples.llms.shared.deployment import require_bfp16
from .common import ROOT, RUN, REVISION, save_json, file_hash
from .build import decode_module


def rope_lut(capacity):
    inv = 1.0 / (np.float32(10000.0) ** (np.arange(0, 32, 2, dtype=np.float32) / 32))
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv[None, :]
    return np.concatenate([np.cos(angles), np.sin(angles)], axis=1).astype(bfloat16)


def read_bf16(path):
    return np.load(path, mmap_mode="r", allow_pickle=False).view(bfloat16)


def load_cache(path):
    from air_examples.llms.shared.infra.cache import KernelCache
    from air.backend.xrt import XRTCompileArtifact

    cache = KernelCache(cache_dir=str(path))
    info = json.loads((path / cache.MANIFEST_FILE).read_text())
    if info["_toolchain"] != cache._toolchain_id():
        raise RuntimeError(f"build toolchain changed: {path}")
    for name, entry in info["entries"].items():
        binary = path / Path(entry["output_binary"]).name
        insts = path / Path(entry["insts"]).name if entry["insts"] else None
        if not binary.is_file():
            raise FileNotFoundError(binary)
        kernel = entry["kernel"]
        if kernel == "MLIR_AIE" and binary.suffix == ".elf":
            # Early builds retained the XRT default in metadata. Bind to the
            # outer runtime-sequence symbol, recorded in candidate-bound IR.
            import re

            (symbol,) = re.findall(
                r"func\.func @(\w+)\(", (path.parent / f"{name}.mlir").read_text()
            )
            kernel = "main:" + symbol
        cache.artifacts[name] = XRTCompileArtifact(
            str(binary), kernel, str(insts) if insts else None
        )
    return cache


def close_cache(cache):
    cache.close()


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
        assert fd.UNI_DEC == 32 and not fd.W_SPLIT
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
        self.final_norm = read_bf16(weights / "final_norm_weight.npy")
        self.final_bias = read_bf16(weights / "final_norm_bias.npy")
        self.embed = read_bf16(weights / "embedding.npy")
        self.lut = lut
        self.to = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fr = xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        self.decode_y = (fd.HOST_ROUNDS + fd.LAYER_RNDS) * fd.PAYLOAD
        self.ny = self.decode_y + fd.UNI_LM * fd.VOCAB_SIZE_PADDED
        lengths = [
            3072,
            self.w.size,
            32 * fd.RMS_LAYER + fd.ROPE_W_LEN + 3072,
            self.ny,
            32 * capacity * fd.KVSZ_TOK,
        ]
        self.bos = [
            xrt.bo(self.dev, int(n) * 2, xrt.bo.host_only, self.kernel.group_id(3 + i))
            for i, n in enumerate(lengths)
        ]
        self.bos[1].write(self.w.view(np.int16), 0)
        self.bos[1].sync(self.to)
        rms = np.concatenate([self.norms, self.lut[0], self.final_norm])
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
        x = np.zeros(3072, bfloat16)
        x[:2560] = self.embed[token]
        self.bos[0].write(x.view(np.int16), 0)
        self.bos[0].sync(self.to)
        layer_lut = np.ascontiguousarray(self.lut[pos])
        self.bos[2].write(layer_lut.view(np.int16), self.norms.size * 2)
        self.bos[2].sync(self.to, layer_lut.nbytes, self.norms.size * 2)
        state = self.kernel(3, self.insts["ib"], size, *self.bos).wait(60000)
        self.submissions += 1
        if not str(state).endswith("COMPLETED"):
            raise RuntimeError(f"fused decode position {pos}: {state}")
        self.bos[0].sync(self.fr)
        return np.frombuffer(self.bos[0].map(), dtype=bfloat16, count=3072).copy()

    def close(self):
        self.bos = []
        self.insts = None
        self.kernel = None
        self.contexts = None
        self.dev = None


class PhiSession:
    def __init__(self, task, runtime):
        start = time.perf_counter()
        self.root = Path(runtime["candidate_root"]).resolve()
        self.work = Path(runtime["work_dir"]).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        selected = runtime["configuration"]
        expected = next(p for p in task["prompts"] if p["id"] == selected["id"])
        for key in ("tokens", "prefill_build_length", "capacity", "ids_sha256"):
            if selected[key] != expected[key]:
                raise ValueError("fixed configuration differs: " + key)
        if task["model"]["revision"] != REVISION:
            raise ValueError("checkpoint differs")
        self.p, self.capacity = selected["prefill_build_length"], selected["capacity"]
        if (self.p, self.capacity) not in ((512, 1024), (1024, 2048)):
            raise ValueError("unsupported native configuration")
        base = self.root / RUN
        build = base / selected["id"] / "prefill"
        decode = base / selected["id"] / "decode"
        head = base / "shared/head"
        weights = base / "weights"
        manifest = json.loads((weights / "manifest.json").read_text())
        if manifest["logical_manifest_sha256"] != task["assets"]["q4_manifest_sha256"]:
            raise ValueError("logical Q4 identity differs")
        build_info = json.loads((build / "build.json").read_text())
        require_bfp16(build_info)
        if build_info["sequence_length"] != self.p:
            raise ValueError("compiled P differs")
        self.prefill_gemm_bfp16 = build_info["gemm_bfp16"]
        self.cache = load_cache(build / "cache")
        self.head_cache = load_cache(head / "cache")
        self.lut = rope_lut(self.capacity)
        self.decode = FusedDecode(decode, weights, self.lut, self.capacity)
        norms = read_bf16(weights / "norms.npy")
        self.weights = [
            dict(
                {
                    k: read_bf16(weights / f"layer{l}_{k}.npy")
                    for k in ("qkv", "o", "fc1", "down")
                },
                norms=norms[l],
            )
            for l in range(32)
        ]
        from .prefill import Prefill

        self.prefiller = Prefill(self.cache, self.weights, self.p, self.lut)
        self.head_parts = [read_bf16(weights / f"head{i}.npy") for i in range(4)]
        self.head_submissions = 0
        self.position = None
        self.closed = False
        self.request_number = 0
        self.artifacts = {
            str(f.resolve()): file_hash(f)
            for path, pattern in (
                (build / "cache", "*.elf"),
                (decode, "decode_L*.xclbin"),
                (decode, "decode_L*.bin"),
                (head / "cache", "*.elf"),
            )
            for f in path.glob(pattern)
        }
        self.loading_seconds = time.perf_counter() - start
        save_json(self.work / "session.json", self.describe_execution())

    def head(self, hidden):
        args = [
            np.ascontiguousarray(hidden, dtype=bfloat16),
            self.decode.final_norm,
            self.decode.final_bias,
        ]
        for part in self.head_parts:
            args.extend([part, np.zeros(12800, bfloat16)])
        args.append(np.zeros(3072, bfloat16))
        outs = [4, 6, 8, 10]
        ret = self.head_cache.load_and_run(
            "head",
            {"output_format": "elf", "instance_name": "phi_head"},
            *args,
            output_indices=outs,
            static_input_indices={1, 2, 3, 5, 7, 9},
            intermediate_indices={4, 6, 8, 10, 11},
        )
        self.head_submissions += 1
        return np.concatenate([ret[i] for i in outs]).astype(np.float32)

    def prefill(self, prompt_ids):
        if self.closed:
            raise RuntimeError("closed session")
        ids = np.asarray(prompt_ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) != self.p or np.any((ids < 0) | (ids >= 51200)):
            raise ValueError("invalid prompt IDs")
        self.position = None
        self.request_number += 1
        x = np.zeros((self.p, 3072), bfloat16)
        x[:, :2560] = self.decode.embed[ids]
        x, keys, values = self.prefiller.run(x)
        self._pending_kv = (keys, values, self.p)
        logits = self.head(x[-1])
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
            raise ValueError("decode absolute position differs")
        if not 0 <= input_token < 51200 or not current_pos < min(self.capacity, 2048):
            raise ValueError("token or native position out of bounds")
        logits = self.head(self.decode.step(input_token, current_pos))
        self.position += 1
        return logits

    def synchronize(self):
        pass  # All submissions wait and read back before returning.

    def describe_execution(self):
        return dict(
            model="microsoft/phi-2",
            revision=REVISION,
            blocks=32,
            prefill_build_length=self.p,
            kv_capacity=self.capacity,
            native_context=2048,
            decode_native_body_submissions_per_token=1,
            head_native_submissions_per_output=1,
            decode_body="32 device layer waves, parallel attention/MLP semantics; head waves excluded",
            head="one submission: final biased LayerNorm then four 12800-row BF16 reconstructions of independent canonical Q4 head",
            prefill_native_submissions_per_layer=11,
            cpu=[
                "embedding lookup",
                "KV reset/handoff",
                "dispatch/transfers/layout/readback",
                "sampling and finite-logit checks in caller",
            ],
            logical_head_dim=80,
            physical_head_dim=128,
            rotary_dim=32,
            rope_theta=10000,
            rope_scaling=None,
            rope_lut_dtype="BF16",
            layernorm="FP32 mean/variance over 2560, epsilon1e-5, biased; nearest-even BF16 output",
            gelu="gelu_new primitive LUT with original-HF BF16 operation rounding",
            bias="BF16 affine coefficients multiplied by exact one in padded NPU input coordinates",
            prefill_gemm="canonical Q4 reconstruction rounded to BF16, "
            + ("BFP16 emulation" if self.prefill_gemm_bfp16 else "native BF16 multiply")
            + ", FP32 accumulation",
            prefill_gemm_bfp16=self.prefill_gemm_bfp16,
            decode_projection="affine Q4NX BFP16 emulation with FP32 accumulation",
            attention="prefill BFP16; decode BF16 vector products, online softmax; scale 1/sqrt(80)",
            loading_seconds=self.loading_seconds,
            artifacts=self.artifacts,
            request_count=self.request_number,
            decode_submissions=self.decode.submissions,
            head_submissions=self.head_submissions,
            prefill_submissions=self.prefiller.submissions,
        )

    def close(self):
        if not self.closed:
            save_json(self.work / "session-final.json", self.describe_execution())
            close_cache(self.cache)
            close_cache(self.head_cache)
            self.decode.close()
            self.closed = True


def create_session(*, task: dict, runtime: dict):
    return PhiSession(task, runtime)
