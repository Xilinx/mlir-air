# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""BF16/BFP prepass: reference error, isolation, repeatability and wall timing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from ml_dtypes import bfloat16

from bfp_context_prepass import BF16ContextReference as ContextPrepass
from bfp_context_prepass import BFPContextPrepass as NPUContextPrepass
from bfp_reference import bfp_operand
from common.context_reference import rmsnorm, rope


def error_metrics(actual, expected):
    actual = np.asarray(actual, np.float64)
    expected = np.asarray(expected, np.float64)
    delta = actual - expected
    return dict(
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30)),
        max_abs=float(np.max(np.abs(delta), initial=0)),
        scaled_max=float(
            np.max(np.abs(delta), initial=0)
            / max(np.max(np.abs(expected), initial=0), 1e-30)
        ),
        bf16_equal_fraction=float(
            np.mean(
                actual.astype(bfloat16).view(np.uint16)
                == expected.astype(bfloat16).view(np.uint16)
            )
        ),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--fixture", type=Path, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="starting",
        passed=False,
        cases=[],
        fixture=str(args.fixture),
        projection_tolerance_relative_l2=2e-5,
        projection_tolerance_scaled_max=2e-5,
        numerical_reference="BFP operand conversion with conv_even; FP32 accumulation",
        bf16_reference_note="Unrounded BF16/FP32 reference errors are accuracy metrics, not BFP implementation gates. The model-token gate separately assesses the production arithmetic against BF16.",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        reference = ContextPrepass()
        fc_bfp = bfp_operand(reference.fc)
        kv_bfp = bfp_operand(np.concatenate(reference.kv_weights, axis=0))
        device = NPUContextPrepass(args.artifacts, reference)
        device.warmup()

        def checked_stages(x, positions):
            actual = device.chunk(x, positions, capture=True)
            raw = device.last_raw
            fc_expected = bfp_operand(x) @ fc_bfp.T
            normalized = rmsnorm(
                raw["fused"], reference.hidden_norm, reference.eps
            ).astype(bfloat16)
            kv_expected = bfp_operand(actual[0]) @ kv_bfp.T
            keys = []
            for layer in range(5):
                k = raw["raw_kv"][:, layer * 2048 : layer * 2048 + 1024].reshape(
                    -1, 8, 128
                )
                k = rmsnorm(k, reference.k_norm[layer], reference.eps)
                keys.append(rope(k, positions, reference.theta).reshape(-1, 1024))
            stages = {
                "fc_bfp": error_metrics(raw["fused"], fc_expected),
                "kv_bfp": error_metrics(raw["raw_kv"], kv_expected),
                "key_norm_rope": error_metrics(actual[1], np.stack(keys)),
                "feature_norm": error_metrics(actual[0], normalized),
            }
            stages["elementwise"] = {}
            for name, got, want, atol in (
                ("fc_bfp", raw["fused"], fc_expected, 0.0015),
                ("kv_bfp", raw["raw_kv"], kv_expected, 0.0015),
                ("feature_norm", actual[0], normalized, 0.05),
                ("key_norm_rope", actual[1], np.stack(keys), 0.0015),
            ):
                got = np.asarray(got, np.float32)
                want = np.asarray(want, np.float32)
                bad = int(
                    np.count_nonzero(
                        ~np.isfinite(got)
                        | (np.abs(got - want) > atol + 0.016 * np.abs(want))
                    )
                )
                stages["elementwise"][name] = dict(
                    elements=got.size, rtol=0.016, atol=atol, out_of_tolerance=bad
                )
                assert bad == 0, (name, stages["elementwise"][name])
            bits_a = actual[0].astype(bfloat16).view(np.uint16).astype(np.int32)
            bits_b = normalized.view(np.uint16).astype(np.int32)
            stages["feature_norm_max_bf16_steps"] = int(
                np.abs(bits_a - bits_b).max(initial=0)
            )
            for name in ("fc_bfp", "kv_bfp", "key_norm_rope"):
                assert (
                    stages[name]["relative_l2"] < 2e-5
                    and stages[name]["scaled_max"] < 2e-5
                ), (name, stages)
            assert stages["feature_norm_max_bf16_steps"] <= 1, stages
            values = (
                raw["raw_kv"]
                .reshape(len(positions), 5, 2048)[:, :, 1024:]
                .transpose(1, 0, 2)
            )
            assert np.array_equal(actual[2], values)
            return actual, stages

        with np.load(args.fixture) as z:
            taps = z["taps"].view(bfloat16).astype(np.float32).reshape(-1, 20480)
        rng = np.random.default_rng(20260915)
        inputs = [
            ("prompt", taps[:8]),
            ("random", rng.normal(0, 0.25, (8, 20480)).astype(np.float32)),
            (
                "row_scales",
                (
                    rng.normal(size=(8, 20480)) * np.geomspace(0.01, 100, 8)[:, None]
                ).astype(np.float32),
            ),
            ("identical", np.repeat(taps[:1], 8, axis=0)),
            ("zeros", np.zeros((8, 20480), np.float32)),
        ]
        for name, x in inputs:
            for n in (1, 2, 3, 4, 5, 6, 7, 8):
                positions = np.arange(n) + ((0 if name == "prompt" else 2051))
                expected = reference.run(x[:n], positions)
                actual, stages = checked_stages(x[:n], positions)
                errors = {
                    key: error_metrics(a, b)
                    for key, a, b in zip(("features", "k", "v"), actual, expected)
                }
                repeated = device.run(x[:n], positions)
                exact = all(np.array_equal(a, b) for a, b in zip(actual, repeated))
                full = device.run(x, np.arange(8) + int(positions[0]))
                isolated = all(
                    np.array_equal(a, b[:n] if j == 0 else b[:, :n])
                    for j, (a, b) in enumerate(zip(actual, full))
                )
                row = dict(
                    input=name,
                    rows=n,
                    errors=errors,
                    stages=stages,
                    repeat_exact=exact,
                    padding_exact=isolated,
                )
                result["cases"].append(row)
                save()
                print(json.dumps(row), flush=True)
                assert exact and isolated, row
        positions = np.array([0, 1, 31, 127, 511, 2048, 4088, 4095])
        expected = reference.run(taps[:8], positions)
        actual, stages = checked_stages(taps[:8], positions)
        errors = {
            key: error_metrics(a, b)
            for key, a, b in zip(("features", "k", "v"), actual, expected)
        }
        result["mixed_positions"] = {
            "positions": positions.tolist(),
            "errors": errors,
            "stages": stages,
        }
        device.chunk(taps[:8], np.arange(8), capture=True)
        fused = taps[:8].astype(bfloat16).astype(np.float32) @ reference.fc.T
        projected = (
            device.maps[6].astype(np.float32)
            @ np.concatenate(reference.kv_weights, axis=0).T
        )
        result["projection_errors"] = {
            "feature_projection": error_metrics(device.last_raw["fused"], fused),
            "context_projection": error_metrics(device.last_raw["raw_kv"], projected),
        }
        result["timings"] = []
        for n in range(1, 9):
            cpu_ms = []
            npu_ms = []
            calls = []
            for repeat in range(args.repeats):
                for mode in (("cpu", "npu") if repeat % 2 == 0 else ("npu", "cpu")):
                    start = time.perf_counter()
                    (reference if mode == "cpu" else device).run(
                        taps[:n], np.arange(n) + 19
                    )
                    elapsed = (time.perf_counter() - start) * 1000
                    (cpu_ms if mode == "cpu" else npu_ms).append(elapsed)
                    if mode == "npu":
                        calls.extend(device.last_calls)
            result["timings"].append(
                dict(
                    rows=n,
                    cpu_ms=cpu_ms,
                    npu_ms=npu_ms,
                    calls=calls,
                    cpu_median_ms=float(np.median(cpu_ms)),
                    npu_median_ms=float(np.median(npu_ms)),
                )
            )
            save()
        # Exercise chunk boundaries and the full prompt-context preparation.
        result["initial_prompt"] = []
        for n in sorted(set([9, 17, len(taps)])):
            x = np.resize(taps, (n, 20480))
            positions = np.arange(n)
            start = time.perf_counter()
            expected = reference.run(x, positions)
            cpu_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            actual = device.run(x, positions)
            npu_ms = (time.perf_counter() - start) * 1000
            errors = {
                key: error_metrics(a, b)
                for key, a, b in zip(("features", "k", "v"), actual, expected)
            }
            independent = [
                checked_stages(x[i : i + 8], positions[i : i + 8])[0]
                for i in range(0, n, 8)
            ]
            exact = all(
                np.array_equal(
                    actual[j],
                    np.concatenate(
                        [c[j] for c in independent], axis=0 if j == 0 else 1
                    ),
                )
                for j in range(3)
            )
            result["initial_prompt"].append(
                dict(
                    rows=n,
                    cpu_ms=cpu_ms,
                    npu_ms=npu_ms,
                    errors=errors,
                    chunk_exact=exact,
                )
            )
            assert exact
        result.update(status="passed", passed=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
