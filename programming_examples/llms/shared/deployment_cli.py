# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared command-line interface for the six affine-Q4 deployment examples."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .deployment import build_root, configuration, file_hash, read_task, runtime, spec


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@contextmanager
def device_lock(path):
    # Cooperates with other AIR examples sharing the device on this host.
    import fcntl

    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checked_logits(value, vocab):
    z = np.asarray(value, dtype=np.float32)
    if z.shape != (vocab,) or not np.isfinite(z).all():
        raise ValueError(
            f"expected {vocab} finite full-vocabulary logits; got {z.shape}"
        )
    return z, int(np.argmax(z))


def request(session, selected, vocab, outputs, capture=False):
    """Fresh request: t1 includes first-logit checks and greedy selection.

    Physical KV reset/handoff follows t1 and is charged to decode/request time.
    All native calls block through synchronization and readback.
    """
    tokens, logits = [], []
    t0 = time.perf_counter()
    z, token = checked_logits(session.prefill(selected["token_ids"]), vocab)
    t1 = time.perf_counter()
    tokens.append(token)
    if capture:
        logits.append(z.copy())
    session.finish_prefill()
    handoff_end = time.perf_counter()
    for position in range(selected["tokens"], selected["tokens"] + outputs - 1):
        z, token = checked_logits(session.decode_step(token, position), vocab)
        tokens.append(token)
        if capture:
            logits.append(z.copy())
    session.synchronize()
    end = time.perf_counter()
    return dict(
        tokens=tokens,
        logits=np.stack(logits) if capture else None,
        ttft_seconds=t1 - t0,
        decode_seconds=end - t1,
        request_seconds=end - t0,
        handoff_seconds=handoff_end - t1,
        t0=t0,
        t1=t1,
        end=end,
    )


def save_capture(path, selected, result, model):
    np.savez(
        path,
        logits=result["logits"],
        chosen=np.asarray(result["tokens"], dtype=np.int64),
        input_ids=np.asarray(selected["token_ids"], dtype=np.int64),
        model=np.asarray(model),
        revision=np.asarray(spec(model)["revision"]),
    )


def compare_capture(candidate, reference, selected, model):
    from air_examples.llms.verify.comparators import compute_topk_set_check

    s = spec(model)
    records = []
    for path in (candidate, reference):
        with np.load(path, allow_pickle=False) as data:
            if (
                str(data["model"]) != model
                or str(data["revision"]) != s["revision"]
                or not np.array_equal(data["input_ids"], selected["token_ids"])
            ):
                raise ValueError(f"capture model/input identity differs: {path}")
            z = data["logits"]
            if z.shape != (32, s["vocab_size"]) or not np.isfinite(z).all():
                raise ValueError(
                    f"expected complete finite (32, vocab) capture: {path}"
                )
            chosen = np.argmax(z, axis=1).tolist()
            if not np.array_equal(chosen, data["chosen"]):
                raise ValueError(f"capture greedy choices disagree with logits: {path}")
            # Full stable ordering handles ties at the top-5 boundary exactly.
            top5 = np.argsort(-z, axis=1, kind="stable")[:, :5].tolist()
            records.append((chosen, top5))
    c, r = records
    result = compute_topk_set_check(c[0], c[1], r[0], r[1], k=5).to_dict()
    result.update(
        candidate_sha256=file_hash(candidate), reference_sha256=file_hash(reference)
    )
    return result


def checkpoint(model, requested):
    if requested:
        return Path(requested).resolve()
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE

    s = spec(model)
    return Path(
        snapshot_download(
            s["checkpoint"],
            revision=s["revision"],
            allow_patterns=["*.safetensors", "*.json", "*.py"],
            local_files_only=HF_HUB_OFFLINE,
        )
    )


def reference(model, selected, source, out, threads):
    """Original checkpoint on CPU BF16, free-running greedy, no Q4 reference."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    s = spec(model)
    if file_hash(source / "config.json") != s["config_sha256"]:
        raise ValueError("reference config differs from pinned checkpoint")
    for shard in s["source_shards"]:
        if file_hash(source / shard["file"]) != shard["sha256"]:
            raise ValueError("reference checkpoint shard differs: " + shard["file"])
    for entry in s.get("reference_code", []):
        if file_hash(source / entry["file"]) != entry["sha256"]:
            raise ValueError("reference HF implementation differs: " + entry["file"])
    expected = "4.39.3" if model == "openelm_1b_q4nx" else "5.3.0"
    if transformers.__version__ != expected:
        raise ValueError(
            f"{model} reference requires transformers=={expected}; use a separate reference environment"
        )
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    net = AutoModelForCausalLM.from_pretrained(
        str(source),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=model == "openelm_1b_q4nx",
        attn_implementation=s["reference_attention"],
    )
    net.cpu().eval()
    tokens, logits, past = [], [], None
    ids = torch.tensor([selected["token_ids"]], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(32):
            result = net(
                input_ids=ids, past_key_values=past, use_cache=True, return_dict=True
            )
            z, token = checked_logits(
                result.logits[0, -1].float().cpu().numpy(), s["vocab_size"]
            )
            logits.append(z.copy())
            tokens.append(token)
            past = result.past_key_values
            ids = torch.tensor([[token]], dtype=torch.long)
    save_capture(out, selected, dict(logits=np.stack(logits), tokens=tokens), model)
    write_json(
        out.with_suffix(".json"),
        dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            source=str(source),
            config=selected,
            dtype="BF16",
            device="CPU",
            checkpoint=s["checkpoint"],
            revision=s["revision"],
            outputs=32,
            ignore_eos=True,
            sha256=file_hash(out),
        ),
    )


def main(model):
    s = spec(model)
    parser = argparse.ArgumentParser(
        description=s["name"] + " on NPU2, affine Q4 / BFP16 prefill"
    )
    parser.add_argument(
        "command",
        choices=("info", "weights", "compile", "reference", "verify", "run", "profile"),
    )
    parser.add_argument(
        "--prefill",
        type=int,
        default=512,
        help="native input length; capacity follows model.json",
    )
    parser.add_argument("--build-dir", type=Path, default=build_root(model))
    parser.add_argument(
        "--q4-dir", type=Path, help="reuse a canonical affine-Q4 directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local pinned HF checkpoint; otherwise download the pinned revision",
    )
    parser.add_argument(
        "--input-ids",
        type=Path,
        help="JSON list of exactly P token IDs (default: packaged input)",
    )
    parser.add_argument(
        "--part", choices=("all", "kernels", "decode", "prefill", "head"), default="all"
    )
    parser.add_argument(
        "--out", type=Path, help="new output directory for this invocation"
    )
    parser.add_argument(
        "--capacity",
        type=int,
        help="compile --part decode: build the decode templates for this KV "
        "capacity into --out, outside the native configurations "
        "(bench/sweep_decode.py)",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="directory containing p<P>.npz from the reference command",
    )
    parser.add_argument(
        "--outputs",
        type=int,
        default=32,
        help="run output count; verify=32, profile=128",
    )
    parser.add_argument("--threads", type=int, default=8, help="CPU reference threads")
    parser.add_argument("--lock", default="/tmp/mlir-air-npu.lock")
    parser.add_argument(
        "--check-reset",
        action="store_true",
        help="verify: repeat the capture after an intervening changed request",
    )
    args = parser.parse_args()
    if args.command == "info":
        print(json.dumps(s, indent=2))
        return
    os.environ["AIR_TARGET_DEVICE"] = "npu2"
    os.environ["AIR_MODEL_BUILD_DIR"] = str(args.build_dir.resolve())
    if args.q4_dir:
        os.environ["AIR_MODEL_Q4_DIR"] = str(args.q4_dir.resolve())
    selected = configuration(
        model,
        args.prefill,
        json.loads(args.input_ids.read_text()) if args.input_ids else None,
    )
    base = build_root(model)
    if args.command == "weights":
        q4 = Path(os.environ.get("AIR_MODEL_Q4_DIR", base / "logical-q4"))
        if not (q4 / "COMPLETE.json").exists():
            from .prepare_affine_q4 import prepare

            prepare(checkpoint(model, args.checkpoint), q4, s)
        packer = importlib.import_module(f"air_examples.llms.{model}.weights")
        packer.prepare(base / "weights", read_task(model))
        return
    if args.command == "compile":
        build = importlib.import_module(f"air_examples.llms.{model}.build")
        if args.capacity is not None:
            if args.part != "decode" or args.out is None:
                parser.error(
                    "--capacity builds decode templates only: --part decode --out DIR"
                )
            build.build_decode(args.out.resolve(), args.capacity)
            print("Compilation passed.")
            return
        folder = base / selected["id"]
        if args.part == "kernels":
            build.build_decode_kernels(folder / "decode", selected["capacity"])
            return
        if args.part in ("all", "decode"):
            build.build_decode(folder / "decode", selected["capacity"])
        if args.part in ("all", "prefill"):
            build.build_prefill(
                folder / "prefill",
                selected["tokens"],
                rms_impl=s["prefill_rms"][selected["id"]],
            )
        if args.part in ("all", "head"):
            build.build_head(base / "shared/head")
        print("Compilation passed.")
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = (args.out or base / "runs" / (stamp + "-" + args.command)).resolve()
    out.mkdir(parents=True, exist_ok=args.command == "reference")
    if args.command == "reference":
        if (out / (selected["id"] + ".npz")).exists():
            raise ValueError(
                "reference capture already exists; choose a new output directory"
            )
        reference(
            model,
            selected,
            checkpoint(model, args.checkpoint),
            out / (selected["id"] + ".npz"),
            args.threads,
        )
        return
    outputs = {"verify": 32, "profile": 128}.get(args.command, args.outputs)
    if outputs < 1 or selected["tokens"] + outputs - 1 > selected["capacity"]:
        raise ValueError("output count exceeds the compiled capacity")
    if args.command == "verify" and not args.reference_dir:
        parser.error(
            "verify requires --reference-dir from a prior CPU reference command"
        )
    if (
        args.command == "verify"
        and not (args.reference_dir / (selected["id"] + ".npz")).is_file()
    ):
        parser.error("missing reference capture for " + selected["id"])
    task = read_task(model, selected)
    factory = importlib.import_module(
        f"air_examples.llms.{model}.session"
    ).create_session
    with device_lock(args.lock):
        session = factory(task=task, runtime=runtime(model, selected, out / "session"))
        try:
            if args.command == "profile":
                warmups, measured = [], []
                while sum(r["request_seconds"] for r in warmups) < 5:
                    warmups.append(request(session, selected, s["vocab_size"], outputs))
                    write_json(
                        out / "samples.json", dict(warmups=warmups, requests=measured)
                    )
                while (
                    len(measured) < 3
                    or sum(r["request_seconds"] for r in measured) < 10
                ):
                    measured.append(
                        request(session, selected, s["vocab_size"], outputs)
                    )
                    write_json(
                        out / "samples.json", dict(warmups=warmups, requests=measured)
                    )
                summary = dict(
                    median_ttft_seconds=float(
                        np.median([r["ttft_seconds"] for r in measured])
                    ),
                    mean_request_seconds=float(
                        np.mean([r["request_seconds"] for r in measured])
                    ),
                    decode_tokens_per_second=(outputs - 1)
                    * len(measured)
                    / sum(r["decode_seconds"] for r in measured),
                )
                summary["prefill_tokens_per_second"] = (
                    selected["tokens"] / summary["median_ttft_seconds"]
                )
                write_json(
                    out / "timing.json",
                    dict(
                        model=model,
                        configuration=selected,
                        outputs=outputs,
                        ttft_boundary="first token selected; physical KV handoff charged after t1",
                        warmups=warmups,
                        requests=measured,
                        summary=summary,
                    ),
                )
                print(json.dumps(summary, indent=2))
            else:
                result = request(
                    session, selected, s["vocab_size"], outputs, capture=True
                )
                path = out / (selected["id"] + ".npz")
                save_capture(path, selected, result, model)
                print("Generated token IDs:", result["tokens"])
                if args.command == "verify":
                    gate = compare_capture(
                        path, args.reference_dir / path.name, selected, model
                    )
                    write_json(out / "gate.json", gate)
                    print(json.dumps(gate, indent=2))
                    if gate["status"] != "OK":
                        raise SystemExit(1)
                    if args.check_reset:
                        changed = dict(selected, token_ids=list(selected["token_ids"]))
                        changed["token_ids"][0] = (changed["token_ids"][0] + 1) % s[
                            "vocab_size"
                        ]
                        request(session, changed, s["vocab_size"], 2)
                        repeated = request(
                            session, selected, s["vocab_size"], outputs, capture=True
                        )
                        save_capture(out / "repeat.npz", selected, repeated, model)
                        same = np.array_equal(result["logits"], repeated["logits"])
                        write_json(
                            out / "reset.json",
                            dict(
                                bitwise_logits_equal=bool(same),
                                protocol="A(32 outputs), changed first input B(2 outputs), A(32 outputs)",
                            ),
                        )
                        if not same:
                            raise ValueError(
                                "capture differs after an intervening changed request"
                            )
        finally:
            session.close()
