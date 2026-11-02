# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""CPU reference of the d3 gate request set: the package's own runtime
(d3_runtime.D3: transformers' qwen3_5 backbone in BF16, noncausal full
attention, FP32 readout) on CPU with batch_size=1, one forward pass per
question. Saves per question the token ids, option logits and probabilities,
plus the identity of the code, weights and libraries.

  python3 reference.py --model edge --out reference_edge.json

  --model edge  vllm-sr/d3-edge as published (BF16)
  --model lite  vllm-sr/d3-lite with every weight FastFlowLM's D3-Lite-NPU2
                bundle quantizes replaced by its exact dequantized value
                (Q4K d*sc*q + dmin*m, Q8 scale*q; rounded to the bf16
                parameters), the embedding from the bundle's Q8 lm_head (the
                model is tied); norms, conv, dt_bias, A_log and the readout
                as published (the bundle's are the same values)
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
sys.modules.setdefault("air_examples", types.ModuleType("air_examples")).__path__ = [
    str(HERE.parents[1])
]

from air_examples.llms.d3_edge import d3_weights as W3  # noqa: E402

HEADS, DH = 8, 256


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def requests_sha256(path):
    """The request set's identity: the sha256 of its content serialized as
    json.dumps(..., indent=1), so it does not depend on the file's layout."""
    doc = json.loads(Path(path).read_text())
    return hashlib.sha256(json.dumps(doc, indent=1).encode()).hexdigest()


def _cos(a, b):
    a, b = (np.asarray(v, np.float64).ravel() for v in (a, b))
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def lite_weights(rt):
    """Replace the backbone's quantized weights with the bundle's values;
    every one is checked against the published weight by cosine."""
    import torch

    q = W3.LiteBundle()
    bb = rt.backbone
    params = dict(bb.named_parameters())
    kinds = bb.config.text_config.layer_types
    pairs = []
    for L, kind in enumerate(kinds):
        b, h = f"model.layers.{L}.", f"language_model.layers.{L}."
        if kind == "linear_attention":
            s = b + "linear_attn."
            pairs += [
                (s + "qkv_proj.weight", h + "linear_attn.in_proj_qkv.weight"),
                (b + "self_attn.gate_proj.weight", h + "linear_attn.in_proj_z.weight"),
                (s + "ssm_beta_proj.weight", h + "linear_attn.in_proj_b.weight"),
                (s + "ssm_alpha_proj.weight", h + "linear_attn.in_proj_a.weight"),
                (s + "ssm_out_proj.weight", h + "linear_attn.out_proj.weight"),
            ]
        else:
            pairs += [
                (b + f"self_attn.{p}.weight", h + f"self_attn.{p}.weight")
                for p in ("q_proj", "k_proj", "v_proj", "o_proj")
            ]
        pairs += [
            (b + f"mlp.{p}.weight", h + f"mlp.{p}.weight")
            for p in ("gate_proj", "up_proj", "down_proj")
        ]
    worst = 1.0
    with torch.no_grad():
        for bn, hn in pairs:
            p = params[hn]
            R, K = p.shape
            w = q.dequant(bn, R, K)
            if bn.endswith("self_attn.q_proj.weight"):
                # the bundle's 8 queries, then 8 gates -> HF's per head [q | gate]
                w = w.reshape(2, HEADS, DH, K).transpose(1, 0, 2, 3).reshape(R, K)
            c = _cos(w, p.float().numpy())
            assert c > 0.99, (bn, c)
            worst = min(worst, c)
            p.copy_(torch.from_numpy(np.ascontiguousarray(w)).to(p.dtype))
        emb = bb.get_input_embeddings().weight
        e = q.embed().astype(np.float32)
        assert _cos(e, emb.float().numpy()) > 0.9999
        emb.copy_(torch.from_numpy(e).to(emb.dtype))
    return dict(
        package=f"{W3.LITE_REPO}@{W3.LITE_REVISION}",
        bundle_sha256=sha256(q.path),
        weights="bundle Q4K/Q8 exact dequant -> bf16; embedding = lm_head Q8",
        min_weight_cosine=worst,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("edge", "lite"), default="edge")
    ap.add_argument("--requests", default=str(HERE / "requests.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    import torch
    import transformers

    torch.set_num_threads(args.threads)
    # lite: the runtime loads the published BF16 weights, then lite_weights
    # replaces them
    root = W3.model_dir(args.model, weights=True)
    sys.path.insert(0, str(root))
    from d3_runtime import D3, kernel_report

    rt = D3.from_pretrained(str(root), device="cpu", batch_size=1, verify="full")
    assert rt.device.type == "cpu" and rt.fast is None and rt.encoder is not None
    repo, rev = W3.MODELS[args.model]
    identity = dict(repo=repo, revision=rev)
    if args.model == "lite":
        identity.update(lite_weights(rt))
    doc = json.loads(Path(args.requests).read_text())
    out = []
    t0 = time.time()
    for req in doc["requests"]:
        prepared = rt.prepare(req["state"], req["questions"])
        assert not prepared.errors, prepared.errors
        probs = {}
        for key in prepared.runnable:
            seq = prepared.sequences[key]
            count = len(prepared.questions[key].keys)
            logits = rt.logits([seq], [count])[0, :count]
            p = torch.softmax(logits / rt.temperature, dim=-1)
            probs[key] = p.tolist()
            out.append(
                dict(
                    request=req["id"],
                    question=key,
                    type=req["questions"][key]["type"],
                    bucket=req["bucket"],
                    n_tokens=len(seq),
                    input_ids=seq,
                    option_keys=prepared.questions[key].keys,
                    logits=logits.tolist(),
                    probabilities=p.tolist(),
                    answer=int(torch.argmax(p)),
                )
            )
            print(
                f"{req['id']}/{key} n={len(seq)} t={time.time() - t0:.0f}s", flush=True
            )
        # the public answers of the request, from the same probabilities
        out_req = rt.respond(prepared, probs, 0)
        for rec in out:
            if rec["request"] == req["id"]:
                rec["system_one_answer"] = out_req["answers"][rec["question"]]
    identity.update(
        weights_sha256=sha256(root / "model-00001-of-00001.safetensors"),
        readout_sha256=sha256(root / "readout.safetensors"),
        runtime_sha256=sha256(root / "d3_runtime.py"),
        format_sha256=sha256(root / "d3_format.py"),
        requests_sha256=requests_sha256(args.requests),
        transformers=transformers.__version__,
        torch=torch.__version__,
        python=platform.python_version(),
        device="cpu",
        dtype="bfloat16 backbone, float32 readout",
        batch_size=1,
        attention_mode=rt.attention_mode,
        temperature=rt.temperature,
        kernels=kernel_report(),
        seconds=time.time() - t0,
    )
    Path(args.out).write_text(json.dumps(dict(identity=identity, questions=out)))
    print(json.dumps(identity, indent=1))


if __name__ == "__main__":
    main()
