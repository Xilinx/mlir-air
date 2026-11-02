# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""System One answers of the d3 decision models on NPU2: the upstream
runtime's prompt format and tokenizer, one forward pass per question on the
fused prefill device (shared/fused_prefill/d3.py, D3Prefill), the 255-way
FP32 readout of the last prompt position and the option softmax on the
host.

    python3 d3_npu.py --build BUILD --model edge [--wform hilo]
"""

import argparse
import json
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
from air_examples.llms.shared.fused_prefill import d3  # noqa: E402


class D3Npu:
    """System One on the NPU, as d3_runtime.D3.system_one answers it."""

    def __init__(self, build_dir, model="edge", wform=None):
        from safetensors.numpy import load_file

        self.model = model
        self.pf = d3.D3Prefill(build_dir, model, wform)
        self.wform = self.pf.wform
        self.root = self.pf.root
        dc = json.loads((self.root / "decision_config.json").read_text())
        assert dc["attention_mode"] == "noncausal_full_attention", dc
        assert dc["pooling"] == "last", dc
        assert dc.get("readout_dtype", "float32") == "float32", dc
        self.temperature = float(dc.get("temperature", 1.0))
        r = load_file(str(self.root / "readout.safetensors"))["weight"]
        self.readout = np.ascontiguousarray(r, np.float32)  # [255, d]
        self.times = self.pf.times

    def logits(self, ids):
        """The 255 code logits (FP32) at the last position."""
        h = self.pf.forward(ids)
        t0 = time.perf_counter()
        z = self.readout @ h
        self.pf._t("host", t0)
        return z

    def probabilities(self, ids, count):
        return self.softmax(self.logits(ids)[:count])

    def softmax(self, z):
        """The option probabilities of the option logits z."""
        z = np.asarray(z, np.float64) / self.temperature
        e = np.exp(z - z.max())
        return (e / e.sum()).astype(np.float32)

    # ---- System One ----

    def runtime_helpers(self):
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        import d3_runtime

        return d3_runtime

    def prepare(self, state, questions):
        """(key, normalized question, input ids) of every question: the
        package's own prompt format and tokenizer."""
        R = self.runtime_helpers()
        from d3_format import answer_codes
        from transformers import AutoTokenizer

        if not hasattr(self, "tok"):
            self.tok = AutoTokenizer.from_pretrained(str(self.root))
            self.codes, _ = answer_codes(self.tok)
        out = []
        for key, q in questions.items():
            nq = R.normalize_question(q)
            text = R.render(self.tok, "d3", state, nq.rendered, self.codes)
            out.append((key, nq, self.tok(text, add_special_tokens=False)["input_ids"]))
        return out

    def system_one(self, state, questions):
        """Typed answers, one forward pass per question."""
        R = self.runtime_helpers()
        answers, tokens = {}, 0
        for key, nq, ids in self.prepare(state, questions):
            tokens += len(ids)
            p = self.probabilities(ids, len(nq.keys)).tolist()
            answers[key] = R.product_answer(nq, p)
        return {
            "model": W3.MODELS[self.model][0],
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


QUICKSTART = {
    "state": "The order arrived damaged and the customer wants a replacement.",
    "questions": {
        "route": {
            "type": "choice",
            "criteria": {
                "returns": "Refunds and replacements",
                "billing": "Payments",
                "general": None,
            },
        }
    },
}


def add_model_args(ap):
    ap.add_argument("--build", required=True)
    ap.add_argument("--model", choices=("edge", "lite"), default="edge")
    ap.add_argument(
        "--wform",
        choices=list(d3.WFORMS),
        default=None,
        help=f"edge's weight form (default {d3.DEFAULT_WFORM})",
    )


def main():
    ap = argparse.ArgumentParser()
    add_model_args(ap)
    a = ap.parse_args()
    m = D3Npu(a.build, a.model, a.wform)
    t0 = time.perf_counter()
    r = m.system_one(QUICKSTART["state"], QUICKSTART["questions"])
    print(json.dumps(r, indent=1))
    print(f"{time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
