# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""DFlash token captures compared with llms/verify's top-5 comparator.

Run NPU and HF captures in separate processes, then compare 32-token traces.
With --inputs, decode follows the saved prefill's first token. Otherwise use
NPU prefill and capture the full DFlash path on two short prompts.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
VERIFY = HERE.parents[1] / "verify"
sys.path.insert(0, str(VERIFY))
from comparators import compute_topk_set_check, topk_token_ids
from verify_runner import _generate_with_topk
from model_paths import TOKENIZER_MODEL, TOKENIZER_REVISION
from prepare_inputs import load_prefill
from provenance import runtime_identity
from runtime import load_pair, tokenize_prompt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(a):
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    r = dict(
        runtime=runtime_identity(),
        status="running",
        phase=a.phase,
        self_validated=True,
        source_sha256=sha(__file__),
        comparator_sha256=sha(VERIFY / "comparators.py"),
        requested_tokens=32,
        k=5,
        stop_on_eos=False,
        captures=[],
        comparisons=[],
        mode=(
            "saved-prefill decode after first token"
            if a.inputs
            else "NPU prefill + DFlash decode"
        ),
    )

    def save():
        a.output.write_text(json.dumps(r, indent=2) + "\n")

    save()
    try:
        if a.inputs:
            all_cases = json.loads((a.inputs / "cases.json").read_text())["cases"]
            cases = [all_cases[i] for i in (0, 16)]
        else:
            prompts = [
                s.strip()
                for s in (VERIFY / "prompts/instruct.txt").read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")
            ][:2]
            assert len(prompts) == 2
            cases = [
                dict(id=f"prompt-{i}", prompt=text) for i, text in enumerate(prompts)
            ]
        if a.phase == "hf":
            from resources import prepare
            from runners.hf_runner import HfRunner

            # Only the pinned files are cached. A bare snapshot_download asks
            # for the whole repo and fails offline.
            (model,) = (
                r["snapshot"]
                for r in prepare(reference=True, offline=True)
                if r["repo"] == TOKENIZER_MODEL
            )
            runner = HfRunner(
                model,
                SimpleNamespace(n_layers=36, emb_dim=2560, n_kv_heads=8, head_dim=128),
                4096,
                lite_mode=True,
            )
            r["reference_model"] = dict(
                repo=TOKENIZER_MODEL, revision=TOKENIZER_REVISION
            )
        elif a.phase == "npu":
            if a.inputs:
                loop = load_pair(a.target, a.draft, prepass_artifacts=a.prepass)
            else:
                from runtime import load_pipeline

                loop, engine = load_pipeline(
                    a.target,
                    a.draft,
                    prepass_artifacts=a.prepass,
                    prefill_artifacts=a.prefill_artifacts,
                )
            r["manifests"] = {
                k: sha(getattr(a, k) / "manifest.json")
                for k in ("target", "draft", "prepass")
            }
            loop.prepass.warmup()
            original = loop.target.verify
            latest = {}

            def capture_call(*args, **kwargs):
                result = original(*args, **kwargs)
                latest["logits"] = result.logits
                return result

            loop.target.verify = capture_call
        else:
            x = json.loads(a.npu_capture.read_text())
            y = json.loads(a.hf_capture.read_text())
            assert x["status"] == y["status"] == "captured"
            assert x["mode"] == y["mode"] == r["mode"]
            assert (
                x["comparator_sha256"]
                == y["comparator_sha256"]
                == r["comparator_sha256"]
            )
            r["capture_sha256"] = dict(npu=sha(a.npu_capture), hf=sha(a.hf_capture))
        for number, case in enumerate(cases):
            if a.inputs:
                pf = load_prefill(a.inputs, case)
                ids = pf["input_ids"] + [int(pf["first"])]
            else:
                _, ids = tokenize_prompt(case["prompt"])
            if a.phase == "compare":
                n, h = x["captures"][number], y["captures"][number]
                assert n["input_ids"] == h["input_ids"] == ids
                assert all(
                    len(t) == 32
                    for t in (n["chosen"], n["topk"], h["chosen"], h["topk"])
                )
                assert all(len(t) == 5 for t in n["topk"] + h["topk"])
                verdict = compute_topk_set_check(
                    test_chosen=n["chosen"],
                    test_topk=n["topk"],
                    ref_chosen=h["chosen"],
                    ref_topk=h["topk"],
                    k=5,
                    prompt_idx=number,
                    prompt_text=case["id"],
                )
                divergence = next(
                    (
                        i
                        for i, (u, v) in enumerate(zip(n["chosen"], h["chosen"]))
                        if u != v
                    ),
                    None,
                )
                r["comparisons"].append(
                    dict(
                        **asdict(verdict),
                        first_divergence=divergence,
                        visited_positions=32 if divergence is None else divergence + 1,
                    )
                )
            else:
                if a.phase == "hf":
                    chosen, topk = _generate_with_topk(
                        runner, np.asarray(ids, np.int64), 32, 5
                    )
                else:
                    if not a.inputs:
                        pf, _ = engine.prepare(ids)
                    loop.reset(pf)
                    topk = [] if a.inputs else [topk_token_ids(pf["logits"], 5)]

                    def observe(row, active):
                        topk.extend(
                            topk_token_ids(latest["logits"][i], 5)
                            for i in range(row["emitted"])
                        )

                    generated = loop.generate(
                        33 if a.inputs else 32, eos_ids=(), observer=observe
                    )
                    chosen = (
                        generated.output_ids[1:] if a.inputs else generated.output_ids
                    )
                assert len(chosen) == len(topk) == 32 and all(len(t) == 5 for t in topk)
                assert all(int(c) == int(t[0]) for c, t in zip(chosen, topk))
                r["captures"].append(
                    dict(
                        id=case["id"],
                        input_ids=ids,
                        chosen=list(map(int, chosen)),
                        topk=[list(map(int, t)) for t in topk],
                    )
                )
            save()
            print(
                json.dumps(dict(phase=a.phase, id=case["id"], complete=True)),
                flush=True,
            )
        if a.phase == "compare":
            r["passed"] = all(c["status"] == "OK" for c in r["comparisons"])
            r["status"] = "passed" if r["passed"] else "failed"
        else:
            r["status"] = "captured"
        save()
        assert r.get("passed", True), r["comparisons"]
    except BaseException as error:
        r.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("npu", "hf", "compare"), required=True)
    p.add_argument("--output", type=Path, required=True)
    for k in (
        "inputs",
        "target",
        "draft",
        "prepass",
        "prefill-artifacts",
        "npu-capture",
        "hf-capture",
    ):
        p.add_argument("--" + k, type=Path)
    main(p.parse_args())
