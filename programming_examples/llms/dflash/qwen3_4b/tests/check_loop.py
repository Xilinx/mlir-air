# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Actual B8 DFlash loop against its matching-arithmetic sequential oracle."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
from pathlib import Path
import numpy as np
from runtime import load_pair
from prepare_inputs import load_prefill
from checks import sha, write
from speculative import accept_greedy, autoregressive
from verify_loop import signature


def host_gate(batch=8):
    checked = 0
    for accepted in range(batch):
        block = [100] + list(range(1, batch))
        prediction = list(range(1, batch + 1))
        if accepted < batch - 1:
            block[accepted + 1] = 99
        sequence = list(range(1, accepted + 2))
        for remaining in range(batch + 2):
            for eos in (None, *sequence):
                d = accept_greedy(
                    block, prediction, remaining, () if eos is None else (eos,)
                )
                expected = sequence[:remaining]
                stopped = eos in expected if eos is not None else False
                if stopped:
                    expected = expected[: expected.index(eos) + 1]
                assert (d.accepted_drafts, d.tokens, d.stopped_on_eos) == (
                    accepted,
                    expected,
                    stopped,
                )
                checked += 1
    return checked


def main(a):
    a.output.parent.mkdir(parents=True, exist_ok=True)
    assert not a.output.exists()
    r = dict(
        status="loading",
        passed=False,
        source_sha256=sha(__file__),
        host_cases=host_gate(batch=8),
        target_manifest_sha256=sha(a.target / "manifest.json"),
        draft_manifest_sha256=sha(a.draft / "manifest.json"),
        prepass_manifest_sha256=sha(a.prepass / "manifest.json"),
        inputs_sha256=sha(a.inputs / "cases.json"),
    )

    def save():
        write(a.output, r)

    save()
    try:
        case = json.loads((a.inputs / "cases.json").read_text())["cases"][a.case_index]
        prefill = load_prefill(a.inputs, case)
        r.update(
            case_id=case["id"],
            prefill_sha256=sha(a.inputs / case["cache_file"]),
            tokens_requested=a.tokens,
        )
        loop = load_pair(a.target, a.draft, prepass_artifacts=a.prepass)
        assert loop.batch == 8
        loop.prepass.warmup()
        ck = {}
        r["checks"] = ck
        r["status"] = "oracle"
        save()
        oracle = autoregressive(loop.target, prefill, a.tokens, eos_ids=())
        committed = len(prefill["input_ids"]) + len(oracle.output_ids) - 1

        def committed_state():
            # Preserve the same committed-word check without copying the
            # unused C4096 tail into a temporary full-cache host array.
            loop.target.bo["kv"].sync(loop.target.FROM)
            state = np.frombuffer(loop.target.bo["kv"].map(), np.uint16).reshape(
                loop.target.kv_shape
            )
            return state[:, :, :committed].copy()

        oracle_kv = committed_state()
        r["status"] = "dflash"
        save()
        runs = []
        states = []
        round_checks = []

        def observe(row, active):
            end = row["position"] + row["emitted"]
            active.target.bo["kv"].sync(active.target.FROM)
            state = np.frombuffer(active.target.bo["kv"].map(), np.uint16).reshape(
                active.target.kv_shape
            )
            exact = bool(np.array_equal(state[:, :, :end], oracle_kv[:, :, :end]))
            round_checks.append(
                dict(position=row["position"], committed_end=end, exact=exact)
            )
            assert exact, round_checks[-1]

        for _ in range(2):
            loop.reset(prefill)
            g = loop.generate(a.tokens, eos_ids=(), observer=observe)
            runs.append(g)
            states.append(committed_state())
        first, second = runs
        ck.update(
            same_arithmetic_AR_exact=first.output_ids == oracle.output_ids,
            repeats_exact=first.output_ids == second.output_ids,
            decisions_exact=signature(first) == signature(second),
            committed_KV_exact=all(np.array_equal(s, oracle_kv) for s in states),
            every_round_committed_KV_exact=all(x["exact"] for x in round_checks),
        )
        r.update(
            checks=ck,
            output_ids=first.output_ids,
            oracle_ids=oracle.output_ids,
            rounds=first.rounds,
            rounds_checked=len(first.rounds),
            round_kv_checks=round_checks,
            scope="Frozen saved-prefill decode; no EOS stopping; matching-arithmetic target oracle, repeat decisions, committed KV; host EOS/clipping covered exhaustively",
        )
        r.update(
            passed=all(ck.values()), status="passed" if all(ck.values()) else "failed"
        )
        save()
        assert r["passed"], ck
        print(
            json.dumps(
                dict(passed=True, id=case["id"], rounds=len(first.rounds), checks=ck)
            ),
            flush=True,
        )
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        save()
        print(repr(e), flush=True)
        os._exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("target", "draft", "prepass", "inputs", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--case-index", type=int, default=0)
    p.add_argument("--tokens", type=int, default=32)
    main(p.parse_args())
