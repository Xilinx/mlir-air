# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""B8 full outputs and state gate, one model resident and bounded KV scratch.

Peer runs use separate processes and compare hashes covering every output word.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
from target_runtime import TargetDecoder
from draft_weights import DraftWeights
from model_paths import (
    default_draft_cache,
    default_draft_snapshot,
    resolve_target_model,
)
from prepare_inputs import Q4nxModel
from checks import sha, write


def digest(a):
    h = hashlib.sha256()
    a = np.asarray(a)
    if a.ndim < 2:
        h.update(np.ascontiguousarray(a).view(np.uint8))
    else:
        for row in a:
            h.update(np.ascontiguousarray(row).view(np.uint8))
    return h.hexdigest()


def decoder(artifact, role):
    if role == "target":
        return TargetDecoder(artifact, stream_weights=True)
    return TargetDecoder(
        artifact,
        packed_cache=default_draft_cache(),
        model_weights=DraftWeights(
            Q4nxModel(resolve_target_model()), snapshot=default_draft_snapshot()
        ),
        stream_weights=True,
    )


def main(a):
    assert not a.output.exists()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    r = dict(
        passed=False,
        status="loading",
        cases=[],
        source_sha256=sha(__file__),
        artifact_sha256=sha(a.artifact / "manifest.json"),
    )

    def save():
        write(a.output, r)

    save()
    try:
        print(json.dumps(dict(phase="load", artifact=str(a.artifact))), flush=True)
        m = decoder(a.artifact, a.role)
        assert m.batch == 8
        state = np.frombuffer(m.bo["kv"].map(), np.uint16).reshape(m.kv_shape)
        ids = np.array([47, 2698, 374, 279, 6864, 315, 9822, 30])
        for p in a.prefixes:
            checks = {}
            row = dict(prefix=p, checks=checks, call_hashes={})
            r["cases"].append(row)

            def seed(future=4):
                rng = np.random.default_rng(941 + p)
                slab = np.empty(m.kv_shape[1:], np.uint16)
                for layer in range(m.layers):
                    slab.fill(np.asarray(future, bfloat16).view(np.uint16))
                    slab[:, :p] = (
                        rng.normal(0, 0.25, slab[:, :p].shape)
                        .astype(bfloat16)
                        .view(np.uint16)
                    )
                    m.bo["kv"].write(
                        slab.reshape(-1).view(np.int16), layer * slab.nbytes
                    )
                m.bo["kv"].sync(m.TO)

            def call(tokens, kind, future=4):
                r.update(status="running", current=dict(prefix=p, kind=kind))
                save()
                seed(future)
                prefix_before = digest(state[:, :, :p])
                tail_before = digest(state[:, :, p + 8 :])
                for name in ("x", "y"):
                    m.bo[name].write(
                        np.full(m.n[name], 0x7FC1, np.uint16).view(np.int16), 0
                    )
                    m.bo[name].sync(m.TO)
                v = m.verify(tokens, p)
                hidden = m.hidden()
                m.bo["kv"].sync(m.FROM)
                assert (
                    np.isfinite(v.logits).all()
                    and np.isfinite(hidden.view(bfloat16).astype(np.float32)).all()
                )
                checks[kind + "_prefix_unchanged"] = (
                    digest(state[:, :, :p]) == prefix_before
                )
                checks[kind + "_tail_unchanged"] = (
                    digest(state[:, :, p + 8 :]) == tail_before
                )
                hashes = [digest(v.logits), digest(hidden), digest(state)]
                row["call_hashes"][kind] = hashes
                return v.logits, hidden, hashes, digest(state[:, :, : p + 8])

            ref = call(ids, "first")
            for rep in range(2):
                other = call(ids, f"repeat{rep}")
                checks[f"repeat{rep}_exact"] = ref[2] == other[2]
            other = call(ids, "future", -8)
            checks["future_invisible"] = (
                ref[2][:2] == other[2][:2] and ref[3] == other[3]
            )
            altered = ids.copy()
            altered[-1] = 13
            other = call(altered, "last_row")
            if a.role == "target":
                checks["causal_rows0to6_exact"] = np.array_equal(
                    other[0][:7], ref[0][:7]
                ) and np.array_equal(other[1][:, :7], ref[1][:, :7])
            else:
                checks["row0_sees_row7"] = not np.array_equal(
                    other[1][1, 0], ref[1][1, 0]
                )
                rng = np.random.default_rng(20261008 + p)
                keys = (
                    rng.normal(0, 0.2, (m.layers, 8, 1024))
                    .astype(bfloat16)
                    .view(np.uint16)
                )
                vals = (
                    rng.normal(0, 0.2, (m.layers, 8, 1024))
                    .astype(bfloat16)
                    .view(np.uint16)
                )
                start = max(0, p - 8)
                seed()
                before = (digest(state[:, :, :start]), digest(state[:, :, start + 8 :]))
                m.append_context_kv(keys, vals, start)
                m.bo["kv"].sync(m.FROM)
                checks["injection_other_words_unchanged"] = before == (
                    digest(state[:, :, :start]),
                    digest(state[:, :, start + 8 :]),
                )
                checks["injected_KV_exact"] = all(
                    np.array_equal(
                        state[:, off : off + 2, start : start + 8],
                        x.reshape(m.layers, 8, 2, 512).transpose(0, 2, 1, 3),
                    )
                    for off, x in ((0, keys), (2, vals))
                )
            row.update(
                checks={k: bool(v) for k, v in checks.items()},
                logits_elements=ref[0].size,
                hidden_elements=ref[1].size,
                kv_words=state.size,
                output_hashes=ref[2],
            )
            save()
            assert all(checks.values()), row
            print(
                json.dumps(dict(prefix=p, passed=True, checks=len(checks))), flush=True
            )
        r.update(passed=True, status="passed")
        save()
    except BaseException as e:
        r.update(status="failed", error=repr(e))
        save()
        print(repr(e), flush=True)
        os._exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("artifact", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--role", choices=("target", "drafter"), required=True)
    p.add_argument(
        "--prefixes", type=int, nargs="+", default=[0, 7, 16, 31, 64, 112, 1024, 4088]
    )
    main(p.parse_args())
