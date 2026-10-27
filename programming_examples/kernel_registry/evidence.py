# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Opt-in, evidence-bound kernel records; legacy registry lookup is unchanged.

Records are local proposals until reviewed. File hashes bind evidence, not trust:
the contributor/reviewer still establishes that the reported experiment occurred.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

KEY_FIELDS = {
    "operation",
    "shape",
    "input_dtype",
    "output_dtype",
    "compute",
    "precision",
    "layout",
    "target",
    "toolchain",
    "timing_boundary",
    "rtol",
    "atol",
}
STATES = {"PASS", "FAIL", "UNKNOWN", "NOT_RUN"}


def validate_key(key):
    if not isinstance(key, dict) or set(key) != KEY_FIELDS:
        raise ValueError(f"key must contain exactly {sorted(KEY_FIELDS)}")
    for name in KEY_FIELDS - {"shape", "rtol", "atol"}:
        if not isinstance(key[name], str) or not key[name].strip():
            raise ValueError("empty key field: " + name)
    if (
        not isinstance(key["shape"], list)
        or not key["shape"]
        or any(type(n) is not int or n <= 0 for n in key["shape"])
    ):
        raise ValueError("shape must contain positive integer dimensions")
    for name in ("rtol", "atol"):
        if (
            type(key[name]) not in (int, float)
            or not math.isfinite(key[name])
            or key[name] < 0
        ):
            raise ValueError("key needs finite nonnegative " + name)


def evidence_file(root, item):
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError("evidence file needs path and sha256")
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence path must be relative without parent traversal")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("evidence file is missing or escapes the evidence root")
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("evidence hash mismatch: " + item["path"])
    return path


def validate_record(record, root):
    """Validate metadata, hashes and report consistency, including failed records."""
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if (
        record.get("schema_version") != 1
        or not isinstance(record.get("id"), str)
        or not record["id"].strip()
    ):
        raise ValueError("record needs schema_version=1 and a nonempty id")
    validate_key(record["key"])
    if record.get("kind") not in ("primitive", "megakernel") or not record.get("abi"):
        raise ValueError("record needs primitive/megakernel kind and ABI description")
    if record["kind"] == "megakernel" and not record.get("graph"):
        raise ValueError("megakernel record needs its fused graph")
    gate = record["validation"]
    if gate.get("status") not in STATES:
        raise ValueError("invalid validation status")
    files = record["files"]
    for name in ("source", "artifact", "reference", "validation"):
        if name not in files:
            raise ValueError("missing evidence binding: " + name)
    paths = {name: evidence_file(root, value) for name, value in files.items()}
    if json.loads(paths["validation"].read_text(encoding="utf-8")) != gate:
        raise ValueError("validation metadata differs from saved report")
    bindings = {
        name: files[name]["sha256"] for name in ("source", "artifact", "reference")
    }
    if gate.get("bindings") != bindings:
        raise ValueError(
            "validation report does not bind this source/artifact/reference"
        )
    if gate["status"] == "PASS":
        for name in ("rtol", "atol"):
            value = gate.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("PASS needs finite nonnegative " + name)
            if value != record["key"][name]:
                raise ValueError("validation tolerance differs from query contract")
        shape = record.get("output_shape")
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            raise ValueError("PASS needs its complete output_shape")
        if (
            type(gate.get("compared_elements")) is not int
            or gate["compared_elements"] <= 0
            or gate["compared_elements"] != math.prod(shape)
            or type(gate.get("mismatch_count")) is not int
            or gate["mismatch_count"] != 0
            or gate.get("finite") is not True
        ):
            raise ValueError(
                "PASS needs complete finite elementwise comparison with zero mismatches"
            )
    timing = record.get("timing")
    if timing is not None:
        if (
            "timing" not in paths
            or json.loads(paths["timing"].read_text(encoding="utf-8")) != timing
        ):
            raise ValueError("timing metadata differs from saved report")
        if timing.get("bindings") != bindings:
            raise ValueError(
                "timing report does not bind this source/artifact/reference"
            )
        samples = timing.get("samples_seconds")
        if (
            timing.get("boundary") != record["key"]["timing_boundary"]
            or not isinstance(samples, list)
            or not samples
            or any(
                type(s) not in (int, float) or not math.isfinite(s) or s <= 0
                for s in samples
            )
        ):
            raise ValueError(
                "timing requires positive finite samples at the queried boundary"
            )
    return record


def select_verified(records, key, root):
    """Exact contract match; select the lowest mean latency among evidenced PASSes.

    This strict API never falls back to legacy mean-error/timing-only entries.
    Returns None when no measured verified match exists.
    """
    validate_key(key)
    candidates = []
    ids = set()
    for record in records:
        validate_record(record, root)
        if record["id"] in ids:
            raise ValueError("duplicate record id: " + record["id"])
        ids.add(record["id"])
        if (
            record["key"] == key
            and record["validation"]["status"] == "PASS"
            and record.get("timing")
        ):
            candidates.append(record)
    if not candidates:
        return None
    return copy.deepcopy(
        min(
            candidates,
            key=lambda r: (statistics.mean(r["timing"]["samples_seconds"]), r["id"]),
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("records", type=Path, help="JSON list of local records")
    p.add_argument("--root", type=Path, help="evidence root; default: records parent")
    p.add_argument("--query", type=Path, help="optional exact key JSON")
    a = p.parse_args()
    try:
        records = json.loads(a.records.read_text(encoding="utf-8"))
        root = a.root or a.records.parent
        if not isinstance(records, list):
            raise ValueError("records must be a JSON list")
        if a.query:
            selected = select_verified(
                records, json.loads(a.query.read_text(encoding="utf-8")), root
            )
            print(
                json.dumps(
                    dict(
                        status="VERIFIED_MATCH" if selected else "NO_VERIFIED_MATCH",
                        record=selected,
                    ),
                    indent=2,
                )
            )
        else:
            for record in records:
                validate_record(record, root)
            print(json.dumps(dict(status="RECORDS_CONSISTENT", records=len(records))))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        p.exit(1, f"Evidence check failed: {exc}\n")


if __name__ == "__main__":
    main()
