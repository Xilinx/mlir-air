# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fetch and verify the fixed 80-prompt benchmark bundle (no NPU needed)."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_inputs(inputs):
    inputs = Path(inputs)
    expected = json.loads((HERE / "benchmark/cases.json").read_text())
    actual = json.loads((inputs / "cases.json").read_text())
    if actual != expected:
        raise ValueError("Benchmark manifest differs from the frozen reviewer cohort")
    for case in actual["cases"]:
        path = inputs / case["cache_file"]
        if sha(path) != case["cache_sha256"]:
            raise ValueError(f"Prefill checksum mismatch: {case['id']}")
    return actual


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--destination", type=Path, default=HERE / "build/benchmark_bundle")
    ap.add_argument("--archive", type=Path, help="Use a previously downloaded archive")
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Require the bundle or archive to be cached",
    )
    ap.add_argument(
        "--verify",
        type=Path,
        metavar="INPUTS",
        help="Only verify an extracted inputs directory",
    )
    args = ap.parse_args()
    if args.verify:
        verify_inputs(args.verify)
        print("Verified all 80 frozen prompt states")
        return
    dest = args.destination.resolve()
    if dest.exists():
        verify_inputs(dest / "inputs")
        bundle = json.loads((dest / "bundle.json").read_text())
        for name, identity in bundle["files"].items():
            if sha(dest / name) != identity["sha256"]:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        print(f"Already verified: {dest}")
        return
    asset = json.loads((HERE / "benchmark/download.json").read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    archive = args.archive or dest.parent / asset["archive"]
    if not archive.exists():
        if args.archive or args.offline:
            raise FileNotFoundError(
                f"Missing {archive}; run make download once with network access"
            )
        temporary = archive.with_suffix(archive.suffix + ".partial")
        print(
            f"Downloading {asset['bytes'] / 1e6:.1f} MB from {asset['url']}", flush=True
        )
        urllib.request.urlretrieve(asset["url"], temporary)
        temporary.rename(archive)
    if sha(archive) != asset["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    with tempfile.TemporaryDirectory(prefix="dflash-extract-", dir=dest.parent) as temp:
        unpacked = Path(temp) / "bundle"
        unpacked.mkdir()
        with tarfile.open(archive) as packed:
            packed.extractall(unpacked, filter="data")
        verify_inputs(unpacked / "inputs")
        bundle = json.loads((unpacked / "bundle.json").read_text())
        for name, identity in bundle["files"].items():
            if sha(unpacked / name) != identity["sha256"]:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        shutil.move(unpacked, dest)
    print(f"Verified benchmark bundle: {dest}")


if __name__ == "__main__":
    main()
