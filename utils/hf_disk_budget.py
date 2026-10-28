#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Size the HF weights the nightly runner caches, from programming_examples/llms/hf_models.txt.

The download patterns live here so the size this reports is what the
"Download LLM Weights" workflow fetches. Exits 1 when the total is over --budget-gb.
With --prune-cache, instead delete the local HF cache's model revisions the list
no longer names.
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, scan_cache_dir
from huggingface_hub.utils import filter_repo_objects

# Weight/config/tokenizer files. *.q4nx: FastFlowLM bundles. *.jinja: chat
# templates newer repos (LFM2) ship outside tokenizer_config.json.
ALLOW = [
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.q4nx",
    "*.jinja",
]
# Files no loader reads, which would otherwise cost disk or, failing to fetch on
# the runner's flaky link, mark the snapshot incomplete: Llama's original/ dir,
# licence blurbs, and Mistral's consolidated.safetensors (a copy of the shards).
IGNORE = ["original/*", "LICENSE*", "USE_POLICY*", "consolidated*.safetensors"]
# Repos whose pinned snapshot carries the HF implementation Transformers loads.
EXTRA_ALLOW = {
    "apple/OpenELM-1_1B-Instruct": ["configuration_openelm.py", "modeling_openelm.py"],
}

DEFAULT_LIST = (
    Path(__file__).resolve().parents[1] / "programming_examples/llms/hf_models.txt"
)


def allow_patterns(repo):
    return ALLOW + EXTRA_ALLOW.get(repo, [])


def read_models(path):
    entries = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            repo, _, revision = line.partition("@")
            entries.append((line, repo, revision or None))
    return entries


def prune_cache(entries, dry_run):
    """Keep each listed pin, and the revision `main` points at for an unpinned entry."""
    wanted = {}
    for _, repo, revision in entries:
        wanted.setdefault(repo, set()).add(revision)
    doomed = []
    for repo in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        pins = wanted.get(repo.repo_id)
        if pins is None:
            keep = set()
        else:
            keep = {r.commit_hash for r in repo.revisions if r.commit_hash in pins}
            if None in pins:
                main = {r.commit_hash for r in repo.revisions if "main" in r.refs}
                if not main:
                    print(f"skip {repo.repo_id}: no cached revision is `main`")
                    continue
                keep |= main
        for r in repo.revisions:
            if r.commit_hash not in keep:
                print(f"prune {repo.repo_id}@{r.commit_hash} ({r.size_on_disk_str})")
                doomed.append(r.commit_hash)
    if not doomed:
        print("nothing to prune")
        return 0
    strategy = scan_cache_dir().delete_revisions(*doomed)
    print(
        f"{'would free' if dry_run else 'freeing'} {strategy.expected_freed_size_str}"
    )
    if not dry_run:
        strategy.execute()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=DEFAULT_LIST)
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--prune-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.prune_cache:
        return prune_cache(read_models(args.models), args.dry_run)

    api = HfApi(token=False)
    seen = set()  # (repo, blob id): revisions of one repo share blobs in the cache
    total = 0
    rows = []
    for entry, repo, revision in read_models(args.models):
        info = api.model_info(repo, revision=revision, files_metadata=True)
        files = list(
            filter_repo_objects(
                info.siblings,
                allow_patterns=allow_patterns(repo),
                ignore_patterns=IGNORE,
                key=lambda s: s.rfilename,
            )
        )
        new = 0
        for s in files:
            blob = s.lfs.sha256 if s.lfs else s.blob_id
            if (repo, blob) not in seen:
                seen.add((repo, blob))
                new += s.size or 0
        total += new
        rows.append((new, entry))

    for size, entry in sorted(rows, reverse=True):
        print(f"{size / 2**30:8.1f} GiB  {entry}")
    print(f"{total / 2**30:8.1f} GiB  TOTAL ({len(rows)} entries)")
    if args.budget_gb is not None:
        if total > args.budget_gb * 2**30:
            print(
                f"ERROR: HF weights need {total / 2**30:.1f} GiB, over the "
                f"{args.budget_gb:.0f} GiB budget for the nightly runner's disk. "
                "Drop an entry, or raise HF_BUDGET_GB in "
                ".github/workflows/checkLLMDiskBudget.yml if the runner has the room."
            )
            return 1
        print(f"OK: within the {args.budget_gb:.0f} GiB budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
