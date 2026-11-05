#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Size the HF weights the nightly runner caches, from programming_examples/llms/hf_models.txt.

The download patterns live here so the size this reports is what the
"Download LLM Weights" workflow fetches. Exits 1 when the total is over --budget-gb.
With --prune-cache, instead delete the local HF cache's model revisions the list
no longer names, and the files IGNORE names in the revisions it keeps.
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
# licence blurbs, Mistral's consolidated.safetensors (a copy of the shards), and
# the audio and vision towers FastFlowLM bundles carry for FastFlowLM's own
# runtime (the examples read model.q4nx).
IGNORE = [
    "original/*",
    "LICENSE*",
    "USE_POLICY*",
    "consolidated*.safetensors",
    "audio_weight.q4nx",
    "vision_weight.q4nx",
]
# Repos whose pinned snapshot carries the HF implementation Transformers loads;
# the example's offline snapshot_download asks for every *.py of it.
EXTRA_ALLOW = {
    "apple/OpenELM-1_1B-Instruct": ["*.py"],
    # the d3 package's runtime, which its CPU reference imports
    "vllm-sr/d3-edge": ["*.py"],
    "vllm-sr/d3-lite": ["*.py"],
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
    """Keep each listed pin, and the revision `main` points at for an unpinned
    entry; in those, delete the files IGNORE names."""
    wanted = {}
    for _, repo, revision in entries:
        wanted.setdefault(repo, set()).add(revision)
    doomed, ignored = [], []
    for repo in sorted(scan_cache_dir().repos, key=lambda r: r.repo_id):
        if repo.repo_type != "model":
            continue
        pins = wanted.get(repo.repo_id)
        if pins is None:
            keep = set()
        else:
            # A pin is a commit hash, or a branch/tag name the cache records as a ref.
            keep = {
                r.commit_hash
                for r in repo.revisions
                if r.commit_hash in pins or pins & set(r.refs)
            }
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
        ignored += ignored_files(repo, keep)
    freed = prune_files(ignored, dry_run)
    if not doomed:
        print("no revision to prune")
    else:
        strategy = scan_cache_dir().delete_revisions(*doomed)
        print(
            f"{'would free' if dry_run else 'freeing'} "
            f"{strategy.expected_freed_size_str} of revisions"
        )
        if not dry_run:
            strategy.execute()
    print(f"{'would free' if dry_run else 'freed'} {freed / 2**30:.1f} GiB of files")
    return 0


def ignored_files(repo, keep):
    """(snapshot path, blob path, size) of the files IGNORE names in the
    revisions `keep` of cached `repo`, with whether the blob may go: no other
    file of the repo's kept revisions links to it."""
    out, used = [], {}
    for r in repo.revisions:
        if r.commit_hash not in keep:
            continue
        for f in r.files:
            rel = str(f.file_path.relative_to(r.snapshot_path))
            hit = any(filter_repo_objects([rel], allow_patterns=IGNORE))
            used.setdefault(f.blob_path, set()).add(hit)
            if hit:
                out.append(
                    (repo.repo_id, rel, f.file_path, f.blob_path, f.size_on_disk)
                )
    return [(*o, used[o[3]] == {True}) for o in out]


def prune_files(files, dry_run):
    """Delete the snapshot links in `files` (ignored_files) and the blobs no
    kept file links to; the bytes freed. A repo's blob may itself link into a
    content store the cache shares across repos; that content goes when no
    repo's blob links to it any more."""
    freed, stored = 0, []
    for repo, rel, link, blob, size, blob_goes in files:
        print(f"prune {repo}: {rel} ({size / 2**30:.2f} GiB)")
        if blob_goes:
            freed += size
        if dry_run:
            continue
        link.unlink(missing_ok=True)
        if blob_goes:
            if blob.is_symlink():
                stored.append(blob.resolve())
            blob.unlink(missing_ok=True)
    if stored:
        hub = Path(scan_cache_dir().cache_dir)
        linked = {b.resolve() for b in hub.glob("*/blobs/*") if b.is_symlink()}
        for content in stored:
            if content not in linked:
                content.unlink(missing_ok=True)
    return freed


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
