# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Pack the drafter's device weights from the pinned checkpoints. The target
packs its own from model.q4nx when it loads."""

import argparse
import json
from pathlib import Path

import paths  # noqa: F401
from common.build_identity import digest
from draft_weights import DraftWeights, packed_cache
from model_paths import TARGET_MODEL, default_draft_cache, resolve_target_model
from weights_b8 import gw


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--offline", action="store_true")
    a = p.parse_args()
    qm = gw.Q4nxModel(resolve_target_model(TARGET_MODEL, offline=a.offline))
    cache = packed_cache(default_draft_cache(), DraftWeights(target_model=qm))
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(
        json.dumps(
            dict(
                target=str(qm.path),
                draft_cache=str(cache),
                draft_identity=json.loads(cache.with_suffix(".json").read_text()),
                draft_cache_sha256=digest(cache),
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"drafter weights in {cache}")


if __name__ == "__main__":
    main()
