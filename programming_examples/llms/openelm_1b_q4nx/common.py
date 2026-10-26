# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Fixed task geometry and shared, production-independent capture utilities."""

from dataclasses import dataclass
from pathlib import Path
import os
import hashlib
import json
import sys
import time

import numpy as np
from ml_dtypes import bfloat16

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for path in (HERE.parent / "llama32_1b", HERE.parent / "verify"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from air_examples.llms.verify.comparators import topk_token_ids

RUN = Path(os.environ.get("AIR_MODEL_BUILD_DIR", str(HERE / "build")))
REVISION = "effd796da2a77361d7360e45e54c7cc14bc5df2a"
LAYERS = [
    (16, 4, 1024),
    (16, 4, 1280),
    (16, 4, 1536),
    (20, 5, 1792),
    (20, 5, 2048),
    (20, 5, 2304),
    (20, 5, 2560),
    (20, 5, 2816),
    (20, 5, 3072),
    (20, 5, 3328),
    (24, 6, 3584),
    (24, 6, 3840),
    (24, 6, 4096),
    (24, 6, 4608),
    (24, 6, 4608),
    (24, 6, 5120),
    (24, 6, 5376),
    (24, 6, 5632),
    (28, 7, 5888),
    (28, 7, 6144),
    (28, 7, 6400),
    (28, 7, 6656),
    (28, 7, 6912),
    (28, 7, 7168),
    (32, 8, 7424),
    (32, 8, 7680),
    (32, 8, 7936),
    (32, 8, 8192),
]


def read_task():
    from air_examples.llms.shared.deployment import read_task

    return read_task("openelm_1b_q4nx")


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)
