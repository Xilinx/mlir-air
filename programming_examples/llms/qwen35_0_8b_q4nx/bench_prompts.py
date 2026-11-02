# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The timing prompt of a bucket: the gate prompts' text repeated, its ids cut
at exactly P tokens (bench.py, and reference.py --timing-prompts)."""

from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "verify" / "prompts" / "instruct.txt"


def prompt_ids(tok, P):
    lines = [
        l.strip()
        for l in PROMPTS.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    text = " ".join(lines)
    ids = []
    while len(ids) < P:
        text = text + " " + text
        ids = tok(text)["input_ids"]
    return ids[:P]
