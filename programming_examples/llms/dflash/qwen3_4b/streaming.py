# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Incremental text output for committed Qwen tokens."""

from __future__ import annotations

import sys


class TextStream:
    """Decode the accumulated IDs so split UTF-8 characters survive BPE boundaries.

    Only a trailing replacement character is held back: a later token may
    complete its bytes. Disabling whitespace cleanup keeps printed prefixes
    stable. This helper never accepts or rejects draft tokens itself.
    """

    def __init__(self, tokenizer, output=None):
        self.tokenizer = tokenizer
        self.output = sys.stdout if output is None else output
        self.ids = []
        self.printed = ""

    def _write(self, *, final=False):
        text = self.tokenizer.decode(
            self.ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if not final:
            text = text.rstrip("\ufffd")
        if not text.startswith(self.printed):
            raise ValueError("Tokenizer changed an already streamed text prefix")
        delta = text[len(self.printed) :]
        if delta:
            self.output.write(delta)
            self.output.flush()
        self.printed = text

    def put(self, ids):
        self.ids.extend(int(token) for token in ids)
        self._write()

    def finish(self):
        self._write(final=True)
