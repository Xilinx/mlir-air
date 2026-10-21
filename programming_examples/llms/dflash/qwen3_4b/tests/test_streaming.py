# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Offline streaming checks, including UTF-8 split across committed rounds."""

import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming import TextStream


class ByteTokenizer:
    """Represent byte-level tokens without a downloaded tokenizer or NPU."""

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens and not clean_up_tokenization_spaces
        return bytes(token for token in ids if token != 256).decode(
            "utf-8", errors="replace"
        )


class FlushedOutput(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushed = []

    def flush(self):
        self.flushed.append(self.getvalue())


class StreamingTests(unittest.TestCase):
    def test_unicode_and_whitespace_across_round_boundaries(self):
        text = "Paris.  中文 café 🙂\n\tcode = 'λ'\n"
        ids = list(text.encode("utf-8"))
        for block_size in (1, 2, 3, 7, 8):
            with self.subTest(block_size=block_size):
                output = FlushedOutput()
                stream = TextStream(ByteTokenizer(), output)
                for start in range(0, len(ids), block_size):
                    stream.put(ids[start : start + block_size])
                    self.assertNotIn("\ufffd", output.getvalue())
                stream.put([256])  # EOS is part of the committed IDs, not text.
                stream.finish()
                self.assertEqual(output.getvalue(), text)
                self.assertEqual(stream.ids, ids + [256])
                self.assertGreater(len(output.flushed), 1)
                self.assertEqual(output.flushed[-1], text)

    def test_first_token_and_each_committed_chunk_are_flushed(self):
        output = FlushedOutput()
        stream = TextStream(ByteTokenizer(), output)
        stream.put([ord("A")])
        self.assertEqual(output.flushed, ["A"])
        stream.put([ord("B"), ord("C")])
        self.assertEqual(output.flushed, ["A", "ABC"])
        stream.finish()
        self.assertEqual(output.getvalue(), "ABC")

    def test_truncated_unicode_is_flushed_on_finish(self):
        output = FlushedOutput()
        stream = TextStream(ByteTokenizer(), output)
        stream.put([0xE4])
        self.assertEqual(output.getvalue(), "")
        stream.finish()
        self.assertEqual(output.getvalue(), "\ufffd")

    def test_empty_and_eos_only_output(self):
        for ids in ([], [256]):
            output = FlushedOutput()
            stream = TextStream(ByteTokenizer(), output)
            stream.put(ids)
            stream.finish()
            self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
