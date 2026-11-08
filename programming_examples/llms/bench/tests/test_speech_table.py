# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Speech-to-text models: their profile's real-time factor reaches the record
and the history, and they render in their own table rather than as LLM
rows."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
EXAMPLES = BENCH.parents[1]


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(EXAMPLES / "generate_readme.py")
append_history = _load(BENCH / "append_history.py")

PROFILE = """\
24 requests (3 rounds of 8 clips), warm-up 8.7 s, measured 23.3 s
Audio: 10.3 s, 33.1 tokens per request
Front end (log-mel, convs): 11.7 ms
Encoder: 469.5 ms
Time to first token (TTFT): 0.5312 s
Decode: 13.61 ms per token (73.48 tok/s)
Request: 972.8 ms
Real-time factor: 0.0901
"""


def _extract(text, model):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    out = subprocess.run(
        [sys.executable, str(BENCH / "extract_perf.py"), f.name, "--model", model],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    Path(f.name).unlink()
    return json.loads(out)


def _row(model, ttft, decode, rtf=None):
    metrics = {"ttft_ms": ttft, "decode_tokens_per_sec": decode, "context_len": None}
    if rtf is not None:
        metrics["real_time_factor"] = rtf
    return {
        "model": model,
        "timestamp_utc": "2026-11-08T04:00:00Z",
        "verify_status": "pass",
        "metrics": metrics,
    }


def _page(recs):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(recs, f)
    page = generate_readme.render_llm_benchmark(f.name)
    Path(f.name).unlink()
    return page


class ExtractTest(unittest.TestCase):
    def test_speech_profile(self):
        m = _extract(PROFILE, "whisper_v3_turbo_q4nx")["metrics"]
        self.assertEqual(m["ttft_ms"], 531.2)
        self.assertEqual(m["decode_tokens_per_sec"], 73.48)
        self.assertEqual(m["real_time_factor"], 0.0901)

    def test_llm_profile_has_no_real_time_factor(self):
        text = "Time to first token (TTFT): 0.25s\nDecode: (30.0 tok/s)\n"
        self.assertNotIn("real_time_factor", _extract(text, "x")["metrics"])


class HistoryTest(unittest.TestCase):
    def test_real_time_factor_round_trips(self):
        rec = _row("whisper_v3_turbo_q4nx", 531.2, 73.48, 0.0901)
        (flat,) = append_history._flat_rows([rec], run_id=1)
        self.assertEqual(flat["real_time_factor"], 0.0901)
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            f.write(json.dumps(flat) + "\n")
        (back,) = generate_readme.load_llm_history(f.name)
        Path(f.name).unlink()
        self.assertEqual(back["metrics"]["real_time_factor"], 0.0901)


class TableTest(unittest.TestCase):
    def test_own_table_not_an_llm_row(self):
        page = _page(
            [
                _row("whisper_v3_turbo_q4nx", 531.2, 73.48, 0.0901),
                _row("llama32_1b_q4nx", 120.0, 60.0),
            ]
        )
        start = page.index("### Speech-to-text")
        speech = page[start:]
        self.assertIn("whisper-large-v3-turbo", speech)
        self.assertIn("| 531.2 | 73.48 | 0.0901 |", speech)
        self.assertNotIn("whisper", page[:start])

    def test_no_speech_rows_no_section(self):
        self.assertEqual(generate_readme.render_speech_benchmark([]), "")

    def test_flm_link_outside_automodel(self):
        cell = generate_readme._llm_model_cell("whisper_v3_turbo_q4nx", "")
        self.assertIn("/src/common/whisper/modeling_whisper.cpp", cell)
        self.assertIn("huggingface.co/openai/whisper-large-v3-turbo", cell)


if __name__ == "__main__":
    unittest.main()
