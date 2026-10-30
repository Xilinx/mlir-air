# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Accuracy and task scores as columns of the LLM page's decode and prefill
tables (generate_readme.py)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[3]


def _load(path):
    # By file path: programming_examples/ itself must stay off sys.path.
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_readme = _load(EXAMPLES / "generate_readme.py")

KRACKAN = generate_readme.hardware_label("krackan-point")
HALO = generate_readme.hardware_label("strix-halo")
MODEL = "qwen3_4b_q4nx"


def sweep(model=MODEL, hardware="krackan-point"):
    return {
        "model": model,
        "hardware": hardware,
        "verify_status": "pass",
        "points": [{"context_len": 2048, "decode_tokens_per_sec": 19.0}],
    }


def prefill_sweep(model=MODEL, hardware="krackan-point"):
    return {
        "model": model,
        "hardware": hardware,
        "verify_status": "pass",
        "points": [{"prefill_len": 2048, "ttft_ms": 1800.0}],
    }


def accuracy(ref, path, kld, ctx=2048, hardware="krackan-point", day=1):
    return {
        "model": MODEL,
        "hardware": hardware,
        "timestamp_utc": f"2026-11-{day:02d}T00:00:00Z",
        "air_sha": "0123456789",
        "ref_kind": ref,
        "path": path,
        "ctx": ctx,
        "n_tokens": 4092,
        "kld_mean": kld,
        "kld_se": 0.001,
        "same_top": 0.9,
        "ln_ppl_ratio": 0.01,
        "ln_ppl_ratio_se": 0.001,
    }


def task(name, npu, ref, hardware="krackan-point"):
    return {
        "model": MODEL,
        "hardware": hardware,
        "timestamp_utc": "2026-11-01T00:00:00Z",
        "air_sha": "0123456789",
        "task": name,
        "n": 100,
        "npu": npu,
        "ref": ref,
        "npu_only_right": 2,
        "ref_only_right": 3,
        "mcnemar_p": 0.5,
    }


def load(acc_rows, task_rows=()):
    with tempfile.TemporaryDirectory() as tmp:
        acc, tasks = Path(tmp, "accuracy.ndjson"), Path(tmp, "tasks.ndjson")
        acc.write_text("".join(json.dumps(r) + "\n" for r in acc_rows))
        tasks.write_text("".join(json.dumps(r) + "\n" for r in task_rows))
        return generate_readme.load_llm_accuracy(acc, tasks)


def row(table, model=MODEL):
    """The cells of the table row whose Model cell links to `model`."""
    (line,) = [l for l in table.splitlines() if f"llms/{model}/" in l]
    return [c.strip() for c in line.strip("|").split(" | ")]


def text(cell):
    """A cell without its tooltip markup."""
    return generate_readme.re.sub(r"<[^>]+>", "", cell)


class AccuracyColumns(unittest.TestCase):
    def test_no_accuracy_leaves_the_tables_as_they_were(self):
        for render, recs in (
            (generate_readme.render_llm_sweep, [sweep()]),
            (generate_readme.render_llm_prefill_sweep, [prefill_sweep()]),
        ):
            self.assertEqual(render(recs), render(recs, accuracy={}))
            self.assertNotIn("KLD", render(recs))

    def test_decode_row_shows_same_weights_kld_and_tasks(self):
        acc = load(
            [accuracy("dequant", "decode", 0.03), accuracy("bf16", "decode", 0.08)],
            [task("gsm8k", 0.80, 0.75), task("mmlu", 0.72, 0.73)],
        )
        cells = row(generate_readme.render_llm_sweep([sweep()], accuracy=acc))
        self.assertEqual(
            [text(c) for c in cells[-4:-1]], ["0.0300", "80 / 75", "72 / 73"]
        )
        self.assertEqual(cells[1], KRACKAN)

    def test_prefill_row_shows_prefill_kld(self):
        acc = load([accuracy("dequant", "prefill", 0.0033)])
        table = generate_readme.render_llm_prefill_sweep(
            [prefill_sweep()], accuracy=acc
        )
        self.assertEqual(text(row(table)[-2]), "0.0033")

    def test_accuracy_from_other_hardware_is_named_on_the_row(self):
        # Speed timed on Krackan Point, accuracy measured on Strix Halo: one
        # row, which names both.
        acc = load([accuracy("dequant", "decode", 0.02, hardware="strix-halo")])
        cells = row(generate_readme.render_llm_sweep([sweep()], accuracy=acc))
        self.assertEqual(text(cells[-4]), "0.0200")
        self.assertIn(KRACKAN, cells[1])
        self.assertIn(f"accuracy: {HALO}", cells[1])

    def test_newest_measurement_wins_whatever_its_hardware(self):
        acc = load(
            [
                accuracy("dequant", "decode", 0.05, hardware="strix-halo", day=1),
                accuracy("dequant", "decode", 0.03, day=2),
            ]
        )
        cells = row(generate_readme.render_llm_sweep([sweep()], accuracy=acc))
        self.assertEqual(text(cells[-4]), "0.0300")
        self.assertEqual(cells[1], KRACKAN)

    def test_longer_contexts_go_in_the_tooltip(self):
        acc = load(
            [
                accuracy("dequant", "decode", 0.03),
                accuracy("dequant", "decode", 0.04, ctx=8192),
            ]
        )
        cell = row(generate_readme.render_llm_sweep([sweep()], accuracy=acc))[-4]
        self.assertEqual(text(cell), "0.0300")
        self.assertIn("8k: KLD 0.0400", cell)

    def test_unmeasured_cells_are_blank(self):
        # A dash in these tables means a context the speed sweep skipped.
        acc = load([accuracy("dequant", "decode", 0.03)])
        cells = row(
            generate_readme.render_llm_sweep(
                [sweep(), sweep("llama32_1b")], accuracy=acc
            ),
            "llama32_1b",
        )
        self.assertEqual(cells[-4:-1], ["", "", ""])

    def test_llm_page_hides_the_sidebars(self):
        page = generate_readme.generate_readme(section="llm")
        self.assertTrue(page.startswith("---\nhide:\n  - navigation\n  - toc\n---\n"))


if __name__ == "__main__":
    unittest.main()
