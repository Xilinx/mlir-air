# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bounded-memory BF16 prefill weights, using AIR's unchanged Q4NX dequantizer.

Converted tensors live in read-only .npy mappings. Conversion and verification
work on whole Q4NX row blocks; neither expands a full vocabulary head in RAM.
The file representation is uint16 so NumPy preserves BF16 bits portably.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys

import numpy as np
from ml_dtypes import bfloat16

from paths import HERE

sys.path.insert(0, str(HERE.parents[1] / "qwen3_8b_q4nx"))
from qwen3_8b_q4nx_weights import Q4nxModel, BLOCK_BF16

SCHEMA = "qwen3-8b-prefill-mapped-bf16-v1"
PROJECTIONS = {
    "q": "wq",
    "k": "wk",
    "v": "wv",
    "o": "wo",
    "gate": "w_gate",
    "up": "w_up",
    "down": "w_down",
}
NORMS = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "q_norm": "self_attn.q_norm",
    "k_norm": "self_attn.k_norm",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def drop_pages(array):
    """Release this process's mapped pages after its synchronous last reader."""
    current = array
    while current is not None:
        if isinstance(current, np.memmap):
            current._mmap.madvise(mmap.MADV_DONTNEED)
            return
        current = getattr(current, "base", None)


def discard_file_cache(path):
    # These are this example's own converted-cache files, never original models.
    with Path(path).open("rb") as f:
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def plain(model, name):
    item = model._hdr[name]
    lo, hi = item["data_offsets"]
    assert item["dtype"] == "BF16" and hi - lo == int(np.prod(item["shape"])) * 2
    return (
        model._mm[model._base + lo : model._base + hi]
        .view(bfloat16)
        .reshape(item["shape"])
    )


def dequant_rows(model, name, rows, width, start, count):
    """Select complete row blocks, then call the unmodified AIR reference."""
    if (
        start % 32
        or count % 32
        or width % 256
        or not 0 <= start < start + count <= rows
    ):
        raise ValueError("Q4NX dequantization slice must contain complete row blocks")
    item = model._hdr[name]
    assert item["shape"][0] == (rows // 32) * (width // 256)
    lo, hi = item["data_offsets"]
    block_bytes = BLOCK_BF16 * 2
    assert hi - lo == item["shape"][0] * block_bytes
    offset = model._base + lo + (start // 32) * (width // 256) * block_bytes
    size = (count // 32) * (width // 256) * block_bytes

    class RowSlice(Q4nxModel):
        def __init__(self):
            # Keep the normal reader/dequantizer methods, with a bounded view
            # of the original bytes and offsets relative to that view.
            self._mm = model._mm[offset : offset + size]
            self._base = 0
            self._hdr = {
                name: {
                    "shape": [(count // 32) * (width // 256)],
                    "data_offsets": [0, size],
                }
            }

    return RowSlice().dequant(name, count, width)


def definitions():
    result = []
    for layer in range(36):
        for kind, field in PROJECTIONS.items():
            suffix, rows, width = Q4nxModel._PROJ[kind]
            result.append(
                (
                    f"L{layer:02d}_{field}.npy",
                    f"model.layers.{layer}.{suffix}.weight",
                    rows,
                    width,
                    True,
                )
            )
    result.append(("lm_head.npy", "lm_head.weight", 151936, 4096, False))
    return result


def convert(source, output):
    output.mkdir(parents=True, exist_ok=False)
    model = Q4nxModel(str(source))
    record = dict(
        schema=SCHEMA,
        status="preparing",
        model=str(source.resolve()),
        model_sha256=sha(source),
        dequantizer_sha256=sha(Path(sys.modules[Q4nxModel.__module__].__file__)),
        source_sha256=sha(__file__),
        tensors={},
    )

    def save():
        (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    for file, name, rows, width, transpose in definitions():
        # Pad the final vocabulary partition on disk, avoiding a full host copy.
        shape = (width, rows) if transpose else (163840, width)
        dest = np.lib.format.open_memmap(
            output / file, mode="w+", dtype=np.uint16, shape=shape
        )
        for start in range(0, rows, 256):
            n = min(256, rows - start)
            bf = (
                dequant_rows(model, name, rows, width, start, n)
                .astype(bfloat16)
                .view(np.uint16)
            )
            if transpose:
                dest[:, start : start + n] = bf.T
            else:
                dest[start : start + n] = bf
        if not transpose:
            dest[rows:] = 0
        dest.flush()
        drop_pages(dest)
        del dest
        record["tensors"][file] = dict(
            tensor=name, shape=shape, transpose=transpose, sha256=sha(output / file)
        )
        discard_file_cache(output / file)
        drop_pages(model._mm)
        save()
        if file.endswith("_w_down.npy") or file == "lm_head.npy":
            print(
                f'Converted {file}; completed {len(record["tensors"])}/253 tensors',
                flush=True,
            )
    record["status"] = "prepared"
    save()


def verify(source, root, output):
    """Check all converted bits with a different chunk size; anchor real layers
    against the original whole-tensor method, which is affordable per layer.
    """
    if output.exists():
        raise FileExistsError(output)
    model = Q4nxModel(str(source))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "prepared" and manifest["schema"] == SCHEMA
    assert manifest["model_sha256"] == sha(source)
    checked = 0
    for file, name, rows, width, transpose in definitions():
        actual = np.load(root / file, mmap_mode="r")
        assert sha(root / file) == manifest["tensors"][file]["sha256"]
        for start in range(0, rows, 512):
            n = min(512, rows - start)
            reference = (
                dequant_rows(model, name, rows, width, start, n)
                .astype(bfloat16)
                .view(np.uint16)
            )
            selected = (
                actual[:, start : start + n].T
                if transpose
                else actual[start : start + n]
            )
            assert np.array_equal(selected, reference), (name, start)
            checked += selected.size
        if not transpose:
            assert not actual[rows:].any()
        drop_pages(actual)
        discard_file_cache(root / file)
        drop_pages(model._mm)
    anchors = []
    for layer in (0, 17, 35):
        original = model.layer_weights(layer)
        for kind, field in PROJECTIONS.items():
            file = root / f"L{layer:02d}_{field}.npy"
            actual = np.load(file, mmap_mode="r")
            assert np.array_equal(actual, original[kind].view(np.uint16)), (layer, kind)
            drop_pages(actual)
            discard_file_cache(file)
        del original
        drop_pages(model._mm)
        anchors.append(layer)
        print(f"Whole-layer AIR reference exact: layer {layer}", flush=True)
    result = dict(
        passed=True,
        self_validated=True,
        manifest_sha256=sha(root / "manifest.json"),
        source_sha256=sha(__file__),
        tensors=253,
        checked_elements=checked,
        whole_layer_reference_exact=anchors,
        vocabulary_padding_zero=True,
        reference="unchanged AIR Q4nxModel.dequant; whole-layer anchors plus complete chunked BF16-bit coverage",
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


class MappedWeights:
    def __init__(self, root, source):
        from qwen3_4b_weights import LayerWeights

        self.root = Path(root)
        record = json.loads((self.root / "manifest.json").read_text())
        if record["schema"] != SCHEMA or record["status"] != "prepared":
            raise ValueError(
                "Prepare and verify the BF16 weight cache before NPU execution"
            )
        if record["model_sha256"] != sha(source):
            raise ValueError("Prefill weight-cache model identity differs")
        for file, entry in record["tensors"].items():
            if sha(self.root / file) != entry["sha256"]:
                raise ValueError(f"Prefill BF16 weight checksum mismatch: {file}")
            discard_file_cache(self.root / file)
        self.model = Q4nxModel(str(source))
        self.embed_table = plain(self.model, "model.embed_tokens.weight")
        self.final_norm = plain(self.model, "model.norm.weight")
        self.lm_head = np.load(self.root / "lm_head.npy", mmap_mode="r").view(bfloat16)
        self.layers = []
        for layer in range(36):
            values = {
                field: np.load(
                    self.root / f"L{layer:02d}_{field}.npy", mmap_mode="r"
                ).view(bfloat16)
                for field in PROJECTIONS.values()
            }
            values.update(
                {
                    field: plain(
                        self.model, f"model.layers.{layer}.{suffix}.weight"
                    ).copy()
                    for field, suffix in NORMS.items()
                }
            )
            self.layers.append(LayerWeights(**values))

    def release_layer(self, layer):
        for field in PROJECTIONS.values():
            drop_pages(getattr(self.layers[layer], field))
            discard_file_cache(self.root / f"L{layer:02d}_{field}.npy")

    def release_head(self):
        drop_pages(self.lm_head)
        discard_file_cache(self.root / "lm_head.npy")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--verify", type=Path)
    args = ap.parse_args()
    if args.verify:
        verify(args.model, args.cache, args.verify)
    else:
        convert(args.model, args.cache)


if __name__ == "__main__":
    main()
