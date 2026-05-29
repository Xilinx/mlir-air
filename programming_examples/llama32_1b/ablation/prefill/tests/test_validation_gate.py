"""Test the prefill validation gate against the committed goldens."""

import os

import numpy as np
import pytest
from ml_dtypes import bfloat16

from validate import validate_against_golden, GoldenMismatch

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden")


def _load(filename):
    npz = np.load(os.path.join(GOLDEN_DIR, filename))
    return {k: npz[k] for k in npz.files}


def test_rms_gemms_rope_passes_on_exact_match():
    g = _load("golden_rms_gemms_rope_prefill.npz")
    validate_against_golden(g, GOLDEN_DIR, "golden_rms_gemms_rope_prefill.npz")


def test_rms_gemms_rope_raises_on_byte_diff():
    g = _load("golden_rms_gemms_rope_prefill.npz")
    perturbed = {k: v.copy() for k, v in g.items()}
    arr = perturbed["normed"].view(np.uint8).copy()
    arr[0] ^= 0x01
    perturbed["normed"] = arr.view(bfloat16).reshape(g["normed"].shape)
    with pytest.raises(GoldenMismatch, match="normed"):
        validate_against_golden(
            perturbed, GOLDEN_DIR, "golden_rms_gemms_rope_prefill.npz"
        )


def test_o_ffn_passes_on_exact_match():
    g = _load("golden_o_ffn_prefill.npz")
    validate_against_golden(g, GOLDEN_DIR, "golden_o_ffn_prefill.npz")


def test_o_ffn_raises_on_byte_diff():
    g = _load("golden_o_ffn_prefill.npz")
    perturbed = {k: v.copy() for k, v in g.items()}
    arr = perturbed["output"].view(np.uint8).copy()
    arr[0] ^= 0x01
    perturbed["output"] = arr.view(bfloat16).reshape(g["output"].shape)
    with pytest.raises(GoldenMismatch, match="output"):
        validate_against_golden(perturbed, GOLDEN_DIR, "golden_o_ffn_prefill.npz")
