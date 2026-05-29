"""Verify Plan 1's validate.py works against the new decode goldens.

Two goldens: golden_rms_gemv_rope_decode.npz and golden_o_gemv_ffn_decode.npz.
For each, two tests:
  1. Loading the golden and validating it against itself MUST pass.
  2. Mutating one byte and re-validating MUST raise GoldenMismatch.

These tests do NOT touch the NPU.
"""

import os

import numpy as np
from ml_dtypes import bfloat16

from validate import GoldenMismatch, validate_against_golden

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "golden")


def _load(name):
    return np.load(os.path.join(GOLDEN_DIR, name))


def test_rms_gemv_rope_passes_on_exact_match():
    npz = _load("golden_rms_gemv_rope_decode.npz")
    cell_outputs = {key: npz[key] for key in npz.files}
    validate_against_golden(cell_outputs, GOLDEN_DIR, "golden_rms_gemv_rope_decode.npz")


def test_rms_gemv_rope_raises_on_byte_diff():
    npz = _load("golden_rms_gemv_rope_decode.npz")
    perturbed = {k: npz[k].copy() for k in npz.files}
    arr = perturbed["normed"].view(np.uint8).copy()
    arr[0] ^= 0x01  # flip one bit
    perturbed["normed"] = arr.view(bfloat16).reshape(npz["normed"].shape)
    try:
        validate_against_golden(
            perturbed, GOLDEN_DIR, "golden_rms_gemv_rope_decode.npz"
        )
        raise AssertionError("expected GoldenMismatch")
    except GoldenMismatch:
        pass


def test_o_gemv_ffn_passes_on_exact_match():
    npz = _load("golden_o_gemv_ffn_decode.npz")
    cell_outputs = {key: npz[key] for key in npz.files}
    validate_against_golden(cell_outputs, GOLDEN_DIR, "golden_o_gemv_ffn_decode.npz")


def test_o_gemv_ffn_raises_on_byte_diff():
    npz = _load("golden_o_gemv_ffn_decode.npz")
    perturbed = {k: npz[k].copy() for k in npz.files}
    arr = perturbed["output"].view(np.uint8).copy()
    arr[0] ^= 0x01
    perturbed["output"] = arr.view(bfloat16).reshape(npz["output"].shape)
    try:
        validate_against_golden(perturbed, GOLDEN_DIR, "golden_o_gemv_ffn_decode.npz")
        raise AssertionError("expected GoldenMismatch")
    except GoldenMismatch:
        pass
