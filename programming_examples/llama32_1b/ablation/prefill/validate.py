"""Per-cell validation — parameterized version of Plan 1's validate.py.

Plan 1's validate.py hardcodes the golden filename to
"golden_rms_gemv_rope.npz". For prefill we have two goldens, so we
parameterize the filename. The byte-equality contract is identical.
"""

import os

import numpy as np


class GoldenMismatch(AssertionError):
    """Raised when a cell's output diverges from the committed golden."""


def validate_against_golden(cell_outputs: dict, golden_dir: str, npz_filename: str):
    """Compare every key in cell_outputs to the matching array in
    <golden_dir>/<npz_filename>. Raise GoldenMismatch on any diff."""
    npz = np.load(os.path.join(golden_dir, npz_filename))
    for key in npz.files:
        if key not in cell_outputs:
            raise GoldenMismatch(f"cell missing output '{key}'")
        gv = npz[key]
        cv = cell_outputs[key]
        if cv.shape != gv.shape:
            raise GoldenMismatch(
                f"{key}: shape mismatch cell={cv.shape} golden={gv.shape}"
            )
        if cv.dtype.itemsize != gv.dtype.itemsize:
            raise GoldenMismatch(f"{key}: itemsize mismatch")
        if cv.tobytes() != gv.tobytes():
            from ml_dtypes import bfloat16 as _bf16

            cf = (
                cv.view(np.uint8).view(_bf16).astype(np.float32)
                if cv.dtype != np.float32
                else cv
            )
            gf = (
                gv.view(np.uint8).view(_bf16).astype(np.float32)
                if gv.dtype != np.float32
                else gv
            )
            max_abs = float(np.max(np.abs(cf - gf)))
            max_rel = float(np.max(np.abs((cf - gf) / (np.abs(gf) + 1e-9))))
            raise GoldenMismatch(
                f"{key}: byte mismatch  max_abs={max_abs:.4g}  max_rel={max_rel:.4g}"
            )
