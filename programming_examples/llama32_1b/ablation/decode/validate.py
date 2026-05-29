"""Re-export Plan 1's parameterized bit-exact validation gate.

Plan 1's validate.py accepts a `golden_filename` parameter, so the same
function works for decode goldens too — just pass a different filename.
"""

from prefill.validate import (
    validate_against_golden,
    GoldenMismatch,
)

__all__ = ["validate_against_golden", "GoldenMismatch"]
