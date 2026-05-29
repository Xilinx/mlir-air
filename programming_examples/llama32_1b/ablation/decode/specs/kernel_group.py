"""Re-export Plan 1's KernelGroupSpec dataclasses (single source of truth).

Decode specs (rms_gemv_rope, o_gemv_ffn) and cells reference these. Keeping
one definition prevents drift across the three plans.
"""

from prefill.specs.kernel_group import (
    SubLaunchSpec,
    BatonLink,
    KernelGroupSpec,
    validate_baton_links,
)

__all__ = ["SubLaunchSpec", "BatonLink", "KernelGroupSpec", "validate_baton_links"]
