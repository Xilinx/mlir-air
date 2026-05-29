"""Frozen dataclasses describing a multi-launch kernel-group's structure.

A KernelGroupSpec is consumed by parameterized cells (cell_a/b/c/d) so that
the same cell logic works for any kernel-group whose spec is provided.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SubLaunchSpec:
    """One sub-launch's standalone definition.

    Used by Cell A/B/C to invoke the sub-launch as its own xrt.run() call.
    Cell D ignores SubLaunchSpec entirely (it uses the merged ELF).
    """

    name: str  # "rmsnorm" | "q_gemm" | "rope_q" | ...
    builder_ref: Callable  # returns a 1-launch mlir.Module at production shape
    build_kwargs: dict  # passed verbatim to builder_ref
    weight_slot_in_standalone: (
        int | None
    )  # arg slot of the standalone call holding the weight (or None)
    output_slot_in_standalone: int  # arg slot of the standalone call holding the output


@dataclass(frozen=True)
class BatonLink:
    """An intermediate-BO alias to apply in Cell C.

    The producer's output BO becomes the consumer's input BO; the host
    skips writing the consumer's input slot via intermediate_indices.
    """

    producer_idx: int  # index into KernelGroupSpec.sub_launches
    producer_out_slot: int  # output slot of producer's standalone signature
    consumer_idx: (
        int  # index into KernelGroupSpec.sub_launches (must be > producer_idx)
    )
    consumer_in_slot: int  # input slot of consumer's standalone signature


@dataclass(frozen=True)
class KernelGroupSpec:
    """Full description of a multi-launch kernel-group for ablation."""

    name: str  # "rms_gemms_rope" | "o_ffn"
    sub_launches: tuple  # tuple of SubLaunchSpec (frozen)
    merged_arg_signature: (
        tuple  # tuple of arg-name strings matching production merged ELF args
    )
    weight_slots: frozenset  # slots in merged signature that are weights/LUTs (Cell D static_input_indices)
    intermediate_slots: (
        frozenset  # slots in merged signature that are kernel-overwritten intermediates
    )
    output_slots_for_validation: tuple  # slots whose bytes go in the golden npz
    baton_links: tuple  # tuple of BatonLink (Cell C aliases these intermediate BOs)


def validate_baton_links(sub_launches, baton_links):
    """Sanity check: each link's consumer must come after its producer in the sequence."""
    for link in baton_links:
        if link.consumer_idx <= link.producer_idx:
            raise ValueError(
                f"baton link consumer_idx={link.consumer_idx} must be greater than "
                f"producer_idx={link.producer_idx}"
            )
        if link.producer_idx >= len(sub_launches):
            raise ValueError(f"producer_idx {link.producer_idx} out of range")
        if link.consumer_idx >= len(sub_launches):
            raise ValueError(f"consumer_idx {link.consumer_idx} out of range")
