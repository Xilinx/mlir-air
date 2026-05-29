"""Unit tests for the KernelGroupSpec dataclasses."""

import pytest
from specs.kernel_group import SubLaunchSpec, BatonLink, KernelGroupSpec


def _dummy_builder():
    return None  # Spec test doesn't need a real builder


def test_sublaunch_spec_is_frozen():
    s = SubLaunchSpec(
        name="rms",
        builder_ref=_dummy_builder,
        build_kwargs={"emb_dim": 2048},
        weight_slot_in_standalone=1,
        output_slot_in_standalone=2,
    )
    with pytest.raises((AttributeError, TypeError)):  # frozen
        s.name = "other"


def test_baton_link_orders_by_indices():
    link = BatonLink(
        producer_idx=0, producer_out_slot=2, consumer_idx=1, consumer_in_slot=1
    )
    assert link.consumer_idx > link.producer_idx


def test_kernel_group_spec_holds_sublaunches():
    sub = SubLaunchSpec("rms", _dummy_builder, {}, 1, 2)
    spec = KernelGroupSpec(
        name="rms_gemms_rope",
        sub_launches=(sub,),  # tuple — frozen dataclass
        merged_arg_signature=("x_in", "norm_w", "normed"),
        weight_slots=frozenset({1}),
        intermediate_slots=frozenset({2}),
        output_slots_for_validation=(2,),
        baton_links=(),
    )
    assert spec.name == "rms_gemms_rope"
    assert len(spec.sub_launches) == 1


def test_baton_link_consumer_must_follow_producer():
    """A baton link with consumer_idx <= producer_idx is meaningless;
    spec dataclass tolerates it but a validator rejects."""
    from specs.kernel_group import validate_baton_links

    sub_a = SubLaunchSpec("a", _dummy_builder, {}, 1, 2)
    sub_b = SubLaunchSpec("b", _dummy_builder, {}, 1, 2)
    bad = BatonLink(
        producer_idx=1, producer_out_slot=2, consumer_idx=0, consumer_in_slot=1
    )
    with pytest.raises(ValueError, match="consumer_idx"):
        validate_baton_links([sub_a, sub_b], [bad])
