"""KV cache state must be deterministic and per-trial resettable."""

import numpy as np

from cells.kv_cache import build_initial_kv_cache, reset_position

CONFIG = {
    "n_layers": 16,
    "n_kv_heads": 8,
    "head_dim": 64,
    "max_seq": 2048,
}


def test_initial_cache_is_deterministic():
    c1 = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    c2 = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    assert c1["k_cache"].tobytes() == c2["k_cache"].tobytes()
    assert c1["v_cache"].tobytes() == c2["v_cache"].tobytes()
    assert c1["current_pos"] == 7
    assert c2["current_pos"] == 7


def test_initial_cache_zeros_after_prompt_len():
    cache = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    # Positions 7..max_seq-1 must be zeros
    after = cache["k_cache"][:, :, 7:, :]
    assert np.all(after.view(np.uint8) == 0)
    after_v = cache["v_cache"][:, :, 7:, :]
    assert np.all(after_v.view(np.uint8) == 0)


def test_initial_cache_nonzero_in_prompt_range():
    cache = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    # At least some entries in [0:7] must be non-zero
    pre = cache["k_cache"][:, :, :7, :]
    assert not np.all(pre.view(np.uint8) == 0)


def test_reset_position_zeros_target_slot_only():
    cache = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    # Simulate a kernel writing to position 7 in layer 0
    cache["k_cache"][0, :, 7, :] = 99.0
    cache["v_cache"][0, :, 7, :] = -42.0
    # Reset should zero position 7 across ALL layers
    reset_position(cache, 7)
    assert np.all(cache["k_cache"][:, :, 7, :].view(np.uint8) == 0)
    assert np.all(cache["v_cache"][:, :, 7, :].view(np.uint8) == 0)
    # Positions 0..6 must be untouched (still match a fresh init)
    fresh = build_initial_kv_cache(CONFIG, prompt_len=7, seed=42)
    assert (
        cache["k_cache"][:, :, :7, :].tobytes()
        == fresh["k_cache"][:, :, :7, :].tobytes()
    )
    assert (
        cache["v_cache"][:, :, :7, :].tobytes()
        == fresh["v_cache"][:, :, :7, :].tobytes()
    )
