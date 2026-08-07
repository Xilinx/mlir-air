# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real recorded observations from a LeRobot dataset, as batches the model takes.

The synthetic path (`build_oracle_batch` in smolvla_inference) feeds
seeded-random images and a zero state: deterministic and needing nothing
downloaded, which is what the gate and CI want. This is the other input source, selected by `INPUT=real` on `make run`
and `make verify`.

Not a CLI -- the entry points are `make run` and `make verify`.
"""

from __future__ import annotations

DEFAULT_DATASET = "lerobot/droid_100"


def load(dataset=DEFAULT_DATASET):
    """Open a LeRobot dataset for frame-by-frame reads.

    `video_backend="pyav"` is not a preference. LeRobot defaults to torchcodec,
    which fails to load against some torch builds (it cannot find a matching
    FFmpeg/libtorchcodec); pyav ships with lerobot[dataset] and decodes the same
    files. Every multi-camera LeRobot dataset is video-backed, so this applies
    to all of them.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(dataset, video_backend="pyav")


def camera_keys(meta):
    """The dataset's camera keys, in feature order.

    Read rather than hardcoded, so pointing DATASET at another LeRobot dataset
    works without editing anything. Order decides which feeds survive the 1- and
    2-camera subsets.
    """
    return [
        k
        for k, v in meta.features.items()
        if v.get("dtype") in ("image", "video") and "images" in k
    ]


def sample_indices(meta, n_frames, n_episodes):
    """Frame indices spread across episodes rather than taken consecutively.

    At 15 fps adjacent frames are nearly identical, so 100 in a row would be
    statistically one frame. This takes them from the middle of each episode's
    span, cycling until n_frames is reached.
    """
    eps = meta.episodes
    spans = [
        (int(eps["dataset_from_index"][i]), int(eps["dataset_to_index"][i]))
        for i in range(n_episodes)
    ]
    per_ep = max(1, n_frames // len(spans))
    idxs = []
    for start, end in spans:
        span = end - start
        for j in range(per_ep):
            if len(idxs) >= n_frames:
                return idxs
            idxs.append(start + span * (2 * j + 1) // (2 * per_ep))
    return idxs


def build_batch(policy, tok, keys, n_cameras, sample):
    """One dataset frame -> a batch the policy accepts, with n_cameras feeds.

    Renaming a dataset's camera key onto the slot the checkpoint expects wires a
    real camera to a real slot; it is not synthesising input. State widths
    generally differ (droid_100 is 7, the checkpoint 6) -- padded or truncated
    here, which lerobot would do anyway via pad_vector to max_state_dim=32. That
    is numerically legal and semantically meaningless, which is fine for a
    CPU-vs-NPU comparison and would not be for a task-quality claim.
    """
    import torch

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    want_state = policy.config.input_features["observation.state"].shape[0]

    b = {}
    for i in range(1, n_cameras + 1):
        b[f"observation.images.camera{i}"] = (
            sample[keys[i - 1]][None].to(torch.float32).contiguous()
        )

    state = sample["observation.state"].to(torch.float32).ravel()
    state = (
        state[:want_state]
        if state.numel() >= want_state
        else torch.nn.functional.pad(state, (0, want_state - state.numel()))
    )
    b["observation.state"] = state[None]
    b[OBS_LANGUAGE_TOKENS] = tok["input_ids"]
    b[OBS_LANGUAGE_ATTENTION_MASK] = tok["attention_mask"].bool()
    return b


def batches(policy, prompt, n_cameras, dataset=DEFAULT_DATASET, n_frames=100):
    """Yield (frame_index, batch) for n_frames real observations."""
    ds = load(dataset)
    keys = camera_keys(ds.meta)
    if n_cameras > len(keys):
        raise SystemExit(
            f"{dataset} has {len(keys)} camera(s); asked for {n_cameras}. "
            f"Available: {keys}"
        )
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        [prompt],
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    for idx in sample_indices(ds.meta, n_frames, ds.num_episodes):
        yield idx, build_batch(policy, tok, keys, n_cameras, ds[idx])
