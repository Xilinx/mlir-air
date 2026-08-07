# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run the unmodified CPU model on its own, and dump the result.

`make cpu-baseline` -> smolvla_oracle.npz, holding the (1, 50, 6) action chunk:
the tensor the robot would actually execute. Needs torch + lerobot (see
requirements.txt); no NPU.

This is for INSPECTION -- SmolVLA is a three-stage model and it is useful to be
able to run the reference on its own and look at what it produces. **`make
verify` does not read this file.** The gate computes its CPU reference live, in
the same process, so that it cannot go stale against a checkpoint or lerobot
upgrade (and so that the verify lit test works on a clean checkout, where this
gitignored .npz does not exist).

The baseline is upstream lerobot itself, not a reimplementation in this repo,
so the comparison is "what does swapping in the NPU vision encoder change"
rather than "do two of my own implementations agree".

The noise is pinned to zero because the action expert is a flow-matching
denoiser seeded from noise; without that, two runs would differ by sampling
variance alone. `build_oracle_batch` is imported rather than copied so this and
the gate can never drift onto different inputs.

Porting another stage to the NPU
--------------------------------
This file used to dump eight more arrays -- per-layer backbone hidden states,
the KV cache, prefix pad masks, position ids -- for the backbone and action
expert ports, which measured slower than the CPU and are not part of this
example. That code, and the NPU implementations that consumed it, are on the
`smolvla` branch, which is also where the recipe and the traps worth knowing
before re-deriving it are written up.
"""

import numpy as np
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smolvla_inference import DEFAULT_MODEL, build_oracle_batch, fixed_noise

OUT = "smolvla_oracle.npz"


def main():
    torch.manual_seed(0)
    policy = SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()

    policy.reset()
    with torch.no_grad():
        action_chunk = policy.predict_action_chunk(
            build_oracle_batch(policy), noise=fixed_noise(policy)
        )

    action_chunk = action_chunk.detach().float().numpy()
    np.savez(OUT, action_chunk=action_chunk)
    print(f"[oracle] wrote {OUT}: action_chunk{action_chunk.shape}")


if __name__ == "__main__":
    main()
