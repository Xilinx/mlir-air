# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""DFlash token captures compared with llms/verify's top-5 comparator.

There is no BF16 checkpoint of this model, so the reference is
reference_b8's float32 forward of the target package's own weights, over
the prompt and the NPU's tokens (teacher forced).
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dflash_model import MODEL  # noqa: E402
from model_paths import TARGET_MODEL, resolve_target_model  # noqa: E402
from resources import prepare  # noqa: E402
from common.check_model import main  # noqa: E402

_qm = None


def reference(prompt_ids, tokens, k):
    import reference_b8
    from weights_b8 import gw

    global _qm
    if _qm is None:
        _qm = gw.Q4nxModel(resolve_target_model(TARGET_MODEL, offline=True))
    ids = list(prompt_ids) + list(tokens[:-1])
    _, logits = reference_b8.forward(_qm, ids, gw.NUM_LAYERS)
    rows = logits[len(prompt_ids) - 1 :]
    topk = np.argsort(-rows, axis=1, kind="stable")[:, :k]
    return topk[:, 0].tolist(), topk.tolist()


def load_prefill(root, case):
    raise NotImplementedError("the model check uses NPU prefill")


if __name__ == "__main__":
    main(MODEL, prepare=prepare, load_prefill=load_prefill, reference=reference)
