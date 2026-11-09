# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The gate's audio: the 73 LibriSpeech dev-clean utterances of
hf-internal-testing/librispeech_asr_dummy at a pinned revision, in file
order. `verify` takes the first N_VERIFY, `verify-full` all of them."""

import io

DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FILE = "clean/validation-00000-of-00001.parquet"
N_VERIFY = 8


def clips(n=None):
    """[(id, float32 16 kHz audio, reference text)] of the first n clips."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, FILE, repo_type="dataset", revision=REVISION)
    rows = pq.read_table(path).to_pylist()[:n]
    out = []
    for r in rows:
        a, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16000, (r["id"], sr)
        out.append((r["id"], a, r["text"]))
    return out
