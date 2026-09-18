#!/usr/bin/env python3
"""Turn a Hugging Face Qwen3 checkpoint into what gen.py's chain expects.

The chain reads one flat little-endian float32 blob. This writes that blob plus
a manifest naming the float offset and shape of every tensor in it, so gen.py
does not have to agree with this script about the order by convention.

safetensors is parsed here rather than imported: the format is an 8-byte
little-endian header length, a JSON header, then the raw tensor data, and the
only dtype in a Qwen3 checkpoint is bfloat16, which widens to float32 by
shifting left 16 bits. That avoids needing torch or the safetensors package on
a machine where neither is installed.

The layouts are the ones the chain indexes with, which are the transposes of
what Hugging Face stores: HF keeps a linear layer as [out, in] and every matmul
in the chain reduces over the leading axis.

    ./weights.py /shared/erweiw/qwen3-0.6b /shared/erweiw/qwen3-0.6b/air
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np


def read_safetensors(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            assert meta["dtype"] == "BF16", f"{name}: {meta['dtype']} not handled"
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = np.frombuffer(f.read(e - s), dtype="<u2")
            out[name] = (
                (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
            )
    return out


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    src, dst = Path(argv[1]), Path(argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    L = cfg["num_hidden_layers"]
    dim = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    nq = cfg["num_attention_heads"]
    nkv = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    vocab = cfg["vocab_size"]
    qw = nq * hd
    qkvo = (nq + 2 * nkv) * hd

    t = read_safetensors(src / "model.safetensors")
    emb = t["model.embed_tokens.weight"]
    assert emb.shape == (vocab, dim), emb.shape
    tied = cfg.get("tie_word_embeddings", False)
    lm = emb if tied else t["lm_head.weight"]

    def layerstack(fn, shape):
        a = np.empty((L,) + shape, dtype=np.float32)
        for l in range(L):
            a[l] = fn(l)
        return a

    pre = "model.layers.%d."
    tensors = {
        # [vocab, dim]: the embedding is gathered by token id
        "Emb": emb.astype(np.float32),
        # [dim, vocab]: the lm head reduces over dim
        "Wlm": np.ascontiguousarray(lm.astype(np.float32).T),
        "Nf": t["model.norm.weight"].astype(np.float32),
        "N1": layerstack(lambda l: t[(pre % l) + "input_layernorm.weight"], (dim,)),
        "N2": layerstack(
            lambda l: t[(pre % l) + "post_attention_layernorm.weight"], (dim,)
        ),
        # q_norm in [0, hd), k_norm in [hd, 2hd)
        "QKN": layerstack(
            lambda l: np.concatenate(
                [
                    t[(pre % l) + "self_attn.q_norm.weight"],
                    t[(pre % l) + "self_attn.k_norm.weight"],
                ]
            ),
            (2 * hd,),
        ),
        # one fused projection, q then k then v, transposed to reduce over dim
        "Wqkv": layerstack(
            lambda l: np.concatenate(
                [
                    t[(pre % l) + "self_attn.q_proj.weight"],
                    t[(pre % l) + "self_attn.k_proj.weight"],
                    t[(pre % l) + "self_attn.v_proj.weight"],
                ],
                0,
            ).T.astype(np.float32),
            (dim, qkvo),
        ),
        "Wo": layerstack(
            lambda l: t[(pre % l) + "self_attn.o_proj.weight"].T.astype(np.float32),
            (qw, dim),
        ),
        # gate then up, the fusion Fleet also uses
        "Wgu": layerstack(
            lambda l: np.concatenate(
                [
                    t[(pre % l) + "mlp.gate_proj.weight"],
                    t[(pre % l) + "mlp.up_proj.weight"],
                ],
                0,
            ).T.astype(np.float32),
            (dim, 2 * inter),
        ),
        "Wd": layerstack(
            lambda l: t[(pre % l) + "mlp.down_proj.weight"].T.astype(np.float32),
            (inter, dim),
        ),
    }

    offsets, cur = {}, 0
    blob = dst / "weights.f32"
    with open(blob, "wb") as f:
        for name, a in tensors.items():
            a = np.ascontiguousarray(a, dtype="<f4")
            offsets[name] = {
                "offset": cur,
                "count": int(a.size),
                "shape": list(a.shape),
            }
            f.write(a.tobytes())
            cur += a.size
            print(
                f"  {name:5s} {str(list(a.shape)):24s} " f"{a.size * 4 / 1e6:8.1f} MB",
                file=sys.stderr,
            )
    manifest = {
        "blob": blob.name,
        "floats": cur,
        "config": {
            "layers": L,
            "dim": dim,
            "inter": inter,
            "heads": nq,
            "kv_heads": nkv,
            "head_dim": hd,
            "vocab": vocab,
            "rope_theta": cfg["rope_theta"],
            "rms_eps": cfg["rms_norm_eps"],
            "tie_word_embeddings": tied,
        },
        "tensors": offsets,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"wrote {blob} ({cur * 4 / 1e9:.2f} GB) and {dst / 'manifest.json'}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
