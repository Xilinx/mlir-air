# SPDX-License-Identifier: MIT
"""Real action-expert data for the NPU expert probes, plus an fp32 NumPy reference.

`capture()` runs lerobot's pure-CPU `predict_action_chunk` on the gate's batch
(zero noise) and records, for each of the denoising calls into the expert: its
input embeddings, attention mask, position ids, the cached prefix K/V of every
layer, and lerobot's output. `expert_weights()` pulls the expert's weights.
`reference_expert()` recomputes the expert in fp32 from those; checking it
against lerobot's output validates the reference before the NPU is compared
to it.

Run directly to capture into build/expert_capture.npz and check the reference.
"""

import sys
from pathlib import Path

import numpy as np

import backbone_npu as bn  # noqa: F401  (sys.path setup)

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "build" / "expert_capture.npz"
NH, NKV, HD = 15, 5, 64


def load_policy():
    sys.path.insert(0, str(HERE.parent))
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from smolvla_inference import DEFAULT_MODEL

    return SmolVLAPolicy.from_pretrained(DEFAULT_MODEL).eval()


def capture(policy):
    import torch
    from smolvla_inference import build_oracle_batch, fixed_noise

    vwe = policy.model.vlm_with_expert
    orig = vwe.forward
    steps = []

    def f32(t):
        return t.detach().float().numpy()

    def wrapped(*a, **kw):
        embs, pkv = kw.get("inputs_embeds"), kw.get("past_key_values")
        if pkv is None or embs is None or embs[0] is not None:
            return orig(*a, **kw)
        rec = dict(
            x=f32(embs[1][0]),
            mask=kw["attention_mask"][0].numpy(),
            pos=kw["position_ids"][0].numpy(),
            k=np.stack(
                [
                    f32(pkv.layers[i].keys[0].transpose(0, 1))
                    for i in range(len(pkv.layers))
                ]
            ),
            v=np.stack(
                [
                    f32(pkv.layers[i].values[0].transpose(0, 1))
                    for i in range(len(pkv.layers))
                ]
            ),
        )
        out = orig(*a, **kw)
        rec["out"] = f32(out[0][1][0])
        steps.append(rec)
        return out

    vwe.forward = wrapped
    try:
        policy.reset()
        with torch.no_grad():
            chunk = policy.predict_action_chunk(
                build_oracle_batch(policy), noise=fixed_noise(policy)
            )
    finally:
        vwe.forward = orig
    d = {key: np.stack([s[key] for s in steps]) for key in steps[0]}
    d["chunk"] = chunk.float().numpy()
    return d


def expert_weights(policy):
    """Per layer: f32 [in, out] matrices and norm weights, plus the final norm."""
    ex = policy.model.vlm_with_expert.lm_expert
    t = lambda p: p.detach().float().numpy()  # noqa: E731
    layers = []
    for layer in ex.layers:
        a, m = layer.self_attn, layer.mlp
        lin = lambda mod: t(
            getattr(mod, "lin", mod).weight
        ).T  # noqa: E731  (kv memo wrappers)
        layers.append(
            dict(
                anorm=t(layer.input_layernorm.weight),
                fnorm=t(layer.post_attention_layernorm.weight),
                wq=lin(a.q_proj),
                wk=lin(a.k_proj),
                wv=lin(a.v_proj),
                wo=lin(a.o_proj),
                wg=lin(m.gate_proj),
                wu=lin(m.up_proj),
                wd=lin(m.down_proj),
            )
        )
    eps = ex.layers[0].input_layernorm.variance_epsilon
    theta = getattr(ex.config, "rope_theta", 10000.0)
    return layers, dict(
        norm=t(ex.norm.weight), eps=eps, theta=theta, self_every=vwe_self_every(policy)
    )


def vwe_self_every(policy):
    return policy.model.vlm_with_expert.self_attn_every_n_layers


def rope(x, pos, theta):
    """lerobot apply_rope: halves, x [S, H, D]."""
    d = x.shape[-1] // 2
    inv = 1.0 / theta ** (np.arange(d, dtype=np.float64) * 2 / x.shape[-1])
    ang = pos[:, None].astype(np.float64) * inv[None]
    c, s = np.cos(ang)[:, None].astype(np.float32), np.sin(ang)[:, None].astype(
        np.float32
    )
    x1, x2 = x[..., :d], x[..., d:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def rms(a, g, eps):
    return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps) * g


def is_self(l, meta):
    return l % meta["self_every"] == 0


def layer_kv(w, l, meta, k_cache, v_cache, h, pos):
    """(K, V) [keys, NKV*HD] expert layer l attends to; h is its normed input.

    RoPE positions are relative to the first action token: a self layer's prefix
    keys are rotated back by it, so every layer's queries (and a self layer's own
    keys) use positions 0..S-1. Scores only depend on position differences.
    """
    kc, vc = k_cache.reshape(len(k_cache), NKV, HD), v_cache.reshape(len(v_cache), -1)
    p0 = int(pos.min())
    if is_self(l, meta):
        kc = rope(kc, np.full(len(kc), -p0), meta["theta"]).reshape(len(kc), -1)
        k = rope((h @ w["wk"]).reshape(-1, NKV, HD), pos - p0, meta["theta"]).reshape(
            len(pos), -1
        )
        return np.concatenate([kc, k]), np.concatenate([vc, h @ w["wv"]])
    return kc.reshape(len(kc), -1) @ w["wk"], vc @ w["wv"]


def layer_mask(l, meta, mask, n_prefix):
    return mask if is_self(l, meta) else mask[:, :n_prefix]


def ref_layer(x, w, meta, k, v, mask):
    """fp32 expert layer on given K/V [keys, NKV*HD] and bool mask [S, keys]."""
    s, eps = x.shape[0], meta["eps"]
    h = rms(x, w["anorm"], eps)
    q = rope((h @ w["wq"]).reshape(s, NH, HD), np.arange(s), meta["theta"])
    kk, vv = k.reshape(-1, NKV, HD), v.reshape(-1, NKV, HD)
    att = np.empty((s, NH, HD), np.float32)
    for hh in range(NH):
        sc = q[:, hh] @ kk[:, hh // (NH // NKV)].T / np.sqrt(HD)
        sc = np.where(mask, sc, -np.inf)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        att[:, hh] = (p / p.sum(-1, keepdims=True)) @ vv[:, hh // (NH // NKV)]
    x = att.reshape(s, -1) @ w["wo"] + x
    n2 = rms(x, w["fnorm"], eps)
    g = n2 @ w["wg"]
    return (g / (1 + np.exp(-g)) * (n2 @ w["wu"])) @ w["wd"] + x


def reference_expert(x, layers, meta, k_cache, v_cache, mask, pos, collect=None):
    """fp32 expert forward for one denoising call. x [S, 720]; returns the final-normed output.
    collect: list to append (layer input, K, V, mask) per layer, then the last layer's output.
    """
    for l, w in enumerate(layers):
        k, v = layer_kv(
            w, l, meta, k_cache[l], v_cache[l], rms(x, w["anorm"], meta["eps"]), pos
        )
        m = layer_mask(l, meta, mask, len(k_cache[l]))
        if collect is not None:
            collect.append((x, k, v, m))
        x = ref_layer(x, w, meta, k, v, m)
    if collect is not None:
        collect.append(x)
    return rms(x, meta["norm"], meta["eps"])


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    policy = load_policy()
    if "--reuse" in sys.argv and CAPTURE.exists():
        d = dict(np.load(CAPTURE))
    else:
        d = capture(policy)
        CAPTURE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CAPTURE, **d)
    print({k: v.shape for k, v in d.items()})
    layers, meta = expert_weights(policy)
    print("meta", {k: v for k, v in meta.items() if k != "norm"}, "layers", len(layers))
    m0 = d["mask"][0]
    print(
        "prefix keys",
        d["k"].shape[2],
        "valid prefix keys",
        int(m0[0, : d["k"].shape[2]].sum()),
        "suffix mask causal?",
        bool(
            (
                m0[:, d["k"].shape[2] :]
                == np.tril(np.ones_like(m0[:, d["k"].shape[2] :]))
            ).all()
        ),
        "pos",
        d["pos"][0][:3],
        "...",
        d["pos"][0][-1],
    )
    for i in range(len(d["x"])):
        ref = reference_expert(
            d["x"][i], layers, meta, d["k"][i], d["v"][i], d["mask"][i], d["pos"][i]
        )
        print(f"step {i}: reference vs lerobot cosine {cos(ref, d['out'][i]):.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
