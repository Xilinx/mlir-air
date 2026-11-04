# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Checks the FLM-bundle loader and the CPU reference against HF openai/gpt-oss-20b.
#   weights: every tensor of the given layers, bundle vs HF (MXFP4 experts must
#            match exactly; Q4_1-requantised attention is reported).
#   forward: HF transformers logits vs Reference on HF weights (forward math)
#            vs Reference on the bundle (loader + quantisation).
import argparse
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpt_oss_20b_q4nx_weights as g  # noqa: E402

HF_REPO = "openai/gpt-oss-20b"


def hf_dir():
    try:
        return Path(snapshot_download(HF_REPO, local_files_only=True))
    except Exception:
        return Path(
            snapshot_download(HF_REPO, allow_patterns=["*.json", "*.safetensors"])
        )


def mxfp4_hf(blocks, scales):
    """HF MXFP4 [.., rows, G, 16] + e8m0 [.., rows, G] -> float32 [.., rows, G*32];
    low nibble is the even element."""
    lo, hi = g._E2M1[(blocks & 0xF).long()], g._E2M1[(blocks >> 4).long()]
    v = torch.stack((lo, hi), dim=-1).flatten(-2)
    return (v * g._e8m0(scales)[..., None]).flatten(-2)


class HfWeights:
    """Same interface as GptOssQ4nx, reading the HF checkpoint."""

    def __init__(self, path=None):
        path = Path(path or hf_dir())
        self._files = [
            safe_open(str(p), framework="pt")
            for p in sorted(path.glob("*.safetensors"))
        ]
        self._where = {k: f for f in self._files for k in f.keys()}
        self._expert_cache = {}

    def t(self, name):
        return self._where[name].get_tensor(name)

    def f(self, name):
        return self.t(name).float()

    def embed(self, ids):
        if not hasattr(self, "_embed"):
            self._embed = self.t("model.embed_tokens.weight")
        return self._embed[ids].float()

    def lm_head(self):
        return self.f("lm_head.weight")

    def attn(self, layer):
        p = f"model.layers.{layer}.self_attn."
        return dict(
            wq=self.f(p + "q_proj.weight"),
            wk=self.f(p + "k_proj.weight"),
            wv=self.f(p + "v_proj.weight"),
            wo=self.f(p + "o_proj.weight"),
            bq=self.f(p + "q_proj.bias"),
            bk=self.f(p + "k_proj.bias"),
            bv=self.f(p + "v_proj.bias"),
            bo=self.f(p + "o_proj.bias"),
            sinks=self.f(p + "sinks"),
        )

    def norms(self, layer):
        p = f"model.layers.{layer}."
        return self.f(p + "input_layernorm.weight"), self.f(
            p + "post_attention_layernorm.weight"
        )

    def router(self, layer):
        p = f"model.layers.{layer}.mlp.router."
        return self.f(p + "weight"), self.f(p + "bias")

    def expert(self, layer, e):
        key = (layer, e)
        if key not in self._expert_cache:
            if len(self._expert_cache) >= 64:
                self._expert_cache.clear()
            p = f"model.layers.{layer}.mlp.experts."

            def get(n):
                return self._where[p + n].get_slice(p + n)[e]

            gu = mxfp4_hf(get("gate_up_proj_blocks"), get("gate_up_proj_scales"))
            dn = mxfp4_hf(get("down_proj_blocks"), get("down_proj_scales"))
            gub = get("gate_up_proj_bias").float()
            self._expert_cache[key] = (
                gu[0::2],
                gu[1::2],
                dn,
                gub[0::2],
                gub[1::2],
                get("down_proj_bias").float(),
            )
        return self._expert_cache[key]

    def final_norm(self):
        return self.f("model.norm.weight")


def rel(a, b):
    return float((a - b).norm() / b.norm()) if b.norm() > 0 else float((a - b).norm())


def check_weights(layers):
    q, h = g.GptOssQ4nx(), HfWeights()
    ok = True
    for layer in layers:
        aq, ah = q.attn(layer), h.attn(layer)
        for k in aq:
            r = rel(aq[k], ah[k])
            print(f"L{layer} attn.{k:6s} rel {r:.4f}")
            # biases/sinks are bf16 copies; projections are Q4_1 requantised
            ok &= r < (1e-6 if k.startswith("b") or k == "sinks" else 0.15)
        for a, b, n in zip(q.norms(layer), h.norms(layer), ("norm1", "norm2")):
            ok &= torch.equal(a, b)
            print(f"L{layer} {n} exact {torch.equal(a, b)}")
        (wq, bq), (wh, bh) = q.router(layer), h.router(layer)
        print(
            f"L{layer} router exact {torch.equal(wq, wh)} bias exact {torch.equal(bq, bh)}"
        )
        ok &= torch.equal(wq, wh) and torch.equal(bq, bh)
        # decode-layout router copy: [45 col blocks, (2 tiles x 64 cols), 16 rows]
        rd = q.raw(f"model.layers.{layer}.mlp.router.weight").float()
        rd = rd.reshape(45, 2, 64, 16).permute(1, 3, 0, 2).reshape(32, 2880)
        print(f"L{layer} router decode layout exact {torch.equal(rd, wh)}")
        ok &= torch.equal(rd, wh)
        for e in range(g.N_EXPERTS):
            eq, eh = q.expert(layer, e), h.expert(layer, e)
            same = [torch.equal(a, b) for a, b in zip(eq, eh)]
            if not all(same):
                print(
                    f"L{layer} expert {e} exact {same} rel {[round(rel(a, b), 5) for a, b in zip(eq, eh)]}"
                )
                ok = False
        print(f"L{layer} experts checked ({g.N_EXPERTS})")
        # the bundle also ships the expert biases as plain tensors
        p = f"model.layers.{layer}.mlp.experts."
        for n, i in (("gate_proj_bias", 3), ("up_proj_bias", 4), ("down_proj_bias", 5)):
            plain = q.vec(p + n)
            same = all(
                torch.equal(
                    plain[e, : q.expert(layer, e)[i].numel()], q.expert(layer, e)[i]
                )
                for e in range(g.N_EXPERTS)
            )
            print(f"L{layer} {n} plain==in-block {same}")
            ok &= same
    if layers:
        r = rel(q.lm_head(), h.lm_head())
        print(f"lm_head rel {r:.4f}")
        ok &= r < 0.15
        e = torch.arange(0, g.VOCAB, 997)
        ok &= torch.equal(q.embed(e), h.embed(e))
        print(f"embed exact {torch.equal(q.embed(e), h.embed(e))}")
    print("WEIGHTS", "PASS" if ok else "FAIL")
    return ok


def cos(a, b):
    return float(
        torch.nn.functional.cosine_similarity(
            a.flatten().double(), b.flatten().double(), dim=0
        )
    )


def run_ref(w, ids, decode_steps):
    """Prefill `ids`, then greedily decode; returns per-step logits."""
    ref = g.Reference(w)
    out = [ref.forward(ids)]
    for _ in range(decode_steps):
        out.append(ref.forward([int(out[-1].argmax())]))
    return out


def check_forward(prompt, decode_steps):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    path = hf_dir()
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(prompt)["input_ids"]
    print(f"prompt {prompt!r} -> {len(ids)} tokens")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True)
    ).eval()
    hf_steps, cur = [], list(ids)
    with torch.no_grad():
        for _ in range(decode_steps + 1):
            logit = model(torch.tensor([cur])).logits[0, -1].float()
            hf_steps.append(logit)
            cur.append(int(logit.argmax()))
    del model
    print(f"HF bf16 done in {time.time() - t0:.0f}s: {tok.decode(cur[len(ids):])!r}")

    ok = True
    for name, w in (
        ("ref(HF weights)", HfWeights(path)),
        ("ref(FLM q4nx)", g.GptOssQ4nx()),
    ):
        t0 = time.time()
        steps = run_ref(w, ids, decode_steps)
        toks = [int(s.argmax()) for s in steps]
        print(f"{name} {time.time() - t0:.0f}s: {tok.decode(toks)!r}")
        for i, (a, b) in enumerate(zip(steps, hf_steps)):
            print(
                f"  step {i}: top1 {int(a.argmax())} vs HF {int(b.argmax())}  cos {cos(a, b):.5f}"
            )
        # each step is fed its own greedy token, so compare only up to the first divergence
        ok &= toks[0] == int(hf_steps[0].argmax())
    print("FORWARD", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["weights", "forward"])
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 11, 23])
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--decode-steps", type=int, default=4)
    a = ap.parse_args()
    good = (
        check_weights(a.layers)
        if a.what == "weights"
        else check_forward(a.prompt, a.decode_steps)
    )
    sys.exit(0 if good else 1)
