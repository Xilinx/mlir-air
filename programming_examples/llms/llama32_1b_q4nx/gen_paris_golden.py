#!/usr/bin/env python3
# Regenerate the HF golden that run_paris_gen.py consumes at /tmp/paris_golden:
#   weights/lm_head.f32.bin      [VOCAB_SIZE, K]  f32
#   weights/final_norm.f32.bin   [K]              f32
#   weights/embed_tokens.f32.bin [VOCAB_SIZE, K]  f32
#   rope_cos32.f32.bin           [32, head_dim]   f32   (positions 0..31)
#   rope_sin32.f32.bin           [32, head_dim]   f32
#   meta.json                    {prompt_ids, hf_greedy_ids}
# Run with a HuggingFace-enabled python venv (transformers + torch), HF_HUB_OFFLINE=1.
import os, json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("HF_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
OUT = os.environ.get("PARIS_GOLDEN", "/tmp/paris_golden")
PROMPT = os.environ.get("PARIS_PROMPT", "The capital of France is")
os.makedirs(OUT + "/weights", exist_ok=True)

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
cfg = m.config
D = cfg.hidden_size
head_dim = getattr(cfg, "head_dim", D // cfg.num_attention_heads)
print(f"[golden] D={D} head_dim={head_dim} vocab={cfg.vocab_size}", flush=True)

# --- weights (f32, row-major) ---
m.lm_head.weight.detach().to(torch.float32).numpy().tofile(
    OUT + "/weights/lm_head.f32.bin"
)
m.model.norm.weight.detach().to(torch.float32).numpy().tofile(
    OUT + "/weights/final_norm.f32.bin"
)
m.model.embed_tokens.weight.detach().to(torch.float32).numpy().tofile(
    OUT + "/weights/embed_tokens.f32.bin"
)

# --- rope cos/sin for positions 0..31 via the model's own rotary embedding ---
# (guarantees the exact theta/scaling HF uses; rotate-half convention, dim=head_dim)
pos = torch.arange(32).unsqueeze(0)  # [1, 32]
dummy = torch.zeros(1, 32, D)
with torch.no_grad():
    cos, sin = m.model.rotary_emb(dummy, pos)  # each [1, 32, head_dim]
cos = cos[0].to(torch.float32).numpy().astype(np.float32)  # [32, head_dim]
sin = sin[0].to(torch.float32).numpy().astype(np.float32)
assert cos.shape == (32, head_dim), cos.shape
cos.tofile(OUT + "/rope_cos32.f32.bin")
sin.tofile(OUT + "/rope_sin32.f32.bin")

# --- confirm HF greedy emits "Paris" and record golden ids ---
ids = tok(PROMPT, return_tensors="pt").input_ids
with torch.no_grad():
    g = m.generate(ids, max_new_tokens=8, do_sample=False)
cont = g[0, ids.shape[1] :].tolist()
meta = {
    "prompt": PROMPT,
    "prompt_ids": ids[0].tolist(),
    "hf_greedy_ids": cont,
    "hf_greedy_text": tok.decode(cont),
}
json.dump(meta, open(OUT + "/meta.json", "w"), indent=2)
print(f"[golden] prompt_ids = {meta['prompt_ids']}", flush=True)
print(f"[golden] HF greedy  = {cont}  -> {meta['hf_greedy_text']!r}", flush=True)
print(
    f"[golden] first token {cont[0]} {'== 12366 (Paris) OK' if cont[0]==12366 else '!= 12366'}",
    flush=True,
)
print(f"[golden] wrote {OUT}", flush=True)
