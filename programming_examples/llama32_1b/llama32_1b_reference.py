# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU reference implementation of LLAMA-3.2-1B forward pass.

Pure NumPy in F32 for numerical verification against NPU results.
All intermediate computations are done in F32 (weights are cast from BF16
at use time) to provide a high-accuracy reference.

LLAMA-3.2-1B config:
  16 layers, emb_dim=2048, n_heads=32, head_dim=64, n_kv_heads=8,
  hidden_dim=8192, vocab_size=128256, BF16, rope_base=500000
"""

import argparse
import numpy as np
from ml_dtypes import bfloat16

from llama32_1b_weights import (
    LlamaConfig,
    LayerWeights,
    LlamaWeights,
    load_weights,
    generate_rope_lut,
)


def rms_norm(x, weight, eps=1e-5):
    """RMS normalization: x / sqrt(mean(x^2) + eps) * weight.

    Args:
        x: (M, N) input array in F32.
        weight: (N,) learned scale parameter.
        eps: Small constant for numerical stability.

    Returns:
        (M, N) normalized and scaled array in F32.
    """
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    # Compute RMS per row
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def apply_rope(x, lut):
    """Apply Rotary Position Embedding using a precomputed LUT.

    Uses half-split convention (matching HuggingFace Llama):
    pairs (x[i], x[i + dim//2]) with rotation angle theta_i.

    LUT layout: [cos_0, ..., cos_{half-1}, sin_0, ..., sin_{half-1}]

    Args:
        x: (seq_len, head_dim) input for one head.
        lut: (seq_len, head_dim) with concatenated [cos..., sin...].

    Returns:
        (seq_len, head_dim) with RoPE applied.
    """
    x = np.asarray(x, dtype=np.float32)
    lut = np.asarray(lut, dtype=np.float32)
    dim = x.shape[-1]
    half = dim // 2

    cos_vals = lut[:, :half]
    sin_vals = lut[:, half:]

    x1 = x[:, :half]
    x2 = x[:, half:]

    out = np.empty_like(x)
    out[:, :half] = x1 * cos_vals - x2 * sin_vals
    out[:, half:] = x1 * sin_vals + x2 * cos_vals
    return out


def silu(x):
    """SiLU activation: x * sigmoid(x).

    Args:
        x: Input array (any shape) in F32.

    Returns:
        SiLU-activated array with the same shape.
    """
    x = np.asarray(x, dtype=np.float32)
    return x * (1.0 / (1.0 + np.exp(-x)))


def swiglu(gate, up):
    """SwiGLU gating: SiLU(gate) * up.

    Args:
        gate: Gate input array in F32.
        up: Up-projection input array in F32.

    Returns:
        Element-wise SiLU(gate) * up.
    """
    return silu(gate) * np.asarray(up, dtype=np.float32)


def ffn_full_reference(x, ffn_norm_weight, w_gate, w_up, w_down, eps=1e-5):
    """CPU F32 reference for the full FFN block:
    RMSNorm -> Gate -> Up -> SwiGLU -> Down -> Residual Add.

    Args:
        x: (seq_len, emb_dim) input (residual state)
        ffn_norm_weight: (emb_dim,) RMSNorm weight
        w_gate: (emb_dim, hidden_dim) gate projection weight
        w_up: (emb_dim, hidden_dim) up projection weight
        w_down: (hidden_dim, emb_dim) down projection weight
        eps: RMSNorm epsilon

    Returns:
        (seq_len, emb_dim) bfloat16: x + down_proj(SwiGLU(gate, up))
    """
    x_f32 = x.astype(np.float32)
    normed = rms_norm(x_f32, ffn_norm_weight, eps)
    gate = normed @ w_gate.astype(np.float32)
    up = normed @ w_up.astype(np.float32)
    down = swiglu(gate, up) @ w_down.astype(np.float32)
    return (x_f32 + down).astype(bfloat16)


def softmax(x, axis=-1):
    """Numerically stable softmax.

    Args:
        x: Input array in F32.
        axis: Axis along which to compute softmax.

    Returns:
        Softmax probabilities with the same shape as x.
    """
    x = np.asarray(x, dtype=np.float32)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def attention_reference(q, k, v, n_heads, n_kv_heads):
    """Multi-head attention with Grouped Query Attention (GQA).

    Args:
        q: (seq_len, n_heads * head_dim) -- already projected and RoPE'd.
        k: (seq_len, n_kv_heads * head_dim) -- already projected and RoPE'd.
        v: (seq_len, n_kv_heads * head_dim) -- already projected.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads (for GQA).

    Returns:
        (seq_len, n_heads * head_dim) attention output.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)

    seq_len = q.shape[0]
    head_dim = q.shape[1] // n_heads
    group_size = n_heads // n_kv_heads

    # Reshape to per-head views
    # q: (seq_len, n_heads, head_dim) -> (n_heads, seq_len, head_dim)
    q = q.reshape(seq_len, n_heads, head_dim).transpose(1, 0, 2)
    # k: (seq_len, n_kv_heads, head_dim) -> (n_kv_heads, seq_len, head_dim)
    k = k.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)
    # v: (seq_len, n_kv_heads, head_dim) -> (n_kv_heads, seq_len, head_dim)
    v = v.reshape(seq_len, n_kv_heads, head_dim).transpose(1, 0, 2)

    scale = 1.0 / np.sqrt(head_dim)

    # Causal mask: mask[i][j] = 0 if j <= i, else -inf
    causal_mask = np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)

    # Compute attention for each query head
    out_heads = np.empty((n_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(n_heads):
        kv_idx = h // group_size
        # scores: (seq_len, seq_len)
        scores = q[h] @ k[kv_idx].T * scale
        scores = scores + causal_mask
        probs = softmax(scores, axis=-1)
        out_heads[h] = probs @ v[kv_idx]

    # Reshape back: (n_heads, seq_len, head_dim) -> (seq_len, n_heads * head_dim)
    out = out_heads.transpose(1, 0, 2).reshape(seq_len, n_heads * head_dim)
    return out


def transformer_block(x, layer_weights, rope_lut, config):
    """Single transformer block with attention and FFN.

    Args:
        x: (seq_len, emb_dim) input in F32.
        layer_weights: LayerWeights for this layer.
        rope_lut: (seq_len, head_dim) RoPE lookup table.
        config: LlamaConfig with model hyperparameters.

    Returns:
        (output, intermediates) where output is (seq_len, emb_dim) in F32
        and intermediates is a dict mapping step names to arrays.
    """
    x = np.asarray(x, dtype=np.float32)
    intermediates = {}
    seq_len = x.shape[0]
    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim

    # --- Self-attention ---

    # 1. Pre-attention RMS norm
    normed = rms_norm(x, layer_weights.attn_norm)
    intermediates["attn_norm"] = normed

    # 2-4. QKV projections
    wq = np.asarray(layer_weights.wq, dtype=np.float32)
    wk = np.asarray(layer_weights.wk, dtype=np.float32)
    wv = np.asarray(layer_weights.wv, dtype=np.float32)
    q = normed @ wq  # (seq_len, n_heads * head_dim) = (seq_len, 2048)
    k = normed @ wk  # (seq_len, n_kv_heads * head_dim) = (seq_len, 512)
    v = normed @ wv  # (seq_len, n_kv_heads * head_dim) = (seq_len, 512)
    intermediates["q"] = q
    intermediates["k"] = k
    intermediates["v"] = v

    # 5. Apply RoPE to Q (per-head)
    # Reshape Q: (seq_len, n_heads, head_dim) -> process each head independently
    q_heads = q.reshape(seq_len, n_heads, head_dim)
    q_roped_heads = np.empty_like(q_heads)
    for h in range(n_heads):
        q_roped_heads[:, h, :] = apply_rope(
            q_heads[:, h, :].reshape(seq_len, head_dim), rope_lut[:seq_len]
        )
    q_roped = q_roped_heads.reshape(seq_len, n_heads * head_dim)
    intermediates["q_roped"] = q_roped

    # 6. Apply RoPE to K (per-head)
    k_heads = k.reshape(seq_len, n_kv_heads, head_dim)
    k_roped_heads = np.empty_like(k_heads)
    for h in range(n_kv_heads):
        k_roped_heads[:, h, :] = apply_rope(
            k_heads[:, h, :].reshape(seq_len, head_dim), rope_lut[:seq_len]
        )
    k_roped = k_roped_heads.reshape(seq_len, n_kv_heads * head_dim)
    intermediates["k_roped"] = k_roped

    # 7. Attention
    attn_out = attention_reference(q_roped, k_roped, v, n_heads, n_kv_heads)
    intermediates["attn_out"] = attn_out

    # 8. Output projection
    wo = np.asarray(layer_weights.wo, dtype=np.float32)
    proj = attn_out @ wo  # (seq_len, emb_dim)
    intermediates["proj"] = proj

    # 9. Residual connection
    res1 = x + proj
    intermediates["res1"] = res1

    # --- Feed-forward network ---

    # 10. Pre-FFN RMS norm
    normed2 = rms_norm(res1, layer_weights.ffn_norm)
    intermediates["ffn_norm"] = normed2

    # 11-12. Gate and Up projections
    w_gate = np.asarray(layer_weights.w_gate, dtype=np.float32)
    w_up = np.asarray(layer_weights.w_up, dtype=np.float32)
    gate = normed2 @ w_gate  # (seq_len, hidden_dim) = (seq_len, 8192)
    up = normed2 @ w_up  # (seq_len, hidden_dim) = (seq_len, 8192)
    intermediates["gate"] = gate
    intermediates["up"] = up

    # 13. SwiGLU activation
    swiglu_out = swiglu(gate, up)
    intermediates["swiglu"] = swiglu_out

    # 14. Down projection
    w_down = np.asarray(layer_weights.w_down, dtype=np.float32)
    down = swiglu_out @ w_down  # (seq_len, emb_dim) = (seq_len, 2048)
    intermediates["down"] = down

    # 15. Residual connection
    output = res1 + down
    intermediates["output"] = output

    return output, intermediates


def forward(token_ids, weights, config, rope_lut=None):
    """Full LLAMA-3.2-1B forward pass.

    Args:
        token_ids: (seq_len,) integer array of token IDs.
        weights: LlamaWeights containing all model parameters.
        config: LlamaConfig with model hyperparameters.
        rope_lut: Optional precomputed (seq_len, head_dim) RoPE LUT.
            If None, one will be generated using generate_rope_lut.

    Returns:
        logits: (seq_len, vocab_size) in F32.
    """
    seq_len = len(token_ids)

    # Generate RoPE LUT if not provided
    if rope_lut is None:
        rope_lut = generate_rope_lut(config=config, seq_len=seq_len)
    rope_lut = np.asarray(rope_lut, dtype=np.float32)

    # 1. Token embedding (CPU lookup)
    embed_table = np.asarray(weights.embed_table, dtype=np.float32)
    x = embed_table[token_ids]  # (seq_len, emb_dim)

    # 2. Transformer blocks
    for i in range(config.n_layers):
        x, _ = transformer_block(x, weights.layers[i], rope_lut, config)

    # 3. Final RMS norm
    x = rms_norm(x, weights.final_norm)

    # 4. Language model head (CPU GEMM)
    lm_head = np.asarray(weights.lm_head, dtype=np.float32)
    logits = x @ lm_head.T  # (seq_len, vocab_size)

    return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CPU reference forward pass for LLAMA-3.2-1B"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or local path (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Input prompt (default: 'The capital of France is')",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Sequence length to pad/truncate to (default: 128)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare output against HuggingFace transformers reference",
    )
    args = parser.parse_args()

    # Load weights
    config = LlamaConfig()
    print(f"Loading weights from {args.model}...")
    weights = load_weights(args.model, config=config)
    print(f"  Config: {config}")
    print(
        f"  Layers: {config.n_layers}, emb_dim: {config.emb_dim}, "
        f"n_heads: {config.n_heads}, n_kv_heads: {config.n_kv_heads}, "
        f"hidden_dim: {config.hidden_dim}, vocab_size: {config.vocab_size}"
    )

    # Tokenize
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Token IDs ({len(token_ids)} tokens): {token_ids}")

    # Pad or truncate to seq_len
    if len(token_ids) > args.seq_len:
        token_ids = token_ids[: args.seq_len]
        print(f"Truncated to {args.seq_len} tokens")
    elif len(token_ids) < args.seq_len:
        # Pad with EOS token (or 0 if no EOS)
        pad_token = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        original_len = len(token_ids)
        token_ids = token_ids + [pad_token] * (args.seq_len - len(token_ids))
        print(
            f"Padded from {original_len} to {args.seq_len} tokens "
            f"(pad_token={pad_token})"
        )

    token_ids = np.array(token_ids, dtype=np.int64)

    # Run forward pass
    print(f"\nRunning forward pass (seq_len={args.seq_len})...")
    logits = forward(token_ids, weights, config)
    print(f"Output logits shape: {logits.shape}")

    # Get the prediction at the last real token position
    # (the position just before padding starts, or the last position if no padding)
    prompt_len = len(tokenizer.encode(args.prompt))
    pred_pos = min(prompt_len - 1, args.seq_len - 1)

    # Top-5 predicted next tokens
    next_token_logits = logits[pred_pos]
    top5_indices = np.argsort(next_token_logits)[-5:][::-1]
    top5_probs = softmax(next_token_logits)

    print(f"\nTop-5 predicted next tokens (position {pred_pos}):")
    for rank, idx in enumerate(top5_indices):
        token_str = tokenizer.decode([idx])
        prob = top5_probs[idx]
        print(
            f"  {rank + 1}. '{token_str}' (id={idx}, logit={next_token_logits[idx]:.4f}, "
            f"prob={prob:.4f})"
        )

    # Optional: verify against HuggingFace transformers
    if args.verify:
        print("\n--- Verification against HuggingFace transformers ---")
        try:
            import torch
            from transformers import AutoModelForCausalLM

            print("Loading HuggingFace model...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float32
            )
            hf_model.eval()

            with torch.no_grad():
                input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
                hf_output = hf_model(input_ids)
                hf_logits = hf_output.logits[0].numpy()  # (seq_len, vocab_size)

            print(f"HF logits shape: {hf_logits.shape}")
            print(f"Our logits shape: {logits.shape}")

            # Compare at the prediction position
            our_next = logits[pred_pos]
            hf_next = hf_logits[pred_pos]

            # Absolute and relative error
            abs_diff = np.abs(our_next - hf_next)
            max_abs_err = np.max(abs_diff)
            mean_abs_err = np.mean(abs_diff)

            # Relative error (avoid division by zero)
            denom = np.maximum(np.abs(hf_next), 1e-8)
            rel_diff = abs_diff / denom
            max_rel_err = np.max(rel_diff)
            mean_rel_err = np.mean(rel_diff)

            print(f"\nError at position {pred_pos}:")
            print(f"  Max  absolute error: {max_abs_err:.6f}")
            print(f"  Mean absolute error: {mean_abs_err:.6f}")
            print(f"  Max  relative error: {max_rel_err:.6f}")
            print(f"  Mean relative error: {mean_rel_err:.6f}")

            # Check if top-1 predictions match
            our_top1 = np.argmax(our_next)
            hf_top1 = np.argmax(hf_next)
            match = our_top1 == hf_top1
            print(f"\nTop-1 prediction match: {'YES' if match else 'NO'}")
            print(f"  Ours: '{tokenizer.decode([our_top1])}' (id={our_top1})")
            print(f"  HF:   '{tokenizer.decode([hf_top1])}' (id={hf_top1})")

            # Overall logits correlation
            correlation = np.corrcoef(our_next, hf_next)[0, 1]
            print(f"  Logits correlation: {correlation:.8f}")

            if match and correlation > 0.999:
                print("\nVERIFICATION PASSED")
            else:
                print("\nVERIFICATION FAILED")

        except ImportError as e:
            print(f"Cannot verify: {e}")
            print("Install torch and transformers: pip install torch transformers")
