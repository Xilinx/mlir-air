"""Run inference with the cache-level breakdown profiler enabled, then dump
per-stage timing. Used to investigate where per-token host overhead lives.

Usage:
  python3 bench_breakdown.py [--use-int4-decode] [--n-tokens 30]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

# Enable the breakdown profiler by monkey-patching KernelCache's default.
from llama_kernel_builder import cache as _cache_mod

_orig_init = _cache_mod.KernelCache.__init__


def _patched_init(self, cache_dir=None, verbose=False, profiler=None):
    if profiler is None:
        profiler = _cache_mod.Profiler(enabled=True)
    _orig_init(self, cache_dir=cache_dir, verbose=verbose, profiler=profiler)


_cache_mod.KernelCache.__init__ = _patched_init


# Argv shim: build_session expects argparse Namespace.
parser = argparse.ArgumentParser()
parser.add_argument("--use-int4-decode", action="store_true")
parser.add_argument("--n-tokens", type=int, default=30)
parser.add_argument("--prompt", type=str,
                    default="Write a long story about a dragon flying through forests and mountains.")
parser.add_argument("--model", type=str, default="instruct")
args_user = parser.parse_args()


class _Args:
    compile_only = False
    run_only = True
    n_tokens = args_user.n_tokens
    profile = False  # we use breakdown profiler instead
    verify = False
    cpu_attn = False
    verbose = False
    prompt = args_user.prompt
    synthetic_weights = False
    model = args_user.model
    use_int4_decode = args_user.use_int4_decode
    interactive = False


from llama32_1b_inference import build_session, run_once

session = build_session(_Args)

print(f"\n{'='*70}")
print(f"BREAKDOWN BENCH — int4={_Args.use_int4_decode}, n_tokens={_Args.n_tokens}")
print(f"{'='*70}\n")

# Reset breakdown stats so weight-preload calls don't pollute per-token stats.
session.decode_cache.profiler.kernel_breakdowns = {}
session.decode_cache.profiler.kernel_times = {}

t0 = time.perf_counter()
generated, prompt_len_actual = run_once(
    session,
    _Args.prompt,
    n_tokens=_Args.n_tokens,
    profile=False,
    verify=False,
    cpu_attn=False,
    on_token=None,
)
t_total = time.perf_counter() - t0

n_generated = len(generated) - 1
print(f"\nGenerated {n_generated} tokens in {t_total:.2f}s ({n_generated/t_total:.2f} tok/s)")
print(f"Time/token (incl prefill amortized): {t_total/n_generated*1000:.1f} ms")

# Per-token breakdown: kernel_breakdowns aggregates ALL calls (prefill + decode +
# preload). We care only about decode-loop calls. Approx: total calls / n_generated.
session.decode_cache.profiler.report()

# Per-token decode-only breakdown summary
print(f"\n{'='*70}")
print(f"PER-TOKEN DECODE BREAKDOWN (decode_cache only)")
print(f"{'='*70}")
bd = session.decode_cache.profiler.kernel_breakdowns
for name in sorted(bd.keys()):
    entries = bd[name]
    if len(entries) <= 1:
        continue
    # Per-decode-token: assume the last n_generated entries are decode-loop calls
    decode_entries = entries[-n_generated:]
    if not decode_entries:
        continue
    avg_w = sum(e["write_ms"] for e in decode_entries) / len(decode_entries)
    avg_k = sum(e["kernel_ms"] for e in decode_entries) / len(decode_entries)
    avg_r = sum(e["read_ms"] for e in decode_entries) / len(decode_entries)
    print(
        f"  {name:25s} write={avg_w:5.2f}ms  npu_run={avg_k:5.2f}ms  read={avg_r:5.2f}ms  total={avg_w+avg_k+avg_r:5.2f}ms"
    )
