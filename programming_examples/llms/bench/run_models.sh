#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Nightly perf driver: for each model, run compile -> verify -> profile and
# extract per-model perf JSON. Processes ALL models (does not abort on the
# first failure) so a run reports every model's status, then exits non-zero
# if any model's compile or verify failed (strict gate). Perf parsing is
# best-effort: a missing metric is recorded as null, not a failure.
#
# Usage:  run_models.sh "<model1> <model2> ..." <output_dir>
# Env:    PEANO_INSTALL_DIR (required), BENCH_N_TOKENS, BENCH_PROMPT,
#         BENCH_RUNNER, BENCH_AIR_SHA, BENCH_AIE_HASH, BENCH_PEANO_VER
#
# Note: intentionally no `set -e` — per-model failures are handled explicitly.
set -uo pipefail

MODELS="${1:?usage: run_models.sh \"<models>\" <output_dir>}"
OUT="${2:?usage: run_models.sh \"<models>\" <output_dir>}"

SRCDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLMS_DIR="$(dirname "$SRCDIR")"
EXTRACT="$SRCDIR/extract_perf.py"

: "${BENCH_N_TOKENS:=128}"
: "${BENCH_PROMPT:=What is the capital of France?}"
: "${BENCH_RUNNER:=}"
: "${BENCH_AIR_SHA:=}"
: "${BENCH_AIE_HASH:=}"
: "${BENCH_PEANO_VER:=}"

mkdir -p "$OUT"
overall_fail=0

emit_perf() {  # <model> <verify-status> <profile-file-or-/dev/null>
  python3 "$EXTRACT" "$3" \
    --model "$1" --verify-status "$2" --runner "$BENCH_RUNNER" \
    --mlir-air-sha "$BENCH_AIR_SHA" --mlir-aie-hash "$BENCH_AIE_HASH" \
    --llvm-aie-version "$BENCH_PEANO_VER" \
    --n-tokens "$BENCH_N_TOKENS" --prompt "$BENCH_PROMPT" \
    > "$OUT/$1.perf.json"
}

for model in $MODELS; do
  dir="$LLMS_DIR/$model"
  echo "::group::[$model] compile + verify + profile"
  status="unknown"

  if [ ! -f "$dir/Makefile" ]; then
    echo "[$model] ERROR: no Makefile at $dir"
    status="no-makefile"; overall_fail=1
    emit_perf "$model" "fail" /dev/null
    echo "::endgroup::"; echo "[$model] STATUS=$status"; continue
  fi

  # Install this model's Python deps (models may pin different versions).
  [ -f "$dir/requirements.txt" ] && pip install -q -r "$dir/requirements.txt" || true

  if ! timeout 1800 make -C "$dir" compile PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR"; then
    echo "[$model] ERROR: compile failed"
    status="compile-fail"; overall_fail=1
    emit_perf "$model" "fail" /dev/null
    echo "::endgroup::"; echo "[$model] STATUS=$status"; continue
  fi

  # verify: authoritative signal is the "[verify] PASS" marker (same as the
  # lit FileCheck gate), not just the exit code.
  timeout 1800 make -C "$dir" verify PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" \
    2>&1 | tee "$OUT/$model.verify.txt"
  if grep -q '\[verify\] PASS' "$OUT/$model.verify.txt"; then
    verify_status="pass"
  else
    echo "[$model] ERROR: verify did not PASS"
    verify_status="fail"; status="verify-fail"; overall_fail=1
    emit_perf "$model" "fail" /dev/null
    echo "::endgroup::"; echo "[$model] STATUS=$status"; continue
  fi

  # profile: best-effort perf capture (failure here is not job-fatal).
  if timeout 900 make -C "$dir" profile PEANO_INSTALL_DIR="$PEANO_INSTALL_DIR" \
      N_TOKENS="$BENCH_N_TOKENS" PROMPT="$BENCH_PROMPT" \
      2>&1 | tee "$OUT/$model.profile.txt"; then
    status="ok"
    emit_perf "$model" "pass" "$OUT/$model.profile.txt"
  else
    echo "[$model] WARNING: profile failed; recording verify pass with null perf"
    status="profile-fail"
    emit_perf "$model" "pass" /dev/null
  fi

  echo "::endgroup::"
  echo "[$model] STATUS=$status $(python3 -c "import json;d=json.load(open('$OUT/$model.perf.json'))['metrics'];print('ttft_ms='+str(d['ttft_ms']),'tok/s='+str(d['decode_tokens_per_sec']))")"
done

echo "===== per-model summary ====="
for model in $MODELS; do
  [ -f "$OUT/$model.perf.json" ] && python3 -c "
import json;d=json.load(open('$OUT/$model.perf.json'))
print(f\"  {d['model']:<18} verify={d['verify_status']:<5} ttft_ms={d['metrics']['ttft_ms']} tok/s={d['metrics']['decode_tokens_per_sec']}\")"
done

exit "$overall_fail"
