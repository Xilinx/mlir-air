#!/usr/bin/env bash
#===- run_all.sh -----------------------------------*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
#===------------------------------------------------------------------===//
#
# Run every GPU execution test and decide pass/fail, rather than printing
# numbers for someone to read. Each test reports a count of things that came
# out wrong; anything but zero fails, and a test that prints no count at all
# fails too -- a silent run is a run that did not check anything.
#
# Needs a GPU. Set up the environment first:
#
#   export MLIR_AIR_INSTALL_DIR=<mlir-air build dir>
#   export LLVM_INSTALL_DIR=<llvm install dir>
#   export PATH=$MLIR_AIR_INSTALL_DIR/bin:$LLVM_INSTALL_DIR/bin:$PATH
#   test/gpu/run_all.sh
#
#===------------------------------------------------------------------===//

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMEOUT="${TIMEOUT:-600}"

# The 4k GEMM says "Output Matched!" instead of counting; everything else ends
# its output with "<something> = <count>".
declare -A EXPECT=( [4k_4k_mul]="Output Matched!" )

# A test name is a directory under test/gpu/ unless it appears here, which lets
# one generator be run in more than one configuration. All three have to be in
# the suite: M = 1 and M > 1 take different paths through the piece
# decomposition, and only a run with more than one decode step exercises the
# iteration versioning that lets one launch reuse the same task graph.
declare -A DIR=( [megakernel_gen_prefill]=megakernel_gen
                 [megakernel_gen_decode]=megakernel_gen
                 [megakernel_gen_chunked]=megakernel_gen )
declare -A ENVV=( [megakernel_gen_prefill]="TOKENS=4 LAYERS=2"
                  [megakernel_gen_decode]="STEPS=3 TOKENS=2 LAYERS=2"
                  # A 5-token prompt in a window of 2: the steps carry 2, 2, 1,
                  # 1, 1 tokens off one static task graph. The 1 in the middle
                  # is the ragged tail of the prompt, so three different
                  # counts, not just "prefill then decode".
                  [megakernel_gen_chunked]="STEPS=5 TOKENS=2 LAYERS=2 PROMPT_LEN=5" )

TESTS=(
  4k_4k_mul
  chiplet_identity
  chiplet_cooperative
  persistent_worker
  two_level_signal
  task_queue
  event_gated
  megakernel_gemm
  megakernel_layers
  megakernel_decode
  megakernel_attention
  megakernel_gen
  megakernel_gen_prefill
  megakernel_gen_decode
  megakernel_gen_chunked
  gang_task
  scheduler_broadcast
  gang_mmajor
)

fails=0
for t in "${TESTS[@]}"; do
  printf '%-24s ' "$t"
  log=$(mktemp)
  if ! env TMPDIR="${TMPDIR_BASE:-/tmp}/air_runall_$t" ${ENVV[$t]:-} timeout "$TIMEOUT" \
        bash "$SCRIPT_DIR/${DIR[$t]:-$t}/run.sh" > "$log" 2>&1; then
    echo "FAIL (runner exited nonzero; $log)"
    fails=$((fails + 1))
    continue
  fi
  if [ -n "${EXPECT[$t]:-}" ]; then
    if grep -qF "${EXPECT[$t]}" "$log"; then
      echo "pass"
      rm -f "$log"
    else
      echo "FAIL (expected '${EXPECT[$t]}'; $log)"
      fails=$((fails + 1))
    fi
    continue
  fi
  # Last line is "<label> = <count>"; the count must be zero.
  last=$(tail -1 "$log")
  count=$(printf '%s' "$last" | sed -n 's/.*= *\([0-9][0-9]*\) *$/\1/p')
  if [ -z "$count" ]; then
    echo "FAIL (no count in output, so nothing was checked; $log)"
    fails=$((fails + 1))
  elif [ "$count" != "0" ]; then
    echo "FAIL ($last)"
    fails=$((fails + 1))
  else
    echo "pass"
    rm -f "$log"
  fi
done

echo
if [ "$fails" -eq 0 ]; then
  echo "all ${#TESTS[@]} GPU tests passed"
else
  echo "$fails of ${#TESTS[@]} GPU tests failed"
fi
exit $((fails > 0))
