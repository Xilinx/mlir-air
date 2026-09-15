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
)

fails=0
for t in "${TESTS[@]}"; do
  printf '%-24s ' "$t"
  log=$(mktemp)
  if ! TMPDIR="${TMPDIR_BASE:-/tmp}/air_runall_$t" timeout "$TIMEOUT" \
        bash "$SCRIPT_DIR/$t/run.sh" > "$log" 2>&1; then
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
