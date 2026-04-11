#!/bin/bash
# Launch symmetric heap test with N ranks (default 2).
# Each rank gets its own GPU via LOCAL_RANK.
#
# Usage: bash run_symmetric_heap_test.sh [num_ranks]

set -e

NUM_RANKS=${1:-2}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_BIN="${SCRIPT_DIR}/../../runtime_lib/airgpu/test_symmetric_heap"

if [ ! -f "$TEST_BIN" ]; then
  echo "Error: test binary not found at $TEST_BIN"
  echo "Build it first: cd runtime_lib/airgpu && make test_symmetric_heap"
  exit 1
fi

echo "=== Symmetric heap test: ${NUM_RANKS} ranks ==="

PIDS=()
PASS=1

for i in $(seq 0 $((NUM_RANKS - 1))); do
  RANK=$i WORLD_SIZE=$NUM_RANKS LOCAL_RANK=$i \
    "$TEST_BIN" 2>&1 | sed "s/^/[rank $i] /" &
  PIDS+=($!)
done

for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    PASS=0
  fi
done

if [ $PASS -eq 1 ]; then
  echo "=== ALL ${NUM_RANKS} RANKS PASSED ==="
else
  echo "=== SOME RANKS FAILED ==="
  exit 1
fi
