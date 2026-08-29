#!/bin/bash
# bench_mha.sh — Sweep MHA causal benchmark across sequence lengths on NPU2
#
# Usage: ./bench_mha.sh
#
# Environment variables (optional overrides):
#   WARMUP=10        Number of warmup iterations per config
#   ITERATIONS=20    Number of measurement iterations per config
#   SEQ_LENGTHS      Space-separated list (default: "256 512 1024 2048 4096 8192 16384 32768")
#
# Requires: PEANO_INSTALL_DIR, XILINX_XRT set; aie-opt on PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Fixed attention parameters
DK=64
DV=64
LKP=64
LQP=256
NUM_HEADS=12
NUM_KV_HEADS=12

# Benchmark parameters
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-20}"
SEQ_LENGTHS=(${SEQ_LENGTHS:-256 512 1024 2048 4096 8192 16384 32768})

# Build directory (Peano-based)
BUILD_DIR="build_peano"

# Output CSV
CSV_FILE="bench_mha_results.csv"

# Result arrays
declare -a R_SEQ R_AVG_TIME R_MIN_TIME R_MAX_TIME R_AVG_GFLOPS R_PEAK_GFLOPS

echo "============================================================"
echo " MHA Causal Benchmark Sweep"
echo " num_heads=$NUM_HEADS, dk=$DK, dv=$DV, causal=true"
echo " Warmup=$WARMUP, Iterations=$ITERATIONS"
echo " Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "============================================================"
echo ""

# Step 1: Compile kernel .o once (tile sizes are constant across seq lengths)
echo ">>> Compiling kernel (attn.o) ..."
make -f "$SCRIPT_DIR/Makefile" compile-kernel \
    DK=$DK DV=$DV LKP=$LKP LQP=$LQP
echo ""

# Step 2: Build test_elf.exe once
echo ">>> Building test_elf.exe ..."
make -f "$SCRIPT_DIR/Makefile" build-test-exe
echo ""

# Step 3: Sweep sequence lengths
for S in "${SEQ_LENGTHS[@]}"; do
    echo "============================================================"
    echo " LQ=LK=$S"
    echo "============================================================"

    # Compile MLIR → ELF (compile-only, causal)
    echo ">>> Compiling MLIR for seq_len=$S ..."
    cd "$BUILD_DIR"
    PEANO_INSTALL_DIR="${PEANO_INSTALL_DIR}" python3 "$SCRIPT_DIR/attn.py" \
        --lk "$S" --lq "$S" --lkp "$LKP" --lqp "$LQP" \
        --dk "$DK" --dv "$DV" \
        --num-heads "$NUM_HEADS" --num-kv-heads "$NUM_KV_HEADS" \
        --compile-mode compile-only --causal
    cd ..

    # Run profiler
    echo ">>> Profiling seq_len=$S (warmup=$WARMUP, iterations=$ITERATIONS) ..."
    LOGFILE="/tmp/bench_mha_${S}.log"
    cd "$BUILD_DIR"
    ./test_elf.exe -e air.elf -k "main:attention_bf16" \
        --lq "$S" --lk "$S" --dk "$DK" --dv "$DV" --num-heads "$NUM_HEADS" \
        -w "$WARMUP" -n "$ITERATIONS" 2>&1 | tee "$LOGFILE"
    cd ..

    # Parse results
    avg_time=$(grep "Avg NPU attention time:" "$LOGFILE" | awk '{print $5}' | tr -d 'us.')
    min_time=$(grep "Min NPU attention time:" "$LOGFILE" | awk '{print $5}' | tr -d 'us.')
    max_time=$(grep "Max NPU attention time:" "$LOGFILE" | awk '{print $5}' | tr -d 'us.')
    avg_gflops=$(grep "Avg NPU gflops:" "$LOGFILE" | awk '{print $4}')
    peak_gflops=$(grep "Max NPU gflops:" "$LOGFILE" | awk '{print $4}')

    R_SEQ+=("$S")
    R_AVG_TIME+=("$avg_time")
    R_MIN_TIME+=("$min_time")
    R_MAX_TIME+=("$max_time")
    R_AVG_GFLOPS+=("$avg_gflops")
    R_PEAK_GFLOPS+=("$peak_gflops")

    echo ""
done

# Step 4: Print summary table
echo ""
echo "============================================================"
echo " RESULTS: MHA Causal Benchmark (${NUM_HEADS} heads, dk=${DK}, dv=${DV})"
echo "============================================================"
echo ""
printf "%-10s | %15s | %15s | %15s | %12s | %12s\n" \
    "Seq Len" "Avg Time (us)" "Min Time (us)" "Max Time (us)" "Avg GFLOPS" "Peak GFLOPS"
printf "%-10s-|-%15s-|-%15s-|-%15s-|-%12s-|-%12s\n" \
    "----------" "---------------" "---------------" "---------------" "------------" "------------"

for i in "${!R_SEQ[@]}"; do
    printf "%-10s | %15s | %15s | %15s | %12s | %12s\n" \
        "${R_SEQ[$i]}" "${R_AVG_TIME[$i]}" "${R_MIN_TIME[$i]}" "${R_MAX_TIME[$i]}" \
        "${R_AVG_GFLOPS[$i]}" "${R_PEAK_GFLOPS[$i]}"
done

# Step 5: Write CSV
echo "seq_len,avg_time_us,min_time_us,max_time_us,avg_gflops,peak_gflops" > "$CSV_FILE"
for i in "${!R_SEQ[@]}"; do
    echo "${R_SEQ[$i]},${R_AVG_TIME[$i]},${R_MIN_TIME[$i]},${R_MAX_TIME[$i]},${R_AVG_GFLOPS[$i]},${R_PEAK_GFLOPS[$i]}" >> "$CSV_FILE"
done

echo ""
echo "Results saved to $CSV_FILE"
