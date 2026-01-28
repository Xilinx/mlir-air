#!/usr/bin/env bash

##===- utils/aircc-gpu.sh - GPU compilation script for AIR --*- Script -*-===##
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
##===----------------------------------------------------------------------===##
#
# This script compiles AIR MLIR files to GPU (ROCDL) target.
#
# Usage:
#   aircc-gpu.sh [options] <input.mlir>
#
# Options:
#   -o <output>       Output file (default: stdout)
#   --gpu-arch <arch> GPU architecture (default: gfx942)
#   --tmpdir <dir>    Temporary directory (default: /tmp/aircc_gpu_$$)
#   -v, --verbose     Verbose output
#   -h, --help        Show help
#
# Example:
#   ./utils/aircc-gpu.sh --gpu-arch=gfx942 -o output.mlir input.mlir
#
##===----------------------------------------------------------------------===##

set -e

# Default values
GPU_ARCH="gfx942"
GPU_RUNTIME="HIP"
OUTPUT=""
TMPDIR=""
VERBOSE=0
INPUT=""

# Find the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Tool paths - try to find in build directory or PATH
if [ -x "$PROJECT_ROOT/build/bin/air-opt" ]; then
    AIR_OPT="$PROJECT_ROOT/build/bin/air-opt"
else
    AIR_OPT=$(which air-opt 2>/dev/null || echo "")
fi

if [ -x "$PROJECT_ROOT/llvm/install/bin/mlir-opt" ]; then
    MLIR_OPT="$PROJECT_ROOT/llvm/install/bin/mlir-opt"
else
    MLIR_OPT=$(which mlir-opt 2>/dev/null || echo "")
fi

usage() {
    cat << EOF
Usage: $(basename "$0") [options] <input.mlir>

Compile AIR MLIR to GPU (ROCDL) target.

Options:
  -o <output>       Output file (default: stdout)
  --gpu-arch <arch> GPU architecture (default: gfx942)
                    Common values: gfx942 (MI300), gfx90a (MI200), gfx908 (MI100)
  --gpu-runtime <r> GPU runtime: HIP or OpenCL (default: HIP)
  --tmpdir <dir>    Temporary directory for intermediate files
  -v, --verbose     Verbose output
  -h, --help        Show this help message

Examples:
  # Compile for MI300X
  $(basename "$0") --gpu-arch=gfx942 -o output.mlir input.mlir

  # Compile with verbose output
  $(basename "$0") -v --gpu-arch=gfx942 input.mlir

  # Keep intermediate files
  $(basename "$0") --tmpdir=/tmp/mytest input.mlir
EOF
    exit 0
}

log() {
    if [ "$VERBOSE" -eq 1 ]; then
        echo "[aircc-gpu] $*" >&2
    fi
}

error() {
    echo "Error: $*" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o)
            OUTPUT="$2"
            shift 2
            ;;
        --gpu-arch)
            GPU_ARCH="$2"
            shift 2
            ;;
        --gpu-arch=*)
            GPU_ARCH="${1#*=}"
            shift
            ;;
        --gpu-runtime)
            GPU_RUNTIME="$2"
            shift 2
            ;;
        --gpu-runtime=*)
            GPU_RUNTIME="${1#*=}"
            shift
            ;;
        --tmpdir)
            TMPDIR="$2"
            shift 2
            ;;
        --tmpdir=*)
            TMPDIR="${1#*=}"
            shift
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            error "Unknown option: $1"
            ;;
        *)
            if [ -z "$INPUT" ]; then
                INPUT="$1"
            else
                error "Multiple input files not supported"
            fi
            shift
            ;;
    esac
done

# Validate input
if [ -z "$INPUT" ]; then
    error "No input file specified. Use -h for help."
fi

if [ ! -f "$INPUT" ]; then
    error "Input file not found: $INPUT"
fi

# Validate tools
if [ -z "$AIR_OPT" ] || [ ! -x "$AIR_OPT" ]; then
    error "air-opt not found. Please build mlir-air or add it to PATH."
fi

if [ -z "$MLIR_OPT" ] || [ ! -x "$MLIR_OPT" ]; then
    error "mlir-opt not found. Please build LLVM or add it to PATH."
fi

# Create temporary directory
if [ -z "$TMPDIR" ]; then
    TMPDIR=$(mktemp -d /tmp/aircc_gpu_XXXXXX)
    CLEANUP_TMPDIR=1
else
    mkdir -p "$TMPDIR"
    CLEANUP_TMPDIR=0
fi

cleanup() {
    if [ "$CLEANUP_TMPDIR" -eq 1 ] && [ -d "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi
}
trap cleanup EXIT

log "Input: $INPUT"
log "GPU arch: $GPU_ARCH"
log "GPU runtime: $GPU_RUNTIME"
log "Temp dir: $TMPDIR"
log "air-opt: $AIR_OPT"
log "mlir-opt: $MLIR_OPT"

# Step 1: AIR to ROCDL
STEP1_OUT="$TMPDIR/01_rocdl.mlir"
log "Step 1: AIR to ROCDL dialect"
"$AIR_OPT" "$INPUT" \
    -air-to-rocdl \
    -canonicalize -cse \
    -o "$STEP1_OUT"

# Step 2: Lower affine and outline GPU kernels
STEP2_OUT="$TMPDIR/02_gpu_outlined.mlir"
log "Step 2: Lower affine and outline GPU kernels"
"$MLIR_OPT" "$STEP1_OUT" \
    --pass-pipeline="builtin.module(func.func(lower-affine, convert-scf-to-cf), gpu-kernel-outlining)" \
    -o "$STEP2_OUT"

# Step 3: Convert to ROCDL and generate binary
STEP3_OUT="$TMPDIR/03_rocdl_binary.mlir"
log "Step 3: Convert to ROCDL and generate GPU binary"
"$MLIR_OPT" "$STEP2_OUT" \
    --pass-pipeline="builtin.module(rocdl-attach-target{chip=$GPU_ARCH O=3}, gpu.module(convert-gpu-to-rocdl{chipset=$GPU_ARCH runtime=$GPU_RUNTIME}, reconcile-unrealized-casts), gpu-module-to-binary)" \
    -o "$STEP3_OUT"

# Step 4: Final lowering to LLVM
STEP4_OUT="$TMPDIR/04_llvm.mlir"
log "Step 4: Final lowering to LLVM"
"$MLIR_OPT" "$STEP3_OUT" \
    --pass-pipeline="builtin.module(func.func(gpu-async-region), gpu-to-llvm, convert-to-llvm, reconcile-unrealized-casts)" \
    -o "$STEP4_OUT"

# Output
if [ -n "$OUTPUT" ]; then
    cp "$STEP4_OUT" "$OUTPUT"
    log "Output written to: $OUTPUT"
else
    cat "$STEP4_OUT"
fi

log "GPU compilation complete"
if [ "$VERBOSE" -eq 1 ]; then
    echo "Intermediate files in: $TMPDIR" >&2
    echo "  Step 1 (ROCDL): $STEP1_OUT" >&2
    echo "  Step 2 (GPU outlined): $STEP2_OUT" >&2
    echo "  Step 3 (ROCDL binary): $STEP3_OUT" >&2
    echo "  Step 4 (LLVM): $STEP4_OUT" >&2
fi
