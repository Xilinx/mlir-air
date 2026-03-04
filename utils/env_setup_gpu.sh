##===- utils/env_setup_gpu.sh - Setup mlir-air env for GPU builds --*- Script -*-===##
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
##===----------------------------------------------------------------------===##
#
# This script sets up the environment to run mlir-air GPU tools.
# It must be sourced, not executed.
#
# Usage:
#   source utils/env_setup_gpu.sh [mlir-air-dir] [llvm-dir]
#
# Arguments (optional - will auto-detect if not provided):
#   mlir-air-dir - Path to mlir-air installation (default: ./install)
#   llvm-dir     - Path to LLVM installation (default: ./llvm/install)
#
# Example:
#   source utils/env_setup_gpu.sh
#   source utils/env_setup_gpu.sh ./install ./llvm/install
#
##===----------------------------------------------------------------------===##

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Set defaults or use provided arguments
MLIR_AIR_DIR="${1:-$PROJECT_DIR/install}"
LLVM_DIR="${2:-$PROJECT_DIR/llvm/install}"

# Convert to absolute paths
MLIR_AIR_DIR="$(realpath "$MLIR_AIR_DIR" 2>/dev/null || echo "$MLIR_AIR_DIR")"
LLVM_DIR="$(realpath "$LLVM_DIR" 2>/dev/null || echo "$LLVM_DIR")"

# Verify directories exist
if [ ! -d "$MLIR_AIR_DIR" ]; then
    echo "ERROR: MLIR-AIR directory not found: $MLIR_AIR_DIR"
    echo "Please build mlir-air first or provide the correct path."
    return 1
fi

if [ ! -d "$LLVM_DIR" ]; then
    echo "ERROR: LLVM directory not found: $LLVM_DIR"
    echo "Please provide the correct LLVM installation path."
    return 1
fi

# Check for Python bindings in build directory (preferred) or install directory
PYTHON_DIR=""
if [ -d "$PROJECT_DIR/build/python/air/_mlir_libs" ]; then
    PYTHON_DIR="$PROJECT_DIR/build/python"
elif [ -d "$MLIR_AIR_DIR/python/air/_mlir_libs" ]; then
    PYTHON_DIR="$MLIR_AIR_DIR/python"
elif [ -d "$MLIR_AIR_DIR/python" ]; then
    # Fallback to install python dir even without _mlir_libs
    PYTHON_DIR="$MLIR_AIR_DIR/python"
fi

if [ -z "$PYTHON_DIR" ]; then
    echo "WARNING: Python bindings not found. aircc.py may not work."
    echo "Checked: $PROJECT_DIR/build/python"
    echo "         $MLIR_AIR_DIR/python"
fi

# Export environment variables
export MLIR_AIR_INSTALL_DIR="$MLIR_AIR_DIR"
export LLVM_INSTALL_DIR="$LLVM_DIR"

# Update PATH
export PATH="$MLIR_AIR_DIR/bin:$LLVM_DIR/bin:$PATH"

# Update PYTHONPATH
if [ -n "$PYTHON_DIR" ]; then
    export PYTHONPATH="$PYTHON_DIR:$PYTHONPATH"
fi

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$MLIR_AIR_DIR/lib:$LLVM_DIR/lib:$LD_LIBRARY_PATH"

echo "=== MLIR-AIR GPU Environment Setup ==="
echo "MLIR_AIR_INSTALL_DIR: $MLIR_AIR_DIR"
echo "LLVM_INSTALL_DIR:     $LLVM_DIR"
echo "PYTHONPATH:           $PYTHON_DIR"
echo ""
echo "Tools available:"
which air-opt 2>/dev/null && echo "  - air-opt: $(which air-opt)"
which aircc.py 2>/dev/null && echo "  - aircc.py: $(which aircc.py)"
which mlir-opt 2>/dev/null && echo "  - mlir-opt: $(which mlir-opt)"
echo ""
echo "Example usage:"
echo "  aircc.py --target gpu --gpu-arch gfx942 -o output.mlir input.mlir"
echo ""
