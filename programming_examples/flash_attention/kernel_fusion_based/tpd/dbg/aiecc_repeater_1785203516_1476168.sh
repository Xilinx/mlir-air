#!/bin/bash
set -e
# Repeater script for: routing
echo "Original MLIR Diagnostics:"
cat << 'DIAGNOSTICS_EOF'
Unable to find a legal routing
DIAGNOSTICS_EOF
echo ""

MLIR_FILE='./dbg/aiecc_failure_1785203516_1476168.mlir'
PASS_PIPELINE='builtin.module(aie.device(aie-create-pathfinder-flows))'
aie-opt --mlir-print-ir-after-all --mlir-disable-threading --pass-pipeline="$PASS_PIPELINE" "$MLIR_FILE"
