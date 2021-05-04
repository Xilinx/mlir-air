python cnv2_ca.py > ca.mlir; aten-opt --aten-to-air ca.mlir > ca.air.mlir; aten-opt --air-expand-graph ca.air.mlir
