python cnv2_ca.py > ca.mlir; air-opt --aten-to-xten ca.mlir > ca.air.mlir; air-opt --air-name-layers ca.air.mlir > ca.air_named.mlir; aten-opt --air-expand-graph ca.air_named.mlir
