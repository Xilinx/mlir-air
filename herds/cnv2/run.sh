python3 $1.py > $1.mlir; air-opt --aten-to-air $1.mlir > $1.air.mlir; air-opt --air-name-layers $1.air.mlir > $1.air_named.mlir; aten-opt --air-expand-graph $1.air_named.mlir
