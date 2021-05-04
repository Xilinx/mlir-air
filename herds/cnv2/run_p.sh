python cnv2.py > p.mlir; aten-opt --aten-to-air p.mlir > p.air.mlir; aten-opt --air-name-layers p.air.mlir > p.air_named.mlir; aten-opt --air-expand-graph p.air_named.mlir
