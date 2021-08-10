python cnv2.py > p.mlir; air-opt --aten-to-xten p.mlir > p.air.mlir; air-opt --air-name-layers p.air.mlir > p.air_named.mlir; aten-opt --air-expand-graph p.air_named.mlir
