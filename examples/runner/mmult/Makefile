# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

all: mmult.deps.mlir

mmult.linalg-memrefs.mlir: mmult.py
	python3 $< -dump-linalg > $@ 

# change linalg on tensors into linalg on memrefs
# mmult.linalg-memrefs.mlir: mmult.mlir
# 	air-opt \
# 		-linalg-bufferize \
# 		-func-bufferize \
# 		-tensor-bufferize \
# 		-finalizing-bufferize \
# 		-buffer-results-to-out-params \
# 		-o $@ $<

mmult.tiled.mlir: mmult.linalg-memrefs.mlir
	air-opt -o $@ $< \
			-air-linalg-codegen='l1-tile-size=32,32,32'
			
mmult.air.mlir: mmult.tiled.mlir
	air-opt -o $@ $< \
			-air-par-to-herd \
			-air-copy-to-dma \
			-canonicalize -cse

mmult.deps.mlir: mmult.air.mlir
	air-opt -o $@ $< \
			-air-dependency

clean::
	rm -rf mmult.*mlir* out.dot trace.out

