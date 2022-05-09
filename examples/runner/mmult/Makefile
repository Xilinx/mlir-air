
all: mmult.deps.mlir

mmult.mlir: mmult.py
	python3 $< -dump-linalg > $@ 

# change linalg on tensors into linalg on memrefs
mmult.linalg-memrefs.mlir: mmult.mlir
	mlir-opt \
		-linalg-bufferize \
		-func-bufferize \
		-tensor-bufferize \
		-finalizing-bufferize \
		-buffer-results-to-out-params \
		-o $@ $<

mmult.tiled.mlir: mmult.linalg-memrefs.mlir
	air-opt -o $@ $< \
			-air-linalg-codegen='l1-tile-size=32,32,32'
			
mmult.air.mlir: mmult.tiled.mlir
	air-opt -o $@ $< \
			-affine-to-air \
			-canonicalize -cse

mmult.deps.mlir: mmult.air.mlir
	air-opt -o $@ $< \
			-air-dependency

clean::
	rm -rf mmult.*mlir* out.dot trace.out

