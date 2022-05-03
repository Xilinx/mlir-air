
all: mmult.air.mlir

mmult.mlir: mmult.py
	python3 $< > $@ 

# change linalg on tensors into linalg on memrefs
mmult.linalg-memrefs.mlir: mmult.mlir
	mlir-opt \
		--linalg-bufferize \
		--func-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o $@ $<

mmult.air.mlir: mmult.linalg-memrefs.mlir
	air-opt -o $@ $< \
			-buffer-results-to-out-params \
			-air-linalg-codegen='l1-tile-size=32,32,32' \
			-affine-to-air \
			-air-dependency

mmult.air.a: mmult.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 -v $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a
