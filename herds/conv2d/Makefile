
include ../config.make

#all: conv2d_air.a test_cpu.exe
all: conv2d.air.mlir 
conv2d.torch.mlir: conv2d.py
	python3 $< > a.$@
	torch-mlir-opt --torch-refine-types --torch-refine-public-return a.$@ -o $@

conv2d.mlir: conv2d.torch.mlir
	torch-mlir-opt --torch-backend-to-linalg-on-tensors-backend-pipeline -o $@ $<


# change linalg on tensors into linalg on memrefs
conv2d.linalg-memrefs.mlir: conv2d.mlir
	mlir-opt \
		--linalg-bufferize \
		-o b.$@ $<
	mlir-opt \
		--tensor-constant-bufferize \
		-o c.$@ b.$@
	mlir-opt \
		--func-bufferize \
		-o d.$@ c.$@
	mlir-opt \
		--tensor-bufferize \
		-o e.$@ d.$@
	mlir-opt \
		--finalizing-bufferize \
		-o $@ e.$@
	
conv2d.air.mlir: conv2d.linalg-memrefs.mlir
	$(AIR_OPT)  -o 1.$@   $< -buffer-results-to-out-params -cse
	patch < 1.conv2d.air.patch
	# this patch remove bias functionality to avoid linalg.generic op
	$(AIR_OPT)  -o 2.$@ 1.$@ -air-linalg-codegen='herd-size=4,4 l1-tile-size=1,32,16,16,16,3,3 l1-tile-permute=0,3,1,2,4,5,6'
	$(AIR_OPT)  -o $@ 2.$@ -air-par-to-herd -air-copy-to-dma -cse -canonicalize 

clean:
	rm -rf *.mlir
