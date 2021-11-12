# General Makefile for tests

include ../config.make

all: herd.exe

%.exe: %.o test_library.o add_one.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive add_one.air.a -Wl,--no-whole-archive

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

# add_one.mlir: add_one.py
# 	python3 $< > $@ 

add_one.affine.mlir: add_one.mlir
	$(ATEN_OPT) -aten-to-xten \
				-xten-to-affine \
				-o $@ $<

add_one.affine-tensors.mlir: add_one.affine.mlir
	torch-mlir-opt \
		--torch-verify-invariants-before-backend-lowering \
		--convert-torch-to-linalg \
		--convert-torch-to-std \
		--convert-torch-to-scf \
		--std-expand \
		--canonicalize \
		--resolve-shaped-type-result-dims \
		--torch-func-backend-type-conversion \
		--cse \
		--torch-finalizing-backend-type-conversion \
		-o $@ $<

# change linalg on tensors into linalg on memrefs
add_one.affine-memrefs.mlir: add_one.affine-tensors.mlir
	mlir-opt \
		--linalg-bufferize \
		--func-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o $@ $<

add_one.tiled_0.mlir: add_one.affine-memrefs.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-tile-sizes=1,1,16,16' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one.tiled_1.mlir: add_one.tiled_0.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-copy-fast-space=2' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one.tiled_2.mlir: add_one.tiled_1.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-tile-sizes=16,16' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one.air.mlir: add_one.tiled_2.mlir
	$(AIR_OPT) -air-herd-assign=herd-assign-depth=0 \
				-affine-to-air \
				-cse \
				-o $@ $<

add_one.air.a: add_one.air.mlir
	aircc.py --sysroot=${SYSROOT} -o $@ -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project add_one.*.mlir *.elf *.exe *.o *.a 
