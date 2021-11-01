# General Makefile for tests

include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  vecmul.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive vecmul.air.a -Wl,--no-whole-archive

vecmul.mlir: vecmul.py
	python $< > $@

# It should be possbile to do this step with the pass pipeline
# 'torch-backend-to-linalg-on-tensors-backend-pipeline' but it 
# doesn't seem to work so the pipeline is written out here instead.
vecmul.linalg-tensors.mlir: vecmul.mlir
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
vecmul.linalg-memrefs.mlir: vecmul.linalg-tensors.mlir
	air-opt \
		--func-bufferize \
		--buffer-results-to-out-params \
		--linalg-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		--canonicalize \
		--cse \
		-o $@ $<

vecmul_tiled.mlir: vecmul.linalg-memrefs.mlir
	$(AIR_OPT) \
		--air-linalg-codegen \
		--canonicalize --cse \
		-o $@ $<

vecmul.air.mlir: vecmul_tiled.mlir
	$(AIR_OPT) -affine-to-air \
				-cse \
				-o $@ $<

vecmul.air.a: vecmul.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project vecmul*mlir* *.elf *.exe *.o *.a
