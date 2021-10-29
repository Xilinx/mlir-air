
include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  mmult.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive mmult.air.a -Wl,--no-whole-archive

mmult.mlir: mmult.py
	python3 $< > $@ 

# It should be possbile to do this step with the pass pipeline
# 'torch-backend-to-linalg-on-tensors-backend-pipeline' but it 
# doesn't seem to work so the pipeline is written out here instead.
mmult.linalg-tensors.mlir: mmult.mlir
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
mmult.linalg-memrefs.mlir: mmult.linalg-tensors.mlir
	mlir-opt \
		--linalg-bufferize \
		--func-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o $@ $<

mmult.air.mlir: mmult.linalg-memrefs.mlir
	$(AIR_OPT)  -o $@ $< \
			-air-linalg-codegen='herd-size=2,2' \
			-affine-to-air \
			-convert-linalg-to-affine-loops

mmult.air.a: mmult.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 -v $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a
