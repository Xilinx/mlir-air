
include ../config.make

all: herd.exe

%.exe: %.o test_library.o mmult.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive mmult.air.a -Wl,--no-whole-archive

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

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
	$(AIR_OPT)  -o $@ $< \
			-buffer-results-to-out-params \
			-air-linalg-codegen='herd-size=2,2' \
			-affine-to-air \
			-convert-linalg-to-affine-loops

mmult.air.a: mmult.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a