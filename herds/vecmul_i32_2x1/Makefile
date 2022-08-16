# General Makefile for tests

include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  vecmul.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive vecmul.air.a -Wl,--no-whole-archive

vecmul.mlir: vecmul.py
	python $< > $@

# change linalg on tensors into linalg on memrefs
vecmul.linalg-memrefs.mlir: vecmul.mlir
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
		--air-linalg-codegen='herd-size=2,1' \
		--canonicalize --cse \
		-o $@ $<

vecmul.air.mlir: vecmul_tiled.mlir
	$(AIR_OPT) -air-par-to-herd -air-copy-to-dma \
				-cse \
				-o $@ $<

vecmul.air.a: vecmul.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project vecmul*mlir* *.elf *.exe *.o *.a
