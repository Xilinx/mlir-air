# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o mmult.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive mmult.air.a -Wl,--no-whole-archive -lm

mmult.mlir: mmult.py
	python $< > $@

# change linalg on tensors into linalg on memrefs
mmult.linalg-memrefs.mlir: mmult.mlir
	mlir-opt \
		--linalg-bufferize --cse \
		--func-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o $@ $<

mmult.air.mlir: mmult.linalg-memrefs.mlir
	$(AIR_OPT) -o $@ $< \
			-buffer-results-to-out-params \
			-air-linalg-codegen='l1-tile-size=32,32,32' \
			-air-par-to-herd -air-copy-to-dma \
			-canonicalize -cse \
			-convert-linalg-to-affine-loops

mmult.air.a: mmult.air.mlir
	aircc.py -v -o $@ --sysroot=${SYSROOT} $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a
