# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang++-12
CFLAGS = -O2 -std=c++17 \
	-I$(AIR_ROOT)/runtime_lib/airhost/include \
	-I/path/to/mlir-air/install/include
LDFLAGS = -L/path/to/mlir-air/install/lib -lmlir_async_runtime
LDFLAGS += -L$(AIR_ROOT)/runtime_lib/aircpu -laircpu

test.exe: main.o mmult.async.o
	$(CC) $(LDFLAGS) -lm -lstdc++ -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<


mmult.async.llvm.mlir: mmult.mlir
	air-opt -o $@ $< \
		-async-to-async-runtime -async-runtime-ref-counting \
		-async-runtime-ref-counting-opt -convert-linalg-to-affine-loops \
		-expand-strided-metadata -lower-affine -convert-scf-to-cf \
		-convert-async-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm \
		-convert-func-to-llvm -canonicalize -cse

mmult.async.ll: mmult.async.llvm.mlir
	air-translate --mlir-to-llvmir -o $@ $<

mmult.async.bc: mmult.async.ll
	opt -O2 -o $@ $<

mmult.async.o: mmult.async.bc
	clang -g -O0 -o $@ -c $<

clean:
	rm -rf mmult.air.* mmult.async.* *.elf *.exe *.o *.a
