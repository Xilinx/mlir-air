# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang++-12
CFLAGS = -O2 -std=c++17 \
	-I$(AIR_ROOT)/runtime_lib/airhost/include \
	-I$(AIR_ROOT)/include
LDFLAGS = -L$(AIR_ROOT)/lib -lmlir_async_runtime
LDFLAGS += -L$(AIR_ROOT)/runtime_lib/aircpu -laircpu

test.exe: main.o producer_consumer.async.o
	$(CC) $(LDFLAGS) -lm -lstdc++ -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

producer_consumer.async.llvm.mlir: producer_consumer.mlir
	air-opt -o $@ $< \
		-async-to-async-runtime -async-runtime-ref-counting \
		-async-runtime-ref-counting-opt -convert-linalg-to-affine-loops \
		-expand-strided-metadata -lower-affine -convert-scf-to-cf \
		-convert-async-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm \
		-convert-func-to-llvm -canonicalize -cse

producer_consumer.async.ll: producer_consumer.async.llvm.mlir
	air-translate --mlir-to-llvmir -o $@ $<

producer_consumer.async.bc: producer_consumer.async.ll
	opt -O2 -o $@ $<

producer_consumer.async.o: producer_consumer.async.bc
	clang -g -O0 -o $@ -c $<

clean:
	rm -rf producer_consumer.air.* producer_consumer.async.* *.elf *.exe *.o *.a
