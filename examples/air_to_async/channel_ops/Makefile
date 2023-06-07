# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang++-12
CFLAGS = -O2 -std=c++17 \
	-I$(AIR_ROOT)/runtime_lib/airhost/include \
	-I/path/to/mlir-air/utils/llvm/mlir/include
LDFLAGS = -L/path/to/mlir-air/install/lib -lmlir_async_runtime
LDFLAGS += -L$(AIR_ROOT)/runtime_lib/aircpu -laircpu

test.exe: main.o channel.async.o
	$(CC) $(LDFLAGS) -lm -lstdc++ -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<


channel.async.mlir: channel_op_lowering.mlir
	air-opt -buffer-results-to-out-params -air-to-async -o $@ $<

channel.async.llvm.mlir: channel.async.mlir
	air-opt -o $@ $< \
		-async-to-async-runtime -async-runtime-ref-counting \
		-async-runtime-ref-counting-opt -convert-linalg-to-affine-loops \
		-expand-strided-metadata -lower-affine -convert-scf-to-cf \
		-convert-async-to-llvm -convert-memref-to-llvm -convert-cf-to-llvm \
		-convert-func-to-llvm -canonicalize -cse

channel.async.ll: channel.async.llvm.mlir
	air-translate --mlir-to-llvmir -o $@ $<

channel.async.bc: channel.async.ll
	opt -O2 -o $@ $<

channel.async.o: channel.async.bc
	clang -g -O0 -o $@ -c $<

channel.air.a: channel.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $<

clean:
	rm -rf air_project channel.async.* channel.*mlir* *.elf *.exe *.o *.a
