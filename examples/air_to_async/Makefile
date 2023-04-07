# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang++-12
CFLAGS = -O2 -std=c++17 \
	-I$(AIR_ROOT)/runtime_lib/airhost/include \
	-I/path/to/llvm-project/mlir/include
LDFLAGS = -L/path/to/build/llvm/lib -lmlir_async_runtime
LDFLAGS += -L$(AIR_ROOT)/runtime_lib/aircpu -laircpu

test.exe: main.o mmult.async.o
	$(CC) $(LDFLAGS) -lm -lstdc++ -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

mmult.mlir: mmult.py
	python3 $< > $@ 

# change linalg on tensors into linalg on memrefs
mmult.linalg-memrefs.mlir: mmult.mlir
	torch-mlir-opt \
		--refback-mlprogram-bufferize \
		--linalg-bufferize --cse \
		--func-bufferize \
		--tensor-bufferize \
		--finalizing-bufferize \
		-o $@ $<

mmult.air.mlir: mmult.linalg-memrefs.mlir
	air-opt -o $@ $< \
		-buffer-results-to-out-params \
		-air-linalg-codegen \
		-air-par-to-herd \
		-air-copy-to-dma \
		-canonicalize -cse

mmult.async.mlir: mmult.air.mlir
	air-opt -buffer-results-to-out-params -air-to-async -o $@ $<

mmult.async.llvm.mlir: mmult.async.mlir
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

mmult.air.a: mmult.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $<

clean:
	rm -rf air_project mmult.async.* mmult.*mlir* *.elf *.exe *.o *.a
