# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang-10
CFLAGS += -O0 -std=c++17 -I$(AIR_ROOT)/runtime_lib/airhost/include

all: test.exe

test.exe: memory.o main.o test.o
	$(CC) $(LDFLAGS) -lm -lstdc++ -o $@ $^

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.cpu.mlir: %.mlir
	air-opt  -o $@ $< -air-to-cpu -cse

%.llvmir.mlir: %.cpu.mlir
	air-opt  -o $@ $< \
					-convert-linalg-to-affine-loops \
					-lower-affine -convert-scf-to-std \
					-convert-std-to-llvm \
					-convert-memref-to-llvm \
					-canonicalize -cse

%.ll: %.llvmir.mlir
	aie-translate -o $@ $< --mlir-to-llvmir

%.opt.bc: %.ll
	opt -O3 -o $@ $<

%.o: %.opt.bc
	clang -O3 -Wno-override-module -g -c -o $@ $<

out.dot: test.exe
	test.exe

%.pdf: %.dot
	dot -Tpdf $< -o $@ 

clean:
	rm -rf *.exe *.o *.ll *.bc *.llvmir.mlir *.cpu.mlir *.dot *.pdf
