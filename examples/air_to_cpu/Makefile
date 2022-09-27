# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

AIR_ROOT = $(dir $(shell which air-opt))/..
CC = clang-10
CFLAGS += -O3 -std=c++17 -I$(AIR_ROOT)/runtime_lib/airhost/include

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
					-lower-affine -convert-scf-to-cf \
					-convert-func-to-llvm \
					-convert-cf-to-llvm \
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
