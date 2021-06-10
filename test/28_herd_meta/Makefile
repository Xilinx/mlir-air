
SOURCE_FILES := test.cpp herd.o
TEST_EXE := test.exe

include ../config.make

all: test.exe

herd.llvm.mlir: herd.airrt.mlir
	aten-opt -o $@ $< -airrt-to-llvm

herd.o: herd.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc --filetype=obj -o $@

# herd.so: herd.o
# 	clang-8 -shared $< -o $@

clean::
	rm -rf *.exe *.so *.o herd.llvm.mlir
