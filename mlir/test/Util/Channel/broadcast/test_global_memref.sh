export PATH=/root/mlir-air/build_assert/bin/:$PATH
export PATH=/root/mlir-air/llvm/build/bin/:$PATH

air-opt broadcast.mlir \
	-air-to-async \
	-buffer-results-to-out-params \
	-async-to-async-runtime \
	-async-runtime-ref-counting \
	-async-runtime-ref-counting-opt \
	-convert-linalg-to-affine-loops \
	-expand-strided-metadata \
	-lower-affine \
	-convert-scf-to-cf \
	-convert-async-to-llvm \
	-convert-memref-to-llvm \
	-convert-cf-to-llvm \
	-convert-func-to-llvm \
	-canonicalize -cse \
	-o broadcast.llvm.mlir


air-translate broadcast.llvm.mlir \
	--mlir-to-llvmir \
	-o broadcast.ll


clang broadcast.ll -O2 -std=c++17 -c -o broadcast.o

clang main.cpp -O2 -std=c++17 \
	-I/root/mlir-air/build_assert/runtime_lib/x86_64/airhost/include \
	-c \
	-o main.o

clang main.o broadcast.o \
	-L/root/mlir-air/build_assert/runtime_lib/x86_64/runtime_lib \
	-laircpu \
	-L/root/mlir-air/llvm/build/lib \
	-lmlir_async_runtime \
	-o test.exe