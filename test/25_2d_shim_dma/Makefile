#SOURCE_FILES := aie.1.mlir test.cpp
#TEST_EXE := test.exe

include ../config.make


.PHONY: all
all: test

air_project/air.mlir.copyherd.inc: air.mlir
	aircc.py -row-offset=2 -col-offset=7 $<

test.exe: air_project/air.mlir.copyherd.inc test.o 
	$(CC) test.o \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl \
		-o $@

.PHONY: test
test: test.exe aie_ctrl.so 

air_project/aie_ctrl.o: air_project/air.mlir.copyherd.inc

aie_ctrl.so: air_project/air.mlir.o
	$(CC) -shared $< -o $@

#all: test.exe aie_ctrl.so core_7_2.elf

# core_7_2.elf: aie.1.mlir
# 	aiecc.py --no-xbridge $<

# aie.0.mlir: air.mlir Makefile
# 	air-opt $< -air-to-aie="air-to-aie-output-prefix=./ air-to-aie-row-offset=2 air-to-aie-col-offset=7" \
# 			-o /dev/null

# aie.1.mlir: aie.0.mlir
# 	air-opt -lower-affine -convert-scf-to-std -o $@ $<

# aie_ctrl.mlir: air.mlir Makefile
# 	air-opt $< -air-to-aie="air-to-aie-output-prefix=/dev/null air-to-aie-row-offset=2 air-to-aie-col-offset=7" \
# 			-air-to-std \
# 			-airrt-to-llvm \
# 			-aten-lowering \
# 			-o $@

# aie_ctrl.llvm.mlir: aie_ctrl.mlir
# 	air-opt \
# 		-air-return-elimination \
# 		--lower-affine \
# 		--convert-scf-to-std \
# 		--convert-std-to-llvm='emit-c-wrappers'  \
# 		-o $@ $<

# aie_ctrl.o: aie_ctrl.llvm.mlir
# 	mlir-translate --mlir-to-llvmir $< | llc --march=aarch64 --filetype=obj -o $@

# aie_ctrl.so: aie_ctrl.o
# 	$(CC) -shared $< -o $@

# clean::
# 	rm -rf acdc_project *.elf aie.cpp aie.?.mlir
# 	rm -rf aie_ctrl.* test.exe
