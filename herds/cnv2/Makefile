# General Makefile for tests

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
ATEN_OPT = aten-opt

AIECC = aiecc.py
SYSROOT = /group/xrlabs/platforms/pynq-vck190-sysroot/
CC = clang --target=aarch64-linux-gnu -std=c++11 --sysroot=$(SYSROOT) \
   -I$(SYSROOT)/opt/xaiengine/include -L$(SYSROOT)/opt/xaiengine/lib -fuse-ld=lld -rdynamic -lxaiengine

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
endif

CFLAGS += -I../../lib/include

all: aie_ctrl.so elfs

%.o: %.cpp aie_inc.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: $(ARM_OBJ_DIR)/%.o
	$(CC) $^ \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-o $@

cnv2.mlir: cnv2.py
	python3 $< > $@

aie_inc.cpp: aie.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
			   	--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

cnv2_tiled_0.mlir: cnv2.mlir
	$(ATEN_OPT) -aten-to-air \
				-air-to-affine \
				-affine-loop-opt -affine-opt-tile-sizes 64 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

cnv2_tiled_1.mlir: cnv2_tiled_0.mlir
	$(ATEN_OPT) -affine-loop-opt -affine-opt-copy-depths 0 -affine-opt-copy-fast-space 2 \
				-cse \
				-o $@ $<

cnv2_tiled_2.mlir: cnv2_tiled_1.mlir
	$(ATEN_OPT) -affine-loop-normalize \
				-affine-loop-opt -affine-opt-tile-sizes 2 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<
cnv2_tiled_3.mlir: cnv2_tiled_2.mlir
	$(ATEN_OPT) -affine-loop-normalize \
				-affine-loop-opt -affine-opt-tile-sizes 2 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

cnv2_air.mlir: cnv2_tiled_3.mlir
	$(ATEN_OPT) -acap-herd-assign \
				-herd-assign-depth 1 \
				-affine-to-air \
				-cse \
				-o $@ $<

aie.mlir: cnv2_air.mlir
	$(ATEN_OPT) -air-to-aie \
				-air-to-aie-emit-while-loop=true \
				-air-to-aie-row-offset 2 \
				-air-to-aie-col-offset 7 \
				-air-to-aie-output-file $@ $< \
				-o aie_t.mlir
peano.mlir: aie.mlir
	$(ATEN_OPT) -lower-affine -o $@ $< 

elfs: peano.mlir
	$(AIECC) --no-xbridge peano.mlir


aie_ctrl.mlir: cnv2_air.mlir
	$(ATEN_OPT) -air-to-aie $< -air-to-aie-output-file /dev/null \
				-air-to-aie-row-offset 2 \
				-air-to-aie-col-offset 7 \
				-air-to-std \
				-aten-lowering \
				-o $@

aie_ctrl.llvm.mlir: aie_ctrl.mlir
	$(ATEN_OPT) -air-return-elimination \
		--lower-affine \
		--convert-scf-to-std \
		--convert-std-to-llvm='emit-c-wrappers'  \
		-o $@ $<

aie_ctrl.o: aie_ctrl.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc --march=aarch64 --filetype=obj -o $@

aie_ctrl.so: aie_ctrl.o
	$(CC) -shared $< -o $@

%.elf: $(AIE_SRC_DIR)/main.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)
	rm -rf aie_ctrl.so aie_ctrl.o
	rm -rf *.mlir
