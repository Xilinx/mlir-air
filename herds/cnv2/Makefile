# General Makefile for tests

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
AIR_OPT = air-opt

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

DESIGN=cnv6

$(DESIGN).mlir: $(DESIGN).py
	python3 $< > $@

aie_inc.cpp: aie.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
			   	--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

$(DESIGN)_airin.mlir: $(DESIGN).mlir
	$(AIR_OPT) -aten-to-air \
				-air-name-layers \
				-air-expand-graph \
				-o $@ $<

$(DESIGN).air.mlir: $(DESIGN)_airin.mlir
	air-opt  -o $@ $< \
			-air-to-linalg \
			-air-linalg-codegen='herd-size=1,1' \
			-affine-to-air \
			-convert-linalg-to-affine-loops
				-o $@ $<

aie.mlir: $(DESIGN).air.mlir
        $(AIR_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=./}" $< \

peano.mlir: aie.mlir
	$(AIR_OPT) -lower-affine -o $@ $< 

elfs: peano.mlir
	$(AIECC) --no-xbridge peano.mlir

aie_ctrl.mlir: $(DESIGN).air.mlir
	$(AIR_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=/dev/null}"\
				--convert-vector-to-llvm \
				-air-to-std \
				-airrt-to-llvm \
				-aten-lowering \
				-o $@ $<

aie_ctrl.llvm.mlir: aie_ctrl.mlir
        $(AIR_OPT) -air-return-elimination \
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
