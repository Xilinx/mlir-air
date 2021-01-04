# General Makefile for tests

AIE_SRC_DIR = chess
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
ATEN_OPT = aten-opt

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
endif

CFLAGS += -I../../lib/include
all: herd.exe add_one.so

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

aie_inc.cpp: aie.mlir
	$(AIE_OPT) --aie-create-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

add_one_tiled.mlir: add_one.mlir
	$(ATEN_OPT) -aten-to-air \
				-air-to-affine \
				-affine-loop-opt -affine-opt-copy-depths 1 -affine-opt-tile-sizes 32,32 \
				-cse \
				-o $@ $<

add_one_lowered.mlir: add_one_tiled.mlir
	$(ATEN_OPT) -affine-to-air \
				-aten-to-std \
				-return-elimination \
				-cse \
				-o $@ $<

add_one.llvm.mlir: add_one_lowered.mlir
	mlir-opt --lower-affine \
		--convert-scf-to-std \
		--convert-std-to-llvm='emit-c-wrappers'  \
		-o $@ $<

add_one.o: add_one.llvm.mlir
	mlir-translate --mlir-to-llvmir $< | llc --march=aarch64 --filetype=obj -o $@

add_one.so: add_one.o
	$(CC) -shared $< -o $@

%.elf: $(AIE_SRC_DIR)/main.cc
	cd ./$(AIE_SRC_DIR) && \
	xchessmk $*.prx && \
	cp work/Release_LLVM/$*.prx/$* ../$*.elf

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)/work
	rm -rf add_one.so add_one.o add_one.llvm.mlir add_one_lowered.mlir add_one_tiled.mlir
