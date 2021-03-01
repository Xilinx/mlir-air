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

add_one.mlir: add_one.py
	python3 $< > $@ 

aie_inc.cpp: aie.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
			   --aie-create-flows --aie-find-flows --aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

add_one_tiled_0.mlir: add_one.mlir
	$(ATEN_OPT) -aten-to-air \
				-air-to-affine \
				-affine-loop-opt -affine-opt-copy-depths 0 -affine-opt-copy-fast-space 2 -affine-opt-tile-sizes 64 \
				-cse \
				-o $@ $<

add_one_tiled_1.mlir: add_one_tiled_0.mlir
	$(ATEN_OPT) -affine-loop-normalize \
				-affine-loop-opt -affine-opt-tile-sizes 2 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_air.mlir: add_one_tiled_1.mlir
	$(ATEN_OPT) -acap-herd-assign \
				-herd-assign-depth 0 \
				-affine-to-air \
				-cse \
				-o $@ $<

aie.mlir: add_one_air.mlir
	$(ATEN_OPT) -air-to-aie \
				-air-to-aie-output-file $@ $< \
				-o /dev/null

add_one_lowered.mlir: add_one_tiled_1.mlir
	$(ATEN_OPT) -affine-to-air \
				-air-to-std \
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
	rm -rf add_one.so add_one.o
	rm -rf add_one_air.mlir add_one.llvm.mlir add_one_lowered.mlir add_one_tiled*.mlir
