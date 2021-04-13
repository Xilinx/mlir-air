# General Makefile for tests

AIE_SRC_DIR = acdc_project
ARM_SRC_DIR = .
ARM_OBJ_DIR = .

AIE_OPT = aie-opt
AIE_XLATE = aie-translate
ATEN_OPT = aten-opt

AIECC = aiecc.py

uname_p := $(shell uname -p)
ifeq ($(uname_p),aarch64)
	CFLAGS += -I/opt/xaiengine/include
	LDFLAGS += -L/opt/xaiengine/lib
endif

CFLAGS += -I../../lib/include

all: herd.exe aie_ctrl.so elfs

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

vecmul.mlir: vecmul.py
	python3 $< > $@

aie_inc.cpp: aie.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
				--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

vecmul_tiled_0.mlir: vecmul.mlir
	$(ATEN_OPT) -aten-to-air \
				-air-to-affine \
				-affine-loop-opt -affine-opt-tile-sizes 64 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

vecmul_tiled_1.mlir: vecmul_tiled_0.mlir
	$(ATEN_OPT) -affine-loop-opt -affine-opt-copy-depths 0 -affine-opt-copy-fast-space 2 \
				-cse \
				-o $@ $<

vecmul_tiled_2.mlir: vecmul_tiled_1.mlir
	$(ATEN_OPT) -affine-loop-normalize \
				-affine-loop-opt -affine-opt-tile-sizes 2 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<
vecmul_tiled_3.mlir: vecmul_tiled_2.mlir
	$(ATEN_OPT) -affine-loop-normalize \
				-affine-loop-opt -affine-opt-tile-sizes 2 \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

vecmul_air.mlir: vecmul_tiled_3.mlir
	$(ATEN_OPT) -acap-herd-assign=herd-assign-depth=1 \
				-affine-super-vectorize="virtual-vector-size=8" \
				-affine-to-air \
				-cse \
				-o $@ $<

aie.mlir: vecmul_air.mlir
	$(ATEN_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-file=$@}" $< \
				-o /dev/null
peano.mlir: aie.mlir
	$(ATEN_OPT) -lower-affine -o $@ $<

elfs: peano.mlir
	$(AIECC) -v --no-xbridge peano.mlir


aie_ctrl.mlir: vecmul_air.mlir
	$(ATEN_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-file=/dev/null}"\
				-air-to-std \
				-aten-lowering \
				-o $@ $<

aie_ctrl.llvm.mlir: aie_ctrl.mlir
	$(ATEN_OPT) -air-return-elimination \
		--lower-affine \
		--convert-scf-to-std \
		--convert-vector-to-llvm \
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
