# General Makefile for tests

include ../config.make

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

add_one.mlir: add_one.py
	python3 $< > $@ 

aie_inc.cpp: aie.herd_0.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
				--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

add_one_tiled_0.mlir: add_one.mlir
	$(AIR_OPT) -aten-to-xten \
				-xten-to-affine \
				-affine-loop-opt='affine-opt-tile-sizes=32,32' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_tiled_1.mlir: add_one_tiled_0.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-copy-fast-space=2' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_tiled_2.mlir: add_one_tiled_1.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-tile-sizes=4,4' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_air.mlir: add_one_tiled_2.mlir
	$(AIR_OPT) -acap-herd-assign=herd-assign-depth=0 \
				-affine-to-air \
				-cse \
				-o $@ $<

aie.herd_0.mlir: add_one_air.mlir
	$(AIR_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=./}" $< \
				-o /dev/null

peano.mlir: aie.herd_0.mlir
	$(AIR_OPT) -lower-affine -o $@ $< 

elfs: peano.mlir
	$(AIECC) --no-xbridge peano.mlir


aie_ctrl.mlir: add_one_air.mlir
	$(AIR_OPT) --pass-pipeline="air-to-aie{air-to-aie-emit-while-loop=true air-to-aie-row-offset=2 air-to-aie-col-offset=7 air-to-aie-output-prefix=/dev/null}"\
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
	mlir-translate --mlir-to-llvmir $< | opt -O3 | llc --march=aarch64 --filetype=obj -o $@

aie_ctrl.so: aie_ctrl.o
	$(CC) -shared $< -o $@

clean:
	rm -rf aie_inc.cpp *.o *.exe *.elf ./$(AIE_SRC_DIR)
	rm -rf aie_ctrl.so aie_ctrl.o
	rm -rf *.mlir
