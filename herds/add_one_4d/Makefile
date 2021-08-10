# General Makefile for tests

include ../config.make

all: herd.exe

%.exe: %.o test_library.o add_one.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive add_one.air.a -Wl,--no-whole-archive

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

add_one.mlir: add_one.py
	python3 $< > $@ 

aie_inc.cpp: aie.herd_0.mlir
	$(AIE_OPT) -lower-affine --convert-scf-to-std \
				--aie-create-flows --aie-find-flows \
				--aie-assign-buffer-addresses $^ | $(AIE_XLATE) --aie-generate-xaie -o $@

add_one_tiled_0.mlir: add_one.mlir
	$(AIR_OPT) -aten-to-xten \
				-xten-to-affine \
				-affine-loop-opt='affine-opt-tile-sizes=1,1,16,16' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_tiled_1.mlir: add_one_tiled_0.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=1 affine-opt-copy-fast-space=2' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one_tiled_2.mlir: add_one_tiled_1.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-tile-sizes=16,16' \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one.air.mlir: add_one_tiled_2.mlir
	$(AIR_OPT) -acap-herd-assign=herd-assign-depth=0 \
				-affine-to-air \
				-affine-loop-normalize -simplify-affine-structures \
				-cse \
				-o $@ $<

add_one.air.a: add_one.air.mlir
	aircc.py -o $@ -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a *.mlir
