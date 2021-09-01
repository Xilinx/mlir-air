# General Makefile for tests

include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  vecmul.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive vecmul.air.a -Wl,--no-whole-archive

vecmul.mlir: vecmul.py
	python3 $< > $@

vecmul.affine.mlir: vecmul.mlir
	$(ATEN_OPT) -aten-to-xten \
				-xten-to-affine \
				-o $@ $<

vecmul_tiled_0.mlir: vecmul.affine.mlir
	$(AIR_OPT) -air-automatic-tiling='loop-tile-sizes=64' \
				-simplify-affine-structures \
				-cse \
				-o $@ $<

vecmul_tiled_1.mlir: vecmul_tiled_0.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=0 affine-opt-copy-fast-space=2' \
				-cse \
				-o $@ $<

vecmul_tiled_2.mlir: vecmul_tiled_1.mlir
	$(AIR_OPT) -air-automatic-tiling='loop-tile-sizes=4,4' \
				-simplify-affine-structures \
				-cse \
				-o $@ $<

vecmul.air.mlir: vecmul_tiled_2.mlir
	$(AIR_OPT) -air-herd-assign=herd-assign-depth=1 \
				-affine-to-air \
				-cse \
				-o $@ $<

vecmul.air.a: vecmul.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=3 -col-offset=20 $<

clean::
	rm -rf air_project vecmul*mlir* *.elf *.exe *.o *.a
