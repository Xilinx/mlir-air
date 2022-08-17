# General Makefile for tests

include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  vecmul.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive vecmul.air.a -Wl,--no-whole-archive

vecmul.linalg.mlir: vecmul.py
	python3 $< > $@
vecmul.affine.mlir: vecmul.linalg.mlir
	$(ATEN_OPT) -linalg-bufferize -func-bufferize -tensor-bufferize -finalizing-bufferize -convert-linalg-to-affine-loops \
				-o $@ $<

vecmul.affine_attr.mlir: vecmul.affine.mlir
	cat $< | awk 'BEGIN {CURLY = 0; AFFINE = 0} { if ($$1 == "affine.store") AFFINE = 1; if (AFFINE > 0 && $$1 == "}") CURLY += 1; if (CURLY == 1) {print $$0 " {affine_opt_label = \"\"}" ; CURLY = 0; AFFINE = 0}  else print $$0}' > $@

vecmul_tiled_0.mlir: vecmul.affine_attr.mlir
	$(AIR_OPT) -air-automatic-tiling='loop-tile-sizes=256' \
				-affine-simplify-structures \
				-cse \
				-o $@ $<

vecmul_tiled_1.mlir: vecmul_tiled_0.mlir
	$(AIR_OPT) -affine-loop-opt='affine-opt-copy-depths=0 affine-opt-copy-fast-space=2' \
				-cse \
				-o $@ $<

vecmul_tiled_2.mlir: vecmul_tiled_1.mlir
	$(AIR_OPT) -air-automatic-tiling='loop-tile-sizes=4,4' \
				-affine-simplify-structures \
				-cse \
				-o $@ $<

vecmul.air.mlir: vecmul_tiled_2.mlir
	$(AIR_OPT) -air-herd-assign=herd-assign-depth=0 \
				-air-copy-to-dma \
				-cse \
				-o $@ $<

vecmul.air.a: vecmul.air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=3 -col-offset=20 $<

clean::
	rm -rf air_project vecmul*mlir* *.elf *.exe *.o *.a
