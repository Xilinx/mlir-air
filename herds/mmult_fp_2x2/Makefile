
include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  mmult.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive mmult.air.a -Wl,--no-whole-archive

mmult.mlir: mmult.py
	python3 $< > $@ 

mmult.linalg.mlir: mmult.mlir
	$(ATEN_OPT) -aten-to-xten \
				-xten-to-linalg \
				-o $@ $<

mmult.air.mlir: mmult.linalg.mlir
	$(AIR_OPT)  -o $@ $< \
			-air-linalg-codegen='herd-size=2,2' \
			-air-par-to-herd -air-copy-to-dma \
			-convert-linalg-to-affine-loops

mmult.air.a: mmult.air.mlir
	aircc.py --sysroot=${SYSROOT} -o $@ -row-offset=2 -col-offset=7 $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a
