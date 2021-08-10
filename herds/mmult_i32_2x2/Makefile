
include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o  mmult.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive mmult.air.a -Wl,--no-whole-archive

mmult.mlir: mmult.py
	python3 $< > $@ 

mmult.air.mlir: mmult.mlir
	air-opt  -o $@ $< \
			-aten-to-xten \
			-xten-to-linalg \
			-air-linalg-codegen='herd-size=2,2' \
			-affine-to-air \
			-convert-linalg-to-affine-loops

mmult.air.a: mmult.air.mlir
	aircc.py -o $@ -row-offset=2 -col-offset=7 -v -cc $(CC) $<

clean::
	rm -rf air_project mmult.*mlir* *.elf *.exe *.o *.a
