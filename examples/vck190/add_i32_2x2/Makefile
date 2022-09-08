
include ../config.make

all: herd.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o add.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive add.air.a -Wl,--no-whole-archive -lm

add.air.a: air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} -row-offset=5 -col-offset=17 $<

clean::
	rm -rf air_project *.elf *.exe *.o *.a
