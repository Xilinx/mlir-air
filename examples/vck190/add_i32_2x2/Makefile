# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include ../config.make

all: main.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o add.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive add.air.a -Wl,--no-whole-archive -lm

add.air.a: air.mlir
	aircc.py -o $@ --sysroot=${SYSROOT} $<

clean::
	rm -rf air_project *.elf *.exe *.o *.a
