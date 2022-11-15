# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include ../config.make

all: main.exe

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: %.o test_library.o beefmaker.air.a
	$(CC) $(LDFLAGS) -o $@ $< test_library.o -Wl,--whole-archive beefmaker.air.a -Wl,--no-whole-archive -lm

beefmaker_kernel.o: chess/beefmaker_kernel.cc
	xchesscc -p me -P $(CARDANO)/data/cervino/lib -c $<

beefmaker.air.a: air.mlir beefmaker_kernel.o
	aircc.py -o $@ --host-target=aarch64-linux-gnu -xbridge --sysroot=${SYSROOT} $<

clean::
	rm -rf air_project *.elf* *.exe *.o *.a
