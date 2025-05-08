# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

targetname := $(shell basename ${srcdir})

all: run4x4

print:
	${powershell} python3 ${srcdir}/run.py -p

run4x4:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 4 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run2x4:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run2x2:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 2 --herd-n 2 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run1x4:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 1 --herd-n 4 --m 512 --n 512 --k 512 --compile-mode compile-and-run

run4x1:
	mkdir -p build
	cd build && ${powershell} xchesscc_wrapper aie2 -c ${srcdir}/mm.cc -o mm.o
	cd build && ${powershell} python3 ${srcdir}/run.py --herd-m 4 --herd-n 1 --m 512 --n 512 --k 512 --compile-mode compile-and-run

clean:
	rm -rf build __pycache__
