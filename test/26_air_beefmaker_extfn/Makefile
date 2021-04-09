SOURCE_FILES := aie.mlir test.cpp
TEST_EXE := test.exe

include ../config.make

.PHONY: all
all: beefmaker_kernel.o core_7_2.elf test.exe

beefmaker_kernel.o: chess/beefmaker_kernel.cc
	xchesscc -p me -P ${CARDANO}/data/cervino/lib -c $<

aie.mlir: air.mlir Makefile
	aten-opt $< -air-to-aie="air-to-aie-output-file=$@ air-to-aie-row-offset=2 air-to-aie-col-offset=7" \
			-o /dev/null

core_7_2.elf: aie.mlir
	aiecc.py --no-xbridge aie.mlir

clean::
	rm -rf beefmaker_kernel.o acdc_project aie.mlir