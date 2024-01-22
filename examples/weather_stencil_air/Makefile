# (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
# SPDX-License-Identifier: MIT

ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

i32: i32_chess i32.elf

build: 
	xchessmk test.prx
sim: 
	xca_udm_dbg -P ${AIETOOLS_ROOT}/data/cervino/lib/ -t sim.tcl

i32_chess: 
	xchesscc -p me -P ${AIETOOLS_ROOT}/data/cervino/lib/ -L ${AIETOOLS_ROOT}/data/cervino/lib/ -c ./hdiff_lap.cc ./hdiff_flux1.cc ./hdiff_flux2.cc

i32.elf: 
	aiecc.py aie_$(b).mlir \
	 ../../elfutils/libelf/libelf.a \
	-I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib -I$(ACDC_AIE)/runtime_lib/x86_64/test_lib/include -I$(ACDC_AIR)/runtime_lib/airhost/include -L$(ACDC_AIR)/runtime_lib/airhost $(ACDC_AIE)/runtime_lib/x86_64/test_lib/src/test_library.cpp -I${ROCM_ROOT}/include -Wl,-rpath,${ROCM_ROOT}/lib ${ROCM_ROOT}/lib/libhsa-runtime64.so.1.9.0 ./test_$(b).cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -ldl -o test_$(b).elf

clean:
	rm -rf work *mem acdc_project core_* *.elf *.mlir.prj chesswork*


