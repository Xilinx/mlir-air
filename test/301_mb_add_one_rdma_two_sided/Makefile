ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..
OBJFILES=pcie-ernic.o pcie-ernic-dev-mem-allocator.o

all: test.elf driver1

driver1: ${OBJFILES} driver1.o
	g++ -I$(ACDC_AIR)/airhost/include -o driver1 driver1.o ${OBJFILES}

driver1.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c driver1.c

pcie-ernic.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c $(ACDC_AIR)/runtime_lib/airhost/pcie-ernic.cpp

pcie-ernic-dev-mem-allocator.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c $(ACDC_AIR)/runtime_lib/airhost/pcie-ernic-dev-mem-allocator.cpp


test.elf: aie.mlir test.cpp
	aiecc.py -v --no-xchesscc --no-xbridge --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
            -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib $(ACDC_AIE)/runtime_lib/test_library.cpp \
            -L$(ACDC_AIR)/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 


clean::
	rm -rf acdc_project *.elf
	rm -f ${OBJFILES}
	rm -f driver1
	rm -f driver1.o
