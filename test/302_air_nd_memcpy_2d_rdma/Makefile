# General Makefile for tests

CC=clang
ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..
OBJFILES=pcie-ernic.o pcie-ernic-dev-mem-allocator.o

LDFLAGS = -fuse-ld=lld -rdynamic \
	  -DLIBXAIENGINEV2 \
		-lxaiengine \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-lm \
		-Wl,--no-as-needed -ldl

CFLAGS += -g -I/opt/xaiengine/include
CFLAGS += -std=c++17 -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib \
          -DAIR_LIBXAIE_ENABLE
LDFLAGS += -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib
LDFLAGS += -L$(ACDC_AIR)/runtime_lib/airhost

all: herd.exe driver1

%.exe: 2d_nd.air.a test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ACDC_AIE)/runtime_lib/test_library.cpp test.cpp -Wl,--whole-archive 2d_nd.air.a -Wl,--no-whole-archive

2d_nd.air.a: air.mlir
	aircc.py -v --sysroot= --host-target=x86_64-amd-linux-gnu -row-offset=4 -col-offset=5 $< -o $@

driver1: ${OBJFILES} driver1.o
	g++ -I$(ACDC_AIR)/airhost/include -o driver1 driver1.o ${OBJFILES}

driver1.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c driver1.c

pcie-ernic.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c $(ACDC_AIR)/runtime_lib/airhost/pcie-ernic.cpp

pcie-ernic-dev-mem-allocator.o:
	g++ -I$(ACDC_AIR)/runtime_lib/airhost/include -c $(ACDC_AIR)/runtime_lib/airhost/pcie-ernic-dev-mem-allocator.cpp


clean::
	rm -rf air_project *.elf *.exe *.o *.a 
	rm -f ${OBJFILES}
	rm -f driver1
	rm -f driver1.o
