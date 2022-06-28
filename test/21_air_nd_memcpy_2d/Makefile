# General Makefile for tests

CC=clang
ACDC_AIR = $(dir $(shell which aie-opt))/..

LDFLAGS = -fuse-ld=lld -rdynamic \
	  -DLIBXAIENGINEV2 \
		-lxaiengine \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-lm \
		-Wl,--no-as-needed -ldl

CFLAGS += -g -I/opt/xaiengine/include
CFLAGS += -std=c++17 -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIR)/runtime_lib \
          -DAIR_LIBXAIE_ENABLE
LDFLAGS += -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib
LDFLAGS += -L$(ACDC_AIR)/runtime_lib/airhost

all: herd.exe

%.exe: 2d_nd.air.a test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ACDC_AIR)/runtime_lib/test_library.cpp test.cpp -Wl,--whole-archive 2d_nd.air.a -Wl,--no-whole-archive

2d_nd.air.a: air.mlir
	aircc.py -v --sysroot= -row-offset=4 -col-offset=5 $< -o $@

clean::
	rm -rf air_project *.elf *.exe *.o *.a 
