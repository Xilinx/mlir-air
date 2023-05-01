# General Makefile for tests

CC=clang
ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

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

all: test.exe

%.exe: test.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(ACDC_AIE)/runtime_lib/test_library.cpp test.cpp -Wl,--whole-archive -Wl,--no-whole-archive

clean::
	rm -rf test.exe
