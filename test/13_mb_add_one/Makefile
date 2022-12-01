ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

all: test.elf

test.elf: aie.mlir test.cpp
	aiecc.py -v --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
						-I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib $(ACDC_AIE)/runtime_lib/test_library.cpp \
						-L$(ACDC_AIR)/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 

clean::
	rm -rf acdc_project *.elf
