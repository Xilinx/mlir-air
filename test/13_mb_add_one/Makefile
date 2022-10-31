ACDC_AIR = $(dir $(shell which aie-opt))/..

all: test.elf

test.elf: aie.mlir test.cpp
	aiecc.py -v --aie-generate-xaiev2 --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
						-I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIR)/runtime_lib $(ACDC_AIR)/runtime_lib/test_library.cpp \
						-L$(ACDC_AIR)/runtime_lib/airhost test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 

clean::
	rm -rf acdc_project *.elf
