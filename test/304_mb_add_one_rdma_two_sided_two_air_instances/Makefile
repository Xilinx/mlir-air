ACDC_AIE = $(dir $(shell which aie-opt))/..
ACDC_AIR = $(dir $(shell which air-opt))/..

all: host0_test.elf host1_test.elf

host0_test.elf: aie.mlir host0_test.cpp
	aiecc.py -v --no-xchesscc --no-xbridge --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
            -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib $(ACDC_AIE)/runtime_lib/test_library.cpp \
            -L$(ACDC_AIR)/runtime_lib/airhost host0_test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 


host1_test.elf: aie.mlir host1_test.cpp
	aiecc.py -v --no-xchesscc --no-xbridge --aie-generate-xaiev2 --host-target=x86_64-amd-linux-gnu --sysroot= $< -I/opt/xaiengine/include -L/opt/xaiengine/lib -Wl,-R/opt/xaiengine/lib \
            -I$(ACDC_AIR)/runtime_lib/airhost/include -I$(ACDC_AIE)/runtime_lib $(ACDC_AIE)/runtime_lib/test_library.cpp \
            -L$(ACDC_AIR)/runtime_lib/airhost host1_test.cpp -Wl,--whole-archive -lairhost -Wl,--no-whole-archive -lstdc++ -o $@ 

clean::
	rm -rf acdc_project *.elf
