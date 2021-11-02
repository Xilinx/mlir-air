#SOURCE_FILES := aie.1.mlir test.cpp
#TEST_EXE := test.exe

include ../config.make


.PHONY: all
all: test aie_ctrl.so

air_project/air.mlir.copyherd.inc: air.mlir
	aircc.py -row-offset=2 -col-offset=7 $<

test.exe: air_project/air.mlir.copyherd.inc test.o 
	$(CC) test.o \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl \
		-o $@

.PHONY: test
test: test.exe aie_ctrl.so 

air_project/aie_ctrl.o: air_project/air.mlir.copyherd.inc

aie_ctrl.so: air_project/air.mlir.o
	$(CC) -shared $< -o $@

clean::
	rm -rf air_project/ *.elf *.a *.exe *.so *.o
