include ../config.make

.PHONY: all
all: test 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

air.a: air.mlir
	aircc.py --sysroot=${SYSROOT} -row-offset=4 -col-offset=16 $< -o $@

air_test.exe: air.a air_test.o 
	$(CC) air_test.o \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
    -I./air_project/copyherd \
		-Wl,--whole-archive air.a -Wl,--no-whole-archive \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl \
		-o $@

.PHONY: test
test: air_test.exe 

clean::
	rm -rf air_project/ *.elf *.a *.exe *.so *.o
