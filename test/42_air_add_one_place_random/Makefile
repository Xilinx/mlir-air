include ../config.make

.PHONY: all
all: test run 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

air.a: air.mlir 
	aircc.py --sysroot=${SYSROOT} -row-offset=$(shell shuf -i 1-8 -n 1) -col-offset=$(shell shuf -i 0-39 -n 1) $< -o $@

test.exe: air.a test.o 
	$(CC) test.o \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
    -I./air_project/herd_0 \
		-Wl,--whole-archive air.a -Wl,--no-whole-archive \
		-Wl,--whole-archive -lairhost -Wl,--no-whole-archive \
		-lstdc++ \
		-ldl \
		-o $@

.PHONY: test
test: test.exe 

run:
	sudo ./test.exe

clean::
	rm -rf air_project/ *.elf *.a *.exe *.so *.o
