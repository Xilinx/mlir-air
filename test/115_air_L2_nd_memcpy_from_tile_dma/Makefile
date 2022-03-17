include ../config.make

.PHONY: all
all: test 

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

air.a: air.mlir 
	aircc.py -v --sysroot=${SYSROOT} -row-offset=2 -col-offset=7 $< -o $@

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
