# Simplified Makefile for tests

ARM_SRC_DIR = .
ARM_OBJ_DIR = .
uname_p := $(shell uname -p)

ifeq ($(uname_p),x86_64)
	CFLAGS = -I../../include
	LDFLAGS = -L../../lib
endif

ifeq ($(uname_p),aarch64)
	CFLAGS = -I/opt/xaiengine/include -I../../lib/include
	LDFLAGS = -L/opt/xaiengine/lib
endif

all: test.exe

$(ARM_OBJ_DIR)/%.o: $(ARM_SRC_DIR)/%.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.exe: $(ARM_OBJ_DIR)/%.o
	$(CC) $^ \
		$(LDFLAGS) \
		-rdynamic \
		-lxaiengine \
		-lmetal \
		-lopen_amp \
		-ldl \
		-o $@

clean::
	rm -rf *.o *.exe 