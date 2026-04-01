# SPDX-License-Identifier: MIT
# Build libairgpu.so — VMem-backed GPU runtime for mlir-air

ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc

CXXFLAGS  = -std=c++17 -fPIC -O2 -Wall -Wextra -Wno-unused-parameter
INCLUDES  = -I$(ROCM_PATH)/include -I.
LDFLAGS   = -shared -L$(ROCM_PATH)/lib -lamdhip64

TARGET    = libairgpu.so
SOURCES   = GpuRuntimeWrappers.cpp GpuTestUtils.cpp vmem_allocator.cpp
OBJECTS   = $(SOURCES:.cpp=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(HIPCC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET) test_vmem

# Quick smoke test: allocate and free via VMem
test: $(TARGET)
	@echo "=== Smoke test: VMem allocator ==="
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -o test_vmem ../../test/gpu/test_vmem.cpp -L. -lairgpu -Wl,-rpath,.
	./test_vmem
	@echo "=== PASSED ==="
