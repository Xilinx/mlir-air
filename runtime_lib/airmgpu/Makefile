# SPDX-License-Identifier: MIT
# Build libair_mgpu_runtime.so — VMem-backed GPU runtime for MLIR-AIR

ROCM_PATH ?= /opt/rocm
HIPCC     := $(ROCM_PATH)/bin/hipcc

CXXFLAGS  = -std=c++17 -fPIC -O2 -Wall -Wextra -Wno-unused-parameter
INCLUDES  = -I$(ROCM_PATH)/include -I.
LDFLAGS   = -shared -L$(ROCM_PATH)/lib -lamdhip64

TARGET    = libair_mgpu_runtime.so
SOURCES   = air_mgpu_runtime.cpp vmem_allocator.cpp
OBJECTS   = $(SOURCES:.cpp=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(HIPCC) $(LDFLAGS) -o $@ $^

%.o: %.cpp
	$(HIPCC) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(TARGET)

# Quick smoke test: allocate and free via VMem
test: $(TARGET)
	@echo "=== Smoke test: dlopen and call mgpuMemAlloc ==="
	$(HIPCC) $(INCLUDES) -o test_vmem test_vmem.cpp -L. -lair_mgpu_runtime -Wl,-rpath,.
	./test_vmem
	@echo "=== PASSED ==="
