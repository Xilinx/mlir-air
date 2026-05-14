# Multi-process symmetric-heap multi-GPU e2e — air.launch / air.segment /
# air.herd hierarchy + air.rank wrapping.
#
# These tests express the GPU kernel structure declaratively via the AIR
# compute-model hierarchy (per docs/AIRComputeModel.md §2.3, on AMD MI3xx
# the herd is the GPU kernel boundary). After lowering through the
# existing AIR-on-GPU passes, the IR is functionally equivalent to
# ../handwritten/cacheline.mlir.
#
# Lowering chain:
#   air-rank-to-mgpu             # phase 3 — inlines air.rank
#   air-translate-to-llvm        # phase 2 — expands air.translate
#   air-to-rocdl                 # existing — AIR ops → ROCDL primitives
#   air-gpu-outlining            # existing — air.launch/segment/herd → gpu.func
#   <standard MLIR GPU pipeline> # convert-gpu-to-rocdl, gpu-module-to-binary, etc.
#
# Variants (selected via INPUT):
#   cacheline  Wrap of ../handwritten/cacheline.mlir using the full AIR
#              hierarchy. Producer/consumer kernel bodies live inside
#              air.herd ops within an air.launch/air.segment nest.
#
# Usage:
#   make                       # default: INPUT=cacheline NUM_RANKS=2
#   make NUM_RANKS=4
#   make clean
#
# Required environment (auto-detected when sourced via env_setup_gpu.sh):
#   MLIR_AIR_INSTALL_DIR  — path containing lib/libairgpu.so
#   LLVM_INSTALL_DIR      — path containing bin/mlir-opt + lib/libmlir_*.so
#
# This Makefile is intentionally self-contained — no included files, no
# sourced helpers. Other multi_gpu/<level>/ subdirs each have their own
# complete Makefile so that each phase's PR touches only its own dir.

SHELL       := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

INPUT      ?= cacheline
NUM_RANKS  ?= 2
TMPDIR     ?= /tmp/air_multi_gpu_air_hierarchy

SCRIPT_DIR := $(patsubst %/,%,$(dir $(realpath $(firstword $(MAKEFILE_LIST)))))

# Derive install dirs from PATH if not explicitly provided.
LLVM_INSTALL_DIR     ?= $(shell dirname "$$(dirname "$$(command -v mlir-opt)")" 2>/dev/null)
MLIR_AIR_INSTALL_DIR ?= $(shell dirname "$$(dirname "$$(command -v air-opt)")" 2>/dev/null)
LLVM_LIB_DIR         ?= $(LLVM_INSTALL_DIR)/lib
AIRGPU_LIB           ?= $(MLIR_AIR_INSTALL_DIR)/lib/libairgpu.so

ifeq ($(filter $(INPUT),cacheline),)
$(error Unknown INPUT=$(INPUT); expected 'cacheline')
endif

SRC_MLIR    := $(SCRIPT_DIR)/$(INPUT).mlir
POST_LOWER  := $(TMPDIR)/$(INPUT)_post_lower.mlir
LOWERED     := $(TMPDIR)/$(INPUT)_lowered.mlir

.PHONY: run clean check-preconditions
.DEFAULT_GOAL := run

$(TMPDIR):
	@mkdir -p $@

# Step 1a: AIR-level lowerings.
#   air-rank-to-mgpu    — inlines air.rank (resolves rank ID at runtime)
#   air-translate-to-llvm — expands air.translate to memref descriptor rebase
#   air-to-rocdl + air-gpu-outlining — converts air.launch/segment/herd
#                                      into gpu.launch_func + gpu.func
$(POST_LOWER): $(SRC_MLIR) | $(TMPDIR)
	@echo "Step 1a: AIR-level lowering ($(INPUT))"
	air-opt $< -air-rank-to-mgpu --air-translate-to-llvm \
	    -air-to-rocdl -air-gpu-outlining -o $@

# Step 1b: standard MLIR GPU compile chain. Same pipeline as
# ../handwritten/Makefile (the lowered output is structurally a superset
# of the corresponding handwritten test).
$(LOWERED): $(POST_LOWER)
	@echo "Step 1b: Compile gpu.module to AMDGPU binary + finalize host"
	mlir-opt $< \
	    --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary,func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
	    -o $@

check-preconditions:
	@if [ ! -d "$(LLVM_LIB_DIR)" ]; then                                       \
	  echo "ERROR: LLVM_LIB_DIR=$(LLVM_LIB_DIR) does not exist." >&2;         \
	  echo "       Source utils/env_setup_gpu.sh or set LLVM_INSTALL_DIR."    \
	       >&2;                                                                \
	  exit 1;                                                                  \
	fi
	@if [ ! -f "$(AIRGPU_LIB)" ]; then                                         \
	  echo "ERROR: AIRGPU_LIB=$(AIRGPU_LIB) does not exist." >&2;             \
	  echo "       Source utils/env_setup_gpu.sh or set"                      \
	       "MLIR_AIR_INSTALL_DIR." >&2;                                        \
	  exit 1;                                                                  \
	fi
	@if [ "$(NUM_RANKS)" -lt 2 ]; then                                         \
	  echo "ERROR: NUM_RANKS=$(NUM_RANKS); requires >= 2 ranks (producer +"   \
	       "consumer)." >&2;                                                   \
	  exit 1;                                                                  \
	fi
	@if [ -n "$${HIP_VISIBLE_DEVICES:-}" ]; then                               \
	  NUM_GPUS=$$(echo "$$HIP_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .);     \
	else                                                                       \
	  NUM_GPUS=$$(grep -l '^simd_count [1-9]'                                  \
	      /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | wc -l); \
	fi;                                                                        \
	if [ "$$NUM_GPUS" -lt "$(NUM_RANKS)" ]; then                               \
	  echo "ERROR: need >= $(NUM_RANKS) GPUs to validate cross-rank XGMI"     \
	       "traffic; found $$NUM_GPUS." >&2;                                   \
	  echo "       This test refuses to colocate ranks on a single GPU"      \
	       "because it would silently" >&2;                                    \
	  echo "       bypass the symmetric-heap path and report false PASSes." \
	       >&2;                                                                \
	  exit 1;                                                                  \
	fi

run: check-preconditions $(LOWERED)
	@echo "Step 2: Run as $(NUM_RANKS) processes"
	@export AIRGPU_JOB_ID="$${AIRGPU_JOB_ID:-$$$$}";                     \
	PIDS=();                                                              \
	PASS=1;                                                               \
	for i in $$(seq 0 $$(($(NUM_RANKS) - 1))); do                         \
	  ( set -o pipefail;                                                  \
	    RANK=$$i WORLD_SIZE=$(NUM_RANKS) LOCAL_RANK=0                     \
	    HIP_VISIBLE_DEVICES=$$i                                           \
	    mlir-runner --entry-point-result=void                             \
	        --shared-libs="$(LLVM_LIB_DIR)/libmlir_rocm_runtime.so"       \
	        --shared-libs="$(AIRGPU_LIB)"                                 \
	        --shared-libs="$(LLVM_LIB_DIR)/libmlir_runner_utils.so"       \
	        --shared-libs="$(LLVM_LIB_DIR)/libmlir_c_runner_utils.so"     \
	        $(LOWERED) 2>&1 | sed "s/^/[rank $$i] /") &                   \
	  PIDS+=($$!);                                                        \
	done;                                                                 \
	for pid in "$${PIDS[@]}"; do                                          \
	  if ! wait "$$pid"; then PASS=0; fi;                                 \
	done;                                                                 \
	if [ $$PASS -eq 1 ]; then                                             \
	  echo "=== ALL $(NUM_RANKS) RANKS PASSED ===";                       \
	else                                                                  \
	  echo "=== SOME RANKS FAILED ===";                                   \
	  exit 1;                                                             \
	fi

clean:
	rm -rf $(TMPDIR)
