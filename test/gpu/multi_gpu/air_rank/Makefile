# Multi-process symmetric-heap multi-GPU e2e — air.rank wrapped tests.
#
# These tests express the multi-process world declaratively via
# `air.rank (%rid) in (%rsize = %c2) { ... }`. The air-rank-to-mgpu
# pass (Phase 3) replaces the air.rank op with body-inlined IR that
# resolves %rid from mgpuGetRank() at runtime and brackets the
# enclosing function with mgpuSymmetricHeapInit / Destroy.
#
# Each variant in this dir is a 1:1 wrap of the corresponding test in
# ../handwritten/. After lowering through air-rank-to-mgpu the IR is
# functionally equivalent to the handwritten reference.
#
# Variants (selected via INPUT):
#   cacheline  Wrap of ../handwritten/cacheline.mlir (producer/consumer,
#              1-to-1, cache-line atomicity).
#   allgather  Wrap of ../handwritten/allgather.mlir (many-to-many SIMD,
#              cache-line atomicity).
#
# Usage:
#   make                       # default: INPUT=cacheline NUM_RANKS=2
#   make INPUT=allgather
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
TMPDIR     ?= /tmp/air_multi_gpu_air_rank

SCRIPT_DIR := $(patsubst %/,%,$(dir $(realpath $(firstword $(MAKEFILE_LIST)))))

# Derive install dirs from PATH if not explicitly provided. Matches the
# original run.sh fallback (`dirname $(dirname $(which mlir-opt))`).
LLVM_INSTALL_DIR     ?= $(shell dirname "$$(dirname "$$(command -v mlir-opt)")" 2>/dev/null)
MLIR_AIR_INSTALL_DIR ?= $(shell dirname "$$(dirname "$$(command -v air-opt)")" 2>/dev/null)
LLVM_LIB_DIR         ?= $(LLVM_INSTALL_DIR)/lib
AIRGPU_LIB           ?= $(MLIR_AIR_INSTALL_DIR)/lib/libairgpu.so

# Auto-detect the GPU chip target. Override via `make GFX_TARGET=gfx942`
# for cross-compilation or on heterogeneous systems.
# Detection order:
#   1. amdgpu-arch (ROCm's clang tool; ships at /opt/rocm/llvm/bin/).
#   2. KFD topology: gfx_target_version is major*10000 + minor*100 + step,
#      and the gfx name is gfx<major><minor><step> (e.g. 90500 -> gfx950,
#      90402 -> gfx942). Holds for the MI3xx family this test targets.
GFX_TARGET ?= $(or \
    $(shell { command -v amdgpu-arch || echo /opt/rocm/llvm/bin/amdgpu-arch; } | xargs -r -I{} {} 2>/dev/null | head -1), \
    $(shell awk '/^gfx_target_version [1-9]/{v=$$2; printf "gfx%d%d%d", v/10000, (v%10000)/100, v%100; exit}' /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null))

ifeq ($(filter $(INPUT),cacheline allgather),)
$(error Unknown INPUT=$(INPUT); expected 'cacheline' or 'allgather')
endif

SRC_MLIR  := $(SCRIPT_DIR)/$(INPUT).mlir
POST_RANK := $(TMPDIR)/$(INPUT)_post_rank.mlir
LOWERED   := $(TMPDIR)/$(INPUT)_lowered.mlir

.PHONY: run clean check-preconditions
.DEFAULT_GOAL := run

$(TMPDIR):
	@mkdir -p $@

# Step 1a: lower air.rank to mgpu* runtime + expand air.translate.
$(POST_RANK): $(SRC_MLIR) | $(TMPDIR)
	@echo "Step 1a: Lower air.rank to mgpu* + expand air.translate ($(INPUT))"
	air-opt $< -air-rank-to-mgpu --air-translate-to-llvm -o $@

# Step 1b: compile gpu.module to AMDGPU binary + finalize host. Same
# pipeline as ../handwritten/Makefile (the lowered output is structurally
# a superset of the corresponding handwritten test).
$(LOWERED): $(POST_RANK)
	@echo "Step 1b: Compile gpu.module to AMDGPU binary + finalize host"
	mlir-opt $< \
	    --pass-pipeline='builtin.module(rocdl-attach-target{chip=$(GFX_TARGET) O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=$(GFX_TARGET) runtime=HIP},reconcile-unrealized-casts),gpu-module-to-binary,func.func(gpu-async-region,convert-scf-to-cf),gpu-to-llvm,convert-to-llvm,reconcile-unrealized-casts)' \
	    -o $@

# Refuse to launch if NUM_RANKS < 2 (no peer to talk to), if fewer
# physical GPUs than NUM_RANKS (would silently bypass XGMI and report
# false-positive PASSes by colocating ranks on one GPU), or if the
# install paths are missing (mlir-runner would fail at dlopen with a
# more cryptic message).
check-preconditions:
	@if [ -z "$(GFX_TARGET)" ]; then                                           \
	  echo "ERROR: Could not auto-detect GFX_TARGET (chip name like gfx950)." \
	       >&2;                                                                \
	  echo "       Install ROCm so /opt/rocm/llvm/bin/amdgpu-arch exists, or" \
	       "set GFX_TARGET=gfx<NNN> explicitly." >&2;                          \
	  exit 1;                                                                  \
	fi
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

# Step 2: fork NUM_RANKS processes, each pinned to its own GPU via
# HIP_VISIBLE_DEVICES. mlir-runner's gpu.launch_func handler (and any
# nested call into libmlir_rocm_runtime.so) only ever sees one device,
# so it can't accidentally launch on the wrong one. Every rank still
# sees device 0 internally, so airgpu uses LOCAL_RANK=0.
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
