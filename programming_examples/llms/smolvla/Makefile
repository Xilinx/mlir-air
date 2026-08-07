# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# SmolVLA on NPU2 (MLIR-AIR) — SigLIP vision encoder + connector on the NPU.
#
# Pipeline: NPU vision tower + connector -> CPU prefix assembly -> CPU language
# backbone -> CPU action expert, all in ONE process. The backbone and the action
# expert stay on the CPU because they measure SLOWER on the NPU at their shapes
# (seq 256 and seq 50); see README.md.
#
# Correctness is a continuous-output REGRESSION gate on the (1,50,6) action
# chunk against the unmodified CPU model — not the token-set gate the
# autoregressive LLM siblings use, because SmolVLA does not emit tokens.
#
# One process: with the mlir-air environment sourced and requirements.txt
# installed, one interpreter has torch + lerobot AND air + pyxrt, so the NPU
# stage runs in-process through smolvla_runtime, with weights, ELFs and the XRT
# context loaded once.
#
# On a machine where several sessions share one NPU, wrap the device-touching
# targets the way the sibling examples are wrapped:
#     flock -x -w 1800 /tmp/mlir-air-npu.lock make verify
# The recipes deliberately do NOT take that lock themselves — doing so would
# deadlock exactly that invocation. `shared/infra/cache.py` separately holds
# /tmp/npu.lock around each dispatch, which keeps concurrent runs correct; the
# outer lock is what keeps a timing run from being interleaved with someone
# else's.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# The interpreter that runs the model. It needs torch + lerobot (installed by
# requirements.txt) AND air + pyxrt (from the sourced mlir-air environment) in
# ONE process, which is what makes the in-process NPU path work. Defaults to
# the environment's python3; override if your lerobot install lives elsewhere:
#     make verify LEROBOT_PYTHON=/path/to/venv/bin/python
LEROBOT_PYTHON ?= python3
NPU_PYTHON     ?= python3
BUILD_DIR      ?= build
REPS           ?= 5

# Input selection, shared by run / verify.
#   INPUT=synthetic  seeded-random images; deterministic, nothing to
#                    download. The default, and what the gate and the lit
#                    tests run.
#   INPUT=real       frames from a LeRobot dataset; downloads on first use.
# CAMERAS is how many feeds to supply. The model accepts any non-empty subset
# and the NPU needs no recompile -- the count is only how many times the host
# vision loop runs. 3 is the shipping configuration.
INPUT          ?= synthetic
CAMERAS        ?= 3
DATASET        ?= lerobot/droid_100
FRAMES         ?= 100

.PHONY: help compile cpu-baseline run verify profile clean

help:
	@echo "=================================================================="
	@echo " SmolVLA on NPU2 — SigLIP vision encoder on the NPU, one process"
	@echo "=================================================================="
	@echo "  make compile   Build every vision ELF (no NPU dispatch, no download)"
	@echo "  make cpu-baseline  Run the unmodified CPU model; dumps it for inspection"
	@echo "  make run       One end-to-end forward; prints the action chunk"
	@echo "  make verify    THE GATE: action chunk vs the pure-CPU model (PASS/FAIL)"
	@echo "  make profile   CPU vs NPU, interleaved + warmed, with the ELF breakdown"
	@echo "  make clean     Remove the kernel cache and build artifacts"
	@echo ""
	@echo "  run / verify take INPUT=synthetic|real and CAMERAS=1|2|3"
	@echo "  profile takes CAMERAS only (pixel values do not change NPU work)"
	@echo "    make verify INPUT=real            survey 100 real frames, always exits 0"
	@echo "    make verify CAMERAS=1             synthetic, 1 camera"
	@echo ""
	@echo "  LEROBOT_PYTHON=$(LEROBOT_PYTHON)"
	@echo "  (needs torch + lerobot; with the mlir-air env sourced it also has"
	@echo "   air/pyxrt, which is what makes the single-process path work)"

## Build every vision ELF through AIR -> AIE -> aiecc -> Peano.
## No NPU dispatch and no HuggingFace download, so this runs anywhere the
## toolchain is installed.
compile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(NPU_PYTHON) $(srcdir)/smolvla_inference.py --compile-only

## Run the unmodified lerobot model on CPU and dump the action chunk to
## smolvla_oracle.npz. For inspecting what the baseline does -- `make verify`
## does NOT read this file, it computes its reference live.
## CPU only — no NPU, so no lock.
cpu-baseline:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(LEROBOT_PYTHON) $(srcdir)/smolvla_cpu_baseline.py

## One end-to-end forward. INPUT/CAMERAS select the input; the defaults are
## the shipping configuration.
run:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(LEROBOT_PYTHON) $(srcdir)/smolvla_inference.py \
		--input $(INPUT) --cameras $(CAMERAS) --dataset $(DATASET)

## THE GATE at its defaults (INPUT=synthetic, CAMERAS=3): action chunk vs the
## unmodified CPU model, cosine >= 0.99 and normalized MSE <= 0.04, exits
## non-zero on FAIL. This is what CI and the lit test run.
##
## INPUT=real surveys FRAMES frames of DATASET instead and reports the
## distribution. It always exits 0 -- it answers how much headroom the
## precision has, not whether the build is broken.
verify:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(LEROBOT_PYTHON) $(srcdir)/verify_adapter.py \
		--input $(INPUT) --cameras $(CAMERAS) \
		--dataset $(DATASET) --frames $(FRAMES)

## Pure CPU against NPU vision: one process, both arms warmed, interleaved,
## median of REPS, plus the per-ELF device breakdown.
profile:
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(LEROBOT_PYTHON) $(srcdir)/smolvla_inference.py \
		--profile --reps $(REPS) --cameras $(CAMERAS)

clean:
	rm -rf $(srcdir)/$(BUILD_DIR) $(srcdir)/__pycache__
	rm -rf $(srcdir)/../verify/reports
	@echo "$(BUILD_DIR)/ and verify/reports/ removed. Run 'make compile' to rebuild."
