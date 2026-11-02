# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Gemma4-E2B image + text on NPU2. The vision tower lives here; the text model
# is ../gemma4_e2b_q4nx, whose fused prefill and fused decode this example
# drives (`make compile` builds both there).

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
TEXT := $(srcdir)/../gemma4_e2b_q4nx

IMAGE ?=
IMAGE_ARG := $(if $(IMAGE),--image $(IMAGE),)
PROMPT ?=
PROMPT_ARG := $(if $(PROMPT),--prompt "$(PROMPT)",)
## Soft tokens per image (70, 140, 280, 560, 1120); empty = the model's 280.
MST ?=
MST_ARG := $(if $(MST),--max-soft-tokens $(MST),)
N_TOKENS ?= 100
## Longest context the decode serves: a 1120-soft-token image turn is ~1100
## prompt tokens, plus N_TOKENS.
RUN_LBUILD ?= 2048
FUSED_DIR ?= $(CURDIR)/_fused_prefill
override FUSED_DIR := $(abspath $(FUSED_DIR))

## Generated files (host library, ELF cache) go under the directory make runs
## in, so builds in different directories do not share them.
export G4_VISION_BUILD_DIR ?= $(CURDIR)/_vision_build

## Tower gate floor, NPU soft tokens vs HF.
TOWER_COS ?= 0.99
## Turns timed by `make profile`, after one warm-up turn, at PROFILE_MST soft
## tokens per image.
REPEAT ?= 3
PROFILE_MST ?= 70

## Image-token budgets `make profile-image-sweep` measures.
IMAGE_BUDGETS ?= 70,140,280,560,1120

.PHONY: help compile reference vision diagnosis run verify verify-image profile \
        profile-image-sweep clean

help:
	@echo "Gemma4-E2B vision (NPU2)"
	@echo "  make compile       Text model's fused prefill + decode (../gemma4_e2b_q4nx)"
	@echo "  make reference     NumPy tower vs HF (no NPU)"
	@echo "  make vision        NPU tower vs HF (cosine >= $(TOWER_COS))"
	@echo "  make diagnosis     NPU tower, per-layer cosine vs the numpy tower"
	@echo "  make run           Describe IMAGE (default: the shapes image)"
	@echo "  make verify        Keyword gate on the shapes image + image-turn top-k gate"
	@echo "  make profile       TTFT and decode tok/s, model resident"
	@echo "  make profile-image-sweep  TTFT per image-token budget ($(IMAGE_BUDGETS))"
	@echo "  IMAGE=<path> PROMPT=<text> MST=<soft tokens> N_TOKENS=$(N_TOKENS)"

compile:
	@$(MAKE) --no-print-directory -f $(TEXT)/Makefile compile-fused-prefill \
	  compile-decode-elf FUSED_DIR=$(FUSED_DIR) RUN_LBUILD=$(RUN_LBUILD) \
	  PEANO_INSTALL_DIR=$(PEANO_INSTALL_DIR)

reference:
	$(PYTHON) $(srcdir)/gemma4_vision_reference.py $(IMAGE_ARG) $(MST_ARG)

vision:
	$(PYTHON) $(srcdir)/run_vision.py --min-cosine $(TOWER_COS) $(IMAGE_ARG) $(MST_ARG)

diagnosis:
	$(PYTHON) $(srcdir)/run_vision.py --diagnosis $(IMAGE_ARG) $(MST_ARG)

run:
	$(PYTHON) $(srcdir)/gemma4_vision_inference.py --fused-dir $(FUSED_DIR) \
	  --n-tokens $(N_TOKENS) $(IMAGE_ARG) $(PROMPT_ARG) $(MST_ARG)

## Two gates: the description of the shapes image must name both shapes, their
## colors and a position; and the first 32 tokens of each image turn in
## verify_image.py must pass the top-k check against HF bf16.
verify: verify-image
	$(PYTHON) $(srcdir)/gemma4_vision_inference.py --fused-dir $(FUSED_DIR) \
	  --n-tokens $(N_TOKENS) --gate $(MST_ARG)

verify-image:
	$(PYTHON) $(srcdir)/verify_image.py --fused-dir $(FUSED_DIR)

profile:
	$(PYTHON) $(srcdir)/gemma4_vision_inference.py --fused-dir $(FUSED_DIR) \
	  --n-tokens $(N_TOKENS) --repeat $(REPEAT) --max-soft-tokens $(PROFILE_MST) \
	  $(IMAGE_ARG) $(PROMPT_ARG)

## TTFT and decode tok/s at each image-token budget, written to
## gemma4_e2b_vision.image_sweep.json in the current directory.
profile-image-sweep:
	$(PYTHON) $(srcdir)/bench_image.py --fused-dir $(FUSED_DIR) \
	  --budgets $(IMAGE_BUDGETS) --repeat $(REPEAT) $(IMAGE_ARG)

clean:
	rm -rf $(G4_VISION_BUILD_DIR) air_project
