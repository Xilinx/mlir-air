# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Decode kernels as make targets. A kernel is recompiled only when its source, a
# header it includes, its flags or the Peano compiler changed, so building
# several templates in a row compiles the kernels once.
#
# Set before including this file:
#   KERNEL_DIR       directory the .o / .ll files are written to
#   KERNEL_SRC_DIRS  directories searched, in order, for <name>.cc
#   KERNEL_FLAGS     Peano flags for every kernel
#   O2_KERNELS       kernels built to KERNEL_DIR/<name>.o at -O2, plus O2_EXTRA
#   LL_KERNELS       kernels built to KERNEL_DIR/<name>.ll at -$(LL_OPT) with
#                    -DDECODE_INLINE_ATTN, plus LL_EXTRA. The builder merges
#                    these into the core (link_with_mode="merge") rather than
#                    linking an object.
#   <name>_FLAGS     optional flags for one kernel only
#
# Then build them with the decode-kernels target, or depend on
# $(DECODE_KERNEL_FILES). A clean rule removes $(DECODE_KERNEL_CLEAN).

# These rules must not become the including Makefile's default goal.
_decode_kernels_goal := $(.DEFAULT_GOAL)

LL_OPT ?= O1

DECODE_KERNEL_FILES := $(O2_KERNELS:%=$(KERNEL_DIR)/%.o) \
                       $(LL_KERNELS:%=$(KERNEL_DIR)/%.ll)
KERNEL_FLAGS_FILE := $(KERNEL_DIR)/.kernel_flags
DECODE_KERNEL_CLEAN := $(DECODE_KERNEL_FILES) $(DECODE_KERNEL_FILES:=.d) \
                       $(KERNEL_FLAGS_FILE)
PEANO_CXX = $(PEANO_INSTALL_DIR)/bin/clang++

vpath %.cc $(KERNEL_SRC_DIRS)

# Everything that decides what a kernel compiles to, other than its sources.
# The file is rewritten only when this changes, so its date tells the kernels
# when to rebuild.
_kernel_flags = $(PEANO_CXX) $(KERNEL_FLAGS) O2: $(O2_EXTRA) \
  LL: -$(LL_OPT) $(LL_EXTRA) \
  $(foreach k,$(O2_KERNELS) $(LL_KERNELS),$(k): $($(k)_FLAGS))

.PHONY: decode-kernels _kernel_flags_check
decode-kernels: $(DECODE_KERNEL_FILES)
	@:

$(KERNEL_FLAGS_FILE): _kernel_flags_check
	@mkdir -p $(KERNEL_DIR)
	@echo '$(_kernel_flags)' | cmp -s - $@ || echo '$(_kernel_flags)' > $@

$(KERNEL_DIR)/%.o: %.cc $(KERNEL_FLAGS_FILE) $(PEANO_CXX)
	@echo "    $(notdir $<) -> $(notdir $@) (-O2)"
	@$(PEANO_CXX) $(KERNEL_FLAGS) $(O2_EXTRA) $($*_FLAGS) -O2 \
	  -MMD -MP -MF $@.d -c $< -o $@

$(KERNEL_DIR)/%.ll: %.cc $(KERNEL_FLAGS_FILE) $(PEANO_CXX)
	@echo "    $(notdir $<) -> $(notdir $@) (-$(LL_OPT), inline .ll)"
	@$(PEANO_CXX) $(KERNEL_FLAGS) $(LL_EXTRA) $($*_FLAGS) -$(LL_OPT) \
	  -DDECODE_INLINE_ATTN -S -emit-llvm -MMD -MP -MF $@.d $< -o $@

-include $(DECODE_KERNEL_FILES:=.d)

.DEFAULT_GOAL := $(_decode_kernels_goal)
