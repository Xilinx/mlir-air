# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The decode kernels, described for ../kernel_rules.mk. Building several
# templates in a row compiles the kernels once.
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

# The decode-kernels target must not become the including Makefile's default
# goal.
_decode_kernels_goal := $(.DEFAULT_GOAL)

LL_OPT ?= O1

DECODE_KERNEL_FILES := $(O2_KERNELS:%=$(KERNEL_DIR)/%.o) \
                       $(LL_KERNELS:%=$(KERNEL_DIR)/%.ll)
DECODE_KERNEL_CLEAN := $(DECODE_KERNEL_FILES) $(DECODE_KERNEL_FILES:=.d) \
                       $(DECODE_KERNEL_FILES:=.flags)

# KERNEL_FLAGS may run a builder query, so it is expanded once, on first use,
# rather than once per kernel.
_decode_flags = $(eval _decode_flags := $$(KERNEL_FLAGS))$(_decode_flags)

_decode_src = $(or $(firstword $(wildcard $(KERNEL_SRC_DIRS:%=%/$(1).cc))),$(1).cc)

$(foreach k,$(O2_KERNELS), \
  $(eval $(KERNEL_DIR)/$(k).o_SRC := $(call _decode_src,$(k))) \
  $(eval $(KERNEL_DIR)/$(k).o_FLAGS = \
    $$(_decode_flags) $$(O2_EXTRA) $$($(k)_FLAGS) -O2 -c))
$(foreach k,$(LL_KERNELS), \
  $(eval $(KERNEL_DIR)/$(k).ll_SRC := $(call _decode_src,$(k))) \
  $(eval $(KERNEL_DIR)/$(k).ll_FLAGS = \
    $$(_decode_flags) $$(LL_EXTRA) $$($(k)_FLAGS) -$$(LL_OPT) \
    -DDECODE_INLINE_ATTN -S -emit-llvm))

KERNEL_CXX = $(PEANO_INSTALL_DIR)/bin/clang++
KERNEL_OUTS := $(DECODE_KERNEL_FILES)
include $(dir $(lastword $(MAKEFILE_LIST)))../kernel_rules.mk

.PHONY: decode-kernels
decode-kernels: $(DECODE_KERNEL_FILES)
	@:

.DEFAULT_GOAL := $(_decode_kernels_goal)
