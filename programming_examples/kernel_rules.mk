# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Rules for compiled kernel outputs (.o or .ll). Each output is rebuilt only
# when its source, a header that source includes, its flags or the compiler
# changed.
#
# Set before including this file:
#   KERNEL_CXX     the compiler
#   KERNEL_OUTS    the output files
#   <out>_SRC      source of each output
#   <out>_FLAGS    all flags for each output, including -c or -S -emit-llvm,
#                  quoted for the shell
#
# Each output keeps its flags in <out>.flags and its header dependencies in
# <out>.d, next to it.
#
# With AIR_COMPILE_CACHE set, each compile goes through
# python/air/backend/compile_cache.py, which reuses an identical earlier one.

# These rules must not become the including Makefile's default goal.
_kernel_rules_goal := $(.DEFAULT_GOAL)

_kernel_cache := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))../python/air/backend/compile_cache.py)
KERNEL_LAUNCHER ?= $(if $(AIR_COMPILE_CACHE),$(or $(PYTHON),python3) $(_kernel_cache) cc)

.PHONY: _kernel_flags_check

# Everything other than headers that decides an output. <out>.flags is
# rewritten only when it changes, so its date tells the output when to rebuild.
# $(file) writes it as is, so quotes in the flags survive.
_kernel_sig = $(KERNEL_CXX) $($(1)_SRC) $($(1)_FLAGS)

define _kernel_rule
$(1).flags: _kernel_flags_check
	@$$(shell mkdir -p $$(@D))$$(file >$$@.new,$$(call _kernel_sig,$(1)))
	@cmp -s $$@.new $$@ && rm $$@.new || mv $$@.new $$@

$(1): $$($(1)_SRC) $(1).flags $$(KERNEL_CXX)
	@echo "    $$(notdir $$<) -> $$(notdir $$@)"
	@$$(KERNEL_LAUNCHER) $$(KERNEL_CXX) $$($(1)_FLAGS) -MMD -MP -MF $$@.d $$< -o $$@
endef

$(foreach o,$(KERNEL_OUTS),$(eval $(call _kernel_rule,$(o))))

-include $(KERNEL_OUTS:=.d)

.DEFAULT_GOAL := $(_kernel_rules_goal)
