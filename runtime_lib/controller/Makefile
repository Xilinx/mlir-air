# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# explicit list of known platforms
platforms = vck190 vck5000

# a specific platform can be built by calling:
# make <platform>
#
# All platforms are built without any parameters:
# make
#
# A specific platform can be cleaned by calling:
# make <platform>-clean
#
# All platforms are cleaned by calling:
# make clean

.PHONY: all
all: $(platforms)

define PLATFORM_RULES =
 $(1):
	$(MAKE) -f Makefile.$(1)
 $(1)-clean:
	$(MAKE) -f Makefile.$(1) clean
endef

# This expands to dynamically create rules for each platform
$(foreach prog,$(platforms),$(eval $(call PLATFORM_RULES,$(prog))))

# This aggregates the 'clean' targets for all platforms
clean_targets=$(foreach plat,$(platforms),$(plat)-clean)

clean: $(clean_targets)
