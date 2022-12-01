#!/bin/bash

#===----------------------------------------------------------------------===
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#===----------------------------------------------------------------------===

# Getting the BDF of a Xilinx device. 
BAR0_PATH="/sys/bus/pci/devices/0000:"$(lspci -d 10ee: | head -n 1 | awk '{print $1;}')"/resource0"
BAR0_SIZE=$(stat -c %s $BAR0_PATH)
BAR2_PATH="/sys/bus/pci/devices/0000:"$(lspci -d 10ee: | head -n 1 | awk '{print $1;}')"/resource2"
BAR2_SIZE=$(stat -c %s $BAR2_PATH)
BAR4_PATH="/sys/bus/pci/devices/0000:"$(lspci -d 10ee: | head -n 1 | awk '{print $1;}')"/resource4"
BAR4_SIZE=$(stat -c %s $BAR4_PATH)

echo "#define BAR0_DEV_FILE \""$BAR0_PATH"\"" > ${1}/pcie-bdf.h
echo "#define BAR0_SIZE "$BAR0_SIZE >> ${1}/pcie-bdf.h
echo "#define BAR2_DEV_FILE \""$BAR2_PATH"\"" >> ${1}/pcie-bdf.h
echo "#define BAR2_SIZE "$BAR2_SIZE >> ${1}/pcie-bdf.h
echo "#define BAR4_DEV_FILE \""$BAR4_PATH"\"" >> ${1}/pcie-bdf.h
echo "#define BAR4_SIZE "$BAR4_SIZE >> ${1}/pcie-bdf.h
