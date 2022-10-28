#!/bin/bash

#===----------------------------------------------------------------------===
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
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
