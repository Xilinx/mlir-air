#!/bin/bash

#===----------------------------------------------------------------------===
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#===----------------------------------------------------------------------===

#These are the two commands that we want to run:
#echo 50 > /proc/sys/vm/nr_hugepages
#sudo setpci -s 09:00.0 COMMAND=0x06
#But only want to run them if necessary

# Checking if master bit is set and if the bar is disabled
BAR_DISABLED=$(lspci -vd 10ee:)

# If the BAR is disabled, will run setpci to enable them
if ! $(grep -q "master" <<< "$BAR_DISABLED")  && $(grep -q "disabled" <<< "$BAR_DISABLED"); then
    echo "Running setpci"
    # This will get the BDF and set the command register in the device
    setpci -s $(lspci -d 10ee: | head -n 1 | awk '{print $1;}') COMMAND=0x06
fi

# If we have no huge pages enabled in the system, will add 50
NUM_HUGE_PAGES=$(cat /proc/meminfo | grep HugePages_Total: | awk -F'HugePages_Total:' '{print $2}')
if [[ $NUM_HUGE_PAGES -eq "0" ]]; then
    echo "Found "$NUM_HUGE_PAGES" huge pages in system. Not enough. Adding 50."
    echo 50 > /proc/sys/vm/nr_hugepages
fi

echo "PCI Initialized"

