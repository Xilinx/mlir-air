#!/bin/bash

#===----------------------------------------------------------------------===
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#===----------------------------------------------------------------------===

# Search for a PCI device by vendor ID/device ID, and remove it
# This prevents drivers from accessing the device and helps ensure
# that it will be quiescent during FPGA & firmware updates

function usage()
{
	echo "Remove a PCI device by vendor ID/device ID"
	echo "$0 <vendor id> <device id>"
	echo ""
}

# sudo is required to remove a PCI device
#if [[ $EUID != 0 ]]; then
#	echo "You must be root to run this script"
#	exit -5
#fi

if [[ -z $1 ]]; then
	echo "You must specify a vendor ID"
	usage
	exit -1
fi
if [[ -z $2 ]]; then
	echo "You must specify a device ID"
	usage
	exit -2
fi
vendor_id=$1
device_id=$2

cmd="lspci -D -d $vendor_id:$device_id"
dev=$($cmd)

if [[ $? != 0 ]]; then
	# echo "Bad vendor ID or device ID"
	exit -3
fi

if [[ -z $dev ]]; then
	echo "No device found"
	exit -4
fi

echo "Removing $dev"

# extract BDF of the device
bdf=${dev%%\ *}

echo 1 > "/sys/bus/pci/devices/$bdf/remove"
