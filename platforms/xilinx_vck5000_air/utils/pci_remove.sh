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
