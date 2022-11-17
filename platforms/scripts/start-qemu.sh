#!/bin/bash
#
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# pre-requisite packages for running QEMU
# apt install libncursesw5

if [[ $EUID != 0 ]]; then
	echo "You must be root (i.e. sudo) to run this script because it manipulates hardware devices"
	exit -1
fi

# get the absolute location of the setup script
# we expect the config files to be stored in the same directory
# because this script is run as 'sudo' the environment is different from the user's
# we must use the full path to the realpath program, and drop sudo access temporarily
logname=$(logname)
realpath=$(sudo -u $logname /usr/bin/realpath $0)
script_path=${realpath%/*}
script_path=platforms/scripts
echo "script Path $script_path"

# Get the necessary user-configured paths
. $script_path/env_config
[[ $? != 0 ]] && exit -1


# enable VFIO driver to bind to our device(s)
for devid in "${devids[@]}"; do
	echo "Registering $devid"
	echo $devid > /sys/bus/pci/drivers/vfio-pci/new_id
done

# unbind the device(s) from the host and bind them to VFIO
for bdf in ${bdfs[*]}; do
	bdf_nodomain=${bdf#*:}
	echo "Rebinding $bdf_nodomain"
	if [[ -e /sys/bus/pci/devices/$bdf/driver/unbind ]]; then
		echo $bdf > /sys/bus/pci/devices/$bdf/driver/unbind

		# bind to VFIO driver
		echo $bdf > /sys/bus/pci/drivers/vfio-pci/bind

		if [[ -z $ASSIGN ]]; then
			ASSIGN="-device vfio-pci"
		fi

		# add device assignment to QEMU command line
		ASSIGN+=",host=$bdf_nodomain"
	fi
done


# run QEMU
$QEMU_PATH/qemu-system-x86_64 \
	-machine q35,kernel-irqchip=split \
	-m 1G \
	-nographic \
	-kernel $IMAGE_PATH/bzImage \
	-drive file=$IMAGE_PATH/rootfs.ext2,if=virtio,format=raw \
	-append "rootwait root=/dev/vda console=tty1 console=ttyS0" \
	$ASSIGN  \
	-net nic,model=virtio -net user

# unregister the device IDs from VFIO
for devid in "${devids[@]}"; do
	echo "Unregistering $devid"
	echo $devid > /sys/bus/pci/drivers/vfio-pci/remove_id
done

# unbind the device(s) from VFIO to allow the host to find them again
for bdf in ${bdfs[*]}; do
	if [[ -e /sys/bus/pci/devices/$bdf/driver/unbind ]]; then
		echo "Unbinding $bdf"
		echo $bdf > /sys/bus/pci/devices/$bdf/driver/unbind
	fi
done
