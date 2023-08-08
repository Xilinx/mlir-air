#!/bin/bash
#
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# load a new FPGA image + firmware
# Usually, they are both contained in a single .PDI file. If not, set the ELF_FILE variable in addition to PDI_FILE.
#
# Parameters (as environment variables)
# ELF_FILE				path to the ELF file with the controller firmware
# PDI_FILE				path to the PDI file with the FPGA image
# CARD_IDX				index of the card to be programmed, in JTAG scan-chain order (as seen by 'xsct')
# DEVICE_ID				PCI BDI address of the card to be programmed
# VIVADO_INSTALL_DIR	install directory of Vivado
# DRIVER_DIR	    directory of built driver

VIVADO_INSTALL_DIR=${VIVADO_INSTALL_DIR:=/proj/xbuilds/2022.1_released/installs/lin64/Vivado/2022.1}
DEVICE_ID=${DEVICE_ID:=0000:21:00.0}
DRIVER_DIR=${DRIVER_DIR:="../../driver"}

# these variables are used by load.tcl so they must be exported
export CARD_IDX=${CARD_IDX:=0}

# PDI file is optional
if [ -v PDI_FILE ]; then
	if [[ ! -e $PDI_FILE ]]; then
		echo "Can't find $PDI_FILE"
		exit
	fi
	export PDI_FILE=$PDI_FILE
fi

# ELF file is optional
if [ -v ELF_FILE ]; then
	if [[ ! -e $ELF_FILE ]]; then
		echo "Can't find $ELF_FILE"
		exit
	fi
	export ELF_FILE=$ELF_FILE
fi

# make sure at least one file is being programmed
if [[ ! -v ELF_FILE && ! -v PDI_FILE ]]; then
	echo "You must set at least one of ELF_FILE or PDI_FILE to be programmed"
	exit
fi

# get the path of this script - it is the script path
this_script=$(realpath $0)
script_path=${this_script%/*}

# load environment for hw_server and xsct
. $VIVADO_INSTALL_DIR/settings64.sh

# start the hardware server in daemon mode (if it is not already running)
hw_server -d

# if the driver is loaded, stop it
driver_loaded=0
lsmod | grep amdair
if [[ $? == 0 ]]; then
  echo "Unloading driver"
  sudo rmmod amdair
  driver_loaded=1
fi

# remove the PCI device in case it is already loaded
echo "Removing device"
sudo sh -c "echo 1 > /sys/bus/pci/devices/$DEVICE_ID/remove"

# load the FPGA image and firmware
echo "Loading FPGA and firmware"
xsct $script_path/load.tcl

# rescan PCIe bus
echo "Scanning for new devices"
sudo sh -c "echo '1' > /sys/bus/pci/rescan"

# reload the driver if needed
if [[ $driver_loaded == 1 ]]; then
  echo "Reloading driver"
  sudo insmod $DRIVER_DIR/amdair.ko
fi
