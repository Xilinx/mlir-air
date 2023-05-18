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

VIVADO_INSTALL_DIR=${VIVADO_INSTALL_DIR:=/proj/xbuilds/2022.1_released/installs/lin64/Vivado/2022.1}
DEVICE_ID=${DEVICE_ID:=0000:21:00.0}

# these variables are used by load.tcl so they must be exported
export CARD_IDX=${CARD_IDX:=0}

# PDI file is optional
if [ -v PDI_FILE ]; then
	export PDI_FILE=$PDI_FILE
fi

# ELF file is optional
if [ -v ELF_FILE ]; then
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

# remove the PCI device in case it is already loaded
echo "Removing device"
sudo sh -c "echo 1 > /sys/bus/pci/devices/$DEVICE_ID/remove"

# load the FPGA image and firmware
echo "Loading FPGA and firmware"
xsct $script_path/load.tcl

# rescan PCIe bus
echo "Scanning for new devices"
sudo sh -c "echo '1' > /sys/bus/pci/rescan"
