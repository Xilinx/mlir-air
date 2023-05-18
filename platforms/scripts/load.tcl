# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Load either a PDI file or ELF file (or both) to a VCK5000
# Intended to be used with xsct, with JTAG over USB
#
# If you are loading an FPGA image (PDI file) it is recommended to use the
# shell script wrapper (e.g. program_vck5000.sh) instead, since it performs
# additional actions such as removing the PCIe device from Linux beforehand.
#
# If you only want to load firmware, it can be done like this:
#    ELF_FILE=acdc_agent.elf xsct load.tcl
#
# A specific card can be targeted when more than one is installed:
#    CARD_IDX=1 ELF_FILE=acdc_agent.elf xsct load.tcl

# if the card index is not set, default to 0
if {[info exists ::env(CARD_IDX)]} {
	set card_idx $::env(CARD_IDX)
} else {
	set card_idx 0
}
puts "Using card index $card_idx"

if {[info exists ::env(PDI_FILE)]} {
	set pdi_file $::env(PDI_FILE)
	puts "Using PDI $pdi_file"
}

if {[info exists ::env(ELF_FILE)]} {
	set elf_file $::env(ELF_FILE)
	puts "Using ELF $elf_file"
}

# Initiate hardware manager's server
connect

# get a list of all VCK5000 cards
set cards [ targets -target-properties -filter { name =~ "Versal xcvc1902*" }]

# validate the selected card index
set num_cards [ llength $cards ]
if { $card_idx > $num_cards } {
	puts "Invalid card index $card_idx; only $num_cards detected"
	exit
}

# get the JTAG index and context of the board
# The index is used for resetting, the context is used to distinguish child
# nodes with the same name belonging to different cards
set ta_board [ dict get [ lindex $cards $card_idx ] target_id ]
set card_jtag_ctx [ dict get [ lindex $cards $card_idx ] jtag_device_ctx ]

# get the PMC target ID
# compare with the parent JTAG context to be sure we found the right one
set pmcs [ targets -target-properties -filter { name =~ "PMC*" }]
foreach pmc $pmcs {
	if { [ dict get $pmc parent_ctx ] == "JTAG-$card_jtag_ctx" } {
		set ta_pmc [ dict get $pmc target_id ]
	}
}
if { ! [info exists ta_pmc]} {
	puts "Couldn't find a valid PMC target!"
	exit
}

# get the ARM target ID
# The ARM core #0 is a child of the APU, which is a child of the board
set apus [ targets -target-properties -filter { name =~ "APU*" }]
foreach apu $apus {
	if { [ dict get $apu parent_ctx ] == "JTAG-$card_jtag_ctx" } {
		set apu_jtag_ctx [ dict get $apu jtag_device_ctx ]
	}
}
if { ! [info exists apu_jtag_ctx]} {
	puts "Couldn't find a valid APU target!"
	exit
}
foreach arm [ targets -target-properties -filter { name =~ "Cortex-A72 #0" }] {
	if  {[ dict get $arm jtag_device_ctx ] == $apu_jtag_ctx } {
		set ta_arm0 [ dict get $arm target_id ]
	}
}
if { ! [info exists ta_arm0]} {
	puts "Couldn't find a valid ARM target!"
	exit
}

if {[info exists pdi_file]} {
	# reset the board
	ta $ta_board
	rst

	# target the PMC and program it with the PDI file
	ta $ta_pmc
	device program $pdi_file
}

# select ARM processor 0 and load ELF
# If the processor gets into suspended state, this will fail:
# 6  Cortex-A72 #0 (Suspended, EL3(S)/A64)
if {[info exists elf_file]} {
	ta $ta_arm0
	rst -processor -clear-registers -skip-activate-subsystem
	dow $elf_file
	con
}
