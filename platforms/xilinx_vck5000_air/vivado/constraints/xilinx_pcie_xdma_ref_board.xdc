# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

#
##########################################################################################################################
# Vivado - PCIe GUI / User Configuration
##########################################################################################################################
#
# Link Speed   - Gen3 - 8.0 Gb/s
# Link Width   - X4
# AXIST Width  - 256-bit
# AXIST Frequ  - 125
# Core Clock   - 500 MHz
# Pipe Clock   - 125 MHz (Gen1) / 250 MHz (Gen2/Gen3/Gen4) / 500 MHz (Gen4)
#
# Family       - versal
# Part         - xcvc1902
# Package      - vsva2197
# Speed grade  - -2MP
# Xilinx RefBrd- VCK190_ES
#
##########################################################################################################################
# # # #                            User Time Names / User Time Groups / Time Specs                                 # # # #
##########################################################################################################################
create_clock -period 10.000 -name sys_clk [get_ports sys_clk_p]
#create_clock -name aclk -period 8 [get_nets aclk0_clk]
#
#set_property IOSTANDARD LVCMOS18 [get_ports sys_rst_n]
#set_property PACKAGE_PIN K35 [get_ports sys_rst_n]
#set_false_path -from [get_ports sys_rst_n]
#set_property PULLUP true [get_ports sys_rst_n]
#
##########################################################################################################################
# # # #                                                                                                            # # # #
##########################################################################################################################
## sys_rst
#set_property PACKAGE_PIN H34 [get_ports led_0]
## user_link_up
#set_property PACKAGE_PIN J33 [get_ports led_1]
## Clock Up/Heart Beat(HB)
#set_property PACKAGE_PIN K36 [get_ports led_2]
## cfg_current_speed[0] => 00: Gen1; 01: Gen2; 10:Gen3; 11:Gen4
#set_property PACKAGE_PIN L35 [get_ports led_3]
#set_property IOSTANDARD LVCMOS18 [get_ports led_*]
##########################################################################################################################
# # # #                                                                                                            # # # #
##########################################################################################################################
#

#set_property USER_CLOCK_ROOT {X9Y2} [get_nets -of_objects [get_pins -hierarchical -filter NAME=~*/phy_clk_i/bufg_gt_coreclk/O]]
#set_property USER_CLOCK_ROOT {X9Y2} [get_nets -of_objects [get_pins -hierarchical -filter NAME=~*/phy_clk_i/bufg_gt_pclk/O]]
#set_property USER_CLOCK_ROOT {X9Y2} [get_nets -of_objects [get_pins -hierarchical -filter NAME=~*/phy_clk_i/bufg_gt_userclk/O]]
#set_property LOC GTY_REFCLK_X1Y0 [get_cells -hierarchical -filter REF_NAME==IBUFDS_GTE5]
#set_property LOC GTY_QUAD_X1Y0   [get_cells $gt_quads -filter NAME=~*/gt_quad_0/*]

#set PCIEINST serial_pcie_top.pcie_4_0_pipe_inst/pcie_4_0_e5_inst
#set_property LOC PCIE40_X1Y0 [get_cells $PCIEINST]
set_property LOC PCIE40_X0Y1 [get_cells qdma_ep_i/qdma_host_mem_support/pcie/inst/serial_pcie_top.pcie_4_0_pipe_inst/pcie_4_0_e5_inst]

#
set_property PACKAGE_PIN R36 [get_ports sys_clk_p]
set_property PACKAGE_PIN R37 [get_ports sys_clk_n]

#set_property PACKAGE_PIN AB28 [get_ports {pci_exp_rxp[0]}]
set_property PACKAGE_PIN AC41 [get_ports {pci_exp_rxp[0]}]
#set_property PACKAGE_PIN AB39 [get_ports {pci_exp_rxn[0]}]
set_property PACKAGE_PIN AC42 [get_ports {pci_exp_rxn[0]}]
set_property PACKAGE_PIN AB38 [get_ports {pci_exp_txp[0]}]
set_property PACKAGE_PIN AB39 [get_ports {pci_exp_txn[0]}]
set_property PACKAGE_PIN AA41 [get_ports {pci_exp_rxp[1]}]
set_property PACKAGE_PIN AA42 [get_ports {pci_exp_rxn[1]}]
set_property PACKAGE_PIN AA36 [get_ports {pci_exp_txp[1]}]
set_property PACKAGE_PIN AA37 [get_ports {pci_exp_txn[1]}]
set_property PACKAGE_PIN W41 [get_ports {pci_exp_rxp[2]}]
set_property PACKAGE_PIN W42 [get_ports {pci_exp_rxn[2]}]
set_property PACKAGE_PIN Y38 [get_ports {pci_exp_txp[2]}]
set_property PACKAGE_PIN Y39 [get_ports {pci_exp_txn[2]}]
set_property PACKAGE_PIN U41 [get_ports {pci_exp_rxp[3]}]
set_property PACKAGE_PIN U42 [get_ports {pci_exp_rxn[3]}]
set_property PACKAGE_PIN V38 [get_ports {pci_exp_txp[3]}]
#set_property PACKAGE_PIN V29  [get_ports {pci_exp_txn[3]}]
set_property PACKAGE_PIN V39  [get_ports {pci_exp_txn[3]}]

#########################################################################
set_property BITSTREAM.GENERAL.COMPRESS true [current_design]
set_property BITSTREAM.GENERAL.WRITE0FRAMES No [current_design]
set_property BITSTREAM.GENERAL.PROCESSALLVEAMS true [current_design]
########################################################################
#set_multicycle_path -setup -from [get_pins -hierarchical -filter {NAME =~ *phy_pipeline/pcie_ltssm_state_chain/with_ff_chain.ff_chain_gen[0].sync_rst.ff_chain_reg[1][*]/C}] -to [get_pins -hierarchical -filter {NAME =~ */*gt_quad_*/inst/quad_inst/PCIELTSSM[*]}] 2
#set_multicycle_path -hold  -from [get_pins -hierarchical -filter {NAME =~ *phy_pipeline/pcie_ltssm_state_chain/with_ff_chain.ff_chain_gen[0].sync_rst.ff_chain_reg[1][*]/C}] -to [get_pins -hierarchical -filter {NAME =~ */*gt_quad_*/inst/quad_inst/PCIELTSSM[*]}] 1



