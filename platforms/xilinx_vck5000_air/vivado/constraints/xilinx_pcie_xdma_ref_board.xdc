# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
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
create_clock -name aclk -period 8 [get_nets aclk0_clk]
create_clock -period 6.206 -name mrmac_0_gt_ref_clk_p -waveform {0.000 3.103} [get_ports mrmac_0_gt_ref_clk_p]
create_clock -period 6.206 -name mrmac_1_gt_ref_clk_p -waveform {0.000 3.103} [get_ports mrmac_1_gt_ref_clk_p]

set_property LOC PCIE40_X0Y1 [get_cells qdma_ep_i/qdma_host_mem_support/pcie/inst/serial_pcie_top.pcie_4_0_pipe_inst/pcie_4_0_e5_inst]

set_property PACKAGE_PIN R36 [get_ports sys_clk_p]
set_property PACKAGE_PIN R37 [get_ports sys_clk_n]

set_property PACKAGE_PIN AB28 [get_ports {pci_exp_rxp[0]}]
set_property PACKAGE_PIN AB39 [get_ports {pci_exp_rxn[0]}]
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
set_property PACKAGE_PIN V29  [get_ports {pci_exp_txn[3]}]

##### GTY Bank 203
set_property PACKAGE_PIN D2 [get_ports {mrmac_0_gt_rxp_in[0]}]
set_property PACKAGE_PIN C2 [get_ports {mrmac_0_gt_rxn_in[0]}]
set_property PACKAGE_PIN G1 [get_ports {mrmac_0_gt_txp_out[0]}]
set_property PACKAGE_PIN F1 [get_ports {mrmac_0_gt_txn_out[0]}]
set_property PACKAGE_PIN B3 [get_ports {mrmac_0_gt_rxp_in[1]}]
set_property PACKAGE_PIN A3 [get_ports {mrmac_0_gt_rxn_in[1]}]
set_property PACKAGE_PIN G3 [get_ports {mrmac_0_gt_txp_out[1]}]
set_property PACKAGE_PIN F3 [get_ports {mrmac_0_gt_txn_out[1]}]
set_property PACKAGE_PIN D4 [get_ports {mrmac_0_gt_rxp_in[2]}]
set_property PACKAGE_PIN C4 [get_ports {mrmac_0_gt_rxn_in[2]}]
set_property PACKAGE_PIN G5 [get_ports {mrmac_0_gt_txp_out[2]}]
set_property PACKAGE_PIN F5 [get_ports {mrmac_0_gt_txn_out[2]}]
set_property PACKAGE_PIN B5 [get_ports {mrmac_0_gt_rxp_in[3]}]
set_property PACKAGE_PIN A5 [get_ports {mrmac_0_gt_rxn_in[3]}]
set_property PACKAGE_PIN E6 [get_ports {mrmac_0_gt_txp_out[3]}]
set_property PACKAGE_PIN D6 [get_ports {mrmac_0_gt_txn_out[3]}]
# GTREFCLK 0 Configured as Output (Recovered Clock) which connects to 8A34001 CLK1_IN
# GTREFCLK 1 ( Driven by 8A34001 Q1 )
set_property PACKAGE_PIN B15 [get_ports mrmac_0_gt_ref_clk_p]
set_property PACKAGE_PIN A15 [get_ports mrmac_0_gt_ref_clk_n]
# placement for MRMAC core

##### GTY Bank 204
set_property PACKAGE_PIN B7 [get_ports {mrmac_1_gt_rxp_in[0]}]
set_property PACKAGE_PIN A7 [get_ports {mrmac_1_gt_rxn_in[0]}]
set_property PACKAGE_PIN E8 [get_ports {mrmac_1_gt_txp_out[0]}]
set_property PACKAGE_PIN D8 [get_ports {mrmac_1_gt_txn_out[0]}]
set_property PACKAGE_PIN B9 [get_ports {mrmac_1_gt_rxp_in[1]}]
set_property PACKAGE_PIN A9 [get_ports {mrmac_1_gt_rxn_in[1]}]
set_property PACKAGE_PIN E10 [get_ports {mrmac_1_gt_txp_out[1]}]
set_property PACKAGE_PIN D10 [get_ports {mrmac_1_gt_txn_out[1]}]
set_property PACKAGE_PIN B11 [get_ports {mrmac_1_gt_rxp_in[2]}]
set_property PACKAGE_PIN A11 [get_ports {mrmac_1_gt_rxn_in[2]}]
set_property PACKAGE_PIN E12 [get_ports {mrmac_1_gt_txp_out[2]}]
set_property PACKAGE_PIN D12 [get_ports {mrmac_1_gt_txn_out[2]}]
set_property PACKAGE_PIN B13 [get_ports {mrmac_1_gt_rxp_in[3]}]
set_property PACKAGE_PIN A13 [get_ports {mrmac_1_gt_rxn_in[3]}]
set_property PACKAGE_PIN E14 [get_ports {mrmac_1_gt_txp_out[3]}]
set_property PACKAGE_PIN D14 [get_ports {mrmac_1_gt_txn_out[3]}]
# GTREFCLK 0 Configured as Output (Recovered Clock) which connects to 8A34001 CLK1_IN
# GTREFCLK 1 ( Driven by 8A34001 Q1 )
set_property PACKAGE_PIN D18 [get_ports mrmac_1_gt_ref_clk_p]
set_property PACKAGE_PIN C18 [get_ports mrmac_1_gt_ref_clk_n]
# placement for MRMAC core


set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */gt_quad_base/inst/quad_inst/CH0_TXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */i_*_axis_clk_wiz_*/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0}]] 2.800
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */i_*_axis_clk_wiz_*/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */gt_quad_base/inst/quad_inst/CH0_TXOUTCLK}]] 2.800
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH2_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH3_RXOUTCLK}]]
set_false_path -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH0_RXOUTCLK}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */quad_inst/CH1_RXOUTCLK}]]
set_max_delay -datapath_only -from [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */MMCME5_inst/CLKOUT0}]] -to [get_clocks -of_objects [get_pins -hierarchical -filter {NAME =~ */mbufg_gt_*/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1*}]] 1.552

set_false_path -from [get_clocks -of_objects [get_pins inst_mrmac_0/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/BUFG_clkout1_inst/O]] -to [get_clocks -of_objects [get_pins {inst_mrmac_0/i_mrmac_0_exdes_support_wrapper/mrmac_0_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_0/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]]
set_false_path -from [get_clocks -of_objects [get_pins inst_mrmac_1/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/BUFG_clkout1_inst/O]] -to [get_clocks -of_objects [get_pins {inst_mrmac_1/i_mrmac_0_exdes_support_wrapper/mrmac_1_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_0/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_0/i_mrmac_0_exdes_support_wrapper/mrmac_0_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_0/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_0/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/BUFG_clkout1_inst/O]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_1/i_mrmac_0_exdes_support_wrapper/mrmac_1_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_0/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_1/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/BUFG_clkout1_inst/O]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_0/i_mrmac_0_exdes_support_wrapper/mrmac_0_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_1/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_0/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/BUFG_clkout1_inst/O]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_0/i_mrmac_0_exdes_support_wrapper/mrmac_0_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_1_2/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_0/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_1/i_mrmac_0_exdes_support_wrapper/mrmac_1_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_1/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_1/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins {inst_mrmac_1/i_mrmac_0_exdes_support_wrapper/mrmac_1_exdes_support_i/mrmac_0_gt_wrapper/mbufg_gt_1_2/U0/USE_MBUFG_GT.GEN_MBUFG_GT[0].MBUFG_GT_U/O1}]] -to [get_clocks -of_objects [get_pins inst_mrmac_1/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins qdma_ep_i/clk_wizard_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]] -to [get_clocks -of_objects [get_pins inst_mrmac_0/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]]
set_false_path -from [get_clocks -of_objects [get_pins qdma_ep_i/clk_wizard_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]] -to [get_clocks -of_objects [get_pins inst_mrmac_1/i_mrmac_0_axis_clk_wiz_0/inst/clock_primitive_inst/MMCME5_inst/CLKOUT0]]


set_property BITSTREAM.GENERAL.COMPRESS true [current_design]
set_property BITSTREAM.GENERAL.WRITE0FRAMES No [current_design]
set_property BITSTREAM.GENERAL.PROCESSALLVEAMS true [current_design]
