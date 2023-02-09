# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

proc numberOfCPUs {} {
    return 8

    # Windows puts it in an environment variable
    global tcl_platform env
    if {$tcl_platform(platform) eq "windows"} {
        return $env(NUMBER_OF_PROCESSORS)
    }

    # Check for sysctl (OSX, BSD)
    set sysctl [auto_execok "sysctl"]
    if {[llength $sysctl]} {
        if {![catch {exec {*}$sysctl -n "hw.ncpu"} cores]} {
            return $cores
        }
    }

    # Assume Linux, which has /proc/cpuinfo, but be careful
    if {![catch {open "/proc/cpuinfo"} f]} {
        set cores [regexp -all -line {^processor\s} [read $f]]
        close $f
        if {$cores > 0} {
            return $cores
        }
    }

    # No idea what the actual number of cores is; exhausted all our options
    # Fall back to returning 1; there must be at least that because we're running on it!
    return 1
}

################################################################
# This is a generated script based on design: qdma_ep
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################

namespace eval _tcl {
proc get_script_folder {} {
   set script_path [file normalize [info script]]
   set script_folder [file dirname $script_path]
   return $script_folder
}
}
variable script_folder
set script_folder [_tcl::get_script_folder]

################################################################
# Check if script is running in correct Vivado version.
################################################################
set scripts_vivado_version 2022.1
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

set_param board.repoPaths /wrk/xcohdnobkup1/jmelber/vck5000/boards/vck5000/production_silicon/1.1

################################################################
# START
################################################################

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xcvc1902-vsvd1760-2MP-e-S
   set_property platform.extensible "true" [current_project]
   set_property BOARD_PART xilinx.com:vck5000:part0:1.0 [current_project]
}


# CHANGE DESIGN NAME HERE
variable design_name
set design_name project_1

# If you do not already have an existing IP Integrator design open,
# you can create a design using the following command:
#    create_bd_design $design_name

# Creating design if needed
set errMsg ""
set nRet 0

set cur_design [current_bd_design -quiet]
set list_cells [get_bd_cells -quiet]

if { ${design_name} eq "" } {
   # USE CASES:
   #    1) Design_name not set

   set errMsg "Please set the variable <design_name> to a non-empty value."
   set nRet 1

} elseif { ${cur_design} ne "" && ${list_cells} eq "" } {
   # USE CASES:
   #    2): Current design opened AND is empty AND names same.
   #    3): Current design opened AND is empty AND names diff; design_name NOT in project.
   #    4): Current design opened AND is empty AND names diff; design_name exists in project.

   if { $cur_design ne $design_name } {
      common::send_gid_msg -ssname BD::TCL -id 2001 -severity "INFO" "Changing value of <design_name> from <$design_name> to <$cur_design> since current design is empty."
      set design_name [get_property NAME $cur_design]
   }
   common::send_gid_msg -ssname BD::TCL -id 2002 -severity "INFO" "Constructing design in IPI design <$cur_design>..."

} elseif { ${cur_design} ne "" && $list_cells ne "" && $cur_design eq $design_name } {
   # USE CASES:
   #    5) Current design opened AND has components AND same names.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 1
} elseif { [get_files -quiet ${design_name}.bd] ne "" } {
   # USE CASES: 
   #    6) Current opened design, has components, but diff names, design_name exists in project.
   #    7) No opened design, design_name exists in project.

   set errMsg "Design <$design_name> already exists in your project, please set the variable <design_name> to another value."
   set nRet 2

} else {
   # USE CASES:
   #    8) No opened design, design_name not in project.
   #    9) Current opened design, has components, but diff names, design_name not in project.

   common::send_gid_msg -ssname BD::TCL -id 2003 -severity "INFO" "Currently there is no design <$design_name> in project, so creating one..."

   create_bd_design $design_name

   common::send_gid_msg -ssname BD::TCL -id 2004 -severity "INFO" "Making design <$design_name> as current_bd_design."
   current_bd_design $design_name

}

common::send_gid_msg -ssname BD::TCL -id 2005 -severity "INFO" "Currently the variable <design_name> is equal to \"$design_name\"."

if { $nRet != 0 } {
   catch {common::send_gid_msg -ssname BD::TCL -id 2006 -severity "ERROR" $errMsg}
   return $nRet
}

set bCheckIPsPassed 1
##################################################################
# CHECK IPs
##################################################################
set bCheckIPs 1
if { $bCheckIPs == 1 } {
   set list_check_ips "\ 
xilinx.com:ip:ai_engine:2.0\
xilinx.com:ip:axi_bram_ctrl:4.1\
xilinx.com:ip:axi_cdma:4.1\
xilinx.com:ip:axi_dbg_hub:2.0\
xilinx.com:ip:axi_noc:1.0\
xilinx.com:ip:axi_traffic_gen:3.0\
xilinx.com:ip:axis_ila:1.1\
xilinx.com:ip:axis_vio:1.0\
xilinx.com:ip:versal_cips:3.2\
xilinx.com:ip:clk_wizard:1.0\
xilinx.com:ip:emb_mem_gen:1.0\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:qdma:4.0\
xilinx.com:ip:smartconnect:1.0\
xilinx.com:ip:xlconstant:1.1\
xilinx.com:ip:util_ds_buf:2.2\
xilinx.com:ip:gt_quad_base:1.1\
xilinx.com:ip:pcie_versal:1.0\
xilinx.com:ip:pcie_phy_versal:1.0\
"

   set list_ips_missing ""
   common::send_gid_msg -ssname BD::TCL -id 2011 -severity "INFO" "Checking if the following IPs exist in the project's IP catalog: $list_check_ips ."

   foreach ip_vlnv $list_check_ips {
      set ip_obj [get_ipdefs -all $ip_vlnv]
      if { $ip_obj eq "" } {
         lappend list_ips_missing $ip_vlnv
      }
   }

   if { $list_ips_missing ne "" } {
      catch {common::send_gid_msg -ssname BD::TCL -id 2012 -severity "ERROR" "The following IPs are not found in the IP Catalog:\n  $list_ips_missing\n\nResolution: Please add the repository containing the IP(s) to the project." }
      set bCheckIPsPassed 0
   }

}

if { $bCheckIPsPassed != 1 } {
  common::send_gid_msg -ssname BD::TCL -id 2023 -severity "WARNING" "Will not continue with creation of design due to the error(s) above."
  return 3
}

##################################################################
# DESIGN PROCs
##################################################################


# Hierarchical cell: qdma_host_mem_support
proc create_hier_cell_qdma_host_mem_support { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_hier_cell_qdma_host_mem_support() - Empty argument(s)!"}
     return
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create cell and set as current instance
  set hier_obj [create_bd_cell -type hier $nameHier]
  current_bd_instance $hier_obj

  # Create interface pins
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_cq

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 m_axis_rc

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:pcie4_cfg_control_rtl:1.0 pcie_cfg_control

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:pcie4_cfg_msix_rtl:1.0 pcie_cfg_external_msix_without_msi

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie_cfg_fc_rtl:1.1 pcie_cfg_fc

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:pcie3_cfg_interrupt_rtl:1.0 pcie_cfg_interrupt

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie3_cfg_msg_received_rtl:1.0 pcie_cfg_mesg_rcvd

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie3_cfg_mesg_tx_rtl:1.0 pcie_cfg_mesg_tx

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:pcie4_cfg_mgmt_rtl:1.0 pcie_cfg_mgmt

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie4_cfg_status_rtl:1.0 pcie_cfg_status

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:axis_rtl:1.0 pcie_dbg_m_axis

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 pcie_dbg_s_axis

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:gt_rtl:1.0 pcie_mgt

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie3_transmit_fc_rtl:1.0 pcie_transmit_fc

  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:pcie_ext_pipe_rtl:1.0 pipe_ep

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_cc

  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:axis_rtl:1.0 s_axis_rq


  # Create pins
  create_bd_pin -dir I pcie_dbg_aclk
  create_bd_pin -dir I pcie_dbg_aresetn
  create_bd_pin -dir O phy_rdy_out
  create_bd_pin -dir I -type rst sys_reset
  create_bd_pin -dir O -type clk user_clk
  create_bd_pin -dir O user_lnk_up
  create_bd_pin -dir O -type rst user_reset

  # Create instance: bufg_gt_sysclk, and set properties
  set bufg_gt_sysclk [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 bufg_gt_sysclk ]
  set_property -dict [ list \
   CONFIG.C_BUFG_GT_SYNC {true} \
   CONFIG.C_BUF_TYPE {BUFG_GT} \
 ] $bufg_gt_sysclk

  # Create instance: const_1b1, and set properties
  set const_1b1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_1b1 ]
  set_property -dict [ list \
   CONFIG.CONST_VAL {1} \
   CONFIG.CONST_WIDTH {1} \
 ] $const_1b1

  # Create instance: gt_quad_0, and set properties
  set gt_quad_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:gt_quad_base:1.1 gt_quad_0 ]
  set_property -dict [ list \
   CONFIG.PORTS_INFO_DICT {\
     LANE_SEL_DICT {PROT0 {RX0 RX1 RX2 RX3 TX0 TX1 TX2 TX3}}\
     GT_TYPE {GTY}\
     REG_CONF_INTF {APB3_INTF}\
     BOARD_PARAMETER {}\
   } \
   CONFIG.PROT_OUTCLK_VALUES {\
CH0_RXOUTCLK 250 CH0_TXOUTCLK 500 CH1_RXOUTCLK 250 CH1_TXOUTCLK 500\
CH2_RXOUTCLK 250 CH2_TXOUTCLK 500 CH3_RXOUTCLK 250 CH3_TXOUTCLK 500} \
   CONFIG.REFCLK_STRING {\
HSCLK0_LCPLLGTREFCLK0 refclk_PROT0_R0_100_MHz_unique1 HSCLK0_RPLLGTREFCLK0\
refclk_PROT0_R0_100_MHz_unique1 HSCLK1_LCPLLGTREFCLK0\
refclk_PROT0_R0_100_MHz_unique1 HSCLK1_RPLLGTREFCLK0\
refclk_PROT0_R0_100_MHz_unique1} \
 ] $gt_quad_0

  # Create instance: pcie, and set properties
  set pcie [ create_bd_cell -type ip -vlnv xilinx.com:ip:pcie_versal:1.0 pcie ]
  set_property -dict [ list \
   CONFIG.AXISTEN_IF_CQ_ALIGNMENT_MODE {Address_Aligned} \
   CONFIG.AXISTEN_IF_RC_STRADDLE {true} \
   CONFIG.AXISTEN_IF_RQ_ALIGNMENT_MODE {DWORD_Aligned} \
   CONFIG.MSI_X_OPTIONS {MSI-X_External} \
   CONFIG.PF0_DEVICE_ID {B034} \
   CONFIG.PF0_INTERRUPT_PIN {INTA} \
   CONFIG.PF0_LINK_STATUS_SLOT_CLOCK_CONFIG {true} \
   CONFIG.PF0_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF0_MSIX_CAP_PBA_OFFSET {34000} \
   CONFIG.PF0_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF0_MSIX_CAP_TABLE_OFFSET {30000} \
   CONFIG.PF0_MSIX_CAP_TABLE_SIZE {007} \
   CONFIG.PF0_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF0_REVISION_ID {00} \
   CONFIG.PF0_SRIOV_VF_DEVICE_ID {C034} \
   CONFIG.PF0_SUBSYSTEM_ID {0007} \
   CONFIG.PF0_SUBSYSTEM_VENDOR_ID {10EE} \
   CONFIG.PF0_Use_Class_Code_Lookup_Assistant {false} \
   CONFIG.PF1_DEVICE_ID {B134} \
   CONFIG.PF1_INTERRUPT_PIN {INTA} \
   CONFIG.PF1_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF1_MSIX_CAP_PBA_OFFSET {34000} \
   CONFIG.PF1_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF1_MSIX_CAP_TABLE_OFFSET {30000} \
   CONFIG.PF1_MSIX_CAP_TABLE_SIZE {007} \
   CONFIG.PF1_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF1_REVISION_ID {00} \
   CONFIG.PF1_SRIOV_VF_DEVICE_ID {C134} \
   CONFIG.PF1_SUBSYSTEM_ID {0007} \
   CONFIG.PF1_SUBSYSTEM_VENDOR_ID {10EE} \
   CONFIG.PF1_Use_Class_Code_Lookup_Assistant {false} \
   CONFIG.PF2_DEVICE_ID {0007} \
   CONFIG.PF2_INTERRUPT_PIN {NONE} \
   CONFIG.PF2_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF2_MSIX_CAP_PBA_OFFSET {00000000} \
   CONFIG.PF2_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF2_MSIX_CAP_TABLE_OFFSET {00000000} \
   CONFIG.PF2_MSIX_CAP_TABLE_SIZE {000} \
   CONFIG.PF2_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF2_REVISION_ID {00} \
   CONFIG.PF2_SRIOV_VF_DEVICE_ID {C234} \
   CONFIG.PF2_SUBSYSTEM_ID {0007} \
   CONFIG.PF2_SUBSYSTEM_VENDOR_ID {10EE} \
   CONFIG.PF2_Use_Class_Code_Lookup_Assistant {false} \
   CONFIG.PF3_DEVICE_ID {B334} \
   CONFIG.PF3_INTERRUPT_PIN {NONE} \
   CONFIG.PF3_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF3_MSIX_CAP_PBA_OFFSET {00000000} \
   CONFIG.PF3_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF3_MSIX_CAP_TABLE_OFFSET {00000000} \
   CONFIG.PF3_MSIX_CAP_TABLE_SIZE {000} \
   CONFIG.PF3_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF3_REVISION_ID {00} \
   CONFIG.PF3_SRIOV_VF_DEVICE_ID {C334} \
   CONFIG.PF3_SUBSYSTEM_ID {0007} \
   CONFIG.PF3_SUBSYSTEM_VENDOR_ID {10EE} \
   CONFIG.PF3_Use_Class_Code_Lookup_Assistant {false} \
   CONFIG.PF4_DEVICE_ID {B434} \
   CONFIG.PF4_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF4_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF4_SRIOV_VF_DEVICE_ID {C434} \
   CONFIG.PF5_DEVICE_ID {B534} \
   CONFIG.PF5_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF5_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF5_SRIOV_VF_DEVICE_ID {C534} \
   CONFIG.PF6_DEVICE_ID {B634} \
   CONFIG.PF6_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF6_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF6_SRIOV_VF_DEVICE_ID {C634} \
   CONFIG.PF7_DEVICE_ID {B734} \
   CONFIG.PF7_MSIX_CAP_PBA_BIR {BAR_1:0} \
   CONFIG.PF7_MSIX_CAP_TABLE_BIR {BAR_1:0} \
   CONFIG.PF7_SRIOV_VF_DEVICE_ID {C734} \
   CONFIG.PL_DISABLE_LANE_REVERSAL {FALSE} \
   CONFIG.PL_LINK_CAP_MAX_LINK_SPEED {8.0_GT/s} \
   CONFIG.PL_LINK_CAP_MAX_LINK_WIDTH {X4} \
   CONFIG.REF_CLK_FREQ {100_MHz} \
   CONFIG.TL_PF_ENABLE_REG {2} \
   CONFIG.acs_ext_cap_enable {false} \
   CONFIG.axisten_freq {125} \
   CONFIG.axisten_if_enable_client_tag {true} \
   CONFIG.axisten_if_enable_msg_route_override {TRUE} \
   CONFIG.axisten_if_width {256_bit} \
   CONFIG.cfg_ext_if {false} \
   CONFIG.cfg_mgmt_if {true} \
   CONFIG.copy_pf0 {true} \
   CONFIG.coreclk_freq {500} \
   CONFIG.dedicate_perst {false} \
   CONFIG.device_port_type {PCI_Express_Endpoint_device} \
   CONFIG.en_dbg_descramble {false} \
   CONFIG.en_ext_clk {FALSE} \
   CONFIG.en_l23_entry {false} \
   CONFIG.en_parity {false} \
   CONFIG.en_transceiver_status_ports {false} \
   CONFIG.enable_auto_rxeq {False} \
   CONFIG.enable_ccix {FALSE} \
   CONFIG.enable_code {0000} \
   CONFIG.enable_dvsec {FALSE} \
   CONFIG.enable_gen4 {true} \
   CONFIG.enable_ibert {false} \
   CONFIG.enable_jtag_dbg {false} \
   CONFIG.enable_more_clk {false} \
   CONFIG.ext_pcie_cfg_space_enabled {false} \
   CONFIG.ext_xvc_vsec_enable {false} \
   CONFIG.extended_tag_field {true} \
   CONFIG.insert_cips {false} \
   CONFIG.lane_order {Bottom} \
   CONFIG.legacy_ext_pcie_cfg_space_enabled {false} \
   CONFIG.mcap_enablement {None} \
   CONFIG.mode_selection {Advanced} \
   CONFIG.pcie_blk_locn {X0Y1} \
   CONFIG.pcie_link_debug {true} \
   CONFIG.pcie_link_debug_axi4_st {true} \
   CONFIG.pf0_ari_enabled {false} \
   CONFIG.pf0_bar0_64bit {true} \
   CONFIG.pf0_bar0_enabled {true} \
   CONFIG.pf0_bar0_prefetchable {false} \
   CONFIG.pf0_bar0_scale {Megabytes} \
   CONFIG.pf0_bar0_size {64} \
   CONFIG.pf0_bar1_64bit {false} \
   CONFIG.pf0_bar1_enabled {false} \
   CONFIG.pf0_bar1_prefetchable {false} \
   CONFIG.pf0_bar1_scale {Kilobytes} \
   CONFIG.pf0_bar1_size {128} \
   CONFIG.pf0_bar2_64bit {true} \
   CONFIG.pf0_bar2_enabled {true} \
   CONFIG.pf0_bar2_prefetchable {false} \
   CONFIG.pf0_bar2_scale {Megabytes} \
   CONFIG.pf0_bar2_size {512} \
   CONFIG.pf0_bar2_type {Memory} \
   CONFIG.pf0_bar3_64bit {false} \
   CONFIG.pf0_bar3_enabled {false} \
   CONFIG.pf0_bar3_prefetchable {false} \
   CONFIG.pf0_bar3_scale {Kilobytes} \
   CONFIG.pf0_bar3_size {128} \
   CONFIG.pf0_bar4_64bit {true} \
   CONFIG.pf0_bar4_enabled {true} \
   CONFIG.pf0_bar4_prefetchable {false} \
   CONFIG.pf0_bar4_scale {Megabytes} \
   CONFIG.pf0_bar4_size {1} \
   CONFIG.pf0_bar4_type {Memory} \
   CONFIG.pf0_bar5_enabled {false} \
   CONFIG.pf0_bar5_prefetchable {false} \
   CONFIG.pf0_bar5_scale {Kilobytes} \
   CONFIG.pf0_bar5_size {128} \
   CONFIG.pf0_base_class_menu {Memory_controller} \
   CONFIG.pf0_class_code_base {05} \
   CONFIG.pf0_class_code_interface {00} \
   CONFIG.pf0_class_code_sub {80} \
   CONFIG.pf0_dll_feature_cap_enabled {false} \
   CONFIG.pf0_expansion_rom_enabled {false} \
   CONFIG.pf0_margining_cap_enabled {false} \
   CONFIG.pf0_msi_enabled {false} \
   CONFIG.pf0_msix_enabled {true} \
   CONFIG.pf0_pl16_cap_enabled {false} \
   CONFIG.pf0_sub_class_interface_menu {Other_memory_controller} \
   CONFIG.pf1_bar0_64bit {true} \
   CONFIG.pf1_bar0_enabled {true} \
   CONFIG.pf1_bar0_prefetchable {false} \
   CONFIG.pf1_bar0_scale {Megabytes} \
   CONFIG.pf1_bar0_size {64} \
   CONFIG.pf1_bar1_64bit {false} \
   CONFIG.pf1_bar1_enabled {false} \
   CONFIG.pf1_bar1_prefetchable {false} \
   CONFIG.pf1_bar1_scale {Kilobytes} \
   CONFIG.pf1_bar1_size {128} \
   CONFIG.pf1_bar2_64bit {true} \
   CONFIG.pf1_bar2_enabled {true} \
   CONFIG.pf1_bar2_prefetchable {false} \
   CONFIG.pf1_bar2_scale {Megabytes} \
   CONFIG.pf1_bar2_size {512} \
   CONFIG.pf1_bar2_type {Memory} \
   CONFIG.pf1_bar3_64bit {false} \
   CONFIG.pf1_bar3_enabled {false} \
   CONFIG.pf1_bar3_prefetchable {false} \
   CONFIG.pf1_bar3_scale {Kilobytes} \
   CONFIG.pf1_bar3_size {128} \
   CONFIG.pf1_bar4_64bit {true} \
   CONFIG.pf1_bar4_enabled {true} \
   CONFIG.pf1_bar4_prefetchable {false} \
   CONFIG.pf1_bar4_scale {Megabytes} \
   CONFIG.pf1_bar4_size {1} \
   CONFIG.pf1_bar4_type {Memory} \
   CONFIG.pf1_bar5_enabled {false} \
   CONFIG.pf1_bar5_prefetchable {false} \
   CONFIG.pf1_bar5_scale {Kilobytes} \
   CONFIG.pf1_bar5_size {128} \
   CONFIG.pf1_base_class_menu {Memory_controller} \
   CONFIG.pf1_class_code_base {05} \
   CONFIG.pf1_class_code_interface {00} \
   CONFIG.pf1_class_code_sub {80} \
   CONFIG.pf1_expansion_rom_enabled {false} \
   CONFIG.pf1_msi_enabled {false} \
   CONFIG.pf1_msix_enabled {true} \
   CONFIG.pf1_sub_class_interface_menu {Other_memory_controller} \
   CONFIG.pf1_vendor_id {10EE} \
   CONFIG.pf2_bar0_64bit {true} \
   CONFIG.pf2_bar0_enabled {true} \
   CONFIG.pf2_bar0_prefetchable {false} \
   CONFIG.pf2_bar0_scale {Megabytes} \
   CONFIG.pf2_bar0_size {64} \
   CONFIG.pf2_bar1_64bit {false} \
   CONFIG.pf2_bar1_enabled {false} \
   CONFIG.pf2_bar1_prefetchable {false} \
   CONFIG.pf2_bar1_scale {Kilobytes} \
   CONFIG.pf2_bar1_size {128} \
   CONFIG.pf2_bar2_64bit {true} \
   CONFIG.pf2_bar2_enabled {true} \
   CONFIG.pf2_bar2_prefetchable {false} \
   CONFIG.pf2_bar2_scale {Megabytes} \
   CONFIG.pf2_bar2_size {512} \
   CONFIG.pf2_bar2_type {Memory} \
   CONFIG.pf2_bar3_64bit {false} \
   CONFIG.pf2_bar3_enabled {false} \
   CONFIG.pf2_bar3_prefetchable {false} \
   CONFIG.pf2_bar3_scale {Kilobytes} \
   CONFIG.pf2_bar3_size {128} \
   CONFIG.pf2_bar4_64bit {true} \
   CONFIG.pf2_bar4_enabled {true} \
   CONFIG.pf2_bar4_prefetchable {false} \
   CONFIG.pf2_bar4_scale {Megabytes} \
   CONFIG.pf2_bar4_size {1} \
   CONFIG.pf2_bar4_type {Memory} \
   CONFIG.pf2_bar5_enabled {false} \
   CONFIG.pf2_bar5_prefetchable {false} \
   CONFIG.pf2_bar5_scale {Kilobytes} \
   CONFIG.pf2_bar5_size {128} \
   CONFIG.pf2_base_class_menu {Memory_controller} \
   CONFIG.pf2_class_code_base {05} \
   CONFIG.pf2_class_code_interface {00} \
   CONFIG.pf2_class_code_sub {80} \
   CONFIG.pf2_expansion_rom_enabled {false} \
   CONFIG.pf2_msi_enabled {false} \
   CONFIG.pf2_msix_enabled {true} \
   CONFIG.pf2_sub_class_interface_menu {Other_memory_controller} \
   CONFIG.pf2_vendor_id {10EE} \
   CONFIG.pf3_bar0_64bit {true} \
   CONFIG.pf3_bar0_enabled {true} \
   CONFIG.pf3_bar0_prefetchable {false} \
   CONFIG.pf3_bar0_scale {Megabytes} \
   CONFIG.pf3_bar0_size {64} \
   CONFIG.pf3_bar1_64bit {false} \
   CONFIG.pf3_bar1_enabled {false} \
   CONFIG.pf3_bar1_prefetchable {false} \
   CONFIG.pf3_bar1_scale {Kilobytes} \
   CONFIG.pf3_bar1_size {128} \
   CONFIG.pf3_bar2_64bit {true} \
   CONFIG.pf3_bar2_enabled {true} \
   CONFIG.pf3_bar2_prefetchable {false} \
   CONFIG.pf3_bar2_scale {Megabytes} \
   CONFIG.pf3_bar2_size {512} \
   CONFIG.pf3_bar2_type {Memory} \
   CONFIG.pf3_bar3_64bit {false} \
   CONFIG.pf3_bar3_enabled {false} \
   CONFIG.pf3_bar3_prefetchable {false} \
   CONFIG.pf3_bar3_scale {Kilobytes} \
   CONFIG.pf3_bar3_size {128} \
   CONFIG.pf3_bar4_64bit {true} \
   CONFIG.pf3_bar4_enabled {true} \
   CONFIG.pf3_bar4_prefetchable {false} \
   CONFIG.pf3_bar4_scale {Megabytes} \
   CONFIG.pf3_bar4_size {1} \
   CONFIG.pf3_bar4_type {Memory} \
   CONFIG.pf3_bar5_enabled {false} \
   CONFIG.pf3_bar5_prefetchable {false} \
   CONFIG.pf3_bar5_scale {Kilobytes} \
   CONFIG.pf3_bar5_size {128} \
   CONFIG.pf3_base_class_menu {Memory_controller} \
   CONFIG.pf3_class_code_base {05} \
   CONFIG.pf3_class_code_interface {00} \
   CONFIG.pf3_class_code_sub {80} \
   CONFIG.pf3_expansion_rom_enabled {false} \
   CONFIG.pf3_msi_enabled {false} \
   CONFIG.pf3_msix_enabled {true} \
   CONFIG.pf3_sub_class_interface_menu {Other_memory_controller} \
   CONFIG.pf3_vendor_id {10EE} \
   CONFIG.pf4_bar0_64bit {true} \
   CONFIG.pf4_bar0_scale {Megabytes} \
   CONFIG.pf4_bar0_size {64} \
   CONFIG.pf4_bar2_64bit {true} \
   CONFIG.pf4_bar2_enabled {true} \
   CONFIG.pf4_bar2_scale {Megabytes} \
   CONFIG.pf4_bar2_size {512} \
   CONFIG.pf4_bar2_type {Memory} \
   CONFIG.pf4_bar4_64bit {true} \
   CONFIG.pf4_bar4_enabled {true} \
   CONFIG.pf4_bar4_scale {Megabytes} \
   CONFIG.pf4_bar4_size {1} \
   CONFIG.pf4_bar4_type {Memory} \
   CONFIG.pf5_bar0_64bit {true} \
   CONFIG.pf5_bar0_scale {Megabytes} \
   CONFIG.pf5_bar0_size {64} \
   CONFIG.pf5_bar2_64bit {true} \
   CONFIG.pf5_bar2_enabled {true} \
   CONFIG.pf5_bar2_scale {Megabytes} \
   CONFIG.pf5_bar2_size {512} \
   CONFIG.pf5_bar2_type {Memory} \
   CONFIG.pf5_bar4_64bit {true} \
   CONFIG.pf5_bar4_enabled {true} \
   CONFIG.pf5_bar4_scale {Megabytes} \
   CONFIG.pf5_bar4_size {1} \
   CONFIG.pf5_bar4_type {Memory} \
   CONFIG.pf6_bar0_64bit {true} \
   CONFIG.pf6_bar0_scale {Megabytes} \
   CONFIG.pf6_bar0_size {64} \
   CONFIG.pf6_bar2_64bit {true} \
   CONFIG.pf6_bar2_enabled {true} \
   CONFIG.pf6_bar2_scale {Megabytes} \
   CONFIG.pf6_bar2_size {512} \
   CONFIG.pf6_bar2_type {Memory} \
   CONFIG.pf6_bar4_64bit {true} \
   CONFIG.pf6_bar4_enabled {true} \
   CONFIG.pf6_bar4_scale {Megabytes} \
   CONFIG.pf6_bar4_size {1} \
   CONFIG.pf6_bar4_type {Memory} \
   CONFIG.pf7_bar0_64bit {true} \
   CONFIG.pf7_bar0_scale {Megabytes} \
   CONFIG.pf7_bar0_size {64} \
   CONFIG.pf7_bar2_64bit {true} \
   CONFIG.pf7_bar2_enabled {true} \
   CONFIG.pf7_bar2_scale {Megabytes} \
   CONFIG.pf7_bar2_size {512} \
   CONFIG.pf7_bar2_type {Memory} \
   CONFIG.pf7_bar4_64bit {true} \
   CONFIG.pf7_bar4_enabled {true} \
   CONFIG.pf7_bar4_scale {Megabytes} \
   CONFIG.pf7_bar4_size {1} \
   CONFIG.pf7_bar4_type {Memory} \
   CONFIG.pipe_sim {true} \
   CONFIG.sys_reset_polarity {ACTIVE_LOW} \
   CONFIG.userclk2_freq {500} \
   CONFIG.vendor_id {10EE} \
 ] $pcie

  # Create instance: pcie_phy, and set properties
  set pcie_phy [ create_bd_cell -type ip -vlnv xilinx.com:ip:pcie_phy_versal:1.0 pcie_phy ]
  set_property -dict [ list \
   CONFIG.PL_LINK_CAP_MAX_LINK_SPEED {8.0_GT/s} \
   CONFIG.PL_LINK_CAP_MAX_LINK_WIDTH {X4} \
   CONFIG.aspm {No_ASPM} \
   CONFIG.async_mode {SRNS} \
   CONFIG.disable_double_pipe {YES} \
   CONFIG.en_gt_pclk {false} \
   CONFIG.ins_loss_profile {Add-in_Card} \
   CONFIG.lane_order {Bottom} \
   CONFIG.lane_reversal {false} \
   CONFIG.phy_async_en {true} \
   CONFIG.phy_coreclk_freq {500_MHz} \
   CONFIG.phy_refclk_freq {100_MHz} \
   CONFIG.phy_userclk2_freq {125_MHz} \
   CONFIG.phy_userclk_freq {125_MHz} \
   CONFIG.pipeline_stages {1} \
   CONFIG.sim_model {NO} \
   CONFIG.tx_preset {4} \
 ] $pcie_phy

  # Create instance: refclk_ibuf, and set properties
  set refclk_ibuf [ create_bd_cell -type ip -vlnv xilinx.com:ip:util_ds_buf:2.2 refclk_ibuf ]
  set_property -dict [ list \
   CONFIG.C_BUF_TYPE {IBUFDSGTE} \
 ] $refclk_ibuf

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins pcie_refclk] [get_bd_intf_pins refclk_ibuf/CLK_IN_D]
  connect_bd_intf_net -intf_net Conn2 [get_bd_intf_pins pcie_mgt] [get_bd_intf_pins pcie_phy/pcie_mgt]
  connect_bd_intf_net -intf_net Conn3 [get_bd_intf_pins m_axis_cq] [get_bd_intf_pins pcie/m_axis_cq]
  connect_bd_intf_net -intf_net Conn4 [get_bd_intf_pins m_axis_rc] [get_bd_intf_pins pcie/m_axis_rc]
  connect_bd_intf_net -intf_net Conn5 [get_bd_intf_pins pcie_cfg_fc] [get_bd_intf_pins pcie/pcie_cfg_fc]
  connect_bd_intf_net -intf_net Conn6 [get_bd_intf_pins pcie_cfg_interrupt] [get_bd_intf_pins pcie/pcie_cfg_interrupt]
  connect_bd_intf_net -intf_net Conn7 [get_bd_intf_pins pcie_cfg_mesg_rcvd] [get_bd_intf_pins pcie/pcie_cfg_mesg_rcvd]
  connect_bd_intf_net -intf_net Conn8 [get_bd_intf_pins pcie_cfg_mesg_tx] [get_bd_intf_pins pcie/pcie_cfg_mesg_tx]
  connect_bd_intf_net -intf_net Conn9 [get_bd_intf_pins s_axis_cc] [get_bd_intf_pins pcie/s_axis_cc]
  connect_bd_intf_net -intf_net Conn10 [get_bd_intf_pins s_axis_rq] [get_bd_intf_pins pcie/s_axis_rq]
  connect_bd_intf_net -intf_net Conn11 [get_bd_intf_pins pcie_cfg_control] [get_bd_intf_pins pcie/pcie_cfg_control]
  connect_bd_intf_net -intf_net Conn12 [get_bd_intf_pins pcie_cfg_external_msix_without_msi] [get_bd_intf_pins pcie/pcie_cfg_external_msix_without_msi]
  connect_bd_intf_net -intf_net Conn13 [get_bd_intf_pins pcie_cfg_mgmt] [get_bd_intf_pins pcie/pcie_cfg_mgmt]
  connect_bd_intf_net -intf_net Conn14 [get_bd_intf_pins pcie_cfg_status] [get_bd_intf_pins pcie/pcie_cfg_status]
  connect_bd_intf_net -intf_net Conn15 [get_bd_intf_pins pcie_transmit_fc] [get_bd_intf_pins pcie/pcie_transmit_fc]
  connect_bd_intf_net -intf_net Conn16 [get_bd_intf_pins pipe_ep] [get_bd_intf_pins pcie/pcie_ext_pipe_ep]
  connect_bd_intf_net -intf_net gt_quad_0_GT0_BUFGT [get_bd_intf_pins gt_quad_0/GT0_BUFGT] [get_bd_intf_pins pcie_phy/GT_BUFGT]
  connect_bd_intf_net -intf_net gt_quad_0_GT_Serial [get_bd_intf_pins gt_quad_0/GT_Serial] [get_bd_intf_pins pcie_phy/GT0_Serial]
  connect_bd_intf_net -intf_net pcie_dbg_s_axis_1 [get_bd_intf_pins pcie_dbg_s_axis] [get_bd_intf_pins pcie/pcie_dbg_s_axis]
  connect_bd_intf_net -intf_net pcie_pcie_dbg_m_axis [get_bd_intf_pins pcie_dbg_m_axis] [get_bd_intf_pins pcie/pcie_dbg_m_axis]
  connect_bd_intf_net -intf_net pcie_phy_GT_RX0 [get_bd_intf_pins gt_quad_0/RX0_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_RX0]
  connect_bd_intf_net -intf_net pcie_phy_GT_RX1 [get_bd_intf_pins gt_quad_0/RX1_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_RX1]
  connect_bd_intf_net -intf_net pcie_phy_GT_RX2 [get_bd_intf_pins gt_quad_0/RX2_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_RX2]
  connect_bd_intf_net -intf_net pcie_phy_GT_RX3 [get_bd_intf_pins gt_quad_0/RX3_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_RX3]
  connect_bd_intf_net -intf_net pcie_phy_GT_TX0 [get_bd_intf_pins gt_quad_0/TX0_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_TX0]
  connect_bd_intf_net -intf_net pcie_phy_GT_TX1 [get_bd_intf_pins gt_quad_0/TX1_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_TX1]
  connect_bd_intf_net -intf_net pcie_phy_GT_TX2 [get_bd_intf_pins gt_quad_0/TX2_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_TX2]
  connect_bd_intf_net -intf_net pcie_phy_GT_TX3 [get_bd_intf_pins gt_quad_0/TX3_GT_IP_Interface] [get_bd_intf_pins pcie_phy/GT_TX3]
  connect_bd_intf_net -intf_net pcie_phy_gt_rxmargin_q0 [get_bd_intf_pins gt_quad_0/gt_rxmargin_intf] [get_bd_intf_pins pcie_phy/gt_rxmargin_q0]
  connect_bd_intf_net -intf_net pcie_phy_mac_rx [get_bd_intf_pins pcie/phy_mac_rx] [get_bd_intf_pins pcie_phy/phy_mac_rx]
  connect_bd_intf_net -intf_net pcie_phy_mac_tx [get_bd_intf_pins pcie/phy_mac_tx] [get_bd_intf_pins pcie_phy/phy_mac_tx]
  connect_bd_intf_net -intf_net pcie_phy_phy_mac_command [get_bd_intf_pins pcie/phy_mac_command] [get_bd_intf_pins pcie_phy/phy_mac_command]
  connect_bd_intf_net -intf_net pcie_phy_phy_mac_rx_margining [get_bd_intf_pins pcie/phy_mac_rx_margining] [get_bd_intf_pins pcie_phy/phy_mac_rx_margining]
  connect_bd_intf_net -intf_net pcie_phy_phy_mac_status [get_bd_intf_pins pcie/phy_mac_status] [get_bd_intf_pins pcie_phy/phy_mac_status]
  connect_bd_intf_net -intf_net pcie_phy_phy_mac_tx_drive [get_bd_intf_pins pcie/phy_mac_tx_drive] [get_bd_intf_pins pcie_phy/phy_mac_tx_drive]
  connect_bd_intf_net -intf_net pcie_phy_phy_mac_tx_eq [get_bd_intf_pins pcie/phy_mac_tx_eq] [get_bd_intf_pins pcie_phy/phy_mac_tx_eq]

  # Create port connections
  connect_bd_net -net bufg_gt_sysclk_BUFG_GT_O [get_bd_pins bufg_gt_sysclk/BUFG_GT_O] [get_bd_pins gt_quad_0/apb3clk] [get_bd_pins pcie/sys_clk] [get_bd_pins pcie_phy/phy_refclk]
  connect_bd_net -net const_1b1_dout [get_bd_pins bufg_gt_sysclk/BUFG_GT_CE] [get_bd_pins const_1b1/dout]
  connect_bd_net -net gt_quad_0_ch0_phyready [get_bd_pins gt_quad_0/ch0_phyready] [get_bd_pins pcie_phy/ch0_phyready]
  connect_bd_net -net gt_quad_0_ch0_phystatus [get_bd_pins gt_quad_0/ch0_phystatus] [get_bd_pins pcie_phy/ch0_phystatus]
  connect_bd_net -net gt_quad_0_ch0_rxoutclk [get_bd_pins gt_quad_0/ch0_rxoutclk] [get_bd_pins pcie_phy/gt_rxoutclk]
  connect_bd_net -net gt_quad_0_ch0_txoutclk [get_bd_pins gt_quad_0/ch0_txoutclk] [get_bd_pins pcie_phy/gt_txoutclk]
  connect_bd_net -net gt_quad_0_ch1_phyready [get_bd_pins gt_quad_0/ch1_phyready] [get_bd_pins pcie_phy/ch1_phyready]
  connect_bd_net -net gt_quad_0_ch1_phystatus [get_bd_pins gt_quad_0/ch1_phystatus] [get_bd_pins pcie_phy/ch1_phystatus]
  connect_bd_net -net gt_quad_0_ch2_phyready [get_bd_pins gt_quad_0/ch2_phyready] [get_bd_pins pcie_phy/ch2_phyready]
  connect_bd_net -net gt_quad_0_ch2_phystatus [get_bd_pins gt_quad_0/ch2_phystatus] [get_bd_pins pcie_phy/ch2_phystatus]
  connect_bd_net -net gt_quad_0_ch3_phyready [get_bd_pins gt_quad_0/ch3_phyready] [get_bd_pins pcie_phy/ch3_phyready]
  connect_bd_net -net gt_quad_0_ch3_phystatus [get_bd_pins gt_quad_0/ch3_phystatus] [get_bd_pins pcie_phy/ch3_phystatus]
  connect_bd_net -net pcie_dbg_aclk_1 [get_bd_pins pcie_dbg_aclk] [get_bd_pins pcie/pcie_dbg_aclk]
  connect_bd_net -net pcie_dbg_aresetn_1 [get_bd_pins pcie_dbg_aresetn] [get_bd_pins pcie/pcie_dbg_aresetn]
  connect_bd_net -net pcie_pcie_ltssm_state [get_bd_pins pcie/pcie_ltssm_state] [get_bd_pins pcie_phy/pcie_ltssm_state]
  connect_bd_net -net pcie_phy_gt_pcieltssm [get_bd_pins gt_quad_0/pcieltssm] [get_bd_pins pcie_phy/gt_pcieltssm]
  connect_bd_net -net pcie_phy_gtrefclk [get_bd_pins gt_quad_0/GT_REFCLK0] [get_bd_pins pcie_phy/gtrefclk]
  connect_bd_net -net pcie_phy_pcierstb [get_bd_pins gt_quad_0/ch0_pcierstb] [get_bd_pins gt_quad_0/ch1_pcierstb] [get_bd_pins gt_quad_0/ch2_pcierstb] [get_bd_pins gt_quad_0/ch3_pcierstb] [get_bd_pins pcie_phy/pcierstb]
  connect_bd_net -net pcie_phy_phy_coreclk [get_bd_pins pcie/phy_coreclk] [get_bd_pins pcie_phy/phy_coreclk]
  connect_bd_net -net pcie_phy_phy_mcapclk [get_bd_pins pcie/phy_mcapclk] [get_bd_pins pcie_phy/phy_mcapclk]
  connect_bd_net -net pcie_phy_phy_pclk [get_bd_pins gt_quad_0/ch0_rxusrclk] [get_bd_pins gt_quad_0/ch0_txusrclk] [get_bd_pins gt_quad_0/ch1_rxusrclk] [get_bd_pins gt_quad_0/ch1_txusrclk] [get_bd_pins gt_quad_0/ch2_rxusrclk] [get_bd_pins gt_quad_0/ch2_txusrclk] [get_bd_pins gt_quad_0/ch3_rxusrclk] [get_bd_pins gt_quad_0/ch3_txusrclk] [get_bd_pins pcie/phy_pclk] [get_bd_pins pcie_phy/phy_pclk]
  connect_bd_net -net pcie_phy_phy_userclk [get_bd_pins pcie/phy_userclk] [get_bd_pins pcie_phy/phy_userclk]
  connect_bd_net -net pcie_phy_phy_userclk2 [get_bd_pins pcie/phy_userclk2] [get_bd_pins pcie_phy/phy_userclk2]
  connect_bd_net -net pcie_phy_rdy_out [get_bd_pins phy_rdy_out] [get_bd_pins pcie/phy_rdy_out]
  connect_bd_net -net pcie_user_clk [get_bd_pins user_clk] [get_bd_pins pcie/user_clk]
  connect_bd_net -net pcie_user_lnk_up [get_bd_pins user_lnk_up] [get_bd_pins pcie/user_lnk_up]
  connect_bd_net -net pcie_user_reset [get_bd_pins user_reset] [get_bd_pins pcie/user_reset]
  connect_bd_net -net refclk_ibuf_IBUF_DS_ODIV2 [get_bd_pins bufg_gt_sysclk/BUFG_GT_I] [get_bd_pins refclk_ibuf/IBUF_DS_ODIV2]
  connect_bd_net -net refclk_ibuf_IBUF_OUT [get_bd_pins pcie/sys_clk_gt] [get_bd_pins pcie_phy/phy_gtrefclk] [get_bd_pins refclk_ibuf/IBUF_OUT]
  connect_bd_net -net sys_reset_1 [get_bd_pins sys_reset] [get_bd_pins pcie/sys_reset] [get_bd_pins pcie_phy/phy_rst_n]

  # Restore current instance
  current_bd_instance $oldCurInst
}


# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
     set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
     return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
     return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj


  # Create interface ports
  set ddr4_c0_sysclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_c0_sysclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $ddr4_c0_sysclk

  set ddr4_c1_sysclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_c1_sysclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $ddr4_c1_sysclk

  set ddr4_c2_sysclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_c2_sysclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $ddr4_c2_sysclk

  set ddr4_c3_sysclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_c3_sysclk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $ddr4_c3_sysclk

  set ddr4_sdram_c0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_sdram_c0 ]

  set ddr4_sdram_c1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_sdram_c1 ]

  set ddr4_sdram_c2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_sdram_c2 ]

  set ddr4_sdram_c3 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_sdram_c3 ]

  set pcie_mgt [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:gt_rtl:1.0 pcie_mgt ]

  set pcie_refclk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 pcie_refclk ]

  set pipe_ep [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:pcie_ext_pipe_rtl:1.0 pipe_ep ]


  # Create ports

  # Create instance: ai_engine_0, and set properties
  set ai_engine_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:ai_engine:2.0 ai_engine_0 ]

  set_property -dict [ list \
   CONFIG.CATEGORY {NOC} \
 ] [get_bd_intf_pins /ai_engine_0/S00_AXI]

  # Create instance: axi_bram_ctrl_0, and set properties
  set axi_bram_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0 ]
  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
 ] $axi_bram_ctrl_0

  # Create instance: axi_bram_ctrl_1, and set properties
  set axi_bram_ctrl_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_1 ]
  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
 ] $axi_bram_ctrl_1

  # Create instance: axi_cdma_0, and set properties
  set axi_cdma_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_cdma:4.1 axi_cdma_0 ]
  set_property -dict [ list \
   CONFIG.C_ADDR_WIDTH {64} \
   CONFIG.C_M_AXI_MAX_BURST_LEN {256} \
 ] $axi_cdma_0

  # Create instance: axi_dbg_hub_0, and set properties
  set axi_dbg_hub_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dbg_hub:2.0 axi_dbg_hub_0 ]
  set_property -dict [ list \
   CONFIG.C_NUM_DEBUG_CORES {1} \
 ] $axi_dbg_hub_0

  # Create instance: axi_noc_0, and set properties
  set axi_noc_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.0 axi_noc_0 ]
  set_property -dict [ list \
   CONFIG.CH0_DDR4_0_BOARD_INTERFACE {ddr4_sdram_c0} \
   CONFIG.CH0_DDR4_1_BOARD_INTERFACE {ddr4_sdram_c1} \
   CONFIG.CH0_DDR4_2_BOARD_INTERFACE {ddr4_sdram_c2} \
   CONFIG.CH0_DDR4_3_BOARD_INTERFACE {ddr4_sdram_c3} \
   CONFIG.LOGO_FILE {data/noc_mc.png} \
   CONFIG.MC_ADDR_BIT9 {CA3} \
   CONFIG.MC_CHAN_REGION1 {DDR_LOW1} \
   CONFIG.NUM_CLKS {9} \
   CONFIG.NUM_MC {4} \
   CONFIG.NUM_MCP {1} \
   CONFIG.NUM_MI {2} \
   CONFIG.NUM_NSI {1} \
   CONFIG.NUM_SI {10} \
   CONFIG.sys_clk0_BOARD_INTERFACE {ddr4_c0_sysclk} \
   CONFIG.sys_clk1_BOARD_INTERFACE {ddr4_c1_sysclk} \
   CONFIG.sys_clk2_BOARD_INTERFACE {ddr4_c2_sysclk} \
   CONFIG.sys_clk3_BOARD_INTERFACE {ddr4_c3_sysclk} \
 ] $axi_noc_0

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CATEGORY {aie} \
 ] [get_bd_intf_pins /axi_noc_0/M00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.APERTURES {{0x201_0000_0000 1G}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /axi_noc_0/M01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /axi_noc_0/S00_AXI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {load} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
 ] [get_bd_intf_pins /axi_noc_0/S00_INI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /axi_noc_0/S01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /axi_noc_0/S02_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /axi_noc_0/S03_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /axi_noc_0/S04_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /axi_noc_0/S05_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /axi_noc_0/S06_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /axi_noc_0/S07_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /axi_noc_0/S08_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {256} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
   CONFIG.DEST_IDS {M01_AXI:0x100:M00_AXI:0x140} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /axi_noc_0/S09_AXI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M01_AXI:S03_AXI:S04_AXI:S09_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk0]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M00_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk1]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk2]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S01_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk3]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S02_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk4]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S05_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk5]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S06_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk6]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S07_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk7]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S08_AXI} \
 ] [get_bd_pins /axi_noc_0/aclk8]

  # Create instance: axi_noc_1, and set properties
  set axi_noc_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.0 axi_noc_1 ]
  set_property -dict [ list \
   CONFIG.NUM_CLKS {0} \
   CONFIG.NUM_MI {0} \
   CONFIG.NUM_NMI {1} \
   CONFIG.NUM_SI {0} \
 ] $axi_noc_1

  # Create instance: axi_traffic_gen_0_read, and set properties
  set axi_traffic_gen_0_read [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_0_read ]
  set_property -dict [ list \
   CONFIG.ATG_OPTIONS {High Level Traffic} \
   CONFIG.C_ATG_REPEAT_TYPE {One_Shot} \
   CONFIG.C_ATG_STATIC_HLTP_INCR {true} \
   CONFIG.DATA_SIZE_AVG {1} \
   CONFIG.DATA_SIZE_MAX {256} \
   CONFIG.DATA_TRAFFIC_PATTERN {Fixed} \
   CONFIG.DATA_TRANS_TYPE {Read_Only} \
   CONFIG.DATA_WRITE_SHARE {0} \
   CONFIG.MASTER_AXI_WIDTH {64} \
   CONFIG.MASTER_HIGH_ADDRESS {0x00001FFF} \
   CONFIG.TRAFFIC_PROFILE {Data} \
 ] $axi_traffic_gen_0_read

  # Create instance: axi_traffic_gen_1_write, and set properties
  set axi_traffic_gen_1_write [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_traffic_gen:3.0 axi_traffic_gen_1_write ]
  set_property -dict [ list \
   CONFIG.ATG_OPTIONS {High Level Traffic} \
   CONFIG.C_ATG_STATIC_HLTP_INCR {true} \
   CONFIG.DATA_READ_SHARE {0} \
   CONFIG.DATA_SIZE_AVG {1} \
   CONFIG.DATA_SIZE_MAX {256} \
   CONFIG.DATA_TRAFFIC_PATTERN {Fixed} \
   CONFIG.DATA_TRANS_TYPE {Write_Only} \
   CONFIG.DATA_WRITE_SHARE {50} \
   CONFIG.MASTER_AXI_WIDTH {64} \
   CONFIG.MASTER_BASE_ADDRESS {0x00000000} \
   CONFIG.MASTER_HIGH_ADDRESS {0x00001FFF} \
   CONFIG.TRAFFIC_PROFILE {Data} \
 ] $axi_traffic_gen_1_write

  # Create instance: axis_ila_0, and set properties
  set axis_ila_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_ila:1.1 axis_ila_0 ]
  set_property -dict [ list \
   CONFIG.C_BRAM_CNT {0} \
   CONFIG.C_MON_TYPE {Interface_Monitor} \
   CONFIG.C_NUM_MONITOR_SLOTS {2} \
   CONFIG.C_SLOT {0} \
   CONFIG.C_SLOT_0_APC_EN {1} \
 ] $axis_ila_0

  # Create instance: axis_vio_0, and set properties
  set axis_vio_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axis_vio:1.0 axis_vio_0 ]
  set_property -dict [ list \
   CONFIG.C_EN_PROBE_IN_ACTIVITY {0} \
   CONFIG.C_NUM_PROBE_IN {0} \
   CONFIG.C_NUM_PROBE_OUT {2} \
 ] $axis_vio_0

  # Create instance: cips_0, and set properties
  set cips_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.2 cips_0 ]
  set_property -dict [ list \
   CONFIG.BOOT_MODE {Custom} \
   CONFIG.CLOCK_MODE {Custom} \
   CONFIG.DDR_MEMORY_MODE {Custom} \
   CONFIG.DESIGN_MODE {1} \
   CONFIG.IO_CONFIG_MODE {Custom} \
   CONFIG.PS_BOARD_INTERFACE {Custom} \
   CONFIG.PS_PL_CONNECTIVITY_MODE {Custom} \
   CONFIG.PS_PMC_CONFIG {\
     AURORA_LINE_RATE_GPBS {12.5}\
     BOOT_MODE {Custom}\
     BOOT_SECONDARY_PCIE_ENABLE {0}\
     CLOCK_MODE {Custom}\
     COHERENCY_MODE {Custom}\
     CPM_PCIE0_TANDEM {None}\
     DDR_MEMORY_MODE {Custom}\
     DEBUG_MODE {Custom}\
     DESIGN_MODE {1}\
     DEVICE_INTEGRITY_MODE {Custom}\
     DIS_AUTO_POL_CHECK {0}\
     GT_REFCLK_MHZ {156.25}\
     INIT_CLK_MHZ {125}\
     INV_POLARITY {0}\
     IO_CONFIG_MODE {Custom}\
     JTAG_USERCODE {0x0}\
     OT_EAM_RESP {SRST}\
     PCIE_APERTURES_DUAL_ENABLE {0}\
     PCIE_APERTURES_SINGLE_ENABLE {0}\
     PERFORMANCE_MODE {Custom}\
     PL_SEM_GPIO_ENABLE {0}\
     PMC_ALT_REF_CLK_FREQMHZ {33.333}\
     PMC_BANK_0_IO_STANDARD {LVCMOS1.8}\
     PMC_BANK_1_IO_STANDARD {LVCMOS1.8}\
     PMC_CIPS_MODE {ADVANCE}\
     PMC_CORE_SUBSYSTEM_LOAD {10}\
     PMC_CRP_CFU_REF_CTRL_ACT_FREQMHZ {295.833038}\
     PMC_CRP_CFU_REF_CTRL_DIVISOR0 {4}\
     PMC_CRP_CFU_REF_CTRL_FREQMHZ {300}\
     PMC_CRP_CFU_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_DFT_OSC_REF_CTRL_ACT_FREQMHZ {400}\
     PMC_CRP_DFT_OSC_REF_CTRL_DIVISOR0 {3}\
     PMC_CRP_DFT_OSC_REF_CTRL_FREQMHZ {400}\
     PMC_CRP_DFT_OSC_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_EFUSE_REF_CTRL_ACT_FREQMHZ {100.000000}\
     PMC_CRP_EFUSE_REF_CTRL_FREQMHZ {100.000000}\
     PMC_CRP_EFUSE_REF_CTRL_SRCSEL {IRO_CLK/4}\
     PMC_CRP_HSM0_REF_CTRL_ACT_FREQMHZ {32.870338}\
     PMC_CRP_HSM0_REF_CTRL_DIVISOR0 {36}\
     PMC_CRP_HSM0_REF_CTRL_FREQMHZ {33.333}\
     PMC_CRP_HSM0_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_HSM1_REF_CTRL_ACT_FREQMHZ {131.481354}\
     PMC_CRP_HSM1_REF_CTRL_DIVISOR0 {9}\
     PMC_CRP_HSM1_REF_CTRL_FREQMHZ {133.333}\
     PMC_CRP_HSM1_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_I2C_REF_CTRL_ACT_FREQMHZ {100}\
     PMC_CRP_I2C_REF_CTRL_DIVISOR0 {12}\
     PMC_CRP_I2C_REF_CTRL_FREQMHZ {100}\
     PMC_CRP_I2C_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_LSBUS_REF_CTRL_ACT_FREQMHZ {147.916519}\
     PMC_CRP_LSBUS_REF_CTRL_DIVISOR0 {8}\
     PMC_CRP_LSBUS_REF_CTRL_FREQMHZ {150}\
     PMC_CRP_LSBUS_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_NOC_REF_CTRL_ACT_FREQMHZ {999.999023}\
     PMC_CRP_NOC_REF_CTRL_FREQMHZ {1000}\
     PMC_CRP_NOC_REF_CTRL_SRCSEL {NPLL}\
     PMC_CRP_NPI_REF_CTRL_ACT_FREQMHZ {295.833038}\
     PMC_CRP_NPI_REF_CTRL_DIVISOR0 {4}\
     PMC_CRP_NPI_REF_CTRL_FREQMHZ {300}\
     PMC_CRP_NPI_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_NPLL_CTRL_CLKOUTDIV {4}\
     PMC_CRP_NPLL_CTRL_FBDIV {120}\
     PMC_CRP_NPLL_CTRL_SRCSEL {REF_CLK}\
     PMC_CRP_NPLL_TO_XPD_CTRL_DIVISOR0 {2}\
     PMC_CRP_OSPI_REF_CTRL_ACT_FREQMHZ {200}\
     PMC_CRP_OSPI_REF_CTRL_DIVISOR0 {4}\
     PMC_CRP_OSPI_REF_CTRL_FREQMHZ {200}\
     PMC_CRP_OSPI_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_PL0_REF_CTRL_ACT_FREQMHZ {99.999901}\
     PMC_CRP_PL0_REF_CTRL_DIVISOR0 {10}\
     PMC_CRP_PL0_REF_CTRL_FREQMHZ {100}\
     PMC_CRP_PL0_REF_CTRL_SRCSEL {NPLL}\
     PMC_CRP_PL1_REF_CTRL_ACT_FREQMHZ {100}\
     PMC_CRP_PL1_REF_CTRL_DIVISOR0 {3}\
     PMC_CRP_PL1_REF_CTRL_FREQMHZ {334}\
     PMC_CRP_PL1_REF_CTRL_SRCSEL {NPLL}\
     PMC_CRP_PL2_REF_CTRL_ACT_FREQMHZ {100}\
     PMC_CRP_PL2_REF_CTRL_DIVISOR0 {3}\
     PMC_CRP_PL2_REF_CTRL_FREQMHZ {334}\
     PMC_CRP_PL2_REF_CTRL_SRCSEL {NPLL}\
     PMC_CRP_PL3_REF_CTRL_ACT_FREQMHZ {100}\
     PMC_CRP_PL3_REF_CTRL_DIVISOR0 {3}\
     PMC_CRP_PL3_REF_CTRL_FREQMHZ {334}\
     PMC_CRP_PL3_REF_CTRL_SRCSEL {NPLL}\
     PMC_CRP_PL5_REF_CTRL_FREQMHZ {400}\
     PMC_CRP_PPLL_CTRL_CLKOUTDIV {2}\
     PMC_CRP_PPLL_CTRL_FBDIV {71}\
     PMC_CRP_PPLL_CTRL_SRCSEL {REF_CLK}\
     PMC_CRP_PPLL_TO_XPD_CTRL_DIVISOR0 {1}\
     PMC_CRP_QSPI_REF_CTRL_ACT_FREQMHZ {300}\
     PMC_CRP_QSPI_REF_CTRL_DIVISOR0 {4}\
     PMC_CRP_QSPI_REF_CTRL_FREQMHZ {300}\
     PMC_CRP_QSPI_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_SDIO0_REF_CTRL_ACT_FREQMHZ {200}\
     PMC_CRP_SDIO0_REF_CTRL_DIVISOR0 {6}\
     PMC_CRP_SDIO0_REF_CTRL_FREQMHZ {200}\
     PMC_CRP_SDIO0_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_SDIO1_REF_CTRL_ACT_FREQMHZ {200}\
     PMC_CRP_SDIO1_REF_CTRL_DIVISOR0 {6}\
     PMC_CRP_SDIO1_REF_CTRL_FREQMHZ {200}\
     PMC_CRP_SDIO1_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_SD_DLL_REF_CTRL_ACT_FREQMHZ {1200}\
     PMC_CRP_SD_DLL_REF_CTRL_DIVISOR0 {1}\
     PMC_CRP_SD_DLL_REF_CTRL_FREQMHZ {1200}\
     PMC_CRP_SD_DLL_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_SWITCH_TIMEOUT_CTRL_ACT_FREQMHZ {1.000000}\
     PMC_CRP_SWITCH_TIMEOUT_CTRL_DIVISOR0 {100}\
     PMC_CRP_SWITCH_TIMEOUT_CTRL_FREQMHZ {1}\
     PMC_CRP_SWITCH_TIMEOUT_CTRL_SRCSEL {IRO_CLK/4}\
     PMC_CRP_SYSMON_REF_CTRL_ACT_FREQMHZ {295.833038}\
     PMC_CRP_SYSMON_REF_CTRL_FREQMHZ {295.833038}\
     PMC_CRP_SYSMON_REF_CTRL_SRCSEL {NPI_REF_CLK}\
     PMC_CRP_TEST_PATTERN_REF_CTRL_ACT_FREQMHZ {200}\
     PMC_CRP_TEST_PATTERN_REF_CTRL_DIVISOR0 {6}\
     PMC_CRP_TEST_PATTERN_REF_CTRL_FREQMHZ {200}\
     PMC_CRP_TEST_PATTERN_REF_CTRL_SRCSEL {PPLL}\
     PMC_CRP_USB_SUSPEND_CTRL_ACT_FREQMHZ {0.200000}\
     PMC_CRP_USB_SUSPEND_CTRL_DIVISOR0 {500}\
     PMC_CRP_USB_SUSPEND_CTRL_FREQMHZ {0.2}\
     PMC_CRP_USB_SUSPEND_CTRL_SRCSEL {IRO_CLK/4}\
     PMC_EXTERNAL_TAMPER {{ENABLE 0} {IO NONE}}\
     PMC_EXTERNAL_TAMPER_1 {{ENABLE 0} {IO None}}\
     PMC_EXTERNAL_TAMPER_2 {{ENABLE 0} {IO None}}\
     PMC_EXTERNAL_TAMPER_3 {{ENABLE 0} {IO None}}\
     PMC_GPIO0_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 0 .. 25}}}\
     PMC_GPIO1_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 26 .. 51}}}\
     PMC_GPIO_EMIO_PERIPHERAL_ENABLE {0}\
     PMC_GPIO_EMIO_WIDTH {64}\
     PMC_GPIO_EMIO_WIDTH_HDL {64}\
     PMC_GPI_ENABLE {0}\
     PMC_GPI_WIDTH {32}\
     PMC_GPO_ENABLE {0}\
     PMC_GPO_WIDTH {32}\
     PMC_HSM0_CLK_ENABLE {1}\
     PMC_HSM1_CLK_ENABLE {1}\
     PMC_I2CPMC_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 2 .. 3}}}\
     PMC_MIO0 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO1 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO10 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO11 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO12 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO13 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO14 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO15 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO16 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO17 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 1} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO18 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO19 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO2 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO20 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO21 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO22 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO23 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO24 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO25 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO26 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO27 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO28 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO29 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO3 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO30 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO31 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO32 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO33 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO34 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO35 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO36 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO37 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA high}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}}\
     PMC_MIO38 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA\
default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO39 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA\
default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO4 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO40 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO41 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO42 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO43 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO44 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO45 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO46 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO47 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO48 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Unassigned}}\
     PMC_MIO49 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Unassigned}}\
     PMC_MIO5 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO50 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Unassigned}}\
     PMC_MIO51 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO6 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 12mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 1} {SLEW fast} {USAGE Reserved}}\
     PMC_MIO7 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO8 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO9 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PMC_MIO_EN_FOR_PL_PCIE {1}\
     PMC_MIO_TREE_PERIPHERALS {################UART 0#UART 0####################GPIO\
1#PCIE#PCIE######################################}\
     PMC_MIO_TREE_SIGNALS {################rxd#txd####################gpio_1_pin[37]#reset1_n#reset2_n######################################}\
     PMC_NOC_PMC_ADDR_WIDTH {64}\
     PMC_NOC_PMC_DATA_WIDTH {128}\
     PMC_OSPI_COHERENCY {0}\
     PMC_OSPI_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 0 .. 11}} {MODE Single}}\
     PMC_OSPI_ROUTE_THROUGH_FPD {0}\
     PMC_PL_ALT_REF_CLK_FREQMHZ {33.333}\
     PMC_PMC_NOC_ADDR_WIDTH {64}\
     PMC_PMC_NOC_DATA_WIDTH {128}\
     PMC_QSPI_COHERENCY {0}\
     PMC_QSPI_FBCLK {{ENABLE 1} {IO {PMC_MIO 6}}}\
     PMC_QSPI_PERIPHERAL_DATA_MODE {x1}\
     PMC_QSPI_PERIPHERAL_ENABLE {0}\
     PMC_QSPI_PERIPHERAL_MODE {Single}\
     PMC_QSPI_ROUTE_THROUGH_FPD {0}\
     PMC_REF_CLK_FREQMHZ {33.3333}\
     PMC_SD0 {{CD_ENABLE 0} {CD_IO {PMC_MIO 24}} {POW_ENABLE 0} {POW_IO {PMC_MIO 17}}\
{RESET_ENABLE 0} {RESET_IO {PMC_MIO 17}} {WP_ENABLE 0} {WP_IO {PMC_MIO\
25}}}\
     PMC_SD0_COHERENCY {0}\
     PMC_SD0_DATA_TRANSFER_MODE {4Bit}\
     PMC_SD0_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x00} {CLK_200_SDR_OTAP_DLY 0x00}\
{CLK_50_DDR_ITAP_DLY 0x00} {CLK_50_DDR_OTAP_DLY 0x00}\
{CLK_50_SDR_ITAP_DLY 0x00} {CLK_50_SDR_OTAP_DLY 0x00} {ENABLE\
0} {IO {PMC_MIO 13 .. 25}}}\
     PMC_SD0_ROUTE_THROUGH_FPD {0}\
     PMC_SD0_SLOT_TYPE {SD 2.0}\
     PMC_SD0_SPEED_MODE {default speed}\
     PMC_SD1 {{CD_ENABLE 0} {CD_IO {PMC_MIO 2}} {POW_ENABLE 0} {POW_IO {PMC_MIO 12}}\
{RESET_ENABLE 0} {RESET_IO {PMC_MIO 12}} {WP_ENABLE 0} {WP_IO {PMC_MIO\
1}}}\
     PMC_SD1_COHERENCY {0}\
     PMC_SD1_DATA_TRANSFER_MODE {4Bit}\
     PMC_SD1_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x00} {CLK_200_SDR_OTAP_DLY 0x00}\
{CLK_50_DDR_ITAP_DLY 0x00} {CLK_50_DDR_OTAP_DLY 0x00}\
{CLK_50_SDR_ITAP_DLY 0x00} {CLK_50_SDR_OTAP_DLY 0x00} {ENABLE\
0} {IO {PMC_MIO 0 .. 11}}}\
     PMC_SD1_ROUTE_THROUGH_FPD {0}\
     PMC_SD1_SLOT_TYPE {SD 2.0}\
     PMC_SD1_SPEED_MODE {default speed}\
     PMC_SHOW_CCI_SMMU_SETTINGS {0}\
     PMC_SMAP_PERIPHERAL {{ENABLE 0} {IO {32 Bit}}}\
     PMC_TAMPER_EXTMIO_ENABLE {0}\
     PMC_TAMPER_EXTMIO_ERASE_BBRAM {0}\
     PMC_TAMPER_EXTMIO_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_GLITCHDETECT_ENABLE {0}\
     PMC_TAMPER_GLITCHDETECT_ENABLE_1 {0}\
     PMC_TAMPER_GLITCHDETECT_ENABLE_2 {0}\
     PMC_TAMPER_GLITCHDETECT_ENABLE_3 {0}\
     PMC_TAMPER_GLITCHDETECT_ERASE_BBRAM {0}\
     PMC_TAMPER_GLITCHDETECT_ERASE_BBRAM_1 {0}\
     PMC_TAMPER_GLITCHDETECT_ERASE_BBRAM_2 {0}\
     PMC_TAMPER_GLITCHDETECT_ERASE_BBRAM_3 {0}\
     PMC_TAMPER_GLITCHDETECT_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_GLITCHDETECT_RESPONSE_1 {SYS INTERRUPT}\
     PMC_TAMPER_GLITCHDETECT_RESPONSE_2 {SYS INTERRUPT}\
     PMC_TAMPER_GLITCHDETECT_RESPONSE_3 {SYS INTERRUPT}\
     PMC_TAMPER_JTAGDETECT_ENABLE {0}\
     PMC_TAMPER_JTAGDETECT_ENABLE_1 {0}\
     PMC_TAMPER_JTAGDETECT_ENABLE_2 {0}\
     PMC_TAMPER_JTAGDETECT_ENABLE_3 {0}\
     PMC_TAMPER_JTAGDETECT_ERASE_BBRAM {0}\
     PMC_TAMPER_JTAGDETECT_ERASE_BBRAM_1 {0}\
     PMC_TAMPER_JTAGDETECT_ERASE_BBRAM_2 {0}\
     PMC_TAMPER_JTAGDETECT_ERASE_BBRAM_3 {0}\
     PMC_TAMPER_JTAGDETECT_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_JTAGDETECT_RESPONSE_1 {SYS INTERRUPT}\
     PMC_TAMPER_JTAGDETECT_RESPONSE_2 {SYS INTERRUPT}\
     PMC_TAMPER_JTAGDETECT_RESPONSE_3 {SYS INTERRUPT}\
     PMC_TAMPER_SUP_0_31_ENABLE {0}\
     PMC_TAMPER_SUP_0_31_ERASE_BBRAM {0}\
     PMC_TAMPER_SUP_0_31_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_SUP_128_151_ENABLE {0}\
     PMC_TAMPER_SUP_128_151_ERASE_BBRAM {0}\
     PMC_TAMPER_SUP_128_151_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_SUP_32_63_ENABLE {0}\
     PMC_TAMPER_SUP_32_63_ERASE_BBRAM {0}\
     PMC_TAMPER_SUP_32_63_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_SUP_64_95_ENABLE {0}\
     PMC_TAMPER_SUP_64_95_ERASE_BBRAM {0}\
     PMC_TAMPER_SUP_64_95_ERASE_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_SUP_64_95_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_SUP_96_127_ENABLE {0}\
     PMC_TAMPER_SUP_96_127_ERASE_BBRAM {0}\
     PMC_TAMPER_SUP_96_127_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_TEMPERATURE_ENABLE {0}\
     PMC_TAMPER_TEMPERATURE_ENABLE_1 {0}\
     PMC_TAMPER_TEMPERATURE_ENABLE_2 {0}\
     PMC_TAMPER_TEMPERATURE_ENABLE_3 {0}\
     PMC_TAMPER_TEMPERATURE_ERASE_BBRAM {0}\
     PMC_TAMPER_TEMPERATURE_ERASE_BBRAM_1 {0}\
     PMC_TAMPER_TEMPERATURE_ERASE_BBRAM_2 {0}\
     PMC_TAMPER_TEMPERATURE_ERASE_BBRAM_3 {0}\
     PMC_TAMPER_TEMPERATURE_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_TEMPERATURE_RESPONSE_1 {SYS INTERRUPT}\
     PMC_TAMPER_TEMPERATURE_RESPONSE_2 {SYS INTERRUPT}\
     PMC_TAMPER_TEMPERATURE_RESPONSE_3 {SYS INTERRUPT}\
     PMC_TAMPER_TRIGGER_ERASE_BBRAM {0}\
     PMC_TAMPER_TRIGGER_ERASE_BBRAM_1 {0}\
     PMC_TAMPER_TRIGGER_ERASE_BBRAM_2 {0}\
     PMC_TAMPER_TRIGGER_ERASE_BBRAM_3 {0}\
     PMC_TAMPER_TRIGGER_REGISTER {0}\
     PMC_TAMPER_TRIGGER_REGISTER_1 {0}\
     PMC_TAMPER_TRIGGER_REGISTER_2 {0}\
     PMC_TAMPER_TRIGGER_REGISTER_3 {0}\
     PMC_TAMPER_TRIGGER_RESPONSE {SYS INTERRUPT}\
     PMC_TAMPER_TRIGGER_RESPONSE_1 {SYS INTERRUPT}\
     PMC_TAMPER_TRIGGER_RESPONSE_2 {SYS INTERRUPT}\
     PMC_TAMPER_TRIGGER_RESPONSE_3 {SYS INTERRUPT}\
     PMC_USE_CFU_SEU {0}\
     PMC_USE_NOC_PMC_AXI0 {0}\
     PMC_USE_NOC_PMC_AXI1 {0}\
     PMC_USE_NOC_PMC_AXI2 {0}\
     PMC_USE_NOC_PMC_AXI3 {0}\
     PMC_USE_PL_PMC_AUX_REF_CLK {0}\
     PMC_USE_PMC_NOC_AXI0 {1}\
     PMC_USE_PMC_NOC_AXI1 {0}\
     PMC_USE_PMC_NOC_AXI2 {0}\
     PMC_USE_PMC_NOC_AXI3 {0}\
     PMC_WDT_PERIOD {100}\
     PMC_WDT_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 0}}}\
     POWER_REPORTING_MODE {Custom}\
     PSPMC_MANUAL_CLK_ENABLE {0}\
     PS_A72_ACTIVE_BLOCKS {2}\
     PS_A72_LOAD {90}\
     PS_BANK_2_IO_STANDARD {LVCMOS1.8}\
     PS_BANK_3_IO_STANDARD {LVCMOS1.8}\
     PS_BOARD_INTERFACE {Custom}\
     PS_CAN0_CLK {{ENABLE 0} {IO {PMC_MIO 0}}}\
     PS_CAN0_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 8 .. 9}}}\
     PS_CAN1_CLK {{ENABLE 0} {IO {PMC_MIO 0}}}\
     PS_CAN1_PERIPHERAL {{ENABLE 0} {IO {PS_MIO 16 .. 17}}}\
     PS_CRF_ACPU_CTRL_ACT_FREQMHZ {1349.998657}\
     PS_CRF_ACPU_CTRL_DIVISOR0 {1}\
     PS_CRF_ACPU_CTRL_FREQMHZ {1350}\
     PS_CRF_ACPU_CTRL_SRCSEL {APLL}\
     PS_CRF_APLL_CTRL_CLKOUTDIV {2}\
     PS_CRF_APLL_CTRL_FBDIV {81}\
     PS_CRF_APLL_CTRL_SRCSEL {REF_CLK}\
     PS_CRF_APLL_TO_XPD_CTRL_DIVISOR0 {4}\
     PS_CRF_DBG_FPD_CTRL_ACT_FREQMHZ {394.444061}\
     PS_CRF_DBG_FPD_CTRL_DIVISOR0 {3}\
     PS_CRF_DBG_FPD_CTRL_FREQMHZ {400}\
     PS_CRF_DBG_FPD_CTRL_SRCSEL {PPLL}\
     PS_CRF_DBG_TRACE_CTRL_ACT_FREQMHZ {300}\
     PS_CRF_DBG_TRACE_CTRL_DIVISOR0 {3}\
     PS_CRF_DBG_TRACE_CTRL_FREQMHZ {300}\
     PS_CRF_DBG_TRACE_CTRL_SRCSEL {PPLL}\
     PS_CRF_FPD_LSBUS_CTRL_ACT_FREQMHZ {149.999847}\
     PS_CRF_FPD_LSBUS_CTRL_DIVISOR0 {9}\
     PS_CRF_FPD_LSBUS_CTRL_FREQMHZ {150}\
     PS_CRF_FPD_LSBUS_CTRL_SRCSEL {APLL}\
     PS_CRF_FPD_TOP_SWITCH_CTRL_ACT_FREQMHZ {824.999207}\
     PS_CRF_FPD_TOP_SWITCH_CTRL_DIVISOR0 {1}\
     PS_CRF_FPD_TOP_SWITCH_CTRL_FREQMHZ {825}\
     PS_CRF_FPD_TOP_SWITCH_CTRL_SRCSEL {RPLL}\
     PS_CRL_CAN0_REF_CTRL_ACT_FREQMHZ {100}\
     PS_CRL_CAN0_REF_CTRL_DIVISOR0 {12}\
     PS_CRL_CAN0_REF_CTRL_FREQMHZ {100}\
     PS_CRL_CAN0_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_CAN1_REF_CTRL_ACT_FREQMHZ {100}\
     PS_CRL_CAN1_REF_CTRL_DIVISOR0 {12}\
     PS_CRL_CAN1_REF_CTRL_FREQMHZ {100}\
     PS_CRL_CAN1_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_CPM_TOPSW_REF_CTRL_ACT_FREQMHZ {591.666077}\
     PS_CRL_CPM_TOPSW_REF_CTRL_DIVISOR0 {2}\
     PS_CRL_CPM_TOPSW_REF_CTRL_FREQMHZ {600}\
     PS_CRL_CPM_TOPSW_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_CPU_R5_CTRL_ACT_FREQMHZ {591.666077}\
     PS_CRL_CPU_R5_CTRL_DIVISOR0 {2}\
     PS_CRL_CPU_R5_CTRL_FREQMHZ {600}\
     PS_CRL_CPU_R5_CTRL_SRCSEL {PPLL}\
     PS_CRL_DBG_LPD_CTRL_ACT_FREQMHZ {394.444061}\
     PS_CRL_DBG_LPD_CTRL_DIVISOR0 {3}\
     PS_CRL_DBG_LPD_CTRL_FREQMHZ {400}\
     PS_CRL_DBG_LPD_CTRL_SRCSEL {PPLL}\
     PS_CRL_DBG_TSTMP_CTRL_ACT_FREQMHZ {394.444061}\
     PS_CRL_DBG_TSTMP_CTRL_DIVISOR0 {3}\
     PS_CRL_DBG_TSTMP_CTRL_FREQMHZ {400}\
     PS_CRL_DBG_TSTMP_CTRL_SRCSEL {PPLL}\
     PS_CRL_GEM0_REF_CTRL_ACT_FREQMHZ {125}\
     PS_CRL_GEM0_REF_CTRL_DIVISOR0 {4}\
     PS_CRL_GEM0_REF_CTRL_FREQMHZ {125}\
     PS_CRL_GEM0_REF_CTRL_SRCSEL {NPLL}\
     PS_CRL_GEM1_REF_CTRL_ACT_FREQMHZ {125}\
     PS_CRL_GEM1_REF_CTRL_DIVISOR0 {4}\
     PS_CRL_GEM1_REF_CTRL_FREQMHZ {125}\
     PS_CRL_GEM1_REF_CTRL_SRCSEL {NPLL}\
     PS_CRL_GEM_TSU_REF_CTRL_ACT_FREQMHZ {250}\
     PS_CRL_GEM_TSU_REF_CTRL_DIVISOR0 {2}\
     PS_CRL_GEM_TSU_REF_CTRL_FREQMHZ {250}\
     PS_CRL_GEM_TSU_REF_CTRL_SRCSEL {NPLL}\
     PS_CRL_I2C0_REF_CTRL_ACT_FREQMHZ {100}\
     PS_CRL_I2C0_REF_CTRL_DIVISOR0 {12}\
     PS_CRL_I2C0_REF_CTRL_FREQMHZ {100}\
     PS_CRL_I2C0_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_I2C1_REF_CTRL_ACT_FREQMHZ {100}\
     PS_CRL_I2C1_REF_CTRL_DIVISOR0 {12}\
     PS_CRL_I2C1_REF_CTRL_FREQMHZ {100}\
     PS_CRL_I2C1_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_IOU_SWITCH_CTRL_ACT_FREQMHZ {249.999756}\
     PS_CRL_IOU_SWITCH_CTRL_DIVISOR0 {2}\
     PS_CRL_IOU_SWITCH_CTRL_FREQMHZ {250}\
     PS_CRL_IOU_SWITCH_CTRL_SRCSEL {NPLL}\
     PS_CRL_LPD_LSBUS_CTRL_ACT_FREQMHZ {149.999863}\
     PS_CRL_LPD_LSBUS_CTRL_DIVISOR0 {11}\
     PS_CRL_LPD_LSBUS_CTRL_FREQMHZ {150}\
     PS_CRL_LPD_LSBUS_CTRL_SRCSEL {RPLL}\
     PS_CRL_LPD_TOP_SWITCH_CTRL_ACT_FREQMHZ {591.666077}\
     PS_CRL_LPD_TOP_SWITCH_CTRL_DIVISOR0 {2}\
     PS_CRL_LPD_TOP_SWITCH_CTRL_FREQMHZ {600}\
     PS_CRL_LPD_TOP_SWITCH_CTRL_SRCSEL {PPLL}\
     PS_CRL_PSM_REF_CTRL_ACT_FREQMHZ {394.444061}\
     PS_CRL_PSM_REF_CTRL_DIVISOR0 {3}\
     PS_CRL_PSM_REF_CTRL_FREQMHZ {400}\
     PS_CRL_PSM_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_RPLL_CTRL_CLKOUTDIV {2}\
     PS_CRL_RPLL_CTRL_FBDIV {99}\
     PS_CRL_RPLL_CTRL_SRCSEL {REF_CLK}\
     PS_CRL_RPLL_TO_XPD_CTRL_DIVISOR0 {2}\
     PS_CRL_SPI0_REF_CTRL_ACT_FREQMHZ {200}\
     PS_CRL_SPI0_REF_CTRL_DIVISOR0 {6}\
     PS_CRL_SPI0_REF_CTRL_FREQMHZ {200}\
     PS_CRL_SPI0_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_SPI1_REF_CTRL_ACT_FREQMHZ {200}\
     PS_CRL_SPI1_REF_CTRL_DIVISOR0 {6}\
     PS_CRL_SPI1_REF_CTRL_FREQMHZ {200}\
     PS_CRL_SPI1_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_TIMESTAMP_REF_CTRL_ACT_FREQMHZ {99.999901}\
     PS_CRL_TIMESTAMP_REF_CTRL_DIVISOR0 {5}\
     PS_CRL_TIMESTAMP_REF_CTRL_FREQMHZ {100}\
     PS_CRL_TIMESTAMP_REF_CTRL_SRCSEL {NPLL}\
     PS_CRL_UART0_REF_CTRL_ACT_FREQMHZ {99.999901}\
     PS_CRL_UART0_REF_CTRL_DIVISOR0 {5}\
     PS_CRL_UART0_REF_CTRL_FREQMHZ {100}\
     PS_CRL_UART0_REF_CTRL_SRCSEL {NPLL}\
     PS_CRL_UART1_REF_CTRL_ACT_FREQMHZ {100}\
     PS_CRL_UART1_REF_CTRL_DIVISOR0 {12}\
     PS_CRL_UART1_REF_CTRL_FREQMHZ {100}\
     PS_CRL_UART1_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_USB0_BUS_REF_CTRL_ACT_FREQMHZ {20}\
     PS_CRL_USB0_BUS_REF_CTRL_DIVISOR0 {60}\
     PS_CRL_USB0_BUS_REF_CTRL_FREQMHZ {20}\
     PS_CRL_USB0_BUS_REF_CTRL_SRCSEL {PPLL}\
     PS_CRL_USB3_DUAL_REF_CTRL_ACT_FREQMHZ {20}\
     PS_CRL_USB3_DUAL_REF_CTRL_DIVISOR0 {60}\
     PS_CRL_USB3_DUAL_REF_CTRL_FREQMHZ {10}\
     PS_CRL_USB3_DUAL_REF_CTRL_SRCSEL {PPLL}\
     PS_DDRC_ENABLE {1}\
     PS_DDR_RAM_HIGHADDR_OFFSET {0x800000000}\
     PS_DDR_RAM_LOWADDR_OFFSET {0x80000000}\
     PS_ENET0_MDIO {{ENABLE 0} {IO {PMC_MIO 50 .. 51}}}\
     PS_ENET0_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 26 .. 37}}}\
     PS_ENET1_MDIO {{ENABLE 0} {IO {PMC_MIO 50 .. 51}}}\
     PS_ENET1_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 38 .. 49}}}\
     PS_EN_AXI_STATUS_PORTS {0}\
     PS_EN_PORTS_CONTROLLER_BASED {0}\
     PS_EXPAND_CORESIGHT {0}\
     PS_EXPAND_FPD_SLAVES {0}\
     PS_EXPAND_GIC {0}\
     PS_EXPAND_LPD_SLAVES {0}\
     PS_FPD_INTERCONNECT_LOAD {90}\
     PS_FTM_CTI_IN0 {0}\
     PS_FTM_CTI_IN1 {0}\
     PS_FTM_CTI_IN2 {0}\
     PS_FTM_CTI_IN3 {0}\
     PS_FTM_CTI_OUT0 {0}\
     PS_FTM_CTI_OUT1 {0}\
     PS_FTM_CTI_OUT2 {0}\
     PS_FTM_CTI_OUT3 {0}\
     PS_GEM0_COHERENCY {0}\
     PS_GEM0_ROUTE_THROUGH_FPD {0}\
     PS_GEM1_COHERENCY {0}\
     PS_GEM1_ROUTE_THROUGH_FPD {0}\
     PS_GEM_TSU {{ENABLE 0} {IO {PS_MIO 24}}}\
     PS_GEM_TSU_CLK_PORT_PAIR {0}\
     PS_GEN_IPI0_ENABLE {1}\
     PS_GEN_IPI0_MASTER {A72}\
     PS_GEN_IPI1_ENABLE {1}\
     PS_GEN_IPI1_MASTER {A72}\
     PS_GEN_IPI2_ENABLE {1}\
     PS_GEN_IPI2_MASTER {A72}\
     PS_GEN_IPI3_ENABLE {1}\
     PS_GEN_IPI3_MASTER {A72}\
     PS_GEN_IPI4_ENABLE {1}\
     PS_GEN_IPI4_MASTER {A72}\
     PS_GEN_IPI5_ENABLE {1}\
     PS_GEN_IPI5_MASTER {A72}\
     PS_GEN_IPI6_ENABLE {1}\
     PS_GEN_IPI6_MASTER {A72}\
     PS_GEN_IPI_PMCNOBUF_ENABLE {1}\
     PS_GEN_IPI_PMCNOBUF_MASTER {PMC}\
     PS_GEN_IPI_PMC_ENABLE {1}\
     PS_GEN_IPI_PMC_MASTER {PMC}\
     PS_GEN_IPI_PSM_ENABLE {1}\
     PS_GEN_IPI_PSM_MASTER {PSM}\
     PS_GPIO2_MIO_PERIPHERAL {{ENABLE 0} {IO {PS_MIO 0 .. 25}}}\
     PS_GPIO_EMIO_PERIPHERAL_ENABLE {0}\
     PS_GPIO_EMIO_WIDTH {32}\
     PS_HSDP0_REFCLK {0}\
     PS_HSDP1_REFCLK {0}\
     PS_HSDP_EGRESS_TRAFFIC {JTAG}\
     PS_HSDP_INGRESS_TRAFFIC {JTAG}\
     PS_HSDP_MODE {NONE}\
     PS_HSDP_SAME_EGRESS_AS_INGRESS_TRAFFIC {1}\
     PS_I2C0_PERIPHERAL {{ENABLE 0} {IO {PS_MIO 2 .. 3}}}\
     PS_I2C1_PERIPHERAL {{ENABLE 0} {IO {PS_MIO 0 .. 1}}}\
     PS_I2CSYSMON_PERIPHERAL {{ENABLE 0} {IO {PS_MIO 23 .. 24}}}\
     PS_IRQ_USAGE {{CH0 0} {CH1 0} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0} {CH15\
0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 0} {CH9 0}}\
     PS_LPDMA0_COHERENCY {0}\
     PS_LPDMA0_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA1_COHERENCY {0}\
     PS_LPDMA1_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA2_COHERENCY {0}\
     PS_LPDMA2_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA3_COHERENCY {0}\
     PS_LPDMA3_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA4_COHERENCY {0}\
     PS_LPDMA4_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA5_COHERENCY {0}\
     PS_LPDMA5_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA6_COHERENCY {0}\
     PS_LPDMA6_ROUTE_THROUGH_FPD {0}\
     PS_LPDMA7_COHERENCY {0}\
     PS_LPDMA7_ROUTE_THROUGH_FPD {0}\
     PS_LPD_DMA_CHANNEL_ENABLE {{CH0 0} {CH1 0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6\
0} {CH7 0}}\
     PS_LPD_DMA_CH_TZ {{CH0 NonSecure} {CH1 NonSecure} {CH2 NonSecure} {CH3 NonSecure}\
{CH4 NonSecure} {CH5 NonSecure} {CH6 NonSecure} {CH7 NonSecure}}\
     PS_LPD_DMA_ENABLE {0}\
     PS_LPD_INTERCONNECT_LOAD {90}\
     PS_MIO0 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO1 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO10 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO11 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO12 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO13 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO14 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO15 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO16 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO17 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO18 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO19 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO2 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO20 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO21 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO22 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO23 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO24 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO25 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO3 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO4 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO5 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO6 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO7 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO8 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO9 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_M_AXI_FPD_DATA_WIDTH {128}\
     PS_M_AXI_GP4_DATA_WIDTH {128}\
     PS_M_AXI_LPD_DATA_WIDTH {128}\
     PS_NOC_PS_CCI_DATA_WIDTH {128}\
     PS_NOC_PS_NCI_DATA_WIDTH {128}\
     PS_NOC_PS_PCI_DATA_WIDTH {128}\
     PS_NOC_PS_PMC_DATA_WIDTH {128}\
     PS_NUM_F2P0_INTR_INPUTS {1}\
     PS_NUM_F2P1_INTR_INPUTS {1}\
     PS_NUM_FABRIC_RESETS {0}\
     PS_OCM_ACTIVE_BLOCKS {1}\
     PS_PCIE1_PERIPHERAL_ENABLE {0}\
     PS_PCIE2_PERIPHERAL_ENABLE {0}\
     PS_PCIE_EP_RESET1_IO {PMC_MIO 38}\
     PS_PCIE_EP_RESET2_IO {PMC_MIO 39}\
     PS_PCIE_PERIPHERAL_ENABLE {0}\
     PS_PCIE_RESET {{ENABLE 1}}\
     PS_PCIE_ROOT_RESET1_IO {None}\
     PS_PCIE_ROOT_RESET1_IO_DIR {output}\
     PS_PCIE_ROOT_RESET1_POLARITY {Active Low}\
     PS_PCIE_ROOT_RESET2_IO {None}\
     PS_PCIE_ROOT_RESET2_IO_DIR {output}\
     PS_PCIE_ROOT_RESET2_POLARITY {Active Low}\
     PS_PL_CONNECTIVITY_MODE {Custom}\
     PS_PL_DONE {0}\
     PS_PL_PASS_AXPROT_VALUE {0}\
     PS_PMCPL_CLK0_BUF {1}\
     PS_PMCPL_CLK1_BUF {1}\
     PS_PMCPL_CLK2_BUF {1}\
     PS_PMCPL_CLK3_BUF {1}\
     PS_PMCPL_IRO_CLK_BUF {1}\
     PS_PMU_PERIPHERAL_ENABLE {0}\
     PS_PS_ENABLE {0}\
     PS_PS_NOC_CCI_DATA_WIDTH {128}\
     PS_PS_NOC_NCI_DATA_WIDTH {128}\
     PS_PS_NOC_PCI_DATA_WIDTH {128}\
     PS_PS_NOC_PMC_DATA_WIDTH {128}\
     PS_PS_NOC_RPU_DATA_WIDTH {128}\
     PS_R5_ACTIVE_BLOCKS {2}\
     PS_R5_LOAD {90}\
     PS_RPU_COHERENCY {0}\
     PS_SLR_TYPE {master}\
     PS_SMON_PL_PORTS_ENABLE {0}\
     PS_SPI0 {{GRP_SS0_ENABLE 0} {GRP_SS0_IO {PMC_MIO 15}} {GRP_SS1_ENABLE 0}\
{GRP_SS1_IO {PMC_MIO 14}} {GRP_SS2_ENABLE 0} {GRP_SS2_IO {PMC_MIO 13}}\
{PERIPHERAL_ENABLE 0} {PERIPHERAL_IO {PMC_MIO 12 .. 17}}}\
     PS_SPI1 {{GRP_SS0_ENABLE 0} {GRP_SS0_IO {PS_MIO 9}} {GRP_SS1_ENABLE 0}\
{GRP_SS1_IO {PS_MIO 8}} {GRP_SS2_ENABLE 0} {GRP_SS2_IO {PS_MIO 7}}\
{PERIPHERAL_ENABLE 0} {PERIPHERAL_IO {PS_MIO 6 .. 11}}}\
     PS_S_AXI_ACE_DATA_WIDTH {128}\
     PS_S_AXI_ACP_DATA_WIDTH {128}\
     PS_S_AXI_FPD_DATA_WIDTH {128}\
     PS_S_AXI_GP2_DATA_WIDTH {128}\
     PS_S_AXI_LPD_DATA_WIDTH {128}\
     PS_TCM_ACTIVE_BLOCKS {2}\
     PS_TIE_MJTAG_TCK_TO_GND {1}\
     PS_TRACE_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 30 .. 47}}}\
     PS_TRACE_WIDTH {2Bit}\
     PS_TRISTATE_INVERTED {1}\
     PS_TTC0_CLK {{ENABLE 0} {IO {PS_MIO 6}}}\
     PS_TTC0_PERIPHERAL_ENABLE {0}\
     PS_TTC0_REF_CTRL_ACT_FREQMHZ {100}\
     PS_TTC0_REF_CTRL_FREQMHZ {100}\
     PS_TTC0_WAVEOUT {{ENABLE 0} {IO {PS_MIO 7}}}\
     PS_TTC1_CLK {{ENABLE 0} {IO {PS_MIO 12}}}\
     PS_TTC1_PERIPHERAL_ENABLE {0}\
     PS_TTC1_REF_CTRL_ACT_FREQMHZ {100}\
     PS_TTC1_REF_CTRL_FREQMHZ {100}\
     PS_TTC1_WAVEOUT {{ENABLE 0} {IO {PS_MIO 13}}}\
     PS_TTC2_CLK {{ENABLE 0} {IO {PS_MIO 2}}}\
     PS_TTC2_PERIPHERAL_ENABLE {0}\
     PS_TTC2_REF_CTRL_ACT_FREQMHZ {100}\
     PS_TTC2_REF_CTRL_FREQMHZ {100}\
     PS_TTC2_WAVEOUT {{ENABLE 0} {IO {PS_MIO 3}}}\
     PS_TTC3_CLK {{ENABLE 0} {IO {PS_MIO 16}}}\
     PS_TTC3_PERIPHERAL_ENABLE {0}\
     PS_TTC3_REF_CTRL_ACT_FREQMHZ {100}\
     PS_TTC3_REF_CTRL_FREQMHZ {100}\
     PS_TTC3_WAVEOUT {{ENABLE 0} {IO {PS_MIO 17}}}\
     PS_TTC_APB_CLK_TTC0_SEL {APB}\
     PS_TTC_APB_CLK_TTC1_SEL {APB}\
     PS_TTC_APB_CLK_TTC2_SEL {APB}\
     PS_TTC_APB_CLK_TTC3_SEL {APB}\
     PS_UART0_BAUD_RATE {115200}\
     PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 16 .. 17}}}\
     PS_UART0_RTS_CTS {{ENABLE 0} {IO {PS_MIO 2 .. 3}}}\
     PS_UART1_BAUD_RATE {115200}\
     PS_UART1_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 4 .. 5}}}\
     PS_UART1_RTS_CTS {{ENABLE 0} {IO {PMC_MIO 6 .. 7}}}\
     PS_UNITS_MODE {Custom}\
     PS_USB3_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 13 .. 25}}}\
     PS_USB_COHERENCY {0}\
     PS_USB_ROUTE_THROUGH_FPD {0}\
     PS_USE_ACE_LITE {0}\
     PS_USE_APU_EVENT_BUS {0}\
     PS_USE_APU_INTERRUPT {0}\
     PS_USE_AXI4_EXT_USER_BITS {0}\
     PS_USE_BSCAN_USER1 {0}\
     PS_USE_BSCAN_USER2 {0}\
     PS_USE_BSCAN_USER3 {0}\
     PS_USE_BSCAN_USER4 {0}\
     PS_USE_CAPTURE {0}\
     PS_USE_CLK {0}\
     PS_USE_DEBUG_TEST {0}\
     PS_USE_DIFF_RW_CLK_S_AXI_FPD {0}\
     PS_USE_DIFF_RW_CLK_S_AXI_GP2 {0}\
     PS_USE_DIFF_RW_CLK_S_AXI_LPD {0}\
     PS_USE_ENET0_PTP {0}\
     PS_USE_ENET1_PTP {0}\
     PS_USE_FIFO_ENET0 {0}\
     PS_USE_FIFO_ENET1 {0}\
     PS_USE_FIXED_IO {0}\
     PS_USE_FPD_AXI_NOC0 {1}\
     PS_USE_FPD_AXI_NOC1 {1}\
     PS_USE_FPD_CCI_NOC {1}\
     PS_USE_FPD_CCI_NOC0 {0}\
     PS_USE_FPD_CCI_NOC1 {0}\
     PS_USE_FPD_CCI_NOC2 {0}\
     PS_USE_FPD_CCI_NOC3 {0}\
     PS_USE_FTM_GPI {0}\
     PS_USE_FTM_GPO {0}\
     PS_USE_HSDP_PL {0}\
     PS_USE_MJTAG_TCK_TIE_OFF {0}\
     PS_USE_M_AXI_FPD {1}\
     PS_USE_M_AXI_LPD {1}\
     PS_USE_NOC_FPD_AXI0 {0}\
     PS_USE_NOC_FPD_AXI1 {0}\
     PS_USE_NOC_FPD_CCI0 {0}\
     PS_USE_NOC_FPD_CCI1 {0}\
     PS_USE_NOC_LPD_AXI0 {0}\
     PS_USE_NOC_PS_PCI_0 {0}\
     PS_USE_NOC_PS_PMC_0 {0}\
     PS_USE_NPI_CLK {0}\
     PS_USE_NPI_RST {0}\
     PS_USE_PL_FPD_AUX_REF_CLK {0}\
     PS_USE_PL_LPD_AUX_REF_CLK {0}\
     PS_USE_PMC {0}\
     PS_USE_PMCPL_CLK0 {1}\
     PS_USE_PMCPL_CLK1 {0}\
     PS_USE_PMCPL_CLK2 {0}\
     PS_USE_PMCPL_CLK3 {0}\
     PS_USE_PMCPL_IRO_CLK {0}\
     PS_USE_PSPL_IRQ_FPD {0}\
     PS_USE_PSPL_IRQ_LPD {0}\
     PS_USE_PSPL_IRQ_PMC {0}\
     PS_USE_PS_NOC_PCI_0 {0}\
     PS_USE_PS_NOC_PCI_1 {0}\
     PS_USE_PS_NOC_PMC_0 {0}\
     PS_USE_PS_NOC_PMC_1 {0}\
     PS_USE_RPU_EVENT {0}\
     PS_USE_RPU_INTERRUPT {0}\
     PS_USE_RTC {0}\
     PS_USE_SMMU {0}\
     PS_USE_STARTUP {0}\
     PS_USE_STM {0}\
     PS_USE_S_ACP_FPD {0}\
     PS_USE_S_AXI_ACE {0}\
     PS_USE_S_AXI_FPD {0}\
     PS_USE_S_AXI_GP2 {0}\
     PS_USE_S_AXI_LPD {0}\
     PS_USE_TRACE_ATB {0}\
     PS_WDT0_REF_CTRL_ACT_FREQMHZ {100}\
     PS_WDT0_REF_CTRL_FREQMHZ {100}\
     PS_WDT0_REF_CTRL_SEL {NONE}\
     PS_WDT1_REF_CTRL_ACT_FREQMHZ {100}\
     PS_WDT1_REF_CTRL_FREQMHZ {100}\
     PS_WDT1_REF_CTRL_SEL {NONE}\
     PS_WWDT0_CLK {{ENABLE 0} {IO {PMC_MIO 0}}}\
     PS_WWDT0_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 0 .. 5}}}\
     PS_WWDT1_CLK {{ENABLE 0} {IO {PMC_MIO 6}}}\
     PS_WWDT1_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 6 .. 11}}}\
     SEM_ERROR_HANDLE_OPTIONS {Detect & Correct}\
     SEM_EVENT_LOG_OPTIONS {Log & Notify}\
     SEM_MEM_BUILT_IN_SELF_TEST {0}\
     SEM_MEM_ENABLE_ALL_TEST_FEATURE {0}\
     SEM_MEM_ENABLE_SCAN_AFTER {0}\
     SEM_MEM_GOLDEN_ECC {0}\
     SEM_MEM_GOLDEN_ECC_SW {0}\
     SEM_MEM_SCAN {0}\
     SEM_NPI_BUILT_IN_SELF_TEST {0}\
     SEM_NPI_ENABLE_ALL_TEST_FEATURE {0}\
     SEM_NPI_ENABLE_SCAN_AFTER {0}\
     SEM_NPI_GOLDEN_CHECKSUM_SW {0}\
     SEM_NPI_SCAN {0}\
     SEM_TIME_INTERVAL_BETWEEN_SCANS {0}\
     SMON_ALARMS {Set_Alarms_On}\
     SMON_ENABLE_INT_VOLTAGE_MONITORING {0}\
     SMON_ENABLE_TEMP_AVERAGING {0}\
     SMON_INTERFACE_TO_USE {None}\
     SMON_INT_MEASUREMENT_ALARM_ENABLE {0}\
     SMON_INT_MEASUREMENT_AVG_ENABLE {0}\
     SMON_INT_MEASUREMENT_ENABLE {0}\
     SMON_INT_MEASUREMENT_MODE {0}\
     SMON_INT_MEASUREMENT_TH_HIGH {0}\
     SMON_INT_MEASUREMENT_TH_LOW {0}\
     SMON_MEAS0 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_103} {SUPPLY_NUM 0}}\
     SMON_MEAS1 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_104} {SUPPLY_NUM 0}}\
     SMON_MEAS10 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_206}\
{SUPPLY_NUM 0}}\
     SMON_MEAS100 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS101 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS102 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS103 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS104 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS105 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS106 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS107 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS108 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS109 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS11 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_103} {SUPPLY_NUM\
0}}\
     SMON_MEAS110 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS111 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS112 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS113 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS114 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS115 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS116 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS117 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS118 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS119 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS12 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_104} {SUPPLY_NUM\
0}}\
     SMON_MEAS120 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS121 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS122 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS123 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS124 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS125 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS126 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS127 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS128 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS129 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS13 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_105} {SUPPLY_NUM\
0}}\
     SMON_MEAS130 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS131 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS132 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS133 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS134 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS135 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS136 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS137 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS138 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS139 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS14 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_106} {SUPPLY_NUM\
0}}\
     SMON_MEAS140 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS141 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS142 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS143 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS144 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS145 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS146 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS147 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS148 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS149 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS15 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_200} {SUPPLY_NUM\
0}}\
     SMON_MEAS150 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS151 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS152 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS153 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS154 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS155 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS156 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS157 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS158 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS159 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS16 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_201} {SUPPLY_NUM\
0}}\
     SMON_MEAS160 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS161 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS162 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCINT}}\
     SMON_MEAS163 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCAUX}}\
     SMON_MEAS164 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_RAM}}\
     SMON_MEAS165 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_SOC}}\
     SMON_MEAS166 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PSFP}}\
     SMON_MEAS167 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PSLP}}\
     SMON_MEAS168 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCAUX_PMC}}\
     SMON_MEAS169 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PMC}}\
     SMON_MEAS17 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_202} {SUPPLY_NUM\
0}}\
     SMON_MEAS170 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS171 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS172 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS173 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS174 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS175 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103}}\
     SMON_MEAS18 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_203} {SUPPLY_NUM\
0}}\
     SMON_MEAS19 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_204} {SUPPLY_NUM\
0}}\
     SMON_MEAS2 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_105} {SUPPLY_NUM 0}}\
     SMON_MEAS20 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_205} {SUPPLY_NUM\
0}}\
     SMON_MEAS21 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCC_206} {SUPPLY_NUM\
0}}\
     SMON_MEAS22 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_103} {SUPPLY_NUM\
0}}\
     SMON_MEAS23 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_104} {SUPPLY_NUM\
0}}\
     SMON_MEAS24 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_105} {SUPPLY_NUM\
0}}\
     SMON_MEAS25 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_106} {SUPPLY_NUM\
0}}\
     SMON_MEAS26 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_200} {SUPPLY_NUM\
0}}\
     SMON_MEAS27 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_201} {SUPPLY_NUM\
0}}\
     SMON_MEAS28 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_202} {SUPPLY_NUM\
0}}\
     SMON_MEAS29 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_203} {SUPPLY_NUM\
0}}\
     SMON_MEAS3 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_106} {SUPPLY_NUM 0}}\
     SMON_MEAS30 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_204} {SUPPLY_NUM\
0}}\
     SMON_MEAS31 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_205} {SUPPLY_NUM\
0}}\
     SMON_MEAS32 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVTT_206} {SUPPLY_NUM\
0}}\
     SMON_MEAS33 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCAUX} {SUPPLY_NUM 0}}\
     SMON_MEAS34 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCAUX_PMC} {SUPPLY_NUM 0}}\
     SMON_MEAS35 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCAUX_SMON} {SUPPLY_NUM 0}}\
     SMON_MEAS36 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCINT} {SUPPLY_NUM 0}}\
     SMON_MEAS37 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_306} {SUPPLY_NUM 0}}\
     SMON_MEAS38 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_406} {SUPPLY_NUM 0}}\
     SMON_MEAS39 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_500} {SUPPLY_NUM 0}}\
     SMON_MEAS4 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_200} {SUPPLY_NUM 0}}\
     SMON_MEAS40 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_501} {SUPPLY_NUM 0}}\
     SMON_MEAS41 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_502} {SUPPLY_NUM 0}}\
     SMON_MEAS42 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {4 V unipolar}} {NAME VCCO_503} {SUPPLY_NUM 0}}\
     SMON_MEAS43 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_700} {SUPPLY_NUM 0}}\
     SMON_MEAS44 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_701} {SUPPLY_NUM 0}}\
     SMON_MEAS45 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_702} {SUPPLY_NUM 0}}\
     SMON_MEAS46 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_703} {SUPPLY_NUM 0}}\
     SMON_MEAS47 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_704} {SUPPLY_NUM 0}}\
     SMON_MEAS48 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_705} {SUPPLY_NUM 0}}\
     SMON_MEAS49 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_706} {SUPPLY_NUM 0}}\
     SMON_MEAS5 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_201} {SUPPLY_NUM 0}}\
     SMON_MEAS50 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_707} {SUPPLY_NUM 0}}\
     SMON_MEAS51 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_708} {SUPPLY_NUM 0}}\
     SMON_MEAS52 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_709} {SUPPLY_NUM 0}}\
     SMON_MEAS53 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_710} {SUPPLY_NUM 0}}\
     SMON_MEAS54 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCCO_711} {SUPPLY_NUM 0}}\
     SMON_MEAS55 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_BATT} {SUPPLY_NUM 0}}\
     SMON_MEAS56 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PMC} {SUPPLY_NUM 0}}\
     SMON_MEAS57 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PSFP} {SUPPLY_NUM 0}}\
     SMON_MEAS58 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_PSLP} {SUPPLY_NUM 0}}\
     SMON_MEAS59 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_RAM} {SUPPLY_NUM 0}}\
     SMON_MEAS6 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_202} {SUPPLY_NUM 0}}\
     SMON_MEAS60 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VCC_SOC} {SUPPLY_NUM 0}}\
     SMON_MEAS61 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE {2 V unipolar}} {NAME VP_VN} {SUPPLY_NUM 0}}\
     SMON_MEAS62 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCCO_711} {SUPPLY_NUM 0}}\
     SMON_MEAS63 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCCO_PKG_306} {SUPPLY_NUM 0}}\
     SMON_MEAS64 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCCO_PKG_406} {SUPPLY_NUM 0}}\
     SMON_MEAS65 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_BATT} {SUPPLY_NUM 0}}\
     SMON_MEAS66 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_700} {SUPPLY_NUM 0}}\
     SMON_MEAS67 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_701} {SUPPLY_NUM 0}}\
     SMON_MEAS68 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_702} {SUPPLY_NUM 0}}\
     SMON_MEAS69 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_703} {SUPPLY_NUM 0}}\
     SMON_MEAS7 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_203} {SUPPLY_NUM 0}}\
     SMON_MEAS70 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_704} {SUPPLY_NUM 0}}\
     SMON_MEAS71 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_705} {SUPPLY_NUM 0}}\
     SMON_MEAS72 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_706} {SUPPLY_NUM 0}}\
     SMON_MEAS73 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_707} {SUPPLY_NUM 0}}\
     SMON_MEAS74 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_708} {SUPPLY_NUM 0}}\
     SMON_MEAS75 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_709} {SUPPLY_NUM 0}}\
     SMON_MEAS76 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_710} {SUPPLY_NUM 0}}\
     SMON_MEAS77 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_711} {SUPPLY_NUM 0}}\
     SMON_MEAS78 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_PKG_306} {SUPPLY_NUM 0}}\
     SMON_MEAS79 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_IO_PKG_406} {SUPPLY_NUM 0}}\
     SMON_MEAS8 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_204} {SUPPLY_NUM 0}}\
     SMON_MEAS80 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_PMC} {SUPPLY_NUM 0}}\
     SMON_MEAS81 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_PSFP} {SUPPLY_NUM 0}}\
     SMON_MEAS82 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_PSLP} {SUPPLY_NUM 0}}\
     SMON_MEAS83 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_RAM} {SUPPLY_NUM 0}}\
     SMON_MEAS84 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VCC_SOC} {SUPPLY_NUM 0}}\
     SMON_MEAS85 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME VP_VN} {SUPPLY_NUM 0}}\
     SMON_MEAS86 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS87 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS88 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS89 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS9 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0}\
{ENABLE 0} {MODE {2 V unipolar}} {NAME GTY_AVCCAUX_205} {SUPPLY_NUM 0}}\
     SMON_MEAS90 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS91 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS92 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS93 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS94 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS95 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS96 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS97 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS98 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEAS99 {{ALARM_ENABLE 0} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN\
0} {ENABLE 0} {MODE None} {NAME GT_AVAUX_PKG_103} {SUPPLY_NUM 0}}\
     SMON_MEASUREMENT_COUNT {62}\
     SMON_MEASUREMENT_LIST {BANK_VOLTAGE:GTY_AVTT-GTY_AVTT_103,GTY_AVTT_104,GTY_AVTT_105,GTY_AVTT_106,GTY_AVTT_200,GTY_AVTT_201,GTY_AVTT_202,GTY_AVTT_203,GTY_AVTT_204,GTY_AVTT_205,GTY_AVTT_206#VCC-GTY_AVCC_103,GTY_AVCC_104,GTY_AVCC_105,GTY_AVCC_106,GTY_AVCC_200,GTY_AVCC_201,GTY_AVCC_202,GTY_AVCC_203,GTY_AVCC_204,GTY_AVCC_205,GTY_AVCC_206#VCCAUX-GTY_AVCCAUX_103,GTY_AVCCAUX_104,GTY_AVCCAUX_105,GTY_AVCCAUX_106,GTY_AVCCAUX_200,GTY_AVCCAUX_201,GTY_AVCCAUX_202,GTY_AVCCAUX_203,GTY_AVCCAUX_204,GTY_AVCCAUX_205,GTY_AVCCAUX_206#VCCO-VCCO_306,VCCO_406,VCCO_500,VCCO_501,VCCO_502,VCCO_503,VCCO_700,VCCO_701,VCCO_702,VCCO_703,VCCO_704,VCCO_705,VCCO_706,VCCO_707,VCCO_708,VCCO_709,VCCO_710,VCCO_711|DEDICATED_PAD:VP-VP_VN|SUPPLY_VOLTAGE:VCC-VCC_BATT,VCC_PMC,VCC_PSFP,VCC_PSLP,VCC_RAM,VCC_SOC#VCCAUX-VCCAUX,VCCAUX_PMC,VCCAUX_SMON#VCCINT-VCCINT}\
     SMON_OT {{THRESHOLD_LOWER 70} {THRESHOLD_UPPER 125}}\
     SMON_PMBUS_ADDRESS {0x0}\
     SMON_PMBUS_UNRESTRICTED {0}\
     SMON_REFERENCE_SOURCE {Internal}\
     SMON_TEMP_AVERAGING_SAMPLES {0}\
     SMON_TEMP_THRESHOLD {0}\
     SMON_USER_TEMP {{THRESHOLD_LOWER 70} {THRESHOLD_UPPER 125} {USER_ALARM_TYPE\
hysteresis}}\
     SMON_VAUX_CH0 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH0} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH1 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH1} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH10 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH10} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH11 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH11} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH12 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH12} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH13 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH13} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH14 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH14} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH15 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH15} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH2 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH2} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH3 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH3} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH4 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH4} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH5 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH5} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH6 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH6} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH7 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH7} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH8 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH8} {SUPPLY_NUM 0}}\
     SMON_VAUX_CH9 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 0} {AVERAGE_EN 0}\
{ENABLE 0} {IO_N PMC_MIO1_500} {IO_P PMC_MIO0_500} {MODE {1 V\
unipolar}} {NAME VAUX_CH9} {SUPPLY_NUM 0}}\
     SMON_VAUX_IO_BANK {MIO_BANK0}\
     SMON_VOLTAGE_AVERAGING_SAMPLES {None}\
     SPP_PSPMC_FROM_CORE_WIDTH {12000}\
     SPP_PSPMC_TO_CORE_WIDTH {12000}\
     SUBPRESET1 {Custom}\
     USE_UART0_IN_DEVICE_BOOT {0}\
     preset {None}\
   } \
   CONFIG.PS_PMC_CONFIG_APPLIED {1} \
 ] $cips_0

  # Create instance: clk_wizard_0, and set properties
  set clk_wizard_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wizard_0 ]
  set_property -dict [ list \
   CONFIG.CLKOUT1_DIVIDE {24.000000} \
   CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
   CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
   CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
   CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
   CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
   CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
   CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {125.000,100.000,100.000,100.000,100.000,100.000,100.000} \
   CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
   CONFIG.CLKOUT_USED {true,false,false,false,false,false,false} \
   CONFIG.RESET_TYPE {ACTIVE_LOW} \
   CONFIG.USE_LOCKED {true} \
   CONFIG.USE_RESET {true} \
 ] $clk_wizard_0

  # Create instance: emb_mem_axi_bridge_mem, and set properties
  set emb_mem_axi_bridge_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:emb_mem_gen:1.0 emb_mem_axi_bridge_mem ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH_A {17} \
   CONFIG.ADDR_WIDTH_B {17} \
   CONFIG.MEMORY_TYPE {True_Dual_Port_RAM} \
   CONFIG.READ_DATA_WIDTH_A {256} \
   CONFIG.READ_DATA_WIDTH_B {256} \
 ] $emb_mem_axi_bridge_mem

  # Create instance: emb_mem_dma_mem, and set properties
  set emb_mem_dma_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:emb_mem_gen:1.0 emb_mem_dma_mem ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH_A {15} \
   CONFIG.ADDR_WIDTH_B {15} \
   CONFIG.MEMORY_TYPE {True_Dual_Port_RAM} \
   CONFIG.READ_DATA_WIDTH_A {256} \
   CONFIG.READ_DATA_WIDTH_B {256} \
 ] $emb_mem_dma_mem

  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: qdma_host_mem, and set properties
  set qdma_host_mem [ create_bd_cell -type ip -vlnv xilinx.com:ip:qdma:4.0 qdma_host_mem ]
  set_property -dict [ list \
   CONFIG.INS_LOSS_NYQ {15} \
   CONFIG.MAILBOX_ENABLE {false} \
   CONFIG.MSI_X_OPTIONS {MSI-X_External} \
   CONFIG.PCIE_BOARD_INTERFACE {Custom} \
   CONFIG.PF0_MSIX_CAP_PBA_BIR_qdma {BAR_1:0} \
   CONFIG.PF0_MSIX_CAP_PBA_OFFSET_qdma {34000} \
   CONFIG.PF0_MSIX_CAP_TABLE_BIR_qdma {BAR_1:0} \
   CONFIG.PF0_MSIX_CAP_TABLE_OFFSET_qdma {30000} \
   CONFIG.PF0_MSIX_CAP_TABLE_SIZE_qdma {007} \
   CONFIG.PF0_SRIOV_CAP_INITIAL_VF {4} \
   CONFIG.PF0_SRIOV_FIRST_VF_OFFSET {1} \
   CONFIG.PF0_SRIOV_FUNC_DEP_LINK {0000} \
   CONFIG.PF0_SRIOV_SUPPORTED_PAGE_SIZE {00000553} \
   CONFIG.PF0_SRIOV_VF_DEVICE_ID {C034} \
   CONFIG.PF1_INTERRUPT_PIN {INTA} \
   CONFIG.PF1_MSIX_CAP_PBA_BIR_qdma {BAR_1:0} \
   CONFIG.PF1_MSIX_CAP_PBA_OFFSET_qdma {34000} \
   CONFIG.PF1_MSIX_CAP_TABLE_BIR_qdma {BAR_1:0} \
   CONFIG.PF1_MSIX_CAP_TABLE_OFFSET_qdma {30000} \
   CONFIG.PF1_MSIX_CAP_TABLE_SIZE_qdma {007} \
   CONFIG.PF1_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF1_SRIOV_CAP_INITIAL_VF {4} \
   CONFIG.PF1_SRIOV_CAP_VER {1} \
   CONFIG.PF1_SRIOV_FIRST_VF_OFFSET {4} \
   CONFIG.PF1_SRIOV_FUNC_DEP_LINK {0001} \
   CONFIG.PF1_SRIOV_SUPPORTED_PAGE_SIZE {00000553} \
   CONFIG.PF1_SRIOV_VF_DEVICE_ID {C134} \
   CONFIG.PF2_INTERRUPT_PIN {INTA} \
   CONFIG.PF2_MSIX_CAP_PBA_BIR_qdma {BAR_1:0} \
   CONFIG.PF2_MSIX_CAP_PBA_OFFSET_qdma {34000} \
   CONFIG.PF2_MSIX_CAP_TABLE_BIR_qdma {BAR_1:0} \
   CONFIG.PF2_MSIX_CAP_TABLE_OFFSET_qdma {30000} \
   CONFIG.PF2_MSIX_CAP_TABLE_SIZE_qdma {007} \
   CONFIG.PF2_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF2_SRIOV_CAP_INITIAL_VF {4} \
   CONFIG.PF2_SRIOV_CAP_VER {1} \
   CONFIG.PF2_SRIOV_FIRST_VF_OFFSET {7} \
   CONFIG.PF2_SRIOV_FUNC_DEP_LINK {0002} \
   CONFIG.PF2_SRIOV_SUPPORTED_PAGE_SIZE {00000553} \
   CONFIG.PF2_SRIOV_VF_DEVICE_ID {C234} \
   CONFIG.PF3_INTERRUPT_PIN {INTA} \
   CONFIG.PF3_MSIX_CAP_PBA_BIR_qdma {BAR_1:0} \
   CONFIG.PF3_MSIX_CAP_PBA_OFFSET_qdma {34000} \
   CONFIG.PF3_MSIX_CAP_TABLE_BIR_qdma {BAR_1:0} \
   CONFIG.PF3_MSIX_CAP_TABLE_OFFSET_qdma {30000} \
   CONFIG.PF3_MSIX_CAP_TABLE_SIZE_qdma {007} \
   CONFIG.PF3_MSI_CAP_MULTIMSGCAP {1_vector} \
   CONFIG.PF3_SRIOV_CAP_INITIAL_VF {4} \
   CONFIG.PF3_SRIOV_CAP_VER {1} \
   CONFIG.PF3_SRIOV_FIRST_VF_OFFSET {10} \
   CONFIG.PF3_SRIOV_FUNC_DEP_LINK {0003} \
   CONFIG.PF3_SRIOV_SUPPORTED_PAGE_SIZE {00000553} \
   CONFIG.PF3_SRIOV_VF_DEVICE_ID {C334} \
   CONFIG.PHY_LP_TXPRESET {4} \
   CONFIG.RX_PPM_OFFSET {0} \
   CONFIG.RX_SSC_PPM {0} \
   CONFIG.SRIOV_CAP_ENABLE {false} \
   CONFIG.SRIOV_FIRST_VF_OFFSET {1} \
   CONFIG.SYS_RST_N_BOARD_INTERFACE {Custom} \
   CONFIG.Shared_Logic {1} \
   CONFIG.Shared_Logic_Both {false} \
   CONFIG.Shared_Logic_Clk {false} \
   CONFIG.Shared_Logic_Gtc {false} \
   CONFIG.acs_ext_cap_enable {false} \
   CONFIG.adv_int_usr {false} \
   CONFIG.alf_cap_enable {false} \
   CONFIG.async_clk_enable {false} \
   CONFIG.axi_aclk_loopback {false} \
   CONFIG.axi_addr_width {64} \
   CONFIG.axi_bypass_64bit_en {false} \
   CONFIG.axi_bypass_prefetchable {false} \
   CONFIG.axi_data_width {256_bit} \
   CONFIG.axi_id_width {4} \
   CONFIG.axi_vip_in_exdes {false} \
   CONFIG.axibar2pciebar_0 {0x0000000000000000} \
   CONFIG.axibar2pciebar_1 {0x0000000000000000} \
   CONFIG.axibar2pciebar_2 {0x0000000000000000} \
   CONFIG.axibar2pciebar_3 {0x0000000000000000} \
   CONFIG.axibar2pciebar_4 {0x0000000000000000} \
   CONFIG.axibar2pciebar_5 {0x0000000000000000} \
   CONFIG.axibar_notranslate {false} \
   CONFIG.axibar_num {1} \
   CONFIG.axil_master_64bit_en {false} \
   CONFIG.axil_master_prefetchable {false} \
   CONFIG.axilite_master_en {false} \
   CONFIG.axilite_master_scale {Megabytes} \
   CONFIG.axilite_master_size {1} \
   CONFIG.axist_bypass_en {true} \
   CONFIG.axist_bypass_scale {Megabytes} \
   CONFIG.axist_bypass_size {1} \
   CONFIG.axisten_freq {125} \
   CONFIG.axisten_if_enable_msg_route {1EFFF} \
   CONFIG.bar0_indicator {1} \
   CONFIG.bar1_indicator {0} \
   CONFIG.bar2_indicator {0} \
   CONFIG.bar3_indicator {0} \
   CONFIG.bar4_indicator {0} \
   CONFIG.bar5_indicator {0} \
   CONFIG.bar_indicator {BAR_0} \
   CONFIG.barlite2 {7} \
   CONFIG.barlite_mb_pf0 {0} \
   CONFIG.barlite_mb_pf1 {0} \
   CONFIG.barlite_mb_pf2 {0} \
   CONFIG.barlite_mb_pf3 {0} \
   CONFIG.bridge_burst {FALSE} \
   CONFIG.bridge_register_access {false} \
   CONFIG.bridge_registers_offset_enable {false} \
   CONFIG.c2h_stream_cpl_col_bit_pos0 {1} \
   CONFIG.c2h_stream_cpl_col_bit_pos1 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos2 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos3 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos4 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos5 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos6 {0} \
   CONFIG.c2h_stream_cpl_col_bit_pos7 {0} \
   CONFIG.c2h_stream_cpl_data_size {8_Bytes} \
   CONFIG.c2h_stream_cpl_err_bit_pos0 {2} \
   CONFIG.c2h_stream_cpl_err_bit_pos1 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos2 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos3 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos4 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos5 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos6 {0} \
   CONFIG.c2h_stream_cpl_err_bit_pos7 {0} \
   CONFIG.c_ats_enable {false} \
   CONFIG.c_m_axi_num_read {8} \
   CONFIG.c_m_axi_num_write {32} \
   CONFIG.c_pri_enable {false} \
   CONFIG.c_s_axi_num_read {8} \
   CONFIG.c_s_axi_num_write {8} \
   CONFIG.c_s_axi_supports_narrow_burst {false} \
   CONFIG.cfg_ext_if {false} \
   CONFIG.cfg_mgmt_if {true} \
   CONFIG.cfg_space_enable {false} \
   CONFIG.comp_timeout {50ms} \
   CONFIG.copy_pf0 {true} \
   CONFIG.copy_sriov_pf0 {true} \
   CONFIG.coreclk_freq {500} \
   CONFIG.csr_axilite_slave {false} \
   CONFIG.csr_module {1} \
   CONFIG.data_mover {false} \
   CONFIG.debug_mode {DEBUG_NONE} \
   CONFIG.dedicate_perst {false} \
   CONFIG.descriptor_bypass_exdes {false} \
   CONFIG.device_port_type {PCI_Express_Endpoint_device} \
   CONFIG.disable_bram_pipeline {false} \
   CONFIG.disable_eq_synchronizer {false} \
   CONFIG.disable_gt_loc {false} \
   CONFIG.disable_user_clock_root {true} \
   CONFIG.dma_2rp {false} \
   CONFIG.dma_intf_sel_qdma {AXI_MM} \
   CONFIG.dma_mode_en {false} \
   CONFIG.dma_reset_source_sel {PCIe_User_Reset} \
   CONFIG.double_quad {false} \
   CONFIG.drp_clk_sel {Internal} \
   CONFIG.dsc_byp_mode {None} \
   CONFIG.dsc_bypass_rd_out {false} \
   CONFIG.dsc_bypass_wr_out {false} \
   CONFIG.en_axi_master_if {false} \
   CONFIG.en_axi_mm_qdma {true} \
   CONFIG.en_axi_slave_if {true} \
   CONFIG.en_axi_st_qdma {false} \
   CONFIG.en_bridge {true} \
   CONFIG.en_bridge_slv {true} \
   CONFIG.en_coreclk_es1 {false} \
   CONFIG.en_dbg_descramble {false} \
   CONFIG.en_debug_ports {false} \
   CONFIG.en_dma_and_bridge {false} \
   CONFIG.en_dma_completion {false} \
   CONFIG.en_ext_ch_gt_drp {false} \
   CONFIG.en_gt_selection {false} \
   CONFIG.en_l23_entry {false} \
   CONFIG.en_pcie_drp {false} \
   CONFIG.en_qdma {true} \
   CONFIG.en_transceiver_status_ports {false} \
   CONFIG.enable_at_ports {false} \
   CONFIG.enable_ats_switch {FALSE} \
   CONFIG.enable_auto_rxeq {False} \
   CONFIG.enable_ccix {FALSE} \
   CONFIG.enable_clock_delay_grp {true} \
   CONFIG.enable_code {0000} \
   CONFIG.enable_dvsec {FALSE} \
   CONFIG.enable_error_injection {false} \
   CONFIG.enable_gen4 {true} \
   CONFIG.enable_ibert {false} \
   CONFIG.enable_jtag_dbg {false} \
   CONFIG.enable_mark_debug {false} \
   CONFIG.enable_more_clk {false} \
   CONFIG.enable_multi_pcie {false} \
   CONFIG.enable_pcie_debug {False} \
   CONFIG.enable_pcie_debug_axi4_st {False} \
   CONFIG.enable_resource_reduction {false} \
   CONFIG.enable_x16 {false} \
   CONFIG.example_design_type {RTL} \
   CONFIG.ext_startup_primitive {false} \
   CONFIG.ext_sys_clk_bufg {false} \
   CONFIG.ext_xvc_vsec_enable {false} \
   CONFIG.flr_enable {false} \
   CONFIG.free_run_freq {100_MHz} \
   CONFIG.functional_mode {QDMA} \
   CONFIG.gen4_eieos_0s7 {true} \
   CONFIG.gt_loc_num {X99Y99} \
   CONFIG.gtcom_in_core_usp {2} \
   CONFIG.gtwiz_in_core_us {1} \
   CONFIG.gtwiz_in_core_usp {1} \
   CONFIG.iep_enable {false} \
   CONFIG.include_baroffset_reg {true} \
   CONFIG.ins_loss_profile {Add-in_Card} \
   CONFIG.insert_cips {false} \
   CONFIG.lane_order {Bottom} \
   CONFIG.lane_reversal {false} \
   CONFIG.last_core_cap_addr {0x100} \
   CONFIG.legacy_cfg_ext_if {false} \
   CONFIG.local_test {false} \
   CONFIG.master_cal_only {true} \
   CONFIG.mcap_enablement {None} \
   CONFIG.mhost_en {false} \
   CONFIG.mode_selection {Advanced} \
   CONFIG.msix_pcie_internal {false} \
   CONFIG.msix_preset {0} \
   CONFIG.msix_type {HARD} \
   CONFIG.mult_pf_des {false} \
   CONFIG.num_queues {512} \
   CONFIG.old_bridge_timeout {false} \
   CONFIG.parity_settings {None} \
   CONFIG.pcie_blk_locn {X0Y1} \
   CONFIG.pcie_extended_tag {true} \
   CONFIG.pcie_id_if {true} \
   CONFIG.pciebar2axibar_axil_master {0x0000000000000000} \
   CONFIG.pciebar2axibar_axist_bypass {0x0000000000000000} \
   CONFIG.pciebar2axibar_xdma {0x0000000000000000} \
   CONFIG.performance {false} \
   CONFIG.performance_exdes {false} \
   CONFIG.pf0_Use_Class_Code_Lookup_Assistant_qdma {false} \
   CONFIG.pf0_ari_enabled {false} \
   CONFIG.pf0_ats_enabled {false} \
   CONFIG.pf0_bar0_64bit_qdma {true} \
   CONFIG.pf0_bar0_enabled_qdma {true} \
   CONFIG.pf0_bar0_index {0} \
   CONFIG.pf0_bar0_prefetchable_qdma {false} \
   CONFIG.pf0_bar0_scale_qdma {Megabytes} \
   CONFIG.pf0_bar0_size_qdma {64} \
   CONFIG.pf0_bar0_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_bar1_64bit_qdma {false} \
   CONFIG.pf0_bar1_enabled_qdma {false} \
   CONFIG.pf0_bar1_index {7} \
   CONFIG.pf0_bar1_prefetchable_qdma {false} \
   CONFIG.pf0_bar1_scale_qdma {Megabytes} \
   CONFIG.pf0_bar1_size_qdma {128} \
   CONFIG.pf0_bar1_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_bar2_64bit_qdma {true} \
   CONFIG.pf0_bar2_enabled_qdma {true} \
   CONFIG.pf0_bar2_index {7} \
   CONFIG.pf0_bar2_prefetchable_qdma {false} \
   CONFIG.pf0_bar2_scale_qdma {Megabytes} \
   CONFIG.pf0_bar2_size_qdma {512} \
   CONFIG.pf0_bar2_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_bar3_64bit_qdma {false} \
   CONFIG.pf0_bar3_enabled_qdma {false} \
   CONFIG.pf0_bar3_index {7} \
   CONFIG.pf0_bar3_prefetchable_qdma {false} \
   CONFIG.pf0_bar3_scale_qdma {Kilobytes} \
   CONFIG.pf0_bar3_size_qdma {128} \
   CONFIG.pf0_bar3_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_bar4_64bit_qdma {true} \
   CONFIG.pf0_bar4_enabled_qdma {true} \
   CONFIG.pf0_bar4_index {7} \
   CONFIG.pf0_bar4_prefetchable_qdma {false} \
   CONFIG.pf0_bar4_scale_qdma {Megabytes} \
   CONFIG.pf0_bar4_size_qdma {1} \
   CONFIG.pf0_bar4_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_bar5_enabled_qdma {false} \
   CONFIG.pf0_bar5_index {7} \
   CONFIG.pf0_bar5_prefetchable_qdma {false} \
   CONFIG.pf0_bar5_scale_qdma {Kilobytes} \
   CONFIG.pf0_bar5_size_qdma {128} \
   CONFIG.pf0_bar5_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf0_base_class_menu_qdma {Memory_controller} \
   CONFIG.pf0_class_code_base_qdma {05} \
   CONFIG.pf0_class_code_interface_qdma {00} \
   CONFIG.pf0_class_code_qdma {058000} \
   CONFIG.pf0_class_code_sub_qdma {80} \
   CONFIG.pf0_device_id {B034} \
   CONFIG.pf0_expansion_rom_enabled_qdma {false} \
   CONFIG.pf0_expansion_rom_scale_qdma {Kilobytes} \
   CONFIG.pf0_expansion_rom_size_qdma {4} \
   CONFIG.pf0_expansion_rom_type_qdma {Expansion_ROM} \
   CONFIG.pf0_interrupt_pin {INTA} \
   CONFIG.pf0_link_status_slot_clock_config {true} \
   CONFIG.pf0_msi_cap_multimsgcap {1_vector} \
   CONFIG.pf0_msi_enabled {false} \
   CONFIG.pf0_msix_cap_pba_bir {BAR_0} \
   CONFIG.pf0_msix_cap_pba_offset {00008FE0} \
   CONFIG.pf0_msix_cap_table_bir {BAR_0} \
   CONFIG.pf0_msix_cap_table_offset {00008000} \
   CONFIG.pf0_msix_cap_table_size {020} \
   CONFIG.pf0_msix_enabled {true} \
   CONFIG.pf0_msix_enabled_qdma {true} \
   CONFIG.pf0_msix_impl_locn {External} \
   CONFIG.pf0_pciebar2axibar_0 {0x0000000800000000} \
   CONFIG.pf0_pciebar2axibar_1 {0x0000000000000000} \
   CONFIG.pf0_pciebar2axibar_2 {0x0000020000000000} \
   CONFIG.pf0_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf0_pciebar2axibar_4 {0x0000020100000000} \
   CONFIG.pf0_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf0_pciebar2axibar_6 {0x0000000000000000} \
   CONFIG.pf0_pri_enabled {false} \
   CONFIG.pf0_rbar_cap_bar0 {0x00000000fff0} \
   CONFIG.pf0_rbar_cap_bar1 {0x000000000000} \
   CONFIG.pf0_rbar_cap_bar2 {0x000000000000} \
   CONFIG.pf0_rbar_cap_bar3 {0x000000000000} \
   CONFIG.pf0_rbar_cap_bar4 {0x000000000000} \
   CONFIG.pf0_rbar_cap_bar5 {0x000000000000} \
   CONFIG.pf0_rbar_num {1} \
   CONFIG.pf0_revision_id {00} \
   CONFIG.pf0_sriov_bar0_64bit {true} \
   CONFIG.pf0_sriov_bar0_enabled {true} \
   CONFIG.pf0_sriov_bar0_prefetchable {true} \
   CONFIG.pf0_sriov_bar0_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar0_size {32} \
   CONFIG.pf0_sriov_bar0_type {DMA} \
   CONFIG.pf0_sriov_bar1_64bit {false} \
   CONFIG.pf0_sriov_bar1_enabled {false} \
   CONFIG.pf0_sriov_bar1_prefetchable {false} \
   CONFIG.pf0_sriov_bar1_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar1_size {4} \
   CONFIG.pf0_sriov_bar1_type {AXI_Bridge_Master} \
   CONFIG.pf0_sriov_bar2_64bit {true} \
   CONFIG.pf0_sriov_bar2_enabled {true} \
   CONFIG.pf0_sriov_bar2_prefetchable {true} \
   CONFIG.pf0_sriov_bar2_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar2_size {4} \
   CONFIG.pf0_sriov_bar2_type {AXI_Lite_Master} \
   CONFIG.pf0_sriov_bar3_64bit {false} \
   CONFIG.pf0_sriov_bar3_enabled {false} \
   CONFIG.pf0_sriov_bar3_prefetchable {false} \
   CONFIG.pf0_sriov_bar3_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar3_size {4} \
   CONFIG.pf0_sriov_bar3_type {AXI_Bridge_Master} \
   CONFIG.pf0_sriov_bar4_64bit {false} \
   CONFIG.pf0_sriov_bar4_enabled {false} \
   CONFIG.pf0_sriov_bar4_prefetchable {false} \
   CONFIG.pf0_sriov_bar4_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar4_size {4} \
   CONFIG.pf0_sriov_bar4_type {AXI_Bridge_Master} \
   CONFIG.pf0_sriov_bar5_64bit {false} \
   CONFIG.pf0_sriov_bar5_enabled {false} \
   CONFIG.pf0_sriov_bar5_prefetchable {false} \
   CONFIG.pf0_sriov_bar5_scale {Kilobytes} \
   CONFIG.pf0_sriov_bar5_size {4} \
   CONFIG.pf0_sriov_bar5_type {AXI_Bridge_Master} \
   CONFIG.pf0_sriov_cap_ver {1} \
   CONFIG.pf0_sub_class_interface_menu_qdma {Other_memory_controller} \
   CONFIG.pf0_subsystem_id {0007} \
   CONFIG.pf0_subsystem_vendor_id {10EE} \
   CONFIG.pf0_vc_cap_enabled {true} \
   CONFIG.pf0_vf_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf0_vf_pciebar2axibar_1 {0x0000000040000000} \
   CONFIG.pf0_vf_pciebar2axibar_2 {0x0000000040000000} \
   CONFIG.pf0_vf_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf0_vf_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf0_vf_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf1_Use_Class_Code_Lookup_Assistant_qdma {false} \
   CONFIG.pf1_bar0_64bit_qdma {true} \
   CONFIG.pf1_bar0_enabled_qdma {true} \
   CONFIG.pf1_bar0_index {0} \
   CONFIG.pf1_bar0_prefetchable_qdma {false} \
   CONFIG.pf1_bar0_scale_qdma {Megabytes} \
   CONFIG.pf1_bar0_size_qdma {64} \
   CONFIG.pf1_bar0_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_bar1_64bit_qdma {false} \
   CONFIG.pf1_bar1_enabled_qdma {false} \
   CONFIG.pf1_bar1_index {7} \
   CONFIG.pf1_bar1_prefetchable_qdma {false} \
   CONFIG.pf1_bar1_scale_qdma {Megabytes} \
   CONFIG.pf1_bar1_size_qdma {128} \
   CONFIG.pf1_bar1_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_bar2_64bit_qdma {true} \
   CONFIG.pf1_bar2_enabled_qdma {true} \
   CONFIG.pf1_bar2_index {7} \
   CONFIG.pf1_bar2_prefetchable_qdma {false} \
   CONFIG.pf1_bar2_scale_qdma {Megabytes} \
   CONFIG.pf1_bar2_size_qdma {512} \
   CONFIG.pf1_bar2_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_bar3_64bit_qdma {false} \
   CONFIG.pf1_bar3_enabled_qdma {false} \
   CONFIG.pf1_bar3_index {7} \
   CONFIG.pf1_bar3_prefetchable_qdma {false} \
   CONFIG.pf1_bar3_scale_qdma {Kilobytes} \
   CONFIG.pf1_bar3_size_qdma {128} \
   CONFIG.pf1_bar3_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_bar4_64bit_qdma {true} \
   CONFIG.pf1_bar4_enabled_qdma {true} \
   CONFIG.pf1_bar4_index {7} \
   CONFIG.pf1_bar4_prefetchable_qdma {false} \
   CONFIG.pf1_bar4_scale_qdma {Megabytes} \
   CONFIG.pf1_bar4_size_qdma {1} \
   CONFIG.pf1_bar4_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_bar5_enabled_qdma {false} \
   CONFIG.pf1_bar5_index {7} \
   CONFIG.pf1_bar5_prefetchable_qdma {false} \
   CONFIG.pf1_bar5_scale_qdma {Kilobytes} \
   CONFIG.pf1_bar5_size_qdma {128} \
   CONFIG.pf1_bar5_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf1_base_class_menu_qdma {Memory_controller} \
   CONFIG.pf1_class_code_base_qdma {05} \
   CONFIG.pf1_class_code_interface_qdma {00} \
   CONFIG.pf1_class_code_qdma {058000} \
   CONFIG.pf1_class_code_sub_qdma {80} \
   CONFIG.pf1_device_id {913F} \
   CONFIG.pf1_expansion_rom_enabled_qdma {false} \
   CONFIG.pf1_expansion_rom_scale_qdma {Kilobytes} \
   CONFIG.pf1_expansion_rom_size_qdma {4} \
   CONFIG.pf1_expansion_rom_type_qdma {Expansion_ROM} \
   CONFIG.pf1_msi_enabled {false} \
   CONFIG.pf1_msix_cap_pba_bir {BAR_0} \
   CONFIG.pf1_msix_cap_pba_offset {00010050} \
   CONFIG.pf1_msix_cap_table_bir {BAR_0} \
   CONFIG.pf1_msix_cap_table_offset {00010040} \
   CONFIG.pf1_msix_cap_table_size {020} \
   CONFIG.pf1_msix_enabled {true} \
   CONFIG.pf1_msix_enabled_qdma {true} \
   CONFIG.pf1_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf1_pciebar2axibar_1 {0x0000000010000000} \
   CONFIG.pf1_pciebar2axibar_2 {0x0000000000000000} \
   CONFIG.pf1_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf1_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf1_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf1_pciebar2axibar_6 {0x0000000000000000} \
   CONFIG.pf1_rbar_cap_bar0 {0x00000000fff0} \
   CONFIG.pf1_rbar_cap_bar1 {0x000000000000} \
   CONFIG.pf1_rbar_cap_bar2 {0x000000000000} \
   CONFIG.pf1_rbar_cap_bar3 {0x000000000000} \
   CONFIG.pf1_rbar_cap_bar4 {0x000000000000} \
   CONFIG.pf1_rbar_cap_bar5 {0x000000000000} \
   CONFIG.pf1_rbar_num {1} \
   CONFIG.pf1_revision_id {00} \
   CONFIG.pf1_sriov_bar0_64bit {true} \
   CONFIG.pf1_sriov_bar0_enabled {true} \
   CONFIG.pf1_sriov_bar0_prefetchable {true} \
   CONFIG.pf1_sriov_bar0_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar0_size {32} \
   CONFIG.pf1_sriov_bar0_type {DMA} \
   CONFIG.pf1_sriov_bar1_64bit {false} \
   CONFIG.pf1_sriov_bar1_enabled {false} \
   CONFIG.pf1_sriov_bar1_prefetchable {false} \
   CONFIG.pf1_sriov_bar1_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar1_size {4} \
   CONFIG.pf1_sriov_bar1_type {AXI_Bridge_Master} \
   CONFIG.pf1_sriov_bar2_64bit {true} \
   CONFIG.pf1_sriov_bar2_enabled {true} \
   CONFIG.pf1_sriov_bar2_prefetchable {true} \
   CONFIG.pf1_sriov_bar2_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar2_size {4} \
   CONFIG.pf1_sriov_bar2_type {AXI_Lite_Master} \
   CONFIG.pf1_sriov_bar3_64bit {false} \
   CONFIG.pf1_sriov_bar3_enabled {false} \
   CONFIG.pf1_sriov_bar3_prefetchable {false} \
   CONFIG.pf1_sriov_bar3_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar3_size {4} \
   CONFIG.pf1_sriov_bar3_type {AXI_Bridge_Master} \
   CONFIG.pf1_sriov_bar4_64bit {false} \
   CONFIG.pf1_sriov_bar4_enabled {false} \
   CONFIG.pf1_sriov_bar4_prefetchable {false} \
   CONFIG.pf1_sriov_bar4_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar4_size {4} \
   CONFIG.pf1_sriov_bar4_type {AXI_Bridge_Master} \
   CONFIG.pf1_sriov_bar5_64bit {false} \
   CONFIG.pf1_sriov_bar5_enabled {false} \
   CONFIG.pf1_sriov_bar5_prefetchable {false} \
   CONFIG.pf1_sriov_bar5_scale {Kilobytes} \
   CONFIG.pf1_sriov_bar5_size {4} \
   CONFIG.pf1_sriov_bar5_type {AXI_Bridge_Master} \
   CONFIG.pf1_sub_class_interface_menu_qdma {Other_memory_controller} \
   CONFIG.pf1_subsystem_id {0007} \
   CONFIG.pf1_subsystem_vendor_id {10EE} \
   CONFIG.pf1_vendor_id {10EE} \
   CONFIG.pf1_vf_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf1_vf_pciebar2axibar_1 {0x0000000050000000} \
   CONFIG.pf1_vf_pciebar2axibar_2 {0x0000000050000000} \
   CONFIG.pf1_vf_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf1_vf_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf1_vf_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf2_Use_Class_Code_Lookup_Assistant_qdma {false} \
   CONFIG.pf2_bar0_64bit_qdma {true} \
   CONFIG.pf2_bar0_enabled_qdma {true} \
   CONFIG.pf2_bar0_index {0} \
   CONFIG.pf2_bar0_prefetchable_qdma {false} \
   CONFIG.pf2_bar0_scale_qdma {Megabytes} \
   CONFIG.pf2_bar0_size_qdma {64} \
   CONFIG.pf2_bar0_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_bar1_64bit_qdma {false} \
   CONFIG.pf2_bar1_enabled_qdma {false} \
   CONFIG.pf2_bar1_index {7} \
   CONFIG.pf2_bar1_prefetchable_qdma {false} \
   CONFIG.pf2_bar1_scale_qdma {Megabytes} \
   CONFIG.pf2_bar1_size_qdma {128} \
   CONFIG.pf2_bar1_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_bar2_64bit_qdma {true} \
   CONFIG.pf2_bar2_enabled_qdma {true} \
   CONFIG.pf2_bar2_index {7} \
   CONFIG.pf2_bar2_prefetchable_qdma {false} \
   CONFIG.pf2_bar2_scale_qdma {Megabytes} \
   CONFIG.pf2_bar2_size_qdma {512} \
   CONFIG.pf2_bar2_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_bar3_64bit_qdma {false} \
   CONFIG.pf2_bar3_enabled_qdma {false} \
   CONFIG.pf2_bar3_index {7} \
   CONFIG.pf2_bar3_prefetchable_qdma {false} \
   CONFIG.pf2_bar3_scale_qdma {Kilobytes} \
   CONFIG.pf2_bar3_size_qdma {128} \
   CONFIG.pf2_bar3_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_bar4_64bit_qdma {true} \
   CONFIG.pf2_bar4_enabled_qdma {true} \
   CONFIG.pf2_bar4_index {7} \
   CONFIG.pf2_bar4_prefetchable_qdma {false} \
   CONFIG.pf2_bar4_scale_qdma {Megabytes} \
   CONFIG.pf2_bar4_size_qdma {1} \
   CONFIG.pf2_bar4_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_bar5_enabled_qdma {false} \
   CONFIG.pf2_bar5_index {7} \
   CONFIG.pf2_bar5_prefetchable_qdma {false} \
   CONFIG.pf2_bar5_scale_qdma {Kilobytes} \
   CONFIG.pf2_bar5_size_qdma {128} \
   CONFIG.pf2_bar5_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf2_base_class_menu_qdma {Memory_controller} \
   CONFIG.pf2_class_code_base_qdma {05} \
   CONFIG.pf2_class_code_interface_qdma {00} \
   CONFIG.pf2_class_code_qdma {058000} \
   CONFIG.pf2_class_code_sub_qdma {80} \
   CONFIG.pf2_device_id {B234} \
   CONFIG.pf2_expansion_rom_enabled_qdma {false} \
   CONFIG.pf2_expansion_rom_scale_qdma {Kilobytes} \
   CONFIG.pf2_expansion_rom_size_qdma {4} \
   CONFIG.pf2_expansion_rom_type_qdma {Expansion_ROM} \
   CONFIG.pf2_msi_enabled {false} \
   CONFIG.pf2_msix_enabled_qdma {true} \
   CONFIG.pf2_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf2_pciebar2axibar_1 {0x0000000020000000} \
   CONFIG.pf2_pciebar2axibar_2 {0x0000000020000000} \
   CONFIG.pf2_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf2_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf2_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf2_pciebar2axibar_6 {0x0000000000000000} \
   CONFIG.pf2_rbar_cap_bar0 {0x00000000fff0} \
   CONFIG.pf2_rbar_cap_bar1 {0x000000000000} \
   CONFIG.pf2_rbar_cap_bar2 {0x000000000000} \
   CONFIG.pf2_rbar_cap_bar3 {0x000000000000} \
   CONFIG.pf2_rbar_cap_bar4 {0x000000000000} \
   CONFIG.pf2_rbar_cap_bar5 {0x000000000000} \
   CONFIG.pf2_rbar_num {1} \
   CONFIG.pf2_revision_id {00} \
   CONFIG.pf2_sriov_bar0_64bit {true} \
   CONFIG.pf2_sriov_bar0_enabled {true} \
   CONFIG.pf2_sriov_bar0_prefetchable {true} \
   CONFIG.pf2_sriov_bar0_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar0_size {32} \
   CONFIG.pf2_sriov_bar0_type {DMA} \
   CONFIG.pf2_sriov_bar1_64bit {false} \
   CONFIG.pf2_sriov_bar1_enabled {false} \
   CONFIG.pf2_sriov_bar1_prefetchable {false} \
   CONFIG.pf2_sriov_bar1_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar1_size {4} \
   CONFIG.pf2_sriov_bar1_type {AXI_Bridge_Master} \
   CONFIG.pf2_sriov_bar2_64bit {true} \
   CONFIG.pf2_sriov_bar2_enabled {true} \
   CONFIG.pf2_sriov_bar2_prefetchable {true} \
   CONFIG.pf2_sriov_bar2_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar2_size {4} \
   CONFIG.pf2_sriov_bar2_type {AXI_Lite_Master} \
   CONFIG.pf2_sriov_bar3_64bit {false} \
   CONFIG.pf2_sriov_bar3_enabled {false} \
   CONFIG.pf2_sriov_bar3_prefetchable {false} \
   CONFIG.pf2_sriov_bar3_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar3_size {4} \
   CONFIG.pf2_sriov_bar3_type {AXI_Bridge_Master} \
   CONFIG.pf2_sriov_bar4_64bit {false} \
   CONFIG.pf2_sriov_bar4_enabled {false} \
   CONFIG.pf2_sriov_bar4_prefetchable {false} \
   CONFIG.pf2_sriov_bar4_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar4_size {4} \
   CONFIG.pf2_sriov_bar4_type {AXI_Bridge_Master} \
   CONFIG.pf2_sriov_bar5_64bit {false} \
   CONFIG.pf2_sriov_bar5_enabled {false} \
   CONFIG.pf2_sriov_bar5_prefetchable {false} \
   CONFIG.pf2_sriov_bar5_scale {Kilobytes} \
   CONFIG.pf2_sriov_bar5_size {4} \
   CONFIG.pf2_sriov_bar5_type {AXI_Bridge_Master} \
   CONFIG.pf2_sub_class_interface_menu_qdma {Other_memory_controller} \
   CONFIG.pf2_subsystem_id {0007} \
   CONFIG.pf2_subsystem_vendor_id {10EE} \
   CONFIG.pf2_vendor_id {10EE} \
   CONFIG.pf2_vf_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf2_vf_pciebar2axibar_1 {0x0000000060000000} \
   CONFIG.pf2_vf_pciebar2axibar_2 {0x0000000060000000} \
   CONFIG.pf2_vf_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf2_vf_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf2_vf_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf3_Use_Class_Code_Lookup_Assistant_qdma {false} \
   CONFIG.pf3_bar0_64bit_qdma {true} \
   CONFIG.pf3_bar0_enabled_qdma {true} \
   CONFIG.pf3_bar0_index {0} \
   CONFIG.pf3_bar0_prefetchable_qdma {false} \
   CONFIG.pf3_bar0_scale_qdma {Megabytes} \
   CONFIG.pf3_bar0_size_qdma {64} \
   CONFIG.pf3_bar0_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_bar1_64bit_qdma {false} \
   CONFIG.pf3_bar1_enabled_qdma {false} \
   CONFIG.pf3_bar1_index {7} \
   CONFIG.pf3_bar1_prefetchable_qdma {false} \
   CONFIG.pf3_bar1_scale_qdma {Megabytes} \
   CONFIG.pf3_bar1_size_qdma {128} \
   CONFIG.pf3_bar1_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_bar2_64bit_qdma {true} \
   CONFIG.pf3_bar2_enabled_qdma {true} \
   CONFIG.pf3_bar2_index {7} \
   CONFIG.pf3_bar2_prefetchable_qdma {false} \
   CONFIG.pf3_bar2_scale_qdma {Megabytes} \
   CONFIG.pf3_bar2_size_qdma {512} \
   CONFIG.pf3_bar2_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_bar3_64bit_qdma {false} \
   CONFIG.pf3_bar3_enabled_qdma {false} \
   CONFIG.pf3_bar3_index {7} \
   CONFIG.pf3_bar3_prefetchable_qdma {false} \
   CONFIG.pf3_bar3_scale_qdma {Kilobytes} \
   CONFIG.pf3_bar3_size_qdma {128} \
   CONFIG.pf3_bar3_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_bar4_64bit_qdma {true} \
   CONFIG.pf3_bar4_enabled_qdma {true} \
   CONFIG.pf3_bar4_index {7} \
   CONFIG.pf3_bar4_prefetchable_qdma {false} \
   CONFIG.pf3_bar4_scale_qdma {Megabytes} \
   CONFIG.pf3_bar4_size_qdma {1} \
   CONFIG.pf3_bar4_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_bar5_enabled_qdma {false} \
   CONFIG.pf3_bar5_index {7} \
   CONFIG.pf3_bar5_prefetchable_qdma {false} \
   CONFIG.pf3_bar5_scale_qdma {Kilobytes} \
   CONFIG.pf3_bar5_size_qdma {128} \
   CONFIG.pf3_bar5_type_qdma {AXI_Bridge_Master} \
   CONFIG.pf3_base_class_menu_qdma {Memory_controller} \
   CONFIG.pf3_class_code_base_qdma {05} \
   CONFIG.pf3_class_code_interface_qdma {00} \
   CONFIG.pf3_class_code_qdma {058000} \
   CONFIG.pf3_class_code_sub_qdma {80} \
   CONFIG.pf3_device_id {B334} \
   CONFIG.pf3_expansion_rom_enabled_qdma {false} \
   CONFIG.pf3_expansion_rom_scale_qdma {Kilobytes} \
   CONFIG.pf3_expansion_rom_size_qdma {4} \
   CONFIG.pf3_expansion_rom_type_qdma {Expansion_ROM} \
   CONFIG.pf3_msi_enabled {false} \
   CONFIG.pf3_msix_enabled_qdma {true} \
   CONFIG.pf3_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf3_pciebar2axibar_1 {0x0000000030000000} \
   CONFIG.pf3_pciebar2axibar_2 {0x0000000030000000} \
   CONFIG.pf3_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf3_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf3_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pf3_pciebar2axibar_6 {0x0000000000000000} \
   CONFIG.pf3_rbar_cap_bar0 {0x00000000fff0} \
   CONFIG.pf3_rbar_cap_bar1 {0x000000000000} \
   CONFIG.pf3_rbar_cap_bar2 {0x000000000000} \
   CONFIG.pf3_rbar_cap_bar3 {0x000000000000} \
   CONFIG.pf3_rbar_cap_bar4 {0x000000000000} \
   CONFIG.pf3_rbar_cap_bar5 {0x000000000000} \
   CONFIG.pf3_rbar_num {1} \
   CONFIG.pf3_revision_id {00} \
   CONFIG.pf3_sriov_bar0_64bit {true} \
   CONFIG.pf3_sriov_bar0_enabled {true} \
   CONFIG.pf3_sriov_bar0_prefetchable {true} \
   CONFIG.pf3_sriov_bar0_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar0_size {32} \
   CONFIG.pf3_sriov_bar0_type {DMA} \
   CONFIG.pf3_sriov_bar1_64bit {false} \
   CONFIG.pf3_sriov_bar1_enabled {false} \
   CONFIG.pf3_sriov_bar1_prefetchable {false} \
   CONFIG.pf3_sriov_bar1_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar1_size {4} \
   CONFIG.pf3_sriov_bar1_type {AXI_Bridge_Master} \
   CONFIG.pf3_sriov_bar2_64bit {true} \
   CONFIG.pf3_sriov_bar2_enabled {true} \
   CONFIG.pf3_sriov_bar2_prefetchable {true} \
   CONFIG.pf3_sriov_bar2_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar2_size {4} \
   CONFIG.pf3_sriov_bar2_type {AXI_Lite_Master} \
   CONFIG.pf3_sriov_bar3_64bit {false} \
   CONFIG.pf3_sriov_bar3_enabled {false} \
   CONFIG.pf3_sriov_bar3_prefetchable {false} \
   CONFIG.pf3_sriov_bar3_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar3_size {4} \
   CONFIG.pf3_sriov_bar3_type {AXI_Bridge_Master} \
   CONFIG.pf3_sriov_bar4_64bit {false} \
   CONFIG.pf3_sriov_bar4_enabled {false} \
   CONFIG.pf3_sriov_bar4_prefetchable {false} \
   CONFIG.pf3_sriov_bar4_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar4_size {4} \
   CONFIG.pf3_sriov_bar4_type {AXI_Bridge_Master} \
   CONFIG.pf3_sriov_bar5_64bit {false} \
   CONFIG.pf3_sriov_bar5_enabled {false} \
   CONFIG.pf3_sriov_bar5_prefetchable {false} \
   CONFIG.pf3_sriov_bar5_scale {Kilobytes} \
   CONFIG.pf3_sriov_bar5_size {4} \
   CONFIG.pf3_sriov_bar5_type {AXI_Bridge_Master} \
   CONFIG.pf3_sub_class_interface_menu_qdma {Other_memory_controller} \
   CONFIG.pf3_subsystem_id {0007} \
   CONFIG.pf3_subsystem_vendor_id {10EE} \
   CONFIG.pf3_vendor_id {10EE} \
   CONFIG.pf3_vf_pciebar2axibar_0 {0x0000000000000000} \
   CONFIG.pf3_vf_pciebar2axibar_1 {0x0000000070000000} \
   CONFIG.pf3_vf_pciebar2axibar_2 {0x0000000070000000} \
   CONFIG.pf3_vf_pciebar2axibar_3 {0x0000000000000000} \
   CONFIG.pf3_vf_pciebar2axibar_4 {0x0000000000000000} \
   CONFIG.pf3_vf_pciebar2axibar_5 {0x0000000000000000} \
   CONFIG.pfch_cache_depth {16} \
   CONFIG.pipe_line_stage {2} \
   CONFIG.pipe_sim {true} \
   CONFIG.pl_link_cap_max_link_speed {8.0_GT/s} \
   CONFIG.pl_link_cap_max_link_width {X4} \
   CONFIG.plltype {QPLL1} \
   CONFIG.rbar_enable {false} \
   CONFIG.ref_clk_freq {100_MHz} \
   CONFIG.rq_rcfg_en {TRUE} \
   CONFIG.rx_detect {Default} \
   CONFIG.select_quad {GTH_Quad_128} \
   CONFIG.set_finite_credit {false} \
   CONFIG.shell_bridge {false} \
   CONFIG.silicon_rev {Pre-Production} \
   CONFIG.sim_model {NO} \
   CONFIG.soft_nic {false} \
   CONFIG.soft_nic_bridge {false} \
   CONFIG.split_dma {true} \
   CONFIG.sys_reset_polarity {ACTIVE_LOW} \
   CONFIG.tandem_enable_rfsoc {false} \
   CONFIG.testname {mm} \
   CONFIG.timeout0_sel {14} \
   CONFIG.timeout1_sel {15} \
   CONFIG.timeout_mult {3} \
   CONFIG.tl_credits_cd {15} \
   CONFIG.tl_credits_ch {15} \
   CONFIG.tl_pf_enable_reg {1} \
   CONFIG.tl_tx_mux_strict_priority {false} \
   CONFIG.type1_membase_memlimit_enable {Disabled} \
   CONFIG.type1_prefetchable_membase_memlimit {Disabled} \
   CONFIG.use_standard_interfaces {true} \
   CONFIG.usplus_es1_seqnum_bypass {false} \
   CONFIG.usr_irq_exdes {false} \
   CONFIG.usr_irq_xdma_type_interface {false} \
   CONFIG.vcu118_board {false} \
   CONFIG.vcu118_ddr_ex {false} \
   CONFIG.vdm_en {true} \
   CONFIG.vdpa_exdes {false} \
   CONFIG.vendor_id {10EE} \
   CONFIG.virtio_en {false} \
   CONFIG.virtio_exdes {false} \
   CONFIG.virtio_perf_exdes {false} \
   CONFIG.vsec_cap_addr {0xE00} \
   CONFIG.vu9p_board {false} \
   CONFIG.vu9p_tul_ex {false} \
   CONFIG.wrb_coal_loop_fix_disable {false} \
   CONFIG.wrb_coal_max_buf {16} \
   CONFIG.xdma_axi_intf_mm {AXI_Memory_Mapped} \
   CONFIG.xdma_axilite_slave {false} \
   CONFIG.xdma_dsc_bypass {false} \
   CONFIG.xdma_en {true} \
   CONFIG.xdma_non_incremental_exdes {false} \
   CONFIG.xdma_num_usr_irq {16} \
   CONFIG.xdma_pcie_64bit_en {false} \
   CONFIG.xdma_pcie_prefetchable {false} \
   CONFIG.xdma_rnum_chnl {4} \
   CONFIG.xdma_rnum_rids {32} \
   CONFIG.xdma_scale {Megabytes} \
   CONFIG.xdma_size {1} \
   CONFIG.xdma_st_infinite_desc_exdes {false} \
   CONFIG.xdma_sts_ports {false} \
   CONFIG.xdma_wnum_chnl {4} \
   CONFIG.xdma_wnum_rids {32} \
   CONFIG.xlnx_ddr_ex {false} \
 ] $qdma_host_mem

  # Create instance: qdma_host_mem_support
  create_hier_cell_qdma_host_mem_support [current_bd_instance .] qdma_host_mem_support

  # Create instance: smartconnect_1, and set properties
  set smartconnect_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1 ]

  # Create instance: smartconnect_3, and set properties
  set smartconnect_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_3 ]
  set_property -dict [ list \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_3

  # Create instance: vcc_1_bit, and set properties
  set vcc_1_bit [ create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 vcc_1_bit ]

  # Create interface connections
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins emb_mem_axi_bridge_mem/BRAM_PORTA]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTB [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTB] [get_bd_intf_pins emb_mem_axi_bridge_mem/BRAM_PORTB]
  connect_bd_intf_net -intf_net axi_bram_ctrl_1_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_1/BRAM_PORTA] [get_bd_intf_pins emb_mem_dma_mem/BRAM_PORTA]
  connect_bd_intf_net -intf_net axi_bram_ctrl_1_BRAM_PORTB [get_bd_intf_pins axi_bram_ctrl_1/BRAM_PORTB] [get_bd_intf_pins emb_mem_dma_mem/BRAM_PORTB]
  connect_bd_intf_net -intf_net axi_cdma_0_M_AXI [get_bd_intf_pins axi_cdma_0/M_AXI] [get_bd_intf_pins axi_noc_0/S03_AXI]
  connect_bd_intf_net -intf_net axi_cdma_0_M_AXI_SG [get_bd_intf_pins axi_cdma_0/M_AXI_SG] [get_bd_intf_pins axi_noc_0/S04_AXI]
  connect_bd_intf_net -intf_net axi_dbg_hub_0_M00_AXIS [get_bd_intf_pins axi_dbg_hub_0/M00_AXIS] [get_bd_intf_pins qdma_host_mem_support/pcie_dbg_s_axis]
  connect_bd_intf_net -intf_net axi_noc_0_CH0_DDR4_0 [get_bd_intf_ports ddr4_sdram_c0] [get_bd_intf_pins axi_noc_0/CH0_DDR4_0]
  connect_bd_intf_net -intf_net axi_noc_0_CH0_DDR4_1 [get_bd_intf_ports ddr4_sdram_c1] [get_bd_intf_pins axi_noc_0/CH0_DDR4_1]
  connect_bd_intf_net -intf_net axi_noc_0_CH0_DDR4_2 [get_bd_intf_ports ddr4_sdram_c2] [get_bd_intf_pins axi_noc_0/CH0_DDR4_2]
  connect_bd_intf_net -intf_net axi_noc_0_CH0_DDR4_3 [get_bd_intf_ports ddr4_sdram_c3] [get_bd_intf_pins axi_noc_0/CH0_DDR4_3]
  connect_bd_intf_net -intf_net axi_noc_0_M00_AXI [get_bd_intf_pins ai_engine_0/S00_AXI] [get_bd_intf_pins axi_noc_0/M00_AXI]
  connect_bd_intf_net -intf_net axi_noc_0_M01_AXI [get_bd_intf_pins axi_bram_ctrl_0/S_AXI] [get_bd_intf_pins axi_noc_0/M01_AXI]
  connect_bd_intf_net -intf_net axi_noc_1_M00_INI [get_bd_intf_pins axi_noc_0/S00_INI] [get_bd_intf_pins axi_noc_1/M00_INI]
  connect_bd_intf_net -intf_net axi_traffic_gen_0_M_AXI [get_bd_intf_pins axi_traffic_gen_0_read/M_AXI] [get_bd_intf_pins smartconnect_1/S00_AXI]
  connect_bd_intf_net -intf_net axi_traffic_gen_1_M_AXI [get_bd_intf_pins axi_traffic_gen_1_write/M_AXI] [get_bd_intf_pins smartconnect_1/S01_AXI]
  connect_bd_intf_net -intf_net cips_0_FPD_AXI_NOC_0 [get_bd_intf_pins axi_noc_0/S00_AXI] [get_bd_intf_pins cips_0/FPD_AXI_NOC_0]
  connect_bd_intf_net -intf_net cips_0_FPD_AXI_NOC_1 [get_bd_intf_pins axi_noc_0/S01_AXI] [get_bd_intf_pins cips_0/FPD_AXI_NOC_1]
  connect_bd_intf_net -intf_net cips_0_FPD_CCI_NOC_0 [get_bd_intf_pins axi_noc_0/S05_AXI] [get_bd_intf_pins cips_0/FPD_CCI_NOC_0]
  connect_bd_intf_net -intf_net cips_0_FPD_CCI_NOC_1 [get_bd_intf_pins axi_noc_0/S06_AXI] [get_bd_intf_pins cips_0/FPD_CCI_NOC_1]
  connect_bd_intf_net -intf_net cips_0_FPD_CCI_NOC_2 [get_bd_intf_pins axi_noc_0/S07_AXI] [get_bd_intf_pins cips_0/FPD_CCI_NOC_2]
  connect_bd_intf_net -intf_net cips_0_FPD_CCI_NOC_3 [get_bd_intf_pins axi_noc_0/S08_AXI] [get_bd_intf_pins cips_0/FPD_CCI_NOC_3]
  connect_bd_intf_net -intf_net cips_0_M_AXI_FPD [get_bd_intf_pins cips_0/M_AXI_FPD] [get_bd_intf_pins smartconnect_3/S00_AXI]
  connect_bd_intf_net -intf_net cips_0_M_AXI_LPD [get_bd_intf_pins axi_dbg_hub_0/S_AXI] [get_bd_intf_pins cips_0/M_AXI_LPD]
  connect_bd_intf_net -intf_net cips_0_PMC_NOC_AXI_0 [get_bd_intf_pins axi_noc_0/S02_AXI] [get_bd_intf_pins cips_0/PMC_NOC_AXI_0]
  connect_bd_intf_net -intf_net ddr4_c0_sysclk_1 [get_bd_intf_ports ddr4_c0_sysclk] [get_bd_intf_pins axi_noc_0/sys_clk0]
  connect_bd_intf_net -intf_net ddr4_c1_sysclk_1 [get_bd_intf_ports ddr4_c1_sysclk] [get_bd_intf_pins axi_noc_0/sys_clk1]
  connect_bd_intf_net -intf_net ddr4_c2_sysclk_1 [get_bd_intf_ports ddr4_c2_sysclk] [get_bd_intf_pins axi_noc_0/sys_clk2]
  connect_bd_intf_net -intf_net ddr4_c3_sysclk_1 [get_bd_intf_ports ddr4_c3_sysclk] [get_bd_intf_pins axi_noc_0/sys_clk3]
  connect_bd_intf_net -intf_net pcie_refclk_1 [get_bd_intf_ports pcie_refclk] [get_bd_intf_pins qdma_host_mem_support/pcie_refclk]
  connect_bd_intf_net -intf_net qdma_host_mem_M_AXI [get_bd_intf_pins axi_bram_ctrl_1/S_AXI] [get_bd_intf_pins qdma_host_mem/M_AXI]
  connect_bd_intf_net -intf_net qdma_host_mem_M_AXI_BRIDGE [get_bd_intf_pins axi_noc_0/S09_AXI] [get_bd_intf_pins qdma_host_mem/M_AXI_BRIDGE]
connect_bd_intf_net -intf_net [get_bd_intf_nets qdma_host_mem_M_AXI_BRIDGE] [get_bd_intf_pins axis_ila_0/SLOT_1_AXI] [get_bd_intf_pins qdma_host_mem/M_AXI_BRIDGE]
  connect_bd_intf_net -intf_net qdma_host_mem_pcie_cfg_control_if [get_bd_intf_pins qdma_host_mem/pcie_cfg_control_if] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_control]
  connect_bd_intf_net -intf_net qdma_host_mem_pcie_cfg_external_msix_without_msi_if [get_bd_intf_pins qdma_host_mem/pcie_cfg_external_msix_without_msi_if] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_external_msix_without_msi]
  connect_bd_intf_net -intf_net qdma_host_mem_pcie_cfg_interrupt [get_bd_intf_pins qdma_host_mem/pcie_cfg_interrupt] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_interrupt]
  connect_bd_intf_net -intf_net qdma_host_mem_pcie_cfg_mgmt_if [get_bd_intf_pins qdma_host_mem/pcie_cfg_mgmt_if] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_mgmt]
  connect_bd_intf_net -intf_net qdma_host_mem_s_axis_cc [get_bd_intf_pins qdma_host_mem/s_axis_cc] [get_bd_intf_pins qdma_host_mem_support/s_axis_cc]
  connect_bd_intf_net -intf_net qdma_host_mem_s_axis_rq [get_bd_intf_pins qdma_host_mem/s_axis_rq] [get_bd_intf_pins qdma_host_mem_support/s_axis_rq]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets qdma_host_mem_s_axis_rq]
  connect_bd_intf_net -intf_net qdma_host_mem_support_m_axis_cq [get_bd_intf_pins qdma_host_mem/m_axis_cq] [get_bd_intf_pins qdma_host_mem_support/m_axis_cq]
  set_property HDL_ATTRIBUTE.DEBUG {true} [get_bd_intf_nets qdma_host_mem_support_m_axis_cq]
  connect_bd_intf_net -intf_net qdma_host_mem_support_m_axis_rc [get_bd_intf_pins qdma_host_mem/m_axis_rc] [get_bd_intf_pins qdma_host_mem_support/m_axis_rc]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_cfg_fc [get_bd_intf_pins qdma_host_mem/pcie_cfg_fc] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_fc]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_cfg_mesg_rcvd [get_bd_intf_pins qdma_host_mem/pcie_cfg_mesg_rcvd] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_mesg_rcvd]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_cfg_mesg_tx [get_bd_intf_pins qdma_host_mem/pcie_cfg_mesg_tx] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_mesg_tx]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_cfg_status [get_bd_intf_pins qdma_host_mem/pcie_cfg_status_if] [get_bd_intf_pins qdma_host_mem_support/pcie_cfg_status]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_dbg_m_axis [get_bd_intf_pins axi_dbg_hub_0/S00_AXIS] [get_bd_intf_pins qdma_host_mem_support/pcie_dbg_m_axis]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_mgt [get_bd_intf_ports pcie_mgt] [get_bd_intf_pins qdma_host_mem_support/pcie_mgt]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pcie_transmit_fc [get_bd_intf_pins qdma_host_mem/pcie_transmit_fc_if] [get_bd_intf_pins qdma_host_mem_support/pcie_transmit_fc]
  connect_bd_intf_net -intf_net qdma_host_mem_support_pipe_ep [get_bd_intf_ports pipe_ep] [get_bd_intf_pins qdma_host_mem_support/pipe_ep]
  connect_bd_intf_net -intf_net smartconnect_1_M00_AXI [get_bd_intf_pins qdma_host_mem/S_AXI_BRIDGE] [get_bd_intf_pins smartconnect_1/M00_AXI]
connect_bd_intf_net -intf_net [get_bd_intf_nets smartconnect_1_M00_AXI] [get_bd_intf_pins axis_ila_0/SLOT_0_AXI] [get_bd_intf_pins smartconnect_1/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_3_M00_AXI [get_bd_intf_pins axi_cdma_0/S_AXI_LITE] [get_bd_intf_pins smartconnect_3/M00_AXI]

  # Create port connections
  connect_bd_net -net ai_engine_0_s00_axi_aclk [get_bd_pins ai_engine_0/s00_axi_aclk] [get_bd_pins axi_noc_0/aclk1]
  connect_bd_net -net cips_0_fpd_axi_noc_axi0_clk [get_bd_pins axi_noc_0/aclk2] [get_bd_pins cips_0/fpd_axi_noc_axi0_clk]
  connect_bd_net -net cips_0_fpd_axi_noc_axi1_clk [get_bd_pins axi_noc_0/aclk3] [get_bd_pins cips_0/fpd_axi_noc_axi1_clk]
  connect_bd_net -net cips_0_fpd_cci_noc_axi0_clk [get_bd_pins axi_noc_0/aclk5] [get_bd_pins cips_0/fpd_cci_noc_axi0_clk]
  connect_bd_net -net cips_0_fpd_cci_noc_axi1_clk [get_bd_pins axi_noc_0/aclk6] [get_bd_pins cips_0/fpd_cci_noc_axi1_clk]
  connect_bd_net -net cips_0_fpd_cci_noc_axi2_clk [get_bd_pins axi_noc_0/aclk7] [get_bd_pins cips_0/fpd_cci_noc_axi2_clk]
  connect_bd_net -net cips_0_fpd_cci_noc_axi3_clk [get_bd_pins axi_noc_0/aclk8] [get_bd_pins cips_0/fpd_cci_noc_axi3_clk]
  connect_bd_net -net cips_0_pl0_ref_clk [get_bd_pins axi_dbg_hub_0/aclk] [get_bd_pins axis_vio_0/clk] [get_bd_pins cips_0/m_axi_lpd_aclk] [get_bd_pins cips_0/pl0_ref_clk] [get_bd_pins qdma_host_mem_support/pcie_dbg_aclk]
  connect_bd_net -net cips_0_pl_pcie0_resetn [get_bd_pins cips_0/pl_pcie0_resetn] [get_bd_pins qdma_host_mem_support/sys_reset]
  connect_bd_net -net cips_0_pl_pcie1_resetn [get_bd_pins axi_dbg_hub_0/aresetn] [get_bd_pins cips_0/pl_pcie1_resetn] [get_bd_pins qdma_host_mem_support/pcie_dbg_aresetn]
  connect_bd_net -net cips_0_pmc_axi_noc_axi0_clk [get_bd_pins axi_noc_0/aclk4] [get_bd_pins cips_0/pmc_axi_noc_axi0_clk]
  connect_bd_net -net clk_wizard_0_clk_out1 [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
  connect_bd_net -net clk_wizard_0_locked [get_bd_pins clk_wizard_0/locked] [get_bd_pins proc_sys_reset_0/dcm_locked]
  connect_bd_net -net qdma_host_mem_axi_aclk [get_bd_pins axi_bram_ctrl_0/s_axi_aclk] [get_bd_pins axi_bram_ctrl_1/s_axi_aclk] [get_bd_pins axi_cdma_0/m_axi_aclk] [get_bd_pins axi_cdma_0/s_axi_lite_aclk] [get_bd_pins axi_noc_0/aclk0] [get_bd_pins axi_traffic_gen_0_read/s_axi_aclk] [get_bd_pins axi_traffic_gen_1_write/s_axi_aclk] [get_bd_pins axis_ila_0/clk] [get_bd_pins cips_0/m_axi_fpd_aclk] [get_bd_pins clk_wizard_0/clk_in1] [get_bd_pins qdma_host_mem/axi_aclk] [get_bd_pins smartconnect_1/aclk] [get_bd_pins smartconnect_3/aclk]
  connect_bd_net -net qdma_host_mem_axi_aresetn [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn] [get_bd_pins axi_bram_ctrl_1/s_axi_aresetn] [get_bd_pins axi_cdma_0/s_axi_lite_aresetn] [get_bd_pins axi_traffic_gen_0_read/s_axi_aresetn] [get_bd_pins axi_traffic_gen_1_write/s_axi_aresetn] [get_bd_pins axis_ila_0/resetn] [get_bd_pins clk_wizard_0/resetn] [get_bd_pins qdma_host_mem/axi_aresetn] [get_bd_pins smartconnect_1/aresetn] [get_bd_pins smartconnect_3/aresetn]
  connect_bd_net -net qdma_host_mem_support_phy_rdy_out [get_bd_pins qdma_host_mem/phy_rdy_out_sd] [get_bd_pins qdma_host_mem_support/phy_rdy_out]
  connect_bd_net -net qdma_host_mem_support_user_clk [get_bd_pins qdma_host_mem/user_clk_sd] [get_bd_pins qdma_host_mem_support/user_clk]
  connect_bd_net -net qdma_host_mem_support_user_lnk_up [get_bd_pins qdma_host_mem/user_lnk_up_sd] [get_bd_pins qdma_host_mem_support/user_lnk_up]
  connect_bd_net -net qdma_host_mem_support_user_reset [get_bd_pins qdma_host_mem/user_reset_sd] [get_bd_pins qdma_host_mem_support/user_reset]
  connect_bd_net -net vcc_1_bit_dout [get_bd_pins qdma_host_mem/qsts_out_rdy] [get_bd_pins qdma_host_mem/soft_reset_n] [get_bd_pins qdma_host_mem/st_rx_msg_rdy] [get_bd_pins qdma_host_mem/tm_dsc_sts_rdy] [get_bd_pins vcc_1_bit/dout]
  connect_bd_net -net vio_start_pat_gen_0 [get_bd_pins axi_traffic_gen_0_read/core_ext_start] [get_bd_pins axis_vio_0/probe_out0]
  connect_bd_net -net vio_start_pat_gen_1 [get_bd_pins axi_traffic_gen_1_write/core_ext_start] [get_bd_pins axis_vio_0/probe_out1]

  # Create address segments
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs axi_noc_0/S03_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data] [get_bd_addr_segs axi_noc_0/S03_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data_SG] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data_SG] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data_SG] [get_bd_addr_segs axi_noc_0/S04_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces axi_cdma_0/Data_SG] [get_bd_addr_segs axi_noc_0/S04_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x00000000 -range 0x00040000 -target_address_space [get_bd_addr_spaces axi_traffic_gen_0_read/Data] [get_bd_addr_segs qdma_host_mem/S_AXI_BRIDGE/BAR0] -force
  assign_bd_address -offset 0x00000000 -range 0x00040000 -target_address_space [get_bd_addr_spaces axi_traffic_gen_1_write/Data] [get_bd_addr_segs qdma_host_mem/S_AXI_BRIDGE/BAR0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_noc_0/S00_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_noc_0/S00_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_noc_0/S01_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_noc_0/S01_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_noc_0/S05_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_noc_0/S05_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_noc_0/S06_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_noc_0/S06_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_2] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_noc_0/S07_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_noc_0/S07_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_3] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_noc_0/S08_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_noc_0/S08_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0xA4000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces cips_0/M_AXI_FPD] [get_bd_addr_segs axi_cdma_0/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x80000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces cips_0/M_AXI_LPD] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces cips_0/PMC_NOC_AXI_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces cips_0/PMC_NOC_AXI_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces cips_0/PMC_NOC_AXI_0] [get_bd_addr_segs axi_noc_0/S02_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces cips_0/PMC_NOC_AXI_0] [get_bd_addr_segs axi_noc_0/S02_AXI/C0_DDR_LOW1x4] -force
  assign_bd_address -offset 0x00000000 -range 0x00008000 -target_address_space [get_bd_addr_spaces qdma_host_mem/M_AXI] [get_bd_addr_segs axi_bram_ctrl_1/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces qdma_host_mem/M_AXI_BRIDGE] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00020000 -target_address_space [get_bd_addr_spaces qdma_host_mem/M_AXI_BRIDGE] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces qdma_host_mem/M_AXI_BRIDGE] [get_bd_addr_segs axi_noc_0/S09_AXI/C0_DDR_LOW0x4] -force
  assign_bd_address -offset 0x000800000000 -range 0x000380000000 -target_address_space [get_bd_addr_spaces qdma_host_mem/M_AXI_BRIDGE] [get_bd_addr_segs axi_noc_0/S09_AXI/C0_DDR_LOW1x4] -force


  # Restore current instance
  current_bd_instance $oldCurInst

  # Create PFM attributes
  set_property PFM_NAME {xilinx:vck5000:xilinx_vck5000_air:0.0} [get_files [current_bd_design].bd]
  set_property PFM.AXI_PORT {M00_AXI {memport "NOC_MASTER"}} [get_bd_cells /axi_noc_0]
  set_property PFM.AXI_PORT {S00_AXI {memport "S_AXI_NOC" sptag "DDR"} S01_AXI {memport "S_AXI_NOC" sptag "DDR"} S02_AXI {memport "S_AXI_NOC" sptag "DDR"} S03_AXI {memport "S_AXI_NOC" sptag "DDR"} S04_AXI {memport "S_AXI_NOC" sptag "DDR"} S05_AXI {memport "S_AXI_NOC" sptag "DDR"} S06_AXI {memport "S_AXI_NOC" sptag "DDR"} S07_AXI {memport "S_AXI_NOC" sptag "DDR"} S08_AXI {memport "S_AXI_NOC" sptag "DDR"} S09_AXI {memport "S_AXI_NOC" sptag "DDR"} S10_AXI {memport "S_AXI_NOC" sptag "DDR"} S11_AXI {memport "S_AXI_NOC" sptag "DDR"} S12_AXI {memport "S_AXI_NOC" sptag "DDR"} S13_AXI {memport "S_AXI_NOC" sptag "DDR"} S14_AXI {memport "S_AXI_NOC" sptag "DDR"} S15_AXI {memport "S_AXI_NOC" sptag "DDR"} S16_AXI {memport "S_AXI_NOC" sptag "DDR"} S17_AXI {memport "S_AXI_NOC" sptag "DDR"} S18_AXI {memport "S_AXI_NOC" sptag "DDR"} S19_AXI {memport "S_AXI_NOC" sptag "DDR"} S20_AXI {memport "S_AXI_NOC" sptag "DDR"} S21_AXI {memport "S_AXI_NOC" sptag "DDR"} S22_AXI {memport "S_AXI_NOC" sptag "DDR"} S23_AXI {memport "S_AXI_NOC" sptag "DDR"} S24_AXI {memport "S_AXI_NOC" sptag "DDR"} S25_AXI {memport "S_AXI_NOC" sptag "DDR"} S26_AXI {memport "S_AXI_NOC" sptag "DDR"} S27_AXI {memport "S_AXI_NOC" sptag "DDR"}} [get_bd_cells /axi_noc_1]
  set_property PFM.CLOCK {clk_out1 {id "12" is_default "true" proc_sys_reset "/proc_sys_reset_0" status "fixed" freq_hz "125000000"}} [get_bd_cells /clk_wizard_0]
  set_property PFM.AXI_PORT {M01_AXI {memport "M_AXI_GP" sptag "" memory ""} M02_AXI {memport "M_AXI_GP" sptag "" memory ""} M03_AXI {memport "M_AXI_GP" sptag "" memory ""} M04_AXI {memport "M_AXI_GP" sptag "" memory ""} M05_AXI {memport "M_AXI_GP" sptag "" memory ""} M06_AXI {memport "M_AXI_GP" sptag "" memory ""} M07_AXI {memport "M_AXI_GP" sptag "" memory ""} M08_AXI {memport "M_AXI_GP" sptag "" memory ""} M09_AXI {memport "M_AXI_GP" sptag "" memory ""} M10_AXI {memport "M_AXI_GP" sptag "" memory ""} M11_AXI {memport "M_AXI_GP" sptag "" memory ""} M12_AXI {memport "M_AXI_GP" sptag "" memory ""} M13_AXI {memport "M_AXI_GP" sptag "" memory ""} M14_AXI {memport "M_AXI_GP" sptag "" memory ""} M15_AXI {memport "M_AXI_GP" sptag "" memory ""}} [get_bd_cells /smartconnect_1]


  validate_bd_design
  save_bd_design
}
# End of create_root_design()


##################################################################
# MAIN FLOW
##################################################################

create_root_design ""

regenerate_bd_layout
save_bd_design

import_files -fileset constrs_1 -norecurse ./constraints/xilinx_pcie_xdma_ref_board.xdc
#import_files -fileset constrs_1 -norecurse ./constraints/xilinx_qdma_pcie_ep_debug.xdc

import_files -norecurse ./imports/

set_property target_constrs_file ./myproj/project_1.srcs/constrs_1/imports/constraints/xilinx_pcie_xdma_ref_board.xdc [current_fileset -constrset]

set_property generate_synth_checkpoint true [get_files -norecurse *.bd]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set_property pfm_name {xilinx:vck5000:xilinx_vck5000_air:0.0} [get_files -norecurse *.bd]
set_property platform.default_output_type "xclbin" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "true" [current_project]
set_property platform.design_intent.external_host "true" [current_project]
set_property platform.design_intent.datacenter "true" [current_project]
set_property platform.uses_pr "false" [current_project]
set_property platform.synth_constraint_files {./constraints/xilinx_pcie_xdma_ref_board.xdc,EARLY,{synthesis implementation}} [current_project]

generate_target all [get_files ./myproj/project_1.srcs/sources_1/bd/project_1/project_1.bd]

update_compile_order -fileset sources_1

write_hw_platform -hw -force -file ./xilinx_vck5000_air.xsa
