# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

proc numberOfCPUs {} {
    return 10

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
# This is a generated script based on design: project_1
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
set scripts_vivado_version 2021.2
set current_vivado_version [version -short]

if { [string first $scripts_vivado_version $current_vivado_version] == -1 } {
   puts ""
   catch {common::send_gid_msg -ssname BD::TCL -id 2041 -severity "ERROR" "This script was generated using Vivado <$scripts_vivado_version> and is being run in <$current_vivado_version> of Vivado. Please run the script in Vivado <$scripts_vivado_version> then open the design in Vivado <$current_vivado_version>. Upgrade the design by running \"Tools => Report => Report IP Status...\", then run write_bd_tcl to create an updated script."}

   return 1
}

set_param board.repoPaths ../../boards/boards/Xilinx/vck190/production/2.3

################################################################
# START
################################################################

# To test this script, run the following commands from Vivado Tcl console:
# source project_1_script.tcl

# If there is no project opened, this script will create a
# project, but make sure you do not have an existing project
# <./myproj/project_1.xpr> in the current working folder.

set list_projs [get_projects -quiet]
if { $list_projs eq "" } {
   create_project project_1 myproj -part xcvc1902-vsva2197-2MP-e-S
   set_property BOARD_PART xilinx.com:vck190:part0:2.3 [current_project]
}

# BlackParrot IP
set ip_repo_paths [get_property ip_repo_paths [current_fileset]]
lappend ip_repo_paths ./blackparrot_ip
set_property ip_repo_paths $ip_repo_paths [current_fileset]
update_ip_catalog

# BlackParrot memory initialization file
import_files -fileset sources_1 -norecurse ./main.mem
set_property used_in_simulation 0 [get_files main.mem]

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
xilinx.com:ip:versal_cips:3.1\
xilinx.com:ip:axi_noc:1.0\
xilinx.com:ip:ai_engine:2.0\
xilinx.com:ip:axi_bram_ctrl:4.1\
xilinx.com:ip:axi_cdma:4.1\
xilinx.com:ip:axi_dbg_hub:2.0\
xilinx.com:ip:axi_gpio:2.0\
xilinx.com:ip:axi_intc:4.1\
xilinx.com:ip:axi_uartlite:2.0\
xilinx.com:ip:axis_ila:1.1\
amd.com:ip:blackparrot:1.0\
xilinx.com:ip:clk_wizard:1.0\
xilinx.com:ip:mutex:2.1\
xilinx.com:ip:emb_mem_gen:1.0\
xilinx.com:ip:mdm:3.2\
xilinx.com:ip:proc_sys_reset:5.0\
xilinx.com:ip:smartconnect:1.0\
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

# BlackParrot Hierarchical Cell
proc create_blackparrot_hier_cell { parentCell nameHier } {

  variable script_folder

  if { $parentCell eq "" || $nameHier eq "" } {
     catch {common::send_gid_msg -ssname BD::TCL -id 2092 -severity "ERROR" "create_blackparrot_hier_cell() - Empty argument(s)!"}
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
  create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M00_AXI
  create_bd_intf_pin -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S00_AXI


  # Create pins
  create_bd_pin -dir I -type rst core_reset
  create_bd_pin -dir I -type rst resetn
  create_bd_pin -dir I -type clk s00_axi_aclk
  create_bd_pin -dir I -type rst s00_axi_aresetn

  # Create instance: axi_bram_ctrl_0, and set properties
  set axi_bram_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0 ]
  set_property -dict [ list \
   CONFIG.DATA_WIDTH {64} \
   CONFIG.SINGLE_PORT_BRAM {0} \
 ] $axi_bram_ctrl_0

  # Create instance: blackparrot_0, and set properties
  set blackparrot_0 [ create_bd_cell -type ip -vlnv amd.com:ip:blackparrot:1.0 blackparrot_0 ]

  # Create instance: emb_mem_gen_0, and set properties
  set emb_mem_gen_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:emb_mem_gen:1.0 emb_mem_gen_0 ]
  set_property -dict [ list \
   CONFIG.ADDR_WIDTH_A {19} \
   CONFIG.ADDR_WIDTH_B {19} \
   CONFIG.ENABLE_32BIT_ADDRESS {true} \
   CONFIG.ENABLE_BYTE_WRITES_A {true} \
   CONFIG.ENABLE_BYTE_WRITES_B {true} \
   CONFIG.MEMORY_INIT_FILE {main.mem} \
   CONFIG.MEMORY_DEPTH {65536} \
   CONFIG.MEMORY_PRIMITIVE {URAM} \
   CONFIG.MEMORY_SIZE {4194304} \
   CONFIG.MEMORY_TYPE {True_Dual_Port_RAM} \
   CONFIG.READ_DATA_WIDTH_A {64} \
   CONFIG.READ_DATA_WIDTH_B {64} \
   CONFIG.READ_LATENCY_A {2} \
   CONFIG.READ_LATENCY_B {2} \
   CONFIG.USE_MEMORY_BLOCK {Stand_Alone} \
   CONFIG.WRITE_DATA_WIDTH_A {64} \
   CONFIG.WRITE_DATA_WIDTH_B {64} \
   CONFIG.WRITE_MODE_A {NO_CHANGE} \
   CONFIG.WRITE_MODE_B {NO_CHANGE} \
 ] $emb_mem_gen_0

  # Create interface connections
  connect_bd_intf_net -intf_net Conn1 [get_bd_intf_pins S00_AXI] [get_bd_intf_pins blackparrot_0/s00_axi]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins emb_mem_gen_0/BRAM_PORTA]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTB [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTB] [get_bd_intf_pins emb_mem_gen_0/BRAM_PORTB]
  connect_bd_intf_net -intf_net blackparrot_0_m00_axi [get_bd_intf_pins blackparrot_0/m00_axi] [get_bd_intf_pins M00_AXI]
  connect_bd_intf_net -intf_net blackparrot_0_m01_axi [get_bd_intf_pins axi_bram_ctrl_0/S_AXI] [get_bd_intf_pins blackparrot_0/m01_axi]

  # Create port connections
  connect_bd_net -net core_reset_1 [get_bd_pins core_reset] [get_bd_pins blackparrot_0/core_reset]
  connect_bd_net -net resetn_1 [get_bd_pins resetn] [get_bd_pins blackparrot_0/resetn]
  connect_bd_net -net s00_axi_aclk_1 [get_bd_pins s00_axi_aclk] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk] [get_bd_pins blackparrot_0/m00_axi_aclk] [get_bd_pins blackparrot_0/m01_axi_aclk] [get_bd_pins blackparrot_0/s00_axi_aclk] [get_bd_pins smartconnect_0/aclk]
  connect_bd_net -net s00_axi_aresetn_1 [get_bd_pins s00_axi_aresetn] [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn] [get_bd_pins blackparrot_0/m00_axi_aresetn] [get_bd_pins blackparrot_0/m01_axi_aresetn] [get_bd_pins blackparrot_0/s00_axi_aresetn] [get_bd_pins smartconnect_0/aresetn]

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
  set ddr4_dimm1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_dimm1 ]

  set ddr4_dimm1_sma_clk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_dimm1_sma_clk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {200000000} \
   ] $ddr4_dimm1_sma_clk

  set uart2_bank306 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:uart_rtl:1.0 uart2_bank306 ]


  # Create ports

  # Create instance: CIPS_0, and set properties
  set CIPS_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.1 CIPS_0 ]
  set_property -dict [ list \
   CONFIG.CLOCK_MODE {Custom} \
   CONFIG.DDR_MEMORY_MODE {Custom} \
   CONFIG.PS_BOARD_INTERFACE {ps_pmc_fixed_io} \
   CONFIG.PS_PL_CONNECTIVITY_MODE {Custom} \
   CONFIG.PS_PMC_CONFIG {\
     CLOCK_MODE {Custom}\
     DDR_MEMORY_MODE {Custom}\
     DESIGN_MODE {1}\
     PMC_CRP_PL0_REF_CTRL_FREQMHZ {99.999992}\
     PMC_GPIO0_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 0 .. 25}}}\
     PMC_GPIO1_MIO_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 26 .. 51}}}\
     PMC_MIO37 {{AUX_IO 0} {DIRECTION out} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA high}\
{PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}}\
     PMC_OSPI_PERIPHERAL {{ENABLE 0} {IO {PMC_MIO 0 .. 11}} {MODE Single}}\
     PMC_QSPI_COHERENCY {0}\
     PMC_QSPI_FBCLK {{ENABLE 1} {IO {PMC_MIO 6}}}\
     PMC_QSPI_PERIPHERAL_DATA_MODE {x4}\
     PMC_QSPI_PERIPHERAL_ENABLE {1}\
     PMC_QSPI_PERIPHERAL_MODE {Dual Parallel}\
     PMC_REF_CLK_FREQMHZ {33.3333}\
     PMC_SD1 {{CD_ENABLE 1} {CD_IO {PMC_MIO 28}} {POW_ENABLE 1} {POW_IO {PMC_MIO 51}}\
{RESET_ENABLE 0} {RESET_IO {PMC_MIO 12}} {WP_ENABLE 0} {WP_IO {PMC_MIO\
1}}}\
     PMC_SD1_COHERENCY {0}\
     PMC_SD1_DATA_TRANSFER_MODE {8Bit}\
     PMC_SD1_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 26 .. 36}}}\
     PMC_SD1_SLOT_TYPE {SD 3.0}\
     PMC_USE_PMC_NOC_AXI0 {1}\
     PS_BOARD_INTERFACE {ps_pmc_fixed_io}\
     PS_CAN1_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 40 .. 41}}}\
     PS_ENET0_MDIO {{ENABLE 1} {IO {PS_MIO 24 .. 25}}}\
     PS_ENET0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 0 .. 11}}}\
     PS_ENET1_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 12 .. 23}}}\
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
     PS_GPIO_EMIO_PERIPHERAL_ENABLE {1}\
     PS_GPIO_EMIO_WIDTH {2}\
     PS_IRQ_USAGE {{CH0 1} {CH1 0} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0} {CH15\
0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 0} {CH9 0}}\
     PS_MIO19 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO21 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO7 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_MIO9 {{AUX_IO 0} {DIRECTION in} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default}\
{PULL disable} {SCHMITT 0} {SLEW slow} {USAGE Reserved}}\
     PS_NUM_FABRIC_RESETS {1}\
     PS_PCIE_RESET {{ENABLE 1}}\
     PS_PL_CONNECTIVITY_MODE {Custom}\
     PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 42 .. 43}}}\
     PS_USB3_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 13 .. 25}}}\
     PS_USE_BSCAN_USER2 {1}\
     PS_USE_FPD_AXI_NOC0 {1}\
     PS_USE_FPD_AXI_NOC1 {1}\
     PS_USE_FPD_CCI_NOC {1}\
     PS_USE_M_AXI_FPD {1}\
     PS_USE_NOC_LPD_AXI0 {1}\
     PS_USE_PMCPL_CLK0 {1}\
     SMON_ALARMS {Set_Alarms_On}\
     SMON_ENABLE_TEMP_AVERAGING {0}\
     SMON_TEMP_AVERAGING_SAMPLES {0}\
     PS_I2C0_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 46 .. 47}}}\
     PS_I2C1_PERIPHERAL {{ENABLE 1} {IO {PMC_MIO 44 .. 45}}}\
   } \
   CONFIG.PS_PMC_CONFIG_APPLIED {1} \
 ] $CIPS_0

  set_property SELECTED_SIM_MODEL tlm  $CIPS_0

  # Create instance: NOC_0, and set properties
  set NOC_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.0 NOC_0 ]
  set_property -dict [ list \
   CONFIG.CH0_DDR4_0_BOARD_INTERFACE {ddr4_dimm1} \
   CONFIG.CONTROLLERTYPE {DDR4_SDRAM} \
   CONFIG.LOGO_FILE {data/noc_mc.png} \
   CONFIG.MC_BA_WIDTH {2} \
   CONFIG.MC_BG_WIDTH {2} \
   CONFIG.MC_CHAN_REGION0 {DDR_LOW0} \
   CONFIG.MC_CHAN_REGION1 {DDR_LOW1} \
   CONFIG.MC_COMPONENT_WIDTH {x8} \
   CONFIG.MC_CONFIG_NUM {config17} \
   CONFIG.MC_DATAWIDTH {64} \
   CONFIG.MC_DDR4_2T {Disable} \
   CONFIG.MC_F1_LPDDR4_MR1 {0x0000} \
   CONFIG.MC_F1_LPDDR4_MR2 {0x0000} \
   CONFIG.MC_F1_TRCD {13750} \
   CONFIG.MC_F1_TRCDMIN {13750} \
   CONFIG.MC_INPUTCLK0_PERIOD {5000} \
   CONFIG.MC_INPUT_FREQUENCY0 {200.000} \
   CONFIG.MC_INTERLEAVE_SIZE {128} \
   CONFIG.MC_MEMORY_DEVICETYPE {UDIMMs} \
   CONFIG.MC_MEMORY_SPEEDGRADE {DDR4-3200AA(22-22-22)} \
   CONFIG.MC_MEMORY_TIMEPERIOD0 {625} \
   CONFIG.MC_NO_CHANNELS {Single} \
   CONFIG.MC_PRE_DEF_ADDR_MAP_SEL {ROW_COLUMN_BANK} \
   CONFIG.MC_RANK {1} \
   CONFIG.MC_ROWADDRESSWIDTH {16} \
   CONFIG.MC_TRC {45750} \
   CONFIG.MC_TRCD {13750} \
   CONFIG.MC_TRCDMIN {13750} \
   CONFIG.MC_TRCMIN {45750} \
   CONFIG.MC_TRP {13750} \
   CONFIG.MC_TRPMIN {13750} \
   CONFIG.NUM_CLKS {12} \
   CONFIG.NUM_MC {1} \
   CONFIG.NUM_MCP {4} \
   CONFIG.NUM_MI {5} \
   CONFIG.NUM_NMI {0} \
   CONFIG.NUM_NSI {1} \
   CONFIG.NUM_SI {11} \
   CONFIG.sys_clk0_BOARD_INTERFACE {ddr4_dimm1_sma_clk} \
 ] $NOC_0

  set_property SELECTED_SIM_MODEL tlm  $NOC_0

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.REGION {0} \
   CONFIG.CATEGORY {aie} \
 ] [get_bd_intf_pins /NOC_0/M00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {64} \
   CONFIG.APERTURES {{0x201_0000_0000 1G}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/M01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.APERTURES {{0x202_4000_0000 1G}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/M02_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {64} \
   CONFIG.APERTURES {{0x800_0000_0000 8T}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/M03_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.APERTURES {{0x202_0000_0000 1G}} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/M04_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S00_AXI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {load} \
   CONFIG.CONNECTIONS {MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
 ] [get_bd_intf_pins /NOC_0/S00_INI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S02_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /NOC_0/S03_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /NOC_0/S04_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /NOC_0/S05_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_rpu} \
 ] [get_bd_intf_pins /NOC_0/S06_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /NOC_0/S07_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {64} \
   CONFIG.CONNECTIONS {M03_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M04_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M03_AXI:0x0:M04_AXI:0x80:M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S08_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S09_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {32} \
   CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M01_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M02_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} M00_AXI { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {M01_AXI:0x1c0:M02_AXI:0x100:M00_AXI:0x200} \
   CONFIG.CATEGORY {pl} \
 ] [get_bd_intf_pins /NOC_0/S10_AXI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M01_AXI:M02_AXI:M03_AXI:M04_AXI:S08_AXI:S09_AXI:S10_AXI} \
 ] [get_bd_pins /NOC_0/aclk0]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
 ] [get_bd_pins /NOC_0/aclk1]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S01_AXI} \
 ] [get_bd_pins /NOC_0/aclk2]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S02_AXI} \
 ] [get_bd_pins /NOC_0/aclk3]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S03_AXI} \
 ] [get_bd_pins /NOC_0/aclk4]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S04_AXI} \
 ] [get_bd_pins /NOC_0/aclk5]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S05_AXI} \
 ] [get_bd_pins /NOC_0/aclk6]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S06_AXI} \
 ] [get_bd_pins /NOC_0/aclk7]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S07_AXI} \
 ] [get_bd_pins /NOC_0/aclk8]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {} \
 ] [get_bd_pins /NOC_0/aclk9]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M00_AXI} \
 ] [get_bd_pins /NOC_0/aclk10]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {} \
 ] [get_bd_pins /NOC_0/aclk11]

  # Create instance: ai_engine_0, and set properties
  set ai_engine_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:ai_engine:2.0 ai_engine_0 ]

  set_property -dict [ list \
   CONFIG.CATEGORY {NOC} \
 ] [get_bd_intf_pins /ai_engine_0/S00_AXI]

  # Create instance: axi_bram_ctrl_0, and set properties
  set axi_bram_ctrl_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_bram_ctrl:4.1 axi_bram_ctrl_0 ]
  set_property -dict [ list \
   CONFIG.DATA_WIDTH {64} \
 ] $axi_bram_ctrl_0

  # Create instance: axi_cdma_1, and set properties
  set axi_cdma_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_cdma:4.1 axi_cdma_1 ]
  set_property -dict [ list \
   CONFIG.C_ADDR_WIDTH {64} \
   CONFIG.C_INCLUDE_SG {1} \
   CONFIG.C_M_AXI_MAX_BURST_LEN {256} \
 ] $axi_cdma_1

  # Create instance: axi_dbg_hub_0, and set properties
  set axi_dbg_hub_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dbg_hub:2.0 axi_dbg_hub_0 ]
  set_property -dict [ list \
   CONFIG.C_NUM_DEBUG_CORES {0} \
 ] $axi_dbg_hub_0

  # Create instance: axi_gpio_0, and set properties
  set axi_gpio_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0 ]
  set_property -dict [ list \
   CONFIG.C_ALL_INPUTS {0} \
   CONFIG.C_ALL_OUTPUTS {1} \
   CONFIG.C_ALL_OUTPUTS_2 {1} \
   CONFIG.C_DOUT_DEFAULT {0x00000001} \
   CONFIG.C_DOUT_DEFAULT_2 {0x00000001} \
   CONFIG.C_GPIO2_WIDTH {1} \
   CONFIG.C_GPIO_WIDTH {1} \
   CONFIG.C_IS_DUAL {1} \
 ] $axi_gpio_0

  # Create instance: axi_intc_0, and set properties
  set axi_intc_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc:4.1 axi_intc_0 ]
  set_property -dict [ list \
   CONFIG.C_ASYNC_INTR {0xFFFFFFFF} \
   CONFIG.C_IRQ_CONNECTION {1} \
 ] $axi_intc_0

  # Create instance: axi_noc_kernel0, and set properties
  set axi_noc_kernel0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.0 axi_noc_kernel0 ]
  set_property -dict [ list \
   CONFIG.NUM_CLKS {0} \
   CONFIG.NUM_MI {0} \
   CONFIG.NUM_NMI {1} \
   CONFIG.NUM_SI {0} \
 ] $axi_noc_kernel0

  set_property SELECTED_SIM_MODEL tlm  $axi_noc_kernel0

  set_property -dict [ list \
   CONFIG.APERTURES {{0x201_4000_0000 1G}} \
 ] [get_bd_intf_pins /axi_noc_kernel0/M00_INI]

  # Create instance: axi_uartlite_0, and set properties
  set axi_uartlite_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_uartlite:2.0 axi_uartlite_0 ]
  set_property -dict [ list \
   CONFIG.C_BAUDRATE {115200} \
   CONFIG.UARTLITE_BOARD_INTERFACE {uart2_bank306} \
   CONFIG.USE_BOARD_FLOW {true} \
 ] $axi_uartlite_0

  # Create instance: blackparrot_hier_0
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_0

  # Create instance: blackparrot_hier_1
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_1

  # Create instance: blackparrot_hier_2
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_2

  # Create instance: blackparrot_hier_3
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_3

  # Create instance: blackparrot_hier_4
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_4

  # Create instance: blackparrot_hier_5
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_5

  # Create instance: blackparrot_hier_6
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_6

  # Create instance: blackparrot_hier_7
  create_blackparrot_hier_cell [current_bd_instance .] blackparrot_hier_7

  # configure ID for each BP core
  set_property CONFIG.C_DEVICE_ID {0} [get_bd_cells blackparrot_hier_0/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {1} [get_bd_cells blackparrot_hier_1/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {2} [get_bd_cells blackparrot_hier_2/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {3} [get_bd_cells blackparrot_hier_3/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {4} [get_bd_cells blackparrot_hier_4/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {5} [get_bd_cells blackparrot_hier_5/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {6} [get_bd_cells blackparrot_hier_6/blackparrot_0]
  set_property CONFIG.C_DEVICE_ID {7} [get_bd_cells blackparrot_hier_7/blackparrot_0]

  # configure AXI offset for each BP core
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80000000000} [get_bd_cells blackparrot_hier_0/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80100000000} [get_bd_cells blackparrot_hier_1/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80200000000} [get_bd_cells blackparrot_hier_2/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80300000000} [get_bd_cells blackparrot_hier_3/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80400000000} [get_bd_cells blackparrot_hier_4/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80500000000} [get_bd_cells blackparrot_hier_5/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80600000000} [get_bd_cells blackparrot_hier_6/blackparrot_0]
  set_property CONFIG.C_S00_AXI_BASEADDR {0x80700000000} [get_bd_cells blackparrot_hier_7/blackparrot_0]

  # reset memory init file for the platform using a project relative path
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_0/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_1/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_2/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_3/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_4/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_5/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_6/emb_mem_gen_0]
  set_property CONFIG.MEMORY_INIT_FILE [get_files main.mem] [get_bd_cells blackparrot_hier_7/emb_mem_gen_0]

  # Create instance: clk_wizard_0, and set properties
  set clk_wizard_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard:1.0 clk_wizard_0 ]
  set_property -dict [ list \
   CONFIG.CLKOUT2_DIVIDE {20.000000} \
   CONFIG.CLKOUT3_DIVIDE {10.000000} \
   CONFIG.CLKOUT4_DIVIDE {15.000000} \
   CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
   CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
   CONFIG.CLKOUT_GROUPING {Auto,Auto,Auto,Auto,Auto,Auto,Auto} \
   CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
   CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
   CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
   CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {100.000,150,300,200,100.000,100.000,100.000} \
   CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
   CONFIG.CLKOUT_USED {true,true,true,true,false,false,false} \
   CONFIG.JITTER_SEL {Min_O_Jitter} \
   CONFIG.RESET_TYPE {ACTIVE_LOW} \
   CONFIG.USE_LOCKED {true} \
   CONFIG.USE_PHASE_ALIGNMENT {true} \
   CONFIG.USE_RESET {true} \
 ] $clk_wizard_0

  # Create instance: emb_mem_gen_0, and set properties
  set emb_mem_gen_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:emb_mem_gen:1.0 emb_mem_gen_0 ]
  set_property -dict [ list \
   CONFIG.MEMORY_TYPE {True_Dual_Port_RAM} \
 ] $emb_mem_gen_0

  # Create instance: mdm_0, and set properties
  set mdm_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:mdm:3.2 mdm_0 ]
  set_property -dict [ list \
   CONFIG.C_ADDR_SIZE {64} \
   CONFIG.C_MB_DBG_PORTS {0} \
   CONFIG.C_M_AXI_ADDR_WIDTH {64} \
   CONFIG.C_USE_BSCAN {2} \
 ] $mdm_0

  # Create instance: mutex_0, and set properties
  set mutex_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:mutex:2.1 mutex_0 ]
  set_property -dict [ list \
   CONFIG.C_NUM_AXI {1} \
   CONFIG.C_NUM_MUTEX {32} \
 ] $mutex_0

  # Create instance: proc_sys_reset_0, and set properties
  set proc_sys_reset_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0 ]

  # Create instance: proc_sys_reset_1, and set properties
  set proc_sys_reset_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_1 ]

  # Create instance: proc_sys_reset_2, and set properties
  set proc_sys_reset_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_2 ]

  # Create instance: smartconnect_0, and set properties
  set smartconnect_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0 ]
  set_property -dict [ list \
   CONFIG.NUM_MI {3} \
   CONFIG.NUM_SI {8} \
 ] $smartconnect_0

  # Create instance: smartconnect_1, and set properties
  set smartconnect_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1 ]
  set_property -dict [ list \
   CONFIG.NUM_CLKS {1} \
   CONFIG.NUM_MI {1} \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_1

  # Create instance: smartconnect_2, and set properties
  set smartconnect_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_2 ]
  set_property -dict [ list \
   CONFIG.NUM_MI {9} \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_2

  # Create instance: smartconnect_3, and set properties
  set smartconnect_3 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_3 ]
  set_property -dict [ list \
   CONFIG.NUM_SI {1} \
 ] $smartconnect_3

  # Create interface connections
  connect_bd_intf_net -intf_net CIPS_0_BSCAN_USER2 [get_bd_intf_pins CIPS_0/BSCAN_USER2] [get_bd_intf_pins mdm_0/BSCAN]
  connect_bd_intf_net -intf_net CIPS_0_IF_PMC_NOC_AXI_0 [get_bd_intf_pins CIPS_0/PMC_NOC_AXI_0] [get_bd_intf_pins NOC_0/S07_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_0 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_0] [get_bd_intf_pins NOC_0/S00_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_1 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_1] [get_bd_intf_pins NOC_0/S01_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_2 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_2] [get_bd_intf_pins NOC_0/S02_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_3 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_3] [get_bd_intf_pins NOC_0/S03_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_0 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_0] [get_bd_intf_pins NOC_0/S04_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_1 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_1] [get_bd_intf_pins NOC_0/S05_AXI]
  connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_RPU_0 [get_bd_intf_pins CIPS_0/LPD_AXI_NOC_0] [get_bd_intf_pins NOC_0/S06_AXI]
  connect_bd_intf_net -intf_net CIPS_0_M_AXI_GP0 [get_bd_intf_pins CIPS_0/M_AXI_FPD] [get_bd_intf_pins smartconnect_1/S00_AXI]
  connect_bd_intf_net -intf_net NOC_0_CH0_DDR4_0 [get_bd_intf_ports ddr4_dimm1] [get_bd_intf_pins NOC_0/CH0_DDR4_0]
  connect_bd_intf_net -intf_net NOC_0_M00_AXI [get_bd_intf_pins NOC_0/M00_AXI] [get_bd_intf_pins ai_engine_0/S00_AXI]
  connect_bd_intf_net -intf_net NOC_0_M01_AXI [get_bd_intf_pins NOC_0/M01_AXI] [get_bd_intf_pins axi_bram_ctrl_0/S_AXI]
  connect_bd_intf_net -intf_net NOC_0_M02_AXI [get_bd_intf_pins NOC_0/M02_AXI] [get_bd_intf_pins axi_dbg_hub_0/S_AXI]
  connect_bd_intf_net -intf_net NOC_0_M03_AXI [get_bd_intf_pins NOC_0/M03_AXI] [get_bd_intf_pins smartconnect_2/S00_AXI]
  connect_bd_intf_net -intf_net NOC_0_M04_AXI [get_bd_intf_pins NOC_0/M04_AXI] [get_bd_intf_pins smartconnect_3/S00_AXI]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTA [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTA] [get_bd_intf_pins emb_mem_gen_0/BRAM_PORTA]
  connect_bd_intf_net -intf_net axi_bram_ctrl_0_BRAM_PORTB [get_bd_intf_pins axi_bram_ctrl_0/BRAM_PORTB] [get_bd_intf_pins emb_mem_gen_0/BRAM_PORTB]
  connect_bd_intf_net -intf_net axi_cdma_1_M_AXI [get_bd_intf_pins NOC_0/S09_AXI] [get_bd_intf_pins axi_cdma_1/M_AXI]
  connect_bd_intf_net -intf_net axi_cdma_1_M_AXI_SG [get_bd_intf_pins NOC_0/S10_AXI] [get_bd_intf_pins axi_cdma_1/M_AXI_SG]
  connect_bd_intf_net -intf_net axi_noc_kernel0_M00_INI [get_bd_intf_pins NOC_0/S00_INI] [get_bd_intf_pins axi_noc_kernel0/M00_INI]
  connect_bd_intf_net -intf_net axi_uartlite_0_UART [get_bd_intf_ports uart2_bank306] [get_bd_intf_pins axi_uartlite_0/UART]
  connect_bd_intf_net -intf_net blackparrot_hier_0_M00_AXI [get_bd_intf_pins blackparrot_hier_0/M00_AXI] [get_bd_intf_pins smartconnect_0/S00_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_1_M00_AXI [get_bd_intf_pins blackparrot_hier_1/M00_AXI] [get_bd_intf_pins smartconnect_0/S01_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_2_M00_AXI [get_bd_intf_pins blackparrot_hier_2/M00_AXI] [get_bd_intf_pins smartconnect_0/S02_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_3_M00_AXI [get_bd_intf_pins blackparrot_hier_3/M00_AXI] [get_bd_intf_pins smartconnect_0/S03_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_4_M00_AXI [get_bd_intf_pins blackparrot_hier_4/M00_AXI] [get_bd_intf_pins smartconnect_0/S04_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_5_M00_AXI [get_bd_intf_pins blackparrot_hier_5/M00_AXI] [get_bd_intf_pins smartconnect_0/S05_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_6_M00_AXI [get_bd_intf_pins blackparrot_hier_6/M00_AXI] [get_bd_intf_pins smartconnect_0/S06_AXI]
  connect_bd_intf_net -intf_net blackparrot_hier_7_M00_AXI [get_bd_intf_pins blackparrot_hier_7/M00_AXI] [get_bd_intf_pins smartconnect_0/S07_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M00_AXI [get_bd_intf_pins NOC_0/S08_AXI] [get_bd_intf_pins smartconnect_0/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M01_AXI [get_bd_intf_pins axi_uartlite_0/S_AXI] [get_bd_intf_pins smartconnect_0/M01_AXI]
  connect_bd_intf_net -intf_net smartconnect_0_M02_AXI [get_bd_intf_pins axi_cdma_1/S_AXI_LITE] [get_bd_intf_pins smartconnect_0/M02_AXI]
  connect_bd_intf_net -intf_net smartconnect_1_M00_AXI [get_bd_intf_pins axi_intc_0/s_axi] [get_bd_intf_pins smartconnect_1/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M00_AXI [get_bd_intf_pins axi_gpio_0/S_AXI] [get_bd_intf_pins smartconnect_2/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M01_AXI [get_bd_intf_pins blackparrot_hier_0/s00_axi] [get_bd_intf_pins smartconnect_2/M01_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M02_AXI [get_bd_intf_pins blackparrot_hier_1/s00_axi] [get_bd_intf_pins smartconnect_2/M02_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M03_AXI [get_bd_intf_pins blackparrot_hier_2/s00_axi] [get_bd_intf_pins smartconnect_2/M03_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M04_AXI [get_bd_intf_pins blackparrot_hier_3/s00_axi] [get_bd_intf_pins smartconnect_2/M04_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M05_AXI [get_bd_intf_pins blackparrot_hier_4/s00_axi] [get_bd_intf_pins smartconnect_2/M05_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M06_AXI [get_bd_intf_pins blackparrot_hier_5/s00_axi] [get_bd_intf_pins smartconnect_2/M06_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M07_AXI [get_bd_intf_pins blackparrot_hier_6/s00_axi] [get_bd_intf_pins smartconnect_2/M07_AXI]
  connect_bd_intf_net -intf_net smartconnect_3_M00_AXI [get_bd_intf_pins mutex_0/S0_AXI] [get_bd_intf_pins smartconnect_3/M00_AXI]
  connect_bd_intf_net -intf_net smartconnect_2_M08_AXI [get_bd_intf_pins blackparrot_hier_7/s00_axi] [get_bd_intf_pins smartconnect_2/M08_AXI]
  connect_bd_intf_net -intf_net sys_clk0_0_1 [get_bd_intf_ports ddr4_dimm1_sma_clk] [get_bd_intf_pins NOC_0/sys_clk0]

  # Create port connections
  connect_bd_net -net CIPS_0_pl_clk0 [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins clk_wizard_0/clk_in1]
  connect_bd_net -net CIPS_0_pl_resetn1 [get_bd_pins CIPS_0/pl0_resetn] [get_bd_pins clk_wizard_0/resetn] [get_bd_pins proc_sys_reset_0/ext_reset_in] [get_bd_pins proc_sys_reset_1/ext_reset_in] [get_bd_pins proc_sys_reset_2/ext_reset_in]
  connect_bd_net -net CIPS_0_ps_pmc_noc_axi0_clk [get_bd_pins CIPS_0/pmc_axi_noc_axi0_clk] [get_bd_pins NOC_0/aclk8]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi0_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi0_clk] [get_bd_pins NOC_0/aclk1]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi1_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi1_clk] [get_bd_pins NOC_0/aclk2]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi2_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi2_clk] [get_bd_pins NOC_0/aclk3]
  connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi3_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi3_clk] [get_bd_pins NOC_0/aclk4]
  connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi0_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi0_clk] [get_bd_pins NOC_0/aclk5]
  connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi1_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi1_clk] [get_bd_pins NOC_0/aclk6]
  connect_bd_net -net CIPS_0_ps_ps_noc_rpu_axi0_clk [get_bd_pins CIPS_0/lpd_axi_noc_clk] [get_bd_pins NOC_0/aclk7]
  connect_bd_net -net ai_engine_0_s00_axi_aclk [get_bd_pins NOC_0/aclk10] [get_bd_pins ai_engine_0/s00_axi_aclk]
  connect_bd_net -net axi_gpio_0_gpio2_io_o [get_bd_pins axi_gpio_0/gpio2_io_o] [get_bd_pins proc_sys_reset_0/aux_reset_in]
  connect_bd_net -net axi_gpio_0_gpio_io_o [get_bd_pins axi_gpio_0/gpio_io_o] [get_bd_pins blackparrot_hier_0/resetn] [get_bd_pins blackparrot_hier_1/resetn] [get_bd_pins blackparrot_hier_2/resetn] [get_bd_pins blackparrot_hier_3/resetn] [get_bd_pins blackparrot_hier_4/resetn] [get_bd_pins blackparrot_hier_5/resetn] [get_bd_pins blackparrot_hier_6/resetn] [get_bd_pins blackparrot_hier_7/resetn]
  connect_bd_net -net axi_intc_0_irq [get_bd_pins CIPS_0/pl_ps_irq0] [get_bd_pins axi_intc_0/irq]
  connect_bd_net -net clk_wizard_0_clk_out1 [get_bd_pins CIPS_0/m_axi_fpd_aclk] [get_bd_pins NOC_0/aclk0] [get_bd_pins NOC_0/aclk9] [get_bd_pins axi_bram_ctrl_0/s_axi_aclk] [get_bd_pins axi_cdma_1/m_axi_aclk] [get_bd_pins axi_cdma_1/s_axi_lite_aclk] [get_bd_pins axi_dbg_hub_0/aclk] [get_bd_pins axi_gpio_0/s_axi_aclk] [get_bd_pins axi_intc_0/s_axi_aclk] [get_bd_pins axi_uartlite_0/s_axi_aclk] [get_bd_pins blackparrot_hier_0/s00_axi_aclk] [get_bd_pins blackparrot_hier_1/s00_axi_aclk] [get_bd_pins blackparrot_hier_2/s00_axi_aclk] [get_bd_pins blackparrot_hier_3/s00_axi_aclk] [get_bd_pins blackparrot_hier_4/s00_axi_aclk] [get_bd_pins blackparrot_hier_5/s00_axi_aclk] [get_bd_pins blackparrot_hier_6/s00_axi_aclk] [get_bd_pins blackparrot_hier_7/s00_axi_aclk] [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins mutex_0/S0_AXI_ACLK] [get_bd_pins proc_sys_reset_0/slowest_sync_clk] [get_bd_pins smartconnect_0/aclk] [get_bd_pins smartconnect_1/aclk] [get_bd_pins smartconnect_2/aclk] [get_bd_pins smartconnect_3/aclk]
  connect_bd_net -net clk_wizard_0_clk_out2 [get_bd_pins clk_wizard_0/clk_out2] [get_bd_pins proc_sys_reset_1/slowest_sync_clk]
  connect_bd_net -net clk_wizard_0_clk_out3 [get_bd_pins clk_wizard_0/clk_out3] [get_bd_pins proc_sys_reset_2/slowest_sync_clk]
  connect_bd_net -net clk_wizard_0_clk_out4 [get_bd_pins NOC_0/aclk11] [get_bd_pins clk_wizard_0/clk_out4]
  connect_bd_net -net clk_wizard_0_locked [get_bd_pins clk_wizard_0/locked] [get_bd_pins proc_sys_reset_0/dcm_locked] [get_bd_pins proc_sys_reset_1/dcm_locked] [get_bd_pins proc_sys_reset_2/dcm_locked]
  connect_bd_net -net mdm_0_Debug_SYS_Rst [get_bd_pins mdm_0/Debug_SYS_Rst] [get_bd_pins proc_sys_reset_0/mb_debug_sys_rst]
  connect_bd_net -net proc_sys_reset_0_mb_reset [get_bd_pins blackparrot_hier_0/core_reset] [get_bd_pins blackparrot_hier_1/core_reset] [get_bd_pins blackparrot_hier_2/core_reset] [get_bd_pins blackparrot_hier_3/core_reset] [get_bd_pins blackparrot_hier_4/core_reset] [get_bd_pins blackparrot_hier_5/core_reset] [get_bd_pins blackparrot_hier_6/core_reset] [get_bd_pins blackparrot_hier_7/core_reset] [get_bd_pins proc_sys_reset_0/mb_reset]
  connect_bd_net -net proc_sys_reset_0_peripheral_aresetn [get_bd_pins axi_bram_ctrl_0/s_axi_aresetn] [get_bd_pins axi_cdma_1/s_axi_lite_aresetn] [get_bd_pins axi_dbg_hub_0/aresetn] [get_bd_pins axi_gpio_0/s_axi_aresetn] [get_bd_pins axi_intc_0/s_axi_aresetn] [get_bd_pins axi_uartlite_0/s_axi_aresetn] [get_bd_pins blackparrot_hier_0/s00_axi_aresetn] [get_bd_pins blackparrot_hier_1/s00_axi_aresetn] [get_bd_pins blackparrot_hier_2/s00_axi_aresetn] [get_bd_pins blackparrot_hier_3/s00_axi_aresetn] [get_bd_pins blackparrot_hier_4/s00_axi_aresetn] [get_bd_pins blackparrot_hier_5/s00_axi_aresetn] [get_bd_pins blackparrot_hier_6/s00_axi_aresetn] [get_bd_pins blackparrot_hier_7/s00_axi_aresetn] [get_bd_pins mutex_0/S0_AXI_ARESETN] [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn] [get_bd_pins smartconnect_1/aresetn] [get_bd_pins smartconnect_2/aresetn] [get_bd_pins smartconnect_3/aresetn]

  # Create address segments
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs NOC_0/S00_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs NOC_0/S04_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs NOC_0/S00_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs NOC_0/S04_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs NOC_0/S01_AXI/C1_DDR_LOW0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs NOC_0/S05_AXI/C1_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs NOC_0/S01_AXI/C1_DDR_LOW1] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs NOC_0/S05_AXI/C1_DDR_LOW1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs NOC_0/S02_AXI/C2_DDR_LOW0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs NOC_0/S06_AXI/C2_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs NOC_0/S06_AXI/C2_DDR_LOW1] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs NOC_0/S02_AXI/C2_DDR_LOW1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs NOC_0/S07_AXI/C3_DDR_LOW0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs NOC_0/S03_AXI/C3_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs NOC_0/S07_AXI/C3_DDR_LOW1] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs NOC_0/S03_AXI/C3_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0xA4000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/M_AXI_FPD] [get_bd_addr_segs axi_intc_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data] [get_bd_addr_segs NOC_0/S09_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data_SG] [get_bd_addr_segs NOC_0/S10_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data] [get_bd_addr_segs NOC_0/S09_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data_SG] [get_bd_addr_segs NOC_0/S10_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data_SG] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data_SG] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_0/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_1/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_2/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_3/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_4/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_5/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_6/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x00000000 -range 0x40000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs NOC_0/S08_AXI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x020100000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x80000000 -range 0x00080000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m01_axi] [get_bd_addr_segs blackparrot_hier_7/axi_bram_ctrl_0/S_AXI/Mem0] -force
  assign_bd_address -offset 0x44A00000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs axi_cdma_1/S_AXI_LITE/Reg] -force
  assign_bd_address -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0] -force
  assign_bd_address -offset 0x090000000000 -range 0x00001000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs axi_gpio_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x40600000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs axi_uartlite_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020200000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs mutex_0/S0_AXI/Reg] -force
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_0/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_1/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_2/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_3/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_4/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_5/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_6/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080700000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_7/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_0/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080100000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_1/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080200000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_2/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080300000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_3/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080400000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_4/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080500000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_5/blackparrot_0/s00_axi/mem]
  assign_bd_address -offset 0x080600000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces blackparrot_hier_7/blackparrot_0/m00_axi] [get_bd_addr_segs blackparrot_hier_6/blackparrot_0/s00_axi/mem]

  # Exclude Address Segments
  exclude_bd_addr_seg -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0]
  exclude_bd_addr_seg -offset 0x020240000000 -range 0x00200000 -target_address_space [get_bd_addr_spaces axi_cdma_1/Data_SG] [get_bd_addr_segs axi_dbg_hub_0/S_AXI_DBG_HUB/Mem0]

  # Restore current instance
  current_bd_instance $oldCurInst

  # Create PFM attributes
  set_property PFM_NAME {xilinx:vck190:xilinx_vck190_air:1.0} [get_files [current_bd_design].bd]
  set_property PFM.AXI_PORT {S11_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S12_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S13_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S14_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S15_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S16_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S17_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S18_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S19_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S20_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S21_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S22_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S23_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S24_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S25_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"} S26_AXI {memport "S_AXI_NOC" sptag "DDR" memory "" is_range "true"}} [get_bd_cells /NOC_0]
  set_property PFM.IRQ {intr {id 0 range 32}} [get_bd_cells /axi_intc_0]
  set_property PFM.CLOCK {clk_out1 {id "1" is_default "false" proc_sys_reset "proc_sys_reset_0" status "fixed"} clk_out2 {id "0" is_default "true" proc_sys_reset "/proc_sys_reset_1" status "fixed"} clk_out3 {id "2" is_default "false" proc_sys_reset "/proc_sys_reset_2" status "fixed"}} [get_bd_cells /clk_wizard_0]
  set_property PFM.AXI_PORT {M01_AXI {memport "M_AXI_GP" sptag "" memory "" is_range "true"} M02_AXI {memport "M_AXI_GP" sptag "" memory "" is_range "true"} M03_AXI {memport "M_AXI_GP" sptag "" memory "" is_range "true"} M04_AXI {memport "M_AXI_GP" sptag "" memory "" is_range "true"}} [get_bd_cells /smartconnect_1]


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

set_property generate_synth_checkpoint true [get_files -norecurse *.bd]
make_wrapper -files [get_files ./myproj/project_1.srcs/sources_1/bd/project_1/project_1.bd] -top
add_files -norecurse ./myproj/project_1.srcs/sources_1/bd/project_1/hdl/project_1_wrapper.v
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

set_property platform.default_output_type "sd_card" [current_project]
set_property platform.design_intent.embedded "true" [current_project]
set_property platform.design_intent.server_managed "false" [current_project]
set_property platform.design_intent.external_host "false" [current_project]
set_property platform.design_intent.datacenter "false" [current_project]

generate_target all [get_files project_1.bd]
update_compile_order -fileset sources_1

write_hw_platform -force ./xilinx_vck190_air.xsa
