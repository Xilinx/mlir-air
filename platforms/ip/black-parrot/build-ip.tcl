# Copyright(C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPDX-License-Identifier: MIT
#

set ip_name            $::env(IP_NAME)
set project_name       $::env(PROJECT_NAME)
set project_dir        $::env(PROJECT_DIR)
set project_top_module $::env(TOP_MODULE)
set project_part       $::env(PART)
set project_ip_dir     $::env(PROJECT_IP_DIR)
set flist              $::env(FLIST)
set vendor             amd.com
set vendor_name        {Advanced Micro Devices, Inc.}
set library            ip
set version            1.0
set taxonomy           {/Embedded_Processing/Processor}

source $::env(TCL_DIR)/vivado-parse-flist.tcl

set vlist [vivado_parse_flist ${flist}]
set vsources_list  [lindex $vlist 0]
set vincludes_list [lindex $vlist 1]
set vdefines_list  [lindex $vlist 2]

#
# create project and load in all the files
#

create_project -force -part ${project_part} ${project_name} ${project_dir}

puts ${vsources_list}
puts ${vdefines_list}

if {[string equal [get_filesets -quiet sources_1] ""]} {
  create_fileset -srcset sources_1
}

if {[string equal [get_filesets -quiet sim_1] ""]} {
  create_fileset -simset sim_1
}

add_files -norecurse ${vsources_list}
set_property file_type SystemVerilog [get_files ${vsources_list}]
set_property include_dirs ${vincludes_list} [get_filesets sources_1]
set_property verilog_define ${vdefines_list} [get_filesets sources_1]
set_property top ${project_top_module} [get_filesets sources_1]
update_compile_order -fileset sources_1

set_property top ${project_top_module} [get_filesets sim_1]
set_property include_dirs ${vincludes_list} [get_filesets sim_1]
set_property verilog_define ${vdefines_list} [get_filesets sim_1]
update_compile_order -fileset sim_1


ipx::package_project -root_dir ${project_ip_dir} -vendor ${vendor} -library ${library} -taxonomy ${taxonomy} -import_files -set_current false
ipx::unload_core ${project_ip_dir}/component.xml
ipx::edit_ip_in_project -upgrade true -name tmp_edit_project -directory ${project_ip_dir} ${project_ip_dir}/component.xml
update_compile_order -fileset sources_1
set_property vendor ${vendor} [ipx::current_core]
set_property library ${library} [ipx::current_core]
set_property name ${ip_name} [ipx::current_core]
set_property version ${version} [ipx::current_core]
set_property display_name ${ip_name} [ipx::current_core]
set_property description ${ip_name} [ipx::current_core]
set_property vendor_display_name ${vendor_name} [ipx::current_core]
# addressing and memory information
ipx::remove_all_memory_map [ipx::current_core]
ipx::add_memory_map s00_axi [ipx::current_core]
set_property slave_memory_map_ref s00_axi [ipx::get_bus_interfaces s00_axi -of_objects [ipx::current_core]]
ipx::add_address_block mem [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]
set_property usage memory [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
set_property access read-write [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
set_property range_dependency {pow(2,(spirit:decode(id('MODELPARAM_VALUE.C_S00_AXI_ADDR_WIDTH')) - 1) + 1)} [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
set_property range_resolve_type dependent [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
set_property width_dependency {(spirit:decode(id('MODELPARAM_VALUE.C_S00_AXI_DATA_WIDTH')) - 1) + 1} [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
set_property width_resolve_type dependent [ipx::get_address_blocks mem -of_objects [ipx::get_memory_maps s00_axi -of_objects [ipx::current_core]]]
# core_reset
ipx::add_bus_parameter POLARITY [ipx::get_bus_interfaces core_reset -of_objects [ipx::current_core]]
set_property value ACTIVE_HIGH [ipx::get_bus_parameters POLARITY -of_objects [ipx::get_bus_interfaces core_reset -of_objects [ipx::current_core]]]
# clocks
set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]
ipx::add_bus_parameter FREQ_TOLERANCE_HZ [ipx::get_bus_interfaces m00_axi_aclk -of_objects [ipx::current_core]]
set_property value -1 [ipx::get_bus_parameters FREQ_TOLERANCE_HZ -of_objects [ipx::get_bus_interfaces m00_axi_aclk -of_objects [ipx::current_core]]]
ipx::add_bus_parameter FREQ_TOLERANCE_HZ [ipx::get_bus_interfaces m01_axi_aclk -of_objects [ipx::current_core]]
set_property value -1 [ipx::get_bus_parameters FREQ_TOLERANCE_HZ -of_objects [ipx::get_bus_interfaces m01_axi_aclk -of_objects [ipx::current_core]]]
ipx::add_bus_parameter FREQ_TOLERANCE_HZ [ipx::get_bus_interfaces s00_axi_aclk -of_objects [ipx::current_core]]
set_property value -1 [ipx::get_bus_parameters FREQ_TOLERANCE_HZ -of_objects [ipx::get_bus_interfaces s00_axi_aclk -of_objects [ipx::current_core]]]

ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
ipx::move_temp_component_back -component [ipx::current_core]
close_project -delete
set_property ip_repo_paths ${project_ip_dir} [current_project]
update_ip_catalog

