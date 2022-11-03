# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

setws build

set script_directory [file dirname [file normalize [info script]]]

set root_directory [file normalize [file join $script_directory "../.."]]

platform create -name "mb" -hw [file join ${root_directory} "platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa"]

::common::get_property NAME [hsi::get_cells -filter {IP_TYPE==PROCESSOR}]
domain create -name microblaze -proc microblaze_0 

platform generate

app create -lang c++ -name acdc_agent -platform mb -domain microblaze -template "Empty Application (C++)"
#app config -name acdc_agent build-config Release
#app config -name acdc_agent compiler-misc {-O1}

importsources -name acdc_agent -soft-link -path [file join $root_directory "runtime_lib/controller"]
app config -name acdc_agent include-path [file join $root_directory "runtime_lib/airhost/include"]

app build -name acdc_agent
