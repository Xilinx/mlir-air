# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

setws build

set script_directory [file dirname [file normalize [info script]]]

set root_directory [file normalize [file join $script_directory "../.."]]

platform create -name "mb" -hw [file join ${root_directory} "platforms/xilinx_vck190_air/vivado/xilinx_vck190_air.xsa"]
#platform create -name "arm" -hw [file join ${root_directory} "platforms/xilinx_vck5000_air/vivado/xilinx_vck5000_air.xsa"]

#domain create -name airrt_arm -proc psv_cortexa72_0 -os standalone
::common::get_property NAME [hsi::get_cells -filter {IP_TYPE==PROCESSOR}]
domain create -name microblaze -proc microblaze_0 

platform generate

app create -lang c++ -name acdc_agent -platform mb -domain microblaze -template "Empty Application (C++)"
#app create -lang c++ -name acdc_agent -platform arm -domain airrt_arm -template "Empty Application (C++)"
#app config -name acdc_agent build-config Release
#app config -name acdc_agent compiler-misc {-O1}

importsources -name acdc_agent -soft-link -path [file join $root_directory "runtime_lib/controller"]
app config -name acdc_agent include-path [file join $root_directory "runtime_lib/airhost/include"]

app build -name acdc_agent
