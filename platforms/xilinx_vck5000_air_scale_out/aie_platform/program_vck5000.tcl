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

puts "Please enter the card index to program: "

set data [gets stdin]
scan $data "%d" myidx

# Open vivado hardware manager to program device
open_hw_manager
# Initiate harwdare manager's server
connect_hw_server -url localhost:3121
# Set the current target card based on index 0->N
current_hw_target [lindex [get_hw_targets] $myidx]
# Open the current target card
open_hw_target
# Set the current target device. ID 1 for xcvc1902_*
current_hw_device [lindex [get_hw_devices] 1]
# Refresht the device and setup probes because that's what the GUI does
refresh_hw_device -update_hw_probes false [lindex [lindex [get_hw_devices] 1] 0]
# Set the programming file to the final PDI with the firmware
set_property PROGRAM.FILE {./final_vck5000.pdi} [lindex [get_hw_devices] 1]
# Initiate device programming
program_hw_devices [lindex [get_hw_devices] 1]
