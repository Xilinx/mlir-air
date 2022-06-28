open_hw_manager
connect_hw_server -url localhost:3121
open_hw_target
current_hw_device [get_hw_devices xcvc1902_1]
refresh_hw_device -update_hw_probes false [lindex [get_hw_devices xcvc1902_1] 0]
set_property PROGRAM.FILE {./final_vck5000.pdi} [get_hw_devices xcvc1902_1]
program_hw_devices [get_hw_devices xcvc1902_1]
