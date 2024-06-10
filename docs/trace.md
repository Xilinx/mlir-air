# Auto Trace Generation in AIR

## Usage

To enable this feature, 

* provide the `insert-trace-packet-flow=true` option to the `air-to-aie` pass, and
* specify the `trace-size`, `trace-offset` options to the `airrt-to-npu` pass. 

Trace can then be generated for all compute tiles (cores) and memtiles, unless there is a routing congestion when the build might fail.

`trace-size` defines the buffer size allocated to hold the trace data, represented in bytes. Currently, this value is chosen by the user empirically, depending on the number of cores traced and how frequent the event might be triggered.

`trace-offset` defines the offset when the trace data are appended to the output. It might be inferred from the code in the future. In addition, it is for now hard coded that the trace data are dumped to `ddr_id = 2`.

One such example is provided in `test/xrt/01_air_to_npu`, and the generated trace file can be further processed through [parse_trace.py](https://github.com/Xilinx/mlir-aie/blob/main/programming_examples/utils/parse_trace.py).


Currently, in this pariticular example and when trace is enabled, the entire column of core tiles is shifted to the right by one and all trace data comes out via the second column's shim tile. This is a workaround for the congestion that the `South` port is running out and the bottom row of core tiles (i.e. the 2nd row of the whole array) cannot be routed as `Trace->South->West/East`, once it hits the switchbox of memtile.

## air-to-aie
Inside this pass, the packet flows are inserted when `insert-trace-packet-flow=true`. The source of the flow is `channel = 0` of the trace port and the destination is `channel = 1` of the shim tile in the same column. 

One possible future improvement can be allowing user to specify which channel/shim tile to use, or having an allocation algorithm in place. In addition, the current assumption is everything else apart from the trace are using circult-switch connections, without detecting any potential conflict in the packet id.

## airrt-to-npu
This pass is responsible for inserting trace-related `NpuWrite32Op` to `func.func`. The details of these operations have already been documented in [MLIR-AIE](https://github.com/Xilinx/mlir-aie/blob/resnet/docs/Tracing.md), except the extra support for timestamp synchronization across multiple traces. 

To have the synchronization, the following steps are required:

* make the internal timer of each tile reset, when the event `BROADCAST_15` is detected. The address is `0x34000` and `0x94000` for the NPU compute tile and memtile respectively. The event id is `122` and `157` respectively according to this [header file](https://github.com/Xilinx/aie-rt/blob/main-aie/driver/src/events/xaie_events_aieml.h).
* set the start of the trace triggered by `BROADCAST_15` as well, with the address as `0x340D0` and `0x940D0`.
* for the bottom left tile (0, 0), reset the timer when `USER_EVENT_1` is detected. The address to write is `0x34000` and the event id is `127`.
* use `USER_EVENT_1` to trigger `BROADCAST_15`. This is done by writing `127` to address `0x3404C`.
* actually trigger `USER_EVENT_1` by writing `127` to address `0x34008`.

So far, the values of these operations (such as specifying which events or ports to monitor) and the addresses are all hard coded. In the future, they might also be exposed as user options and depend on the `TargetModel` as well.
