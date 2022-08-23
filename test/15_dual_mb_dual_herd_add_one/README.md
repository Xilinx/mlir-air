# Test 15_dual_mb_dual_herd_add_one
## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to/from Shim DMA |
| Shim DMA | 2 channels used (S2MM MM2S) |
| Tile DMA | Both directions used in 2 tiles | 
| Microblaze | Multiple queues to multiple herd controllers |
| Microblaze | Asynchronous packet dispatch and wait |

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>