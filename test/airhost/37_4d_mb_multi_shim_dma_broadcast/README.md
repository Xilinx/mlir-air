# Test 37_4d_mb_multi_shim_dma_broadcast

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Logical dialect  | Pathfinder flows are used to broadcast from Shim DMA |
| Shim DMA | Multiple channels used |
| Tile DMA | Both directions used in 4 tiles | 
| Microblaze | Two Shim DMAs are used |
| Microblaze Shim DMA Packet| Microblaze Shim ND DMA memcpy packet used to program Shim DMA. 4D DMA copy used to push data in/out of the herd |
| Microblaze Simultaneous Shim DMA| Microblaze interleaves buffer descriptors as necessary for channels that are stalled. |

-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc. All rights reserved.</p>