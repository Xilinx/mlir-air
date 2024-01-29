# Test 38_4d_mb_shim_dma_channel_stress

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Shim DMA | All 4 channels used |
| Tile DMA | Both directions used in 4 tiles | 
| Microblaze | 8 Shim DMAs are used simulaneously |
| Microblaze Shim DMA Packet| Microblaze Shim ND DMA memcpy packet used to program Shim DMA. 4D DMA copy used to push data in/out of the herd |
| Microblaze Simultaneous Shim DMA| Microblaze interleaves buffer descriptors as necessary for channels that are stalled. |

-----

<p align="center">Copyright&copy; 2019-2022 Advanced Micro Devices, Inc. All rights reserved.</p>