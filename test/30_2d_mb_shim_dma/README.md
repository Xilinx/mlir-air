# Test 30_2d_mb_shim_dma

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Shim DMA | 2 channels used |
| Tile DMA | Both directions used in 2 tiles | 
| Microblaze Shim DMA Packet| Microblaze Shim ND DMA memcpy packet used to program Shim DMA. 2D DMA copy used to push data in/out of the herd |
| Microblaze Simultaneous Shim DMA| Microblaze interleaves buffer descriptors as necessary for channels that are stalled. |

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>