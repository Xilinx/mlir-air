# Test 18_2d_tile_dma_transpose

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to/from Shim DMA |
| Shim DMA | 2 channels used (S2MM MM2S) |
| Tile DMA | Both directions used in one tile | 
| Tile DMA | 2D X and Y address generation used to transpose a matrix | 
| Microblaze | ND memcpy packet is used |

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>