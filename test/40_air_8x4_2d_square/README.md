# Test 40_air_8x4_2d_square

## Coverage

| Coverage | How |
| -------- | --- |
| AIR dialect  | Set up 8x4 herd where each core receives the same input and distributes to a different location. |
| AIR dialect  | nd memcpy sends an ND memcpy packet to Shim DMAs. |
| Shim DMA | All of the available shim DMAs used with both channels |
| Tile DMA | Both directions used in 4 tiles | 
| Microblaze Shim DMA Packet | Microblaze Shim ND DMA memcpy packet used to program Shim DMA. |

-----

<p align="center">Copyright&copy; 2019-2022 AMD/Xilinx</p>