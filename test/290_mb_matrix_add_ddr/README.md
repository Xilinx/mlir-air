# Test 290_mb_matrix_add_ddr
## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Shim DMA | 3 channels used and data in DDR |
| Tile DMA | Both directions used in tile DMA | 
| Microblaze Shim DMA Packet| Microblaze Shim ND DMA memcpy packet used to program Shim DMA. 4D DMA copy used to push data in/out of the herd |\
| Microblaze Simultaneous Shim DMA| Microblaze interleaves buffer descriptors as necessary for channels that are stalled. |
