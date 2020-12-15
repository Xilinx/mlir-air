# Test 11_mb_shim_dma_from_tile_dma

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Microblaze Shim DMA Packet| Microblaze Shim DMA memcpy packet used to program Shim DMA. MM2S0 used to pull data from the herd |
| Tile DMA | S2MM0 used to push data to streams from the tile memory | 