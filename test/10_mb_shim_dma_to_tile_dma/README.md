# Test 10_mb_shim_dma_to_tile_dma

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Microblaze Shim DMA Packet| Microblaze Shim DMA memcpy packet used to program Shim DMA. MM2S0 used to push data into the herd |
| Tile DMA | S2MM0 used to pull data from streams into the tile memory |
