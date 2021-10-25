# Test 03_mb_lock_rel

## Coverage

| Coverage | How |
| -------- | --- |
| Physical dialect | Shim DMA is set up |
| Logical dialect  | Flows are used to route to bottom of herd |
| Shim DMA | MM2S0 used to push data into the herd |
| Tile DMA | S2MM0 used to pull data from streams into the tile memory | 
| Tile locks | Check the value of locks |
| Microblaze | MicroBlaze used as packet processor |
| Herd init | HSA packet used to initialize herd |
| Lock commands | HSA packet release lock |
