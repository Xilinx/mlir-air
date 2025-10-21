# Matrix multiplication
Matrix multiplication with bf16 datatype, supports variable tile sizes for m and n, and for k at L2 and L1 level.
Use `make run4x4` to compile and run the design on the NPU.
Use `make runner` to simulate the design on air-runner and the find the expected latency
Use `make profile` to run the design on hardware and measure the average latency
You can adjust the tile sizes by appending `... TILE_M=64` to the command, this can be done for "TILE_M=[]", "TILE_K_L2=[]", "TILE_K_L1=[]" and "TILE_N"
This tile sizes can be set for `run4x4`, `runner` and `profile`. For example `make run4x4 TILE_M=64 TILE_K_L2=128 TILE_K_L1=32 TILE_N=64`.
Any tile sizes left out are set to the default values which are shown below:
TILE_M=128
TILE_K_L2=64
TILE_K_L1=32
TILE_N=64