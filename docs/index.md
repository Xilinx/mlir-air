<div class="air-hero" markdown>

<p class="air-eyebrow">Open-source MLIR compiler infrastructure</p>

# MLIR-AIR

<p class="air-tagline">Map structured loop and tensor programs onto AMD NPUs and Versal AI Engine arrays.</p>

[Get started](getting_started/){ .md-button .md-button--primary }
[A simple AIR program](#a-simple-air-program){ .md-button }

</div>

<div class="air-intro" markdown>

On AMD AI Engine–based NPUs, compute cores, DMA engines, and communication
resources execute concurrently and are coordinated through explicit data
movement and synchronization. Achieving high performance therefore requires
constructing dataflow organizations that are sufficiently pipelined to overlap
communication with computation, balanced across processing stages, compatible
with finite on-chip memory and routing resources, and free from cyclic
dependencies that can lead to deadlock.

MLIR-AIR provides compiler abstractions and transformation infrastructure for
exploring this design space. It preserves a common representation while a design
is progressively transformed across discrete choices in tiling, placement,
buffering, communication, and synchronization. Through its Python APIs,
developers can derive and evaluate related implementations without reconstructing
each candidate directly in MLIR-AIE, making systematic design-space exploration
substantially more accessible.

</div>

<div class="air-cards" markdown>

<a class="air-card" href="getting_started/">
<span class="air-card-icon">🚀</span>
<span class="air-card-title">Get Started</span>
<span class="air-card-desc">Build from source and run designs on Ryzen AI or Versal.</span>
</a>

<a class="air-card" href="programming_examples/">
<span class="air-card-icon">📊</span>
<span class="air-card-title">Operator Dashboard</span>
<span class="air-card-desc">Operators and examples with live NPU1/NPU2 test status.</span>
</a>

<a class="air-card" href="llms/">
<span class="air-card-icon">🧠</span>
<span class="air-card-title">LLMs on NPU</span>
<span class="air-card-desc">Decoder-only models running end-to-end on NPU2, with a nightly benchmark.</span>
</a>

<a class="air-card" href="https://arxiv.org/abs/2510.14871">
<span class="air-card-icon">📄</span>
<span class="air-card-title">Read the Paper</span>
<span class="air-card-desc">From Loop Nests to Silicon: Mapping AI Workloads onto AMD NPUs with MLIR-AIR.</span>
</a>

</div>

## A simple AIR program

This elementwise-add design distributes a 4096-element vector across two herd
cores. The segment stages both inputs from L3 into L2. Each core then brings its
tile into L1, computes `C = A + B`, and writes the result back. The following
tabs show the Python source and the MLIR-AIR IR generated from it.

=== "Python API"

    ```python
    from air import api as air
    from air.api.types import dtype_of

    NUM_TILES = 2


    def build_module(n, tile_n, np_dtype_in):
        assert n % (tile_n * NUM_TILES) == 0
        dt = dtype_of(np_dtype_in)

        # Function arguments reside in L3 (system memory).
        A = air.tensor([n], dt)
        B = air.tensor([n], dt)
        C = air.tensor([n], dt)

        with air.launch(name="eltwise_add") as launch:

            @launch.body
            def _():
                with air.segment(name="segment_0") as seg:

                    @seg.body
                    def _():
                        # seg.private() places these in L2: storage owned by the
                        # segment and shared by its herd.
                        l2_a = air.alloc([n], dt, scope=seg.private())
                        l2_b = air.alloc([n], dt, scope=seg.private())
                        l2_c = air.alloc([n], dt, scope=seg.private())

                        # L3 -> L2: stage both input vectors in the MemTile level.
                        air.ops.load(l2_a, A)
                        air.ops.load(l2_b, B)

                        # The tile grid `range(0, n, tile_n)` is strip-mined onto
                        # NUM_TILES cores by the DSL, and l2_a/l2_b/l2_c are
                        # carried in as operands automatically.
                        with air.herd(
                            [range(0, n, tile_n)],
                            name="herd_0",
                            shape=(NUM_TILES,),
                        ) as h:

                            @h.body
                            def _(tx):
                                # tx counts tiles, not elements.
                                i0 = tx * tile_n

                                # h.private() places these in L1: each core has
                                # its own tile-sized buffer.
                                a = air.alloc([tile_n], dt, scope=h.private())
                                b = air.alloc([tile_n], dt, scope=h.private())
                                c = air.alloc([tile_n], dt, scope=h.private())

                                # L2 -> L1: bring this core's tiles into local memory.
                                air.ops.load(a, l2_a[i0 : i0 + tile_n])
                                air.ops.load(b, l2_b[i0 : i0 + tile_n])

                                # Compute C = A + B in each core's local memory.
                                c[:] = a[:] + b[:]

                                # L1 -> L2: return the completed tile to the segment.
                                air.ops.store(c, l2_c[i0 : i0 + tile_n])

                        # L2 -> L3: write the full output vector to the result.
                        air.ops.store(l2_c, C)

        return launch
    ```

=== "MLIR-AIR IR"

    ```mlir
    #map = affine_map<()[s0, s1] -> (s0 * 2048 + s1 * 512)>
    module {
      func.func @eltwise_add(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>) {
        %c1 = arith.constant 1 : index
        %c1_0 = arith.constant 1 : index
        air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<4096xf32>, memref<4096xf32>, memref<4096xf32> {
          air.segment @segment_0  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<4096xf32>, memref<4096xf32>, memref<4096xf32> {
            %alloc = memref.alloc() : memref<4096xf32, 1 : i32>
            %alloc_1 = memref.alloc() : memref<4096xf32, 1 : i32>
            %alloc_2 = memref.alloc() : memref<4096xf32, 1 : i32>
            air.dma_memcpy_nd (%alloc[] [] [], %arg10[] [] []) : (memref<4096xf32, 1 : i32>, memref<4096xf32>)
            air.dma_memcpy_nd (%alloc_1[] [] [], %arg11[] [] []) : (memref<4096xf32, 1 : i32>, memref<4096xf32>)
            %c2 = arith.constant 2 : index
            %c1_3 = arith.constant 1 : index
            air.herd @herd_0  tile (%arg13, %arg14) in (%arg15=%c2, %arg16=%c1_3) args(%arg17=%alloc, %arg18=%alloc_1, %arg19=%alloc_2) : memref<4096xf32, 1 : i32>, memref<4096xf32, 1 : i32>, memref<4096xf32, 1 : i32> {
              %c0 = arith.constant 0 : index
              %c4 = arith.constant 4 : index
              %c1_4 = arith.constant 1 : index
              scf.for %arg20 = %c0 to %c4 step %c1_4 {
                %alloc_5 = memref.alloc() : memref<512xf32, 2 : i32>
                %alloc_6 = memref.alloc() : memref<512xf32, 2 : i32>
                %alloc_7 = memref.alloc() : memref<512xf32, 2 : i32>
                %0 = affine.apply #map()[%arg13, %arg20]
                air.dma_memcpy_nd (%alloc_5[] [] [], %arg17[%0] [512] [1]) : (memref<512xf32, 2 : i32>, memref<4096xf32, 1 : i32>)
                %1 = affine.apply #map()[%arg13, %arg20]
                air.dma_memcpy_nd (%alloc_6[] [] [], %arg18[%1] [512] [1]) : (memref<512xf32, 2 : i32>, memref<4096xf32, 1 : i32>)
                %cst = arith.constant 0.000000e+00 : f32
                %c0_8 = arith.constant 0 : index
                %c512 = arith.constant 512 : index
                %c16 = arith.constant 16 : index
                scf.for %arg21 = %c0_8 to %c512 step %c16 {
                  %3 = vector.transfer_read %alloc_5[%arg21], %cst {in_bounds = [true]} : memref<512xf32, 2 : i32>, vector<16xf32>
                  %4 = vector.transfer_read %alloc_6[%arg21], %cst {in_bounds = [true]} : memref<512xf32, 2 : i32>, vector<16xf32>
                  %5 = arith.addf %3, %4 : vector<16xf32>
                  vector.transfer_write %5, %alloc_7[%arg21] {in_bounds = [true]} : vector<16xf32>, memref<512xf32, 2 : i32>
                }
                memref.dealloc %alloc_6 : memref<512xf32, 2 : i32>
                memref.dealloc %alloc_5 : memref<512xf32, 2 : i32>
                %2 = affine.apply #map()[%arg13, %arg20]
                air.dma_memcpy_nd (%arg19[%2] [512] [1], %alloc_7[] [] []) : (memref<4096xf32, 1 : i32>, memref<512xf32, 2 : i32>)
                memref.dealloc %alloc_7 : memref<512xf32, 2 : i32>
              }
            }
            memref.dealloc %alloc_1 : memref<4096xf32, 1 : i32>
            memref.dealloc %alloc : memref<4096xf32, 1 : i32>
            air.dma_memcpy_nd (%arg12[] [] [], %alloc_2[] [] []) : (memref<4096xf32>, memref<4096xf32, 1 : i32>)
            memref.dealloc %alloc_2 : memref<4096xf32, 1 : i32>
          }
        }
        return
      }
    }
    ```

To build and run this design, see
[`programming_examples/eltwise_add_with_l2`](https://github.com/Xilinx/mlir-air/tree/main/programming_examples/eltwise_add_with_l2).

For more details on the AIR compute and memory model — launches, segments,
herds, and how data movement maps onto hardware — read the
[AIR Compute Model](AIRComputeModel/).
