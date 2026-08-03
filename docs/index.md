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

The Python API constructs this elementwise-add design as MLIR-AIR IR. Two herd
cores each process a slice of a 4096-element vector, staging data L3 → L2 → L1,
computing `C = A + B` in local memory, and writing the result back. Switch tabs
to inspect the same program in either representation.

=== "Python API"

    ```python
    from air.ir import *
    from air.dialects.affine import apply as affine_apply
    from air.dialects.air import *
    from air.dialects.memref import AllocOp, DeallocOp, load, store
    from air.dialects.func import FuncOp
    from air.dialects.scf import for_, yield_
    from air.backend.xrt_runner import type_mapper

    range_ = for_


    @module_builder
    def build_module(n, tile_n, np_dtype_in):
        a_size = [n]
        xrt_dtype_in = type_mapper(np_dtype_in)
        num_tiles = 2
        assert n % (tile_n * num_tiles) == 0

        # Function arguments reside in L3 (system memory).
        l3memrefTy = MemRefType.get(a_size, xrt_dtype_in)

        # A segment owns L2 storage shared by its herd.
        l2MemrefTy = MemRefType.get(
            shape=a_size,
            element_type=xrt_dtype_in,
            memory_space=IntegerAttr.get(T.i32(), MemorySpace.L2),
        )

        # Each herd core has its own tile-sized L1 buffer.
        l1MemrefTy = MemRefType.get(
            shape=[tile_n],
            element_type=xrt_dtype_in,
            memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
        )

        @FuncOp.from_py_func(l3memrefTy, l3memrefTy, l3memrefTy)
        def eltwise_add(arg0, arg1, arg2):
            @launch(operands=[arg0, arg1, arg2], sizes=[1, 1])
            def launch_body(_ivx, _ivy, _sx, _sy, arg0_l, arg1_l, arg2_l):
                @segment(name="segment_0", operands=[arg0_l, arg1_l, arg2_l])
                def segment_body(arg0_s, arg1_s, arg2_s):
                    l2_a = AllocOp(l2MemrefTy, [], [])
                    l2_b = AllocOp(l2MemrefTy, [], [])
                    l2_out = AllocOp(l2MemrefTy, [], [])

                    # L3 -> L2: stage both input vectors in the MemTile level.
                    dma_memcpy_nd(l2_a, arg0_s)
                    dma_memcpy_nd(l2_b, arg1_s)

                    @herd(
                        name="herd_0",
                        sizes=[1, num_tiles],
                        operands=[l2_a, l2_b, l2_out],
                    )
                    def herd_body(_tx, _ty, _sx, _sy, a, b, c):
                        l1_a = AllocOp(l1MemrefTy, [], [])
                        l1_b = AllocOp(l1MemrefTy, [], [])
                        l1_out = AllocOp(l1MemrefTy, [], [])

                        for _iv in range_(0, n, tile_n * num_tiles):
                            offset_map = AffineMap.get(
                                0,
                                2,
                                [
                                    AffineExpr.get_add(
                                        AffineSymbolExpr.get(0),
                                        AffineExpr.get_mul(
                                            AffineSymbolExpr.get(1),
                                            AffineConstantExpr.get(tile_n),
                                        ),
                                    )
                                ],
                            )
                            offset = affine_apply(offset_map, [_iv, _ty])

                            # L2 -> L1: bring this core's tiles into local memory.
                            dma_memcpy_nd(
                                l1_a, a, src_offsets=[offset],
                                src_sizes=[tile_n], src_strides=[1],
                            )
                            dma_memcpy_nd(
                                l1_b, b, src_offsets=[offset],
                                src_sizes=[tile_n], src_strides=[1],
                            )

                            # Compute C = A + B in each core's local memory.
                            for i in range_(tile_n):
                                val = arith.addf(load(l1_a, [i]), load(l1_b, [i]))
                                store(val, l1_out, [i])
                                yield_([])

                            # L1 -> L2: return the completed tile to the segment.
                            dma_memcpy_nd(
                                c, l1_out, dst_offsets=[offset],
                                dst_sizes=[tile_n], dst_strides=[1],
                            )
                            DeallocOp(l1_a)
                            DeallocOp(l1_b)
                            DeallocOp(l1_out)
                            yield_([])

                    # L2 -> L3: write the full output vector to the result.
                    dma_memcpy_nd(arg2_s, l2_out)
                    DeallocOp(l2_a)
                    DeallocOp(l2_b)
                    DeallocOp(l2_out)
    ```

=== "MLIR-AIR IR"

    ```mlir
    #map = affine_map<()[s0, s1] -> (s0 + s1 * 512)>
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
            %c1_3 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            air.herd @herd_0  tile (%arg13, %arg14) in (%arg15=%c1_3, %arg16=%c2) args(%arg17=%alloc, %arg18=%alloc_1, %arg19=%alloc_2) : memref<4096xf32, 1 : i32>, memref<4096xf32, 1 : i32>, memref<4096xf32, 1 : i32> {
              %alloc_4 = memref.alloc() : memref<512xf32, 2 : i32>
              %alloc_5 = memref.alloc() : memref<512xf32, 2 : i32>
              %alloc_6 = memref.alloc() : memref<512xf32, 2 : i32>
              %c0 = arith.constant 0 : index
              %c4096 = arith.constant 4096 : index
              %c1024 = arith.constant 1024 : index
              scf.for %arg20 = %c0 to %c4096 step %c1024 {
                %0 = affine.apply #map()[%arg20, %arg14]
                air.dma_memcpy_nd (%alloc_4[] [] [], %arg17[%0] [512] [1]) : (memref<512xf32, 2 : i32>, memref<4096xf32, 1 : i32>)
                air.dma_memcpy_nd (%alloc_5[] [] [], %arg18[%0] [512] [1]) : (memref<512xf32, 2 : i32>, memref<4096xf32, 1 : i32>)
                %c0_7 = arith.constant 0 : index
                %c512 = arith.constant 512 : index
                %c1_8 = arith.constant 1 : index
                scf.for %arg21 = %c0_7 to %c512 step %c1_8 {
                  %1 = memref.load %alloc_4[%arg21] : memref<512xf32, 2 : i32>
                  %2 = memref.load %alloc_5[%arg21] : memref<512xf32, 2 : i32>
                  %3 = arith.addf %1, %2 : f32
                  memref.store %3, %alloc_6[%arg21] : memref<512xf32, 2 : i32>
                }
                air.dma_memcpy_nd (%arg19[%0] [512] [1], %alloc_6[] [] []) : (memref<4096xf32, 1 : i32>, memref<512xf32, 2 : i32>)
                memref.dealloc %alloc_4 : memref<512xf32, 2 : i32>
                memref.dealloc %alloc_5 : memref<512xf32, 2 : i32>
                memref.dealloc %alloc_6 : memref<512xf32, 2 : i32>
              }
            }
            air.dma_memcpy_nd (%arg12[] [] [], %alloc_2[] [] []) : (memref<4096xf32>, memref<4096xf32, 1 : i32>)
            memref.dealloc %alloc : memref<4096xf32, 1 : i32>
            memref.dealloc %alloc_1 : memref<4096xf32, 1 : i32>
            memref.dealloc %alloc_2 : memref<4096xf32, 1 : i32>
          }
        }
        return
      }
    }
    ```

The Python tab is a condensed excerpt — renamed and reflowed for readability —
of the CI-covered
[`programming_examples/eltwise_add_with_l2/`](https://github.com/Xilinx/mlir-air/tree/main/programming_examples/eltwise_add_with_l2)
example; see that directory for the exact, runnable source and its XRT test
harness. The IR tab is the verbatim `--print-module-only` output of that example
at `n=4096`, `tile_n=512`.

For more details on the AIR compute and memory model — launches, segments,
herds, and how data movement maps onto hardware — read the
[AIR Compute Model](AIRComputeModel/).
