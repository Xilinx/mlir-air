// RUN: air-opt %s -test-transform-dialect-interpreter

func.func @matmul_on_buffers(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) {
  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%arg0, %arg1 : memref<1024x1024xf32>, memref<1024x1024xf32>) outs(%arg2 : memref<1024x1024xf32>)
  return
}
// Tile the linalg operation and convert to air.herd + air.launch
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %1, %outer_tile_loops:3 = transform.air.linalg_tile %0 [64, 64, 64]
    %2, %inner_tile_loops:3 = transform.air.linalg_tile %1 [32, 32, 32]
    %3 = transform.air.linalg_promote %2
    %4 = transform.air.par_to_herd %inner_tile_loops#0
    %5 = transform.air.par_to_launch %outer_tile_loops#0
  }
}
// Convert memref.copy to air.dma_memcpy_nd
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_copy : benefit(1) {
    %args = pdl.operands
    %results = pdl.types
    %op = pdl.operation "memref.copy"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    pdl.rewrite %op with "transform.dialect"
  }
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb2(%arg1: !pdl.operation):
    %4 = pdl_match @match_copy in %arg1 : (!pdl.operation) -> !pdl.operation
    %5 = transform.air.copy_to_dma %4
  }
}
