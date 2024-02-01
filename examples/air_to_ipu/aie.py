import air
import air.compiler.util
from air.dialects import linalg, tensor, arith, func, memref
from air.ir import *
import air.passmanager
from air.dialects import air as airdialect
from air._mlir_libs._airMlir import _run_air_transform as run_air_transform
import sys
def matmul_on_tensors(m, n, k, dtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            MemRefType.get((m, k), dtype), MemRefType.get((k, n), dtype))
        def forward(lhs, rhs):
            out = memref.AllocOp(MemRefType.get((m, n), dtype), [], [])
            zero = arith.ConstantOp(dtype, 0)
            zero_fill = linalg.fill(zero, outs=[out])
            linalg.matmul(lhs, rhs, outs=[out])
            return out
    return module

with air.ir.Context() as ctx, Location.unknown():
    airdialect.register_dialect(ctx)
    air_module = matmul_on_tensors(2048, 2048, 512, IntegerType.get_signless(width = 32))
    
    ################################################
    ## Tiling
    ################################################

    pm = air.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module.operation)
    with open('air_input.mlir', 'w') as f:
        f.write(str(air_module))
    
    transform_ir_string = """
    transform.with_pdl_patterns {
    ^bb0(%arg0: !pdl.operation):
        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul_1, %loops:2 = transform.air.linalg_tile %matmul [64, 64, 0]
            %fill_1 = transform.air.fuse_into_containing_op %fill into %loops#1
            transform.air.linalg_promote %fill_1 {"operands_to_promote"=[1], "memory_space"="L2"}
            transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[2], "memory_space"="L2"}
            transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[0,1], "memory_space"="L2"}
            %matmul_2, %loops_2:2 = transform.air.linalg_tile %matmul_1 [32, 32, 0]
            %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %loops_2#1
            transform.air.linalg_promote %fill_2 {"operands_to_promote"=[1], "memory_space"="L1"}
            transform.air.linalg_promote %matmul_2 {"operands_to_promote"=[2], "memory_space"="L1"}
            %matmul_3, %reduction_loop = transform.air.linalg_tile %matmul_2 [0, 0, 32]
            transform.air.linalg_promote %matmul_3 {"operands_to_promote"=[0,1], "memory_space"="L1"}
        }
    }
    """
    transform_ir = Module.parse(transform_ir_string)
    run_air_transform(transform_ir, air_module)
    
    with open('air_tiled.mlir', 'w') as f:
        f.write(str(air_module))
    
    ################################################
    ## Binding scf.paralell to air hierarchies
    ################################################

    pipeline = "builtin.module("+",".join([
        "buffer-results-to-out-params",
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{has-air-segment=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open('air_sync.mlir', 'w') as f:
        f.write(str(air_module))
    
    ################################################
    ## Extract event dependency and optimize schedule
    ################################################

    pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "canonicalize", "cse",
        "air-label-scf-for-to-ping-pong",
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    # Not sure why parsing the ir solves the segmentation fault...
    air_module = Module.parse(str(air_module))
    pipeline = "builtin.module("+",".join([
        "air-ping-pong-transform{keep-memref-dealloc=true}",
        "air-dealias-memref",
        "canonicalize", "cse",
        "air-isolate-async-dma-loop-nests",
        "air-specialize-channel-wrap-and-stride",
        "canonicalize", "cse",
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open('aircc_input.mlir', 'w') as f:
        f.write(str(air_module))
    
    ################################################
    ## Place herd to segment
    ################################################

    air_async_module = Module.parse(str(air_module))
    pipeline = "builtin.module("+",".join([
        "func.func(air-collapse-herd)",
        'canonicalize', 'cse',
        "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor=0}",
        'canonicalize', 'cse',
        'func.func(air-renumber-dma)',
        'func.func(convert-linalg-to-loops)',
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open('air_placed.mlir', 'w') as f:
        f.write(str(air_module))
    
    # ################################################
    # ## MLIR-AIR to MLIR-AIE
    # ################################################
    
    pipeline = "builtin.module("+",".join([
        'air-to-aie{row-offset=2 col-offset=0 device=ipu emit-while-loop=true}',
        'canonicalize',
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open('aircc_decomp_aiecc.mlir', 'w') as f:
        f.write(str(air_module))
    
    ################################################
    ## MLIR-AIR runtime lowering
    ################################################

    pipeline = "builtin.module("+",".join([
      'air-to-std',
      'func.func(affine-loop-tile{tile-sizes=4,4})',
      'func.func(air-unroll-outer-affine-loops{depth=2})',
      'affine-expand-index-ops',
      'airrt-to-ipu',
      'canonicalize',
    ])+')'
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    with open('aie.mlir', 'w') as f:
        f.write(str(air_module))
