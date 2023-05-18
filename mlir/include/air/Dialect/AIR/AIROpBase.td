//===- AIROpBase.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_OPS_BASE
#define AIR_OPS_BASE

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def air_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "xilinx::air";
  let description = [{
    The `air` dialect is used to describe cooperating arrays of AIE cores
    and the bulk data movement between AIE cores and a memory hierarchy.
    Its primary features are collective operations for executing work on
    groups of AIE cores and multi-dimensional DMA copy operations.

    ### Herd
    A `herd` in the `air` dialect is a one or two dimensional array of adjacent
    AIE cores executing the same code region. A `herd` operation defines an
    iteration space representing the spatial parallelism of the array of cores.
    Code within a `herd` can directly access L1 memory but must use DMA
    operations or channels to access other levels of memory.

    ### Segment
    A `segment` in the `air` dialect represents a physically contiguous
    grouping of AIE cores, L1 and L2 memory resources, and controllers
    sufficient to implement the herds, memory allocations, data movement,
    and synchronization contained in the `segment` code region. Code within
    a `segment` can allocate L2 memory but must use DMA operations or
    channels to access other levels of memory. A `segment` can optionally
    define an iteration space which represents a spatial unroll of the
    `segment`. That is, it allows the `segment` to be "stamped out" multiple
    times, also multiplying the physical resources required by the `segment`.

    ### Launch
    A `launch` in the `air` dialect groups `segments` and L3 allocations that
    must be co-resident within a device when execution of the `launch`
    code region begins. A `launch` operation can optionally define a parallel
    iteration space.

    ### Memory Hierarchy
    There are three levels of memory in the air dialect:
    - `L1` memory corresponds to the AIE tile memory.
    - `L2` memory corresponds to pools of on-device memory such as of URAMs,
      memory tiles, or other memory resources.
    - `L3` memory corresponds to off-chip memory such as DDR.
    Each memory space has an enum defined in the air::MemorySpace namespace.

    ### DMA Memory Copies
    The `dma_memcpy_nd` operation in `air` dialect describes asynchronous
    multi-dimensional copy operations between levels of memory.

    ### Channels
    A `channel` in the `air` dialect is a representation of data movement...

    ### Asynchronous operations
    Operations in air dialect can be asynchronous. An asynchronous token
    interface is provided to synchronize between operations.

  }];
  let useDefaultTypePrinterParser = 1;
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

def air_AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def air_AnyScalar : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;

def L1:  I32EnumAttrCase<"L1", 2>;
def L2:  I32EnumAttrCase<"L2", 1>;
def L3:  I32EnumAttrCase<"L3", 0>;

def MemorySpace: I32EnumAttr<"MemorySpace", "AIR Memory Space IDs",
  [L1, L2, L3]> {

  let cppNamespace = "xilinx::air";
}

def air_AsyncToken : DialectType<
  air_Dialect, CPred<"$_self.isa<xilinx::air::AsyncTokenType>()">, "async token type">,
             BuildableType<"xilinx::air::AsyncTokenType::get($_builder.getContext())">;

def air_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for AIR operations that execute asynchronously.

    AIR operations implementing this interface take a list of dependencies
    as `air.async.token` arguments and optionally return a `air.async.token`.

    The op doesn't start executing until all ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op is scheduled for execution on the
    device and returns immediately, without waiting for the execution to
    complete. If the op does not return a token, the op will wait for
    execution to complete.
  }];
  let cppNamespace = "xilinx::air";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        ::xilinx::air::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Erases a token from the list of async dependencies.
      }],
      "void", "eraseAsyncDependency", (ins "unsigned":$index),
      [{}], [{
        ::xilinx::air::eraseAsyncDependency(this->getOperation(), index);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "Value", "getAsyncToken"
    >
  ];
}

def air_DmaMemcpyInterface : OpInterface<"DmaMemcpyInterface"> {
  let description = [{}];
  let cppNamespace = "xilinx::air";
  let methods = [
    InterfaceMethod<"description",
    "unsigned", "getNumDims"
    >,
    InterfaceMethod<"description",
    "Value", "getSrcMemref"
    >,
    InterfaceMethod<"description",
    "Value", "getDstMemref"
    >,
    InterfaceMethod<"description",
    "Value", "getSrcMemrefDim", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "Value", "getDstMemrefDim", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "Value", "getLength"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}

def air_HierarchyInterface : OpInterface<"HierarchyInterface"> {
  let description = [{
    Interface for AIR hierarchy ops `air.herd`, `air.segment` and `air.launch`

    Operations implementing this interface define an iteration space, support
    arguments for implementing "isolated from above" and have an integer `id`
    attribute.
  }];
  let cppNamespace = "xilinx::air";
  let methods = [
    InterfaceMethod<"description",
    "ArrayRef<BlockArgument>", "getIds"
    >,
    InterfaceMethod<"description",
    "ArrayRef<BlockArgument>", "getSize"
    >,
    InterfaceMethod<"description",
    "OperandRange", "getSizeOperands"
    >,
    InterfaceMethod<"description",
    "unsigned", "getNumKernelOperands"
    >,
    InterfaceMethod<"description",
    "ArrayRef<BlockArgument>", "getKernelArguments"
    >,
    InterfaceMethod<"description",
    "Value", "getKernelOperand", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "BlockArgument", "getKernelArgument", (ins "unsigned":$i)
    >,
    InterfaceMethod<"description",
    "unsigned", "getNumDims"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}

def air_ChannelInterface : OpInterface<"ChannelInterface"> {
  let description = [{
    Interface for AIR channel ops.
  }];
  let cppNamespace = "xilinx::air";
  let methods = [
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
    InterfaceMethod<"description",
    "StringRef", "getChanName"
    >,
    InterfaceMethod<"",
    "OperandRange", "getIndices"
    >,
    InterfaceMethod<"",
    "OperandRange", "getOffsets"
    >,
    InterfaceMethod<"",
    "OperandRange", "getSizes"
    >,
    InterfaceMethod<"",
    "OperandRange", "getStrides"
    >,
    InterfaceMethod<"",
    "Value", "getMemref"
    >,
  ];
}
#endif
