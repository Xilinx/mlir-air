//===- AIROpBase.td ----------------------------------------*- tablegen -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_OPS_BASE
#define AIR_OPS_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def air_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "xilinx::air";
  let description = [{
    The `air` dialect describes spatial computation on accelerators with an
    explicit memory hierarchy and explicit data movement. It is
    architecture-independent: the same IR lowers to AMD NPUs (AI Engine arrays)
    and to AMD GPUs. The normative semantics of each operation are defined in
    the [AIR Compute Model](AIRComputeModel.md).

    ### Hierarchy
    Computation is organized in three nested levels. Each level defines an
    iteration space and a body region that is isolated from above.
    - `launch` groups segments and L3 allocations that must be co-resident on
      the device. Iterations of a `launch` may execute in parallel.
    - `segment` groups the processing elements, L2 memory, and data-movement
      resources required by its body. The iterations of a `segment` execute in
      parallel, each on its own copy of these resources.
    - `herd` is an array of processing elements executing the same body; each
      point of its iteration space executes on a distinct processing element.
      Code in a `herd` accesses L1 memory directly and reaches other memory
      levels only through data-movement operations.

    ### Memory Hierarchy
    - `L1`: local memory of a single processing element.
    - `L2`: on-device memory shared within a `segment`.
    - `L3`: off-chip or host-accessible memory.

    The memory spaces are enumerated in `air::MemorySpace`.

    ### Data Movement
    `dma_memcpy_nd` is an asynchronous multi-dimensional copy between memory
    levels. `channel` declares a stream between producers and consumers, which
    `channel.put` and `channel.get` write to and read from. Puts and gets on a
    channel must balance on every control-flow path.

    ### Asynchrony
    Operations may execute asynchronously and return an `!air.token`. Token
    operands order an operation after its producers, and `wait_all` joins
    multiple tokens.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  // The op assembly formats print inherent attributes in `attr-dict`, which
  // the strict-properties default rejects.
  let useStrictPropertiesInAssemblyFormat = 0;
}

// Base class for AIR dialect attributes.
class air_Attr<string name, list<Trait> traits = []>
    : AttrDef<air_Dialect, name, traits>;

class air_Type<string name, string typeMnemonic>
    : TypeDef<air_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def air_AsyncTokenType : air_Type<"AsyncToken", "async.token"> {
  let summary = "Token that orders asynchronous AIR operations";
}

def air_UniverseType : air_Type<"Universe", "universe"> {
  let summary = "Bounded pool of devices or hosts that air.rank runs on";
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

// Symmetric-heap memory space.
//
// Marker attribute used as a memref `memory_space` to indicate the memref
// is backed by the symmetric-heap (cross-rank XGMI-accessible HBM on GPU
// targets). Unlike the integer MemorySpace enum (L1/L2/L3 = NPU memory
// hierarchy), this attribute does not participate in the strict per-level
// access constraints — direct loads/stores from inside `air.herd` bodies
// are allowed because GPU kernels can reach HBM (and peer-mapped HBM)
// directly. The `air-symmetric-alloc-to-mgpu` pass dispatches on this
// attribute to lower `memref.alloc` ops to `mgpuSymmetricAlloc` runtime
// calls.
//
// Usage:
//   memref<32xi32, #air.symmetric_heap>
def SymmetricHeapMemorySpace_Attr : air_Attr<"SymmetricHeapMemorySpace"> {
  let mnemonic = "symmetric_heap";
  let summary = "Symmetric-heap memory space (cross-rank XGMI-accessible HBM)";
}

def air_AsyncToken : DialectType<
  air_Dialect, CPred<"llvm::isa<xilinx::air::AsyncTokenType>($_self)">, "async token type">,
             BuildableType<"xilinx::air::AsyncTokenType::get($_builder.getContext())">;

def air_Universe : DialectType<
  air_Dialect, CPred<"llvm::isa<xilinx::air::UniverseType>($_self)">, "universe type">,
             BuildableType<"xilinx::air::UniverseType::get($_builder.getContext())">;

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

def air_HierarchyInterface : OpInterface<"HierarchyInterface"> {
  let description = [{
    Interface for AIR hierarchy ops `air.herd`, `air.segment` and `air.launch`

    Operations implementing this interface define an iteration space, support
    arguments for implementing "isolated from above" and have an integer `id`
    attribute.
  }];
  let cppNamespace = "xilinx::air";
  let methods = [
    InterfaceMethod<"Get the block arguments representing the iteration space indices (e.g., %x, %y for a 2D herd)",
    "ArrayRef<BlockArgument>", "getIds"
    >,
    InterfaceMethod<"Get the block arguments representing the iteration space sizes (e.g., %sx, %sy for a 2D herd)",
    "ArrayRef<BlockArgument>", "getSize"
    >,
    InterfaceMethod<"Get the operands that define the iteration space sizes",
    "OperandRange", "getSizeOperands"
    >,
    InterfaceMethod<"Get the number of kernel operands passed to the hierarchy operation",
    "unsigned", "getNumKernelOperands"
    >,
    InterfaceMethod<"Get all block arguments representing kernel operands (arguments passed from outside)",
    "ArrayRef<BlockArgument>", "getKernelArguments"
    >,
    InterfaceMethod<"Get all operands passed as kernel arguments to the hierarchy operation",
    "OperandRange", "getKernelOperands"
    >,
    InterfaceMethod<"Get the i-th kernel operand",
    "Value", "getKernelOperand", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Get the i-th kernel argument (block argument corresponding to kernel operand)",
    "BlockArgument", "getKernelArgument", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Get the number of dimensions in the iteration space",
    "unsigned", "getNumDims"
    >,
    InterfaceMethod<"Get the unique identifier for this hierarchy operation",
    "int32_t", "getId"
    >,
    InterfaceMethod<"Get the block argument that corresponds to the given kernel operand",
    "BlockArgument", "getTiedKernelArgument", (ins "Value":$i)
    >,
    InterfaceMethod<"Get the kernel operand that corresponds to the given block argument",
    "Value", "getTiedKernelOperand", (ins "BlockArgument":$i)
    >,
    InterfaceMethod<"Get the body region of the hierarchy operation",
    "Region&", "getBody"
    >,
    InterfaceMethod<"Append new kernel operands to the hierarchy operation",
    "void", "appendKernelOperands", (ins "ValueRange":$i)
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
    InterfaceMethod<"Access pattern offsets, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedOffsets"
    >,
    InterfaceMethod<"Access pattern sizes, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedSizes"
    >,
    InterfaceMethod<"Access pattern strides, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedStrides"
    >,
    InterfaceMethod<"Replace the access pattern offsets",
    "void", "setMixedOffsets", (ins "::llvm::ArrayRef<::mlir::OpFoldResult>":$values)
    >,
    InterfaceMethod<"Replace the access pattern sizes",
    "void", "setMixedSizes", (ins "::llvm::ArrayRef<::mlir::OpFoldResult>":$values)
    >,
    InterfaceMethod<"Replace the access pattern strides",
    "void", "setMixedStrides", (ins "::llvm::ArrayRef<::mlir::OpFoldResult>":$values)
    >,
    InterfaceMethod<"",
    "Value", "getMemref"
    >,
    InterfaceMethod<"Get optional padding-before attribute",
    "std::optional<DenseI32ArrayAttr>", "getPadBeforeAttr", (ins),
    [{}], [{
      if (auto attr = this->getOperation()->template
              getAttrOfType<DenseI32ArrayAttr>("pad_before"))
        return attr;
      return std::nullopt;
    }]
    >,
    InterfaceMethod<"Get optional padding-after attribute",
    "std::optional<DenseI32ArrayAttr>", "getPadAfterAttr", (ins),
    [{}], [{
      if (auto attr = this->getOperation()->template
              getAttrOfType<DenseI32ArrayAttr>("pad_after"))
        return attr;
      return std::nullopt;
    }]
    >,
  ];
}

def air_MemcpyInterface : OpInterface<"MemcpyInterface"> {
  let description = [{}];
  let cppNamespace = "xilinx::air";
  let methods = [
    InterfaceMethod<"description",
    "Value", "getSrcMemref"
    >,
    InterfaceMethod<"description",
    "Value", "getDstMemref"
    >,
    InterfaceMethod<"Source-side offsets, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedSrcOffsets"
    >,
    InterfaceMethod<"Destination-side offsets, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedDstOffsets"
    >,
    InterfaceMethod<"Source-side sizes, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedSrcSizes"
    >,
    InterfaceMethod<"Destination-side sizes, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedDstSizes"
    >,
    InterfaceMethod<"Source-side strides, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedSrcStrides"
    >,
    InterfaceMethod<"Destination-side strides, as mixed static/dynamic entries",
    "::llvm::SmallVector<::mlir::OpFoldResult>", "getMixedDstStrides"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}

// See `AttrBasedSymbolVisibility` in AIRDialect.h. Use alongside `Symbol` and
// `SymbolName`.
def air_AttrBasedSymbolVisibility : NativeOpTrait<"AttrBasedSymbolVisibility"> {
  string cppNamespace = "::xilinx::air";
}

#endif
