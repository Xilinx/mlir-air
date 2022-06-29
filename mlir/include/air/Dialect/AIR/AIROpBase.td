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

    ### Herds
    A *herd* in the `air` dialect is a one or two dimensional array of adjacent
    AIE cores executing the same code region.

    ### Memory Hierarchy
    There are three levels of memory in the air dialect:
    - L1 memory corresponds to the AIE tile memory.
    - L2 memory corresponds to pools of on-device memory such as of URAMs,
      BRAMs, or other memory resources.
    - L3 memory corresponds to off-chip memory such as DDR.

    The individual cores of a herd have direct access to L1 memory with
    standard load and store operations. Access to all other levels of memory
    is through DMA copy operations only.

    ### Asynchronous operations
    Operations in air dialect can be asynchronous. An asynchronous token
    interface is provided to synchronize between operations.

  }];
  let useDefaultTypePrinterParser = 1;
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
       "OpResult", "getAsyncToken", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.asyncToken().template dyn_cast_or_null<OpResult>();
      }]
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
#endif
