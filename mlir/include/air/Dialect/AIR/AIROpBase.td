#ifndef AIR_OPS_BASE
#define AIR_OPS_BASE

include "mlir/IR/OpBase.td"

include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.td"

def air_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "xilinx::air";
}

def air_AnyScalarOrTensor : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                           AnyFloat.predicate,
                                           AnyTensor.predicate]>,
                                          "scalar-or-tensor">;

def air_AnyScalar : TypeConstraint<Or<[AnySignlessInteger.predicate,
                                   AnyFloat.predicate]>,
                                 "scalar">;

def air_AnyOptionalTensor : AnyTypeOf<[
  AnyTensor,
  Basicpy_NoneType
], "tensor or None">;

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
    "Value", "getLength"
    >,
    InterfaceMethod<"description",
    "int32_t", "getId"
    >,
  ];
}
#endif
