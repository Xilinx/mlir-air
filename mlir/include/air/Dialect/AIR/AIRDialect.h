// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
//===- AIRDialect.h - Dialect definition for the AIR IR ----------------===//
//
// Copyright 2020 Xilinx
//
//===---------------------------------------------------------------------===//

#ifndef MLIR_AIR_DIALECT_H
#define MLIR_AIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "npcomp/Dialect/Basicpy/IR/BasicpyDialect.h"

#include <map>
#include "Util.h"

using namespace mlir;

namespace xilinx {
namespace air {

void registerAIRRtTranslations();

struct HerdDim2 {
  Value x;
  Value y;
};

class AsyncTokenType
    : public Type::TypeBase<AsyncTokenType, Type, TypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
};

// Adds a `air.async.token` to the front of the argument list.
void addAsyncDependency(Operation *op, Value token);

}
}

#include "AIROpInterfaces.h.inc"
#include "AIRDialect.h.inc"
#include "AIREnums.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "AIR.h.inc"

#endif
