// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

using namespace mlir;
using namespace xilinx::airrt;

#include "air/Dialect/AIRRt/AIRRtOpsDialect.cpp.inc"

void AIRRtDialect::initialize() {
  addTypes<EventType>();
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/AIRRt/AIRRtOps.cpp.inc"
      >();
  addTypes<TensorType>();
}

Type AIRRtDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'event' types.
  if (keyword == "event")
    return EventType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown airrt type: " + keyword);
  return Type();
}

void AIRRtDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<EventType>([&](Type) { os << "event"; })
      .Default([](Type) { llvm_unreachable("unexpected 'airrt' type"); });
}
