//===- AIRRtOps.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"

#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"

using namespace mlir;
using namespace xilinx::airrt;

namespace xilinx {
namespace airrt {

//===----------------------------------------------------------------------===//
// ModuleMetadataOp
//===----------------------------------------------------------------------===//

void ModuleMetadataOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  p.printRegion(getSegments(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult ModuleMetadataOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, std::nullopt, false))
    return failure();
  ModuleMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

void SegmentMetadataOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  p.printRegion(getHerds(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult SegmentMetadataOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, std::nullopt, false))
    return failure();
  SegmentMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                        result.location);
  return success();
}

} // namespace airrt
} // namespace xilinx

#define GET_OP_CLASSES
#include "air/Dialect/AIRRt/AIRRtOps.cpp.inc"
