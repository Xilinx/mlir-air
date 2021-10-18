//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

//===----------------------------------------------------------------------===//
// ModuleMetadataOp
//===----------------------------------------------------------------------===//

static void printModuleMetadataOp(OpAsmPrinter &p, ModuleMetadataOp &op) {
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
  p.printRegion(op.herds(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseModuleMetadataOp(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();
  ModuleMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// HerdMetadataOp
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "air/Dialect/AIRRt/AIRRtOps.cpp.inc"
