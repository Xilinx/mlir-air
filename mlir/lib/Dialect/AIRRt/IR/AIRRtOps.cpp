//===- AIRRtOps.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
  p.printRegion(partitions(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult ModuleMetadataOp::parse(OpAsmParser &parser,
                                         OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, false))
    return failure();
  ModuleMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                     result.location);
  return success();
}

void PartitionMetadataOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());
  p.printRegion(herds(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult PartitionMetadataOp::parse(OpAsmParser &parser,
                                       OperationState &result) {
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, llvm::None, false))
    return failure();
  PartitionMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                        result.location);
  return success();
}

} // namespace airrt
} // namespace xilinx

#define GET_OP_CLASSES
#include "air/Dialect/AIRRt/AIRRtOps.cpp.inc"
