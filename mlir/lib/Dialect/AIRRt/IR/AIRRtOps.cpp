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

#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

using namespace mlir;

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
  if (parser.parseRegion(*body, {}, false))
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
  if (parser.parseRegion(*body, {}, false))
    return failure();
  SegmentMetadataOp::ensureTerminator(*body, parser.getBuilder(),
                                      result.location);
  return success();
}

//===----------------------------------------------------------------------===//
// DmaMemcpyNdOp
//===----------------------------------------------------------------------===//

void DmaMemcpyNdOp::build(OpBuilder &b, OperationState &result,
                          TypeRange resultTypes, Value id, Value x, Value y,
                          Value memref, ArrayRef<OpFoldResult> offsets,
                          ArrayRef<OpFoldResult> lengths,
                          ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets, staticLengths, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicLengths, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(lengths, dynamicLengths, staticLengths);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  build(b, result, resultTypes, id, x, y, memref, dynamicOffsets,
        dynamicLengths, dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticLengths),
        b.getDenseI64ArrayAttr(staticStrides));
}

void DmaMemcpyNdOp::build(OpBuilder &b, OperationState &result,
                          TypeRange resultTypes, Value id, Value x, Value y,
                          Value memref, ValueRange offsets, ValueRange lengths,
                          ValueRange strides) {
  SmallVector<int64_t> allDynamic(offsets.size(), ShapedType::kDynamic);
  SmallVector<int64_t> allDynamicLengths(lengths.size(), ShapedType::kDynamic);
  SmallVector<int64_t> allDynamicStrides(strides.size(), ShapedType::kDynamic);
  build(b, result, resultTypes, id, x, y, memref, offsets, lengths, strides,
        b.getDenseI64ArrayAttr(allDynamic),
        b.getDenseI64ArrayAttr(allDynamicLengths),
        b.getDenseI64ArrayAttr(allDynamicStrides));
}

// Replace one of the three mixed lists in place. The static array and the
// matching variadic operand group must be updated together to stay consistent.
static void setMixedList(DmaMemcpyNdOp op, ArrayRef<OpFoldResult> values,
                         StringRef staticAttrName,
                         MutableOperandRange dynamicOperands) {
  SmallVector<int64_t> staticValues;
  SmallVector<Value> dynamicValues;
  dispatchIndexOpFoldResults(values, dynamicValues, staticValues);
  dynamicOperands.assign(dynamicValues);
  op->setAttr(staticAttrName, OpBuilder(op).getDenseI64ArrayAttr(staticValues));
}

void DmaMemcpyNdOp::setMixedOffsets(ArrayRef<OpFoldResult> offsets) {
  setMixedList(*this, offsets, getStaticOffsetsAttrName(), getOffsetsMutable());
}

void DmaMemcpyNdOp::setMixedLengths(ArrayRef<OpFoldResult> lengths) {
  setMixedList(*this, lengths, getStaticLengthsAttrName(), getLengthsMutable());
}

void DmaMemcpyNdOp::setMixedStrides(ArrayRef<OpFoldResult> strides) {
  setMixedList(*this, strides, getStaticStridesAttrName(), getStridesMutable());
}

LogicalResult DmaMemcpyNdOp::verify() {
  auto verifyList = [&](StringRef name, ArrayRef<int64_t> staticValues,
                        OperandRange dynamicValues) -> LogicalResult {
    if (staticValues.size() != kNumDims)
      return emitOpError("expected ")
             << kNumDims << " " << name << ", got " << staticValues.size();
    return verifyListOfOperandsOrIntegers(*this, name, kNumDims, staticValues,
                                          dynamicValues);
  };
  if (failed(verifyList("offsets", getStaticOffsets(), getOffsets())))
    return failure();
  if (failed(verifyList("lengths", getStaticLengths(), getLengths())))
    return failure();
  if (failed(verifyList("strides", getStaticStrides(), getStrides())))
    return failure();
  return success();
}

} // namespace airrt
} // namespace xilinx

#define GET_OP_CLASSES
#include "air/Dialect/AIRRt/AIRRtOps.cpp.inc"
