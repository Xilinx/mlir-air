//===- AIRRtDialect.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

using namespace mlir;

#include "air/Dialect/AIRRt/AIRRtOpsDialect.cpp.inc"

namespace xilinx::airrt {

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

static LogicalResult FoldWaitAll(WaitAllOp op, PatternRewriter &rewriter) {
  SmallVector<Value> operands = op->getOperands();
  // If wait all has no results, then it is blocking; else, it is not blocking
  // and is only used to join tokens.
  if ((op.getNumResults() == 0 && operands.empty()) ||
      (op.getNumResults() != 0 && op.use_empty())) {
    rewriter.eraseOp(op);
    return success();
  }

  // If an operand of a wait_all is another wait_all, then fold them into one.
  llvm::SetVector<Value> newOperands;
  newOperands.insert(operands.begin(), operands.end());
  for (auto i = newOperands.begin(), e = newOperands.end(); i != e; ++i) {
    auto wa = llvm::dyn_cast_if_present<WaitAllOp>(i->getDefiningOp());
    if (!wa)
      continue;
    newOperands.erase(i);
    newOperands.insert(wa.getOperands().begin(), wa.getOperands().end());
    rewriter.replaceOpWithNewOp<WaitAllOp>(op, op.getResultTypes(),
                                           newOperands.takeVector());
    return success();
  }

  return failure();
}

static LogicalResult FoldAlloc(AllocOp op, PatternRewriter &rewriter) {
  auto memref = op.getResult();
  if (!llvm::range_size(memref.getUsers())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

static LogicalResult FoldDealloc(DeallocOp op, PatternRewriter &rewriter) {
  auto memref = op.getOperand();
  if (llvm::range_size(memref.getUsers()) == 1) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

void WaitAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldWaitAll);
}

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  patterns.add(FoldAlloc);
}

void DeallocOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldDealloc);
}

} // namespace xilinx::airrt
