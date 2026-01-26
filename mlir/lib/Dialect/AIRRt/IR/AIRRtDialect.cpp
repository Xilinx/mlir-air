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
    auto newOp = rewriter.replaceOpWithNewOp<WaitAllOp>(
        op, op.getResultTypes(), newOperands.takeVector());
    // Preserve attributes from the original operation (e.g., air.segment_end,
    // air.launch_end).
    newOp->setAttrs(op->getAttrs());
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

static LogicalResult FoldSegmentLoad(SegmentLoadOp op,
                                     PatternRewriter &rewriter) {
  // Look for a previous segment_load with the same sym_name in the same block.
  auto symName = op.getSymName();

  for (Operation *prevOp = op->getPrevNode(); prevOp;
       prevOp = prevOp->getPrevNode()) {
    // Check if prevOp itself is a SegmentLoadOp.
    if (auto prevSegmentLoad = dyn_cast<SegmentLoadOp>(prevOp)) {
      if (prevSegmentLoad.getSymName() == symName) {
        rewriter.replaceOp(op, prevSegmentLoad.getResults());
        return success();
      }
    }
  }
  return failure();
}

// Helper to check if two HerdLoadOps are equivalent.
static bool areHerdLoadsEquivalent(HerdLoadOp a, HerdLoadOp b) {
  if (a.getSymName() != b.getSymName())
    return false;
  if (a->getAttr("segment_name") != b->getAttr("segment_name"))
    return false;
  auto aRtp = a.getRtp();
  auto bRtp = b.getRtp();
  if (aRtp.size() != bRtp.size())
    return false;
  for (size_t i = 0; i < aRtp.size(); ++i) {
    if (aRtp[i] != bRtp[i])
      return false;
  }
  return true;
}

static LogicalResult FoldHerdLoad(HerdLoadOp op, PatternRewriter &rewriter) {
  // Look for a previous herd_load with the same sym_name, segment_name
  // attribute, and rtp operands in the same block.

  for (Operation *prevOp = op->getPrevNode(); prevOp;
       prevOp = prevOp->getPrevNode()) {
    // Check if prevOp itself is a HerdLoadOp.
    if (auto prevHerdLoad = dyn_cast<HerdLoadOp>(prevOp)) {
      if (areHerdLoadsEquivalent(op, prevHerdLoad)) {
        rewriter.replaceOp(op, prevHerdLoad.getResults());
        return success();
      }
    }
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

void SegmentLoadOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add(FoldSegmentLoad);
}

void HerdLoadOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add(FoldHerdLoad);
}

} // namespace xilinx::airrt
