//===- AIRDialect.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#include "air/Dialect/AIR/AIRDialect.cpp.inc"

void airDialect::initialize() {
  addTypes<AsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/AIR/AIR.cpp.inc"
      >();
}

Type airDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'async token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown air type: " + keyword);
  return Type();
}

void airDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'air' type"); });
}

template <class T>
static LogicalResult canonicalizeHierarchyOpArgs(T op,
                                                 PatternRewriter &rewriter) {

  // make a list of new hierarchy operands
  SmallVector<Value> newOperands;
  SmallVector<int> newOperandsIdx;
  for (int i = 0, e = op.getNumKernelOperands(); i < e; i++) {
    auto arg = op.getKernelArgument(i);
    // don't include unused operands
    if (arg.getUsers().empty())
      continue;
    newOperands.push_back(op.getKernelOperand(i));
    newOperandsIdx.push_back(i);
  }

  // make a list of new async token operands
  SmallVector<Value> newAsyncDeps;
  for (auto v : op.getAsyncDependencies()) {
    // don't include duplicates
    if (std::find(std::begin(newAsyncDeps), std::end(newAsyncDeps), v) !=
        std::end(newAsyncDeps))
      continue;
    // don't include wait_all ops with no operands
    if (auto wa = dyn_cast_if_present<WaitAllOp>(v.getDefiningOp()))
      if (wa.getAsyncDependencies().size() == 0)
        continue;
    newAsyncDeps.push_back(v);
  }

  // if the operands won't change, return
  if (newOperands.size() == op.getNumKernelOperands() &&
      newAsyncDeps.size() == op.getAsyncDependencies().size())
    return failure();

  IRMapping remap;
  auto newOp =
      rewriter.create<T>(op.getLoc(), newAsyncDeps, op.getSizeOperands(),
                         newOperands, op->getNumResults() > 0, op->getAttrs());

  rewriter.setInsertionPointToStart(&newOp.getBody().front());
  for (auto p : llvm::zip(op.getSize(), newOp.getSize()))
    remap.map(std::get<0>(p), std::get<1>(p));
  for (auto p : llvm::zip(op.getIds(), newOp.getIds()))
    remap.map(std::get<0>(p), std::get<1>(p));

  int newIdx = 0;
  for (int i : newOperandsIdx)
    remap.map(op.getKernelArgument(i), newOp.getKernelArgument(newIdx++));

  auto &ops = op.getBody().front().getOperations();
  for (auto oi = ops.begin(), oe = --ops.end(); oi != oe; ++oi)
    rewriter.clone(*oi, remap);

  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//

void air::addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.asArrayRef())
    sizes.push_back(size);
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getDenseI32ArrayAttr(sizes));
}

void air::eraseAsyncDependency(Operation *op, unsigned index) {
  assert(index + 1 <= op->getNumOperands() && "Index out of range");
  op->eraseOperands(index);
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.asArrayRef())
    sizes.push_back(size);
  --sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getDenseI32ArrayAttr(sizes));
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependenciesUnsorted) {

  if (asyncTokenType)
    printer << "async ";
  if (asyncDependenciesUnsorted.empty())
    return;

  // The values can be sorted by their order in a basic block, but only if they
  // all have defining ops in the same basic block. We go through all the
  // values, and check that they have defining ops in the same block.
  bool canSort = [&]() {
    auto v0 = asyncDependenciesUnsorted[0];
    if (!v0.getDefiningOp())
      return false;
    auto block = v0.getDefiningOp()->getBlock();
    for (auto v : asyncDependenciesUnsorted) {
      auto op = v.getDefiningOp();
      if (!op)
        return false;
      auto b = op->getBlock();
      if (b != block)
        return false;
    }
    return true;
  }();

  printer << "[";

  if (!canSort) {
    llvm::interleaveComma(asyncDependenciesUnsorted, printer);
  } else {
    SmallVector<Value> asyncDependencies(asyncDependenciesUnsorted);
    llvm::sort(asyncDependencies, [&](Value a, Value b) {
      return a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
    });
    llvm::interleaveComma(asyncDependencies, printer);
  }
  printer << "] ";
}

//
// LaunchOp
//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange asyncDependencies, ValueRange sizes,
                     ValueRange launchOperands, bool isAsync,
                     ArrayRef<NamedAttribute> attrs) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));

  for (auto attr : attrs)
    if (attr.getName() == getOperandSegmentSizeAttr())
      continue;
    else
      result.addAttribute(attr.getName(), attr.getValue());

  Region *r = result.addRegion();
  Block *body = new Block();
  for (Value v : sizes) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  for (Value v : launchOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
  LaunchOp::ensureTerminator(*r, builder, result.location);
}

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange sizes, ValueRange launchOperands) {

  build(builder, result, {}, sizes, launchOperands, false);
}

void LaunchOp::print(OpAsmPrinter &p) {

  p << ' ';

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this,
                         (getAsyncToken() ? getAsyncToken().getType() : Type()),
                         getAsyncDependencies());
  p << "(";
  p.printOperands(getIds());
  p << ") in (";
  auto sizeArgs = getSize();
  auto sizeOpers = getSizeOperands();
  for (int i = 0, e = getNumDims(); i < e; i++) {
    if (i)
      p << ", ";
    p << sizeArgs[i] << "=";
    p << sizeOpers[i];
  }
  p << ")";

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
        if (attr.getName() == getOperandSegmentSizeAttr())
          return false;
        if (attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          return false;
        return true;
      }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  if (nameAttr && getBody().front().getOperations().size() == 1)
    return;
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  if (parser.parseArgumentList(tileArgs, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();

  tileSize.resize(tileArgs.size());
  tileSizeRef.resize(tileArgs.size());
  for (unsigned i = 0; i < tileArgs.size(); ++i) {
    if (parser.parseArgument(tileSizeRef[i]) || parser.parseEqual() ||
        parser.parseOperand(tileSize[i]))
      return failure();
    (void)parser.parseOptionalComma();
  }

  if (parser.parseRParen())
    return failure();

  Type indexType = parser.getBuilder().getIndexType();

  tileArgs.append(tileSizeRef);
  for (auto &a : tileArgs)
    a.type = indexType;

  auto tokenType = AsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependencies, tokenType, result.operands))
    return failure();
  if (parser.resolveOperands(tileSize, indexType, result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::Argument argument;
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseArgument(argument) || parser.parseEqual() ||
            parser.parseOperand(operand))
          return failure();
        kernelArguments.push_back(argument);
        kernelOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
      if (parser.parseColonTypeList(types))
        return failure();
    }
  }

  for (int i = 0, e = kernelOperands.size(); i < e; i++) {
    kernelArguments[i].type = types[i];
    tileArgs.push_back(kernelArguments[i]);
    if (parser.resolveOperand(kernelOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();

  auto regionResult = parser.parseOptionalRegion(*body, tileArgs);
  LaunchOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  if (!regionResult.has_value()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void LaunchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<LaunchOp>);
}

ArrayRef<BlockArgument> LaunchOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> LaunchOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange LaunchOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned LaunchOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

Value LaunchOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> LaunchOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument LaunchOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned LaunchOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

//
// SegmentOp
//

void SegmentOp::build(OpBuilder &builder, OperationState &result,
                      ValueRange asyncDependencies, ValueRange sizes,
                      ValueRange segmentOperands, bool isAsync,
                      ArrayRef<NamedAttribute> attrs) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(segmentOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(segmentOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));

  for (auto attr : attrs)
    if (attr.getName() == getOperandSegmentSizeAttr())
      continue;
    else
      result.addAttribute(attr.getName(), attr.getValue());

  Region *r = result.addRegion();
  Block *body = new Block();
  for (Value v : sizes) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  for (Value v : segmentOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
  SegmentOp::ensureTerminator(*r, builder, result.location);
}

void SegmentOp::build(OpBuilder &builder, OperationState &result,
                      ValueRange sizes, ValueRange segmentOperands) {

  build(builder, result, {}, sizes, segmentOperands, false);
}

void SegmentOp::print(OpAsmPrinter &p) {

  p << ' ';
  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this,
                         (getAsyncToken() ? getAsyncToken().getType() : Type()),
                         getAsyncDependencies());

  if (getNumDims()) {
    p << " unroll(";
    p.printOperands(getIds());
    p << ") in (";
    auto sizeArgs = getSize();
    auto sizeOpers = getSizeOperands();
    for (int i = 0, e = getNumDims(); i < e; i++) {
      if (i)
        p << ", ";
      p << sizeArgs[i] << "=";
      p << sizeOpers[i];
    }
    p << ")";
  }

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
        if (attr.getName() == getOperandSegmentSizeAttr())
          return false;
        if (attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          return false;
        return true;
      }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  if (nameAttr && getBody().front().getOperations().size() == 1)
    return;
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult SegmentOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  Type indexType = parser.getBuilder().getIndexType();

  auto tokenType = AsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependencies, tokenType, result.operands))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("unroll"))) {
    if (parser.parseArgumentList(tileArgs, OpAsmParser::Delimiter::Paren) ||
        parser.parseKeyword("in") || parser.parseLParen())
      return failure();

    tileSize.resize(tileArgs.size());
    tileSizeRef.resize(tileArgs.size());
    for (unsigned i = 0; i < tileArgs.size(); ++i) {
      if (parser.parseArgument(tileSizeRef[i]) || parser.parseEqual() ||
          parser.parseOperand(tileSize[i]))
        return failure();
      (void)parser.parseOptionalComma();
    }

    if (parser.parseRParen())
      return failure();

    tileArgs.append(tileSizeRef);
    for (auto &a : tileArgs)
      a.type = indexType;

    if (parser.resolveOperands(tileSize, indexType, result.operands))
      return failure();
  }

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::Argument argument;
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseArgument(argument) || parser.parseEqual() ||
            parser.parseOperand(operand))
          return failure();
        kernelArguments.push_back(argument);
        kernelOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
      if (parser.parseColonTypeList(types))
        return failure();
    }
  }

  for (int i = 0, e = kernelOperands.size(); i < e; i++) {
    kernelArguments[i].type = types[i];
    tileArgs.push_back(kernelArguments[i]);
    if (parser.resolveOperand(kernelOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  auto regionResult = parser.parseOptionalRegion(*body, tileArgs);
  SegmentOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  if (!regionResult.has_value()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void SegmentOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<SegmentOp>);
}

ArrayRef<BlockArgument> SegmentOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> SegmentOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange SegmentOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned SegmentOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

Value SegmentOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> SegmentOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument SegmentOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned SegmentOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

//
// HerdOp
//

void HerdOp::build(OpBuilder &builder, OperationState &result,
                   ValueRange asyncDependencies, ValueRange sizes,
                   ValueRange launchOperands, bool isAsync,
                   ArrayRef<NamedAttribute> attrs) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));

  for (auto attr : attrs)
    if (attr.getName() == getOperandSegmentSizeAttr())
      continue;
    else
      result.addAttribute(attr.getName(), attr.getValue());

  Region *r = result.addRegion();
  Block *body = new Block();
  for (Value v : sizes) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  for (Value v : launchOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
  HerdOp::ensureTerminator(*r, builder, result.location);
}

void HerdOp::build(OpBuilder &builder, OperationState &result, ValueRange sizes,
                   ValueRange launchOperands) {

  build(builder, result, {}, sizes, launchOperands);
}

void HerdOp::print(OpAsmPrinter &p) {

  p << ' ';

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this,
                         (getAsyncToken() ? getAsyncToken().getType() : Type()),
                         getAsyncDependencies());
  p << " tile (";
  p.printOperands(getIds());
  p << ") in (";
  auto sizeArgs = getSize();
  auto sizeOpers = getSizeOperands();
  for (int i = 0, e = getNumDims(); i < e; i++) {
    if (i)
      p << ", ";
    p << sizeArgs[i] << "=";
    p << sizeOpers[i];
  }
  p << ")";

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i = 0, e = getNumKernelOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
        if (attr.getName() == getOperandSegmentSizeAttr())
          return false;
        if (attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          return false;
        return true;
      }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  if (nameAttr && getBody().front().getOperations().size() == 1)
    return;
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

ParseResult HerdOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  if (parser.parseKeyword("tile"))
    return failure();

  if (parser.parseArgumentList(tileArgs, OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();

  tileSize.resize(tileArgs.size());
  tileSizeRef.resize(tileArgs.size());
  for (unsigned i = 0; i < tileArgs.size(); ++i) {
    if (parser.parseArgument(tileSizeRef[i]) || parser.parseEqual() ||
        parser.parseOperand(tileSize[i]))
      return failure();
    (void)parser.parseOptionalComma();
  }

  if (parser.parseRParen())
    return failure();

  Type indexType = parser.getBuilder().getIndexType();

  tileArgs.append(tileSizeRef);
  for (auto &a : tileArgs)
    a.type = indexType;

  auto tokenType = AsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependencies, tokenType, result.operands))
    return failure();
  if (parser.resolveOperands(tileSize, indexType, result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::Argument argument;
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseArgument(argument) || parser.parseEqual() ||
            parser.parseOperand(operand))
          return failure();
        kernelArguments.push_back(argument);
        kernelOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
      if (parser.parseColonTypeList(types))
        return failure();
    }
  }

  for (int i = 0, e = kernelOperands.size(); i < e; i++) {
    kernelArguments[i].type = types[i];
    tileArgs.push_back(kernelArguments[i]);
    if (parser.resolveOperand(kernelOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();

  auto regionResult = parser.parseOptionalRegion(*body, tileArgs);
  HerdOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  if (!regionResult.has_value()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void HerdOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<HerdOp>);
}

ArrayRef<BlockArgument> HerdOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> HerdOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange HerdOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned HerdOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

Value HerdOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> HerdOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(4);
}

BlockArgument HerdOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned HerdOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

uint64_t HerdOp::getNumCols() {
  auto cols = getSizeOperands()[0].getDefiningOp();
  return cast<arith::ConstantIndexOp>(cols).value();
}

uint64_t HerdOp::getNumRows() {
  auto rows = getSizeOperands()[1].getDefiningOp();
  return cast<arith::ConstantIndexOp>(rows).value();
}

//
// Asynchronous execute
//

LogicalResult ExecuteOp::verify() {
  assert(getOperation()->getNumRegions() == 1 && "ExecuteOp has zero region!");
  assert(!getBody().empty() && "ExecuteOp should have non-empty body");

  return success();
}

static LogicalResult FoldExecute(ExecuteOp op, PatternRewriter &rewriter) {

  // if the terminator is the only thing in the ExecuteOp,
  // and the op is unused, then it can be removed.
  auto &body = op.getRegion().front();
  auto et = body.getTerminator();
  if (op.use_empty() && body.getOperations().size() == 1) {
    rewriter.eraseOp(op);
    return success();
  }

  // replace returns of constants with the constant
  int idx = 0;
  for (auto v : et->getOperands()) {
    idx++;
    if (op.getResult(idx).use_empty())
      continue;
    auto o = v.getDefiningOp();
    if (!o)
      continue;
    if (isa<arith::ConstantOp>(o)) {
      op.getResult(idx).replaceAllUsesWith(rewriter.clone(*o)->getResult(0));
      return success();
    }
  }

  // if any of the results are used, return failure()
  for (auto v : op->getResults().drop_front())
    if (!v.use_empty())
      return failure();

  // if we get here then only the async token result has uses.

  // if there are extra results than async token, and none of them are used,
  // then replace the execute with a wait_all no-op.
  if (op->getNumResults() > 1) {
    op.getResult(0).replaceAllUsesWith(
        rewriter
            .create<WaitAllOp>(op->getLoc(), op->getResult(0).getType(),
                               op->getOperands())
            .getResult(0));
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

void ExecuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldExecute);
}

Operation *ExecuteOp::getChildOp() {
  auto child_op =
      &getOperation()->getRegion(0).getBlocks().front().getOperations().front();
  return child_op;
}

//
// WaitAllOp
//

static LogicalResult FoldWaitAll(WaitAllOp op, PatternRewriter &rewriter) {
  SmallVector<Value> operands;
  for (auto o : op->getOperands())
    if (std::find(operands.begin(), operands.end(), o) == std::end(operands))
      operands.push_back(o);

  // Erase wait_all with no operands and no uses
  if (op.use_empty() && !operands.size()) {
    rewriter.eraseOp(op);
    return success();
  }

  // remove duplicate operands
  if (op->getNumOperands() != operands.size()) {
    rewriter.replaceOpWithNewOp<WaitAllOp>(op, op.getResultTypes(), operands);
    return success();
  }

  // If an operand of a wait_all is a wait_all without operands,
  // then we can remove it from the operand list.
  for (auto i = operands.begin(), e = operands.end(); i != e; ++i) {
    auto wa = llvm::dyn_cast_if_present<WaitAllOp>(i->getDefiningOp());
    if (!wa)
      continue;
    if (wa->getNumOperands())
      continue;
    operands.erase(i);
    rewriter.replaceOpWithNewOp<WaitAllOp>(op, op.getResultTypes(), operands);
    return success();
  }

  // If async wait_all has a single operand, forward it to any uses
  if (op.getAsyncDependencies().size() == 1 && op.getResults().size() == 1) {
    rewriter.replaceOp(op, op.getAsyncDependencies()[0]);
    return success();
  }

  return failure();
}

void WaitAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldWaitAll);
}

// Get strides from MemRefType.
static SmallVector<Value> extractStridesFromMemrefType(MemRefType memrefTy,
                                                       OpBuilder &builder) {
  SmallVector<Value> strides;
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides = getStridesAndOffset(memrefTy, layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return strides;
  }

  for (auto s : layout_strides)
    strides.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));

  return strides;
}

// Get sizes from MemRefType.
static SmallVector<Value> extractSizesFromMemrefType(MemRefType memrefTy,
                                                     OpBuilder &builder) {
  SmallVector<Value> sizes;
  for (auto s : memrefTy.getShape())
    sizes.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));
  return sizes;
}

// Get offsets from memref::SubviewOp.
static void extractOffsetsFromSubview(memref::SubViewOp subview,
                                      OpBuilder &builder,
                                      SmallVector<Value> &offsets) {
  auto subview_offsets = subview.getOffsets().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto loc = subview.getLoc();

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
}

static LogicalResult canonicalizeAIRDmaOperands(OpBuilder builder,
                                                SmallVector<Value> &offsets,
                                                SmallVector<Value> &sizes,
                                                SmallVector<Value> &strides,
                                                MemRefType memref) {
  // Increase vector sizes up to memref size. When offsets, sizes and strides
  // are all empty, then it implies that the whole memref is accessed in the
  // default order.
  auto max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  auto target_dim_size = std::max(max_dim_size, (size_t)memref.getRank());
  if (max_dim_size && offsets.size() < target_dim_size) {
    for (unsigned i = offsets.size(); i < target_dim_size; i++) {
      offsets.insert(offsets.begin(), builder.create<arith::ConstantIndexOp>(
                                          builder.getUnknownLoc(), 0));
    }
  }
  if (max_dim_size && sizes.size() < target_dim_size) {
    for (unsigned i = sizes.size(); i < target_dim_size; i++) {
      sizes.insert(sizes.begin(), builder.create<arith::ConstantIndexOp>(
                                      builder.getUnknownLoc(), 1));
    }
  }
  int memref_size = 1;
  for (auto size : memref.getShape())
    memref_size *= size;
  if (max_dim_size && strides.size() < target_dim_size) {
    for (unsigned i = strides.size(); i < target_dim_size; i++) {
      strides.insert(strides.begin(),
                     builder.create<arith::ConstantIndexOp>(
                         builder.getUnknownLoc(), memref_size));
    }
  }

  // Reduce highest dimensions if more than memref size
  while (strides.size() > target_dim_size && getConstantIntValue(strides[0]) &&
         *getConstantIntValue(strides[0]) == memref_size) {
    strides.erase(strides.begin());
  }
  while (sizes.size() > target_dim_size && getConstantIntValue(sizes[0]) &&
         *getConstantIntValue(sizes[0]) == 1) {
    sizes.erase(sizes.begin());
  }
  while (offsets.size() > std::min(sizes.size(), strides.size()) &&
         getConstantIntValue(offsets[0]) &&
         *getConstantIntValue(offsets[0]) == 0) {
    offsets.erase(offsets.begin());
  }

  if (offsets.size() != sizes.size() || sizes.size() != strides.size())
    return failure();

  return success();
}

static LogicalResult ComposeMemrefOp(Value memref, PatternRewriter &rewriter,
                                     Value &input_memref,
                                     SmallVector<Value> &offsets,
                                     SmallVector<Value> &sizes,
                                     SmallVector<Value> &strides) {

  auto memref_type = llvm::dyn_cast<MemRefType>(memref.getType());
  if (!memref_type)
    return failure();
  auto defop = memref.getDefiningOp();
  if (!defop)
    return failure();
  auto loc = defop->getLoc();

  // Get a chain of memref ops that produce the memref consumed by the memcpy
  // op.
  std::vector<Operation *> memrefOpVec;
  bool exit = false;
  while (defop && !exit) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(defop)) {
      memrefOpVec.push_back(defop);
      defop = transposeOp.getIn().getDefiningOp();
    } else if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(defop)) {
      memrefOpVec.push_back(defop);
      defop = viewLikeOp.getViewSource().getDefiningOp();
    } else
      exit = true;
  }
  if (memrefOpVec.empty())
    return failure();

  // Revert the vector of memref ops, as it was built with push_back.
  std::reverse(memrefOpVec.begin(), memrefOpVec.end());

  // Init. memref type and offsets at the front of the vector of memref ops.
  auto constZero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  MemRefType input_ty;
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOpVec[0])) {
    // Init. offsets
    extractOffsetsFromSubview(subviewOp, rewriter, offsets);
    // Init. memref type
    input_ty = subviewOp.getSourceType();
    input_memref = subviewOp.getViewSource();
  } else if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOpVec[0])) {
    // Init. memref type
    input_ty = llvm::cast<MemRefType>(transposeOp.getIn().getType());
    input_memref = transposeOp.getIn();
    // Init. offsets
    offsets.clear();
    for (unsigned i = 0; i < transposeOp.getPermutation().getNumInputs(); i++)
      offsets.push_back(constZero);
  } else if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(memrefOpVec[0])) {
    // Init. memref type
    input_ty = llvm::cast<MemRefType>(viewLikeOp.getViewSource().getType());
    input_memref = viewLikeOp.getViewSource();
    // Init. offsets
    offsets.clear();
    for (unsigned i = 0; i < input_ty.getRank(); i++)
      offsets.push_back(constZero);
  } else
    return failure();

  // Compose memref type as the memref propagates through the chain of memref
  // ops.
  for (auto memrefOp : memrefOpVec) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      if (transposeOp.getPermutation().getNumInputs() != offsets.size())
        continue;
      offsets =
          applyPermutationMap<Value>(transposeOp.getPermutation(), offsets);
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      // Init. offsets
      for (int i = (int)expandShapeOp.getReassociationIndices().size() - 1;
           i >= 0; i--) {
        if (expandShapeOp.getReassociationIndices()[i].size() <= 1)
          continue;
        for (unsigned j = 1;
             j < expandShapeOp.getReassociationIndices()[i].size(); j++)
          offsets.insert(offsets.begin() + i,
                         rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      if (subviewOp != memrefOpVec.front() && !subviewOp.hasZeroOffset())
        subviewOp->emitOpError(
            "is not the source op in a chain of memref layout transformation "
            "ops, but applies a non-zero offset. This feature is NYI, and "
            "leads to unexpected behaviour.");
    }
  }

  // Memref type at sink memref op.
  input_ty =
      llvm::cast<MemRefType>(memrefOpVec.back()->getResultTypes().front());

  // Compose sizes and strides from the output memref type's layout.
  strides = extractStridesFromMemrefType(input_ty, rewriter);
  sizes = extractSizesFromMemrefType(input_ty, rewriter);

  return canonicalizeAIRDmaOperands(rewriter, offsets, sizes, strides,
                                    input_ty);
}

//
// Dma op
//

static LogicalResult
ComposeMemrefOpOnDmaMemcpyNdSrc(DmaMemcpyNdOp op, PatternRewriter &rewriter) {

  Value memref = op.getSrcMemref();
  if (!memref)
    return failure();
  auto loc = op->getLoc();
  Value input_memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getSrcOffsets();
  if (!offsets.empty())
    return failure();
  sizes = op.getSrcSizes();
  if (!sizes.empty())
    return failure();
  strides = op.getSrcStrides();
  if (!strides.empty())
    return failure();

  if (failed(ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes,
                             strides))) {
    return failure();
  }

  auto newOp = rewriter.create<air::DmaMemcpyNdOp>(
      loc, op->getResultTypes(), op.getAsyncDependencies(), op.getDstMemref(),
      op.getDstOffsets(), op.getDstSizes(), op.getDstStrides(), input_memref,
      offsets, sizes, strides);

  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
  }

  rewriter.eraseOp(op);

  return success();
}

static LogicalResult
ComposeMemrefOpOnDmaMemcpyNdDst(DmaMemcpyNdOp op, PatternRewriter &rewriter) {

  Value memref = op.getDstMemref();
  if (!memref)
    return failure();
  auto loc = op->getLoc();
  Value input_memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getDstOffsets();
  if (!offsets.empty())
    return failure();
  sizes = op.getDstSizes();
  if (!sizes.empty())
    return failure();
  strides = op.getDstStrides();
  if (!strides.empty())
    return failure();

  if (failed(ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes,
                             strides))) {
    return failure();
  }

  auto newOp = rewriter.create<air::DmaMemcpyNdOp>(
      loc, op->getResultTypes(), op.getAsyncDependencies(), input_memref,
      offsets, sizes, strides, op.getSrcMemref(), op.getSrcOffsets(),
      op.getSrcSizes(), op.getSrcStrides());

  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
  }

  rewriter.eraseOp(op);

  return success();
}

void DmaMemcpyNdOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnDmaMemcpyNdSrc);
  patterns.add(ComposeMemrefOpOnDmaMemcpyNdDst);
}

//
// Channel put op
//

static LogicalResult ComposeMemrefOpOnChannelPut(ChannelPutOp op,
                                                 PatternRewriter &rewriter) {

  Value memref = op.getMemref();
  if (!memref)
    return failure();
  auto loc = op->getLoc();

  // Init. memref type and offsets from memref's defining op's input type
  Value input_memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getOffsets();
  if (!offsets.empty())
    return failure();
  sizes = op.getSizes();
  if (!sizes.empty())
    return failure();
  strides = op.getStrides();
  if (!strides.empty())
    return failure();

  if (failed(ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes,
                             strides))) {
    return failure();
  }

  auto newOp = rewriter.create<air::ChannelPutOp>(
      loc, op->getResultTypes(), op.getAsyncDependencies(), op.getChanName(),
      op.getIndices(), input_memref, offsets, sizes, strides);

  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
  }

  rewriter.eraseOp(op);

  return success();
}

void ChannelPutOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnChannelPut);
}

//
// Channel get op
//

static LogicalResult ComposeMemrefOpOnChannelGet(ChannelGetOp op,
                                                 PatternRewriter &rewriter) {

  Value memref = op.getMemref();
  if (!memref)
    return failure();
  auto loc = op->getLoc();

  // Init. memref type and offsets from memref's defining op's input type
  Value input_memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getOffsets();
  if (!offsets.empty())
    return failure();
  sizes = op.getSizes();
  if (!sizes.empty())
    return failure();
  strides = op.getStrides();
  if (!strides.empty())
    return failure();

  if (failed(ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes,
                             strides))) {
    return failure();
  }

  auto newOp = rewriter.create<air::ChannelGetOp>(
      loc, op->getResultTypes(), op.getAsyncDependencies(), op.getChanName(),
      op.getIndices(), input_memref, offsets, sizes, strides);

  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
  }

  rewriter.eraseOp(op);

  return success();
}

void ChannelGetOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnChannelGet);
}

//
// Channel op
//

LogicalResult ChannelOp::verify() {
  if (isBroadcast()) {
    auto bundle_size = getSize();
    auto broadcast_shape = getBroadcastShape();
    if (bundle_size.size() != broadcast_shape.size())
      return emitOpError("bundle rank should match broadcast_shape rank");
  }
  return success();
}

int ChannelOp::getBroadcastDimension() {
  int broadcastDim = -1;
  auto bundle_size = getSize();
  auto broadcast_shape = getBroadcastShape();
  if (isBroadcast()) {
    for (int i = 0; i < (int)bundle_size.size(); i++) {
      if (dyn_cast<IntegerAttr>(bundle_size[i]).getInt() !=
          dyn_cast<IntegerAttr>(broadcast_shape[i]).getInt()) {
        broadcastDim = i;
        break;
      }
    }
  }
  return broadcastDim;
}

static LogicalResult FoldChannel(ChannelOp op, PatternRewriter &rewriter) {

  Operation *parent = op;
  std::vector<Operation *> parent_sts;
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      parent_sts.push_back(parent);
    }
  }
  if (parent_sts.empty()) {
    return failure();
  }
  for (auto st : parent_sts) {
    if (mlir::SymbolTable::lookupSymbolIn(st, op.getSymName())) {
      auto attr =
          op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

      std::vector<ChannelPutOp> puts;
      std::vector<ChannelGetOp> gets;
      st->walk([&](Operation *o) {
        if (auto put = dyn_cast<air::ChannelPutOp>(o)) {
          if (put.getChanName() == attr) {
            puts.push_back(put);
          }
        } else if (auto get = dyn_cast<air::ChannelGetOp>(o)) {
          if (get.getChanName() == attr) {
            gets.push_back(get);
          }
        }
      });
      if (puts.empty() && gets.empty()) {
        rewriter.eraseOp(op);
        return success();
      }
    }
  }
  return failure();
}

void ChannelOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldChannel);
}

//
// Custom op
//

void CustomOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange asyncDependencies, ValueRange customOperands,
                     bool isAsync, ArrayRef<NamedAttribute> attrs) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(AsyncTokenType::get(builder.getContext()));
  result.addOperands(customOperands);

  SmallVector<int32_t, 8> segmentSizes(2, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = static_cast<int32_t>(customOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));

  for (auto attr : attrs)
    if (attr.getName() == getOperandSegmentSizeAttr())
      continue;
    else
      result.addAttribute(attr.getName(), attr.getValue());
}

void CustomOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange customOperands) {

  build(builder, result, {}, customOperands, false);
}

void CustomOp::print(OpAsmPrinter &p) {

  p << ' ';

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this,
                         (getAsyncToken() ? getAsyncToken().getType() : Type()),
                         getAsyncDependencies());

  if (!getCustomOperands().empty()) {
    auto operands = getCustomOperands();
    p << " operands (";
    for (int i = 0, e = getCustomOperands().size(); i < e; i++) {
      if (i)
        p << ", ";
      p << operands[i];
    }
    p << ") : ";
    for (int i = 0, e = getCustomOperands().size(); i < e; i++) {
      if (i)
        p << ", ";
      p << operands[i].getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
      llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
        if (attr.getName() == getOperandSegmentSizeAttr())
          return false;
        if (attr.getName() == mlir::SymbolTable::getSymbolAttrName())
          return false;
        return true;
      }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
}

ParseResult CustomOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  auto tokenType = AsyncTokenType::get(parser.getBuilder().getContext());
  if (parser.resolveOperands(asyncDependencies, tokenType, result.operands))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand, 4> customOperands;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("operands"))) {
    if (parser.parseLParen())
      return failure();
    if (parser.parseOptionalRParen()) {
      do {
        OpAsmParser::UnresolvedOperand operand;
        if (parser.parseOperand(operand))
          return failure();
        customOperands.push_back(operand);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
      if (parser.parseColonTypeList(types))
        return failure();
    }
  }

  for (int i = 0, e = customOperands.size(); i < e; i++) {
    if (parser.resolveOperand(customOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  SmallVector<int32_t, 8> segmentSizes(2, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = customOperands.size();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

#include "air/Dialect/AIR/AIROpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.cpp.inc"
