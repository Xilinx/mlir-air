//===- AIRDialect.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
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

  for (Operation &o : op.getRegion().front().getOperations())
    rewriter.clone(o, remap);

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
                                   OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async ";
  if (asyncDependencies.empty())
    return;
  printer << "[";
  llvm::interleaveComma(asyncDependencies, printer);
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
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
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
  ensureTerminator(*body, parser.getBuilder(), result.location);

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
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
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
  ensureTerminator(*body, parser.getBuilder(), result.location);

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
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
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
  ensureTerminator(*body, parser.getBuilder(), result.location);

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
// HerdPipelineOp
//

LogicalResult HerdPipelineOp::verify() {
  auto direction = (*this)->getAttrOfType<StringAttr>("direction");
  if (!direction)
    return emitOpError() << "expects 'direction' attribute";

  return success();
}

SmallVector<PipelineStageOp, 8> HerdPipelineOp::getStages() {
  SmallVector<PipelineStageOp, 8> stages;
  for (auto &o : getBody().front().getOperations()) {
    if (auto stage = dyn_cast<PipelineStageOp>(o))
      stages.push_back(stage);
  }
  return stages;
}

//
// PipelineStageOp
//

ParseResult PipelineStageOp::parse(OpAsmParser &parser,
                                   OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseAssignmentList(kernelArguments, kernelOperands))
      return failure();
    if (parser.parseColonTypeList(types))
      return failure();
  }

  for (int i = 0, e = kernelOperands.size(); i < e; i++) {
    kernelArguments[i].type = types[i];
    if (parser.resolveOperand(kernelOperands[i], types[i], result.operands))
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, kernelArguments, false))
    return failure();

  SmallVector<Type, 4> retTypes;
  if (parser.parseOptionalColon())
    return success();

  if (parser.parseTypeList(retTypes))
    return failure();

  result.addTypes(retTypes);
  return success();
}

void PipelineStageOp::print(OpAsmPrinter &p) {

  if (getNumOperands()) {
    auto args = getBody().front().getArguments();
    p << " args(";
    for (int i = 0, e = getNumOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << args[i] << "=";
      p << getOperand(i);
    }
    p << ") : ";
    for (int i = 0, e = getNumOperands(); i < e; i++) {
      if (i)
        p << ", ";
      p << getOperand(i).getType();
    }
  }

  p << " ";
  if ((*this)->getAttrs().size()) {
    p << "attributes ";
    p.printOptionalAttrDict((*this)->getAttrs());
    p << " ";
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  if ((*this)->getNumResults())
    p << " : ";
  for (Type type : (*this)->getResultTypes())
    p.printType(type);
}

unsigned PipelineStageOp::getStageId() {
  auto stages = getOperation()->getParentOfType<HerdPipelineOp>().getStages();
  for (unsigned idx = 0; idx < stages.size(); idx++)
    if (stages[idx] == *this)
      return idx;
  llvm_unreachable("Could not find stage in parent");
  return -1;
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
  // if the execute body is empty, replace the execute with a wait_all no-op
  if (body.getOperations().size() == 1) {
    op.getResult(0).replaceAllUsesWith(
        rewriter
            .create<WaitAllOp>(op->getLoc(), op->getResult(0).getType(),
                               op->getOperands())
            .getResult(0));
    return success();
  }

  return failure();
}

void ExecuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add(FoldExecute);
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

//
// Channel op
//

LogicalResult ChannelOp::verify() {
  if (isBroadcast()) {
    auto bundle_size = getSize();
    auto broadcast_shape = getBroadcastShape();
    if (bundle_size.size() != broadcast_shape.size())
      return emitOpError("bundle size should match broadcast_shape size");
    int diffDims = 0;
    int broadcastDim = -1;
    for (int i = 0; i < (int)bundle_size.size(); i++)
      if (dyn_cast<IntegerAttr>(bundle_size[i]).getInt() !=
          dyn_cast<IntegerAttr>(broadcast_shape[i]).getInt()) {
        diffDims++;
        broadcastDim = i;
      }
    if (diffDims > 1)
      return emitOpError("bundle sizes and broadcast_shape should only differ "
                         "along one dimension");
    if (dyn_cast<IntegerAttr>(bundle_size[broadcastDim]).getInt() != 1)
      return emitOpError("along the broadcast dimension the index in the "
                         "channel bundle sizes should be equal to 1");
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
  while (parent = parent->getParentOp()) {
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

  Type indexType = parser.getBuilder().getIndexType();

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
