// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/TypeSwitch.h"
#include <iostream>
using namespace mlir;
using namespace xilinx::air;
//using namespace xilinx::xten;

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

static LogicalResult removeUnusedArguments(HerdOp op,
                                           PatternRewriter &rewriter) {
  SmallVector<Value, 32> newOperands;
  SmallVector<int, 32> newOperandsIdx;
  for (int i = 0, e = op.getNumKernelOperands(); i < e; i++) {
    auto arg = op.getKernelArgument(i);
    if (!arg.getUsers().empty()) {
      newOperands.push_back(op.getKernelOperand(i));
      newOperandsIdx.push_back(i);
    }
  }
  if (newOperands.size() == op.getNumKernelOperands())
    return failure();

  BlockAndValueMapping remap;
  auto newOp = rewriter.create<HerdOp>(
      op.getLoc(), op.getAsyncDependencies(), op.getSizeOperands(),
      newOperands, op->getNumResults() > 0);
  rewriter.setInsertionPointToStart(&newOp.body().front());
  remap.map(op.getSize()[0], newOp.getSize()[0]);
  remap.map(op.getSize()[1], newOp.getSize()[1]);
  remap.map(op.getIds()[0], newOp.getIds()[0]);
  remap.map(op.getIds()[1], newOp.getIds()[1]);

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
namespace xilinx {
namespace air {

void addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getValues<APInt>())
    sizes.push_back(size.getSExtValue());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}

void eraseAsyncDependency(Operation *op, unsigned index) {
  assert(index + 1 <= op->getNumOperands() && "Index out of range");
  op->eraseOperands(index);
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getValues<APInt>())
    sizes.push_back(size.getSExtValue());
  --sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
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

}
}

//
// LaunchOp
//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange asyncDependencies,
                     ValueRange sizes, ValueRange launchOperands, 
                     bool isAsync) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(air::AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));

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

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << "(";
  p.printOperands(getIds());
  p << ") in (";
  auto sizeArgs = getSize();
  auto sizeOpers = getSizeOperands();
  for (int i=0,e=getNumDims(); i<e; i++) {
    if (i) p << ", ";
    p << sizeArgs[i] << "=";
    p << sizeOpers[i];
  }
  p << ")";

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
          if (attr.getName() == OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
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
  if (nameAttr &&
      body().front().getOperations().size() == 1)
    return;
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                                    result.attributes);

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

  auto tokenType = xilinx::air::AsyncTokenType::get(parser.getBuilder().getContext());
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

  for (int i=0,e=kernelOperands.size(); i<e; i++) {
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

  if (!regionResult.hasValue()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(segmentSizes));
  return success();
}

void LaunchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add(removeUnusedArguments);
}

ArrayRef<BlockArgument> LaunchOp::getIds() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> LaunchOp::getSize() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange LaunchOp::getSizeOperands() {
  auto start = asyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned LaunchOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - getNumDims();
}

Value LaunchOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> LaunchOp::getKernelArguments() {
  return body().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument LaunchOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned LaunchOp::getNumDims() {
  auto size_attr_name = OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseIntElementsAttr>(size_attr_name);
  SmallVector<APInt, 4> segment_sizes{size_attr.begin(), size_attr.end()};
  return segment_sizes[1].getZExtValue();
}

//
// PartitionOp
//

void PartitionOp::build(OpBuilder &builder, OperationState &result,
                     ValueRange asyncDependencies,
                     ValueRange sizes, ValueRange partitionOperands, 
                     bool isAsync) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(air::AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(partitionOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(partitionOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));

  Region *r = result.addRegion();
  Block *body = new Block();
  for (Value v : sizes) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  for (Value v : partitionOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
}

void PartitionOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange sizes, ValueRange partitionOperands) {

  build(builder, result, {}, sizes, partitionOperands, false);
}

void PartitionOp::print(OpAsmPrinter &p) {

  p << ' ';
  auto nameAttr = (*this)->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }
  
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  
  if (getNumDims()){
    p << " unroll(";
    p.printOperands(getIds());
    p << ") in (";
    auto sizeArgs = getSize();
    auto sizeOpers = getSizeOperands();
    for (int i=0,e=getNumDims(); i<e; i++) {
      if (i) p << ", ";
      p << sizeArgs[i] << "=";
      p << sizeOpers[i];
    }
    p << ")";
  }

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
          if (attr.getName() == OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
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
  if (nameAttr &&
    body().front().getOperations().size() == 1)
  return;
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult PartitionOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                                    result.attributes);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  Type indexType = parser.getBuilder().getIndexType();

  auto tokenType = xilinx::air::AsyncTokenType::get(parser.getBuilder().getContext());
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

  for (int i=0,e=kernelOperands.size(); i<e; i++) {
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

  if (!regionResult.hasValue()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(segmentSizes));
  return success();
}

void PartitionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add(removeUnusedArguments);
}

ArrayRef<BlockArgument> PartitionOp::getIds() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> PartitionOp::getSize() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange PartitionOp::getSizeOperands() {
  auto start = asyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned PartitionOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - getNumDims();
}

Value PartitionOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> PartitionOp::getKernelArguments() {
  return body().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument PartitionOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned PartitionOp::getNumDims() {
  auto size_attr_name = OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseIntElementsAttr>(size_attr_name);
  SmallVector<APInt, 4> segment_sizes{size_attr.begin(), size_attr.end()};
  return segment_sizes[1].getZExtValue();
}

//
// LaunchHerdOp
//

void HerdOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange asyncDependencies, ValueRange sizes,
                         ValueRange launchOperands, bool isAsync) {

  result.addOperands(asyncDependencies);
  if (isAsync)
    result.addTypes(air::AsyncTokenType::get(builder.getContext()));
  result.addOperands(sizes);
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = sizes.size();
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));

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

void HerdOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange sizes, ValueRange launchOperands) {

  build(builder, result, {}, sizes, launchOperands);
}

void HerdOp::print(OpAsmPrinter &p) {

  p << ' ';

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    p.printSymbolName(nameAttr);
    p << ' ';
  }

  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << " tile (";
  p.printOperands(getIds());
  p << ") in (";
  auto sizeArgs = getSize();
  auto sizeOpers = getSizeOperands();
  for (int i=0,e=getNumDims(); i<e; i++) {
    if (i) p << ", ";
    p << sizeArgs[i] << "=";
    p << sizeOpers[i];
  }
  p << ")";

  if (getNumKernelOperands()) {
    auto args = getKernelArguments();
    p << " args(";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << getKernelOperand(i);
    }
    p << ") : ";
    for (int i=0,e=getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range((*this)->getAttrs(), [&](NamedAttribute attr) {
          if (attr.getName() == OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr())
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
  if (nameAttr &&
      body().front().getOperations().size() == 1)
    return;
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult HerdOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::Argument, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileSize;
  SmallVector<OpAsmParser::Argument, 4> tileSizeRef;

  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                                    result.attributes);

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

  auto tokenType = xilinx::air::AsyncTokenType::get(parser.getBuilder().getContext());
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

  for (int i=0,e=kernelOperands.size(); i<e; i++) {
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

  if (!regionResult.hasValue()) {
    if (!nameAttr)
      return failure();
    for (auto ta : tileArgs)
      body->addArgument(ta.type, result.location);
  }

  SmallVector<int32_t, 8> segmentSizes(3, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[1] = tileSize.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(segmentSizes));
  return success();
}

void HerdOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add(removeUnusedArguments);
}

ArrayRef<BlockArgument> HerdOp::getIds() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> HerdOp::getSize() {
  auto s = body().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange HerdOp::getSizeOperands() {
  auto start = asyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned HerdOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - 2;
}

Value HerdOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + 2 + i);
}

ArrayRef<BlockArgument> HerdOp::getKernelArguments() {
  return body().front().getArguments().drop_front(4);
}

BlockArgument HerdOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned HerdOp::getNumDims() {
  auto size_attr_name = OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseIntElementsAttr>(size_attr_name);
  SmallVector<APInt, 4> segment_sizes{size_attr.begin(), size_attr.end()};
  return segment_sizes[1].getZExtValue();
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
  for (auto &o : body().front().getOperations()) {
    if (auto stage = dyn_cast<air::PipelineStageOp>(o))
      stages.push_back(stage);
  }
  return stages;
}

//
// PipelineStageOp
//

ParseResult
PipelineStageOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::Argument, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseAssignmentList(kernelArguments, kernelOperands))
      return failure();
    if (parser.parseColonTypeList(types))
      return failure();
  }

  for (int i=0,e=kernelOperands.size(); i<e; i++) {
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
    auto args = body().front().getArguments();
    p << " args(";
    for (int i=0,e=getNumOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << getOperand(i);
    }
    p << ") : ";
    for (int i=0,e=getNumOperands(); i<e; i++) {
      if (i) p << ", ";
      p << getOperand(i).getType();
    }
  }

  p << " ";
  if ((*this)->getAttrs().size()) {
    p << "attributes ";
    p.printOptionalAttrDict((*this)->getAttrs());
    p << " ";
  }
  p.printRegion(body(), /*printEntryBlockArgs=*/false);

  if ((*this)->getNumResults())
    p << " : ";
  for (Type type : (*this)->getResultTypes())
    p.printType(type);
}

unsigned PipelineStageOp::getStageId() {
  auto stages = getOperation()->getParentOfType<HerdPipelineOp>().getStages();
  for (unsigned idx = 0; idx<stages.size(); idx++)
    if (stages[idx] == *this)
      return idx;
  llvm_unreachable("Could not find stage in parent");
  return -1;
}

//
// Asynchronous region
//

LogicalResult RegionOp::verify() {
  assert(getOperation()->getNumRegions() == 1 && "RegionOp has zero region!");
  assert(!body().empty() && "RegionOp should have non-empty body");

  return success();
}

#include "air/Dialect/AIR/AIROpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.cpp.inc"
