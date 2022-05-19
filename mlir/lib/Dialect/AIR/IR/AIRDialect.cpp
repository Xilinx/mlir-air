// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"

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
// LaunchHerdOp
//

void HerdLaunchOp::build(OpBuilder &builder, OperationState &result,
                         HerdDim2 herdSize, ValueRange launchOperands) {

  result.addOperands({herdSize.x, herdSize.y});
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(4, 1);
  segmentSizes.front() = 0; // Initially no async dependencies.
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));

  Region *r = result.addRegion();
  Block *body = new Block();
  SmallVector<Type, 4> argtypes(4, builder.getIndexType());
  SmallVector<Location, 4> arglocs(4, builder.getUnknownLoc());
  body->addArguments(argtypes, arglocs);
  for (Value v : launchOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
}

void HerdLaunchOp::build(OpBuilder &builder, OperationState &result,
                         ValueRange asyncDependencies,
                         HerdDim2 herdSize, ValueRange launchOperands) {

  result.addOperands(asyncDependencies);
  result.addTypes(air::AsyncTokenType::get(builder.getContext()));
  result.addOperands({herdSize.x, herdSize.y});
  result.addOperands(launchOperands);

  SmallVector<int32_t, 8> segmentSizes(4, 1);
  segmentSizes.front() = 0; // Initially no async dependencies.
  segmentSizes.back() = static_cast<int32_t>(launchOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));

  Region *r = result.addRegion();
  Block *body = new Block();
  SmallVector<Type, 4> argtypes(4, builder.getIndexType());
  SmallVector<Location, 4> arglocs(4, builder.getUnknownLoc());
  body->addArguments(argtypes, arglocs);
  for (Value v : launchOperands) {
    body->addArgument(v.getType(), builder.getUnknownLoc());
  }
  r->push_back(body);
}

void HerdLaunchOp::print(OpAsmPrinter &p) {

  auto num_async_deps = asyncDependencies().size();
  p << ' ';
  printAsyncDependencies(p, *this, (asyncToken() ? asyncToken().getType() : Type()), asyncDependencies());
  p << " tile (";
  p << getTileIds().x << ", ";
  p << getTileIds().y << ") in (";
  p << getHerdSize().x << "=";
  p << getOperand(num_async_deps + 0) << ", ";
  p << getHerdSize().y << "=";
  p << getOperand(num_async_deps + 1) << ")";

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
          return (OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()
            != attr.getName());
        }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult HerdLaunchOp::parse(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> tileArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 2> tileSize(2);
  SmallVector<OpAsmParser::UnresolvedOperand, 2> tileSizeRef(2);

  Type asyncTokenType = nullptr;
  if (parseAsyncDependencies(parser, asyncTokenType, asyncDependencies))
    return failure();
  if (asyncTokenType)
    result.addTypes(asyncTokenType);

  if (parser.parseKeyword("tile"))
    return failure();

  if (parser.parseRegionArgumentList(tileArgs, /*requiredOperandCount=*/2,
                                     OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();

  for (int i = 0; i < 2; ++i) {
    if (i != 0 && parser.parseComma())
      return failure();
    if (parser.parseRegionArgument(tileSizeRef[i]) || parser.parseEqual() ||
        parser.parseOperand(tileSize[i]))
      return failure();
  }

  if (parser.parseRParen())
    return failure();

  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, 4> dataTypes(4, index);

  tileArgs.push_back(tileSizeRef[0]);
  tileArgs.push_back(tileSizeRef[1]);

  auto tokenType = xilinx::air::AsyncTokenType::get(parser.getBuilder().getContext());
  parser.resolveOperands(asyncDependencies, tokenType, result.operands);
  parser.resolveOperands(tileSize, index, result.operands);

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    do {
      OpAsmParser::UnresolvedOperand argument;
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseRegionArgument(argument) || parser.parseEqual() ||
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
    tileArgs.push_back(kernelArguments[i]);
    dataTypes.push_back(types[i]);
    parser.resolveOperand(kernelOperands[i], types[i], result.operands);
  }

  parser.parseOptionalAttrDictWithKeyword(result.attributes);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, tileArgs, dataTypes))
    return failure();

  SmallVector<int32_t, 8> segmentSizes(4, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = kernelOperands.size();
  result.addAttribute(OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getI32VectorAttr(segmentSizes));
  return success();
}

HerdDim2 HerdLaunchOp::getTileIds() {
  auto args = body().front().getArguments();
  return HerdDim2{args[0], args[1]};
}

HerdDim2 HerdLaunchOp::getHerdSize() {
  auto args = body().front().getArguments();
  return HerdDim2{args[2], args[3]};
}

HerdDim2 HerdLaunchOp::getHerdSizeOperands() {
  auto opers = getOperands().drop_front(asyncDependencies().size());
  return HerdDim2{opers[0], opers[1]};
}

unsigned HerdLaunchOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - 2;
}

Value HerdLaunchOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + 2 + i);
}

ArrayRef<BlockArgument> HerdLaunchOp::getKernelArguments() {
  return body().front().getArguments().drop_front(4);
}

BlockArgument HerdLaunchOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
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

  SmallVector<OpAsmParser::UnresolvedOperand, 4> args;

  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelOperands;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    do {
      OpAsmParser::UnresolvedOperand argument;
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseRegionArgument(argument) || parser.parseEqual() ||
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
    args.push_back(kernelArguments[i]);
    parser.resolveOperand(kernelOperands[i], types[i], result.operands);
  }

  parser.parseOptionalAttrDictWithKeyword(result.attributes);

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args, types))
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
