// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/IR/Attributes.h"
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

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::OperandType> &asyncDependencies) {
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
  printer << "]";
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

static LogicalResult verify(HerdLaunchOp op) {
  return success();
}

static void printHerdLaunchOp(OpAsmPrinter &p, HerdLaunchOp op) {

  auto num_async_deps = op.asyncDependencies().size();
  if (num_async_deps)
    p << ' ';
  printAsyncDependencies(p, op, (op.asyncToken() ? op.asyncToken().getType() : Type()), op.asyncDependencies());
  p << " tile (";
  p << op.getTileIds().x << ", ";
  p << op.getTileIds().y << ") in (";
  p << op.getHerdSize().x << "=";
  p << op.getOperand(num_async_deps + 0) << ", ";
  p << op.getHerdSize().y << "=";
  p << op.getOperand(num_async_deps + 1) << ")";

  if (op.getNumKernelOperands()) {
    auto args = op.getKernelArguments();
    p << " args(";
    for (int i=0,e=op.getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << op.getKernelOperand(i);
    }
    p << ") : ";
    for (int i=0,e=op.getNumKernelOperands(); i<e; i++) {
      if (i) p << ", ";
      p << op.getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(op->getAttrs(), [&](NamedAttribute attr) {
          return (OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()
            != attr.getName());
        }));
  p << " ";
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
    p << " ";
  }
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

static ParseResult
parseHerdLaunchOp(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 4> asyncDependencies;
  SmallVector<OpAsmParser::OperandType, 4> tileArgs;
  SmallVector<OpAsmParser::OperandType, 2> tileSize(2);
  SmallVector<OpAsmParser::OperandType, 2> tileSizeRef(2);

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

  SmallVector<OpAsmParser::OperandType, 4> kernelOperands;
  SmallVector<OpAsmParser::OperandType, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    do {
      OpAsmParser::OperandType argument;
      OpAsmParser::OperandType operand;
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

static ParseResult
parsePipelineStageOp(OpAsmParser &parser, OperationState &result) {

  SmallVector<OpAsmParser::OperandType, 4> args;

  SmallVector<OpAsmParser::OperandType, 4> kernelOperands;
  SmallVector<OpAsmParser::OperandType, 4> kernelArguments;
  SmallVector<Type, 4> types;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();
    do {
      OpAsmParser::OperandType argument;
      OpAsmParser::OperandType operand;
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

//
// PipelineStageOp
//

static LogicalResult verify(PipelineStageOp op) {
  return success();
}

static void printPipelineStageOp(OpAsmPrinter &p, PipelineStageOp op) {

  if (op.getNumOperands()) {
    auto args = op.body().front().getArguments();
    p << " args(";
    for (int i=0,e=op.getNumOperands(); i<e; i++) {
      if (i) p << ", ";
      p << args[i] << "=";
      p << op.getOperand(i);
    }
    p << ") : ";
    for (int i=0,e=op.getNumOperands(); i<e; i++) {
      if (i) p << ", ";
      p << op.getOperand(i).getType();
    }
  }

  p << " ";
  if (op->getAttrs().size()) {
    p << "attributes ";
    p.printOptionalAttrDict(op->getAttrs());
    p << " ";
  }
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);

  if (op->getNumResults())
    p << " : ";
  for (Type type : op->getResultTypes())
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

static LogicalResult verify(RegionOp op) {
  assert(op.getOperation()->getNumRegions() == 1 && "RegionOp has zero region!");
  assert(!op.body().empty() && "RegionOp should have non-empty body");

  return success();
}

#include "air/Dialect/AIR/AIROpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.cpp.inc"
