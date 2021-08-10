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
      .Default([](Type) { llvm_unreachable("unexpected 'gpu' type kind"); });
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
  body->addArguments(argtypes);
  for (Value v : launchOperands) {
    body->addArgument(v.getType());
  }
  r->push_back(body);
}

static LogicalResult verify(HerdLaunchOp op) {
  return success();
}

static void printHerdLaunchOp(OpAsmPrinter &p, HerdLaunchOp op) {

  p << HerdLaunchOp::getOperationName();

  auto num_async_deps = op.asyncDependencies().size();
  if (num_async_deps) {
    p << " [";
    llvm::interleaveComma(op.asyncDependencies(), p);
    p << "] ";
  }

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
      if (i) p << ",";
      p << op.getKernelOperand(i).getType();
    }
  }

  SmallVector<NamedAttribute, 8> filteredAttrs(
        llvm::make_filter_range(op->getAttrs(), [&](NamedAttribute attr) {
          return (OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()
            != attr.first.strref());
        }));
  if (filteredAttrs.size()) {
    p << "attributes";
    p.printOptionalAttrDict(filteredAttrs);
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
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }

  if (parser.parseOperandList(asyncDependencies, OpAsmParser::Delimiter::OptionalSquare))
    return failure();

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

  parser.resolveOperands(asyncDependencies, asyncTokenType, result.operands);
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

// static ParseResult
// parseLaunchFuncOperands(OpAsmParser &parser,
//                         SmallVectorImpl<OpAsmParser::OperandType> &argNames,
//                         SmallVectorImpl<Type> &argTypes) {
//   if (parser.parseOptionalKeyword("args"))
//     return success();
//   SmallVector<NamedAttrList, 4> argAttrs;
//   bool isVariadic = false;
//   return impl::parseFunctionArgumentList(parser, /*allowAttributes=*/false,
//                                          /*allowVariadic=*/false, argNames,
//                                          argTypes, argAttrs, isVariadic);
// }

// static void printLaunchFuncOperands(OpAsmPrinter &printer, Operation *,
//                                     OperandRange operands, TypeRange types) {
//   if (operands.empty())
//     return;
//   printer << "args(";
//   llvm::interleaveComma(llvm::zip(operands, types), printer,
//                         [&](const auto &pair) {
//                           printer.printOperand(std::get<0>(pair));
//                           printer << " : ";
//                           printer.printType(std::get<1>(pair));
//                         });
//   printer << ")";
// }

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
  for (auto size : sizeAttr.getIntValues())
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

#include "air/Dialect/AIR/AIROpInterfaces.cpp.inc"

using namespace mlir::NPCOMP;

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.cpp.inc"
