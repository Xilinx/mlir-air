//===- AIRDialect.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/TypeSwitch.h"

#include <iostream>

using namespace mlir;

#include "air/Dialect/AIR/AIRDialect.cpp.inc"

namespace xilinx {

void air::airDialect::initialize() {
  addTypes<AsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "air/Dialect/AIR/AIR.cpp.inc"
      >();
}

Type air::airDialect::parseType(DialectAsmParser &parser) const {
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

void air::airDialect::printType(Type type, DialectAsmPrinter &os) const {
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

  // if the operands won't change, return
  if (newOperands.size() == op.getNumKernelOperands())
    return failure();

  IRMapping remap;
  auto newOp = T::create(rewriter, op.getLoc(), op.getAsyncDependencies(),
                         op.getSizeOperands(), newOperands,
                         op->getNumResults() > 0, op->getAttrs());

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
    asyncTokenType = parser.getBuilder().getType<air::AsyncTokenType>();
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

template <class OpT>
static LogicalResult CanonicalizeAsyncOpDeps(OpT op,
                                             PatternRewriter &rewriter) {
  auto getAllReadAccess = [](Operation *op) {
    SmallVector<Value> operands;
    if (auto linalgop = dyn_cast<linalg::LinalgOp>(op)) {
      for (auto oper : linalgop.getDpsInputs())
        operands.push_back(oper);
    } else if (auto memref_copy = dyn_cast<memref::CopyOp>(op)) {
      operands.push_back(memref_copy.getSource());
    } else if (auto memcpy = mlir::dyn_cast<xilinx::air::MemcpyInterface>(op)) {
      if (memcpy.getSrcMemref())
        operands.push_back(memcpy.getSrcMemref());
    } else { // If unknown op, then assume all operands are read.
      for (auto oper : op->getOperands()) {
        if (!isa<MemRefType>(oper.getType()))
          continue;
        operands.push_back(oper);
      }
    }
    return operands;
  };
  auto getAllMemrefsReadByOp = [getAllReadAccess](Operation *o) {
    llvm::SetVector<Value> memrefs;
    auto opReadAccesses = getAllReadAccess(o);
    memrefs.insert(opReadAccesses.begin(), opReadAccesses.end());
    SmallVector<Region *> regions;
    for (auto &region : o->getRegions())
      regions.push_back(&region);
    // If air.wait_all, then we analyze the dependency by collecting all
    // operations that depend on it.
    auto waitAllOp = dyn_cast_if_present<air::WaitAllOp>(o);
    if (waitAllOp && waitAllOp.getAsyncToken()) {
      for (auto user : waitAllOp.getAsyncToken().getUsers()) {
        auto userReadAccesses = getAllReadAccess(user);
        memrefs.insert(userReadAccesses.begin(), userReadAccesses.end());
        for (auto &region : user->getRegions())
          regions.push_back(&region);
      }
    }
    for (auto region : regions) {
      visitUsedValuesDefinedAbove(*region, [&memrefs,
                                            getAllReadAccess](OpOperand *use) {
        if (llvm::is_contained(getAllReadAccess(use->getOwner()), use->get()))
          memrefs.insert(use->get());
      });
    }
    return memrefs;
  };
  auto getAllWriteAccess = [](Operation *op) {
    SmallVector<Value> operands;
    if (auto linalgop = dyn_cast<linalg::LinalgOp>(op)) {
      for (auto oper :
           llvm::concat<Value>(linalgop.getDpsInits(), linalgop->getResults()))
        operands.push_back(oper);
    } else if (auto memref_copy = dyn_cast<memref::CopyOp>(op)) {
      operands.push_back(memref_copy.getTarget());
    } else if (auto memcpy = mlir::dyn_cast<xilinx::air::MemcpyInterface>(op)) {
      if (memcpy.getDstMemref())
        operands.push_back(memcpy.getDstMemref());
    } else { // If unknown op, then assume all operands and results are written
             // to.
      for (auto oper :
           llvm::concat<Value>(op->getOperands(), op->getResults())) {
        if (!isa<MemRefType>(oper.getType()))
          continue;
        operands.push_back(oper);
      }
    }
    return operands;
  };
  auto getAllMemrefsWrittenByOp = [getAllWriteAccess](Operation *o) {
    llvm::SetVector<Value> memrefs;
    auto opWriteAccesses = getAllWriteAccess(o);
    memrefs.insert(opWriteAccesses.begin(), opWriteAccesses.end());
    SmallVector<Region *> regions;
    for (auto &region : o->getRegions())
      regions.push_back(&region);
    // If air.wait_all, then we analyze the dependency by collecting all
    // operations that depend on it.
    auto waitAllOp = dyn_cast_if_present<air::WaitAllOp>(o);
    if (waitAllOp && waitAllOp.getAsyncToken()) {
      for (auto user : waitAllOp.getAsyncToken().getUsers()) {
        auto userWriteAccesses = getAllWriteAccess(user);
        memrefs.insert(userWriteAccesses.begin(), userWriteAccesses.end());
        for (auto &region : user->getRegions())
          regions.push_back(&region);
      }
    }
    for (auto region : regions) {
      visitUsedValuesDefinedAbove(*region, [&memrefs,
                                            getAllWriteAccess](OpOperand *use) {
        if (llvm::is_contained(getAllWriteAccess(use->getOwner()), use->get()))
          memrefs.insert(use->get());
      });
    }
    return memrefs;
  };
  auto getAllSymbolRefAccess = [](Operation *o) {
    SmallVector<SymbolRefAttr> result;
    for (NamedAttribute attr : o->getAttrs()) {
      // Skip attributes that define a symbol name
      if (attr.getName() == "sym_name")
        continue;
      if (auto symRef = dyn_cast<SymbolRefAttr>(attr.getValue())) {
        result.push_back(symRef);
      }
      // Also check for ArrayAttr containing SymbolRefAttrs
      else if (auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue())) {
        for (Attribute elem : arrayAttr) {
          if (auto symRef = dyn_cast<SymbolRefAttr>(elem)) {
            result.push_back(symRef);
          }
        }
      }
    }
    return result;
  };
  auto collectAllSymbolRefAttrsInRegion = [](Region &region) {
    SmallVector<SymbolRefAttr> result;
    region.walk([&](Operation *op) {
      for (NamedAttribute attr : op->getAttrs()) {
        Attribute value = attr.getValue();
        // Direct SymbolRefAttr
        if (auto sym = llvm::dyn_cast<SymbolRefAttr>(value))
          result.push_back(sym);
        // Array of SymbolRefAttr
        else if (auto arr = llvm::dyn_cast<ArrayAttr>(value)) {
          for (Attribute elem : arr) {
            if (auto sym = llvm::dyn_cast<SymbolRefAttr>(elem))
              result.push_back(sym);
          }
        }
      }
    });
    return result;
  };
  auto getSymbolRefsUsedByOp =
      [getAllSymbolRefAccess, collectAllSymbolRefAttrsInRegion](Operation *o) {
        llvm::SetVector<SymbolRefAttr> result;
        auto opSymbolRefAccesses = getAllSymbolRefAccess(o);
        result.insert(opSymbolRefAccesses.begin(), opSymbolRefAccesses.end());
        if (isa<air::AsyncOpInterface>(o))
          return result;
        // If op isn't an air.async op, then collect symref uses in its regions.
        SmallVector<Region *> regions;
        for (auto &region : o->getRegions())
          regions.push_back(&region);
        for (auto region : regions) {
          auto symRefsInRegion = collectAllSymbolRefAttrsInRegion(*region);
          result.insert(symRefsInRegion.begin(), symRefsInRegion.end());
        }
        return result;
      };
  auto memrefsReadBySinkOp = getAllMemrefsReadByOp(op.getOperation());
  auto memrefsWrittenBySinkOp = getAllMemrefsWrittenByOp(op.getOperation());
  auto resourcesUsedBySinkOp = getSymbolRefsUsedByOp(op.getOperation());
  // make a list of new async token operands
  std::function<void(SmallVector<Value>, SmallVector<Value> &)>
      getDirectDependenciesGreedily;
  getDirectDependenciesGreedily = [&getDirectDependenciesGreedily](
                                      SmallVector<Value> depList,
                                      SmallVector<Value> &directDeps) {
    for (auto v : depList) {
      if (auto wa = dyn_cast_if_present<air::WaitAllOp>(v.getDefiningOp()))
        getDirectDependenciesGreedily(wa.getAsyncDependencies(), directDeps);
      else
        directDeps.push_back(v);
    }
    return;
  };
  llvm::SetVector<Value> newAsyncDeps; // don't include duplicates
  SmallVector<Value> directDeps;
  getDirectDependenciesGreedily(op.getAsyncDependencies(), directDeps);
  for (auto v : directDeps) {
    // don't include any false (RAR) dependencies
    if (v.getDefiningOp()) {
      auto memrefsReadBySourceOp = getAllMemrefsReadByOp(v.getDefiningOp());
      auto memrefsWrittenBySourceOp =
          getAllMemrefsWrittenByOp(v.getDefiningOp());
      auto resourcesUsedBySourceOp = getSymbolRefsUsedByOp(v.getDefiningOp());
      bool sourceOpTouchesMemref =
          llvm::range_size(llvm::concat<Value>(memrefsReadBySourceOp,
                                               memrefsWrittenBySourceOp)) != 0;
      bool sinkOpTouchesMemref =
          llvm::range_size(llvm::concat<Value>(memrefsReadBySinkOp,
                                               memrefsWrittenBySinkOp)) != 0;
      if (sourceOpTouchesMemref && sinkOpTouchesMemref) {
        bool RAWNotFound = llvm::none_of(
            memrefsWrittenBySourceOp, [&memrefsReadBySinkOp](Value v) {
              return llvm::is_contained(memrefsReadBySinkOp, v);
            });
        bool WARNotFound = llvm::none_of(
            memrefsReadBySourceOp, [&memrefsWrittenBySinkOp](Value v) {
              return llvm::is_contained(memrefsWrittenBySinkOp, v);
            });
        bool WAWNotFound = llvm::none_of(
            memrefsWrittenBySourceOp, [&memrefsWrittenBySinkOp](Value v) {
              return llvm::is_contained(memrefsWrittenBySinkOp, v);
            });
        bool noSharedResource = llvm::none_of(
            resourcesUsedBySourceOp, [&resourcesUsedBySinkOp](SymbolRefAttr r) {
              return llvm::is_contained(resourcesUsedBySinkOp, r);
            });
        if (RAWNotFound && WARNotFound && WAWNotFound && noSharedResource)
          continue;
      }
    }
    newAsyncDeps.insert(v);
  }

  // don't include a dependency of another dependency
  auto getDepsOfDeps = [](llvm::SetVector<Value> deps) {
    llvm::SetVector<Value> depsOfDeps;
    for (auto v : deps) {
      if (auto asyncOperand =
              dyn_cast_if_present<air::AsyncOpInterface>(v.getDefiningOp())) {
        auto deps = asyncOperand.getAsyncDependencies();
        depsOfDeps.insert(deps.begin(), deps.end());
      }
    }
    return depsOfDeps;
  };
  llvm::SetVector<Value> erased;
  for (auto v : newAsyncDeps) {
    if (llvm::is_contained(getDepsOfDeps(newAsyncDeps), v))
      erased.insert(v);
  }
  for (auto e : erased)
    newAsyncDeps.remove(e);

  // if the operands won't change, return
  if (newAsyncDeps.size() == op.getAsyncDependencies().size())
    return failure();

  while (op.getAsyncDependencies().size())
    op.eraseAsyncDependency(0);
  for (auto v : newAsyncDeps)
    op.addAsyncDependency(v);
  auto newOp = rewriter.clone(*op.getOperation());
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

// Enforce that, within op's region, all loop-carried dependency tokens to
// scf.for loops must enter as iter_args.
template <class OpT>
static LogicalResult
CanonicalizeAsyncLoopCarriedDepsInRegion(OpT op, PatternRewriter &rewriter) {
  // Get async tokens used by ops within region.getOps(), which are defined
  // above this region.
  auto getUsedAsyncTokensDefinedAbove = [](Region *region,
                                           SetVector<Value> &regionTokens) {
    for (auto &o : region->getOps()) {
      for (auto oper : o.getOperands()) {
        if (!isa<air::AsyncTokenType>(oper.getType()))
          continue;
        if (!oper.getParentRegion()->isProperAncestor(region))
          continue;
        regionTokens.insert(oper);
      }
    }
    return;
  };
  // Get all scf.for ops which might require loop-carried token
  // canonicalization.
  SetVector<scf::ForOp> candidateForOps;
  op.getBody().template walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](Operation *o) {
        if (o->hasTrait<OpTrait::IsIsolatedFromAbove>())
          return WalkResult::skip();
        if (auto forOp = dyn_cast<scf::ForOp>(o)) {
          SetVector<Value> regionTokens;
          getUsedAsyncTokensDefinedAbove(&forOp.getRegion(), regionTokens);
          if (regionTokens.empty())
            return WalkResult::advance();
          if (llvm::none_of(forOp.getRegionIterArgs(), [](BlockArgument arg) {
                return isa<air::AsyncTokenType>(arg.getType());
              }))
            return WalkResult::advance();
          candidateForOps.insert(forOp);
        }
        return WalkResult::advance();
      });
  if (candidateForOps.empty())
    return failure();
  // Enforce that all loop-carried tokens must be passed in from the iter_args.
  for (auto forOp : candidateForOps) {
    SetVector<Value> regionTokens;
    getUsedAsyncTokensDefinedAbove(&forOp.getRegion(), regionTokens);
    BlockArgument loopCarriedTokenArg =
        *llvm::find_if(forOp.getRegionIterArgs(), [](BlockArgument arg) {
          return isa<air::AsyncTokenType>(arg.getType());
        });
    for (auto tok : regionTokens)
      replaceAllUsesInRegionWith(tok, loopCarriedTokenArg, forOp.getRegion());
    rewriter.setInsertionPoint(forOp);
    for (auto arg : forOp.getInitArgs()) {
      if (isa<air::AsyncTokenType>(arg.getType()))
        regionTokens.insert(arg);
    }
    auto newWaitAll =
        air::WaitAllOp::create(rewriter, forOp->getLoc(),
                               air::AsyncTokenType::get(forOp->getContext()),
                               regionTokens.takeVector());
    forOp
        .getInitsMutable()[loopCarriedTokenArg.getArgNumber() -
                           forOp.getNumInductionVars()]
        .assign(newWaitAll.getAsyncToken());
  }
  return success();
}

//
// LaunchOp
//

void air::LaunchOp::build(OpBuilder &builder, OperationState &result,
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

void air::LaunchOp::build(OpBuilder &builder, OperationState &result,
                          ValueRange sizes, ValueRange launchOperands) {

  build(builder, result, {}, sizes, launchOperands, false);
}

void air::LaunchOp::print(OpAsmPrinter &p) {

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

ParseResult air::LaunchOp::parse(OpAsmParser &parser, OperationState &result) {

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

void air::LaunchOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<LaunchOp>);
  patterns.add(CanonicalizeAsyncOpDeps<LaunchOp>);
  patterns.add(CanonicalizeAsyncLoopCarriedDepsInRegion<LaunchOp>);
}

ArrayRef<BlockArgument> air::LaunchOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> air::LaunchOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange air::LaunchOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned air::LaunchOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

OperandRange air::LaunchOp::getKernelOperands() {
  return getOperands().drop_front(getAsyncDependencies().size() + getNumDims());
}

Value air::LaunchOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> air::LaunchOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument air::LaunchOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned air::LaunchOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

//
// SegmentOp
//

void air::SegmentOp::build(OpBuilder &builder, OperationState &result,
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
  air::SegmentOp::ensureTerminator(*r, builder, result.location);
}

void air::SegmentOp::build(OpBuilder &builder, OperationState &result,
                           ValueRange sizes, ValueRange segmentOperands) {

  build(builder, result, {}, sizes, segmentOperands, false);
}

void air::SegmentOp::print(OpAsmPrinter &p) {

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

ParseResult air::SegmentOp::parse(OpAsmParser &parser, OperationState &result) {

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

  auto tokenType = air::AsyncTokenType::get(parser.getBuilder().getContext());
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
  air::SegmentOp::ensureTerminator(*body, parser.getBuilder(), result.location);

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

void air::SegmentOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<air::SegmentOp>);
  patterns.add(CanonicalizeAsyncOpDeps<air::SegmentOp>);
  patterns.add(CanonicalizeAsyncLoopCarriedDepsInRegion<air::SegmentOp>);
}

ArrayRef<BlockArgument> air::SegmentOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> air::SegmentOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange air::SegmentOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned air::SegmentOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

OperandRange air::SegmentOp::getKernelOperands() {
  return getOperands().drop_front(getAsyncDependencies().size() + getNumDims());
}

Value air::SegmentOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> air::SegmentOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(getNumDims() * 2);
}

BlockArgument air::SegmentOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned air::SegmentOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

/// Utility function to verify that all memref.alloc operations within a region
/// have a memory space greater than or equal to the specified minimum.
/// Returns failure if any alloc violates the constraint.
template <typename OpT>
static LogicalResult verifyAllocMemorySpace(OpT op, unsigned minMemorySpace,
                                            StringRef opName) {
  WalkResult result =
      op.getBody().walk([&](memref::AllocOp allocOp) -> WalkResult {
        auto memrefType = allocOp.getType();
        // Get memory space (defaults to 0 if not specified)
        unsigned memorySpace = memrefType.getMemorySpaceAsInt();

        if (memorySpace < minMemorySpace) {
          allocOp.emitOpError()
              << "memref.alloc inside " << opName
              << " must have memory space >= " << minMemorySpace
              << ", but found memory space " << memorySpace;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });

  if (result.wasInterrupted())
    return failure();

  return success();
}

LogicalResult air::SegmentOp::verify() {
  return verifyAllocMemorySpace(*this, /*minMemorySpace=*/1, "air.segment");
}

//
// HerdOp
//

void air::HerdOp::build(OpBuilder &builder, OperationState &result,
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
  air::HerdOp::ensureTerminator(*r, builder, result.location);
}

void air::HerdOp::build(OpBuilder &builder, OperationState &result,
                        ValueRange sizes, ValueRange launchOperands) {

  build(builder, result, {}, sizes, launchOperands);
}

void air::HerdOp::print(OpAsmPrinter &p) {

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

ParseResult air::HerdOp::parse(OpAsmParser &parser, OperationState &result) {

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
  air::HerdOp::ensureTerminator(*body, parser.getBuilder(), result.location);

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

void air::HerdOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add(canonicalizeHierarchyOpArgs<HerdOp>);
  patterns.add(CanonicalizeAsyncOpDeps<HerdOp>);
  patterns.add(CanonicalizeAsyncLoopCarriedDepsInRegion<HerdOp>);
}

ArrayRef<BlockArgument> air::HerdOp::getIds() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.take_front(n);
}

ArrayRef<BlockArgument> air::HerdOp::getSize() {
  auto s = getBody().front().getArguments();
  auto n = getNumDims();
  return s.slice(n, n);
}

OperandRange air::HerdOp::getSizeOperands() {
  auto start = getAsyncDependencies().size();
  auto n = getNumDims();
  return getOperands().slice(start, n);
}

unsigned air::HerdOp::getNumKernelOperands() {
  return getNumOperands() - getAsyncDependencies().size() - getNumDims();
}

OperandRange air::HerdOp::getKernelOperands() {
  return getOperands().drop_front(getAsyncDependencies().size() + getNumDims());
}

Value air::HerdOp::getKernelOperand(unsigned i) {
  return getOperand(getAsyncDependencies().size() + getNumDims() + i);
}

ArrayRef<BlockArgument> air::HerdOp::getKernelArguments() {
  return getBody().front().getArguments().drop_front(4);
}

BlockArgument air::HerdOp::getKernelArgument(unsigned i) {
  return getKernelArguments()[i];
}

unsigned air::HerdOp::getNumDims() {
  auto size_attr_name = getOperandSegmentSizeAttr();
  auto size_attr = (*this)->getAttrOfType<DenseI32ArrayAttr>(size_attr_name);
  auto segment_sizes = size_attr.asArrayRef();
  return segment_sizes[1];
}

uint64_t air::HerdOp::getNumCols() {
  auto cols = getSizeOperands()[0].getDefiningOp();
  return cast<arith::ConstantIndexOp>(cols).value();
}

uint64_t air::HerdOp::getNumRows() {
  auto rows = getSizeOperands()[1].getDefiningOp();
  return cast<arith::ConstantIndexOp>(rows).value();
}

LogicalResult air::HerdOp::verify() {
  return verifyAllocMemorySpace(*this, /*minMemorySpace=*/2, "air.herd");
}

//
// Asynchronous execute
//

LogicalResult air::ExecuteOp::verify() {
  if (getOperation()->getNumRegions() != 1)
    return emitOpError("ExecuteOp has zero region.");
  if (getRegion().empty())
    return emitOpError("ExecuteOp should have non-empty region.");
  if (getBody().empty())
    return emitOpError("ExecuteOp should have non-empty body.");

  return success();
}

static LogicalResult FoldExecute(air::ExecuteOp op, PatternRewriter &rewriter) {

  // if the terminator is the only thing in the ExecuteOp,
  // and the op is unused, then it can be removed.
  auto &body = op.getRegion().front();
  auto et = body.getTerminator();
  if (op.use_empty() && body.getOperations().size() == 1) {
    rewriter.eraseOp(op);
    return success();
  }

  // replace returns of (1) constants with the constant, and (2) values not
  // defined within the execute with its original value
  int idx = 0;
  for (auto v : et->getOperands()) {
    idx++;
    if (op.getResult(idx).use_empty())
      continue;
    if (!op.getRegion().isAncestor(v.getParentRegion())) {
      rewriter.replaceAllUsesWith(op.getResult(idx), v);
      return success();
    }
    if (auto constOp =
            dyn_cast_if_present<arith::ConstantOp>(v.getDefiningOp())) {
      rewriter.replaceAllUsesWith(op.getResult(idx),
                                  rewriter.clone(*constOp)->getResult(0));
      return success();
    }
  }

  // if any of the results are used, return failure()
  for (auto v : op->getResults().drop_front())
    if (!v.use_empty())
      return failure();

  // if we get here then only the async token result has uses.

  // Don't fold if body contains ops with memory write side effects
  // (e.g., memref.store). These ops must be preserved even if the
  // execute's async token becomes unused due to wait_all folding.
  for (auto &bodyOp : body.without_terminator()) {
    if (auto memEffects = dyn_cast<MemoryEffectOpInterface>(bodyOp)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      memEffects.getEffects(effects);
      for (auto &effect : effects)
        if (isa<MemoryEffects::Write>(effect.getEffect()))
          return failure();
    }
  }

  // if there are extra results than async token, and none of them are used,
  // then replace the execute with a wait_all no-op.
  if (op->getNumResults() > 1) {
    op.getResult(0).replaceAllUsesWith(
        air::WaitAllOp::create(rewriter, op->getLoc(),
                               op->getResult(0).getType(), op->getOperands())
            .getResult(0));
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

void air::ExecuteOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(FoldExecute);
  patterns.add(CanonicalizeAsyncOpDeps<ExecuteOp>);
  patterns.add(CanonicalizeAsyncLoopCarriedDepsInRegion<ExecuteOp>);
}

//
// WaitAllOp
//

static LogicalResult FoldWaitAll(air::WaitAllOp op, PatternRewriter &rewriter) {

  // Erase wait_all with no operands and no uses
  if (op.use_empty() && !op->getOperands().size()) {
    rewriter.eraseOp(op);
    return success();
  }

  // If async wait_all has a single operand, forward it to any uses
  if (op.getAsyncDependencies().size() == 1 && op.getResults().size() == 1) {
    rewriter.replaceOp(op, op.getAsyncDependencies()[0]);
    return success();
  }

  // If all of async wait_all's users have AsyncOpInterface, fold it into its
  // users
  if (op.getResults().size() == 1 &&
      llvm::all_of(op.getResults().front().getUsers(), [](Operation *user) {
        return isa_and_present<air::AsyncOpInterface>(user);
      })) {
    SmallVector<Operation *> users;
    for (auto user : op.getResults().front().getUsers()) {
      users.push_back(user);
    }
    for (auto user : users) {
      air::AsyncOpInterface asyncUser =
          dyn_cast_if_present<air::AsyncOpInterface>(user);
      for (int i = asyncUser.getAsyncDependencies().size() - 1; i >= 0; i--) {
        if (asyncUser.getAsyncDependencies()[i] == op.getResults().front())
          asyncUser.eraseAsyncDependency(i);
      }
      for (auto dep : op.getAsyncDependencies())
        asyncUser.addAsyncDependency(dep);
    }
    rewriter.eraseOp(op);
    return success();
  }

  return failure();
}

void air::WaitAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(FoldWaitAll);
  patterns.add(CanonicalizeAsyncOpDeps<air::WaitAllOp>);
}

// Get strides from MemRefType.
static SmallVector<Value> extractStridesFromMemrefType(MemRefType memrefTy,
                                                       OpBuilder &builder) {
  SmallVector<Value> strides;
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides = memrefTy.getStridesAndOffset(layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return strides;
  }

  for (auto s : layout_strides)
    strides.push_back(getValueOrCreateConstantIndexOp(
        builder, builder.getUnknownLoc(), builder.getIndexAttr(s)));

  return strides;
}

// Get sizes from MemRefType.
static SmallVector<Value> extractSizesFromMemrefType(MemRefType memrefTy,
                                                     OpBuilder &builder) {
  SmallVector<Value> sizes;
  for (auto s : memrefTy.getShape())
    sizes.push_back(getValueOrCreateConstantIndexOp(
        builder, builder.getUnknownLoc(), builder.getIndexAttr(s)));
  return sizes;
}

// Generic template to extract offsets from memref operations.
template <typename OpType>
static void extractOffsetsFromOp(OpType op, OpBuilder &builder,
                                 SmallVector<Value> &offsets) {
  auto op_offsets = op.getOffsets().begin();
  auto static_offsets = op.getStaticOffsets();
  auto loc = op.getLoc();

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(getValueOrCreateConstantIndexOp(
          builder, loc, builder.getIndexAttr(o)));
    else
      offsets.push_back(*op_offsets++);
  }
}

// Generic template to extract strides from memref operations.
template <typename OpType>
static void extractStridesFromOp(OpType op, OpBuilder &builder,
                                 SmallVector<Value> &strides) {
  auto op_strides = op.getStrides().begin();
  auto static_strides = op.getStaticStrides();
  auto loc = op.getLoc();

  for (auto o : static_strides) {
    if (o >= 0)
      strides.push_back(getValueOrCreateConstantIndexOp(
          builder, loc, builder.getIndexAttr(o)));
    else
      strides.push_back(*op_strides++);
  }
}

// Check if the access pattern represents the default (contiguous, row-major)
// data access pattern.
static bool isDefaultDataAccessPattern(SmallVector<Value> memcpy_sizes,
                                       SmallVector<Value> memcpy_strides) {
  if (memcpy_sizes.size() != memcpy_strides.size())
    return false;
  // If the sizes and strides were already accessing the memref in default
  // order, then wraps and strides are not needed
  if (memcpy_sizes.empty() || memcpy_strides.empty())
    return true;
  if (memcpy_sizes.size() == 1 && memcpy_strides.size() == 1) {
    auto stepsize = mlir::getConstantIntValue(memcpy_strides[0]);
    if (stepsize && *stepsize == 1)
      return true;
  }
  unsigned stride_factor = 1;
  for (int i = memcpy_sizes.size() - 1; i >= 0; i--) {
    auto stepsize = mlir::getConstantIntValue(memcpy_strides[i]);
    if (!stepsize)
      return false;
    auto wrap = mlir::getConstantIntValue(memcpy_sizes[i]);
    if (!wrap)
      return false;
    if (*wrap == 1 && *stepsize == 0)
      continue; // dummy dimension.
    if (*stepsize != stride_factor)
      return false;
    stride_factor *= *wrap;
  }
  return true;
}

// Check if the volume of sizes equals the volume of the memref.
// Return true if equal, and return false if any size value is not constant,
// or memref shape isn't static.
static bool isVolumeEqualToMemrefVolume(SmallVector<Value> memcpy_sizes,
                                        BaseMemRefType memref) {
  // Return false if memref doesn't have static shape
  if (!memref.hasStaticShape())
    return false;

  // Calculate memref volume
  int64_t memref_volume = 1;
  for (auto dim : memref.getShape()) {
    memref_volume *= dim;
  }

  // Calculate sizes volume
  int64_t sizes_volume = 1;
  for (auto size : memcpy_sizes) {
    auto constant_size = mlir::getConstantIntValue(size);
    if (!constant_size)
      return false; // Size value is not constant
    sizes_volume *= *constant_size;
  }

  return memref_volume == sizes_volume;
}

static LogicalResult canonicalizeEmptyLists(SmallVector<Value> &offsets,
                                            SmallVector<Value> &sizes,
                                            SmallVector<Value> &strides,
                                            BaseMemRefType memref) {
  // Check if the access pattern represents the default (contiguous, row-major)
  // data access pattern. If so, clear all lists to their canonical empty form.
  if (offsets.empty() && sizes.empty() && strides.empty())
    return failure();
  if (isDefaultDataAccessPattern(sizes, strides) &&
      isVolumeEqualToMemrefVolume(sizes, memref)) {
    offsets.clear();
    sizes.clear();
    strides.clear();
    return success();
  }
  return failure();
}

static LogicalResult canonicalizeAIRDmaOperands(OpBuilder builder,
                                                SmallVector<Value> &offsets,
                                                SmallVector<Value> &sizes,
                                                SmallVector<Value> &strides,
                                                MemRefType memref) {
  // Increase vector sizes up to memref size. When offsets, sizes and strides
  // are all empty, then it implies that the whole memref is accessed in the
  // default order.
  auto loc = builder.getUnknownLoc();
  auto max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  auto target_dim_size = max_dim_size;
  if (max_dim_size && offsets.size() < target_dim_size) {
    for (unsigned i = offsets.size(); i < target_dim_size; i++) {
      offsets.insert(offsets.begin(),
                     getValueOrCreateConstantIndexOp(builder, loc,
                                                     builder.getIndexAttr(0)));
    }
  }
  if (max_dim_size && sizes.size() < target_dim_size) {
    for (unsigned i = sizes.size(); i < target_dim_size; i++) {
      sizes.insert(sizes.begin(), getValueOrCreateConstantIndexOp(
                                      builder, loc, builder.getIndexAttr(1)));
    }
  }
  int memref_size = 1;
  for (auto size : memref.getShape())
    memref_size *= size;
  if (max_dim_size && strides.size() < target_dim_size) {
    for (unsigned i = strides.size(); i < target_dim_size; i++) {
      strides.insert(strides.begin(),
                     getValueOrCreateConstantIndexOp(
                         builder, loc, builder.getIndexAttr(memref_size)));
    }
  }

  if (offsets.size() != sizes.size() || sizes.size() != strides.size())
    return failure();

  return success();
}

/// Combine mixed offsets with existing offsets and strides in-place:
///   offsets[i] = sourceOffset[i] + opOffset[i] *
///   (applyStrideWhenFoldingOffsets ? stride[i] : 1)
///
/// This function handles offset composition for memref operations like subview
/// and reinterpret_cast.
/// - When applyStrideWhenFoldingOffsets=true: applies stride scaling (for
/// reinterpret_cast/subview)
/// - When applyStrideWhenFoldingOffsets=false: no stride scaling (for air
/// offset folding)
/// - Folds fully-static cases to index constants
/// - Uses affine.apply when stride is a known constant (to keep it affine)
/// - Falls back to arith ops for general dynamic cases
static void combineMixedOffsetsInPlace(
    mlir::PatternRewriter &rewriter, SmallVector<OpFoldResult> mixedOffsets,
    SmallVectorImpl<mlir::Value> &offsets, ArrayRef<mlir::Value> strides,
    bool applyStrideWhenFoldingOffsets = false) {

  const auto loc = rewriter.getUnknownLoc();
  assert(offsets.size() == strides.size() &&
         "offsets/strides must have same length");
  assert(offsets.size() == mixedOffsets.size() &&
         "offsets/strides must match the memref op's rank");

  auto makeIndexCst = [&](int64_t v) -> Value {
    return getValueOrCreateConstantIndexOp(rewriter, loc,
                                           rewriter.getIndexAttr(v));
  };

  auto addExprToExpr = [](AffineExpr &exprA, AffineExpr exprB) {
    exprA = (exprA) ? (exprA + exprB) : (exprB);
    return;
  };

  for (auto it : llvm::enumerate(mixedOffsets)) {
    const unsigned i = it.index();
    OpFoldResult opOffset = it.value();
    Value sourceOffset = offsets[i];
    Value sourceStride = strides[i];

    // Static opOffset?
    Attribute opOffsetAttr = llvm::dyn_cast_if_present<Attribute>(opOffset);

    // Try to read constants from Value-typed sourceOffset/sourceStride.
    auto sourceOffsetConst = getConstantIntValue(sourceOffset);
    auto sourceStrideConst = getConstantIntValue(sourceStride);
    // Case 1: everything static -> fold to a constant Value and overwrite.
    if (opOffsetAttr && sourceOffsetConst && sourceStrideConst) {
      auto opOff = mlir::cast<IntegerAttr>(opOffsetAttr).getInt();
      auto srcOff = *sourceOffsetConst;
      auto srcStr = *sourceStrideConst;
      offsets[i] = makeIndexCst(opOff * srcStr + srcOff);
      continue;
    }

    // Case 2: stride is a known constant -> use affine.apply.
    if (sourceStrideConst) {
      AffineExpr expr, newExpr;
      SmallVector<Value> affineOperands;

      // Start with sourceOffset (const or symbol).
      if (sourceOffsetConst) {
        newExpr = rewriter.getAffineConstantExpr(*sourceOffsetConst);
      } else {
        newExpr = rewriter.getAffineSymbolExpr(affineOperands.size());
        affineOperands.push_back(sourceOffset);
      }
      addExprToExpr(expr, newExpr);

      // Add opOffset * strideConst.
      newExpr = AffineExpr();
      int64_t k = (applyStrideWhenFoldingOffsets) ? *sourceStrideConst : 1;
      if (opOffsetAttr) {
        newExpr = rewriter.getAffineConstantExpr(
            mlir::cast<IntegerAttr>(opOffsetAttr).getInt() * k);
      } else {
        newExpr = rewriter.getAffineSymbolExpr(affineOperands.size()) * k;
        affineOperands.push_back(mlir::cast<Value>(opOffset));
      }
      addExprToExpr(expr, newExpr);

      AffineMap map = AffineMap::get(/*dimCount=*/0,
                                     /*symCount=*/affineOperands.size(), expr);
      offsets[i] =
          affine::AffineApplyOp::create(rewriter, loc, map, affineOperands)
              .getResult();
      continue;
    }

    // Case 3: general dynamic -> arith: sourceOffset + opOffset * sourceStride.
    Value mulTerm;
    if (opOffsetAttr) {
      Value opOffCst =
          makeIndexCst(mlir::cast<IntegerAttr>(opOffsetAttr).getInt());
      mulTerm = arith::MulIOp::create(rewriter, loc, opOffCst, sourceStride);
    } else {
      Value opOffsetVal = mlir::cast<Value>(opOffset);
      mulTerm = arith::MulIOp::create(rewriter, loc, opOffsetVal, sourceStride);
    }
    offsets[i] = arith::AddIOp::create(rewriter, loc, sourceOffset, mulTerm);
  }
}

/// Combine a reinterpret_cast's offset with the first dimension's existing
/// offset:
///   offsets[0] = sourceOffset[0] + reinterpretCastOffset
///
/// - Folds fully-static cases to an index constant.
/// - Falls back to arith ops otherwise.
static void
combineReinterpretCastOffsetInPlace(mlir::PatternRewriter &rewriter,
                                    memref::ReinterpretCastOp reinterpretCastOp,
                                    SmallVectorImpl<mlir::Value> &offsets,
                                    ArrayRef<mlir::Value> strides) {

  combineMixedOffsetsInPlace(rewriter, reinterpretCastOp.getMixedOffsets(),
                             offsets, strides,
                             /*applyStrideWhenFoldingOffsets*/ true);
}

/// Combine a subview's op offset with its source offset/stride in-place:
///   offsets[i] = sourceOffset[i] + opOffset[i] * stride[i]
///
/// - Folds fully-static cases to an index constant.
/// - Uses affine.apply when the stride is a known constant (to keep it affine).
/// - Falls back to arith ops otherwise.
template <typename SubViewLikeOp>
static void combineSubviewOffsetsInPlace(mlir::PatternRewriter &rewriter,
                                         SubViewLikeOp subviewOp,
                                         SmallVectorImpl<mlir::Value> &offsets,
                                         ArrayRef<mlir::Value> strides) {

  combineMixedOffsetsInPlace(rewriter, subviewOp.getMixedOffsets(), offsets,
                             strides, /*applyStrideWhenFoldingOffsets*/ true);
}

static LogicalResult ComposeMemrefOp(Value memref, PatternRewriter &rewriter,
                                     Value &input_memref,
                                     SmallVector<Value> &offsets,
                                     SmallVector<Value> &sizes,
                                     SmallVector<Value> &strides) {

  auto memref_type = llvm::dyn_cast<BaseMemRefType>(memref.getType());
  if (!memref_type)
    return rewriter.notifyMatchFailure(rewriter.getUnknownLoc(),
                                       "not operating on MemRef");
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
    } else if (auto castOp = dyn_cast<memref::CastOp>(defop)) {
      memrefOpVec.push_back(defop);
      defop = castOp.getSource().getDefiningOp();
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

  // Init. source memref and offsets at the front of the vector of memref ops.
  auto constZero =
      getValueOrCreateConstantIndexOp(rewriter, loc, rewriter.getIndexAttr(0));
  auto constOne =
      getValueOrCreateConstantIndexOp(rewriter, loc, rewriter.getIndexAttr(1));
  SmallVector<Value> initialOffsets, initialStrides;
  if (isa<ViewLikeOpInterface>(memrefOpVec[0]) &&
      isa<OffsetSizeAndStrideOpInterface>(memrefOpVec[0])) {
    auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(memrefOpVec[0]);
    auto offSizStrOp = dyn_cast<OffsetSizeAndStrideOpInterface>(memrefOpVec[0]);
    input_memref = viewLikeOp.getViewSource();
    extractOffsetsFromOp(offSizStrOp, rewriter, initialOffsets);
    extractStridesFromOp(offSizStrOp, rewriter, initialStrides);
  } else if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOpVec[0])) {
    input_memref = transposeOp.getIn();
    initialOffsets.clear();
    initialStrides.clear();
    for (unsigned i = 0; i < transposeOp.getPermutation().getNumInputs(); i++) {
      initialOffsets.push_back(constZero);
      initialStrides.push_back(constOne);
    }
  } else if (auto expandShapeOp =
                 dyn_cast<memref::ExpandShapeOp>(memrefOpVec[0])) {
    input_memref = expandShapeOp.getSrc();
    initialOffsets.clear();
    initialStrides.clear();
    for (unsigned i = 0;
         i < llvm::cast<MemRefType>(input_memref.getType()).getRank(); i++) {
      initialOffsets.push_back(constZero);
      initialStrides.push_back(constOne);
    }
  } else if (auto collapseShapeOp =
                 dyn_cast<memref::CollapseShapeOp>(memrefOpVec[0])) {
    input_memref = collapseShapeOp.getSrc();
    initialOffsets.clear();
    initialStrides.clear();
    for (unsigned i = 0;
         i < llvm::cast<MemRefType>(input_memref.getType()).getRank(); i++) {
      initialOffsets.push_back(constZero);
      initialStrides.push_back(constOne);
    }
  } else if (auto castOp = dyn_cast<memref::CastOp>(memrefOpVec[0])) {
    input_memref = castOp.getSource();
    initialOffsets.clear();
    initialStrides.clear();
    for (unsigned i = 0;
         i < llvm::cast<MemRefType>(input_memref.getType()).getRank(); i++) {
      initialOffsets.push_back(constZero);
      initialStrides.push_back(constOne);
    }
  } else
    return failure();

  // Compose offsets as the memref type propagates through the chain of memref
  // ops.
  for (auto memrefOp : memrefOpVec) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      if (transposeOp.getPermutation().getNumInputs() != initialOffsets.size())
        continue;
      initialOffsets = applyPermutationMap<Value>(transposeOp.getPermutation(),
                                                  initialOffsets);
      initialStrides = applyPermutationMap<Value>(transposeOp.getPermutation(),
                                                  initialStrides);
    } else if (auto castOp = dyn_cast<memref::CastOp>(memrefOp)) {
      // memref.cast doesn't change data layout, so no offset/stride adjustments
      // needed
      continue;
    } else if (auto reinterpretCastOp =
                   dyn_cast<memref::ReinterpretCastOp>(memrefOp)) {
      // For reinterpret_cast operations that aren't the first in the chain,
      // we need to combine their offset with existing offsets
      if (reinterpretCastOp != memrefOpVec.front() &&
          !reinterpretCastOp.hasZeroOffset()) {
        combineReinterpretCastOffsetInPlace(rewriter, reinterpretCastOp,
                                            initialOffsets, initialStrides);
      }
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      for (int i = (int)expandShapeOp.getReassociationIndices().size() - 1;
           i >= 0; i--) {
        if (expandShapeOp.getReassociationIndices()[i].size() <= 1)
          continue;
        for (unsigned j = 1;
             j < expandShapeOp.getReassociationIndices()[i].size(); j++) {
          initialOffsets.insert(initialOffsets.begin() + i, constZero);
          initialStrides.insert(initialStrides.begin() + i, constOne);
        }
      }
    } else if (auto collapseShapeOp =
                   dyn_cast<memref::CollapseShapeOp>(memrefOp)) {
      // For each collapsed dimension, keep the offset/stride of the first
      // source dim in the group.
      SmallVector<Value> newOffsets, newStrides;
      size_t srcIdx = 0;
      for (size_t i = 0; i < collapseShapeOp.getReassociationIndices().size();
           ++i) {
        // The first source dim in the group determines the collapsed
        // offset/stride.
        newOffsets.push_back(initialOffsets[srcIdx]);
        newStrides.push_back(initialStrides[srcIdx]);
        // Skip to the next group.
        srcIdx += collapseShapeOp.getReassociationIndices()[i].size();
      }
      initialOffsets = newOffsets;
      initialStrides = newStrides;
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      if (subviewOp != memrefOpVec.front() && !subviewOp.hasZeroOffset()) {
        combineSubviewOffsetsInPlace(rewriter, subviewOp, initialOffsets,
                                     initialStrides);
      }
    }
  }

  // Memref type at the sink memref op.
  MemRefType sink_memref_ty =
      llvm::cast<MemRefType>(memrefOpVec.back()->getResultTypes().front());

  // // Compose sizes and strides from the output memref type's layout.
  if (strides.empty())
    strides = extractStridesFromMemrefType(sink_memref_ty, rewriter);
  if (sizes.empty())
    sizes = extractSizesFromMemrefType(sink_memref_ty, rewriter);

  // Compose offsets with any offsets already in place.
  if (initialOffsets.size() && offsets.size()) {
    while (initialOffsets.size() < initialStrides.size()) {
      initialOffsets.insert(initialOffsets.begin(), constZero);
    }
    while (initialOffsets.size() < offsets.size()) {
      initialOffsets.insert(initialOffsets.begin(), constZero);
      initialStrides.insert(initialStrides.begin(), constOne);
    }
    while (offsets.size() < initialOffsets.size()) {
      offsets.insert(offsets.begin(), constZero);
    }
    combineMixedOffsetsInPlace(rewriter, mlir::getAsOpFoldResult(offsets),
                               initialOffsets, initialStrides);
  }
  offsets = initialOffsets;

  return canonicalizeAIRDmaOperands(rewriter, offsets, sizes, strides,
                                    sink_memref_ty);
}

//
// Dma op
//

static LogicalResult
ComposeMemrefOpOnDmaMemcpyNdSrc(air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) {

  Value memref = op.getSrcMemref();
  if (!memref)
    return failure();
  Value input_memref = memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getSrcOffsets();
  sizes = op.getSrcSizes();
  strides = op.getSrcStrides();

  auto composeMemrefRes =
      ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes, strides);
  auto canonicalizeListsRes =
      canonicalizeEmptyLists(offsets, sizes, strides,
                             dyn_cast<BaseMemRefType>(input_memref.getType()));

  if (failed(composeMemrefRes) && failed(canonicalizeListsRes))
    return failure();

  rewriter.replaceOpWithNewOp<air::DmaMemcpyNdOp>(
      op, op->getResultTypes(), op.getAsyncDependencies(), op.getDstMemref(),
      op.getDstOffsets(), op.getDstSizes(), op.getDstStrides(), input_memref,
      offsets, sizes, strides);

  return success();
}

static LogicalResult
ComposeMemrefOpOnDmaMemcpyNdDst(air::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) {

  Value memref = op.getDstMemref();
  if (!memref)
    return failure();
  Value input_memref = memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getDstOffsets();
  sizes = op.getDstSizes();
  strides = op.getDstStrides();

  auto composeMemrefRes =
      ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes, strides);
  auto canonicalizeListsRes =
      canonicalizeEmptyLists(offsets, sizes, strides,
                             dyn_cast<BaseMemRefType>(input_memref.getType()));

  if (failed(composeMemrefRes) && failed(canonicalizeListsRes))
    return failure();
  rewriter.replaceOpWithNewOp<air::DmaMemcpyNdOp>(
      op, op->getResultTypes(), op.getAsyncDependencies(), input_memref,
      offsets, sizes, strides, op.getSrcMemref(), op.getSrcOffsets(),
      op.getSrcSizes(), op.getSrcStrides());

  return success();
}

void air::DmaMemcpyNdOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnDmaMemcpyNdSrc);
  patterns.add(ComposeMemrefOpOnDmaMemcpyNdDst);
  patterns.add(CanonicalizeAsyncOpDeps<air::DmaMemcpyNdOp>);
}

//
// Channel put/get utilities
//

/// Return a `memref.dim` or `tensor.dim` for the shape of `v` at `dim`.
static OpFoldResult getDimValue(OpBuilder &builder, Location loc, Value v,
                                int64_t dim) {
  auto type = cast<ShapedType>(v.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getAsOpFoldResult(
      TypeSwitch<Type, Value>(v.getType())
          .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
            return tensor::DimOp::create(builder, loc, v, dim);
          })
          .Case<MemRefType>([&](MemRefType t) -> Value {
            return memref::DimOp::create(builder, loc, v, dim);
          }));
}

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Operation *getSlice(OpBuilder &b, Location loc, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Operation * {
        return tensor::ExtractSliceOp::create(b, loc, source, offsets, sizes,
                                              strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Operation * {
        return memref::SubViewOp::create(b, loc, source, offsets, sizes,
                                         strides);
      })
      .Default([&](Type t) -> Operation * { return nullptr; });
}

/// Materialize the OpFoldResults into Values.
SmallVector<Value> materializeOpFoldResultAsValues(ArrayRef<OpFoldResult> ofrs,
                                                   Location loc,
                                                   OpBuilder &builder) {
  SmallVector<Value> values;
  for (OpFoldResult ofr : ofrs) {
    if (auto val = dyn_cast<Value>(ofr)) {
      values.push_back(val);
    } else if (auto attr = dyn_cast<Attribute>(ofr)) {
      // Create an arith.constant if the OpFoldResult is an Attribute.
      auto constAttr = cast<IntegerAttr>(attr);
      values.push_back(
          arith::ConstantIndexOp::create(builder, loc, constAttr.getInt()));
    }
  }
  return values;
}

// Required by TilingInterface.
//
// Constructs the iteration domain (i.e., a list of Ranges) for a given
// air::ChannelInterface op. This domain represents the loop bounds to iterate
// over the tensor or memref involved in the channel operation.
//
// If the channel op does not explicitly define `sizes`, it infers the domain
// from the full shape of the memref. Otherwise, it uses the (offset, size,
// stride) attributes encoded in the op.
SmallVector<Range> getIterationDomainFromChanIf(OpBuilder &builder,
                                                air::ChannelInterface op) {
  Location loc = op.getLoc();
  Value source = op.getMemref();

  if (op.getSizes().empty()) {
    // Case 1: Sizes are not explicitly provided: use full shape of the memref.
    int64_t operandRank =
        dyn_cast<MemRefType>(op.getMemref().getType()).getRank();
    SmallVector<Range> loopBounds(operandRank);
    Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
    Value one = arith::ConstantIndexOp::create(builder, loc, 1);
    for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
      loopBounds[dim].offset = zero;
      loopBounds[dim].size = getDimValue(builder, loc, source, dim);
      loopBounds[dim].stride = one;
    }
    return loopBounds;
  } else {
    // Case 2: Sizes are explicitly provided: construct the domain directly from
    // op attributes.
    int64_t operandRank = op.getSizes().size();
    SmallVector<Range> loopBounds(operandRank);
    for (auto dim : llvm::seq<int64_t>(0, operandRank)) {
      loopBounds[dim].offset = op.getOffsets()[dim];
      loopBounds[dim].size = op.getSizes()[dim];
      loopBounds[dim].stride = op.getStrides()[dim];
    }
    return loopBounds;
  }
}

// Required by TilingInterface.
//
// Returns a vector of `IteratorType::parallel` for each dimension in the
// channel op. The rank is either derived from the full memref shape or the
// explicit sizes if provided.
SmallVector<utils::IteratorType>
getLoopIteratorTypesFromChanIf(air::ChannelInterface op) {
  int64_t operandRank = 0;

  // If sizes are not provided, infer rank from the memref type.
  if (op.getSizes().empty())
    operandRank = dyn_cast<MemRefType>(op.getMemref().getType()).getRank();
  else
    operandRank = op.getSizes().size();

  // All dimensions are marked as parallel iterators.
  SmallVector<utils::IteratorType> iteratorTypes(operandRank,
                                                 utils::IteratorType::parallel);
  return iteratorTypes;
}

// Required by TilingInterface.
//
// Instantiates a tiled version of an air::ChannelPut or ChannelGet operation.
//
// It creates a sliced view of the original memref using (offsets, sizes,
// strides), and uses that slice to construct a new instance of the same channel
// operation.
//
// Template parameter PutGetTy should be either air::ChannelPutOp or
// air::ChannelGetOp.
template <typename PutGetTy>
FailureOr<TilingResult> getTiledImplementationFromChanIf(
    OpBuilder &builder, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, air::ChannelInterface op) {
  int64_t rank = 0;
  auto oneAttr = builder.getI64IntegerAttr(1);

  // Determine the rank from memref or explicit sizes.
  if (op.getSizes().empty())
    rank = dyn_cast<MemRefType>(op.getMemref().getType()).getRank();
  else
    rank = op.getSizes().size();

  // Compute strides for slicing — use op's strides if available.
  SmallVector<OpFoldResult> strides(rank, oneAttr);
  if (!op.getSizes().empty()) {
    for (auto dim : llvm::seq<int64_t>(0, rank)) {
      strides[dim] = op.getStrides()[dim];
    }
  }

  // Compute a sliced subview of the memref.
  Operation *inputSlice =
      getSlice(builder, op.getLoc(), op.getMemref(), offsets, sizes, strides);
  if (!inputSlice) {
    return op.emitOpError("failed to compute input slice");
  }

  // Convert strides to Values.
  SmallVector<Value> stridesAsValues(rank);
  for (auto dim : llvm::seq<int64_t>(0, rank)) {
    if (op.getStrides().empty())
      stridesAsValues[dim] =
          arith::ConstantIndexOp::create(builder, op.getLoc(), 1);
    else
      stridesAsValues[dim] = op.getStrides()[dim];
  }

  // Clone a new tiled op with the sliced subview and same async/channel
  // attributes.
  air::AsyncOpInterface asyncIf =
      dyn_cast<air::AsyncOpInterface>(op.getOperation());
  PutGetTy tiledOp = PutGetTy::create(
      builder, op.getLoc(), op->getResultTypes(),
      asyncIf.getAsyncDependencies(), op.getChanName(), op.getIndices(),
      inputSlice->getResult(0),
      materializeOpFoldResultAsValues(offsets, op.getLoc(), builder),
      materializeOpFoldResultAsValues(sizes, op.getLoc(), builder),
      materializeOpFoldResultAsValues(strides, op.getLoc(), builder));

  // Return the tiling result, including the new op and the sliced input.
  return TilingResult{{tiledOp},
                      SmallVector<Value>{},
                      llvm::to_vector(ArrayRef<Operation *>{inputSlice})};
}

//
// Channel put op
//

// Fold memref.cast operations for cascade channels (simplified version)
template <typename OpT>
static LogicalResult FoldMemrefCastOnChannelOp(OpT op,
                                               PatternRewriter &rewriter) {
  Value memref = op.getMemref();
  if (!memref)
    return failure();

  // Check if the memref is directly produced by a memref.cast
  auto castOp = dyn_cast_if_present<memref::CastOp>(memref.getDefiningOp());
  if (!castOp)
    return failure();

  // Only proceed if offsets, sizes, and strides are empty (no explicit access
  // pattern)
  if (!op.getOffsets().empty() || !op.getSizes().empty() ||
      !op.getStrides().empty())
    return failure();

  // Replace the channel op with a new one using the cast's source
  rewriter.replaceOpWithNewOp<OpT>(
      op, op->getResultTypes(), op.getAsyncDependencies(), op.getChanName(),
      op.getIndices(), castOp.getSource(), op.getOffsets(), op.getSizes(),
      op.getStrides());

  return success();
}

template <typename OpT>
static LogicalResult ComposeMemrefOpOnChannelOp(OpT op,
                                                PatternRewriter &rewriter) {

  // Lambda version of `getChannelDeclarationThroughSymbol` method defined in
  // `Util/Utils.cpp`. It is duplicated here because `Util/Utils.cpp` depends on
  // this file, so direct inclusion is not possible.
  auto getChannelDeclarationThroughSymbol = [](air::ChannelInterface op) {
    if (!op)
      // Return an empty ChannelOp if the input operation is invalid.
      return air::ChannelOp();

    // Traverse up through the operation's parents until a symbol table is
    // found.
    Operation *parent = op;
    while ((parent = parent->getParentOp())) {
      if (parent->hasTrait<OpTrait::SymbolTable>()) {
        auto st = mlir::SymbolTable::lookupSymbolIn(parent, op.getChanName());
        if (auto chanOp = dyn_cast_if_present<air::ChannelOp>(st))
          return chanOp;
      }
    }

    // No matching channel declaration found in any enclosing symbol tables.
    return air::ChannelOp();
  };

  // Extract the memref operand from the operation.
  Value memref = op.getMemref();
  if (!memref)
    // If there is no associated memref, signal a failure.
    return failure();
  // Resolve the channel declaration for the given channel interface operation.
  air::ChannelOp chan = getChannelDeclarationThroughSymbol(op);
  if (!chan)
    // If the channel declaration cannot be resolved, signal a failure.
    return failure();
  // If the channel is of type "cascade", try to fold memref.cast but skip full
  // composition
  if (chan.getChannelType() == "cascade")
    return FoldMemrefCastOnChannelOp(op, rewriter);

  // Init. memref type and offsets from memref's defining op's input type
  Value input_memref = memref;
  SmallVector<Value> offsets, sizes, strides;
  offsets = op.getOffsets();
  sizes = op.getSizes();
  strides = op.getStrides();

  auto composeMemrefRes =
      ComposeMemrefOp(memref, rewriter, input_memref, offsets, sizes, strides);
  auto canonicalizeListsRes =
      canonicalizeEmptyLists(offsets, sizes, strides,
                             dyn_cast<BaseMemRefType>(input_memref.getType()));

  if (failed(composeMemrefRes) && failed(canonicalizeListsRes))
    return failure();

  rewriter.replaceOpWithNewOp<OpT>(
      op, op->getResultTypes(), op.getAsyncDependencies(), op.getChanName(),
      op.getIndices(), input_memref, offsets, sizes, strides);

  return success();
}

// The following methods are required by TilingInterface.
SmallVector<Range> air::ChannelPutOp::getIterationDomain(OpBuilder &builder) {
  air::ChannelInterface put = dyn_cast<air::ChannelInterface>(getOperation());
  return getIterationDomainFromChanIf(builder, put);
}

SmallVector<utils::IteratorType> air::ChannelPutOp::getLoopIteratorTypes() {
  air::ChannelInterface put = dyn_cast<air::ChannelInterface>(getOperation());
  return getLoopIteratorTypesFromChanIf(put);
}

FailureOr<TilingResult>
air::ChannelPutOp::getTiledImplementation(OpBuilder &builder,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes) {
  air::ChannelInterface put = dyn_cast<air::ChannelInterface>(getOperation());
  return getTiledImplementationFromChanIf<air::ChannelPutOp>(builder, offsets,
                                                             sizes, put);
}

LogicalResult air::ChannelPutOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  // An optional result (air::AsyncTokenType) may be returned, but it doesn't
  // represent any tile.
  return success();
}

void air::ChannelPutOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnChannelOp<air::ChannelPutOp>);
  patterns.add(CanonicalizeAsyncOpDeps<air::ChannelPutOp>);
}

//
// Channel get op
//

// The following methods are required by TilingInterface.
SmallVector<Range> air::ChannelGetOp::getIterationDomain(OpBuilder &builder) {
  air::ChannelInterface get = dyn_cast<air::ChannelInterface>(getOperation());
  return getIterationDomainFromChanIf(builder, get);
}

SmallVector<utils::IteratorType> air::ChannelGetOp::getLoopIteratorTypes() {
  air::ChannelInterface get = dyn_cast<air::ChannelInterface>(getOperation());
  return getLoopIteratorTypesFromChanIf(get);
}

FailureOr<TilingResult>
air::ChannelGetOp::getTiledImplementation(OpBuilder &builder,
                                          ArrayRef<OpFoldResult> offsets,
                                          ArrayRef<OpFoldResult> sizes) {
  air::ChannelInterface get = dyn_cast<air::ChannelInterface>(getOperation());
  return getTiledImplementationFromChanIf<air::ChannelGetOp>(builder, offsets,
                                                             sizes, get);
}

LogicalResult air::ChannelGetOp::getResultTilePosition(
    OpBuilder &builder, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  // An optional result (air::AsyncTokenType) may be returned, but it doesn't
  // represent any tile.
  return success();
}

void air::ChannelGetOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add(ComposeMemrefOpOnChannelOp<air::ChannelGetOp>);
  patterns.add(CanonicalizeAsyncOpDeps<air::ChannelGetOp>);
}

//
// Channel op
//

LogicalResult air::ChannelOp::verify() {
  if (isBroadcast()) {
    auto bundle_size = getSize();
    auto broadcast_shape = getBroadcastShape();
    if (bundle_size.size() != broadcast_shape.size())
      return emitOpError("bundle rank should match broadcast_shape rank");
  }
  return success();
}

int air::ChannelOp::getBroadcastDimension() {
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

static LogicalResult FoldChannel(air::ChannelOp op, PatternRewriter &rewriter) {

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

      std::vector<air::ChannelPutOp> puts;
      std::vector<air::ChannelGetOp> gets;
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

void air::ChannelOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add(FoldChannel);
}

//
// Custom op
//

void air::CustomOp::build(OpBuilder &builder, OperationState &result,
                          ValueRange asyncDependencies,
                          ValueRange customOperands, bool isAsync,
                          ArrayRef<NamedAttribute> attrs) {

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

void air::CustomOp::build(OpBuilder &builder, OperationState &result,
                          ValueRange customOperands) {

  build(builder, result, {}, customOperands, false);
}

void air::CustomOp::print(OpAsmPrinter &p) {

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

ParseResult air::CustomOp::parse(OpAsmParser &parser, OperationState &result) {

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

} // namespace xilinx

#include "air/Dialect/AIR/AIROpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIR.cpp.inc"
