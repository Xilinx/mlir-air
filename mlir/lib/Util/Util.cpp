//===- Util.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/Util.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "iostream"

#define DEBUG_TYPE "air-util"

using namespace mlir;

namespace xilinx {

const StringLiteral air::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

static std::string getMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = llvm::dyn_cast<const MemRefType>(ty)) {
    ret << "M";
    ret << mrt.getMemorySpaceAsInt();
    if (mrt.hasStaticShape()) {
      auto shape = mrt.getShape();
      for (auto s : shape)
        ret << s << "x";
    } else if (mrt.hasRank()) {
      ret << "D" << mrt.getRank();
    }
    const Type elem = mrt.getElementType();
    ret << getMangledType(elem);
  } else if (FloatType ft = llvm::dyn_cast<FloatType>(ty)) {
    ret << "F" << ft.getWidth();
  } else if (const IntegerType it = llvm::dyn_cast<const IntegerType>(ty)) {
    ret << "I" << it.getWidth();
  } else if (const IndexType it = llvm::dyn_cast<const IndexType>(ty)) {
    ret << "I64";
  } else if (llvm::dyn_cast<air::AsyncTokenType>(ty)) {
    ret << "E";
  } else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

static std::string getMangledFuncName(ModuleOp module, std::string prefix,
                                      FunctionType fnTy) {
  std::string sep = "_";

  auto resultTy = fnTy.getResults();
  auto operTy = fnTy.getInputs();

  std::string ret = prefix;
  for (const Type t : resultTy)
    ret = ret + sep + "r" + getMangledType(t);
  for (const Type t : operTy)
    ret = ret + sep + getMangledType(t);

  return ret;
}

func::FuncOp air::getMangledFunction(ModuleOp module, std::string prefix,
                                     ArrayRef<Value> operands,
                                     ArrayRef<Type> retTys) {
  Builder builder(module);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix, fnTy);
  auto fn = module.lookupSymbol<func::FuncOp>(fnName);

  if (!fn) {
    fn = func::FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  return fn;
}

LogicalResult air::normalizeLoop(affine::AffineForOp afo) {
  auto ubMap = afo.getUpperBoundMap();
  auto lbMap = afo.getLowerBoundMap();
  auto ctx = afo.getContext();
  auto loc = afo.getLoc();

  auto step_expr = getAffineConstantExpr(afo.getStepAsInt(), ctx);

  auto ub_expr = ubMap.getResult(0);
  auto lb_expr = lbMap.getResult(0);
  auto sub_expr = ub_expr - lb_expr;
  auto new_ub_expr = sub_expr.ceilDiv(step_expr);

  auto iv = afo.getInductionVar();

  afo.setLowerBoundMap(AffineMap::get(0, 0, getAffineConstantExpr(0, ctx)));
  afo.setUpperBoundMap(AffineMap::get(0, 0, new_ub_expr));
  afo.setStep(1);

  auto dim0_expr = getAffineDimExpr(0, ctx);
  auto iv_expr = dim0_expr * step_expr + lb_expr;
  auto iv_map = AffineMap::get(1, 0, iv_expr);
  auto builder = OpBuilder::atBlockBegin(afo.getBody());
  auto new_iv = builder.create<affine::AffineApplyOp>(loc, iv_map, iv);
  SmallPtrSet<Operation *, 1> keep{new_iv};
  iv.replaceAllUsesExcept(new_iv, keep);
  return success();
}

uint64_t air::getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t air::getTensorVolume(const Type ty) {
  if (auto t = llvm::dyn_cast<ShapedType>(ty)) {
    return getTensorVolume(t);
  } else {
    return 1;
  }
}

SmallVector<int> air::getTensorShape(const ShapedType ty) {
  if (!ty.hasRank())
    return SmallVector<int>(1);
  SmallVector<int> shape = {};
  for (auto &d : ty.getShape())
    shape.push_back(d);
  return shape;
}

SmallVector<int> air::getTensorShape(const Type ty) {
  if (auto t = llvm::dyn_cast<ShapedType>(ty)) {
    return getTensorShape(t);
  } else {
    return SmallVector<int>(1);
  }
}

std::string air::getElementTypeAsString(const mlir::Type ty) {
  if (auto st = llvm::dyn_cast<mlir::ShapedType>(ty)) {
    return to_string(st.getElementType());
  } else {
    return to_string(ty);
  }
}

// An incomplete lookup table of common data types
uint64_t air::getElementSizeInBytes(const mlir::Type ty) {
  if (auto memrefTy = llvm::cast<BaseMemRefType>(ty)) {
    return memrefTy.getElementTypeBitWidth() / 8;
  }
  auto typeAsString = getElementTypeAsString(ty);
  if (typeAsString == "i32")
    return 4;
  else if (typeAsString == "i16")
    return 2;
  else if (typeAsString == "i8")
    return 1;
  else if (typeAsString == "f32")
    return 4;
  else if (typeAsString == "f16")
    return 2;
  else if (typeAsString == "bf16")
    return 2;
  else
    return 0;
}

// Get the parent scf.for op of an iter_arg
scf::ForOp air::getForRegionIterArgsOwner(Value val) {
  auto ivArg = llvm::dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return scf::ForOp();
  if (!ivArg.getOwner()) {
    val.getDefiningOp()->emitOpError("unlinked block argument");
    return scf::ForOp();
  }
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<scf::ForOp>(containingOp);
}

// Get the parent scf.parallel op of an init_val
scf::ParallelOp air::getParallelRegionInitValsOwner(Operation *op, Value val) {
  if (auto parent_parallel_op = op->getParentOfType<scf::ParallelOp>()) {
    for (auto init_val : parent_parallel_op.getInitVals()) {
      if (init_val == val)
        return parent_parallel_op;
    }
  }
  return scf::ParallelOp();
}

// Get the parent air.launch_herd op of a tile id
air::HerdOp air::getHerdArgOwner(Value val) {
  auto ivArg = llvm::dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return air::HerdOp();
  if (!ivArg.getOwner()) {
    val.getDefiningOp()->emitOpError("unlinked block argument");
    return air::HerdOp();
  }
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HerdOp>(containingOp);
}

// Get the parent air.hierarchy op of a tile id
air::HierarchyInterface air::getHierarchyArgOwner(Value val) {
  auto ivArg = llvm::dyn_cast<BlockArgument>(val);
  if (!ivArg)
    return air::HierarchyInterface();
  if (!ivArg.getOwner()) {
    val.getDefiningOp()->emitOpError("unlinked block argument");
    return air::HierarchyInterface();
  }
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HierarchyInterface>(containingOp);
}

// Get a static scf.for trip count as int
std::optional<int64_t> air::getStaticScfForTripCountAsInt(scf::ForOp for_op) {
  std::optional<int64_t> output = std::nullopt;
  // Get for loop iteration count
  std::optional<int64_t> LbCstOp =
      mlir::getConstantIntValue(for_op.getLowerBound());
  std::optional<int64_t> UbCstOp =
      mlir::getConstantIntValue(for_op.getUpperBound());
  std::optional<int64_t> StepCstOp =
      mlir::getConstantIntValue(for_op.getStep());
  if (LbCstOp && UbCstOp && StepCstOp)
    output = llvm::divideCeilSigned((*UbCstOp - *LbCstOp), *StepCstOp);
  return output;
}

// Get a static affine.for trip count as int
std::optional<int64_t>
air::getStaticAffineForTripCountAsInt(affine::AffineForOp for_op) {
  std::optional<int64_t> output = std::nullopt;
  if (for_op.hasConstantBounds()) {
    output = llvm::divideCeilSigned(
        (for_op.getConstantUpperBound() - for_op.getConstantLowerBound()),
        for_op.getStepAsInt());
  }
  return output;
}

// Get operation's "id" attribute
int air::getIdAttr(Operation *op) {
  auto idAttr = op->getAttrOfType<IntegerAttr>("id");
  if (idAttr)
    return idAttr.getInt();
  else
    return -1;
}

// Re-assign operation ids for air::MemcpyInterface ops.
void air::renumberMemcpyIfOps(Region *region) {
  unsigned id = 0;
  region->walk([&](air::MemcpyInterface memcpyOpIf) {
    memcpyOpIf->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(memcpyOpIf->getContext(), 32), ++id));
  });
}
// Re-assign operation ids for air::MemcpyInterface ops, and keep a reverse_map
// which recovers the original op ids.
void air::renumberMemcpyIfOps(Region *region, std::map<int, int> &reverse_map) {
  unsigned id = 0;
  region->walk([&](air::MemcpyInterface memcpyOpIf) {
    // Update a reverse map for op ids
    if (memcpyOpIf->hasAttr("id")) {
      reverse_map[id + 1] = memcpyOpIf.getId();
    }
    memcpyOpIf->setAttr(
        "id", mlir::IntegerAttr::get(
                  mlir::IntegerType::get(memcpyOpIf->getContext(), 32), ++id));
  });
}

// Return op name as string
std::string air::to_string(Operation *op) {
  return op->getName().getStringRef().str();
}

// Return mlir type name as string
std::string air::to_string(mlir::Type t) {
  std::string type_str;
  llvm::raw_string_ostream rso(type_str);
  t.print(rso);
  return type_str;
}

// Create channel name as string
std::string air::createChannelName(Operation *scope) {
  if (!scope->hasTrait<OpTrait::SymbolTable>()) {
    scope->emitOpError("has no symbol table trait");
  }
  std::string new_cname = "channel_0";
  std::string cname = "channel";
  int which_try = 0;
  while (mlir::SymbolTable::lookupSymbolIn(scope, new_cname))
    new_cname = cname + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Return memory space as string
std::string air::getMemorySpaceAsString(Value memref) {
  if (!llvm::isa<BaseMemRefType>(memref.getType())) {
    memref.getDefiningOp()->emitOpError("value returned is not a memref");
    return "";
  }
  auto memory_space_as_int =
      llvm::dyn_cast<BaseMemRefType>(memref.getType()).getMemorySpaceAsInt();
  std::string memorySpaceStr = "";
  if (memory_space_as_int == (int)air::MemorySpace::L1) {
    memorySpaceStr = "L1";
  } else if (memory_space_as_int == (int)air::MemorySpace::L2) {
    memorySpaceStr = "L2";
  } else if (memory_space_as_int == (int)air::MemorySpace::L3) {
    memorySpaceStr = "L3";
  } else {
    memref.getDefiningOp()->emitOpError(
        "value returned has an unexpected memory space");
  }
  return memorySpaceStr;
}

// Returns the first affine if op in block; nullptr otherwise
affine::AffineIfOp air::getAffineIfInBlock(mlir::Block *block) {
  for (auto op : block->getOps<affine::AffineIfOp>()) {
    return op;
  }
  return affine::AffineIfOp();
}

// Returns the first air.dma op in block; nullptr otherwise
air::DmaMemcpyNdOp air::getAIRDmaInBlock(mlir::Block *block) {
  for (auto op : block->getOps<air::DmaMemcpyNdOp>()) {
    return op;
  }
  return air::DmaMemcpyNdOp();
}

// Erase a kernel operand from air.hierarchy op
void air::eraseAIRHierarchyOperand(air::HierarchyInterface op, unsigned index) {
  if (index + 1 > op->getNumOperands()) {
    op->emitOpError("index out of range");
    return;
  }
  auto numAsyncDeps = dyn_cast<air::AsyncOpInterface>(op.getOperation())
                          .getAsyncDependencies()
                          .size();
  auto removed_operand_index = index + numAsyncDeps + op.getNumDims();
  op->eraseOperands(removed_operand_index);
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.asArrayRef()) {
    sizes.push_back(size);
  }
  // Find which bin the erased operand belongs to in OperandSegmentSizes
  unsigned sum = 0;
  unsigned i = 0;
  for (auto s : sizes) {
    sum += s;
    if (sum > removed_operand_index) {
      sizes[i]--;
      break;
    }
    i++;
  }
  op->setAttr(attrName, Builder(op->getContext()).getDenseI32ArrayAttr(sizes));
}

// Get channel declaration through channel symbol
air::ChannelOp
air::getChannelDeclarationThroughSymbol(air::ChannelInterface op) {
  if (!op)
    return air::ChannelOp();
  Operation *parent = op;
  while ((parent = parent->getParentOp())) {
    if (parent->hasTrait<OpTrait::SymbolTable>()) {
      auto st = mlir::SymbolTable::lookupSymbolIn(parent, op.getChanName());
      if (auto chanOp = dyn_cast_if_present<air::ChannelOp>(st))
        return chanOp;
    }
  }
  return air::ChannelOp();
}

// Get ChannelPutOp through ChannelOp
std::vector<air::ChannelPutOp>
air::getChannelPutOpThroughSymbol(air::ChannelOp channel, Operation *scope) {

  if (!scope)
    scope = channel->getParentOfType<ModuleOp>();

  auto attr =
      channel->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  std::vector<ChannelPutOp> channelPuts;
  scope->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](air::ChannelPutOp put) {
        if (put.getChanName() == attr) {
          channelPuts.push_back(put);
          WalkResult::advance();
        }
        WalkResult::advance();
      });

  return channelPuts;
}

// Get ChannelGetOps through ChannelOp
std::vector<air::ChannelGetOp>
air::getChannelGetOpThroughSymbol(air::ChannelOp channel, Operation *scope) {

  if (!scope)
    scope = channel->getParentOfType<ModuleOp>();

  auto attr =
      channel->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
  std::vector<ChannelGetOp> channelGets;
  scope->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](air::ChannelGetOp get) {
        if (get.getChanName() == attr) {
          channelGets.push_back(get);
          WalkResult::advance();
        }
        WalkResult::advance();
      });

  return channelGets;
}

// Get the other channel op through channel symbol
std::vector<air::ChannelGetOp>
air::getTheOtherChannelOpThroughSymbol(air::ChannelPutOp put) {
  auto channel_op = getChannelDeclarationThroughSymbol(
      dyn_cast<air::ChannelInterface>(put.getOperation()));
  return getChannelGetOpThroughSymbol(channel_op);
}

// Get channel type from air.channel_interface ops.
FailureOr<StringRef> air::getChannelType(air::MemcpyInterface memcpyIfOp) {
  if (!memcpyIfOp)
    return failure();
  auto chanIfOp = dyn_cast<air::ChannelInterface>(memcpyIfOp.getOperation());
  if (!chanIfOp)
    return StringRef("dma_stream");
  auto chanOp = getChannelDeclarationThroughSymbol(chanIfOp);
  if (chanOp) {
    return chanOp.getChannelType();
  }
  return failure();
}

// Get the other channel op through channel symbol
std::vector<air::ChannelPutOp>
air::getTheOtherChannelOpThroughSymbol(air::ChannelGetOp get) {
  auto channel_op = getChannelDeclarationThroughSymbol(
      dyn_cast<air::ChannelInterface>(get.getOperation()));
  return getChannelPutOpThroughSymbol(channel_op);
}

std::vector<air::ChannelInterface>
air::getTheOtherChannelOpThroughSymbol(air::ChannelInterface op) {
  if (auto put = dyn_cast<air::ChannelPutOp>(op.getOperation())) {
    auto vec = getTheOtherChannelOpThroughSymbol(put);
    std::vector<air::ChannelInterface> output;
    for (auto v : vec)
      output.push_back(dyn_cast<air::ChannelInterface>(v.getOperation()));
    return output;
  } else if (auto get = dyn_cast<air::ChannelGetOp>(op.getOperation())) {
    auto vec = getTheOtherChannelOpThroughSymbol(get);
    std::vector<air::ChannelInterface> output;
    for (auto v : vec)
      output.push_back(dyn_cast<air::ChannelInterface>(v.getOperation()));
    return output;
  } else
    return std::vector<air::ChannelInterface>();
}

// Get integer index to metadataArray, from channel bundle indices.
std::optional<int>
air::getIndexToMetadataArrayFromChannelIndices(air::ChannelInterface op) {
  std::optional<int> res = std::nullopt;
  std::vector<unsigned> indicesUint =
      air::convertVecOfConstIndexToVecOfUInt(op.getIndices());
  auto channelBundleSize =
      air::getChannelDeclarationThroughSymbol(op).getSize();

  auto arrayAttrToUIntVector =
      [](mlir::ArrayAttr attr) -> std::vector<unsigned int> {
    std::vector<unsigned int> vec;
    for (mlir::Attribute a : attr) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(a))
        vec.push_back(static_cast<unsigned int>(intAttr.getInt()));
      else
        llvm::errs() << "Warning: Non-integer attribute in ArrayAttr\n";
    }
    return vec;
  };

  std::vector<unsigned int> uIntVec = arrayAttrToUIntVector(channelBundleSize);
  res = air::getIteratorFromMDVector(uIntVec, indicesUint);
  return res;
}

// Get sizes from integerset
void air::getSizesFromIntegerSet(MLIRContext *ctx, IntegerSet int_set,
                                 SmallVector<int, 2> &lbs_int,
                                 SmallVector<int, 2> &ubs_int) {

  auto constraints = int_set.getConstraints();
  auto eqFlags = int_set.getEqFlags();

  // Get an affine expression set made of zeros
  SmallVector<AffineExpr, 2> zero_syms;
  for (unsigned i = 0; i < int_set.getNumSymbols(); i++) {
    zero_syms.push_back(getAffineConstantExpr(0, ctx));
  }

  // Fill in lbs and ubs vectors by evaluating affine set
  for (unsigned i = 0; i < int_set.getNumSymbols(); i++) {
    int c_iter = 0;
    for (auto c : constraints) {
      if (c.isSymbolicOrConstant()) {
        auto newC = c.replaceSymbols(zero_syms);
        auto expr =
            dyn_cast<AffineConstantExpr>(simplifyAffineExpr(newC, 0, 1));
        int v = expr.getValue();
        if (c.isFunctionOfSymbol(i)) {
          if (eqFlags[c_iter]) {
            v = abs(v);
            lbs_int[i] = v;
            ubs_int[i] = v;
          } else {
            if (v > 0)
              ubs_int[i] = v;
            else if (v < 0)
              lbs_int[i] = -v;
            else
              lbs_int[i] = v;
          }
        }
      }
      c_iter++;
    }
  }
}

// Get spatial sizes from spatial loop
void air::getSizesFromSpatialLoop(Operation *spatial_loop,
                                  SmallVector<int, 2> &lbs_spatial,
                                  SmallVector<int, 2> &ubs_spatial) {
  if (auto scf_par = dyn_cast<scf::ParallelOp>(spatial_loop)) {
    for (unsigned i = 0; i < scf_par.getLowerBound().size(); i++) {
      auto lbCstOp =
          scf_par.getLowerBound()[i].getDefiningOp<arith::ConstantIndexOp>();
      auto ubCstOp =
          scf_par.getUpperBound()[i].getDefiningOp<arith::ConstantIndexOp>();
      auto stepCstOp =
          scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
      lbs_spatial.push_back(
          llvm::divideCeilSigned(lbCstOp.value(), stepCstOp.value()));
      ubs_spatial.push_back(
          llvm::divideCeilSigned(ubCstOp.value(), stepCstOp.value()) - 1);
    }
  } else if (auto hier = dyn_cast<air::HierarchyInterface>(spatial_loop)) {
    for (unsigned i = 0; i < hier.getSizeOperands().size(); i++) {
      lbs_spatial.push_back(0);
      ubs_spatial.push_back(hier.getSizeOperands()[i]
                                .getDefiningOp<arith::ConstantIndexOp>()
                                .value() -
                            1);
    }
  }
}

// Get else sizes from affine.if. Assumption: rectangular input, then and else
// sizes only
void air::getElseSizesFromAffineIf(SmallVector<int, 2> &lbs_in,
                                   SmallVector<int, 2> &ubs_in,
                                   SmallVector<int, 2> &lbs_then,
                                   SmallVector<int, 2> &ubs_then) {
  for (unsigned i = 0; i < lbs_in.size(); i++) {
    if ((lbs_in[i] != lbs_then[i])) {
      ubs_in[i] = lbs_then[i] - 1;
      lbs_in[i] = lbs_in[i];
      return;
    } else if ((ubs_in[i] != ubs_then[i])) {
      lbs_in[i] = ubs_then[i] + 1;
      ubs_in[i] = ubs_in[i];
      return;
    }
  }
}

// Check if op hits affine.if condition
bool air::positionHitsAffineIfCondition(Operation *op,
                                        std::vector<unsigned> position) {
  std::vector<Operation *> affine_if_nest;
  Operation *spatial_loop = nullptr;
  getAffineIfNestAndSpatialLoopFromOp(op, affine_if_nest, spatial_loop);
  return positionHitsAffineIfCondition(op, spatial_loop, affine_if_nest,
                                       position);
}

// Walk affine.if then and else blocks and check if current core lies in
// condition
bool air::positionHitsAffineIfCondition(Operation *op, Operation *spatial_loop,
                                        std::vector<Operation *> affine_if_nest,
                                        std::vector<unsigned> position) {
  SmallVector<int, 2> lbs_spatial;
  SmallVector<int, 2> ubs_spatial;
  getSizesFromSpatialLoop(spatial_loop, lbs_spatial, ubs_spatial);

  // Walk through affine.if nest (in reverse order through vector)
  for (auto it = affine_if_nest.rbegin(); it != affine_if_nest.rend(); ++it) {
    auto affine_if = dyn_cast<affine::AffineIfOp>(*it);
    // Get then integerset sizes
    SmallVector<int, 2> lbs_int = {0, 0};
    SmallVector<int, 2> ubs_int = {0, 0};
    IntegerSet int_set = affine_if.getIntegerSet();
    getSizesFromIntegerSet(affine_if->getContext(), int_set, lbs_int, ubs_int);
    // If found then block containing op
    if (affine_if.getThenBlock()->findAncestorOpInBlock(*op)) {
      bool hit = true;
      for (unsigned i = 0; i < lbs_int.size(); i++) {
        if ((position[i] < (unsigned)lbs_int[i]) ||
            (position[i] > (unsigned)ubs_int[i])) {
          hit = false;
        }
      }
      return hit;
    }
    // Else keep going, while updating the spatial sizes wrt else condition
    else {
      getElseSizesFromAffineIf(lbs_spatial, ubs_spatial, lbs_int, ubs_int);
    }
  }
  // If op isn't in any then blocks in affine.if nest
  bool hit = true;
  for (unsigned i = 0; i < lbs_spatial.size(); i++) {
    if ((position[i] < (unsigned)lbs_spatial[i]) ||
        (position[i] > (unsigned)ubs_spatial[i])) {
      hit = false;
    }
  }
  return hit;
}

// Get parent affine.if nest and ancestor spatial loop from op
Operation *air::getAffineIfNestAndSpatialLoopFromOp(
    Operation *op, std::vector<Operation *> &affine_if_nest,
    Operation *&spatial_loop) {
  Operation *parent = op;
  while ((!isa<scf::ParallelOp>(parent)) &&
         (!isa<air::HierarchyInterface>(parent))) {
    if (isa<affine::AffineIfOp>(parent)) {
      affine_if_nest.push_back(parent);
    }
    parent = parent->getParentOp();
    if (isa<func::FuncOp>(parent)) {
      // Found affine.if not filtering on a spatial loop (air.hierarchy or
      // scf.parallel)
      return nullptr;
    }
  }
  // Skip over the first parent hierarchy or parallel loop
  spatial_loop = parent;
  parent = parent->getParentOp();
  return parent;
}

// Evaluate the condition lower and upper boundaries that the specified op is
// hitting, from an affine if nest. Assumes a rectangular condition bound
// region.
SmallVector<std::pair<int, int>>
air::getRectangularConditionBoundsThroughAffineIfs(
    Operation *op, Operation *spatial_loop,
    std::vector<Operation *> affine_if_nest) {
  SmallVector<int, 2> lbs_spatial;
  SmallVector<int, 2> ubs_spatial;
  air::getSizesFromSpatialLoop(spatial_loop, lbs_spatial, ubs_spatial);

  // Walk through affine.if nest (in reverse order through vector)
  for (auto it = affine_if_nest.rbegin(); it != affine_if_nest.rend(); ++it) {
    auto affine_if = dyn_cast<affine::AffineIfOp>(*it);
    // Get then integerset sizes
    SmallVector<int, 2> lbs_int = {0, 0};
    SmallVector<int, 2> ubs_int = {0, 0};
    IntegerSet int_set = affine_if.getIntegerSet();
    air::getSizesFromIntegerSet(affine_if->getContext(), int_set, lbs_int,
                                ubs_int);
    // If found then block containing op
    SmallVector<std::pair<int, int>> conditionBounds; // <LB, UB>
    if (affine_if.getThenBlock()->findAncestorOpInBlock(*op)) {
      for (unsigned i = 0; i < lbs_int.size(); i++)
        conditionBounds.push_back(std::make_pair(lbs_int[i], ubs_int[i]));
      return conditionBounds;
    }
    // Else keep going, while updating the spatial sizes wrt else condition
    else {
      air::getElseSizesFromAffineIf(lbs_spatial, ubs_spatial, lbs_int, ubs_int);
    }
  }

  // If op isn't in any then blocks in affine.if nest
  SmallVector<std::pair<int, int>> conditionBounds; // <LB, UB>
  for (unsigned i = 0; i < lbs_spatial.size(); i++)
    conditionBounds.push_back(std::make_pair(lbs_spatial[i], ubs_spatial[i]));
  return conditionBounds;
}

// Evaluate the integer value of affine set expression if the only symbolic
// identifier is replaced with zero
int air::evaluateSymbolEqualityInSet(AffineExpr c, MLIRContext *ctx) {
  if (!c.isSymbolicOrConstant())
    return 0; // Constraint has dimension identifier.
  SmallVector<AffineExpr, 2> zero_syms{
      getAffineConstantExpr(0, ctx),
      getAffineConstantExpr(0, ctx),
  };
  auto newC = c.replaceSymbols(zero_syms);
  auto expr = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(newC, 0, 1));
  if (!expr)
    return 0;
  int result = expr.getValue();
  // Both + and - constant eval are legal for AffineExpr
  return (result >= 0) ? (result) : (-result);
}

// Check if an operand of an operation is read or write access
char air::checkOpOperandReadOrWrite(Value v, Operation *owner) {
  for (auto &op_operand : owner->getOpOperands()) {
    if (op_operand.is(v)) {
      return checkOpOperandReadOrWrite(op_operand);
    }
  }
  // Value is not an opoperand of the operation
  return 'e';
}
char air::checkOpOperandReadOrWrite(mlir::OpOperand &op_operand) {
  auto owner = op_operand.getOwner();
  // If used in DmaMemcpy Op
  if (auto dma = dyn_cast<xilinx::air::DmaMemcpyNdOp>(owner)) {
    if (op_operand.is(dma.getSrcMemref())) {
      return 'r';
    } else if (op_operand.is(dma.getDstMemref())) {
      return 'w';
    } else {
      return 'u';
    }
  }
  // If used in Channel Put Op
  else if (auto channel_put = dyn_cast<xilinx::air::ChannelPutOp>(owner)) {
    if (op_operand.is(channel_put.getSrc())) {
      return 'r';
    } else {
      return 'u';
    }
  }
  // If used in Channel Get Op
  else if (auto channel_get = dyn_cast<xilinx::air::ChannelGetOp>(owner)) {
    if (op_operand.is(channel_get.getDst())) {
      return 'w';
    } else {
      return 'u';
    }
  }
  // If used in a linalg op
  else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(owner)) {
    if (op_operand.getOperandNumber() <
        linalgop.getNumDpsInputs() + linalgop.getNumDpsInits()) {
      return 'r';
    } else if (op_operand.getOperandNumber() >= linalgop.getNumDpsInputs() &&
               op_operand.getOperandNumber() - linalgop.getNumDpsInputs() <
                   linalgop.getNumDpsInits()) {
      return 'w';
    } else {
      return 'u';
    }
  }
  // If unknown op
  else
    return 'u';
}

// Convert a vector of SSA returned from arith::ConstantIndexOp into a vector of
// uints
std::vector<unsigned>
air::convertVecOfConstIndexToVecOfUInt(SmallVector<Value> svec) {
  std::vector<unsigned> output;
  for (auto v : svec) {
    auto constOp = getConstantIntValue(v);
    if (!constOp)
      return std::vector<unsigned>();
    output.push_back(*constOp);
  }
  return output;
}

// Get iterator corresponding to a position in a multi-dimensional vector
unsigned air::getIteratorFromMDVector(std::vector<unsigned> dims,
                                      std::vector<unsigned> position) {
  if (dims.size() != position.size())
    return 0;

  std::reverse(position.begin(), position.end());
  unsigned output = 0;
  for (int i = dims.size() - 1; i >= 0; i--) { // In reversed order
    unsigned scale_factor = 1;
    for (int j = 0; j < i; j++) {
      scale_factor *= dims[i];
    }
    output += scale_factor * position[i];
  }
  return output;
}

// Get coordinates corresponding to a position in a multi-dimensional vector
// from an iterator
std::vector<unsigned> air::getMDVectorFromIterator(std::vector<unsigned> dims,
                                                   unsigned iter) {
  std::vector<unsigned> output;
  if (dims.size() > 1) {
    for (int i = dims.size() - 1; i >= 0; i--) { // reversed order
      unsigned denominator = 1;
      for (int j = 0; j < i; j++) {
        denominator *= dims[j];
      }
      output.push_back((iter / (denominator)) % dims[i]);
    }
    // Reverse to original order
    std::reverse(output.begin(), output.end());
  } else if (dims.size() == 1)
    output.push_back(iter);
  return output;
}

// Recursively trace back in defining ops
void air::getDefiningOpsToOperands(Operation *op,
                                   SmallVector<Operation *> &def_ops) {
  if (!op)
    return;
  for (auto oper : op->getOperands()) {
    if (auto def_op = oper.getDefiningOp()) {
      getDefiningOpsToOperands(def_op, def_ops);
      def_ops.push_back(def_op);
    }
  }
}

// Erase dims; recalculate offsets if needed.
LogicalResult eraseWrapNStrideDim(OpBuilder builder,
                                  SmallVector<int> erase_dims,
                                  SmallVector<Value> &offsets,
                                  SmallVector<Value> &sizes,
                                  SmallVector<Value> &strides) {
  auto original_insert_point = builder.saveInsertionPoint();
  bool erased = false;
  // Multiply adjacent wraps.
  auto multiplyAdjWraps = [](OpBuilder builder, int erase_dim,
                             SmallVector<Value> &sizes) {
    auto const_size = getConstantIntValue(sizes[erase_dim]);
    if (!const_size)
      return false; // non-static wrap, NYI.
    if ((int)sizes.size() - 1 == erase_dim)
      return false; // Already the last wrap dimension.
    auto const_size_next = getConstantIntValue(sizes[erase_dim + 1]);
    if (!const_size_next)
      return false; // non-static wrap, NYI.
    sizes[erase_dim + 1] = builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), (*const_size) * (*const_size_next));
    return true;
  };
  // For a given offset[i], find the first offset[j] such that stride[j] is
  // divisible by stride[i], so that offset[i] can be composed onto offset[j].
  auto findFirstComposableOffsetIdx = [](int i, SmallVector<Value> offsets,
                                         SmallVector<Value> strides) {
    auto constStrideI = getConstantIntValue(strides[i]);
    std::optional<int> output = std::nullopt;
    for (int j = i + 1; j < (int)strides.size(); j++) {
      if (!getConstantIntValue(offsets[j]))
        continue; // Currently unable to compose offset[i] expr onto another
                  // offset[j] expr.
      auto constStrideJ = getConstantIntValue(strides[j]);
      if ((*constStrideI) % (*constStrideJ) == 0) {
        output = j;
        return output;
      }
    }
    return output;
  };
  for (auto i : erase_dims) {
    auto const_offset = getConstantIntValue(offsets[i]);
    if (const_offset && *const_offset == 0) {
      erased |= multiplyAdjWraps(builder, i, sizes);
      offsets.erase(offsets.begin() + i);
      sizes.erase(sizes.begin() + i);
      strides.erase(strides.begin() + i);
      erased = true;
      continue;
    }
    // Propagate any non-zero offset and non-unit wrap to the next dimension
    if (offsets.begin() + i + 1 == offsets.end())
      continue;
    auto const_stride = getConstantIntValue(strides[i]);
    if (!const_stride) {
      emitError(builder.getUnknownLoc(), "non-static stride, NYI.");
      return failure();
    }
    auto j = findFirstComposableOffsetIdx(i, offsets, strides);
    if (!j)
      continue;
    auto const_offset_next = getConstantIntValue(offsets[*j]);
    auto const_stride_next = getConstantIntValue(strides[*j]);
    // Attempting to compose i-th offset onto another offset.
    if (const_offset) {
      offsets[*j] = builder.create<arith::ConstantIndexOp>(
          builder.getUnknownLoc(),
          (*const_stride) * (*const_offset) / (*const_stride_next) +
              (*const_offset_next));
    } else {
      // Get affine.apply which produces the offset ssa
      Operation *offset_producer = offsets[i].getDefiningOp();
      if (auto castOp =
              dyn_cast_if_present<arith::IndexCastOp>(offset_producer)) {
        offsets[i] = castOp.getIn();
        offset_producer = castOp.getIn().getDefiningOp();
      } else if (auto addOp =
                     dyn_cast_if_present<arith::AddIOp>(offset_producer)) {
        auto newAffineApply =
            air::consructComposedAffineApplyOpFromArithAddI(builder, addOp);
        if (!newAffineApply) {
          addOp->emitOpError("failed to convert to affine_apply");
          return failure();
        }
        offsets[i] = newAffineApply.getResult();
        offset_producer = newAffineApply;
      } else if (auto mulOp =
                     dyn_cast_if_present<arith::MulIOp>(offset_producer)) {
        auto newAffineApply =
            air::consructComposedAffineApplyOpFromArithMulI(builder, mulOp);
        if (!newAffineApply) {
          mulOp->emitOpError("failed to convert to affine_apply");
          return failure();
        }
        offsets[i] = newAffineApply.getResult();
        offset_producer = newAffineApply;
      }
      if (!offset_producer) {
        if (auto afo = affine::getForInductionVarOwner(offsets[i])) {
          builder.setInsertionPointToStart(afo.getBody());
        } else if (auto sfo = scf::getForInductionVarOwner(offsets[i])) {
          builder.setInsertionPointToStart(sfo.getBody());
        } else if (auto aho = air::getHierarchyArgOwner(offsets[i])) {

        } else
          continue;
        // Create a new affine.apply on affine.for ind. vars, as handle for
        // subsequent offset composition.
        auto dim0_expr = getAffineDimExpr(0, builder.getContext());
        auto iv_map = AffineMap::get(1, 0, dim0_expr);
        offset_producer = builder.create<affine::AffineApplyOp>(
            builder.getUnknownLoc(), iv_map, offsets[i]);
      }
      if (auto exec = dyn_cast<air::ExecuteOp>(offset_producer))
        offset_producer = &exec.getChildOps().front();
      auto affine_apply = dyn_cast<affine::AffineApplyOp>(offset_producer);
      if (!affine_apply) {
        offset_producer->emitOpError("unknown ssa offset producer, NYI.");
        return failure();
      }
      if (affine_apply->getNumOperands() > 1)
        continue;
      // Compose affine map
      SmallVector<AffineExpr, 8> exprReplacements(
          affine_apply.getAffineMap().getResults().begin(),
          affine_apply.getAffineMap().getResults().end());
      if (exprReplacements.size() > 1)
        continue;
      bool affineApplyOnDim = exprReplacements.front().isFunctionOfDim(0);
      bool affineApplyOnSymbol = exprReplacements.front().isFunctionOfSymbol(0);
      AffineExpr offset_expr = AffineExpr();
      if (affineApplyOnDim)
        offset_expr = builder.getAffineDimExpr(0);
      else if (affineApplyOnSymbol)
        offset_expr = builder.getAffineSymbolExpr(0);
      else
        continue;
      auto stride_expr = builder.getAffineConstantExpr(*const_stride);
      auto next_stride_expr = builder.getAffineConstantExpr(*const_stride_next);
      offset_expr = offset_expr * stride_expr;
      offset_expr = offset_expr.ceilDiv(next_stride_expr);
      offset_expr =
          offset_expr + builder.getAffineConstantExpr(*const_offset_next);
      AffineMap next_offset_map = AffineMap();
      if (affineApplyOnDim) {
        offset_expr = offset_expr.replaceDims(exprReplacements);
        next_offset_map = AffineMap::get(1, 0, offset_expr);
      } else if (affineApplyOnSymbol) {
        offset_expr = offset_expr.replaceSymbols(exprReplacements);
        next_offset_map = AffineMap::get(0, 1, offset_expr);
      }
      // Apply affine map
      builder.setInsertionPoint(affine_apply);
      if (auto exec =
              dyn_cast_if_present<air::ExecuteOp>(affine_apply->getParentOp()))
        builder.setInsertionPoint(exec);
      auto newAffineApply =
          dyn_cast<affine::AffineApplyOp>(builder.clone(*affine_apply));
      newAffineApply.setMap(next_offset_map);
      offsets[i] = newAffineApply->getResult(0);
      offsets[*j] = offsets[i];
    }
    erased |= multiplyAdjWraps(builder, i, sizes);
    offsets.erase(offsets.begin() + i);
    sizes.erase(sizes.begin() + i);
    strides.erase(strides.begin() + i);
    erased = true;
  }
  builder.restoreInsertionPoint(original_insert_point);
  if (erased)
    return success();
  return failure();
};

// Find the largest factor of 'num' which is not larger than 'max'. Ref:
// https://github.com/nod-ai/iree-amd-aie/blob/main/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/AMDAIEUtils.cpp#L334
int air::findLargestFactor(int num, int max) {
  assert(max > 0 && "No factors less than or equal to 0 exist");

  // Do O(1) instead of O(sqrt(num)) computation for this common case.
  if (num <= max) {
    return num;
  }

  int largestLowFactor = 1;
  for (int lowFactor = 2; lowFactor <= max; ++lowFactor) {
    const int highFactor = num / lowFactor;

    // This early exit is what makes this O(sqrt(num)) instead of O(num).
    if (highFactor < lowFactor)
      return largestLowFactor;

    const bool areActuallyFactors = num % lowFactor == 0;
    if (areActuallyFactors) {
      // We're certain that here lowFactor <= highFactor, and highFactor is
      // descending in this loop. So we can return immediately if highFactor is
      // good.
      if (highFactor <= max)
        return highFactor;
      largestLowFactor = lowFactor;
    }
  }
  return largestLowFactor;
}

// Canonicalize wrap and stride lists by removing redundant dimensions.
LogicalResult air::canonicalizeWrapAndStrideList(
    OpBuilder &builder, SmallVector<Value> &offsets, SmallVector<Value> &sizes,
    SmallVector<Value> &strides, int memref_volume, int maxSize) {
  // AIE2 hardware constraints. TODO: import these info from target model.
  const int AIE2_STRIDE_UPPER_BOUND = 1048576;
  bool listsHaveChanged = false;
  OpBuilder::InsertionGuard guard(builder);
  // Match offsets size with sizes and strides
  auto max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  if (max_dim_size && offsets.size() < max_dim_size) {
    for (auto i = offsets.size(); i < max_dim_size; i++) {
      offsets.insert(offsets.begin(), builder.create<arith::ConstantIndexOp>(
                                          builder.getUnknownLoc(), 0));
    }
    listsHaveChanged = true;
  }

  // If maxSize is given, check whether any size value goes beyond maxSize.
  for (int i = sizes.size() - 1; i >= 0; i--) {
    if (isDefaultDataAccessPattern(sizes, strides))
      // Default access pattern, despite being multi-dimensional, can get
      // collapsed into a one-dimensional data stream and not subject to maxSize
      // limitation.
      break;
    if (maxSize <= 0)
      break;
    auto const_wrap = *getConstantIntValue(sizes[i]);
    auto const_stride = *getConstantIntValue(strides[i]);
    if (const_wrap <= maxSize)
      continue;
    // Found dimension with illegal wrap. Tiling. (Prefers smaller outer wrap
    // values, as long as stride fits)
    int a_wrap = findLargestFactor(const_wrap, maxSize);
    int b_wrap = llvm::divideCeilSigned(const_wrap, a_wrap);
    int new_a_stride = const_stride * a_wrap;
    if (memref_volume != 1)
      new_a_stride %= memref_volume; // Avoids striding out of memory size, if
                                     // memref is ranked
    int inner_wrap =
        (new_a_stride > AIE2_STRIDE_UPPER_BOUND) ? (b_wrap) : (a_wrap);
    int outer_wrap =
        (new_a_stride > AIE2_STRIDE_UPPER_BOUND) ? (a_wrap) : (b_wrap);
    sizes[i] = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),
                                                      inner_wrap);
    sizes.insert(sizes.begin() + i, builder.create<arith::ConstantIndexOp>(
                                        builder.getUnknownLoc(), outer_wrap));
    auto new_const_stride = const_stride * inner_wrap;
    if (memref_volume != 1)
      new_const_stride %= memref_volume; // Avoids striding out of memory size,
                                         // if memref is ranked
    strides.insert(strides.begin() + i,
                   builder.create<arith::ConstantIndexOp>(
                       builder.getUnknownLoc(), new_const_stride));
    offsets.insert(offsets.begin() + i, builder.create<arith::ConstantIndexOp>(
                                            builder.getUnknownLoc(), 0));
    i++;
    listsHaveChanged = true;
  }

  // Canonicalize dimensions with size = 1
  SmallVector<int> erase_dims;
  for (int i = sizes.size() - 1; i >= 0; i--) {
    auto const_size = getConstantIntValue(sizes[i]);
    if (!const_size)
      continue;
    if (*const_size != 1)
      continue;
    erase_dims.push_back(i);
  }

  listsHaveChanged |=
      eraseWrapNStrideDim(builder, erase_dims, offsets, sizes, strides)
          .succeeded();
  erase_dims.clear();

  // Canonicalize adjacent dimensions
  if (!sizes.empty()) {
    for (int i = sizes.size() - 1; i >= 1; i--) {
      auto const_size = getConstantIntValue(sizes[i]);
      auto const_stride = getConstantIntValue(strides[i]);
      auto const_size_prev = getConstantIntValue(sizes[i - 1]);
      // Check for max size constraint, and skip if the final size exceeds it
      // after composition. The exception is when the access pattern is in
      // default access pattern, meaning that the data access pattern is a
      // single stream without multi-dimensional wrapping and striding.
      if (maxSize > 0 && *const_size_prev * (*const_size) > maxSize)
        if (!isDefaultDataAccessPattern(sizes, strides))
          continue;
      auto const_stride_prev = getConstantIntValue(strides[i - 1]);
      if (*const_stride_prev == *const_size * *const_stride)
        listsHaveChanged |=
            eraseWrapNStrideDim(builder, SmallVector<int>{i - 1}, offsets,
                                sizes, strides)
                .succeeded();
    }
  }

  // If default data access pattern, then clear the offsets, sizes and strides.
  if (offsets.size() == 1 && sizes.size() == 1 && strides.size() == 1) {
    if (getConstantIntValue(offsets[0]) && getConstantIntValue(sizes[0]) &&
        getConstantIntValue(strides[0])) {
      if (*getConstantIntValue(strides[0]) == 1 &&
          *getConstantIntValue(sizes[0]) == memref_volume) {
        offsets.erase(offsets.begin());
        sizes.erase(sizes.begin());
        strides.erase(strides.begin());
        listsHaveChanged = true;
      }
    }
  }

  if (listsHaveChanged)
    return success();
  else
    return failure();
}

// Fold perfectly nested for loops as extra entries in wraps and strides. This
// method does not directly mutate the for op nor the data movement operation.
// It only generates the offsets, wraps and strides list after loop folding.
LogicalResult air::foldForLoopNestAsExtendedSizesAndStrides(
    OpBuilder builder, Operation *for_op, Operation *channel_op,
    SmallVector<Value> &offsets, SmallVector<Value> &wraps,
    SmallVector<Value> &strides, Value memref) {
  auto loc = for_op->getLoc();

  // Fold for loops into channel op's wrap and stride fields
  SmallVector<Operation *> for_loops;
  Operation *parent = channel_op;
  while (parent != for_op) {
    parent = parent->getParentOp();
    if (auto sfo = dyn_cast<scf::ForOp>(parent)) {
      for_loops.push_back(parent);
    } else if (auto afo = dyn_cast<affine::AffineForOp>(parent)) {
      for_loops.push_back(parent);
    }
  }

  // Evaluate offset from affine map.
  auto evalOffsetFromAffineMap = [&](MLIRContext *ctx, AffineMap map) {
    SmallVector<std::optional<int64_t>> zeroSyms(map.getNumSymbols(),
                                                 std::optional<int64_t>{0});
    SmallVector<std::optional<int64_t>> zeroDims(map.getNumDims(),
                                                 std::optional<int64_t>{0});
    return air::evaluateConstantsInMap(map, zeroSyms, zeroDims, ctx);
  };
  for (auto o : for_loops) {
    int64_t stepSize = -1;
    int loop_lower_bound = 0;
    Value iv = nullptr;
    if (auto afo = dyn_cast<affine::AffineForOp>(o)) {
      iv = afo.getInductionVar();
      loop_lower_bound = afo.getConstantLowerBound();
      stepSize = afo.getStepAsInt();
    } else if (auto sfo = dyn_cast<scf::ForOp>(o)) {
      iv = sfo.getInductionVar();
      if (auto cst_lower_bound = mlir::getConstantIntValue(sfo.getLowerBound()))
        loop_lower_bound = *cst_lower_bound;
      stepSize = *mlir::getConstantIntValue(sfo.getStep());
    }
    int64_t ind_var_factor = 0;
    for (int i = offsets.size() - 1; i >= 0; i--) {
      Value offsetVal = offsets[i];
      // Propagate through cast op
      if (auto cast = offsetVal.getDefiningOp<CastOpInterface>()) {
        if (cast->getNumOperands() == 1)
          offsetVal = cast->getOperand(0);
      }
      if (iv && offsetVal == iv) {
        ind_var_factor = *getConstantIntValue(strides[i]) * stepSize;
        offsets[i] = builder.template create<arith::ConstantIndexOp>(
            loc, loop_lower_bound);
        break;
      } else if (iv && offsetVal.getDefiningOp()) {
        Operation *iv_consumer = offsetVal.getDefiningOp();
        if (auto exec = dyn_cast<air::ExecuteOp>(iv_consumer))
          iv_consumer = &exec.getChildOps().front();
        if (auto affop = dyn_cast<affine::AffineApplyOp>(iv_consumer)) {
          auto idx = llvm::find_if(affop.getOperands(),
                                   [iv](Value oper) { return oper == iv; });
          if (idx == affop.getOperands().end())
            continue;
          auto map = affop.getAffineMap();
          int64_t map_offset =
              evalOffsetFromAffineMap(for_op->getContext(), map).value();
          ind_var_factor = *getConstantIntValue(strides[i]);
          SmallVector<std::optional<int64_t>> stepSizeAsSymVec(
              affop.getMap().getNumSymbols(), std::optional<int64_t>{0});
          SmallVector<std::optional<int64_t>> stepSizeAsDimVec(
              affop.getMap().getNumDims(), std::optional<int64_t>{0});
          if (idx - affop.getOperands().begin() < affop.getMap().getNumDims())
            stepSizeAsDimVec[idx - affop.getOperands().begin()] = stepSize;
          else
            stepSizeAsSymVec[idx - affop.getOperands().begin() -
                             affop.getMap().getNumDims()] = stepSize;
          int64_t map_gradient =
              air::evaluateConstantsInMap(
                  map, stepSizeAsSymVec, stepSizeAsDimVec, for_op->getContext())
                  .value() -
              map_offset;
          ind_var_factor *= map_gradient;
        } else if (auto arithop = dyn_cast<arith::AddIOp>(iv_consumer)) {
          ind_var_factor = stepSize;
        }
      }
    }
    int trip_count = -1;
    if (auto afo = dyn_cast<affine::AffineForOp>(o))
      trip_count = *getStaticAffineForTripCountAsInt(afo);
    else if (auto sfo = dyn_cast<scf::ForOp>(o))
      trip_count = *getStaticScfForTripCountAsInt(sfo);
    Value new_wrap =
        builder.template create<arith::ConstantIndexOp>(loc, trip_count);
    // Skip stride modulo if memref is unranked.
    int64_t new_stride_value =
        getTensorVolume(memref.getType()) == 1
            ? ind_var_factor
            : ind_var_factor % getTensorVolume(memref.getType());
    Value new_stride =
        builder.template create<arith::ConstantIndexOp>(loc, new_stride_value);

    // Check for compliance with DMA BD hardware limitation (<= 1M). Note that
    // the exception is when stepSize = previous highest wrap, because in that
    // case the large stride shall simply get canonicalized away.
    if (llvm::divideCeilSigned(
            new_stride_value * getElementSizeInBytes(memref.getType()), 4) >
        0x100000) {
      if (stepSize != *getConstantIntValue(wraps[0])) {
        return failure();
      }
    }

    // Insert new dimension into the wraps and strides list.
    offsets.insert(offsets.begin(),
                   builder.template create<arith::ConstantIndexOp>(loc, 0));
    wraps.insert(wraps.begin(), new_wrap);
    strides.insert(strides.begin(), new_stride);
  }
  return success();
}

// If wrap-and-stride lists are empty, populate them with default data access
// layout (contiguous, row-major).
void air::populateDefaultWrapsAndStrides(OpBuilder builder, Value memref,
                                         SmallVector<Value> &offsets,
                                         SmallVector<Value> &wraps,
                                         SmallVector<Value> &strides) {
  auto loc = builder.getUnknownLoc();
  if (offsets.empty() && wraps.empty() && strides.empty()) {
    auto memref_shape = getTensorShape(memref.getType());
    int current_stride = getTensorVolume(memref.getType());
    for (unsigned i = 0; i < memref_shape.size(); i++) {
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
      wraps.push_back(
          builder.create<arith::ConstantIndexOp>(loc, memref_shape[i]));
      current_stride /= memref_shape[i];
      strides.push_back(
          builder.create<arith::ConstantIndexOp>(loc, current_stride));
    }
  }
}

// Check if the wraps and strides imply the default (contiguous, row-major) data
// access pattern.
bool air::isDefaultDataAccessPattern(SmallVector<Value> memcpy_sizes,
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
    auto wrap = mlir::getConstantIntValue(memcpy_sizes[i]);
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
bool air::isVolumeEqualToMemrefVolume(SmallVector<Value> memcpy_sizes,
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

// Get the memref size along a given dimension, that the access pattern actually
// covers.
SmallVector<int64_t>
air::getEffectiveMemrefSizeFromAccessPattern(SmallVector<int> memref_shape,
                                             SmallVector<Value> sizes,
                                             SmallVector<Value> strides) {
  SmallVector<int64_t> access_bounds(memref_shape.size(), 1);
  for (int i = sizes.size() - 1; i >= 0; i--) {
    int current_memref_volume = 1;
    for (int j = memref_shape.size() - 1; j >= 0; j--) {
      current_memref_volume *= memref_shape[j];
      if (llvm::divideFloorSigned(*getConstantIntValue(strides[i]),
                                  current_memref_volume))
        continue;
      int64_t bound =
          llvm::divideFloorSigned(*getConstantIntValue(strides[i]),
                                  current_memref_volume / memref_shape[j]) *
          *getConstantIntValue(sizes[i]);
      access_bounds[j] = std::max(access_bounds[j], bound);
    }
  }
  return access_bounds;
}

// Get the overall data access pattern from air.channel ops which access the
// memref.
SmallVector<int64_t> air::getDataAccessShapeFromMemcpyOp(
    Value memref,
    SmallVector<
        std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>>
        patterns) {
  auto memref_shape = getTensorShape(memref.getType());
  SmallVector<int64_t> overall_access_bounds(memref_shape.size(), 1);
  for (auto pattern : patterns) {
    SmallVector<int64_t> access_bounds(memref_shape.size(), 1);
    // Empty offsets list means default data access pattern spaning the entire
    // memref
    if (std::get<0>(pattern).empty())
      for (unsigned i = 0; i < memref_shape.size(); i++)
        access_bounds[i] = memref_shape[i];
    else
      access_bounds = getEffectiveMemrefSizeFromAccessPattern(
          memref_shape, std::get<1>(pattern), std::get<2>(pattern));
    // Update overall access bounds.
    for (unsigned i = 0; i < memref_shape.size(); i++)
      overall_access_bounds[i] =
          std::max(overall_access_bounds[i], access_bounds[i]);
  }
  return overall_access_bounds;
}

static void updateAccessPatternByScfForNest(
    std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
        &pattern,
    SmallVector<Value> indices, OpBuilder builder, Region *untilReg = nullptr) {
  auto loc = builder.getUnknownLoc();
  auto updateWrapAndStride = [&](int stepSize, int tripCount, int i) {
    std::get<1>(pattern)[i] =
        builder.create<arith::ConstantIndexOp>(loc, tripCount);
    std::get<2>(pattern)[i] = builder.create<arith::ConstantIndexOp>(
        loc, stepSize * (*getConstantIntValue(std::get<2>(pattern)[i])));
  };
  // Infer data access pattern's sizes from parent scf.for loop and any affine
  // op applied on the induction variable
  auto inferDataAccessSizes = [](scf::ForOp scfForOp, air::ExecuteOp execOp,
                                 Value index) {
    int scfForTripCount = *air::getStaticScfForTripCountAsInt(scfForOp);
    // If scf.for's iv applies affine::DelinerizeIndexOp
    if (auto delinearizeOp = dyn_cast<affine::AffineDelinearizeIndexOp>(
            &execOp.getChildOps().front())) {
      int resIdx =
          llvm::find(execOp.getResults(), index) - execOp.getResults().begin();
      auto constBasis =
          getConstantIntValue(delinearizeOp.getMixedBasis()[resIdx]);
      if (!constBasis)
        delinearizeOp->emitOpError("basis ") << resIdx << " is non-static.";
      else
        scfForTripCount = *constBasis;
    }
    return scfForTripCount;
  };
  int dim = -1;
  for (auto index : indices) {
    dim++;
    if (getConstantIntValue(index))
      continue;
    if (auto scfForOp = scf::getForInductionVarOwner(index)) {
      if (untilReg && !untilReg->isAncestor(scfForOp->getParentRegion()))
        continue; // Out of scope
      updateWrapAndStride(*getConstantIntValue(scfForOp.getStep()),
                          *air::getStaticScfForTripCountAsInt(scfForOp), dim);
    }
    if (!index.getDefiningOp())
      continue;
    if (auto execOp = dyn_cast<air::ExecuteOp>(index.getDefiningOp())) {
      for (auto &childOp : execOp.getChildOps())
        for (auto oper : childOp.getOperands())
          if (auto scfForOp = scf::getForInductionVarOwner(oper)) {
            if (untilReg && !untilReg->isAncestor(scfForOp->getParentRegion()))
              continue; // Out of scope
            int scfForTripCount = inferDataAccessSizes(scfForOp, execOp, index);
            updateWrapAndStride(*getConstantIntValue(scfForOp.getStep()),
                                scfForTripCount, dim);
          }
    }
  }
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
air::writeAccessPattern(air::ChannelInterface chanOp) {
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
      pattern;
  for (auto offset : chanOp.getOffsets())
    std::get<0>(pattern).push_back(offset);
  for (auto size : chanOp.getSizes())
    std::get<1>(pattern).push_back(size);
  for (auto stride : chanOp.getStrides())
    std::get<2>(pattern).push_back(stride);
  return pattern;
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
air::writeAccessPattern(memref::SubViewOp subview, Region *commonReg) {
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
      pattern;
  auto subview_offsets = subview.getOffsets().begin();
  auto subview_sizes = subview.getSizes().begin();
  auto subview_strides = subview.getStrides().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto static_sizes = subview.getStaticSizes();
  auto static_strides = subview.getStaticStrides();
  // Get strided layout from subview op's output MemRefType
  if (auto strided = llvm::dyn_cast<StridedLayoutAttr>(
          llvm::cast<MemRefType>(subview.getResult().getType()).getLayout()))
    static_strides = strided.getStrides();

  auto loc = subview.getLoc();
  OpBuilder builder(subview);
  for (auto o : static_offsets) {
    if (o >= 0)
      std::get<0>(pattern).push_back(
          builder.create<arith::ConstantIndexOp>(loc, o));
    else
      std::get<0>(pattern).push_back(*subview_offsets++);
  }
  for (auto o : static_sizes) {
    if (o >= 0)
      std::get<1>(pattern).push_back(
          builder.create<arith::ConstantIndexOp>(loc, o));
    else
      std::get<1>(pattern).push_back(*subview_sizes++);
  }
  if (static_sizes.size() < static_strides.size()) {
    subview->emitOpError(
        "isn't standard or rank-reduced memref.subview. Not supported.");
    return pattern;
  }
  // If rank-reduced memref.subview, then pad strides with 1s.
  for (unsigned i = 0; i < static_sizes.size() - static_strides.size(); i++)
    std::get<2>(pattern).push_back(
        builder.create<arith::ConstantIndexOp>(loc, 1));
  for (auto o : static_strides) {
    if (o >= 0)
      std::get<2>(pattern).push_back(
          builder.create<arith::ConstantIndexOp>(loc, o));
    else
      std::get<2>(pattern).push_back(*subview_strides++);
  }
  updateAccessPatternByScfForNest(pattern, std::get<0>(pattern), builder,
                                  commonReg);
  return pattern;
}

// Helper function to check if a value ultimately depends on herd tile indices,
// either directly or through intermediate operations.
static bool dependsOnHerdTileIndex(Value index) {
  // Direct check - is this value a herd argument?
  if (air::getHerdArgOwner(index))
    return true;

  // Check if defined by an operation that uses herd indices
  auto defOp = index.getDefiningOp();
  if (!defOp)
    return false;

  // Check air.execute wrapping
  if (auto exec = dyn_cast<air::ExecuteOp>(defOp)) {
    for (auto &childOp : exec.getChildOps()) {
      for (auto oper : childOp.getOperands()) {
        if (air::getHerdArgOwner(oper))
          return true;
      }
    }
  }

  // Check operands of the defining operation
  for (auto oper : defOp->getOperands()) {
    if (air::getHerdArgOwner(oper))
      return true;
  }

  return false;
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
air::writeAccessPattern(mlir::vector::TransferReadOp readOp) {
  OpBuilder builder(readOp);
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
      pattern;
  auto vectorTy = llvm::cast<VectorType>(readOp.getVector().getType());
  auto memrefTy = llvm::cast<BaseMemRefType>(readOp.getBase().getType());
  if (!vectorTy) {
    readOp->emitOpError("Not a vector");
    return pattern;
  }
  if (!memrefTy) {
    readOp->emitOpError("Not a memref");
    return pattern;
  }
  // Initialize wraps and strides based on the unshrunk memref shape.
  populateDefaultWrapsAndStrides(builder, readOp.getBase(),
                                 std::get<0>(pattern), std::get<1>(pattern),
                                 std::get<2>(pattern));

  // Detect herd tile indices in the indices and adjust sizes accordingly.
  // If an index depends on a herd tile index, that dimension only accesses
  // a size of 1 per tile.
  auto indices = readOp.getIndices();
  for (unsigned i = 0; i < indices.size() && i < std::get<1>(pattern).size();
       i++) {
    if (dependsOnHerdTileIndex(indices[i])) {
      // This index is a herd tile index - set size to 1 for this dimension
      std::get<1>(pattern)[i] =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    }
  }

  // Update wraps based on vector shape and vector access patterns.
  unsigned rankOffset =
      vectorTy.getShape().size() >= std::get<1>(pattern).size()
          ? 0
          : std::get<1>(pattern).size() - vectorTy.getShape().size();
  for (unsigned i = rankOffset; i < std::get<1>(pattern).size(); i++)
    std::get<1>(pattern)[i] = builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), vectorTy.getShape()[i - rankOffset]);
  updateAccessPatternByScfForNest(pattern, readOp.getIndices(), builder);
  return pattern;
}

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
air::writeAccessPattern(mlir::vector::TransferWriteOp writeOp) {
  OpBuilder builder(writeOp);
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
      pattern;
  auto memrefTy = llvm::cast<BaseMemRefType>(writeOp.getBase().getType());
  auto vectorTy = llvm::cast<VectorType>(writeOp.getVector().getType());
  if (!vectorTy) {
    writeOp->emitOpError("Not a vector");
    return pattern;
  }
  if (!memrefTy) {
    writeOp->emitOpError("Not a memref");
    return pattern;
  }
  // Initialize wraps and strides based on the unshrunk memref shape.
  populateDefaultWrapsAndStrides(builder, writeOp.getBase(),
                                 std::get<0>(pattern), std::get<1>(pattern),
                                 std::get<2>(pattern));

  // Detect herd tile indices in the indices and adjust sizes accordingly.
  // If an index depends on a herd tile index, that dimension only accesses
  // a size of 1 per tile.
  auto indices = writeOp.getIndices();
  for (unsigned i = 0; i < indices.size() && i < std::get<1>(pattern).size();
       i++) {
    if (dependsOnHerdTileIndex(indices[i])) {
      // This index is a herd tile index - set size to 1 for this dimension
      std::get<1>(pattern)[i] =
          builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    }
  }

  // Update wraps based on vector shape and vector access patterns.
  unsigned rankOffset =
      vectorTy.getShape().size() >= std::get<1>(pattern).size()
          ? 0
          : std::get<1>(pattern).size() - vectorTy.getShape().size();
  for (unsigned i = rankOffset; i < std::get<1>(pattern).size(); i++)
    std::get<1>(pattern)[i] = builder.create<arith::ConstantIndexOp>(
        builder.getUnknownLoc(), vectorTy.getShape()[i - rankOffset]);
  updateAccessPatternByScfForNest(pattern, writeOp.getIndices(), builder);
  return pattern;
}

SmallVector<int64_t> air::getDataAccessShapeFromMemcpyOp(
    Value memref, SmallVector<air::ChannelInterface> chanUsers) {
  SmallVector<
      std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>>
      accessPatterns;
  for (auto chanUser : chanUsers) {
    accessPatterns.push_back(writeAccessPattern(chanUser));
  }
  return getDataAccessShapeFromMemcpyOp(memref, accessPatterns);
}
SmallVector<int64_t>
air::getDataAccessShapeFromMemcpyOp(Value memref,
                                    SmallVector<Operation *> users) {
  if (users.empty())
    return SmallVector<int64_t>();
  SmallVector<
      std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>>
      accessPatterns;
  // Get a common ancestor region for all users
  Region *commonAncestorReg = findCommonRegionContainingAllAncestors(
      users, users.front()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());

  for (auto user : users) {
    if (auto chanUser = dyn_cast<air::ChannelInterface>(user))
      accessPatterns.push_back(writeAccessPattern(chanUser));
    else if (auto svUser = dyn_cast<memref::SubViewOp>(user))
      accessPatterns.push_back(writeAccessPattern(svUser, commonAncestorReg));
    else if (auto vecReadUser = dyn_cast<mlir::vector::TransferReadOp>(user))
      accessPatterns.push_back(writeAccessPattern(vecReadUser));
    else if (auto vecWriteUser = dyn_cast<mlir::vector::TransferWriteOp>(user))
      accessPatterns.push_back(writeAccessPattern(vecWriteUser));
  }
  return getDataAccessShapeFromMemcpyOp(memref, accessPatterns);
}

// Update strides after memref shrinkage. Assuming there is only one dimension
// being shrunk.
SmallVector<int>
air::getUpdatedStridesAfterShrinkage(SmallVector<int> old_memref_shape,
                                     SmallVector<int64_t> new_memref_shape,
                                     SmallVector<Value> strides) {
  SmallVector<int> new_strides(strides.size(), -1);
  int shrinkage_volume = 1;
  int shrinkage_factor = 1;
  for (int j = old_memref_shape.size() - 1; j >= 0; j--) {
    shrinkage_volume *= old_memref_shape[j];
    if (old_memref_shape[j] != new_memref_shape[j]) {
      shrinkage_factor =
          llvm::divideCeilSigned(old_memref_shape[j], new_memref_shape[j]);
      break;
    }
  }
  for (int i = strides.size() - 1; i >= 0; i--) {
    if (llvm::divideFloorSigned(*getConstantIntValue(strides[i]),
                                shrinkage_volume))
      new_strides[i] = llvm::divideCeilSigned(*getConstantIntValue(strides[i]),
                                              shrinkage_factor);
    else
      new_strides[i] = *getConstantIntValue(strides[i]);
  }
  return new_strides;
}

// Update offsets after memref shrinkage.
SmallVector<int>
air::getUpdatedOffsetsAfterShrinkage(SmallVector<int> old_memref_shape,
                                     SmallVector<int64_t> new_memref_shape,
                                     SmallVector<Value> offsets) {
  SmallVector<int> new_offsets(offsets.size(), -1);
  for (int i = 0; i < (int)offsets.size(); i++) {
    int memref_idx = i + old_memref_shape.size() - offsets.size();
    if (memref_idx >= 0) {
      // Reset offset to zero, if the user air.channel put/get has offset being
      // variant wrt a parent spatial iteration space (e.g. air.herd,
      // scf.parallel).
      if (offsets[i].getDefiningOp()) {
        if (auto exec = dyn_cast<air::ExecuteOp>(offsets[i].getDefiningOp())) {
          for (auto &childOp : exec.getChildOps())
            for (auto oper : childOp.getOperands())
              if (getHerdArgOwner(oper))
                new_offsets[i] = 0;
        }
      } else {
        // If offset is some block argument
        if (getHerdArgOwner(offsets[i]))
          new_offsets[i] = 0;
        else if (scf::getParallelForInductionVarOwner(offsets[i]))
          new_offsets[i] = 0;
        else if (scf::getForInductionVarOwner(offsets[i]))
          continue;
        else
          return SmallVector<int>(); // Offset is block argument to an unknown
                                     // iteration space.
      }
    }
  }
  return new_offsets;
}

// Given a dimension on wrap-and-stride list, infer the dimension on memref that
// this pattern spans completely on.
std::optional<int>
air::getMemrefDimFromOffsetDim(int dimOnOffset, SmallVector<Value> offsets,
                               SmallVector<Value> strides,
                               SmallVector<int> memrefShape) {
  std::optional<int> output = std::nullopt;
  if (memrefShape.empty())
    return std::nullopt;
  if (offsets.empty())
    return std::nullopt;
  if (dimOnOffset >= (int)offsets.size()) {
    return std::nullopt; // Dimension exceeds offsets rank.
  }

  // Get stride value which corresponds to accessing each memref dimension,
  // highest dimension first.
  int memrefRank = memrefShape.size();
  SmallVector<int> memrefDimStrides(memrefRank, 1);
  int currentStride = 1;
  for (int i = memrefRank - 1; i >= 0; i--) {
    memrefDimStrides[i] = currentStride;
    currentStride *= memrefShape[i];
  }

  // Find the dimension on memref shape with given stride value.
  auto strideVal = getConstantIntValue(strides[dimOnOffset]);
  if (!strideVal) {
    return std::nullopt; // Non-static stride value in data access pattern, NYI.
  }
  for (unsigned i = 0; i < memrefDimStrides.size(); i++)
    if (*strideVal == memrefDimStrides[i]) {
      output = i;
      return output;
    }
  return std::nullopt;
}

// Given a dimension on memref shape, infer the dimension on wrap-and-stride
// list that spans on this memref dimension.
std::optional<int>
air::getOffsetDimFromMemrefDim(int dimOnMemref, SmallVector<Value> strides,
                               SmallVector<int> memrefShape) {
  std::optional<int> output = std::nullopt;
  if (memrefShape.empty())
    return std::nullopt;
  if (strides.empty())
    return std::nullopt;
  if (dimOnMemref >= (int)memrefShape.size()) {
    return std::nullopt; // Dimension exceeds memref rank.
  }

  // Get stride value which corresponds to accessing the current memref
  // dimension.
  int memrefRank = memrefShape.size();
  int memrefStride = 1;
  for (int i = memrefRank - 1; i > dimOnMemref; i--)
    memrefStride *= memrefShape[i];

  // Find the dimension on wrap-and-stride list with given stride value.
  for (unsigned i = 0; i < strides.size(); i++) {
    auto strideVal = getConstantIntValue(strides[i]);
    if (!strideVal) {
      return std::nullopt; // Non-static stride value in data access pattern,
                           // NYI.
    }
    if (*strideVal == memrefStride) {
      output = i;
      return output;
    }
  }
  return std::nullopt;
}

// Evaluate the affine expression of affine map on a sparse vector of constant
// ints.
std::optional<int64_t>
air::evaluateConstantsInMap(AffineMap map,
                            SmallVector<std::optional<int64_t>> symAndDimInputs,
                            MLIRContext *ctx) {
  std::optional<int64_t> output = std::nullopt;
  if (map.getNumSymbols() == symAndDimInputs.size()) {
    return evaluateConstantsInMap(map, symAndDimInputs,
                                  SmallVector<std::optional<int64_t>>{}, ctx);
  } else if (map.getNumDims() == symAndDimInputs.size()) {
    return evaluateConstantsInMap(map, SmallVector<std::optional<int64_t>>{},
                                  symAndDimInputs, ctx);
  } else
    return output;
}
std::optional<int64_t> air::evaluateConstantsInMap(
    AffineMap map, SmallVector<std::optional<int64_t>> symbolInputs,
    SmallVector<std::optional<int64_t>> dimInputs, MLIRContext *ctx) {
  std::optional<int64_t> output = std::nullopt;
  if (map.getNumSymbols() != symbolInputs.size())
    return output;
  if (map.getNumDims() != dimInputs.size())
    return output;
  auto newmap = map;
  for (unsigned i = 0; i < map.getNumSymbols(); i++) {
    if (!symbolInputs[i])
      continue;
    auto c = getAffineConstantExpr(*symbolInputs[i], ctx);
    newmap =
        newmap.replace(getAffineSymbolExpr(i, ctx), c, 0, map.getNumSymbols());
  }
  for (unsigned i = 0; i < map.getNumDims(); i++) {
    if (!dimInputs[i])
      continue;
    auto c = getAffineConstantExpr(*dimInputs[i], ctx);
    newmap = newmap.replace(getAffineDimExpr(i, ctx), c, map.getNumDims(), 0);
  }
  output = simplifyAffineMap(newmap).getSingleConstantResult();
  return output;
}

// Extend the lookupOrDefault method to operate on a vector of values.
Value air::lookupOrDefaultRange(Value v, IRMapping &remap) {
  return remap.lookupOrDefault(v);
}
SmallVector<Value> air::lookupOrDefaultRange(SmallVectorImpl<Value> &vec,
                                             IRMapping &remap) {
  SmallVector<Value> output;
  for (auto v : vec) {
    output.push_back(remap.lookupOrDefault(v));
  }
  return output;
}
SmallVector<Value> air::lookupOrDefaultRange(OperandRange vec,
                                             IRMapping &remap) {
  SmallVector<Value> output;
  for (auto v : vec) {
    output.push_back(remap.lookupOrDefault(v));
  }
  return output;
}

// Extend isPure method to operate on air.execute.
bool air::isPure(Operation *op) {
  bool result = mlir::isPure(op);
  if (auto execOp = dyn_cast<air::ExecuteOp>(op))
    result = llvm::all_of(execOp.getChildOps(), [](Operation &childOp) {
      return mlir::isPure(&childOp);
    });
  return result;
}

// Return if the given block contains N ops which are impure and aren't async
// wait ops (such as air.wait_all).
bool air::hasNImpureOps(Block *block, unsigned N) {
  unsigned counter = 0;
  for (auto &o : block->without_terminator()) {
    if (air::isPure(&o))
      continue;
    if (isa<air::WaitAllOp>(o))
      continue;
    counter++;
  }
  return counter == N;
}

// Return if the given block contains N ops or not, not counting the block's
// terminator.
bool air::hasNElements(Block *block, unsigned N) {
  return llvm::range_size(block->without_terminator()) == N;
}

// Get backward slice to a vector of values, within a specified region.
void air::getBackwardSliceInRegion(OpBuilder builder, Region *region,
                                   SmallVectorImpl<Value> &vals,
                                   SetVector<Operation *> &backwardSlices) {
  BackwardSliceOptions bsOptions{
      [&](Operation *o) { return region->isAncestor(o->getParentRegion()); }};
  if (!region)
    return;
  for (auto val : vals) {
    auto valDefOp = val.getDefiningOp();
    if (!valDefOp)
      continue;
    if (!region->isAncestor(valDefOp->getParentRegion()))
      continue;
    // Get backward slices
    SetVector<Operation *> valBS;
    (void)getBackwardSlice(valDefOp, &valBS, bsOptions);
    for (auto b : valBS) {
      backwardSlices.insert(b);
    }
    backwardSlices.insert(valDefOp);
  }
}

// Buffer all allocations of L3 memref directly within the func op's body into
// the func op's arguments.
struct BufferMemrefToFuncArgsPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {

    if (funcOp.isExternal())
      return failure();

    SmallVector<Type, 6> memrefTypes;
    llvm::SetVector<Value> memrefs;
    for (auto &op : funcOp.getFunctionBody().getOps()) {
      if (!isa<memref::AllocOp, memref::AssumeAlignmentOp>(op))
        continue;
      for (auto res : op.getResults()) {
        BaseMemRefType resType = dyn_cast<BaseMemRefType>(res.getType());
        if (!resType)
          continue;
        if (resType.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
          memrefs.insert(res);
      }
    }
    for (auto memref : memrefs)
      memrefTypes.push_back(memref.getType());
    if (memrefs.empty())
      return failure();

    // Append memref to function's arguments.
    auto functionType = funcOp.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(
        llvm::concat<const Type>(functionType.getInputs(), memrefTypes));
    auto newFunctionType = FunctionType::get(funcOp.getContext(), newArgTypes,
                                             functionType.getResults());
    funcOp.setType(newFunctionType);

    // Add the new arguments to the entry block if the function is not external.
    Location loc = funcOp.getLoc();
    for (Value v : memrefs) {
      auto newArg = funcOp.front().addArgument(v.getType(), loc);
      v.replaceAllUsesWith(newArg);
    }
    return success();
  }

private:
};

void air::populateBufferMemrefToFuncArgsPattern(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<BufferMemrefToFuncArgsPattern>(ctx);
}

// Find a common region that contains all ops, or ancestors of ops, until it
// reaches a specified parent op.
Region *
air::findCommonRegionContainingAllAncestors(SmallVector<Operation *> ops,
                                            Operation *until) {
  Region *region = ops.front()->getParentRegion();
  while (llvm::any_of(ops, [region](Operation *op) {
    return !region->isAncestor(op->getParentRegion());
  })) {
    // Failed to find any shared region within the parent IsolatedFromAbove op
    // body.
    if (until && region->getParentOp() == until)
      return nullptr;
    region = region->getParentRegion();
    if (!region)
      return nullptr;
  }
  return region;
}

// A lite version of OperationEquivalence::isRegionEquivalentTo which only
// checks for const value equivalences.
bool air::isRegionEquivalentTo(Region *lhs, Region *rhs) {
  auto blocksEquivalent = [&](Block &lBlock, Block &rBlock) {
    // Check block arguments.
    if (lBlock.getNumArguments() != rBlock.getNumArguments())
      return false;

    for (auto argPair :
         llvm::zip(lBlock.getArguments(), rBlock.getArguments())) {
      Value curArg = std::get<0>(argPair);
      Value otherArg = std::get<1>(argPair);
      if (curArg.getType() != otherArg.getType())
        return false;
    }

    auto opsEquivalent = [&](Operation &lOp, Operation &rOp) {
      // Check for op equality (recursively).
      if (!air::isEquivalentTo(&lOp, &rOp))
        return false;
      return true;
    };
    return llvm::all_of_zip(lBlock, rBlock, opsEquivalent);
  };
  return llvm::all_of_zip(*lhs, *rhs, blocksEquivalent);
}

// A lite version of OperationEquivalence::isEquivalentTo which only checks for
// const value equivalences.
bool air::isEquivalentTo(Operation *lhs, Operation *rhs) {
  // 1. Compare the operation properties.
  if (lhs->getName() != rhs->getName() ||
      lhs->getRawDictionaryAttrs() != rhs->getRawDictionaryAttrs() ||
      lhs->getNumRegions() != rhs->getNumRegions() ||
      lhs->getNumSuccessors() != rhs->getNumSuccessors() ||
      lhs->getNumOperands() != rhs->getNumOperands() ||
      lhs->getNumResults() != rhs->getNumResults() ||
      !lhs->getName().compareOpProperties(lhs->getPropertiesStorage(),
                                          rhs->getPropertiesStorage()))
    return false;

  // 2. Compare operands.
  for (auto operandPair : llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    Value curArg = std::get<0>(operandPair);
    Value otherArg = std::get<1>(operandPair);
    if (curArg.getType() != otherArg.getType())
      return false;
    auto curConst = getConstantIntValue(curArg);
    auto otherConst = getConstantIntValue(otherArg);
    if (curConst && otherConst)
      if (curConst != otherConst)
        return false;
  }

  // 3. Compare result types and mark results as equivalent.
  for (auto resultPair : llvm::zip(lhs->getResults(), rhs->getResults())) {
    Value curArg = std::get<0>(resultPair);
    Value otherArg = std::get<1>(resultPair);
    if (curArg.getType() != otherArg.getType())
      return false;
  }

  // 4. Compare regions.
  for (auto regionPair : llvm::zip(lhs->getRegions(), rhs->getRegions()))
    if (!air::isRegionEquivalentTo(&std::get<0>(regionPair),
                                   &std::get<1>(regionPair)))
      return false;

  return true;
}

// Generate composed affine apply op from arith addi op operating on Index
// values.
affine::AffineApplyOp
air::consructComposedAffineApplyOpFromArithAddI(OpBuilder &builder,
                                                arith::AddIOp addOp) {
  if (!addOp)
    return affine::AffineApplyOp();
  if (llvm::any_of(addOp->getOperands(),
                   [](Value operand) { return !operand.getType().isIndex(); }))
    return affine::AffineApplyOp();
  builder.setInsertionPoint(addOp);
  auto map = AffineMap::get(
      0, 2, builder.getAffineSymbolExpr(0) + builder.getAffineSymbolExpr(1));
  return affine::makeComposedAffineApply(
      builder, addOp.getLoc(), map,
      getAsOpFoldResult({addOp.getLhs(), addOp.getRhs()}));
}

// Generate composed affine apply op from arith muli op operating on Index
// values.
affine::AffineApplyOp
air::consructComposedAffineApplyOpFromArithMulI(OpBuilder &builder,
                                                arith::MulIOp mulOp) {
  if (!mulOp)
    return affine::AffineApplyOp();
  if (llvm::any_of(mulOp->getOperands(),
                   [](Value operand) { return !operand.getType().isIndex(); }))
    return affine::AffineApplyOp();
  builder.setInsertionPoint(mulOp);
  auto map = AffineMap::get(
      0, 2, builder.getAffineSymbolExpr(0) * builder.getAffineSymbolExpr(1));
  return affine::makeComposedAffineApply(
      builder, mulOp.getLoc(), map,
      getAsOpFoldResult({mulOp.getLhs(), mulOp.getRhs()}));
}

/// Get bands of loops that are valid to tile from the top-level of `f`.
/// Ref: mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
void air::getTopLevelTileableBands(
    func::FuncOp f, std::vector<SmallVector<affine::AffineForOp, 6>> &bands) {
  // Get maximal perfect nest of 'affine.for' ops starting from root
  // (inclusive).
  for (affine::AffineForOp forOp : f.getOps<affine::AffineForOp>()) {
    SmallVector<affine::AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, forOp);
    if (isTilingValid(band))
      bands.push_back(band);
  }
}

Operation *air::cloneOpAndOperands(RewriterBase &rewriter, IRMapping &remap,
                                   Operation *op,
                                   function_ref<bool(Operation *)> canClone) {
  // 1) Collect backward slice of 'op' (producers).
  SetVector<Operation *> backwardSlice;
  BackwardSliceOptions bsOptions{/*filter=*/canClone};
  (void)getBackwardSlice(op, &backwardSlice, bsOptions);
  // 2) Clone producers if not already mapped.
  for (auto b : backwardSlice)
    if (air::isPure(b))
      rewriter.clone(*b, remap);
  // 3) Finally clone the target op itself.
  auto new_op = rewriter.clone(*op, remap);
  return new_op;
}

bool air::opOrAncestorIsDominantOver(Operation *a, Operation *b) {
  Region *commonRegion = air::findCommonRegionContainingAllAncestors(
      SmallVector<Operation *>{a, b}, nullptr);
  auto aAncestor = commonRegion->findAncestorOpInRegion(*a);
  auto bAncestor = commonRegion->findAncestorOpInRegion(*b);
  if (!aAncestor || !bAncestor)
    return false;
  DominanceInfo domInfo(aAncestor);
  return domInfo.properlyDominates(aAncestor, bAncestor);
}

} // namespace xilinx
