//===- Util.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/Util.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-util"

using namespace mlir;
using namespace xilinx;

const StringLiteral air::LinalgTransforms::kLinalgTransformMarker =
    "__internal_linalg_transform__";

static std::string getMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
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
  } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  } else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
    ret << "I64";
  } else if (ty.dyn_cast<air::AsyncTokenType>()) {
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

LogicalResult air::normalizeLoop(AffineForOp afo) {
  auto ubMap = afo.getUpperBoundMap();
  auto lbMap = afo.getLowerBoundMap();
  auto ctx = afo.getContext();
  auto loc = afo.getLoc();

  auto step_expr = getAffineConstantExpr(afo.getStep(), ctx);

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
  auto new_iv = builder.create<AffineApplyOp>(loc, iv_map, iv);
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
  if (auto t = ty.dyn_cast<ShapedType>()) {
    return getTensorVolume(t);
  } else {
    return 1;
  }
}

// Get the parent scf.for op of an iter_arg
scf::ForOp air::getForRegionIterArgsOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return scf::ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
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
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return air::HerdOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HerdOp>(containingOp);
}

// Get the parent air.hierarchy op of a tile id
air::HierarchyInterface air::getHierarchyArgOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return air::HierarchyInterface();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HierarchyInterface>(containingOp);
}

// Get operation's "id" attribute
int air::getIdAttr(Operation *op) {
  auto idAttr = op->getAttrOfType<IntegerAttr>("id");
  assert(idAttr && "op has no attribute named 'id'");
  return idAttr.getInt();
}

// Renumber the DMA ops
void air::renumberDmaOps(func::FuncOp func, std::string mode) {
  unsigned id = 0;
  if (mode == "global") {
    // Renumber DMA ops per entire module
    func->walk([&](Operation *func_dma) {
      if (dyn_cast<xilinx::air::DmaMemcpyInterface>(func_dma)) {
        func_dma->setAttr(
            "id",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(func_dma->getContext(), 32), ++id));
      }
    });
  } else if (mode == "herd") {
    for (auto herd : func.getOps<xilinx::air::HerdOp>()) {
      id = 0;
      // Renumber DMA ops per air herd
      herd->walk([&](Operation *herd_dma) {
        if (dyn_cast<xilinx::air::DmaMemcpyInterface>(herd_dma)) {
          herd_dma->setAttr(
              "id",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(herd_dma->getContext(), 32), ++id));
        }
      });
    }
  } else
    assert(false && "Unknown dma renumber mode. Supported modes: global, herd");
}

// Return op name as string
std::string air::to_string(Operation *op) {
  return op->getName().getStringRef().str();
}

// Return memory space as string
std::string air::getMemorySpaceAsString(Value memref) {
  assert(memref.getType().isa<MemRefType>() && "value is not a memref");
  auto memory_space_as_int =
      memref.getType().dyn_cast<MemRefType>().getMemorySpaceAsInt();
  std::string memorySpaceStr;
  if (memory_space_as_int == (int)air::MemorySpace::L1) {
    memorySpaceStr = "L1";
  } else if (memory_space_as_int == (int)air::MemorySpace::L2) {
    memorySpaceStr = "L2";
  } else if (memory_space_as_int == (int)air::MemorySpace::L3) {
    memorySpaceStr = "L3";
  } else
    assert(false && "unknown memory space");
  return memorySpaceStr;
}

// Returns the first affine if op in block; nullptr otherwise
mlir::AffineIfOp air::getAffineIfInBlock(mlir::Block *block) {
  for (auto op : block->getOps<mlir::AffineIfOp>()) {
    return op;
  }
  return mlir::AffineIfOp();
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
  assert(index + 1 <= op->getNumOperands() && "Index out of range");
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
  int32_t sum = 0;
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
  auto module = op->getParentOfType<ModuleOp>();
  return dyn_cast<air::ChannelOp>(module.lookupSymbol(op.getChanName()));
}

// Get ChannelPutOp through ChannelOp
air::ChannelPutOp air::getChannelPutOpThroughSymbol(air::ChannelOp channel) {
  auto module = channel->getParentOfType<ModuleOp>();
  auto attr =
      channel->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

  air::ChannelPutOp output = nullptr;
  module.walk([&](Operation *op) {
    if (auto put = dyn_cast<air::ChannelPutOp>(op)) {
      if (put.getChanName() == attr) {
        output = put;
      }
    }
  });

  if (output)
    return output;
  else
    return ChannelPutOp();
}

// Get ChannelGetOp through ChannelOp
air::ChannelGetOp air::getChannelGetOpThroughSymbol(air::ChannelOp channel) {
  auto module = channel->getParentOfType<ModuleOp>();
  auto attr =
      channel->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

  air::ChannelGetOp output = nullptr;
  module.walk([&](Operation *op) {
    if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
      if (get.getChanName() == attr) {
        output = get;
      }
    }
  });

  if (output)
    return output;
  else
    return ChannelGetOp();
}

// Get the other channel op through channel symbol
air::ChannelGetOp
air::getTheOtherChannelOpThroughSymbol(air::ChannelPutOp put) {
  auto module = put->getParentOfType<ModuleOp>();
  auto channel_op = getChannelDeclarationThroughSymbol(
      dyn_cast<air::ChannelInterface>(put.getOperation()));
  auto attr =
      channel_op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

  air::ChannelGetOp output = nullptr;
  module.walk([&](Operation *op) {
    if (auto get = dyn_cast<air::ChannelGetOp>(op)) {
      if (get.getChanName() == attr) {
        output = get;
      }
    }
  });

  if (output)
    return output;
  else
    return ChannelGetOp();
}

// Get the other channel op through channel symbol
air::ChannelPutOp
air::getTheOtherChannelOpThroughSymbol(air::ChannelGetOp get) {
  auto module = get->getParentOfType<ModuleOp>();
  auto channel_op = getChannelDeclarationThroughSymbol(
      dyn_cast<air::ChannelInterface>(get.getOperation()));
  auto attr =
      channel_op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());

  air::ChannelPutOp output = nullptr;
  module.walk([&](Operation *op) {
    if (auto put = dyn_cast<ChannelPutOp>(op)) {
      if (put.getChanName() == attr) {
        output = put;
      }
    }
  });

  if (output)
    return output;
  else
    return ChannelPutOp();
}
