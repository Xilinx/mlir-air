// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include "air/Util/Outliner.h"

#include <vector>

#define DEBUG_TYPE "air-outliner"

using namespace mlir;

namespace xilinx {
namespace air {

CallOp AIROutliner::outline(AffineForOp forOp, std::string fname) {

  llvm::SetVector<Value> outline_args;

  auto loc = forOp.getLoc();

  for (Value v : forOp.getOperands())
    outline_args.insert(v);

  auto &region = forOp.getRegion();
  llvm::SetVector<Value> region_args;
  getUsedValuesDefinedAbove(region, region_args);

  for (Value v : region_args) {
    auto o = v.getDefiningOp();
    if (o && isa<arith::ConstantOp>(o)) {
      auto builder = OpBuilder::atBlockBegin(forOp.getBody());
      auto c = builder.clone(*o);
      replaceAllUsesInRegionWith(v, c->getResult(0), region);
    }
    else {
      outline_args.insert(v);
    }
  }

  std::vector<mlir::Type> arg_types, ret_types;
  for (auto a : outline_args)
    arg_types.push_back(a.getType());

  auto module = forOp->getParentOfType<mlir::ModuleOp>();
  auto func_type = mlir::FunctionType::get(forOp.getContext(), arg_types, ret_types);

  std::string new_fname = fname;
  int which_try = 0;
  while (module.lookupSymbol(new_fname))
    new_fname = fname + "_" + std::to_string(++which_try);
  fname = new_fname;

  auto function = mlir::FuncOp::create(loc, fname,
                                       func_type, /* attrs = */ {});
  module.push_back(function);

  auto &entryBlock = *function.addEntryBlock();

  BlockAndValueMapping mapper;
  int idx = 0;
  for (auto a : outline_args)
    mapper.map(a, entryBlock.getArgument(idx++));

  auto body_builder = OpBuilder::atBlockBegin(&entryBlock);
  Operation* clone = body_builder.clone(*forOp.getOperation(), mapper);
  assert(clone);

  body_builder.create<ReturnOp>(loc);

  OpBuilder call_builder(forOp);
  return call_builder.create<CallOp>(loc, function, outline_args.getArrayRef());
}

CallOp AIROutliner::outline(std::vector<mlir::Operation*> ops, std::string fname) {
  if (!ops.size())
    return nullptr;

  auto module = ops[0]->getParentOfType<mlir::ModuleOp>();
  Block *bb = nullptr;
  std::vector<Operation*> outline_ops;
  std::vector<Value> outline_rets;
  std::vector<mlir::Type> ret_types;
  std::vector<Value> outline_args;

  for (Operation *op : ops) {
    assert((!bb || bb == op->getBlock()) && "operations must be in same basic block");
    bb = op->getBlock();
    for (Value v : op->getOperands()) {
      auto def = v.getDefiningOp();
      if (def && dyn_cast<arith::ConstantOp>(def))
        outline_ops.push_back(def);
      else
        outline_args.push_back(v);
    }
    outline_ops.push_back(op);
    for (Value v : op->getResults()) {
      outline_rets.push_back(v);
      ret_types.push_back(v.getType());
    }
  }

  auto context = ops[0]->getContext();
  auto loc = ops[0]->getLoc();

  std::vector<mlir::Type> arg_types;
  for (auto a : outline_args)
    arg_types.push_back(a.getType());

  std::string new_fname = fname;
  int which_try = 0;
  while (module.lookupSymbol(new_fname))
    new_fname = fname + "_" + std::to_string(++which_try);
  fname = new_fname;

  auto func_type = mlir::FunctionType::get(context, arg_types, ret_types);
  auto function = mlir::FuncOp::create(loc, fname,
                                       func_type, /* attrs = */ {});

  auto &entryBlock = *function.addEntryBlock();

  BlockAndValueMapping mapper;
  int idx = 0;
  for (auto a : outline_args)
    mapper.map(a, entryBlock.getArgument(idx++));

  SmallVector<Value, 4> rets;
  for (Operation *op : outline_ops) {
    Operation* clone = op->clone(mapper);
    for (auto p : llvm::zip(op->getResults(), clone->getResults()))
      mapper.map(std::get<0>(p), std::get<1>(p));
    entryBlock.push_back(clone);
    auto results = clone->getResults();
    rets.append(results.begin(), results.end());
  }

  auto builder = OpBuilder(ops[0]);
  auto call = builder.create<CallOp>(loc, function, outline_args);

  auto func_builder = OpBuilder::atBlockEnd(&entryBlock);
  func_builder.create<ReturnOp>(loc, rets);

  idx = 0;
  for (auto r : call.getResults()) {
    outline_rets[idx++].replaceAllUsesWith(r);
  }
  for (auto op : ops)
    op->erase();

  module.push_back(function);
  return call;
}

} // namespace aten
} // namespace xilinx
