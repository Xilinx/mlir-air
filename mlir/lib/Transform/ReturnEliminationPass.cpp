// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

// This is an empty module pass.  Copy it to make your own pass.

#include "air/Transform/ReturnEliminationPass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include <vector>
#include <set>

#include "PassDetail.h"

#define DEBUG_TYPE "return-elimination"

using namespace mlir;

namespace {

class ReturnEliminationPass : public PassWrapper<ReturnEliminationPass,
                                                 OperationPass<ModuleOp>> {

public:
  ReturnEliminationPass() {}

  void runOn(Operation *op) {
    auto module = getOperation();

    if (visitedOps.count(op))
      return;
    visitedOps.insert(op);

    if (auto callOp = dyn_cast<CallOp>(op)) {

      auto builder = std::make_unique<mlir::OpBuilder>(op);

      std::vector<Type> tys;
      for (auto t : callOp.getCalleeType().getInputs())
        tys.push_back(t);
      for (auto t : callOp.getCalleeType().getResults())
        tys.push_back(t);

      auto newFnTy = FunctionType::get(op->getContext(), tys, {});
      std::string newFnName = callOp.callee().str()+"_out";

      if (!module.lookupSymbol<FuncOp>(newFnName)) {
        auto fn = FuncOp::create(op->getLoc(), newFnName, newFnTy);
        fn.setPrivate();
        module.push_back(fn);
      }

      std::vector<Value> newCallArgs{callOp.arg_operand_begin(),
                                      callOp.arg_operand_end()};

      for (auto v : callOp.getResults()) {
        if (!v.getType().isa<MemRefType>())
          llvm_unreachable("function returns non-memref");
        if (!valueMap.count(v)) {
          valueMap[v] = builder->create<memref::AllocOp>(op->getLoc(),
                                                 v.getType().cast<MemRefType>());
        }
        v.replaceAllUsesWith(valueMap[v]);
        newCallArgs.push_back(valueMap[v]);
      }

      /*auto newCallOp =*/ builder->create<CallOp>(op->getLoc(),
                                               newFnName,
                                               ArrayRef<Type>{},
                                               newCallArgs);
      erasedOps.insert(op);
      auto fn = module.lookupSymbol<FuncOp>(callOp.callee());
      if (fn && fn.use_empty()) erasedOps.insert(fn);
    }
    else if ( isa<memref::AllocOp>(op) ) {
      Value v = op->getResult(0);
      if (valueMap.count(v)) {
        v.replaceAllUsesWith(valueMap[v]);
        erasedOps.insert(op);
      }
    }
    // else if ( isa<xilinx::air::AllocOp>(op) ) {
    // }
    else {
      //getModule().dump();
      //op->dump();
      //llvm_unreachable("unhandled operation type");
    }

    for (Value v : op->getOperands()) {
      if (!v.getType().isa<MemRefType>())
        continue;
      if (v.isa<BlockArgument>())
        continue;
      if (v.getDefiningOp())
        runOn(v.getDefiningOp());
    }

  }

  void runOnOperation() override {

    auto module = getOperation();

    for (auto graph : module.getOps<mlir::FuncOp>()) {
      // assume a single return statement
      if (graph.isExternal())
        return;
      Block &BB = graph.front();
  
      std::vector<ReturnOp> retOps;
      graph.walk([&](Operation *op) {
        if (auto r = dyn_cast<ReturnOp>(op))
          retOps.push_back(r);
      });

      FunctionType funcTy = graph.getType();
      if (funcTy.getNumResults() == 0)
        continue;

      std::vector<Type> newFuncInputTys;

      for (auto ty : funcTy.getInputs())
        newFuncInputTys.push_back(ty);

      for (auto ty : funcTy.getResults())
        newFuncInputTys.push_back(ty);

      FunctionType newFuncTy = FunctionType::get(module.getContext(), newFuncInputTys, {});
      graph.setType(newFuncTy);

      Operation *retOp = retOps.front();
      auto builder = std::make_unique<mlir::OpBuilder>(retOp);

      builder->create<ReturnOp>(retOp->getLoc());

      std::vector<Value> operands{retOp->getOperands().begin(),
                                  retOp->getOperands().end()};

      retOp->dropAllReferences();
      erasedOps.insert(retOp);

      for (Value v : operands)
        valueMap[v] = BB.addArgument(v.getType());

      for (Value v : operands) {
        if (!v.getType().isa<MemRefType>())
          llvm_unreachable("graph function returns non-memref");
        if (v.getDefiningOp())
          runOn(v.getDefiningOp());
      }

      for (auto oi=BB.rbegin(),oe=BB.rend(); oi!=oe; ++oi) {
        Operation *o = &*oi;
        for (Value v : o->getResults()) {
          if (v.getType().isa<MemRefType>()) {
            runOn(o);
            break;
          }
        }
      }

      for (Operation *o : erasedOps)
        o->erase();
    }
  }

private:
  llvm::DenseMap<Value,Value> valueMap;
  std::set<Operation*> visitedOps;
  std::set<Operation*> erasedOps;
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createReturnEliminationPass() {
  return std::make_unique<ReturnEliminationPass>();
}

} // namespace air
} // namespace xilinx

