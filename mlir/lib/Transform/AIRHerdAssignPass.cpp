// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"

#include "air/Transform/AIRHerdAssignPass.h"
#include "air/Util/Util.h"

#include <vector>
#include <deque>

#include "PassDetail.h"

#define DEBUG_TYPE "air-herd-assign"

using namespace mlir;

static llvm::cl::OptionCategory clOptionsCategory(DEBUG_TYPE " options");

namespace {

class AIRHerdAssignPass : public PassWrapper<AIRHerdAssignPass,
                                              OperationPass<ModuleOp>> {

public:

  AIRHerdAssignPass() = default;
  AIRHerdAssignPass(const AIRHerdAssignPass &pass) {}

  Option<int> HerdAssignDepth{*this, "herd-assign-depth",
                                     llvm::cl::desc("herd assign depth"),
                                     llvm::cl::init(0)};

  void lowerAddOneHerd(Operation *kernelCallOp) {

    auto module = getOperation();

    std::vector<AffineForOp> fors;

    auto f = kernelCallOp->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_back(f);

    f = f->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_back(f);

    f = f->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_back(f);

    f = f->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_back(f);

    xilinx::air::normalizeLoop(fors[3]);
    xilinx::air::normalizeLoop(fors[2]);

    LLVM_DEBUG(llvm::outs() << "\nNormalize loops:\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    xilinx::air::coalesceLoops(fors[3], fors[2]);
    LLVM_DEBUG(llvm::outs() << "\nCoalesce loops:\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    {
      OpBuilder builder(fors[3]);
      auto ctx = fors[3].getContext();
      auto loc = fors[3].getLoc();
      auto ub_map_0 = fors[3].getUpperBoundMap();
      assert(ub_map_0.isSingleConstant());
      int64_t ub_0 = ub_map_0.getSingleConstantResult();

      auto affine_par = builder.create<AffineParallelOp>(loc,
                                                         std::vector<Type>{},
                                                         std::vector<AtomicRMWKind>{},
                                                         std::vector<int64_t>{ub_0});

      fors[3].getBody()->back().erase();
      affine_par.getBody()->getOperations().splice(affine_par.getBody()->begin(),
                                                   fors[3].getBody()->getOperations());
      fors[3].getInductionVar().replaceAllUsesWith(affine_par.getIVs()[0]);
      fors[3].erase();

      Block *BB = kernelCallOp->getBlock();
      AffineStoreOp ast = nullptr;
      AffineLoadOp ald = nullptr;
      for (auto &op : BB->getOperations()) {
        if (auto load = dyn_cast<AffineLoadOp>(&op))
          ald = load;
        else if (auto store = dyn_cast<AffineStoreOp>(&op))
          ast = store;
      }
      kernelCallOp->setOperand(0, ald.memref());
      kernelCallOp->setOperand(1, ast.memref());
      SmallVector<Type, 1> retTys;
      SmallVector<Type, 2> tys;
      for (auto t : kernelCallOp->getOperandTypes())
        tys.push_back(t);
      cast<FuncOp>(module.lookupSymbol(cast<CallOp>(kernelCallOp).getCallee())).setType(FunctionType::get(ctx, tys, retTys));
      kernelCallOp->moveBefore(fors[1]);
      fors[0].erase();
      fors[1].erase();
    }
    LLVM_DEBUG(llvm::outs() << "\noutput:\n");
    LLVM_DEBUG(module.print(llvm::outs()));
  }

  void lowerConv2dHerd(Operation *kernelCallOp) {

    auto module = getOperation();

    unsigned herdRows = 1;
    unsigned herdCols = 1;

    std::deque<AffineForOp> fors;

    auto f = kernelCallOp->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_front(f);
    f = f->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_front(f);
    f = f->getParentOfType<AffineForOp>();
    assert(f);
    fors.push_front(f);

    for (auto f : fors)
      xilinx::air::normalizeLoop(f);

    LLVM_DEBUG(llvm::outs() << "\nNormalize loops:\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    xilinx::air::coalesceLoops(fors[1], fors[2]);
    LLVM_DEBUG(llvm::outs() << "\nCoalesce loops:\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    {
      OpBuilder builder(fors[0]);
      auto ctx = fors[0].getContext();
      auto loc = fors[0].getLoc();
      auto ub_map_0 = fors[0].getUpperBoundMap();
      auto ub_map_1 = fors[1].getUpperBoundMap();
      assert(ub_map_0.isSingleConstant() && ub_map_1.isSingleConstant());
      int64_t ub_0 = ub_map_0.getSingleConstantResult();
      int64_t ub_1 = ub_map_1.getSingleConstantResult();

      auto affine_par = builder.create<AffineParallelOp>(loc,
                                                         std::vector<Type>{},
                                                         std::vector<AtomicRMWKind>{},
                                                         std::vector<int64_t>{ub_0,ub_1});

      fors[0].getBody()->back().erase();
      affine_par.getBody()->getOperations().splice(affine_par.getBody()->begin(),
                                                   fors[0].getBody()->getOperations());
      fors[0].getInductionVar().replaceAllUsesWith(affine_par.getIVs()[0]);
      fors[0].erase();
      fors[1].getBody()->back().erase();
      affine_par.getBody()->getOperations().splice(Block::iterator(fors[1].getOperation()),
                                                   fors[1].getBody()->getOperations());
      fors[1].getInductionVar().replaceAllUsesWith(affine_par.getIVs()[1]);
      fors[1].erase();

      builder.setInsertionPoint(kernelCallOp);
      auto herd_row_expr = getAffineDimExpr(0, ctx) % getAffineConstantExpr(herdRows, ctx);
      auto herd_row = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, herd_row_expr),
                                                    affine_par.getIVs()[0]);
      auto herd_col_expr = getAffineDimExpr(0, ctx) % getAffineConstantExpr(herdCols, ctx);
      auto herd_col = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, herd_col_expr),
                                                    affine_par.getIVs()[1]);
      auto num_ops = kernelCallOp->getNumOperands();
      kernelCallOp->setOperand(num_ops-4, herd_row);
      kernelCallOp->setOperand(num_ops-3, herd_col);
      kernelCallOp->setOperand(num_ops-2, affine_par.getIVs()[0]);
      kernelCallOp->setOperand(num_ops-1, affine_par.getIVs()[1]);
    }

    LLVM_DEBUG(llvm::outs() << "\nSpatial for loop:\n");
    LLVM_DEBUG(module.print(llvm::outs()));

  }

  void loopsToParallel(ArrayRef<AffineForOp> nest, int depth)
  {
    assert((int)nest.size() > depth+1);
    AffineForOp outer = nest[depth];
    AffineForOp inner = nest[depth+1];

    xilinx::air::normalizeLoop(inner);
    xilinx::air::normalizeLoop(outer);
    {
      OpBuilder builder(outer);
      auto loc = outer.getLoc();
      auto ub_map_0 = outer.getUpperBoundMap();
      auto ub_map_1 = inner.getUpperBoundMap();
      assert(ub_map_0.isSingleConstant() && ub_map_1.isSingleConstant());
      int64_t ub_0 = ub_map_0.getSingleConstantResult();
      int64_t ub_1 = ub_map_1.getSingleConstantResult();

      auto affine_par = builder.create<AffineParallelOp>(loc,
                                                         std::vector<Type>{},
                                                         std::vector<AtomicRMWKind>{},
                                                         std::vector<int64_t>{ub_0,ub_1});

      outer.getBody()->back().erase();
      affine_par.getBody()->getOperations().splice(affine_par.getBody()->begin(),
                                                   outer.getBody()->getOperations());
      outer.getInductionVar().replaceAllUsesWith(affine_par.getIVs()[0]);
      outer.erase();

      inner.getBody()->back().erase();
      affine_par.getBody()->getOperations().splice(Block::iterator(inner.getOperation()),
                                                   inner.getBody()->getOperations());
      inner.getInductionVar().replaceAllUsesWith(affine_par.getIVs()[1]);
      inner.erase();
    }
  }

  void runOnOperation() override {

    auto module = getOperation();

    LLVM_DEBUG(llvm::outs() << "Starting herd assignment\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    //
    // Herd assignment
    //

    for (auto f : module.getOps<FuncOp>()) {
      std::vector<SmallVector<AffineForOp, 6>> bands;
      getTileableBands(f, &bands);
      for (auto &band : bands) {
        auto stringAttr = band[0]->getAttrOfType<StringAttr>(
          "affine_opt_label");
        if (!stringAttr) continue;
        int depth = HerdAssignDepth;
        loopsToParallel(band, depth);
        LLVM_DEBUG(llvm::outs() << "finished band\n");
        LLVM_DEBUG(module.print(llvm::outs()));
      }
    }

    //std::vector<CallOp> dmaOps;
    for (auto f : module.getOps<FuncOp>()) {
      std::vector<CallOp> kernelOps;
      f.walk([&](Operation *o) {
        if (auto co = dyn_cast<CallOp>(o)) {
          if (co.getCallee().startswith("acap_conv2d_hw_kernel")) {
            kernelOps.push_back(co);
          }
        }
      });
    //     else if (co.getCallee().startswith("acap_add_one_hw_kernel")) {
    //       kernelOps.push_back(co);
    //     }
    //     else if (co.getCallee().startswith("acap_dma_outline")) {
    //       //dmaOps.push_back(co);
    //     }
    //   }
    // });
      for (auto co : kernelOps) {
        if (co.getCallee().startswith("acap_conv2d_hw_kernel")) {
          lowerConv2dHerd(co);
        }
      }
    }
    //   else if (co.getCallee().startswith("acap_add_one_hw_kernel")) {
    //     lowerAddOneHerd(co);
    //   }
    // }
  }

private:

};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdAssignPass() {
  return std::make_unique<AIRHerdAssignPass>();
}

} // namespace air
} // namespace xilinx
