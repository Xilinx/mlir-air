//===- AIRHerdPlacementPass.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependency.h"
#include "air/Util/Dependency.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "air/Transform/AIRHerdPlacementPass.h"
#include "air/Util/Util.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-place-herds"

namespace {

class Herd {

public:
  Herd(int32_t numRows, int32_t numCols, uint32_t number, std::string name)
      : numRows(numRows), numCols(numCols), number(number), name(name) {
    size = numRows * numCols;
  }

  int32_t getNumRows() const { return numRows; }
  int32_t getNumCols() const { return numCols; }
  std::string getName() const { return name; }
  uint32_t getNumber() const { return number; }
  int32_t getLocX() const { return locX; }
  int32_t getLocY() const { return locY; }
  int32_t getSize() const { return size; }

  void setLocX(int32_t x) { locX = x; }
  void setLocY(int32_t y) { locY = y; }

  void printHerd() const {
    llvm::outs() << "name: " << name << ", numRows: " << numRows
                 << ", number: " << number << ", numCols: " << numCols
                 << ", x_loc: " << locX << ", y_loc: " << locY
                 << ", size: " << size << "\n";
  }

private:
  int32_t numRows;
  int32_t numCols;
  uint32_t number;
  std::string name;
  int32_t size;

  int32_t locX = -1;
  int32_t locY = -1;
};

struct HerdComparision {
  inline bool operator()(const std::unique_ptr<Herd> &l,
                         const std::unique_ptr<Herd> &r) {
    return l->getSize() > r->getSize();
  }
};

class Partition {

public:
  Partition(int numRows, int numCols, int anchorPointRow, int anchorPointCol)
      : numRows(numRows), numCols(numCols), anchorPointRow(anchorPointRow),
        anchorPointCol(anchorPointCol) {
    grid.assign(numRows, std::vector<int>(numCols, -1));
  }

  std::vector<std::vector<int>> grid;

  int32_t getNumRows() const { return numRows; }
  int32_t getNumCols() const { return numCols; }
  int32_t getAnchorPointRow() const { return anchorPointRow; }
  int32_t getAnchorPointCol() const { return anchorPointCol; }
  int32_t getLocX() const { return locX; }
  int32_t getLocY() const { return locY; }

  bool isLegalPlacement(std::unique_ptr<Herd> &herd, int32_t row,
                        int32_t col) const {
    for (int i = numRows - row - herd->getNumRows(); i < numRows - row; i++) {
      for (int j = col; j < herd->getNumCols() + col; j++) {
        // build down and to the right

        if (i < 0 || j >= numCols) {
          return false;
        }

        if (i >= numRows || j < 0) {
          return false;
        }

        if (grid[i][j] != -1) {
          return false;
        }
      }
    }
    return true;
  }

  // row and col refer to the top left corner location of the herd
  void placeHerd(std::unique_ptr<Herd> &herd, int32_t row, int32_t col) {
    for (int i = numRows - row - herd->getNumRows(); i < numRows - row; i++) {
      for (int j = col; j < herd->getNumCols() + col; j++) {
        grid[i][j] = herd->getNumber();
      }
    }
  }

  void printPartition() const {
    for (uint32_t i = 0; i < grid.size(); i++) {
      for (uint32_t j = 0; j < grid[i].size(); j++) {
        llvm::outs() << grid[i][j] << " ";
      }
      llvm::outs() << "\n";
    }
    llvm::outs() << "\n";
  }

private:
  int32_t numRows;
  int32_t numCols;
  int32_t anchorPointRow;
  int32_t anchorPointCol;
  int32_t locX;
  int32_t locY;
};

class AIRHerdPlacementPass
    : public AIRHerdPlacementPassBase<AIRHerdPlacementPass> {

public:
  AIRHerdPlacementPass() = default;
  AIRHerdPlacementPass(const AIRHerdPlacementPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
  }

  void runOnOperation() override {

    if (numRows < 0 || numCols < 0 || anchorPointRow < 0 ||
        anchorPointCol < 0) {
      llvm::errs() << "Ensure all input parameters are greater than zero.\n";
      return;
    }
    auto module = getOperation();

    OpBuilder module_builder(module);

    // Number of the current herd
    uint32_t number = 0;
    std::vector<std::unique_ptr<Herd>> unplacedHerds;
    std::vector<xilinx::air::HerdOp> herdOps;
    for (auto f : module.getOps<func::FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto herd = dyn_cast<xilinx::air::HerdOp>(op)) {
          herdOps.push_back(herd);
          std::string name = "herd";
          if (auto attr = herd->getAttrOfType<StringAttr>(
                  SymbolTable::getSymbolAttrName()))
            name = attr.getValue().str();
          SmallVector<Value, 2> herd_size = herd.getSizeOperands();
          if (!isa<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()) ||
              !isa<arith::ConstantIndexOp>(herd_size[1].getDefiningOp())) {
            llvm::errs() << "Only constant sized herds are supported\n";
            return;
          }
          int64_t herd_size_x =
              cast<arith::ConstantIndexOp>(herd_size[0].getDefiningOp())
                  .value();
          int64_t herd_size_y =
              cast<arith::ConstantIndexOp>(herd_size[1].getDefiningOp())
                  .value();

          std::unique_ptr<Herd> herdPtr =
              std::make_unique<Herd>(herd_size_x, herd_size_y, number, name);
          unplacedHerds.push_back(std::move(herdPtr));

          number++;
        }
      });
    }
    std::unique_ptr<Partition> partition = std::make_unique<Partition>(
        numRows, numCols, anchorPointRow, anchorPointCol);
    std::sort(unplacedHerds.begin(), unplacedHerds.end(), HerdComparision());
    std::vector<std::unique_ptr<Herd>> placedHerds;
    naivePlacement(partition, unplacedHerds, placedHerds);

    if (unplacedHerds.size() != 0) {
      module.emitError("No valid placement found.");
      for (uint32_t i = 0; i < unplacedHerds.size(); i++) {
        herdOps[unplacedHerds[i]->getNumber()]->emitOpError("\nUnplaced herd: ")
            << unplacedHerds[i]->getName() << "\n";
      }
      return;
    }

    auto xLocName = xilinx::air::HerdOp::getColOffsetAttrName();
    auto yLocName = xilinx::air::HerdOp::getRowOffsetAttrName();

    for (uint32_t i = 0; i < herdOps.size(); i++) {

      int32_t herdIndex = placedHerds[i]->getNumber();
      herdOps[herdIndex]->setAttr(
          yLocName,
          IntegerAttr::get(
              IntegerType::get(herdOps[herdIndex]->getContext(), 64),
              placedHerds[i]->getLocY() + partition->getAnchorPointRow()));
      herdOps[herdIndex]->setAttr(
          xLocName,
          IntegerAttr::get(
              IntegerType::get(herdOps[herdIndex]->getContext(), 64),
              placedHerds[i]->getLocX() + partition->getAnchorPointCol()));
    }
    return;
  }

private:
  // Performs placement, trying to place the first herd on the anchor point
  // first, moving from left -> right, up a row, then left -> right again. Will
  // try to place each remaining unplaced herd in each open partition tile.
  void naivePlacement(std::unique_ptr<Partition> &partition,
                      std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                      std::vector<std::unique_ptr<Herd>> &placedHerds) {
    for (int64_t i = 0; i < partition->getNumRows(); i++) {
      for (int64_t j = 0; j < partition->getNumCols(); j++) {
        if (partition->grid[partition->getNumRows() - i - 1][j] == -1) {
          for (uint32_t k = 0; k < unplacedHerds.size(); k++) {
            bool legalPlace =
                partition->isLegalPlacement(unplacedHerds[k], i, j);
            if (legalPlace) {
              partition->placeHerd(unplacedHerds[k], i, j);
              unplacedHerds[k]->setLocX(j);
              unplacedHerds[k]->setLocY(i + unplacedHerds[k]->getNumRows() - 1);
              placedHerds.push_back(std::move(unplacedHerds[k]));
              unplacedHerds.erase(unplacedHerds.begin() + k);
              if (unplacedHerds.size() == 0) {
                return;
              }
              break;
            }
          }
        }
      }
    }
    return;
  }

}; // end AIRHerdPlacementPass

} // end namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdPlacementPass() {
  return std::make_unique<AIRHerdPlacementPass>();
}

} // namespace air
} // namespace xilinx