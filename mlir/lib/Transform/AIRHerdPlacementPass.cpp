//===- AIRHerdPlacementPass.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "air/Transform/AIRHerdPlacementPass.h"
#include "air/Util/Util.h"

#include <algorithm>
#include <map>
#include <numeric> 
#include <string>
#include <vector>
#include <iostream>
#include <memory>


using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-place-herds"

namespace {

    // These should be determined by either looking up the grid dimensions or taking them 
    // right from the input; however, there is no operation defined yet.
    static const int32_t rowSize = 8;
    static const int32_t colSize = 10;

class Herd {

public:
  Herd(int32_t numRows, int32_t numCols, uint32_t number, std::string name):
    numRows(numRows), numCols(numCols), number(number), name(name) {
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
    llvm::outs() << "name: "    << name   
                 << ", numRows: " << numRows
                 << ", number: " << number
                 << ", numCols: " << numCols 
                 << ", x_loc: " << locX 
                 << ", y_loc: " << locY 
                 << ", size: " << size
                 << "\n";  
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
  inline bool operator() (const std::unique_ptr<Herd> &l, const std::unique_ptr<Herd> &r)
  {
      return l->getSize() > r->getSize();
  }
};

class Grid {

public:
  Grid(int numRows, int numCols):numRows(numRows), numCols(numCols) {
    grid.assign(numRows, std::vector<int>(numCols, -1));
  }
  
  std::vector<std::vector<int>> grid;

  int32_t getNumRows() const { return numRows; }
  int32_t getNumCols() const { return numCols; }
  int32_t getLocX() const { return locX; }
  int32_t getLocY() const { return locY; }

  bool isLegalPlacement (std::unique_ptr<Herd> &herd, int32_t row, int32_t col) const {
    for (int i = numRows - 1 - row; i < herd->getNumRows() + numRows - 1 - row; i++) {
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
    for (int i = numRows - 1 - row; i < herd->getNumRows() + numRows - 1 - row; i++) {
      for (int j = col; j < herd->getNumCols() + col; j++) {
        grid[i][j] = herd->getNumber();
      }
    }
  }

  void printGrid() const {
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
  int32_t locX;
  int32_t locY;
};

class AIRHerdPlacementPass : public AIRHerdPlacementPassBase<AIRHerdPlacementPass> {

public:
  AIRHerdPlacementPass() = default;
  AIRHerdPlacementPass(const AIRHerdPlacementPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
  }

  void runOnOperation() override {

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
          if (auto attr =
              herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
            name = attr.getValue().str();
          SmallVector<Value, 2> herd_size = herd.getSizeOperands();
          if (!isa<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()) ||
              !isa<arith::ConstantIndexOp>(herd_size[1].getDefiningOp())) {
            llvm::errs() << "Only constant sized herds are supported";
            return;
          }
          int64_t herd_size_x =
            cast<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()).value();
          int64_t herd_size_y =
            cast<arith::ConstantIndexOp>(herd_size[1].getDefiningOp()).value();
          
          std::unique_ptr<Herd> herdPtr = std::make_unique<Herd>(herd_size_x, herd_size_y, number, name);
          unplacedHerds.push_back(std::move(herdPtr));

          number++;
          }
      });
    }
    std::unique_ptr<Grid> grid = std::make_unique<Grid>(rowSize, colSize);
    std::sort(unplacedHerds.begin(), unplacedHerds.end(), HerdComparision());
    std::vector<std::unique_ptr<Herd>> placedHerds;
    naivePlacement(grid, unplacedHerds, placedHerds);

    if (unplacedHerds.size() != 0) {
      llvm::outs() << "Valid placement Not found." << "\n";
      for (uint32_t i = 0; i < unplacedHerds.size(); i++) {
        unplacedHerds[i]->printHerd();
      }
      return;
    }

    auto col_name = xilinx::air::HerdOp::getColOffsetAttrName();
    auto row_name = xilinx::air::HerdOp::getRowOffsetAttrName();

    for (uint32_t i = 0; i < herdOps.size(); i++) {
      for (uint32_t j = 0; j < placedHerds.size(); j++) {
        std::string herdName;
        if (auto attr =
                herdOps[i]->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          herdName = attr.getValue().str();
        }
        if (herdName == placedHerds[j]->getName()) {

          herdOps[i]->setAttr(row_name, IntegerAttr::get(IntegerType::get(herdOps[i]->getContext(), 64),
                                          placedHerds[j]->getLocX()));
          herdOps[i]->setAttr(col_name, IntegerAttr::get(IntegerType::get(herdOps[i]->getContext(), 64),
                                          placedHerds[j]->getLocY()));
        }
      }
    }
    return;
  }

private:

  // Performs placement, trying to place the first herd in the file first, moving from top 
  // left -> right, down a row, then left -> right again. Will try to place each remaining
  // unplaced herd in each open partition tile.
  void naivePlacement(std::unique_ptr<Grid> &grid, std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                      std::vector<std::unique_ptr<Herd>> &placedHerds) {
    // i starts at n - 1, not n
    for (int64_t i = grid->getNumRows(); i-- > 0 ;) {
      for (int64_t j = 0; j < grid->getNumCols(); j++) {
        if (grid->grid[grid->getNumRows() - i - 1][j] == -1) {
          for (uint32_t k = 0; k < unplacedHerds.size(); k++) {
            bool legalPlace = grid->isLegalPlacement(unplacedHerds[k], i, j);
            if(legalPlace) {
              grid->placeHerd(unplacedHerds[k], i, j);
              unplacedHerds[k]->setLocX(i);
              unplacedHerds[k]->setLocY(j);
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