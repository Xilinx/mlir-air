//===- AIRHerdPlacementPass.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Transform/AIRHerdPlacementPass.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::affine;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-place-herds"

namespace {

class Herd {

public:
  Herd(air::HerdOp herd, int32_t numRows, int32_t numCols, uint32_t number)
      : herdOp(herd), numRows(numRows), numCols(numCols), number(number) {
    size = numRows * numCols;
  }

  HerdOp getHerdOp() const { return herdOp; }
  int32_t getNumRows() const { return numRows; }
  int32_t getNumCols() const { return numCols; }
  std::string getName() const {
    std::string name = "herd";
    if (auto attr =
            herdOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      name = attr.getValue().str();
    return name;
  }
  uint32_t getNumber() const { return number; }
  int32_t getLocX() const { return locX; }
  int32_t getLocY() const { return locY; }
  int32_t getSize() const { return size; }

  void setLocX(int32_t x) { locX = x; }
  void setLocY(int32_t y) { locY = y; }

  void printHerd() const {
    llvm::outs() << "name: " << getName() << ", numRows: " << numRows
                 << ", number: " << number << ", numCols: " << numCols
                 << ", x_loc: " << locX << ", y_loc: " << locY
                 << ", size: " << size << "\n";
  }

private:
  air::HerdOp herdOp;
  int32_t numRows;
  int32_t numCols;
  uint32_t number;
  int32_t size;

  int32_t locX = -1;
  int32_t locY = -1;
};

class Segment {

public:
  Segment(int numRows, int numCols, int anchorPointRow, int anchorPointCol)
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

  void printSegment() const {
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

    if (clNumRows < 0 || clNumCols < 0 || clAnchorPointRow < 0 ||
        clAnchorPointCol < 0) {
      llvm::errs() << "Ensure all input parameters are greater than zero.\n";
      return;
    }

    auto module = getOperation();

    // Place herds in segments
    module.walk([&](air::SegmentOp part) {
      std::vector<std::unique_ptr<Herd>> segmentHerds;
      part.walk([&](air::HerdOp herd) {
        auto herd_size_x = herd.getNumCols();
        auto herd_size_y = herd.getNumRows();
        auto number = segmentHerds.size();
        auto herdPtr =
            std::make_unique<Herd>(herd, herd_size_y, herd_size_x, number);
        segmentHerds.push_back(std::move(herdPtr));
      });

      // If the size and offset attributes of the segment op are set then use
      // them. Otherwise use the values from the command line.
      auto num_rows_op = part.getNumRows();
      auto num_cols_op = part.getNumCols();
      auto row_offset_op = part.getRowOffset();
      auto col_offset_op = part.getColOffset();

      auto num_rows = num_rows_op ? *num_rows_op : clNumRows;
      auto num_cols = num_cols_op ? *num_cols_op : clNumCols;
      auto row_offset = row_offset_op ? *row_offset_op : clAnchorPointRow;
      auto col_offset = col_offset_op ? *col_offset_op : clAnchorPointCol;
      auto segment =
          std::make_unique<Segment>(num_rows, num_cols, row_offset, col_offset);
      placeHerdsInSegment(segmentHerds, segment);

      auto intTy = IntegerType::get(part->getContext(), 64);
      part->setAttr(part.getRowOffsetAttrName(),
                    IntegerAttr::get(intTy, row_offset));
      part->setAttr(part.getColOffsetAttrName(),
                    IntegerAttr::get(intTy, col_offset));
      part->setAttr(part.getNumRowsAttrName(),
                    IntegerAttr::get(intTy, num_rows));
      part->setAttr(part.getNumColsAttrName(),
                    IntegerAttr::get(intTy, num_cols));
    });

    module.walk([&](func::FuncOp f) {
      f.walk([&](air::HerdOp herd) {
        std::vector<std::unique_ptr<Herd>> unplacedHerds;

        // Place herds not in segments
        std::unique_ptr<Segment> segment = std::make_unique<Segment>(
            clNumRows, clNumCols, clAnchorPointRow, clAnchorPointCol);

        if (herd->getParentOfType<air::SegmentOp>())
          return;

        // Any pre-placed herds are assumed to be outside if the area being used
        // by the placement pass.
        if (herd.getRowOffset() && herd.getColOffset())
          return;

        auto herd_size_x = herd.getNumCols();
        auto herd_size_y = herd.getNumRows();
        auto number = unplacedHerds.size();
        std::unique_ptr<Herd> herdPtr =
            std::make_unique<Herd>(herd, herd_size_y, herd_size_x, number);
        unplacedHerds.push_back(std::move(herdPtr));

        placeHerdsInSegment(unplacedHerds, segment);
      });
    });
    return;
  }

private:
  void placeHerdsInSegment(std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                           std::unique_ptr<Segment> &segment) {

    std::sort(
        unplacedHerds.begin(), unplacedHerds.end(),
        [](const std::unique_ptr<Herd> &l, const std::unique_ptr<Herd> &r) {
          return l->getSize() > r->getSize();
        });

    std::vector<std::unique_ptr<Herd>> placedHerds;
    naivePlacement(segment, unplacedHerds, placedHerds);

    if (unplacedHerds.size() != 0) {
      getOperation().emitError("No valid placement found.");
      for (uint32_t i = 0; i < unplacedHerds.size(); i++) {
        unplacedHerds[i]->getHerdOp()->emitOpError("\nUnplaced herd: ")
            << unplacedHerds[i]->getName() << "\n";
      }
      return;
    }

    auto xLocName = xilinx::air::HerdOp::getColOffsetAttrName();
    auto yLocName = xilinx::air::HerdOp::getRowOffsetAttrName();

    for (auto &herd : placedHerds) {
      auto herdOp = herd->getHerdOp();
      herdOp->setAttr(
          yLocName,
          IntegerAttr::get(IntegerType::get(herdOp->getContext(), 64),
                           herd->getLocY() + segment->getAnchorPointRow()));
      herdOp->setAttr(
          xLocName,
          IntegerAttr::get(IntegerType::get(herdOp->getContext(), 64),
                           herd->getLocX() + segment->getAnchorPointCol()));
    }
  }

  // Performs placement, trying to place the first herd on the anchor point
  // first, moving from left -> right, up a row, then left -> right again. Will
  // try to place each remaining unplaced herd in each open segment tile.
  void naivePlacement(std::unique_ptr<Segment> &segment,
                      std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                      std::vector<std::unique_ptr<Herd>> &placedHerds) {
    for (int64_t i = 0; i < segment->getNumRows(); i++) {
      for (int64_t j = 0; j < segment->getNumCols(); j++) {
        if (segment->grid[segment->getNumRows() - i - 1][j] == -1) {
          for (uint32_t k = 0; k < unplacedHerds.size(); k++) {
            bool legalPlace = segment->isLegalPlacement(unplacedHerds[k], i, j);
            if (legalPlace) {
              segment->placeHerd(unplacedHerds[k], i, j);
              unplacedHerds[k]->setLocX(j);
              unplacedHerds[k]->setLocY(i);
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
