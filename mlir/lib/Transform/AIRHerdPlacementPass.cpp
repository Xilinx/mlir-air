//===- AIRHerdPlacementPass.cpp ---------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

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
#include <set>
#include <string>
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "air-place-herds"

namespace xilinx {
namespace air {

// Represents a cascade connection between two herds
struct CascadeConnection {
  std::string producerHerdName;
  std::string consumerHerdName;
  air::ChannelOp channelOp;
};

// Represents a shared L1 memref connection between two herds
struct SharedL1Connection {
  std::string herd1Name;
  std::string herd2Name;
  Value sharedMemref;
};

class Herd {

public:
  Herd(air::HerdOp herd, int32_t numRows, int32_t numCols, uint32_t number) {
    this->herdOps.push_back(herd);
    this->numRows = numRows;
    this->numCols = numCols;
    this->number = number;
    this->size = numRows * numCols;
  }

  std::vector<air::HerdOp> getHerdOps() const { return herdOps; }
  air::HerdOp getHerdOp(int i) const { return herdOps[i]; }
  void addHerdOp(air::HerdOp h) { herdOps.push_back(h); }
  int32_t getNumRows() const { return numRows; }
  int32_t getNumCols() const { return numCols; }
  std::string getName(int i) const {
    std::string name = "herd";
    if (auto attr = herdOps[i]->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      name = attr.getValue().str();
    return name;
  }
  uint32_t getNumber() const { return number; }
  int32_t getLocX() const { return locX; }
  int32_t getLocY() const { return locY; }
  int32_t getSize() const { return size; }

  void setLocX(int32_t x) { locX = x; }
  void setLocY(int32_t y) { locY = y; }

  void printHerd(int i) const {
    llvm::outs() << "name: " << getName(i) << ", numRows: " << numRows
                 << ", number: " << number << ", numCols: " << numCols
                 << ", x_loc: " << locX << ", y_loc: " << locY
                 << ", size: " << size << "\n";
  }

private:
  std::vector<air::HerdOp> herdOps;
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
    : public xilinx::air::impl::AIRHerdPlacementPassBase<AIRHerdPlacementPass> {

public:
  AIRHerdPlacementPass() = default;
  AIRHerdPlacementPass(const AIRHerdPlacementPass &pass) {}
  AIRHerdPlacementPass(const AIRHerdPlacementPassOptions &options)
      : AIRHerdPlacementPassBase(options) {}

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
      std::vector<CascadeConnection> cascadeConnections;
      std::vector<SharedL1Connection> sharedL1Connections;

      // Collect herds
      part.walk([&](air::HerdOp herd) {
        std::string name;
        if (auto attr = herd->getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName())) {
          name = attr.getValue().str();
          for (auto &h : segmentHerds) {
            if (name == h->getName(0)) {
              // Found herds sharing the same symbolic name
              h->addHerdOp(herd);
              return;
            }
          }
        }
        auto herd_size_x = herd.getNumCols();
        auto herd_size_y = herd.getNumRows();
        auto number = segmentHerds.size();
        auto herdPtr =
            std::make_unique<Herd>(herd, herd_size_y, herd_size_x, number);
        segmentHerds.push_back(std::move(herdPtr));
      });

      // Analyze cascade channel connections
      analyzeCascadeConnections(part, cascadeConnections);

      // Analyze shared L1 memref connections
      analyzeSharedL1Connections(part, sharedL1Connections);

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

      placeHerdsInSegment(segmentHerds, segment, cascadeConnections,
                          sharedL1Connections);

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

        std::vector<CascadeConnection> emptyCascadeConnections;
        placeHerdsInSegment(unplacedHerds, segment, emptyCascadeConnections);
      });
    });
    return;
  }

private:
  // Get herd name from a herd op
  std::string getHerdName(air::HerdOp herd) {
    if (auto attr =
            herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return attr.getValue().str();
    return "";
  }

  // Find the herd that contains a given channel put/get operation
  std::string findParentHerdName(Operation *op) {
    if (auto herd = op->getParentOfType<air::HerdOp>()) {
      return getHerdName(herd);
    }
    return "";
  }

  // Analyze shared L1 memref connections within a segment
  // Detects when the same L1 memref (memory space 2) is accessed by multiple
  // herds
  void analyzeSharedL1Connections(
      air::SegmentOp segment,
      std::vector<SharedL1Connection> &sharedL1Connections) {

    // Map: L1 memref value -> set of herds that access it
    DenseMap<Value, std::set<std::string>> memrefToHerds;

    // Walk through all herds and collect their L1 memref arguments
    segment.walk([&](air::HerdOp herd) {
      std::string herdName = getHerdName(herd);
      if (herdName.empty())
        return;

      // Check herd's kernel arguments for L1 memrefs
      // The herd operation has args that map segment-level values to herd
      // arguments
      for (auto arg : herd.getKernelOperands()) {
        auto memrefType = dyn_cast_if_present<MemRefType>(arg.getType());
        if (!memrefType)
          continue;

        // Check if memory space is L1
        if (air::isL1(memrefType)) {
          memrefToHerds[arg].insert(herdName);
          LLVM_DEBUG(llvm::dbgs() << "Found L1 memref accessed by herd "
                                  << herdName << "\n");
        }
      }
    });

    // Create connections for memrefs shared by multiple herds
    for (auto &entry : memrefToHerds) {
      Value memref = entry.first;
      const std::set<std::string> &herds = entry.second;

      if (herds.size() > 1) {
        // Create connections for all pairs of herds sharing this memref
        std::vector<std::string> herdList(herds.begin(), herds.end());
        for (size_t i = 0; i < herdList.size(); i++) {
          for (size_t j = i + 1; j < herdList.size(); j++) {
            SharedL1Connection conn;
            conn.herd1Name = herdList[i];
            conn.herd2Name = herdList[j];
            conn.sharedMemref = memref;
            sharedL1Connections.push_back(conn);

            LLVM_DEBUG(llvm::dbgs()
                       << "Found shared L1 connection: " << herdList[i]
                       << " <-> " << herdList[j] << "\n");
          }
        }
      }
    }
  }

  // Analyze cascade channel connections within a segment
  void analyzeCascadeConnections(
      air::SegmentOp segment,
      std::vector<CascadeConnection> &cascadeConnections) {

    // Find all cascade channels in the module
    auto module = segment->getParentOfType<ModuleOp>();
    if (!module)
      return;

    // Collect cascade channel declarations
    std::map<StringRef, air::ChannelOp> cascadeChannels;
    module.walk([&](air::ChannelOp channelOp) {
      if (channelOp.getChannelType() == "npu_cascade") {
        cascadeChannels[channelOp.getSymName()] = channelOp;
      }
    });

    if (cascadeChannels.empty())
      return;

    // For each cascade channel, find the producer (put) and consumer (get)
    // herds
    for (auto &entry : cascadeChannels) {
      StringRef channelName = entry.first;
      air::ChannelOp channelOp = entry.second;
      std::set<std::string> producerHerds;
      std::set<std::string> consumerHerds;

      // Walk through the segment to find channel put/get operations
      segment.walk([&, channelName](air::ChannelPutOp putOp) {
        if (putOp.getChanName() == channelName) {
          std::string herdName = findParentHerdName(putOp);
          if (!herdName.empty()) {
            producerHerds.insert(herdName);
          }
        }
      });

      segment.walk([&, channelName](air::ChannelGetOp getOp) {
        if (getOp.getChanName() == channelName) {
          std::string herdName = findParentHerdName(getOp);
          if (!herdName.empty()) {
            consumerHerds.insert(herdName);
          }
        }
      });

      // One CascadeConnection per channel-level (producer, consumer) edge.
      // Downstream maps (herdToProducers / herdToConsumers) dedupe by
      // collecting into std::set, so multiple channels between the same
      // herd pair (e.g. Q+K+V from rope to attn) don't false-positive
      // multi-producer detection in neighborAwarePlacement.
      for (const auto &producer : producerHerds) {
        for (const auto &consumer : consumerHerds) {
          if (producer == consumer)
            continue;
          CascadeConnection conn;
          conn.producerHerdName = producer;
          conn.consumerHerdName = consumer;
          conn.channelOp = channelOp;
          cascadeConnections.push_back(conn);

          LLVM_DEBUG(llvm::dbgs()
                     << "Found cascade connection: " << producer << " -> "
                     << consumer << " via channel " << channelName << "\n");
        }
      }
    }
  }

  // Find herd index by name in the herds vector
  int findHerdIdxByName(std::vector<std::unique_ptr<Herd>> &herds,
                        const std::string &name) {
    for (uint32_t i = 0; i < herds.size(); i++) {
      if (herds[i]->getName(0) == name) {
        return i;
      }
    }
    return -1;
  }

  // Check if two herds are cascade neighbors (west-to-east or north-to-south)
  bool areCascadeNeighbors(Herd *producer, Herd *consumer) {
    if (producer->getLocX() < 0 || consumer->getLocX() < 0)
      return false;

    int32_t prodX = producer->getLocX();
    int32_t prodY = producer->getLocY();
    int32_t consX = consumer->getLocX();
    int32_t consY = consumer->getLocY();

    // West-to-east: producer's right edge touches consumer's left edge
    bool westToEast =
        (prodX + producer->getNumCols() == consX) && (prodY == consY);

    // North-to-south: producer's bottom edge touches consumer's top edge
    bool northToSouth =
        (prodX == consX) && (prodY == consY + consumer->getNumRows());

    return westToEast || northToSouth;
  }

  // Read the (x_loc, y_loc) pin off a Herd by inspecting every wrapped
  // air.herd op. Returns std::nullopt if no pin is set. If sibling ops
  // (multiple HerdOps sharing one symbolic name) carry mismatched pins
  // an op-error is emitted on the second op and std::nullopt is returned.
  // Coordinates are kept in int64_t to match the IntegerAttr width.
  std::optional<std::pair<int64_t, int64_t>>
  getHerdPin(const std::unique_ptr<Herd> &h) {
    std::optional<std::pair<int64_t, int64_t>> pin;
    for (air::HerdOp op : h->getHerdOps()) {
      auto col = op.getColOffset();
      auto row = op.getRowOffset();
      if (!col || !row) {
        if (pin) {
          op->emitOpError("disagrees with sibling air.herd '")
              << h->getName(0)
              << "': one carries x_loc/y_loc, another does not";
          return std::nullopt;
        }
        continue;
      }
      std::pair<int64_t, int64_t> here{static_cast<int64_t>(*col),
                                       static_cast<int64_t>(*row)};
      if (pin && *pin != here) {
        op->emitOpError("disagrees with sibling air.herd '")
            << h->getName(0) << "': x_loc/y_loc (" << here.first << ", "
            << here.second << ") vs (" << pin->first << ", " << pin->second
            << ")";
        return std::nullopt;
      }
      pin = here;
    }
    return pin;
  }

  // Pre-place any user-pinned herds (those with both x_loc and y_loc set)
  // before the placer runs. Pinned herds are moved from `unplaced` to
  // `placed` and their cells marked occupied so subsequent placement
  // routes around them. If a pin is illegal (out of segment bounds or
  // overlapping an already-pinned cell), a warning is emitted naming the
  // herd, the original physical coordinates, and the segment extent, and
  // the herd falls through to the regular placer.
  void extractPinnedHerds(std::unique_ptr<Segment> &segment,
                          std::vector<std::unique_ptr<Herd>> &unplaced,
                          std::vector<std::unique_ptr<Herd>> &placed) {
    std::vector<std::unique_ptr<Herd>> remaining;
    remaining.reserve(unplaced.size());
    const int32_t anchorCol = segment->getAnchorPointCol();
    const int32_t anchorRow = segment->getAnchorPointRow();
    const int32_t segCols = segment->getNumCols();
    const int32_t segRows = segment->getNumRows();
    for (auto &h : unplaced) {
      auto pin = getHerdPin(h);
      if (!pin) {
        remaining.push_back(std::move(h));
        continue;
      }
      const int64_t physX = pin->first;
      const int64_t physY = pin->second;
      const int64_t relX = physX - anchorCol;
      const int64_t relY = physY - anchorRow;

      // Bounds check before narrowing so out-of-range pins are reported
      // distinctly from overlaps with already-placed pins.
      auto reportIllegal = [&](StringRef reason) {
        h->getHerdOp(0)->emitWarning()
            << "ignoring user-pinned x_loc/y_loc (" << physX << ", " << physY
            << ") on air.herd '" << h->getName(0) << "': " << reason
            << " (segment anchor (" << anchorCol << ", " << anchorRow
            << "), extent " << segCols << "x" << segRows << "); "
            << "falling back to automatic placement";
      };
      if (relX < 0 || relY < 0 || relX + h->getNumCols() > segCols ||
          relY + h->getNumRows() > segRows) {
        reportIllegal("position is outside the segment");
        remaining.push_back(std::move(h));
        continue;
      }
      const int32_t pinX = static_cast<int32_t>(relX);
      const int32_t pinY = static_cast<int32_t>(relY);
      if (!segment->isLegalPlacement(h, pinY, pinX)) {
        reportIllegal("position overlaps a previously-pinned herd");
        remaining.push_back(std::move(h));
        continue;
      }
      segment->placeHerd(h, pinY, pinX);
      h->setLocX(pinX);
      h->setLocY(pinY);
      placed.push_back(std::move(h));
    }
    unplaced = std::move(remaining);
  }

  void placeHerdsInSegment(
      std::vector<std::unique_ptr<Herd>> &unplacedHerds,
      std::unique_ptr<Segment> &segment,
      std::vector<CascadeConnection> &cascadeConnections,
      std::vector<SharedL1Connection> sharedL1Connections = {}) {

    std::vector<std::unique_ptr<Herd>> placedHerds;

    // Honor pre-set x_loc/y_loc attributes on herds: pinned herds are
    // claimed first so the neighbor-aware placer routes unpinned herds
    // around them. (The func-level path in runOnOperation() takes the
    // opposite stance and skips pinned herds entirely, on the assumption
    // that they live outside the placement region; segment-scoped pins
    // here are inside, so we reserve their cells.)
    extractPinnedHerds(segment, unplacedHerds, placedHerds);

    // If there are cascade or shared L1 connections, use neighbor-aware
    // placement
    if (!cascadeConnections.empty() || !sharedL1Connections.empty()) {
      neighborAwarePlacement(segment, unplacedHerds, placedHerds,
                             cascadeConnections, sharedL1Connections);
    } else {
      // Sort by size (largest first) for naive placement
      std::sort(
          unplacedHerds.begin(), unplacedHerds.end(),
          [](const std::unique_ptr<Herd> &l, const std::unique_ptr<Herd> &r) {
            return l->getSize() > r->getSize();
          });
      naivePlacement(segment, unplacedHerds, placedHerds);
    }

    if (unplacedHerds.size() != 0) {
      getOperation().emitError("No valid placement found.");
      for (uint32_t i = 0; i < unplacedHerds.size(); i++) {
        for (uint32_t j = 0; j < unplacedHerds[i]->getHerdOps().size(); j++)
          unplacedHerds[i]->getHerdOp(j)->emitOpError("\nUnplaced herd: ")
              << unplacedHerds[i]->getName(j) << "\n";
      }
      return;
    }

    auto xLocName = xilinx::air::HerdOp::getColOffsetAttrName();
    auto yLocName = xilinx::air::HerdOp::getRowOffsetAttrName();

    for (auto &herd : placedHerds) {
      for (auto herdOp : herd->getHerdOps()) {
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
  }

  // Build a topological order for herds based on cascade dependencies
  // Returns herds ordered such that producers come before consumers
  std::vector<std::string> buildCascadeTopologicalOrder(
      std::vector<std::unique_ptr<Herd>> &herds,
      std::vector<CascadeConnection> &cascadeConnections) {
    // Adjacency uses set<string> so multi-channel pairs collapse to one edge.
    std::map<std::string, std::set<std::string>> producerToConsumers;
    std::set<std::string> cascadeConnected;
    for (const auto &conn : cascadeConnections) {
      producerToConsumers[conn.producerHerdName].insert(conn.consumerHerdName);
      cascadeConnected.insert(conn.producerHerdName);
      cascadeConnected.insert(conn.consumerHerdName);
    }

    std::map<std::string, int> inDegree;
    for (auto &herd : herds)
      inDegree[herd->getName(0)] = 0;
    for (const auto &entry : producerToConsumers)
      for (const auto &consumer : entry.second)
        inDegree[consumer]++;

    // Seed Kahn's queue with cascade-connected roots only. Non-cascade
    // herds get appended at the end so they don't claim rows the cascade
    // chain needs to stack into.
    std::vector<std::string> order;
    std::vector<std::string> queue;
    for (const auto &name : cascadeConnected)
      if (inDegree[name] == 0)
        queue.push_back(name);

    while (!queue.empty()) {
      std::string current = queue.back();
      queue.pop_back();
      order.push_back(current);
      for (const auto &consumer : producerToConsumers[current]) {
        if (--inDegree[consumer] == 0)
          queue.push_back(consumer);
      }
    }

    // Tail: non-cascade herds, plus any cascade herds left behind by
    // a (malformed) cyclic cascade graph.
    for (auto &herd : herds) {
      std::string name = herd->getName(0);
      if (std::find(order.begin(), order.end(), name) == order.end())
        order.push_back(name);
    }

    return order;
  }

  // Check if two herds are neighbors (adjacent tiles)
  bool areNeighbors(Herd *herd1, Herd *herd2) {
    if (herd1->getLocX() < 0 || herd2->getLocX() < 0)
      return false;

    int32_t x1 = herd1->getLocX();
    int32_t y1 = herd1->getLocY();
    int32_t x2 = herd2->getLocX();
    int32_t y2 = herd2->getLocY();

    // Check if herd2 is east of herd1
    bool eastNeighbor = (x1 + herd1->getNumCols() == x2) && (y1 == y2);
    // Check if herd2 is west of herd1
    bool westNeighbor = (x2 + herd2->getNumCols() == x1) && (y1 == y2);
    // Check if herd2 is north of herd1
    bool northNeighbor = (x1 == x2) && (y1 == y2 + herd2->getNumRows());
    // Check if herd2 is south of herd1
    bool southNeighbor = (x1 == x2) && (y2 == y1 + herd1->getNumRows());

    return eastNeighbor || westNeighbor || northNeighbor || southNeighbor;
  }

  // Place a herd adjacent to ALL of its L1 neighbors (if possible)
  bool
  placeAdjacentToAllL1Neighbors(std::unique_ptr<Segment> &segment,
                                std::unique_ptr<Herd> &herd,
                                const std::vector<Herd *> &placedL1Neighbors) {

    if (placedL1Neighbors.empty()) {
      return false;
    }

    // Try to find a position that is adjacent to ALL placed L1 neighbors
    for (int64_t y = 0; y < segment->getNumRows(); y++) {
      for (int64_t x = 0; x < segment->getNumCols(); x++) {
        if (!segment->isLegalPlacement(herd, y, x)) {
          continue;
        }

        // Temporarily set location to check adjacency
        int32_t origX = herd->getLocX();
        int32_t origY = herd->getLocY();
        herd->setLocX(x);
        herd->setLocY(y);

        // Check if this position is adjacent to all placed L1 neighbors
        bool adjacentToAll = true;
        for (Herd *neighbor : placedL1Neighbors) {
          if (!areNeighbors(herd.get(), neighbor)) {
            adjacentToAll = false;
            break;
          }
        }

        // Reset location
        herd->setLocX(origX);
        herd->setLocY(origY);

        if (adjacentToAll) {
          segment->placeHerd(herd, y, x);
          herd->setLocX(x);
          herd->setLocY(y);
          LLVM_DEBUG(llvm::dbgs()
                     << "Placed " << herd->getName(0) << " at (" << x << ", "
                     << y << ") adjacent to all " << placedL1Neighbors.size()
                     << " L1 neighbors\n");
          return true;
        }
      }
    }

    return false;
  }

  // Place a herd adjacent to another herd (any direction)
  bool placeAdjacentToHerd(std::unique_ptr<Segment> &segment,
                           std::unique_ptr<Herd> &herd, Herd *neighbor) {
    int32_t neighX = neighbor->getLocX();
    int32_t neighY = neighbor->getLocY();

    // Try east
    int32_t candidateX = neighX + neighbor->getNumCols();
    int32_t candidateY = neighY;
    if (segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " east of "
                              << neighbor->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }

    // Try west
    candidateX = neighX - herd->getNumCols();
    candidateY = neighY;
    if (candidateX >= 0 &&
        segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " west of "
                              << neighbor->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }

    // Try north
    candidateX = neighX;
    candidateY = neighY + neighbor->getNumRows();
    if (candidateY < segment->getNumRows() &&
        segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " north of "
                              << neighbor->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }

    // Try south
    candidateX = neighX;
    candidateY = neighY - herd->getNumRows();
    if (candidateY >= 0 &&
        segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " south of "
                              << neighbor->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }

    return false;
  }

  // Extent (rows south or cols east) needed to stack the longest cascade
  // chain rooted at each herd, excluding the herd itself. `herdExtent`
  // supplies the per-herd dimension (numRows for south stacking, numCols
  // for east stacking) summed along the longest path.
  // `visiting` guards malformed cyclic cascade graphs from infinite recursion.
  std::map<std::string, int> computeCascadeChainExtent(
      const std::map<std::string, std::set<std::string>> &herdToConsumers,
      const std::map<std::string, int> &herdExtent) {
    std::map<std::string, int> extent;
    std::set<std::string> visiting;
    std::function<int(const std::string &)> dfs =
        [&](const std::string &name) -> int {
      auto it = extent.find(name);
      if (it != extent.end())
        return it->second;
      if (!visiting.insert(name).second)
        return 0; // cycle: break recursion
      int maxChild = 0;
      auto consIt = herdToConsumers.find(name);
      if (consIt != herdToConsumers.end()) {
        for (const auto &c : consIt->second) {
          auto hIt = herdExtent.find(c);
          int childExtent = (hIt != herdExtent.end()) ? hIt->second : 1;
          maxChild = std::max(maxChild, dfs(c) + childExtent);
        }
      }
      visiting.erase(name);
      extent[name] = maxChild;
      return maxChild;
    };
    for (const auto &entry : herdToConsumers)
      dfs(entry.first);
    return extent;
  }

  // Neighbor-aware placement algorithm that handles both cascade and shared L1
  // connections
  void
  neighborAwarePlacement(std::unique_ptr<Segment> &segment,
                         std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                         std::vector<std::unique_ptr<Herd>> &placedHerds,
                         std::vector<CascadeConnection> &cascadeConnections,
                         std::vector<SharedL1Connection> &sharedL1Connections) {

    LLVM_DEBUG(llvm::dbgs()
               << "Starting neighbor-aware placement with "
               << cascadeConnections.size() << " cascade connections and "
               << sharedL1Connections.size() << " shared L1 connections\n");

    // Build maps for cascade relationships (directional: producer -> consumer).
    // set<string> collapses multi-channel pairs (Q+K+V from rope to attn) to
    // a single edge so multi-producer detection below isn't false-positive.
    std::map<std::string, std::set<std::string>> herdToConsumers;
    std::map<std::string, std::set<std::string>> herdToProducers;

    for (const auto &conn : cascadeConnections) {
      herdToConsumers[conn.producerHerdName].insert(conn.consumerHerdName);
      herdToProducers[conn.consumerHerdName].insert(conn.producerHerdName);
    }

    // Extent needed for the longest cascade chain rooted at each herd, in
    // both directions: south rows for north-to-south chains and east cols
    // for west-to-east chains.
    std::map<std::string, int> herdHeight;
    std::map<std::string, int> herdWidth;
    for (auto &h : unplacedHerds) {
      herdHeight[h->getName(0)] = h->getNumRows();
      herdWidth[h->getName(0)] = h->getNumCols();
    }
    auto chainSouthRows =
        computeCascadeChainExtent(herdToConsumers, herdHeight);
    auto chainEastCols = computeCascadeChainExtent(herdToConsumers, herdWidth);

    // Build map for shared L1 relationships (bidirectional)
    std::map<std::string, std::set<std::string>> herdToL1Neighbors;
    for (const auto &conn : sharedL1Connections) {
      herdToL1Neighbors[conn.herd1Name].insert(conn.herd2Name);
      herdToL1Neighbors[conn.herd2Name].insert(conn.herd1Name);
    }

    // Find consumers with multiple producers - these need coordinated placement
    std::set<std::string> multiProducerConsumers;
    for (const auto &entry : herdToProducers) {
      if (entry.second.size() > 1) {
        multiProducerConsumers.insert(entry.first);
        LLVM_DEBUG(llvm::dbgs() << "Consumer " << entry.first << " has "
                                << entry.second.size() << " producers\n");
      }
    }

    // Find herds that have multiple L1 neighbors - they need coordinated
    // placement
    std::set<std::string> multiL1NeighborHerds;
    for (const auto &entry : herdToL1Neighbors) {
      if (entry.second.size() > 1) {
        multiL1NeighborHerds.insert(entry.first);
        LLVM_DEBUG(llvm::dbgs() << "Herd " << entry.first << " has "
                                << entry.second.size() << " L1 neighbors\n");
      }
    }

    // Place herds in topological order (based on cascade connections)
    auto topOrder =
        buildCascadeTopologicalOrder(unplacedHerds, cascadeConnections);

    LLVM_DEBUG({
      llvm::dbgs() << "Topological order: ";
      for (const auto &name : topOrder) {
        llvm::dbgs() << name << " ";
      }
      llvm::dbgs() << "\n";
    });

    // Track planned consumer positions for coordinating multi-producer
    // placement
    std::map<std::string, std::pair<int32_t, int32_t>> plannedConsumerPositions;

    // Place each herd in topological order
    for (const auto &herdName : topOrder) {
      int herdIdx = findHerdIdxByName(unplacedHerds, herdName);
      if (herdIdx < 0)
        continue;

      auto &herd = unplacedHerds[herdIdx];
      bool placed = false;

      // Check if this herd is a consumer with multiple producers (cascade)
      auto prodIt = herdToProducers.find(herdName);
      if (prodIt != herdToProducers.end() && prodIt->second.size() > 1) {
        placed = placeConsumerWithMultipleProducers(segment, herd, placedHerds,
                                                    prodIt->second);
      }

      // Check if this herd has a single placed producer (cascade)
      if (!placed && prodIt != herdToProducers.end()) {
        for (const auto &producerName : prodIt->second) {
          Herd *placedProducer = findPlacedHerd(placedHerds, producerName);
          if (placedProducer) {
            placed = placeAdjacentToProducer(segment, herd, placedProducer);
            break;
          }
        }
      }

      // Check if this herd shares L1 with a herd that has multiple L1 neighbors
      // (coordinate placement so the multi-neighbor herd can be adjacent to
      // all) This must be checked BEFORE regular adjacent placement
      if (!placed) {
        auto l1It = herdToL1Neighbors.find(herdName);
        if (l1It != herdToL1Neighbors.end()) {
          for (const auto &neighborName : l1It->second) {
            if (multiL1NeighborHerds.count(neighborName)) {
              placed = placeForMultiL1NeighborHerd(
                  segment, herd, herdName, neighborName,
                  herdToL1Neighbors[neighborName], placedHerds,
                  plannedConsumerPositions, unplacedHerds);
              if (placed)
                break;
            }
          }
        }
      }

      // Check if this herd shares L1 memory with any placed herd
      if (!placed) {
        auto l1It = herdToL1Neighbors.find(herdName);
        if (l1It != herdToL1Neighbors.end()) {
          // Collect all placed L1 neighbors
          std::vector<Herd *> placedL1Neighbors;
          for (const auto &neighborName : l1It->second) {
            Herd *placedNeighbor = findPlacedHerd(placedHerds, neighborName);
            if (placedNeighbor) {
              placedL1Neighbors.push_back(placedNeighbor);
            }
          }

          if (!placedL1Neighbors.empty()) {
            // Try to place adjacent to ALL L1 neighbors first
            if (placedL1Neighbors.size() > 1) {
              placed = placeAdjacentToAllL1Neighbors(segment, herd,
                                                     placedL1Neighbors);
              if (placed) {
                LLVM_DEBUG(llvm::dbgs()
                           << "Placed " << herdName << " adjacent to all "
                           << placedL1Neighbors.size() << " L1 neighbors\n");
              }
            }

            // Fallback: place adjacent to at least one L1 neighbor
            if (!placed) {
              for (Herd *placedNeighbor : placedL1Neighbors) {
                placed = placeAdjacentToHerd(segment, herd, placedNeighbor);
                if (placed) {
                  LLVM_DEBUG(llvm::dbgs()
                             << "Placed " << herdName << " adjacent to "
                             << placedNeighbor->getName(0)
                             << " (shared L1, partial)\n");
                  break;
                }
              }
            }
          }
        }
      }

      // Check if this producer's consumer has multiple producers (cascade)
      if (!placed) {
        auto consIt = herdToConsumers.find(herdName);
        if (consIt != herdToConsumers.end()) {
          for (const auto &consumerName : consIt->second) {
            if (multiProducerConsumers.count(consumerName)) {
              placed = placeProducerForMultiConsumer(
                  segment, herd, herdName, consumerName,
                  herdToProducers[consumerName], placedHerds,
                  plannedConsumerPositions, unplacedHerds);
              if (placed)
                break;
            }
          }
        }
      }

      // Fallback: standard placement with neighbor awareness
      if (!placed) {
        auto consIt = herdToConsumers.find(herdName);
        bool hasConsumers =
            consIt != herdToConsumers.end() && !consIt->second.empty();
        auto l1It = herdToL1Neighbors.find(herdName);
        bool hasL1Neighbors =
            l1It != herdToL1Neighbors.end() && !l1It->second.empty();

        // Cascade producers stack south when both ends are multi-column
        // (north-to-south chain) and east when both ends are multi-row
        // (west-to-east chain). Per-tile cascade adjacency requires room
        // for the rest of the chain in that direction.
        bool needsRoomSouth = false;
        bool needsRoomEast = false;
        if (hasConsumers) {
          for (const auto &consumerName : consIt->second) {
            int consIdx = findHerdIdxByName(unplacedHerds, consumerName);
            if (consIdx < 0)
              continue;
            auto &cons = unplacedHerds[consIdx];
            if (herd->getNumCols() > 1 && cons->getNumCols() > 1)
              needsRoomSouth = true;
            if (herd->getNumRows() > 1 && cons->getNumRows() > 1)
              needsRoomEast = true;
          }
        }

        // Reserve enough extent in the chain direction for the rest of it.
        int requiredSouthRows = 1;
        if (needsRoomSouth) {
          auto rowsIt = chainSouthRows.find(herdName);
          if (rowsIt != chainSouthRows.end() && rowsIt->second > 0)
            requiredSouthRows = rowsIt->second;
        }
        int requiredEastCols = 1;
        if (needsRoomEast) {
          auto colsIt = chainEastCols.find(herdName);
          if (colsIt != chainEastCols.end() && colsIt->second > 0)
            requiredEastCols = colsIt->second;
        }

        for (int64_t i = 0; i < segment->getNumRows() && !placed; i++) {
          for (int64_t j = 0; j < segment->getNumCols() && !placed; j++) {
            if (segment->grid[segment->getNumRows() - i - 1][j] == -1) {
              if (segment->isLegalPlacement(herd, i, j)) {
                bool goodPosition = true;
                if (hasConsumers || hasL1Neighbors) {
                  // Ensure room for neighbor in any direction
                  bool roomEast = (j + herd->getNumCols() + requiredEastCols <=
                                   segment->getNumCols());
                  bool roomWest = (j > 0);
                  bool roomSouth = (i >= requiredSouthRows);
                  bool roomNorth =
                      (i + herd->getNumRows() < segment->getNumRows());
                  if (needsRoomSouth)
                    goodPosition = roomSouth;
                  else if (needsRoomEast)
                    goodPosition = roomEast;
                  else
                    goodPosition =
                        roomEast || roomWest || roomSouth || roomNorth;
                }
                if (goodPosition) {
                  segment->placeHerd(herd, i, j);
                  herd->setLocX(j);
                  herd->setLocY(i);
                  placed = true;
                  LLVM_DEBUG(llvm::dbgs()
                             << "Placed " << herdName << " at (" << j << ", "
                             << i << ") using fallback\n");
                }
              }
            }
          }
        }
      }

      if (placed) {
        placedHerds.push_back(std::move(unplacedHerds[herdIdx]));
        unplacedHerds.erase(unplacedHerds.begin() + herdIdx);
      }
    }

    // Place any remaining herds using naive placement
    if (!unplacedHerds.empty()) {
      naivePlacement(segment, unplacedHerds, placedHerds);
    }
  }

  // Find a placed herd by name
  Herd *findPlacedHerd(std::vector<std::unique_ptr<Herd>> &placedHerds,
                       const std::string &name) {
    for (auto &h : placedHerds) {
      if (h->getName(0) == name) {
        return h.get();
      }
    }
    return nullptr;
  }

  // Try placing a herd to the east of the producer.
  bool tryPlaceEast(std::unique_ptr<Segment> &segment,
                    std::unique_ptr<Herd> &herd, Herd *producer) {
    int32_t candidateX = producer->getLocX() + producer->getNumCols();
    int32_t candidateY = producer->getLocY();
    if (segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " east of "
                              << producer->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }
    return false;
  }

  // Try placing a herd to the south of the producer.
  bool tryPlaceSouth(std::unique_ptr<Segment> &segment,
                     std::unique_ptr<Herd> &herd, Herd *producer) {
    int32_t candidateX = producer->getLocX();
    int32_t candidateY = producer->getLocY() - herd->getNumRows();
    if (candidateY >= 0 &&
        segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " south of "
                              << producer->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }
    return false;
  }

  // Place a herd adjacent to its cascade producer. Cascade requires
  // per-tile adjacency at matching indices, so multi-column herds must
  // be stacked vertically (south) and multi-row herds side-by-side (east).
  bool placeAdjacentToProducer(std::unique_ptr<Segment> &segment,
                               std::unique_ptr<Herd> &herd, Herd *producer) {
    // When both herds span multiple columns, stacking south keeps
    // corresponding column indices adjacent (north-to-south cascade).
    // Otherwise default to east-first (west-to-east cascade).
    bool preferSouth = producer->getNumCols() > 1 && herd->getNumCols() > 1;
    if (preferSouth) {
      if (tryPlaceSouth(segment, herd, producer))
        return true;
      return tryPlaceEast(segment, herd, producer);
    }
    if (tryPlaceEast(segment, herd, producer))
      return true;
    return tryPlaceSouth(segment, herd, producer);
  }

  // Place a consumer that has multiple producers
  bool placeConsumerWithMultipleProducers(
      std::unique_ptr<Segment> &segment, std::unique_ptr<Herd> &consumer,
      std::vector<std::unique_ptr<Herd>> &placedHerds,
      const std::set<std::string> &producerNames) {

    // Collect all placed producers
    std::vector<Herd *> placedProducers;
    for (const auto &name : producerNames) {
      Herd *p = findPlacedHerd(placedHerds, name);
      if (p) {
        placedProducers.push_back(p);
      }
    }

    if (placedProducers.empty()) {
      return false;
    }

    // Try to find a position that is adjacent to ALL placed producers
    // For each potential position, check if it's a valid cascade neighbor
    for (int64_t y = 0; y < segment->getNumRows(); y++) {
      for (int64_t x = 0; x < segment->getNumCols(); x++) {
        if (!segment->isLegalPlacement(consumer, y, x)) {
          continue;
        }

        // Check if this position is adjacent to all placed producers
        bool adjacentToAll = true;
        for (Herd *prod : placedProducers) {
          int32_t prodX = prod->getLocX();
          int32_t prodY = prod->getLocY();

          // Check west-to-east: producer is immediately west
          bool westToEast = (prodX + prod->getNumCols() == x) && (prodY == y);
          // Check north-to-south: producer is immediately north
          bool northToSouth =
              (prodX == x) && (prodY == y + consumer->getNumRows());

          if (!westToEast && !northToSouth) {
            adjacentToAll = false;
            break;
          }
        }

        if (adjacentToAll) {
          segment->placeHerd(consumer, y, x);
          consumer->setLocX(x);
          consumer->setLocY(y);
          LLVM_DEBUG(llvm::dbgs()
                     << "Placed consumer " << consumer->getName(0) << " at ("
                     << x << ", " << y << ") adjacent to all "
                     << placedProducers.size() << " producers\n");
          return true;
        }
      }
    }

    // If we can't be adjacent to all, try being adjacent to at least one
    if (!placedProducers.empty()) {
      return placeAdjacentToProducer(segment, consumer, placedProducers[0]);
    }

    return false;
  }

  // Place a herd that shares L1 with a multi-L1-neighbor herd
  // Coordinate placement so the multi-neighbor herd can be adjacent to all
  bool placeForMultiL1NeighborHerd(
      std::unique_ptr<Segment> &segment, std::unique_ptr<Herd> &herd,
      const std::string &herdName, const std::string &multiNeighborHerdName,
      const std::set<std::string> &allL1Neighbors,
      std::vector<std::unique_ptr<Herd>> &placedHerds,
      std::map<std::string, std::pair<int32_t, int32_t>>
          &plannedCenterPositions,
      std::vector<std::unique_ptr<Herd>> &unplacedHerds) {

    // Find how many L1 neighbors for this multi-neighbor herd are already
    // placed
    std::vector<Herd *> alreadyPlacedNeighbors;
    for (const auto &name : allL1Neighbors) {
      Herd *p = findPlacedHerd(placedHerds, name);
      if (p) {
        alreadyPlacedNeighbors.push_back(p);
      }
    }

    // If no neighbors placed yet, this is the first one
    // Place it and plan where the multi-neighbor herd should go
    if (alreadyPlacedNeighbors.empty()) {
      // For a herd with 2 neighbors, we want them in an L-shape:
      //   N1 at (x, y) - first neighbor
      //   Center at (x, y+1) - multi-neighbor herd will go here
      //   N2 at (x+1, y+1) - second neighbor goes east of center
      //
      // This ensures center is adjacent to both N1 (north) and N2 (west)
      for (int64_t y = 0; y < segment->getNumRows() - 1; y++) {
        for (int64_t x = 0; x < segment->getNumCols() - 1; x++) {
          if (segment->isLegalPlacement(herd, y, x)) {
            // Plan center position north of this herd
            int32_t centerX = x;
            int32_t centerY = y + herd->getNumRows();

            // Verify there's room for center and another neighbor east of
            // center
            if (centerY < segment->getNumRows() &&
                centerX + 1 < segment->getNumCols()) {
              segment->placeHerd(herd, y, x);
              herd->setLocX(x);
              herd->setLocY(y);

              // Record planned center position
              plannedCenterPositions[multiNeighborHerdName] = {centerX,
                                                               centerY};

              LLVM_DEBUG(llvm::dbgs()
                         << "Placed first L1 neighbor " << herdName << " at ("
                         << x << ", " << y << "), planned center "
                         << multiNeighborHerdName << " at (" << centerX << ", "
                         << centerY << ")\n");
              return true;
            }
          }
        }
      }
    } else {
      // Other neighbors already placed, position this one relative to planned
      // center
      auto it = plannedCenterPositions.find(multiNeighborHerdName);
      if (it != plannedCenterPositions.end()) {
        int32_t centerX = it->second.first;
        int32_t centerY = it->second.second;

        // Look up the center (multi-neighbor) herd to get its actual dimensions
        int centerIdx = findHerdIdxByName(unplacedHerds, multiNeighborHerdName);
        if (centerIdx < 0) {
          // Center herd not found in unplacedHerds, may already be placed or
          // invalid
          return false;
        }
        auto &centerHerd = unplacedHerds[centerIdx];

        // Place this neighbor east of the planned center position
        // Use center herd's width to ensure neighbor's left edge touches
        // center's right edge
        int32_t neighX = centerX + centerHerd->getNumCols();
        int32_t neighY = centerY;

        if (neighX < segment->getNumCols() &&
            segment->isLegalPlacement(herd, neighY, neighX)) {
          segment->placeHerd(herd, neighY, neighX);
          herd->setLocX(neighX);
          herd->setLocY(neighY);
          LLVM_DEBUG(llvm::dbgs()
                     << "Placed subsequent L1 neighbor " << herdName << " at ("
                     << neighX << ", " << neighY
                     << ") east of planned center position\n");
          return true;
        }

        // Try north of center instead
        // Use center herd's height to ensure neighbor's bottom edge touches
        // center's top edge
        neighX = centerX;
        neighY = centerY + centerHerd->getNumRows();

        if (neighY < segment->getNumRows() &&
            segment->isLegalPlacement(herd, neighY, neighX)) {
          segment->placeHerd(herd, neighY, neighX);
          herd->setLocX(neighX);
          herd->setLocY(neighY);
          LLVM_DEBUG(llvm::dbgs()
                     << "Placed subsequent L1 neighbor " << herdName << " at ("
                     << neighX << ", " << neighY
                     << ") north of planned center position\n");
          return true;
        }
      }
    }

    return false;
  }

  // Place a producer that feeds a multi-producer consumer
  // Coordinate with other producers to ensure consumer can be adjacent to all
  bool placeProducerForMultiConsumer(
      std::unique_ptr<Segment> &segment, std::unique_ptr<Herd> &producer,
      const std::string &producerName, const std::string &consumerName,
      const std::set<std::string> &allProducerNames,
      std::vector<std::unique_ptr<Herd>> &placedHerds,
      std::map<std::string, std::pair<int32_t, int32_t>>
          &plannedConsumerPositions,
      std::vector<std::unique_ptr<Herd>> &unplacedHerds) {

    // Find how many producers for this consumer are already placed
    std::vector<Herd *> alreadyPlacedProducers;
    for (const auto &name : allProducerNames) {
      Herd *p = findPlacedHerd(placedHerds, name);
      if (p) {
        alreadyPlacedProducers.push_back(p);
      }
    }

    // If no producers placed yet, this is the first one
    // Place it and plan where the consumer should go
    if (alreadyPlacedProducers.empty()) {
      // Look up the consumer herd to get its actual dimensions
      int consumerIdx = findHerdIdxByName(unplacedHerds, consumerName);
      if (consumerIdx < 0) {
        // Consumer not found in unplacedHerds
        return false;
      }
      auto &consumer = unplacedHerds[consumerIdx];

      // Find the maximum height among other producers (for space planning)
      int32_t maxOtherProducerHeight = 1; // Default minimum
      for (const auto &otherProdName : allProducerNames) {
        if (otherProdName != producerName) {
          int otherIdx = findHerdIdxByName(unplacedHerds, otherProdName);
          if (otherIdx >= 0) {
            maxOtherProducerHeight = std::max(
                maxOtherProducerHeight, unplacedHerds[otherIdx]->getNumRows());
          }
        }
      }

      // Find a position that leaves room for consumer east and another producer
      // north. The second producer goes north of consumer, so we need room for:
      // consumer height + max other producer height above consY
      for (int64_t y = 1; y < segment->getNumRows();
           y++) { // Start at y=1 to leave room north
        for (int64_t x = 0; x < segment->getNumCols() - 1;
             x++) { // Leave room east
          if (segment->isLegalPlacement(producer, y, x)) {
            // Plan consumer position east of this producer
            int32_t consX = x + producer->getNumCols();
            int32_t consY = y;

            // Verify there's room for another producer north of consumer
            // position. The second producer's Y will be consY + consumer
            // height, and it needs maxOtherProducerHeight rows above that.
            int32_t requiredTopY =
                consY + consumer->getNumRows() + maxOtherProducerHeight;
            if (requiredTopY <= segment->getNumRows()) {
              segment->placeHerd(producer, y, x);
              producer->setLocX(x);
              producer->setLocY(y);

              // Record planned consumer position
              plannedConsumerPositions[consumerName] = {consX, consY};

              LLVM_DEBUG(llvm::dbgs()
                         << "Placed first producer " << producerName << " at ("
                         << x << ", " << y << "), planned consumer "
                         << consumerName << " at (" << consX << ", " << consY
                         << ")\n");
              return true;
            }
          }
        }
      }
    } else {
      // Other producers already placed, position this one relative to planned
      // consumer
      auto it = plannedConsumerPositions.find(consumerName);
      if (it != plannedConsumerPositions.end()) {
        int32_t consX = it->second.first;
        int32_t consY = it->second.second;

        // Look up the consumer herd to get its actual dimensions
        int consumerIdx = findHerdIdxByName(unplacedHerds, consumerName);
        if (consumerIdx < 0) {
          // Consumer not found in unplacedHerds, may already be placed or
          // invalid
          return false;
        }
        auto &consumer = unplacedHerds[consumerIdx];

        // Place this producer north of the planned consumer position
        // Use consumer's height to ensure producer's bottom edge touches
        // consumer's top edge
        int32_t prodX = consX;
        int32_t prodY = consY + consumer->getNumRows();

        if (prodY < segment->getNumRows() &&
            segment->isLegalPlacement(producer, prodY, prodX)) {
          segment->placeHerd(producer, prodY, prodX);
          producer->setLocX(prodX);
          producer->setLocY(prodY);
          LLVM_DEBUG(llvm::dbgs()
                     << "Placed subsequent producer " << producerName << " at ("
                     << prodX << ", " << prodY
                     << ") north of planned consumer position\n");
          return true;
        }
      }
    }

    return false;
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

} // end namespace air
} // end namespace xilinx

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdPlacementPass() {
  return std::make_unique<AIRHerdPlacementPass>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createAIRHerdPlacementPass(const AIRHerdPlacementPassOptions &options) {
  return std::make_unique<AIRHerdPlacementPass>(options);
}

} // namespace air
} // namespace xilinx
