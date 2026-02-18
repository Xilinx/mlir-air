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

      placeHerdsInSegment(segmentHerds, segment, cascadeConnections);

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
      if (channelOp.getChannelType() == "cascade") {
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

      // Create cascade connections for each producer-consumer pair
      for (const auto &producer : producerHerds) {
        for (const auto &consumer : consumerHerds) {
          if (producer != consumer) {
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

  void placeHerdsInSegment(std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                           std::unique_ptr<Segment> &segment,
                           std::vector<CascadeConnection> &cascadeConnections) {

    std::vector<std::unique_ptr<Herd>> placedHerds;

    // If there are cascade connections, use cascade-aware placement
    if (!cascadeConnections.empty()) {
      cascadeAwarePlacement(segment, unplacedHerds, placedHerds,
                            cascadeConnections);
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
    // Build adjacency list: producer -> consumers
    std::map<std::string, std::vector<std::string>> producerToConsumers;
    std::map<std::string, int> inDegree;
    std::set<std::string> allHerds;

    for (auto &herd : herds) {
      std::string name = herd->getName(0);
      allHerds.insert(name);
      inDegree[name] = 0;
    }

    for (const auto &conn : cascadeConnections) {
      producerToConsumers[conn.producerHerdName].push_back(
          conn.consumerHerdName);
      inDegree[conn.consumerHerdName]++;
    }

    // Kahn's algorithm for topological sort
    std::vector<std::string> order;
    std::vector<std::string> queue;

    for (const auto &name : allHerds) {
      if (inDegree[name] == 0) {
        queue.push_back(name);
      }
    }

    while (!queue.empty()) {
      std::string current = queue.back();
      queue.pop_back();
      order.push_back(current);

      for (const auto &consumer : producerToConsumers[current]) {
        inDegree[consumer]--;
        if (inDegree[consumer] == 0) {
          queue.push_back(consumer);
        }
      }
    }

    // Add any remaining herds not in cascade chains
    for (auto &herd : herds) {
      std::string name = herd->getName(0);
      if (std::find(order.begin(), order.end(), name) == order.end()) {
        order.push_back(name);
      }
    }

    return order;
  }

  // Cascade-aware placement algorithm
  void
  cascadeAwarePlacement(std::unique_ptr<Segment> &segment,
                        std::vector<std::unique_ptr<Herd>> &unplacedHerds,
                        std::vector<std::unique_ptr<Herd>> &placedHerds,
                        std::vector<CascadeConnection> &cascadeConnections) {

    LLVM_DEBUG(llvm::dbgs()
               << "Starting cascade-aware placement with "
               << cascadeConnections.size() << " cascade connections\n");

    // Build maps for cascade relationships
    std::map<std::string, std::vector<std::string>> herdToConsumers;
    std::map<std::string, std::vector<std::string>> herdToProducers;

    for (const auto &conn : cascadeConnections) {
      herdToConsumers[conn.producerHerdName].push_back(conn.consumerHerdName);
      herdToProducers[conn.consumerHerdName].push_back(conn.producerHerdName);
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

    // Place herds in topological order
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

      // Check if this herd is a consumer with multiple producers
      auto prodIt = herdToProducers.find(herdName);
      if (prodIt != herdToProducers.end() && prodIt->second.size() > 1) {
        // This is a consumer with multiple producers
        // Try to find a position where it can be adjacent to all placed
        // producers
        placed = placeConsumerWithMultipleProducers(segment, herd, placedHerds,
                                                    prodIt->second);
      }

      // Check if this herd has a single placed producer
      if (!placed && prodIt != herdToProducers.end()) {
        for (const auto &producerName : prodIt->second) {
          Herd *placedProducer = findPlacedHerd(placedHerds, producerName);
          if (placedProducer) {
            placed = placeAdjacentToProducer(segment, herd, placedProducer);
            break;
          }
        }
      }

      // Check if this producer's consumer has multiple producers
      // If so, coordinate placement with other producers
      if (!placed) {
        auto consIt = herdToConsumers.find(herdName);
        if (consIt != herdToConsumers.end()) {
          for (const auto &consumerName : consIt->second) {
            if (multiProducerConsumers.count(consumerName)) {
              // This producer feeds a multi-producer consumer
              // Plan a coordinated placement
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

      // Fallback: standard placement with cascade awareness
      if (!placed) {
        auto consIt = herdToConsumers.find(herdName);
        bool hasConsumers =
            consIt != herdToConsumers.end() && !consIt->second.empty();

        for (int64_t i = 0; i < segment->getNumRows() && !placed; i++) {
          for (int64_t j = 0; j < segment->getNumCols() && !placed; j++) {
            if (segment->grid[segment->getNumRows() - i - 1][j] == -1) {
              if (segment->isLegalPlacement(herd, i, j)) {
                bool goodPosition = true;
                if (hasConsumers) {
                  // Ensure room for consumer east or south
                  bool roomEast =
                      (j + herd->getNumCols() < segment->getNumCols());
                  bool roomSouth = (i > 0);
                  goodPosition = roomEast || roomSouth;
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

  // Place a herd adjacent to its producer (east or south)
  bool placeAdjacentToProducer(std::unique_ptr<Segment> &segment,
                               std::unique_ptr<Herd> &herd, Herd *producer) {
    int32_t prodX = producer->getLocX();
    int32_t prodY = producer->getLocY();

    // Try east (west-to-east cascade)
    int32_t candidateX = prodX + producer->getNumCols();
    int32_t candidateY = prodY;
    if (segment->isLegalPlacement(herd, candidateY, candidateX)) {
      segment->placeHerd(herd, candidateY, candidateX);
      herd->setLocX(candidateX);
      herd->setLocY(candidateY);
      LLVM_DEBUG(llvm::dbgs() << "Placed " << herd->getName(0) << " east of "
                              << producer->getName(0) << " at (" << candidateX
                              << ", " << candidateY << ")\n");
      return true;
    }

    // Try south (north-to-south cascade)
    candidateX = prodX;
    candidateY = prodY - herd->getNumRows();
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

  // Place a consumer that has multiple producers
  bool placeConsumerWithMultipleProducers(
      std::unique_ptr<Segment> &segment, std::unique_ptr<Herd> &consumer,
      std::vector<std::unique_ptr<Herd>> &placedHerds,
      const std::vector<std::string> &producerNames) {

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

  // Place a producer that feeds a multi-producer consumer
  // Coordinate with other producers to ensure consumer can be adjacent to all
  bool placeProducerForMultiConsumer(
      std::unique_ptr<Segment> &segment, std::unique_ptr<Herd> &producer,
      const std::string &producerName, const std::string &consumerName,
      const std::vector<std::string> &allProducerNames,
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
      // Find a position that leaves room for consumer east and another producer
      // north
      for (int64_t y = 1; y < segment->getNumRows();
           y++) { // Start at y=1 to leave room north
        for (int64_t x = 0; x < segment->getNumCols() - 1;
             x++) { // Leave room east
          if (segment->isLegalPlacement(producer, y, x)) {
            // Plan consumer position east of this producer
            int32_t consX = x + producer->getNumCols();
            int32_t consY = y;

            // Verify there's room for another producer north of consumer
            // position
            if (consY + 1 < segment->getNumRows()) {
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

        // Place this producer north of the planned consumer position
        int32_t prodX = consX;
        int32_t prodY = consY + producer->getNumRows();

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
