//===- AIRMergeUnrolledDevicesPass.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/PassDetail.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "air-merge-unrolled-devices"

using namespace mlir;

namespace xilinx {
namespace air {

/// Extract the base segment name from a device symbol name.
/// E.g., "segment_with_unroll_0_0" -> "segment_with_unroll"
static std::string extractBaseSegmentName(StringRef fullName) {
  // Pattern: {base}_{unroll_x}_{unroll_y}
  // Find the last two underscore-separated numeric components
  size_t lastUnderscore = fullName.rfind('_');
  if (lastUnderscore == StringRef::npos)
    return fullName.str();

  StringRef afterLast = fullName.substr(lastUnderscore + 1);
  // Check if it's a number
  int dummy;
  if (afterLast.getAsInteger(10, dummy))
    return fullName.str(); // Not a number, return as-is

  // Find the second-to-last underscore
  StringRef prefix = fullName.substr(0, lastUnderscore);
  size_t secondLastUnderscore = prefix.rfind('_');
  if (secondLastUnderscore == StringRef::npos)
    return fullName.str();

  StringRef afterSecondLast = prefix.substr(secondLastUnderscore + 1);
  if (afterSecondLast.getAsInteger(10, dummy))
    return fullName.str(); // Not a number

  // Both are numbers, so the base name is everything before the second-to-last
  // underscore
  return prefix.substr(0, secondLastUnderscore).str();
}

/// Get the number of columns for an AIE device type.
static int getDeviceColumns(AIE::AIEDevice device) {
  switch (device) {
  case AIE::AIEDevice::npu2_1col:
    return 1;
  case AIE::AIEDevice::npu2_2col:
    return 2;
  case AIE::AIEDevice::npu2_4col:
    return 4;
  case AIE::AIEDevice::npu2:
    return 8;
  default:
    return 4; // Default assumption
  }
}

/// Compute the merged device type from sub-device type and number of devices.
/// This reverses the computeSubDeviceType logic from AIRToAIEPass.
static AIE::AIEDevice computeMergedDeviceType(AIE::AIEDevice subDevice,
                                              int numDevices) {
  int subCols = getDeviceColumns(subDevice);
  int totalCols = subCols * numDevices;

  if (totalCols >= 8)
    return AIE::AIEDevice::npu2;
  else if (totalCols >= 4)
    return AIE::AIEDevice::npu2_4col;
  else if (totalCols >= 2)
    return AIE::AIEDevice::npu2_2col;
  else
    return AIE::AIEDevice::npu2_1col;
}

class AIRMergeUnrolledDevicesPass
    : public impl::AIRMergeUnrolledDevicesBase<AIRMergeUnrolledDevicesPass> {

public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    // Group devices by base segment name
    llvm::StringMap<SmallVector<AIE::DeviceOp>> deviceGroups;

    module.walk([&](AIE::DeviceOp device) {
      if (!device->hasAttr("segment_unroll_x"))
        return;
      std::string baseName = extractBaseSegmentName(device.getSymName());
      deviceGroups[baseName].push_back(device);
    });

    // Process each group
    for (auto &entry : deviceGroups) {
      StringRef baseName = entry.first();
      auto &devices = entry.second;
      if (devices.size() <= 1)
        continue; // Nothing to merge

      LLVM_DEBUG(llvm::dbgs() << "Merging " << devices.size()
                              << " devices for segment: " << baseName << "\n");

      // Sort by unroll_x, then unroll_y
      llvm::sort(devices, [](AIE::DeviceOp a, AIE::DeviceOp b) {
        int64_t ax = a->getAttrOfType<IntegerAttr>("segment_unroll_x").getInt();
        int64_t bx = b->getAttrOfType<IntegerAttr>("segment_unroll_x").getInt();
        if (ax != bx)
          return ax < bx;
        int64_t ay = a->getAttrOfType<IntegerAttr>("segment_unroll_y").getInt();
        int64_t by = b->getAttrOfType<IntegerAttr>("segment_unroll_y").getInt();
        return ay < by;
      });

      // Compute column width per device using device type
      AIE::AIEDevice subDeviceType = devices[0].getDevice();
      int deviceWidth = getDeviceColumns(subDeviceType);
      LLVM_DEBUG(llvm::dbgs() << "  Device width: " << deviceWidth << "\n");

      // Compute merged device type
      AIE::AIEDevice mergedType =
          computeMergedDeviceType(subDeviceType, devices.size());
      LLVM_DEBUG(llvm::dbgs() << "  Merged device type: "
                              << AIE::stringifyAIEDevice(mergedType) << "\n");

      // Create merged device (insert before the first device)
      builder.setInsertionPoint(devices[0]);
      auto mergedDevice = builder.create<AIE::DeviceOp>(
          devices[0].getLoc(),
          AIE::AIEDeviceAttr::get(builder.getContext(), mergedType));
      mergedDevice->setAttr(SymbolTable::getSymbolAttrName(),
                            builder.getStringAttr(baseName));

      // Copy DLTI data layout attribute if present
      if (auto dltiAttr = devices[0]->getAttr("dlti.dl_spec"))
        mergedDevice->setAttr("dlti.dl_spec", dltiAttr);

      // Create the device body with a terminator
      builder.createBlock(&mergedDevice.getRegion());
      builder.create<AIE::EndOp>(mergedDevice.getLoc());

      // Clone ops from each source device with tile offset
      for (auto [idx, srcDevice] : llvm::enumerate(devices)) {
        int64_t unrollX =
            srcDevice->getAttrOfType<IntegerAttr>("segment_unroll_x").getInt();
        int colOffset = unrollX * deviceWidth;
        LLVM_DEBUG(llvm::dbgs() << "  Cloning device " << idx
                                << " with col offset " << colOffset << "\n");
        cloneDeviceOpsWithOffset(builder, srcDevice, mergedDevice, colOffset,
                                 idx);
      }

      // Merge airrt.segment_metadata entries
      mergeSegmentMetadata(module, baseName, devices);

      // Erase original devices
      for (auto device : devices)
        device.erase();
    }
  }

private:
  /// Clone all ops from srcDevice to mergedDevice, offsetting tile columns.
  /// unrollIdx is used to make symbol names unique across devices.
  void cloneDeviceOpsWithOffset(OpBuilder &builder, AIE::DeviceOp srcDevice,
                                AIE::DeviceOp mergedDevice, int colOffset,
                                int64_t unrollIdx) {
    IRMapping mapping;
    builder.setInsertionPoint(mergedDevice.getBody()->getTerminator());

    // First pass: clone TileOps with offset and build mapping
    for (auto tileOp : srcDevice.getOps<AIE::TileOp>()) {
      int newCol = tileOp.getCol() + colOffset;
      int row = tileOp.getRow();

      // Check if a tile at this location already exists in merged device
      AIE::TileOp existingTile = nullptr;
      for (auto t : mergedDevice.getOps<AIE::TileOp>()) {
        if (t.getCol() == newCol && t.getRow() == row) {
          existingTile = t;
          break;
        }
      }

      if (existingTile) {
        mapping.map(tileOp.getResult(), existingTile.getResult());
      } else {
        auto newTile =
            builder.create<AIE::TileOp>(tileOp.getLoc(), newCol, row);
        mapping.map(tileOp.getResult(), newTile.getResult());
      }
    }

    // Second pass: clone all other ops (except terminator)
    for (auto &op : srcDevice.getBody()->getOperations()) {
      // Skip TileOps (already handled) and terminator
      if (isa<AIE::TileOp, AIE::EndOp>(op))
        continue;

      auto *clonedOp = builder.clone(op, mapping);

      // If the op has a sym_name attribute, make it unique by appending unroll
      // index. EXCEPT for ShimDMAAllocationOp - these names are already unique
      // (they contain the unroll coordinates) and are referenced by host-side
      // metadataArray in air.channel.put/get ops.
      if (auto symNameAttr = clonedOp->getAttrOfType<StringAttr>("sym_name")) {
        if (!isa<AIE::ShimDMAAllocationOp>(clonedOp)) {
          std::string newName = symNameAttr.getValue().str() + "_unroll_" +
                                std::to_string(unrollIdx);
          clonedOp->setAttr("sym_name", builder.getStringAttr(newName));
        }
      }
    }
  }

  /// Merge airrt.segment_metadata entries for the given devices.
  void mergeSegmentMetadata(ModuleOp module, StringRef baseName,
                            ArrayRef<AIE::DeviceOp> devices) {
    // Find airrt.module_metadata
    airrt::ModuleMetadataOp moduleMeta = nullptr;
    module.walk([&](airrt::ModuleMetadataOp op) { moduleMeta = op; });
    if (!moduleMeta)
      return;

    // Collect all segment metadata ops for the devices
    SmallVector<airrt::SegmentMetadataOp> segmentMetas;
    for (auto device : devices) {
      StringRef deviceName = device.getSymName();
      for (auto segMeta : moduleMeta.getSegments()
                              .front()
                              .getOps<airrt::SegmentMetadataOp>()) {
        if (segMeta.getSymName() == deviceName)
          segmentMetas.push_back(segMeta);
      }
    }

    if (segmentMetas.empty())
      return;

    // Create merged segment metadata
    OpBuilder builder(moduleMeta.getContext());
    builder.setInsertionPoint(segmentMetas[0]);
    auto mergedMeta = builder.create<airrt::SegmentMetadataOp>(
        builder.getUnknownLoc(), baseName);
    builder.createBlock(&mergedMeta.getHerds());
    builder.create<airrt::SegmentMetadataTerminatorOp>(builder.getUnknownLoc());

    // Clone herd metadata from all segments into the merged one
    for (auto segMeta : segmentMetas) {
      builder.setInsertionPoint(mergedMeta.getBody()->getTerminator());
      for (auto &op : segMeta.getBody()->getOperations()) {
        if (isa<airrt::SegmentMetadataTerminatorOp>(op))
          continue;
        builder.clone(op);
      }
    }

    // Copy any dma_allocations attribute from first segment
    if (auto dmaAllocs = segmentMetas[0]->getAttr("dma_allocations"))
      mergedMeta->setAttr("dma_allocations", dmaAllocs);

    // Erase original segment metadata ops
    for (auto segMeta : segmentMetas)
      segMeta.erase();
  }
};

std::unique_ptr<mlir::Pass> createAIRMergeUnrolledDevicesPass() {
  return std::make_unique<AIRMergeUnrolledDevicesPass>();
}

} // namespace air
} // namespace xilinx
