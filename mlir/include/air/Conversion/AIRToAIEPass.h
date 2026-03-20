//===- AIRToAIEPass.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_AIE_PASS_H
#define AIR_TO_AIE_PASS_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#if AIR_ENABLE_AIE
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#endif

namespace mlir {
class ModuleOp;
class OpBuilder;
class RewriterBase;
} // namespace mlir

namespace xilinx {

#if AIR_ENABLE_AIE
namespace AIE {
class DeviceOp;
class TileOp;
class BufferOp;
} // namespace AIE
#endif

namespace air {
class SegmentOp;
class HerdOp;
class LaunchOp;

mlir::FailureOr<mlir::ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                                air::SegmentOp segment);
std::unique_ptr<mlir::Pass> createAIRToAIEPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToAIEPass(const AIRToAIEOptions &options);

std::unique_ptr<mlir::Pass> createAIRSplitDevicesPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRLinalgToFuncPass();

#if AIR_ENABLE_AIE

/// Options shared between AIRToAIE and AIRHierarchyToAIE passes.
struct AIRToAIEConversionOptions {
  int64_t col_offset;
  int64_t row_offset;
  bool emit_while;
  bool emit_herd_lock;
  bool generate_shim_dma;
  bool insert_trace_packet_flow;
  bool use_packet_flow_at_shim_dmas;
  bool use_lock_race_condition_fix;
  AIE::AIEDevice device;
};

/// Create aie.device ops with aie.tile/aie.core for each segment/herd.
void createAIEModulesAndOutlineCores(
    mlir::ModuleOp module,
    std::vector<std::tuple<AIE::DeviceOp, air::HerdOp,
                           AIRToAIEConversionOptions>> &aie_modules,
    std::map<AIE::TileOp, air::HerdOp> &tileToHerdMap,
    AIRToAIEConversionOptions &options);

/// Clone L2/L3 data movement ops into the aie.device region.
void cloneL2AndL3MemcpysToDeviceOp(mlir::OpBuilder &builder,
                                   AIE::DeviceOp aie_device,
                                   mlir::ModuleOp module, bool clone_l2,
                                   bool clone_l3,
                                   bool lock_race_condition_fix,
                                   air::LaunchOp targetLaunch);

/// Specialize affine.if / scf.if on herd tile coordinates.
void specializeHerdAffineIf(AIE::DeviceOp m);

/// Lower air.execute regions to sequential code.
void lowerAirExecute(AIE::DeviceOp d);

/// Lower scf.for iter_args carrying air.async.token.
void lowerScfAirTokens(AIE::DeviceOp m);

/// Specialize air.channel bundles into per-index channels.
void specializeChannelBundle(
    AIE::DeviceOp &d, std::map<std::string, std::string> &chan_to_chan_map);

/// Remove orphaned channels with only puts or only gets.
void removeOrphanedChannels(AIE::DeviceOp &d);

/// Create the --air-hierarchy-to-aie pass.
std::unique_ptr<mlir::Pass> createAIRHierarchyToAIEPass();

#endif // AIR_ENABLE_AIE

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_PASS_H
