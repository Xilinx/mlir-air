//===- AirChannelToConduitPass.cpp - air-opt wrapper for Pass B -*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Thin wrapper that makes the mlir-aie AirChannelToConduit pass (Pass B) and
// the AirChannelIndexFlattener pre-pass available to the mlir-air tool
// (air-opt) when AIR_ENABLE_AIE is set.
//
// The pass implementations live in mlir-aie:
//   mlir-aie/lib/Dialect/Conduit/Transforms/AirChannelToConduit.cpp
//   mlir-aie/lib/Dialect/Conduit/Transforms/AirChannelIndexFlattener.cpp
//
// This file only pulls in the mlir-aie headers and the pass registration
// mechanism; no pass logic is duplicated here.
//
// Registration is performed inside registerConversionPasses() in Passes.cpp
// via the generated GEN_PASS_REGISTRATION macro, which calls
// xilinx::conduit::createAirChannelToConduitPass() and
// xilinx::conduit::createAirChannelIndexFlattenerPass() as constructors.
//
// Invocation:
//   air-opt --allow-unregistered-dialect --air-channel-to-conduit <file.mlir>
//   air-opt --allow-unregistered-dialect --air-channel-flatten-indices <file.mlir>
//
//===----------------------------------------------------------------------===//

// This file is intentionally minimal: the pass is registered via Passes.td
// (AirChannelToConduit / AirChannelIndexFlattener definitions) and Passes.cpp
// (GEN_PASS_REGISTRATION). The constructor functions are provided by
// ConduitTransforms (mlir-aie) which is listed in CONVERSION_LINK_LIBS.
//
// No additional code is required here.
