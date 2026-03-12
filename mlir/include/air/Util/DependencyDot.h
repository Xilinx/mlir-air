//===- DependencyDot.h - DOT graph visualization for dependency ---*- C++
//-*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Utility functions to serialize dependency graphs to GraphViz DOT format.
// No external dependencies (no boost, no RTTI).
//
//===----------------------------------------------------------------------===//

#pragma once

#include "air/Util/Dependency.h"
#include <string>

namespace xilinx {
namespace air {

/// Write a single dependency graph to a DOT file.
/// Nodes include color, shape, and label (asyncEventName + description).
/// If timing info is present (start_time/end_time != 0), it appears in the
/// label.
void writeDotGraph(const dependencyGraph::Graph &g,
                   const std::string &filename);

/// Write all graphs in the hierarchy to individual DOT files.
/// Generates: host.dot, air.launch_1.dot, air.segment_1_1.dot, etc.
/// Creates the output directory if it doesn't exist.
void dumpDotGraphFiles(const dependencyGraph &globalGraph,
                       const std::string &dumpDir);

/// Write the entire graph hierarchy to a single DOT file using
/// DOT subgraph clusters for hierarchical nesting.
void dumpCombinedDotGraph(const dependencyGraph &globalGraph,
                          const std::string &filename);

} // namespace air
} // namespace xilinx
