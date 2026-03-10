//===- DependencyDot.cpp - DOT graph visualization for dependency -*- C++
//-*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/DependencyDot.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace xilinx::air;

/// Escape characters that are special in DOT label strings.
static std::string escapeDotLabel(const std::string &s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s) {
    if (c == '"')
      result += "\\\"";
    else if (c == '\\')
      result += "\\\\";
    else if (c == '\n')
      result += "\\n";
    else
      result += c;
  }
  return result;
}

/// Build the label string for a dependency node.
/// Individual fields are escaped, then joined with DOT newline sequences.
static std::string buildNodeLabel(const dependencyNodeEntry &node) {
  std::string label = escapeDotLabel(node.asyncEventName);
  if (!node.detailed_description.empty())
    label += "\\n" + escapeDotLabel(node.detailed_description);
  if (node.start_time != 0 && node.end_time != 0)
    label += "\\n[" + std::to_string(node.start_time) + "-" +
             std::to_string(node.end_time) + "]";
  return label;
}

/// Get a sanitized name for a hierarchy op (e.g. "air.launch").
/// Dots are replaced with underscores for use in filenames.
static std::string getHierarchyName(mlir::Operation *op) {
  if (!op)
    return "unknown";
  std::string name = op->getName().getStringRef().str();
  // Replace dots with underscores for filename safety.
  for (char &c : name)
    if (c == '.')
      c = '_';
  return name;
}

void xilinx::air::writeDotGraph(const dependencyGraph::Graph &g,
                                const std::string &filename) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  if (ec)
    return;

  os << "digraph G {\n";
  os << "  rankdir=LR;\n";

  auto vertices = g.getVertices();

  // Write vertices.
  for (auto v : vertices) {
    const auto &node = g[v];
    std::string label = buildNodeLabel(node);

    os << "  v" << v << " [label=\"" << label << "\"";
    if (!node.color.empty())
      os << ", color=\"" << node.color << "\", style=filled";
    if (!node.shape.empty())
      os << ", shape=\"" << node.shape << "\"";
    os << "];\n";
  }

  // Write edges.
  for (auto v : vertices) {
    for (auto adj : g.adjacentVertices(v)) {
      os << "  v" << v << " -> v" << adj << ";\n";
    }
  }

  os << "}\n";
}

/// Recursively dump per-level DOT files.
static void dumpGraphRecursive(const dependencyGraph &graph,
                               const std::string &dir,
                               const std::string &prefix) {
  // Write this level's graph.
  writeDotGraph(graph.g, dir + prefix + ".dot");

  // Recurse into subgraphs.
  for (unsigned i = 0, e = graph.subgraphs.size(); i < e; ++i) {
    const auto &sub = graph.subgraphs[i];
    std::string subPrefix = prefix + "_" + getHierarchyName(sub.hierarchyOp) +
                            "_" + std::to_string(i + 1);
    dumpGraphRecursive(sub, dir, subPrefix);
  }
}

void xilinx::air::dumpDotGraphFiles(const dependencyGraph &globalGraph,
                                    const std::string &dumpDir) {
  std::string dir = dumpDir;
  if (!dir.empty()) {
    // Create the directory (ignore if already exists).
    llvm::sys::fs::create_directories(dir);
    if (dir.back() != '/')
      dir += '/';
  }

  dumpGraphRecursive(globalGraph, dir, "host");
}

/// Recursive helper for combined DOT graph with subgraph clusters.
static void writeDotSubgraph(llvm::raw_fd_ostream &os,
                             const dependencyGraph &graph,
                             const std::string &prefix, unsigned depth,
                             unsigned &clusterId) {
  std::string indent(depth * 2, ' ');

  // Determine label for this subgraph.
  std::string label;
  if (graph.hierarchyOp)
    label = graph.hierarchyOp->getName().getStringRef().str();
  if (!graph.position.empty()) {
    label += " [";
    for (unsigned i = 0; i < graph.position.size(); ++i) {
      if (i > 0)
        label += ",";
      label += std::to_string(graph.position[i]);
    }
    label += "]";
  }

  if (depth > 0) {
    os << indent << "subgraph cluster_" << clusterId++ << " {\n";
    os << indent << "  label=\"" << escapeDotLabel(label) << "\";\n";
    os << indent << "  style=dashed;\n";
  }

  auto vertices = graph.g.getVertices();

  // Emit vertices.
  for (auto v : vertices) {
    const auto &node = graph.g[v];
    std::string nodeLabel = buildNodeLabel(node);

    os << indent << "  " << prefix << "v" << v << " [label=\"" << nodeLabel
       << "\"";
    if (!node.color.empty())
      os << ", color=\"" << node.color << "\", style=filled";
    if (!node.shape.empty())
      os << ", shape=\"" << node.shape << "\"";
    os << "];\n";
  }

  // Emit edges.
  for (auto v : vertices) {
    for (auto adj : graph.g.adjacentVertices(v)) {
      os << indent << "  " << prefix << "v" << v << " -> " << prefix << "v"
         << adj << ";\n";
    }
  }

  // Recurse into subgraphs.
  for (unsigned i = 0, e = graph.subgraphs.size(); i < e; ++i) {
    std::string subPrefix = prefix + "s" + std::to_string(i) + "_";
    writeDotSubgraph(os, graph.subgraphs[i], subPrefix, depth + 1, clusterId);
  }

  if (depth > 0) {
    os << indent << "}\n";
  }
}

void xilinx::air::dumpCombinedDotGraph(const dependencyGraph &globalGraph,
                                       const std::string &filename) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  if (ec)
    return;

  unsigned clusterId = 0;
  os << "digraph G {\n";
  os << "  rankdir=LR;\n";
  writeDotSubgraph(os, globalGraph, "", 0, clusterId);
  os << "}\n";
}
