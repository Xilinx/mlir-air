// Copyright (C) 2023, Xilinx Inc. All rights reserved.
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "air/Util/DirectedAdjacencyMap.h"

namespace xilinx {
namespace air {

using VertexId = DirectedAdjacencyMap::VertexId;

// Kahn's algorithm. See https://en.wikipedia.org/wiki/Topological_sorting
std::vector<VertexId> DirectedAdjacencyMap::getSchedule() const {

  std::vector<VertexId> toProcess;

  std::vector<uint64_t> nRemaining(numVertices());
  for (uint64_t i = 0; i < numVertices(); ++i) {
    nRemaining[i] = bwdEdges[i].size();
    if (nRemaining[i] == 0) {
      toProcess.push_back(i);
    }
  }

  std::vector<VertexId> schedule;
  schedule.reserve(numVertices());
  while (!toProcess.empty()) {
    auto nxt = toProcess.back();
    toProcess.pop_back();
    schedule.push_back(nxt);
    for (auto n : fwdEdges[nxt]) {
      --nRemaining[n];
      if (nRemaining[n] == 0) {
        toProcess.push_back(n);
      }
    }
  }

  return schedule;
}

void DirectedAdjacencyMap::addEdge(VertexId src, VertexId dst) {
  assert(src < numVertices());
  assert(dst < numVertices());
  fwdEdges[src].insert(dst);
  bwdEdges[dst].insert(src);
}

void DirectedAdjacencyMap::removeEdge(VertexId src, VertexId dst) {
  if (src >= numVertices() || dst >= numVertices()) {
    return;
  }
  if (fwdEdges[src].count(dst) > 0) {
    fwdEdges[src].erase(dst);
    bwdEdges[dst].erase(src);
  }
}

std::vector<VertexId> DirectedAdjacencyMap::getVertices() const {
  std::vector<VertexId> vs;
  vs.reserve(numVertices());
  for (uint64_t i = 0; i < numVertices(); ++i) {
    vs.push_back(i);
  }
  return vs;
}

std::vector<std::vector<bool>> DirectedAdjacencyMap::getClosure() const {

  std::vector<std::vector<bool>> closure(
      numVertices(), std::vector<bool>(numVertices(), false));

  auto schedule = getSchedule();
  for (uint64_t i = 0; i < numVertices(); ++i) {
    auto vertexId = schedule[numVertices() - i - 1];
    closure[vertexId][vertexId] = true;
    for (auto nxt : adjacentVertices(vertexId)) {
      for (uint64_t j = 0; j < numVertices(); ++j) {
        closure[vertexId][j] = closure[vertexId][j] || closure[nxt][j];
      }
    }
  }
  return closure;
}

void DirectedAdjacencyMap::applyTransitiveReduction() {
  auto closure = getClosure();
  for (uint64_t i = 0; i < numVertices(); ++i) {
    std::set<VertexId> reducedEdges;
    for (auto nxt : adjacentVertices(i)) {
      bool retain = true;
      for (auto other : adjacentVertices(i)) {
        if (nxt == other) {
          continue;
        }
        // if nxt is in the transitive closure of other, don't keep it.
        if (closure[other][nxt]) {
          retain = false;
        }
      }
      if (retain) {
        reducedEdges.insert(nxt);
      }
    }
    fwdEdges[i] = reducedEdges;
  }
  updateBwdEdgesFromFwdEdges();
}

void DirectedAdjacencyMap::updateBwdEdgesFromFwdEdges() {
  bwdEdges.clear();
  bwdEdges.resize(numVertices());
  for (uint64_t i = 0; i < numVertices(); ++i) {
    for (auto nxt : fwdEdges[i]) {
      bwdEdges[nxt].insert(i);
    }
  }
}

} // namespace air
} // namespace xilinx
