// Copyright (C) 2023, Xilinx Inc. All rights reserved.
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "air/Util/DirectedAdjacencyMap.h"
#include <stdexcept>

class TestGraph : public xilinx::air::DirectedAdjacencyMap {
public:
  using xilinx::air::DirectedAdjacencyMap::addVertex;
};

using VertexId = TestGraph::VertexId;

void basicTest() {

  TestGraph g;

  g.addVertex();
  g.addVertex();
  g.addVertex();

  g.addEdge(2, 0);
  g.addEdge(1, 0);
  g.addEdge(2, 1);

  if (g.getSchedule() != std::vector<VertexId>{2, 1, 0}) {
    throw std::runtime_error("Incorrect schedule");
  }

  auto closure = g.getClosure();
  decltype(closure) expected = {{1, 0, 0}, {1, 1, 0}, {1, 1, 1}};
  if (closure != expected) {
    throw std::runtime_error("Incorrect closure");
  }

  if (!g.hasEdge(2, 0)) {
    throw std::runtime_error("Edge 2->0 was inserted into graph");
  }

  g.applyTransitiveReduction();

  if (g.hasEdge(2, 0)) {
    throw std::runtime_error("Edge 2->0 not removed in reduction");
  }
}

void templateClassTest() {

  class TGraph : public xilinx::air::TypedDirectedAdjacencyMap<std::string> {};

  TGraph a;
  a.addVertex();
  a.addVertex();

  a[0] = "foo";
  a[1] = "bar";

  if (a[0] != "foo" || a[1] != "bar") {
    throw std::runtime_error("string vertex type setter/getter failure");
  }
}

int main() {
  basicTest();
  templateClassTest();
  return 0;
}
