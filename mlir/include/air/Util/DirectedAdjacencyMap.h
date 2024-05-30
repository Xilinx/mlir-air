// Copyright (C) 2023, Xilinx Inc. All rights reserved.
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstdint>
#include <set>
#include <vector>

namespace xilinx {
namespace air {

/**
 * This class is functionally similar to the boost::adjacency_list class. It
 * provides minimal functionality to support the removal of boost in this
 * project. Additional functionality should be added as required.
 *
 * The class implements a directed acyclic graph, where each node can be queried
 * for outgoing edges and incoming edges.
 * */
class DirectedAdjacencyMap {

public:
  using VertexId = uint64_t;

  /**
   * Insert an edge from \p src to \p dst to the graph, so that \p dst is
   * adjacent to \p src.
   * */
  void addEdge(VertexId src, VertexId dst);

  /**
   * \return The number of vertices which have an edge \b from \p v.
   * */
  uint64_t outDegree(VertexId v) const { return fwdEdges[v].size(); }

  /**
   * \return The number of vertices which have an edge \b to \p v.
   * */
  uint64_t inDegree(VertexId v) const { return bwdEdges[v].size(); }

  /**
   * \return The number of vertices in the graph.
   * */
  uint64_t numVertices() const { return fwdEdges.size(); }

  /**
   * Remove the edge from \p src to \p dst from the graph. If the graph is not
   * in the graph, or if a VertexId is out of range, this function does nothing.
   * */
  void removeEdge(VertexId src, VertexId dst);

  /**
   * \return A vector of all vertices in the graph.
   * */
  std::vector<VertexId> getVertices() const;

  /**
   * \return true of the graph has an edge from \p src to \p dst, and false
   *         otherwise.
   * */
  bool hasEdge(VertexId src, VertexId dst) const {
    return src < numVertices() && (fwdEdges[src].count(dst) > 0);
  }

  /**
   * \return A vector of all vertices which have an edge to \p v.
   * */
  std::vector<VertexId> inverseAdjacentVertices(VertexId v) const {
    return {bwdEdges[v].begin(), bwdEdges[v].end()};
  }

  /**
   * \return A vector of all vertices which have an edge from \p v.
   * */
  std::vector<VertexId> adjacentVertices(VertexId i) const {
    return {fwdEdges[i].begin(), fwdEdges[i].end()};
  }

  /**
   * \return The vertices in topological order. If the graph contains a cycle,
   *         the subset of vertices which can be scheduled is returned, so that
   *         the returned vector is smaller than the number of vertices in the
   *         graph.
   * */
  std::vector<VertexId> getSchedule() const;

  /**
   * \return The topological closure of the graph. See
   *         https://en.wikipedia.org/wiki/Transitive_closure
   *
   *         If out[i][j] is true, then there is a path from i to j (or i == i).
   *
   * */
  std::vector<std::vector<bool>> getClosure() const;

  /**
   * Apply a transitive reduction to this graph. This reduces the number of
   * edges to the minimal set possible which does not change the transitive
   * closure of this graph. See
   * https://en.wikipedia.org/wiki/Transitive_reduction
   *
   * O(n^2) memory, O(n^3) operations (same as boost).
   * */
  void applyTransitiveReduction();

protected:
  // These methods are protected, as they are intended to be called by their
  // derived classes.

  void addVertex() {
    fwdEdges.push_back({});
    bwdEdges.push_back({});
  }

  void clear() {
    fwdEdges = {};
    bwdEdges = {};
  }

  // Given that fwdEdges is up-to-date, update bwdEdges to correspond.
  void updateBwdEdgesFromFwdEdges();

private:
  // The edges.
  std::vector<std::set<VertexId>> fwdEdges;

  // The inverse edges.
  std::vector<std::set<VertexId>> bwdEdges;
};

/**
 * An extension to DirectedAdjacencyMap where vertices have types.
 *
 * \tparam T The type of the vertex.
 * */
template <typename T>
class TypedDirectedAdjacencyMap : public DirectedAdjacencyMap {
public:
  using DirectedAdjacencyMap::VertexId;

  VertexId addVertex() {
    auto id = nodes.size();
    nodes.push_back({});
    DirectedAdjacencyMap::addVertex();
    return id;
  }

  const T &operator[](VertexId v) const {
    assert(v < numVertices());
    return nodes[v];
  }

  T &operator[](VertexId v) {
    // const_cast, really? See Scott Meyers: Effective C++ for this use case.
    return const_cast<T &>(
        static_cast<const TypedDirectedAdjacencyMap<T> &>(*this)[v]);
  }

  void clear() {
    nodes = {};
    DirectedAdjacencyMap::clear();
  }

private:
  std::vector<T> nodes;
};

} // namespace air
} // namespace xilinx
