//===- RunnerNode.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_NODE
#define AIR_UTIL_RUNNER_NODE

namespace xilinx {
namespace air {

class runnerNode {

public:
  // Parent runner node
  runnerNode *parent;
  // Dependency graph that the current runner node processes
  dependencyGraph *ctrl_g;
  // Runner node hierarchy type
  std::string runner_node_type;
  // Each entry is an std::tuple. First element is vertex, second element is
  // vector of resoruces consumed, and third element is thread id.
  // TODO: Replace thread id with id which better reflects resource slots.
  std::vector<
      std::tuple<Graph::vertex_descriptor, std::vector<resource *>, unsigned>>
      wavefront;
  // A vector of vertices processed by the current runner node
  std::vector<Graph::vertex_descriptor> processed_vertices;
  // Sub runner nodes to the current runner node
  std::deque<runnerNode> sub_runner_nodes;
  // Reserved resource hierarchies
  std::vector<resourceHierarchy *> resource_hiers;

  // Private wavefront of each runner node, reserved to interface with resource
  // model
  std::vector<dependencyNodeEntry *> wavefrontNodes() {
    std::vector<dependencyNodeEntry *> output;
    for (auto v : wavefront) {
      output.push_back(&ctrl_g->g[std::get<0>(v)]);
    }
    return output;
  }

  // Push runner "start" signal into wavefront
  void pushStartToWavefront(Graph::vertex_descriptor v) {
    std::vector<resource *> reserved_resources;
    // Allocate resources to this runner
    this->consumeResourceHiersWhenRunnerStarts(reserved_resources);
    auto entry = std::make_tuple(v, reserved_resources, (unsigned)1);
    for (auto i : this->wavefront) {
      assert(std::get<2>(i) != std::get<2>(entry) && "queried thread is busy");
    }
    this->wavefront.push_back(entry);
  }

  // Push an entry to wavefront
  void pushToWavefront(Graph::vertex_descriptor v, unsigned core_id = 0) {
    std::vector<resource *> reserved_resources;
    // Allocate resources to this event
    this->consumeOrReleaseResources(reserved_resources, v);
    // Acquire available thread id for current op
    unsigned max_num_threads_per_core =
        10; // Hardcoded maximum number of threads per core
    unsigned offset = core_id * max_num_threads_per_core;
    unsigned tid = 0;
    for (unsigned i = offset + 1; i < offset + this->wavefront.size() + 2;
         i++) {
      bool tid_i_unavailable = false;
      for (auto j : this->wavefront) {
        if (std::get<2>(j) == i) {
          tid_i_unavailable = true;
        }
      }
      if (!tid_i_unavailable) {
        tid = i;
        break;
      }
    }
    this->wavefront.push_back(std::make_tuple(v, reserved_resources, tid));
  }

  // Initialize sub runner nodes from launch graph tree
  void initRunnerNodesFromLaunchGraph(dependencyGraph &launchGraph) {
    launchGraph.runner_node = this;
    launchGraph.runner_node->channel_token_counts_ptr =
        &(launchGraph.runner_node->channel_token_counts);
    for (auto &partitionGraph : launchGraph.subgraphs) {
      // Create partition runner node
      this->sub_runner_nodes.push_back(
          runnerNode(this, &partitionGraph, "partition", this->dep_ctx,
                     this->sim_granularity,
                     &(launchGraph.runner_node->channel_token_counts)));
      auto current_partition_node = &(this->sub_runner_nodes.back());
      for (auto &herdGraph : partitionGraph.subgraphs) {
        // Create herd runner node
        current_partition_node->sub_runner_nodes.push_back(
            runnerNode(current_partition_node, &herdGraph, "herd",
                       this->dep_ctx, this->sim_granularity,
                       &(launchGraph.runner_node->channel_token_counts)));
      }
    }
    this->addPointerBetweenSubRunnerNodeAndSubCommandGraph();
    for (auto &partition_runner_node : this->sub_runner_nodes) {
      partition_runner_node.addPointerBetweenSubRunnerNodeAndSubCommandGraph();
    }
  }

  // Get time stamps from wavefront
  void getTimeStampsFromWavefront(std::vector<uint64_t> &next_times) {
    for (auto it = this->wavefront.begin(); it != this->wavefront.end(); it++) {
      auto command_node = this->ctrl_g->g[std::get<0>(*it)];
      if (command_node.is_started() && (command_node.end_time)) {
        next_times.push_back(command_node.end_time);
      }
    }
  }

  // Execute an mlir op in runner node
  void executeOpImpls(Graph::vertex_descriptor it, uint64_t time) {
    Graph G = this->ctrl_g->g;
    auto node = G[it];
    if (node.asyncEventType == "start") {
      this->executeOp(it);
    } else if (auto Op = dyn_cast<xilinx::air::HierarchyInterface>(node.op)) {
      for (auto sub_dependency_graph : node.nextDependencyGraphs) {
        auto sub_runner_node = sub_dependency_graph->runner_node;
        this->executeOp(Op, time, sub_runner_node, it);
      }
    } else if (auto Op = dyn_cast<scf::ForOp>(node.op)) {
      this->executeOp(Op, it);
    } else if (dyn_cast<scf::YieldOp>(node.op) &&
               getScfParentOpFromYieldOp<scf::ForOp>(
                   dyn_cast<scf::YieldOp>(node.op))) {
      auto Op = dyn_cast<scf::YieldOp>(node.op);
      auto parent_for_op =
          dyn_cast<scf::ForOp>(getScfParentOpFromYieldOp<scf::ForOp>(Op));
      this->executeOp(Op, time, parent_for_op, it);
    } else if (auto Op = dyn_cast<air::ChannelPutOp>(node.op)) {
      this->executeOp(Op, it);
    } else if (auto Op = dyn_cast<air::ChannelGetOp>(node.op)) {
      this->executeOp(Op, it);
    } else {
      this->executeOp(it);
    }
  }

  // Recursively reset all vertices in for loop body
  void resetGraphBetweenTwoVertices(Graph::vertex_descriptor start_v,
                                    Graph::vertex_descriptor end_v, Graph &G,
                                    uint64_t time) {

    this->resetVertex(start_v, G, time);

    if (start_v == end_v)
      return;

    SmallVector<Graph::vertex_descriptor, 1> vertices;
    if (this->hasPath(start_v, end_v, G, vertices)) {
      for (auto v : vertices) {
        this->resetVertex(v, G, time);
        // If v is a hierarchy op, then recursively clear the entire subgraph
        if (G[v].asyncEventType == "hierarchy") {
          for (auto sub_c : G[v].nextDependencyGraphs) {
            auto start = sub_c->start_vertex;
            auto terminator_v = sub_c->terminator_vertex;
            auto sub_g = sub_c->g;
            auto sub_runner = sub_c->runner_node;
            sub_runner->resetGraphBetweenTwoVertices(start, terminator_v, sub_g,
                                                     time);
          }
        }
        // Else if v is an scf.for op, then clear the cached trip count from
        // runner node
        else if (G[v].asyncEventType == "for_loop") {
          // Clear for loop trip count from runner node's cache
          for (auto it = this->loop_trip_count.begin();
               it != this->loop_trip_count.end(); it++) {
            if (std::get<0>(*it) == (unsigned)getIdAttr(G[v].op)) {
              this->loop_trip_count.erase(it);
              it--;
            }
          }
        }
      }
    }
  }

  // Remove vertices in vector a which already exist in vector b
  void removeRepeatedVertices(std::vector<Graph::vertex_descriptor> &a,
                              std::vector<Graph::vertex_descriptor> b) {
    for (auto v : b) {
      this->removeVertexFromVertices(a, v);
    }
  }

  // Consume tokens upon op execution
  void consumeLoopYieldedTokens(Graph::vertex_descriptor it) {

    Graph &G = this->ctrl_g->g;
    auto inv_adj_set = boost::inv_adjacent_vertices(it, G);
    for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second;
         ++inv_adj_v) {
      if (G[*inv_adj_v].asyncEventType == "for_loop") {
        int th = this->tokenCountThresholdForExecution(
            G[it].op); // Consume all iter_arg tokens
        assert(G[it].token_count >= th * this->tokenSpatialFactor(G[it].op));
        G[it].token_count -= th * this->tokenSpatialFactor(G[it].op);
        return;
      }
    }
  }

  // Check if all dependencies of an async op have been fulfilled
  bool checkAllDependenciesFulfillment(
      std::vector<std::pair<dependencyNodeEntry, std::string>> dep_list,
      dependencyNodeEntry node, uint64_t time, bool isBlocking) {
    bool dep_fulfilled = true;
    if (isBlocking) {
      dep_fulfilled = true;
      for (auto &dep : dep_list) {
        dep_fulfilled =
            dep_fulfilled && this->checkEachDependenceFulfillment(
                                 dep, node, this->ctrl_g->position, time);
      }
    } else {
      dep_fulfilled = false;
      for (auto &dep : dep_list) {
        dep_fulfilled =
            dep_fulfilled || this->checkEachDependenceFulfillment(
                                 dep, node, this->ctrl_g->position, time);
      }
    }
    return dep_fulfilled;
  }

  // Find all vertices adjacent to given vertices in graph
  void findAdjacentVerticesToProcessed(
      std::vector<Graph::vertex_descriptor> &adjacent_vertices) {
    Graph G = this->ctrl_g->g;
    for (auto v : this->processed_vertices) {
      auto adj_set = boost::adjacent_vertices(v, G);
      for (auto v1 = adj_set.first; v1 != adj_set.second; ++v1) {
        bool found_duplicate = false;
        for (auto v2 : adjacent_vertices) {
          if (*v1 == v2) {
            found_duplicate = true;
          }
        }
        bool is_in_vertices = false;
        for (auto v3 : this->processed_vertices) {
          if (*v1 == v3) {
            is_in_vertices = true;
          }
        }
        if (!found_duplicate && !is_in_vertices) {
          adjacent_vertices.push_back(*v1);
        }
      }
    }
  }

  void buildVertexDependencyList(
      Graph::vertex_descriptor v,
      std::vector<std::pair<dependencyNodeEntry, std::string>> &dep_list) {
    Graph G = this->ctrl_g->g;
    // If current vertex is ChannelGet, then add implicit ChannelPut vertex to
    // dep list
    if (air::ChannelGetOp channel_get = dyn_cast<air::ChannelGetOp>(G[v].op)) {
      dep_list.push_back(std::make_pair(G[v], "sym"));
    }
    auto inv_adj_set = boost::inv_adjacent_vertices(v, G);
    for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second;
         ++inv_adj_v) {
      // If dependent on a hierarchy op, then push its terminator into dep_list
      // instead
      if (G[*inv_adj_v].asyncEventType == "hierarchy") {
        for (auto sub_g : G[*inv_adj_v].nextDependencyGraphs) {
          auto terminator_v = sub_g->terminator_vertex;
          auto &terminator_node = sub_g->g[terminator_v];
          dep_list.push_back(std::make_pair(terminator_node, "ssa"));
        }
      } else if (G[*inv_adj_v].asyncEventType == "for_loop") {
        pushToDepListIfAffineIfHit(dep_list, G[*inv_adj_v],
                                   this->ctrl_g->position, "ssa_loop_yield");
      } else {
        pushToDepListIfAffineIfHit(dep_list, G[*inv_adj_v],
                                   this->ctrl_g->position, "ssa");
      }
    }
  }

  // Remove ops in affine.if which aren't running on this core
  void removeOpsFilteredOutByAffineIf(
      std::vector<Graph::vertex_descriptor> &candidates) {
    Graph &G = this->ctrl_g->g;
    for (auto it = candidates.begin(); it != candidates.end(); ++it) {
      auto op = G[*it].op;
      if (op->getParentOfType<mlir::AffineIfOp>()) {
        std::vector<Operation *> affine_if_nest;
        Operation *spatial_loop = nullptr;
        getAffineIfNestAndSpatialLoopFromOp(op, affine_if_nest, spatial_loop);
        if (!positionHitsAffineIfCondition(op, spatial_loop, affine_if_nest,
                                           this->ctrl_g->position)) {
          candidates.erase(it);
          it--;
        }
      }
    }
  }

  // Try to reserve resources for an event
  bool checkResourceFulfillmentForOpImpls(dependencyNodeEntry node) {
    return checkResourceFulfillmentForOpImpls(node.op, node.asyncEventName);
  }
  bool checkResourceFulfillmentForOpImpls(Operation *op,
                                          std::string name = "") {
    if (auto Op = dyn_cast<air::PartitionOp>(op)) {
      return this->checkResourceFulfillmentForOp(Op);
    } else if (auto Op = dyn_cast<air::HerdOp>(op)) {
      return this->checkResourceFulfillmentForOp(Op);
    } else if (isa<air::ExecuteOp>(op)) {
      auto child_op = &*(op->getRegions().front().getOps().begin());
      if (name == "AllocOp") {
        auto Op = dyn_cast<memref::AllocOp>(child_op);
        return this->checkResourceFulfillmentForOp(Op);
      } else if (name == "DeallocOp") {
        auto Op = dyn_cast<memref::DeallocOp>(child_op);
        return this->checkResourceFulfillmentForOp(Op);
      }
    }
    return true;
  }

  runnerNode(runnerNode *parent = nullptr, dependencyGraph *ctrl_g = nullptr,
             std::string runner_node_type = "",
             dependencyContext *dep_ctx = nullptr,
             std::string sim_granularity = "",
             std::vector<std::pair<std::string, unsigned>>
                 *channel_token_counts_ptr = nullptr)
      : parent(parent), ctrl_g(ctrl_g), runner_node_type(runner_node_type),
        dep_ctx(dep_ctx), sim_granularity(sim_granularity),
        channel_token_counts_ptr(channel_token_counts_ptr) {}

  ~runnerNode() {
    wavefront.clear();
    processed_vertices.clear();
    loop_trip_count.clear();
    sub_runner_nodes.clear();
    channel_token_counts.clear();
  }

private:
  // Dependency graph helper functions.
  dependencyCanonicalizer canonicalizer;
  // Dependency graph context.
  xilinx::air::dependencyContext *dep_ctx;
  // Simulation granularity.
  std::string sim_granularity;
  // Each entry is an std::tuple. First element is for op's id, second element
  // is the loop's async token id, and third element is trip counter.
  std::vector<std::tuple<unsigned, unsigned, unsigned>> loop_trip_count;
  // Each entry is a std::pair. First element is the channel name, second
  // element is the token count.
  std::vector<std::pair<std::string, unsigned>> channel_token_counts;
  // Pointer to the channel token count vector. Currently, all runner nodes
  // below launch node use launch runner node's channel token count vector.
  std::vector<std::pair<std::string, unsigned>> *channel_token_counts_ptr;

  // Get a pool of available resources
  void getColumnsPool(std::vector<resource *> &resource_pool) {
    for (auto res_hier : this->resource_hiers) {
      auto dev = static_cast<device *>(res_hier);
      for (auto column : dev->columns) {
        if (!column->isReserved) {
          resource_pool.push_back(column);
        }
      }
    }
  }
  void getTilesPool(std::vector<resource *> &resource_pool) {
    for (auto res_hier : this->resource_hiers) {
      auto col = static_cast<column *>(res_hier);
      for (auto tile : col->tiles) {
        if (!tile->isReserved) {
          resource_pool.push_back(tile);
        }
      }
    }
  }
  void getColumnsPoolFromParent(std::vector<resource *> &resource_pool) {
    auto parent_runner_node = this->parent;
    parent_runner_node->getColumnsPool(resource_pool);
  }
  void getTilesPoolFromParent(std::vector<resource *> &resource_pool) {
    auto parent_runner_node = this->parent;
    parent_runner_node->getTilesPool(resource_pool);
  }
  double getMemoriesPool(std::vector<resource *> &resource_pool,
                         bool free_memory_only = true) {
    // If free_memory_only is true, then get free memory resources. If false,
    // then get used memory resources.
    if (this->runner_node_type == "launch") {
      // L3 is external memory, and is therefore not modelled in runner for
      // memory size contention.
      return DBL_MAX;
    }
    double memory_pool = 0;
    for (auto res_hier : this->resource_hiers) {
      // Get L2 memories from columns
      if (auto col = static_cast<column *>(res_hier)) {
        auto col_mem = col->column_mem;
        auto free_memory = col_mem->bytes - col_mem->bytes_used;
        // If returning free memories
        if (free_memory_only && free_memory > 0.0f) {
          resource_pool.push_back(col_mem);
          memory_pool += free_memory;
        }
        // Else if returning used memories
        else if (!free_memory_only && col_mem->bytes_used > 0.0f) {
          resource_pool.push_back(col_mem);
          memory_pool += col_mem->bytes_used;
        }
      }
      // Get L1 memories from tiles
      else if (auto til = static_cast<tile *>(res_hier)) {
        auto tile_mem = til->tile_mem;
        auto free_memory = tile_mem->bytes - tile_mem->bytes_used;
        // If returning free memories
        if (free_memory_only && free_memory > 0.0f) {
          resource_pool.push_back(tile_mem);
          memory_pool += free_memory;
        }
        // Else if returning used memories
        else if ((!free_memory_only) && tile_mem->bytes_used > 0.0f) {
          resource_pool.push_back(tile_mem);
          memory_pool += tile_mem->bytes_used;
        }
      }
    }
    return memory_pool;
  }

  // Get resource cost
  unsigned getResourceCost(Operation *op, std::string attrName) {
    unsigned usage_count = 1;
    if (op->hasAttr(attrName)) {
      auto size =
          extractFromI64ArrayAttr(op->getAttrOfType<mlir::ArrayAttr>(attrName));
      for (auto &s : size) {
        usage_count *= s;
      }
      return usage_count;
    }
    // If no resource cost attr passed, then disable resource contention
    // modelling
    // TODO: this state will become an error once resource modelling is complete
    else
      return 0;
  }
  double getMemoryCostInBytes(MemRefType ty, Operation *op) {
    // Get number of bytes per element in tensor
    double datawidth = 0;
    auto d = this->getDeviceHier();
    assert(d);
    if (auto bytes = d->datatypes[getElementTypeAsString(ty)]) {
      datawidth = bytes;
    } else {
      assert(false && "data type not found in JSON model");
    }
    // Get resource usage multipler for spatial ops which are batch dispatched
    auto spatial_op = this->getAncestorSpatialLoopFromOp(op);
    unsigned usage_multiplier = this->getBatchDispatchCount(spatial_op);
    return getTensorVolume(ty) * datawidth * usage_multiplier;
  }

  // Reserve resources
  void
  allocateRunnerNodeToResourceHiers(std::vector<resource *> resource_pool,
                                    std::vector<resource *> &reserved_resources,
                                    unsigned usage_count) {
    // A previously emitted error should have captured this
    assert(usage_count <= resource_pool.size() &&
           "failed to reserve resources");
    for (unsigned i = 0; i < usage_count; i++) {
      resource_pool[i]->isReserved = true;
      reserved_resources.push_back(resource_pool[i]);
      // Update current runner node's resource hierarchy allocation
      if (auto hier = static_cast<resourceHierarchy *>(resource_pool[i])) {
        this->resource_hiers.push_back(hier);
      }
    }
  }
  void allocateRunnerNodeToAllocateMemory(
      std::vector<resource *> resource_pool,
      std::vector<resource *> &reserved_resources, double memory_allocated) {
    double remaining = memory_allocated;
    for (auto res : resource_pool) {
      auto mem = static_cast<memory *>(res);
      assert(mem);
      auto free_memory = mem->bytes - mem->bytes_used;
      if (free_memory >= remaining) {
        mem->bytes_used += remaining;
        remaining = 0;
        reserved_resources.push_back(res);
        break;
      } else {
        mem->bytes_used = mem->bytes;
        remaining -= free_memory;
        reserved_resources.push_back(res);
        // keep going, until all memory costs are deducted
      }
    }
  }
  void allocateRunnerNodeToDeallocateMemory(
      std::vector<resource *> resource_pool,
      std::vector<resource *> &reserved_resources, double memory_deallocated) {
    double remaining = memory_deallocated;
    for (auto res : resource_pool) {
      auto mem = static_cast<memory *>(res);
      assert(mem);
      if (mem->bytes_used >= remaining) {
        mem->bytes_used -= remaining;
        remaining = 0;
        reserved_resources.push_back(res);
        break;
      } else {
        remaining -= mem->bytes_used;
        mem->bytes_used = 0;
        reserved_resources.push_back(res);
        // keep going, until enough memory has been deallocated
      }
    }
  }

  // Consume resource hierarchies when sub-runner starts
  void consumeResourceHiersWhenRunnerStarts(
      std::vector<resource *> &reserved_resources) {
    auto parent_runner_node = this->parent;
    std::string parent_runner_type = "";
    if (parent_runner_node) {
      parent_runner_type = parent_runner_node->runner_node_type;
    } else {
      parent_runner_type = "func";
    }
    auto status = parent_runner_node->checkResourceFulfillmentForOpImpls(
        this->ctrl_g->hierarchyOp);
    if (status) {
      this->allocateEventToResourcesImpls(reserved_resources);
    } else {
      this->ctrl_g->hierarchyOp->emitError(
          "Failed to allocate resources to dispatch hierarchy op");
    }
  }
  void consumeOrReleaseResources(std::vector<resource *> &reserved_resources,
                        Graph::vertex_descriptor v) {
    Graph &G = this->ctrl_g->g;
    auto status = this->checkResourceFulfillmentForOpImpls(G[v]);
    if (status) {
      this->allocateEventToResourcesImpls(reserved_resources, G[v].op,
                                          G[v].asyncEventName);
    } else {
      this->ctrl_g->hierarchyOp->emitError(
          "Failed to allocate resources to dispatch op");
    }
  }

  // Try to reserve resources for an event
  bool checkResourceFulfillmentForOp(air::PartitionOp Op) {
    std::vector<resource *> resource_hier_pool;
    this->getColumnsPool(resource_hier_pool);
    // Get resource cost
    unsigned column_count =
        this->getResourceCost(Op.getOperation(), "column_usage");
    if (column_count <= resource_hier_pool.size()) {
      return true;
    } else
      return false;
  }
  bool checkResourceFulfillmentForOp(air::HerdOp Op) {
    std::vector<resource *> resource_hier_pool;
    this->getTilesPool(resource_hier_pool);
    // Get resource cost
    unsigned tile_count = this->getBatchDispatchCount(Op.getOperation());
    if (tile_count <= resource_hier_pool.size()) {
      return true;
    } else
      return false;
  }
  bool checkResourceFulfillmentForOp(memref::AllocOp Op) {
    // Get a pool of free memories
    std::vector<resource *> resource_pool;
    double memory_pool = this->getMemoriesPool(resource_pool);
    // Get memory allocation size
    MemRefType ty = Op.getMemref().getType().cast<MemRefType>();
    double memory_allocated = this->getMemoryCostInBytes(ty, Op.getOperation());
    if (memory_allocated <= memory_pool) {
      return true;
    } else
      return false;
  }
  bool checkResourceFulfillmentForOp(memref::DeallocOp Op) {
    // Get a pool of used memories
    std::vector<resource *> resource_pool;
    double memory_pool = this->getMemoriesPool(resource_pool, false);
    // Get memory allocation size
    MemRefType ty = Op.getMemref().getType().cast<MemRefType>();
    double memory_deallocated =
        this->getMemoryCostInBytes(ty, Op.getOperation());
    if (memory_deallocated <= memory_pool) {
      return true;
    } else
      return false;
  }

  // Allocate event to resources
  void
  allocateEventToResourcesImpls(std::vector<resource *> &reserved_resources,
                                Operation *op = nullptr,
                                std::string name = "") {
    if (op) {
      if (isa<air::ExecuteOp>(op)) {
        auto child_op = &*(op->getRegions().front().getOps().begin());
        // Memory allocation/deallocation
        if (name == "AllocOp") {
          auto Op = dyn_cast<memref::AllocOp>(child_op);
          this->allocateEventToResources(Op, reserved_resources);
        } else if (name == "DeallocOp") {
          auto Op = dyn_cast<memref::DeallocOp>(child_op);
          this->allocateEventToResources(Op, reserved_resources);
        }
      }
      // Hierarchy terminator ops release resource hierarchies (devices, columns
      // or tiles)
      else if (isa<air::PartitionTerminatorOp>(op)) {
        for (auto res : this->resource_hiers) {
          res->isReserved = false;
        }
      } else if (isa<air::HerdTerminatorOp>(op)) {
        for (auto res : this->resource_hiers) {
          res->isReserved = false;
        }
      }
    } else {
      if (auto Op = dyn_cast<air::PartitionOp>(this->ctrl_g->hierarchyOp)) {
        this->allocateEventToResources(Op, reserved_resources);
      } else if (auto Op = dyn_cast<air::HerdOp>(this->ctrl_g->hierarchyOp)) {
        this->allocateEventToResources(Op, reserved_resources);
      }
    }
  }
  void allocateEventToResources(air::PartitionOp Op,
                                std::vector<resource *> &reserved_resources) {
    std::vector<resource *> resource_hier_pool;
    // Get resource pool
    this->getColumnsPoolFromParent(resource_hier_pool);
    // Get resource cost
    unsigned column_count = this->getResourceCost(Op, "column_usage");
    // Reserve resource
    this->allocateRunnerNodeToResourceHiers(resource_hier_pool,
                                            reserved_resources, column_count);
  }
  void allocateEventToResources(air::HerdOp Op,
                                std::vector<resource *> &reserved_resources) {
    std::vector<resource *> resource_hier_pool;
    // Get resource pool
    this->getTilesPoolFromParent(resource_hier_pool);
    // Get resource cost
    unsigned tile_count = this->getBatchDispatchCount(Op.getOperation());
    // Reserve resource
    this->allocateRunnerNodeToResourceHiers(resource_hier_pool,
                                            reserved_resources, tile_count);
  }
  void allocateEventToResources(memref::AllocOp Op,
                                std::vector<resource *> &reserved_resources) {
    // Get a pool of free memories
    std::vector<resource *> resource_pool;
    this->getMemoriesPool(resource_pool);
    // Get memory size in bytes
    MemRefType ty = Op.getMemref().getType().cast<MemRefType>();
    double memory_allocated = this->getMemoryCostInBytes(ty, Op.getOperation());
    // Reserve resource
    this->allocateRunnerNodeToAllocateMemory(resource_pool, reserved_resources,
                                             memory_allocated);
  }
  void allocateEventToResources(memref::DeallocOp Op,
                                std::vector<resource *> &reserved_resources) {
    // Get a pool of used memories
    std::vector<resource *> resource_pool;
    this->getMemoriesPool(resource_pool, false);
    // Get memory size in bytes
    MemRefType ty = Op.getMemref().getType().cast<MemRefType>();
    double memory_deallocated =
        this->getMemoryCostInBytes(ty, Op.getOperation());
    // Reserve resource
    this->allocateRunnerNodeToDeallocateMemory(
        resource_pool, reserved_resources, memory_deallocated);
  }

  void executeOp(xilinx::air::HierarchyInterface op, uint64_t time,
                 runnerNode *sub_runner_node, Graph::vertex_descriptor it) {
    // Initialize sub runner and sub graph prior to execution
    Graph &G = sub_runner_node->ctrl_g->g;
    auto sub_start_v = sub_runner_node->ctrl_g->start_vertex;
    auto sub_terminator_v = sub_runner_node->ctrl_g->terminator_vertex;
    sub_runner_node->resetGraphBetweenTwoVertices(sub_start_v, sub_terminator_v,
                                                  G, time);
    sub_runner_node->loop_trip_count.clear();

    // Start sub-runner node by pushing start node into its wavefront
    sub_runner_node->ctrl_g->g[sub_start_v].start_time = time;
    sub_runner_node->ctrl_g->g[sub_start_v].end_time = time;
    assert(!sub_runner_node->wavefront.size() && "Sub runner node is busy");
    sub_runner_node->pushStartToWavefront(sub_start_v);

    sub_runner_node->processed_vertices.clear();

    this->processed_vertices.push_back(it);
  }

  void executeOp(scf::YieldOp op, uint64_t time, scf::ForOp for_op,
                 Graph::vertex_descriptor it) {
    Graph &G = this->ctrl_g->g;

    // Get async tokens ready to iterate at scf.yield
    std::vector<unsigned> token_ids;
    std::vector<bool> token_is_still_iterating;
    this->getReadyTokensAtScfYield(token_ids, op, time, G);

    // For loop trip counter
    bool trip_count_fulfilled = false;
    for (auto &count_entry : this->loop_trip_count) {
      if (std::get<0>(count_entry) ==
          (unsigned)getIdAttr(for_op.getOperation())) {
        for (int i = token_ids.size() - 1; i >= 0; i--) {
          if (std::get<1>(count_entry) == token_ids[i]) {
            if (std::get<2>(count_entry)) {
              // Decrement token count if this token still needs to iterate
              std::get<2>(count_entry)--;
            }
            if (!std::get<2>(count_entry)) {
              // If this token's iteration cound is fulfilled, then delete this
              // token from ready token list
              token_is_still_iterating.push_back(false);
            } else {
              token_is_still_iterating.push_back(true);
            }
          }
        }
      }
    }

    // If all async tokens' trip counts are fulfilled
    bool allAsyncTokensFulfilled = true;
    for (auto &count_entry : this->loop_trip_count) {
      if (std::get<0>(count_entry) ==
          (unsigned)getIdAttr(for_op.getOperation())) {
        if (std::get<2>(count_entry)) {
          allAsyncTokensFulfilled = false;
        }
      }
    }

    if (allAsyncTokensFulfilled) {
      this->processed_vertices.push_back(it);
      trip_count_fulfilled = true;
    } else {
      // If trip count unfulfilled, then iterate.
      // Clear start_time and end_time of all ops in loop body.
      // From processed_vertices, remove all ops which are in loop body.
      for (unsigned i = 0; i < token_ids.size(); i++) {
        // Get the yielded token in the next loop iteration (at the beginning of
        // the loop)
        auto next_iter_token = for_op.getRegionIterArgs()[token_ids[i]];
        assert(next_iter_token);

        // Search for vertices corresponding to the next-iteration incarnations
        // of this token
        auto for_v = this->canonicalizer
                         .getVertexFromOp(for_op.getOperation(),
                                          *(this->dep_ctx), "front")
                         .first;
        auto adj_set = boost::adjacent_vertices(for_v, G);
        for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
          auto adj_op = G[*adj_v].op;
          assert(adj_op);
          for (auto d : adj_op->getOperands()) {
            if (d == next_iter_token) {
              // To start the next loop iteration:
              // (1) reset graph wrt this token
              this->resetGraphBetweenTwoVertices(*adj_v, it, G, time);
              // (2) release the token locks, if the token is still iterating
              if (token_is_still_iterating[i]) {
                G[*adj_v].token_count += this->tokenSpatialFactor(G[*adj_v].op);
              }
            }
          }
        }
      }
    }
  }

  void executeOp(scf::ForOp op, Graph::vertex_descriptor it) {
    // Get for loop trip count
    auto lb = op.getLowerBound().getDefiningOp();
    int64_t lbv = cast<arith::ConstantIndexOp>(lb).value();
    auto ub = op.getUpperBound().getDefiningOp();
    int64_t ubv = cast<arith::ConstantIndexOp>(ub).value();
    auto step = op.getStep().getDefiningOp();
    int64_t stepv = cast<arith::ConstantIndexOp>(step).value();

    // (ubv - lbv) / stepv, fast round up
    int64_t trip_count = (ubv - lbv + stepv - 1) / stepv;

    // Update for loop trip count per async token
    for (unsigned i = 0; i < op.getRegionIterArgs().size(); i++) {
      this->loop_trip_count.push_back(
          std::make_tuple(getIdAttr(op.getOperation()), i, trip_count));
    }

    // Release the locks for all async tokens adjacent to scf.for, to initiate
    // the first iteration.
    Graph &G = this->ctrl_g->g;
    auto adj_set = boost::adjacent_vertices(it, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
      G[*adj_v].token_count +=
          this->tokenCountThresholdForExecution(G[*adj_v].op) *
          this->tokenSpatialFactor(
              G[*adj_v].op); // Lock number = number of dependent iter_args
    }

    this->processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelPutOp op, Graph::vertex_descriptor it) {
    auto spatial_factor = this->tokenSpatialFactor(op.getOperation());
    bool found_entry = false;
    for (auto &entry : *channel_token_counts_ptr) {
      if ((!found_entry) && entry.first == op.getChanName().str()) {
        entry.second += spatial_factor;
        found_entry = true;
      }
    }
    if (!found_entry) {
      channel_token_counts_ptr->push_back(
          std::make_pair(op.getChanName().str(), spatial_factor));
    }

    this->processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelGetOp op, Graph::vertex_descriptor it) {
    // Get spatial factor for op
    auto spatial_factor = this->tokenSpatialFactor(op.getOperation());

    bool found_entry = false;
    // for (auto &entry : launch_runner_node.channel_token_count) {
    for (auto &entry : *channel_token_counts_ptr) {
      if ((!found_entry) && entry.first == op.getChanName().str()) {
        entry.second -= spatial_factor;
        found_entry = true;
      }
    }
    assert(found_entry && "Cannot find channel symbol name in launch runner");

    this->processed_vertices.push_back(it);
  }

  void executeOp(Graph::vertex_descriptor it) {
    this->processed_vertices.push_back(it);
  }

  // Adds pointer between runner node and command graph
  void addPointerBetweenSubRunnerNodeAndSubCommandGraph() {
    for (auto r_it = std::begin(this->sub_runner_nodes);
         r_it != std::end(this->sub_runner_nodes); ++r_it) {
      r_it->ctrl_g->runner_node = &(*r_it);
    }
  }

  // Reset a vertex in dependency graph
  void resetVertex(Graph::vertex_descriptor v, Graph &G, uint64_t time) {

    // Remove start_v from processed_vertices
    this->removeVertexFromVertices(this->processed_vertices, v);

    // Reset node's start_time and end_time, if the async event represented by
    // the vertex is complete
    if (G[v].is_started() && G[v].is_done(time)) {
      G[v].start_end_time_log.push_back(
          std::make_pair(G[v].start_time, G[v].end_time));
      G[v].start_time = 0;
      G[v].end_time = 0;
    }
  }

  // Remove a vertex from a vector of vertices
  void removeVertexFromVertices(std::vector<Graph::vertex_descriptor> &vector,
                                Graph::vertex_descriptor a) {
    if (vector.size()) {
      for (auto it = vector.begin(); it != vector.end(); ++it) {
        if (*it == a) {
          vector.erase(it);
          it--;
        }
      }
    }
  }

  bool hasPath(Graph::vertex_descriptor start_v, Graph::vertex_descriptor end_v,
               Graph &G, SmallVector<Graph::vertex_descriptor, 1> &vec) {

    vec.push_back(start_v);
    if (start_v == end_v)
      return true;
    int pathCount = 0;
    auto adj_set = boost::adjacent_vertices(start_v, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
      SmallVector<Graph::vertex_descriptor, 1> tmp_vec;
      if (this->hasPath(*adj_v, end_v, G, tmp_vec)) {
        pathCount++;
        // Concatenate
        vec.insert(vec.end(), tmp_vec.begin(), tmp_vec.end());
      }
    }
    if (pathCount)
      return true;
    vec.pop_back();
    return false;
  }

  // Get a vector of async tokens which are ready to advance to the next loop
  // iteration at scf.yield
  void getReadyTokensAtScfYield(std::vector<unsigned> &token_ids,
                                scf::YieldOp op, uint64_t time, Graph &G) {

    unsigned token_id = 0;
    for (auto operand : op->getOperands()) {
      // With scf.for possibly having multiple tokens, check for the token ids
      // which are ready to advance to the next iteration
      auto dep_op = operand.getDefiningOp();
      auto dep_entry =
          canonicalizer.getVertexFromOp(dep_op, *(this->dep_ctx), "front");
      auto &dep_node = G[dep_entry.first];

      // Check each token's dependence fulfillment at scf.yield
      std::string node_type = "ssa";
      auto dep_pair_entry = std::make_pair(dep_node, node_type);
      if (checkEachDependenceFulfillment(dep_pair_entry, time)) {
        token_ids.push_back(token_id);
      }

      token_id++;
    }
  }

  // Check if a channel dependence has been fulfilled
  bool checkChannelDependenceFulfillment(dependencyNodeEntry dep_node,
                                         std::vector<unsigned> position) {
    auto channel_op = dyn_cast<air::ChannelInterface>(dep_node.op);
    assert(channel_op);
    std::string chan_name = channel_op.getChanName().str();
    unsigned th = (position.size())
                      ? (this->tokenSpatialFactor(dep_node.op, position))
                      : (1);
    bool found_entry = false;
    for (auto entry : *channel_token_counts_ptr) {
      if ((!found_entry) && entry.first == chan_name) {
        found_entry = true;
        if (entry.second < th) {
          return false;
        }
      }
    }
    if (!found_entry) {
      return false;
    }
    return true;
  }

  // Check if a dependence has been fulfilled
  bool checkEachDependenceFulfillment(
      std::pair<dependencyNodeEntry, std::string> &dep,
      dependencyNodeEntry &node, std::vector<unsigned> position,
      uint64_t time) {
    dependencyNodeEntry &dep_node = dep.first;
    if (dep.second == "ssa") {
      if ((!dep_node.is_started()) || (!dep_node.is_done(time))) {
        // If source and sink of dep are both under the same loop
        if (node.op && dep_node.op &&
            this->shareInnerMostForLoop(node.op, dep_node.op)) {
          // Check node's timestamp log, in case if it has executed in previous
          // loop iterations
          unsigned dep_iter_count = dep_node.start_end_time_log.size();
          unsigned node_iter_count = node.start_end_time_log.size();
          if (node.is_started() && node.is_done(time))
            node_iter_count++;
          if (dep_iter_count <= node_iter_count) {
            return false;
          }
        } else
          return false;
      }
    } else if (dep.second == "ssa_loop_yield") {
      // Threshold token_count for dep fulfillment = how many iter_args does
      // node depend on
      int th = this->tokenCountThresholdForExecution(node.op);
      if (node.token_count < th * this->tokenSpatialFactor(node.op, position)) {
        return false;
      }
    } else if (dep.second == "sym") {
      if (!checkChannelDependenceFulfillment(dep_node, position)) {
        return false;
      }
    } else {
      assert(false && "Unknown async token type");
    }
    return true;
  }

  // Check if a dependence has been fulfilled
  bool checkEachDependenceFulfillment(
      std::pair<dependencyNodeEntry, std::string> &dep, uint64_t time) {
    if (dep.second == "ssa") {
      assert(dep.first.start_time >= 0);
      if ((!dep.first.is_started()) || (!dep.first.is_done(time))) {
        // If source and sink of dep are both under the same loop
        return false;
      }
    } else if (dep.second == "ssa_loop_yield") {
      // Threshold token_count for dep fulfillment = how many iter_args does
      // node depend on
      return false;
    } else if (dep.second == "sym") {
      dependencyNodeEntry &dep_node = dep.first;
      if (!this->checkChannelDependenceFulfillment(dep_node, {})) {
        return false;
      }
    } else {
      assert(false && "Unknown async token type");
    }
    return true;
  }

  // Check if two ops are under the same scf for loop
  bool shareInnerMostForLoop(Operation *a, Operation *b) {
    if (a) {
      if (auto a_parent_for = a->getParentOfType<scf::ForOp>()) {
        if (b) {
          if (auto b_parent_for = b->getParentOfType<scf::ForOp>()) {
            if (a_parent_for == b_parent_for) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  // Calculate the number of spatially parallel tokens produced/consumed per op
  unsigned tokenSpatialFactor(Operation *op) {
    unsigned output = 1;
    // If op is producer to a channel broadcast, then bump up token count by
    // fanout
    if (isa<air::ChannelPutOp>(op)) {
      auto channel_op = dyn_cast<air::ChannelInterface>(op);
      auto chan = getChannelDeclarationThroughSymbol(channel_op);
      if (chan->hasAttr("broadcast_shape")) {
        auto size = extractFromI64ArrayAttr(chan->getAttr("broadcast_shape"));
        for (auto s : size) {
          output *= s;
        }
        size = extractFromI64ArrayAttr(chan.getSize());
        for (auto s : size) {
          output /= s;
        }
      }
    }
    for (auto parent = op->getParentOp(); !isa<func::FuncOp>(parent);
         parent = parent->getParentOp()) {
      if (auto scf_par = dyn_cast<scf::ParallelOp>(parent)) {
        for (unsigned i = 0; i < scf_par.getNumLoops(); i++) {
          auto lbCstOp = scf_par.getLowerBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto ubCstOp = scf_par.getUpperBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto stepCstOp =
              scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
          int64_t tripCount = mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(),
                                            stepCstOp.value());
          output *= tripCount;
        }
      } else if (auto hier = dyn_cast<air::HierarchyInterface>(parent)) {
        if (this->sim_granularity == "core" && isa<air::HerdOp>(parent)) {
        } else {
          output *= canonicalizer.getTripCountInHierarchyOp(hier);
        }
      } else if (auto affine_if = dyn_cast<mlir::AffineIfOp>(parent)) {
        // Fast forward through affine.if nest
        std::vector<Operation *> affine_if_nest;
        Operation *spatial_loop = nullptr;
        parent = getAffineIfNestAndSpatialLoopFromOp(parent, affine_if_nest,
                                                     spatial_loop);

        // If showing cores
        auto herd = dyn_cast<air::HerdOp>(spatial_loop);
        if (herd && this->sim_granularity == "core") {
          output =
              (positionHitsAffineIfCondition(op, spatial_loop, affine_if_nest,
                                             this->ctrl_g->position))
                  ? (output)
                  : (0);
        } else {
          unsigned size =
              this->getSizeThroughAffineIf(op, spatial_loop, affine_if_nest);
          output *= size;
        }
      }
    }
    return output;
  }

  // Calculate the number of spatially parallel tokens produced/consumed per op
  unsigned tokenSpatialFactor(Operation *op, std::vector<unsigned> position) {
    unsigned output = 1;
    // If op is producer to a channel broadcast, then bump up token count by
    // fanout
    if (isa<air::ChannelPutOp>(op)) {
      auto channel_op = dyn_cast<air::ChannelInterface>(op);
      auto chan = getChannelDeclarationThroughSymbol(channel_op);
      if (chan->hasAttr("broadcast_shape")) {
        auto size = extractFromI64ArrayAttr(chan->getAttr("broadcast_shape"));
        for (auto s : size) {
          output *= s;
        }
        size = extractFromI64ArrayAttr(chan.getSize());
        for (auto s : size) {
          output /= s;
        }
      }
    }
    for (auto parent = op->getParentOp(); !isa<func::FuncOp>(parent);
         parent = parent->getParentOp()) {
      if (auto scf_par = dyn_cast<scf::ParallelOp>(parent)) {
        for (unsigned i = 0; i < scf_par.getNumLoops(); i++) {
          auto lbCstOp = scf_par.getLowerBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto ubCstOp = scf_par.getUpperBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto stepCstOp =
              scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
          int64_t tripCount = mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(),
                                            stepCstOp.value());
          output *= tripCount;
        }
      } else if (auto hier = dyn_cast<air::HierarchyInterface>(parent)) {
        if (this->sim_granularity == "core" && isa<air::HerdOp>(parent)) {
        } else {
          output *= this->canonicalizer.getTripCountInHierarchyOp(hier);
        }
      } else if (auto affine_if = dyn_cast<mlir::AffineIfOp>(parent)) {
        // Fast forward through affine.if nest
        std::vector<Operation *> affine_if_nest;
        Operation *spatial_loop = nullptr;
        parent = getAffineIfNestAndSpatialLoopFromOp(parent, affine_if_nest,
                                                     spatial_loop);

        // If showing cores
        auto herd = dyn_cast<air::HerdOp>(spatial_loop);
        if (herd && this->sim_granularity == "core") {
          output = (positionHitsAffineIfCondition(op, spatial_loop,
                                                  affine_if_nest, position))
                       ? (output)
                       : (0);
        } else {
          unsigned size =
              this->getSizeThroughAffineIf(op, spatial_loop, affine_if_nest);
          output *= size;
        }
      }
    }
    return output;
  }

  // Async for loop race condition: calculate the minimum number of tokens
  // required for dep fulfillment
  unsigned tokenCountThresholdForExecution(Operation *op) {
    // Threshold token_count for dep fulfillment = how many iter_args does node
    // depend on
    unsigned th = 0;
    if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
      for (auto token : async_op.getAsyncDependencies()) {
        if (getForRegionIterArgsOwner(token)) {
          th++;
        }
      }
    } else {
      th = 1;
    }
    return th;
  }

  // Walk affine.if then and else blocks and infer block sizes of op's ancestor
  unsigned getSizeThroughAffineIf(Operation *op, Operation *spatial_loop,
                                  std::vector<Operation *> affine_if_nest) {
    unsigned output = 1;
    SmallVector<int, 2> lbs_spatial;
    SmallVector<int, 2> ubs_spatial;
    getSizesFromSpatialLoop(spatial_loop, lbs_spatial, ubs_spatial);

    // Walk through affine.if nest (in reverse order through vector)
    for (auto it = affine_if_nest.rbegin(); it != affine_if_nest.rend(); ++it) {
      auto affine_if = dyn_cast<mlir::AffineIfOp>(*it);
      // Get then integerset sizes
      SmallVector<int, 2> lbs_int = {0, 0};
      SmallVector<int, 2> ubs_int = {0, 0};
      IntegerSet int_set = affine_if.getIntegerSet();
      getSizesFromIntegerSet(affine_if->getContext(), int_set, lbs_int,
                             ubs_int);
      // If found then block containing op
      if (affine_if.getThenBlock()->findAncestorOpInBlock(*op)) {
        for (unsigned i = 0; i < lbs_int.size(); i++) {
          output *= ubs_int[i] - lbs_int[i] + 1;
        }
        return output;
      }
      // Else keep going, while updating the spatial sizes wrt else condition
      else {
        getElseSizesFromAffineIf(lbs_spatial, ubs_spatial, lbs_int, ubs_int);
      }
    }
    // If op isn't in any then blocks in affine.if nest
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      output *= ubs_spatial[i] - lbs_spatial[i] + 1;
    }
    return output;
  }

  // Get dispatch size of spatial op
  unsigned getResourceUsageMultiplier(Operation *op,
                                      bool dispatchesSingleResource = true) {
    unsigned resource_usage = 1;
    if (dispatchesSingleResource) {
      return resource_usage;
    }
    // Are iterations of an op dispatched individually?
    // TODO: formalize for scf.parallel, air.launch, air.partition and air.herd.
    SmallVector<int, 2> lbs_spatial;
    SmallVector<int, 2> ubs_spatial;
    getSizesFromSpatialLoop(op, lbs_spatial, ubs_spatial);
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      resource_usage *= ubs_spatial[i] - lbs_spatial[i] + 1;
    }
    return resource_usage;
  }

  // Check if spatial op is batch-dispatched in current simulation granularity
  unsigned getBatchDispatchCount(Operation *op) {
    if (isa<air::HerdOp>(op)) {
      if (this->sim_granularity == "core") {
        return this->getResourceUsageMultiplier(op, true);
      } else if (this->sim_granularity == "herd") {
        return this->getResourceUsageMultiplier(op, false);
      }
      // TODO: add other simulation granularities
    } else if (isa<air::PartitionOp>(op)) {
      return 1;
    } else if (isa<air::LaunchOp>(op)) {
      return 1;
    } else if (isa<scf::ParallelOp>(op)) {
      return 1;
    }
    assert(false && "TODO: add other simulation granularities");
    return 1;
  }

  bool pushToDepListIfAffineIfHit(
      std::vector<std::pair<dependencyNodeEntry, std::string>> &dep_list,
      dependencyNodeEntry &node, std::vector<unsigned> position,
      std::string dep_type = "") {
    bool pushed = false;
    if (this->sim_granularity == "core" && node.op &&
        node.op->getParentOfType<mlir::AffineIfOp>()) {
      std::vector<Operation *> affine_if_nest;
      Operation *spatial_loop = nullptr;
      getAffineIfNestAndSpatialLoopFromOp(node.op, affine_if_nest,
                                          spatial_loop);
      if (positionHitsAffineIfCondition(node.op, spatial_loop, affine_if_nest,
                                        this->ctrl_g->position)) {
        dep_list.push_back(std::make_pair(node, dep_type));
        pushed = true;
      }
    } else {
      dep_list.push_back(std::make_pair(node, dep_type));
      pushed = true;
    }
    return pushed;
  }

  // Get parent launch runner node
  runnerNode *getParentLaunchRunner() {
    runnerNode *parent_runner = this;
    while (parent_runner->runner_node_type != "launch") {
      if (!parent_runner->parent) {
        return nullptr;
      }
      parent_runner = parent_runner->parent;
    }
    return parent_runner;
  }

  // Get device resource hierarchy
  device *getDeviceHier() {
    runnerNode *parent_launch = this->getParentLaunchRunner();
    if (!parent_launch)
      return nullptr;
    // TODO: separate device properties (datatypes) from device resoruces
    auto d = static_cast<device *>(parent_launch->resource_hiers[0]);
    return d;
  }

  // Get ancestor spatial loop from op
  Operation *getAncestorSpatialLoopFromOp(Operation *op) {
    Operation *parent = op;
    while ((!isa<scf::ParallelOp>(parent)) &&
           (!isa<air::HierarchyInterface>(parent))) {
      parent = parent->getParentOp();
      if (isa<func::FuncOp>(parent)) {
        return nullptr;
      }
    }
    return parent;
  }

}; // runnerNode

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_NODE