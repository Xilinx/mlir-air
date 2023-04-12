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
  // Resource hierarchies which are allocated to this runner node
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
    for (auto &segmentGraph : launchGraph.subgraphs) {
      // Create segment runner node
      this->sub_runner_nodes.push_back(
          runnerNode(this, &segmentGraph, "segment", this->dep_ctx,
                     this->sim_granularity,
                     &(launchGraph.runner_node->channel_token_counts)));
      auto current_segment_node = &(this->sub_runner_nodes.back());
      for (auto &herdGraph : segmentGraph.subgraphs) {
        // Create herd runner node
        current_segment_node->sub_runner_nodes.push_back(
            runnerNode(current_segment_node, &herdGraph, "herd",
                       this->dep_ctx, this->sim_granularity,
                       &(launchGraph.runner_node->channel_token_counts)));
      }
    }
    this->addPointerBetweenSubRunnerNodeAndSubCommandGraph();
    for (auto &segment_runner_node : this->sub_runner_nodes) {
      segment_runner_node.addPointerBetweenSubRunnerNodeAndSubCommandGraph();
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
               getScfParentOpFromYieldOp<scf::ForOp>(node.op)) {
      auto Op = dyn_cast<scf::YieldOp>(node.op);
      auto parent_for_op =
          dyn_cast<scf::ForOp>(getScfParentOpFromYieldOp<scf::ForOp>(node.op));
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
        assert(G[it].token_count >= th * this->tokenSpatialFactorForDependency(G[it].op));
        G[it].token_count -= th * this->tokenSpatialFactorForDependency(G[it].op);
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
    // At any point in time, if segment or herd op fails to allocate enough
    // resources, then the entire launch is invalid due to failing to allocate
    // enough resources upon launch.
    if (auto Op = dyn_cast<air::SegmentOp>(op)) {
      bool result = this->checkResourceFulfillmentForOp(Op);
      if (!result) {
        op->emitOpError("isn't allocated with enough resources to run");
      }
      return result;
    } else if (auto Op = dyn_cast<air::HerdOp>(op)) {
      bool result = this->checkResourceFulfillmentForOp(Op);
      if (!result) {
        op->emitOpError("isn't allocated with enough resources to run");
      }
      return result;
    }
    // If the ops below fails to be allocated with enough resources, then defer their execution until enough resources are freed up.
    else if (auto Op = dyn_cast<air::ChannelPutOp>(op)){
      return (bool)this->checkResourceFulfillmentForOp(Op);
    }
    else if (auto Op = dyn_cast<air::ChannelGetOp>(op)){
      return (bool)this->checkResourceFulfillmentForOp(Op);
    }
    else if (isa<air::ExecuteOp>(op)) {
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
    resource_hiers.clear();
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
  // A log of operations which are partially dispatched due to resource contention.
  // Keys: op pointer; mapped: # of ops dispatched.
  std::map<Operation *, std::pair<unsigned, std::vector<resource *>>> ops_in_progress;

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
  void getPortsPool(std::vector<resource *> &resource_pool, unsigned memory_space_as_int) {
    auto mem_str = lookUpMemorySpaceFromInt(memory_space_as_int);
    for (auto res_hier : this->resource_hiers) {
      // Get ports from devices
      if (this->runner_node_type == "launch") {
        auto dev = static_cast<device *>(res_hier);
        auto dev_ports = dev->ports;
        for (auto p : dev_ports[mem_str]){
          if (!p->isReserved) {
            resource_pool.push_back(p);
          }
        }
      }
      // Get ports from columns
      else if (this->runner_node_type == "segment") {
        auto col = static_cast<column *>(res_hier);
        auto col_ports = col->ports;
        for (auto p : col_ports[mem_str]){
          if (!p->isReserved) {
            resource_pool.push_back(p);
          }
        }
      }
      // Get ports from tiles
      else if (this->runner_node_type == "herd") {
        auto til = static_cast<tile *>(res_hier);
        auto til_ports = til->ports;
        for (auto p : til_ports[mem_str]){
          if (!p->isReserved) {
            resource_pool.push_back(p);
          }
        }
      }
    }
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
  void allocateRunnerNodeToPorts(
      std::vector<resource *> resource_pool,
      std::vector<resource *> &reserved_resources, unsigned usage_count) {
    for (unsigned i = 0; i < usage_count; i++) {
      resource_pool[i]->isReserved = true;
      reserved_resources.push_back(resource_pool[i]);
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
    this->allocateEventToResourcesImpls(reserved_resources);
  }
  void consumeOrReleaseResources(std::vector<resource *> &reserved_resources,
                                 Graph::vertex_descriptor v) {
    Graph &G = this->ctrl_g->g;
    this->allocateEventToResourcesImpls(reserved_resources, G[v].op,
                                        G[v].asyncEventName);
  }

  // Try to reserve resources for an event
  bool checkResourceFulfillmentForOp(air::SegmentOp Op) {
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
    // Note: forced to use dispatch multiplier to get tile count, since it is
    // checking for the entire herd op.
    unsigned tile_count = this->getBatchDispatchCount(Op.getOperation(), true);
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
  // Return how many events can be dispatched at this point in time.
  unsigned checkResourceFulfillmentForOp(air::ChannelPutOp putOp) {

    // std::cout << "Start checking for put\n";

    // Get src memory spaces
    MemRefType srcTy = putOp.getSrc().getType().cast<MemRefType>();
    auto srcSpace = srcTy.getMemorySpaceAsInt();

    // Get a pool of available ports from src
    std::vector<resource *> src_resource_pool;
    this->getPortsPool(src_resource_pool, srcSpace);

    // Get launch runner node
    auto launch_runner = this;
    while(launch_runner->runner_node_type != "launch"){
      launch_runner = launch_runner->parent;
    }

    // Check how many remaining dispatches are there for this dynamically dispatched event
    unsigned remaining = launch_runner->getRemainingDispatchesForDynamicDispatch(putOp.getOperation());

    // std::cout << "remaining: " << remaining << "\n";
    // std::cout << "pool size: " << src_resource_pool.size() << "\n";

    // std::cout << "Finish checking for put\n";

    return std::min(remaining, (unsigned)src_resource_pool.size());
  }
  unsigned checkResourceFulfillmentForOp(air::ChannelGetOp getOp) {

    // std::cout << "Start checking for get\n";

    // Get dst memory spaces
    MemRefType dstTy = getOp.getDst().getType().cast<MemRefType>();
    auto dstSpace = dstTy.getMemorySpaceAsInt();

    // Get a pool of available ports from dst
    std::vector<resource *> dst_resource_pool;
    this->getPortsPool(dst_resource_pool, dstSpace);

    // Get launch runner node
    auto launch_runner = this;
    while(launch_runner->runner_node_type != "launch"){
      launch_runner = launch_runner->parent;
    }

    unsigned remaining = launch_runner->getRemainingDispatchesForDynamicDispatch(getOp);

    // std::cout << "remaining: " << remaining << "\n";
    // std::cout << "pool size: " << dst_resource_pool.size() << "\n";

    // std::cout << "Finish checking for get\n";

    return std::min(remaining, (unsigned)dst_resource_pool.size());
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
      else if (isa<air::SegmentTerminatorOp>(op)) {
        for (auto res : this->resource_hiers) {
          res->isReserved = false;
        }
      } else if (isa<air::HerdTerminatorOp>(op)) {
        for (auto res : this->resource_hiers) {
          res->isReserved = false;
        }
      }
      else if (auto Op = dyn_cast<air::ChannelPutOp>(op)){
        this->allocateEventToResources(Op, reserved_resources);
      }
      else if (auto Op = dyn_cast<air::ChannelGetOp>(op)){
        this->allocateEventToResources(Op, reserved_resources);
      }
    } else {
      if (auto Op = dyn_cast<air::SegmentOp>(this->ctrl_g->hierarchyOp)) {
        this->allocateEventToResources(Op, reserved_resources);
      } else if (auto Op = dyn_cast<air::HerdOp>(this->ctrl_g->hierarchyOp)) {
        this->allocateEventToResources(Op, reserved_resources);
      }
    }
  }
  void allocateEventToResources(air::SegmentOp Op,
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
  void allocateEventToResources(air::ChannelPutOp Op,
                                std::vector<resource *> &reserved_resources){
    // Get src memory spaces
    MemRefType srcTy = Op.getSrc().getType().cast<MemRefType>();
    auto srcSpace = srcTy.getMemorySpaceAsInt();
    
    auto chan_interface = dyn_cast<air::ChannelInterface>(Op.getOperation());
    unsigned dispatched = 0;

    // Check how many evnets need to be dispatched in this op
    unsigned total = this->tokenSpatialFactorForResource<air::HierarchyInterface>(Op.getOperation(), {});
    this->allocateEventToResources(chan_interface, reserved_resources, srcSpace, dispatched);

    // Update tokens
    auto spatial_factor = this->tokenSpatialFactorForDependency(Op.getOperation());
    // TODO: Unify tokenSpatialFactorForDependency with tokenSpatialFactorForResource
    // Current solution: use proportion of dispatch progress to discount token spatial factor
    spatial_factor = spatial_factor * dispatched / total;
    bool found_entry = false;
    for (auto &entry : *channel_token_counts_ptr) {
      if ((!found_entry) && entry.first == Op.getChanName().str()) {
        entry.second += spatial_factor;
        found_entry = true;
      }
    }
    if (!found_entry) {
      channel_token_counts_ptr->push_back(
          std::make_pair(Op.getChanName().str(), spatial_factor));
    }
  }
  void allocateEventToResources(air::ChannelGetOp Op,
                                std::vector<resource *> &reserved_resources){
    // Get dst memory spaces
    MemRefType dstTy = Op.getDst().getType().cast<MemRefType>();
    auto dstSpace = dstTy.getMemorySpaceAsInt();
    
    auto chan_interface = dyn_cast<air::ChannelInterface>(Op.getOperation());
    unsigned dispatched = 0;
    // Check how many evnets need to be dispatched in this op
    unsigned total = this->tokenSpatialFactorForResource<air::HierarchyInterface>(Op.getOperation(), {});
    this->allocateEventToResources(chan_interface, reserved_resources, dstSpace, dispatched);

    // Update tokens
    auto spatial_factor = this->tokenSpatialFactorForDependency(Op.getOperation());
    spatial_factor = spatial_factor * dispatched / total;

    bool found_entry = false;
    for (auto &entry : *channel_token_counts_ptr) {
      if ((!found_entry) && entry.first == Op.getChanName().str()) {
        entry.second -= spatial_factor;
        found_entry = true;
      }
    }
    assert(found_entry && "Cannot find channel symbol name in launch runner");
  }

  void allocateEventToResources(air::ChannelInterface Op,
                                std::vector<resource *> &reserved_resources, unsigned memSpace, unsigned &dispatched) {
    
    if (isa<air::ChannelPutOp>(Op.getOperation())){
      // std::cout << "Start allocating to put\n";
    }
    if (isa<air::ChannelGetOp>(Op.getOperation())){
      // std::cout << "Start allocating to get\n";
    }

    // Get a pool of available ports
    std::vector<resource *> resource_pool;
    this->getPortsPool(resource_pool, memSpace);

    // Get launch runner node
    auto launch_runner = this;
    while(launch_runner->runner_node_type != "launch"){
      launch_runner = launch_runner->parent;
    }

    // Check how many remaining dispatches for get op, by checking the progress difference to put op
    int remaining = 0;
    if (auto getOp = dyn_cast<air::ChannelGetOp>(Op.getOperation())){
      remaining = launch_runner->getRemainingDispatchesForDynamicDispatch(getOp);
    }
    else if (isa<air::ChannelPutOp>(Op.getOperation())){
      remaining = launch_runner->getRemainingDispatchesForDynamicDispatch(Op.getOperation());
    }

    // Check how many events can be dispatched at this time
    dispatched = std::min(remaining, (int)resource_pool.size());

    if (isa<air::ChannelGetOp>(Op.getOperation())){
      // std::cout << "remaining: " << remaining << "\n";
    }

    // Dispatch all events that can be dispatched
    this->allocateRunnerNodeToPorts(
        resource_pool, reserved_resources, dispatched);

    // Keep track of how many remaining events to dispatch in this op
    if (launch_runner->ops_in_progress.count(Op.getOperation())){
      launch_runner->ops_in_progress[Op.getOperation()].first += dispatched;
      for (auto res : reserved_resources){
        launch_runner->ops_in_progress[Op.getOperation()].second.push_back(res);
      }
    }
    else {
      std::vector<resource *> res = reserved_resources;
      auto entry = std::make_pair(dispatched, res);
      launch_runner->ops_in_progress.insert(std::make_pair(Op.getOperation(), entry));
    }

    // std::cout << "Finish allocating to chan\n";
  }

  // Get broadcast size from channel declaration
  unsigned getBCastSizeFromChannelDeclaration(Operation * op){
    auto chan_op = dyn_cast<air::ChannelInterface>(op);
    if (!chan_op) return 1;
    auto chan_declr = getChannelDeclarationThroughSymbol(chan_op);
    if (chan_declr->hasAttr("broadcast_shape")) {
      unsigned bcast_size = 1;
      auto size = extractFromI64ArrayAttr(chan_declr->getAttr("broadcast_shape"));
      for (auto s : size) {
        bcast_size *= s;
      }
      size = extractFromI64ArrayAttr(chan_declr.getSize());
      for (auto s : size) {
        bcast_size /= s;
      }
      return bcast_size;
    }
    else return 1;
  }

  // Get the number of dispatches already executed in a dynamically dispatched event
  unsigned getAlreadyDispatchedForDynamicDispatch(Operation * op){
    unsigned already_dispatched = 0;
    if (this->ops_in_progress.count(op)){
      already_dispatched = this->ops_in_progress[op].first;
    }
    // Event's dynamic dispatch log not found
    else {
      already_dispatched = 0;
    }
    return already_dispatched;
  }

  // Get the number of remaining dispatches in a dynamically dispatched event (e.g., events in scf.parallel)
  unsigned getRemainingDispatchesForDynamicDispatch(Operation * op){
    // Only launch runner node holds ops_in_progress cache
    if (this->runner_node_type != "launch"){
      return 0;
    }
    
    // Check how many events in total
    unsigned total = this->tokenSpatialFactorForResource<air::HierarchyInterface>(op, {});
    unsigned already_dispatched = this->getAlreadyDispatchedForDynamicDispatch(op);
    // std::cout << air::to_string(op) << " in depth: already_dispatched = " << already_dispatched << "\n";

    // Check how many remaining evnets need to be dispatched in this op
    unsigned remaining = total - already_dispatched;
    return remaining;
  }
  unsigned getRemainingDispatchesForDynamicDispatch(air::ChannelGetOp getOp){
    // Only launch runner node holds ops_in_progress cache
    if (this->runner_node_type != "launch"){
      return 0;
    }

    // Check how many remaining dispatches for get op, by checking the progress difference to put op
    auto putOp = getTheOtherChannelOpThroughSymbol(getOp);
    unsigned get_dispatched = this->getAlreadyDispatchedForDynamicDispatch(getOp.getOperation());
    unsigned put_dispatched = this->getAlreadyDispatchedForDynamicDispatch(putOp[0].getOperation());

    // Channel broadcast
    unsigned bcast_factor = this->getBCastSizeFromChannelDeclaration(getOp.getOperation());
    int remaining = put_dispatched * bcast_factor - get_dispatched;
    remaining = std::max(remaining, 0);
    return (unsigned)remaining;
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

        // To start the next loop iteration:
        // reset graph wrt this token
        std::vector<Graph::vertex_descriptor> reset_vertices_start;
        // Get vertices adjacent to the next-iteration-incarnation of this
        // yielded token
        this->getVerticesAdjToNextIterToken(reset_vertices_start, G, for_v,
                                      next_iter_token);
        // Get vertex inversely adjacent to this yielded token
        auto reset_vertices_end =
            this->getVertexInvAdjToLoopYieldedToken(op->getOperands()[token_ids[i]]);
        for (auto adj_v : reset_vertices_start) {
          this->resetGraphBetweenTwoVertices(adj_v, reset_vertices_end, G, time);
          // release the token locks, if the token is still iterating
          if (token_is_still_iterating[i]) {
            G[adj_v].token_count += this->tokenSpatialFactorForDependency(G[adj_v].op);
          }
        }
        // Reset scf.yield
        this->resetVertex(it, G, time);
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
          this->tokenSpatialFactorForDependency(
              G[*adj_v].op); // Lock number = number of dependent iter_args
    }

    this->processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelPutOp op, Graph::vertex_descriptor it) {
    // Get launch runner node
    auto launch_runner = this;
    while(launch_runner->runner_node_type != "launch"){
      launch_runner = launch_runner->parent;
    }

    // Check if this op has been completely dispatched
    unsigned processed = 1;
    unsigned total_count = this->tokenSpatialFactorForResource<air::HierarchyInterface>(op, {});
    if (launch_runner->ops_in_progress.count(op.getOperation())){
      processed = launch_runner->ops_in_progress[op.getOperation()].first;
      if (processed == total_count){
        this->processed_vertices.push_back(it);
      }
    }
    else assert(false);
  }

  void executeOp(air::ChannelGetOp op, Graph::vertex_descriptor it) {

    // std::cout << "Start executing get\n";

    // Get launch runner node
    auto launch_runner = this;
    while((launch_runner->runner_node_type != "launch")){
      launch_runner = launch_runner->parent;
    }

    // Get op progress
    auto put = air::getTheOtherChannelOpThroughSymbol(op);
    auto put_processed = launch_runner->getAlreadyDispatchedForDynamicDispatch(put[0].getOperation());
    auto get_processed = launch_runner->getAlreadyDispatchedForDynamicDispatch(op.getOperation());
    unsigned bcast_factor = launch_runner->getBCastSizeFromChannelDeclaration(op.getOperation());
    unsigned total_count = this->tokenSpatialFactorForResource<air::HierarchyInterface>(op, {});

    // Calculate how many src and dst ports to deallocate
    unsigned put_reserved_count = 0;
    for (auto p : launch_runner->ops_in_progress[put[0].getOperation()].second){
      if (p->isReserved){
        put_reserved_count++;
      }
    }
    // std::cout << "put_reserved count: " << put_reserved_count << "\n";
    unsigned get_reserved_count = 0;
    for (auto g : launch_runner->ops_in_progress[op.getOperation()].second){
      if (g->isReserved){
        get_reserved_count++;
      }
    }
    // std::cout << "get_reserved count: " << get_reserved_count << "\n";

    unsigned put_to_deallocate = 0;
    unsigned get_to_deallocate = 0;
    if (put_reserved_count * bcast_factor > get_reserved_count){
      put_to_deallocate = mlir::floorDiv(get_reserved_count, bcast_factor);
      get_to_deallocate = get_reserved_count;
    }
    else {
      put_to_deallocate = put_reserved_count;
      get_to_deallocate = put_reserved_count * bcast_factor;
    }



    // Deallocate src and dst ports
    // unsigned to_deallocate = std::min(put_reserved_count, get_reserved_count);
    unsigned put_deallocate_count = 0;
    for (auto p : launch_runner->ops_in_progress[put[0].getOperation()].second){
      if (p->isReserved){
        p->isReserved = false;
        put_deallocate_count ++;
      }
      if (put_deallocate_count == put_to_deallocate){
        break;
      }
    }
    unsigned get_deallocate_count = 0;
    for (auto g : launch_runner->ops_in_progress[op.getOperation()].second){
      if (g->isReserved){
        g->isReserved = false;
        get_deallocate_count ++;
      }
      if (get_deallocate_count == get_to_deallocate){
        break;
      }
    }

    // std::cout << "put_processed count: " << put_processed << "\n";
    // std::cout << "get_processed count: " << get_processed << "\n";
    // std::cout << "total count: " << total_count << "\n";

    // If data movement is complete, clear put and get progresses
    if ((put_processed * bcast_factor == total_count) && (get_processed == total_count)){
      this->processed_vertices.push_back(it);
      launch_runner->ops_in_progress[op.getOperation()].first = 0;
      launch_runner->ops_in_progress[op.getOperation()].second.clear();
      launch_runner->ops_in_progress[put[0].getOperation()].first = 0;
      launch_runner->ops_in_progress[put[0].getOperation()].second.clear();
    }
    // Else, continue dispatching get events
    else{
      // launch_runner->ops_in_progress[op.getOperation()].first = to_deallocate;
      launch_runner->ops_in_progress[op.getOperation()].first = total_count - get_deallocate_count;
      launch_runner->ops_in_progress[put[0].getOperation()].first = mlir::floorDiv(total_count, bcast_factor) - put_deallocate_count;
    }

    // std::cout << "Finish executing get\n";
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

  // Get vertices adjacent to the next-iteration-incarnation of a yielded async
  // token
  void
  getVerticesAdjToNextIterToken(std::vector<Graph::vertex_descriptor> &adj_vs,
                                Graph &G, Graph::vertex_descriptor v,
                                Value next_iter_token) {
    auto adj_set = boost::adjacent_vertices(v, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
      auto adj_op = G[*adj_v].op;
      assert(adj_op);
      for (auto d : adj_op->getOperands()) {
        if (d == next_iter_token) {
          adj_vs.push_back(*adj_v);
        }
      }
    }
  }

  // Get vertex inversely adjacent to this yielded token
  Graph::vertex_descriptor getVertexInvAdjToLoopYieldedToken(Value token) {
    // If vertex is air.execute, then return the terminator using "back" flag
    auto token_op = token.getDefiningOp();
    Graph::vertex_descriptor reset_vertices_end;
    if (isa<air::ExecuteOp>(token_op)) {
      reset_vertices_end =
          canonicalizer.getVertexFromOp(token_op, *(this->dep_ctx), "back").first;
    } else if (auto forop = dyn_cast<scf::ForOp>(token_op)) {
      auto forop_terminator = forop.getBody()->getTerminator();
      reset_vertices_end =
          canonicalizer.getVertexFromOp(forop_terminator, *(this->dep_ctx), "back")
              .first;
    } else {
      reset_vertices_end =
          canonicalizer.getVertexFromOp(token_op, *(this->dep_ctx), "back").first;
    }
    return reset_vertices_end;
  }

  // Check if a channel dependence has been fulfilled
  bool checkChannelDependenceFulfillment(dependencyNodeEntry dep_node,
                                         std::vector<unsigned> position) {
    auto channel_op = dyn_cast<air::ChannelInterface>(dep_node.op);
    assert(channel_op);
    std::string chan_name = channel_op.getChanName().str();
    unsigned th = (position.size())
                      ? (this->tokenSpatialFactorForDependency(dep_node.op, position))
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
      if (node.token_count < th * this->tokenSpatialFactorForDependency(node.op, position)) {
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
  unsigned tokenSpatialFactorForDependency(Operation *op) {
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
  unsigned tokenSpatialFactorForDependency(Operation *op, std::vector<unsigned> position) {
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

  // Get batch-dispatched count up until type
  template <typename T>
  unsigned tokenSpatialFactorForResource(Operation *op, std::vector<unsigned> position) {
    unsigned output = 1;
    auto parent = op;
    while ((!isa<T>(parent)) && !(isa<func::FuncOp>(parent))){
      parent = parent->getParentOp();
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
    // TODO: formalize for scf.parallel, air.launch, air.segment and air.herd.
    SmallVector<int, 2> lbs_spatial;
    SmallVector<int, 2> ubs_spatial;
    getSizesFromSpatialLoop(op, lbs_spatial, ubs_spatial);
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      resource_usage *= ubs_spatial[i] - lbs_spatial[i] + 1;
    }
    return resource_usage;
  }

  // Check if spatial op is batch-dispatched in current simulation granularity
  unsigned getBatchDispatchCount(Operation *op, bool use_multiplier = false) {
    if (isa<air::HerdOp>(op)) {
      if (this->sim_granularity == "herd" || use_multiplier) {
        return this->getResourceUsageMultiplier(op, false);
      } else if (this->sim_granularity == "core") {
        return this->getResourceUsageMultiplier(op, true);
      }
      // TODO: add other simulation granularities
    } else if (isa<air::SegmentOp>(op)) {
      return 1;
    } else if (isa<air::LaunchOp>(op)) {
      return 1;
    } else if (isa<scf::ParallelOp>(op)) {
      return this->getResourceUsageMultiplier(op, false);
    }
    assert(false && "TODO: add other simulation granularities");
    return 1;
  }

  // Get batch-dispatched count up until type
  template <typename T>
  unsigned getBatchDispatchCountUpUntilType(Operation *op) {
    unsigned spatial_factor = 1;
    while ((!isa<T>(op)) && !(isa<func::FuncOp>(op))){
      if (isa<scf::ParallelOp>(op)){
        spatial_factor *= this->getBatchDispatchCount(op);
      }
      else if (isa<air::HierarchyInterface>(op)){
        spatial_factor *= this->getBatchDispatchCount(op);
      }
      op = op->getParentOp();
    }
    // If ending with hierarchy op, then the last op's spatial factor also needs to be incorporated
    if (isa<air::HierarchyInterface>(op)){
      spatial_factor *= this->getBatchDispatchCount(op);
    }
    return spatial_factor;
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