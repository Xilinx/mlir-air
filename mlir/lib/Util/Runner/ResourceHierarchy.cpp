//===- ResourceHierarchy.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_RESOURCE_HIERARCHY
#define AIR_UTIL_RUNNER_RESOURCE_HIERARCHY

#include "air/Util/Runner.h"

namespace xilinx {
namespace air {

// Resource hierarchy node entry. Contains sub resources and sub hierarchies.
class resourceHierarchy : public resource {

public:
  std::vector<resourceHierarchy *> sub_resource_hiers;
  std::vector<resource *> resources;

  resourceHierarchy(std::string name = "", resource *parent = nullptr) {
    this->set_name(name);
    this->set_parent(parent);
    this->reset_reservation();
  }

  ~resourceHierarchy() { sub_resource_hiers.clear(); }

private:
}; // resourceHierarchy

class tile : public resourceHierarchy {

public:
  memory *tile_mem;
  unsigned idx;
  // Keys: port direction (inbound/outbound); mapped: vector of ports.
  std::map<std::string, std::vector<port *>> ports;

  void set_tile_id(unsigned idx) { this->idx = idx; }

  void set_memory(memory *mem) { this->tile_mem = mem; }

  void set_memory(llvm::json::Object *memObject) {
    if (memObject) {
      auto ms = memObject->getString("memory_space");
      auto bytes = memObject->getNumber("bytes");
      this->resource_assertion(ms.has_value(),
                               "memory_space not found for memory object");
      this->resource_assertion(bytes != 0.0f,
                               "memory size is zero bytes for memory object");
      memory *mem = new memory(ms.value().str(), *bytes);
      this->set_memory(mem);
    } else {
      this->tile_mem = nullptr;
    }
  }

  void set_ports(llvm::json::Object *portsObject) {
    if (portsObject) {
      auto inboundPortsObject = portsObject->getObject("inbound");
      if (inboundPortsObject) {
        auto inbound_port_count = inboundPortsObject->getInteger("count");
        if (inbound_port_count) {
          std::vector<port *> inbound_port_vec;
          for (unsigned i = 0; i < *inbound_port_count; i++) {
            auto bytes_per_second =
                inboundPortsObject->getNumber("bytes_per_second");
            auto latency = inboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L1_inbound", bytes_per_second, i, latency);
            inbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("inbound", inbound_port_vec));
        }
      }

      auto outboundPortsObject = portsObject->getObject("outbound");
      if (outboundPortsObject) {
        auto outbound_port_count = outboundPortsObject->getInteger("count");
        if (outbound_port_count) {
          std::vector<port *> outbound_port_vec;
          for (unsigned i = 0; i < *outbound_port_count; i++) {
            auto bytes_per_second =
                outboundPortsObject->getNumber("bytes_per_second");
            auto latency = outboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L1_outbound", bytes_per_second, i, latency);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      this->resource_assertion(false, "JSON object 'portsObject' not found");
    }
  }

  tile(resource *parent, llvm::json::Object *tileObject, unsigned idx) {
    this->set_tile_id(idx);
    this->set_memory(tileObject->getObject("memory"));
    this->set_ports(tileObject->getObject("ports"));
    this->reset_reservation();
  }

  ~tile() {}

private:
}; // tile

class du : public resourceHierarchy {

public:
  memory *du_mem;
  std::vector<tile *> tiles;
  std::vector<unsigned> shape;
  // Keys: port direction (inbound/outbound); mapped: vector of ports.
  std::map<std::string, std::vector<port *>> ports;
  unsigned idx;

  du() {}

  du(resource *parent, llvm::json::Object *duObject, unsigned idx) {
    this->set_du_id(idx);
    this->set_memory(duObject->getObject("memory"));
    this->set_tiles(duObject->getObject("tiles"));
    this->set_ports(duObject->getObject("ports"));
    this->set_shape(duObject->getObject("tiles")->getArray("count"));
    this->reset_reservation();
  }

  ~du() {}

  void set_du_id(unsigned idx) { this->idx = idx; }

  void set_memory(memory *mem) { this->du_mem = mem; }

  void set_memory(llvm::json::Object *memObject) {
    if (memObject) {
      auto ms = memObject->getString("memory_space");
      auto bytes = memObject->getNumber("bytes");
      this->resource_assertion(ms.has_value(),
                               "memory_space not found for memory object");
      this->resource_assertion(bytes != 0.0f,
                               "memory size is zero bytes for memory object");
      memory *mem = new memory(ms.value().str(), *bytes);
      this->set_memory(mem);
    } else {
      this->du_mem = nullptr;
    }
  }

  void set_tiles(llvm::json::Object *tilesObject) {
    if (tilesObject) {
      // Get total number of tiles in device
      unsigned total_count = 1;
      auto tileArray = tilesObject->getArray("count");
      for (auto it = tileArray->begin(), ie = tileArray->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        auto val = jv.getAsInteger();
        total_count *= *val;
      }
      for (unsigned i = 0; i < total_count; i++) {
        tile *new_tile = new tile(this, tilesObject, i);
        this->tiles.push_back(new_tile);
      }
    } else {
      this->resource_assertion(false, "JSON object 'tilesObject' not found");
    }
  }

  void set_ports(llvm::json::Object *portsObject) {
    if (portsObject) {
      auto inboundPortsObject = portsObject->getObject("inbound");
      if (inboundPortsObject) {
        auto inbound_port_count = inboundPortsObject->getInteger("count");
        if (inbound_port_count) {
          std::vector<port *> inbound_port_vec;
          for (unsigned i = 0; i < *inbound_port_count; i++) {
            auto bytes_per_second =
                inboundPortsObject->getNumber("bytes_per_second");
            auto latency = inboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L2_inbound", bytes_per_second, i, latency);
            inbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("inbound", inbound_port_vec));
        }
      }

      auto outboundPortsObject = portsObject->getObject("outbound");
      if (outboundPortsObject) {
        auto outbound_port_count = outboundPortsObject->getInteger("count");
        if (outbound_port_count) {
          std::vector<port *> outbound_port_vec;
          for (unsigned i = 0; i < *outbound_port_count; i++) {
            auto bytes_per_second =
                outboundPortsObject->getNumber("bytes_per_second");
            auto latency = outboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L2_outbound", bytes_per_second, i, latency);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      this->resource_assertion(false, "JSON object 'portsObject' not found");
    }
  }

  // Get the shape of each DU (in tiles)
  void set_shape(llvm::json::Array *sizesObject) {
    for (auto it = sizesObject->begin(), ie = sizesObject->end(); it != ie;
         ++it) {
      llvm::json::Value jv = *it;
      auto val = jv.getAsInteger();
      this->shape.push_back(*val);
    }
  }

private:
  // int config_speed; //Until we model prog mem as L3->L2->L1 dma memcopies,
  //   //we can just map fixed-rate transfers onto the L1 progmem ports, no
  //   other
  //   //segments necessary.

  // std::map<std::string, port*> ports;

  // llvm::json::Array* connectivity_json;

}; // du

// Device hierarchy node entry.
class device : public resourceHierarchy {

public:
  unsigned clock;
  std::vector<resourceHierarchy *> sub_resource_hiers;
  std::vector<resource *> resources;
  std::map<std::string, double> datatypes;
  // Key pair: <src, dst>; mapped: vector of port pointers
  // TODO: deprecate this.
  std::map<std::pair<unsigned, unsigned>, port *> interfaces;
  // How an op is priced. Four entries, and the names answer the reader's
  // question -- I have an op, where does its cost come from?
  //
  //   cost_model.op_costs        ops with a body: a cycle expression over the
  //                              op's own shape, or a throughput rate. Keyed by
  //                              op name, or by `air.op_cost` when the op name
  //                              does not identify the work.
  //   cost_model.opaque_costs    ops with no body (air.custom): a fixed
  //                              latency, keyed by the op's symbol. No formula,
  //                              because there is nothing to derive one from.
  //   cost_model.transfer_costs  data movement: bandwidth and time of flight,
  //                              keyed by `air.transfer_cost`. Absent, a
  //                              transfer is priced by its memory-space
  //                              interface.
  //   cost_model.fallback        used when none of the above matched: scalars
  //                              feeding the built-in instruction-count
  //                              estimate.
  //
  // Plural names are maps of named entries; `fallback` is singular because it
  // is one entry, not a map.
  std::map<std::string, kernel *> op_costs;
  std::vector<du *> dus;
  // Keys: port direction (inbound/outbound); mapped: vector of ports.
  std::map<std::string, std::vector<port *>> ports;

  // cost_model.transfer_costs: named ways of pricing a data movement.
  //
  // Same mechanism as op_costs -- select an entry by a key -- but the key has
  // no default. An op_costs key falls back to the op name, so that table is
  // always consulted; a transfer with no `air.transfer_cost` attribute never
  // reaches this table and is priced by its memory-space interface, exactly as
  // before this existed.
  //
  // It is needed because the only thing a transfer could be keyed on was its
  // (src, dst) memory-space pair, so two transfers between the same two levels
  // always cost the same. That is wrong whenever a machine prices them
  // differently for a reason the memory spaces do not capture -- a second
  // interconnect between the same levels is one such reason, and another, which
  // needs no extra hardware, is that one of them overlaps with compute and
  // never reaches the critical path. The runner has no other way to say either.
  //
  // Named for the cost model rather than for a wire, because which of those an
  // arch author means is their business and not the runner's.
  struct transferCost {
    double data_rate = 0; // bytes per second; unset means "use the interface"
    double latency = 0;   // cycles of time of flight
    bool has_data_rate = false;
    bool has_latency = false;
  };
  std::map<std::string, transferCost> transfer_costs;

  void set_transfer_costs(llvm::json::Object *obj) {
    if (!obj)
      return;
    for (auto &entry : *obj) {
      auto *o = entry.second.getAsObject();
      if (!o)
        continue;
      transferCost tc;
      if (auto v = o->getNumber("bytes_per_second")) {
        this->resource_assertion(
            *v > 0, "transfer cost bytes_per_second must be positive");
        tc.data_rate = *v;
        tc.has_data_rate = true;
      }
      if (auto v = o->getNumber("latency")) {
        this->resource_assertion(*v >= 0,
                                 "transfer cost latency must not be negative");
        tc.latency = *v;
        tc.has_latency = true;
      }
      this->transfer_costs[entry.first.str()] = tc;
    }
  }

  // The entry a channel names, or nullptr if it names none or names one the
  // arch does not define. Lookup failure falls back silently and on purpose: an
  // arch that does not describe the distinction is still a valid model of the
  // same IR, just a coarser one.
  const transferCost *getTransferCostEntry(llvm::StringRef name) {
    if (name.empty())
      return nullptr;
    auto it = this->transfer_costs.find(name.str());
    return it == this->transfer_costs.end() ? nullptr : &it->second;
  }

  // How the throughput model prices a linalg body. These were fixed constants
  // in the runner, chosen for AIE: a herd body instance is one core, a kernel
  // is an external function call costing ~100 cycles to enter, and a vector
  // lane is 8 wide. They are properties of a machine, not of the simulator, so
  // a model that is not AIE can say otherwise. The defaults reproduce the
  // previous behaviour exactly.
  double cores_per_kernel_instance = 1;
  double default_ops_per_core_per_cycle = 8;
  uint64_t kernel_invocation_overhead = 100;

  void set_fallback(llvm::json::Object *model) {
    if (!model)
      return;
    // Validate here rather than at the point of use: the first two divide a
    // work count, and the third is cast to an unsigned, so a zero or negative
    // in the JSON would surface much later as a division by zero or as an
    // overhead of billions of cycles.
    if (auto v = model->getNumber("cores_per_kernel_instance")) {
      this->resource_assertion(*v > 0,
                               "cores_per_kernel_instance must be positive");
      this->cores_per_kernel_instance = *v;
    }
    if (auto v = model->getNumber("default_ops_per_core_per_cycle")) {
      this->resource_assertion(
          *v > 0, "default_ops_per_core_per_cycle must be positive");
      this->default_ops_per_core_per_cycle = *v;
    }
    if (auto v = model->getNumber("kernel_invocation_overhead")) {
      this->resource_assertion(*v >= 0,
                               "kernel_invocation_overhead must not be "
                               "negative");
      this->kernel_invocation_overhead = (uint64_t)*v;
    }
  }

  void set_clock(std::optional<double> clk) {
    if (clk) {
      this->set_clock(*clk);
    } else
      this->set_clock((unsigned)0);
  }

  void set_clock(unsigned clock) { this->clock = clock; }

  void set_datatypes(llvm::json::Array *datatypeObjects) {
    for (auto it = datatypeObjects->begin(), ie = datatypeObjects->end();
         it != ie; ++it) {
      llvm::json::Value jv = *it;
      llvm::json::Object *datatypeObject = jv.getAsObject();
      if (datatypeObject) {
        this->resource_assertion(datatypeObject->getString("name").has_value(),
                                 "datatypeObject has no name");
        this->resource_assertion(datatypeObject->getNumber("bytes").has_value(),
                                 "datatypeObject has no byte count");
        std::string name = datatypeObject->getString("name").value().str();
        double bytes = datatypeObject->getNumber("bytes").value();
        this->datatypes.insert(std::make_pair(name, bytes));
      }
    }
  }

  void set_interfaces() {
    for (unsigned s = 0; s < 3; s++) {
      for (unsigned d = 0; d < 3; d++) {
        double b_s = this->getDataRateFromMemorySpace(s, "outbound");
        double b_d = this->getDataRateFromMemorySpace(d, "inbound");
        double b = std::min(b_s, b_d);
        // Bandwidth is the bottleneck of the two ends, but time-of-flight is
        // paid at both: data serialises out of the source and back in at the
        // destination, so the two latencies add.
        double l = this->getLatencyFromMemorySpace(s, "outbound") +
                   this->getLatencyFromMemorySpace(d, "inbound");
        port *new_port = new port(this, s, d, b, l);
        this->interfaces.insert({{s, d}, new_port});
      }
    }
  }

  void set_op_costs(llvm::json::Object *kernelObjects) {
    // Absent is legal -- a model that prices nothing by name still runs, on
    // the fallback. Dereferencing unconditionally used to be safe only because
    // the sole caller read a top-level key; it now reads a nested one, so
    // there are two ways to arrive here with nothing.
    if (!kernelObjects)
      return;
    for (auto it = kernelObjects->begin(), ie = kernelObjects->end(); it != ie;
         ++it) {
      llvm::json::Object *kernelObject = it->second.getAsObject();
      if (kernelObject) {
        kernel *new_kernel = new kernel(this, kernelObject);
        this->op_costs.insert(std::make_pair(
            kernelObject->getString("name").value(), new_kernel));
      }
    }
  }

  void set_dus(llvm::json::Object *dusObject) {
    if (dusObject) {
      // Get total number of dus in device
      unsigned total_count = 1;
      auto countArray = dusObject->getArray("count");
      for (auto it = countArray->begin(), ie = countArray->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        auto val = jv.getAsInteger();
        total_count *= *val;
      }
      for (unsigned i = 0; i < total_count; i++) {
        du *new_col = new du(this, dusObject, i);
        this->dus.push_back(new_col);
      }
    } else {
      this->resource_assertion(false, "JSON model 'dusObject' not found");
    }
  }

  void set_ports(llvm::json::Object *portsObject) {
    if (portsObject) {
      auto inboundPortsObject = portsObject->getObject("inbound");
      if (inboundPortsObject) {
        auto inbound_port_count = inboundPortsObject->getInteger("count");
        if (inbound_port_count) {
          std::vector<port *> inbound_port_vec;
          for (unsigned i = 0; i < *inbound_port_count; i++) {
            auto bytes_per_second =
                inboundPortsObject->getNumber("bytes_per_second");
            auto latency = inboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L3_inbound", bytes_per_second, i, latency);
            inbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("inbound", inbound_port_vec));
        }
      }

      auto outboundPortsObject = portsObject->getObject("outbound");
      if (outboundPortsObject) {
        auto outbound_port_count = outboundPortsObject->getInteger("count");
        if (outbound_port_count) {
          std::vector<port *> outbound_port_vec;
          for (unsigned i = 0; i < *outbound_port_count; i++) {
            auto bytes_per_second =
                outboundPortsObject->getNumber("bytes_per_second");
            auto latency = outboundPortsObject->getNumber("latency");
            port *new_port =
                new port(this, "L3_outbound", bytes_per_second, i, latency);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      this->resource_assertion(false, "JSON model 'portsObject' not found");
    }
  }

  void setup_device_parameters(llvm::json::Object *nameObject = nullptr,
                               std::optional<double> clk = 0,
                               llvm::json::Array *datatypeObjects = nullptr,
                               llvm::json::Object *kernelsObject = nullptr,
                               llvm::json::Object *parentObject = nullptr) {
    this->set_name(nameObject);
    this->set_clock(clk);
    this->set_datatypes(datatypeObjects);
    this->set_interfaces();
    this->set_op_costs(kernelsObject);
    // TODO: get parent from parentObject, for multi-device modelling.
  }

  void setup_device_resources(llvm::json::Object *dusObject = nullptr,
                              llvm::json::Object *portsObject = nullptr) {
    this->set_dus(dusObject);
    this->set_ports(portsObject);
  }

  // Get the representative port serving a memory space in a given direction,
  // or nullptr if the model does not describe one.
  port *getPortFromMemorySpace(unsigned memory_space,
                               std::string port_direction) {
    auto ms = symbolizeMemorySpace(memory_space);
    if (!ms)
      return nullptr;
    if (*ms == air::MemorySpace::L3) {
      if (this->ports.count(port_direction))
        return this->ports[port_direction][0];
      return nullptr;
    } else if (*ms == air::MemorySpace::L2) {
      if (this->dus.size() && this->dus[0]->ports.count(port_direction))
        return this->dus[0]->ports[port_direction][0];
      return nullptr;
    } else if (*ms == air::MemorySpace::L1) {
      if (this->dus.size() && this->dus[0]->tiles.size() &&
          this->dus[0]->tiles[0]->ports.count(port_direction))
        return this->dus[0]->tiles[0]->ports[port_direction][0];
      return nullptr;
    }
    return nullptr;
  }

  double getDataRateFromMemorySpace(unsigned memory_space,
                                    std::string port_direction) {
    auto *p = this->getPortFromMemorySpace(memory_space, port_direction);
    return p ? p->data_rate : 0;
  }

  double getLatencyFromMemorySpace(unsigned memory_space,
                                   std::string port_direction) {
    auto *p = this->getPortFromMemorySpace(memory_space, port_direction);
    return p ? p->latency : 0;
  }

  device(std::string name = "", resource *parent = nullptr,
         unsigned clock = 0) {
    this->set_name(name);
    this->set_parent(parent);
    this->set_clock(clock);
    this->reset_reservation();
  }

  device(llvm::json::Object *model) {
    // Everything that prices an op lives under one `cost_model` object. See
    // set_op_costs() for what the four entries under it are and why they are
    // named the way they are.
    auto *costs = model->getObject("cost_model");
    // Everything that prices an op moved under `cost_model`. Say so, rather
    // than silently pricing the whole model on the fallback: a file written
    // against the old flat layout is otherwise indistinguishable from one that
    // deliberately prices nothing.
    this->resource_assertion(
        costs != nullptr,
        "arch model has no 'cost_model' object. The pricing tables live under "
        "it now: cost_model.op_costs (was 'kernels'), cost_model.opaque_costs "
        "(was 'custom_kernels'), cost_model.fallback (was 'compute_model'), "
        "and cost_model.transfer_costs");
    this->setup_device_resources(model->getObject("dus"),
                                 model->getObject("noc"));
    this->setup_device_parameters(
        model->getObject("devicename"), model->getNumber("clock"),
        model->getArray("datatypes"),
        costs ? costs->getObject("op_costs") : nullptr, nullptr);
    this->set_fallback(costs ? costs->getObject("fallback") : nullptr);
    this->set_transfer_costs(costs ? costs->getObject("transfer_costs")
                                   : nullptr);
    this->reset_reservation();
  }

  ~device() { sub_resource_hiers.clear(); }

private:
}; // device

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE_HIERARCHY