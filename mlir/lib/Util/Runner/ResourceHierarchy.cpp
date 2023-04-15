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
      assert(ms && bytes != 0.0f);
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
            port *new_port = new port(this, "L1_inbound", bytes_per_second, i);
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
            port *new_port = new port(this, "L1_outbound", bytes_per_second, i);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      assert(false);
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
  // Keys: port direction (inbound/outbound); mapped: vector of ports.
  std::map<std::string, std::vector<port *>> ports;
  unsigned idx;

  du() {}

  du(resource *parent, llvm::json::Object *duObject, unsigned idx) {
    this->set_du_id(idx);
    this->set_memory(duObject->getObject("memory"));
    this->set_tiles(duObject->getObject("tiles"));
    this->set_ports(duObject->getObject("ports"));
    this->reset_reservation();
  }

  ~du() {}

  void set_du_id(unsigned idx) { this->idx = idx; }

  void set_memory(memory *mem) { this->du_mem = mem; }

  void set_memory(llvm::json::Object *memObject) {
    if (memObject) {
      auto ms = memObject->getString("memory_space");
      auto bytes = memObject->getNumber("bytes");
      assert(ms && bytes != 0.0f);
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
      assert(false);
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
            port *new_port = new port(this, "L2_inbound", bytes_per_second, i);
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
            port *new_port = new port(this, "L2_outbound", bytes_per_second, i);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      assert(false);
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
  std::map<std::string, kernel *> kernels;
  std::vector<du *> dus;
  // Keys: port direction (inbound/outbound); mapped: vector of ports.
  std::map<std::string, std::vector<port *>> ports;

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
        assert(datatypeObject->getString("name") &&
               datatypeObject->getNumber("bytes"));
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
        port *new_port = new port(this, s, d, b);
        this->interfaces.insert({{s, d}, new_port});
      }
    }
  }

  void set_kernels(llvm::json::Object *kernelObjects) {
    for (auto it = kernelObjects->begin(), ie = kernelObjects->end(); it != ie;
         ++it) {
      llvm::json::Object *kernelObject = it->second.getAsObject();
      if (kernelObject) {
        kernel *new_kernel = new kernel(this, kernelObject);
        this->kernels.insert(std::make_pair(
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
      assert(false);
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
            port *new_port = new port(this, "L3_inbound", bytes_per_second, i);
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
            port *new_port = new port(this, "L3_outbound", bytes_per_second, i);
            outbound_port_vec.push_back(new_port);
          }
          this->ports.insert(std::make_pair("outbound", outbound_port_vec));
        }
      }
    } else {
      assert(false);
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
    this->set_kernels(kernelsObject);
    // TODO: get parent from parentObject, for multi-device modelling.
  }

  void setup_device_resources(llvm::json::Object *dusObject = nullptr,
                              llvm::json::Object *portsObject = nullptr) {
    this->set_dus(dusObject);
    this->set_ports(portsObject);
  }

  double getDataRateFromMemorySpace(unsigned memory_space,
                                    std::string port_direction) {
    if (lookUpMemorySpaceFromInt(memory_space) == "L3") {
      if (this->ports.count(port_direction))
        return this->ports[port_direction][0]->data_rate;
      else
        return 0;
    } else if (lookUpMemorySpaceFromInt(memory_space) == "L2") {
      if (this->dus.size()) {
        if (this->dus[0]->ports.count(port_direction))
          return this->dus[0]->ports[port_direction][0]->data_rate;
        else
          return 0;
      } else
        return 0;
    } else if (lookUpMemorySpaceFromInt(memory_space) == "L1") {
      if (this->dus.size()) {
        if (this->dus[0]->tiles.size()) {
          if (this->dus[0]->tiles[0]->ports.count(port_direction))
            return this->dus[0]->tiles[0]->ports[port_direction][0]->data_rate;
          else
            return 0;
        } else
          return 0;
      } else
        return 0;
    } else
      return 0;
  }

  device(std::string name = "", resource *parent = nullptr,
         unsigned clock = 0) {
    this->set_name(name);
    this->set_parent(parent);
    this->set_clock(clock);
    this->reset_reservation();
  }

  device(llvm::json::Object *model) {
    this->setup_device_resources(model->getObject("dus"),
                                 model->getObject("noc"));
    this->setup_device_parameters(
        model->getObject("devicename"), model->getNumber("clock"),
        model->getArray("datatypes"), model->getObject("kernels"), nullptr);
    this->reset_reservation();
  }

  ~device() { sub_resource_hiers.clear(); }

private:
}; // device

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE_HIERARCHY