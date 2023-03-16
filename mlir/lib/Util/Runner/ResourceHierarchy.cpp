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

  void set_tile_id(unsigned idx) { this->idx = idx; }

  void set_memory(memory *mem) { this->tile_mem = mem; }

  void set_memory(llvm::json::Object *memObject) {
    if (memObject) {
      auto ms = memObject->getInteger("memory_space");
      auto bytes = memObject->getNumber("bytes");
      assert(ms && bytes != 0.0f);
      memory *mem = new memory(*ms, *bytes);
      this->set_memory(mem);
    } else {
      this->tile_mem = nullptr;
    }
  }

  tile(resource *parent, llvm::json::Object *tileObject, unsigned idx) {
    this->set_tile_id(idx);
    this->set_memory(tileObject->getObject("memory"));
    this->reset_reservation();
  }

  ~tile() {}

private:
}; // tile

class column : public resourceHierarchy {

public:
  memory *column_mem;
  // std::map<std::vector<unsigned>, tile *> tiles;
  std::vector<tile *> tiles;
  // std::vector<unsigned> coordinates;
  unsigned idx;

  column() {}

  column(resource *parent, llvm::json::Object *columnObject, unsigned idx) {
    this->set_column_id(idx);
    this->set_memory(columnObject->getObject("memory"));
    this->set_tiles(columnObject->getObject("tiles"));
    this->reset_reservation();
  }

  ~column() {}

  // port* get_port(std::string port_name) {
  //   return ports[port_name];
  // }

  void set_column_id(unsigned idx) { this->idx = idx; }

  void set_memory(memory *mem) { this->column_mem = mem; }

  void set_memory(llvm::json::Object *memObject) {
    if (memObject) {
      auto ms = memObject->getInteger("memory_space");
      auto bytes = memObject->getNumber("bytes");
      assert(ms && bytes != 0.0f);
      memory *mem = new memory(*ms, *bytes);
      this->set_memory(mem);
    } else {
      this->column_mem = nullptr;
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
        llvm::json::Object *tileObject = jv.getAsObject();
        auto val = tileObject->getInteger("num");
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

private:
  // int config_speed; //Until we model prog mem as L3->L2->L1 dma memcopies,
  //   //we can just map fixed-rate transfers onto the L1 progmem ports, no
  //   other
  //   //segments necessary.

  // std::map<std::string, port*> ports;

  // llvm::json::Array* connectivity_json;

}; // column

// Device hierarchy node entry.
class device : public resourceHierarchy {

public:
  unsigned clock;
  std::vector<resourceHierarchy *> sub_resource_hiers;
  std::vector<resource *> resources;
  std::map<std::string, double> datatypes;
  // Key pair: <src, dst>; mapped: vector of port pointers
  std::map<std::pair<unsigned, unsigned>, std::vector<port *>> ports;
  std::map<std::string, kernel *> kernels;
  std::vector<column *> columns;

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

  void set_ports(llvm::json::Array *portObjects) {
    for (auto it = portObjects->begin(), ie = portObjects->end(); it != ie;
         ++it) {
      llvm::json::Value jv = *it;
      llvm::json::Object *portObject = jv.getAsObject();
      if (portObject) {
        auto srcSpace = portObject->getNumber("src");
        auto dstSpace = portObject->getNumber("dst");
        auto bps = portObject->getNumber("bytes_per_second");
        assert(srcSpace && dstSpace && bps);
        unsigned s = *srcSpace;
        unsigned d = *dstSpace;
        double b = *bps;
        if (ports.count({s, d})) {
          auto idx = ports[{s, d}].size() - 1;
          port *new_port = new port(this, s, d, b, idx);
          ports[{s, d}].push_back(new_port);
        } else {
          ports.insert({{s, d}, {}});
          port *new_port = new port(this, s, d, b, 0);
          ports[{s, d}].push_back(new_port);
        }
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

  void set_columns(llvm::json::Object *columnsObject) {
    if (columnsObject) {
      // Get total number of columns in device
      unsigned total_count = 1;
      auto countArray = columnsObject->getArray("count");
      for (auto it = countArray->begin(), ie = countArray->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        llvm::json::Object *countObject = jv.getAsObject();
        auto val = countObject->getInteger("num");
        total_count *= *val;
      }
      for (unsigned i = 0; i < total_count; i++) {
        column *new_col = new column(this, columnsObject, i);
        this->columns.push_back(new_col);
      }
    } else {
      assert(false);
    }
  }

  void setup_device_parameters(llvm::json::Object *nameObject = nullptr,
                               std::optional<double> clk = 0,
                               llvm::json::Array *datatypeObjects = nullptr,
                               llvm::json::Array *portsObject = nullptr,
                               llvm::json::Object *kernelsObject = nullptr,
                               llvm::json::Object *parentObject = nullptr) {
    this->set_name(nameObject);
    this->set_clock(clk);
    this->set_datatypes(datatypeObjects);
    this->set_ports(portsObject);
    this->set_kernels(kernelsObject);
    // TODO: get parent from parentObject, for multi-device modelling.
  }

  void setup_device_resources(llvm::json::Object *columnsObject = nullptr) {
    this->set_columns(columnsObject);
  }

  device(std::string name = "", resource *parent = nullptr,
         unsigned clock = 0) {
    this->set_name(name);
    this->set_parent(parent);
    this->set_clock(clock);
    this->reset_reservation();
  }

  device(llvm::json::Object *model) {
    this->setup_device_parameters(
        model->getObject("devicename"), model->getNumber("clock"),
        model->getArray("datatypes"), model->getArray("interfaces"),
        model->getObject("kernels"), nullptr);
    this->setup_device_resources(model->getObject("columns"));
    this->reset_reservation();
  }

  ~device() { sub_resource_hiers.clear(); }

private:
}; // device

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE_HIERARCHY