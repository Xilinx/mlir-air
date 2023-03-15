//===- Resource.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_RESOURCE
#define AIR_UTIL_RUNNER_RESOURCE

#include "air/Util/Runner.h"

namespace xilinx {
namespace air {

// Resource node entry for resource model
class resource {

public:
  std::string name;
  resource *parent;

  void set_name(llvm::json::Object *nameObject) {
    if (nameObject)
      this->set_name(nameObject->getString("devicename").value().str());
    else
      this->set_name("");
  }

  void set_name(std::string name) { this->name = name; }

  void set_parent(resource *parent) { this->parent = parent; }

  resource(std::string name = "", resource *parent = nullptr)
      : name(name), parent(parent) {}

  ~resource() {}

private:
};

class port : public resource {
public:
  double data_rate;
  std::map<std::string, port> connected_ports;

  port() {}

  port(llvm::json::Object &json_template) {
    name = json_template.getString("type").value().str() + "_" +
           json_template.getString("idx").value().str();
    data_rate = json_template.getInteger("bytes_per_cycle").value();
  }

  port(port *base) {
    name = base->name;
    data_rate = base->data_rate;
    // TODO: Copy port Connections as well
    // May be challenging due to hierarchy --
    // need to copy while copying regions/tiles etc.
  }

  port(resource *parent, unsigned src, unsigned dst, double data_rate,
       unsigned idx) {
    this->set_name("L" + std::to_string(src) + "_to_" + "L" +
                   std::to_string(dst) + "_" + std::to_string(idx));
    this->set_data_rate(data_rate);
    this->set_parent(parent);
  }

  ~port() {}

  void add_connection(port *p) { connected_ports[p->name] = p; }

  void set_data_rate(long bytes_per_cycle) {
    this->data_rate = bytes_per_cycle;
  }

private:
}; // port

class streamPort : port { // Do Ports need Master/Slave assignment? Probably...
public:
private:
}; // streamPort

class MMPort : port {
public:
private:
}; // MMPort

class kernel {

public:
  resource *parent;
  std::string name;
  double efficiency;
  int ops_per_core_per_cycle;

  void set_vector_size(std::optional<int> vectorSize) {
    if (vectorSize)
      this->ops_per_core_per_cycle = *vectorSize;
    else
      this->ops_per_core_per_cycle = 0;
  }

  void set_efficiency(std::optional<double> eff) {
    if (eff)
      this->ops_per_core_per_cycle = *eff;
    else
      this->ops_per_core_per_cycle = 0;
  }

  kernel() {}

  kernel(resource *parent, llvm::json::Object *kernelObject) {
    this->name = kernelObject->getString("name").value().str();
    this->parent = parent;
    this->set_efficiency(kernelObject->getNumber("efficiency"));
    this->set_vector_size(kernelObject->getInteger("ops_per_core_per_cycle"));
    // this->num_fmt = kernelObject->getString("format").value().str();
    // this->total_ops = kernelObject->getInteger("ops").value();
    // for(llvm::json::Value rgn : *kernelObject->getArray("supported_regions"))
    // {
    //   llvm::json::Object rgn_obj = *rgn.getAsObject();
    //   this->efficiency[rgn_obj.getString("region").value().str()]
    //     = rgn_obj.getInteger("efficiency").value();
    // }
  }

  ~kernel() {}

private:
}; // kernel

class memory : public resource {

public:
  unsigned memory_space;
  double bytes;

  void set_memory_space(unsigned ms) { this->memory_space = ms; }

  void set_bytes(double b) { this->bytes = b; }

  memory(unsigned ms, double b) {
    this->set_memory_space(ms);
    this->set_bytes(b);
  }

  ~memory() {}

private:
}; // memory

class tile : public resource {

public:
  memory *tile_mem;
  std::vector<unsigned> coordinates;

  void set_tile_coordinates(std::vector<unsigned> coords) {
    for (auto c : coords) {
      this->coordinates.push_back(c);
    }
  }

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

  tile(resource *parent, llvm::json::Object *tileObject,
       std::vector<unsigned> coordinates) {
    this->set_tile_coordinates(coordinates);
    this->set_memory(tileObject->getObject("memory"));
  }

  ~tile() {}

private:
}; // tile

class column : public resource {

public:
  memory *column_mem;
  std::map<std::vector<unsigned>, tile *> tiles;
  std::vector<unsigned> coordinates;

  column() {}

  column(resource *parent, llvm::json::Object *columnObject,
         std::vector<unsigned> coords) {
    this->set_column_coordinates(coords);
    this->set_memory(columnObject->getObject("memory"));
  }

  ~column() {}

  // port* get_port(std::string port_name) {
  //   return ports[port_name];
  // }

  void set_column_coordinates(std::vector<unsigned> coords) {
    for (auto c : coords) {
      this->coordinates.push_back(c);
    }
  }

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
      std::vector<unsigned> counts;
      std::vector<unsigned> coordinates;
      auto tileArray = tilesObject->getArray("count");
      for (auto it = tileArray->begin(), ie = tileArray->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        llvm::json::Object *tileObject = jv.getAsObject();
        auto val = tileObject->getInteger("num");
        counts.push_back(*val);
        total_count *= *val;
        coordinates.push_back(0); // Zero initialization
      }
      for (unsigned i = 0; i < total_count; i++) {
        tile *new_tile = new tile(this, tilesObject, coordinates);
        this->tiles.insert(std::make_pair(coordinates, new_tile));

        // Keep track of coordinates
        coordinates[0]++;
        for (unsigned d = 0; d < coordinates.size() - 1; d++) {
          if (coordinates[d] >= counts[d]) {
            coordinates[d] = 0;
            coordinates[d + 1]++;
          }
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

}; // column

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE