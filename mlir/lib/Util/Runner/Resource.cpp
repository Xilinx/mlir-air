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

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE