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
  }

  ~resourceHierarchy() { sub_resource_hiers.clear(); }

private:
};

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

  void setup_device(llvm::json::Object *nameObject = nullptr,
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
    // TODO: get parent from parentObject. But does device have parent?
  }

  device(std::string name = "", resource *parent = nullptr,
         unsigned clock = 0) {
    this->set_name(name);
    this->set_parent(parent);
    this->set_clock(clock);
  }

  device(llvm::json::Object *nameObject, std::optional<double> clk,
         llvm::json::Array *datatypeObjects, llvm::json::Array *portsObject,
         llvm::json::Object *kernelsObject, llvm::json::Object *parentObject) {
    this->setup_device(nameObject, clk, datatypeObjects, portsObject,
                       kernelsObject, parentObject);
  }

  device(llvm::json::Object *model) {
    this->setup_device(model->getObject("devicename"),
                       model->getNumber("clock"), model->getArray("datatypes"),
                       model->getArray("interfaces"),
                       model->getObject("kernels"), nullptr);
  }

  ~device() { sub_resource_hiers.clear(); }

private:
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE_HIERARCHY