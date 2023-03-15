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
  std::map<std::vector<unsigned>, column *> columns;

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
      std::vector<unsigned> counts;
      std::vector<unsigned> coordinates;
      auto countArray = columnsObject->getArray("count");
      for (auto it = countArray->begin(), ie = countArray->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        llvm::json::Object *countObject = jv.getAsObject();
        auto val = countObject->getInteger("num");
        counts.push_back(*val);
        total_count *= *val;
        coordinates.push_back(0); // Zero initialization
      }
      for (unsigned i = 0; i < total_count; i++) {
        column *new_col = new column(this, columnsObject, coordinates);
        this->columns.insert(std::make_pair(coordinates, new_col));

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
    // TODO: get parent from parentObject. But does device have parent?
  }

  void setup_device_resources(llvm::json::Object *columnsObject = nullptr) {
    this->set_columns(columnsObject);
  }

  device(std::string name = "", resource *parent = nullptr,
         unsigned clock = 0) {
    this->set_name(name);
    this->set_parent(parent);
    this->set_clock(clock);
  }

  device(llvm::json::Object *model) {
    this->setup_device_parameters(
        model->getObject("devicename"), model->getNumber("clock"),
        model->getArray("datatypes"), model->getArray("interfaces"),
        model->getObject("kernels"), nullptr);
    this->setup_device_resources(model->getObject("columns"));
  }

  ~device() { sub_resource_hiers.clear(); }

private:
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE_HIERARCHY