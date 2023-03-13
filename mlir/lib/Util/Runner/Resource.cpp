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
    if (nameObject) {
      this->set_name(nameObject->getString("devicename").value().str());
    }
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
    this->name = "L" + std::to_string(src) + "_to_" + "L" +
                 std::to_string(dst) + "_" + std::to_string(idx);
    this->data_rate = data_rate;
    this->parent = parent;
  }

  ~port() {}

  void add_connection(port *p) { connected_ports[p->name] = p; }

  void set_data_rate(long bytes_per_cycle) { data_rate = bytes_per_cycle; }

private:
  double data_rate;
  std::map<std::string, port> connected_ports;

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

  kernel() {}

  kernel(resource *parent, llvm::json::Object *kernelObject) {
    this->name = kernelObject->getString("name").value().str();
    this->parent = parent;
    this->efficiency = kernelObject->getNumber("efficiency").value();
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
  // unsigned total_ops;
  // std::string num_fmt;
  // std::map<std::string,float> efficiency;

}; // kernel

// Resource hierarchy node entry. Contains sub resources and sub hierarchies.
class resourceHierarchy : public resource {

public:
  // unsigned clock;
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

  void set_clock(llvm::json::Object *clockObject) {
    if (clockObject) {
      this->set_clock(clockObject->getNumber("clock").value());
    }
  }

  void set_clock(unsigned clock) { this->clock = clock; }

  // TODO: datatype shoud be array of objects, as a dictionary of supported
  // datatypes
  void set_datatypes(llvm::json::Object *datatypeObject) {
    if (datatypeObject) {
      if (datatypeObject->getObject("name")) {
        if (datatypeObject->getObject("bytes")) {
          std::string name = datatypeObject->getString("name").value().str();
          double bytes = datatypeObject->getNumber("bytes").value();
          this->datatypes.insert(std::make_pair(name, bytes));
        }
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
                    llvm::json::Object *clockObject = nullptr,
                    llvm::json::Object *datatypeObject = nullptr,
                    llvm::json::Array *portsObject = nullptr,
                    llvm::json::Object *kernelsObject = nullptr,
                    llvm::json::Object *parentObject = nullptr) {
    this->set_name(nameObject);
    this->set_clock(clockObject);
    this->set_datatypes(datatypeObject);
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

  device(llvm::json::Object *nameObject, llvm::json::Object *clockObject,
         llvm::json::Object *datatypeObject, llvm::json::Array *portsObject,
         llvm::json::Object *kernelsObject, llvm::json::Object *parentObject) {
    this->setup_device(nameObject, clockObject, datatypeObject, portsObject,
                       kernelsObject, parentObject);
  }

  device(llvm::json::Object *model) {
    this->setup_device(model->getObject("devicename"),
                       model->getObject("clock"), model->getObject("datatype"),
                       model->getArray("interfaces"),
                       model->getObject("kernels"), nullptr);
  }

  ~device() { sub_resource_hiers.clear(); }

private:
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_RESOURCE