//===- AIRTargets.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "AIRTargets.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;
using namespace xilinx;

namespace xilinx {
namespace air {

namespace {

static llvm::cl::opt<int>
    gridNumRows("num-rows",
                llvm::cl::desc("Number of rows of AIEs in the grid"),
                llvm::cl::init(0));
static llvm::cl::opt<int>
    gridNumCols("num-cols",
                llvm::cl::desc("Number of columns of AIEs in the grid"),
                llvm::cl::init(0));

llvm::json::Value attrToJSON(Attribute &attr) {
  if (auto a = attr.dyn_cast<StringAttr>()) {
    return llvm::json::Value(a.getValue().str());
  } else if (auto array_attr = attr.dyn_cast<ArrayAttr>()) {
    llvm::json::Array arrayJSON;
    for (auto a : array_attr)
      arrayJSON.push_back(attrToJSON(a));
    return llvm::json::Value(std::move(arrayJSON));
  } else if (auto dict_attr = attr.dyn_cast<DictionaryAttr>()) {
    llvm::json::Object dictJSON;
    for (auto a : dict_attr) {
      auto ident = a.getName();
      auto attr = a.getValue();
      dictJSON[ident.str()] = attrToJSON(attr);
    }
    return llvm::json::Value(std::move(dictJSON));
  } else if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
    return llvm::json::Value(int_attr.getInt());
  } else
    return llvm::json::Value(std::string(""));
}
} // namespace

void registerAIRRtTranslations() {

  TranslateFromMLIRRegistration registrationMMap(
      "airrt-generate-json", "Transform airrt metadata to JSON",
      [](ModuleOp module, raw_ostream &output) {
        llvm::json::Object moduleJSON;
        for (auto module_meta : module.getOps<airrt::ModuleMetadataOp>()) {
          llvm::json::Object segmentJSON;
          for (auto segment_meta :
               module_meta.getOps<airrt::SegmentMetadataOp>()) {
            for (auto herd_meta :
                 segment_meta.getOps<airrt::HerdMetadataOp>()) {
              llvm::json::Object herdJSON;
              for (auto a : herd_meta->getAttrs()) {
                auto ident = a.getName();
                auto attr = a.getValue();
                herdJSON[ident.str()] = attrToJSON(attr);
              }
              segmentJSON[herd_meta.getSymName()] =
                  llvm::json::Value(std::move(herdJSON));
            }
            for (auto a : segment_meta->getAttrs()) {
              auto ident = a.getName();
              auto attr = a.getValue();
              segmentJSON[ident.str()] = attrToJSON(attr);
            }
            moduleJSON[segment_meta.getSymName()] =
                llvm::json::Value(std::move(segmentJSON));
          }
        }
        llvm::json::Value topv(std::move(moduleJSON));
        std::string ret;
        llvm::raw_string_ostream ss(ret);
        ss << llvm::formatv("{0:2}", topv) << "\n";
        output << ss.str();
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<xilinx::AIE::AIEDialect, xilinx::air::airDialect,
                        xilinx::airrt::AIRRtDialect, func::FuncDialect,
                        cf::ControlFlowDialect, arith::ArithDialect,
                        memref::MemRefDialect, vector::VectorDialect,
                        LLVM::LLVMDialect, scf::SCFDialect,
                        affine::AffineDialect>();
      });
  TranslateFromMLIRRegistration registrationXJSON(
      "air-herds-to-json", "Transform herd information to JSON",
      [](ModuleOp module, raw_ostream &output) {
        // boilerplate to give dimensions to the visualizer
        output << "{\n\t\"switchbox00\": {\n\t\t\"row\": " << gridNumRows - 1
               << ", "
               << "\n\t\t\"col\": " << gridNumCols - 1 << "\n\t}, ";
        output << "\n\t\"segment\": [ ";
        return AIRHerdsToJSON(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<air::airDialect, func::FuncDialect, arith::ArithDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        affine::AffineDialect, linalg::LinalgDialect>();
      });
}

} // namespace air
} // namespace xilinx
