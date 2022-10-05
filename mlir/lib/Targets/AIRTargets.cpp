//===- AIRTargets.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
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

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"

using namespace mlir;
using namespace xilinx;

namespace xilinx {
namespace air {

namespace {

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
      "airrt-generate-json",
      [](ModuleOp module, raw_ostream &output) {
        llvm::json::Object moduleJSON;
        for (auto module_meta : module.getOps<airrt::ModuleMetadataOp>()) {
          llvm::json::Object partitionJSON;
          for (auto partition_meta :
               module_meta.getOps<airrt::PartitionMetadataOp>()) {
            for (auto herd_meta :
                 partition_meta.getOps<airrt::HerdMetadataOp>()) {
              llvm::json::Object herdJSON;
              for (auto a : herd_meta->getAttrs()) {
                auto ident = a.getName();
                auto attr = a.getValue();
                herdJSON[ident.str()] = attrToJSON(attr);
              }
              partitionJSON[herd_meta.getSymName()] =
                  llvm::json::Value(std::move(herdJSON));
            }
            for (auto a : partition_meta->getAttrs()) {
              auto ident = a.getName();
              auto attr = a.getValue();
              partitionJSON[ident.str()] = attrToJSON(attr);
            }
            moduleJSON[partition_meta.getSymName()] =
                llvm::json::Value(std::move(partitionJSON));
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
        registry.insert<xilinx::air::airDialect, xilinx::airrt::AIRRtDialect,
                        func::FuncDialect, cf::ControlFlowDialect,
                        arith::ArithmeticDialect, memref::MemRefDialect,
                        vector::VectorDialect, LLVM::LLVMDialect,
                        scf::SCFDialect, AffineDialect>();
      });
}

} // namespace air
} // namespace xilinx
