// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "mlir/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Translation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/Export.h"

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
    }
    else if (auto array_attr = attr.dyn_cast<ArrayAttr>() ) {
      llvm::json::Array arrayJSON;
      for (auto a : array_attr)
        arrayJSON.push_back(attrToJSON(a));
      return llvm::json::Value(std::move(arrayJSON));
    }
    else if (auto dict_attr = attr.dyn_cast<DictionaryAttr>()) {
      llvm::json::Object dictJSON;
      for (auto a : dict_attr) {
        auto ident = a.first;
        auto attr = a.second;
        dictJSON[ident.str()] = attrToJSON(attr);
      }
      return llvm::json::Value(std::move(dictJSON));
    }
    else if (auto int_attr = attr.dyn_cast<IntegerAttr>()) {
      return llvm::json::Value(int_attr.getInt());
    }
    else return llvm::json::Value(std::string(""));
  }
}
  void registerAIRRtTranslations() {

  TranslateFromMLIRRegistration
    registrationMMap("airrt-generate-json", [](ModuleOp module, raw_ostream &output) {
      llvm::json::Object moduleJSON;
      for (auto module_meta : module.getOps<airrt::ModuleMetadataOp>()) {
        for (auto herd_meta : module_meta.getOps<airrt::HerdMetadataOp>()) {
          llvm::json::Object herdJSON;
          for (auto a : herd_meta->getAttrs()) {
            auto ident = a.first;
            auto attr = a.second;
            herdJSON[ident.str()] = attrToJSON(attr);
          }
          moduleJSON[herd_meta.sym_name()] = llvm::json::Value(std::move(herdJSON));
        }
      }
      llvm::json::Value topv(std::move(moduleJSON));
      std::string ret;
      llvm::raw_string_ostream ss(ret);
      ss << llvm::formatv("{0:2}",topv) << "\n";
      output << ss.str();
      return success();
    }, 
    [](DialectRegistry &registry) {
      registry.insert<xilinx::air::airDialect,
                      xilinx::airrt::AIRRtDialect,
                      StandardOpsDialect,
                      memref::MemRefDialect,
                      vector::VectorDialect,
                      LLVM::LLVMDialect,
                      scf::SCFDialect,
                      AffineDialect>();
    });

}

} // namespace air
} // namespace xilinx
