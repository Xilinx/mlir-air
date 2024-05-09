//===- AIRRtToLLVMPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#define DEBUG_TYPE "airrt-to-llvm-pass"

using namespace mlir;
using namespace xilinx;

namespace {
#define GEN_PASS_DEF_AIRRTTOLLVM
#include "air/Conversion/Passes.h.inc"

// struct shim_desc_t {
//   int64_t *location_data;
//   int64_t *channel_data;
// }
LLVM::LLVMStructType getShimDescriptorType(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(ctx,
                                          {
                                              // int64_t[64]* location data
                                              LLVM::LLVMPointerType::get(ctx),
                                              // int64_t[64]* channel data
                                              LLVM::LLVMPointerType::get(ctx),
                                          });
}

// struct herd_desc_t {
//   int64_t name_length;
//   char *name;
//   shim_desc_t *shim_desc;
// }
LLVM::LLVMStructType getHerdDescriptorType(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(ctx,
                                          {
                                              // int64_t name_length
                                              IntegerType::get(ctx, 64),
                                              // char *name
                                              LLVM::LLVMPointerType::get(ctx),
                                              // shim_desc_t *shim_desc
                                              LLVM::LLVMPointerType::get(ctx),
                                          });
}

// struct air_segment_desc_t {
//   int64_t name_length;
//   char *name;
//   uint64_t herd_length;
//   air_herd_desc_t **herd_descs;
// };
LLVM::LLVMStructType getSegmentDescriptorType(MLIRContext *ctx,
                                              int64_t herd_length) {
  return LLVM::LLVMStructType::getLiteral(
      ctx, {
               // int64_t name_length;
               IntegerType::get(ctx, 64),
               // char *name;
               LLVM::LLVMPointerType::get(ctx),
               // uint64_t herd_length;
               IntegerType::get(ctx, 64),
               // air_herd_desc_t *herd_descs[herd_length];
               LLVM::LLVMPointerType::get(ctx),
           });
};

// struct module_desc_t {
//   int64_t length;
//   herd_desc_t *herd_descs[length];
// }
LLVM::LLVMStructType getModuleDescriptorType(MLIRContext *ctx,
                                             ArrayRef<int64_t> herd_count) {
  return LLVM::LLVMStructType::getLiteral(ctx,
                                          {
                                              // int64_t length
                                              IntegerType::get(ctx, 64),
                                              // herd_desc_t *herd_descs[length]
                                              LLVM::LLVMPointerType::get(ctx),
                                          });
}

LLVM::GlobalOp getOrCreateAIRString(OpBuilder builder, ModuleOp module,
                                    StringRef str) {
  std::string llvmSymbolName = std::string("__airrt_string_") + str.str();
  auto global = module.lookupSymbol(llvmSymbolName);
  if (!global) {
    auto arrayTy = LLVM::LLVMArrayType::get(
        IntegerType::get(builder.getContext(), 8), str.size());
    auto loc = builder.getUnknownLoc();
    global = builder.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        llvmSymbolName, builder.getStringAttr(str));
  }
  return cast<LLVM::GlobalOp>(global);
}

LLVM::GlobalOp
createSegmentDescriptor(OpBuilder builder, ModuleOp module,
                        ArrayRef<LLVM::GlobalOp> herd_descs,
                        xilinx::airrt::SegmentMetadataOp segment) {
  auto ctx = module.getContext();
  auto loc = builder.getUnknownLoc();

  auto descTy = getSegmentDescriptorType(ctx, herd_descs.size());

  std::string segment_name = "segment";
  if (auto attr =
          segment->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    segment_name = attr.getValue().str();

  auto segmentName = getOrCreateAIRString(builder, module, segment_name);

  auto arrayElemTy = LLVM::LLVMPointerType::get(ctx);
  auto arrayTy = LLVM::LLVMArrayType::get(arrayElemTy, herd_descs.size());
  std::string str_name = "__airrt_segment_herd_descriptors";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto herd_descs_global = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&herd_descs_global.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i = 0, e = herd_descs.size(); i < e; i++) {
      auto a = builder.create<LLVM::BitcastOp>(
          loc, arrayElemTy,
          builder.create<LLVM::AddressOfOp>(loc, herd_descs[i]));
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, a, builder.getDenseI64ArrayAttr({i}));
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  str_name = "__airrt_segment_descriptor";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
      loc, descTy, /*isConstant=*/true, LLVM::Linkage::External, str_name,
      /*value=*/Attribute());
  if (1) {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&descGlobal.getInitializerRegion());
    Value desc = builder.create<LLVM::UndefOp>(loc, descTy);

    auto segmentNameArray = builder.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(ctx), segmentName.getSymNameAttr());
    auto segmentNameLen = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(ctx, 64),
        builder.getI32IntegerAttr(segment_name.size()));

    builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                segmentName.getType(), segmentNameArray,
                                ArrayRef<LLVM::GEPArg>{0, 0});

    // length of the array of herd_desc_t
    auto herd_descs_len = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(ctx, 64),
        builder.getI64IntegerAttr(herd_descs.size()));

    auto herd_descs_global_addr =
        builder.create<LLVM::AddressOfOp>(loc, herd_descs_global);

    desc = builder.create<LLVM::InsertValueOp>(loc, desc, segmentNameLen,
                                               builder.getDenseI64ArrayAttr(0));

    auto segmentNameArrayPtr = builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(ctx), segmentNameArray);
    desc = builder.create<LLVM::InsertValueOp>(loc, desc, segmentNameArrayPtr,
                                               builder.getDenseI64ArrayAttr(1));

    desc = builder.create<LLVM::InsertValueOp>(loc, desc, herd_descs_len,
                                               builder.getDenseI64ArrayAttr(2));

    auto herd_descs_ptr = builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(ctx), herd_descs_global_addr);
    desc = builder.create<LLVM::InsertValueOp>(loc, desc, herd_descs_ptr,
                                               builder.getDenseI64ArrayAttr(3));

    builder.create<LLVM::ReturnOp>(loc, desc);
  }
  return descGlobal;
}

LLVM::GlobalOp createModuleDescriptor(OpBuilder builder, ModuleOp module,
                                      ArrayRef<LLVM::GlobalOp> segment_descs,
                                      ArrayRef<int64_t> segment_herd_count) {
  auto ctx = module.getContext();
  auto loc = builder.getUnknownLoc();
  auto descTy = getModuleDescriptorType(ctx, segment_herd_count);
  auto arrayElemTy = LLVM::LLVMPointerType::get(ctx);
  auto arrayTy =
      LLVM::LLVMArrayType::get(arrayElemTy, segment_herd_count.size());
  std::string str_name = "__airrt_module_segment_descriptors";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto segment_descs_global = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&segment_descs_global.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i = 0, e = segment_descs.size(); i < e; i++) {
      auto a = builder.create<LLVM::BitcastOp>(
          loc, arrayElemTy,
          builder.create<LLVM::AddressOfOp>(loc, segment_descs[i]));
      data = builder.create<LLVM::InsertValueOp>(
          loc, data, a, builder.getDenseI64ArrayAttr({i}));
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  str_name = "__airrt_module_descriptor";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
      loc, descTy, /*isConstant=*/true, LLVM::Linkage::External, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&descGlobal.getInitializerRegion());
    Value desc = builder.create<LLVM::UndefOp>(loc, descTy);

    // length of the array of herd_desc_t
    auto segment_descs_len = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(ctx, 64),
        builder.getI64IntegerAttr(segment_descs.size()));

    auto segment_descs_global_addr = builder.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(ctx),
        builder.create<LLVM::AddressOfOp>(loc, segment_descs_global));

    desc = builder.create<LLVM::InsertValueOp>(loc, desc, segment_descs_len,
                                               builder.getDenseI64ArrayAttr(0));

    desc = builder.create<LLVM::InsertValueOp>(
        loc, desc, segment_descs_global_addr, builder.getDenseI64ArrayAttr(1));

    builder.create<LLVM::ReturnOp>(loc, desc);
  }
  return descGlobal;
}

LLVM::GlobalOp createHerdDescriptor(OpBuilder builder, ModuleOp module,
                                    LLVM::GlobalOp shim_desc,
                                    xilinx::airrt::HerdMetadataOp herd) {
  auto ctx = module.getContext();
  builder.setInsertionPointAfter(shim_desc);
  auto loc = builder.getUnknownLoc();

  auto descTy = getHerdDescriptorType(ctx);

  std::string herd_name = "herd";
  if (auto attr =
          herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    herd_name = attr.getValue().str();

  auto herdName = getOrCreateAIRString(builder, module, herd_name);

  std::string str_name = "__airrt_herd_descriptor";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
      loc, descTy, /*isConstant=*/true, LLVM::Linkage::External, str_name,
      /*value=*/Attribute());

  builder.createBlock(&descGlobal.getInitializerRegion());

  Value desc = builder.create<LLVM::UndefOp>(loc, descTy);
  auto herdNameArray = builder.create<LLVM::AddressOfOp>(
      loc, LLVM::LLVMPointerType::get(ctx), herdName.getSymNameAttr());
  auto herdNameLen = builder.create<LLVM::ConstantOp>(
      loc, IntegerType::get(ctx, 64),
      builder.getI32IntegerAttr(herd_name.size()));

  auto herdNamePtr = builder.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(ctx),
      builder.create<LLVM::GEPOp>(loc, LLVM::LLVMPointerType::get(ctx),
                                  herdName.getType(), herdNameArray,
                                  ArrayRef<LLVM::GEPArg>{0, 0}));

  desc = builder.create<LLVM::InsertValueOp>(loc, desc, herdNameLen,
                                             builder.getDenseI64ArrayAttr({0}));
  desc = builder.create<LLVM::InsertValueOp>(loc, desc, herdNamePtr,
                                             builder.getDenseI64ArrayAttr({1}));

  Value shimDescPtr = builder.create<LLVM::BitcastOp>(
      loc, LLVM::LLVMPointerType::get(ctx),
      builder.create<LLVM::AddressOfOp>(loc, shim_desc));
  desc = builder.create<LLVM::InsertValueOp>(loc, desc, shimDescPtr,
                                             builder.getDenseI64ArrayAttr({2}));

  builder.create<LLVM::ReturnOp>(loc, desc);
  return descGlobal;
}

LLVM::GlobalOp createShimDescriptor(OpBuilder builder, ModuleOp module,
                                    int64_t cols[16][8][8],
                                    int64_t chans[16][8][8]) {
  auto ctx = module.getContext();
  auto loc = builder.getUnknownLoc();
  auto descTy = getShimDescriptorType(ctx);
  auto arrayTy =
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 64), 16 * 8 * 8);

  // construct the location data global array + initializer
  std::string str_name = "__airrt_shim_location_data";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto locArrayGlobal = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&locArrayGlobal.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
          auto c = builder.create<LLVM::ConstantOp>(
              loc, IntegerType::get(ctx, 64),
              builder.getI64IntegerAttr(cols[i][j][k]));
          data = builder.create<LLVM::InsertValueOp>(
              loc, data, c,
              builder.getDenseI64ArrayAttr({i * 8 * 8 + j * 8 + k}));
        }
      }
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  // construct the channel data global array + initializer
  str_name = "__airrt_shim_channel_data";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto chanArrayGlobal = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&chanArrayGlobal.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i = 0; i < 16; i++) {
      for (int j = 0; j < 8; j++) {
        for (int k = 0; k < 8; k++) {
          auto c = builder.create<LLVM::ConstantOp>(
              loc, IntegerType::get(ctx, 64),
              builder.getI32IntegerAttr(chans[i][j][k]));
          data = builder.create<LLVM::InsertValueOp>(
              loc, data, c,
              builder.getDenseI64ArrayAttr({i * 8 * 8 + j * 8 + k}));
        }
      }
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  // construct the shim descriptor + initializer
  str_name = "__airrt_shim_descriptor";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
      loc, descTy, /*isConstant=*/true, LLVM::Linkage::Internal, str_name,
      /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&descGlobal.getInitializerRegion());

    Value desc = builder.create<LLVM::UndefOp>(loc, descTy);

    Type arrayPtrTy = LLVM::LLVMPointerType::get(ctx);
    Value locArrayPtr = builder.create<LLVM::BitcastOp>(
        loc, arrayPtrTy,
        builder.create<LLVM::AddressOfOp>(loc, locArrayGlobal));
    desc = builder.create<LLVM::InsertValueOp>(
        loc, desc, locArrayPtr, builder.getDenseI64ArrayAttr({0}));

    Value chanArrayPtr = builder.create<LLVM::BitcastOp>(
        loc, arrayPtrTy,
        builder.create<LLVM::AddressOfOp>(loc, chanArrayGlobal));
    desc = builder.create<LLVM::InsertValueOp>(
        loc, desc, chanArrayPtr, builder.getDenseI64ArrayAttr({1}));

    builder.create<LLVM::ReturnOp>(loc, desc);
  }
  return descGlobal;
}

class ModuleMetadataToLLVMConversion
    : public OpRewritePattern<xilinx::airrt::ModuleMetadataOp> {
public:
  using OpRewritePattern<xilinx::airrt::ModuleMetadataOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::ModuleMetadataOp op,
                                PatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    SmallVector<LLVM::GlobalOp, 4> segment_descs;
    SmallVector<int64_t, 4> segment_herd_count;
    auto &segment_block = op.getSegments().front();
    for (auto segment_meta :
         segment_block.getOps<xilinx::airrt::SegmentMetadataOp>()) {

      SmallVector<LLVM::GlobalOp, 4> herd_descs;
      auto &herd_block = segment_meta.getHerds().front();
      for (auto herd_meta :
           herd_block.getOps<xilinx::airrt::HerdMetadataOp>()) {

        int64_t cols[16][8][8] = {{{0}}};
        int64_t chans[16][8][8] = {{{0}}};

        // "dma_allocations" attribute is an array of DictAttr
        ArrayAttr shim_attr =
            herd_meta->getAttrOfType<ArrayAttr>("dma_allocations");
        assert(shim_attr);
        for (auto &shim_alloc : shim_attr) {
          auto shim_alloc_dict = shim_alloc.cast<DictionaryAttr>();
          auto id = shim_alloc_dict.get("id").cast<IntegerAttr>().getInt();
          auto row = shim_alloc_dict.get("row").cast<IntegerAttr>().getInt();
          auto col = shim_alloc_dict.get("col").cast<IntegerAttr>().getInt();
          auto channel =
              shim_alloc_dict.get("channel").cast<IntegerAttr>().getInt();
          auto location =
              shim_alloc_dict.get("location").cast<IntegerAttr>().getInt();
          cols[id - 1][row][col] = location;
          chans[id - 1][row][col] = channel;
        }

        auto shim_desc = createShimDescriptor(rewriter, module, cols, chans);
        herd_descs.push_back(
            createHerdDescriptor(rewriter, module, shim_desc, herd_meta));
      }
      segment_herd_count.push_back(herd_descs.size());
      segment_descs.push_back(
          createSegmentDescriptor(rewriter, module, herd_descs, segment_meta));
    }
    auto desc = createModuleDescriptor(rewriter, module, segment_descs,
                                       segment_herd_count);
    rewriter.replaceOp(op, desc->getResults());

    return success();
  }
};

class SegmentLoadToLLVMConversion
    : public OpRewritePattern<xilinx::airrt::SegmentLoadOp> {
public:
  using OpRewritePattern<xilinx::airrt::SegmentLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::SegmentLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    SmallVector<Type> retTys{IntegerType::get(ctx, 64)};
    SmallVector<Type, 1> tys{LLVM::LLVMPointerType::get(ctx)};
    auto functionTy = FunctionType::get(ctx, tys, retTys);

    auto module = op->getParentOfType<ModuleOp>();

    rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
    auto segment_name = getOrCreateAIRString(rewriter, module, op.getSymName());

    auto funcOp = dyn_cast_if_present<func::FuncOp>(
        module.lookupSymbol("__airrt_segment_load"));
    if (!funcOp) {
      funcOp = rewriter.create<func::FuncOp>(
          op->getLoc(), "__airrt_segment_load", functionTy);
      funcOp.setPrivate();
    }
    rewriter.setInsertionPoint(op);

    auto segment_name_addr =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), segment_name);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto segment_name_addr_cast = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), ptrTy, segment_name_addr);
    SmallVector<Value, 2> operands{segment_name_addr_cast};

    auto call = rewriter.create<func::CallOp>(
        op->getLoc(), retTys, SymbolRefAttr::get(funcOp), operands);
    rewriter.replaceOp(op, call->getResults());
    return success();
  }
};

class HerdLoadToLLVMConversion
    : public OpRewritePattern<xilinx::airrt::HerdLoadOp> {
public:
  using OpRewritePattern<xilinx::airrt::HerdLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::HerdLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto ctx = op->getContext();
    SmallVector<Type> retTys{IntegerType::get(ctx, 64)};
    SmallVector<Type, 1> tys{LLVM::LLVMPointerType::get(ctx)};
    auto functionTy = FunctionType::get(ctx, tys, retTys);

    auto module = op->getParentOfType<ModuleOp>();

    rewriter.setInsertionPoint(op->getParentOfType<func::FuncOp>());
    auto herd_name = getOrCreateAIRString(rewriter, module, op.getSymName());

    auto funcOp = dyn_cast_if_present<func::FuncOp>(
        module.lookupSymbol("__airrt_herd_load"));
    if (!funcOp) {
      funcOp = rewriter.create<func::FuncOp>(op->getLoc(), "__airrt_herd_load",
                                             functionTy);
      funcOp.setPrivate();
    }
    rewriter.setInsertionPoint(op);

    auto herd_name_addr =
        rewriter.create<LLVM::AddressOfOp>(op->getLoc(), herd_name);
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto herd_name_addr_cast =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), ptrTy, herd_name_addr);
    SmallVector<Value, 2> operands{herd_name_addr_cast};

    auto call = rewriter.create<func::CallOp>(
        op->getLoc(), retTys, SymbolRefAttr::get(funcOp), operands);
    rewriter.replaceOp(op, call->getResults());
    return success();
  }
};

LogicalResult lowerDmaNdMemcpy(Operation *op, PatternRewriter &rewriter,
                               std::string fnName) {
  auto ctx = op->getContext();
  auto loc = op->getLoc();

  SmallVector<Type, 6> tys;
  SmallVector<Type, 1> retTys;
  SmallVector<Value, 16> operands;

  auto i32Ty = IntegerType::get(ctx, 32);
  auto signalTy = LLVM::LLVMPointerType::get(ctx);
  tys.push_back(signalTy);
  if (op->getNumResults()) {
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(1));
    auto signal = rewriter.create<LLVM::AllocaOp>(loc, signalTy, i32Ty, one, 4);
    operands.push_back(signal);
  } else {
    auto nullV = rewriter.create<LLVM::ZeroOp>(loc, signalTy).getResult();
    operands.push_back(nullV);
  }

  for (auto o : op->getOperands()) {
    tys.push_back(o.getType());
    operands.push_back(o);
  }

  MemRefType memrefTy = tys[4].cast<MemRefType>();
  tys[4] = MemRefType::get(
      std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
      memrefTy.getElementType(), memrefTy.getLayout(),
      memrefTy.getMemorySpace());

  operands[4] = rewriter.create<memref::CastOp>(loc, tys[4], operands[4]);

  // mangle the name by appending '_<rank>d<space><type>'
  llvm::raw_string_ostream ss(fnName);
  ss << "_" << memrefTy.getRank();
  ss << "d" << memrefTy.getMemorySpaceAsInt();
  memrefTy.getElementType().print(ss);

  auto module = op->getParentOfType<ModuleOp>();
  auto fn = module.lookupSymbol<func::FuncOp>(fnName);
  if (!fn) {
    auto fnTy = FunctionType::get(ctx, tys, retTys);
    fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  rewriter.create<func::CallOp>(loc, retTys, SymbolRefAttr::get(fn), operands);
  if (op->getNumResults()) {
    rewriter.replaceOp(op, operands[0]);
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult lowerNdMemcpy(Operation *op, PatternRewriter &rewriter,
                            std::string fnName) {
  auto ctx = op->getContext();
  auto loc = op->getLoc();

  auto dmaOp = cast<xilinx::airrt::MemcpyNdOp>(op);
  SmallVector<Type, 6> tys;
  SmallVector<Value, 16> operands;

  SmallVector<Type, 1> retTys(op->getNumResults(),
                              LLVM::LLVMPointerType::get(ctx));

  auto i32Ty = IntegerType::get(ctx, 32);
  auto signalTy = LLVM::LLVMPointerType::get(ctx);
  if (op->getNumResults()) {
    auto one = rewriter.create<LLVM::ConstantOp>(loc, i32Ty,
                                                 rewriter.getI32IntegerAttr(1));
    auto signal = rewriter.create<LLVM::AllocaOp>(loc, signalTy, i32Ty, one, 4);
    operands.push_back(signal);
  } else {
    auto nullV = rewriter.create<LLVM::ZeroOp>(loc, signalTy).getResult();
    operands.push_back(nullV);
  }

  MemRefType dstMemRefTy = llvm::cast<MemRefType>(dmaOp.getDst().getType());
  MemRefType srcMemRefTy = llvm::cast<MemRefType>(dmaOp.getSrc().getType());

  for (auto o : op->getOperands())
    operands.push_back(o);

  operands[1] = rewriter.create<memref::CastOp>(
      loc,
      MemRefType::get(
          std::vector<int64_t>(dstMemRefTy.getRank(), ShapedType::kDynamic),
          dstMemRefTy.getElementType(), dstMemRefTy.getLayout(),
          dstMemRefTy.getMemorySpace()),
      operands[1]);
  operands[2] = rewriter.create<memref::CastOp>(
      loc,
      MemRefType::get(
          std::vector<int64_t>(srcMemRefTy.getRank(), ShapedType::kDynamic),
          srcMemRefTy.getElementType(), srcMemRefTy.getLayout(),
          srcMemRefTy.getMemorySpace()),
      operands[2]);

  for (auto o : operands)
    tys.push_back(o.getType());

  // mangle the name by appending '_<rank>d<space><type>'
  llvm::raw_string_ostream ss(fnName);
  ss << "_" << dstMemRefTy.getRank();
  ss << "d" << dstMemRefTy.getMemorySpaceAsInt();
  dstMemRefTy.getElementType().print(ss);
  ss << "_" << srcMemRefTy.getRank();
  ss << "d" << srcMemRefTy.getMemorySpaceAsInt();
  srcMemRefTy.getElementType().print(ss);

  auto module = op->getParentOfType<ModuleOp>();
  auto fn = module.lookupSymbol<func::FuncOp>(fnName);
  if (!fn) {
    auto fnTy = FunctionType::get(ctx, tys, retTys);
    fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  auto call = rewriter.create<func::CallOp>(op->getLoc(), retTys,
                                            SymbolRefAttr::get(fn), operands);
  if (op->getNumResults()) {
    rewriter.replaceOp(op, call.getResults());
  } else {
    rewriter.eraseOp(op);
  }
  return success();
}

class DmaMemcpyNdToLLVMConversion
    : public OpRewritePattern<xilinx::airrt::DmaMemcpyNdOp> {
public:
  using OpRewritePattern<xilinx::airrt::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::DmaMemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    return lowerDmaNdMemcpy(op, rewriter, "__airrt_dma_nd_memcpy");
  }
};

class MemcpyNdToLLVMConversion
    : public OpRewritePattern<xilinx::airrt::MemcpyNdOp> {
public:
  using OpRewritePattern<xilinx::airrt::MemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::MemcpyNdOp op,
                                PatternRewriter &rewriter) const override {
    return lowerNdMemcpy(op, rewriter, "__airrt_nd_memcpy");
  }
};

class L1AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getType();
    if (op.getType().getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    auto alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(),
        MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                        memrefTy.getLayout(), 0));
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class L1DeallocOpConversion : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.create<memref::DeallocOp>(op.getLoc(), adaptor.getMemref());
    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineStoreOpConversion
    : public OpConversionPattern<affine::AffineStoreOp> {
public:
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class L1MemRefLoadOpConversion : public OpConversionPattern<memref::LoadOp> {
public:
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    auto load =
        rewriter.create<memref::LoadOp>(op.getLoc(), op->getResultTypes(),
                                        adaptor.getOperands(), op->getAttrs());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class L1MemRefStoreOpConversion : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineLoadOpConversion
    : public OpConversionPattern<affine::AffineLoadOp> {
public:
  using OpConversionPattern<affine::AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    auto load = rewriter.create<affine::AffineLoadOp>(
        op.getLoc(), op->getResultTypes(), adaptor.getOperands(),
        op->getAttrs());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class L2AllocOpConversion : public OpRewritePattern<xilinx::airrt::AllocOp> {
public:
  using OpRewritePattern<xilinx::airrt::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::AllocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;

    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2)
      return failure();

    tys.push_back(IndexType::get(ctx));
    retTys.push_back(MemRefType::get(
        std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
        memrefTy.getElementType(), memrefTy.getLayout(),
        memrefTy.getMemorySpace()));

    auto size = xilinx::air::getTensorVolume(memrefTy);
    operands.push_back(
        rewriter.create<arith::ConstantIndexOp>(op->getLoc(), size));

    auto module = op->getParentOfType<ModuleOp>();

    std::string fnName = "__airrt_alloc_L2";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << memrefTy.getRank();
    ss << "d" << memrefTy.getMemorySpaceAsInt();
    memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
    auto castOp = rewriter.create<memref::CastOp>(op->getLoc(), memrefTy,
                                                  callOp.getResult(0));
    rewriter.replaceOp(op, castOp->getResults());
    return success();
  }
};

class L2DeallocOpConversion
    : public OpRewritePattern<xilinx::airrt::DeallocOp> {
public:
  using OpRewritePattern<xilinx::airrt::DeallocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xilinx::airrt::DeallocOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> operands;
    SmallVector<Type, 1> tys;
    SmallVector<Type, 1> retTys;
    auto ctx = op->getContext();

    auto memrefTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2)
      return failure();

    tys.push_back(MemRefType::get(
        std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
        memrefTy.getElementType(), memrefTy.getLayout(),
        memrefTy.getMemorySpace()));
    operands.push_back(
        rewriter.create<memref::CastOp>(op->getLoc(), tys[0], op.getMemref()));

    auto module = op->getParentOfType<ModuleOp>();

    std::string fnName = "__airrt_dealloc_L2";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << memrefTy.getRank();
    ss << "d" << memrefTy.getMemorySpaceAsInt();
    memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    rewriter.create<func::CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn),
                                  operands);
    rewriter.eraseOp(op);
    return success();
  }
};

class WaitAllOpConversion
    : public OpConversionPattern<xilinx::airrt::WaitAllOp> {
public:
  using OpConversionPattern<xilinx::airrt::WaitAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto module = op->getParentOfType<ModuleOp>();
    auto ctx = op->getContext();

    SmallVector<Type, 8> tys(operands.size(), LLVM::LLVMPointerType::get(ctx));
    SmallVector<Type, 1> retTys(op->getNumResults(),
                                LLVM::LLVMPointerType::get(ctx));

    std::string fnName = "__airrt_wait_all";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << retTys.size() << "_" << operands.size();

    auto fn = module.lookupSymbol<func::FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = func::FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, retTys,
                                              SymbolRefAttr::get(fn), operands);
    return success();
  }
};

class ScfYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getOperands()};
    SmallVector<Type> retTys;
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, retTys, operands);
    return success();
  }
};

class ScfReduceOpConversion : public OpConversionPattern<scf::ReduceOp> {
public:
  using OpConversionPattern<scf::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        rewriter.create<scf::ReduceOp>(op->getLoc(), adaptor.getOperands());
    auto body = &op.getRegion(0).front();
    auto newBody = &newOp.getRegion(0).front();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());
    rewriter.eraseOp(op);
    return success();
  }
};

class ScfReduceReturnOpConversion
    : public OpConversionPattern<scf::ReduceReturnOp> {
public:
  using OpConversionPattern<scf::ReduceReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    SmallVector<Type, 2> retTys;
    rewriter.replaceOpWithNewOp<scf::ReduceReturnOp>(op, retTys, operands);
    return success();
  }
};

class ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newFor = rewriter.create<scf::ForOp>(
        op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    auto body = op.getBody();
    auto newBody = newFor.getBody();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());

    rewriter.replaceOp(op, newFor.getResults());
    return success();
  }
};

class ScfIfOpConversion : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type> retTys;
    if (typeConverter->convertTypes(op.getResultTypes(), retTys).failed())
      return failure();

    bool hasElseBlock = op.elseBlock() != nullptr;
    auto newIf = rewriter.create<scf::IfOp>(op->getLoc(), retTys,
                                            op.getCondition(), hasElseBlock);

    auto &thenOps = op.thenBlock()->getOperations();
    auto &newThenOps = newIf.thenBlock()->getOperations();
    newThenOps.splice(newThenOps.begin(), thenOps, thenOps.begin(),
                      thenOps.end());

    if (!hasElseBlock)
      return success();

    auto &elseOps = op.elseBlock()->getOperations();
    auto &newElseOps = newIf.elseBlock()->getOperations();
    newElseOps.splice(newElseOps.begin(), elseOps, elseOps.begin(),
                      elseOps.end());

    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

class ScfParOpConversion : public OpConversionPattern<scf::ParallelOp> {
public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newPar = rewriter.create<scf::ParallelOp>(
        op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitVals());
    auto body = op.getBody();
    auto newBody = newPar.getBody();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());

    rewriter.replaceOp(op, newPar.getResults());
    return success();
  }
};

class CallOpConversion : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type> retTys;
    if (typeConverter->convertTypes(op.getResultTypes(), retTys).failed())
      return failure();

    auto callee = adaptor.getCallee();
    if (callee.starts_with("__airrt_"))
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    auto sym = module.lookupSymbol(callee);
    if (!sym)
      return failure();

    auto funcOp = dyn_cast<func::FuncOp>(sym);
    if (!funcOp)
      return failure();

    if (retTys.size() == 0 && funcOp.isExternal()) {
      rewriter.eraseOp(op);
      return success();
    }

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), adaptor.getCallee(), retTys, adaptor.getOperands());
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

class AIRRtToLLVM : public impl::AIRRtToLLVMBase<AIRRtToLLVM> {

public:
  AIRRtToLLVM() {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    LLVMTypeConverter converter(context);

    converter.addConversion([&](Type type) -> std::optional<Type> {
      // convert L1 memrefs to L3
      if (auto memref = llvm::dyn_cast<MemRefType>(type))
        if (memref.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1)
          return mlir::MemRefType::get(memref.getShape(),
                                       memref.getElementType(),
                                       memref.getLayout(), 0);
      if (auto t = llvm::dyn_cast<xilinx::airrt::EventType>(type))
        return LLVM::LLVMPointerType::get(context);
      return std::optional<Type>(type);
    });

    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return std::optional<Value>(cast.getResult(0));
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(context);
    patterns.add<ModuleMetadataToLLVMConversion, SegmentLoadToLLVMConversion,
                 HerdLoadToLLVMConversion, DmaMemcpyNdToLLVMConversion,
                 MemcpyNdToLLVMConversion, L2AllocOpConversion,
                 L2DeallocOpConversion>(context);
    patterns.add<ScfIfOpConversion, ScfYieldOpConversion,
                 ScfReduceReturnOpConversion, ScfReduceOpConversion,
                 ScfForOpConversion, ScfParOpConversion, L1AllocOpConversion,
                 L1AffineLoadOpConversion, L1AffineStoreOpConversion,
                 L1MemRefLoadOpConversion, L1MemRefStoreOpConversion,
                 L1DeallocOpConversion, WaitAllOpConversion, CallOpConversion>(
        converter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           arith::ArithDialect, affine::AffineDialect,
                           scf::SCFDialect, memref::MemRefDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() == 0);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (llvm::cast<MemRefType>(op.getMemref().getType())
                  .getMemorySpaceAsInt() == 0);
    });

    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          return (op.getMemref()
                      .getType()
                      .cast<MemRefType>()
                      .getMemorySpaceAsInt() !=
                  (int)xilinx::air::MemorySpace::L1);
        });

    target.addDynamicallyLegalOp<affine::AffineLoadOp>(
        [&](affine::AffineLoadOp op) {
          return (llvm::cast<MemRefType>(op.getMemref().getType())
                      .getMemorySpaceAsInt() !=
                  (int)xilinx::air::MemorySpace::L1);
        });

    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      return (llvm::cast<MemRefType>(op.getMemref().getType())
                  .getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1);
    });

    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      return (llvm::cast<MemRefType>(op.getMemref().getType())
                  .getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1);
    });

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      for (auto t : op.getOperandTypes()) {
        if (auto mty = llvm::dyn_cast<MemRefType>(t))
          if (mty.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1)
            return false;
      }
      for (auto t : op.getResultTypes()) {
        if (auto mty = llvm::dyn_cast<MemRefType>(t))
          if (mty.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1)
            return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      for (auto o : op.getRegionIterArgs()) {
        if (llvm::isa<xilinx::airrt::EventType>(o.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ParallelOp>([&](scf::ParallelOp op) {
      for (auto o : op.getInitVals()) {
        if (llvm::isa<xilinx::airrt::EventType>(o.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      for (auto v : op.getOperands()) {
        if (llvm::isa<xilinx::airrt::EventType>(v.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceOp>([&](scf::ReduceOp op) {
      for (auto oper : op.getOperands()) {
        if (llvm::isa<xilinx::airrt::EventType>(oper.getType()))
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ReduceReturnOp>(
        [&](scf::ReduceReturnOp op) {
          if (llvm::isa<xilinx::airrt::EventType>(op.getResult().getType()))
            return false;
          else
            return true;
        });

    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      for (auto v : op.getResults()) {
        if (llvm::isa<xilinx::airrt::EventType>(v.getType()))
          return false;
      }
      return true;
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering AIRRt\n");
      signalPassFailure();
    }

    SmallVector<func::FuncOp> erased_extern;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() && func.symbolKnownUseEmpty(module))
        erased_extern.push_back(func);
      else
        func->setAttr("llvm.emit_c_interface",
                      UnitAttr::get(func.getContext()));
    }
    for (auto e : erased_extern)
      e.erase();
  }

private:
};

} // namespace

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToLLVMPass() {
  return std::make_unique<AIRRtToLLVM>();
}

} // namespace airrt
} // namespace xilinx
