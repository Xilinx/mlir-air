// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


#include "PassDetail.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#define DEBUG_TYPE "airrt-to-llvm-pass"

using namespace mlir;
using namespace xilinx::air;

namespace {

// struct shim_desc_t {
//   int64_t *location_data;
//   int64_t *channel_data;
// }
LLVM::LLVMStructType getShimDescriptorType(MLIRContext *ctx ) {
  return LLVM::LLVMStructType::getLiteral(ctx,{
    // int64_t[64]* location data
    LLVM::LLVMPointerType::get(
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 64), 16*8*8)),
    // int64_t[64]* channel data
    LLVM::LLVMPointerType::get(
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 64), 16*8*8)),
  });
}

// struct herd_desc_t {
//   int32_t name_length;
//   char *name;
//   shim_desc_t *shim_desc;
// }
LLVM::LLVMStructType getHerdDescriptorType(MLIRContext *ctx ) {
  return LLVM::LLVMStructType::getLiteral(ctx,{
    // int32_t name_length
    IntegerType::get(ctx, 32),
    // char *name
    LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)),
    // shim_desc_t *shim_desc
    LLVM::LLVMPointerType::get(getShimDescriptorType(ctx)),
  });
}

// struct module_desc_t {
//   int64_t length;
//   herd_desc_t *herd_descs[length];
// }
LLVM::LLVMStructType getModuleDescriptorType(MLIRContext *ctx, int64_t length) {
  return LLVM::LLVMStructType::getLiteral(ctx,{
    // int64_t length
    IntegerType::get(ctx, 64),
    // herd_desc_t *herd_descs[length]
    LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(
      LLVM::LLVMPointerType::get(getHerdDescriptorType(ctx)),length)),
  });
}

LLVM::GlobalOp createModuleDescriptor(OpBuilder builder,
                                      ModuleOp module,
                                      std::vector<LLVM::GlobalOp> &herd_descs)
{
  auto ctx = module.getContext();
  auto loc = builder.getUnknownLoc();
  auto descTy = getModuleDescriptorType(ctx, herd_descs.size());
  auto arrayTy =
    LLVM::LLVMArrayType::get(
      LLVM::LLVMPointerType::get(getHerdDescriptorType(ctx)),
                                 herd_descs.size());
  std::string str_name = "__air_module_herd_descriptors";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto herd_descs_global = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
      str_name, /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&herd_descs_global.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i=0,e=herd_descs.size(); i<e; i++) {
      auto a = builder.create<LLVM::AddressOfOp>(loc, herd_descs[i]);
      data = builder.create<LLVM::InsertValueOp>(
              loc, data, a, builder.getI32ArrayAttr({i}));
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }
  
  str_name = "__air_module_descriptor";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
    loc, descTy, /*isConstant=*/true, LLVM::Linkage::External,
    str_name, /*value=*/Attribute());
  if (1) {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&descGlobal.getInitializerRegion());
    Value desc = builder.create<LLVM::UndefOp>(loc, descTy);

    // length of the array of herd_desc_t
    auto herd_descs_len = 
      builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64),
                                      builder.getI64IntegerAttr(herd_descs.size()));

    auto herd_descs_global_addr = builder.create<LLVM::AddressOfOp>(loc, herd_descs_global);

    desc = builder.create<LLVM::InsertValueOp>(loc, desc, herd_descs_len,
                                               builder.getI32ArrayAttr(0));

    desc = builder.create<LLVM::InsertValueOp>(loc, desc, herd_descs_global_addr,
                                               builder.getI32ArrayAttr(1));

    builder.create<LLVM::ReturnOp>(loc, desc);
  }
  return descGlobal;
}

LLVM::GlobalOp getOrCreateAIRString(OpBuilder builder, ModuleOp module, StringRef str)
{
  std::string llvmSymbolName = std::string("__air_string_") + str.str();
  auto global = module.lookupSymbol(llvmSymbolName);
  if (!global) {
    auto arrayTy =
      LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8),
                               str.size());
    auto loc = builder.getUnknownLoc();
    global = builder.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        llvmSymbolName, builder.getStringAttr(str));
  }
  return cast<LLVM::GlobalOp>(global);
}

LLVM::GlobalOp createHerdDescriptor(OpBuilder builder, ModuleOp module,
                                    LLVM::GlobalOp shim_desc,
                                    xilinx::airrt::HerdMetadataOp herd)
{
  auto ctx = module.getContext();
  builder.setInsertionPointAfter(shim_desc);
  auto loc = builder.getUnknownLoc();

  auto descTy = getHerdDescriptorType(ctx);

  std::string herd_name = "herd";
  if (auto attr = herd->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    herd_name = attr.getValue().str();

  auto herdName = getOrCreateAIRString(builder, module, herd_name);

  std::string str_name = "__air_herd_descriptor";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
    loc, descTy, /*isConstant=*/true, LLVM::Linkage::External,
    str_name, /*value=*/Attribute());

  builder.createBlock(&descGlobal.getInitializerRegion());

  Value desc = builder.create<LLVM::UndefOp>(loc, descTy);
  auto herdNameArray = builder.create<LLVM::AddressOfOp>(loc, herdName);
  auto herdNameLen = 
    builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32),
                                      builder.getI32IntegerAttr(herd_name.size()));

  auto c0 = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 32),
                                              builder.getI32IntegerAttr(0));
  auto herdNamePtr = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8)), herdNameArray,
      ValueRange({c0, c0}));

  desc = builder.create<LLVM::InsertValueOp>(loc, desc, herdNameLen,
                                              builder.getI64ArrayAttr({0}));
  desc = builder.create<LLVM::InsertValueOp>(loc, desc, herdNamePtr,
                                              builder.getI64ArrayAttr({1}));

  Value shimDescPtr = builder.create<LLVM::AddressOfOp>(loc, shim_desc);
  desc = builder.create<LLVM::InsertValueOp>(loc, desc, shimDescPtr,
                                              builder.getI64ArrayAttr({2}));

  builder.create<LLVM::ReturnOp>(loc, desc);
  return descGlobal;
}

LLVM::GlobalOp createShimDescriptor(OpBuilder builder,
                                    ModuleOp module,
                                    int64_t cols[16][8][8],
                                    int64_t chans[16][8][8])
{
  auto ctx = module.getContext();
  auto loc = builder.getUnknownLoc();
  auto descTy = getShimDescriptorType(ctx);
  auto arrayTy = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 64), 16*8*8);

  // construct the location data global array + initializer
  std::string str_name = "__air_shim_location_data";
  int which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto locArrayGlobal = builder.create<LLVM::GlobalOp>(
    loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
    str_name, /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&locArrayGlobal.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i=0; i<16; i++) {
      for (int j=0; j<8; j++) {
        for (int k=0; k<8; k++) {
          auto c = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64),
                                                    builder.getI64IntegerAttr(cols[i][j][k]));
          data = builder.create<LLVM::InsertValueOp>(loc, data, c,
                                                      builder.getI64ArrayAttr({i*8*8+j*8+k}));
        }
      }
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  // construct the channel data global array + initializer
  str_name = "__air_shim_channel_data";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto chanArrayGlobal = builder.create<LLVM::GlobalOp>(
    loc, arrayTy, /*isConstant=*/true, LLVM::Linkage::Internal,
    str_name, /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&chanArrayGlobal.getInitializerRegion());
    Value data = builder.create<LLVM::UndefOp>(loc, arrayTy);
    for (int i=0; i<16; i++) {
      for (int j=0; j<8; j++) {
        for (int k=0; k<8; k++) {
          auto c = builder.create<LLVM::ConstantOp>(loc, IntegerType::get(ctx, 64),
                                                    builder.getI32IntegerAttr(chans[i][j][k]));
          data = builder.create<LLVM::InsertValueOp>(loc, data, c,
                                                      builder.getI32ArrayAttr({i*8*8+j*8+k}));
        }
      }
    }
    builder.create<LLVM::ReturnOp>(loc, data);
  }

  // construct the shim descriptor + initializer
  str_name = "__air_shim_descriptor";
  which_try = 0;
  while (module.lookupSymbol(str_name))
    str_name = str_name + "_" + std::to_string(++which_try);
  auto descGlobal = builder.create<LLVM::GlobalOp>(
    loc, descTy, /*isConstant=*/true, LLVM::Linkage::Internal,
    str_name, /*value=*/Attribute());
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.createBlock(&descGlobal.getInitializerRegion());

    Value desc = builder.create<LLVM::UndefOp>(loc, descTy);

    Value locArrayPtr = builder.create<LLVM::AddressOfOp>(loc, locArrayGlobal);
    desc = builder.create<LLVM::InsertValueOp>(loc, desc, locArrayPtr,
                                                builder.getI64ArrayAttr({0}));

    Value chanArrayPtr = builder.create<LLVM::AddressOfOp>(loc, chanArrayGlobal);
    desc = builder.create<LLVM::InsertValueOp>(loc, desc, chanArrayPtr,
                                                builder.getI64ArrayAttr({1}));

    builder.create<LLVM::ReturnOp>(loc, desc);
  }
  return descGlobal;
}


class ModuleMetadataToLLVMConversion : public OpRewritePattern<xilinx::airrt::ModuleMetadataOp> {
public:
  using OpRewritePattern<xilinx::airrt::ModuleMetadataOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::ModuleMetadataOp op,
                  PatternRewriter &rewriter) const override
  {
    auto module = op->getParentOfType<ModuleOp>();
    std::vector<LLVM::GlobalOp> herd_descs;
    for (auto &herd_op : op.herds().front()) {
      auto herd_meta = dyn_cast<xilinx::airrt::HerdMetadataOp>(herd_op);
      if (!herd_meta) continue;

      int64_t cols[16][8][8] = {{{0}}};
      int64_t chans[16][8][8] = {{{0}}};

      // "shim_allocations" attribute is an array of DictAttr
      ArrayAttr shim_attr = herd_meta->getAttrOfType<ArrayAttr>("shim_allocations");
      assert(shim_attr);
      for (auto &shim_alloc : shim_attr) {
        auto shim_alloc_dict = shim_alloc.cast<DictionaryAttr>();
        auto id = shim_alloc_dict.get("id").cast<IntegerAttr>().getInt();
        auto row = shim_alloc_dict.get("row").cast<IntegerAttr>().getInt();
        auto col= shim_alloc_dict.get("col").cast<IntegerAttr>().getInt();
        auto channel = shim_alloc_dict.get("channel").cast<IntegerAttr>().getInt();
        auto location = shim_alloc_dict.get("location").cast<IntegerAttr>().getInt();
        cols[id-1][row][col] = location;
        chans[id-1][row][col] = channel;
      }

      auto shim_desc = createShimDescriptor(rewriter, module, cols, chans);
      herd_descs.push_back(
        createHerdDescriptor(rewriter, module, shim_desc, herd_meta));
    }
    auto desc = createModuleDescriptor(rewriter, module, herd_descs);
    rewriter.replaceOp(op, desc->getResults());

    return success();
  }
};

class HerdLoadToLLVMConversion : public OpRewritePattern<xilinx::airrt::HerdLoadOp> {
public:
  using OpRewritePattern<xilinx::airrt::HerdLoadOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::HerdLoadOp op,
                  PatternRewriter &rewriter) const override
  {
    auto ctx = op->getContext();
    auto retTy = IntegerType::get(ctx, 64);
    SmallVector<Type, 1> tys{
      LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8))};
    auto functionTy = LLVM::LLVMFunctionType::get(retTy, tys);

    auto module = op->getParentOfType<ModuleOp>();

    rewriter.setInsertionPoint(op->getParentOfType<FuncOp>());
    auto herd_name = getOrCreateAIRString(rewriter, module, op.sym_name());

    auto funcOpSym = module.lookupSymbol("air_herd_load");
    LLVM::LLVMFuncOp funcOp = nullptr;
    if (funcOpSym)
      funcOp = cast<LLVM::LLVMFuncOp>(funcOpSym);
    else
      funcOp = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(), "air_herd_load",
                                        functionTy, LLVM::Linkage::External);
    rewriter.setInsertionPoint(op);

    auto herd_name_addr = rewriter.create<LLVM::AddressOfOp>(op->getLoc(), herd_name);
    auto ptrTy = LLVM::LLVMPointerType::get(IntegerType::get(ctx, 8));
    auto herd_name_addr_cast = rewriter.create<LLVM::BitcastOp>(op->getLoc(), ptrTy, herd_name_addr);
    SmallVector<Value, 2> operands{herd_name_addr_cast};

    LLVM::CallOp call = rewriter.create<LLVM::CallOp>(op->getLoc(), funcOp, operands);
    rewriter.replaceOp(op, call->getResults());
    return success();
  }
};

LogicalResult
lowerDmaMemcpy(Operation* op, PatternRewriter &rewriter, std::string fnName)
{
  auto ctx = op->getContext();

  SmallVector<Type, 6> tys;
  SmallVector<Type, 1> retTys;
  for (auto o : op->getOperands())
    tys.push_back(o.getType());
  MemRefType memrefTy = tys[3].cast<MemRefType>();
  tys[3] = MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                           memrefTy.getElementType(),
                           memrefTy.getLayout(),
                           memrefTy.getMemorySpace());
  auto module = op->getParentOfType<ModuleOp>();

  auto fnTy = FunctionType::get(ctx, tys, retTys);
  auto fn = module.lookupSymbol<FuncOp>(fnName);
  if (!fn) {
    fn = FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }
  SmallVector<Value, 16> operands = op->getOperands();
  operands[3] = rewriter.create<memref::CastOp>(op->getLoc(), operands[3], tys[3]);
  CallOp call = rewriter.create<CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
  rewriter.replaceOp(op, call->getResults());
  return success();
}

LogicalResult
lowerDmaNdMemcpy(Operation* op, PatternRewriter &rewriter, std::string fnName)
{
  auto ctx = op->getContext();

  SmallVector<Type, 6> tys;
  SmallVector<Type, 1> retTys;
  for (auto o : op->getOperands())
    tys.push_back(o.getType());
  MemRefType memrefTy = tys[3].cast<MemRefType>();
  tys[3] = MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                           memrefTy.getElementType(),
                           memrefTy.getLayout(),
                           memrefTy.getMemorySpace());

  SmallVector<Value, 16> operands = op->getOperands();
  operands[3] = rewriter.create<memref::CastOp>(op->getLoc(), operands[3], tys[3]);

  // mangle the name by appending '_<rank>d<space><type>'
  llvm::raw_string_ostream ss(fnName);
  ss << "_" << memrefTy.getRank();
  ss << "d" << memrefTy.getMemorySpaceAsInt();
  memrefTy.getElementType().print(ss);

  auto module = op->getParentOfType<ModuleOp>();
  auto fn = module.lookupSymbol<FuncOp>(fnName);
  if (!fn) {
    auto fnTy = FunctionType::get(ctx, tys, retTys);
    fn = FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  CallOp call = rewriter.create<CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
  rewriter.replaceOp(op, call->getResults());
  return success();
}

LogicalResult
lowerNdMemcpy(Operation* op, PatternRewriter &rewriter, std::string fnName)
{
  auto ctx = op->getContext();
  auto dmaOp = cast<xilinx::airrt::MemcpyNdOp>(op);
  SmallVector<Type, 6> tys;
  SmallVector<Type, 1> retTys;

  MemRefType dstMemRefTy = dmaOp.dst().getType().cast<MemRefType>();
  MemRefType srcMemRefTy = dmaOp.src().getType().cast<MemRefType>();

  SmallVector<Value, 16> operands = op->getOperands();
  operands[0] = rewriter.create<memref::CastOp>(op->getLoc(), operands[0],
           MemRefType::get(std::vector<int64_t>(dstMemRefTy.getRank(), -1),
                           dstMemRefTy.getElementType(),
                           dstMemRefTy.getLayout(),
                           dstMemRefTy.getMemorySpace()));
  operands[1] = rewriter.create<memref::CastOp>(op->getLoc(), operands[1],
           MemRefType::get(std::vector<int64_t>(srcMemRefTy.getRank(), -1),
                           srcMemRefTy.getElementType(),
                           srcMemRefTy.getLayout(),
                           srcMemRefTy.getMemorySpace()));

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
  auto fn = module.lookupSymbol<FuncOp>(fnName);
  if (!fn) {
    auto fnTy = FunctionType::get(ctx, tys, retTys);
    fn = FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  CallOp call = rewriter.create<CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
  rewriter.replaceOp(op, call->getResults());
  return success();
}

class DmaMemcpyToLLVMConversion : public OpRewritePattern<xilinx::airrt::DmaMemcpyOp> {
public:
  using OpRewritePattern<xilinx::airrt::DmaMemcpyOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::DmaMemcpyOp op,
                  PatternRewriter &rewriter) const override
  {
    auto space = op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    if (space == (int)xilinx::air::MemorySpace::L3)
      return lowerDmaMemcpy(op, rewriter, "air_shim_memcpy");
    if (space == (int)xilinx::air::MemorySpace::L2)
      return lowerDmaMemcpy(op, rewriter, "air_L1L2_memcpy");
    return failure();
  }
};

class DmaMemcpy2dToLLVMConversion : public OpRewritePattern<xilinx::airrt::DmaMemcpy2dOp> {
public:
  using OpRewritePattern<xilinx::airrt::DmaMemcpy2dOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::DmaMemcpy2dOp op,
                  PatternRewriter &rewriter) const override
  {
    auto space = op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    if (space == (int)xilinx::air::MemorySpace::L3)
      return lowerDmaMemcpy(op, rewriter, "air_shim_memcpy2d");
    if (space == (int)xilinx::air::MemorySpace::L2)
      return lowerDmaMemcpy(op, rewriter, "air_L1L2_memcpy2d");
    return failure();
  }
};

class DmaMemcpy4dToLLVMConversion : public OpRewritePattern<xilinx::airrt::DmaMemcpy4dOp> {
public:
  using OpRewritePattern<xilinx::airrt::DmaMemcpy4dOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::DmaMemcpy4dOp op,
                  PatternRewriter &rewriter) const override
  {
    auto space = op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt();
    if (space == (int)xilinx::air::MemorySpace::L3)
      return lowerDmaMemcpy(op, rewriter, "air_shim_memcpy4d");
    if (space == (int)xilinx::air::MemorySpace::L2)
      return lowerDmaMemcpy(op, rewriter, "air_L1L2_memcpy4d");
    return failure();
  }
};

class DmaMemcpyNdToLLVMConversion : public OpRewritePattern<xilinx::airrt::DmaMemcpyNdOp> {
public:
  using OpRewritePattern<xilinx::airrt::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::DmaMemcpyNdOp op,
                  PatternRewriter &rewriter) const override
  {
    return lowerDmaNdMemcpy(op, rewriter, "air_dma_nd_memcpy");
  }
};

class MemcpyNdToLLVMConversion : public OpRewritePattern<xilinx::airrt::MemcpyNdOp> {
public:
  using OpRewritePattern<xilinx::airrt::MemcpyNdOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(xilinx::airrt::MemcpyNdOp op,
                  PatternRewriter &rewriter) const override
  {
    return lowerNdMemcpy(op, rewriter, "air_nd_memcpy");
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

    auto memrefTy = op.memref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.create<memref::DeallocOp>(op.getLoc(), adaptor.memref());
    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineStoreOpConversion : public OpConversionPattern<AffineStoreOp> {
public:
  using OpConversionPattern<AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.memref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.create<AffineStoreOp>(op.getLoc(), adaptor.value(),
                                   adaptor.memref(), adaptor.indices());
    rewriter.eraseOp(op);
    return success();
  }
};
class L1AffineLoadOpConversion : public OpConversionPattern<AffineLoadOp> {
public:
  using OpConversionPattern<AffineLoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AffineLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.memref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    // auto ty = adaptor.memref().getType();
    auto load = rewriter.create<AffineLoadOp>(op.getLoc(), adaptor.memref(),
                                              adaptor.indices());
    rewriter.replaceOp(op, load.getResult());
    load.dump();
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

    auto memrefTy = op.getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2)
      return failure();

    tys.push_back(IndexType::get(ctx));
    retTys.push_back(MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                           memrefTy.getElementType(),
                           memrefTy.getLayout(),
                           memrefTy.getMemorySpace()));

    auto size = getTensorVolume(memrefTy);
    operands.push_back(rewriter.create<arith::ConstantIndexOp>(op->getLoc(), size));

    auto module = op->getParentOfType<ModuleOp>();

    std::string fnName = "air_alloc_L2";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << memrefTy.getRank();
    ss << "d" << memrefTy.getMemorySpaceAsInt();
    memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    auto callOp = rewriter.create<CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
    auto castOp = rewriter.create<memref::CastOp>(op->getLoc(), memrefTy, callOp.getResult(0));
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

    auto memrefTy = op.memref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L2)
      return failure();

    tys.push_back(MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                           memrefTy.getElementType(),
                           memrefTy.getLayout(),
                           memrefTy.getMemorySpace()));
    operands.push_back(rewriter.create<memref::CastOp>(op->getLoc(), tys[0], op.memref()));

    auto module = op->getParentOfType<ModuleOp>();
    
    std::string fnName = "air_dealloc_L2";
    llvm::raw_string_ostream ss(fnName);
    ss << "_" << memrefTy.getRank();
    ss << "d" << memrefTy.getMemorySpaceAsInt();
    memrefTy.getElementType().print(ss);

    auto fn = module.lookupSymbol<FuncOp>(fnName);
    if (!fn) {
      auto fnTy = FunctionType::get(ctx, tys, retTys);
      fn = FuncOp::create(rewriter.getUnknownLoc(), fnName, fnTy);
      fn.setPrivate();
      module.push_back(fn);
    }

    rewriter.create<CallOp>(op->getLoc(), retTys, SymbolRefAttr::get(fn), operands);
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRRtToLLVM : public AIRRtToLLVMBase<AIRRtToLLVM> {

public:
  AIRRtToLLVM() {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
     registry.insert<LLVM::LLVMDialect,
                     memref::MemRefDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    LLVMTypeConverter converter(context);

    converter.addConversion([&](Type type) -> Optional<Type> {
      // convert L1 memrefs to L3
      if (auto memref = type.dyn_cast<MemRefType>())
        if (memref.getMemorySpaceAsInt() == (int)xilinx::air::MemorySpace::L1)
          return mlir::MemRefType::get(memref.getShape(),
                                       memref.getElementType(),
                                       memref.getLayout(), 0);
      return type;
    });

    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    OwningRewritePatternList patterns(context);
    patterns.add<ModuleMetadataToLLVMConversion, HerdLoadToLLVMConversion,
                 DmaMemcpyToLLVMConversion, DmaMemcpy2dToLLVMConversion,
                 DmaMemcpy4dToLLVMConversion, DmaMemcpyNdToLLVMConversion,
                 MemcpyNdToLLVMConversion, L2AllocOpConversion,
                 L2DeallocOpConversion>(context);
    patterns.add<L1AllocOpConversion, L1AffineLoadOpConversion,
                 L1AffineStoreOpConversion, L1DeallocOpConversion>(converter,
                                                                   context);
    populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns, converter);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect,
                          StandardOpsDialect,
                          arith::ArithmeticDialect,
                          AffineDialect,
                          scf::SCFDialect,
                          memref::MemRefDialect>();

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() == 0);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() ==
              0);
    });

    target.addDynamicallyLegalOp<AffineStoreOp>([&](AffineStoreOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
              (int)xilinx::air::MemorySpace::L1);
    });

    target.addDynamicallyLegalOp<AffineLoadOp>([&](AffineLoadOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
              (int)xilinx::air::MemorySpace::L1);
    });

    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering AIRRt\n");
      signalPassFailure();
    }

    for (auto func : module.getOps<FuncOp>())
      func->setAttr("llvm.emit_c_interface", UnitAttr::get(func.getContext()));
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
