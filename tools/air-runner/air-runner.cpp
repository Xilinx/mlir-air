// (c) Copyright 2019-2022 Xilinx Inc. All Rights Reserved.

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/ToolUtilities.h"

#include <deque>
#include <float.h>
#include <list>
#include <map>
#include <sstream>
#include <vector>

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/CostModel.h"

#include "air/InitAll.h"

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

static bool verbose = false;

using namespace mlir;

namespace xilinx {
namespace air {

class AIRRunner {

  const int TRACE_PID_QUEUE = 0;
  const int TRACE_PID_ALLOC = 1;
  const int TRACE_PID_STATS = 2;

  void executeOp(arith::ConstantIndexOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    auto attr = op->getAttrOfType<IntegerAttr>("value");
    out[0] = attr.getValue().sextOrTrunc(INDEX_WIDTH);
  }

  void executeOp(arith::ConstantIntOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    auto attr = op->getAttrOfType<IntegerAttr>("value");
    out[0] = attr.getValue();
  }

  void executeOp(arith::AddIOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] =
        llvm::any_cast<llvm::APInt>(in[0]) + llvm::any_cast<llvm::APInt>(in[1]);
  }

  void executeOp(arith::AddFOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = llvm::any_cast<llvm::APFloat>(in[0]) +
             llvm::any_cast<llvm::APFloat>(in[1]);
  }

  void executeOp(arith::SubIOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] =
        llvm::any_cast<llvm::APInt>(in[0]) - llvm::any_cast<llvm::APInt>(in[1]);
  }

  void executeOp(arith::SubFOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = llvm::any_cast<llvm::APFloat>(in[0]) +
             llvm::any_cast<llvm::APFloat>(in[1]);
  }

  void executeOp(arith::CmpIOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    assert(0 && "unsupported op");
    llvm::APInt in0 = llvm::any_cast<llvm::APInt>(in[0]);
    llvm::APInt in1 = llvm::any_cast<llvm::APInt>(in[1]);
    llvm::APInt out0(1, applyCmpPredicate(op.getPredicate(), in0, in1));
    out[0] = out0;
  }

  void executeOp(arith::CmpFOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    assert(0 && "unsupported op");
    llvm::APFloat in0 = llvm::any_cast<llvm::APFloat>(in[0]);
    llvm::APFloat in1 = llvm::any_cast<llvm::APFloat>(in[1]);
    llvm::APInt out0(1, applyCmpPredicate(op.getPredicate(), in0, in1));
    out[0] = out0;
  }

  void executeOp(arith::MulIOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] =
        llvm::any_cast<llvm::APInt>(in[0]) * llvm::any_cast<llvm::APInt>(in[1]);
  }

  void executeOp(arith::MulFOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = llvm::any_cast<llvm::APFloat>(in[0]) *
             llvm::any_cast<llvm::APFloat>(in[1]);
  }

  void executeOp(arith::DivFOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = llvm::any_cast<llvm::APFloat>(in[0]) /
             llvm::any_cast<llvm::APFloat>(in[1]);
  }

  void executeOp(arith::IndexCastOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = in[0];
  }

  void executeOp(memref::LoadOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    llvm::ArrayRef<int64_t> shape = op.getMemRefType().getShape();
    unsigned address = 0;
    for (unsigned i = 0; i < shape.size(); i++) {
      address = address * shape[i] +
                llvm::any_cast<llvm::APInt>(in[i + 1]).getZExtValue();
    }
    unsigned ptr = llvm::any_cast<unsigned>(in[0]);
    assert(ptr < store.size());
    auto &ref = store[ptr];
    assert(address < ref.size());
    //  LLVM_DEBUG(llvm::dbgs() << "Load " << ref[address] << " from " << ptr <<
    //  "[" << address << "]\n");
    llvm::Any result = ref[address];
    out[0] = result;
  }

  void executeOp(xilinx::air::DmaMemcpyInterface op,
                 std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    return;
    // llvm::ArrayRef<int64_t> shape =
    // op.getType().cast<TensorType>().getShape(); unsigned address = 0;
    // unsigned ptr = llvm::any_cast<unsigned>(in[0]);
    // assert(ptr < store.size());
    // auto &ref = store[ptr];
    // assert(address < ref.size());
    // //  LLVM_DEBUG(llvm::dbgs() << "Load " << ref[address] << " from " << ptr
    // << "[" << address << "]\n"); llvm::Any result = ref[address]; out[0] =
    // result;
  }

  void executeOp(memref::StoreOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    llvm::ArrayRef<int64_t> shape = op.getMemRefType().getShape();
    unsigned address = 0;
    for (unsigned i = 0; i < shape.size(); i++) {
      address = address * shape[i] +
                llvm::any_cast<llvm::APInt>(in[i + 2]).getZExtValue();
    }
    unsigned ptr = llvm::any_cast<unsigned>(in[1]);
    assert(ptr < store.size());
    auto &ref = store[ptr];
    //  LLVM_DEBUG(llvm::dbgs() << "Store " << in[0] << " to " << ptr << "[" <<
    //  address << "]\n");
    assert(address < ref.size());
    ref[address] = in[0];
  }

  // void executeOp(xilinx::air::DmaStoreOp op,
  //                std::vector<llvm::Any> &in,
  //                std::vector<llvm::Any> &out) {
  //   return;
  //   llvm::ArrayRef<int64_t> shape =
  //   op.tensor().getType().cast<mlir::ShapedType>().getShape(); unsigned
  //   address = 0; unsigned ptr = llvm::any_cast<unsigned>(in[1]); assert(ptr <
  //   store.size()); auto &ref = store[ptr];
  //   //  LLVM_DEBUG(llvm::dbgs() << "Store " << in[0] << " to " << ptr << "["
  //   << address << "]\n"); assert(address < ref.size()); ref[address] = in[0];
  // }

  void executeOp(memref::AllocOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    out[0] = allocateMemRef(op.getType(), in);
    // unsigned ptr = llvm::any_cast<unsigned>(out[0]);
  }

  // void executeOp(xilinx::air::AllocOp op,
  //                std::vector<llvm::Any> &in,
  //                std::vector<llvm::Any> &out) {
  //   out[0] = allocateMemRef(op.getType().cast<MemRefType>(), in);
  //   //unsigned ptr = llvm::any_cast<unsigned>(out[0]);
  // }

  void executeOp(memref::DeallocOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    return;
    unsigned ptr = llvm::any_cast<unsigned>(in[0]);
    deallocateMemRef(ptr);
  }

  // void executeOp(mlir::AffineForOp op, std::vector<llvm::Any> &in,
  //                std::vector<llvm::Any> &out) {
  //   out.resize(1);
  //   out[0] = op.getConstantUpperBound();
  // }

  // void executeOp(mlir::AffineParallelOp op, std::vector<llvm::Any> &in,
  //                std::vector<llvm::Any> &out) {
  //   out.resize(2);
  //   out[0] = op.getIVs()[0];
  //   out[1] = op.getIVs()[1];
  // }

  void executeOp(scf::ForOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    assert(!op->getNumResults());
    // int i = 0;
    // for (auto r : op.getResults())
    //   out[i++] = llvm::APInt(64,0);
  }

  void executeOp(xilinx::air::HerdLaunchOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    if (op->getNumResults()) {
      auto res = op->getResult(0);
      if (res.getType().isa<xilinx::air::AsyncTokenType>()) {
        out[0] = llvm::any_cast<llvm::APInt>(valueMap[res]) - llvm::APInt(64, 1);
        valueMap[res] = out[0];
      }
    }
  }

#if HAVE_REGION_OP
  void executeOp(xilinx::air::RegionOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    if (op->getNumResults()) {
      auto res = op->getResult(0);
      if (res.getType().isa<xilinx::air::AsyncTokenType>()) {
        out[0] = llvm::any_cast<llvm::APInt>(valueMap[res]) - llvm::APInt(64, 1);
        valueMap[res] = out[0];
      }
    }
  }
#endif

  void executeOp(xilinx::air::WaitAllOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    if (op->getNumResults())
      out[0] = llvm::APInt(64, 0);
  }

  bool executeOpImpls(mlir::Operation &op, std::vector<llvm::Any> &inValues,
                      std::vector<llvm::Any> &outValues) {
    if (auto Op = dyn_cast<arith::ConstantIndexOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::ConstantIntOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::AddIOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::AddFOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::SubIOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::SubFOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::CmpIOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::CmpFOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::MulIOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::MulFOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::DivFOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<arith::IndexCastOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<memref::AllocOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<memref::DeallocOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<scf::ForOp>(op))
      executeOp(Op, inValues, outValues);
    // else if (auto Op = dyn_cast<mlir::AffineForOp>(op))
    //   executeOp(Op, inValues, outValues);
    // else if (auto Op = dyn_cast<mlir::AffineParallelOp>(op))
    //   executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::HerdLaunchOp>(op))
      executeOp(Op, inValues, outValues);
#if HAVE_REGION_OP
    else if (auto Op = dyn_cast<xilinx::air::RegionOp>(op))
      executeOp(Op, inValues, outValues);
#endif
    else if (auto Op = dyn_cast<xilinx::air::WaitAllOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::DmaMemcpyInterface>(op))
      executeOp(Op, inValues, outValues);
    else
      return false;
    return true;
  }

  bool executeOp(Operation &op) {
    int i = 0;
    std::vector<llvm::Any> inValues(op.getNumOperands());
    std::vector<llvm::Any> outValues(op.getNumResults());
    LLVM_DEBUG(llvm::dbgs() << "OP:  " << op.getName() << "\n");
    for (Value in : op.getOperands()) {
      inValues[i++] = valueMap[in];
    }

    if (!executeOpImpls(op, inValues, outValues))
      return false;

    // record result in valuemap
    i = 0;
    for (Value out : op.getResults()) {
      LLVM_DEBUG(debugArg("OUT", out, outValues[i], time));
      valueMap[out] = outValues[i];
      // timeMap[out] = time;
      i++;
    }

    if (auto afo = dyn_cast<AffineForOp>(op)) {
      valueMap[afo.getInductionVar()] = outValues[0];
    }
    if (auto apo = dyn_cast<AffineParallelOp>(op)) {
      valueMap[apo.getIVs()[0]] = outValues[0];
      valueMap[apo.getIVs()[1]] = outValues[1];
    }

    return true;
  }

#if 0
void executeOp(mlir::Op op,
               std::vector<llvm::Any> &in,
               std::vector<llvm::Any> &out) {
}
#endif

  void debugArg(const std::string &head, mlir::Value op,
                const llvm::APInt &value, uint64_t time) {
    LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = " << value
                            << " (llvm::APInt<" << value.getBitWidth() << ">) @"
                            << time << "\n");
  }

  void debugArg(const std::string &head, mlir::Value op,
                const llvm::APFloat &value, uint64_t time) {
    LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = ";
               value.print(llvm::dbgs());
               llvm::dbgs() << " ("
                            << "float"
                            << ") @" << time << "\n");
  }

  void debugArg(const std::string &head, mlir::Value op, const llvm::Any &value,
                uint64_t time) {
    if (llvm::any_isa<llvm::APInt>(value)) {
      debugArg(head, op, llvm::any_cast<llvm::APInt>(value), time);
    } else if (llvm::any_isa<llvm::APFloat>(value)) {
      debugArg(head, op, llvm::any_cast<llvm::APFloat>(value), time);
    } else if (llvm::any_isa<unsigned>(value)) {
      // Represents an allocated buffer.
      LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = Buffer "
                              << llvm::any_cast<unsigned>(value) << "\n");
    } else {
      // llvm_unreachable("unknown type");
    }
  }

public:
  AIRRunner(llvm::raw_ostream &trace_stream, llvm::json::Value &json_model)
      : traceStream(trace_stream), jsonModel(json_model), time(1) {}

  // Allocate a new matrix with dimensions given by the type, in the
  // given store.  Return the pseudo-pointer to the new matrix in the
  // store (i.e. the first dimension index)
  unsigned allocateMemRef(mlir::MemRefType type, std::vector<llvm::Any> &in) {
    llvm::ArrayRef<int64_t> shape = type.getShape();
    auto memorySpace = type.getMemorySpaceAsInt();
    auto volume = getTensorVolume(type);
    auto model = jsonModel.getAsObject();
    unsigned datawidth = 0;
    if (auto dt = model->getObject("datatype"))
      if (auto bytes = dt->getNumber("bytes"))
        datawidth = *bytes;
    assert(datawidth);
    uint64_t bytes = volume * datawidth;
    unsigned ptr = store.size();
    store.resize(ptr + 1);
    store[ptr].resize(1 /*bytes*/);
    if (verbose)
      llvm::outs() << "alloc " << ptr << " space " << memorySpace << " size "
                   << bytes << "\n";
    // mlir::Type elementType = type.getElementType();
    // int width = elementType.getIntOrFloatBitWidth();
    //  for (int i = 0; i < bytes; i++) {
    //    if (elementType.isa<mlir::IntegerType>()) {
    //      store[ptr][i] = llvm::APInt(width, 0);
    //    }
    //    else if (elementType.isa<mlir::FloatType>()) {
    //      store[ptr][i] = llvm::APFloat(0.0);
    //    }
    //    else {
    //      llvm_unreachable("Unknown result type!\n");
    //    }
    //  }
    //  emitTraceEvent(traceStream,
    //                 "tensor "+std::to_string(ptr)+" space
    //                 "+std::to_string(memorySpace)+" size
    //                 "+std::to_string(bytes), "layer", "B", time, ptr,
    //                 TRACE_PID_ALLOC);
    return ptr;
  }

  void deallocateMemRef(unsigned ptr) {
    return;
    // assert(store[ptr].size());
    auto allocationSize = store[ptr].size();
    if (verbose)
      llvm::outs() << "dealloc " << ptr << "\n";
    store[ptr].resize(0);
    // emitTraceEvent(traceStream, "dealloc", "layer", "E", time, ptr,
    // TRACE_PID_ALLOC);
  }

  std::string printAnyValueWithType(mlir::Type type, llvm::Any &value) {
    std::stringstream out;
    if (type.isa<mlir::IntegerType>() || type.isa<mlir::IndexType>()) {
      out << llvm::any_cast<llvm::APInt>(value).getSExtValue();
      return out.str();
    } else if (type.isa<mlir::FloatType>()) {
      out << llvm::any_cast<llvm::APFloat>(value).convertToDouble();
      return out.str();
    } else if (type.isa<mlir::NoneType>()) {
      return "none";
    } else {
      llvm_unreachable("Unknown result type!");
    }
  }

  void scheduleIfNeeded(std::list<mlir::Operation *> &readyList,
                        llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                        mlir::Operation *op) {
    if (std::find(readyList.begin(), readyList.end(), op) == readyList.end()) {
      readyList.push_back(op);
    }
  }
  void scheduleUses(std::list<mlir::Operation *> &readyList,
                    llvm::DenseMap<mlir::Value, llvm::Any> &valueMap,
                    mlir::Value value) {
    for (auto &use : value.getUses()) {
      scheduleIfNeeded(readyList, valueMap, use.getOwner());
    }
  }

  void emitTraceStart(llvm::raw_ostream &s) { s << "[\n"; }

  void emitTraceEnd(llvm::raw_ostream &s) { s << "{}]\n"; }

  void emitTraceEvent(llvm::raw_ostream &s, std::string name, std::string cat,
                      std::string ph, uint64_t ts, int64_t tid, int64_t pid) {
    s << "{\n";
    s << "  \"name\": \"" << name << "\","
      << "\n";
    s << "  \"cat\": \"" << cat << "\","
      << "\n";
    s << "  \"ph\": \"" << ph << "\","
      << "\n";
    s << "  \"ts\": " << ts << ","
      << "\n";
    s << "  \"pid\": " << pid << ","
      << "\n";
    s << "  \"tid\": " << tid << ","
      << "\n";
    s << "  \"args\": "
      << "{}"
      << ""
      << "\n";
    s << "},\n";
  }

  uint64_t getTensorVolume(const mlir::ShapedType ty) {

    if (!ty.hasRank())
      return 1;

    uint64_t volume = 1;
    for (auto &d : ty.getShape())
      volume *= d;
    return volume;
  }

  uint64_t getTensorVolume(const mlir::Type ty) {
    if (auto t = ty.dyn_cast<mlir::ShapedType>()) {
      return getTensorVolume(t);
    } else {
      return 1;
    }
  }

  uint64_t getTransferCost(unsigned srcSpace, unsigned dstSpace,
                           mlir::Type ty) {
    return getTransferCost(srcSpace, dstSpace, getTensorVolume(ty));
  }

  uint64_t getTransferCost(unsigned srcSpace, unsigned dstSpace,
                           int64_t volume) {
    std::map<std::pair<unsigned, unsigned>, double> interface_bw;

    // defaults
    interface_bw.insert({{0, 1}, 100});
    interface_bw.insert({{1, 0}, 100});
    interface_bw.insert({{1, 2}, DBL_MAX});
    interface_bw.insert({{2, 1}, DBL_MAX});
    double cps = 0.0f;

    // override of defaults
    auto model = jsonModel.getAsObject();
    unsigned datawidth = 0;
    // if interfaces exists, assume everthing else exists
    if (model && model->getArray("interfaces")) {
      auto interfaces = model->getArray("interfaces");
      assert(interfaces);

      for (auto it = interfaces->begin(), ie = interfaces->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        llvm::json::Object *interface = jv.getAsObject();
        assert(interface);
        auto srcSpace = interface->getNumber("src");
        auto dstSpace = interface->getNumber("dst");
        auto bps = interface->getNumber("bytes_per_second");
        assert(srcSpace && dstSpace && bps);
        unsigned s = *srcSpace;
        unsigned d = *dstSpace;
        double b = *bps;
        if (interface_bw.count({s, d}))
          interface_bw[{s, d}] = b;
        else
          interface_bw.insert({{s, d}, b});
      }
      if (auto d = model->getNumber("clock"))
        cps = *d;
      if (auto dt = model->getObject("datatype"))
        if (auto bytes = dt->getNumber("bytes"))
          datawidth = *bytes;
    }
    assert(cps != 0.0f && datawidth);

    double bytes = volume * datawidth;
    double bps = interface_bw[{srcSpace, dstSpace}];
    double seconds = bytes / bps;
    return (uint64_t)ceil(seconds * cps);
  }

  // model the memory tranfer time for a layer
  std::tuple<uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>
  modelLayerMemoryTime(mlir::Operation *op) {
    const int num_mem = 2;
    std::vector<uint64_t> ld_xfer_time(num_mem, 0);
    std::vector<uint64_t> st_xfer_time(num_mem, 0);

    auto model = jsonModel.getAsObject();
    if (!model)
      assert(model);
    auto memories = model->getObject("memories");
    if (!memories)
      assert(memories);

    unsigned idx = 0;
    // for (Value o : op->getOperands()) {
    //   if (auto tty = o.getType().dyn_cast<TensorType>()) {
    //     if (auto tensor_load =
    //     dyn_cast<xilinx::air::DmaLoadOp>(o.getDefiningOp())) {
    //       auto space =
    //       tensor_load.memref().getType().cast<MemRefType>().getMemorySpace();
    //       uint64_t ld_vol = 0;
    //       uint64_t st_vol = 0;
    //       // if (auto stats =
    //       mlir::dyn_cast<xilinx::aten::StatisticsOpInterface>(op)) {
    //       //   // make the cost 1 for dram
    //       //   if (space != 0) {
    //       //     ld_vol = stats.getOperandTransferVolume(idx, true);
    //       //     st_vol = stats.getOperandTransferVolume(idx, false);
    //       //   }
    //       //   else {
    //       //     ld_vol = getTensorVolume(tensor_load.memref().getType());
    //       //   }
    //       // } else {
    //         ld_vol = getTensorVolume(tensor_load.memref().getType());
    //       // }
    //       if (ld_vol) ld_xfer_time[space] += getTransferCost(space,
    //       space+1/*2*/, ld_vol); if (st_vol) st_xfer_time[space] +=
    //       getTransferCost(space+1/*2*/, space, st_vol);
    //     }
    //   }
    //   idx++;
    // }

    // idx = 0;
    // for (Value r : op->getResults()) {
    //   if (auto tty = r.getType().dyn_cast<TensorType>()) {
    //     for (auto user : r.getUsers()) {
    //       if (auto tensor_store = dyn_cast<xilinx::air::DmaStoreOp>(user)) {
    //         auto space =
    //         tensor_store.memref().getType().cast<MemRefType>().getMemorySpace();
    //         uint64_t ld_vol = 0;
    //         uint64_t st_vol = 0;
    //         // if (auto stats =
    //         mlir::dyn_cast<xilinx::aten::StatisticsOpInterface>(op)) {
    //         //   // make the cost 1 for dram
    //         //   if (space != 0) {
    //         //     st_vol = stats.getResultTransferVolume(idx, true);
    //         //     ld_vol = stats.getResultTransferVolume(idx, false);
    //         //   }
    //         //   else {
    //         //     st_vol = getTensorVolume(tensor_store.memref().getType());
    //         //   }
    //         // } else {
    //           st_vol = getTensorVolume(tensor_store.memref().getType());
    //         // }
    //         if (ld_vol) ld_xfer_time[space] += getTransferCost(space,
    //         space+1/*2*/, ld_vol); if (st_vol) st_xfer_time[space] +=
    //         getTransferCost(space+1/*2*/, space, st_vol);
    //       }
    //     }
    //   }
    //   idx++;
    // }

    time = 0;
    for (int i = 0; i < num_mem; i++) {
      // llvm::outs() << "memory[" << i << "] ld time: " << ld_xfer_time[i] << "
      // st time: " << st_xfer_time[i] << "\n";
      auto mem = memories->getObject(std::to_string(i));
      if (!mem)
        assert(mem);
      auto type = mem->getString("type");
      if (!type)
        assert(type);

      uint64_t t;
      if (*type == "duplex")
        t = std::max(st_xfer_time[i], ld_xfer_time[i]);
      else if (*type == "simplex")
        t = st_xfer_time[i] + ld_xfer_time[i];
      else
        llvm_unreachable("bad memory type in device model");

      time = std::max(t, time);
    }

    return std::tie(time, ld_xfer_time, st_xfer_time);
  }

  struct CommandQueueEntry {
    mlir::Operation *op;
    uint64_t start_time;
    uint64_t end_time;
    uint64_t compute_op_cost;
    uint64_t compute_xfer_cost;
    uint64_t queue_ready_time;

    using LaunchFn = std::function<void(Operation *)>;
    LaunchFn launch_callback_fn;

    std::vector<uint64_t> ld_xfer_time;
    std::vector<uint64_t> st_xfer_time;

    bool is_started() { return (start_time != 0) && (end_time != 0); }
    bool is_done(uint64_t t) { return t >= end_time; }

    CommandQueueEntry(mlir::Operation *o, LaunchFn launch_fn = nullptr)
        : op(o), start_time(0), end_time(0), compute_op_cost(0),
          compute_xfer_cost(0), queue_ready_time(0),
          launch_callback_fn(launch_fn) {}

    CommandQueueEntry &operator=(const CommandQueueEntry &) = delete;
  };

  uint64_t modelOp(CommandQueueEntry &c) {
    mlir::Operation *op = c.op;
    uint64_t execution_time = 1;

    if (auto Op = mlir::dyn_cast<xilinx::air::WaitAllOp>(op)) {
      execution_time = 1;
    }
    else if (auto Op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)) {

      MemRefType srcTy = Op.getSrcMemref().getType().cast<MemRefType>();
      MemRefType dstTy = Op.getDstMemref().getType().cast<MemRefType>();
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor 
      if (getTensorVolume(srcTy) <= getTensorVolume(dstTy))
        execution_time = getTransferCost(srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(srcSpace, dstSpace, dstTy);
    }
    else if (auto Op = mlir::dyn_cast<linalg::LinalgOp>(op)) {
      auto opCounts = xilinx::air::CostModel().getOpCounts(op);
      std::string memops = "reads;writes;";
      std::string cpuops = "arith.mulf;arith.addf;arith.muli;arith.addi;";
      uint64_t memory_op_count = 0;
      uint64_t compute_op_count = 0;
      for (auto &p : opCounts.map) {
        auto name = std::get<0>(p);
        auto count = std::get<1>(p);
        if (memops.find(name) != std::string::npos)
          memory_op_count += count;
        else if (cpuops.find(name) != std::string::npos)
          compute_op_count += count;
        else if (verbose)
          llvm::outs() << name << " not counted\n";
      }
      c.compute_xfer_cost = memory_op_count;

      if (compute_op_count) {
        // defaults
        double num_cores = 16;
        double ops_per_core_per_cycle = 8; // vector width for this type
        double cycles_per_second = 1e9;
        double efficiency = 1.0f;

        auto model = jsonModel.getAsObject();
        assert(model);

        // if kernels exists, assume everthing else exists
        if (model && model->getObject("kernels")) {
          // device level override of defaults
          if (auto d = model->getNumber("cores"))
            num_cores = *d;
          if (auto d = model->getNumber("ops_per_core_per_cycle"))
            ops_per_core_per_cycle = *d;
          if (auto d = model->getNumber("clock"))
            cycles_per_second = *d;
          if (auto d = model->getNumber("efficiency"))
            efficiency = *d;

          // kernel level override of defaults
          auto kernels = model->getObject("kernels");
          assert(kernels && "kernels not found in JSON model");

          if (kernels) {
            auto kernel = kernels->getObject(op->getName().getStringRef());
            if (kernel) {
              if (auto d = kernel->getNumber("cores"))
                num_cores = *d;
              if (auto d = kernel->getNumber("ops_per_core_per_cycle"))
                ops_per_core_per_cycle = *d;
              if (auto d = kernel->getNumber("clock"))
                cycles_per_second = *d;
              if (auto d = kernel->getNumber("efficiency"))
                efficiency = *d;
            }
          }
        }

        double ops_per_cycle = num_cores * ops_per_core_per_cycle * efficiency;
        assert(ops_per_cycle > 0 && "ops per cycle in model must be greater than zero");

        double cycles = ceil(compute_op_count / ops_per_cycle);
        c.compute_op_cost = cycles;
      }
      execution_time = std::max(c.compute_op_cost, c.compute_xfer_cost);
    }
    else {
      LLVM_DEBUG(llvm::errs()
                 << "WARNING: execution time not modeled for op: '");
      LLVM_DEBUG(llvm::errs() << to_string(op) << "'\n");
      execution_time = 1;
    }
    return execution_time;
  }

  std::string to_string(Operation *op) {
    return op->getName().getStringRef().str();
  }

  std::string to_string(CommandQueueEntry &c) { return to_string(c.op); }

  void processQueue(std::deque<CommandQueueEntry> &q, uint64_t time) {
    if (q.size() == 0)
      return;

    CommandQueueEntry &c = q.front();

    if (c.is_started()) {
      if (c.is_done(time)) {
        if (verbose) {
          llvm::outs() << "finish: '";
          c.op->print(llvm::outs());
          llvm::outs() << "' @ " << time << "\n";
        }

        // execute
        executeOp(*c.op);
        if (c.launch_callback_fn)
          c.launch_callback_fn(c.op);

        // emit trace event end
        emitTraceEvent(traceStream, to_string(c), "layer", "E", time,
                       (size_t)(void *)&q, TRACE_PID_QUEUE);

        if (c.compute_xfer_cost && c.compute_op_cost) {
          if (c.compute_op_cost >= c.compute_xfer_cost) {
            emitTraceEvent(traceStream, "compute_bound", "stats", "B",
                           c.start_time, 0, TRACE_PID_STATS);
            emitTraceEvent(traceStream, "compute_bound", "stats", "E",
                           c.end_time, 0, TRACE_PID_STATS);
          } else {
            emitTraceEvent(traceStream, "memory_bound", "stats", "B",
                           c.start_time, 0, TRACE_PID_STATS);
            emitTraceEvent(traceStream, "memory_bound", "stats", "E",
                           c.end_time, 0, TRACE_PID_STATS);
          }
          if (c.compute_op_cost) {
            std::stringstream cat;
            cat << "compute time";
            emitTraceEvent(traceStream, cat.str(), "stats", "B", c.start_time,
                           100, TRACE_PID_STATS);
            emitTraceEvent(traceStream, cat.str(), "stats", "E",
                           c.start_time + c.compute_op_cost, 100,
                           TRACE_PID_STATS);
          }
          if (c.compute_xfer_cost) {
            std::stringstream cat;
            cat << "transfer time";
            emitTraceEvent(traceStream, cat.str(), "stats", "B", c.start_time,
                           101, TRACE_PID_STATS);
            emitTraceEvent(traceStream, cat.str(), "stats", "E",
                           c.start_time + c.compute_xfer_cost, 101,
                           TRACE_PID_STATS);
          }
        }
        for (int i = 0, e = c.ld_xfer_time.size(); i < e; i++) {
          if (!c.ld_xfer_time[i])
            continue;
          std::stringstream cat;
          cat << "mem " << i << " load";
          emitTraceEvent(traceStream, cat.str(), "stats", "B", c.start_time,
                         (i + 1) * 200, TRACE_PID_STATS);
          emitTraceEvent(traceStream, cat.str(), "stats", "E",
                         c.start_time + c.ld_xfer_time[i], (i + 1) * 200,
                         TRACE_PID_STATS);
        }
        for (int i = 0, e = c.st_xfer_time.size(); i < e; i++) {
          if (!c.st_xfer_time[i])
            continue;
          std::stringstream cat;
          cat << "mem " << i << " store";
          emitTraceEvent(traceStream, cat.str(), "stats", "B", c.start_time,
                         (i + 1) * 200 + 1, TRACE_PID_STATS);
          emitTraceEvent(traceStream, cat.str(), "stats", "E",
                         c.start_time + c.st_xfer_time[i], (i + 1) * 200 + 1,
                         TRACE_PID_STATS);
        }

        // done, return.
        q.pop_front();
        // return;
      } else {
        // running...
        if (verbose) {
          llvm::outs() << "running: '";
          c.op->print(llvm::outs());
          llvm::outs() << "' @ " << time << " - " << c.end_time << "\n";
        }
        // in-order, return.
        return;
      }
    }

    if (!q.size()) {
      if (verbose)
        llvm::outs() << "queue empty @ " << time << "\n";
      return;
    }

    CommandQueueEntry &c_next = q.front();

    if (!c_next.queue_ready_time)
      c_next.queue_ready_time = time;

    bool ready = true;
    for (Value in : c_next.op->getOperands()) {
      if (!valueMap.count(in))
        ;//ready = false;
      else if (in.getType().isa<xilinx::air::AsyncTokenType>() && !valueMap.count(in))
//               llvm::any_cast<llvm::APInt>(valueMap[in]) != 0)
        ready = false;
    }

    if (!ready) {
      if (verbose) {
        llvm::outs() << "not ready: '";
        c_next.op->print(llvm::outs());
        llvm::outs() << "' @ " << time << "\n";
      }
      return;
    }

    c_next.start_time = time;
    c_next.end_time = time + modelOp(c_next);
    if (verbose) {
      llvm::outs() << "start: '";
      c_next.op->print(llvm::outs());
      llvm::outs() << "' @ " << time << " - " << c_next.end_time << "\n";
    }

    // emit trace event begin
    if (time > c_next.queue_ready_time) {
      emitTraceEvent(traceStream, "stall", "layer", "B",
                     c_next.queue_ready_time, (size_t)(void *)&q,
                     TRACE_PID_QUEUE);
      emitTraceEvent(traceStream, "stall", "layer", "E", time,
                     (size_t)(void *)&q, TRACE_PID_QUEUE);
    }
    emitTraceEvent(traceStream, to_string(c_next), "layer", "B", time,
                   (size_t)(void *)&q, TRACE_PID_QUEUE);

    return;
  }

#if HAVE_REGION_OP
  void scheduleAIRRegion(xilinx::air::RegionOp &r,
                         std::deque<CommandQueueEntry> *kernelOps,
                         std::deque<CommandQueueEntry> *dataOps,
                         std::deque<CommandQueueEntry> herdOps[16][16]) {

    if (r->getNumResults() && r->getResult(0).getType().isa<xilinx::air::AsyncTokenType>())
      valueMap[r->getResult(0)] = llvm::APInt(64, 1);

    kernelOps->push_back(
        CommandQueueEntry(r.getOperation(), [=](Operation *op) {
          auto o = cast<xilinx::air::RegionOp>(op);
        }));
  }
#endif

  void scheduleScfFor(mlir::scf::ForOp &fo,
                      std::deque<CommandQueueEntry> *kernelOps,
                      std::deque<CommandQueueEntry> *dataOps,
                      std::deque<CommandQueueEntry> herdOps[16][16]) {
    auto ub = fo.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lb = fo.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step = fo.getStep().getDefiningOp<arith::ConstantIndexOp>();

    assert(ub && lb && step);
    auto r = ub.value() - lb.value();
    auto trip_count = mlir::ceilDiv(r, step.value());

    // kernelOps->push_back(CommandQueueEntry(fo.getOperation()));

    for (auto i = 0; i < trip_count; i++) {
      scheduleRegion(fo.getRegion(), kernelOps, dataOps, herdOps);
    }
    return;
  }

  // void scheduleAffineFor(mlir::AffineForOp &afo,
  //                        std::deque<CommandQueueEntry> *kernelOps,
  //                        std::deque<CommandQueueEntry> *dataOps,
  //                        std::deque<CommandQueueEntry> herdOps[16][16]) {
  //   assert(afo.hasConstantBounds());
  //   auto r = afo.getConstantUpperBound() - afo.getConstantLowerBound();
  //   auto trip_count = mlir::ceilDiv((int64_t)r, afo.getStep());

  //   //kernelOps->push_back(CommandQueueEntry(afo.getOperation()));

  //   for (auto i = 0; i < trip_count; i++) {
  //     scheduleRegion(afo.getRegion(), kernelOps, dataOps, herdOps);
  //   }
  //   return;
  // }

  // void scheduleAffineParallel(mlir::AffineParallelOp &apo,
  //                             std::deque<CommandQueueEntry> *kernelOps,
  //                             std::deque<CommandQueueEntry> *dataOps,
  //                             std::deque<CommandQueueEntry> herdOps[16][16])
  //                             {
  //   assert(apo.getNumDims() == 2);
  //   auto ub_rows =
  //       apo.upperBoundsMap().getResult(0).cast<AffineConstantExpr>().getValue();
  //   auto ub_cols =
  //       apo.upperBoundsMap().getResult(1).cast<AffineConstantExpr>().getValue();

  //   kernelOps->push_back(CommandQueueEntry(apo.getOperation()));

  //   for (auto row = 0; row < ub_rows; row++) {
  //     for (auto col = 0; col < ub_cols; col++) {
  //       auto herd_row = row % 2;
  //       auto herd_col = col % 12;
  //       scheduleRegion(apo.getRegion(), herdOps[herd_col][herd_row],
  //                      herdOps[herd_col][herd_row], herdOps);
  //     }
  //   }
  // }

  void scheduleHerdLaunch(xilinx::air::HerdLaunchOp &hlo,
                          std::deque<CommandQueueEntry> *kernelOps,
                          std::deque<CommandQueueEntry> *dataOps,
                          std::deque<CommandQueueEntry> herdOps[16][16]) {

    int64_t cols = cast<arith::ConstantIndexOp>(
                       hlo.getHerdSizeOperands().x.getDefiningOp())
                       .value();
    int64_t rows = cast<arith::ConstantIndexOp>(
                       hlo.getHerdSizeOperands().y.getDefiningOp())
                       .value();

    if (hlo->getNumResults())
      valueMap[hlo->getResult(0)] = APInt(64, cols * rows);

    if (hlo->getNumResults())
      valueMap[hlo->getResult(0)] = APInt(64, 1);

    llvm::DenseMap<Value, llvm::Any> *vmap = &valueMap;
    kernelOps->push_back(
        CommandQueueEntry(hlo.getOperation(), [=](Operation *op) {
          auto o = cast<xilinx::air::HerdLaunchOp>(op);
          auto args = o.getKernelArguments();
          BlockAndValueMapping map;
          for (int i = 0, e = o.getNumKernelOperands(); i < e; i++) {
            auto operand = o.getKernelOperand(i);
            if (vmap->count(operand))
              (*vmap)[args[i]] = (*vmap)[o.getKernelOperand(i)];
            map.map(args[i], o.getKernelOperand(i));
          }
          for (auto row = 0; row < rows; row++) {
            for (auto col = 0; col < cols; col++) {
              auto r = new Block();
              //o.getRegion().front().cloneInto(r, map);
              scheduleBlock(*r, &herdOps[col][row],
                             &herdOps[col][row], herdOps);
            }
          }
        }));
  }

  void scheduleBlock(mlir::Block &block,
                     std::deque<CommandQueueEntry> *kernelOps,
                     std::deque<CommandQueueEntry> *dataOps,
                     std::deque<CommandQueueEntry> herdOps[16][16]) {
    Operation &o = block.getOperations().front();
    Operation *op = &o;
    while (op) {

      if (isa<arith::ConstantOp>(op) || isa<arith::ConstantIndexOp>(op)) {
        executeOp(*op);
      } else if (isa<memref::AllocOp>(op) || isa<memref::DeallocOp>(op) ||
                 isa<xilinx::air::WaitAllOp>(op)) {
        kernelOps->push_back(CommandQueueEntry(op));
      }
#if HAVE_REGION_OP
      else if (auto r = dyn_cast<xilinx::air::RegionOp>(op)) {
        scheduleAIRRegion(r, kernelOps, dataOps, herdOps);
      }
#endif
      else if (auto hlo = dyn_cast<xilinx::air::HerdLaunchOp>(op)) {
        scheduleHerdLaunch(hlo, kernelOps, dataOps, herdOps);
      }
      else if (auto dmaOpIf = dyn_cast<air::DmaMemcpyInterface>(op)) {
        kernelOps->push_back( CommandQueueEntry(op) );
      }
      else if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op)) {
        kernelOps->push_back(CommandQueueEntry(op));
      }
      else if (auto sfo = dyn_cast<mlir::scf::ForOp>(op)) {
        scheduleScfFor(sfo, kernelOps, dataOps, herdOps);
      } else {
        ; // op->dump();
        ; // llvm_unreachable("unexpected operation");
      }
      op = op->getNextNode();
    }
  }

  void scheduleRegion(mlir::Region &region,
                      std::deque<CommandQueueEntry> *kernelOps,
                      std::deque<CommandQueueEntry> *dataOps,
                      std::deque<CommandQueueEntry> herdOps[16][16]) {
    for (auto &b : region.getBlocks())
      scheduleBlock(b, kernelOps, dataOps, herdOps);
  }

  void scheduleFunction(mlir::FuncOp &toplevel) {
    std::deque<CommandQueueEntry> kernelOps;
    std::deque<CommandQueueEntry> dataOps_Load;
    std::deque<CommandQueueEntry> dataOps_Store;
    std::deque<CommandQueueEntry> herdOps[16][16];

    // push everything onto the queues in program order
    scheduleRegion(toplevel.getRegion(), &kernelOps, &dataOps_Load, herdOps);

    uint64_t time = 1;

    bool running = true;
    while (running) {
      if (verbose)
        llvm::outs() << "time: " << time << "\n";

      // processQueue(dataOps_Store, time);
      processQueue(dataOps_Load, time);
      processQueue(kernelOps, time);
      for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
          processQueue(herdOps[y][x], time);
          processQueue(herdOps[y][x], time);
        }
      }

      std::vector<uint64_t> next_times;
      if (dataOps_Store.size()) {
        if (dataOps_Store.front().is_started() &&
            (dataOps_Store.front().end_time))
          next_times.push_back(dataOps_Store.front().end_time);
      }
      if (dataOps_Load.size()) {
        if (dataOps_Load.front().is_started() &&
            (dataOps_Load.front().end_time))
          next_times.push_back(dataOps_Load.front().end_time);
      }
      if (kernelOps.size()) {
        if (kernelOps.front().is_started() && (kernelOps.front().end_time))
          next_times.push_back(kernelOps.front().end_time);
      }
      for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
          if (herdOps[y][x].size()) {
            if (herdOps[y][x].front().is_started() &&
                (herdOps[y][x].front().end_time))
              next_times.push_back(herdOps[y][x].front().end_time);
          }
        }
      }
      running = dataOps_Store.size() || dataOps_Load.size() || kernelOps.size();
      for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
          running |= herdOps[y][x].size();
          running |= herdOps[y][x].size();
        }
      }
      uint64_t next_time;
      if (next_times.size())
        next_time = *std::min_element(next_times.begin(), next_times.end());
      time = std::max(time + 1, next_time);
    }

    for (unsigned ptr = 0, end = store.size(); ptr != end; ++ptr) {
      if (store[ptr].size()) {
        emitTraceEvent(traceStream, "dealloc", "layer", "E", time, ptr,
                       TRACE_PID_ALLOC);
        store[ptr].resize(0);
      }
    }
  }

  // todo private:
public:
  // The valueMap associates each SSA statement in the program
  // (represented by a Value*) with it's corresponding value.
  llvm::DenseMap<Value, llvm::Any> valueMap;

  // The timeMap associates each value with the time it was created.
  llvm::DenseMap<Value, uint64_t> timeMap;

private:
  llvm::raw_ostream &traceStream;
  std::vector<llvm::Any> results;
  std::vector<uint64_t> resultTimes;

  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.
  std::vector<std::vector<llvm::Any>> store;

  llvm::json::Value &jsonModel;

  uint64_t time;
}; // AIRRunner

} // namespace air
} // namespace xilinx

namespace {

LogicalResult run(int argc, char **argv, llvm::StringRef toolName) {

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> jsonFileName(
      "m", llvm::cl::desc("json model filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("arch.json"));

  static llvm::cl::opt<std::string> topLevelFunction(
      "f", llvm::cl::desc("top-level function name"),
      llvm::cl::value_desc("function"), llvm::cl::init("graph"));

  static llvm::cl::opt<bool> clVerbose("v", llvm::cl::desc("verbose"),
                                       llvm::cl::value_desc("bool"),
                                       llvm::cl::init(false));

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);

  verbose = clVerbose;

  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto json_file = openInputFile(jsonFileName, &errorMessage);
  if (!json_file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto processBuffer = [&](std::unique_ptr<llvm::MemoryBuffer> ownedBuffer,
                           raw_ostream &os) {
    MLIRContext context;
    DialectRegistry registry;
    registerAllDialects(registry);
    registry.insert<xilinx::air::airDialect>();
    context.appendDialectRegistry(registry);

    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), llvm::SMLoc());

    OwningModuleRef module(parseSourceFile(sourceMgr, &context));
    if (!module)
      return failure();

    Operation *topOp = module->lookupSymbol(topLevelFunction);
    // The toplevel function can accept any number of operands, and returns
    // any number of results.
    if (!topOp) {
      llvm::errs() << "Toplevel function " << topLevelFunction
                   << " not found!\n";
      return failure();
    }

    // We need three things in a function-type independent way.
    // The type signature of the function.
    FunctionType ftype;
    // The arguments of the entry block.
    Block::BlockArgListType blockArgs;

    std::string json_str = json_file->getBuffer().str();
    StringRef sr(json_str);
    auto jsonModel = llvm::json::parse(sr);
    if (!jsonModel)
      llvm_unreachable("failed to parse model json\n");

    xilinx::air::AIRRunner runner(os, *jsonModel);

    // The number of inputs to the function in the IR.
    unsigned numInputs = 0;
    unsigned numOutputs = 0;

    if (FuncOp toplevel = module->lookupSymbol<FuncOp>(topLevelFunction)) {
      ftype = toplevel.getType();
      Block &entryBlock = toplevel.getBody().front();
      blockArgs = entryBlock.getArguments();

      // Get the primary inputs of toplevel off the command line.
      numInputs = ftype.getNumInputs();
      numOutputs = ftype.getNumResults();
    } else {
      llvm_unreachable("Function not supported.\n");
    }

    std::vector<std::string> inputArgs;

    runner.emitTraceStart(os);

    for (unsigned i = 0; i < numInputs; i++) {
      Type type = ftype.getInput(i);
      if (auto tensorTy = type.dyn_cast<MemRefType>()) {
        // We require this memref type to be fully specified.
        std::vector<llvm::Any> nothing;
        unsigned buffer = runner.allocateMemRef(tensorTy, nothing);
        runner.valueMap[blockArgs[i]] = buffer;
        // runner.timeMap[blockArgs[i]] = 0;
      } else {
        llvm_unreachable("Only memref arguments are supported.\n");
      }
    }

    std::vector<llvm::Any> results(numOutputs);
    std::vector<uint64_t> resultTimes(numOutputs);
    if (FuncOp toplevel = module->lookupSymbol<FuncOp>(topLevelFunction)) {
      runner.scheduleFunction(toplevel);
    }
    runner.emitTraceEnd(os);
    return success();
  };
  if (failed(processBuffer(std::move(input), output->os())))
    return failure();

  output->keep();
  return success();
}

} // namespace

int main(int argc, char *argv[]) {
  return failed(run(argc, argv, "AIR MLIR Modeling Tool"));
}