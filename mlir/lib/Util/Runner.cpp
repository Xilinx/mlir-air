// (c) Copyright 2019-2022 Xilinx Inc. All Rights Reserved.

#include "air/Util/Runner.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/CostModel.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"

#include <deque>
#include <float.h>
#include <list>
#include <map>
#include <sstream>
#include <vector>

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

using namespace mlir;

namespace xilinx {
namespace air {

class AIRRunner::AIRRunner_impl {

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

  void decrementAsyncTokens(Operation *op, std::vector<llvm::Any> &in,
                            std::vector<llvm::Any> &out) {

    for (unsigned i = 0, e = op->getNumResults(); i < e; i++) {
      auto r = op->getResult(i);
      if (r.getType().isa<xilinx::air::AsyncTokenType>()) {
        valueMap[r] =
            llvm::any_cast<llvm::APInt>(valueMap[r]) - llvm::APInt(64, 1);
        assert(llvm::any_cast<llvm::APInt>(valueMap[r]).getSExtValue() >= 0);
        // valueMap[r] = out[i];
      }
    }
  }

  void executeOp(scf::ParallelOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {}

  void executeOp(xilinx::air::HerdLaunchOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    decrementAsyncTokens(op, in, out);
  }

  void executeOp(xilinx::air::DmaMemcpyInterface op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    decrementAsyncTokens(op, in, out);
  }

  void executeOp(xilinx::air::RegionOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    // decrementAsyncTokens(op, in, out);
  }

  void executeOp(xilinx::air::RegionTerminatorOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    auto regionOp = op->getParentOfType<xilinx::air::RegionOp>();
    std::vector<llvm::Any> v(regionOp->getNumResults(), llvm::APInt(64, 0));
    decrementAsyncTokens(regionOp, v, v);
  }

  void executeOp(xilinx::air::WaitAllOp op, std::vector<llvm::Any> &in,
                 std::vector<llvm::Any> &out) {
    decrementAsyncTokens(op, in, out);
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
    // else if (auto Op = dyn_cast<scf::ForOp>(op))
    //   executeOp(Op, inValues, outValues);
    // else if (auto Op = dyn_cast<scf::YieldOp>(op))
    // executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<scf::ParallelOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::RegionOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::RegionTerminatorOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::HerdLaunchOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::WaitAllOp>(op))
      executeOp(Op, inValues, outValues);
    else if (auto Op = dyn_cast<xilinx::air::DmaMemcpyInterface>(op))
      executeOp(Op, inValues, outValues);
    else
      return false;
    return true;
  }

  bool executeOp(Operation &op) {
    // int i = 0;
    std::vector<llvm::Any> inValues(op.getNumOperands());
    std::vector<llvm::Any> outValues(op.getNumResults());
    // LLVM_DEBUG(llvm::dbgs() << "OP:  " << op.getName() << "\n");
    // for (Value in : op.getOperands()) {
    //   inValues[i++] = valueMap[in];
    // }

    if (!executeOpImpls(op, inValues, outValues))
      return false;

    // record result in valuemap
    // i = 0;
    // for (Value out : op.getResults()) {
    //   LLVM_DEBUG(debugArg("OUT", out, outValues[i], time));
    //   valueMap[out] = outValues[i];
    //   timeMap[out] = time;
    //   i++;
    // }

    return true;
  }

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
  AIRRunner_impl(llvm::raw_ostream &trace_stream, llvm::json::Value &json_model,
                 bool verbose = false)
      : traceStream(trace_stream), jsonModel(json_model), time(1) {

    auto model = jsonModel.getAsObject();
    if (auto ds = model->getNumber("num_dispatch_queues"))
      dispatch_slots = (unsigned)(*ds);
    if (auto hs = model->getNumber("num_herd_slots"))
      herd_slots = (unsigned)(*hs);
    LLVM_DEBUG(llvm::dbgs() << "herd slots: " << herd_slots << "\n");
    LLVM_DEBUG(llvm::dbgs() << "dispatch slots: " << dispatch_slots << "\n");
  }

  // Allocate a new matrix with dimensions given by the type, in the
  // given store.  Return the pseudo-pointer to the new matrix in the
  // store (i.e. the first dimension index)
  unsigned allocateMemRef(mlir::MemRefType type, std::vector<llvm::Any> &in) {
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
    LLVM_DEBUG(llvm::dbgs() << "alloc " << ptr << " space " << memorySpace
                            << " size " << bytes << "\n");
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
    // auto allocationSize = store[ptr].size();
    LLVM_DEBUG(llvm::dbgs() << "dealloc " << ptr << "\n");
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

    // unsigned idx = 0;
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
      // llvm::dbgs() << "memory[" << i << "] ld time: " << ld_xfer_time[i] << "
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
    } else if (auto Op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)) {

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
    } else if (auto Op = mlir::dyn_cast<linalg::LinalgOp>(op)) {
      auto opCounts = xilinx::air::CostModel().getOpCounts(op);
      std::string skip = "footprint";
      std::string memops = "reads;writes;";
      std::string cpuops = "math.rsqrt;";
      cpuops += "arith.mulf;arith.divf;arith.addf;arith.subf;arith.truncf;"
                "arith.cmpf;arith.maxf;";
      cpuops += "arith.muli;arith.divi;arith.addi;arith.subi;arith.trunci;"
                "arith.cmpi;arith.maxi";
      cpuops += "std.select";
      uint64_t memory_op_count = 0;
      uint64_t compute_op_count = 0;
      for (auto &p : opCounts.map) {
        auto name = std::get<0>(p);
        auto count = std::get<1>(p);
        if (memops.find(name) != std::string::npos)
          memory_op_count += count;
        else if (cpuops.find(name) != std::string::npos)
          compute_op_count += count;
        else if (skip.find(name) == std::string::npos)
          LLVM_DEBUG(llvm::dbgs() << name << " not counted\n");
      }
      c.compute_xfer_cost = 0; // memory_op_count;

      if (compute_op_count) {
        // defaults
        double num_cores = 1;
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
        assert(ops_per_cycle > 0 &&
               "ops per cycle in model must be greater than zero");

        double cycles = ceil(compute_op_count / ops_per_cycle);
        c.compute_op_cost = cycles;
      }
      execution_time = std::max(c.compute_op_cost, c.compute_xfer_cost);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: execution time not modeled for op: '");
      LLVM_DEBUG(llvm::dbgs() << to_string(op) << "'\n");
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
        LLVM_DEBUG(llvm::dbgs() << "finish: '");
        LLVM_DEBUG(c.op->print(llvm::dbgs()));
        LLVM_DEBUG(llvm::dbgs() << "' @ " << time << "\n");

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
        LLVM_DEBUG(llvm::dbgs() << "running: '");
        LLVM_DEBUG(c.op->print(llvm::dbgs()));
        LLVM_DEBUG(llvm::dbgs()
                   << "' @ " << time << " - " << c.end_time << "\n");
        // in-order, return.
        return;
      }
    }

    if (!q.size()) {
      LLVM_DEBUG(llvm::dbgs() << "queue empty @ " << time << "\n");
      return;
    }

    CommandQueueEntry &c_next = q.front();

    if (!c_next.queue_ready_time)
      c_next.queue_ready_time = time;

    bool ready = true;
    for (Value in : c_next.op->getOperands()) {
      if (in.getType().isa<xilinx::air::AsyncTokenType>()) {
        if (!valueMap.count(in))
          ready = false;
        else if (llvm::any_cast<llvm::APInt>(valueMap[in]) != 0) {
          LLVM_DEBUG(llvm::dbgs()
                     << "count @ " << llvm::any_cast<llvm::APInt>(valueMap[in])
                     << "\n");
          ready = false;
        }
      }
    }

    if (!ready) {
      LLVM_DEBUG(llvm::dbgs() << "not ready: '");
      LLVM_DEBUG(c_next.op->print(llvm::dbgs()));
      LLVM_DEBUG(llvm::dbgs() << "' @ " << time << "\n");
      return;
    }

    c_next.start_time = time;
    c_next.end_time = time + modelOp(c_next);
    LLVM_DEBUG(llvm::dbgs() << "start: '");
    LLVM_DEBUG(c_next.op->print(llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs()
               << "' @ " << time << " - " << c_next.end_time << "\n");

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

  void scheduleAIRRegion(
      xilinx::air::RegionOp &ro, std::deque<CommandQueueEntry> *programQueue,
      std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
      std::deque<CommandQueueEntry> tileQueue[16][16]) {

    for (auto r : ro->getResults()) {
      if (r.getType().isa<xilinx::air::AsyncTokenType>()) {
        valueMap[r] = llvm::APInt(64, 1);
      }
    }

    programQueue->push_back(CommandQueueEntry(ro.getOperation()));
    scheduleBlock(ro->getRegion(0).front(), programQueue, dispatchQueue,
                  tileQueue);
  }

  void
  scheduleScfFor(mlir::scf::ForOp &fo,
                 std::deque<CommandQueueEntry> *programQueue,
                 std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
                 std::deque<CommandQueueEntry> tileQueue[16][16]) {
    auto ub = fo.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lb = fo.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step = fo.getStep().getDefiningOp<arith::ConstantIndexOp>();

    assert(ub && lb && step);
    auto r = ub.value() - lb.value();
    auto trip_count = mlir::ceilDiv(r, step.value());

    // programQueue->push_back(CommandQueueEntry(fo.getOperation()));

    for (auto r : fo->getResults()) {
      if (r.getType().isa<xilinx::air::AsyncTokenType>()) {
        valueMap[r] = llvm::APInt(64, 1);
      }
    }

    BlockAndValueMapping map;
    // for the first iteration of the loop, map the iter_args to the
    // operands of the scf.for
    for (auto t : llvm::zip(fo.getRegionIterArgs(), fo.getIterOperands()))
      map.map(std::get<0>(t), std::get<1>(t));
    for (auto i = 0; i < trip_count; i++) {
      auto bb = new mlir::Block();
      fo.getRegion().push_back(bb);
      auto bldr = OpBuilder::atBlockEnd(bb);
      for (auto &o : fo.getRegion().front()) {
        auto c = bldr.clone(o, map);
        // for the next iteration of the loop, map the iter_args to the
        // operands of the scf.yield
        if (isa<scf::YieldOp>(c)) {
          map.clear();
          for (auto t : llvm::zip(fo.getRegionIterArgs(), c->getOperands()))
            map.map(std::get<0>(t), std::get<1>(t));
        }
      }
      scheduleBlock(*bb, programQueue, dispatchQueue, tileQueue);
    }
    // replace the results of the scf.for with the operands of the
    // final scf.yield operation
    for (auto &a : fo.getIterOpOperands()) {
      auto v = map.lookupOrNull(fo.getRegionIterArgForOpOperand(a));
      fo.getResultForOpOperand(a).replaceAllUsesWith(v);
    }
    return;
  }

  void scheduleScfParallel(
      mlir::scf::ParallelOp &po, std::deque<CommandQueueEntry> *programQueue,
      std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
      std::deque<CommandQueueEntry> tileQueue[16][16]) {

    int64_t trip_count = 1;
    for (auto t :
         llvm::zip(po.getLowerBound(), po.getUpperBound(), po.getStep())) {
      auto lb = std::get<0>(t).getDefiningOp<arith::ConstantIndexOp>();
      auto ub = std::get<1>(t).getDefiningOp<arith::ConstantIndexOp>();
      auto step = std::get<2>(t).getDefiningOp<arith::ConstantIndexOp>();
      assert(ub && lb && step);
      auto r = ub.value() - lb.value();
      trip_count *= mlir::ceilDiv(r, step.value());
    }

    programQueue->push_back(
        CommandQueueEntry(po.getOperation(), [=](Operation *op) {
          auto spo = cast<scf::ParallelOp>(op);

          auto initOperands = spo.getInitVals();
          Value lastResult = initOperands[0];
          for (auto i = 0; i < trip_count; i++) {
            BlockAndValueMapping map;
            auto bb = new mlir::Block();
            spo.getRegion().push_back(bb);
            auto bldr = OpBuilder::atBlockEnd(bb);
            for (auto &o : spo.getRegion().front().getOperations()) {
              if (auto rop = dyn_cast<scf::ReduceOp>(o)) {
                auto &rbb = rop.getRegion().front();
                map.map(rbb.getArgument(0), lastResult);
                map.map(rbb.getArgument(1), map.lookupOrDefault(rop.getOperand()));
                for (auto &ro : rbb) {
                  if (auto rrop = dyn_cast<scf::ReduceReturnOp>(ro))
                    lastResult = map.lookupOrDefault(rrop.getOperand());
                  else
                    bldr.clone(ro, map);
                }
              } else {
                bldr.clone(o, map);
              }
              bb->dump();
            }
            scheduleBlock(*bb, &(dispatchQueue->data()[i % dispatch_slots]),
                          dispatchQueue, tileQueues[i % herd_slots]);
          }
          spo.getResult(0).replaceAllUsesWith({lastResult});
        }));
    return;
  }

  void scheduleHerdLaunch(
      xilinx::air::HerdLaunchOp &hlo,
      std::deque<CommandQueueEntry> *programQueue,
      std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
      std::deque<CommandQueueEntry> tileQueue[16][16]) {

    int64_t cols = cast<arith::ConstantIndexOp>(
                       hlo.getHerdSizeOperands().x.getDefiningOp())
                       .value();
    int64_t rows = cast<arith::ConstantIndexOp>(
                       hlo.getHerdSizeOperands().y.getDefiningOp())
                       .value();

    if (hlo->getNumResults())
      valueMap[hlo->getResult(0)] = APInt(64, rows * cols + 1);

    programQueue->push_back(
        CommandQueueEntry(hlo.getOperation(), [=](Operation *op) {
          auto ho = cast<xilinx::air::HerdLaunchOp>(op);
          auto tokenTy = xilinx::air::AsyncTokenType::get(op->getContext());
          // SmallVector<Value, 16> exit_tokens;
          for (auto row = 0; row < rows; row++) {
            for (auto col = 0; col < cols; col++) {
              BlockAndValueMapping map;
              auto bb = new mlir::Block();
              ho.getRegion().push_back(bb);
              auto bldr = OpBuilder::atBlockEnd(bb);
              // insert a blocking wait on all input dependencies of the launch
              // op
              bldr.create<xilinx::air::WaitAllOp>(op->getLoc(),
                                                  SmallVector<Type, 1>{},
                                                  ho.getAsyncDependencies());
              SmallVector<Value, 8> exit_deps;
              for (auto &o : ho.getRegion().front()) {
                auto c = bldr.clone(o, map);
                // collect each token created in the block
                for (auto r : c->getResults())
                  if (r.getType().isa<xilinx::air::AsyncTokenType>())
                    exit_deps.push_back(r);
              }
              Operation *t = nullptr;
              if (ho->getNumResults()) {
                // Create an async wait_all on all tokens created in the block.
                // When the wait_all runs it decrements the herd result event.
                t = bldr.create<xilinx::air::WaitAllOp>(
                    op->getLoc(), SmallVector<Type, 1>{tokenTy}, exit_deps);
                // exit_tokens.push_back(t.getResult(0));
              }
              scheduleBlock(*bb, &tileQueue[col][row], dispatchQueue,
                            tileQueue);
              if (t) {
                valueMap[t->getResult(0)] = APInt(64, 2);
                tileQueue[col][row].push_back(
                    CommandQueueEntry(t, [=](Operation *tOp) {
                      valueMap[op->getResult(0)] =
                          llvm::any_cast<llvm::APInt>(
                              valueMap[op->getResult(0)]) -
                          llvm::APInt(64, 1);
                    }));
              }
            }
          }
        }));
  }

  void scheduleAIRAsyncOp(Operation *op, std::deque<CommandQueueEntry> *q) {
    q->push_back(CommandQueueEntry(op));
    for (auto r : op->getResults()) {
      if (r.getType().isa<xilinx::air::AsyncTokenType>()) {
        valueMap[r] = APInt(64, 1);
      }
    }
  }

  void
  scheduleBlock(mlir::Block &block, std::deque<CommandQueueEntry> *programQueue,
                std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
                std::deque<CommandQueueEntry> tileQueue[16][16]) {
    if (!block.getOperations().size())
      return;
    Operation &o = block.getOperations().front();
    Operation *op = &o;
    while (op) {

      if (isa<arith::ConstantOp>(op) || isa<arith::ConstantIndexOp>(op)) {
        executeOp(*op);
      } else if (isa<memref::AllocOp>(op) || isa<memref::DeallocOp>(op)) {
        programQueue->push_back(CommandQueueEntry(op));
      } else if (isa<xilinx::air::WaitAllOp>(op) ||
                 isa<xilinx::air::DmaMemcpyInterface>(op) ||
                 isa<xilinx::air::RegionTerminatorOp>(op)) {
        scheduleAIRAsyncOp(op, programQueue);
      } else if (auto r = dyn_cast<xilinx::air::RegionOp>(op)) {
        scheduleAIRRegion(r, programQueue, dispatchQueue, tileQueue);
      } else if (auto hlo = dyn_cast<xilinx::air::HerdLaunchOp>(op)) {
        scheduleHerdLaunch(hlo, programQueue, dispatchQueue, tileQueue);
      } else if (auto linalgOp = mlir::dyn_cast<linalg::LinalgOp>(op)) {
        programQueue->push_back(CommandQueueEntry(op));
      } else if (auto sfo = dyn_cast<mlir::scf::ForOp>(op)) {
        scheduleScfFor(sfo, programQueue, dispatchQueue, tileQueue);
      } else if (auto spo = dyn_cast<mlir::scf::ParallelOp>(op)) {
        scheduleScfParallel(spo, programQueue, dispatchQueue, tileQueue);
      } else {
        ; // op->dump();
        ; // llvm_unreachable("unexpected operation");
      }
      op = op->getNextNode();
    }
  }

  void
  scheduleRegion(mlir::Region &region,
                 std::deque<CommandQueueEntry> *programQueue,
                 std::array<std::deque<CommandQueueEntry>, 32> *dispatchQueue,
                 std::deque<CommandQueueEntry> tileQueue[16][16]) {
    for (auto &b : region.getBlocks())
      scheduleBlock(b, programQueue, dispatchQueue, tileQueue);
  }

  void scheduleFunction(func::FuncOp &toplevel) {
    std::deque<CommandQueueEntry> programQueue;
    std::array<std::deque<CommandQueueEntry>, 32> dispatchQueue;

    // push everything onto the queues in program order
    scheduleRegion(toplevel.getRegion(), &programQueue, &dispatchQueue,
                   tileQueues[0]);

    uint64_t time = 1;

    bool running = true;
    while (running) {
      LLVM_DEBUG(llvm::dbgs() << "time: " << time << "\n");

      running = false;
      std::vector<uint64_t> next_times;

      processQueue(programQueue, time);
      if (programQueue.size()) {
        running = true;
        if (programQueue.front().is_started() &&
            (programQueue.front().end_time))
          next_times.push_back(programQueue.front().end_time);
      }
      for (unsigned i = 0; i < dispatchQueue.size(); i++) {
        processQueue(dispatchQueue[i], time);
        if (dispatchQueue[i].size()) {
          running = true;
          if (dispatchQueue[i].front().is_started() &&
              (dispatchQueue[i].front().end_time))
            next_times.push_back(dispatchQueue[i].front().end_time);
        }
      }
      for (int i = 0; i < 4; i++) {
        for (int y = 0; y < 16; y++) {
          for (int x = 0; x < 16; x++) {
            processQueue(tileQueues[i][y][x], time);
            if (tileQueues[i][y][x].size()) {
              running = true;
              if (tileQueues[i][y][x].front().is_started() &&
                  (tileQueues[i][y][x].front().end_time))
                next_times.push_back(tileQueues[i][y][x].front().end_time);
            }
          }
        }
      }

      uint64_t next_time = 0;
      if (next_times.size())
        next_time = *std::min_element(next_times.begin(), next_times.end());
      time = std::max(time + 1, next_time);
      // if (time > 5000000)
      //   running = false;
    }

    for (unsigned ptr = 0, end = store.size(); ptr != end; ++ptr) {
      if (store[ptr].size()) {
        emitTraceEvent(traceStream, "dealloc", "layer", "E", time, ptr,
                       TRACE_PID_ALLOC);
        store[ptr].resize(0);
      }
    }
    llvm::dbgs() << "Finished at time " << time << "\n";
  }

private:
  llvm::raw_ostream &traceStream;
  llvm::json::Value &jsonModel;
  uint64_t time;

  std::vector<llvm::Any> results;
  std::vector<uint64_t> resultTimes;

  // The valueMap associates each SSA statement in the program
  // (represented by a Value*) with it's corresponding value.
  llvm::DenseMap<Value, llvm::Any> valueMap;
  // The timeMap associates each value with the time it was created.
  // llvm::DenseMap<Value, uint64_t> timeMap;

  std::deque<CommandQueueEntry> tileQueues[32][16][16];

  // The store associates each allocation in the program
  // (represented by a int) with a vector of values which can be
  // accessed by it.
  std::vector<std::vector<llvm::Any>> store;

  unsigned dispatch_slots;
  unsigned herd_slots;

}; // AIRRunner_impl

AIRRunner::AIRRunner(llvm::raw_ostream &trace_stream,
                     llvm::json::Value &json_model, bool verbose) {
  impl = std::make_unique<AIRRunner_impl>(trace_stream, json_model, verbose);
  if (verbose) {
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType(DEBUG_TYPE);
  }
}

AIRRunner::~AIRRunner() {}

void AIRRunner::emitTraceStart(llvm::raw_ostream &s) {
  impl->emitTraceStart(s);
}

void AIRRunner::emitTraceEnd(llvm::raw_ostream &s) { impl->emitTraceEnd(s); }

void AIRRunner::scheduleFunction(func::FuncOp &toplevel) {
  impl->scheduleFunction(toplevel);
}

} // namespace air
} // namespace xilinx