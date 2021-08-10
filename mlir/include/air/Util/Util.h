#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace xilinx {
namespace air {

void coalesceLoops(AffineForOp outer, AffineForOp inner);

void normalizeLoop(AffineForOp afo);

FuncOp getATenFn(ModuleOp module, std::string fnName, ArrayRef<Value> operands, ArrayRef<Type> retTys);

uint64_t getTensorVolume(const ShapedType ty);

uint64_t getTensorVolume(const Type ty);

}
}