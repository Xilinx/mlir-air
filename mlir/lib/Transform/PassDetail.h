#ifndef AIR_TRANSFORM_PASSDETAIL_H
#define AIR_TRANSFORM_PASSDETAIL_H

#include "mlir/Pass/Pass.h"

namespace xilinx {
namespace air {

using namespace mlir;

#define GEN_PASS_CLASSES
#include "air/Transform/Passes.h.inc"

} // namespace air
} // namespace xilinx

#endif // AIR_TRANSFORM_PASSDETAIL_H