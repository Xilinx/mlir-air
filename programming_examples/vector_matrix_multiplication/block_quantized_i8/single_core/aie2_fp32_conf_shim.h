// Peano-only shim for stock Xilinx/aie_api on aie2 (npu1).
// aie_api's aie2 fp32 multiply path calls mul_elem_16_conf/mac_elem_16_conf,
// but Peano (llvm-aie) implements only the plain mul_elem_16/mac_elem_16 for
// aie2; the _conf forms exist only for aie2p. aie_api always passes conf=0
// (plain multiply/accumulate), so forward to the plain intrinsics.
#if defined(__AIENGINE__) && !defined(__chess__)
#include <aiev2intrin.h>
__attribute__((always_inline)) inline
v16accfloat mul_elem_16_conf(v16float a, v16float b, int /*conf*/) {
  return mul_elem_16(a, b);
}
__attribute__((always_inline)) inline
v16accfloat mac_elem_16_conf(v16float a, v16float b, v16accfloat acc,
                             int, int, int) {
  return mac_elem_16(a, b, acc);
}
#endif
