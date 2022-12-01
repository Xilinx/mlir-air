//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t72 = AIE.tile(7, 2)
  %t73 = AIE.tile(7, 3)
  %t74 = AIE.tile(7, 4)
  %t75 = AIE.tile(7, 5)
  %t82 = AIE.tile(8, 2)
  %t83 = AIE.tile(8, 3)
  %t84 = AIE.tile(8, 4)
  %t85 = AIE.tile(8, 5)

}
