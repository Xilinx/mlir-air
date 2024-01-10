//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

module {
  %t72 = aie.tile(7, 2)
  %t73 = aie.tile(7, 3)
  %t74 = aie.tile(7, 4)
  %t75 = aie.tile(7, 5)
  %t82 = aie.tile(8, 2)
  %t83 = aie.tile(8, 3)
  %t84 = aie.tile(8, 4)
  %t85 = aie.tile(8, 5)

}
