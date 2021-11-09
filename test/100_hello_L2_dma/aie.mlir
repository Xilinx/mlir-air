// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

module @aie  {
  %0 = AIE.tile(7, 1)
  %1 = AIE.tile(7, 0)
  %2 = AIE.tile(0, 0)
  %3 = AIE.tile(7, 4)
}
