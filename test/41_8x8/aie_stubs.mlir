//===- aie_stubs.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

module {
  // shim tiles
  %t020 = AIE.tile(2, 0)
  %t030 = AIE.tile(3, 0)
  %t060 = AIE.tile(6, 0)
  %t070 = AIE.tile(7, 0)
  %t100 = AIE.tile(10, 0)
  %t110 = AIE.tile(11, 0)
  %t180 = AIE.tile(18, 0)
  %t190 = AIE.tile(19, 0)
  %t260 = AIE.tile(26, 0)
  %t270 = AIE.tile(27, 0)
  %t340 = AIE.tile(34, 0)
  %t350 = AIE.tile(35, 0)
  %t420 = AIE.tile(42, 0)
  %t430 = AIE.tile(43, 0)
  %t460 = AIE.tile(46, 0)
  %t470 = AIE.tile(47, 0)

  // col 1
  %t191 = AIE.tile(19, 1)
  %t192 = AIE.tile(19, 2)
  %t193 = AIE.tile(19, 3)
  %t194 = AIE.tile(19, 4)
  %t195 = AIE.tile(19, 5)
  %t196 = AIE.tile(19, 6)
  %t197 = AIE.tile(19, 7)
  %t198 = AIE.tile(19, 8)

  // col 2
  %t201 = AIE.tile(20, 1)
  %t202 = AIE.tile(20, 2)
  %t203 = AIE.tile(20, 3)
  %t204 = AIE.tile(20, 4)
  %t205 = AIE.tile(20, 5)
  %t206 = AIE.tile(20, 6)
  %t207 = AIE.tile(20, 7)
  %t208 = AIE.tile(20, 8)

  // col 3
  %t211 = AIE.tile(21, 1)
  %t212 = AIE.tile(21, 2)
  %t213 = AIE.tile(21, 3)
  %t214 = AIE.tile(21, 4)
  %t215 = AIE.tile(21, 5)
  %t216 = AIE.tile(21, 6)
  %t217 = AIE.tile(21, 7)
  %t218 = AIE.tile(21, 8)

  // col 4
  %t221 = AIE.tile(22, 1)
  %t222 = AIE.tile(22, 2)
  %t223 = AIE.tile(22, 3)
  %t224 = AIE.tile(22, 4)
  %t225 = AIE.tile(22, 5)
  %t226 = AIE.tile(22, 6)
  %t227 = AIE.tile(22, 7)
  %t228 = AIE.tile(22, 8)

  // col 5
  %t231 = AIE.tile(23, 1)
  %t232 = AIE.tile(23, 2)
  %t233 = AIE.tile(23, 3)
  %t234 = AIE.tile(23, 4)
  %t235 = AIE.tile(23, 5)
  %t236 = AIE.tile(23, 6)
  %t237 = AIE.tile(23, 7)
  %t238 = AIE.tile(23, 8)

  // col 6
  %t241 = AIE.tile(24, 1)
  %t242 = AIE.tile(24, 2)
  %t243 = AIE.tile(24, 3)
  %t244 = AIE.tile(24, 4)
  %t245 = AIE.tile(24, 5)
  %t246 = AIE.tile(24, 6)
  %t247 = AIE.tile(24, 7)
  %t248 = AIE.tile(24, 8)

  // col 7
  %t251 = AIE.tile(25, 1)
  %t252 = AIE.tile(25, 2)
  %t253 = AIE.tile(25, 3)
  %t254 = AIE.tile(25, 4)
  %t255 = AIE.tile(25, 5)
  %t256 = AIE.tile(25, 6)
  %t257 = AIE.tile(25, 7)
  %t258 = AIE.tile(25, 8)

  // col 8
  %t261 = AIE.tile(26, 1)
  %t262 = AIE.tile(26, 2)
  %t263 = AIE.tile(26, 3)
  %t264 = AIE.tile(26, 4)
  %t265 = AIE.tile(26, 5)
  %t266 = AIE.tile(26, 6)
  %t267 = AIE.tile(26, 7)
  %t268 = AIE.tile(26, 8)

  // A row 1
  AIE.flow(%t020, "DMA" : 0, %t191, "East" : 3)
  AIE.flow(%t020, "DMA" : 0, %t191, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t201, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t211, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t221, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t231, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t241, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t251, "DMA" : 0)
  AIE.flow(%t201, "West" : 3, %t261, "DMA" : 0)
  // A row 2
  AIE.flow(%t060, "DMA" : 0, %t192, "East" : 3)
  AIE.flow(%t060, "DMA" : 0, %t192, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t202, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t212, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t222, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t232, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t242, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t252, "DMA" : 0)
  AIE.flow(%t202, "West" : 3, %t262, "DMA" : 0)
  // A row 3
  AIE.flow(%t100, "DMA" : 0, %t193, "East" : 3)
  AIE.flow(%t100, "DMA" : 0, %t193, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t203, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t213, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t223, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t233, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t243, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t253, "DMA" : 0)
  AIE.flow(%t203, "West" : 3, %t263, "DMA" : 0)
  // A row 4
  AIE.flow(%t180, "DMA" : 0, %t194, "East" : 3)
  AIE.flow(%t180, "DMA" : 0, %t194, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t204, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t214, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t224, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t234, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t244, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t254, "DMA" : 0)
  AIE.flow(%t204, "West" : 3, %t264, "DMA" : 0)
  // A row 5
  AIE.flow(%t255, "East" : 3, %t195, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t205, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t215, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t225, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t235, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t245, "DMA" : 0)
  AIE.flow(%t255, "East" : 3, %t255, "DMA" : 0)
  AIE.flow(%t260, "DMA" : 0, %t265, "DMA" : 0)
  AIE.flow(%t260, "DMA" : 0, %t265, "West" : 3)
  // A row 6
  AIE.flow(%t256, "East" : 3, %t196, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t206, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t216, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t226, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t236, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t246, "DMA" : 0)
  AIE.flow(%t256, "East" : 3, %t256, "DMA" : 0)
  AIE.flow(%t340, "DMA" : 0, %t266, "DMA" : 0)
  AIE.flow(%t340, "DMA" : 0, %t266, "West" : 3)
  // A row 7
  AIE.flow(%t257, "East" : 3, %t197, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t207, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t217, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t227, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t237, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t247, "DMA" : 0)
  AIE.flow(%t257, "East" : 3, %t257, "DMA" : 0)
  AIE.flow(%t420, "DMA" : 0, %t267, "DMA" : 0)
  AIE.flow(%t420, "DMA" : 0, %t267, "West" : 3)
  // A row 8
  AIE.flow(%t258, "East" : 3, %t198, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t208, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t218, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t228, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t238, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t248, "DMA" : 0)
  AIE.flow(%t258, "East" : 3, %t258, "DMA" : 0)
  AIE.flow(%t460, "DMA" : 0, %t268, "DMA" : 0)
  AIE.flow(%t460, "DMA" : 0, %t268, "West" : 3)

  // B col 1
  AIE.flow(%t030, "DMA" : 0, %t191, "North" : 0)
  AIE.flow(%t030, "DMA" : 0, %t191, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t192, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t193, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t194, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t195, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t196, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t197, "DMA" : 1)
  AIE.flow(%t192, "South" : 0, %t198, "DMA" : 1)
  // B col 2
  AIE.flow(%t070, "DMA" : 0, %t201, "North" : 0)
  AIE.flow(%t070, "DMA" : 0, %t201, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t202, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t203, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t204, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t205, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t206, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t207, "DMA" : 1)
  AIE.flow(%t202, "South" : 0, %t208, "DMA" : 1)
  // B col 3
  AIE.flow(%t110, "DMA" : 0, %t211, "North" : 0)
  AIE.flow(%t110, "DMA" : 0, %t211, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t212, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t213, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t214, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t215, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t216, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t217, "DMA" : 1)
  AIE.flow(%t212, "South" : 0, %t218, "DMA" : 1)
  // B col 4
  AIE.flow(%t190, "DMA" : 0, %t221, "North" : 0)
  AIE.flow(%t190, "DMA" : 0, %t221, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t222, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t223, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t224, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t225, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t226, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t227, "DMA" : 1)
  AIE.flow(%t222, "South" : 0, %t228, "DMA" : 1)
  // B col 5
  AIE.flow(%t270, "DMA" : 0, %t231, "North" : 0)
  AIE.flow(%t270, "DMA" : 0, %t231, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t232, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t233, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t234, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t235, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t236, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t237, "DMA" : 1)
  AIE.flow(%t232, "South" : 0, %t238, "DMA" : 1)
  // B col 6
  AIE.flow(%t350, "DMA" : 0, %t241, "North" : 0)
  AIE.flow(%t350, "DMA" : 0, %t241, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t242, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t243, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t244, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t245, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t246, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t247, "DMA" : 1)
  AIE.flow(%t242, "South" : 0, %t248, "DMA" : 1)
  // B col 7
  AIE.flow(%t430, "DMA" : 0, %t251, "North" : 0)
  AIE.flow(%t430, "DMA" : 0, %t251, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t252, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t253, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t254, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t255, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t256, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t257, "DMA" : 1)
  AIE.flow(%t252, "South" : 0, %t258, "DMA" : 1)
  // B col 8
  AIE.flow(%t470, "DMA" : 0, %t261, "North" : 3)
  AIE.flow(%t470, "DMA" : 0, %t261, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t262, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t263, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t264, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t265, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t266, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t267, "DMA" : 1)
  AIE.flow(%t262, "South" : 3, %t268, "DMA" : 1)

}

