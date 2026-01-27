//===- GPUKernelOutlinePass.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TO_GPU_OUTLINE
#define CONVERT_TO_GPU_OUTLINE

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createGPUKernelOutlinePass();

} // namespace air
} // namespace xilinx
#endif // CONVERT_TO_GPU_OUTLINE
//===- GPUKernelOutlinePass.h -------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TO_GPU_OUTLINE
#define CONVERT_TO_GPU_OUTLINE

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createGPUKernelOutlinePass();

} // namespace air
} // namespace xilinx
#endif // CONVERT_TO_GPU_OUTLINE
