//===- lineGeneratorStitcher.aie.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
// Date: July 21st 2021
// 
//===----------------------------------------------------------------------===//


module @vectorBroadCast {

    %tile12 = AIE.tile(1, 2)

    %buf12_0 = AIE.buffer(%tile12) { sym_name = "outVector" } : memref<16xi32>
    %buf12_1 = AIE.buffer(%tile12) { sym_name = "outScalar" } : memref<16xi32> 

    %lock12_0 = AIE.lock(%tile12, 0)
    
    //single line generator filled with same value passed in as argument
    func @generateLineVector(%value: i32, %lineOut:memref<16xi32>) -> () {
        %c0 = constant 0 : index
        %lineWidth = constant 16 : index
        %c8 = constant 8 : index

        //create vector of %value
        %vectorOfValue = vector.broadcast %value: i32 to vector<8xi32>
        
        scf.for %indexInLine = %c0 to %lineWidth step %c8 {
            vector.transfer_write %vectorOfValue, %lineOut[%indexInLine] : vector<8xi32>, memref<16xi32>
        }
        return
    }

    //single line generator filled with same value passed in as argument
    func @generateLineScalar(%value: i32, %lineOut:memref<16xi32>) -> () {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %lineWidth = constant 16 : index
        
        scf.for %indexInLine = %c0 to %lineWidth step %c1 {
            memref.store %value, %lineOut[%indexInLine] : memref<16xi32>
        }
        return
    }
        
    //line generator and stitcher in tile12

    %core12 = AIE.core(%tile12) {
        %value0 = constant 5 : i32

        //acquire output buffer
        AIE.useLock(%lock12_0, "Acquire", 0) // acquire for produce
        
        call @generateLineVector(%value0,%buf12_0) : (i32, memref<16xi32>) -> ()
        call @generateLineScalar(%value0,%buf12_1) : (i32, memref<16xi32>) -> ()

        AIE.useLock(%lock12_0, "Release", 1) // release for consume

        AIE.end
    }
}
