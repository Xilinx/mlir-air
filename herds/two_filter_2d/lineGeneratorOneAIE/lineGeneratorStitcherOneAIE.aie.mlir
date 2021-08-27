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


module @lineGeneratorStitcherOneAIE {


    %tile11 = AIE.tile(1, 2)
    %tile12 = AIE.tile(1, 3)

    %buf11_0 = AIE.buffer(%tile11) { sym_name = "c0" } : memref<16xi32> //memref<%lineWidth x i32> //coarse circular bank 0
    %buf11_1 = AIE.buffer(%tile11) { sym_name = "c1" } : memref<16xi32> //memref<%lineWidth x i32> //coarse circular bank 1
    %buf11_2 = AIE.buffer(%tile11) { sym_name = "c2" } : memref<16xi32> //memref<%lineWidth x i32> //coarse circular bank 1

    %lock11_0 = AIE.lock(%tile11, 0)
    %lock11_1 = AIE.lock(%tile11, 1)
    %lock11_2 = AIE.lock(%tile11, 2)

    %buff12_0 = AIE.buffer(%tile12) { sym_name = "out" } :  memref<10x16xi32> // memref<%height x %lineWidth x i32>//result in bank 0
    %lock12_0 = AIE.lock(%tile12, 0)
    
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

    //single line store function
    func @storeLineScalar(%lineIn:memref<16xi32>, %row:index, %bufferOut:memref<10x16xi32>) -> () {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %lineWidth = constant 16 : index

        scf.for %indexInLine = %c0 to %lineWidth step %c1 {
            %value0 = memref.load %lineIn[%indexInLine] : memref<16xi32>
            memref.store %value0, %bufferOut[%row,%indexInLine] : memref<10x16xi32>
        }
        return
    }
    
    //line generator and stitcher in tile12

    %core12 = AIE.core(%tile12) {
        %c0 = constant 0 : index
        %height = constant 10 : index
        %M = constant 1 : index

        //acquire output buffer
        AIE.useLock(%lock12_0, "Acquire", 0, 0) // acquire for produce


        scf.for %indexInHeight = %c0 to %height step %M {
            AIE.useLock(%lock11_0, "Acquire", 0, 0) // acquire for produce
            %value0 = std.index_cast %indexInHeight : index to i32
            call @generateLineScalar(%value0,%buf11_0) : (i32, memref<16xi32>) -> ()
            AIE.useLock(%lock11_0, "Release", 1, 0) // release for consume

            AIE.useLock(%lock11_0, "Acquire", 1, 0) // acquire for consume
            call @storeLineScalar(%buf11_0,%indexInHeight,%buff12_0) : (memref<16xi32>,index,memref<10x16xi32>) -> ()
            AIE.useLock(%lock11_0, "Release", 0, 0) // release for produce
        }

        AIE.useLock(%lock12_0, "Release", 1, 0) // release for consume

        AIE.end
    }
}
