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


module @lineGeneratorStitcherTwoAIEMultipleBufferElements {

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
    
    //line generator in tile11
    %core11 = AIE.core(%tile11) {
        //hand unroll while we do not know how to calculate lock index with modulo (maybe needs to be done in C++ generating this code from higher level dialect)
        //%c16 = constant 16 : index
        //AIE.useLock(%lock11_{%c16 mod 3}, "Acquire", 1) 
        
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %c9 = constant 9 : index
        %height = constant 9 : index
        %M = constant 3 : index

        scf.for %indexInHeight = %c0 to %height step %M { //NOTE: manually unrolled so if height is set to 10, iteration 9 will still happen, + 2 unrolled out of bound
            AIE.useLock(%lock11_0, "Acquire", 0) // acquire for produce
            %value0 = std.index_cast %indexInHeight : index to i32
            call @generateLineScalar(%value0,%buf11_0) : (i32, memref<16xi32>) -> ()
            AIE.useLock(%lock11_0, "Release", 1) // release for consume

            AIE.useLock(%lock11_1, "Acquire", 0) // acquire for produce
            %indexInHeightPlus1 = addi %indexInHeight, %c1 : index
            %value1 = std.index_cast %indexInHeightPlus1 : index to i32
            call @generateLineScalar(%value1,%buf11_1) : (i32, memref<16xi32>) -> ()
            AIE.useLock(%lock11_1, "Release", 1) // release for consume

            AIE.useLock(%lock11_2, "Acquire", 0) // acquire for produce
            %indexInHeightPlus2 = addi %indexInHeight, %c2 : index
            %value2 = std.index_cast %indexInHeightPlus2 : index to i32
            call @generateLineScalar(%value2,%buf11_2) : (i32, memref<16xi32>) -> ()
            AIE.useLock(%lock11_2, "Release", 1) // release for consume
        }

        // process remainder of height not multiple of M: height%M or 10 modulus 3 = 1
        AIE.useLock(%lock11_0, "Acquire", 0) // acquire for produce
        %value9 = std.index_cast %c9 : index to i32
        call @generateLineScalar(%value9,%buf11_0) : (i32, memref<16xi32>) -> ()
        AIE.useLock(%lock11_0, "Release", 1) // release for consume

        AIE.end
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
    
    func @storeLineVector(%lineIn:memref<16xi32>, %row:index, %bufferOut:memref<10x16xi32>) -> () {
        %c0 = constant 0 : index
        %lineWidth = constant 16 : index
        %c8 = constant 8 : index

        scf.for %indexInLine = %c0 to %lineWidth step %c8 {
            %cst = constant 0 : i32
            %tmpVector = vector.transfer_read %lineIn[%indexInLine], %cst : memref<16xi32>,vector<8xi32>
            vector.transfer_write %tmpVector, %bufferOut[%row,%indexInLine] : vector<8xi32>, memref<10x16xi32>
        }
        return
    }

    //stitcher

    %core12 = AIE.core(%tile12) {
        %c0 = constant 0 : index
        %c1 = constant 1 : index
        %c2 = constant 2 : index
        %c9 = constant 9 : index
        %height = constant 9 : index
        %M = constant 3 : index

        //acquire output buffer
        AIE.useLock(%lock12_0, "Acquire", 0) // acquire for produce
        
        scf.for %rowNumber = %c0 to %height step %M { //NOTE: manually unrolled so if height is set to 10, iteration 9 will still happen, + 2 unrolled out of bound
            AIE.useLock(%lock11_0, "Acquire", 1) // acquire for consume
            call @storeLineScalar(%buf11_0,%rowNumber,%buff12_0) : (memref<16xi32>,index,memref<10x16xi32>) -> ()
            AIE.useLock(%lock11_0, "Release", 0) // release for produce

            AIE.useLock(%lock11_1, "Acquire", 1) // acquire for consume
            %rowNumberPlus1 = std.addi %rowNumber, %c1 : index
            call @storeLineScalar(%buf11_1,%rowNumberPlus1,%buff12_0) : (memref<16xi32>,index,memref<10x16xi32>) -> ()
            AIE.useLock(%lock11_1, "Release", 0) // release for produce

            AIE.useLock(%lock11_2, "Acquire", 1) // acquire for consume
            %rowNumberPlus2 = std.addi %rowNumber, %c2 : index
            call @storeLineScalar(%buf11_2,%rowNumberPlus2,%buff12_0) : (memref<16xi32>,index,memref<10x16xi32>) -> ()
            AIE.useLock(%lock11_2, "Release", 0) // release for produce

        }
        // process remainder of height not multiple of M: height%M or 10 modulus 3 = 1
        AIE.useLock(%lock11_0, "Acquire", 1) // acquire for consume
        call @storeLineScalar(%buf11_0,%c9,%buff12_0) : (memref<16xi32>,index,memref<10x16xi32>) -> ()
        AIE.useLock(%lock11_0, "Release", 0) // release for produce

        AIE.useLock(%lock12_0, "Release", 1) // release for consume

        AIE.end
    }
}
