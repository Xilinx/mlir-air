// module {
//   air.channel @channel_0 [1, 1]
//   air.channel @channel_1 [1, 1]
//   func.func @test() {
//     %c1 = arith.constant 1 : index
//     %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) {
//       %async_token, %results = air.execute -> (memref<256xi32>) {
//         %alloc = memref.alloc() : memref<256xi32>
//         air.execute_terminator %alloc : memref<256xi32>
//       }
//       %1 = air.channel.put async [%async_token]  @channel_0[] (%results[] [] []) : (memref<256xi32>)
//       %2 = air.partition async  {
//         %c1_0 = arith.constant 1 : index
//         %async_token_1, %results_2 = air.execute -> (memref<256xi32, 1>) {
//           %alloc = memref.alloc() : memref<256xi32, 1>
//           air.execute_terminator %alloc : memref<256xi32, 1>
//         }
//         %3 = air.channel.get async [%async_token_1]  @channel_0[] (%results_2[] [] []) : (memref<256xi32, 1>)
//         %4 = air.channel.put async [%3]  @channel_1[] (%results_2[] [] []) : (memref<256xi32, 1>)
//         %5 = air.herd @herd_0 async [%4]  tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c1_0) {
//           %async_token_3, %results_4 = air.execute -> (memref<256xi32, 2>) {
//             %alloc = memref.alloc() : memref<256xi32, 2>
//             air.execute_terminator %alloc : memref<256xi32, 2>
//           }
//           %6 = air.channel.get async [%async_token_3]  @channel_1[] (%results_4[] [] []) : (memref<256xi32, 2>)
//           air.herd_terminator
//         }
//         air.partition_terminator
//       }
//       air.launch_terminator
//     }
//     return
//   }
// }

// module @aie.partition_0 {
//   %0 = AIE.tile(1, 1)
//   %1 = AIE.core(%0) {
//     %alloc = memref.alloc() : memref<256xi32, 2>
//     %3 = air.channel.get async [%2]  @channel_1[] (%alloc[] [] []) : (memref<256xi32, 2>)
//     AIE.end
//   } {elf_file = "partition_0_core_1_1.elf"}
//   air.channel @channel_1 [1, 1]
// }

// module @aie.partition_0 {
//   %0 = AIE.tile(2, 1)
//   %1 = AIE.tile(2, 0)
//   %2 = AIE.objectFifo.createObjectFifo(%1, {%0}, 1) : !AIE.objectFifo<memref<256xi32>>
//   %3 = AIE.core(%0) {
//     %4 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<256xi32>>, 1) : !AIE.objectFifoSubview<memref<256xi32>>
//     %5 = AIE.objectFifo.subview.access %4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
//     AIE.end
//   } {elf_file = "partition_0_core_1_1.elf"}
// }

module @aie.partition_0 {
  %0 = AIE.tile(2, 1)
  %1 = AIE.tile(2, 0)
  AIE.flow(%1, DMA : 0, %0, DMA : 0)
  %2 = AIE.buffer(%0) {sym_name = "of_1_buff_0"} : memref<256xi32>
  %3 = AIE.lock(%0, 0) {sym_name = "of_1_lock_0"}
  %4 = AIE.core(%0) {
    AIE.useLock(%3, Acquire, 1)
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
  %5 = AIE.mem(%0) {
    %6 = AIE.dmaStart(S2MM, 0, ^bb1, ^bb2)
  ^bb1:  // 2 preds: ^bb0, ^bb1
    AIE.useLock(%3, Acquire, 0)
    AIE.dmaBd(<%2 : memref<256xi32>, 0, 256>, 0)
    AIE.useLock(%3, Release, 1)
    AIE.nextBd ^bb1
  ^bb2:  // pred: ^bb0
    AIE.end
  }
}
