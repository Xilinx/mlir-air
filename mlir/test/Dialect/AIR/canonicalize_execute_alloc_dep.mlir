// RUN: air-opt -canonicalize -split-input-file %s | FileCheck %s

// Reproducer for issue #1559 (race #2/#3 family). Inside a merged herd body,
// `MergeAIRHerdsPattern` adds an inter-herd barrier dep from the previous
// herd's last token (a fill `scf.for` writing the shared output buffer) to
// the next herd's freshly-allocated input buffers. CanonicalizeAsyncOpDeps
// would previously drop this dep because:
//   1. `getAllWriteAccess` for the alloc execute incorrectly classified its
//      RESULT (the freshly-allocated memref) as "written" by the execute.
//   2. The fill scf.for's "writes" set is the predecessor herd's shared
//      output buffer (kernel arg).
//   3. These two memrefs are different SSA values → no RAW/WAR/WAW match
//      → dep dropped.
//
// The dep is load-bearing: it's the only chain ensuring the matmul scf.for
// (which transitively waits on the input alloc tokens via channel.gets and
// a wait_all) waits on fill before reading the shared output buffer.
//
// Fix: an `air.execute` whose body just allocates a new memref does not
// "write to" any pre-existing memref. The alloc result is fresh and cannot
// conflict with anything that came before. Special-case `air::ExecuteOp` in
// `getAllWriteAccess` to skip the conservative "all results are written"
// branch — body memref writes are still picked up via the region walk.

// CHECK-LABEL: func.func @execute_alloc_preserves_ordering_dep
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:   memref.store {{.*}} %{{.*}}arg{{.*}} : memref<32x32xi16, 2 : i32>
// The barrier dep on the fill loop must survive on the alloc execute:
// CHECK: air.execute [%[[FILL]]] -> (memref<8xi16, 2 : i32>)
// CHECK: air.execute [%[[FILL]]] -> (memref<8xi16, 2 : i32>)

module {
  func.func @execute_alloc_preserves_ordering_dep() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_alloc, %buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_alloc] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%buf) : memref<32x32xi16, 2 : i32> {
          %c0_i16 = arith.constant 0 : i16
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %wa_init = air.wait_all async
          %fill_loop = scf.for %i = %c0 to %c32 step %c1_h iter_args(%it = %wa_init) -> (!air.async.token) {
            %tk = air.execute [%it] {
              memref.store %c0_i16, %argbuf[%i, %i] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
          // These alloc executes carry an ordering dep on the fill loop;
          // canonicalize must not drop it just because the alloc bodies
          // touch fresh memrefs unrelated to %argbuf.
          %tk_alloc1, %inner1 = air.execute [%fill_loop] -> (memref<8xi16, 2 : i32>) {
            %a = memref.alloc() : memref<8xi16, 2 : i32>
            air.execute_terminator %a : memref<8xi16, 2 : i32>
          }
          %tk_alloc2, %inner2 = air.execute [%fill_loop] -> (memref<8xi16, 2 : i32>) {
            %a = memref.alloc() : memref<8xi16, 2 : i32>
            air.execute_terminator %a : memref<8xi16, 2 : i32>
          }
          %wa = air.wait_all async [%tk_alloc1, %tk_alloc2]
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %wa) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %inner1[%c0] : memref<8xi16, 2 : i32>
              memref.store %v, %inner2[%c0] : memref<8xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}
