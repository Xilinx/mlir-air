// RUN: air-opt -canonicalize -split-input-file %s | FileCheck %s

// Reproducer for the residual race in issue #1559 (race #2/#3 family,
// post-`fb14adaf9`). After `air-ping-pong-transform` restructures the merged
// herd body, the `MergeAIRHerds`-injected barrier dep on the fill scf.for
// migrates to an `air.channel.get` that fetches input data into a freshly-
// allocated input buffer.
//
// Before the fix below, `CanonicalizeAsyncOpDeps` would drop the dep:
//   - source = fill scf.for, writes %output_buf
//   - sink = chan.get, writes %input_buf (no read of %output_buf)
//   - no RAW/WAR/WAW match → dep dropped
//
// But the chan.get is just a synchronization primitive; its async-token
// consumer chain (chan.get → wait_all → matmul scf.for) reads %output_buf.
// Dropping the dep removes the only ordering edge between fill and matmul
// → race on %output_buf.
//
// Fix: when computing the sink op's effective memref accesses for the
// RAW/WAR/WAW pruning, walk forward through the sink's async-token consumer
// chain and union in their accesses too. The chan.get inherits the matmul's
// reads of %output_buf, so RAW(fill, chan.get) matches and the dep is kept.

// CHECK-LABEL: func.func @chan_get_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:   memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// The barrier dep on the fill loop must survive on the chan.get sinks:
// CHECK: air.channel.get async [%[[FILL]]
// CHECK: air.channel.get async [%[[FILL]]

module {
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  func.func @chan_get_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
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
          %tk_in1, %in_buf1 = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %tk_in2, %in_buf2 = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          // These chan.get ops carry a scheduling barrier on the fill loop.
          // The fill loop writes %argbuf; the chan.gets write distinct fresh
          // input buffers — so naive RAW/WAR/WAW analysis would drop the
          // %fill_loop dep. But the matmul scf.for below (the chan.gets'
          // transitive consumer) READS %argbuf, so the barrier is real.
          %g1 = air.channel.get async [%fill_loop, %tk_in1] @ch_a[%tx, %ty] (%in_buf1[] [] []) : (memref<32xi16, 2 : i32>)
          %g2 = air.channel.get async [%fill_loop, %tk_in2] @ch_b[%tx, %ty] (%in_buf2[] [] []) : (memref<32xi16, 2 : i32>)
          %wa = air.wait_all async [%g1, %g2]
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %wa) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              %v1 = memref.load %in_buf1[%k] : memref<32xi16, 2 : i32>
              %v2 = memref.load %in_buf2[%k] : memref<32xi16, 2 : i32>
              %sum = arith.addi %v1, %v2 : i16
              %sum2 = arith.addi %sum, %v : i16
              memref.store %sum2, %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Same scenario as above, but the sink op carrying the fill-loop barrier dep
// is `air.channel.put` instead of `air.channel.get`. The put reads a fresh
// staging buffer (not %argbuf), so the source op's write-set and the sink's
// own read-set don't overlap. The dep is real because the put's transitive
// async-token consumer (the matmul scf.for) reads %argbuf.

// CHECK-LABEL: func.func @chan_put_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:   memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: air.channel.put async [%[[FILL]]

module {
  air.channel @ch_p [1, 1]
  func.func @chan_put_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
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
          %tk_stage, %stage_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          // Sink: chan.put reads %stage_buf (fresh, disjoint from %argbuf).
          // Barrier dep on %fill_loop must survive — downstream loop reads %argbuf.
          %p = air.channel.put async [%fill_loop, %tk_stage] @ch_p[%tx, %ty] (%stage_buf[] [] []) : (memref<32xi16, 2 : i32>)
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %p) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %stage_buf[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Sink op is `air.dma_memcpy_nd`. The DMA copies between two fresh L1 buffers
// (%src_buf → %dst_buf), neither of which is %argbuf. Without the consumer-
// walk, the fill→DMA barrier dep would be dropped: source writes %argbuf,
// sink reads %src_buf and writes %dst_buf — no RAW/WAR/WAW match. The matmul
// loop downstream reads %argbuf, making the barrier real.

// CHECK-LABEL: func.func @dma_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:   memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: air.dma_memcpy_nd async [%[[FILL]]

module {
  func.func @dma_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
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
          %tk_src, %src_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %tk_dst, %dst_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          // Sink: DMA between two fresh buffers. Barrier dep on %fill_loop
          // must survive — downstream loop reads %argbuf via %dma's token.
          %dma = air.dma_memcpy_nd async [%fill_loop, %tk_src, %tk_dst] (%dst_buf[] [] [], %src_buf[] [] []) : (memref<32xi16, 2 : i32>, memref<32xi16, 2 : i32>)
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %dma) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %dst_buf[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Sink op is `air.execute` wrapping a pure `memref.alloc`. The execute body
// does not touch %argbuf — its only memref result is a freshly-allocated
// buffer. Without the consumer-walk, the fill→execute barrier dep would be
// dropped (source writes %argbuf; sink writes a fresh memref result). The
// matmul loop downstream reads %argbuf via the execute's token, making the
// barrier real. This case also exercises the region-walking path of the
// generalized sink-side analysis (air.execute carries a body region).

// CHECK-LABEL: func.func @execute_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:   memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: %{{.*}}, %{{.*}} = air.execute [%[[FILL]]]

module {
  func.func @execute_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
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
          // Sink: air.execute wrapping a pure alloc. Body does not touch
          // %argbuf. Barrier dep on %fill_loop must survive — downstream
          // loop reads %argbuf via the execute's token chain.
          %tk_scratch, %scratch = air.execute [%fill_loop] -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %tk_scratch) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %scratch[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}
