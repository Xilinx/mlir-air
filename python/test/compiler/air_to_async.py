# RUN: %PYTHON %s | FileCheck %s

from textwrap import dedent

from air.dialects.air import register_dialect
from air.mlir.ir import Context, Module, Location
from air.mlir.passmanager import PassManager


# this has a side effect of registering the air passes


def run(f):
    with Context() as ctx:
        register_dialect(ctx)
        print("\nTEST:", f.__name__)
        f()
    return f


src = dedent(
    """\
air.channel @channel_0 [1]
func.func @forward(%arg0 : memref<16x16xi32>, %arg1 : memref<16x16xi32>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  air.channel.put @channel_0[] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  air.channel.get @channel_0[] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
  return
}
"""
)


# CHECK-LABEL: TEST: test_smoke
# CHECK: module {
# CHECK:   air.channel @channel_0 [1]
# CHECK:   func.func @forward(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) {
# CHECK:     %c0 = arith.constant 0 : index
# CHECK:     %c1 = arith.constant 1 : index
# CHECK:     %c8 = arith.constant 8 : index
# CHECK:     %c16 = arith.constant 16 : index
# CHECK:     air.channel.put  @channel_0[] (%arg0[%c0, %c0] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
# CHECK:     air.channel.get  @channel_0[] (%arg1[%c8, %c8] [%c8, %c8] [%c16, %c1]) : (memref<16x16xi32>)
# CHECK:     return
# CHECK:   }
# CHECK: }
@run
def test_smoke():
    with Location.unknown():
        module = Module.parse(src)

    print(module)


# CHECK-LABEL: TEST: test_lower_to_async
# CHECK: module {
# CHECK:   memref.global "private" @channel_0 : memref<i64> = dense<0>
# CHECK:   func.func @forward(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) attributes {llvm.emit_c_interface} {
# CHECK:     %c0 = arith.constant 0 : index
# CHECK:     %c1 = arith.constant 1 : index
# CHECK:     %c8 = arith.constant 8 : index
# CHECK:     %c16 = arith.constant 16 : index
# CHECK:     %0 = memref.get_global @channel_0 : memref<i64>
# CHECK:     %1 = builtin.unrealized_conversion_cast %0 : memref<i64> to memref<i64>
# CHECK:     %2 = builtin.unrealized_conversion_cast %arg0 : memref<16x16xi32> to memref<?x?xi32>
# CHECK:     call @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%1, %2, %c0, %c0, %c8, %c8, %c16, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
# CHECK:     %3 = memref.get_global @channel_0 : memref<i64>
# CHECK:     %4 = builtin.unrealized_conversion_cast %3 : memref<i64> to memref<i64>
# CHECK:     %5 = builtin.unrealized_conversion_cast %arg1 : memref<16x16xi32> to memref<?x?xi32>
# CHECK:     call @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(%4, %5, %c8, %c8, %c8, %c8, %c16, %c1) : (memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) -> ()
# CHECK:     return
# CHECK:   }
# CHECK:   func.func private @air_channel_put_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
# CHECK:   func.func private @air_channel_get_M0I64_M0D2I32_I64_I64_I64_I64_I64_I64(memref<i64>, memref<?x?xi32>, index, index, index, index, index, index) attributes {llvm.emit_c_interface}
# CHECK: }
@run
def test_lower_to_async():
    with Location.unknown():
        module = Module.parse(src)

    PassManager.parse("builtin.module(buffer-results-to-out-params,air-to-async)").run(
        module
    )
    print(module)
