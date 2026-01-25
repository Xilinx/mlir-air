//===- airrt_canonicalize.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize %s | FileCheck %s

// CHECK-LABEL: wait_all_0
// CHECK-NEXT: return
func.func @wait_all_0() -> () {
  %0 = airrt.wait_all : !airrt.event
  airrt.wait_all %0
  return
}

// CHECK-LABEL: wait_all_1
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event, %[[E2:.*]]: !airrt.event) -> !airrt.event {
// CHECK-NEXT:   %[[E4:.*]] = airrt.wait_all %[[E0]], %[[E1]], %[[E2]] : !airrt.event
// CHECK-NEXT: return %[[E4]]
func.func @wait_all_1(%e0 : !airrt.event, %e1 : !airrt.event, %e2 : !airrt.event) -> (!airrt.event) {
  %1 = airrt.wait_all %e0 : !airrt.event
  %2 = airrt.wait_all %e1 : !airrt.event
  %3 = airrt.wait_all %e2 : !airrt.event
  %4 = airrt.wait_all %1 : !airrt.event
  %5 = airrt.wait_all %4, %2 : !airrt.event
  %6 = airrt.wait_all %5, %3 : !airrt.event
  %7 = airrt.wait_all %6 : !airrt.event
  return %7 : !airrt.event
}

// CHECK-LABEL: alloc_dealloc
// CHECK-NEXT: return
func.func @alloc_dealloc() {
  %0 = airrt.alloc : memref<1x4x4x16xi32, 1>
  airrt.dealloc %0 : memref<1x4x4x16xi32, 1>
  return
}

// Test that air.segment_end attribute is preserved during wait_all canonicalization
// CHECK-LABEL: wait_all_segment_end_preserved
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event)
// CHECK-NEXT:   airrt.wait_all {{.*}} {air.segment_end}
// CHECK-NEXT: return
func.func @wait_all_segment_end_preserved(%e0 : !airrt.event, %e1 : !airrt.event) {
  %0 = airrt.wait_all %e0 : !airrt.event
  airrt.wait_all %0, %e1 {air.segment_end}
  return
}

// Test that air.launch_end attribute is preserved during wait_all canonicalization
// CHECK-LABEL: wait_all_launch_end_preserved
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event)
// CHECK-NEXT:   airrt.wait_all {{.*}} {air.launch_end}
// CHECK-NEXT: return
func.func @wait_all_launch_end_preserved(%e0 : !airrt.event, %e1 : !airrt.event) {
  %0 = airrt.wait_all %e0 : !airrt.event
  airrt.wait_all %0, %e1 {air.launch_end}
  return
}

// Test that air.segment_end attribute is preserved when async wait_all ops are folded
// CHECK-LABEL: wait_all_async_segment_end_preserved
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event)
// CHECK-NEXT:   %[[R:.*]] = airrt.wait_all {{.*}} {air.segment_end} : !airrt.event
// CHECK-NEXT: return %[[R]]
func.func @wait_all_async_segment_end_preserved(%e0 : !airrt.event, %e1 : !airrt.event) -> !airrt.event {
  %0 = airrt.wait_all %e0 : !airrt.event
  %1 = airrt.wait_all %0, %e1 {air.segment_end} : !airrt.event
  return %1 : !airrt.event
}

// Test that air.launch_end attribute is preserved when async wait_all ops are folded
// CHECK-LABEL: wait_all_async_launch_end_preserved
// CHECK-SAME: (%[[E0:.*]]: !airrt.event, %[[E1:.*]]: !airrt.event)
// CHECK-NEXT:   %[[R:.*]] = airrt.wait_all {{.*}} {air.launch_end} : !airrt.event
// CHECK-NEXT: return %[[R]]
func.func @wait_all_async_launch_end_preserved(%e0 : !airrt.event, %e1 : !airrt.event) -> !airrt.event {
  %0 = airrt.wait_all %e0 : !airrt.event
  %1 = airrt.wait_all %0, %e1 {air.launch_end} : !airrt.event
  return %1 : !airrt.event
}

// Test that duplicate segment_load ops with the same name are folded
// CHECK-LABEL: segment_load_duplicate
// CHECK: %[[P:.*]] = airrt.segment_load "test_segment"
// CHECK-NOT: airrt.segment_load "test_segment"
// CHECK: return %[[P]]
func.func @segment_load_duplicate() -> i64 {
  %p0 = airrt.segment_load "test_segment" : i64
  %p1 = airrt.segment_load "test_segment" : i64
  return %p1 : i64
}

// Test that different segment_load ops are NOT folded
// CHECK-LABEL: segment_load_different
// CHECK: airrt.segment_load "segment_a"
// CHECK: airrt.segment_load "segment_b"
func.func @segment_load_different() -> (i64, i64) {
  %p0 = airrt.segment_load "segment_a" : i64
  %p1 = airrt.segment_load "segment_b" : i64
  return %p0, %p1 : i64, i64
}

// Test that multiple duplicate segment_load ops are all folded to the first one
// CHECK-LABEL: segment_load_multiple_duplicates
// CHECK: %[[P:.*]] = airrt.segment_load "test_segment"
// CHECK-NOT: airrt.segment_load "test_segment"
// CHECK: return %[[P]], %[[P]], %[[P]]
func.func @segment_load_multiple_duplicates() -> (i64, i64, i64) {
  %p0 = airrt.segment_load "test_segment" : i64
  %p1 = airrt.segment_load "test_segment" : i64
  %p2 = airrt.segment_load "test_segment" : i64
  return %p0, %p1, %p2 : i64, i64, i64
}

// Test that duplicate herd_load ops with the same name and segment_name are folded
// CHECK-LABEL: herd_load_duplicate
// CHECK: %[[H:.*]] = airrt.herd_load "test_herd" () {segment_name = "test_segment"}
// CHECK-NOT: airrt.herd_load "test_herd"
// CHECK: return %[[H]]
func.func @herd_load_duplicate() -> i64 {
  %h0 = airrt.herd_load "test_herd" () {segment_name = "test_segment"} : () -> i64
  %h1 = airrt.herd_load "test_herd" () {segment_name = "test_segment"} : () -> i64
  return %h1 : i64
}

// Test that different herd_load ops (different herd names) are NOT folded
// CHECK-LABEL: herd_load_different_herd
// CHECK: airrt.herd_load "herd_a"
// CHECK: airrt.herd_load "herd_b"
func.func @herd_load_different_herd() -> (i64, i64) {
  %h0 = airrt.herd_load "herd_a" () {segment_name = "test_segment"} : () -> i64
  %h1 = airrt.herd_load "herd_b" () {segment_name = "test_segment"} : () -> i64
  return %h0, %h1 : i64, i64
}

// Test that different herd_load ops (different segment names) are NOT folded
// CHECK-LABEL: herd_load_different_segment
// CHECK: airrt.herd_load "test_herd" () {segment_name = "segment_a"}
// CHECK: airrt.herd_load "test_herd" () {segment_name = "segment_b"}
func.func @herd_load_different_segment() -> (i64, i64) {
  %h0 = airrt.herd_load "test_herd" () {segment_name = "segment_a"} : () -> i64
  %h1 = airrt.herd_load "test_herd" () {segment_name = "segment_b"} : () -> i64
  return %h0, %h1 : i64, i64
}

// Test that multiple duplicate herd_load ops are all folded to the first one
// CHECK-LABEL: herd_load_multiple_duplicates
// CHECK: %[[H:.*]] = airrt.herd_load "test_herd" () {segment_name = "test_segment"}
// CHECK-NOT: airrt.herd_load "test_herd"
// CHECK: return %[[H]], %[[H]], %[[H]]
func.func @herd_load_multiple_duplicates() -> (i64, i64, i64) {
  %h0 = airrt.herd_load "test_herd" () {segment_name = "test_segment"} : () -> i64
  %h1 = airrt.herd_load "test_herd" () {segment_name = "test_segment"} : () -> i64
  %h2 = airrt.herd_load "test_herd" () {segment_name = "test_segment"} : () -> i64
  return %h0, %h1, %h2 : i64, i64, i64
}
