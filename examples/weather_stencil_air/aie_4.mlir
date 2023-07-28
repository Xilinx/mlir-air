// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   

// SPDX-License-Identifier: MIT


module @hdiff_bundle_4 {
//---Generating B-block 0---*-
//---col 2---*-
  %tile2_1 = AIE.tile(2, 1)
  %tile2_2 = AIE.tile(2, 2)
  %tile2_3 = AIE.tile(2, 3)
  %tile2_4 = AIE.tile(2, 4)
//---col 3---*-
  %tile3_1 = AIE.tile(3, 1)
  %tile3_2 = AIE.tile(3, 2)
  %tile3_3 = AIE.tile(3, 3)
  %tile3_4 = AIE.tile(3, 4)
//---col 4---*-
  %tile4_1 = AIE.tile(4, 1)
  %tile4_2 = AIE.tile(4, 2)
  %tile4_3 = AIE.tile(4, 3)
  %tile4_4 = AIE.tile(4, 4)

//---Generating B-block 1---*-
//---col 2---*-
  %tile2_5 = AIE.tile(2, 5)
  %tile2_6 = AIE.tile(2, 6)
  %tile2_7 = AIE.tile(2, 7)
  %tile2_8 = AIE.tile(2, 8)
//---col 3---*-
  %tile3_5 = AIE.tile(3, 5)
  %tile3_6 = AIE.tile(3, 6)
  %tile3_7 = AIE.tile(3, 7)
  %tile3_8 = AIE.tile(3, 8)
//---col 4---*-
  %tile4_5 = AIE.tile(4, 5)
  %tile4_6 = AIE.tile(4, 6)
  %tile4_7 = AIE.tile(4, 7)
  %tile4_8 = AIE.tile(4, 8)

//---Generating B-block 2---*-
//---col 2---*-
  %tile5_1 = AIE.tile(5, 1)
  %tile5_2 = AIE.tile(5, 2)
  %tile5_3 = AIE.tile(5, 3)
  %tile5_4 = AIE.tile(5, 4)
//---col 3---*-
  %tile6_1 = AIE.tile(6, 1)
  %tile6_2 = AIE.tile(6, 2)
  %tile6_3 = AIE.tile(6, 3)
  %tile6_4 = AIE.tile(6, 4)
//---col 4---*-
  %tile7_1 = AIE.tile(7, 1)
  %tile7_2 = AIE.tile(7, 2)
  %tile7_3 = AIE.tile(7, 3)
  %tile7_4 = AIE.tile(7, 4)

//---Generating B-block 3---*-
//---col 2---*-
  %tile5_5 = AIE.tile(5, 5)
  %tile5_6 = AIE.tile(5, 6)
  %tile5_7 = AIE.tile(5, 7)
  %tile5_8 = AIE.tile(5, 8)
//---col 3---*-
  %tile6_5 = AIE.tile(6, 5)
  %tile6_6 = AIE.tile(6, 6)
  %tile6_7 = AIE.tile(6, 7)
  %tile6_8 = AIE.tile(6, 8)
//---col 4---*-
  %tile7_5 = AIE.tile(7, 5)
  %tile7_6 = AIE.tile(7, 6)
  %tile7_7 = AIE.tile(7, 7)
  %tile7_8 = AIE.tile(7, 8)

//---NOC Tile 2---*-
  %tile2_0 = AIE.tile(2, 0)
//---NOC Tile 3---*-
  %tile3_0 = AIE.tile(3, 0)

//---Generating B0 buffers---*-
  %block_0_buf_in_shim_2 = AIE.objectFifo.createObjectFifo(%tile2_0,{%tile2_1,%tile3_1,%tile2_2,%tile3_2,%tile2_3,%tile3_3,%tile2_4,%tile3_4},9 :i32) { sym_name = "block_0_buf_in_shim_2" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_0_buf_row_1_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_1,{%tile3_1},5:i32){ sym_name ="block_0_buf_row_1_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_1_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_1,{%tile4_1},6:i32) { sym_name ="block_0_buf_row_1_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_1_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_1,{%tile4_2},2:i32) { sym_name ="block_0_buf_row_1_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_2_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_2,{%tile3_2},5:i32){ sym_name ="block_0_buf_row_2_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_2_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_2,{%tile4_2},6:i32) { sym_name ="block_0_buf_row_2_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_out_shim_2= AIE.objectFifo.createObjectFifo(%tile4_2,{%tile2_0},5:i32){ sym_name ="block_0_buf_out_shim_2"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_0_buf_row_3_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_3,{%tile3_3},5:i32){ sym_name ="block_0_buf_row_3_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_3_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_3,{%tile4_3},6:i32) { sym_name ="block_0_buf_row_3_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_3_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_3,{%tile4_2},2:i32) { sym_name ="block_0_buf_row_3_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_4_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_4,{%tile3_4},5:i32){ sym_name ="block_0_buf_row_4_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_0_buf_row_4_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_4,{%tile4_4},6:i32) { sym_name ="block_0_buf_row_4_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_0_buf_row_4_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_4,{%tile4_2},2:i32) { sym_name ="block_0_buf_row_4_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
//---Generating B1 buffers---*-
  %block_1_buf_in_shim_2 = AIE.objectFifo.createObjectFifo(%tile2_0,{%tile2_5,%tile3_5,%tile2_6,%tile3_6,%tile2_7,%tile3_7,%tile2_8,%tile3_8},9 :i32) { sym_name = "block_1_buf_in_shim_2" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_1_buf_row_5_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_5,{%tile3_5},5:i32){ sym_name ="block_1_buf_row_5_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_5_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_5,{%tile4_5},6:i32) { sym_name ="block_1_buf_row_5_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_5_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_5,{%tile4_6},2:i32) { sym_name ="block_1_buf_row_5_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_6_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_6,{%tile3_6},5:i32){ sym_name ="block_1_buf_row_6_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_6_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_6,{%tile4_6},6:i32) { sym_name ="block_1_buf_row_6_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_out_shim_2= AIE.objectFifo.createObjectFifo(%tile4_6,{%tile2_0},5:i32){ sym_name ="block_1_buf_out_shim_2"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_1_buf_row_7_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_7,{%tile3_7},5:i32){ sym_name ="block_1_buf_row_7_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_7_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_7,{%tile4_7},6:i32) { sym_name ="block_1_buf_row_7_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_7_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_7,{%tile4_6},2:i32) { sym_name ="block_1_buf_row_7_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_8_inter_lap= AIE.objectFifo.createObjectFifo(%tile2_8,{%tile3_8},5:i32){ sym_name ="block_1_buf_row_8_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_1_buf_row_8_inter_flx1= AIE.objectFifo.createObjectFifo(%tile3_8,{%tile4_8},6:i32) { sym_name ="block_1_buf_row_8_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_1_buf_row_8_out_flx2= AIE.objectFifo.createObjectFifo(%tile4_8,{%tile4_6},2:i32) { sym_name ="block_1_buf_row_8_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
//---Generating B2 buffers---*-
  %block_2_buf_in_shim_3 = AIE.objectFifo.createObjectFifo(%tile3_0,{%tile5_1,%tile6_1,%tile5_2,%tile6_2,%tile5_3,%tile6_3,%tile5_4,%tile6_4},9 :i32) { sym_name = "block_2_buf_in_shim_3" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_2_buf_row_1_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_1,{%tile6_1},5:i32){ sym_name ="block_2_buf_row_1_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_1_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_1,{%tile7_1},6:i32) { sym_name ="block_2_buf_row_1_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_1_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_1,{%tile7_2},2:i32) { sym_name ="block_2_buf_row_1_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_2_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_2,{%tile6_2},5:i32){ sym_name ="block_2_buf_row_2_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_2_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_2,{%tile7_2},6:i32) { sym_name ="block_2_buf_row_2_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_out_shim_3= AIE.objectFifo.createObjectFifo(%tile7_2,{%tile3_0},5:i32){ sym_name ="block_2_buf_out_shim_3"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_2_buf_row_3_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_3,{%tile6_3},5:i32){ sym_name ="block_2_buf_row_3_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_3_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_3,{%tile7_3},6:i32) { sym_name ="block_2_buf_row_3_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_3_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_3,{%tile7_2},2:i32) { sym_name ="block_2_buf_row_3_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_4_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_4,{%tile6_4},5:i32){ sym_name ="block_2_buf_row_4_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_2_buf_row_4_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_4,{%tile7_4},6:i32) { sym_name ="block_2_buf_row_4_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_2_buf_row_4_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_4,{%tile7_2},2:i32) { sym_name ="block_2_buf_row_4_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
//---Generating B3 buffers---*-
  %block_3_buf_in_shim_3 = AIE.objectFifo.createObjectFifo(%tile3_0,{%tile5_5,%tile6_5,%tile5_6,%tile6_6,%tile5_7,%tile6_7,%tile5_8,%tile6_8},9 :i32) { sym_name = "block_3_buf_in_shim_3" } : !AIE.objectFifo<memref<256xi32>> //B block input
  %block_3_buf_row_5_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_5,{%tile6_5},5:i32){ sym_name ="block_3_buf_row_5_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_5_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_5,{%tile7_5},6:i32) { sym_name ="block_3_buf_row_5_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_5_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_5,{%tile7_6},2:i32) { sym_name ="block_3_buf_row_5_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_6_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_6,{%tile6_6},5:i32){ sym_name ="block_3_buf_row_6_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_6_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_6,{%tile7_6},6:i32) { sym_name ="block_3_buf_row_6_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_out_shim_3= AIE.objectFifo.createObjectFifo(%tile7_6,{%tile3_0},5:i32){ sym_name ="block_3_buf_out_shim_3"} : !AIE.objectFifo<memref<256xi32>> //B block output
  %block_3_buf_row_7_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_7,{%tile6_7},5:i32){ sym_name ="block_3_buf_row_7_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_7_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_7,{%tile7_7},6:i32) { sym_name ="block_3_buf_row_7_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_7_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_7,{%tile7_6},2:i32) { sym_name ="block_3_buf_row_7_out_flx2"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_8_inter_lap= AIE.objectFifo.createObjectFifo(%tile5_8,{%tile6_8},5:i32){ sym_name ="block_3_buf_row_8_inter_lap"} : !AIE.objectFifo<memref<256xi32>>
  %block_3_buf_row_8_inter_flx1= AIE.objectFifo.createObjectFifo(%tile6_8,{%tile7_8},6:i32) { sym_name ="block_3_buf_row_8_inter_flx1"} : !AIE.objectFifo<memref<512xi32>>
  %block_3_buf_row_8_out_flx2= AIE.objectFifo.createObjectFifo(%tile7_8,{%tile7_6},2:i32) { sym_name ="block_3_buf_row_8_out_flx2"} : !AIE.objectFifo<memref<256xi32>>

  func.func private @hdiff_lap(%AL: memref<256xi32>,%BL: memref<256xi32>, %CL:  memref<256xi32>, %DL: memref<256xi32>, %EL:  memref<256xi32>,  %OLL1: memref<256xi32>,  %OLL2: memref<256xi32>,  %OLL3: memref<256xi32>,  %OLL4: memref<256xi32>) -> ()
  func.func private @hdiff_flux1(%AF: memref<256xi32>,%BF: memref<256xi32>, %CF:  memref<256xi32>,   %OLF1: memref<256xi32>,  %OLF2: memref<256xi32>,  %OLF3: memref<256xi32>,  %OLF4: memref<256xi32>,  %OFI1: memref<512xi32>,  %OFI2: memref<512xi32>,  %OFI3: memref<512xi32>,  %OFI4: memref<512xi32>,  %OFI5: memref<512xi32>) -> ()
  func.func private @hdiff_flux2( %Inter1: memref<512xi32>,%Inter2: memref<512xi32>, %Inter3: memref<512xi32>,%Inter4: memref<512xi32>,%Inter5: memref<512xi32>,  %Out: memref<256xi32>) -> ()

  %block_0_core2_1 = AIE.core(%tile2_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core3_1 = AIE.core(%tile3_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core4_1 = AIE.core(%tile4_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_1_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core2_2 = AIE.core(%tile2_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core3_2 = AIE.core(%tile3_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_0_core4_2 = AIE.core(%tile4_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_out_shim_2: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_2_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_0_buf_row_1_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_out_shim_2:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core2_3 = AIE.core(%tile2_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core3_3 = AIE.core(%tile3_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core4_3 = AIE.core(%tile4_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_3_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_3_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_0_core2_4 = AIE.core(%tile2_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_0_core3_4 = AIE.core(%tile3_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_0_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_0_core4_4 = AIE.core(%tile4_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_0_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_0_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_0_buf_row_4_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_0_buf_row_4_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core2_5 = AIE.core(%tile2_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core3_5 = AIE.core(%tile3_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core4_5 = AIE.core(%tile4_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_5_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core2_6 = AIE.core(%tile2_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core3_6 = AIE.core(%tile3_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_1_core4_6 = AIE.core(%tile4_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_out_shim_2: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_6_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_5_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_out_shim_2:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core2_7 = AIE.core(%tile2_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core3_7 = AIE.core(%tile3_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core4_7 = AIE.core(%tile4_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_7_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_7_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_1_core2_8 = AIE.core(%tile2_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_1_core3_8 = AIE.core(%tile3_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_1_buf_in_shim_2: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_1_core4_8 = AIE.core(%tile4_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_1_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_1_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_1_buf_row_8_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_1_buf_row_8_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core5_1 = AIE.core(%tile5_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core6_1 = AIE.core(%tile6_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core7_1 = AIE.core(%tile7_1) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_1_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core5_2 = AIE.core(%tile5_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core6_2 = AIE.core(%tile6_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_2_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_2_core7_2 = AIE.core(%tile7_2) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_2_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_out_shim_3: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_1_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_2_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_1_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_out_shim_3:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core5_3 = AIE.core(%tile5_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core6_3 = AIE.core(%tile6_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core7_3 = AIE.core(%tile7_3) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_3_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_3_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_3_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_3_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_2_core5_4 = AIE.core(%tile5_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_2_core6_4 = AIE.core(%tile6_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_2_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_2_core7_4 = AIE.core(%tile7_4) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_2_buf_row_4_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_2_buf_row_4_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_2_buf_row_4_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_2_buf_row_4_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core5_5 = AIE.core(%tile5_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core6_5 = AIE.core(%tile6_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core7_5 = AIE.core(%tile7_5) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_5_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core5_6 = AIE.core(%tile5_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core6_6 = AIE.core(%tile6_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_6_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  // Gathering Tile
  %block_3_core7_6 = AIE.core(%tile7_6) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_6_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_out_shim_3: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
  // Acquire all elements and add in order
      %obj_out_flux_element0 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_flux_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
  // Acquiring outputs from other flux
      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_5_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux3 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from3 = AIE.objectFifo.subview.access %obj_out_subview_flux3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux4 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %final_out_from4 = AIE.objectFifo.subview.access %obj_out_subview_flux4[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

  // Ordering and copying data to gather tile (src-->dst)
      memref.copy %final_out_from1 , %obj_out_flux_element0 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from3 , %obj_out_flux_element2 : memref<256xi32> to memref<256xi32>
      memref.copy %final_out_from4 , %obj_out_flux_element3 : memref<256xi32> to memref<256xi32>
      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_6_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_5_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_out_flx2:!AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_out_shim_3:!AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core5_7 = AIE.core(%tile5_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core6_7 = AIE.core(%tile6_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core7_7 = AIE.core(%tile7_7) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_7_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_7_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_7_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_7_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

  %block_3_core5_8 = AIE.core(%tile5_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row0 = AIE.objectFifo.subview.access %obj_in_subview[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row4 = AIE.objectFifo.subview.access %obj_in_subview[7] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_lap(%row0,%row1,%row2,%row3,%row4,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>,  memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 4)
    AIE.end
  } { link_with="hdiff_lap.o" }

  %block_3_core6_8 = AIE.core(%tile6_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_in_subview = AIE.objectFifo.acquire<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 8) : !AIE.objectFifoSubview<memref<256xi32>>
      %row1 = AIE.objectFifo.subview.access %obj_in_subview[4] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row2 = AIE.objectFifo.subview.access %obj_in_subview[5] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %row3 = AIE.objectFifo.subview.access %obj_in_subview[6] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_lap = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_lap1 = AIE.objectFifo.subview.access %obj_out_subview_lap[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap2 = AIE.objectFifo.subview.access %obj_out_subview_lap[1] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap3 = AIE.objectFifo.subview.access %obj_out_subview_lap[2] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>
      %obj_out_lap4 = AIE.objectFifo.subview.access %obj_out_subview_lap[3] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      %obj_out_subview_flux1 = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_out_flux_inter1 = AIE.objectFifo.subview.access %obj_out_subview_flux1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter2 = AIE.objectFifo.subview.access %obj_out_subview_flux1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter3 = AIE.objectFifo.subview.access %obj_out_subview_flux1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter4 = AIE.objectFifo.subview.access %obj_out_subview_flux1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_out_flux_inter5 = AIE.objectFifo.subview.access %obj_out_subview_flux1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      func.call @hdiff_flux1(%row1,%row2,%row3,%obj_out_lap1,%obj_out_lap2,%obj_out_lap3,%obj_out_lap4, %obj_out_flux_inter1 , %obj_out_flux_inter2, %obj_out_flux_inter3, %obj_out_flux_inter4, %obj_out_flux_inter5) : (memref<256xi32>,memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>, memref<256xi32>,  memref<256xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>,  memref<512xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_inter_lap: !AIE.objectFifo<memref<256xi32>>, 4)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.objectFifo.release<Consume>(%block_3_buf_in_shim_3: !AIE.objectFifo<memref<256xi32>>, 7)
    AIE.end
  } { link_with="hdiff_flux1.o" }

  %block_3_core7_8 = AIE.core(%tile7_8) {
    %lb = arith.constant 0 : index
    %ub = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {  
      %obj_out_subview_flux_inter1 = AIE.objectFifo.acquire<Consume>(%block_3_buf_row_8_inter_flx1: !AIE.objectFifo<memref<512xi32>>, 5): !AIE.objectFifoSubview<memref<512xi32>>
      %obj_flux_inter_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element2 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[1] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element3 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[2] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element4 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[3] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
      %obj_flux_inter_element5 = AIE.objectFifo.subview.access %obj_out_subview_flux_inter1[4] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>

      %obj_out_subview_flux = AIE.objectFifo.acquire<Produce>(%block_3_buf_row_8_out_flx2: !AIE.objectFifo<memref<256xi32>>, 1): !AIE.objectFifoSubview<memref<256xi32>>
      %obj_out_flux_element1 = AIE.objectFifo.subview.access %obj_out_subview_flux[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

      func.call @hdiff_flux2(%obj_flux_inter_element1, %obj_flux_inter_element2,%obj_flux_inter_element3, %obj_flux_inter_element4, %obj_flux_inter_element5,  %obj_out_flux_element1 ) : ( memref<512xi32>,  memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<512xi32>, memref<256xi32>) -> ()

      AIE.objectFifo.release<Consume>(%block_3_buf_row_8_inter_flx1 :!AIE.objectFifo<memref<512xi32>>, 5)
      AIE.objectFifo.release<Produce>(%block_3_buf_row_8_out_flx2 :!AIE.objectFifo<memref<256xi32>>, 1)
    }
    AIE.end
  } { link_with="hdiff_flux2.o" }

}
