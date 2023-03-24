/*
 * Copyright(C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Name:
 *  bsg_mem_1rw_sync_mask_write_byte.v
 *
 *  This module re-implements the same-named module from BaseJump STL in a way that properly maps
 *  to Xilinx BRAM with byte-enable writes.
 *
 *  Bytes are 8 bits wide. els_p must be 2 or greater and a power of 2.
 *
 */

module bsg_mem_1rw_sync_mask_write_byte
  #(parameter els_p=64
   ,parameter addr_width_lp = $clog2(els_p)
   ,parameter data_width_p=64
   ,parameter latch_last_read_p=0
   ,parameter write_mask_width_lp=(data_width_p/8)
   ,parameter enable_clock_gating_p=0
  )
  (input logic                            clk_i
  ,input logic                            reset_i
  ,input logic                            v_i
  ,input logic                            w_i
  ,input logic [addr_width_lp-1:0]        addr_i
  ,input logic [data_width_p-1:0]         data_i
  ,input logic [write_mask_width_lp-1:0]  write_mask_i
  ,output logic [data_width_p-1:0]        data_o
  );

  wire unused = reset_i;

  // RAM
  localparam byte_width_lp = 8;
  (* ram_style = "block" *) logic [(write_mask_width_lp*byte_width_lp)-1:0] mem [els_p-1:0];

  // synchronous write
  logic [write_mask_width_lp-1:0] we;
  assign we = write_mask_i & {write_mask_width_lp{w_i}};

  generate genvar i;
  for (i = 0; i < write_mask_width_lp; i++) begin
    always @(posedge clk_i) begin
      if (we[i]) begin
        mem[addr_i][((i+1)*byte_width_lp)-1:(i*byte_width_lp)] <= data_i[((i+1)*byte_width_lp)-1:(i*byte_width_lp)];
      end
    end
  end
  endgenerate

  // synchronous read
  wire [data_width_p-1:0] data_li = mem[addr_i];

  logic [data_width_p-1:0] data_r;
  assign data_o = data_r;
  if (latch_last_read_p) begin: llr
    // if LLR, only capture read on read_en
    wire read_en = v_i & ~w_i;
    always_ff @(posedge clk_i) begin
      if (read_en) data_r <= data_li;
    end
  end else begin: no_llr
    // if no LLR, capture "read" every cycle, regardless of valid read occurring
    always_ff @(posedge clk_i) begin
      data_r <= data_li;
    end
  end

endmodule

