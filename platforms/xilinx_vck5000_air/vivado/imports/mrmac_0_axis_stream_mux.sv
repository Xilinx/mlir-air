// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

`timescale 1ps/1ps

(* DowngradeIPIdentifiedWarnings="yes" *)
module mrmac_0_axis_stream_mux
  (
   input        [3:0]       clk,
   input        [3:0]       reset,

   input        [2:0]       mux_sel,

   // S_AXIS
   input        [3:0]       s_tvalid,
   input        [7:0][63:0] s_tdata,
   input        [7:0][10:0] s_tkeep,
   input        [3:0]       s_tlast,
   output logic [3:0]       s_tready,

   // M_AXIS
   output logic [3:0]       m_tvalid,
   output logic [7:0][63:0] m_tdata,
   output logic [7:0][10:0] m_tkeep,
   output logic [3:0]       m_tlast,
   input        [3:0]       m_tready

   );

  ///////////////////////////////////////////////
  // Control logic and buffer
  ///////////////////////////////////////////////

  // decode m_tready and s_tvalid based on config
  logic [3:0] int_m_tready_c;
  logic [3:0] int_s_tvalid_c;

  always_comb begin
    case(mux_sel)
      3'b000: begin // 100G
        int_m_tready_c = {1'b0,{3{m_tready[0]}}};
        int_s_tvalid_c = {1'b0,{3{s_tvalid[0]}}};
      end

      3'b010: begin // 2x50G
        int_m_tready_c = {m_tready[2],m_tready[2],m_tready[0],m_tready[0]};
        int_s_tvalid_c = {s_tvalid[2],s_tvalid[2],s_tvalid[0],s_tvalid[0]};
      end

      3'b011: begin // 50G + 2x25G/10G
        int_m_tready_c = {m_tready[2],m_tready[2],m_tready[1],m_tready[0]};
        int_s_tvalid_c = {s_tvalid[2],s_tvalid[2],s_tvalid[1],s_tvalid[0]};
      end

      3'b100: begin // 10G/25G + 50G
        int_m_tready_c = {m_tready[3],m_tready[0],m_tready[0],m_tready[0]};
        int_s_tvalid_c = {s_tvalid[3],s_tvalid[0],s_tvalid[0],s_tvalid[0]};
      end

      3'b110: begin // 2x25G/10G + 50G
        int_m_tready_c = {m_tready[3],m_tready[2],m_tready[0],m_tready[0]};
        int_s_tvalid_c = {s_tvalid[3],s_tvalid[2],s_tvalid[0],s_tvalid[0]};
      end

      default: begin  // 4x25G/10G mode
        int_m_tready_c = m_tready;
        int_s_tvalid_c = s_tvalid;
      end
    endcase
  end

  generate
    genvar a;

    for (a=0; a<4; a=a+1) begin: axis_2stg_buff
      (* shreg_extract = "no" *) logic [1:0][63:0]                   tdata_buff0;
      (* shreg_extract = "no" *) logic [1:0][63:0]                   tdata_buff1;
      (* shreg_extract = "no" *) logic [1:0][10:0]                   tkeep_buff0;
      (* shreg_extract = "no" *) logic [1:0][10:0]                   tkeep_buff1;
      (* shreg_extract = "no" *) logic [1:0]                         tlast_buff;
      logic                               tready_buff;
      (* shreg_extract = "no" *) logic [1:0]                         tvalid_buff;

      always_ff @(posedge clk[a]) begin
        if(reset[a]) begin
          tdata_buff0         <= '0;
          tdata_buff1         <= '0;
          tkeep_buff0         <= '0;
          tkeep_buff1         <= '0;
          tlast_buff          <= '0;
          tready_buff         <= '0;
          tvalid_buff         <= '0;
        end
        else begin
          // slave side is ready when second stage of buffer
          // does not have data queued
          tready_buff         <= !tvalid_buff[1];

          // write data to buffer when slave side ready and
          // slave side valid
          if(int_s_tvalid_c[a] && !tvalid_buff[1]) begin
            if(tvalid_buff[0] && !int_m_tready_c[a]) begin
              // write second stage if master side has write pending
              tdata_buff0[1]  <= s_tdata[a*2];
              tkeep_buff0[1]  <= s_tkeep[a*2];
              tdata_buff1[1]  <= s_tdata[a*2 + 1];
              tkeep_buff1[1]  <= s_tkeep[a*2 + 1];
              tlast_buff[1]   <= s_tlast[a];
              tvalid_buff[1]  <= 1'b1;
            end
            else begin
              // else first stage gets new data
              tdata_buff0[0]  <= s_tdata[a*2];
              tkeep_buff0[0]  <= s_tkeep[a*2];
              tdata_buff1[0]  <= s_tdata[a*2 + 1];
              tkeep_buff1[0]  <= s_tkeep[a*2 + 1];
              tlast_buff[0]   <= s_tlast[a];
              tvalid_buff[1]  <= 1'b0;
              tvalid_buff[0]  <= 1'b1;
            end // else: !if(tvalid_buff[1])
          end // if (int_s_tvalid_c[a] & tready_buff)
          else if(int_m_tready_c[a]) begin
            if(tvalid_buff[1]) begin
              // shift buffer 2nd stage to 1st stage
              tvalid_buff[1]  <= 1'b0;
              tdata_buff0[0]  <= tdata_buff0[1];
              tkeep_buff0[0]  <= tkeep_buff0[1];
              tdata_buff1[0]  <= tdata_buff1[1];
              tkeep_buff1[0]  <= tkeep_buff1[1];
              tlast_buff[0]   <= tlast_buff[1];
              tvalid_buff[0]  <= 1'b1;
            end
            else begin
              // deassert master side valid
              // since no new data or queued data from
              // slave side
              tvalid_buff[0]  <= 1'b0;
            end // else: !if(tvalid_buff[1])
          end // else: !if(int_s_tvalid_c[a] & tready_buff)
        end // else: !if(reset)
      end // always_ff @ (posedge clk)

      // outputs
      always_comb begin
        s_tready[a]            = !tvalid_buff[1];
        m_tdata[a*2]           = tdata_buff0[0];
        m_tkeep[a*2]           = tkeep_buff0[0];
        m_tdata[a*2 + 1]       = tdata_buff1[0];
        m_tkeep[a*2 + 1]       = tkeep_buff1[0];
        m_tlast[a]             = tlast_buff[0];
        m_tvalid[a]            = tvalid_buff[0];
      end
    end // block: axis_2stg_buff

  endgenerate

endmodule


