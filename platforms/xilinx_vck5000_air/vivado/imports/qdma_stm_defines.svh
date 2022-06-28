//-----------------------------------------------------------------------------
//
// (c) Copyright 2012-2012 Xilinx, Inc. All rights reserved.
//
// This file contains confidential and proprietary information
// of Xilinx, Inc. and is protected under U.S. and
// international copyright and other intellectual property
// laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any
// rights to the materials distributed herewith. Except as
// otherwise provided in a valid license issued to you by
// Xilinx, and to the maximum extent permitted by applicable
// law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
// WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
// AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
// BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
// INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
// (2) Xilinx shall not be liable (whether in contract or tort,
// including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature
// related to, arising under or in connection with these
// materials, including for any direct, or any indirect,
// special, incidental, or consequential loss or damage
// (including loss of data, profits, goodwill, or any type of
// loss or damage suffered as a result of any action brought
// by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the
// possibility of the same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-
// safe, or for use in any application requiring fail-safe
// performance, such as life-support or safety devices or
// systems, Class III medical devices, nuclear facilities,
// applications related to the deployment of airbags, or any
// other applications that could lead to death, personal
// injury, or severe property or environmental damage
// (individually and collectively, "Critical
// Applications"). Customer assumes the sole risk and
// liability of any use of Xilinx products in Critical
// Applications, subject only to applicable laws and
// regulations governing limitations on product liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
// PART OF THIS FILE AT ALL TIMES.
//
//-----------------------------------------------------------------------------
//
// Project    : The Xilinx PCI Express DMA 
// File       : qdma_stm_defines.svh
// Version    : 5.0
//-----------------------------------------------------------------------------

`ifndef QDMA_STM_DEFINES_SVH
    `define QDMA_STM_DEFINES_SVH
 
  typedef logic [511:0]                           mdma_int_tdata_exdes_t;
  typedef logic [10:0]                            mdma_qid_exdes_t;
  typedef logic [15:0]                            mdma_dma_buf_len_exdes_t;

    typedef struct packed {
        logic [15:0]            pld_len;
        logic                   req_wrb;
        logic                   eot;
        logic                   zero_cdh;
        logic [5:0]             rsv1;
        logic [3:0]             num_cdh;
        logic [2:0]             num_gl;
    } h2c_stub_tmh_t;

    typedef struct packed {
        logic [95:0]            cdh_data;
        h2c_stub_tmh_t          tmh;
    } h2c_stub_cdh_slot_0_t;

    typedef struct packed {
        logic [127:0]           cdh_slot_2;
        logic [127:0]           cdh_slot_1;
        h2c_stub_cdh_slot_0_t   cdh_slot_0;
        logic [63:0]            rsv4;
        logic [15:0]            rsv3;
        logic [15:0]            tdest;
        logic [9:0]             rsv2;
        logic [5:0]             flow_id;
        logic [4:0]             rsv1;
        logic [10:0]            qid;
    } h2c_stub_hdr_beat_t;

    typedef struct packed {
        logic [1:0]             rsv2;
        logic                   usr_int;
        logic                   eot;
        logic [3:0]             rsv1;
        logic [15:0]            pkt_len;
    } c2h_stub_tmh_t;

    typedef struct packed {
        logic [127:0]           cmp_data_1;
        logic [103:0]           cmp_data_0;
        c2h_stub_tmh_t          tmh;
    } c2h_stub_cmp_t;

    typedef struct packed {
        logic [127:0]           rsv5;
        c2h_stub_cmp_t          cmp;
        logic [63:0]            rsv4;
        logic [15:0]            rsv3;
        logic [15:0]            tdest;
        logic [9:0]             rsv2;
        logic [5:0]             flow_id;
        logic [4:0]             rsv1;
        logic [10:0]            qid;
    } c2h_stub_hdr_beat_t;

    typedef struct packed {
        logic [107:0]           usr_data;
        logic [15:0]            len;        //[19:4]
        logic                   desc_used;  //[3]
        logic [2:0]             rsv;        //[2:0]
    } c2h_stub_std_cmp_ent_t;

 typedef enum logic [1:0]    {
            WRB_DSC_8B_EXDES=0, WRB_DSC_16B_EXDES=1, WRB_DSC_32B_EXDES=2, WRB_DSC_UNKOWN_EXDES=3
        } mdma_c2h_wrb_type_exdes_e;


    typedef struct packed {
        c2h_stub_std_cmp_ent_t  cmp_ent;
        mdma_c2h_wrb_type_exdes_e     cmp_size;
        logic [$bits(c2h_stub_std_cmp_ent_t)/32-1:0] dpar;
    } c2h_stub_std_cmp_t;

    typedef struct packed {
        logic               user_trig;
        logic [2:0]         error_idx;
        logic [2:0]         color_idx;
        logic [2:0]         port_id;
        logic [15:0]        wait_pld_pkt_id;
        logic [1:0]         cmpt_type;
        logic               marker;        // Make sure the pipeline is completely flushed
        mdma_qid_exdes_t          qid;
    } c2h_stub_std_cmp_ctrl_t;

   typedef struct packed {
        logic                       has_cmpt;
        logic [2:0]                 port_id;
        logic                       marker;        // Make sure the pipeline is completely flushed
        mdma_qid_exdes_t            qid;
        mdma_dma_buf_len_exdes_t    len;
    } mdma_c2h_axis_ctrl_exdes_t;

    typedef struct packed {
        mdma_int_tdata_exdes_t    tdata;
        logic [$bits(mdma_int_tdata_exdes_t)/8 - 1 :0]   par; 
    } mdma_c2h_axis_data_exdes_t;
    
   typedef struct packed {
        logic                                           zero_byte;  //[53]
        logic [5:0]                                     mty;        //[52:47]
        logic [31:0]                                    mdata;      //[46:15]
        logic                                           err;        //[14]
        logic [2:0]                                     port_id;    //[13:11]
        mdma_qid_exdes_t                                qid;        //[10:0]
    } mdma_h2c_axis_tuser_exdes_t;
 
   
 

`define XPREG_NORESET_EXDES(clk,q,d)			    \
    always @(posedge clk)			    \
    begin					    \
         `ifdef FOURVALCLKPROP			    \
	    q <= #(TCQ) clk? d : q;			    \
	  `else					    \
	    q <= #(TCQ) d;				    \
	  `endif				    \
     end
`define XSRREG_SYNC_EXDES(clk, reset_n, q,d,rstval)	\
         always @(posedge clk)                    \
         begin                    \
          if (reset_n == 1'b0)            \
              q <= #(TCQ) rstval;                \
          else                    \
          `ifdef FOURVALCLKPROP            \
             q <= #(TCQ) clk ? d : q;            \
           `else                    \
             q <= #(TCQ)  d;                \
           `endif                \
          end
 
 `define XSRREG_ASYNC_EXDES(clk, reset_n, q,d,rstval)	\
              always @(posedge clk or negedge reset_n)    \
              begin                    \
               if (reset_n == 1'b0)            \
                   q <= #(TCQ) rstval;                \
               else                    \
               `ifdef FOURVALCLKPROP            \
                  q <= #(TCQ) clk ? d : q;            \
                `else                    \
                  q <= #(TCQ)  d;                \
                `endif                \
               end
 
 `define XLREGS_SYNC_EXDES(clk, reset_n) \
                     always @(posedge clk)
 `define XLREGS_ASYNC_EXDES(clk, reset_n) \
                     always @(posedge clk or negedge reset_n)
 
               
 

`define XSRREG_XDMA_EXDES(clk, reset_n, q,d,rstval)        \
`ifdef SOFT_IP  \
`XSRREG_SYNC_EXDES (clk, reset_n, q,d,rstval) \
`else   \
`XSRREG_ASYNC_EXDES (clk, reset_n, q,d,rstval)  \
`endif

`define XSRREG_HARD_CLR_EXDES(clk, reset_n, q,d)        \
`ifdef SOFT_IP  \
`XPREG_NORESET_EXDES(clk, q,d) \
`else   \
`XSRREG_ASYNC_EXDES (clk, reset_n, q,d,'h0)  \
`endif

`define XLREG_XDMA_EXDES(clk, reset_n) \
`ifdef SOFT_IP \
`XLREGS_SYNC_EXDES(clk, reset_n) \
`else \
`XLREGS_ASYNC_EXDES(clk, reset_n)  \
`endif


`endif
