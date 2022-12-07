// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

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
