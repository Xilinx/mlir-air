// Copyright(C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// SPDX-License-Identifier: MIT
//
// SystemVerilog wrapper for BlackParrot
//

`include "bp_common_defines.svh"
`include "bp_be_defines.svh"
`include "bp_me_defines.svh"

module blackparrot_wrapper
 import bp_common_pkg::*;
 import bp_be_pkg::*;
 import bp_me_pkg::*;
 #(parameter bp_params_e bp_params_p = e_bp_unicore_air_cfg
   `declare_bp_proc_params(bp_params_p)

   // BP I/O
   // physical address of BP in the system
   , parameter integer C_S00_AXI_DATA_WIDTH   = 64
   , parameter integer C_S00_AXI_ADDR_WIDTH   = 64
   , parameter [63:0] C_S00_AXI_BASEADDR = 64'h800_0000_0000
   , parameter integer C_M00_AXI_DATA_WIDTH   = 64
   , parameter integer C_M00_AXI_ADDR_WIDTH   = 64
   // BP memory
   , parameter integer C_M01_AXI_DATA_WIDTH   = 64
   , parameter integer C_M01_AXI_ADDR_WIDTH   = 32
   , parameter integer C_M01_AXI_ID_WIDTH     = 6
   // Device ID
   , parameter integer C_DEVICE_ID            = 0
   )
  (input wire                                    resetn
   , input wire                                  core_reset

   , input wire                                  s00_axi_aclk
   , input wire                                  s00_axi_aresetn
   , input wire [C_S00_AXI_ADDR_WIDTH-1 : 0]     s00_axi_awaddr
   , input wire [2 : 0]                          s00_axi_awprot
   , input wire                                  s00_axi_awvalid
   , output wire                                 s00_axi_awready
   , input wire [C_S00_AXI_DATA_WIDTH-1 : 0]     s00_axi_wdata
   , input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb
   , input wire                                  s00_axi_wvalid
   , output wire                                 s00_axi_wready
   , output wire [1 : 0]                         s00_axi_bresp
   , output wire                                 s00_axi_bvalid
   , input wire                                  s00_axi_bready
   , input wire [C_S00_AXI_ADDR_WIDTH-1 : 0]     s00_axi_araddr
   , input wire [2 : 0]                          s00_axi_arprot
   , input wire                                  s00_axi_arvalid
   , output wire                                 s00_axi_arready
   , output wire [C_S00_AXI_DATA_WIDTH-1 : 0]    s00_axi_rdata
   , output wire [1 : 0]                         s00_axi_rresp
   , output wire                                 s00_axi_rvalid
   , input wire                                  s00_axi_rready

   , input wire                                  m00_axi_aclk
   , input wire                                  m00_axi_aresetn
   , output wire [C_M00_AXI_ADDR_WIDTH-1 : 0]    m00_axi_awaddr
   , output wire [2 : 0]                         m00_axi_awprot
   , output wire                                 m00_axi_awvalid
   , input wire                                  m00_axi_awready
   , output wire [C_M00_AXI_DATA_WIDTH-1 : 0]    m00_axi_wdata
   , output wire [(C_M00_AXI_DATA_WIDTH/8)-1:0]  m00_axi_wstrb
   , output wire                                 m00_axi_wvalid
   , input wire                                  m00_axi_wready
   , input wire [1 : 0]                          m00_axi_bresp
   , input wire                                  m00_axi_bvalid
   , output wire                                 m00_axi_bready
   , output wire [C_M00_AXI_ADDR_WIDTH-1 : 0]    m00_axi_araddr
   , output wire [2 : 0]                         m00_axi_arprot
   , output wire                                 m00_axi_arvalid
   , input wire                                  m00_axi_arready
   , input wire [C_M00_AXI_DATA_WIDTH-1 : 0]     m00_axi_rdata
   , input wire [1 : 0]                          m00_axi_rresp
   , input wire                                  m00_axi_rvalid
   , output wire                                 m00_axi_rready

   , input wire                                  m01_axi_aclk
   , input wire                                  m01_axi_aresetn
   , output wire [C_M01_AXI_ADDR_WIDTH-1:0]      m01_axi_awaddr
   , output wire                                 m01_axi_awvalid
   , input wire                                  m01_axi_awready
   , output wire [5:0]                           m01_axi_awid
   , output wire                                 m01_axi_awlock
   , output wire [3:0]                           m01_axi_awcache
   , output wire [2:0]                           m01_axi_awprot
   , output wire [7:0]                           m01_axi_awlen
   , output wire [2:0]                           m01_axi_awsize
   , output wire [1:0]                           m01_axi_awburst
   , output wire [3:0]                           m01_axi_awqos

   , output wire [C_M01_AXI_DATA_WIDTH-1:0]      m01_axi_wdata
   , output wire                                 m01_axi_wvalid
   , input wire                                  m01_axi_wready
   , output wire                                 m01_axi_wlast
   , output wire [(C_M01_AXI_DATA_WIDTH/8)-1:0]  m01_axi_wstrb

   , input wire                                  m01_axi_bvalid
   , output wire                                 m01_axi_bready
   , input wire [5:0]                            m01_axi_bid
   , input wire [1:0]                            m01_axi_bresp

   , output wire [C_M01_AXI_ADDR_WIDTH-1:0]      m01_axi_araddr
   , output wire                                 m01_axi_arvalid
   , input wire                                  m01_axi_arready
   , output wire [5:0]                           m01_axi_arid
   , output wire                                 m01_axi_arlock
   , output wire [3:0]                           m01_axi_arcache
   , output wire [2:0]                           m01_axi_arprot
   , output wire [7:0]                           m01_axi_arlen
   , output wire [2:0]                           m01_axi_arsize
   , output wire [1:0]                           m01_axi_arburst
   , output wire [3:0]                           m01_axi_arqos

   , input wire [C_M01_AXI_DATA_WIDTH-1:0]       m01_axi_rdata
   , input wire                                  m01_axi_rvalid
   , output wire                                 m01_axi_rready
   , input wire [5:0]                            m01_axi_rid
   , input wire                                  m01_axi_rlast
   , input wire [1:0]                            m01_axi_rresp
   );

  // BlackParrot reset signal
  wire bp_reset_li = ~s00_axi_aresetn | ~m00_axi_aresetn | ~m01_axi_aresetn | ~resetn | core_reset;

  // subtract base address from AXI address to map into BP-local address
  logic [C_S00_AXI_ADDR_WIDTH-1:0] s00_axi_awaddr_li, s00_axi_araddr_li;
  assign s00_axi_awaddr_li = s00_axi_awaddr - C_S00_AXI_BASEADDR;
  assign s00_axi_araddr_li = s00_axi_araddr - C_S00_AXI_BASEADDR;

  // stub these outputs since they are not driven by bp_axi_top
  assign m01_axi_awqos = '0;
  assign m01_axi_arqos = '0;

  // convert Device ID parameter to an input wire for BP core
  wire [io_noc_did_width_p-1:0] did = io_noc_did_width_p'(C_DEVICE_ID);

  // Notes:
  // BP runs on a single clock - all axi clocks must be the same
  // the clock used is s00_axi_aclk
  // BP has single reset - currently the OR of all AXI and other resets
  bp_axi_top #
    (.bp_params_p(bp_params_p)
     ,.m_axil_addr_width_p(C_M00_AXI_ADDR_WIDTH)
     ,.m_axil_data_width_p(C_M00_AXI_DATA_WIDTH)
     ,.s_axil_addr_width_p(C_S00_AXI_ADDR_WIDTH)
     ,.s_axil_data_width_p(C_S00_AXI_DATA_WIDTH)
     ,.axi_addr_width_p(C_M01_AXI_ADDR_WIDTH)
     ,.axi_data_width_p(C_M01_AXI_DATA_WIDTH)
     ,.axi_id_width_p(C_M01_AXI_ID_WIDTH)
     ,.axi_len_width_p(8) // set for AXI4
     ,.axi_size_width_p(3) // set for AXI4
     )
    blackparrot
    (.clk_i(s00_axi_aclk)
     ,.reset_i(bp_reset_li)
     ,.rt_clk_i(s00_axi_aclk)
     ,.did_i(did)

     // I/O reads/writes from BlackParrot
     ,.m_axil_awaddr_o (m00_axi_awaddr)
     ,.m_axil_awprot_o (m00_axi_awprot)
     ,.m_axil_awvalid_o(m00_axi_awvalid)
     ,.m_axil_awready_i(m00_axi_awready)

     ,.m_axil_wdata_o  (m00_axi_wdata)
     ,.m_axil_wstrb_o  (m00_axi_wstrb)
     ,.m_axil_wvalid_o (m00_axi_wvalid)
     ,.m_axil_wready_i (m00_axi_wready)

     ,.m_axil_bresp_i  (m00_axi_bresp)
     ,.m_axil_bvalid_i (m00_axi_bvalid)
     ,.m_axil_bready_o (m00_axi_bready)

     ,.m_axil_araddr_o (m00_axi_araddr)
     ,.m_axil_arprot_o (m00_axi_arprot)
     ,.m_axil_arvalid_o(m00_axi_arvalid)
     ,.m_axil_arready_i(m00_axi_arready)

     ,.m_axil_rdata_i  (m00_axi_rdata)
     ,.m_axil_rresp_i  (m00_axi_rresp)
     ,.m_axil_rvalid_i (m00_axi_rvalid)
     ,.m_axil_rready_o (m00_axi_rready)

     // I/O reads/writes into BlackParrot
     ,.s_axil_awaddr_i (s00_axi_awaddr_li)
     ,.s_axil_awprot_i (s00_axi_awprot)
     ,.s_axil_awvalid_i(s00_axi_awvalid)
     ,.s_axil_awready_o(s00_axi_awready)

     ,.s_axil_wdata_i  (s00_axi_wdata)
     ,.s_axil_wstrb_i  (s00_axi_wstrb)
     ,.s_axil_wvalid_i (s00_axi_wvalid)
     ,.s_axil_wready_o (s00_axi_wready)

     ,.s_axil_bresp_o  (s00_axi_bresp)
     ,.s_axil_bvalid_o (s00_axi_bvalid)
     ,.s_axil_bready_i (s00_axi_bready)

     ,.s_axil_araddr_i (s00_axi_araddr_li)
     ,.s_axil_arprot_i (s00_axi_arprot)
     ,.s_axil_arvalid_i(s00_axi_arvalid)
     ,.s_axil_arready_o(s00_axi_arready)

     ,.s_axil_rdata_o  (s00_axi_rdata)
     ,.s_axil_rresp_o  (s00_axi_rresp)
     ,.s_axil_rvalid_o (s00_axi_rvalid)
     ,.s_axil_rready_i (s00_axi_rready)

     // Memory access from BlackParrot
     ,.m_axi_awaddr_o   (m01_axi_awaddr)
     ,.m_axi_awvalid_o  (m01_axi_awvalid)
     ,.m_axi_awready_i  (m01_axi_awready)
     ,.m_axi_awid_o     (m01_axi_awid)
     ,.m_axi_awlock_o   (m01_axi_awlock)
     ,.m_axi_awcache_o  (m01_axi_awcache)
     ,.m_axi_awprot_o   (m01_axi_awprot)
     ,.m_axi_awlen_o    (m01_axi_awlen)
     ,.m_axi_awsize_o   (m01_axi_awsize)
     ,.m_axi_awburst_o  (m01_axi_awburst)
     ,.m_axi_awqos_o    () // not driven by bp_axi_top

     ,.m_axi_wdata_o    (m01_axi_wdata)
     ,.m_axi_wvalid_o   (m01_axi_wvalid)
     ,.m_axi_wready_i   (m01_axi_wready)
     ,.m_axi_wid_o      () // AXI3 only, ignore
     ,.m_axi_wlast_o    (m01_axi_wlast)
     ,.m_axi_wstrb_o    (m01_axi_wstrb)

     ,.m_axi_bvalid_i   (m01_axi_bvalid)
     ,.m_axi_bready_o   (m01_axi_bready)
     ,.m_axi_bid_i      (m01_axi_bid)
     ,.m_axi_bresp_i    (m01_axi_bresp)

     ,.m_axi_araddr_o   (m01_axi_araddr)
     ,.m_axi_arvalid_o  (m01_axi_arvalid)
     ,.m_axi_arready_i  (m01_axi_arready)
     ,.m_axi_arid_o     (m01_axi_arid)
     ,.m_axi_arlock_o   (m01_axi_arlock)
     ,.m_axi_arcache_o  (m01_axi_arcache)
     ,.m_axi_arprot_o   (m01_axi_arprot)
     ,.m_axi_arlen_o    (m01_axi_arlen)
     ,.m_axi_arsize_o   (m01_axi_arsize)
     ,.m_axi_arburst_o  (m01_axi_arburst)
     ,.m_axi_arqos_o    () // not driven by bp_axi_top

     ,.m_axi_rdata_i    (m01_axi_rdata)
     ,.m_axi_rvalid_i   (m01_axi_rvalid)
     ,.m_axi_rready_o   (m01_axi_rready)
     ,.m_axi_rid_i      (m01_axi_rid)
     ,.m_axi_rlast_i    (m01_axi_rlast)
     ,.m_axi_rresp_i    (m01_axi_rresp)
     );

endmodule

