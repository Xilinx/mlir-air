// Copyright(C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// SPDX-License-Identifier: MIT
//
// Verilog top-level for Vivado IP block creation
//

module blackparrot
  #(
    // BP I/O
    parameter integer C_S00_AXI_DATA_WIDTH = 64
    , parameter integer C_S00_AXI_ADDR_WIDTH = 64
    , parameter [C_S00_AXI_ADDR_WIDTH-1:0] C_S00_AXI_BASEADDR = 'h800_0000_0000
    , parameter integer C_M00_AXI_DATA_WIDTH = 64
    , parameter integer C_M00_AXI_ADDR_WIDTH = 64
    // BP Memory
    , parameter integer C_M01_AXI_DATA_WIDTH = 64
    , parameter integer C_M01_AXI_ADDR_WIDTH = 32
    , parameter integer C_M01_AXI_ID_WIDTH   = 6
    )
   (// external reset for core
    input wire                                   resetn
    ,input wire                                  core_reset

    ,input wire                                  s00_axi_aclk
    ,input wire                                  s00_axi_aresetn
    ,input wire [C_S00_AXI_ADDR_WIDTH-1 : 0]     s00_axi_awaddr
    ,input wire [2 : 0]                          s00_axi_awprot
    ,input wire                                  s00_axi_awvalid
    ,output wire                                 s00_axi_awready
    ,input wire [C_S00_AXI_DATA_WIDTH-1 : 0]     s00_axi_wdata
    ,input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb
    ,input wire                                  s00_axi_wvalid
    ,output wire                                 s00_axi_wready
    ,output wire [1 : 0]                         s00_axi_bresp
    ,output wire                                 s00_axi_bvalid
    ,input wire                                  s00_axi_bready
    ,input wire [C_S00_AXI_ADDR_WIDTH-1 : 0]     s00_axi_araddr
    ,input wire [2 : 0]                          s00_axi_arprot
    ,input wire                                  s00_axi_arvalid
    ,output wire                                 s00_axi_arready
    ,output wire [C_S00_AXI_DATA_WIDTH-1 : 0]    s00_axi_rdata
    ,output wire [1 : 0]                         s00_axi_rresp
    ,output wire                                 s00_axi_rvalid
    ,input wire                                  s00_axi_rready

    ,input wire                                  m00_axi_aclk
    ,input wire                                  m00_axi_aresetn
    ,output wire [C_M00_AXI_ADDR_WIDTH-1 : 0]    m00_axi_awaddr
    ,output wire [2 : 0]                         m00_axi_awprot
    ,output wire                                 m00_axi_awvalid
    ,input wire                                  m00_axi_awready
    ,output wire [C_M00_AXI_DATA_WIDTH-1 : 0]    m00_axi_wdata
    ,output wire [(C_M00_AXI_DATA_WIDTH/8)-1:0]  m00_axi_wstrb
    ,output wire                                 m00_axi_wvalid
    ,input wire                                  m00_axi_wready
    ,input wire [1 : 0]                          m00_axi_bresp
    ,input wire                                  m00_axi_bvalid
    ,output wire                                 m00_axi_bready
    ,output wire [C_M00_AXI_ADDR_WIDTH-1 : 0]    m00_axi_araddr
    ,output wire [2 : 0]                         m00_axi_arprot
    ,output wire                                 m00_axi_arvalid
    ,input wire                                  m00_axi_arready
    ,input wire [C_M00_AXI_DATA_WIDTH-1 : 0]     m00_axi_rdata
    ,input wire [1 : 0]                          m00_axi_rresp
    ,input wire                                  m00_axi_rvalid
    ,output wire                                 m00_axi_rready

    ,input wire                                  m01_axi_aclk
    ,input wire                                  m01_axi_aresetn
    ,output wire [C_M01_AXI_ADDR_WIDTH-1:0]      m01_axi_awaddr
    ,output wire                                 m01_axi_awvalid
    ,input wire                                  m01_axi_awready
    ,output wire [5:0]                           m01_axi_awid
    ,output wire                                 m01_axi_awlock
    ,output wire [3:0]                           m01_axi_awcache
    ,output wire [2:0]                           m01_axi_awprot
    ,output wire [7:0]                           m01_axi_awlen
    ,output wire [2:0]                           m01_axi_awsize
    ,output wire [1:0]                           m01_axi_awburst
    ,output wire [3:0]                           m01_axi_awqos

    ,output wire [C_M01_AXI_DATA_WIDTH-1:0]      m01_axi_wdata
    ,output wire                                 m01_axi_wvalid
    ,input wire                                  m01_axi_wready
    ,output wire                                 m01_axi_wlast
    ,output wire [(C_M01_AXI_DATA_WIDTH/8)-1:0]  m01_axi_wstrb

    ,input wire                                  m01_axi_bvalid
    ,output wire                                 m01_axi_bready
    ,input wire [5:0]                            m01_axi_bid
    ,input wire [1:0]                            m01_axi_bresp

    ,output wire [C_M01_AXI_ADDR_WIDTH-1:0]      m01_axi_araddr
    ,output wire                                 m01_axi_arvalid
    ,input wire                                  m01_axi_arready
    ,output wire [5:0]                           m01_axi_arid
    ,output wire                                 m01_axi_arlock
    ,output wire [3:0]                           m01_axi_arcache
    ,output wire [2:0]                           m01_axi_arprot
    ,output wire [7:0]                           m01_axi_arlen
    ,output wire [2:0]                           m01_axi_arsize
    ,output wire [1:0]                           m01_axi_arburst
    ,output wire [3:0]                           m01_axi_arqos

    ,input wire [C_M01_AXI_DATA_WIDTH-1:0]       m01_axi_rdata
    ,input wire                                  m01_axi_rvalid
    ,output wire                                 m01_axi_rready
    ,input wire [5:0]                            m01_axi_rid
    ,input wire                                  m01_axi_rlast
    ,input wire [1:0]                            m01_axi_rresp

    );

   blackparrot_wrapper #
     (.C_S00_AXI_DATA_WIDTH (C_S00_AXI_DATA_WIDTH)
      ,.C_S00_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
      ,.C_S00_AXI_BASEADDR(C_S00_AXI_BASEADDR)
      ,.C_M00_AXI_DATA_WIDTH(C_M00_AXI_DATA_WIDTH)
      ,.C_M00_AXI_ADDR_WIDTH(C_M00_AXI_ADDR_WIDTH)
      ,.C_M01_AXI_DATA_WIDTH(C_M01_AXI_DATA_WIDTH)
      ,.C_M01_AXI_ADDR_WIDTH(C_M01_AXI_ADDR_WIDTH)
      ,.C_M01_AXI_ID_WIDTH(C_M01_AXI_ID_WIDTH)
      )
     blackparrot_wrapper_inst
     (.resetn          (resetn)
      ,.core_reset     (core_reset)
      ,.s00_axi_aclk   (s00_axi_aclk)
      ,.s00_axi_aresetn(s00_axi_aresetn)
      ,.s00_axi_awaddr (s00_axi_awaddr)
      ,.s00_axi_awprot (s00_axi_awprot)
      ,.s00_axi_awvalid(s00_axi_awvalid)
      ,.s00_axi_awready(s00_axi_awready)
      ,.s00_axi_wdata  (s00_axi_wdata)
      ,.s00_axi_wstrb  (s00_axi_wstrb)
      ,.s00_axi_wvalid (s00_axi_wvalid)
      ,.s00_axi_wready (s00_axi_wready)
      ,.s00_axi_bresp  (s00_axi_bresp)
      ,.s00_axi_bvalid (s00_axi_bvalid)
      ,.s00_axi_bready (s00_axi_bready)
      ,.s00_axi_araddr (s00_axi_araddr)
      ,.s00_axi_arprot (s00_axi_arprot)
      ,.s00_axi_arvalid(s00_axi_arvalid)
      ,.s00_axi_arready(s00_axi_arready)
      ,.s00_axi_rdata  (s00_axi_rdata)
      ,.s00_axi_rresp  (s00_axi_rresp)
      ,.s00_axi_rvalid (s00_axi_rvalid)
      ,.s00_axi_rready (s00_axi_rready)

      ,.m00_axi_aclk   (m00_axi_aclk)
      ,.m00_axi_aresetn(m00_axi_aresetn)
      ,.m00_axi_awaddr (m00_axi_awaddr)
      ,.m00_axi_awprot (m00_axi_awprot)
      ,.m00_axi_awvalid(m00_axi_awvalid)
      ,.m00_axi_awready(m00_axi_awready)
      ,.m00_axi_wdata  (m00_axi_wdata)
      ,.m00_axi_wstrb  (m00_axi_wstrb)
      ,.m00_axi_wvalid (m00_axi_wvalid)
      ,.m00_axi_wready (m00_axi_wready)
      ,.m00_axi_bresp  (m00_axi_bresp)
      ,.m00_axi_bvalid (m00_axi_bvalid)
      ,.m00_axi_bready (m00_axi_bready)
      ,.m00_axi_araddr (m00_axi_araddr)
      ,.m00_axi_arprot (m00_axi_arprot)
      ,.m00_axi_arvalid(m00_axi_arvalid)
      ,.m00_axi_arready(m00_axi_arready)
      ,.m00_axi_rdata  (m00_axi_rdata)
      ,.m00_axi_rresp  (m00_axi_rresp)
      ,.m00_axi_rvalid (m00_axi_rvalid)
      ,.m00_axi_rready (m00_axi_rready)

      ,.m01_axi_aclk   (m01_axi_aclk)
      ,.m01_axi_aresetn(m01_axi_aresetn)
      ,.m01_axi_awaddr (m01_axi_awaddr)
      ,.m01_axi_awvalid(m01_axi_awvalid)
      ,.m01_axi_awready(m01_axi_awready)
      ,.m01_axi_awid   (m01_axi_awid)
      ,.m01_axi_awlock (m01_axi_awlock)
      ,.m01_axi_awcache(m01_axi_awcache)
      ,.m01_axi_awprot (m01_axi_awprot)
      ,.m01_axi_awlen  (m01_axi_awlen)
      ,.m01_axi_awsize (m01_axi_awsize)
      ,.m01_axi_awburst(m01_axi_awburst)
      ,.m01_axi_awqos  (m01_axi_awqos)

      ,.m01_axi_wdata  (m01_axi_wdata)
      ,.m01_axi_wvalid (m01_axi_wvalid)
      ,.m01_axi_wready (m01_axi_wready)
      ,.m01_axi_wlast  (m01_axi_wlast)
      ,.m01_axi_wstrb  (m01_axi_wstrb)

      ,.m01_axi_bvalid (m01_axi_bvalid)
      ,.m01_axi_bready (m01_axi_bready)
      ,.m01_axi_bid    (m01_axi_bid)
      ,.m01_axi_bresp  (m01_axi_bresp)

      ,.m01_axi_araddr (m01_axi_araddr)
      ,.m01_axi_arvalid(m01_axi_arvalid)
      ,.m01_axi_arready(m01_axi_arready)
      ,.m01_axi_arid   (m01_axi_arid)
      ,.m01_axi_arlock (m01_axi_arlock)
      ,.m01_axi_arcache(m01_axi_arcache)
      ,.m01_axi_arprot (m01_axi_arprot)
      ,.m01_axi_arlen  (m01_axi_arlen)
      ,.m01_axi_arsize (m01_axi_arsize)
      ,.m01_axi_arburst(m01_axi_arburst)
      ,.m01_axi_arqos  (m01_axi_arqos)

      ,.m01_axi_rdata  (m01_axi_rdata)
      ,.m01_axi_rvalid (m01_axi_rvalid)
      ,.m01_axi_rready (m01_axi_rready)
      ,.m01_axi_rid    (m01_axi_rid)
      ,.m01_axi_rlast  (m01_axi_rlast)
      ,.m01_axi_rresp  (m01_axi_rresp)
      );

endmodule

