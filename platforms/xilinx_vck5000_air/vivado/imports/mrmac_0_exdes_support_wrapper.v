//Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
//--------------------------------------------------------------------------------
//Tool Version: Vivado v.2022.1 (lin64) Build 3526262 Mon Apr 18 15:47:01 MDT 2022
//Date        : Wed May 25 10:09:34 2022
//Host        : xsjrdevl169 running 64-bit CentOS Linux release 7.4.1708 (Core)
//Command     : generate_target mrmac_0_exdes_support_wrapper.bd
//Design      : mrmac_0_exdes_support_wrapper
//Purpose     : IP block netlist
//--------------------------------------------------------------------------------
`timescale 1 ps / 1 ps

module mrmac_0_exdes_support_wrapper
   (APB3_INTF_paddr,
    APB3_INTF_penable,
    APB3_INTF_prdata,
    APB3_INTF_pready,
    APB3_INTF_psel,
    APB3_INTF_pslverr,
    APB3_INTF_pwdata,
    APB3_INTF_pwrite,
    CLK_IN_D_clk_n,
    CLK_IN_D_clk_p,
    apb3clk_quad,
    ch0_loopback,
    ch0_rx_usr_clk,
    ch0_rx_usr_clk2,
    ch0_rxrate,
    ch0_rxusrclk,
    ch0_tx_usr_clk,
    ch0_tx_usr_clk2,
    ch0_txrate,
    ch0_txusrclk,
    ch1_loopback,
    ch1_rx_usr_clk,
    ch1_rx_usr_clk2,
    ch1_rxrate,
    ch1_rxusrclk,
    ch1_txrate,
    ch1_txusrclk,
    ch2_loopback,
    ch2_rx_usr_clk,
    ch2_rx_usr_clk2,
    ch2_rxrate,
    ch2_rxusrclk,
    ch2_txrate,
    ch2_txusrclk,
    ch3_loopback,
    ch3_rx_usr_clk,
    ch3_rx_usr_clk2,
    ch3_rxrate,
    ch3_rxusrclk,
    ch3_txrate,
    ch3_txusrclk,
    ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override,
    ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override_value,
    ctl_tx_port0_ctl_tx_send_idle_in,
    ctl_tx_port0_ctl_tx_send_lfi_in,
    ctl_tx_port0_ctl_tx_send_rfi_in,
    ctl_tx_port1_ctl_tx_send_idle_in,
    ctl_tx_port1_ctl_tx_send_lfi_in,
    ctl_tx_port1_ctl_tx_send_rfi_in,
    ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override,
    ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override_value,
    ctl_tx_port2_ctl_tx_send_idle_in,
    ctl_tx_port2_ctl_tx_send_lfi_in,
    ctl_tx_port2_ctl_tx_send_rfi_in,
    ctl_tx_port3_ctl_tx_send_idle_in,
    ctl_tx_port3_ctl_tx_send_lfi_in,
    ctl_tx_port3_ctl_tx_send_rfi_in,
    gt_reset_all_in,
    gt_reset_rx_datapath_in,
    gt_reset_tx_datapath_in,
    gt_rx_reset_done_out,
    gt_rxn_in_0,
    gt_rxp_in_0,
    gt_tx_reset_done_out,
    gt_txn_out_0,
    gt_txp_out_0,
    gtpowergood,
    gtpowergood_in,
    pm_rdy,
    pm_tick,
    rx_alt_serdes_clk,
    rx_axi_clk,
    rx_axis_tdata0,
    rx_axis_tdata1,
    rx_axis_tdata2,
    rx_axis_tdata3,
    rx_axis_tdata4,
    rx_axis_tdata5,
    rx_axis_tdata6,
    rx_axis_tdata7,
    rx_axis_tkeep_user0,
    rx_axis_tkeep_user1,
    rx_axis_tkeep_user2,
    rx_axis_tkeep_user3,
    rx_axis_tkeep_user4,
    rx_axis_tkeep_user5,
    rx_axis_tkeep_user6,
    rx_axis_tkeep_user7,
    rx_axis_tlast_0,
    rx_axis_tlast_1,
    rx_axis_tlast_2,
    rx_axis_tlast_3,
    rx_axis_tvalid_0,
    rx_axis_tvalid_1,
    rx_axis_tvalid_2,
    rx_axis_tvalid_3,
    rx_core_clk,
    rx_core_reset,
    rx_flexif_clk,
    rx_flexif_reset,
    rx_preambleout_rx_preambleout_0,
    rx_preambleout_rx_preambleout_1,
    rx_preambleout_rx_preambleout_2,
    rx_preambleout_rx_preambleout_3,
    rx_serdes_clk,
    rx_serdes_reset,
    rx_ts_clk,
    s_axi_aclk,
    s_axi_araddr,
    s_axi_aresetn,
    s_axi_arready,
    s_axi_arvalid,
    s_axi_awaddr,
    s_axi_awready,
    s_axi_awvalid,
    s_axi_bready,
    s_axi_bresp,
    s_axi_bvalid,
    s_axi_rdata,
    s_axi_rready,
    s_axi_rresp,
    s_axi_rvalid,
    s_axi_wdata,
    s_axi_wready,
    s_axi_wvalid,
    stat_rx_port0_stat_rx_aligned,
    stat_rx_port0_stat_rx_aligned_err,
    stat_rx_port0_stat_rx_axis_err,
    stat_rx_port0_stat_rx_axis_fifo_overflow,
    stat_rx_port0_stat_rx_bad_code,
    stat_rx_port0_stat_rx_bad_fcs,
    stat_rx_port0_stat_rx_bad_preamble,
    stat_rx_port0_stat_rx_bad_sfd,
    stat_rx_port0_stat_rx_bip_err,
    stat_rx_port0_stat_rx_block_lock,
    stat_rx_port0_stat_rx_cl49_82_convert_err,
    stat_rx_port0_stat_rx_ecc_err,
    stat_rx_port0_stat_rx_flex_fifo_ovf,
    stat_rx_port0_stat_rx_flex_fifo_udf,
    stat_rx_port0_stat_rx_flex_mon_fifo_ovf,
    stat_rx_port0_stat_rx_flex_mon_fifo_udf,
    stat_rx_port0_stat_rx_flexif_err,
    stat_rx_port0_stat_rx_framing_err_0,
    stat_rx_port0_stat_rx_got_signal_os,
    stat_rx_port0_stat_rx_hi_ber,
    stat_rx_port0_stat_rx_internal_local_fault,
    stat_rx_port0_stat_rx_invalid_start,
    stat_rx_port0_stat_rx_lane0_vlm_bip7,
    stat_rx_port0_stat_rx_lane0_vlm_bip7_valid,
    stat_rx_port0_stat_rx_local_fault,
    stat_rx_port0_stat_rx_mf_err_0,
    stat_rx_port0_stat_rx_misaligned,
    stat_rx_port0_stat_rx_pcs_bad_code,
    stat_rx_port0_stat_rx_received_local_fault,
    stat_rx_port0_stat_rx_remote_fault,
    stat_rx_port0_stat_rx_status,
    stat_rx_port0_stat_rx_synced,
    stat_rx_port0_stat_rx_synced_err,
    stat_rx_port0_stat_rx_test_pattern_mismatch,
    stat_rx_port0_stat_rx_truncated,
    stat_rx_port0_stat_rx_valid_ctrl_code,
    stat_rx_port0_stat_rx_vl_demuxed,
    stat_rx_port1_stat_rx_axis_err,
    stat_rx_port1_stat_rx_axis_fifo_overflow,
    stat_rx_port1_stat_rx_bad_code,
    stat_rx_port1_stat_rx_bad_fcs,
    stat_rx_port1_stat_rx_bad_preamble,
    stat_rx_port1_stat_rx_bad_sfd,
    stat_rx_port1_stat_rx_block_lock,
    stat_rx_port1_stat_rx_cl49_82_convert_err,
    stat_rx_port1_stat_rx_ecc_err,
    stat_rx_port1_stat_rx_flex_fifo_ovf,
    stat_rx_port1_stat_rx_flex_fifo_udf,
    stat_rx_port1_stat_rx_flex_mon_fifo_ovf,
    stat_rx_port1_stat_rx_flex_mon_fifo_udf,
    stat_rx_port1_stat_rx_flexif_err,
    stat_rx_port1_stat_rx_framing_err_1,
    stat_rx_port1_stat_rx_got_signal_os,
    stat_rx_port1_stat_rx_hi_ber,
    stat_rx_port1_stat_rx_internal_local_fault,
    stat_rx_port1_stat_rx_invalid_start,
    stat_rx_port1_stat_rx_local_fault,
    stat_rx_port1_stat_rx_pcs_bad_code,
    stat_rx_port1_stat_rx_received_local_fault,
    stat_rx_port1_stat_rx_remote_fault,
    stat_rx_port1_stat_rx_status,
    stat_rx_port1_stat_rx_test_pattern_mismatch,
    stat_rx_port1_stat_rx_truncated,
    stat_rx_port1_stat_rx_valid_ctrl_code,
    stat_rx_port2_stat_rx_aligned,
    stat_rx_port2_stat_rx_aligned_err,
    stat_rx_port2_stat_rx_axis_err,
    stat_rx_port2_stat_rx_axis_fifo_overflow,
    stat_rx_port2_stat_rx_bad_code,
    stat_rx_port2_stat_rx_bad_fcs,
    stat_rx_port2_stat_rx_bad_preamble,
    stat_rx_port2_stat_rx_bad_sfd,
    stat_rx_port2_stat_rx_bip_err,
    stat_rx_port2_stat_rx_block_lock,
    stat_rx_port2_stat_rx_cl49_82_convert_err,
    stat_rx_port2_stat_rx_ecc_err,
    stat_rx_port2_stat_rx_flex_fifo_ovf,
    stat_rx_port2_stat_rx_flex_fifo_udf,
    stat_rx_port2_stat_rx_flex_mon_fifo_ovf,
    stat_rx_port2_stat_rx_flex_mon_fifo_udf,
    stat_rx_port2_stat_rx_flexif_err,
    stat_rx_port2_stat_rx_framing_err_2,
    stat_rx_port2_stat_rx_got_signal_os,
    stat_rx_port2_stat_rx_hi_ber,
    stat_rx_port2_stat_rx_internal_local_fault,
    stat_rx_port2_stat_rx_invalid_start,
    stat_rx_port2_stat_rx_lane0_vlm_bip7,
    stat_rx_port2_stat_rx_lane0_vlm_bip7_valid,
    stat_rx_port2_stat_rx_local_fault,
    stat_rx_port2_stat_rx_mf_err_2,
    stat_rx_port2_stat_rx_misaligned,
    stat_rx_port2_stat_rx_pcs_bad_code,
    stat_rx_port2_stat_rx_received_local_fault,
    stat_rx_port2_stat_rx_remote_fault,
    stat_rx_port2_stat_rx_status,
    stat_rx_port2_stat_rx_synced,
    stat_rx_port2_stat_rx_synced_err,
    stat_rx_port2_stat_rx_test_pattern_mismatch,
    stat_rx_port2_stat_rx_truncated,
    stat_rx_port2_stat_rx_valid_ctrl_code,
    stat_rx_port2_stat_rx_vl_demuxed,
    stat_rx_port3_stat_rx_axis_err,
    stat_rx_port3_stat_rx_axis_fifo_overflow,
    stat_rx_port3_stat_rx_bad_code,
    stat_rx_port3_stat_rx_bad_fcs,
    stat_rx_port3_stat_rx_bad_preamble,
    stat_rx_port3_stat_rx_bad_sfd,
    stat_rx_port3_stat_rx_block_lock,
    stat_rx_port3_stat_rx_cl49_82_convert_err,
    stat_rx_port3_stat_rx_ecc_err,
    stat_rx_port3_stat_rx_flex_fifo_ovf,
    stat_rx_port3_stat_rx_flex_fifo_udf,
    stat_rx_port3_stat_rx_flex_mon_fifo_ovf,
    stat_rx_port3_stat_rx_flex_mon_fifo_udf,
    stat_rx_port3_stat_rx_flexif_err,
    stat_rx_port3_stat_rx_framing_err_3,
    stat_rx_port3_stat_rx_got_signal_os,
    stat_rx_port3_stat_rx_hi_ber,
    stat_rx_port3_stat_rx_internal_local_fault,
    stat_rx_port3_stat_rx_invalid_start,
    stat_rx_port3_stat_rx_local_fault,
    stat_rx_port3_stat_rx_pcs_bad_code,
    stat_rx_port3_stat_rx_received_local_fault,
    stat_rx_port3_stat_rx_remote_fault,
    stat_rx_port3_stat_rx_status,
    stat_rx_port3_stat_rx_test_pattern_mismatch,
    stat_rx_port3_stat_rx_truncated,
    stat_rx_port3_stat_rx_valid_ctrl_code,
    stat_tx_port0_stat_tx_axis_err,
    stat_tx_port0_stat_tx_axis_unf,
    stat_tx_port0_stat_tx_bad_fcs,
    stat_tx_port0_stat_tx_cl82_49_convert_err,
    stat_tx_port0_stat_tx_ecc_err,
    stat_tx_port0_stat_tx_flex_fifo_ovf,
    stat_tx_port0_stat_tx_flex_fifo_udf,
    stat_tx_port0_stat_tx_flexif_err,
    stat_tx_port0_stat_tx_frame_error,
    stat_tx_port0_stat_tx_local_fault,
    stat_tx_port0_stat_tx_pcs_bad_code,
    stat_tx_port1_stat_tx_axis_err,
    stat_tx_port1_stat_tx_axis_unf,
    stat_tx_port1_stat_tx_bad_fcs,
    stat_tx_port1_stat_tx_cl82_49_convert_err,
    stat_tx_port1_stat_tx_ecc_err,
    stat_tx_port1_stat_tx_flex_fifo_ovf,
    stat_tx_port1_stat_tx_flex_fifo_udf,
    stat_tx_port1_stat_tx_flexif_err,
    stat_tx_port1_stat_tx_frame_error,
    stat_tx_port1_stat_tx_local_fault,
    stat_tx_port1_stat_tx_pcs_bad_code,
    stat_tx_port2_stat_tx_axis_err,
    stat_tx_port2_stat_tx_axis_unf,
    stat_tx_port2_stat_tx_bad_fcs,
    stat_tx_port2_stat_tx_cl82_49_convert_err,
    stat_tx_port2_stat_tx_ecc_err,
    stat_tx_port2_stat_tx_flex_fifo_ovf,
    stat_tx_port2_stat_tx_flex_fifo_udf,
    stat_tx_port2_stat_tx_flexif_err,
    stat_tx_port2_stat_tx_frame_error,
    stat_tx_port2_stat_tx_local_fault,
    stat_tx_port2_stat_tx_pcs_bad_code,
    stat_tx_port3_stat_tx_axis_err,
    stat_tx_port3_stat_tx_axis_unf,
    stat_tx_port3_stat_tx_bad_fcs,
    stat_tx_port3_stat_tx_cl82_49_convert_err,
    stat_tx_port3_stat_tx_ecc_err,
    stat_tx_port3_stat_tx_flex_fifo_ovf,
    stat_tx_port3_stat_tx_flex_fifo_udf,
    stat_tx_port3_stat_tx_flexif_err,
    stat_tx_port3_stat_tx_frame_error,
    stat_tx_port3_stat_tx_local_fault,
    stat_tx_port3_stat_tx_pcs_bad_code,
    tx_alt_serdes_clk,
    tx_axi_clk,
    tx_axis_tdata0,
    tx_axis_tdata1,
    tx_axis_tdata2,
    tx_axis_tdata3,
    tx_axis_tdata4,
    tx_axis_tdata5,
    tx_axis_tdata6,
    tx_axis_tdata7,
    tx_axis_tkeep_user0,
    tx_axis_tkeep_user1,
    tx_axis_tkeep_user2,
    tx_axis_tkeep_user3,
    tx_axis_tkeep_user4,
    tx_axis_tkeep_user5,
    tx_axis_tkeep_user6,
    tx_axis_tkeep_user7,
    tx_axis_tlast_0,
    tx_axis_tlast_1,
    tx_axis_tlast_2,
    tx_axis_tlast_3,
    tx_axis_tready_0,
    tx_axis_tready_1,
    tx_axis_tready_2,
    tx_axis_tready_3,
    tx_axis_tvalid_0,
    tx_axis_tvalid_1,
    tx_axis_tvalid_2,
    tx_axis_tvalid_3,
    tx_core_clk,
    tx_core_reset,
    tx_flexif_clk,
    tx_preamblein_tx_preamblein_0,
    tx_preamblein_tx_preamblein_1,
    tx_preamblein_tx_preamblein_2,
    tx_preamblein_tx_preamblein_3,
    tx_serdes_reset,
    tx_ts_clk);
  input [15:0]APB3_INTF_paddr;
  input APB3_INTF_penable;
  output [31:0]APB3_INTF_prdata;
  output APB3_INTF_pready;
  input APB3_INTF_psel;
  output APB3_INTF_pslverr;
  input [31:0]APB3_INTF_pwdata;
  input APB3_INTF_pwrite;
  input [0:0]CLK_IN_D_clk_n;
  input [0:0]CLK_IN_D_clk_p;
  input apb3clk_quad;
  input [2:0]ch0_loopback;
  output [0:0]ch0_rx_usr_clk;
  output [0:0]ch0_rx_usr_clk2;
  input [7:0]ch0_rxrate;
  input ch0_rxusrclk;
  output [0:0]ch0_tx_usr_clk;
  output [0:0]ch0_tx_usr_clk2;
  input [7:0]ch0_txrate;
  input ch0_txusrclk;
  input [2:0]ch1_loopback;
  output [0:0]ch1_rx_usr_clk;
  output [0:0]ch1_rx_usr_clk2;
  input [7:0]ch1_rxrate;
  input ch1_rxusrclk;
  input [7:0]ch1_txrate;
  input ch1_txusrclk;
  input [2:0]ch2_loopback;
  output [0:0]ch2_rx_usr_clk;
  output [0:0]ch2_rx_usr_clk2;
  input [7:0]ch2_rxrate;
  input ch2_rxusrclk;
  input [7:0]ch2_txrate;
  input ch2_txusrclk;
  input [2:0]ch3_loopback;
  output [0:0]ch3_rx_usr_clk;
  output [0:0]ch3_rx_usr_clk2;
  input [7:0]ch3_rxrate;
  input ch3_rxusrclk;
  input [7:0]ch3_txrate;
  input ch3_txusrclk;
  input ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override;
  input [7:0]ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override_value;
  input ctl_tx_port0_ctl_tx_send_idle_in;
  input ctl_tx_port0_ctl_tx_send_lfi_in;
  input ctl_tx_port0_ctl_tx_send_rfi_in;
  input ctl_tx_port1_ctl_tx_send_idle_in;
  input ctl_tx_port1_ctl_tx_send_lfi_in;
  input ctl_tx_port1_ctl_tx_send_rfi_in;
  input ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override;
  input [7:0]ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override_value;
  input ctl_tx_port2_ctl_tx_send_idle_in;
  input ctl_tx_port2_ctl_tx_send_lfi_in;
  input ctl_tx_port2_ctl_tx_send_rfi_in;
  input ctl_tx_port3_ctl_tx_send_idle_in;
  input ctl_tx_port3_ctl_tx_send_lfi_in;
  input ctl_tx_port3_ctl_tx_send_rfi_in;
  input [3:0]gt_reset_all_in;
  input [3:0]gt_reset_rx_datapath_in;
  input [3:0]gt_reset_tx_datapath_in;
  output [3:0]gt_rx_reset_done_out;
  input [3:0]gt_rxn_in_0;
  input [3:0]gt_rxp_in_0;
  output [3:0]gt_tx_reset_done_out;
  output [3:0]gt_txn_out_0;
  output [3:0]gt_txp_out_0;
  output gtpowergood;
  input gtpowergood_in;
  output [3:0]pm_rdy;
  input [3:0]pm_tick;
  input [3:0]rx_alt_serdes_clk;
  input [3:0]rx_axi_clk;
  output [63:0]rx_axis_tdata0;
  output [63:0]rx_axis_tdata1;
  output [63:0]rx_axis_tdata2;
  output [63:0]rx_axis_tdata3;
  output [63:0]rx_axis_tdata4;
  output [63:0]rx_axis_tdata5;
  output [63:0]rx_axis_tdata6;
  output [63:0]rx_axis_tdata7;
  output [10:0]rx_axis_tkeep_user0;
  output [10:0]rx_axis_tkeep_user1;
  output [10:0]rx_axis_tkeep_user2;
  output [10:0]rx_axis_tkeep_user3;
  output [10:0]rx_axis_tkeep_user4;
  output [10:0]rx_axis_tkeep_user5;
  output [10:0]rx_axis_tkeep_user6;
  output [10:0]rx_axis_tkeep_user7;
  output rx_axis_tlast_0;
  output rx_axis_tlast_1;
  output rx_axis_tlast_2;
  output rx_axis_tlast_3;
  output rx_axis_tvalid_0;
  output rx_axis_tvalid_1;
  output rx_axis_tvalid_2;
  output rx_axis_tvalid_3;
  input [3:0]rx_core_clk;
  input [3:0]rx_core_reset;
  input [3:0]rx_flexif_clk;
  input [3:0]rx_flexif_reset;
  output [55:0]rx_preambleout_rx_preambleout_0;
  output [55:0]rx_preambleout_rx_preambleout_1;
  output [55:0]rx_preambleout_rx_preambleout_2;
  output [55:0]rx_preambleout_rx_preambleout_3;
  input [3:0]rx_serdes_clk;
  input [3:0]rx_serdes_reset;
  input [3:0]rx_ts_clk;
  input s_axi_aclk;
  input [31:0]s_axi_araddr;
  input s_axi_aresetn;
  output s_axi_arready;
  input s_axi_arvalid;
  input [31:0]s_axi_awaddr;
  output s_axi_awready;
  input s_axi_awvalid;
  input s_axi_bready;
  output [1:0]s_axi_bresp;
  output s_axi_bvalid;
  output [31:0]s_axi_rdata;
  input s_axi_rready;
  output [1:0]s_axi_rresp;
  output s_axi_rvalid;
  input [31:0]s_axi_wdata;
  output s_axi_wready;
  input s_axi_wvalid;
  output stat_rx_port0_stat_rx_aligned;
  output stat_rx_port0_stat_rx_aligned_err;
  output stat_rx_port0_stat_rx_axis_err;
  output stat_rx_port0_stat_rx_axis_fifo_overflow;
  output stat_rx_port0_stat_rx_bad_code;
  output stat_rx_port0_stat_rx_bad_fcs;
  output stat_rx_port0_stat_rx_bad_preamble;
  output stat_rx_port0_stat_rx_bad_sfd;
  output [19:0]stat_rx_port0_stat_rx_bip_err;
  output [19:0]stat_rx_port0_stat_rx_block_lock;
  output stat_rx_port0_stat_rx_cl49_82_convert_err;
  output [1:0]stat_rx_port0_stat_rx_ecc_err;
  output stat_rx_port0_stat_rx_flex_fifo_ovf;
  output stat_rx_port0_stat_rx_flex_fifo_udf;
  output stat_rx_port0_stat_rx_flex_mon_fifo_ovf;
  output stat_rx_port0_stat_rx_flex_mon_fifo_udf;
  output stat_rx_port0_stat_rx_flexif_err;
  output [19:0]stat_rx_port0_stat_rx_framing_err_0;
  output stat_rx_port0_stat_rx_got_signal_os;
  output stat_rx_port0_stat_rx_hi_ber;
  output stat_rx_port0_stat_rx_internal_local_fault;
  output stat_rx_port0_stat_rx_invalid_start;
  output [7:0]stat_rx_port0_stat_rx_lane0_vlm_bip7;
  output stat_rx_port0_stat_rx_lane0_vlm_bip7_valid;
  output stat_rx_port0_stat_rx_local_fault;
  output [19:0]stat_rx_port0_stat_rx_mf_err_0;
  output stat_rx_port0_stat_rx_misaligned;
  output stat_rx_port0_stat_rx_pcs_bad_code;
  output stat_rx_port0_stat_rx_received_local_fault;
  output stat_rx_port0_stat_rx_remote_fault;
  output stat_rx_port0_stat_rx_status;
  output [19:0]stat_rx_port0_stat_rx_synced;
  output [19:0]stat_rx_port0_stat_rx_synced_err;
  output stat_rx_port0_stat_rx_test_pattern_mismatch;
  output stat_rx_port0_stat_rx_truncated;
  output stat_rx_port0_stat_rx_valid_ctrl_code;
  output stat_rx_port0_stat_rx_vl_demuxed;
  output stat_rx_port1_stat_rx_axis_err;
  output stat_rx_port1_stat_rx_axis_fifo_overflow;
  output stat_rx_port1_stat_rx_bad_code;
  output stat_rx_port1_stat_rx_bad_fcs;
  output stat_rx_port1_stat_rx_bad_preamble;
  output stat_rx_port1_stat_rx_bad_sfd;
  output stat_rx_port1_stat_rx_block_lock;
  output stat_rx_port1_stat_rx_cl49_82_convert_err;
  output [1:0]stat_rx_port1_stat_rx_ecc_err;
  output stat_rx_port1_stat_rx_flex_fifo_ovf;
  output stat_rx_port1_stat_rx_flex_fifo_udf;
  output stat_rx_port1_stat_rx_flex_mon_fifo_ovf;
  output stat_rx_port1_stat_rx_flex_mon_fifo_udf;
  output stat_rx_port1_stat_rx_flexif_err;
  output stat_rx_port1_stat_rx_framing_err_1;
  output stat_rx_port1_stat_rx_got_signal_os;
  output stat_rx_port1_stat_rx_hi_ber;
  output stat_rx_port1_stat_rx_internal_local_fault;
  output stat_rx_port1_stat_rx_invalid_start;
  output stat_rx_port1_stat_rx_local_fault;
  output stat_rx_port1_stat_rx_pcs_bad_code;
  output stat_rx_port1_stat_rx_received_local_fault;
  output stat_rx_port1_stat_rx_remote_fault;
  output stat_rx_port1_stat_rx_status;
  output stat_rx_port1_stat_rx_test_pattern_mismatch;
  output stat_rx_port1_stat_rx_truncated;
  output stat_rx_port1_stat_rx_valid_ctrl_code;
  output stat_rx_port2_stat_rx_aligned;
  output stat_rx_port2_stat_rx_aligned_err;
  output stat_rx_port2_stat_rx_axis_err;
  output stat_rx_port2_stat_rx_axis_fifo_overflow;
  output stat_rx_port2_stat_rx_bad_code;
  output stat_rx_port2_stat_rx_bad_fcs;
  output stat_rx_port2_stat_rx_bad_preamble;
  output stat_rx_port2_stat_rx_bad_sfd;
  output [3:0]stat_rx_port2_stat_rx_bip_err;
  output [3:0]stat_rx_port2_stat_rx_block_lock;
  output stat_rx_port2_stat_rx_cl49_82_convert_err;
  output [1:0]stat_rx_port2_stat_rx_ecc_err;
  output stat_rx_port2_stat_rx_flex_fifo_ovf;
  output stat_rx_port2_stat_rx_flex_fifo_udf;
  output stat_rx_port2_stat_rx_flex_mon_fifo_ovf;
  output stat_rx_port2_stat_rx_flex_mon_fifo_udf;
  output stat_rx_port2_stat_rx_flexif_err;
  output [3:0]stat_rx_port2_stat_rx_framing_err_2;
  output stat_rx_port2_stat_rx_got_signal_os;
  output stat_rx_port2_stat_rx_hi_ber;
  output stat_rx_port2_stat_rx_internal_local_fault;
  output stat_rx_port2_stat_rx_invalid_start;
  output [7:0]stat_rx_port2_stat_rx_lane0_vlm_bip7;
  output stat_rx_port2_stat_rx_lane0_vlm_bip7_valid;
  output stat_rx_port2_stat_rx_local_fault;
  output [3:0]stat_rx_port2_stat_rx_mf_err_2;
  output stat_rx_port2_stat_rx_misaligned;
  output stat_rx_port2_stat_rx_pcs_bad_code;
  output stat_rx_port2_stat_rx_received_local_fault;
  output stat_rx_port2_stat_rx_remote_fault;
  output stat_rx_port2_stat_rx_status;
  output [3:0]stat_rx_port2_stat_rx_synced;
  output [3:0]stat_rx_port2_stat_rx_synced_err;
  output stat_rx_port2_stat_rx_test_pattern_mismatch;
  output stat_rx_port2_stat_rx_truncated;
  output stat_rx_port2_stat_rx_valid_ctrl_code;
  output stat_rx_port2_stat_rx_vl_demuxed;
  output stat_rx_port3_stat_rx_axis_err;
  output stat_rx_port3_stat_rx_axis_fifo_overflow;
  output stat_rx_port3_stat_rx_bad_code;
  output stat_rx_port3_stat_rx_bad_fcs;
  output stat_rx_port3_stat_rx_bad_preamble;
  output stat_rx_port3_stat_rx_bad_sfd;
  output stat_rx_port3_stat_rx_block_lock;
  output stat_rx_port3_stat_rx_cl49_82_convert_err;
  output [1:0]stat_rx_port3_stat_rx_ecc_err;
  output stat_rx_port3_stat_rx_flex_fifo_ovf;
  output stat_rx_port3_stat_rx_flex_fifo_udf;
  output stat_rx_port3_stat_rx_flex_mon_fifo_ovf;
  output stat_rx_port3_stat_rx_flex_mon_fifo_udf;
  output stat_rx_port3_stat_rx_flexif_err;
  output stat_rx_port3_stat_rx_framing_err_3;
  output stat_rx_port3_stat_rx_got_signal_os;
  output stat_rx_port3_stat_rx_hi_ber;
  output stat_rx_port3_stat_rx_internal_local_fault;
  output stat_rx_port3_stat_rx_invalid_start;
  output stat_rx_port3_stat_rx_local_fault;
  output stat_rx_port3_stat_rx_pcs_bad_code;
  output stat_rx_port3_stat_rx_received_local_fault;
  output stat_rx_port3_stat_rx_remote_fault;
  output stat_rx_port3_stat_rx_status;
  output stat_rx_port3_stat_rx_test_pattern_mismatch;
  output stat_rx_port3_stat_rx_truncated;
  output stat_rx_port3_stat_rx_valid_ctrl_code;
  output stat_tx_port0_stat_tx_axis_err;
  output stat_tx_port0_stat_tx_axis_unf;
  output stat_tx_port0_stat_tx_bad_fcs;
  output stat_tx_port0_stat_tx_cl82_49_convert_err;
  output [1:0]stat_tx_port0_stat_tx_ecc_err;
  output stat_tx_port0_stat_tx_flex_fifo_ovf;
  output stat_tx_port0_stat_tx_flex_fifo_udf;
  output stat_tx_port0_stat_tx_flexif_err;
  output stat_tx_port0_stat_tx_frame_error;
  output stat_tx_port0_stat_tx_local_fault;
  output [2:0]stat_tx_port0_stat_tx_pcs_bad_code;
  output stat_tx_port1_stat_tx_axis_err;
  output stat_tx_port1_stat_tx_axis_unf;
  output stat_tx_port1_stat_tx_bad_fcs;
  output stat_tx_port1_stat_tx_cl82_49_convert_err;
  output [1:0]stat_tx_port1_stat_tx_ecc_err;
  output stat_tx_port1_stat_tx_flex_fifo_ovf;
  output stat_tx_port1_stat_tx_flex_fifo_udf;
  output stat_tx_port1_stat_tx_flexif_err;
  output stat_tx_port1_stat_tx_frame_error;
  output stat_tx_port1_stat_tx_local_fault;
  output [2:0]stat_tx_port1_stat_tx_pcs_bad_code;
  output stat_tx_port2_stat_tx_axis_err;
  output stat_tx_port2_stat_tx_axis_unf;
  output stat_tx_port2_stat_tx_bad_fcs;
  output stat_tx_port2_stat_tx_cl82_49_convert_err;
  output [1:0]stat_tx_port2_stat_tx_ecc_err;
  output stat_tx_port2_stat_tx_flex_fifo_ovf;
  output stat_tx_port2_stat_tx_flex_fifo_udf;
  output stat_tx_port2_stat_tx_flexif_err;
  output stat_tx_port2_stat_tx_frame_error;
  output stat_tx_port2_stat_tx_local_fault;
  output [2:0]stat_tx_port2_stat_tx_pcs_bad_code;
  output stat_tx_port3_stat_tx_axis_err;
  output stat_tx_port3_stat_tx_axis_unf;
  output stat_tx_port3_stat_tx_bad_fcs;
  output stat_tx_port3_stat_tx_cl82_49_convert_err;
  output [1:0]stat_tx_port3_stat_tx_ecc_err;
  output stat_tx_port3_stat_tx_flex_fifo_ovf;
  output stat_tx_port3_stat_tx_flex_fifo_udf;
  output stat_tx_port3_stat_tx_flexif_err;
  output stat_tx_port3_stat_tx_frame_error;
  output stat_tx_port3_stat_tx_local_fault;
  output [2:0]stat_tx_port3_stat_tx_pcs_bad_code;
  input [3:0]tx_alt_serdes_clk;
  input [3:0]tx_axi_clk;
  input [63:0]tx_axis_tdata0;
  input [63:0]tx_axis_tdata1;
  input [63:0]tx_axis_tdata2;
  input [63:0]tx_axis_tdata3;
  input [63:0]tx_axis_tdata4;
  input [63:0]tx_axis_tdata5;
  input [63:0]tx_axis_tdata6;
  input [63:0]tx_axis_tdata7;
  input [10:0]tx_axis_tkeep_user0;
  input [10:0]tx_axis_tkeep_user1;
  input [10:0]tx_axis_tkeep_user2;
  input [10:0]tx_axis_tkeep_user3;
  input [10:0]tx_axis_tkeep_user4;
  input [10:0]tx_axis_tkeep_user5;
  input [10:0]tx_axis_tkeep_user6;
  input [10:0]tx_axis_tkeep_user7;
  input tx_axis_tlast_0;
  input tx_axis_tlast_1;
  input tx_axis_tlast_2;
  input tx_axis_tlast_3;
  output tx_axis_tready_0;
  output tx_axis_tready_1;
  output tx_axis_tready_2;
  output tx_axis_tready_3;
  input tx_axis_tvalid_0;
  input tx_axis_tvalid_1;
  input tx_axis_tvalid_2;
  input tx_axis_tvalid_3;
  input [3:0]tx_core_clk;
  input [3:0]tx_core_reset;
  input [3:0]tx_flexif_clk;
  input [55:0]tx_preamblein_tx_preamblein_0;
  input [55:0]tx_preamblein_tx_preamblein_1;
  input [55:0]tx_preamblein_tx_preamblein_2;
  input [55:0]tx_preamblein_tx_preamblein_3;
  input [3:0]tx_serdes_reset;
  input [3:0]tx_ts_clk;

  wire [15:0]APB3_INTF_paddr;
  wire APB3_INTF_penable;
  wire [31:0]APB3_INTF_prdata;
  wire APB3_INTF_pready;
  wire APB3_INTF_psel;
  wire APB3_INTF_pslverr;
  wire [31:0]APB3_INTF_pwdata;
  wire APB3_INTF_pwrite;
  wire [0:0]CLK_IN_D_clk_n;
  wire [0:0]CLK_IN_D_clk_p;
  wire apb3clk_quad;
  wire [2:0]ch0_loopback;
  wire [0:0]ch0_rx_usr_clk;
  wire [0:0]ch0_rx_usr_clk2;
  wire [7:0]ch0_rxrate;
  wire ch0_rxusrclk;
  wire [0:0]ch0_tx_usr_clk;
  wire [0:0]ch0_tx_usr_clk2;
  wire [7:0]ch0_txrate;
  wire ch0_txusrclk;
  wire [2:0]ch1_loopback;
  wire [0:0]ch1_rx_usr_clk;
  wire [0:0]ch1_rx_usr_clk2;
  wire [7:0]ch1_rxrate;
  wire ch1_rxusrclk;
  wire [7:0]ch1_txrate;
  wire ch1_txusrclk;
  wire [2:0]ch2_loopback;
  wire [0:0]ch2_rx_usr_clk;
  wire [0:0]ch2_rx_usr_clk2;
  wire [7:0]ch2_rxrate;
  wire ch2_rxusrclk;
  wire [7:0]ch2_txrate;
  wire ch2_txusrclk;
  wire [2:0]ch3_loopback;
  wire [0:0]ch3_rx_usr_clk;
  wire [0:0]ch3_rx_usr_clk2;
  wire [7:0]ch3_rxrate;
  wire ch3_rxusrclk;
  wire [7:0]ch3_txrate;
  wire ch3_txusrclk;
  wire ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override;
  wire [7:0]ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override_value;
  wire ctl_tx_port0_ctl_tx_send_idle_in;
  wire ctl_tx_port0_ctl_tx_send_lfi_in;
  wire ctl_tx_port0_ctl_tx_send_rfi_in;
  wire ctl_tx_port1_ctl_tx_send_idle_in;
  wire ctl_tx_port1_ctl_tx_send_lfi_in;
  wire ctl_tx_port1_ctl_tx_send_rfi_in;
  wire ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override;
  wire [7:0]ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override_value;
  wire ctl_tx_port2_ctl_tx_send_idle_in;
  wire ctl_tx_port2_ctl_tx_send_lfi_in;
  wire ctl_tx_port2_ctl_tx_send_rfi_in;
  wire ctl_tx_port3_ctl_tx_send_idle_in;
  wire ctl_tx_port3_ctl_tx_send_lfi_in;
  wire ctl_tx_port3_ctl_tx_send_rfi_in;
  wire [3:0]gt_reset_all_in;
  wire [3:0]gt_reset_rx_datapath_in;
  wire [3:0]gt_reset_tx_datapath_in;
  wire [3:0]gt_rx_reset_done_out;
  wire [3:0]gt_rxn_in_0;
  wire [3:0]gt_rxp_in_0;
  wire [3:0]gt_tx_reset_done_out;
  wire [3:0]gt_txn_out_0;
  wire [3:0]gt_txp_out_0;
  wire gtpowergood;
  wire gtpowergood_in;
  wire [3:0]pm_rdy;
  wire [3:0]pm_tick;
  wire [3:0]rx_alt_serdes_clk;
  wire [3:0]rx_axi_clk;
  wire [63:0]rx_axis_tdata0;
  wire [63:0]rx_axis_tdata1;
  wire [63:0]rx_axis_tdata2;
  wire [63:0]rx_axis_tdata3;
  wire [63:0]rx_axis_tdata4;
  wire [63:0]rx_axis_tdata5;
  wire [63:0]rx_axis_tdata6;
  wire [63:0]rx_axis_tdata7;
  wire [10:0]rx_axis_tkeep_user0;
  wire [10:0]rx_axis_tkeep_user1;
  wire [10:0]rx_axis_tkeep_user2;
  wire [10:0]rx_axis_tkeep_user3;
  wire [10:0]rx_axis_tkeep_user4;
  wire [10:0]rx_axis_tkeep_user5;
  wire [10:0]rx_axis_tkeep_user6;
  wire [10:0]rx_axis_tkeep_user7;
  wire rx_axis_tlast_0;
  wire rx_axis_tlast_1;
  wire rx_axis_tlast_2;
  wire rx_axis_tlast_3;
  wire rx_axis_tvalid_0;
  wire rx_axis_tvalid_1;
  wire rx_axis_tvalid_2;
  wire rx_axis_tvalid_3;
  wire [3:0]rx_core_clk;
  wire [3:0]rx_core_reset;
  wire [3:0]rx_flexif_clk;
  wire [3:0]rx_flexif_reset;
  wire [55:0]rx_preambleout_rx_preambleout_0;
  wire [55:0]rx_preambleout_rx_preambleout_1;
  wire [55:0]rx_preambleout_rx_preambleout_2;
  wire [55:0]rx_preambleout_rx_preambleout_3;
  wire [3:0]rx_serdes_clk;
  wire [3:0]rx_serdes_reset;
  wire [3:0]rx_ts_clk;
  wire s_axi_aclk;
  wire [31:0]s_axi_araddr;
  wire s_axi_aresetn;
  wire s_axi_arready;
  wire s_axi_arvalid;
  wire [31:0]s_axi_awaddr;
  wire s_axi_awready;
  wire s_axi_awvalid;
  wire s_axi_bready;
  wire [1:0]s_axi_bresp;
  wire s_axi_bvalid;
  wire [31:0]s_axi_rdata;
  wire s_axi_rready;
  wire [1:0]s_axi_rresp;
  wire s_axi_rvalid;
  wire [31:0]s_axi_wdata;
  wire s_axi_wready;
  wire s_axi_wvalid;
  wire stat_rx_port0_stat_rx_aligned;
  wire stat_rx_port0_stat_rx_aligned_err;
  wire stat_rx_port0_stat_rx_axis_err;
  wire stat_rx_port0_stat_rx_axis_fifo_overflow;
  wire stat_rx_port0_stat_rx_bad_code;
  wire stat_rx_port0_stat_rx_bad_fcs;
  wire stat_rx_port0_stat_rx_bad_preamble;
  wire stat_rx_port0_stat_rx_bad_sfd;
  wire [19:0]stat_rx_port0_stat_rx_bip_err;
  wire [19:0]stat_rx_port0_stat_rx_block_lock;
  wire stat_rx_port0_stat_rx_cl49_82_convert_err;
  wire [1:0]stat_rx_port0_stat_rx_ecc_err;
  wire stat_rx_port0_stat_rx_flex_fifo_ovf;
  wire stat_rx_port0_stat_rx_flex_fifo_udf;
  wire stat_rx_port0_stat_rx_flex_mon_fifo_ovf;
  wire stat_rx_port0_stat_rx_flex_mon_fifo_udf;
  wire stat_rx_port0_stat_rx_flexif_err;
  wire [19:0]stat_rx_port0_stat_rx_framing_err_0;
  wire stat_rx_port0_stat_rx_got_signal_os;
  wire stat_rx_port0_stat_rx_hi_ber;
  wire stat_rx_port0_stat_rx_internal_local_fault;
  wire stat_rx_port0_stat_rx_invalid_start;
  wire [7:0]stat_rx_port0_stat_rx_lane0_vlm_bip7;
  wire stat_rx_port0_stat_rx_lane0_vlm_bip7_valid;
  wire stat_rx_port0_stat_rx_local_fault;
  wire [19:0]stat_rx_port0_stat_rx_mf_err_0;
  wire stat_rx_port0_stat_rx_misaligned;
  wire stat_rx_port0_stat_rx_pcs_bad_code;
  wire stat_rx_port0_stat_rx_received_local_fault;
  wire stat_rx_port0_stat_rx_remote_fault;
  wire stat_rx_port0_stat_rx_status;
  wire [19:0]stat_rx_port0_stat_rx_synced;
  wire [19:0]stat_rx_port0_stat_rx_synced_err;
  wire stat_rx_port0_stat_rx_test_pattern_mismatch;
  wire stat_rx_port0_stat_rx_truncated;
  wire stat_rx_port0_stat_rx_valid_ctrl_code;
  wire stat_rx_port0_stat_rx_vl_demuxed;
  wire stat_rx_port1_stat_rx_axis_err;
  wire stat_rx_port1_stat_rx_axis_fifo_overflow;
  wire stat_rx_port1_stat_rx_bad_code;
  wire stat_rx_port1_stat_rx_bad_fcs;
  wire stat_rx_port1_stat_rx_bad_preamble;
  wire stat_rx_port1_stat_rx_bad_sfd;
  wire stat_rx_port1_stat_rx_block_lock;
  wire stat_rx_port1_stat_rx_cl49_82_convert_err;
  wire [1:0]stat_rx_port1_stat_rx_ecc_err;
  wire stat_rx_port1_stat_rx_flex_fifo_ovf;
  wire stat_rx_port1_stat_rx_flex_fifo_udf;
  wire stat_rx_port1_stat_rx_flex_mon_fifo_ovf;
  wire stat_rx_port1_stat_rx_flex_mon_fifo_udf;
  wire stat_rx_port1_stat_rx_flexif_err;
  wire stat_rx_port1_stat_rx_framing_err_1;
  wire stat_rx_port1_stat_rx_got_signal_os;
  wire stat_rx_port1_stat_rx_hi_ber;
  wire stat_rx_port1_stat_rx_internal_local_fault;
  wire stat_rx_port1_stat_rx_invalid_start;
  wire stat_rx_port1_stat_rx_local_fault;
  wire stat_rx_port1_stat_rx_pcs_bad_code;
  wire stat_rx_port1_stat_rx_received_local_fault;
  wire stat_rx_port1_stat_rx_remote_fault;
  wire stat_rx_port1_stat_rx_status;
  wire stat_rx_port1_stat_rx_test_pattern_mismatch;
  wire stat_rx_port1_stat_rx_truncated;
  wire stat_rx_port1_stat_rx_valid_ctrl_code;
  wire stat_rx_port2_stat_rx_aligned;
  wire stat_rx_port2_stat_rx_aligned_err;
  wire stat_rx_port2_stat_rx_axis_err;
  wire stat_rx_port2_stat_rx_axis_fifo_overflow;
  wire stat_rx_port2_stat_rx_bad_code;
  wire stat_rx_port2_stat_rx_bad_fcs;
  wire stat_rx_port2_stat_rx_bad_preamble;
  wire stat_rx_port2_stat_rx_bad_sfd;
  wire [3:0]stat_rx_port2_stat_rx_bip_err;
  wire [3:0]stat_rx_port2_stat_rx_block_lock;
  wire stat_rx_port2_stat_rx_cl49_82_convert_err;
  wire [1:0]stat_rx_port2_stat_rx_ecc_err;
  wire stat_rx_port2_stat_rx_flex_fifo_ovf;
  wire stat_rx_port2_stat_rx_flex_fifo_udf;
  wire stat_rx_port2_stat_rx_flex_mon_fifo_ovf;
  wire stat_rx_port2_stat_rx_flex_mon_fifo_udf;
  wire stat_rx_port2_stat_rx_flexif_err;
  wire [3:0]stat_rx_port2_stat_rx_framing_err_2;
  wire stat_rx_port2_stat_rx_got_signal_os;
  wire stat_rx_port2_stat_rx_hi_ber;
  wire stat_rx_port2_stat_rx_internal_local_fault;
  wire stat_rx_port2_stat_rx_invalid_start;
  wire [7:0]stat_rx_port2_stat_rx_lane0_vlm_bip7;
  wire stat_rx_port2_stat_rx_lane0_vlm_bip7_valid;
  wire stat_rx_port2_stat_rx_local_fault;
  wire [3:0]stat_rx_port2_stat_rx_mf_err_2;
  wire stat_rx_port2_stat_rx_misaligned;
  wire stat_rx_port2_stat_rx_pcs_bad_code;
  wire stat_rx_port2_stat_rx_received_local_fault;
  wire stat_rx_port2_stat_rx_remote_fault;
  wire stat_rx_port2_stat_rx_status;
  wire [3:0]stat_rx_port2_stat_rx_synced;
  wire [3:0]stat_rx_port2_stat_rx_synced_err;
  wire stat_rx_port2_stat_rx_test_pattern_mismatch;
  wire stat_rx_port2_stat_rx_truncated;
  wire stat_rx_port2_stat_rx_valid_ctrl_code;
  wire stat_rx_port2_stat_rx_vl_demuxed;
  wire stat_rx_port3_stat_rx_axis_err;
  wire stat_rx_port3_stat_rx_axis_fifo_overflow;
  wire stat_rx_port3_stat_rx_bad_code;
  wire stat_rx_port3_stat_rx_bad_fcs;
  wire stat_rx_port3_stat_rx_bad_preamble;
  wire stat_rx_port3_stat_rx_bad_sfd;
  wire stat_rx_port3_stat_rx_block_lock;
  wire stat_rx_port3_stat_rx_cl49_82_convert_err;
  wire [1:0]stat_rx_port3_stat_rx_ecc_err;
  wire stat_rx_port3_stat_rx_flex_fifo_ovf;
  wire stat_rx_port3_stat_rx_flex_fifo_udf;
  wire stat_rx_port3_stat_rx_flex_mon_fifo_ovf;
  wire stat_rx_port3_stat_rx_flex_mon_fifo_udf;
  wire stat_rx_port3_stat_rx_flexif_err;
  wire stat_rx_port3_stat_rx_framing_err_3;
  wire stat_rx_port3_stat_rx_got_signal_os;
  wire stat_rx_port3_stat_rx_hi_ber;
  wire stat_rx_port3_stat_rx_internal_local_fault;
  wire stat_rx_port3_stat_rx_invalid_start;
  wire stat_rx_port3_stat_rx_local_fault;
  wire stat_rx_port3_stat_rx_pcs_bad_code;
  wire stat_rx_port3_stat_rx_received_local_fault;
  wire stat_rx_port3_stat_rx_remote_fault;
  wire stat_rx_port3_stat_rx_status;
  wire stat_rx_port3_stat_rx_test_pattern_mismatch;
  wire stat_rx_port3_stat_rx_truncated;
  wire stat_rx_port3_stat_rx_valid_ctrl_code;
  wire stat_tx_port0_stat_tx_axis_err;
  wire stat_tx_port0_stat_tx_axis_unf;
  wire stat_tx_port0_stat_tx_bad_fcs;
  wire stat_tx_port0_stat_tx_cl82_49_convert_err;
  wire [1:0]stat_tx_port0_stat_tx_ecc_err;
  wire stat_tx_port0_stat_tx_flex_fifo_ovf;
  wire stat_tx_port0_stat_tx_flex_fifo_udf;
  wire stat_tx_port0_stat_tx_flexif_err;
  wire stat_tx_port0_stat_tx_frame_error;
  wire stat_tx_port0_stat_tx_local_fault;
  wire [2:0]stat_tx_port0_stat_tx_pcs_bad_code;
  wire stat_tx_port1_stat_tx_axis_err;
  wire stat_tx_port1_stat_tx_axis_unf;
  wire stat_tx_port1_stat_tx_bad_fcs;
  wire stat_tx_port1_stat_tx_cl82_49_convert_err;
  wire [1:0]stat_tx_port1_stat_tx_ecc_err;
  wire stat_tx_port1_stat_tx_flex_fifo_ovf;
  wire stat_tx_port1_stat_tx_flex_fifo_udf;
  wire stat_tx_port1_stat_tx_flexif_err;
  wire stat_tx_port1_stat_tx_frame_error;
  wire stat_tx_port1_stat_tx_local_fault;
  wire [2:0]stat_tx_port1_stat_tx_pcs_bad_code;
  wire stat_tx_port2_stat_tx_axis_err;
  wire stat_tx_port2_stat_tx_axis_unf;
  wire stat_tx_port2_stat_tx_bad_fcs;
  wire stat_tx_port2_stat_tx_cl82_49_convert_err;
  wire [1:0]stat_tx_port2_stat_tx_ecc_err;
  wire stat_tx_port2_stat_tx_flex_fifo_ovf;
  wire stat_tx_port2_stat_tx_flex_fifo_udf;
  wire stat_tx_port2_stat_tx_flexif_err;
  wire stat_tx_port2_stat_tx_frame_error;
  wire stat_tx_port2_stat_tx_local_fault;
  wire [2:0]stat_tx_port2_stat_tx_pcs_bad_code;
  wire stat_tx_port3_stat_tx_axis_err;
  wire stat_tx_port3_stat_tx_axis_unf;
  wire stat_tx_port3_stat_tx_bad_fcs;
  wire stat_tx_port3_stat_tx_cl82_49_convert_err;
  wire [1:0]stat_tx_port3_stat_tx_ecc_err;
  wire stat_tx_port3_stat_tx_flex_fifo_ovf;
  wire stat_tx_port3_stat_tx_flex_fifo_udf;
  wire stat_tx_port3_stat_tx_flexif_err;
  wire stat_tx_port3_stat_tx_frame_error;
  wire stat_tx_port3_stat_tx_local_fault;
  wire [2:0]stat_tx_port3_stat_tx_pcs_bad_code;
  wire [3:0]tx_alt_serdes_clk;
  wire [3:0]tx_axi_clk;
  wire [63:0]tx_axis_tdata0;
  wire [63:0]tx_axis_tdata1;
  wire [63:0]tx_axis_tdata2;
  wire [63:0]tx_axis_tdata3;
  wire [63:0]tx_axis_tdata4;
  wire [63:0]tx_axis_tdata5;
  wire [63:0]tx_axis_tdata6;
  wire [63:0]tx_axis_tdata7;
  wire [10:0]tx_axis_tkeep_user0;
  wire [10:0]tx_axis_tkeep_user1;
  wire [10:0]tx_axis_tkeep_user2;
  wire [10:0]tx_axis_tkeep_user3;
  wire [10:0]tx_axis_tkeep_user4;
  wire [10:0]tx_axis_tkeep_user5;
  wire [10:0]tx_axis_tkeep_user6;
  wire [10:0]tx_axis_tkeep_user7;
  wire tx_axis_tlast_0;
  wire tx_axis_tlast_1;
  wire tx_axis_tlast_2;
  wire tx_axis_tlast_3;
  wire tx_axis_tready_0;
  wire tx_axis_tready_1;
  wire tx_axis_tready_2;
  wire tx_axis_tready_3;
  wire tx_axis_tvalid_0;
  wire tx_axis_tvalid_1;
  wire tx_axis_tvalid_2;
  wire tx_axis_tvalid_3;
  wire [3:0]tx_core_clk;
  wire [3:0]tx_core_reset;
  wire [3:0]tx_flexif_clk;
  wire [55:0]tx_preamblein_tx_preamblein_0;
  wire [55:0]tx_preamblein_tx_preamblein_1;
  wire [55:0]tx_preamblein_tx_preamblein_2;
  wire [55:0]tx_preamblein_tx_preamblein_3;
  wire [3:0]tx_serdes_reset;
  wire [3:0]tx_ts_clk;

  mrmac_0_exdes_support mrmac_0_exdes_support_i
       (.APB3_INTF_paddr(APB3_INTF_paddr),
        .APB3_INTF_penable(APB3_INTF_penable),
        .APB3_INTF_prdata(APB3_INTF_prdata),
        .APB3_INTF_pready(APB3_INTF_pready),
        .APB3_INTF_psel(APB3_INTF_psel),
        .APB3_INTF_pslverr(APB3_INTF_pslverr),
        .APB3_INTF_pwdata(APB3_INTF_pwdata),
        .APB3_INTF_pwrite(APB3_INTF_pwrite),
        .CLK_IN_D_clk_n(CLK_IN_D_clk_n),
        .CLK_IN_D_clk_p(CLK_IN_D_clk_p),
        .apb3clk_quad(apb3clk_quad),
        .ch0_loopback(ch0_loopback),
        .ch0_rx_usr_clk(ch0_rx_usr_clk),
        .ch0_rx_usr_clk2(ch0_rx_usr_clk2),
        .ch0_rxrate(ch0_rxrate),
        .ch0_rxusrclk(ch0_rxusrclk),
        .ch0_tx_usr_clk(ch0_tx_usr_clk),
        .ch0_tx_usr_clk2(ch0_tx_usr_clk2),
        .ch0_txrate(ch0_txrate),
        .ch0_txusrclk(ch0_txusrclk),
        .ch1_loopback(ch1_loopback),
        .ch1_rx_usr_clk(ch1_rx_usr_clk),
        .ch1_rx_usr_clk2(ch1_rx_usr_clk2),
        .ch1_rxrate(ch1_rxrate),
        .ch1_rxusrclk(ch1_rxusrclk),
        .ch1_txrate(ch1_txrate),
        .ch1_txusrclk(ch1_txusrclk),
        .ch2_loopback(ch2_loopback),
        .ch2_rx_usr_clk(ch2_rx_usr_clk),
        .ch2_rx_usr_clk2(ch2_rx_usr_clk2),
        .ch2_rxrate(ch2_rxrate),
        .ch2_rxusrclk(ch2_rxusrclk),
        .ch2_txrate(ch2_txrate),
        .ch2_txusrclk(ch2_txusrclk),
        .ch3_loopback(ch3_loopback),
        .ch3_rx_usr_clk(ch3_rx_usr_clk),
        .ch3_rx_usr_clk2(ch3_rx_usr_clk2),
        .ch3_rxrate(ch3_rxrate),
        .ch3_rxusrclk(ch3_rxusrclk),
        .ch3_txrate(ch3_txrate),
        .ch3_txusrclk(ch3_txusrclk),
        .ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override(ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override),
        .ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override_value(ctl_tx_port0_ctl_tx_lane0_vlm_bip7_override_value),
        .ctl_tx_port0_ctl_tx_send_idle_in(ctl_tx_port0_ctl_tx_send_idle_in),
        .ctl_tx_port0_ctl_tx_send_lfi_in(ctl_tx_port0_ctl_tx_send_lfi_in),
        .ctl_tx_port0_ctl_tx_send_rfi_in(ctl_tx_port0_ctl_tx_send_rfi_in),
        .ctl_tx_port1_ctl_tx_send_idle_in(ctl_tx_port1_ctl_tx_send_idle_in),
        .ctl_tx_port1_ctl_tx_send_lfi_in(ctl_tx_port1_ctl_tx_send_lfi_in),
        .ctl_tx_port1_ctl_tx_send_rfi_in(ctl_tx_port1_ctl_tx_send_rfi_in),
        .ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override(ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override),
        .ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override_value(ctl_tx_port2_ctl_tx_lane0_vlm_bip7_override_value),
        .ctl_tx_port2_ctl_tx_send_idle_in(ctl_tx_port2_ctl_tx_send_idle_in),
        .ctl_tx_port2_ctl_tx_send_lfi_in(ctl_tx_port2_ctl_tx_send_lfi_in),
        .ctl_tx_port2_ctl_tx_send_rfi_in(ctl_tx_port2_ctl_tx_send_rfi_in),
        .ctl_tx_port3_ctl_tx_send_idle_in(ctl_tx_port3_ctl_tx_send_idle_in),
        .ctl_tx_port3_ctl_tx_send_lfi_in(ctl_tx_port3_ctl_tx_send_lfi_in),
        .ctl_tx_port3_ctl_tx_send_rfi_in(ctl_tx_port3_ctl_tx_send_rfi_in),
        .gt_reset_all_in(gt_reset_all_in),
        .gt_reset_rx_datapath_in(gt_reset_rx_datapath_in),
        .gt_reset_tx_datapath_in(gt_reset_tx_datapath_in),
        .gt_rx_reset_done_out(gt_rx_reset_done_out),
        .gt_rxn_in_0(gt_rxn_in_0),
        .gt_rxp_in_0(gt_rxp_in_0),
        .gt_tx_reset_done_out(gt_tx_reset_done_out),
        .gt_txn_out_0(gt_txn_out_0),
        .gt_txp_out_0(gt_txp_out_0),
        .gtpowergood(gtpowergood),
        .gtpowergood_in(gtpowergood_in),
        .pm_rdy(pm_rdy),
        .pm_tick(pm_tick),
        .rx_alt_serdes_clk(rx_alt_serdes_clk),
        .rx_axi_clk(rx_axi_clk),
        .rx_axis_tdata0(rx_axis_tdata0),
        .rx_axis_tdata1(rx_axis_tdata1),
        .rx_axis_tdata2(rx_axis_tdata2),
        .rx_axis_tdata3(rx_axis_tdata3),
        .rx_axis_tdata4(rx_axis_tdata4),
        .rx_axis_tdata5(rx_axis_tdata5),
        .rx_axis_tdata6(rx_axis_tdata6),
        .rx_axis_tdata7(rx_axis_tdata7),
        .rx_axis_tkeep_user0(rx_axis_tkeep_user0),
        .rx_axis_tkeep_user1(rx_axis_tkeep_user1),
        .rx_axis_tkeep_user2(rx_axis_tkeep_user2),
        .rx_axis_tkeep_user3(rx_axis_tkeep_user3),
        .rx_axis_tkeep_user4(rx_axis_tkeep_user4),
        .rx_axis_tkeep_user5(rx_axis_tkeep_user5),
        .rx_axis_tkeep_user6(rx_axis_tkeep_user6),
        .rx_axis_tkeep_user7(rx_axis_tkeep_user7),
        .rx_axis_tlast_0(rx_axis_tlast_0),
        .rx_axis_tlast_1(rx_axis_tlast_1),
        .rx_axis_tlast_2(rx_axis_tlast_2),
        .rx_axis_tlast_3(rx_axis_tlast_3),
        .rx_axis_tvalid_0(rx_axis_tvalid_0),
        .rx_axis_tvalid_1(rx_axis_tvalid_1),
        .rx_axis_tvalid_2(rx_axis_tvalid_2),
        .rx_axis_tvalid_3(rx_axis_tvalid_3),
        .rx_core_clk(rx_core_clk),
        .rx_core_reset(rx_core_reset),
        .rx_flexif_clk(rx_flexif_clk),
        .rx_flexif_reset(rx_flexif_reset),
        .rx_preambleout_rx_preambleout_0(rx_preambleout_rx_preambleout_0),
        .rx_preambleout_rx_preambleout_1(rx_preambleout_rx_preambleout_1),
        .rx_preambleout_rx_preambleout_2(rx_preambleout_rx_preambleout_2),
        .rx_preambleout_rx_preambleout_3(rx_preambleout_rx_preambleout_3),
        .rx_serdes_clk(rx_serdes_clk),
        .rx_serdes_reset(rx_serdes_reset),
        .rx_ts_clk(rx_ts_clk),
        .s_axi_aclk(s_axi_aclk),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_arready(s_axi_arready),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awready(s_axi_awready),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rready(s_axi_rready),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wready(s_axi_wready),
        .s_axi_wvalid(s_axi_wvalid),
        .stat_rx_port0_stat_rx_aligned(stat_rx_port0_stat_rx_aligned),
        .stat_rx_port0_stat_rx_aligned_err(stat_rx_port0_stat_rx_aligned_err),
        .stat_rx_port0_stat_rx_axis_err(stat_rx_port0_stat_rx_axis_err),
        .stat_rx_port0_stat_rx_axis_fifo_overflow(stat_rx_port0_stat_rx_axis_fifo_overflow),
        .stat_rx_port0_stat_rx_bad_code(stat_rx_port0_stat_rx_bad_code),
        .stat_rx_port0_stat_rx_bad_fcs(stat_rx_port0_stat_rx_bad_fcs),
        .stat_rx_port0_stat_rx_bad_preamble(stat_rx_port0_stat_rx_bad_preamble),
        .stat_rx_port0_stat_rx_bad_sfd(stat_rx_port0_stat_rx_bad_sfd),
        .stat_rx_port0_stat_rx_bip_err(stat_rx_port0_stat_rx_bip_err),
        .stat_rx_port0_stat_rx_block_lock(stat_rx_port0_stat_rx_block_lock),
        .stat_rx_port0_stat_rx_cl49_82_convert_err(stat_rx_port0_stat_rx_cl49_82_convert_err),
        .stat_rx_port0_stat_rx_ecc_err(stat_rx_port0_stat_rx_ecc_err),
        .stat_rx_port0_stat_rx_flex_fifo_ovf(stat_rx_port0_stat_rx_flex_fifo_ovf),
        .stat_rx_port0_stat_rx_flex_fifo_udf(stat_rx_port0_stat_rx_flex_fifo_udf),
        .stat_rx_port0_stat_rx_flex_mon_fifo_ovf(stat_rx_port0_stat_rx_flex_mon_fifo_ovf),
        .stat_rx_port0_stat_rx_flex_mon_fifo_udf(stat_rx_port0_stat_rx_flex_mon_fifo_udf),
        .stat_rx_port0_stat_rx_flexif_err(stat_rx_port0_stat_rx_flexif_err),
        .stat_rx_port0_stat_rx_framing_err_0(stat_rx_port0_stat_rx_framing_err_0),
        .stat_rx_port0_stat_rx_got_signal_os(stat_rx_port0_stat_rx_got_signal_os),
        .stat_rx_port0_stat_rx_hi_ber(stat_rx_port0_stat_rx_hi_ber),
        .stat_rx_port0_stat_rx_internal_local_fault(stat_rx_port0_stat_rx_internal_local_fault),
        .stat_rx_port0_stat_rx_invalid_start(stat_rx_port0_stat_rx_invalid_start),
        .stat_rx_port0_stat_rx_lane0_vlm_bip7(stat_rx_port0_stat_rx_lane0_vlm_bip7),
        .stat_rx_port0_stat_rx_lane0_vlm_bip7_valid(stat_rx_port0_stat_rx_lane0_vlm_bip7_valid),
        .stat_rx_port0_stat_rx_local_fault(stat_rx_port0_stat_rx_local_fault),
        .stat_rx_port0_stat_rx_mf_err_0(stat_rx_port0_stat_rx_mf_err_0),
        .stat_rx_port0_stat_rx_misaligned(stat_rx_port0_stat_rx_misaligned),
        .stat_rx_port0_stat_rx_pcs_bad_code(stat_rx_port0_stat_rx_pcs_bad_code),
        .stat_rx_port0_stat_rx_received_local_fault(stat_rx_port0_stat_rx_received_local_fault),
        .stat_rx_port0_stat_rx_remote_fault(stat_rx_port0_stat_rx_remote_fault),
        .stat_rx_port0_stat_rx_status(stat_rx_port0_stat_rx_status),
        .stat_rx_port0_stat_rx_synced(stat_rx_port0_stat_rx_synced),
        .stat_rx_port0_stat_rx_synced_err(stat_rx_port0_stat_rx_synced_err),
        .stat_rx_port0_stat_rx_test_pattern_mismatch(stat_rx_port0_stat_rx_test_pattern_mismatch),
        .stat_rx_port0_stat_rx_truncated(stat_rx_port0_stat_rx_truncated),
        .stat_rx_port0_stat_rx_valid_ctrl_code(stat_rx_port0_stat_rx_valid_ctrl_code),
        .stat_rx_port0_stat_rx_vl_demuxed(stat_rx_port0_stat_rx_vl_demuxed),
        .stat_rx_port1_stat_rx_axis_err(stat_rx_port1_stat_rx_axis_err),
        .stat_rx_port1_stat_rx_axis_fifo_overflow(stat_rx_port1_stat_rx_axis_fifo_overflow),
        .stat_rx_port1_stat_rx_bad_code(stat_rx_port1_stat_rx_bad_code),
        .stat_rx_port1_stat_rx_bad_fcs(stat_rx_port1_stat_rx_bad_fcs),
        .stat_rx_port1_stat_rx_bad_preamble(stat_rx_port1_stat_rx_bad_preamble),
        .stat_rx_port1_stat_rx_bad_sfd(stat_rx_port1_stat_rx_bad_sfd),
        .stat_rx_port1_stat_rx_block_lock(stat_rx_port1_stat_rx_block_lock),
        .stat_rx_port1_stat_rx_cl49_82_convert_err(stat_rx_port1_stat_rx_cl49_82_convert_err),
        .stat_rx_port1_stat_rx_ecc_err(stat_rx_port1_stat_rx_ecc_err),
        .stat_rx_port1_stat_rx_flex_fifo_ovf(stat_rx_port1_stat_rx_flex_fifo_ovf),
        .stat_rx_port1_stat_rx_flex_fifo_udf(stat_rx_port1_stat_rx_flex_fifo_udf),
        .stat_rx_port1_stat_rx_flex_mon_fifo_ovf(stat_rx_port1_stat_rx_flex_mon_fifo_ovf),
        .stat_rx_port1_stat_rx_flex_mon_fifo_udf(stat_rx_port1_stat_rx_flex_mon_fifo_udf),
        .stat_rx_port1_stat_rx_flexif_err(stat_rx_port1_stat_rx_flexif_err),
        .stat_rx_port1_stat_rx_framing_err_1(stat_rx_port1_stat_rx_framing_err_1),
        .stat_rx_port1_stat_rx_got_signal_os(stat_rx_port1_stat_rx_got_signal_os),
        .stat_rx_port1_stat_rx_hi_ber(stat_rx_port1_stat_rx_hi_ber),
        .stat_rx_port1_stat_rx_internal_local_fault(stat_rx_port1_stat_rx_internal_local_fault),
        .stat_rx_port1_stat_rx_invalid_start(stat_rx_port1_stat_rx_invalid_start),
        .stat_rx_port1_stat_rx_local_fault(stat_rx_port1_stat_rx_local_fault),
        .stat_rx_port1_stat_rx_pcs_bad_code(stat_rx_port1_stat_rx_pcs_bad_code),
        .stat_rx_port1_stat_rx_received_local_fault(stat_rx_port1_stat_rx_received_local_fault),
        .stat_rx_port1_stat_rx_remote_fault(stat_rx_port1_stat_rx_remote_fault),
        .stat_rx_port1_stat_rx_status(stat_rx_port1_stat_rx_status),
        .stat_rx_port1_stat_rx_test_pattern_mismatch(stat_rx_port1_stat_rx_test_pattern_mismatch),
        .stat_rx_port1_stat_rx_truncated(stat_rx_port1_stat_rx_truncated),
        .stat_rx_port1_stat_rx_valid_ctrl_code(stat_rx_port1_stat_rx_valid_ctrl_code),
        .stat_rx_port2_stat_rx_aligned(stat_rx_port2_stat_rx_aligned),
        .stat_rx_port2_stat_rx_aligned_err(stat_rx_port2_stat_rx_aligned_err),
        .stat_rx_port2_stat_rx_axis_err(stat_rx_port2_stat_rx_axis_err),
        .stat_rx_port2_stat_rx_axis_fifo_overflow(stat_rx_port2_stat_rx_axis_fifo_overflow),
        .stat_rx_port2_stat_rx_bad_code(stat_rx_port2_stat_rx_bad_code),
        .stat_rx_port2_stat_rx_bad_fcs(stat_rx_port2_stat_rx_bad_fcs),
        .stat_rx_port2_stat_rx_bad_preamble(stat_rx_port2_stat_rx_bad_preamble),
        .stat_rx_port2_stat_rx_bad_sfd(stat_rx_port2_stat_rx_bad_sfd),
        .stat_rx_port2_stat_rx_bip_err(stat_rx_port2_stat_rx_bip_err),
        .stat_rx_port2_stat_rx_block_lock(stat_rx_port2_stat_rx_block_lock),
        .stat_rx_port2_stat_rx_cl49_82_convert_err(stat_rx_port2_stat_rx_cl49_82_convert_err),
        .stat_rx_port2_stat_rx_ecc_err(stat_rx_port2_stat_rx_ecc_err),
        .stat_rx_port2_stat_rx_flex_fifo_ovf(stat_rx_port2_stat_rx_flex_fifo_ovf),
        .stat_rx_port2_stat_rx_flex_fifo_udf(stat_rx_port2_stat_rx_flex_fifo_udf),
        .stat_rx_port2_stat_rx_flex_mon_fifo_ovf(stat_rx_port2_stat_rx_flex_mon_fifo_ovf),
        .stat_rx_port2_stat_rx_flex_mon_fifo_udf(stat_rx_port2_stat_rx_flex_mon_fifo_udf),
        .stat_rx_port2_stat_rx_flexif_err(stat_rx_port2_stat_rx_flexif_err),
        .stat_rx_port2_stat_rx_framing_err_2(stat_rx_port2_stat_rx_framing_err_2),
        .stat_rx_port2_stat_rx_got_signal_os(stat_rx_port2_stat_rx_got_signal_os),
        .stat_rx_port2_stat_rx_hi_ber(stat_rx_port2_stat_rx_hi_ber),
        .stat_rx_port2_stat_rx_internal_local_fault(stat_rx_port2_stat_rx_internal_local_fault),
        .stat_rx_port2_stat_rx_invalid_start(stat_rx_port2_stat_rx_invalid_start),
        .stat_rx_port2_stat_rx_lane0_vlm_bip7(stat_rx_port2_stat_rx_lane0_vlm_bip7),
        .stat_rx_port2_stat_rx_lane0_vlm_bip7_valid(stat_rx_port2_stat_rx_lane0_vlm_bip7_valid),
        .stat_rx_port2_stat_rx_local_fault(stat_rx_port2_stat_rx_local_fault),
        .stat_rx_port2_stat_rx_mf_err_2(stat_rx_port2_stat_rx_mf_err_2),
        .stat_rx_port2_stat_rx_misaligned(stat_rx_port2_stat_rx_misaligned),
        .stat_rx_port2_stat_rx_pcs_bad_code(stat_rx_port2_stat_rx_pcs_bad_code),
        .stat_rx_port2_stat_rx_received_local_fault(stat_rx_port2_stat_rx_received_local_fault),
        .stat_rx_port2_stat_rx_remote_fault(stat_rx_port2_stat_rx_remote_fault),
        .stat_rx_port2_stat_rx_status(stat_rx_port2_stat_rx_status),
        .stat_rx_port2_stat_rx_synced(stat_rx_port2_stat_rx_synced),
        .stat_rx_port2_stat_rx_synced_err(stat_rx_port2_stat_rx_synced_err),
        .stat_rx_port2_stat_rx_test_pattern_mismatch(stat_rx_port2_stat_rx_test_pattern_mismatch),
        .stat_rx_port2_stat_rx_truncated(stat_rx_port2_stat_rx_truncated),
        .stat_rx_port2_stat_rx_valid_ctrl_code(stat_rx_port2_stat_rx_valid_ctrl_code),
        .stat_rx_port2_stat_rx_vl_demuxed(stat_rx_port2_stat_rx_vl_demuxed),
        .stat_rx_port3_stat_rx_axis_err(stat_rx_port3_stat_rx_axis_err),
        .stat_rx_port3_stat_rx_axis_fifo_overflow(stat_rx_port3_stat_rx_axis_fifo_overflow),
        .stat_rx_port3_stat_rx_bad_code(stat_rx_port3_stat_rx_bad_code),
        .stat_rx_port3_stat_rx_bad_fcs(stat_rx_port3_stat_rx_bad_fcs),
        .stat_rx_port3_stat_rx_bad_preamble(stat_rx_port3_stat_rx_bad_preamble),
        .stat_rx_port3_stat_rx_bad_sfd(stat_rx_port3_stat_rx_bad_sfd),
        .stat_rx_port3_stat_rx_block_lock(stat_rx_port3_stat_rx_block_lock),
        .stat_rx_port3_stat_rx_cl49_82_convert_err(stat_rx_port3_stat_rx_cl49_82_convert_err),
        .stat_rx_port3_stat_rx_ecc_err(stat_rx_port3_stat_rx_ecc_err),
        .stat_rx_port3_stat_rx_flex_fifo_ovf(stat_rx_port3_stat_rx_flex_fifo_ovf),
        .stat_rx_port3_stat_rx_flex_fifo_udf(stat_rx_port3_stat_rx_flex_fifo_udf),
        .stat_rx_port3_stat_rx_flex_mon_fifo_ovf(stat_rx_port3_stat_rx_flex_mon_fifo_ovf),
        .stat_rx_port3_stat_rx_flex_mon_fifo_udf(stat_rx_port3_stat_rx_flex_mon_fifo_udf),
        .stat_rx_port3_stat_rx_flexif_err(stat_rx_port3_stat_rx_flexif_err),
        .stat_rx_port3_stat_rx_framing_err_3(stat_rx_port3_stat_rx_framing_err_3),
        .stat_rx_port3_stat_rx_got_signal_os(stat_rx_port3_stat_rx_got_signal_os),
        .stat_rx_port3_stat_rx_hi_ber(stat_rx_port3_stat_rx_hi_ber),
        .stat_rx_port3_stat_rx_internal_local_fault(stat_rx_port3_stat_rx_internal_local_fault),
        .stat_rx_port3_stat_rx_invalid_start(stat_rx_port3_stat_rx_invalid_start),
        .stat_rx_port3_stat_rx_local_fault(stat_rx_port3_stat_rx_local_fault),
        .stat_rx_port3_stat_rx_pcs_bad_code(stat_rx_port3_stat_rx_pcs_bad_code),
        .stat_rx_port3_stat_rx_received_local_fault(stat_rx_port3_stat_rx_received_local_fault),
        .stat_rx_port3_stat_rx_remote_fault(stat_rx_port3_stat_rx_remote_fault),
        .stat_rx_port3_stat_rx_status(stat_rx_port3_stat_rx_status),
        .stat_rx_port3_stat_rx_test_pattern_mismatch(stat_rx_port3_stat_rx_test_pattern_mismatch),
        .stat_rx_port3_stat_rx_truncated(stat_rx_port3_stat_rx_truncated),
        .stat_rx_port3_stat_rx_valid_ctrl_code(stat_rx_port3_stat_rx_valid_ctrl_code),
        .stat_tx_port0_stat_tx_axis_err(stat_tx_port0_stat_tx_axis_err),
        .stat_tx_port0_stat_tx_axis_unf(stat_tx_port0_stat_tx_axis_unf),
        .stat_tx_port0_stat_tx_bad_fcs(stat_tx_port0_stat_tx_bad_fcs),
        .stat_tx_port0_stat_tx_cl82_49_convert_err(stat_tx_port0_stat_tx_cl82_49_convert_err),
        .stat_tx_port0_stat_tx_ecc_err(stat_tx_port0_stat_tx_ecc_err),
        .stat_tx_port0_stat_tx_flex_fifo_ovf(stat_tx_port0_stat_tx_flex_fifo_ovf),
        .stat_tx_port0_stat_tx_flex_fifo_udf(stat_tx_port0_stat_tx_flex_fifo_udf),
        .stat_tx_port0_stat_tx_flexif_err(stat_tx_port0_stat_tx_flexif_err),
        .stat_tx_port0_stat_tx_frame_error(stat_tx_port0_stat_tx_frame_error),
        .stat_tx_port0_stat_tx_local_fault(stat_tx_port0_stat_tx_local_fault),
        .stat_tx_port0_stat_tx_pcs_bad_code(stat_tx_port0_stat_tx_pcs_bad_code),
        .stat_tx_port1_stat_tx_axis_err(stat_tx_port1_stat_tx_axis_err),
        .stat_tx_port1_stat_tx_axis_unf(stat_tx_port1_stat_tx_axis_unf),
        .stat_tx_port1_stat_tx_bad_fcs(stat_tx_port1_stat_tx_bad_fcs),
        .stat_tx_port1_stat_tx_cl82_49_convert_err(stat_tx_port1_stat_tx_cl82_49_convert_err),
        .stat_tx_port1_stat_tx_ecc_err(stat_tx_port1_stat_tx_ecc_err),
        .stat_tx_port1_stat_tx_flex_fifo_ovf(stat_tx_port1_stat_tx_flex_fifo_ovf),
        .stat_tx_port1_stat_tx_flex_fifo_udf(stat_tx_port1_stat_tx_flex_fifo_udf),
        .stat_tx_port1_stat_tx_flexif_err(stat_tx_port1_stat_tx_flexif_err),
        .stat_tx_port1_stat_tx_frame_error(stat_tx_port1_stat_tx_frame_error),
        .stat_tx_port1_stat_tx_local_fault(stat_tx_port1_stat_tx_local_fault),
        .stat_tx_port1_stat_tx_pcs_bad_code(stat_tx_port1_stat_tx_pcs_bad_code),
        .stat_tx_port2_stat_tx_axis_err(stat_tx_port2_stat_tx_axis_err),
        .stat_tx_port2_stat_tx_axis_unf(stat_tx_port2_stat_tx_axis_unf),
        .stat_tx_port2_stat_tx_bad_fcs(stat_tx_port2_stat_tx_bad_fcs),
        .stat_tx_port2_stat_tx_cl82_49_convert_err(stat_tx_port2_stat_tx_cl82_49_convert_err),
        .stat_tx_port2_stat_tx_ecc_err(stat_tx_port2_stat_tx_ecc_err),
        .stat_tx_port2_stat_tx_flex_fifo_ovf(stat_tx_port2_stat_tx_flex_fifo_ovf),
        .stat_tx_port2_stat_tx_flex_fifo_udf(stat_tx_port2_stat_tx_flex_fifo_udf),
        .stat_tx_port2_stat_tx_flexif_err(stat_tx_port2_stat_tx_flexif_err),
        .stat_tx_port2_stat_tx_frame_error(stat_tx_port2_stat_tx_frame_error),
        .stat_tx_port2_stat_tx_local_fault(stat_tx_port2_stat_tx_local_fault),
        .stat_tx_port2_stat_tx_pcs_bad_code(stat_tx_port2_stat_tx_pcs_bad_code),
        .stat_tx_port3_stat_tx_axis_err(stat_tx_port3_stat_tx_axis_err),
        .stat_tx_port3_stat_tx_axis_unf(stat_tx_port3_stat_tx_axis_unf),
        .stat_tx_port3_stat_tx_bad_fcs(stat_tx_port3_stat_tx_bad_fcs),
        .stat_tx_port3_stat_tx_cl82_49_convert_err(stat_tx_port3_stat_tx_cl82_49_convert_err),
        .stat_tx_port3_stat_tx_ecc_err(stat_tx_port3_stat_tx_ecc_err),
        .stat_tx_port3_stat_tx_flex_fifo_ovf(stat_tx_port3_stat_tx_flex_fifo_ovf),
        .stat_tx_port3_stat_tx_flex_fifo_udf(stat_tx_port3_stat_tx_flex_fifo_udf),
        .stat_tx_port3_stat_tx_flexif_err(stat_tx_port3_stat_tx_flexif_err),
        .stat_tx_port3_stat_tx_frame_error(stat_tx_port3_stat_tx_frame_error),
        .stat_tx_port3_stat_tx_local_fault(stat_tx_port3_stat_tx_local_fault),
        .stat_tx_port3_stat_tx_pcs_bad_code(stat_tx_port3_stat_tx_pcs_bad_code),
        .tx_alt_serdes_clk(tx_alt_serdes_clk),
        .tx_axi_clk(tx_axi_clk),
        .tx_axis_tdata0(tx_axis_tdata0),
        .tx_axis_tdata1(tx_axis_tdata1),
        .tx_axis_tdata2(tx_axis_tdata2),
        .tx_axis_tdata3(tx_axis_tdata3),
        .tx_axis_tdata4(tx_axis_tdata4),
        .tx_axis_tdata5(tx_axis_tdata5),
        .tx_axis_tdata6(tx_axis_tdata6),
        .tx_axis_tdata7(tx_axis_tdata7),
        .tx_axis_tkeep_user0(tx_axis_tkeep_user0),
        .tx_axis_tkeep_user1(tx_axis_tkeep_user1),
        .tx_axis_tkeep_user2(tx_axis_tkeep_user2),
        .tx_axis_tkeep_user3(tx_axis_tkeep_user3),
        .tx_axis_tkeep_user4(tx_axis_tkeep_user4),
        .tx_axis_tkeep_user5(tx_axis_tkeep_user5),
        .tx_axis_tkeep_user6(tx_axis_tkeep_user6),
        .tx_axis_tkeep_user7(tx_axis_tkeep_user7),
        .tx_axis_tlast_0(tx_axis_tlast_0),
        .tx_axis_tlast_1(tx_axis_tlast_1),
        .tx_axis_tlast_2(tx_axis_tlast_2),
        .tx_axis_tlast_3(tx_axis_tlast_3),
        .tx_axis_tready_0(tx_axis_tready_0),
        .tx_axis_tready_1(tx_axis_tready_1),
        .tx_axis_tready_2(tx_axis_tready_2),
        .tx_axis_tready_3(tx_axis_tready_3),
        .tx_axis_tvalid_0(tx_axis_tvalid_0),
        .tx_axis_tvalid_1(tx_axis_tvalid_1),
        .tx_axis_tvalid_2(tx_axis_tvalid_2),
        .tx_axis_tvalid_3(tx_axis_tvalid_3),
        .tx_core_clk(tx_core_clk),
        .tx_core_reset(tx_core_reset),
        .tx_flexif_clk(tx_flexif_clk),
        .tx_preamblein_tx_preamblein_0(tx_preamblein_tx_preamblein_0),
        .tx_preamblein_tx_preamblein_1(tx_preamblein_tx_preamblein_1),
        .tx_preamblein_tx_preamblein_2(tx_preamblein_tx_preamblein_2),
        .tx_preamblein_tx_preamblein_3(tx_preamblein_tx_preamblein_3),
        .tx_serdes_reset(tx_serdes_reset),
        .tx_ts_clk(tx_ts_clk));
endmodule
