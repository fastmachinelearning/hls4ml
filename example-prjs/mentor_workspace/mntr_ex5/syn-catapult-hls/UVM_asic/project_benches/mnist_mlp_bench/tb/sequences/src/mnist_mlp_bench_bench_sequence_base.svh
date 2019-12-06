//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Bench Sequence Base
// File            : mnist_mlp_bench_bench_sequence_base.svh
//----------------------------------------------------------------------
//
// Description: This file contains the top level and utility sequences
//     used by test_top. It can be extended to create derivative top
//     level sequences.
//
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
//

class mnist_mlp_bench_bench_sequence_base extends uvmf_sequence_base #(uvm_sequence_item);

  `uvm_object_utils( mnist_mlp_bench_bench_sequence_base );

  // UVMF_CHANGE_ME : Instantiate, construct, and start sequences as needed to create stimulus scenarios.

  // Instantiate sequences here
  typedef ccs_random_sequence #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))    input1_rsc_random_seq_t;
  input1_rsc_random_seq_t input1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))    output1_rsc_random_seq_t;
  output1_rsc_random_seq_t output1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))    const_size_in_1_rsc_random_seq_t;
  const_size_in_1_rsc_random_seq_t const_size_in_1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))    const_size_out_1_rsc_random_seq_t;
  const_size_out_1_rsc_random_seq_t const_size_out_1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_0_0_random_seq_t;
  w2_rsc_0_0_random_seq_t w2_rsc_0_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_1_0_random_seq_t;
  w2_rsc_1_0_random_seq_t w2_rsc_1_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_2_0_random_seq_t;
  w2_rsc_2_0_random_seq_t w2_rsc_2_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_3_0_random_seq_t;
  w2_rsc_3_0_random_seq_t w2_rsc_3_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_4_0_random_seq_t;
  w2_rsc_4_0_random_seq_t w2_rsc_4_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_5_0_random_seq_t;
  w2_rsc_5_0_random_seq_t w2_rsc_5_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_6_0_random_seq_t;
  w2_rsc_6_0_random_seq_t w2_rsc_6_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_7_0_random_seq_t;
  w2_rsc_7_0_random_seq_t w2_rsc_7_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_8_0_random_seq_t;
  w2_rsc_8_0_random_seq_t w2_rsc_8_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_9_0_random_seq_t;
  w2_rsc_9_0_random_seq_t w2_rsc_9_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_10_0_random_seq_t;
  w2_rsc_10_0_random_seq_t w2_rsc_10_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_11_0_random_seq_t;
  w2_rsc_11_0_random_seq_t w2_rsc_11_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_12_0_random_seq_t;
  w2_rsc_12_0_random_seq_t w2_rsc_12_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_13_0_random_seq_t;
  w2_rsc_13_0_random_seq_t w2_rsc_13_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_14_0_random_seq_t;
  w2_rsc_14_0_random_seq_t w2_rsc_14_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_15_0_random_seq_t;
  w2_rsc_15_0_random_seq_t w2_rsc_15_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_16_0_random_seq_t;
  w2_rsc_16_0_random_seq_t w2_rsc_16_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_17_0_random_seq_t;
  w2_rsc_17_0_random_seq_t w2_rsc_17_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_18_0_random_seq_t;
  w2_rsc_18_0_random_seq_t w2_rsc_18_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_19_0_random_seq_t;
  w2_rsc_19_0_random_seq_t w2_rsc_19_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_20_0_random_seq_t;
  w2_rsc_20_0_random_seq_t w2_rsc_20_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_21_0_random_seq_t;
  w2_rsc_21_0_random_seq_t w2_rsc_21_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_22_0_random_seq_t;
  w2_rsc_22_0_random_seq_t w2_rsc_22_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_23_0_random_seq_t;
  w2_rsc_23_0_random_seq_t w2_rsc_23_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_24_0_random_seq_t;
  w2_rsc_24_0_random_seq_t w2_rsc_24_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_25_0_random_seq_t;
  w2_rsc_25_0_random_seq_t w2_rsc_25_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_26_0_random_seq_t;
  w2_rsc_26_0_random_seq_t w2_rsc_26_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_27_0_random_seq_t;
  w2_rsc_27_0_random_seq_t w2_rsc_27_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_28_0_random_seq_t;
  w2_rsc_28_0_random_seq_t w2_rsc_28_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_29_0_random_seq_t;
  w2_rsc_29_0_random_seq_t w2_rsc_29_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_30_0_random_seq_t;
  w2_rsc_30_0_random_seq_t w2_rsc_30_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_31_0_random_seq_t;
  w2_rsc_31_0_random_seq_t w2_rsc_31_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_32_0_random_seq_t;
  w2_rsc_32_0_random_seq_t w2_rsc_32_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_33_0_random_seq_t;
  w2_rsc_33_0_random_seq_t w2_rsc_33_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_34_0_random_seq_t;
  w2_rsc_34_0_random_seq_t w2_rsc_34_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_35_0_random_seq_t;
  w2_rsc_35_0_random_seq_t w2_rsc_35_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_36_0_random_seq_t;
  w2_rsc_36_0_random_seq_t w2_rsc_36_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_37_0_random_seq_t;
  w2_rsc_37_0_random_seq_t w2_rsc_37_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_38_0_random_seq_t;
  w2_rsc_38_0_random_seq_t w2_rsc_38_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_39_0_random_seq_t;
  w2_rsc_39_0_random_seq_t w2_rsc_39_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_40_0_random_seq_t;
  w2_rsc_40_0_random_seq_t w2_rsc_40_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_41_0_random_seq_t;
  w2_rsc_41_0_random_seq_t w2_rsc_41_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_42_0_random_seq_t;
  w2_rsc_42_0_random_seq_t w2_rsc_42_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_43_0_random_seq_t;
  w2_rsc_43_0_random_seq_t w2_rsc_43_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_44_0_random_seq_t;
  w2_rsc_44_0_random_seq_t w2_rsc_44_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_45_0_random_seq_t;
  w2_rsc_45_0_random_seq_t w2_rsc_45_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_46_0_random_seq_t;
  w2_rsc_46_0_random_seq_t w2_rsc_46_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_47_0_random_seq_t;
  w2_rsc_47_0_random_seq_t w2_rsc_47_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_48_0_random_seq_t;
  w2_rsc_48_0_random_seq_t w2_rsc_48_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_49_0_random_seq_t;
  w2_rsc_49_0_random_seq_t w2_rsc_49_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_50_0_random_seq_t;
  w2_rsc_50_0_random_seq_t w2_rsc_50_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_51_0_random_seq_t;
  w2_rsc_51_0_random_seq_t w2_rsc_51_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_52_0_random_seq_t;
  w2_rsc_52_0_random_seq_t w2_rsc_52_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_53_0_random_seq_t;
  w2_rsc_53_0_random_seq_t w2_rsc_53_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_54_0_random_seq_t;
  w2_rsc_54_0_random_seq_t w2_rsc_54_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_55_0_random_seq_t;
  w2_rsc_55_0_random_seq_t w2_rsc_55_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_56_0_random_seq_t;
  w2_rsc_56_0_random_seq_t w2_rsc_56_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_57_0_random_seq_t;
  w2_rsc_57_0_random_seq_t w2_rsc_57_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_58_0_random_seq_t;
  w2_rsc_58_0_random_seq_t w2_rsc_58_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_59_0_random_seq_t;
  w2_rsc_59_0_random_seq_t w2_rsc_59_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_60_0_random_seq_t;
  w2_rsc_60_0_random_seq_t w2_rsc_60_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_61_0_random_seq_t;
  w2_rsc_61_0_random_seq_t w2_rsc_61_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_62_0_random_seq_t;
  w2_rsc_62_0_random_seq_t w2_rsc_62_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))    w2_rsc_63_0_random_seq_t;
  w2_rsc_63_0_random_seq_t w2_rsc_63_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    b2_rsc_random_seq_t;
  b2_rsc_random_seq_t b2_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_0_0_random_seq_t;
  w4_rsc_0_0_random_seq_t w4_rsc_0_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_1_0_random_seq_t;
  w4_rsc_1_0_random_seq_t w4_rsc_1_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_2_0_random_seq_t;
  w4_rsc_2_0_random_seq_t w4_rsc_2_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_3_0_random_seq_t;
  w4_rsc_3_0_random_seq_t w4_rsc_3_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_4_0_random_seq_t;
  w4_rsc_4_0_random_seq_t w4_rsc_4_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_5_0_random_seq_t;
  w4_rsc_5_0_random_seq_t w4_rsc_5_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_6_0_random_seq_t;
  w4_rsc_6_0_random_seq_t w4_rsc_6_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_7_0_random_seq_t;
  w4_rsc_7_0_random_seq_t w4_rsc_7_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_8_0_random_seq_t;
  w4_rsc_8_0_random_seq_t w4_rsc_8_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_9_0_random_seq_t;
  w4_rsc_9_0_random_seq_t w4_rsc_9_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_10_0_random_seq_t;
  w4_rsc_10_0_random_seq_t w4_rsc_10_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_11_0_random_seq_t;
  w4_rsc_11_0_random_seq_t w4_rsc_11_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_12_0_random_seq_t;
  w4_rsc_12_0_random_seq_t w4_rsc_12_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_13_0_random_seq_t;
  w4_rsc_13_0_random_seq_t w4_rsc_13_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_14_0_random_seq_t;
  w4_rsc_14_0_random_seq_t w4_rsc_14_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_15_0_random_seq_t;
  w4_rsc_15_0_random_seq_t w4_rsc_15_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_16_0_random_seq_t;
  w4_rsc_16_0_random_seq_t w4_rsc_16_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_17_0_random_seq_t;
  w4_rsc_17_0_random_seq_t w4_rsc_17_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_18_0_random_seq_t;
  w4_rsc_18_0_random_seq_t w4_rsc_18_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_19_0_random_seq_t;
  w4_rsc_19_0_random_seq_t w4_rsc_19_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_20_0_random_seq_t;
  w4_rsc_20_0_random_seq_t w4_rsc_20_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_21_0_random_seq_t;
  w4_rsc_21_0_random_seq_t w4_rsc_21_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_22_0_random_seq_t;
  w4_rsc_22_0_random_seq_t w4_rsc_22_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_23_0_random_seq_t;
  w4_rsc_23_0_random_seq_t w4_rsc_23_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_24_0_random_seq_t;
  w4_rsc_24_0_random_seq_t w4_rsc_24_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_25_0_random_seq_t;
  w4_rsc_25_0_random_seq_t w4_rsc_25_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_26_0_random_seq_t;
  w4_rsc_26_0_random_seq_t w4_rsc_26_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_27_0_random_seq_t;
  w4_rsc_27_0_random_seq_t w4_rsc_27_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_28_0_random_seq_t;
  w4_rsc_28_0_random_seq_t w4_rsc_28_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_29_0_random_seq_t;
  w4_rsc_29_0_random_seq_t w4_rsc_29_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_30_0_random_seq_t;
  w4_rsc_30_0_random_seq_t w4_rsc_30_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_31_0_random_seq_t;
  w4_rsc_31_0_random_seq_t w4_rsc_31_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_32_0_random_seq_t;
  w4_rsc_32_0_random_seq_t w4_rsc_32_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_33_0_random_seq_t;
  w4_rsc_33_0_random_seq_t w4_rsc_33_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_34_0_random_seq_t;
  w4_rsc_34_0_random_seq_t w4_rsc_34_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_35_0_random_seq_t;
  w4_rsc_35_0_random_seq_t w4_rsc_35_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_36_0_random_seq_t;
  w4_rsc_36_0_random_seq_t w4_rsc_36_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_37_0_random_seq_t;
  w4_rsc_37_0_random_seq_t w4_rsc_37_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_38_0_random_seq_t;
  w4_rsc_38_0_random_seq_t w4_rsc_38_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_39_0_random_seq_t;
  w4_rsc_39_0_random_seq_t w4_rsc_39_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_40_0_random_seq_t;
  w4_rsc_40_0_random_seq_t w4_rsc_40_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_41_0_random_seq_t;
  w4_rsc_41_0_random_seq_t w4_rsc_41_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_42_0_random_seq_t;
  w4_rsc_42_0_random_seq_t w4_rsc_42_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_43_0_random_seq_t;
  w4_rsc_43_0_random_seq_t w4_rsc_43_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_44_0_random_seq_t;
  w4_rsc_44_0_random_seq_t w4_rsc_44_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_45_0_random_seq_t;
  w4_rsc_45_0_random_seq_t w4_rsc_45_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_46_0_random_seq_t;
  w4_rsc_46_0_random_seq_t w4_rsc_46_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_47_0_random_seq_t;
  w4_rsc_47_0_random_seq_t w4_rsc_47_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_48_0_random_seq_t;
  w4_rsc_48_0_random_seq_t w4_rsc_48_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_49_0_random_seq_t;
  w4_rsc_49_0_random_seq_t w4_rsc_49_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_50_0_random_seq_t;
  w4_rsc_50_0_random_seq_t w4_rsc_50_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_51_0_random_seq_t;
  w4_rsc_51_0_random_seq_t w4_rsc_51_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_52_0_random_seq_t;
  w4_rsc_52_0_random_seq_t w4_rsc_52_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_53_0_random_seq_t;
  w4_rsc_53_0_random_seq_t w4_rsc_53_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_54_0_random_seq_t;
  w4_rsc_54_0_random_seq_t w4_rsc_54_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_55_0_random_seq_t;
  w4_rsc_55_0_random_seq_t w4_rsc_55_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_56_0_random_seq_t;
  w4_rsc_56_0_random_seq_t w4_rsc_56_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_57_0_random_seq_t;
  w4_rsc_57_0_random_seq_t w4_rsc_57_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_58_0_random_seq_t;
  w4_rsc_58_0_random_seq_t w4_rsc_58_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_59_0_random_seq_t;
  w4_rsc_59_0_random_seq_t w4_rsc_59_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_60_0_random_seq_t;
  w4_rsc_60_0_random_seq_t w4_rsc_60_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_61_0_random_seq_t;
  w4_rsc_61_0_random_seq_t w4_rsc_61_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_62_0_random_seq_t;
  w4_rsc_62_0_random_seq_t w4_rsc_62_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w4_rsc_63_0_random_seq_t;
  w4_rsc_63_0_random_seq_t w4_rsc_63_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    b4_rsc_random_seq_t;
  b4_rsc_random_seq_t b4_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_0_0_random_seq_t;
  w6_rsc_0_0_random_seq_t w6_rsc_0_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_1_0_random_seq_t;
  w6_rsc_1_0_random_seq_t w6_rsc_1_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_2_0_random_seq_t;
  w6_rsc_2_0_random_seq_t w6_rsc_2_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_3_0_random_seq_t;
  w6_rsc_3_0_random_seq_t w6_rsc_3_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_4_0_random_seq_t;
  w6_rsc_4_0_random_seq_t w6_rsc_4_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_5_0_random_seq_t;
  w6_rsc_5_0_random_seq_t w6_rsc_5_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_6_0_random_seq_t;
  w6_rsc_6_0_random_seq_t w6_rsc_6_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_7_0_random_seq_t;
  w6_rsc_7_0_random_seq_t w6_rsc_7_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_8_0_random_seq_t;
  w6_rsc_8_0_random_seq_t w6_rsc_8_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))    w6_rsc_9_0_random_seq_t;
  w6_rsc_9_0_random_seq_t w6_rsc_9_0_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1))    b6_rsc_random_seq_t;
  b6_rsc_random_seq_t b6_rsc_random_seq;

  // Sequencer handles for each active interface in the environment
  typedef ccs_transaction #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_transaction_t;
  uvm_sequencer #(input1_rsc_transaction_t)  input1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_transaction_t;
  uvm_sequencer #(output1_rsc_transaction_t)  output1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_transaction_t;
  uvm_sequencer #(const_size_in_1_rsc_transaction_t)  const_size_in_1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_transaction_t;
  uvm_sequencer #(const_size_out_1_rsc_transaction_t)  const_size_out_1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_0_0_transaction_t;
  uvm_sequencer #(w2_rsc_0_0_transaction_t)  w2_rsc_0_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_1_0_transaction_t;
  uvm_sequencer #(w2_rsc_1_0_transaction_t)  w2_rsc_1_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_2_0_transaction_t;
  uvm_sequencer #(w2_rsc_2_0_transaction_t)  w2_rsc_2_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_3_0_transaction_t;
  uvm_sequencer #(w2_rsc_3_0_transaction_t)  w2_rsc_3_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_4_0_transaction_t;
  uvm_sequencer #(w2_rsc_4_0_transaction_t)  w2_rsc_4_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_5_0_transaction_t;
  uvm_sequencer #(w2_rsc_5_0_transaction_t)  w2_rsc_5_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_6_0_transaction_t;
  uvm_sequencer #(w2_rsc_6_0_transaction_t)  w2_rsc_6_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_7_0_transaction_t;
  uvm_sequencer #(w2_rsc_7_0_transaction_t)  w2_rsc_7_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_8_0_transaction_t;
  uvm_sequencer #(w2_rsc_8_0_transaction_t)  w2_rsc_8_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_9_0_transaction_t;
  uvm_sequencer #(w2_rsc_9_0_transaction_t)  w2_rsc_9_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_10_0_transaction_t;
  uvm_sequencer #(w2_rsc_10_0_transaction_t)  w2_rsc_10_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_11_0_transaction_t;
  uvm_sequencer #(w2_rsc_11_0_transaction_t)  w2_rsc_11_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_12_0_transaction_t;
  uvm_sequencer #(w2_rsc_12_0_transaction_t)  w2_rsc_12_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_13_0_transaction_t;
  uvm_sequencer #(w2_rsc_13_0_transaction_t)  w2_rsc_13_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_14_0_transaction_t;
  uvm_sequencer #(w2_rsc_14_0_transaction_t)  w2_rsc_14_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_15_0_transaction_t;
  uvm_sequencer #(w2_rsc_15_0_transaction_t)  w2_rsc_15_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_16_0_transaction_t;
  uvm_sequencer #(w2_rsc_16_0_transaction_t)  w2_rsc_16_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_17_0_transaction_t;
  uvm_sequencer #(w2_rsc_17_0_transaction_t)  w2_rsc_17_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_18_0_transaction_t;
  uvm_sequencer #(w2_rsc_18_0_transaction_t)  w2_rsc_18_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_19_0_transaction_t;
  uvm_sequencer #(w2_rsc_19_0_transaction_t)  w2_rsc_19_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_20_0_transaction_t;
  uvm_sequencer #(w2_rsc_20_0_transaction_t)  w2_rsc_20_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_21_0_transaction_t;
  uvm_sequencer #(w2_rsc_21_0_transaction_t)  w2_rsc_21_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_22_0_transaction_t;
  uvm_sequencer #(w2_rsc_22_0_transaction_t)  w2_rsc_22_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_23_0_transaction_t;
  uvm_sequencer #(w2_rsc_23_0_transaction_t)  w2_rsc_23_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_24_0_transaction_t;
  uvm_sequencer #(w2_rsc_24_0_transaction_t)  w2_rsc_24_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_25_0_transaction_t;
  uvm_sequencer #(w2_rsc_25_0_transaction_t)  w2_rsc_25_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_26_0_transaction_t;
  uvm_sequencer #(w2_rsc_26_0_transaction_t)  w2_rsc_26_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_27_0_transaction_t;
  uvm_sequencer #(w2_rsc_27_0_transaction_t)  w2_rsc_27_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_28_0_transaction_t;
  uvm_sequencer #(w2_rsc_28_0_transaction_t)  w2_rsc_28_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_29_0_transaction_t;
  uvm_sequencer #(w2_rsc_29_0_transaction_t)  w2_rsc_29_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_30_0_transaction_t;
  uvm_sequencer #(w2_rsc_30_0_transaction_t)  w2_rsc_30_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_31_0_transaction_t;
  uvm_sequencer #(w2_rsc_31_0_transaction_t)  w2_rsc_31_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_32_0_transaction_t;
  uvm_sequencer #(w2_rsc_32_0_transaction_t)  w2_rsc_32_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_33_0_transaction_t;
  uvm_sequencer #(w2_rsc_33_0_transaction_t)  w2_rsc_33_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_34_0_transaction_t;
  uvm_sequencer #(w2_rsc_34_0_transaction_t)  w2_rsc_34_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_35_0_transaction_t;
  uvm_sequencer #(w2_rsc_35_0_transaction_t)  w2_rsc_35_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_36_0_transaction_t;
  uvm_sequencer #(w2_rsc_36_0_transaction_t)  w2_rsc_36_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_37_0_transaction_t;
  uvm_sequencer #(w2_rsc_37_0_transaction_t)  w2_rsc_37_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_38_0_transaction_t;
  uvm_sequencer #(w2_rsc_38_0_transaction_t)  w2_rsc_38_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_39_0_transaction_t;
  uvm_sequencer #(w2_rsc_39_0_transaction_t)  w2_rsc_39_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_40_0_transaction_t;
  uvm_sequencer #(w2_rsc_40_0_transaction_t)  w2_rsc_40_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_41_0_transaction_t;
  uvm_sequencer #(w2_rsc_41_0_transaction_t)  w2_rsc_41_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_42_0_transaction_t;
  uvm_sequencer #(w2_rsc_42_0_transaction_t)  w2_rsc_42_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_43_0_transaction_t;
  uvm_sequencer #(w2_rsc_43_0_transaction_t)  w2_rsc_43_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_44_0_transaction_t;
  uvm_sequencer #(w2_rsc_44_0_transaction_t)  w2_rsc_44_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_45_0_transaction_t;
  uvm_sequencer #(w2_rsc_45_0_transaction_t)  w2_rsc_45_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_46_0_transaction_t;
  uvm_sequencer #(w2_rsc_46_0_transaction_t)  w2_rsc_46_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_47_0_transaction_t;
  uvm_sequencer #(w2_rsc_47_0_transaction_t)  w2_rsc_47_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_48_0_transaction_t;
  uvm_sequencer #(w2_rsc_48_0_transaction_t)  w2_rsc_48_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_49_0_transaction_t;
  uvm_sequencer #(w2_rsc_49_0_transaction_t)  w2_rsc_49_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_50_0_transaction_t;
  uvm_sequencer #(w2_rsc_50_0_transaction_t)  w2_rsc_50_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_51_0_transaction_t;
  uvm_sequencer #(w2_rsc_51_0_transaction_t)  w2_rsc_51_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_52_0_transaction_t;
  uvm_sequencer #(w2_rsc_52_0_transaction_t)  w2_rsc_52_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_53_0_transaction_t;
  uvm_sequencer #(w2_rsc_53_0_transaction_t)  w2_rsc_53_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_54_0_transaction_t;
  uvm_sequencer #(w2_rsc_54_0_transaction_t)  w2_rsc_54_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_55_0_transaction_t;
  uvm_sequencer #(w2_rsc_55_0_transaction_t)  w2_rsc_55_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_56_0_transaction_t;
  uvm_sequencer #(w2_rsc_56_0_transaction_t)  w2_rsc_56_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_57_0_transaction_t;
  uvm_sequencer #(w2_rsc_57_0_transaction_t)  w2_rsc_57_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_58_0_transaction_t;
  uvm_sequencer #(w2_rsc_58_0_transaction_t)  w2_rsc_58_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_59_0_transaction_t;
  uvm_sequencer #(w2_rsc_59_0_transaction_t)  w2_rsc_59_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_60_0_transaction_t;
  uvm_sequencer #(w2_rsc_60_0_transaction_t)  w2_rsc_60_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_61_0_transaction_t;
  uvm_sequencer #(w2_rsc_61_0_transaction_t)  w2_rsc_61_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_62_0_transaction_t;
  uvm_sequencer #(w2_rsc_62_0_transaction_t)  w2_rsc_62_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_63_0_transaction_t;
  uvm_sequencer #(w2_rsc_63_0_transaction_t)  w2_rsc_63_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  b2_rsc_transaction_t;
  uvm_sequencer #(b2_rsc_transaction_t)  b2_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_0_0_transaction_t;
  uvm_sequencer #(w4_rsc_0_0_transaction_t)  w4_rsc_0_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_1_0_transaction_t;
  uvm_sequencer #(w4_rsc_1_0_transaction_t)  w4_rsc_1_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_2_0_transaction_t;
  uvm_sequencer #(w4_rsc_2_0_transaction_t)  w4_rsc_2_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_3_0_transaction_t;
  uvm_sequencer #(w4_rsc_3_0_transaction_t)  w4_rsc_3_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_4_0_transaction_t;
  uvm_sequencer #(w4_rsc_4_0_transaction_t)  w4_rsc_4_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_5_0_transaction_t;
  uvm_sequencer #(w4_rsc_5_0_transaction_t)  w4_rsc_5_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_6_0_transaction_t;
  uvm_sequencer #(w4_rsc_6_0_transaction_t)  w4_rsc_6_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_7_0_transaction_t;
  uvm_sequencer #(w4_rsc_7_0_transaction_t)  w4_rsc_7_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_8_0_transaction_t;
  uvm_sequencer #(w4_rsc_8_0_transaction_t)  w4_rsc_8_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_9_0_transaction_t;
  uvm_sequencer #(w4_rsc_9_0_transaction_t)  w4_rsc_9_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_10_0_transaction_t;
  uvm_sequencer #(w4_rsc_10_0_transaction_t)  w4_rsc_10_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_11_0_transaction_t;
  uvm_sequencer #(w4_rsc_11_0_transaction_t)  w4_rsc_11_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_12_0_transaction_t;
  uvm_sequencer #(w4_rsc_12_0_transaction_t)  w4_rsc_12_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_13_0_transaction_t;
  uvm_sequencer #(w4_rsc_13_0_transaction_t)  w4_rsc_13_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_14_0_transaction_t;
  uvm_sequencer #(w4_rsc_14_0_transaction_t)  w4_rsc_14_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_15_0_transaction_t;
  uvm_sequencer #(w4_rsc_15_0_transaction_t)  w4_rsc_15_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_16_0_transaction_t;
  uvm_sequencer #(w4_rsc_16_0_transaction_t)  w4_rsc_16_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_17_0_transaction_t;
  uvm_sequencer #(w4_rsc_17_0_transaction_t)  w4_rsc_17_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_18_0_transaction_t;
  uvm_sequencer #(w4_rsc_18_0_transaction_t)  w4_rsc_18_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_19_0_transaction_t;
  uvm_sequencer #(w4_rsc_19_0_transaction_t)  w4_rsc_19_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_20_0_transaction_t;
  uvm_sequencer #(w4_rsc_20_0_transaction_t)  w4_rsc_20_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_21_0_transaction_t;
  uvm_sequencer #(w4_rsc_21_0_transaction_t)  w4_rsc_21_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_22_0_transaction_t;
  uvm_sequencer #(w4_rsc_22_0_transaction_t)  w4_rsc_22_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_23_0_transaction_t;
  uvm_sequencer #(w4_rsc_23_0_transaction_t)  w4_rsc_23_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_24_0_transaction_t;
  uvm_sequencer #(w4_rsc_24_0_transaction_t)  w4_rsc_24_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_25_0_transaction_t;
  uvm_sequencer #(w4_rsc_25_0_transaction_t)  w4_rsc_25_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_26_0_transaction_t;
  uvm_sequencer #(w4_rsc_26_0_transaction_t)  w4_rsc_26_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_27_0_transaction_t;
  uvm_sequencer #(w4_rsc_27_0_transaction_t)  w4_rsc_27_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_28_0_transaction_t;
  uvm_sequencer #(w4_rsc_28_0_transaction_t)  w4_rsc_28_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_29_0_transaction_t;
  uvm_sequencer #(w4_rsc_29_0_transaction_t)  w4_rsc_29_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_30_0_transaction_t;
  uvm_sequencer #(w4_rsc_30_0_transaction_t)  w4_rsc_30_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_31_0_transaction_t;
  uvm_sequencer #(w4_rsc_31_0_transaction_t)  w4_rsc_31_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_32_0_transaction_t;
  uvm_sequencer #(w4_rsc_32_0_transaction_t)  w4_rsc_32_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_33_0_transaction_t;
  uvm_sequencer #(w4_rsc_33_0_transaction_t)  w4_rsc_33_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_34_0_transaction_t;
  uvm_sequencer #(w4_rsc_34_0_transaction_t)  w4_rsc_34_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_35_0_transaction_t;
  uvm_sequencer #(w4_rsc_35_0_transaction_t)  w4_rsc_35_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_36_0_transaction_t;
  uvm_sequencer #(w4_rsc_36_0_transaction_t)  w4_rsc_36_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_37_0_transaction_t;
  uvm_sequencer #(w4_rsc_37_0_transaction_t)  w4_rsc_37_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_38_0_transaction_t;
  uvm_sequencer #(w4_rsc_38_0_transaction_t)  w4_rsc_38_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_39_0_transaction_t;
  uvm_sequencer #(w4_rsc_39_0_transaction_t)  w4_rsc_39_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_40_0_transaction_t;
  uvm_sequencer #(w4_rsc_40_0_transaction_t)  w4_rsc_40_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_41_0_transaction_t;
  uvm_sequencer #(w4_rsc_41_0_transaction_t)  w4_rsc_41_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_42_0_transaction_t;
  uvm_sequencer #(w4_rsc_42_0_transaction_t)  w4_rsc_42_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_43_0_transaction_t;
  uvm_sequencer #(w4_rsc_43_0_transaction_t)  w4_rsc_43_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_44_0_transaction_t;
  uvm_sequencer #(w4_rsc_44_0_transaction_t)  w4_rsc_44_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_45_0_transaction_t;
  uvm_sequencer #(w4_rsc_45_0_transaction_t)  w4_rsc_45_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_46_0_transaction_t;
  uvm_sequencer #(w4_rsc_46_0_transaction_t)  w4_rsc_46_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_47_0_transaction_t;
  uvm_sequencer #(w4_rsc_47_0_transaction_t)  w4_rsc_47_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_48_0_transaction_t;
  uvm_sequencer #(w4_rsc_48_0_transaction_t)  w4_rsc_48_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_49_0_transaction_t;
  uvm_sequencer #(w4_rsc_49_0_transaction_t)  w4_rsc_49_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_50_0_transaction_t;
  uvm_sequencer #(w4_rsc_50_0_transaction_t)  w4_rsc_50_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_51_0_transaction_t;
  uvm_sequencer #(w4_rsc_51_0_transaction_t)  w4_rsc_51_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_52_0_transaction_t;
  uvm_sequencer #(w4_rsc_52_0_transaction_t)  w4_rsc_52_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_53_0_transaction_t;
  uvm_sequencer #(w4_rsc_53_0_transaction_t)  w4_rsc_53_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_54_0_transaction_t;
  uvm_sequencer #(w4_rsc_54_0_transaction_t)  w4_rsc_54_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_55_0_transaction_t;
  uvm_sequencer #(w4_rsc_55_0_transaction_t)  w4_rsc_55_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_56_0_transaction_t;
  uvm_sequencer #(w4_rsc_56_0_transaction_t)  w4_rsc_56_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_57_0_transaction_t;
  uvm_sequencer #(w4_rsc_57_0_transaction_t)  w4_rsc_57_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_58_0_transaction_t;
  uvm_sequencer #(w4_rsc_58_0_transaction_t)  w4_rsc_58_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_59_0_transaction_t;
  uvm_sequencer #(w4_rsc_59_0_transaction_t)  w4_rsc_59_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_60_0_transaction_t;
  uvm_sequencer #(w4_rsc_60_0_transaction_t)  w4_rsc_60_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_61_0_transaction_t;
  uvm_sequencer #(w4_rsc_61_0_transaction_t)  w4_rsc_61_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_62_0_transaction_t;
  uvm_sequencer #(w4_rsc_62_0_transaction_t)  w4_rsc_62_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_63_0_transaction_t;
  uvm_sequencer #(w4_rsc_63_0_transaction_t)  w4_rsc_63_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  b4_rsc_transaction_t;
  uvm_sequencer #(b4_rsc_transaction_t)  b4_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_0_0_transaction_t;
  uvm_sequencer #(w6_rsc_0_0_transaction_t)  w6_rsc_0_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_1_0_transaction_t;
  uvm_sequencer #(w6_rsc_1_0_transaction_t)  w6_rsc_1_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_2_0_transaction_t;
  uvm_sequencer #(w6_rsc_2_0_transaction_t)  w6_rsc_2_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_3_0_transaction_t;
  uvm_sequencer #(w6_rsc_3_0_transaction_t)  w6_rsc_3_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_4_0_transaction_t;
  uvm_sequencer #(w6_rsc_4_0_transaction_t)  w6_rsc_4_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_5_0_transaction_t;
  uvm_sequencer #(w6_rsc_5_0_transaction_t)  w6_rsc_5_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_6_0_transaction_t;
  uvm_sequencer #(w6_rsc_6_0_transaction_t)  w6_rsc_6_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_7_0_transaction_t;
  uvm_sequencer #(w6_rsc_7_0_transaction_t)  w6_rsc_7_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_8_0_transaction_t;
  uvm_sequencer #(w6_rsc_8_0_transaction_t)  w6_rsc_8_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_9_0_transaction_t;
  uvm_sequencer #(w6_rsc_9_0_transaction_t)  w6_rsc_9_0_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1))  b6_rsc_transaction_t;
  uvm_sequencer #(b6_rsc_transaction_t)  b6_rsc_sequencer; 


  // Top level environment configuration handle
  typedef mnist_mlp_env_configuration mnist_mlp_env_configuration_t;
  mnist_mlp_env_configuration_t top_configuration;

  // Configuration handles to access interface BFM's
  ccs_configuration  #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_0_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_1_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_2_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_3_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_4_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_5_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_6_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_7_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_8_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_9_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_10_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_11_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_12_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_13_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_14_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_15_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_16_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_17_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_18_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_19_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_20_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_21_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_22_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_23_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_24_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_25_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_26_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_27_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_28_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_29_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_30_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_31_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_32_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_33_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_34_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_35_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_36_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_37_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_38_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_39_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_40_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_41_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_42_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_43_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_44_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_45_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_46_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_47_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_48_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_49_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_50_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_51_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_52_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_53_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_54_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_55_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_56_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_57_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_58_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_59_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_60_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_61_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_62_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1))  w2_rsc_63_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  b2_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_0_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_1_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_2_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_3_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_4_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_5_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_6_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_7_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_8_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_9_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_10_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_11_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_12_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_13_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_14_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_15_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_16_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_17_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_18_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_19_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_20_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_21_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_22_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_23_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_24_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_25_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_26_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_27_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_28_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_29_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_30_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_31_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_32_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_33_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_34_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_35_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_36_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_37_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_38_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_39_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_40_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_41_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_42_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_43_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_44_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_45_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_46_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_47_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_48_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_49_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_50_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_51_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_52_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_53_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_54_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_55_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_56_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_57_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_58_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_59_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_60_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_61_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_62_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w4_rsc_63_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  b4_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_0_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_1_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_2_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_3_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_4_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_5_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_6_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_7_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_8_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1))  w6_rsc_9_0_config;
  ccs_configuration  #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1))  b6_rsc_config;

  // ****************************************************************************
  function new( string name = "" );
    super.new( name );
    // Retrieve the configuration handles from the uvm_config_db

    // Retrieve top level configuration handle
    if ( !uvm_config_db#(mnist_mlp_env_configuration_t)::get(null,UVMF_CONFIGURATIONS, "TOP_ENV_CONFIG",top_configuration) ) begin
      `uvm_fatal("CFG", "uvm_config_db#(mnist_mlp_env_configuration_t)::get cannot find resource TOP_ENV_CONFIG");
    end

    // Retrieve config handles for all agents
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , input1_rsc_BFM , input1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource input1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , output1_rsc_BFM , output1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource output1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , const_size_in_1_rsc_BFM , const_size_in_1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource const_size_in_1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , const_size_out_1_rsc_BFM , const_size_out_1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource const_size_out_1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_0_0_BFM , w2_rsc_0_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_0_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_1_0_BFM , w2_rsc_1_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_1_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_2_0_BFM , w2_rsc_2_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_2_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_3_0_BFM , w2_rsc_3_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_3_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_4_0_BFM , w2_rsc_4_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_4_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_5_0_BFM , w2_rsc_5_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_5_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_6_0_BFM , w2_rsc_6_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_6_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_7_0_BFM , w2_rsc_7_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_7_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_8_0_BFM , w2_rsc_8_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_8_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_9_0_BFM , w2_rsc_9_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_9_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_10_0_BFM , w2_rsc_10_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_10_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_11_0_BFM , w2_rsc_11_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_11_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_12_0_BFM , w2_rsc_12_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_12_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_13_0_BFM , w2_rsc_13_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_13_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_14_0_BFM , w2_rsc_14_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_14_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_15_0_BFM , w2_rsc_15_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_15_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_16_0_BFM , w2_rsc_16_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_16_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_17_0_BFM , w2_rsc_17_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_17_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_18_0_BFM , w2_rsc_18_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_18_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_19_0_BFM , w2_rsc_19_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_19_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_20_0_BFM , w2_rsc_20_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_20_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_21_0_BFM , w2_rsc_21_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_21_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_22_0_BFM , w2_rsc_22_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_22_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_23_0_BFM , w2_rsc_23_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_23_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_24_0_BFM , w2_rsc_24_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_24_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_25_0_BFM , w2_rsc_25_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_25_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_26_0_BFM , w2_rsc_26_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_26_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_27_0_BFM , w2_rsc_27_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_27_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_28_0_BFM , w2_rsc_28_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_28_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_29_0_BFM , w2_rsc_29_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_29_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_30_0_BFM , w2_rsc_30_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_30_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_31_0_BFM , w2_rsc_31_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_31_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_32_0_BFM , w2_rsc_32_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_32_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_33_0_BFM , w2_rsc_33_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_33_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_34_0_BFM , w2_rsc_34_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_34_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_35_0_BFM , w2_rsc_35_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_35_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_36_0_BFM , w2_rsc_36_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_36_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_37_0_BFM , w2_rsc_37_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_37_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_38_0_BFM , w2_rsc_38_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_38_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_39_0_BFM , w2_rsc_39_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_39_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_40_0_BFM , w2_rsc_40_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_40_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_41_0_BFM , w2_rsc_41_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_41_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_42_0_BFM , w2_rsc_42_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_42_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_43_0_BFM , w2_rsc_43_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_43_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_44_0_BFM , w2_rsc_44_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_44_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_45_0_BFM , w2_rsc_45_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_45_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_46_0_BFM , w2_rsc_46_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_46_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_47_0_BFM , w2_rsc_47_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_47_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_48_0_BFM , w2_rsc_48_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_48_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_49_0_BFM , w2_rsc_49_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_49_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_50_0_BFM , w2_rsc_50_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_50_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_51_0_BFM , w2_rsc_51_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_51_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_52_0_BFM , w2_rsc_52_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_52_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_53_0_BFM , w2_rsc_53_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_53_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_54_0_BFM , w2_rsc_54_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_54_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_55_0_BFM , w2_rsc_55_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_55_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_56_0_BFM , w2_rsc_56_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_56_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_57_0_BFM , w2_rsc_57_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_57_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_58_0_BFM , w2_rsc_58_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_58_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_59_0_BFM , w2_rsc_59_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_59_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_60_0_BFM , w2_rsc_60_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_60_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_61_0_BFM , w2_rsc_61_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_61_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_62_0_BFM , w2_rsc_62_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_62_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w2_rsc_63_0_BFM , w2_rsc_63_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w2_rsc_63_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , b2_rsc_BFM , b2_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource b2_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_0_0_BFM , w4_rsc_0_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_0_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_1_0_BFM , w4_rsc_1_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_1_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_2_0_BFM , w4_rsc_2_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_2_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_3_0_BFM , w4_rsc_3_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_3_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_4_0_BFM , w4_rsc_4_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_4_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_5_0_BFM , w4_rsc_5_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_5_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_6_0_BFM , w4_rsc_6_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_6_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_7_0_BFM , w4_rsc_7_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_7_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_8_0_BFM , w4_rsc_8_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_8_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_9_0_BFM , w4_rsc_9_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_9_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_10_0_BFM , w4_rsc_10_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_10_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_11_0_BFM , w4_rsc_11_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_11_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_12_0_BFM , w4_rsc_12_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_12_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_13_0_BFM , w4_rsc_13_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_13_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_14_0_BFM , w4_rsc_14_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_14_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_15_0_BFM , w4_rsc_15_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_15_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_16_0_BFM , w4_rsc_16_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_16_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_17_0_BFM , w4_rsc_17_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_17_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_18_0_BFM , w4_rsc_18_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_18_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_19_0_BFM , w4_rsc_19_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_19_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_20_0_BFM , w4_rsc_20_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_20_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_21_0_BFM , w4_rsc_21_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_21_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_22_0_BFM , w4_rsc_22_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_22_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_23_0_BFM , w4_rsc_23_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_23_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_24_0_BFM , w4_rsc_24_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_24_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_25_0_BFM , w4_rsc_25_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_25_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_26_0_BFM , w4_rsc_26_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_26_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_27_0_BFM , w4_rsc_27_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_27_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_28_0_BFM , w4_rsc_28_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_28_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_29_0_BFM , w4_rsc_29_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_29_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_30_0_BFM , w4_rsc_30_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_30_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_31_0_BFM , w4_rsc_31_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_31_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_32_0_BFM , w4_rsc_32_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_32_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_33_0_BFM , w4_rsc_33_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_33_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_34_0_BFM , w4_rsc_34_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_34_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_35_0_BFM , w4_rsc_35_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_35_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_36_0_BFM , w4_rsc_36_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_36_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_37_0_BFM , w4_rsc_37_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_37_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_38_0_BFM , w4_rsc_38_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_38_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_39_0_BFM , w4_rsc_39_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_39_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_40_0_BFM , w4_rsc_40_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_40_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_41_0_BFM , w4_rsc_41_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_41_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_42_0_BFM , w4_rsc_42_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_42_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_43_0_BFM , w4_rsc_43_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_43_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_44_0_BFM , w4_rsc_44_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_44_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_45_0_BFM , w4_rsc_45_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_45_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_46_0_BFM , w4_rsc_46_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_46_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_47_0_BFM , w4_rsc_47_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_47_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_48_0_BFM , w4_rsc_48_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_48_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_49_0_BFM , w4_rsc_49_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_49_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_50_0_BFM , w4_rsc_50_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_50_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_51_0_BFM , w4_rsc_51_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_51_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_52_0_BFM , w4_rsc_52_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_52_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_53_0_BFM , w4_rsc_53_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_53_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_54_0_BFM , w4_rsc_54_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_54_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_55_0_BFM , w4_rsc_55_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_55_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_56_0_BFM , w4_rsc_56_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_56_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_57_0_BFM , w4_rsc_57_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_57_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_58_0_BFM , w4_rsc_58_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_58_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_59_0_BFM , w4_rsc_59_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_59_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_60_0_BFM , w4_rsc_60_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_60_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_61_0_BFM , w4_rsc_61_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_61_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_62_0_BFM , w4_rsc_62_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_62_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w4_rsc_63_0_BFM , w4_rsc_63_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w4_rsc_63_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , b4_rsc_BFM , b4_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource b4_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_0_0_BFM , w6_rsc_0_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_0_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_1_0_BFM , w6_rsc_1_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_1_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_2_0_BFM , w6_rsc_2_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_2_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_3_0_BFM , w6_rsc_3_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_3_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_4_0_BFM , w6_rsc_4_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_4_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_5_0_BFM , w6_rsc_5_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_5_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_6_0_BFM , w6_rsc_6_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_6_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_7_0_BFM , w6_rsc_7_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_7_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_8_0_BFM , w6_rsc_8_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_8_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , w6_rsc_9_0_BFM , w6_rsc_9_0_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource w6_rsc_9_0_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , b6_rsc_BFM , b6_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource b6_rsc_BFM" )

    // Assign the sequencer handles from the handles within agent configurations
    input1_rsc_sequencer = input1_rsc_config.get_sequencer();
    output1_rsc_sequencer = output1_rsc_config.get_sequencer();
    const_size_in_1_rsc_sequencer = const_size_in_1_rsc_config.get_sequencer();
    const_size_out_1_rsc_sequencer = const_size_out_1_rsc_config.get_sequencer();
    w2_rsc_0_0_sequencer = w2_rsc_0_0_config.get_sequencer();
    w2_rsc_1_0_sequencer = w2_rsc_1_0_config.get_sequencer();
    w2_rsc_2_0_sequencer = w2_rsc_2_0_config.get_sequencer();
    w2_rsc_3_0_sequencer = w2_rsc_3_0_config.get_sequencer();
    w2_rsc_4_0_sequencer = w2_rsc_4_0_config.get_sequencer();
    w2_rsc_5_0_sequencer = w2_rsc_5_0_config.get_sequencer();
    w2_rsc_6_0_sequencer = w2_rsc_6_0_config.get_sequencer();
    w2_rsc_7_0_sequencer = w2_rsc_7_0_config.get_sequencer();
    w2_rsc_8_0_sequencer = w2_rsc_8_0_config.get_sequencer();
    w2_rsc_9_0_sequencer = w2_rsc_9_0_config.get_sequencer();
    w2_rsc_10_0_sequencer = w2_rsc_10_0_config.get_sequencer();
    w2_rsc_11_0_sequencer = w2_rsc_11_0_config.get_sequencer();
    w2_rsc_12_0_sequencer = w2_rsc_12_0_config.get_sequencer();
    w2_rsc_13_0_sequencer = w2_rsc_13_0_config.get_sequencer();
    w2_rsc_14_0_sequencer = w2_rsc_14_0_config.get_sequencer();
    w2_rsc_15_0_sequencer = w2_rsc_15_0_config.get_sequencer();
    w2_rsc_16_0_sequencer = w2_rsc_16_0_config.get_sequencer();
    w2_rsc_17_0_sequencer = w2_rsc_17_0_config.get_sequencer();
    w2_rsc_18_0_sequencer = w2_rsc_18_0_config.get_sequencer();
    w2_rsc_19_0_sequencer = w2_rsc_19_0_config.get_sequencer();
    w2_rsc_20_0_sequencer = w2_rsc_20_0_config.get_sequencer();
    w2_rsc_21_0_sequencer = w2_rsc_21_0_config.get_sequencer();
    w2_rsc_22_0_sequencer = w2_rsc_22_0_config.get_sequencer();
    w2_rsc_23_0_sequencer = w2_rsc_23_0_config.get_sequencer();
    w2_rsc_24_0_sequencer = w2_rsc_24_0_config.get_sequencer();
    w2_rsc_25_0_sequencer = w2_rsc_25_0_config.get_sequencer();
    w2_rsc_26_0_sequencer = w2_rsc_26_0_config.get_sequencer();
    w2_rsc_27_0_sequencer = w2_rsc_27_0_config.get_sequencer();
    w2_rsc_28_0_sequencer = w2_rsc_28_0_config.get_sequencer();
    w2_rsc_29_0_sequencer = w2_rsc_29_0_config.get_sequencer();
    w2_rsc_30_0_sequencer = w2_rsc_30_0_config.get_sequencer();
    w2_rsc_31_0_sequencer = w2_rsc_31_0_config.get_sequencer();
    w2_rsc_32_0_sequencer = w2_rsc_32_0_config.get_sequencer();
    w2_rsc_33_0_sequencer = w2_rsc_33_0_config.get_sequencer();
    w2_rsc_34_0_sequencer = w2_rsc_34_0_config.get_sequencer();
    w2_rsc_35_0_sequencer = w2_rsc_35_0_config.get_sequencer();
    w2_rsc_36_0_sequencer = w2_rsc_36_0_config.get_sequencer();
    w2_rsc_37_0_sequencer = w2_rsc_37_0_config.get_sequencer();
    w2_rsc_38_0_sequencer = w2_rsc_38_0_config.get_sequencer();
    w2_rsc_39_0_sequencer = w2_rsc_39_0_config.get_sequencer();
    w2_rsc_40_0_sequencer = w2_rsc_40_0_config.get_sequencer();
    w2_rsc_41_0_sequencer = w2_rsc_41_0_config.get_sequencer();
    w2_rsc_42_0_sequencer = w2_rsc_42_0_config.get_sequencer();
    w2_rsc_43_0_sequencer = w2_rsc_43_0_config.get_sequencer();
    w2_rsc_44_0_sequencer = w2_rsc_44_0_config.get_sequencer();
    w2_rsc_45_0_sequencer = w2_rsc_45_0_config.get_sequencer();
    w2_rsc_46_0_sequencer = w2_rsc_46_0_config.get_sequencer();
    w2_rsc_47_0_sequencer = w2_rsc_47_0_config.get_sequencer();
    w2_rsc_48_0_sequencer = w2_rsc_48_0_config.get_sequencer();
    w2_rsc_49_0_sequencer = w2_rsc_49_0_config.get_sequencer();
    w2_rsc_50_0_sequencer = w2_rsc_50_0_config.get_sequencer();
    w2_rsc_51_0_sequencer = w2_rsc_51_0_config.get_sequencer();
    w2_rsc_52_0_sequencer = w2_rsc_52_0_config.get_sequencer();
    w2_rsc_53_0_sequencer = w2_rsc_53_0_config.get_sequencer();
    w2_rsc_54_0_sequencer = w2_rsc_54_0_config.get_sequencer();
    w2_rsc_55_0_sequencer = w2_rsc_55_0_config.get_sequencer();
    w2_rsc_56_0_sequencer = w2_rsc_56_0_config.get_sequencer();
    w2_rsc_57_0_sequencer = w2_rsc_57_0_config.get_sequencer();
    w2_rsc_58_0_sequencer = w2_rsc_58_0_config.get_sequencer();
    w2_rsc_59_0_sequencer = w2_rsc_59_0_config.get_sequencer();
    w2_rsc_60_0_sequencer = w2_rsc_60_0_config.get_sequencer();
    w2_rsc_61_0_sequencer = w2_rsc_61_0_config.get_sequencer();
    w2_rsc_62_0_sequencer = w2_rsc_62_0_config.get_sequencer();
    w2_rsc_63_0_sequencer = w2_rsc_63_0_config.get_sequencer();
    b2_rsc_sequencer = b2_rsc_config.get_sequencer();
    w4_rsc_0_0_sequencer = w4_rsc_0_0_config.get_sequencer();
    w4_rsc_1_0_sequencer = w4_rsc_1_0_config.get_sequencer();
    w4_rsc_2_0_sequencer = w4_rsc_2_0_config.get_sequencer();
    w4_rsc_3_0_sequencer = w4_rsc_3_0_config.get_sequencer();
    w4_rsc_4_0_sequencer = w4_rsc_4_0_config.get_sequencer();
    w4_rsc_5_0_sequencer = w4_rsc_5_0_config.get_sequencer();
    w4_rsc_6_0_sequencer = w4_rsc_6_0_config.get_sequencer();
    w4_rsc_7_0_sequencer = w4_rsc_7_0_config.get_sequencer();
    w4_rsc_8_0_sequencer = w4_rsc_8_0_config.get_sequencer();
    w4_rsc_9_0_sequencer = w4_rsc_9_0_config.get_sequencer();
    w4_rsc_10_0_sequencer = w4_rsc_10_0_config.get_sequencer();
    w4_rsc_11_0_sequencer = w4_rsc_11_0_config.get_sequencer();
    w4_rsc_12_0_sequencer = w4_rsc_12_0_config.get_sequencer();
    w4_rsc_13_0_sequencer = w4_rsc_13_0_config.get_sequencer();
    w4_rsc_14_0_sequencer = w4_rsc_14_0_config.get_sequencer();
    w4_rsc_15_0_sequencer = w4_rsc_15_0_config.get_sequencer();
    w4_rsc_16_0_sequencer = w4_rsc_16_0_config.get_sequencer();
    w4_rsc_17_0_sequencer = w4_rsc_17_0_config.get_sequencer();
    w4_rsc_18_0_sequencer = w4_rsc_18_0_config.get_sequencer();
    w4_rsc_19_0_sequencer = w4_rsc_19_0_config.get_sequencer();
    w4_rsc_20_0_sequencer = w4_rsc_20_0_config.get_sequencer();
    w4_rsc_21_0_sequencer = w4_rsc_21_0_config.get_sequencer();
    w4_rsc_22_0_sequencer = w4_rsc_22_0_config.get_sequencer();
    w4_rsc_23_0_sequencer = w4_rsc_23_0_config.get_sequencer();
    w4_rsc_24_0_sequencer = w4_rsc_24_0_config.get_sequencer();
    w4_rsc_25_0_sequencer = w4_rsc_25_0_config.get_sequencer();
    w4_rsc_26_0_sequencer = w4_rsc_26_0_config.get_sequencer();
    w4_rsc_27_0_sequencer = w4_rsc_27_0_config.get_sequencer();
    w4_rsc_28_0_sequencer = w4_rsc_28_0_config.get_sequencer();
    w4_rsc_29_0_sequencer = w4_rsc_29_0_config.get_sequencer();
    w4_rsc_30_0_sequencer = w4_rsc_30_0_config.get_sequencer();
    w4_rsc_31_0_sequencer = w4_rsc_31_0_config.get_sequencer();
    w4_rsc_32_0_sequencer = w4_rsc_32_0_config.get_sequencer();
    w4_rsc_33_0_sequencer = w4_rsc_33_0_config.get_sequencer();
    w4_rsc_34_0_sequencer = w4_rsc_34_0_config.get_sequencer();
    w4_rsc_35_0_sequencer = w4_rsc_35_0_config.get_sequencer();
    w4_rsc_36_0_sequencer = w4_rsc_36_0_config.get_sequencer();
    w4_rsc_37_0_sequencer = w4_rsc_37_0_config.get_sequencer();
    w4_rsc_38_0_sequencer = w4_rsc_38_0_config.get_sequencer();
    w4_rsc_39_0_sequencer = w4_rsc_39_0_config.get_sequencer();
    w4_rsc_40_0_sequencer = w4_rsc_40_0_config.get_sequencer();
    w4_rsc_41_0_sequencer = w4_rsc_41_0_config.get_sequencer();
    w4_rsc_42_0_sequencer = w4_rsc_42_0_config.get_sequencer();
    w4_rsc_43_0_sequencer = w4_rsc_43_0_config.get_sequencer();
    w4_rsc_44_0_sequencer = w4_rsc_44_0_config.get_sequencer();
    w4_rsc_45_0_sequencer = w4_rsc_45_0_config.get_sequencer();
    w4_rsc_46_0_sequencer = w4_rsc_46_0_config.get_sequencer();
    w4_rsc_47_0_sequencer = w4_rsc_47_0_config.get_sequencer();
    w4_rsc_48_0_sequencer = w4_rsc_48_0_config.get_sequencer();
    w4_rsc_49_0_sequencer = w4_rsc_49_0_config.get_sequencer();
    w4_rsc_50_0_sequencer = w4_rsc_50_0_config.get_sequencer();
    w4_rsc_51_0_sequencer = w4_rsc_51_0_config.get_sequencer();
    w4_rsc_52_0_sequencer = w4_rsc_52_0_config.get_sequencer();
    w4_rsc_53_0_sequencer = w4_rsc_53_0_config.get_sequencer();
    w4_rsc_54_0_sequencer = w4_rsc_54_0_config.get_sequencer();
    w4_rsc_55_0_sequencer = w4_rsc_55_0_config.get_sequencer();
    w4_rsc_56_0_sequencer = w4_rsc_56_0_config.get_sequencer();
    w4_rsc_57_0_sequencer = w4_rsc_57_0_config.get_sequencer();
    w4_rsc_58_0_sequencer = w4_rsc_58_0_config.get_sequencer();
    w4_rsc_59_0_sequencer = w4_rsc_59_0_config.get_sequencer();
    w4_rsc_60_0_sequencer = w4_rsc_60_0_config.get_sequencer();
    w4_rsc_61_0_sequencer = w4_rsc_61_0_config.get_sequencer();
    w4_rsc_62_0_sequencer = w4_rsc_62_0_config.get_sequencer();
    w4_rsc_63_0_sequencer = w4_rsc_63_0_config.get_sequencer();
    b4_rsc_sequencer = b4_rsc_config.get_sequencer();
    w6_rsc_0_0_sequencer = w6_rsc_0_0_config.get_sequencer();
    w6_rsc_1_0_sequencer = w6_rsc_1_0_config.get_sequencer();
    w6_rsc_2_0_sequencer = w6_rsc_2_0_config.get_sequencer();
    w6_rsc_3_0_sequencer = w6_rsc_3_0_config.get_sequencer();
    w6_rsc_4_0_sequencer = w6_rsc_4_0_config.get_sequencer();
    w6_rsc_5_0_sequencer = w6_rsc_5_0_config.get_sequencer();
    w6_rsc_6_0_sequencer = w6_rsc_6_0_config.get_sequencer();
    w6_rsc_7_0_sequencer = w6_rsc_7_0_config.get_sequencer();
    w6_rsc_8_0_sequencer = w6_rsc_8_0_config.get_sequencer();
    w6_rsc_9_0_sequencer = w6_rsc_9_0_config.get_sequencer();
    b6_rsc_sequencer = b6_rsc_config.get_sequencer();


  endfunction

  // ****************************************************************************
  virtual task body();
    // Construct sequences here
    input1_rsc_random_seq     = input1_rsc_random_seq_t::type_id::create("input1_rsc_random_seq");
    output1_rsc_random_seq     = output1_rsc_random_seq_t::type_id::create("output1_rsc_random_seq");
    const_size_in_1_rsc_random_seq     = const_size_in_1_rsc_random_seq_t::type_id::create("const_size_in_1_rsc_random_seq");
    const_size_out_1_rsc_random_seq     = const_size_out_1_rsc_random_seq_t::type_id::create("const_size_out_1_rsc_random_seq");
    w2_rsc_0_0_random_seq     = w2_rsc_0_0_random_seq_t::type_id::create("w2_rsc_0_0_random_seq");
    w2_rsc_1_0_random_seq     = w2_rsc_1_0_random_seq_t::type_id::create("w2_rsc_1_0_random_seq");
    w2_rsc_2_0_random_seq     = w2_rsc_2_0_random_seq_t::type_id::create("w2_rsc_2_0_random_seq");
    w2_rsc_3_0_random_seq     = w2_rsc_3_0_random_seq_t::type_id::create("w2_rsc_3_0_random_seq");
    w2_rsc_4_0_random_seq     = w2_rsc_4_0_random_seq_t::type_id::create("w2_rsc_4_0_random_seq");
    w2_rsc_5_0_random_seq     = w2_rsc_5_0_random_seq_t::type_id::create("w2_rsc_5_0_random_seq");
    w2_rsc_6_0_random_seq     = w2_rsc_6_0_random_seq_t::type_id::create("w2_rsc_6_0_random_seq");
    w2_rsc_7_0_random_seq     = w2_rsc_7_0_random_seq_t::type_id::create("w2_rsc_7_0_random_seq");
    w2_rsc_8_0_random_seq     = w2_rsc_8_0_random_seq_t::type_id::create("w2_rsc_8_0_random_seq");
    w2_rsc_9_0_random_seq     = w2_rsc_9_0_random_seq_t::type_id::create("w2_rsc_9_0_random_seq");
    w2_rsc_10_0_random_seq     = w2_rsc_10_0_random_seq_t::type_id::create("w2_rsc_10_0_random_seq");
    w2_rsc_11_0_random_seq     = w2_rsc_11_0_random_seq_t::type_id::create("w2_rsc_11_0_random_seq");
    w2_rsc_12_0_random_seq     = w2_rsc_12_0_random_seq_t::type_id::create("w2_rsc_12_0_random_seq");
    w2_rsc_13_0_random_seq     = w2_rsc_13_0_random_seq_t::type_id::create("w2_rsc_13_0_random_seq");
    w2_rsc_14_0_random_seq     = w2_rsc_14_0_random_seq_t::type_id::create("w2_rsc_14_0_random_seq");
    w2_rsc_15_0_random_seq     = w2_rsc_15_0_random_seq_t::type_id::create("w2_rsc_15_0_random_seq");
    w2_rsc_16_0_random_seq     = w2_rsc_16_0_random_seq_t::type_id::create("w2_rsc_16_0_random_seq");
    w2_rsc_17_0_random_seq     = w2_rsc_17_0_random_seq_t::type_id::create("w2_rsc_17_0_random_seq");
    w2_rsc_18_0_random_seq     = w2_rsc_18_0_random_seq_t::type_id::create("w2_rsc_18_0_random_seq");
    w2_rsc_19_0_random_seq     = w2_rsc_19_0_random_seq_t::type_id::create("w2_rsc_19_0_random_seq");
    w2_rsc_20_0_random_seq     = w2_rsc_20_0_random_seq_t::type_id::create("w2_rsc_20_0_random_seq");
    w2_rsc_21_0_random_seq     = w2_rsc_21_0_random_seq_t::type_id::create("w2_rsc_21_0_random_seq");
    w2_rsc_22_0_random_seq     = w2_rsc_22_0_random_seq_t::type_id::create("w2_rsc_22_0_random_seq");
    w2_rsc_23_0_random_seq     = w2_rsc_23_0_random_seq_t::type_id::create("w2_rsc_23_0_random_seq");
    w2_rsc_24_0_random_seq     = w2_rsc_24_0_random_seq_t::type_id::create("w2_rsc_24_0_random_seq");
    w2_rsc_25_0_random_seq     = w2_rsc_25_0_random_seq_t::type_id::create("w2_rsc_25_0_random_seq");
    w2_rsc_26_0_random_seq     = w2_rsc_26_0_random_seq_t::type_id::create("w2_rsc_26_0_random_seq");
    w2_rsc_27_0_random_seq     = w2_rsc_27_0_random_seq_t::type_id::create("w2_rsc_27_0_random_seq");
    w2_rsc_28_0_random_seq     = w2_rsc_28_0_random_seq_t::type_id::create("w2_rsc_28_0_random_seq");
    w2_rsc_29_0_random_seq     = w2_rsc_29_0_random_seq_t::type_id::create("w2_rsc_29_0_random_seq");
    w2_rsc_30_0_random_seq     = w2_rsc_30_0_random_seq_t::type_id::create("w2_rsc_30_0_random_seq");
    w2_rsc_31_0_random_seq     = w2_rsc_31_0_random_seq_t::type_id::create("w2_rsc_31_0_random_seq");
    w2_rsc_32_0_random_seq     = w2_rsc_32_0_random_seq_t::type_id::create("w2_rsc_32_0_random_seq");
    w2_rsc_33_0_random_seq     = w2_rsc_33_0_random_seq_t::type_id::create("w2_rsc_33_0_random_seq");
    w2_rsc_34_0_random_seq     = w2_rsc_34_0_random_seq_t::type_id::create("w2_rsc_34_0_random_seq");
    w2_rsc_35_0_random_seq     = w2_rsc_35_0_random_seq_t::type_id::create("w2_rsc_35_0_random_seq");
    w2_rsc_36_0_random_seq     = w2_rsc_36_0_random_seq_t::type_id::create("w2_rsc_36_0_random_seq");
    w2_rsc_37_0_random_seq     = w2_rsc_37_0_random_seq_t::type_id::create("w2_rsc_37_0_random_seq");
    w2_rsc_38_0_random_seq     = w2_rsc_38_0_random_seq_t::type_id::create("w2_rsc_38_0_random_seq");
    w2_rsc_39_0_random_seq     = w2_rsc_39_0_random_seq_t::type_id::create("w2_rsc_39_0_random_seq");
    w2_rsc_40_0_random_seq     = w2_rsc_40_0_random_seq_t::type_id::create("w2_rsc_40_0_random_seq");
    w2_rsc_41_0_random_seq     = w2_rsc_41_0_random_seq_t::type_id::create("w2_rsc_41_0_random_seq");
    w2_rsc_42_0_random_seq     = w2_rsc_42_0_random_seq_t::type_id::create("w2_rsc_42_0_random_seq");
    w2_rsc_43_0_random_seq     = w2_rsc_43_0_random_seq_t::type_id::create("w2_rsc_43_0_random_seq");
    w2_rsc_44_0_random_seq     = w2_rsc_44_0_random_seq_t::type_id::create("w2_rsc_44_0_random_seq");
    w2_rsc_45_0_random_seq     = w2_rsc_45_0_random_seq_t::type_id::create("w2_rsc_45_0_random_seq");
    w2_rsc_46_0_random_seq     = w2_rsc_46_0_random_seq_t::type_id::create("w2_rsc_46_0_random_seq");
    w2_rsc_47_0_random_seq     = w2_rsc_47_0_random_seq_t::type_id::create("w2_rsc_47_0_random_seq");
    w2_rsc_48_0_random_seq     = w2_rsc_48_0_random_seq_t::type_id::create("w2_rsc_48_0_random_seq");
    w2_rsc_49_0_random_seq     = w2_rsc_49_0_random_seq_t::type_id::create("w2_rsc_49_0_random_seq");
    w2_rsc_50_0_random_seq     = w2_rsc_50_0_random_seq_t::type_id::create("w2_rsc_50_0_random_seq");
    w2_rsc_51_0_random_seq     = w2_rsc_51_0_random_seq_t::type_id::create("w2_rsc_51_0_random_seq");
    w2_rsc_52_0_random_seq     = w2_rsc_52_0_random_seq_t::type_id::create("w2_rsc_52_0_random_seq");
    w2_rsc_53_0_random_seq     = w2_rsc_53_0_random_seq_t::type_id::create("w2_rsc_53_0_random_seq");
    w2_rsc_54_0_random_seq     = w2_rsc_54_0_random_seq_t::type_id::create("w2_rsc_54_0_random_seq");
    w2_rsc_55_0_random_seq     = w2_rsc_55_0_random_seq_t::type_id::create("w2_rsc_55_0_random_seq");
    w2_rsc_56_0_random_seq     = w2_rsc_56_0_random_seq_t::type_id::create("w2_rsc_56_0_random_seq");
    w2_rsc_57_0_random_seq     = w2_rsc_57_0_random_seq_t::type_id::create("w2_rsc_57_0_random_seq");
    w2_rsc_58_0_random_seq     = w2_rsc_58_0_random_seq_t::type_id::create("w2_rsc_58_0_random_seq");
    w2_rsc_59_0_random_seq     = w2_rsc_59_0_random_seq_t::type_id::create("w2_rsc_59_0_random_seq");
    w2_rsc_60_0_random_seq     = w2_rsc_60_0_random_seq_t::type_id::create("w2_rsc_60_0_random_seq");
    w2_rsc_61_0_random_seq     = w2_rsc_61_0_random_seq_t::type_id::create("w2_rsc_61_0_random_seq");
    w2_rsc_62_0_random_seq     = w2_rsc_62_0_random_seq_t::type_id::create("w2_rsc_62_0_random_seq");
    w2_rsc_63_0_random_seq     = w2_rsc_63_0_random_seq_t::type_id::create("w2_rsc_63_0_random_seq");
    b2_rsc_random_seq     = b2_rsc_random_seq_t::type_id::create("b2_rsc_random_seq");
    w4_rsc_0_0_random_seq     = w4_rsc_0_0_random_seq_t::type_id::create("w4_rsc_0_0_random_seq");
    w4_rsc_1_0_random_seq     = w4_rsc_1_0_random_seq_t::type_id::create("w4_rsc_1_0_random_seq");
    w4_rsc_2_0_random_seq     = w4_rsc_2_0_random_seq_t::type_id::create("w4_rsc_2_0_random_seq");
    w4_rsc_3_0_random_seq     = w4_rsc_3_0_random_seq_t::type_id::create("w4_rsc_3_0_random_seq");
    w4_rsc_4_0_random_seq     = w4_rsc_4_0_random_seq_t::type_id::create("w4_rsc_4_0_random_seq");
    w4_rsc_5_0_random_seq     = w4_rsc_5_0_random_seq_t::type_id::create("w4_rsc_5_0_random_seq");
    w4_rsc_6_0_random_seq     = w4_rsc_6_0_random_seq_t::type_id::create("w4_rsc_6_0_random_seq");
    w4_rsc_7_0_random_seq     = w4_rsc_7_0_random_seq_t::type_id::create("w4_rsc_7_0_random_seq");
    w4_rsc_8_0_random_seq     = w4_rsc_8_0_random_seq_t::type_id::create("w4_rsc_8_0_random_seq");
    w4_rsc_9_0_random_seq     = w4_rsc_9_0_random_seq_t::type_id::create("w4_rsc_9_0_random_seq");
    w4_rsc_10_0_random_seq     = w4_rsc_10_0_random_seq_t::type_id::create("w4_rsc_10_0_random_seq");
    w4_rsc_11_0_random_seq     = w4_rsc_11_0_random_seq_t::type_id::create("w4_rsc_11_0_random_seq");
    w4_rsc_12_0_random_seq     = w4_rsc_12_0_random_seq_t::type_id::create("w4_rsc_12_0_random_seq");
    w4_rsc_13_0_random_seq     = w4_rsc_13_0_random_seq_t::type_id::create("w4_rsc_13_0_random_seq");
    w4_rsc_14_0_random_seq     = w4_rsc_14_0_random_seq_t::type_id::create("w4_rsc_14_0_random_seq");
    w4_rsc_15_0_random_seq     = w4_rsc_15_0_random_seq_t::type_id::create("w4_rsc_15_0_random_seq");
    w4_rsc_16_0_random_seq     = w4_rsc_16_0_random_seq_t::type_id::create("w4_rsc_16_0_random_seq");
    w4_rsc_17_0_random_seq     = w4_rsc_17_0_random_seq_t::type_id::create("w4_rsc_17_0_random_seq");
    w4_rsc_18_0_random_seq     = w4_rsc_18_0_random_seq_t::type_id::create("w4_rsc_18_0_random_seq");
    w4_rsc_19_0_random_seq     = w4_rsc_19_0_random_seq_t::type_id::create("w4_rsc_19_0_random_seq");
    w4_rsc_20_0_random_seq     = w4_rsc_20_0_random_seq_t::type_id::create("w4_rsc_20_0_random_seq");
    w4_rsc_21_0_random_seq     = w4_rsc_21_0_random_seq_t::type_id::create("w4_rsc_21_0_random_seq");
    w4_rsc_22_0_random_seq     = w4_rsc_22_0_random_seq_t::type_id::create("w4_rsc_22_0_random_seq");
    w4_rsc_23_0_random_seq     = w4_rsc_23_0_random_seq_t::type_id::create("w4_rsc_23_0_random_seq");
    w4_rsc_24_0_random_seq     = w4_rsc_24_0_random_seq_t::type_id::create("w4_rsc_24_0_random_seq");
    w4_rsc_25_0_random_seq     = w4_rsc_25_0_random_seq_t::type_id::create("w4_rsc_25_0_random_seq");
    w4_rsc_26_0_random_seq     = w4_rsc_26_0_random_seq_t::type_id::create("w4_rsc_26_0_random_seq");
    w4_rsc_27_0_random_seq     = w4_rsc_27_0_random_seq_t::type_id::create("w4_rsc_27_0_random_seq");
    w4_rsc_28_0_random_seq     = w4_rsc_28_0_random_seq_t::type_id::create("w4_rsc_28_0_random_seq");
    w4_rsc_29_0_random_seq     = w4_rsc_29_0_random_seq_t::type_id::create("w4_rsc_29_0_random_seq");
    w4_rsc_30_0_random_seq     = w4_rsc_30_0_random_seq_t::type_id::create("w4_rsc_30_0_random_seq");
    w4_rsc_31_0_random_seq     = w4_rsc_31_0_random_seq_t::type_id::create("w4_rsc_31_0_random_seq");
    w4_rsc_32_0_random_seq     = w4_rsc_32_0_random_seq_t::type_id::create("w4_rsc_32_0_random_seq");
    w4_rsc_33_0_random_seq     = w4_rsc_33_0_random_seq_t::type_id::create("w4_rsc_33_0_random_seq");
    w4_rsc_34_0_random_seq     = w4_rsc_34_0_random_seq_t::type_id::create("w4_rsc_34_0_random_seq");
    w4_rsc_35_0_random_seq     = w4_rsc_35_0_random_seq_t::type_id::create("w4_rsc_35_0_random_seq");
    w4_rsc_36_0_random_seq     = w4_rsc_36_0_random_seq_t::type_id::create("w4_rsc_36_0_random_seq");
    w4_rsc_37_0_random_seq     = w4_rsc_37_0_random_seq_t::type_id::create("w4_rsc_37_0_random_seq");
    w4_rsc_38_0_random_seq     = w4_rsc_38_0_random_seq_t::type_id::create("w4_rsc_38_0_random_seq");
    w4_rsc_39_0_random_seq     = w4_rsc_39_0_random_seq_t::type_id::create("w4_rsc_39_0_random_seq");
    w4_rsc_40_0_random_seq     = w4_rsc_40_0_random_seq_t::type_id::create("w4_rsc_40_0_random_seq");
    w4_rsc_41_0_random_seq     = w4_rsc_41_0_random_seq_t::type_id::create("w4_rsc_41_0_random_seq");
    w4_rsc_42_0_random_seq     = w4_rsc_42_0_random_seq_t::type_id::create("w4_rsc_42_0_random_seq");
    w4_rsc_43_0_random_seq     = w4_rsc_43_0_random_seq_t::type_id::create("w4_rsc_43_0_random_seq");
    w4_rsc_44_0_random_seq     = w4_rsc_44_0_random_seq_t::type_id::create("w4_rsc_44_0_random_seq");
    w4_rsc_45_0_random_seq     = w4_rsc_45_0_random_seq_t::type_id::create("w4_rsc_45_0_random_seq");
    w4_rsc_46_0_random_seq     = w4_rsc_46_0_random_seq_t::type_id::create("w4_rsc_46_0_random_seq");
    w4_rsc_47_0_random_seq     = w4_rsc_47_0_random_seq_t::type_id::create("w4_rsc_47_0_random_seq");
    w4_rsc_48_0_random_seq     = w4_rsc_48_0_random_seq_t::type_id::create("w4_rsc_48_0_random_seq");
    w4_rsc_49_0_random_seq     = w4_rsc_49_0_random_seq_t::type_id::create("w4_rsc_49_0_random_seq");
    w4_rsc_50_0_random_seq     = w4_rsc_50_0_random_seq_t::type_id::create("w4_rsc_50_0_random_seq");
    w4_rsc_51_0_random_seq     = w4_rsc_51_0_random_seq_t::type_id::create("w4_rsc_51_0_random_seq");
    w4_rsc_52_0_random_seq     = w4_rsc_52_0_random_seq_t::type_id::create("w4_rsc_52_0_random_seq");
    w4_rsc_53_0_random_seq     = w4_rsc_53_0_random_seq_t::type_id::create("w4_rsc_53_0_random_seq");
    w4_rsc_54_0_random_seq     = w4_rsc_54_0_random_seq_t::type_id::create("w4_rsc_54_0_random_seq");
    w4_rsc_55_0_random_seq     = w4_rsc_55_0_random_seq_t::type_id::create("w4_rsc_55_0_random_seq");
    w4_rsc_56_0_random_seq     = w4_rsc_56_0_random_seq_t::type_id::create("w4_rsc_56_0_random_seq");
    w4_rsc_57_0_random_seq     = w4_rsc_57_0_random_seq_t::type_id::create("w4_rsc_57_0_random_seq");
    w4_rsc_58_0_random_seq     = w4_rsc_58_0_random_seq_t::type_id::create("w4_rsc_58_0_random_seq");
    w4_rsc_59_0_random_seq     = w4_rsc_59_0_random_seq_t::type_id::create("w4_rsc_59_0_random_seq");
    w4_rsc_60_0_random_seq     = w4_rsc_60_0_random_seq_t::type_id::create("w4_rsc_60_0_random_seq");
    w4_rsc_61_0_random_seq     = w4_rsc_61_0_random_seq_t::type_id::create("w4_rsc_61_0_random_seq");
    w4_rsc_62_0_random_seq     = w4_rsc_62_0_random_seq_t::type_id::create("w4_rsc_62_0_random_seq");
    w4_rsc_63_0_random_seq     = w4_rsc_63_0_random_seq_t::type_id::create("w4_rsc_63_0_random_seq");
    b4_rsc_random_seq     = b4_rsc_random_seq_t::type_id::create("b4_rsc_random_seq");
    w6_rsc_0_0_random_seq     = w6_rsc_0_0_random_seq_t::type_id::create("w6_rsc_0_0_random_seq");
    w6_rsc_1_0_random_seq     = w6_rsc_1_0_random_seq_t::type_id::create("w6_rsc_1_0_random_seq");
    w6_rsc_2_0_random_seq     = w6_rsc_2_0_random_seq_t::type_id::create("w6_rsc_2_0_random_seq");
    w6_rsc_3_0_random_seq     = w6_rsc_3_0_random_seq_t::type_id::create("w6_rsc_3_0_random_seq");
    w6_rsc_4_0_random_seq     = w6_rsc_4_0_random_seq_t::type_id::create("w6_rsc_4_0_random_seq");
    w6_rsc_5_0_random_seq     = w6_rsc_5_0_random_seq_t::type_id::create("w6_rsc_5_0_random_seq");
    w6_rsc_6_0_random_seq     = w6_rsc_6_0_random_seq_t::type_id::create("w6_rsc_6_0_random_seq");
    w6_rsc_7_0_random_seq     = w6_rsc_7_0_random_seq_t::type_id::create("w6_rsc_7_0_random_seq");
    w6_rsc_8_0_random_seq     = w6_rsc_8_0_random_seq_t::type_id::create("w6_rsc_8_0_random_seq");
    w6_rsc_9_0_random_seq     = w6_rsc_9_0_random_seq_t::type_id::create("w6_rsc_9_0_random_seq");
    b6_rsc_random_seq     = b6_rsc_random_seq_t::type_id::create("b6_rsc_random_seq");
    fork
      input1_rsc_config.wait_for_reset();
      output1_rsc_config.wait_for_reset();
      const_size_in_1_rsc_config.wait_for_reset();
      const_size_out_1_rsc_config.wait_for_reset();
      w2_rsc_0_0_config.wait_for_reset();
      w2_rsc_1_0_config.wait_for_reset();
      w2_rsc_2_0_config.wait_for_reset();
      w2_rsc_3_0_config.wait_for_reset();
      w2_rsc_4_0_config.wait_for_reset();
      w2_rsc_5_0_config.wait_for_reset();
      w2_rsc_6_0_config.wait_for_reset();
      w2_rsc_7_0_config.wait_for_reset();
      w2_rsc_8_0_config.wait_for_reset();
      w2_rsc_9_0_config.wait_for_reset();
      w2_rsc_10_0_config.wait_for_reset();
      w2_rsc_11_0_config.wait_for_reset();
      w2_rsc_12_0_config.wait_for_reset();
      w2_rsc_13_0_config.wait_for_reset();
      w2_rsc_14_0_config.wait_for_reset();
      w2_rsc_15_0_config.wait_for_reset();
      w2_rsc_16_0_config.wait_for_reset();
      w2_rsc_17_0_config.wait_for_reset();
      w2_rsc_18_0_config.wait_for_reset();
      w2_rsc_19_0_config.wait_for_reset();
      w2_rsc_20_0_config.wait_for_reset();
      w2_rsc_21_0_config.wait_for_reset();
      w2_rsc_22_0_config.wait_for_reset();
      w2_rsc_23_0_config.wait_for_reset();
      w2_rsc_24_0_config.wait_for_reset();
      w2_rsc_25_0_config.wait_for_reset();
      w2_rsc_26_0_config.wait_for_reset();
      w2_rsc_27_0_config.wait_for_reset();
      w2_rsc_28_0_config.wait_for_reset();
      w2_rsc_29_0_config.wait_for_reset();
      w2_rsc_30_0_config.wait_for_reset();
      w2_rsc_31_0_config.wait_for_reset();
      w2_rsc_32_0_config.wait_for_reset();
      w2_rsc_33_0_config.wait_for_reset();
      w2_rsc_34_0_config.wait_for_reset();
      w2_rsc_35_0_config.wait_for_reset();
      w2_rsc_36_0_config.wait_for_reset();
      w2_rsc_37_0_config.wait_for_reset();
      w2_rsc_38_0_config.wait_for_reset();
      w2_rsc_39_0_config.wait_for_reset();
      w2_rsc_40_0_config.wait_for_reset();
      w2_rsc_41_0_config.wait_for_reset();
      w2_rsc_42_0_config.wait_for_reset();
      w2_rsc_43_0_config.wait_for_reset();
      w2_rsc_44_0_config.wait_for_reset();
      w2_rsc_45_0_config.wait_for_reset();
      w2_rsc_46_0_config.wait_for_reset();
      w2_rsc_47_0_config.wait_for_reset();
      w2_rsc_48_0_config.wait_for_reset();
      w2_rsc_49_0_config.wait_for_reset();
      w2_rsc_50_0_config.wait_for_reset();
      w2_rsc_51_0_config.wait_for_reset();
      w2_rsc_52_0_config.wait_for_reset();
      w2_rsc_53_0_config.wait_for_reset();
      w2_rsc_54_0_config.wait_for_reset();
      w2_rsc_55_0_config.wait_for_reset();
      w2_rsc_56_0_config.wait_for_reset();
      w2_rsc_57_0_config.wait_for_reset();
      w2_rsc_58_0_config.wait_for_reset();
      w2_rsc_59_0_config.wait_for_reset();
      w2_rsc_60_0_config.wait_for_reset();
      w2_rsc_61_0_config.wait_for_reset();
      w2_rsc_62_0_config.wait_for_reset();
      w2_rsc_63_0_config.wait_for_reset();
      b2_rsc_config.wait_for_reset();
      w4_rsc_0_0_config.wait_for_reset();
      w4_rsc_1_0_config.wait_for_reset();
      w4_rsc_2_0_config.wait_for_reset();
      w4_rsc_3_0_config.wait_for_reset();
      w4_rsc_4_0_config.wait_for_reset();
      w4_rsc_5_0_config.wait_for_reset();
      w4_rsc_6_0_config.wait_for_reset();
      w4_rsc_7_0_config.wait_for_reset();
      w4_rsc_8_0_config.wait_for_reset();
      w4_rsc_9_0_config.wait_for_reset();
      w4_rsc_10_0_config.wait_for_reset();
      w4_rsc_11_0_config.wait_for_reset();
      w4_rsc_12_0_config.wait_for_reset();
      w4_rsc_13_0_config.wait_for_reset();
      w4_rsc_14_0_config.wait_for_reset();
      w4_rsc_15_0_config.wait_for_reset();
      w4_rsc_16_0_config.wait_for_reset();
      w4_rsc_17_0_config.wait_for_reset();
      w4_rsc_18_0_config.wait_for_reset();
      w4_rsc_19_0_config.wait_for_reset();
      w4_rsc_20_0_config.wait_for_reset();
      w4_rsc_21_0_config.wait_for_reset();
      w4_rsc_22_0_config.wait_for_reset();
      w4_rsc_23_0_config.wait_for_reset();
      w4_rsc_24_0_config.wait_for_reset();
      w4_rsc_25_0_config.wait_for_reset();
      w4_rsc_26_0_config.wait_for_reset();
      w4_rsc_27_0_config.wait_for_reset();
      w4_rsc_28_0_config.wait_for_reset();
      w4_rsc_29_0_config.wait_for_reset();
      w4_rsc_30_0_config.wait_for_reset();
      w4_rsc_31_0_config.wait_for_reset();
      w4_rsc_32_0_config.wait_for_reset();
      w4_rsc_33_0_config.wait_for_reset();
      w4_rsc_34_0_config.wait_for_reset();
      w4_rsc_35_0_config.wait_for_reset();
      w4_rsc_36_0_config.wait_for_reset();
      w4_rsc_37_0_config.wait_for_reset();
      w4_rsc_38_0_config.wait_for_reset();
      w4_rsc_39_0_config.wait_for_reset();
      w4_rsc_40_0_config.wait_for_reset();
      w4_rsc_41_0_config.wait_for_reset();
      w4_rsc_42_0_config.wait_for_reset();
      w4_rsc_43_0_config.wait_for_reset();
      w4_rsc_44_0_config.wait_for_reset();
      w4_rsc_45_0_config.wait_for_reset();
      w4_rsc_46_0_config.wait_for_reset();
      w4_rsc_47_0_config.wait_for_reset();
      w4_rsc_48_0_config.wait_for_reset();
      w4_rsc_49_0_config.wait_for_reset();
      w4_rsc_50_0_config.wait_for_reset();
      w4_rsc_51_0_config.wait_for_reset();
      w4_rsc_52_0_config.wait_for_reset();
      w4_rsc_53_0_config.wait_for_reset();
      w4_rsc_54_0_config.wait_for_reset();
      w4_rsc_55_0_config.wait_for_reset();
      w4_rsc_56_0_config.wait_for_reset();
      w4_rsc_57_0_config.wait_for_reset();
      w4_rsc_58_0_config.wait_for_reset();
      w4_rsc_59_0_config.wait_for_reset();
      w4_rsc_60_0_config.wait_for_reset();
      w4_rsc_61_0_config.wait_for_reset();
      w4_rsc_62_0_config.wait_for_reset();
      w4_rsc_63_0_config.wait_for_reset();
      b4_rsc_config.wait_for_reset();
      w6_rsc_0_0_config.wait_for_reset();
      w6_rsc_1_0_config.wait_for_reset();
      w6_rsc_2_0_config.wait_for_reset();
      w6_rsc_3_0_config.wait_for_reset();
      w6_rsc_4_0_config.wait_for_reset();
      w6_rsc_5_0_config.wait_for_reset();
      w6_rsc_6_0_config.wait_for_reset();
      w6_rsc_7_0_config.wait_for_reset();
      w6_rsc_8_0_config.wait_for_reset();
      w6_rsc_9_0_config.wait_for_reset();
      b6_rsc_config.wait_for_reset();
    join
    // Start RESPONDER sequences here
    fork
    join_none
    // Start INITIATOR sequences here
    fork
      repeat (25) input1_rsc_random_seq.start(input1_rsc_sequencer);
      repeat (25) output1_rsc_random_seq.start(output1_rsc_sequencer);
      repeat (25) const_size_in_1_rsc_random_seq.start(const_size_in_1_rsc_sequencer);
      repeat (25) const_size_out_1_rsc_random_seq.start(const_size_out_1_rsc_sequencer);
      repeat (25) w2_rsc_0_0_random_seq.start(w2_rsc_0_0_sequencer);
      repeat (25) w2_rsc_1_0_random_seq.start(w2_rsc_1_0_sequencer);
      repeat (25) w2_rsc_2_0_random_seq.start(w2_rsc_2_0_sequencer);
      repeat (25) w2_rsc_3_0_random_seq.start(w2_rsc_3_0_sequencer);
      repeat (25) w2_rsc_4_0_random_seq.start(w2_rsc_4_0_sequencer);
      repeat (25) w2_rsc_5_0_random_seq.start(w2_rsc_5_0_sequencer);
      repeat (25) w2_rsc_6_0_random_seq.start(w2_rsc_6_0_sequencer);
      repeat (25) w2_rsc_7_0_random_seq.start(w2_rsc_7_0_sequencer);
      repeat (25) w2_rsc_8_0_random_seq.start(w2_rsc_8_0_sequencer);
      repeat (25) w2_rsc_9_0_random_seq.start(w2_rsc_9_0_sequencer);
      repeat (25) w2_rsc_10_0_random_seq.start(w2_rsc_10_0_sequencer);
      repeat (25) w2_rsc_11_0_random_seq.start(w2_rsc_11_0_sequencer);
      repeat (25) w2_rsc_12_0_random_seq.start(w2_rsc_12_0_sequencer);
      repeat (25) w2_rsc_13_0_random_seq.start(w2_rsc_13_0_sequencer);
      repeat (25) w2_rsc_14_0_random_seq.start(w2_rsc_14_0_sequencer);
      repeat (25) w2_rsc_15_0_random_seq.start(w2_rsc_15_0_sequencer);
      repeat (25) w2_rsc_16_0_random_seq.start(w2_rsc_16_0_sequencer);
      repeat (25) w2_rsc_17_0_random_seq.start(w2_rsc_17_0_sequencer);
      repeat (25) w2_rsc_18_0_random_seq.start(w2_rsc_18_0_sequencer);
      repeat (25) w2_rsc_19_0_random_seq.start(w2_rsc_19_0_sequencer);
      repeat (25) w2_rsc_20_0_random_seq.start(w2_rsc_20_0_sequencer);
      repeat (25) w2_rsc_21_0_random_seq.start(w2_rsc_21_0_sequencer);
      repeat (25) w2_rsc_22_0_random_seq.start(w2_rsc_22_0_sequencer);
      repeat (25) w2_rsc_23_0_random_seq.start(w2_rsc_23_0_sequencer);
      repeat (25) w2_rsc_24_0_random_seq.start(w2_rsc_24_0_sequencer);
      repeat (25) w2_rsc_25_0_random_seq.start(w2_rsc_25_0_sequencer);
      repeat (25) w2_rsc_26_0_random_seq.start(w2_rsc_26_0_sequencer);
      repeat (25) w2_rsc_27_0_random_seq.start(w2_rsc_27_0_sequencer);
      repeat (25) w2_rsc_28_0_random_seq.start(w2_rsc_28_0_sequencer);
      repeat (25) w2_rsc_29_0_random_seq.start(w2_rsc_29_0_sequencer);
      repeat (25) w2_rsc_30_0_random_seq.start(w2_rsc_30_0_sequencer);
      repeat (25) w2_rsc_31_0_random_seq.start(w2_rsc_31_0_sequencer);
      repeat (25) w2_rsc_32_0_random_seq.start(w2_rsc_32_0_sequencer);
      repeat (25) w2_rsc_33_0_random_seq.start(w2_rsc_33_0_sequencer);
      repeat (25) w2_rsc_34_0_random_seq.start(w2_rsc_34_0_sequencer);
      repeat (25) w2_rsc_35_0_random_seq.start(w2_rsc_35_0_sequencer);
      repeat (25) w2_rsc_36_0_random_seq.start(w2_rsc_36_0_sequencer);
      repeat (25) w2_rsc_37_0_random_seq.start(w2_rsc_37_0_sequencer);
      repeat (25) w2_rsc_38_0_random_seq.start(w2_rsc_38_0_sequencer);
      repeat (25) w2_rsc_39_0_random_seq.start(w2_rsc_39_0_sequencer);
      repeat (25) w2_rsc_40_0_random_seq.start(w2_rsc_40_0_sequencer);
      repeat (25) w2_rsc_41_0_random_seq.start(w2_rsc_41_0_sequencer);
      repeat (25) w2_rsc_42_0_random_seq.start(w2_rsc_42_0_sequencer);
      repeat (25) w2_rsc_43_0_random_seq.start(w2_rsc_43_0_sequencer);
      repeat (25) w2_rsc_44_0_random_seq.start(w2_rsc_44_0_sequencer);
      repeat (25) w2_rsc_45_0_random_seq.start(w2_rsc_45_0_sequencer);
      repeat (25) w2_rsc_46_0_random_seq.start(w2_rsc_46_0_sequencer);
      repeat (25) w2_rsc_47_0_random_seq.start(w2_rsc_47_0_sequencer);
      repeat (25) w2_rsc_48_0_random_seq.start(w2_rsc_48_0_sequencer);
      repeat (25) w2_rsc_49_0_random_seq.start(w2_rsc_49_0_sequencer);
      repeat (25) w2_rsc_50_0_random_seq.start(w2_rsc_50_0_sequencer);
      repeat (25) w2_rsc_51_0_random_seq.start(w2_rsc_51_0_sequencer);
      repeat (25) w2_rsc_52_0_random_seq.start(w2_rsc_52_0_sequencer);
      repeat (25) w2_rsc_53_0_random_seq.start(w2_rsc_53_0_sequencer);
      repeat (25) w2_rsc_54_0_random_seq.start(w2_rsc_54_0_sequencer);
      repeat (25) w2_rsc_55_0_random_seq.start(w2_rsc_55_0_sequencer);
      repeat (25) w2_rsc_56_0_random_seq.start(w2_rsc_56_0_sequencer);
      repeat (25) w2_rsc_57_0_random_seq.start(w2_rsc_57_0_sequencer);
      repeat (25) w2_rsc_58_0_random_seq.start(w2_rsc_58_0_sequencer);
      repeat (25) w2_rsc_59_0_random_seq.start(w2_rsc_59_0_sequencer);
      repeat (25) w2_rsc_60_0_random_seq.start(w2_rsc_60_0_sequencer);
      repeat (25) w2_rsc_61_0_random_seq.start(w2_rsc_61_0_sequencer);
      repeat (25) w2_rsc_62_0_random_seq.start(w2_rsc_62_0_sequencer);
      repeat (25) w2_rsc_63_0_random_seq.start(w2_rsc_63_0_sequencer);
      repeat (25) b2_rsc_random_seq.start(b2_rsc_sequencer);
      repeat (25) w4_rsc_0_0_random_seq.start(w4_rsc_0_0_sequencer);
      repeat (25) w4_rsc_1_0_random_seq.start(w4_rsc_1_0_sequencer);
      repeat (25) w4_rsc_2_0_random_seq.start(w4_rsc_2_0_sequencer);
      repeat (25) w4_rsc_3_0_random_seq.start(w4_rsc_3_0_sequencer);
      repeat (25) w4_rsc_4_0_random_seq.start(w4_rsc_4_0_sequencer);
      repeat (25) w4_rsc_5_0_random_seq.start(w4_rsc_5_0_sequencer);
      repeat (25) w4_rsc_6_0_random_seq.start(w4_rsc_6_0_sequencer);
      repeat (25) w4_rsc_7_0_random_seq.start(w4_rsc_7_0_sequencer);
      repeat (25) w4_rsc_8_0_random_seq.start(w4_rsc_8_0_sequencer);
      repeat (25) w4_rsc_9_0_random_seq.start(w4_rsc_9_0_sequencer);
      repeat (25) w4_rsc_10_0_random_seq.start(w4_rsc_10_0_sequencer);
      repeat (25) w4_rsc_11_0_random_seq.start(w4_rsc_11_0_sequencer);
      repeat (25) w4_rsc_12_0_random_seq.start(w4_rsc_12_0_sequencer);
      repeat (25) w4_rsc_13_0_random_seq.start(w4_rsc_13_0_sequencer);
      repeat (25) w4_rsc_14_0_random_seq.start(w4_rsc_14_0_sequencer);
      repeat (25) w4_rsc_15_0_random_seq.start(w4_rsc_15_0_sequencer);
      repeat (25) w4_rsc_16_0_random_seq.start(w4_rsc_16_0_sequencer);
      repeat (25) w4_rsc_17_0_random_seq.start(w4_rsc_17_0_sequencer);
      repeat (25) w4_rsc_18_0_random_seq.start(w4_rsc_18_0_sequencer);
      repeat (25) w4_rsc_19_0_random_seq.start(w4_rsc_19_0_sequencer);
      repeat (25) w4_rsc_20_0_random_seq.start(w4_rsc_20_0_sequencer);
      repeat (25) w4_rsc_21_0_random_seq.start(w4_rsc_21_0_sequencer);
      repeat (25) w4_rsc_22_0_random_seq.start(w4_rsc_22_0_sequencer);
      repeat (25) w4_rsc_23_0_random_seq.start(w4_rsc_23_0_sequencer);
      repeat (25) w4_rsc_24_0_random_seq.start(w4_rsc_24_0_sequencer);
      repeat (25) w4_rsc_25_0_random_seq.start(w4_rsc_25_0_sequencer);
      repeat (25) w4_rsc_26_0_random_seq.start(w4_rsc_26_0_sequencer);
      repeat (25) w4_rsc_27_0_random_seq.start(w4_rsc_27_0_sequencer);
      repeat (25) w4_rsc_28_0_random_seq.start(w4_rsc_28_0_sequencer);
      repeat (25) w4_rsc_29_0_random_seq.start(w4_rsc_29_0_sequencer);
      repeat (25) w4_rsc_30_0_random_seq.start(w4_rsc_30_0_sequencer);
      repeat (25) w4_rsc_31_0_random_seq.start(w4_rsc_31_0_sequencer);
      repeat (25) w4_rsc_32_0_random_seq.start(w4_rsc_32_0_sequencer);
      repeat (25) w4_rsc_33_0_random_seq.start(w4_rsc_33_0_sequencer);
      repeat (25) w4_rsc_34_0_random_seq.start(w4_rsc_34_0_sequencer);
      repeat (25) w4_rsc_35_0_random_seq.start(w4_rsc_35_0_sequencer);
      repeat (25) w4_rsc_36_0_random_seq.start(w4_rsc_36_0_sequencer);
      repeat (25) w4_rsc_37_0_random_seq.start(w4_rsc_37_0_sequencer);
      repeat (25) w4_rsc_38_0_random_seq.start(w4_rsc_38_0_sequencer);
      repeat (25) w4_rsc_39_0_random_seq.start(w4_rsc_39_0_sequencer);
      repeat (25) w4_rsc_40_0_random_seq.start(w4_rsc_40_0_sequencer);
      repeat (25) w4_rsc_41_0_random_seq.start(w4_rsc_41_0_sequencer);
      repeat (25) w4_rsc_42_0_random_seq.start(w4_rsc_42_0_sequencer);
      repeat (25) w4_rsc_43_0_random_seq.start(w4_rsc_43_0_sequencer);
      repeat (25) w4_rsc_44_0_random_seq.start(w4_rsc_44_0_sequencer);
      repeat (25) w4_rsc_45_0_random_seq.start(w4_rsc_45_0_sequencer);
      repeat (25) w4_rsc_46_0_random_seq.start(w4_rsc_46_0_sequencer);
      repeat (25) w4_rsc_47_0_random_seq.start(w4_rsc_47_0_sequencer);
      repeat (25) w4_rsc_48_0_random_seq.start(w4_rsc_48_0_sequencer);
      repeat (25) w4_rsc_49_0_random_seq.start(w4_rsc_49_0_sequencer);
      repeat (25) w4_rsc_50_0_random_seq.start(w4_rsc_50_0_sequencer);
      repeat (25) w4_rsc_51_0_random_seq.start(w4_rsc_51_0_sequencer);
      repeat (25) w4_rsc_52_0_random_seq.start(w4_rsc_52_0_sequencer);
      repeat (25) w4_rsc_53_0_random_seq.start(w4_rsc_53_0_sequencer);
      repeat (25) w4_rsc_54_0_random_seq.start(w4_rsc_54_0_sequencer);
      repeat (25) w4_rsc_55_0_random_seq.start(w4_rsc_55_0_sequencer);
      repeat (25) w4_rsc_56_0_random_seq.start(w4_rsc_56_0_sequencer);
      repeat (25) w4_rsc_57_0_random_seq.start(w4_rsc_57_0_sequencer);
      repeat (25) w4_rsc_58_0_random_seq.start(w4_rsc_58_0_sequencer);
      repeat (25) w4_rsc_59_0_random_seq.start(w4_rsc_59_0_sequencer);
      repeat (25) w4_rsc_60_0_random_seq.start(w4_rsc_60_0_sequencer);
      repeat (25) w4_rsc_61_0_random_seq.start(w4_rsc_61_0_sequencer);
      repeat (25) w4_rsc_62_0_random_seq.start(w4_rsc_62_0_sequencer);
      repeat (25) w4_rsc_63_0_random_seq.start(w4_rsc_63_0_sequencer);
      repeat (25) b4_rsc_random_seq.start(b4_rsc_sequencer);
      repeat (25) w6_rsc_0_0_random_seq.start(w6_rsc_0_0_sequencer);
      repeat (25) w6_rsc_1_0_random_seq.start(w6_rsc_1_0_sequencer);
      repeat (25) w6_rsc_2_0_random_seq.start(w6_rsc_2_0_sequencer);
      repeat (25) w6_rsc_3_0_random_seq.start(w6_rsc_3_0_sequencer);
      repeat (25) w6_rsc_4_0_random_seq.start(w6_rsc_4_0_sequencer);
      repeat (25) w6_rsc_5_0_random_seq.start(w6_rsc_5_0_sequencer);
      repeat (25) w6_rsc_6_0_random_seq.start(w6_rsc_6_0_sequencer);
      repeat (25) w6_rsc_7_0_random_seq.start(w6_rsc_7_0_sequencer);
      repeat (25) w6_rsc_8_0_random_seq.start(w6_rsc_8_0_sequencer);
      repeat (25) w6_rsc_9_0_random_seq.start(w6_rsc_9_0_sequencer);
      repeat (25) b6_rsc_random_seq.start(b6_rsc_sequencer);
    join
    // UVMF_CHANGE_ME : Extend the simulation XXX number of clocks after 
    // the last sequence to allow for the last sequence item to flow 
    // through the design.
    fork
      input1_rsc_config.wait_for_num_clocks(400);
      output1_rsc_config.wait_for_num_clocks(400);
      const_size_in_1_rsc_config.wait_for_num_clocks(400);
      const_size_out_1_rsc_config.wait_for_num_clocks(400);
      w2_rsc_0_0_config.wait_for_num_clocks(400);
      w2_rsc_1_0_config.wait_for_num_clocks(400);
      w2_rsc_2_0_config.wait_for_num_clocks(400);
      w2_rsc_3_0_config.wait_for_num_clocks(400);
      w2_rsc_4_0_config.wait_for_num_clocks(400);
      w2_rsc_5_0_config.wait_for_num_clocks(400);
      w2_rsc_6_0_config.wait_for_num_clocks(400);
      w2_rsc_7_0_config.wait_for_num_clocks(400);
      w2_rsc_8_0_config.wait_for_num_clocks(400);
      w2_rsc_9_0_config.wait_for_num_clocks(400);
      w2_rsc_10_0_config.wait_for_num_clocks(400);
      w2_rsc_11_0_config.wait_for_num_clocks(400);
      w2_rsc_12_0_config.wait_for_num_clocks(400);
      w2_rsc_13_0_config.wait_for_num_clocks(400);
      w2_rsc_14_0_config.wait_for_num_clocks(400);
      w2_rsc_15_0_config.wait_for_num_clocks(400);
      w2_rsc_16_0_config.wait_for_num_clocks(400);
      w2_rsc_17_0_config.wait_for_num_clocks(400);
      w2_rsc_18_0_config.wait_for_num_clocks(400);
      w2_rsc_19_0_config.wait_for_num_clocks(400);
      w2_rsc_20_0_config.wait_for_num_clocks(400);
      w2_rsc_21_0_config.wait_for_num_clocks(400);
      w2_rsc_22_0_config.wait_for_num_clocks(400);
      w2_rsc_23_0_config.wait_for_num_clocks(400);
      w2_rsc_24_0_config.wait_for_num_clocks(400);
      w2_rsc_25_0_config.wait_for_num_clocks(400);
      w2_rsc_26_0_config.wait_for_num_clocks(400);
      w2_rsc_27_0_config.wait_for_num_clocks(400);
      w2_rsc_28_0_config.wait_for_num_clocks(400);
      w2_rsc_29_0_config.wait_for_num_clocks(400);
      w2_rsc_30_0_config.wait_for_num_clocks(400);
      w2_rsc_31_0_config.wait_for_num_clocks(400);
      w2_rsc_32_0_config.wait_for_num_clocks(400);
      w2_rsc_33_0_config.wait_for_num_clocks(400);
      w2_rsc_34_0_config.wait_for_num_clocks(400);
      w2_rsc_35_0_config.wait_for_num_clocks(400);
      w2_rsc_36_0_config.wait_for_num_clocks(400);
      w2_rsc_37_0_config.wait_for_num_clocks(400);
      w2_rsc_38_0_config.wait_for_num_clocks(400);
      w2_rsc_39_0_config.wait_for_num_clocks(400);
      w2_rsc_40_0_config.wait_for_num_clocks(400);
      w2_rsc_41_0_config.wait_for_num_clocks(400);
      w2_rsc_42_0_config.wait_for_num_clocks(400);
      w2_rsc_43_0_config.wait_for_num_clocks(400);
      w2_rsc_44_0_config.wait_for_num_clocks(400);
      w2_rsc_45_0_config.wait_for_num_clocks(400);
      w2_rsc_46_0_config.wait_for_num_clocks(400);
      w2_rsc_47_0_config.wait_for_num_clocks(400);
      w2_rsc_48_0_config.wait_for_num_clocks(400);
      w2_rsc_49_0_config.wait_for_num_clocks(400);
      w2_rsc_50_0_config.wait_for_num_clocks(400);
      w2_rsc_51_0_config.wait_for_num_clocks(400);
      w2_rsc_52_0_config.wait_for_num_clocks(400);
      w2_rsc_53_0_config.wait_for_num_clocks(400);
      w2_rsc_54_0_config.wait_for_num_clocks(400);
      w2_rsc_55_0_config.wait_for_num_clocks(400);
      w2_rsc_56_0_config.wait_for_num_clocks(400);
      w2_rsc_57_0_config.wait_for_num_clocks(400);
      w2_rsc_58_0_config.wait_for_num_clocks(400);
      w2_rsc_59_0_config.wait_for_num_clocks(400);
      w2_rsc_60_0_config.wait_for_num_clocks(400);
      w2_rsc_61_0_config.wait_for_num_clocks(400);
      w2_rsc_62_0_config.wait_for_num_clocks(400);
      w2_rsc_63_0_config.wait_for_num_clocks(400);
      b2_rsc_config.wait_for_num_clocks(400);
      w4_rsc_0_0_config.wait_for_num_clocks(400);
      w4_rsc_1_0_config.wait_for_num_clocks(400);
      w4_rsc_2_0_config.wait_for_num_clocks(400);
      w4_rsc_3_0_config.wait_for_num_clocks(400);
      w4_rsc_4_0_config.wait_for_num_clocks(400);
      w4_rsc_5_0_config.wait_for_num_clocks(400);
      w4_rsc_6_0_config.wait_for_num_clocks(400);
      w4_rsc_7_0_config.wait_for_num_clocks(400);
      w4_rsc_8_0_config.wait_for_num_clocks(400);
      w4_rsc_9_0_config.wait_for_num_clocks(400);
      w4_rsc_10_0_config.wait_for_num_clocks(400);
      w4_rsc_11_0_config.wait_for_num_clocks(400);
      w4_rsc_12_0_config.wait_for_num_clocks(400);
      w4_rsc_13_0_config.wait_for_num_clocks(400);
      w4_rsc_14_0_config.wait_for_num_clocks(400);
      w4_rsc_15_0_config.wait_for_num_clocks(400);
      w4_rsc_16_0_config.wait_for_num_clocks(400);
      w4_rsc_17_0_config.wait_for_num_clocks(400);
      w4_rsc_18_0_config.wait_for_num_clocks(400);
      w4_rsc_19_0_config.wait_for_num_clocks(400);
      w4_rsc_20_0_config.wait_for_num_clocks(400);
      w4_rsc_21_0_config.wait_for_num_clocks(400);
      w4_rsc_22_0_config.wait_for_num_clocks(400);
      w4_rsc_23_0_config.wait_for_num_clocks(400);
      w4_rsc_24_0_config.wait_for_num_clocks(400);
      w4_rsc_25_0_config.wait_for_num_clocks(400);
      w4_rsc_26_0_config.wait_for_num_clocks(400);
      w4_rsc_27_0_config.wait_for_num_clocks(400);
      w4_rsc_28_0_config.wait_for_num_clocks(400);
      w4_rsc_29_0_config.wait_for_num_clocks(400);
      w4_rsc_30_0_config.wait_for_num_clocks(400);
      w4_rsc_31_0_config.wait_for_num_clocks(400);
      w4_rsc_32_0_config.wait_for_num_clocks(400);
      w4_rsc_33_0_config.wait_for_num_clocks(400);
      w4_rsc_34_0_config.wait_for_num_clocks(400);
      w4_rsc_35_0_config.wait_for_num_clocks(400);
      w4_rsc_36_0_config.wait_for_num_clocks(400);
      w4_rsc_37_0_config.wait_for_num_clocks(400);
      w4_rsc_38_0_config.wait_for_num_clocks(400);
      w4_rsc_39_0_config.wait_for_num_clocks(400);
      w4_rsc_40_0_config.wait_for_num_clocks(400);
      w4_rsc_41_0_config.wait_for_num_clocks(400);
      w4_rsc_42_0_config.wait_for_num_clocks(400);
      w4_rsc_43_0_config.wait_for_num_clocks(400);
      w4_rsc_44_0_config.wait_for_num_clocks(400);
      w4_rsc_45_0_config.wait_for_num_clocks(400);
      w4_rsc_46_0_config.wait_for_num_clocks(400);
      w4_rsc_47_0_config.wait_for_num_clocks(400);
      w4_rsc_48_0_config.wait_for_num_clocks(400);
      w4_rsc_49_0_config.wait_for_num_clocks(400);
      w4_rsc_50_0_config.wait_for_num_clocks(400);
      w4_rsc_51_0_config.wait_for_num_clocks(400);
      w4_rsc_52_0_config.wait_for_num_clocks(400);
      w4_rsc_53_0_config.wait_for_num_clocks(400);
      w4_rsc_54_0_config.wait_for_num_clocks(400);
      w4_rsc_55_0_config.wait_for_num_clocks(400);
      w4_rsc_56_0_config.wait_for_num_clocks(400);
      w4_rsc_57_0_config.wait_for_num_clocks(400);
      w4_rsc_58_0_config.wait_for_num_clocks(400);
      w4_rsc_59_0_config.wait_for_num_clocks(400);
      w4_rsc_60_0_config.wait_for_num_clocks(400);
      w4_rsc_61_0_config.wait_for_num_clocks(400);
      w4_rsc_62_0_config.wait_for_num_clocks(400);
      w4_rsc_63_0_config.wait_for_num_clocks(400);
      b4_rsc_config.wait_for_num_clocks(400);
      w6_rsc_0_0_config.wait_for_num_clocks(400);
      w6_rsc_1_0_config.wait_for_num_clocks(400);
      w6_rsc_2_0_config.wait_for_num_clocks(400);
      w6_rsc_3_0_config.wait_for_num_clocks(400);
      w6_rsc_4_0_config.wait_for_num_clocks(400);
      w6_rsc_5_0_config.wait_for_num_clocks(400);
      w6_rsc_6_0_config.wait_for_num_clocks(400);
      w6_rsc_7_0_config.wait_for_num_clocks(400);
      w6_rsc_8_0_config.wait_for_num_clocks(400);
      w6_rsc_9_0_config.wait_for_num_clocks(400);
      b6_rsc_config.wait_for_num_clocks(400);
    join
  endtask

endclass

