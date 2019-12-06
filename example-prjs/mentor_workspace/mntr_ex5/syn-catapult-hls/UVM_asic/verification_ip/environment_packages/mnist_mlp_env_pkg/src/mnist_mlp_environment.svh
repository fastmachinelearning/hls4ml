//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp Environment 
// Unit            : mnist_mlp Environment
// File            : mnist_mlp_environment.svh
//----------------------------------------------------------------------
//                                          
// DESCRIPTION: This environment contains all agents, predictors and
// scoreboards required for the block level design.
//----------------------------------------------------------------------
//



class mnist_mlp_environment #(
  int input1_rsc_WIDTH = 14112,
  bit input1_rsc_RESET_POLARITY = 1,
  bit[1:0] input1_rsc_PROTOCOL_KIND = 3,
  int output1_rsc_WIDTH = 180,
  bit output1_rsc_RESET_POLARITY = 1,
  bit[1:0] output1_rsc_PROTOCOL_KIND = 3,
  int const_size_in_1_rsc_WIDTH = 16,
  bit const_size_in_1_rsc_RESET_POLARITY = 1,
  bit[1:0] const_size_in_1_rsc_PROTOCOL_KIND = 2,
  int const_size_out_1_rsc_WIDTH = 16,
  bit const_size_out_1_rsc_RESET_POLARITY = 1,
  bit[1:0] const_size_out_1_rsc_PROTOCOL_KIND = 2,
  int w2_rsc_0_0_WIDTH = 14112,
  bit w2_rsc_0_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_0_0_PROTOCOL_KIND = 0,
  int w2_rsc_1_0_WIDTH = 14112,
  bit w2_rsc_1_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_1_0_PROTOCOL_KIND = 0,
  int w2_rsc_2_0_WIDTH = 14112,
  bit w2_rsc_2_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_2_0_PROTOCOL_KIND = 0,
  int w2_rsc_3_0_WIDTH = 14112,
  bit w2_rsc_3_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_3_0_PROTOCOL_KIND = 0,
  int w2_rsc_4_0_WIDTH = 14112,
  bit w2_rsc_4_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_4_0_PROTOCOL_KIND = 0,
  int w2_rsc_5_0_WIDTH = 14112,
  bit w2_rsc_5_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_5_0_PROTOCOL_KIND = 0,
  int w2_rsc_6_0_WIDTH = 14112,
  bit w2_rsc_6_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_6_0_PROTOCOL_KIND = 0,
  int w2_rsc_7_0_WIDTH = 14112,
  bit w2_rsc_7_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_7_0_PROTOCOL_KIND = 0,
  int w2_rsc_8_0_WIDTH = 14112,
  bit w2_rsc_8_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_8_0_PROTOCOL_KIND = 0,
  int w2_rsc_9_0_WIDTH = 14112,
  bit w2_rsc_9_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_9_0_PROTOCOL_KIND = 0,
  int w2_rsc_10_0_WIDTH = 14112,
  bit w2_rsc_10_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_10_0_PROTOCOL_KIND = 0,
  int w2_rsc_11_0_WIDTH = 14112,
  bit w2_rsc_11_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_11_0_PROTOCOL_KIND = 0,
  int w2_rsc_12_0_WIDTH = 14112,
  bit w2_rsc_12_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_12_0_PROTOCOL_KIND = 0,
  int w2_rsc_13_0_WIDTH = 14112,
  bit w2_rsc_13_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_13_0_PROTOCOL_KIND = 0,
  int w2_rsc_14_0_WIDTH = 14112,
  bit w2_rsc_14_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_14_0_PROTOCOL_KIND = 0,
  int w2_rsc_15_0_WIDTH = 14112,
  bit w2_rsc_15_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_15_0_PROTOCOL_KIND = 0,
  int w2_rsc_16_0_WIDTH = 14112,
  bit w2_rsc_16_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_16_0_PROTOCOL_KIND = 0,
  int w2_rsc_17_0_WIDTH = 14112,
  bit w2_rsc_17_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_17_0_PROTOCOL_KIND = 0,
  int w2_rsc_18_0_WIDTH = 14112,
  bit w2_rsc_18_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_18_0_PROTOCOL_KIND = 0,
  int w2_rsc_19_0_WIDTH = 14112,
  bit w2_rsc_19_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_19_0_PROTOCOL_KIND = 0,
  int w2_rsc_20_0_WIDTH = 14112,
  bit w2_rsc_20_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_20_0_PROTOCOL_KIND = 0,
  int w2_rsc_21_0_WIDTH = 14112,
  bit w2_rsc_21_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_21_0_PROTOCOL_KIND = 0,
  int w2_rsc_22_0_WIDTH = 14112,
  bit w2_rsc_22_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_22_0_PROTOCOL_KIND = 0,
  int w2_rsc_23_0_WIDTH = 14112,
  bit w2_rsc_23_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_23_0_PROTOCOL_KIND = 0,
  int w2_rsc_24_0_WIDTH = 14112,
  bit w2_rsc_24_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_24_0_PROTOCOL_KIND = 0,
  int w2_rsc_25_0_WIDTH = 14112,
  bit w2_rsc_25_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_25_0_PROTOCOL_KIND = 0,
  int w2_rsc_26_0_WIDTH = 14112,
  bit w2_rsc_26_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_26_0_PROTOCOL_KIND = 0,
  int w2_rsc_27_0_WIDTH = 14112,
  bit w2_rsc_27_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_27_0_PROTOCOL_KIND = 0,
  int w2_rsc_28_0_WIDTH = 14112,
  bit w2_rsc_28_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_28_0_PROTOCOL_KIND = 0,
  int w2_rsc_29_0_WIDTH = 14112,
  bit w2_rsc_29_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_29_0_PROTOCOL_KIND = 0,
  int w2_rsc_30_0_WIDTH = 14112,
  bit w2_rsc_30_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_30_0_PROTOCOL_KIND = 0,
  int w2_rsc_31_0_WIDTH = 14112,
  bit w2_rsc_31_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_31_0_PROTOCOL_KIND = 0,
  int w2_rsc_32_0_WIDTH = 14112,
  bit w2_rsc_32_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_32_0_PROTOCOL_KIND = 0,
  int w2_rsc_33_0_WIDTH = 14112,
  bit w2_rsc_33_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_33_0_PROTOCOL_KIND = 0,
  int w2_rsc_34_0_WIDTH = 14112,
  bit w2_rsc_34_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_34_0_PROTOCOL_KIND = 0,
  int w2_rsc_35_0_WIDTH = 14112,
  bit w2_rsc_35_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_35_0_PROTOCOL_KIND = 0,
  int w2_rsc_36_0_WIDTH = 14112,
  bit w2_rsc_36_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_36_0_PROTOCOL_KIND = 0,
  int w2_rsc_37_0_WIDTH = 14112,
  bit w2_rsc_37_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_37_0_PROTOCOL_KIND = 0,
  int w2_rsc_38_0_WIDTH = 14112,
  bit w2_rsc_38_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_38_0_PROTOCOL_KIND = 0,
  int w2_rsc_39_0_WIDTH = 14112,
  bit w2_rsc_39_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_39_0_PROTOCOL_KIND = 0,
  int w2_rsc_40_0_WIDTH = 14112,
  bit w2_rsc_40_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_40_0_PROTOCOL_KIND = 0,
  int w2_rsc_41_0_WIDTH = 14112,
  bit w2_rsc_41_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_41_0_PROTOCOL_KIND = 0,
  int w2_rsc_42_0_WIDTH = 14112,
  bit w2_rsc_42_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_42_0_PROTOCOL_KIND = 0,
  int w2_rsc_43_0_WIDTH = 14112,
  bit w2_rsc_43_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_43_0_PROTOCOL_KIND = 0,
  int w2_rsc_44_0_WIDTH = 14112,
  bit w2_rsc_44_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_44_0_PROTOCOL_KIND = 0,
  int w2_rsc_45_0_WIDTH = 14112,
  bit w2_rsc_45_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_45_0_PROTOCOL_KIND = 0,
  int w2_rsc_46_0_WIDTH = 14112,
  bit w2_rsc_46_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_46_0_PROTOCOL_KIND = 0,
  int w2_rsc_47_0_WIDTH = 14112,
  bit w2_rsc_47_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_47_0_PROTOCOL_KIND = 0,
  int w2_rsc_48_0_WIDTH = 14112,
  bit w2_rsc_48_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_48_0_PROTOCOL_KIND = 0,
  int w2_rsc_49_0_WIDTH = 14112,
  bit w2_rsc_49_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_49_0_PROTOCOL_KIND = 0,
  int w2_rsc_50_0_WIDTH = 14112,
  bit w2_rsc_50_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_50_0_PROTOCOL_KIND = 0,
  int w2_rsc_51_0_WIDTH = 14112,
  bit w2_rsc_51_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_51_0_PROTOCOL_KIND = 0,
  int w2_rsc_52_0_WIDTH = 14112,
  bit w2_rsc_52_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_52_0_PROTOCOL_KIND = 0,
  int w2_rsc_53_0_WIDTH = 14112,
  bit w2_rsc_53_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_53_0_PROTOCOL_KIND = 0,
  int w2_rsc_54_0_WIDTH = 14112,
  bit w2_rsc_54_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_54_0_PROTOCOL_KIND = 0,
  int w2_rsc_55_0_WIDTH = 14112,
  bit w2_rsc_55_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_55_0_PROTOCOL_KIND = 0,
  int w2_rsc_56_0_WIDTH = 14112,
  bit w2_rsc_56_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_56_0_PROTOCOL_KIND = 0,
  int w2_rsc_57_0_WIDTH = 14112,
  bit w2_rsc_57_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_57_0_PROTOCOL_KIND = 0,
  int w2_rsc_58_0_WIDTH = 14112,
  bit w2_rsc_58_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_58_0_PROTOCOL_KIND = 0,
  int w2_rsc_59_0_WIDTH = 14112,
  bit w2_rsc_59_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_59_0_PROTOCOL_KIND = 0,
  int w2_rsc_60_0_WIDTH = 14112,
  bit w2_rsc_60_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_60_0_PROTOCOL_KIND = 0,
  int w2_rsc_61_0_WIDTH = 14112,
  bit w2_rsc_61_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_61_0_PROTOCOL_KIND = 0,
  int w2_rsc_62_0_WIDTH = 14112,
  bit w2_rsc_62_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_62_0_PROTOCOL_KIND = 0,
  int w2_rsc_63_0_WIDTH = 14112,
  bit w2_rsc_63_0_RESET_POLARITY = 1,
  bit[1:0] w2_rsc_63_0_PROTOCOL_KIND = 0,
  int b2_rsc_WIDTH = 1152,
  bit b2_rsc_RESET_POLARITY = 1,
  bit[1:0] b2_rsc_PROTOCOL_KIND = 0,
  int w4_rsc_0_0_WIDTH = 1152,
  bit w4_rsc_0_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_0_0_PROTOCOL_KIND = 0,
  int w4_rsc_1_0_WIDTH = 1152,
  bit w4_rsc_1_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_1_0_PROTOCOL_KIND = 0,
  int w4_rsc_2_0_WIDTH = 1152,
  bit w4_rsc_2_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_2_0_PROTOCOL_KIND = 0,
  int w4_rsc_3_0_WIDTH = 1152,
  bit w4_rsc_3_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_3_0_PROTOCOL_KIND = 0,
  int w4_rsc_4_0_WIDTH = 1152,
  bit w4_rsc_4_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_4_0_PROTOCOL_KIND = 0,
  int w4_rsc_5_0_WIDTH = 1152,
  bit w4_rsc_5_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_5_0_PROTOCOL_KIND = 0,
  int w4_rsc_6_0_WIDTH = 1152,
  bit w4_rsc_6_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_6_0_PROTOCOL_KIND = 0,
  int w4_rsc_7_0_WIDTH = 1152,
  bit w4_rsc_7_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_7_0_PROTOCOL_KIND = 0,
  int w4_rsc_8_0_WIDTH = 1152,
  bit w4_rsc_8_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_8_0_PROTOCOL_KIND = 0,
  int w4_rsc_9_0_WIDTH = 1152,
  bit w4_rsc_9_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_9_0_PROTOCOL_KIND = 0,
  int w4_rsc_10_0_WIDTH = 1152,
  bit w4_rsc_10_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_10_0_PROTOCOL_KIND = 0,
  int w4_rsc_11_0_WIDTH = 1152,
  bit w4_rsc_11_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_11_0_PROTOCOL_KIND = 0,
  int w4_rsc_12_0_WIDTH = 1152,
  bit w4_rsc_12_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_12_0_PROTOCOL_KIND = 0,
  int w4_rsc_13_0_WIDTH = 1152,
  bit w4_rsc_13_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_13_0_PROTOCOL_KIND = 0,
  int w4_rsc_14_0_WIDTH = 1152,
  bit w4_rsc_14_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_14_0_PROTOCOL_KIND = 0,
  int w4_rsc_15_0_WIDTH = 1152,
  bit w4_rsc_15_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_15_0_PROTOCOL_KIND = 0,
  int w4_rsc_16_0_WIDTH = 1152,
  bit w4_rsc_16_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_16_0_PROTOCOL_KIND = 0,
  int w4_rsc_17_0_WIDTH = 1152,
  bit w4_rsc_17_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_17_0_PROTOCOL_KIND = 0,
  int w4_rsc_18_0_WIDTH = 1152,
  bit w4_rsc_18_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_18_0_PROTOCOL_KIND = 0,
  int w4_rsc_19_0_WIDTH = 1152,
  bit w4_rsc_19_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_19_0_PROTOCOL_KIND = 0,
  int w4_rsc_20_0_WIDTH = 1152,
  bit w4_rsc_20_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_20_0_PROTOCOL_KIND = 0,
  int w4_rsc_21_0_WIDTH = 1152,
  bit w4_rsc_21_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_21_0_PROTOCOL_KIND = 0,
  int w4_rsc_22_0_WIDTH = 1152,
  bit w4_rsc_22_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_22_0_PROTOCOL_KIND = 0,
  int w4_rsc_23_0_WIDTH = 1152,
  bit w4_rsc_23_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_23_0_PROTOCOL_KIND = 0,
  int w4_rsc_24_0_WIDTH = 1152,
  bit w4_rsc_24_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_24_0_PROTOCOL_KIND = 0,
  int w4_rsc_25_0_WIDTH = 1152,
  bit w4_rsc_25_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_25_0_PROTOCOL_KIND = 0,
  int w4_rsc_26_0_WIDTH = 1152,
  bit w4_rsc_26_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_26_0_PROTOCOL_KIND = 0,
  int w4_rsc_27_0_WIDTH = 1152,
  bit w4_rsc_27_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_27_0_PROTOCOL_KIND = 0,
  int w4_rsc_28_0_WIDTH = 1152,
  bit w4_rsc_28_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_28_0_PROTOCOL_KIND = 0,
  int w4_rsc_29_0_WIDTH = 1152,
  bit w4_rsc_29_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_29_0_PROTOCOL_KIND = 0,
  int w4_rsc_30_0_WIDTH = 1152,
  bit w4_rsc_30_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_30_0_PROTOCOL_KIND = 0,
  int w4_rsc_31_0_WIDTH = 1152,
  bit w4_rsc_31_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_31_0_PROTOCOL_KIND = 0,
  int w4_rsc_32_0_WIDTH = 1152,
  bit w4_rsc_32_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_32_0_PROTOCOL_KIND = 0,
  int w4_rsc_33_0_WIDTH = 1152,
  bit w4_rsc_33_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_33_0_PROTOCOL_KIND = 0,
  int w4_rsc_34_0_WIDTH = 1152,
  bit w4_rsc_34_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_34_0_PROTOCOL_KIND = 0,
  int w4_rsc_35_0_WIDTH = 1152,
  bit w4_rsc_35_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_35_0_PROTOCOL_KIND = 0,
  int w4_rsc_36_0_WIDTH = 1152,
  bit w4_rsc_36_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_36_0_PROTOCOL_KIND = 0,
  int w4_rsc_37_0_WIDTH = 1152,
  bit w4_rsc_37_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_37_0_PROTOCOL_KIND = 0,
  int w4_rsc_38_0_WIDTH = 1152,
  bit w4_rsc_38_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_38_0_PROTOCOL_KIND = 0,
  int w4_rsc_39_0_WIDTH = 1152,
  bit w4_rsc_39_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_39_0_PROTOCOL_KIND = 0,
  int w4_rsc_40_0_WIDTH = 1152,
  bit w4_rsc_40_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_40_0_PROTOCOL_KIND = 0,
  int w4_rsc_41_0_WIDTH = 1152,
  bit w4_rsc_41_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_41_0_PROTOCOL_KIND = 0,
  int w4_rsc_42_0_WIDTH = 1152,
  bit w4_rsc_42_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_42_0_PROTOCOL_KIND = 0,
  int w4_rsc_43_0_WIDTH = 1152,
  bit w4_rsc_43_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_43_0_PROTOCOL_KIND = 0,
  int w4_rsc_44_0_WIDTH = 1152,
  bit w4_rsc_44_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_44_0_PROTOCOL_KIND = 0,
  int w4_rsc_45_0_WIDTH = 1152,
  bit w4_rsc_45_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_45_0_PROTOCOL_KIND = 0,
  int w4_rsc_46_0_WIDTH = 1152,
  bit w4_rsc_46_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_46_0_PROTOCOL_KIND = 0,
  int w4_rsc_47_0_WIDTH = 1152,
  bit w4_rsc_47_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_47_0_PROTOCOL_KIND = 0,
  int w4_rsc_48_0_WIDTH = 1152,
  bit w4_rsc_48_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_48_0_PROTOCOL_KIND = 0,
  int w4_rsc_49_0_WIDTH = 1152,
  bit w4_rsc_49_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_49_0_PROTOCOL_KIND = 0,
  int w4_rsc_50_0_WIDTH = 1152,
  bit w4_rsc_50_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_50_0_PROTOCOL_KIND = 0,
  int w4_rsc_51_0_WIDTH = 1152,
  bit w4_rsc_51_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_51_0_PROTOCOL_KIND = 0,
  int w4_rsc_52_0_WIDTH = 1152,
  bit w4_rsc_52_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_52_0_PROTOCOL_KIND = 0,
  int w4_rsc_53_0_WIDTH = 1152,
  bit w4_rsc_53_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_53_0_PROTOCOL_KIND = 0,
  int w4_rsc_54_0_WIDTH = 1152,
  bit w4_rsc_54_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_54_0_PROTOCOL_KIND = 0,
  int w4_rsc_55_0_WIDTH = 1152,
  bit w4_rsc_55_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_55_0_PROTOCOL_KIND = 0,
  int w4_rsc_56_0_WIDTH = 1152,
  bit w4_rsc_56_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_56_0_PROTOCOL_KIND = 0,
  int w4_rsc_57_0_WIDTH = 1152,
  bit w4_rsc_57_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_57_0_PROTOCOL_KIND = 0,
  int w4_rsc_58_0_WIDTH = 1152,
  bit w4_rsc_58_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_58_0_PROTOCOL_KIND = 0,
  int w4_rsc_59_0_WIDTH = 1152,
  bit w4_rsc_59_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_59_0_PROTOCOL_KIND = 0,
  int w4_rsc_60_0_WIDTH = 1152,
  bit w4_rsc_60_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_60_0_PROTOCOL_KIND = 0,
  int w4_rsc_61_0_WIDTH = 1152,
  bit w4_rsc_61_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_61_0_PROTOCOL_KIND = 0,
  int w4_rsc_62_0_WIDTH = 1152,
  bit w4_rsc_62_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_62_0_PROTOCOL_KIND = 0,
  int w4_rsc_63_0_WIDTH = 1152,
  bit w4_rsc_63_0_RESET_POLARITY = 1,
  bit[1:0] w4_rsc_63_0_PROTOCOL_KIND = 0,
  int b4_rsc_WIDTH = 1152,
  bit b4_rsc_RESET_POLARITY = 1,
  bit[1:0] b4_rsc_PROTOCOL_KIND = 0,
  int w6_rsc_0_0_WIDTH = 1152,
  bit w6_rsc_0_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_0_0_PROTOCOL_KIND = 0,
  int w6_rsc_1_0_WIDTH = 1152,
  bit w6_rsc_1_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_1_0_PROTOCOL_KIND = 0,
  int w6_rsc_2_0_WIDTH = 1152,
  bit w6_rsc_2_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_2_0_PROTOCOL_KIND = 0,
  int w6_rsc_3_0_WIDTH = 1152,
  bit w6_rsc_3_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_3_0_PROTOCOL_KIND = 0,
  int w6_rsc_4_0_WIDTH = 1152,
  bit w6_rsc_4_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_4_0_PROTOCOL_KIND = 0,
  int w6_rsc_5_0_WIDTH = 1152,
  bit w6_rsc_5_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_5_0_PROTOCOL_KIND = 0,
  int w6_rsc_6_0_WIDTH = 1152,
  bit w6_rsc_6_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_6_0_PROTOCOL_KIND = 0,
  int w6_rsc_7_0_WIDTH = 1152,
  bit w6_rsc_7_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_7_0_PROTOCOL_KIND = 0,
  int w6_rsc_8_0_WIDTH = 1152,
  bit w6_rsc_8_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_8_0_PROTOCOL_KIND = 0,
  int w6_rsc_9_0_WIDTH = 1152,
  bit w6_rsc_9_0_RESET_POLARITY = 1,
  bit[1:0] w6_rsc_9_0_PROTOCOL_KIND = 0,
  int b6_rsc_WIDTH = 180,
  bit b6_rsc_RESET_POLARITY = 1,
  bit[1:0] b6_rsc_PROTOCOL_KIND = 0)
  extends uvmf_environment_base #(
    .CONFIG_T( mnist_mlp_env_configuration
    #(
     .input1_rsc_WIDTH(input1_rsc_WIDTH),                                
     .input1_rsc_RESET_POLARITY(input1_rsc_RESET_POLARITY),                                
     .input1_rsc_PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND),                                
     .output1_rsc_WIDTH(output1_rsc_WIDTH),                                
     .output1_rsc_RESET_POLARITY(output1_rsc_RESET_POLARITY),                                
     .output1_rsc_PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND),                                
     .const_size_in_1_rsc_WIDTH(const_size_in_1_rsc_WIDTH),                                
     .const_size_in_1_rsc_RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),                                
     .const_size_in_1_rsc_PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND),                                
     .const_size_out_1_rsc_WIDTH(const_size_out_1_rsc_WIDTH),                                
     .const_size_out_1_rsc_RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),                                
     .const_size_out_1_rsc_PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND),                                
     .w2_rsc_0_0_WIDTH(w2_rsc_0_0_WIDTH),                                
     .w2_rsc_0_0_RESET_POLARITY(w2_rsc_0_0_RESET_POLARITY),                                
     .w2_rsc_0_0_PROTOCOL_KIND(w2_rsc_0_0_PROTOCOL_KIND),                                
     .w2_rsc_1_0_WIDTH(w2_rsc_1_0_WIDTH),                                
     .w2_rsc_1_0_RESET_POLARITY(w2_rsc_1_0_RESET_POLARITY),                                
     .w2_rsc_1_0_PROTOCOL_KIND(w2_rsc_1_0_PROTOCOL_KIND),                                
     .w2_rsc_2_0_WIDTH(w2_rsc_2_0_WIDTH),                                
     .w2_rsc_2_0_RESET_POLARITY(w2_rsc_2_0_RESET_POLARITY),                                
     .w2_rsc_2_0_PROTOCOL_KIND(w2_rsc_2_0_PROTOCOL_KIND),                                
     .w2_rsc_3_0_WIDTH(w2_rsc_3_0_WIDTH),                                
     .w2_rsc_3_0_RESET_POLARITY(w2_rsc_3_0_RESET_POLARITY),                                
     .w2_rsc_3_0_PROTOCOL_KIND(w2_rsc_3_0_PROTOCOL_KIND),                                
     .w2_rsc_4_0_WIDTH(w2_rsc_4_0_WIDTH),                                
     .w2_rsc_4_0_RESET_POLARITY(w2_rsc_4_0_RESET_POLARITY),                                
     .w2_rsc_4_0_PROTOCOL_KIND(w2_rsc_4_0_PROTOCOL_KIND),                                
     .w2_rsc_5_0_WIDTH(w2_rsc_5_0_WIDTH),                                
     .w2_rsc_5_0_RESET_POLARITY(w2_rsc_5_0_RESET_POLARITY),                                
     .w2_rsc_5_0_PROTOCOL_KIND(w2_rsc_5_0_PROTOCOL_KIND),                                
     .w2_rsc_6_0_WIDTH(w2_rsc_6_0_WIDTH),                                
     .w2_rsc_6_0_RESET_POLARITY(w2_rsc_6_0_RESET_POLARITY),                                
     .w2_rsc_6_0_PROTOCOL_KIND(w2_rsc_6_0_PROTOCOL_KIND),                                
     .w2_rsc_7_0_WIDTH(w2_rsc_7_0_WIDTH),                                
     .w2_rsc_7_0_RESET_POLARITY(w2_rsc_7_0_RESET_POLARITY),                                
     .w2_rsc_7_0_PROTOCOL_KIND(w2_rsc_7_0_PROTOCOL_KIND),                                
     .w2_rsc_8_0_WIDTH(w2_rsc_8_0_WIDTH),                                
     .w2_rsc_8_0_RESET_POLARITY(w2_rsc_8_0_RESET_POLARITY),                                
     .w2_rsc_8_0_PROTOCOL_KIND(w2_rsc_8_0_PROTOCOL_KIND),                                
     .w2_rsc_9_0_WIDTH(w2_rsc_9_0_WIDTH),                                
     .w2_rsc_9_0_RESET_POLARITY(w2_rsc_9_0_RESET_POLARITY),                                
     .w2_rsc_9_0_PROTOCOL_KIND(w2_rsc_9_0_PROTOCOL_KIND),                                
     .w2_rsc_10_0_WIDTH(w2_rsc_10_0_WIDTH),                                
     .w2_rsc_10_0_RESET_POLARITY(w2_rsc_10_0_RESET_POLARITY),                                
     .w2_rsc_10_0_PROTOCOL_KIND(w2_rsc_10_0_PROTOCOL_KIND),                                
     .w2_rsc_11_0_WIDTH(w2_rsc_11_0_WIDTH),                                
     .w2_rsc_11_0_RESET_POLARITY(w2_rsc_11_0_RESET_POLARITY),                                
     .w2_rsc_11_0_PROTOCOL_KIND(w2_rsc_11_0_PROTOCOL_KIND),                                
     .w2_rsc_12_0_WIDTH(w2_rsc_12_0_WIDTH),                                
     .w2_rsc_12_0_RESET_POLARITY(w2_rsc_12_0_RESET_POLARITY),                                
     .w2_rsc_12_0_PROTOCOL_KIND(w2_rsc_12_0_PROTOCOL_KIND),                                
     .w2_rsc_13_0_WIDTH(w2_rsc_13_0_WIDTH),                                
     .w2_rsc_13_0_RESET_POLARITY(w2_rsc_13_0_RESET_POLARITY),                                
     .w2_rsc_13_0_PROTOCOL_KIND(w2_rsc_13_0_PROTOCOL_KIND),                                
     .w2_rsc_14_0_WIDTH(w2_rsc_14_0_WIDTH),                                
     .w2_rsc_14_0_RESET_POLARITY(w2_rsc_14_0_RESET_POLARITY),                                
     .w2_rsc_14_0_PROTOCOL_KIND(w2_rsc_14_0_PROTOCOL_KIND),                                
     .w2_rsc_15_0_WIDTH(w2_rsc_15_0_WIDTH),                                
     .w2_rsc_15_0_RESET_POLARITY(w2_rsc_15_0_RESET_POLARITY),                                
     .w2_rsc_15_0_PROTOCOL_KIND(w2_rsc_15_0_PROTOCOL_KIND),                                
     .w2_rsc_16_0_WIDTH(w2_rsc_16_0_WIDTH),                                
     .w2_rsc_16_0_RESET_POLARITY(w2_rsc_16_0_RESET_POLARITY),                                
     .w2_rsc_16_0_PROTOCOL_KIND(w2_rsc_16_0_PROTOCOL_KIND),                                
     .w2_rsc_17_0_WIDTH(w2_rsc_17_0_WIDTH),                                
     .w2_rsc_17_0_RESET_POLARITY(w2_rsc_17_0_RESET_POLARITY),                                
     .w2_rsc_17_0_PROTOCOL_KIND(w2_rsc_17_0_PROTOCOL_KIND),                                
     .w2_rsc_18_0_WIDTH(w2_rsc_18_0_WIDTH),                                
     .w2_rsc_18_0_RESET_POLARITY(w2_rsc_18_0_RESET_POLARITY),                                
     .w2_rsc_18_0_PROTOCOL_KIND(w2_rsc_18_0_PROTOCOL_KIND),                                
     .w2_rsc_19_0_WIDTH(w2_rsc_19_0_WIDTH),                                
     .w2_rsc_19_0_RESET_POLARITY(w2_rsc_19_0_RESET_POLARITY),                                
     .w2_rsc_19_0_PROTOCOL_KIND(w2_rsc_19_0_PROTOCOL_KIND),                                
     .w2_rsc_20_0_WIDTH(w2_rsc_20_0_WIDTH),                                
     .w2_rsc_20_0_RESET_POLARITY(w2_rsc_20_0_RESET_POLARITY),                                
     .w2_rsc_20_0_PROTOCOL_KIND(w2_rsc_20_0_PROTOCOL_KIND),                                
     .w2_rsc_21_0_WIDTH(w2_rsc_21_0_WIDTH),                                
     .w2_rsc_21_0_RESET_POLARITY(w2_rsc_21_0_RESET_POLARITY),                                
     .w2_rsc_21_0_PROTOCOL_KIND(w2_rsc_21_0_PROTOCOL_KIND),                                
     .w2_rsc_22_0_WIDTH(w2_rsc_22_0_WIDTH),                                
     .w2_rsc_22_0_RESET_POLARITY(w2_rsc_22_0_RESET_POLARITY),                                
     .w2_rsc_22_0_PROTOCOL_KIND(w2_rsc_22_0_PROTOCOL_KIND),                                
     .w2_rsc_23_0_WIDTH(w2_rsc_23_0_WIDTH),                                
     .w2_rsc_23_0_RESET_POLARITY(w2_rsc_23_0_RESET_POLARITY),                                
     .w2_rsc_23_0_PROTOCOL_KIND(w2_rsc_23_0_PROTOCOL_KIND),                                
     .w2_rsc_24_0_WIDTH(w2_rsc_24_0_WIDTH),                                
     .w2_rsc_24_0_RESET_POLARITY(w2_rsc_24_0_RESET_POLARITY),                                
     .w2_rsc_24_0_PROTOCOL_KIND(w2_rsc_24_0_PROTOCOL_KIND),                                
     .w2_rsc_25_0_WIDTH(w2_rsc_25_0_WIDTH),                                
     .w2_rsc_25_0_RESET_POLARITY(w2_rsc_25_0_RESET_POLARITY),                                
     .w2_rsc_25_0_PROTOCOL_KIND(w2_rsc_25_0_PROTOCOL_KIND),                                
     .w2_rsc_26_0_WIDTH(w2_rsc_26_0_WIDTH),                                
     .w2_rsc_26_0_RESET_POLARITY(w2_rsc_26_0_RESET_POLARITY),                                
     .w2_rsc_26_0_PROTOCOL_KIND(w2_rsc_26_0_PROTOCOL_KIND),                                
     .w2_rsc_27_0_WIDTH(w2_rsc_27_0_WIDTH),                                
     .w2_rsc_27_0_RESET_POLARITY(w2_rsc_27_0_RESET_POLARITY),                                
     .w2_rsc_27_0_PROTOCOL_KIND(w2_rsc_27_0_PROTOCOL_KIND),                                
     .w2_rsc_28_0_WIDTH(w2_rsc_28_0_WIDTH),                                
     .w2_rsc_28_0_RESET_POLARITY(w2_rsc_28_0_RESET_POLARITY),                                
     .w2_rsc_28_0_PROTOCOL_KIND(w2_rsc_28_0_PROTOCOL_KIND),                                
     .w2_rsc_29_0_WIDTH(w2_rsc_29_0_WIDTH),                                
     .w2_rsc_29_0_RESET_POLARITY(w2_rsc_29_0_RESET_POLARITY),                                
     .w2_rsc_29_0_PROTOCOL_KIND(w2_rsc_29_0_PROTOCOL_KIND),                                
     .w2_rsc_30_0_WIDTH(w2_rsc_30_0_WIDTH),                                
     .w2_rsc_30_0_RESET_POLARITY(w2_rsc_30_0_RESET_POLARITY),                                
     .w2_rsc_30_0_PROTOCOL_KIND(w2_rsc_30_0_PROTOCOL_KIND),                                
     .w2_rsc_31_0_WIDTH(w2_rsc_31_0_WIDTH),                                
     .w2_rsc_31_0_RESET_POLARITY(w2_rsc_31_0_RESET_POLARITY),                                
     .w2_rsc_31_0_PROTOCOL_KIND(w2_rsc_31_0_PROTOCOL_KIND),                                
     .w2_rsc_32_0_WIDTH(w2_rsc_32_0_WIDTH),                                
     .w2_rsc_32_0_RESET_POLARITY(w2_rsc_32_0_RESET_POLARITY),                                
     .w2_rsc_32_0_PROTOCOL_KIND(w2_rsc_32_0_PROTOCOL_KIND),                                
     .w2_rsc_33_0_WIDTH(w2_rsc_33_0_WIDTH),                                
     .w2_rsc_33_0_RESET_POLARITY(w2_rsc_33_0_RESET_POLARITY),                                
     .w2_rsc_33_0_PROTOCOL_KIND(w2_rsc_33_0_PROTOCOL_KIND),                                
     .w2_rsc_34_0_WIDTH(w2_rsc_34_0_WIDTH),                                
     .w2_rsc_34_0_RESET_POLARITY(w2_rsc_34_0_RESET_POLARITY),                                
     .w2_rsc_34_0_PROTOCOL_KIND(w2_rsc_34_0_PROTOCOL_KIND),                                
     .w2_rsc_35_0_WIDTH(w2_rsc_35_0_WIDTH),                                
     .w2_rsc_35_0_RESET_POLARITY(w2_rsc_35_0_RESET_POLARITY),                                
     .w2_rsc_35_0_PROTOCOL_KIND(w2_rsc_35_0_PROTOCOL_KIND),                                
     .w2_rsc_36_0_WIDTH(w2_rsc_36_0_WIDTH),                                
     .w2_rsc_36_0_RESET_POLARITY(w2_rsc_36_0_RESET_POLARITY),                                
     .w2_rsc_36_0_PROTOCOL_KIND(w2_rsc_36_0_PROTOCOL_KIND),                                
     .w2_rsc_37_0_WIDTH(w2_rsc_37_0_WIDTH),                                
     .w2_rsc_37_0_RESET_POLARITY(w2_rsc_37_0_RESET_POLARITY),                                
     .w2_rsc_37_0_PROTOCOL_KIND(w2_rsc_37_0_PROTOCOL_KIND),                                
     .w2_rsc_38_0_WIDTH(w2_rsc_38_0_WIDTH),                                
     .w2_rsc_38_0_RESET_POLARITY(w2_rsc_38_0_RESET_POLARITY),                                
     .w2_rsc_38_0_PROTOCOL_KIND(w2_rsc_38_0_PROTOCOL_KIND),                                
     .w2_rsc_39_0_WIDTH(w2_rsc_39_0_WIDTH),                                
     .w2_rsc_39_0_RESET_POLARITY(w2_rsc_39_0_RESET_POLARITY),                                
     .w2_rsc_39_0_PROTOCOL_KIND(w2_rsc_39_0_PROTOCOL_KIND),                                
     .w2_rsc_40_0_WIDTH(w2_rsc_40_0_WIDTH),                                
     .w2_rsc_40_0_RESET_POLARITY(w2_rsc_40_0_RESET_POLARITY),                                
     .w2_rsc_40_0_PROTOCOL_KIND(w2_rsc_40_0_PROTOCOL_KIND),                                
     .w2_rsc_41_0_WIDTH(w2_rsc_41_0_WIDTH),                                
     .w2_rsc_41_0_RESET_POLARITY(w2_rsc_41_0_RESET_POLARITY),                                
     .w2_rsc_41_0_PROTOCOL_KIND(w2_rsc_41_0_PROTOCOL_KIND),                                
     .w2_rsc_42_0_WIDTH(w2_rsc_42_0_WIDTH),                                
     .w2_rsc_42_0_RESET_POLARITY(w2_rsc_42_0_RESET_POLARITY),                                
     .w2_rsc_42_0_PROTOCOL_KIND(w2_rsc_42_0_PROTOCOL_KIND),                                
     .w2_rsc_43_0_WIDTH(w2_rsc_43_0_WIDTH),                                
     .w2_rsc_43_0_RESET_POLARITY(w2_rsc_43_0_RESET_POLARITY),                                
     .w2_rsc_43_0_PROTOCOL_KIND(w2_rsc_43_0_PROTOCOL_KIND),                                
     .w2_rsc_44_0_WIDTH(w2_rsc_44_0_WIDTH),                                
     .w2_rsc_44_0_RESET_POLARITY(w2_rsc_44_0_RESET_POLARITY),                                
     .w2_rsc_44_0_PROTOCOL_KIND(w2_rsc_44_0_PROTOCOL_KIND),                                
     .w2_rsc_45_0_WIDTH(w2_rsc_45_0_WIDTH),                                
     .w2_rsc_45_0_RESET_POLARITY(w2_rsc_45_0_RESET_POLARITY),                                
     .w2_rsc_45_0_PROTOCOL_KIND(w2_rsc_45_0_PROTOCOL_KIND),                                
     .w2_rsc_46_0_WIDTH(w2_rsc_46_0_WIDTH),                                
     .w2_rsc_46_0_RESET_POLARITY(w2_rsc_46_0_RESET_POLARITY),                                
     .w2_rsc_46_0_PROTOCOL_KIND(w2_rsc_46_0_PROTOCOL_KIND),                                
     .w2_rsc_47_0_WIDTH(w2_rsc_47_0_WIDTH),                                
     .w2_rsc_47_0_RESET_POLARITY(w2_rsc_47_0_RESET_POLARITY),                                
     .w2_rsc_47_0_PROTOCOL_KIND(w2_rsc_47_0_PROTOCOL_KIND),                                
     .w2_rsc_48_0_WIDTH(w2_rsc_48_0_WIDTH),                                
     .w2_rsc_48_0_RESET_POLARITY(w2_rsc_48_0_RESET_POLARITY),                                
     .w2_rsc_48_0_PROTOCOL_KIND(w2_rsc_48_0_PROTOCOL_KIND),                                
     .w2_rsc_49_0_WIDTH(w2_rsc_49_0_WIDTH),                                
     .w2_rsc_49_0_RESET_POLARITY(w2_rsc_49_0_RESET_POLARITY),                                
     .w2_rsc_49_0_PROTOCOL_KIND(w2_rsc_49_0_PROTOCOL_KIND),                                
     .w2_rsc_50_0_WIDTH(w2_rsc_50_0_WIDTH),                                
     .w2_rsc_50_0_RESET_POLARITY(w2_rsc_50_0_RESET_POLARITY),                                
     .w2_rsc_50_0_PROTOCOL_KIND(w2_rsc_50_0_PROTOCOL_KIND),                                
     .w2_rsc_51_0_WIDTH(w2_rsc_51_0_WIDTH),                                
     .w2_rsc_51_0_RESET_POLARITY(w2_rsc_51_0_RESET_POLARITY),                                
     .w2_rsc_51_0_PROTOCOL_KIND(w2_rsc_51_0_PROTOCOL_KIND),                                
     .w2_rsc_52_0_WIDTH(w2_rsc_52_0_WIDTH),                                
     .w2_rsc_52_0_RESET_POLARITY(w2_rsc_52_0_RESET_POLARITY),                                
     .w2_rsc_52_0_PROTOCOL_KIND(w2_rsc_52_0_PROTOCOL_KIND),                                
     .w2_rsc_53_0_WIDTH(w2_rsc_53_0_WIDTH),                                
     .w2_rsc_53_0_RESET_POLARITY(w2_rsc_53_0_RESET_POLARITY),                                
     .w2_rsc_53_0_PROTOCOL_KIND(w2_rsc_53_0_PROTOCOL_KIND),                                
     .w2_rsc_54_0_WIDTH(w2_rsc_54_0_WIDTH),                                
     .w2_rsc_54_0_RESET_POLARITY(w2_rsc_54_0_RESET_POLARITY),                                
     .w2_rsc_54_0_PROTOCOL_KIND(w2_rsc_54_0_PROTOCOL_KIND),                                
     .w2_rsc_55_0_WIDTH(w2_rsc_55_0_WIDTH),                                
     .w2_rsc_55_0_RESET_POLARITY(w2_rsc_55_0_RESET_POLARITY),                                
     .w2_rsc_55_0_PROTOCOL_KIND(w2_rsc_55_0_PROTOCOL_KIND),                                
     .w2_rsc_56_0_WIDTH(w2_rsc_56_0_WIDTH),                                
     .w2_rsc_56_0_RESET_POLARITY(w2_rsc_56_0_RESET_POLARITY),                                
     .w2_rsc_56_0_PROTOCOL_KIND(w2_rsc_56_0_PROTOCOL_KIND),                                
     .w2_rsc_57_0_WIDTH(w2_rsc_57_0_WIDTH),                                
     .w2_rsc_57_0_RESET_POLARITY(w2_rsc_57_0_RESET_POLARITY),                                
     .w2_rsc_57_0_PROTOCOL_KIND(w2_rsc_57_0_PROTOCOL_KIND),                                
     .w2_rsc_58_0_WIDTH(w2_rsc_58_0_WIDTH),                                
     .w2_rsc_58_0_RESET_POLARITY(w2_rsc_58_0_RESET_POLARITY),                                
     .w2_rsc_58_0_PROTOCOL_KIND(w2_rsc_58_0_PROTOCOL_KIND),                                
     .w2_rsc_59_0_WIDTH(w2_rsc_59_0_WIDTH),                                
     .w2_rsc_59_0_RESET_POLARITY(w2_rsc_59_0_RESET_POLARITY),                                
     .w2_rsc_59_0_PROTOCOL_KIND(w2_rsc_59_0_PROTOCOL_KIND),                                
     .w2_rsc_60_0_WIDTH(w2_rsc_60_0_WIDTH),                                
     .w2_rsc_60_0_RESET_POLARITY(w2_rsc_60_0_RESET_POLARITY),                                
     .w2_rsc_60_0_PROTOCOL_KIND(w2_rsc_60_0_PROTOCOL_KIND),                                
     .w2_rsc_61_0_WIDTH(w2_rsc_61_0_WIDTH),                                
     .w2_rsc_61_0_RESET_POLARITY(w2_rsc_61_0_RESET_POLARITY),                                
     .w2_rsc_61_0_PROTOCOL_KIND(w2_rsc_61_0_PROTOCOL_KIND),                                
     .w2_rsc_62_0_WIDTH(w2_rsc_62_0_WIDTH),                                
     .w2_rsc_62_0_RESET_POLARITY(w2_rsc_62_0_RESET_POLARITY),                                
     .w2_rsc_62_0_PROTOCOL_KIND(w2_rsc_62_0_PROTOCOL_KIND),                                
     .w2_rsc_63_0_WIDTH(w2_rsc_63_0_WIDTH),                                
     .w2_rsc_63_0_RESET_POLARITY(w2_rsc_63_0_RESET_POLARITY),                                
     .w2_rsc_63_0_PROTOCOL_KIND(w2_rsc_63_0_PROTOCOL_KIND),                                
     .b2_rsc_WIDTH(b2_rsc_WIDTH),                                
     .b2_rsc_RESET_POLARITY(b2_rsc_RESET_POLARITY),                                
     .b2_rsc_PROTOCOL_KIND(b2_rsc_PROTOCOL_KIND),                                
     .w4_rsc_0_0_WIDTH(w4_rsc_0_0_WIDTH),                                
     .w4_rsc_0_0_RESET_POLARITY(w4_rsc_0_0_RESET_POLARITY),                                
     .w4_rsc_0_0_PROTOCOL_KIND(w4_rsc_0_0_PROTOCOL_KIND),                                
     .w4_rsc_1_0_WIDTH(w4_rsc_1_0_WIDTH),                                
     .w4_rsc_1_0_RESET_POLARITY(w4_rsc_1_0_RESET_POLARITY),                                
     .w4_rsc_1_0_PROTOCOL_KIND(w4_rsc_1_0_PROTOCOL_KIND),                                
     .w4_rsc_2_0_WIDTH(w4_rsc_2_0_WIDTH),                                
     .w4_rsc_2_0_RESET_POLARITY(w4_rsc_2_0_RESET_POLARITY),                                
     .w4_rsc_2_0_PROTOCOL_KIND(w4_rsc_2_0_PROTOCOL_KIND),                                
     .w4_rsc_3_0_WIDTH(w4_rsc_3_0_WIDTH),                                
     .w4_rsc_3_0_RESET_POLARITY(w4_rsc_3_0_RESET_POLARITY),                                
     .w4_rsc_3_0_PROTOCOL_KIND(w4_rsc_3_0_PROTOCOL_KIND),                                
     .w4_rsc_4_0_WIDTH(w4_rsc_4_0_WIDTH),                                
     .w4_rsc_4_0_RESET_POLARITY(w4_rsc_4_0_RESET_POLARITY),                                
     .w4_rsc_4_0_PROTOCOL_KIND(w4_rsc_4_0_PROTOCOL_KIND),                                
     .w4_rsc_5_0_WIDTH(w4_rsc_5_0_WIDTH),                                
     .w4_rsc_5_0_RESET_POLARITY(w4_rsc_5_0_RESET_POLARITY),                                
     .w4_rsc_5_0_PROTOCOL_KIND(w4_rsc_5_0_PROTOCOL_KIND),                                
     .w4_rsc_6_0_WIDTH(w4_rsc_6_0_WIDTH),                                
     .w4_rsc_6_0_RESET_POLARITY(w4_rsc_6_0_RESET_POLARITY),                                
     .w4_rsc_6_0_PROTOCOL_KIND(w4_rsc_6_0_PROTOCOL_KIND),                                
     .w4_rsc_7_0_WIDTH(w4_rsc_7_0_WIDTH),                                
     .w4_rsc_7_0_RESET_POLARITY(w4_rsc_7_0_RESET_POLARITY),                                
     .w4_rsc_7_0_PROTOCOL_KIND(w4_rsc_7_0_PROTOCOL_KIND),                                
     .w4_rsc_8_0_WIDTH(w4_rsc_8_0_WIDTH),                                
     .w4_rsc_8_0_RESET_POLARITY(w4_rsc_8_0_RESET_POLARITY),                                
     .w4_rsc_8_0_PROTOCOL_KIND(w4_rsc_8_0_PROTOCOL_KIND),                                
     .w4_rsc_9_0_WIDTH(w4_rsc_9_0_WIDTH),                                
     .w4_rsc_9_0_RESET_POLARITY(w4_rsc_9_0_RESET_POLARITY),                                
     .w4_rsc_9_0_PROTOCOL_KIND(w4_rsc_9_0_PROTOCOL_KIND),                                
     .w4_rsc_10_0_WIDTH(w4_rsc_10_0_WIDTH),                                
     .w4_rsc_10_0_RESET_POLARITY(w4_rsc_10_0_RESET_POLARITY),                                
     .w4_rsc_10_0_PROTOCOL_KIND(w4_rsc_10_0_PROTOCOL_KIND),                                
     .w4_rsc_11_0_WIDTH(w4_rsc_11_0_WIDTH),                                
     .w4_rsc_11_0_RESET_POLARITY(w4_rsc_11_0_RESET_POLARITY),                                
     .w4_rsc_11_0_PROTOCOL_KIND(w4_rsc_11_0_PROTOCOL_KIND),                                
     .w4_rsc_12_0_WIDTH(w4_rsc_12_0_WIDTH),                                
     .w4_rsc_12_0_RESET_POLARITY(w4_rsc_12_0_RESET_POLARITY),                                
     .w4_rsc_12_0_PROTOCOL_KIND(w4_rsc_12_0_PROTOCOL_KIND),                                
     .w4_rsc_13_0_WIDTH(w4_rsc_13_0_WIDTH),                                
     .w4_rsc_13_0_RESET_POLARITY(w4_rsc_13_0_RESET_POLARITY),                                
     .w4_rsc_13_0_PROTOCOL_KIND(w4_rsc_13_0_PROTOCOL_KIND),                                
     .w4_rsc_14_0_WIDTH(w4_rsc_14_0_WIDTH),                                
     .w4_rsc_14_0_RESET_POLARITY(w4_rsc_14_0_RESET_POLARITY),                                
     .w4_rsc_14_0_PROTOCOL_KIND(w4_rsc_14_0_PROTOCOL_KIND),                                
     .w4_rsc_15_0_WIDTH(w4_rsc_15_0_WIDTH),                                
     .w4_rsc_15_0_RESET_POLARITY(w4_rsc_15_0_RESET_POLARITY),                                
     .w4_rsc_15_0_PROTOCOL_KIND(w4_rsc_15_0_PROTOCOL_KIND),                                
     .w4_rsc_16_0_WIDTH(w4_rsc_16_0_WIDTH),                                
     .w4_rsc_16_0_RESET_POLARITY(w4_rsc_16_0_RESET_POLARITY),                                
     .w4_rsc_16_0_PROTOCOL_KIND(w4_rsc_16_0_PROTOCOL_KIND),                                
     .w4_rsc_17_0_WIDTH(w4_rsc_17_0_WIDTH),                                
     .w4_rsc_17_0_RESET_POLARITY(w4_rsc_17_0_RESET_POLARITY),                                
     .w4_rsc_17_0_PROTOCOL_KIND(w4_rsc_17_0_PROTOCOL_KIND),                                
     .w4_rsc_18_0_WIDTH(w4_rsc_18_0_WIDTH),                                
     .w4_rsc_18_0_RESET_POLARITY(w4_rsc_18_0_RESET_POLARITY),                                
     .w4_rsc_18_0_PROTOCOL_KIND(w4_rsc_18_0_PROTOCOL_KIND),                                
     .w4_rsc_19_0_WIDTH(w4_rsc_19_0_WIDTH),                                
     .w4_rsc_19_0_RESET_POLARITY(w4_rsc_19_0_RESET_POLARITY),                                
     .w4_rsc_19_0_PROTOCOL_KIND(w4_rsc_19_0_PROTOCOL_KIND),                                
     .w4_rsc_20_0_WIDTH(w4_rsc_20_0_WIDTH),                                
     .w4_rsc_20_0_RESET_POLARITY(w4_rsc_20_0_RESET_POLARITY),                                
     .w4_rsc_20_0_PROTOCOL_KIND(w4_rsc_20_0_PROTOCOL_KIND),                                
     .w4_rsc_21_0_WIDTH(w4_rsc_21_0_WIDTH),                                
     .w4_rsc_21_0_RESET_POLARITY(w4_rsc_21_0_RESET_POLARITY),                                
     .w4_rsc_21_0_PROTOCOL_KIND(w4_rsc_21_0_PROTOCOL_KIND),                                
     .w4_rsc_22_0_WIDTH(w4_rsc_22_0_WIDTH),                                
     .w4_rsc_22_0_RESET_POLARITY(w4_rsc_22_0_RESET_POLARITY),                                
     .w4_rsc_22_0_PROTOCOL_KIND(w4_rsc_22_0_PROTOCOL_KIND),                                
     .w4_rsc_23_0_WIDTH(w4_rsc_23_0_WIDTH),                                
     .w4_rsc_23_0_RESET_POLARITY(w4_rsc_23_0_RESET_POLARITY),                                
     .w4_rsc_23_0_PROTOCOL_KIND(w4_rsc_23_0_PROTOCOL_KIND),                                
     .w4_rsc_24_0_WIDTH(w4_rsc_24_0_WIDTH),                                
     .w4_rsc_24_0_RESET_POLARITY(w4_rsc_24_0_RESET_POLARITY),                                
     .w4_rsc_24_0_PROTOCOL_KIND(w4_rsc_24_0_PROTOCOL_KIND),                                
     .w4_rsc_25_0_WIDTH(w4_rsc_25_0_WIDTH),                                
     .w4_rsc_25_0_RESET_POLARITY(w4_rsc_25_0_RESET_POLARITY),                                
     .w4_rsc_25_0_PROTOCOL_KIND(w4_rsc_25_0_PROTOCOL_KIND),                                
     .w4_rsc_26_0_WIDTH(w4_rsc_26_0_WIDTH),                                
     .w4_rsc_26_0_RESET_POLARITY(w4_rsc_26_0_RESET_POLARITY),                                
     .w4_rsc_26_0_PROTOCOL_KIND(w4_rsc_26_0_PROTOCOL_KIND),                                
     .w4_rsc_27_0_WIDTH(w4_rsc_27_0_WIDTH),                                
     .w4_rsc_27_0_RESET_POLARITY(w4_rsc_27_0_RESET_POLARITY),                                
     .w4_rsc_27_0_PROTOCOL_KIND(w4_rsc_27_0_PROTOCOL_KIND),                                
     .w4_rsc_28_0_WIDTH(w4_rsc_28_0_WIDTH),                                
     .w4_rsc_28_0_RESET_POLARITY(w4_rsc_28_0_RESET_POLARITY),                                
     .w4_rsc_28_0_PROTOCOL_KIND(w4_rsc_28_0_PROTOCOL_KIND),                                
     .w4_rsc_29_0_WIDTH(w4_rsc_29_0_WIDTH),                                
     .w4_rsc_29_0_RESET_POLARITY(w4_rsc_29_0_RESET_POLARITY),                                
     .w4_rsc_29_0_PROTOCOL_KIND(w4_rsc_29_0_PROTOCOL_KIND),                                
     .w4_rsc_30_0_WIDTH(w4_rsc_30_0_WIDTH),                                
     .w4_rsc_30_0_RESET_POLARITY(w4_rsc_30_0_RESET_POLARITY),                                
     .w4_rsc_30_0_PROTOCOL_KIND(w4_rsc_30_0_PROTOCOL_KIND),                                
     .w4_rsc_31_0_WIDTH(w4_rsc_31_0_WIDTH),                                
     .w4_rsc_31_0_RESET_POLARITY(w4_rsc_31_0_RESET_POLARITY),                                
     .w4_rsc_31_0_PROTOCOL_KIND(w4_rsc_31_0_PROTOCOL_KIND),                                
     .w4_rsc_32_0_WIDTH(w4_rsc_32_0_WIDTH),                                
     .w4_rsc_32_0_RESET_POLARITY(w4_rsc_32_0_RESET_POLARITY),                                
     .w4_rsc_32_0_PROTOCOL_KIND(w4_rsc_32_0_PROTOCOL_KIND),                                
     .w4_rsc_33_0_WIDTH(w4_rsc_33_0_WIDTH),                                
     .w4_rsc_33_0_RESET_POLARITY(w4_rsc_33_0_RESET_POLARITY),                                
     .w4_rsc_33_0_PROTOCOL_KIND(w4_rsc_33_0_PROTOCOL_KIND),                                
     .w4_rsc_34_0_WIDTH(w4_rsc_34_0_WIDTH),                                
     .w4_rsc_34_0_RESET_POLARITY(w4_rsc_34_0_RESET_POLARITY),                                
     .w4_rsc_34_0_PROTOCOL_KIND(w4_rsc_34_0_PROTOCOL_KIND),                                
     .w4_rsc_35_0_WIDTH(w4_rsc_35_0_WIDTH),                                
     .w4_rsc_35_0_RESET_POLARITY(w4_rsc_35_0_RESET_POLARITY),                                
     .w4_rsc_35_0_PROTOCOL_KIND(w4_rsc_35_0_PROTOCOL_KIND),                                
     .w4_rsc_36_0_WIDTH(w4_rsc_36_0_WIDTH),                                
     .w4_rsc_36_0_RESET_POLARITY(w4_rsc_36_0_RESET_POLARITY),                                
     .w4_rsc_36_0_PROTOCOL_KIND(w4_rsc_36_0_PROTOCOL_KIND),                                
     .w4_rsc_37_0_WIDTH(w4_rsc_37_0_WIDTH),                                
     .w4_rsc_37_0_RESET_POLARITY(w4_rsc_37_0_RESET_POLARITY),                                
     .w4_rsc_37_0_PROTOCOL_KIND(w4_rsc_37_0_PROTOCOL_KIND),                                
     .w4_rsc_38_0_WIDTH(w4_rsc_38_0_WIDTH),                                
     .w4_rsc_38_0_RESET_POLARITY(w4_rsc_38_0_RESET_POLARITY),                                
     .w4_rsc_38_0_PROTOCOL_KIND(w4_rsc_38_0_PROTOCOL_KIND),                                
     .w4_rsc_39_0_WIDTH(w4_rsc_39_0_WIDTH),                                
     .w4_rsc_39_0_RESET_POLARITY(w4_rsc_39_0_RESET_POLARITY),                                
     .w4_rsc_39_0_PROTOCOL_KIND(w4_rsc_39_0_PROTOCOL_KIND),                                
     .w4_rsc_40_0_WIDTH(w4_rsc_40_0_WIDTH),                                
     .w4_rsc_40_0_RESET_POLARITY(w4_rsc_40_0_RESET_POLARITY),                                
     .w4_rsc_40_0_PROTOCOL_KIND(w4_rsc_40_0_PROTOCOL_KIND),                                
     .w4_rsc_41_0_WIDTH(w4_rsc_41_0_WIDTH),                                
     .w4_rsc_41_0_RESET_POLARITY(w4_rsc_41_0_RESET_POLARITY),                                
     .w4_rsc_41_0_PROTOCOL_KIND(w4_rsc_41_0_PROTOCOL_KIND),                                
     .w4_rsc_42_0_WIDTH(w4_rsc_42_0_WIDTH),                                
     .w4_rsc_42_0_RESET_POLARITY(w4_rsc_42_0_RESET_POLARITY),                                
     .w4_rsc_42_0_PROTOCOL_KIND(w4_rsc_42_0_PROTOCOL_KIND),                                
     .w4_rsc_43_0_WIDTH(w4_rsc_43_0_WIDTH),                                
     .w4_rsc_43_0_RESET_POLARITY(w4_rsc_43_0_RESET_POLARITY),                                
     .w4_rsc_43_0_PROTOCOL_KIND(w4_rsc_43_0_PROTOCOL_KIND),                                
     .w4_rsc_44_0_WIDTH(w4_rsc_44_0_WIDTH),                                
     .w4_rsc_44_0_RESET_POLARITY(w4_rsc_44_0_RESET_POLARITY),                                
     .w4_rsc_44_0_PROTOCOL_KIND(w4_rsc_44_0_PROTOCOL_KIND),                                
     .w4_rsc_45_0_WIDTH(w4_rsc_45_0_WIDTH),                                
     .w4_rsc_45_0_RESET_POLARITY(w4_rsc_45_0_RESET_POLARITY),                                
     .w4_rsc_45_0_PROTOCOL_KIND(w4_rsc_45_0_PROTOCOL_KIND),                                
     .w4_rsc_46_0_WIDTH(w4_rsc_46_0_WIDTH),                                
     .w4_rsc_46_0_RESET_POLARITY(w4_rsc_46_0_RESET_POLARITY),                                
     .w4_rsc_46_0_PROTOCOL_KIND(w4_rsc_46_0_PROTOCOL_KIND),                                
     .w4_rsc_47_0_WIDTH(w4_rsc_47_0_WIDTH),                                
     .w4_rsc_47_0_RESET_POLARITY(w4_rsc_47_0_RESET_POLARITY),                                
     .w4_rsc_47_0_PROTOCOL_KIND(w4_rsc_47_0_PROTOCOL_KIND),                                
     .w4_rsc_48_0_WIDTH(w4_rsc_48_0_WIDTH),                                
     .w4_rsc_48_0_RESET_POLARITY(w4_rsc_48_0_RESET_POLARITY),                                
     .w4_rsc_48_0_PROTOCOL_KIND(w4_rsc_48_0_PROTOCOL_KIND),                                
     .w4_rsc_49_0_WIDTH(w4_rsc_49_0_WIDTH),                                
     .w4_rsc_49_0_RESET_POLARITY(w4_rsc_49_0_RESET_POLARITY),                                
     .w4_rsc_49_0_PROTOCOL_KIND(w4_rsc_49_0_PROTOCOL_KIND),                                
     .w4_rsc_50_0_WIDTH(w4_rsc_50_0_WIDTH),                                
     .w4_rsc_50_0_RESET_POLARITY(w4_rsc_50_0_RESET_POLARITY),                                
     .w4_rsc_50_0_PROTOCOL_KIND(w4_rsc_50_0_PROTOCOL_KIND),                                
     .w4_rsc_51_0_WIDTH(w4_rsc_51_0_WIDTH),                                
     .w4_rsc_51_0_RESET_POLARITY(w4_rsc_51_0_RESET_POLARITY),                                
     .w4_rsc_51_0_PROTOCOL_KIND(w4_rsc_51_0_PROTOCOL_KIND),                                
     .w4_rsc_52_0_WIDTH(w4_rsc_52_0_WIDTH),                                
     .w4_rsc_52_0_RESET_POLARITY(w4_rsc_52_0_RESET_POLARITY),                                
     .w4_rsc_52_0_PROTOCOL_KIND(w4_rsc_52_0_PROTOCOL_KIND),                                
     .w4_rsc_53_0_WIDTH(w4_rsc_53_0_WIDTH),                                
     .w4_rsc_53_0_RESET_POLARITY(w4_rsc_53_0_RESET_POLARITY),                                
     .w4_rsc_53_0_PROTOCOL_KIND(w4_rsc_53_0_PROTOCOL_KIND),                                
     .w4_rsc_54_0_WIDTH(w4_rsc_54_0_WIDTH),                                
     .w4_rsc_54_0_RESET_POLARITY(w4_rsc_54_0_RESET_POLARITY),                                
     .w4_rsc_54_0_PROTOCOL_KIND(w4_rsc_54_0_PROTOCOL_KIND),                                
     .w4_rsc_55_0_WIDTH(w4_rsc_55_0_WIDTH),                                
     .w4_rsc_55_0_RESET_POLARITY(w4_rsc_55_0_RESET_POLARITY),                                
     .w4_rsc_55_0_PROTOCOL_KIND(w4_rsc_55_0_PROTOCOL_KIND),                                
     .w4_rsc_56_0_WIDTH(w4_rsc_56_0_WIDTH),                                
     .w4_rsc_56_0_RESET_POLARITY(w4_rsc_56_0_RESET_POLARITY),                                
     .w4_rsc_56_0_PROTOCOL_KIND(w4_rsc_56_0_PROTOCOL_KIND),                                
     .w4_rsc_57_0_WIDTH(w4_rsc_57_0_WIDTH),                                
     .w4_rsc_57_0_RESET_POLARITY(w4_rsc_57_0_RESET_POLARITY),                                
     .w4_rsc_57_0_PROTOCOL_KIND(w4_rsc_57_0_PROTOCOL_KIND),                                
     .w4_rsc_58_0_WIDTH(w4_rsc_58_0_WIDTH),                                
     .w4_rsc_58_0_RESET_POLARITY(w4_rsc_58_0_RESET_POLARITY),                                
     .w4_rsc_58_0_PROTOCOL_KIND(w4_rsc_58_0_PROTOCOL_KIND),                                
     .w4_rsc_59_0_WIDTH(w4_rsc_59_0_WIDTH),                                
     .w4_rsc_59_0_RESET_POLARITY(w4_rsc_59_0_RESET_POLARITY),                                
     .w4_rsc_59_0_PROTOCOL_KIND(w4_rsc_59_0_PROTOCOL_KIND),                                
     .w4_rsc_60_0_WIDTH(w4_rsc_60_0_WIDTH),                                
     .w4_rsc_60_0_RESET_POLARITY(w4_rsc_60_0_RESET_POLARITY),                                
     .w4_rsc_60_0_PROTOCOL_KIND(w4_rsc_60_0_PROTOCOL_KIND),                                
     .w4_rsc_61_0_WIDTH(w4_rsc_61_0_WIDTH),                                
     .w4_rsc_61_0_RESET_POLARITY(w4_rsc_61_0_RESET_POLARITY),                                
     .w4_rsc_61_0_PROTOCOL_KIND(w4_rsc_61_0_PROTOCOL_KIND),                                
     .w4_rsc_62_0_WIDTH(w4_rsc_62_0_WIDTH),                                
     .w4_rsc_62_0_RESET_POLARITY(w4_rsc_62_0_RESET_POLARITY),                                
     .w4_rsc_62_0_PROTOCOL_KIND(w4_rsc_62_0_PROTOCOL_KIND),                                
     .w4_rsc_63_0_WIDTH(w4_rsc_63_0_WIDTH),                                
     .w4_rsc_63_0_RESET_POLARITY(w4_rsc_63_0_RESET_POLARITY),                                
     .w4_rsc_63_0_PROTOCOL_KIND(w4_rsc_63_0_PROTOCOL_KIND),                                
     .b4_rsc_WIDTH(b4_rsc_WIDTH),                                
     .b4_rsc_RESET_POLARITY(b4_rsc_RESET_POLARITY),                                
     .b4_rsc_PROTOCOL_KIND(b4_rsc_PROTOCOL_KIND),                                
     .w6_rsc_0_0_WIDTH(w6_rsc_0_0_WIDTH),                                
     .w6_rsc_0_0_RESET_POLARITY(w6_rsc_0_0_RESET_POLARITY),                                
     .w6_rsc_0_0_PROTOCOL_KIND(w6_rsc_0_0_PROTOCOL_KIND),                                
     .w6_rsc_1_0_WIDTH(w6_rsc_1_0_WIDTH),                                
     .w6_rsc_1_0_RESET_POLARITY(w6_rsc_1_0_RESET_POLARITY),                                
     .w6_rsc_1_0_PROTOCOL_KIND(w6_rsc_1_0_PROTOCOL_KIND),                                
     .w6_rsc_2_0_WIDTH(w6_rsc_2_0_WIDTH),                                
     .w6_rsc_2_0_RESET_POLARITY(w6_rsc_2_0_RESET_POLARITY),                                
     .w6_rsc_2_0_PROTOCOL_KIND(w6_rsc_2_0_PROTOCOL_KIND),                                
     .w6_rsc_3_0_WIDTH(w6_rsc_3_0_WIDTH),                                
     .w6_rsc_3_0_RESET_POLARITY(w6_rsc_3_0_RESET_POLARITY),                                
     .w6_rsc_3_0_PROTOCOL_KIND(w6_rsc_3_0_PROTOCOL_KIND),                                
     .w6_rsc_4_0_WIDTH(w6_rsc_4_0_WIDTH),                                
     .w6_rsc_4_0_RESET_POLARITY(w6_rsc_4_0_RESET_POLARITY),                                
     .w6_rsc_4_0_PROTOCOL_KIND(w6_rsc_4_0_PROTOCOL_KIND),                                
     .w6_rsc_5_0_WIDTH(w6_rsc_5_0_WIDTH),                                
     .w6_rsc_5_0_RESET_POLARITY(w6_rsc_5_0_RESET_POLARITY),                                
     .w6_rsc_5_0_PROTOCOL_KIND(w6_rsc_5_0_PROTOCOL_KIND),                                
     .w6_rsc_6_0_WIDTH(w6_rsc_6_0_WIDTH),                                
     .w6_rsc_6_0_RESET_POLARITY(w6_rsc_6_0_RESET_POLARITY),                                
     .w6_rsc_6_0_PROTOCOL_KIND(w6_rsc_6_0_PROTOCOL_KIND),                                
     .w6_rsc_7_0_WIDTH(w6_rsc_7_0_WIDTH),                                
     .w6_rsc_7_0_RESET_POLARITY(w6_rsc_7_0_RESET_POLARITY),                                
     .w6_rsc_7_0_PROTOCOL_KIND(w6_rsc_7_0_PROTOCOL_KIND),                                
     .w6_rsc_8_0_WIDTH(w6_rsc_8_0_WIDTH),                                
     .w6_rsc_8_0_RESET_POLARITY(w6_rsc_8_0_RESET_POLARITY),                                
     .w6_rsc_8_0_PROTOCOL_KIND(w6_rsc_8_0_PROTOCOL_KIND),                                
     .w6_rsc_9_0_WIDTH(w6_rsc_9_0_WIDTH),                                
     .w6_rsc_9_0_RESET_POLARITY(w6_rsc_9_0_RESET_POLARITY),                                
     .w6_rsc_9_0_PROTOCOL_KIND(w6_rsc_9_0_PROTOCOL_KIND),                                
     .b6_rsc_WIDTH(b6_rsc_WIDTH),                                
     .b6_rsc_RESET_POLARITY(b6_rsc_RESET_POLARITY),                                
     .b6_rsc_PROTOCOL_KIND(b6_rsc_PROTOCOL_KIND)                                
    )
  ));
  `uvm_component_param_utils( mnist_mlp_environment #(
                              input1_rsc_WIDTH,
                              input1_rsc_RESET_POLARITY,
                              input1_rsc_PROTOCOL_KIND,
                              output1_rsc_WIDTH,
                              output1_rsc_RESET_POLARITY,
                              output1_rsc_PROTOCOL_KIND,
                              const_size_in_1_rsc_WIDTH,
                              const_size_in_1_rsc_RESET_POLARITY,
                              const_size_in_1_rsc_PROTOCOL_KIND,
                              const_size_out_1_rsc_WIDTH,
                              const_size_out_1_rsc_RESET_POLARITY,
                              const_size_out_1_rsc_PROTOCOL_KIND,
                              w2_rsc_0_0_WIDTH,
                              w2_rsc_0_0_RESET_POLARITY,
                              w2_rsc_0_0_PROTOCOL_KIND,
                              w2_rsc_1_0_WIDTH,
                              w2_rsc_1_0_RESET_POLARITY,
                              w2_rsc_1_0_PROTOCOL_KIND,
                              w2_rsc_2_0_WIDTH,
                              w2_rsc_2_0_RESET_POLARITY,
                              w2_rsc_2_0_PROTOCOL_KIND,
                              w2_rsc_3_0_WIDTH,
                              w2_rsc_3_0_RESET_POLARITY,
                              w2_rsc_3_0_PROTOCOL_KIND,
                              w2_rsc_4_0_WIDTH,
                              w2_rsc_4_0_RESET_POLARITY,
                              w2_rsc_4_0_PROTOCOL_KIND,
                              w2_rsc_5_0_WIDTH,
                              w2_rsc_5_0_RESET_POLARITY,
                              w2_rsc_5_0_PROTOCOL_KIND,
                              w2_rsc_6_0_WIDTH,
                              w2_rsc_6_0_RESET_POLARITY,
                              w2_rsc_6_0_PROTOCOL_KIND,
                              w2_rsc_7_0_WIDTH,
                              w2_rsc_7_0_RESET_POLARITY,
                              w2_rsc_7_0_PROTOCOL_KIND,
                              w2_rsc_8_0_WIDTH,
                              w2_rsc_8_0_RESET_POLARITY,
                              w2_rsc_8_0_PROTOCOL_KIND,
                              w2_rsc_9_0_WIDTH,
                              w2_rsc_9_0_RESET_POLARITY,
                              w2_rsc_9_0_PROTOCOL_KIND,
                              w2_rsc_10_0_WIDTH,
                              w2_rsc_10_0_RESET_POLARITY,
                              w2_rsc_10_0_PROTOCOL_KIND,
                              w2_rsc_11_0_WIDTH,
                              w2_rsc_11_0_RESET_POLARITY,
                              w2_rsc_11_0_PROTOCOL_KIND,
                              w2_rsc_12_0_WIDTH,
                              w2_rsc_12_0_RESET_POLARITY,
                              w2_rsc_12_0_PROTOCOL_KIND,
                              w2_rsc_13_0_WIDTH,
                              w2_rsc_13_0_RESET_POLARITY,
                              w2_rsc_13_0_PROTOCOL_KIND,
                              w2_rsc_14_0_WIDTH,
                              w2_rsc_14_0_RESET_POLARITY,
                              w2_rsc_14_0_PROTOCOL_KIND,
                              w2_rsc_15_0_WIDTH,
                              w2_rsc_15_0_RESET_POLARITY,
                              w2_rsc_15_0_PROTOCOL_KIND,
                              w2_rsc_16_0_WIDTH,
                              w2_rsc_16_0_RESET_POLARITY,
                              w2_rsc_16_0_PROTOCOL_KIND,
                              w2_rsc_17_0_WIDTH,
                              w2_rsc_17_0_RESET_POLARITY,
                              w2_rsc_17_0_PROTOCOL_KIND,
                              w2_rsc_18_0_WIDTH,
                              w2_rsc_18_0_RESET_POLARITY,
                              w2_rsc_18_0_PROTOCOL_KIND,
                              w2_rsc_19_0_WIDTH,
                              w2_rsc_19_0_RESET_POLARITY,
                              w2_rsc_19_0_PROTOCOL_KIND,
                              w2_rsc_20_0_WIDTH,
                              w2_rsc_20_0_RESET_POLARITY,
                              w2_rsc_20_0_PROTOCOL_KIND,
                              w2_rsc_21_0_WIDTH,
                              w2_rsc_21_0_RESET_POLARITY,
                              w2_rsc_21_0_PROTOCOL_KIND,
                              w2_rsc_22_0_WIDTH,
                              w2_rsc_22_0_RESET_POLARITY,
                              w2_rsc_22_0_PROTOCOL_KIND,
                              w2_rsc_23_0_WIDTH,
                              w2_rsc_23_0_RESET_POLARITY,
                              w2_rsc_23_0_PROTOCOL_KIND,
                              w2_rsc_24_0_WIDTH,
                              w2_rsc_24_0_RESET_POLARITY,
                              w2_rsc_24_0_PROTOCOL_KIND,
                              w2_rsc_25_0_WIDTH,
                              w2_rsc_25_0_RESET_POLARITY,
                              w2_rsc_25_0_PROTOCOL_KIND,
                              w2_rsc_26_0_WIDTH,
                              w2_rsc_26_0_RESET_POLARITY,
                              w2_rsc_26_0_PROTOCOL_KIND,
                              w2_rsc_27_0_WIDTH,
                              w2_rsc_27_0_RESET_POLARITY,
                              w2_rsc_27_0_PROTOCOL_KIND,
                              w2_rsc_28_0_WIDTH,
                              w2_rsc_28_0_RESET_POLARITY,
                              w2_rsc_28_0_PROTOCOL_KIND,
                              w2_rsc_29_0_WIDTH,
                              w2_rsc_29_0_RESET_POLARITY,
                              w2_rsc_29_0_PROTOCOL_KIND,
                              w2_rsc_30_0_WIDTH,
                              w2_rsc_30_0_RESET_POLARITY,
                              w2_rsc_30_0_PROTOCOL_KIND,
                              w2_rsc_31_0_WIDTH,
                              w2_rsc_31_0_RESET_POLARITY,
                              w2_rsc_31_0_PROTOCOL_KIND,
                              w2_rsc_32_0_WIDTH,
                              w2_rsc_32_0_RESET_POLARITY,
                              w2_rsc_32_0_PROTOCOL_KIND,
                              w2_rsc_33_0_WIDTH,
                              w2_rsc_33_0_RESET_POLARITY,
                              w2_rsc_33_0_PROTOCOL_KIND,
                              w2_rsc_34_0_WIDTH,
                              w2_rsc_34_0_RESET_POLARITY,
                              w2_rsc_34_0_PROTOCOL_KIND,
                              w2_rsc_35_0_WIDTH,
                              w2_rsc_35_0_RESET_POLARITY,
                              w2_rsc_35_0_PROTOCOL_KIND,
                              w2_rsc_36_0_WIDTH,
                              w2_rsc_36_0_RESET_POLARITY,
                              w2_rsc_36_0_PROTOCOL_KIND,
                              w2_rsc_37_0_WIDTH,
                              w2_rsc_37_0_RESET_POLARITY,
                              w2_rsc_37_0_PROTOCOL_KIND,
                              w2_rsc_38_0_WIDTH,
                              w2_rsc_38_0_RESET_POLARITY,
                              w2_rsc_38_0_PROTOCOL_KIND,
                              w2_rsc_39_0_WIDTH,
                              w2_rsc_39_0_RESET_POLARITY,
                              w2_rsc_39_0_PROTOCOL_KIND,
                              w2_rsc_40_0_WIDTH,
                              w2_rsc_40_0_RESET_POLARITY,
                              w2_rsc_40_0_PROTOCOL_KIND,
                              w2_rsc_41_0_WIDTH,
                              w2_rsc_41_0_RESET_POLARITY,
                              w2_rsc_41_0_PROTOCOL_KIND,
                              w2_rsc_42_0_WIDTH,
                              w2_rsc_42_0_RESET_POLARITY,
                              w2_rsc_42_0_PROTOCOL_KIND,
                              w2_rsc_43_0_WIDTH,
                              w2_rsc_43_0_RESET_POLARITY,
                              w2_rsc_43_0_PROTOCOL_KIND,
                              w2_rsc_44_0_WIDTH,
                              w2_rsc_44_0_RESET_POLARITY,
                              w2_rsc_44_0_PROTOCOL_KIND,
                              w2_rsc_45_0_WIDTH,
                              w2_rsc_45_0_RESET_POLARITY,
                              w2_rsc_45_0_PROTOCOL_KIND,
                              w2_rsc_46_0_WIDTH,
                              w2_rsc_46_0_RESET_POLARITY,
                              w2_rsc_46_0_PROTOCOL_KIND,
                              w2_rsc_47_0_WIDTH,
                              w2_rsc_47_0_RESET_POLARITY,
                              w2_rsc_47_0_PROTOCOL_KIND,
                              w2_rsc_48_0_WIDTH,
                              w2_rsc_48_0_RESET_POLARITY,
                              w2_rsc_48_0_PROTOCOL_KIND,
                              w2_rsc_49_0_WIDTH,
                              w2_rsc_49_0_RESET_POLARITY,
                              w2_rsc_49_0_PROTOCOL_KIND,
                              w2_rsc_50_0_WIDTH,
                              w2_rsc_50_0_RESET_POLARITY,
                              w2_rsc_50_0_PROTOCOL_KIND,
                              w2_rsc_51_0_WIDTH,
                              w2_rsc_51_0_RESET_POLARITY,
                              w2_rsc_51_0_PROTOCOL_KIND,
                              w2_rsc_52_0_WIDTH,
                              w2_rsc_52_0_RESET_POLARITY,
                              w2_rsc_52_0_PROTOCOL_KIND,
                              w2_rsc_53_0_WIDTH,
                              w2_rsc_53_0_RESET_POLARITY,
                              w2_rsc_53_0_PROTOCOL_KIND,
                              w2_rsc_54_0_WIDTH,
                              w2_rsc_54_0_RESET_POLARITY,
                              w2_rsc_54_0_PROTOCOL_KIND,
                              w2_rsc_55_0_WIDTH,
                              w2_rsc_55_0_RESET_POLARITY,
                              w2_rsc_55_0_PROTOCOL_KIND,
                              w2_rsc_56_0_WIDTH,
                              w2_rsc_56_0_RESET_POLARITY,
                              w2_rsc_56_0_PROTOCOL_KIND,
                              w2_rsc_57_0_WIDTH,
                              w2_rsc_57_0_RESET_POLARITY,
                              w2_rsc_57_0_PROTOCOL_KIND,
                              w2_rsc_58_0_WIDTH,
                              w2_rsc_58_0_RESET_POLARITY,
                              w2_rsc_58_0_PROTOCOL_KIND,
                              w2_rsc_59_0_WIDTH,
                              w2_rsc_59_0_RESET_POLARITY,
                              w2_rsc_59_0_PROTOCOL_KIND,
                              w2_rsc_60_0_WIDTH,
                              w2_rsc_60_0_RESET_POLARITY,
                              w2_rsc_60_0_PROTOCOL_KIND,
                              w2_rsc_61_0_WIDTH,
                              w2_rsc_61_0_RESET_POLARITY,
                              w2_rsc_61_0_PROTOCOL_KIND,
                              w2_rsc_62_0_WIDTH,
                              w2_rsc_62_0_RESET_POLARITY,
                              w2_rsc_62_0_PROTOCOL_KIND,
                              w2_rsc_63_0_WIDTH,
                              w2_rsc_63_0_RESET_POLARITY,
                              w2_rsc_63_0_PROTOCOL_KIND,
                              b2_rsc_WIDTH,
                              b2_rsc_RESET_POLARITY,
                              b2_rsc_PROTOCOL_KIND,
                              w4_rsc_0_0_WIDTH,
                              w4_rsc_0_0_RESET_POLARITY,
                              w4_rsc_0_0_PROTOCOL_KIND,
                              w4_rsc_1_0_WIDTH,
                              w4_rsc_1_0_RESET_POLARITY,
                              w4_rsc_1_0_PROTOCOL_KIND,
                              w4_rsc_2_0_WIDTH,
                              w4_rsc_2_0_RESET_POLARITY,
                              w4_rsc_2_0_PROTOCOL_KIND,
                              w4_rsc_3_0_WIDTH,
                              w4_rsc_3_0_RESET_POLARITY,
                              w4_rsc_3_0_PROTOCOL_KIND,
                              w4_rsc_4_0_WIDTH,
                              w4_rsc_4_0_RESET_POLARITY,
                              w4_rsc_4_0_PROTOCOL_KIND,
                              w4_rsc_5_0_WIDTH,
                              w4_rsc_5_0_RESET_POLARITY,
                              w4_rsc_5_0_PROTOCOL_KIND,
                              w4_rsc_6_0_WIDTH,
                              w4_rsc_6_0_RESET_POLARITY,
                              w4_rsc_6_0_PROTOCOL_KIND,
                              w4_rsc_7_0_WIDTH,
                              w4_rsc_7_0_RESET_POLARITY,
                              w4_rsc_7_0_PROTOCOL_KIND,
                              w4_rsc_8_0_WIDTH,
                              w4_rsc_8_0_RESET_POLARITY,
                              w4_rsc_8_0_PROTOCOL_KIND,
                              w4_rsc_9_0_WIDTH,
                              w4_rsc_9_0_RESET_POLARITY,
                              w4_rsc_9_0_PROTOCOL_KIND,
                              w4_rsc_10_0_WIDTH,
                              w4_rsc_10_0_RESET_POLARITY,
                              w4_rsc_10_0_PROTOCOL_KIND,
                              w4_rsc_11_0_WIDTH,
                              w4_rsc_11_0_RESET_POLARITY,
                              w4_rsc_11_0_PROTOCOL_KIND,
                              w4_rsc_12_0_WIDTH,
                              w4_rsc_12_0_RESET_POLARITY,
                              w4_rsc_12_0_PROTOCOL_KIND,
                              w4_rsc_13_0_WIDTH,
                              w4_rsc_13_0_RESET_POLARITY,
                              w4_rsc_13_0_PROTOCOL_KIND,
                              w4_rsc_14_0_WIDTH,
                              w4_rsc_14_0_RESET_POLARITY,
                              w4_rsc_14_0_PROTOCOL_KIND,
                              w4_rsc_15_0_WIDTH,
                              w4_rsc_15_0_RESET_POLARITY,
                              w4_rsc_15_0_PROTOCOL_KIND,
                              w4_rsc_16_0_WIDTH,
                              w4_rsc_16_0_RESET_POLARITY,
                              w4_rsc_16_0_PROTOCOL_KIND,
                              w4_rsc_17_0_WIDTH,
                              w4_rsc_17_0_RESET_POLARITY,
                              w4_rsc_17_0_PROTOCOL_KIND,
                              w4_rsc_18_0_WIDTH,
                              w4_rsc_18_0_RESET_POLARITY,
                              w4_rsc_18_0_PROTOCOL_KIND,
                              w4_rsc_19_0_WIDTH,
                              w4_rsc_19_0_RESET_POLARITY,
                              w4_rsc_19_0_PROTOCOL_KIND,
                              w4_rsc_20_0_WIDTH,
                              w4_rsc_20_0_RESET_POLARITY,
                              w4_rsc_20_0_PROTOCOL_KIND,
                              w4_rsc_21_0_WIDTH,
                              w4_rsc_21_0_RESET_POLARITY,
                              w4_rsc_21_0_PROTOCOL_KIND,
                              w4_rsc_22_0_WIDTH,
                              w4_rsc_22_0_RESET_POLARITY,
                              w4_rsc_22_0_PROTOCOL_KIND,
                              w4_rsc_23_0_WIDTH,
                              w4_rsc_23_0_RESET_POLARITY,
                              w4_rsc_23_0_PROTOCOL_KIND,
                              w4_rsc_24_0_WIDTH,
                              w4_rsc_24_0_RESET_POLARITY,
                              w4_rsc_24_0_PROTOCOL_KIND,
                              w4_rsc_25_0_WIDTH,
                              w4_rsc_25_0_RESET_POLARITY,
                              w4_rsc_25_0_PROTOCOL_KIND,
                              w4_rsc_26_0_WIDTH,
                              w4_rsc_26_0_RESET_POLARITY,
                              w4_rsc_26_0_PROTOCOL_KIND,
                              w4_rsc_27_0_WIDTH,
                              w4_rsc_27_0_RESET_POLARITY,
                              w4_rsc_27_0_PROTOCOL_KIND,
                              w4_rsc_28_0_WIDTH,
                              w4_rsc_28_0_RESET_POLARITY,
                              w4_rsc_28_0_PROTOCOL_KIND,
                              w4_rsc_29_0_WIDTH,
                              w4_rsc_29_0_RESET_POLARITY,
                              w4_rsc_29_0_PROTOCOL_KIND,
                              w4_rsc_30_0_WIDTH,
                              w4_rsc_30_0_RESET_POLARITY,
                              w4_rsc_30_0_PROTOCOL_KIND,
                              w4_rsc_31_0_WIDTH,
                              w4_rsc_31_0_RESET_POLARITY,
                              w4_rsc_31_0_PROTOCOL_KIND,
                              w4_rsc_32_0_WIDTH,
                              w4_rsc_32_0_RESET_POLARITY,
                              w4_rsc_32_0_PROTOCOL_KIND,
                              w4_rsc_33_0_WIDTH,
                              w4_rsc_33_0_RESET_POLARITY,
                              w4_rsc_33_0_PROTOCOL_KIND,
                              w4_rsc_34_0_WIDTH,
                              w4_rsc_34_0_RESET_POLARITY,
                              w4_rsc_34_0_PROTOCOL_KIND,
                              w4_rsc_35_0_WIDTH,
                              w4_rsc_35_0_RESET_POLARITY,
                              w4_rsc_35_0_PROTOCOL_KIND,
                              w4_rsc_36_0_WIDTH,
                              w4_rsc_36_0_RESET_POLARITY,
                              w4_rsc_36_0_PROTOCOL_KIND,
                              w4_rsc_37_0_WIDTH,
                              w4_rsc_37_0_RESET_POLARITY,
                              w4_rsc_37_0_PROTOCOL_KIND,
                              w4_rsc_38_0_WIDTH,
                              w4_rsc_38_0_RESET_POLARITY,
                              w4_rsc_38_0_PROTOCOL_KIND,
                              w4_rsc_39_0_WIDTH,
                              w4_rsc_39_0_RESET_POLARITY,
                              w4_rsc_39_0_PROTOCOL_KIND,
                              w4_rsc_40_0_WIDTH,
                              w4_rsc_40_0_RESET_POLARITY,
                              w4_rsc_40_0_PROTOCOL_KIND,
                              w4_rsc_41_0_WIDTH,
                              w4_rsc_41_0_RESET_POLARITY,
                              w4_rsc_41_0_PROTOCOL_KIND,
                              w4_rsc_42_0_WIDTH,
                              w4_rsc_42_0_RESET_POLARITY,
                              w4_rsc_42_0_PROTOCOL_KIND,
                              w4_rsc_43_0_WIDTH,
                              w4_rsc_43_0_RESET_POLARITY,
                              w4_rsc_43_0_PROTOCOL_KIND,
                              w4_rsc_44_0_WIDTH,
                              w4_rsc_44_0_RESET_POLARITY,
                              w4_rsc_44_0_PROTOCOL_KIND,
                              w4_rsc_45_0_WIDTH,
                              w4_rsc_45_0_RESET_POLARITY,
                              w4_rsc_45_0_PROTOCOL_KIND,
                              w4_rsc_46_0_WIDTH,
                              w4_rsc_46_0_RESET_POLARITY,
                              w4_rsc_46_0_PROTOCOL_KIND,
                              w4_rsc_47_0_WIDTH,
                              w4_rsc_47_0_RESET_POLARITY,
                              w4_rsc_47_0_PROTOCOL_KIND,
                              w4_rsc_48_0_WIDTH,
                              w4_rsc_48_0_RESET_POLARITY,
                              w4_rsc_48_0_PROTOCOL_KIND,
                              w4_rsc_49_0_WIDTH,
                              w4_rsc_49_0_RESET_POLARITY,
                              w4_rsc_49_0_PROTOCOL_KIND,
                              w4_rsc_50_0_WIDTH,
                              w4_rsc_50_0_RESET_POLARITY,
                              w4_rsc_50_0_PROTOCOL_KIND,
                              w4_rsc_51_0_WIDTH,
                              w4_rsc_51_0_RESET_POLARITY,
                              w4_rsc_51_0_PROTOCOL_KIND,
                              w4_rsc_52_0_WIDTH,
                              w4_rsc_52_0_RESET_POLARITY,
                              w4_rsc_52_0_PROTOCOL_KIND,
                              w4_rsc_53_0_WIDTH,
                              w4_rsc_53_0_RESET_POLARITY,
                              w4_rsc_53_0_PROTOCOL_KIND,
                              w4_rsc_54_0_WIDTH,
                              w4_rsc_54_0_RESET_POLARITY,
                              w4_rsc_54_0_PROTOCOL_KIND,
                              w4_rsc_55_0_WIDTH,
                              w4_rsc_55_0_RESET_POLARITY,
                              w4_rsc_55_0_PROTOCOL_KIND,
                              w4_rsc_56_0_WIDTH,
                              w4_rsc_56_0_RESET_POLARITY,
                              w4_rsc_56_0_PROTOCOL_KIND,
                              w4_rsc_57_0_WIDTH,
                              w4_rsc_57_0_RESET_POLARITY,
                              w4_rsc_57_0_PROTOCOL_KIND,
                              w4_rsc_58_0_WIDTH,
                              w4_rsc_58_0_RESET_POLARITY,
                              w4_rsc_58_0_PROTOCOL_KIND,
                              w4_rsc_59_0_WIDTH,
                              w4_rsc_59_0_RESET_POLARITY,
                              w4_rsc_59_0_PROTOCOL_KIND,
                              w4_rsc_60_0_WIDTH,
                              w4_rsc_60_0_RESET_POLARITY,
                              w4_rsc_60_0_PROTOCOL_KIND,
                              w4_rsc_61_0_WIDTH,
                              w4_rsc_61_0_RESET_POLARITY,
                              w4_rsc_61_0_PROTOCOL_KIND,
                              w4_rsc_62_0_WIDTH,
                              w4_rsc_62_0_RESET_POLARITY,
                              w4_rsc_62_0_PROTOCOL_KIND,
                              w4_rsc_63_0_WIDTH,
                              w4_rsc_63_0_RESET_POLARITY,
                              w4_rsc_63_0_PROTOCOL_KIND,
                              b4_rsc_WIDTH,
                              b4_rsc_RESET_POLARITY,
                              b4_rsc_PROTOCOL_KIND,
                              w6_rsc_0_0_WIDTH,
                              w6_rsc_0_0_RESET_POLARITY,
                              w6_rsc_0_0_PROTOCOL_KIND,
                              w6_rsc_1_0_WIDTH,
                              w6_rsc_1_0_RESET_POLARITY,
                              w6_rsc_1_0_PROTOCOL_KIND,
                              w6_rsc_2_0_WIDTH,
                              w6_rsc_2_0_RESET_POLARITY,
                              w6_rsc_2_0_PROTOCOL_KIND,
                              w6_rsc_3_0_WIDTH,
                              w6_rsc_3_0_RESET_POLARITY,
                              w6_rsc_3_0_PROTOCOL_KIND,
                              w6_rsc_4_0_WIDTH,
                              w6_rsc_4_0_RESET_POLARITY,
                              w6_rsc_4_0_PROTOCOL_KIND,
                              w6_rsc_5_0_WIDTH,
                              w6_rsc_5_0_RESET_POLARITY,
                              w6_rsc_5_0_PROTOCOL_KIND,
                              w6_rsc_6_0_WIDTH,
                              w6_rsc_6_0_RESET_POLARITY,
                              w6_rsc_6_0_PROTOCOL_KIND,
                              w6_rsc_7_0_WIDTH,
                              w6_rsc_7_0_RESET_POLARITY,
                              w6_rsc_7_0_PROTOCOL_KIND,
                              w6_rsc_8_0_WIDTH,
                              w6_rsc_8_0_RESET_POLARITY,
                              w6_rsc_8_0_PROTOCOL_KIND,
                              w6_rsc_9_0_WIDTH,
                              w6_rsc_9_0_RESET_POLARITY,
                              w6_rsc_9_0_PROTOCOL_KIND,
                              b6_rsc_WIDTH,
                              b6_rsc_RESET_POLARITY,
                              b6_rsc_PROTOCOL_KIND
                            ))





  typedef ccs_agent #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1)) input1_rsc_agent_t;
  input1_rsc_agent_t input1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1)) output1_rsc_agent_t;
  output1_rsc_agent_t output1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_in_1_rsc_agent_t;
  const_size_in_1_rsc_agent_t const_size_in_1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_out_1_rsc_agent_t;
  const_size_out_1_rsc_agent_t const_size_out_1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_0_0_agent_t;
  w2_rsc_0_0_agent_t w2_rsc_0_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_1_0_agent_t;
  w2_rsc_1_0_agent_t w2_rsc_1_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_2_0_agent_t;
  w2_rsc_2_0_agent_t w2_rsc_2_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_3_0_agent_t;
  w2_rsc_3_0_agent_t w2_rsc_3_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_4_0_agent_t;
  w2_rsc_4_0_agent_t w2_rsc_4_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_5_0_agent_t;
  w2_rsc_5_0_agent_t w2_rsc_5_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_6_0_agent_t;
  w2_rsc_6_0_agent_t w2_rsc_6_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_7_0_agent_t;
  w2_rsc_7_0_agent_t w2_rsc_7_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_8_0_agent_t;
  w2_rsc_8_0_agent_t w2_rsc_8_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_9_0_agent_t;
  w2_rsc_9_0_agent_t w2_rsc_9_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_10_0_agent_t;
  w2_rsc_10_0_agent_t w2_rsc_10_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_11_0_agent_t;
  w2_rsc_11_0_agent_t w2_rsc_11_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_12_0_agent_t;
  w2_rsc_12_0_agent_t w2_rsc_12_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_13_0_agent_t;
  w2_rsc_13_0_agent_t w2_rsc_13_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_14_0_agent_t;
  w2_rsc_14_0_agent_t w2_rsc_14_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_15_0_agent_t;
  w2_rsc_15_0_agent_t w2_rsc_15_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_16_0_agent_t;
  w2_rsc_16_0_agent_t w2_rsc_16_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_17_0_agent_t;
  w2_rsc_17_0_agent_t w2_rsc_17_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_18_0_agent_t;
  w2_rsc_18_0_agent_t w2_rsc_18_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_19_0_agent_t;
  w2_rsc_19_0_agent_t w2_rsc_19_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_20_0_agent_t;
  w2_rsc_20_0_agent_t w2_rsc_20_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_21_0_agent_t;
  w2_rsc_21_0_agent_t w2_rsc_21_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_22_0_agent_t;
  w2_rsc_22_0_agent_t w2_rsc_22_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_23_0_agent_t;
  w2_rsc_23_0_agent_t w2_rsc_23_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_24_0_agent_t;
  w2_rsc_24_0_agent_t w2_rsc_24_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_25_0_agent_t;
  w2_rsc_25_0_agent_t w2_rsc_25_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_26_0_agent_t;
  w2_rsc_26_0_agent_t w2_rsc_26_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_27_0_agent_t;
  w2_rsc_27_0_agent_t w2_rsc_27_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_28_0_agent_t;
  w2_rsc_28_0_agent_t w2_rsc_28_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_29_0_agent_t;
  w2_rsc_29_0_agent_t w2_rsc_29_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_30_0_agent_t;
  w2_rsc_30_0_agent_t w2_rsc_30_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_31_0_agent_t;
  w2_rsc_31_0_agent_t w2_rsc_31_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_32_0_agent_t;
  w2_rsc_32_0_agent_t w2_rsc_32_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_33_0_agent_t;
  w2_rsc_33_0_agent_t w2_rsc_33_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_34_0_agent_t;
  w2_rsc_34_0_agent_t w2_rsc_34_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_35_0_agent_t;
  w2_rsc_35_0_agent_t w2_rsc_35_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_36_0_agent_t;
  w2_rsc_36_0_agent_t w2_rsc_36_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_37_0_agent_t;
  w2_rsc_37_0_agent_t w2_rsc_37_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_38_0_agent_t;
  w2_rsc_38_0_agent_t w2_rsc_38_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_39_0_agent_t;
  w2_rsc_39_0_agent_t w2_rsc_39_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_40_0_agent_t;
  w2_rsc_40_0_agent_t w2_rsc_40_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_41_0_agent_t;
  w2_rsc_41_0_agent_t w2_rsc_41_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_42_0_agent_t;
  w2_rsc_42_0_agent_t w2_rsc_42_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_43_0_agent_t;
  w2_rsc_43_0_agent_t w2_rsc_43_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_44_0_agent_t;
  w2_rsc_44_0_agent_t w2_rsc_44_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_45_0_agent_t;
  w2_rsc_45_0_agent_t w2_rsc_45_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_46_0_agent_t;
  w2_rsc_46_0_agent_t w2_rsc_46_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_47_0_agent_t;
  w2_rsc_47_0_agent_t w2_rsc_47_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_48_0_agent_t;
  w2_rsc_48_0_agent_t w2_rsc_48_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_49_0_agent_t;
  w2_rsc_49_0_agent_t w2_rsc_49_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_50_0_agent_t;
  w2_rsc_50_0_agent_t w2_rsc_50_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_51_0_agent_t;
  w2_rsc_51_0_agent_t w2_rsc_51_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_52_0_agent_t;
  w2_rsc_52_0_agent_t w2_rsc_52_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_53_0_agent_t;
  w2_rsc_53_0_agent_t w2_rsc_53_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_54_0_agent_t;
  w2_rsc_54_0_agent_t w2_rsc_54_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_55_0_agent_t;
  w2_rsc_55_0_agent_t w2_rsc_55_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_56_0_agent_t;
  w2_rsc_56_0_agent_t w2_rsc_56_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_57_0_agent_t;
  w2_rsc_57_0_agent_t w2_rsc_57_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_58_0_agent_t;
  w2_rsc_58_0_agent_t w2_rsc_58_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_59_0_agent_t;
  w2_rsc_59_0_agent_t w2_rsc_59_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_60_0_agent_t;
  w2_rsc_60_0_agent_t w2_rsc_60_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_61_0_agent_t;
  w2_rsc_61_0_agent_t w2_rsc_61_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_62_0_agent_t;
  w2_rsc_62_0_agent_t w2_rsc_62_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_63_0_agent_t;
  w2_rsc_63_0_agent_t w2_rsc_63_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) b2_rsc_agent_t;
  b2_rsc_agent_t b2_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_0_0_agent_t;
  w4_rsc_0_0_agent_t w4_rsc_0_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_1_0_agent_t;
  w4_rsc_1_0_agent_t w4_rsc_1_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_2_0_agent_t;
  w4_rsc_2_0_agent_t w4_rsc_2_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_3_0_agent_t;
  w4_rsc_3_0_agent_t w4_rsc_3_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_4_0_agent_t;
  w4_rsc_4_0_agent_t w4_rsc_4_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_5_0_agent_t;
  w4_rsc_5_0_agent_t w4_rsc_5_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_6_0_agent_t;
  w4_rsc_6_0_agent_t w4_rsc_6_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_7_0_agent_t;
  w4_rsc_7_0_agent_t w4_rsc_7_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_8_0_agent_t;
  w4_rsc_8_0_agent_t w4_rsc_8_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_9_0_agent_t;
  w4_rsc_9_0_agent_t w4_rsc_9_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_10_0_agent_t;
  w4_rsc_10_0_agent_t w4_rsc_10_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_11_0_agent_t;
  w4_rsc_11_0_agent_t w4_rsc_11_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_12_0_agent_t;
  w4_rsc_12_0_agent_t w4_rsc_12_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_13_0_agent_t;
  w4_rsc_13_0_agent_t w4_rsc_13_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_14_0_agent_t;
  w4_rsc_14_0_agent_t w4_rsc_14_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_15_0_agent_t;
  w4_rsc_15_0_agent_t w4_rsc_15_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_16_0_agent_t;
  w4_rsc_16_0_agent_t w4_rsc_16_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_17_0_agent_t;
  w4_rsc_17_0_agent_t w4_rsc_17_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_18_0_agent_t;
  w4_rsc_18_0_agent_t w4_rsc_18_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_19_0_agent_t;
  w4_rsc_19_0_agent_t w4_rsc_19_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_20_0_agent_t;
  w4_rsc_20_0_agent_t w4_rsc_20_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_21_0_agent_t;
  w4_rsc_21_0_agent_t w4_rsc_21_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_22_0_agent_t;
  w4_rsc_22_0_agent_t w4_rsc_22_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_23_0_agent_t;
  w4_rsc_23_0_agent_t w4_rsc_23_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_24_0_agent_t;
  w4_rsc_24_0_agent_t w4_rsc_24_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_25_0_agent_t;
  w4_rsc_25_0_agent_t w4_rsc_25_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_26_0_agent_t;
  w4_rsc_26_0_agent_t w4_rsc_26_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_27_0_agent_t;
  w4_rsc_27_0_agent_t w4_rsc_27_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_28_0_agent_t;
  w4_rsc_28_0_agent_t w4_rsc_28_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_29_0_agent_t;
  w4_rsc_29_0_agent_t w4_rsc_29_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_30_0_agent_t;
  w4_rsc_30_0_agent_t w4_rsc_30_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_31_0_agent_t;
  w4_rsc_31_0_agent_t w4_rsc_31_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_32_0_agent_t;
  w4_rsc_32_0_agent_t w4_rsc_32_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_33_0_agent_t;
  w4_rsc_33_0_agent_t w4_rsc_33_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_34_0_agent_t;
  w4_rsc_34_0_agent_t w4_rsc_34_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_35_0_agent_t;
  w4_rsc_35_0_agent_t w4_rsc_35_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_36_0_agent_t;
  w4_rsc_36_0_agent_t w4_rsc_36_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_37_0_agent_t;
  w4_rsc_37_0_agent_t w4_rsc_37_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_38_0_agent_t;
  w4_rsc_38_0_agent_t w4_rsc_38_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_39_0_agent_t;
  w4_rsc_39_0_agent_t w4_rsc_39_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_40_0_agent_t;
  w4_rsc_40_0_agent_t w4_rsc_40_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_41_0_agent_t;
  w4_rsc_41_0_agent_t w4_rsc_41_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_42_0_agent_t;
  w4_rsc_42_0_agent_t w4_rsc_42_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_43_0_agent_t;
  w4_rsc_43_0_agent_t w4_rsc_43_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_44_0_agent_t;
  w4_rsc_44_0_agent_t w4_rsc_44_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_45_0_agent_t;
  w4_rsc_45_0_agent_t w4_rsc_45_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_46_0_agent_t;
  w4_rsc_46_0_agent_t w4_rsc_46_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_47_0_agent_t;
  w4_rsc_47_0_agent_t w4_rsc_47_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_48_0_agent_t;
  w4_rsc_48_0_agent_t w4_rsc_48_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_49_0_agent_t;
  w4_rsc_49_0_agent_t w4_rsc_49_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_50_0_agent_t;
  w4_rsc_50_0_agent_t w4_rsc_50_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_51_0_agent_t;
  w4_rsc_51_0_agent_t w4_rsc_51_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_52_0_agent_t;
  w4_rsc_52_0_agent_t w4_rsc_52_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_53_0_agent_t;
  w4_rsc_53_0_agent_t w4_rsc_53_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_54_0_agent_t;
  w4_rsc_54_0_agent_t w4_rsc_54_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_55_0_agent_t;
  w4_rsc_55_0_agent_t w4_rsc_55_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_56_0_agent_t;
  w4_rsc_56_0_agent_t w4_rsc_56_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_57_0_agent_t;
  w4_rsc_57_0_agent_t w4_rsc_57_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_58_0_agent_t;
  w4_rsc_58_0_agent_t w4_rsc_58_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_59_0_agent_t;
  w4_rsc_59_0_agent_t w4_rsc_59_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_60_0_agent_t;
  w4_rsc_60_0_agent_t w4_rsc_60_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_61_0_agent_t;
  w4_rsc_61_0_agent_t w4_rsc_61_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_62_0_agent_t;
  w4_rsc_62_0_agent_t w4_rsc_62_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_63_0_agent_t;
  w4_rsc_63_0_agent_t w4_rsc_63_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) b4_rsc_agent_t;
  b4_rsc_agent_t b4_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_0_0_agent_t;
  w6_rsc_0_0_agent_t w6_rsc_0_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_1_0_agent_t;
  w6_rsc_1_0_agent_t w6_rsc_1_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_2_0_agent_t;
  w6_rsc_2_0_agent_t w6_rsc_2_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_3_0_agent_t;
  w6_rsc_3_0_agent_t w6_rsc_3_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_4_0_agent_t;
  w6_rsc_4_0_agent_t w6_rsc_4_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_5_0_agent_t;
  w6_rsc_5_0_agent_t w6_rsc_5_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_6_0_agent_t;
  w6_rsc_6_0_agent_t w6_rsc_6_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_7_0_agent_t;
  w6_rsc_7_0_agent_t w6_rsc_7_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_8_0_agent_t;
  w6_rsc_8_0_agent_t w6_rsc_8_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_9_0_agent_t;
  w6_rsc_9_0_agent_t w6_rsc_9_0;

  typedef ccs_agent #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1)) b6_rsc_agent_t;
  b6_rsc_agent_t b6_rsc;




  typedef mnist_mlp_predictor  #(
                             .input1_rsc_WIDTH(input1_rsc_WIDTH),                                
                             .input1_rsc_RESET_POLARITY(input1_rsc_RESET_POLARITY),                                
                             .input1_rsc_PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND),                                
                             .output1_rsc_WIDTH(output1_rsc_WIDTH),                                
                             .output1_rsc_RESET_POLARITY(output1_rsc_RESET_POLARITY),                                
                             .output1_rsc_PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND),                                
                             .const_size_in_1_rsc_WIDTH(const_size_in_1_rsc_WIDTH),                                
                             .const_size_in_1_rsc_RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),                                
                             .const_size_in_1_rsc_PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND),                                
                             .const_size_out_1_rsc_WIDTH(const_size_out_1_rsc_WIDTH),                                
                             .const_size_out_1_rsc_RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),                                
                             .const_size_out_1_rsc_PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND),                                
                             .w2_rsc_0_0_WIDTH(w2_rsc_0_0_WIDTH),                                
                             .w2_rsc_0_0_RESET_POLARITY(w2_rsc_0_0_RESET_POLARITY),                                
                             .w2_rsc_0_0_PROTOCOL_KIND(w2_rsc_0_0_PROTOCOL_KIND),                                
                             .w2_rsc_1_0_WIDTH(w2_rsc_1_0_WIDTH),                                
                             .w2_rsc_1_0_RESET_POLARITY(w2_rsc_1_0_RESET_POLARITY),                                
                             .w2_rsc_1_0_PROTOCOL_KIND(w2_rsc_1_0_PROTOCOL_KIND),                                
                             .w2_rsc_2_0_WIDTH(w2_rsc_2_0_WIDTH),                                
                             .w2_rsc_2_0_RESET_POLARITY(w2_rsc_2_0_RESET_POLARITY),                                
                             .w2_rsc_2_0_PROTOCOL_KIND(w2_rsc_2_0_PROTOCOL_KIND),                                
                             .w2_rsc_3_0_WIDTH(w2_rsc_3_0_WIDTH),                                
                             .w2_rsc_3_0_RESET_POLARITY(w2_rsc_3_0_RESET_POLARITY),                                
                             .w2_rsc_3_0_PROTOCOL_KIND(w2_rsc_3_0_PROTOCOL_KIND),                                
                             .w2_rsc_4_0_WIDTH(w2_rsc_4_0_WIDTH),                                
                             .w2_rsc_4_0_RESET_POLARITY(w2_rsc_4_0_RESET_POLARITY),                                
                             .w2_rsc_4_0_PROTOCOL_KIND(w2_rsc_4_0_PROTOCOL_KIND),                                
                             .w2_rsc_5_0_WIDTH(w2_rsc_5_0_WIDTH),                                
                             .w2_rsc_5_0_RESET_POLARITY(w2_rsc_5_0_RESET_POLARITY),                                
                             .w2_rsc_5_0_PROTOCOL_KIND(w2_rsc_5_0_PROTOCOL_KIND),                                
                             .w2_rsc_6_0_WIDTH(w2_rsc_6_0_WIDTH),                                
                             .w2_rsc_6_0_RESET_POLARITY(w2_rsc_6_0_RESET_POLARITY),                                
                             .w2_rsc_6_0_PROTOCOL_KIND(w2_rsc_6_0_PROTOCOL_KIND),                                
                             .w2_rsc_7_0_WIDTH(w2_rsc_7_0_WIDTH),                                
                             .w2_rsc_7_0_RESET_POLARITY(w2_rsc_7_0_RESET_POLARITY),                                
                             .w2_rsc_7_0_PROTOCOL_KIND(w2_rsc_7_0_PROTOCOL_KIND),                                
                             .w2_rsc_8_0_WIDTH(w2_rsc_8_0_WIDTH),                                
                             .w2_rsc_8_0_RESET_POLARITY(w2_rsc_8_0_RESET_POLARITY),                                
                             .w2_rsc_8_0_PROTOCOL_KIND(w2_rsc_8_0_PROTOCOL_KIND),                                
                             .w2_rsc_9_0_WIDTH(w2_rsc_9_0_WIDTH),                                
                             .w2_rsc_9_0_RESET_POLARITY(w2_rsc_9_0_RESET_POLARITY),                                
                             .w2_rsc_9_0_PROTOCOL_KIND(w2_rsc_9_0_PROTOCOL_KIND),                                
                             .w2_rsc_10_0_WIDTH(w2_rsc_10_0_WIDTH),                                
                             .w2_rsc_10_0_RESET_POLARITY(w2_rsc_10_0_RESET_POLARITY),                                
                             .w2_rsc_10_0_PROTOCOL_KIND(w2_rsc_10_0_PROTOCOL_KIND),                                
                             .w2_rsc_11_0_WIDTH(w2_rsc_11_0_WIDTH),                                
                             .w2_rsc_11_0_RESET_POLARITY(w2_rsc_11_0_RESET_POLARITY),                                
                             .w2_rsc_11_0_PROTOCOL_KIND(w2_rsc_11_0_PROTOCOL_KIND),                                
                             .w2_rsc_12_0_WIDTH(w2_rsc_12_0_WIDTH),                                
                             .w2_rsc_12_0_RESET_POLARITY(w2_rsc_12_0_RESET_POLARITY),                                
                             .w2_rsc_12_0_PROTOCOL_KIND(w2_rsc_12_0_PROTOCOL_KIND),                                
                             .w2_rsc_13_0_WIDTH(w2_rsc_13_0_WIDTH),                                
                             .w2_rsc_13_0_RESET_POLARITY(w2_rsc_13_0_RESET_POLARITY),                                
                             .w2_rsc_13_0_PROTOCOL_KIND(w2_rsc_13_0_PROTOCOL_KIND),                                
                             .w2_rsc_14_0_WIDTH(w2_rsc_14_0_WIDTH),                                
                             .w2_rsc_14_0_RESET_POLARITY(w2_rsc_14_0_RESET_POLARITY),                                
                             .w2_rsc_14_0_PROTOCOL_KIND(w2_rsc_14_0_PROTOCOL_KIND),                                
                             .w2_rsc_15_0_WIDTH(w2_rsc_15_0_WIDTH),                                
                             .w2_rsc_15_0_RESET_POLARITY(w2_rsc_15_0_RESET_POLARITY),                                
                             .w2_rsc_15_0_PROTOCOL_KIND(w2_rsc_15_0_PROTOCOL_KIND),                                
                             .w2_rsc_16_0_WIDTH(w2_rsc_16_0_WIDTH),                                
                             .w2_rsc_16_0_RESET_POLARITY(w2_rsc_16_0_RESET_POLARITY),                                
                             .w2_rsc_16_0_PROTOCOL_KIND(w2_rsc_16_0_PROTOCOL_KIND),                                
                             .w2_rsc_17_0_WIDTH(w2_rsc_17_0_WIDTH),                                
                             .w2_rsc_17_0_RESET_POLARITY(w2_rsc_17_0_RESET_POLARITY),                                
                             .w2_rsc_17_0_PROTOCOL_KIND(w2_rsc_17_0_PROTOCOL_KIND),                                
                             .w2_rsc_18_0_WIDTH(w2_rsc_18_0_WIDTH),                                
                             .w2_rsc_18_0_RESET_POLARITY(w2_rsc_18_0_RESET_POLARITY),                                
                             .w2_rsc_18_0_PROTOCOL_KIND(w2_rsc_18_0_PROTOCOL_KIND),                                
                             .w2_rsc_19_0_WIDTH(w2_rsc_19_0_WIDTH),                                
                             .w2_rsc_19_0_RESET_POLARITY(w2_rsc_19_0_RESET_POLARITY),                                
                             .w2_rsc_19_0_PROTOCOL_KIND(w2_rsc_19_0_PROTOCOL_KIND),                                
                             .w2_rsc_20_0_WIDTH(w2_rsc_20_0_WIDTH),                                
                             .w2_rsc_20_0_RESET_POLARITY(w2_rsc_20_0_RESET_POLARITY),                                
                             .w2_rsc_20_0_PROTOCOL_KIND(w2_rsc_20_0_PROTOCOL_KIND),                                
                             .w2_rsc_21_0_WIDTH(w2_rsc_21_0_WIDTH),                                
                             .w2_rsc_21_0_RESET_POLARITY(w2_rsc_21_0_RESET_POLARITY),                                
                             .w2_rsc_21_0_PROTOCOL_KIND(w2_rsc_21_0_PROTOCOL_KIND),                                
                             .w2_rsc_22_0_WIDTH(w2_rsc_22_0_WIDTH),                                
                             .w2_rsc_22_0_RESET_POLARITY(w2_rsc_22_0_RESET_POLARITY),                                
                             .w2_rsc_22_0_PROTOCOL_KIND(w2_rsc_22_0_PROTOCOL_KIND),                                
                             .w2_rsc_23_0_WIDTH(w2_rsc_23_0_WIDTH),                                
                             .w2_rsc_23_0_RESET_POLARITY(w2_rsc_23_0_RESET_POLARITY),                                
                             .w2_rsc_23_0_PROTOCOL_KIND(w2_rsc_23_0_PROTOCOL_KIND),                                
                             .w2_rsc_24_0_WIDTH(w2_rsc_24_0_WIDTH),                                
                             .w2_rsc_24_0_RESET_POLARITY(w2_rsc_24_0_RESET_POLARITY),                                
                             .w2_rsc_24_0_PROTOCOL_KIND(w2_rsc_24_0_PROTOCOL_KIND),                                
                             .w2_rsc_25_0_WIDTH(w2_rsc_25_0_WIDTH),                                
                             .w2_rsc_25_0_RESET_POLARITY(w2_rsc_25_0_RESET_POLARITY),                                
                             .w2_rsc_25_0_PROTOCOL_KIND(w2_rsc_25_0_PROTOCOL_KIND),                                
                             .w2_rsc_26_0_WIDTH(w2_rsc_26_0_WIDTH),                                
                             .w2_rsc_26_0_RESET_POLARITY(w2_rsc_26_0_RESET_POLARITY),                                
                             .w2_rsc_26_0_PROTOCOL_KIND(w2_rsc_26_0_PROTOCOL_KIND),                                
                             .w2_rsc_27_0_WIDTH(w2_rsc_27_0_WIDTH),                                
                             .w2_rsc_27_0_RESET_POLARITY(w2_rsc_27_0_RESET_POLARITY),                                
                             .w2_rsc_27_0_PROTOCOL_KIND(w2_rsc_27_0_PROTOCOL_KIND),                                
                             .w2_rsc_28_0_WIDTH(w2_rsc_28_0_WIDTH),                                
                             .w2_rsc_28_0_RESET_POLARITY(w2_rsc_28_0_RESET_POLARITY),                                
                             .w2_rsc_28_0_PROTOCOL_KIND(w2_rsc_28_0_PROTOCOL_KIND),                                
                             .w2_rsc_29_0_WIDTH(w2_rsc_29_0_WIDTH),                                
                             .w2_rsc_29_0_RESET_POLARITY(w2_rsc_29_0_RESET_POLARITY),                                
                             .w2_rsc_29_0_PROTOCOL_KIND(w2_rsc_29_0_PROTOCOL_KIND),                                
                             .w2_rsc_30_0_WIDTH(w2_rsc_30_0_WIDTH),                                
                             .w2_rsc_30_0_RESET_POLARITY(w2_rsc_30_0_RESET_POLARITY),                                
                             .w2_rsc_30_0_PROTOCOL_KIND(w2_rsc_30_0_PROTOCOL_KIND),                                
                             .w2_rsc_31_0_WIDTH(w2_rsc_31_0_WIDTH),                                
                             .w2_rsc_31_0_RESET_POLARITY(w2_rsc_31_0_RESET_POLARITY),                                
                             .w2_rsc_31_0_PROTOCOL_KIND(w2_rsc_31_0_PROTOCOL_KIND),                                
                             .w2_rsc_32_0_WIDTH(w2_rsc_32_0_WIDTH),                                
                             .w2_rsc_32_0_RESET_POLARITY(w2_rsc_32_0_RESET_POLARITY),                                
                             .w2_rsc_32_0_PROTOCOL_KIND(w2_rsc_32_0_PROTOCOL_KIND),                                
                             .w2_rsc_33_0_WIDTH(w2_rsc_33_0_WIDTH),                                
                             .w2_rsc_33_0_RESET_POLARITY(w2_rsc_33_0_RESET_POLARITY),                                
                             .w2_rsc_33_0_PROTOCOL_KIND(w2_rsc_33_0_PROTOCOL_KIND),                                
                             .w2_rsc_34_0_WIDTH(w2_rsc_34_0_WIDTH),                                
                             .w2_rsc_34_0_RESET_POLARITY(w2_rsc_34_0_RESET_POLARITY),                                
                             .w2_rsc_34_0_PROTOCOL_KIND(w2_rsc_34_0_PROTOCOL_KIND),                                
                             .w2_rsc_35_0_WIDTH(w2_rsc_35_0_WIDTH),                                
                             .w2_rsc_35_0_RESET_POLARITY(w2_rsc_35_0_RESET_POLARITY),                                
                             .w2_rsc_35_0_PROTOCOL_KIND(w2_rsc_35_0_PROTOCOL_KIND),                                
                             .w2_rsc_36_0_WIDTH(w2_rsc_36_0_WIDTH),                                
                             .w2_rsc_36_0_RESET_POLARITY(w2_rsc_36_0_RESET_POLARITY),                                
                             .w2_rsc_36_0_PROTOCOL_KIND(w2_rsc_36_0_PROTOCOL_KIND),                                
                             .w2_rsc_37_0_WIDTH(w2_rsc_37_0_WIDTH),                                
                             .w2_rsc_37_0_RESET_POLARITY(w2_rsc_37_0_RESET_POLARITY),                                
                             .w2_rsc_37_0_PROTOCOL_KIND(w2_rsc_37_0_PROTOCOL_KIND),                                
                             .w2_rsc_38_0_WIDTH(w2_rsc_38_0_WIDTH),                                
                             .w2_rsc_38_0_RESET_POLARITY(w2_rsc_38_0_RESET_POLARITY),                                
                             .w2_rsc_38_0_PROTOCOL_KIND(w2_rsc_38_0_PROTOCOL_KIND),                                
                             .w2_rsc_39_0_WIDTH(w2_rsc_39_0_WIDTH),                                
                             .w2_rsc_39_0_RESET_POLARITY(w2_rsc_39_0_RESET_POLARITY),                                
                             .w2_rsc_39_0_PROTOCOL_KIND(w2_rsc_39_0_PROTOCOL_KIND),                                
                             .w2_rsc_40_0_WIDTH(w2_rsc_40_0_WIDTH),                                
                             .w2_rsc_40_0_RESET_POLARITY(w2_rsc_40_0_RESET_POLARITY),                                
                             .w2_rsc_40_0_PROTOCOL_KIND(w2_rsc_40_0_PROTOCOL_KIND),                                
                             .w2_rsc_41_0_WIDTH(w2_rsc_41_0_WIDTH),                                
                             .w2_rsc_41_0_RESET_POLARITY(w2_rsc_41_0_RESET_POLARITY),                                
                             .w2_rsc_41_0_PROTOCOL_KIND(w2_rsc_41_0_PROTOCOL_KIND),                                
                             .w2_rsc_42_0_WIDTH(w2_rsc_42_0_WIDTH),                                
                             .w2_rsc_42_0_RESET_POLARITY(w2_rsc_42_0_RESET_POLARITY),                                
                             .w2_rsc_42_0_PROTOCOL_KIND(w2_rsc_42_0_PROTOCOL_KIND),                                
                             .w2_rsc_43_0_WIDTH(w2_rsc_43_0_WIDTH),                                
                             .w2_rsc_43_0_RESET_POLARITY(w2_rsc_43_0_RESET_POLARITY),                                
                             .w2_rsc_43_0_PROTOCOL_KIND(w2_rsc_43_0_PROTOCOL_KIND),                                
                             .w2_rsc_44_0_WIDTH(w2_rsc_44_0_WIDTH),                                
                             .w2_rsc_44_0_RESET_POLARITY(w2_rsc_44_0_RESET_POLARITY),                                
                             .w2_rsc_44_0_PROTOCOL_KIND(w2_rsc_44_0_PROTOCOL_KIND),                                
                             .w2_rsc_45_0_WIDTH(w2_rsc_45_0_WIDTH),                                
                             .w2_rsc_45_0_RESET_POLARITY(w2_rsc_45_0_RESET_POLARITY),                                
                             .w2_rsc_45_0_PROTOCOL_KIND(w2_rsc_45_0_PROTOCOL_KIND),                                
                             .w2_rsc_46_0_WIDTH(w2_rsc_46_0_WIDTH),                                
                             .w2_rsc_46_0_RESET_POLARITY(w2_rsc_46_0_RESET_POLARITY),                                
                             .w2_rsc_46_0_PROTOCOL_KIND(w2_rsc_46_0_PROTOCOL_KIND),                                
                             .w2_rsc_47_0_WIDTH(w2_rsc_47_0_WIDTH),                                
                             .w2_rsc_47_0_RESET_POLARITY(w2_rsc_47_0_RESET_POLARITY),                                
                             .w2_rsc_47_0_PROTOCOL_KIND(w2_rsc_47_0_PROTOCOL_KIND),                                
                             .w2_rsc_48_0_WIDTH(w2_rsc_48_0_WIDTH),                                
                             .w2_rsc_48_0_RESET_POLARITY(w2_rsc_48_0_RESET_POLARITY),                                
                             .w2_rsc_48_0_PROTOCOL_KIND(w2_rsc_48_0_PROTOCOL_KIND),                                
                             .w2_rsc_49_0_WIDTH(w2_rsc_49_0_WIDTH),                                
                             .w2_rsc_49_0_RESET_POLARITY(w2_rsc_49_0_RESET_POLARITY),                                
                             .w2_rsc_49_0_PROTOCOL_KIND(w2_rsc_49_0_PROTOCOL_KIND),                                
                             .w2_rsc_50_0_WIDTH(w2_rsc_50_0_WIDTH),                                
                             .w2_rsc_50_0_RESET_POLARITY(w2_rsc_50_0_RESET_POLARITY),                                
                             .w2_rsc_50_0_PROTOCOL_KIND(w2_rsc_50_0_PROTOCOL_KIND),                                
                             .w2_rsc_51_0_WIDTH(w2_rsc_51_0_WIDTH),                                
                             .w2_rsc_51_0_RESET_POLARITY(w2_rsc_51_0_RESET_POLARITY),                                
                             .w2_rsc_51_0_PROTOCOL_KIND(w2_rsc_51_0_PROTOCOL_KIND),                                
                             .w2_rsc_52_0_WIDTH(w2_rsc_52_0_WIDTH),                                
                             .w2_rsc_52_0_RESET_POLARITY(w2_rsc_52_0_RESET_POLARITY),                                
                             .w2_rsc_52_0_PROTOCOL_KIND(w2_rsc_52_0_PROTOCOL_KIND),                                
                             .w2_rsc_53_0_WIDTH(w2_rsc_53_0_WIDTH),                                
                             .w2_rsc_53_0_RESET_POLARITY(w2_rsc_53_0_RESET_POLARITY),                                
                             .w2_rsc_53_0_PROTOCOL_KIND(w2_rsc_53_0_PROTOCOL_KIND),                                
                             .w2_rsc_54_0_WIDTH(w2_rsc_54_0_WIDTH),                                
                             .w2_rsc_54_0_RESET_POLARITY(w2_rsc_54_0_RESET_POLARITY),                                
                             .w2_rsc_54_0_PROTOCOL_KIND(w2_rsc_54_0_PROTOCOL_KIND),                                
                             .w2_rsc_55_0_WIDTH(w2_rsc_55_0_WIDTH),                                
                             .w2_rsc_55_0_RESET_POLARITY(w2_rsc_55_0_RESET_POLARITY),                                
                             .w2_rsc_55_0_PROTOCOL_KIND(w2_rsc_55_0_PROTOCOL_KIND),                                
                             .w2_rsc_56_0_WIDTH(w2_rsc_56_0_WIDTH),                                
                             .w2_rsc_56_0_RESET_POLARITY(w2_rsc_56_0_RESET_POLARITY),                                
                             .w2_rsc_56_0_PROTOCOL_KIND(w2_rsc_56_0_PROTOCOL_KIND),                                
                             .w2_rsc_57_0_WIDTH(w2_rsc_57_0_WIDTH),                                
                             .w2_rsc_57_0_RESET_POLARITY(w2_rsc_57_0_RESET_POLARITY),                                
                             .w2_rsc_57_0_PROTOCOL_KIND(w2_rsc_57_0_PROTOCOL_KIND),                                
                             .w2_rsc_58_0_WIDTH(w2_rsc_58_0_WIDTH),                                
                             .w2_rsc_58_0_RESET_POLARITY(w2_rsc_58_0_RESET_POLARITY),                                
                             .w2_rsc_58_0_PROTOCOL_KIND(w2_rsc_58_0_PROTOCOL_KIND),                                
                             .w2_rsc_59_0_WIDTH(w2_rsc_59_0_WIDTH),                                
                             .w2_rsc_59_0_RESET_POLARITY(w2_rsc_59_0_RESET_POLARITY),                                
                             .w2_rsc_59_0_PROTOCOL_KIND(w2_rsc_59_0_PROTOCOL_KIND),                                
                             .w2_rsc_60_0_WIDTH(w2_rsc_60_0_WIDTH),                                
                             .w2_rsc_60_0_RESET_POLARITY(w2_rsc_60_0_RESET_POLARITY),                                
                             .w2_rsc_60_0_PROTOCOL_KIND(w2_rsc_60_0_PROTOCOL_KIND),                                
                             .w2_rsc_61_0_WIDTH(w2_rsc_61_0_WIDTH),                                
                             .w2_rsc_61_0_RESET_POLARITY(w2_rsc_61_0_RESET_POLARITY),                                
                             .w2_rsc_61_0_PROTOCOL_KIND(w2_rsc_61_0_PROTOCOL_KIND),                                
                             .w2_rsc_62_0_WIDTH(w2_rsc_62_0_WIDTH),                                
                             .w2_rsc_62_0_RESET_POLARITY(w2_rsc_62_0_RESET_POLARITY),                                
                             .w2_rsc_62_0_PROTOCOL_KIND(w2_rsc_62_0_PROTOCOL_KIND),                                
                             .w2_rsc_63_0_WIDTH(w2_rsc_63_0_WIDTH),                                
                             .w2_rsc_63_0_RESET_POLARITY(w2_rsc_63_0_RESET_POLARITY),                                
                             .w2_rsc_63_0_PROTOCOL_KIND(w2_rsc_63_0_PROTOCOL_KIND),                                
                             .b2_rsc_WIDTH(b2_rsc_WIDTH),                                
                             .b2_rsc_RESET_POLARITY(b2_rsc_RESET_POLARITY),                                
                             .b2_rsc_PROTOCOL_KIND(b2_rsc_PROTOCOL_KIND),                                
                             .w4_rsc_0_0_WIDTH(w4_rsc_0_0_WIDTH),                                
                             .w4_rsc_0_0_RESET_POLARITY(w4_rsc_0_0_RESET_POLARITY),                                
                             .w4_rsc_0_0_PROTOCOL_KIND(w4_rsc_0_0_PROTOCOL_KIND),                                
                             .w4_rsc_1_0_WIDTH(w4_rsc_1_0_WIDTH),                                
                             .w4_rsc_1_0_RESET_POLARITY(w4_rsc_1_0_RESET_POLARITY),                                
                             .w4_rsc_1_0_PROTOCOL_KIND(w4_rsc_1_0_PROTOCOL_KIND),                                
                             .w4_rsc_2_0_WIDTH(w4_rsc_2_0_WIDTH),                                
                             .w4_rsc_2_0_RESET_POLARITY(w4_rsc_2_0_RESET_POLARITY),                                
                             .w4_rsc_2_0_PROTOCOL_KIND(w4_rsc_2_0_PROTOCOL_KIND),                                
                             .w4_rsc_3_0_WIDTH(w4_rsc_3_0_WIDTH),                                
                             .w4_rsc_3_0_RESET_POLARITY(w4_rsc_3_0_RESET_POLARITY),                                
                             .w4_rsc_3_0_PROTOCOL_KIND(w4_rsc_3_0_PROTOCOL_KIND),                                
                             .w4_rsc_4_0_WIDTH(w4_rsc_4_0_WIDTH),                                
                             .w4_rsc_4_0_RESET_POLARITY(w4_rsc_4_0_RESET_POLARITY),                                
                             .w4_rsc_4_0_PROTOCOL_KIND(w4_rsc_4_0_PROTOCOL_KIND),                                
                             .w4_rsc_5_0_WIDTH(w4_rsc_5_0_WIDTH),                                
                             .w4_rsc_5_0_RESET_POLARITY(w4_rsc_5_0_RESET_POLARITY),                                
                             .w4_rsc_5_0_PROTOCOL_KIND(w4_rsc_5_0_PROTOCOL_KIND),                                
                             .w4_rsc_6_0_WIDTH(w4_rsc_6_0_WIDTH),                                
                             .w4_rsc_6_0_RESET_POLARITY(w4_rsc_6_0_RESET_POLARITY),                                
                             .w4_rsc_6_0_PROTOCOL_KIND(w4_rsc_6_0_PROTOCOL_KIND),                                
                             .w4_rsc_7_0_WIDTH(w4_rsc_7_0_WIDTH),                                
                             .w4_rsc_7_0_RESET_POLARITY(w4_rsc_7_0_RESET_POLARITY),                                
                             .w4_rsc_7_0_PROTOCOL_KIND(w4_rsc_7_0_PROTOCOL_KIND),                                
                             .w4_rsc_8_0_WIDTH(w4_rsc_8_0_WIDTH),                                
                             .w4_rsc_8_0_RESET_POLARITY(w4_rsc_8_0_RESET_POLARITY),                                
                             .w4_rsc_8_0_PROTOCOL_KIND(w4_rsc_8_0_PROTOCOL_KIND),                                
                             .w4_rsc_9_0_WIDTH(w4_rsc_9_0_WIDTH),                                
                             .w4_rsc_9_0_RESET_POLARITY(w4_rsc_9_0_RESET_POLARITY),                                
                             .w4_rsc_9_0_PROTOCOL_KIND(w4_rsc_9_0_PROTOCOL_KIND),                                
                             .w4_rsc_10_0_WIDTH(w4_rsc_10_0_WIDTH),                                
                             .w4_rsc_10_0_RESET_POLARITY(w4_rsc_10_0_RESET_POLARITY),                                
                             .w4_rsc_10_0_PROTOCOL_KIND(w4_rsc_10_0_PROTOCOL_KIND),                                
                             .w4_rsc_11_0_WIDTH(w4_rsc_11_0_WIDTH),                                
                             .w4_rsc_11_0_RESET_POLARITY(w4_rsc_11_0_RESET_POLARITY),                                
                             .w4_rsc_11_0_PROTOCOL_KIND(w4_rsc_11_0_PROTOCOL_KIND),                                
                             .w4_rsc_12_0_WIDTH(w4_rsc_12_0_WIDTH),                                
                             .w4_rsc_12_0_RESET_POLARITY(w4_rsc_12_0_RESET_POLARITY),                                
                             .w4_rsc_12_0_PROTOCOL_KIND(w4_rsc_12_0_PROTOCOL_KIND),                                
                             .w4_rsc_13_0_WIDTH(w4_rsc_13_0_WIDTH),                                
                             .w4_rsc_13_0_RESET_POLARITY(w4_rsc_13_0_RESET_POLARITY),                                
                             .w4_rsc_13_0_PROTOCOL_KIND(w4_rsc_13_0_PROTOCOL_KIND),                                
                             .w4_rsc_14_0_WIDTH(w4_rsc_14_0_WIDTH),                                
                             .w4_rsc_14_0_RESET_POLARITY(w4_rsc_14_0_RESET_POLARITY),                                
                             .w4_rsc_14_0_PROTOCOL_KIND(w4_rsc_14_0_PROTOCOL_KIND),                                
                             .w4_rsc_15_0_WIDTH(w4_rsc_15_0_WIDTH),                                
                             .w4_rsc_15_0_RESET_POLARITY(w4_rsc_15_0_RESET_POLARITY),                                
                             .w4_rsc_15_0_PROTOCOL_KIND(w4_rsc_15_0_PROTOCOL_KIND),                                
                             .w4_rsc_16_0_WIDTH(w4_rsc_16_0_WIDTH),                                
                             .w4_rsc_16_0_RESET_POLARITY(w4_rsc_16_0_RESET_POLARITY),                                
                             .w4_rsc_16_0_PROTOCOL_KIND(w4_rsc_16_0_PROTOCOL_KIND),                                
                             .w4_rsc_17_0_WIDTH(w4_rsc_17_0_WIDTH),                                
                             .w4_rsc_17_0_RESET_POLARITY(w4_rsc_17_0_RESET_POLARITY),                                
                             .w4_rsc_17_0_PROTOCOL_KIND(w4_rsc_17_0_PROTOCOL_KIND),                                
                             .w4_rsc_18_0_WIDTH(w4_rsc_18_0_WIDTH),                                
                             .w4_rsc_18_0_RESET_POLARITY(w4_rsc_18_0_RESET_POLARITY),                                
                             .w4_rsc_18_0_PROTOCOL_KIND(w4_rsc_18_0_PROTOCOL_KIND),                                
                             .w4_rsc_19_0_WIDTH(w4_rsc_19_0_WIDTH),                                
                             .w4_rsc_19_0_RESET_POLARITY(w4_rsc_19_0_RESET_POLARITY),                                
                             .w4_rsc_19_0_PROTOCOL_KIND(w4_rsc_19_0_PROTOCOL_KIND),                                
                             .w4_rsc_20_0_WIDTH(w4_rsc_20_0_WIDTH),                                
                             .w4_rsc_20_0_RESET_POLARITY(w4_rsc_20_0_RESET_POLARITY),                                
                             .w4_rsc_20_0_PROTOCOL_KIND(w4_rsc_20_0_PROTOCOL_KIND),                                
                             .w4_rsc_21_0_WIDTH(w4_rsc_21_0_WIDTH),                                
                             .w4_rsc_21_0_RESET_POLARITY(w4_rsc_21_0_RESET_POLARITY),                                
                             .w4_rsc_21_0_PROTOCOL_KIND(w4_rsc_21_0_PROTOCOL_KIND),                                
                             .w4_rsc_22_0_WIDTH(w4_rsc_22_0_WIDTH),                                
                             .w4_rsc_22_0_RESET_POLARITY(w4_rsc_22_0_RESET_POLARITY),                                
                             .w4_rsc_22_0_PROTOCOL_KIND(w4_rsc_22_0_PROTOCOL_KIND),                                
                             .w4_rsc_23_0_WIDTH(w4_rsc_23_0_WIDTH),                                
                             .w4_rsc_23_0_RESET_POLARITY(w4_rsc_23_0_RESET_POLARITY),                                
                             .w4_rsc_23_0_PROTOCOL_KIND(w4_rsc_23_0_PROTOCOL_KIND),                                
                             .w4_rsc_24_0_WIDTH(w4_rsc_24_0_WIDTH),                                
                             .w4_rsc_24_0_RESET_POLARITY(w4_rsc_24_0_RESET_POLARITY),                                
                             .w4_rsc_24_0_PROTOCOL_KIND(w4_rsc_24_0_PROTOCOL_KIND),                                
                             .w4_rsc_25_0_WIDTH(w4_rsc_25_0_WIDTH),                                
                             .w4_rsc_25_0_RESET_POLARITY(w4_rsc_25_0_RESET_POLARITY),                                
                             .w4_rsc_25_0_PROTOCOL_KIND(w4_rsc_25_0_PROTOCOL_KIND),                                
                             .w4_rsc_26_0_WIDTH(w4_rsc_26_0_WIDTH),                                
                             .w4_rsc_26_0_RESET_POLARITY(w4_rsc_26_0_RESET_POLARITY),                                
                             .w4_rsc_26_0_PROTOCOL_KIND(w4_rsc_26_0_PROTOCOL_KIND),                                
                             .w4_rsc_27_0_WIDTH(w4_rsc_27_0_WIDTH),                                
                             .w4_rsc_27_0_RESET_POLARITY(w4_rsc_27_0_RESET_POLARITY),                                
                             .w4_rsc_27_0_PROTOCOL_KIND(w4_rsc_27_0_PROTOCOL_KIND),                                
                             .w4_rsc_28_0_WIDTH(w4_rsc_28_0_WIDTH),                                
                             .w4_rsc_28_0_RESET_POLARITY(w4_rsc_28_0_RESET_POLARITY),                                
                             .w4_rsc_28_0_PROTOCOL_KIND(w4_rsc_28_0_PROTOCOL_KIND),                                
                             .w4_rsc_29_0_WIDTH(w4_rsc_29_0_WIDTH),                                
                             .w4_rsc_29_0_RESET_POLARITY(w4_rsc_29_0_RESET_POLARITY),                                
                             .w4_rsc_29_0_PROTOCOL_KIND(w4_rsc_29_0_PROTOCOL_KIND),                                
                             .w4_rsc_30_0_WIDTH(w4_rsc_30_0_WIDTH),                                
                             .w4_rsc_30_0_RESET_POLARITY(w4_rsc_30_0_RESET_POLARITY),                                
                             .w4_rsc_30_0_PROTOCOL_KIND(w4_rsc_30_0_PROTOCOL_KIND),                                
                             .w4_rsc_31_0_WIDTH(w4_rsc_31_0_WIDTH),                                
                             .w4_rsc_31_0_RESET_POLARITY(w4_rsc_31_0_RESET_POLARITY),                                
                             .w4_rsc_31_0_PROTOCOL_KIND(w4_rsc_31_0_PROTOCOL_KIND),                                
                             .w4_rsc_32_0_WIDTH(w4_rsc_32_0_WIDTH),                                
                             .w4_rsc_32_0_RESET_POLARITY(w4_rsc_32_0_RESET_POLARITY),                                
                             .w4_rsc_32_0_PROTOCOL_KIND(w4_rsc_32_0_PROTOCOL_KIND),                                
                             .w4_rsc_33_0_WIDTH(w4_rsc_33_0_WIDTH),                                
                             .w4_rsc_33_0_RESET_POLARITY(w4_rsc_33_0_RESET_POLARITY),                                
                             .w4_rsc_33_0_PROTOCOL_KIND(w4_rsc_33_0_PROTOCOL_KIND),                                
                             .w4_rsc_34_0_WIDTH(w4_rsc_34_0_WIDTH),                                
                             .w4_rsc_34_0_RESET_POLARITY(w4_rsc_34_0_RESET_POLARITY),                                
                             .w4_rsc_34_0_PROTOCOL_KIND(w4_rsc_34_0_PROTOCOL_KIND),                                
                             .w4_rsc_35_0_WIDTH(w4_rsc_35_0_WIDTH),                                
                             .w4_rsc_35_0_RESET_POLARITY(w4_rsc_35_0_RESET_POLARITY),                                
                             .w4_rsc_35_0_PROTOCOL_KIND(w4_rsc_35_0_PROTOCOL_KIND),                                
                             .w4_rsc_36_0_WIDTH(w4_rsc_36_0_WIDTH),                                
                             .w4_rsc_36_0_RESET_POLARITY(w4_rsc_36_0_RESET_POLARITY),                                
                             .w4_rsc_36_0_PROTOCOL_KIND(w4_rsc_36_0_PROTOCOL_KIND),                                
                             .w4_rsc_37_0_WIDTH(w4_rsc_37_0_WIDTH),                                
                             .w4_rsc_37_0_RESET_POLARITY(w4_rsc_37_0_RESET_POLARITY),                                
                             .w4_rsc_37_0_PROTOCOL_KIND(w4_rsc_37_0_PROTOCOL_KIND),                                
                             .w4_rsc_38_0_WIDTH(w4_rsc_38_0_WIDTH),                                
                             .w4_rsc_38_0_RESET_POLARITY(w4_rsc_38_0_RESET_POLARITY),                                
                             .w4_rsc_38_0_PROTOCOL_KIND(w4_rsc_38_0_PROTOCOL_KIND),                                
                             .w4_rsc_39_0_WIDTH(w4_rsc_39_0_WIDTH),                                
                             .w4_rsc_39_0_RESET_POLARITY(w4_rsc_39_0_RESET_POLARITY),                                
                             .w4_rsc_39_0_PROTOCOL_KIND(w4_rsc_39_0_PROTOCOL_KIND),                                
                             .w4_rsc_40_0_WIDTH(w4_rsc_40_0_WIDTH),                                
                             .w4_rsc_40_0_RESET_POLARITY(w4_rsc_40_0_RESET_POLARITY),                                
                             .w4_rsc_40_0_PROTOCOL_KIND(w4_rsc_40_0_PROTOCOL_KIND),                                
                             .w4_rsc_41_0_WIDTH(w4_rsc_41_0_WIDTH),                                
                             .w4_rsc_41_0_RESET_POLARITY(w4_rsc_41_0_RESET_POLARITY),                                
                             .w4_rsc_41_0_PROTOCOL_KIND(w4_rsc_41_0_PROTOCOL_KIND),                                
                             .w4_rsc_42_0_WIDTH(w4_rsc_42_0_WIDTH),                                
                             .w4_rsc_42_0_RESET_POLARITY(w4_rsc_42_0_RESET_POLARITY),                                
                             .w4_rsc_42_0_PROTOCOL_KIND(w4_rsc_42_0_PROTOCOL_KIND),                                
                             .w4_rsc_43_0_WIDTH(w4_rsc_43_0_WIDTH),                                
                             .w4_rsc_43_0_RESET_POLARITY(w4_rsc_43_0_RESET_POLARITY),                                
                             .w4_rsc_43_0_PROTOCOL_KIND(w4_rsc_43_0_PROTOCOL_KIND),                                
                             .w4_rsc_44_0_WIDTH(w4_rsc_44_0_WIDTH),                                
                             .w4_rsc_44_0_RESET_POLARITY(w4_rsc_44_0_RESET_POLARITY),                                
                             .w4_rsc_44_0_PROTOCOL_KIND(w4_rsc_44_0_PROTOCOL_KIND),                                
                             .w4_rsc_45_0_WIDTH(w4_rsc_45_0_WIDTH),                                
                             .w4_rsc_45_0_RESET_POLARITY(w4_rsc_45_0_RESET_POLARITY),                                
                             .w4_rsc_45_0_PROTOCOL_KIND(w4_rsc_45_0_PROTOCOL_KIND),                                
                             .w4_rsc_46_0_WIDTH(w4_rsc_46_0_WIDTH),                                
                             .w4_rsc_46_0_RESET_POLARITY(w4_rsc_46_0_RESET_POLARITY),                                
                             .w4_rsc_46_0_PROTOCOL_KIND(w4_rsc_46_0_PROTOCOL_KIND),                                
                             .w4_rsc_47_0_WIDTH(w4_rsc_47_0_WIDTH),                                
                             .w4_rsc_47_0_RESET_POLARITY(w4_rsc_47_0_RESET_POLARITY),                                
                             .w4_rsc_47_0_PROTOCOL_KIND(w4_rsc_47_0_PROTOCOL_KIND),                                
                             .w4_rsc_48_0_WIDTH(w4_rsc_48_0_WIDTH),                                
                             .w4_rsc_48_0_RESET_POLARITY(w4_rsc_48_0_RESET_POLARITY),                                
                             .w4_rsc_48_0_PROTOCOL_KIND(w4_rsc_48_0_PROTOCOL_KIND),                                
                             .w4_rsc_49_0_WIDTH(w4_rsc_49_0_WIDTH),                                
                             .w4_rsc_49_0_RESET_POLARITY(w4_rsc_49_0_RESET_POLARITY),                                
                             .w4_rsc_49_0_PROTOCOL_KIND(w4_rsc_49_0_PROTOCOL_KIND),                                
                             .w4_rsc_50_0_WIDTH(w4_rsc_50_0_WIDTH),                                
                             .w4_rsc_50_0_RESET_POLARITY(w4_rsc_50_0_RESET_POLARITY),                                
                             .w4_rsc_50_0_PROTOCOL_KIND(w4_rsc_50_0_PROTOCOL_KIND),                                
                             .w4_rsc_51_0_WIDTH(w4_rsc_51_0_WIDTH),                                
                             .w4_rsc_51_0_RESET_POLARITY(w4_rsc_51_0_RESET_POLARITY),                                
                             .w4_rsc_51_0_PROTOCOL_KIND(w4_rsc_51_0_PROTOCOL_KIND),                                
                             .w4_rsc_52_0_WIDTH(w4_rsc_52_0_WIDTH),                                
                             .w4_rsc_52_0_RESET_POLARITY(w4_rsc_52_0_RESET_POLARITY),                                
                             .w4_rsc_52_0_PROTOCOL_KIND(w4_rsc_52_0_PROTOCOL_KIND),                                
                             .w4_rsc_53_0_WIDTH(w4_rsc_53_0_WIDTH),                                
                             .w4_rsc_53_0_RESET_POLARITY(w4_rsc_53_0_RESET_POLARITY),                                
                             .w4_rsc_53_0_PROTOCOL_KIND(w4_rsc_53_0_PROTOCOL_KIND),                                
                             .w4_rsc_54_0_WIDTH(w4_rsc_54_0_WIDTH),                                
                             .w4_rsc_54_0_RESET_POLARITY(w4_rsc_54_0_RESET_POLARITY),                                
                             .w4_rsc_54_0_PROTOCOL_KIND(w4_rsc_54_0_PROTOCOL_KIND),                                
                             .w4_rsc_55_0_WIDTH(w4_rsc_55_0_WIDTH),                                
                             .w4_rsc_55_0_RESET_POLARITY(w4_rsc_55_0_RESET_POLARITY),                                
                             .w4_rsc_55_0_PROTOCOL_KIND(w4_rsc_55_0_PROTOCOL_KIND),                                
                             .w4_rsc_56_0_WIDTH(w4_rsc_56_0_WIDTH),                                
                             .w4_rsc_56_0_RESET_POLARITY(w4_rsc_56_0_RESET_POLARITY),                                
                             .w4_rsc_56_0_PROTOCOL_KIND(w4_rsc_56_0_PROTOCOL_KIND),                                
                             .w4_rsc_57_0_WIDTH(w4_rsc_57_0_WIDTH),                                
                             .w4_rsc_57_0_RESET_POLARITY(w4_rsc_57_0_RESET_POLARITY),                                
                             .w4_rsc_57_0_PROTOCOL_KIND(w4_rsc_57_0_PROTOCOL_KIND),                                
                             .w4_rsc_58_0_WIDTH(w4_rsc_58_0_WIDTH),                                
                             .w4_rsc_58_0_RESET_POLARITY(w4_rsc_58_0_RESET_POLARITY),                                
                             .w4_rsc_58_0_PROTOCOL_KIND(w4_rsc_58_0_PROTOCOL_KIND),                                
                             .w4_rsc_59_0_WIDTH(w4_rsc_59_0_WIDTH),                                
                             .w4_rsc_59_0_RESET_POLARITY(w4_rsc_59_0_RESET_POLARITY),                                
                             .w4_rsc_59_0_PROTOCOL_KIND(w4_rsc_59_0_PROTOCOL_KIND),                                
                             .w4_rsc_60_0_WIDTH(w4_rsc_60_0_WIDTH),                                
                             .w4_rsc_60_0_RESET_POLARITY(w4_rsc_60_0_RESET_POLARITY),                                
                             .w4_rsc_60_0_PROTOCOL_KIND(w4_rsc_60_0_PROTOCOL_KIND),                                
                             .w4_rsc_61_0_WIDTH(w4_rsc_61_0_WIDTH),                                
                             .w4_rsc_61_0_RESET_POLARITY(w4_rsc_61_0_RESET_POLARITY),                                
                             .w4_rsc_61_0_PROTOCOL_KIND(w4_rsc_61_0_PROTOCOL_KIND),                                
                             .w4_rsc_62_0_WIDTH(w4_rsc_62_0_WIDTH),                                
                             .w4_rsc_62_0_RESET_POLARITY(w4_rsc_62_0_RESET_POLARITY),                                
                             .w4_rsc_62_0_PROTOCOL_KIND(w4_rsc_62_0_PROTOCOL_KIND),                                
                             .w4_rsc_63_0_WIDTH(w4_rsc_63_0_WIDTH),                                
                             .w4_rsc_63_0_RESET_POLARITY(w4_rsc_63_0_RESET_POLARITY),                                
                             .w4_rsc_63_0_PROTOCOL_KIND(w4_rsc_63_0_PROTOCOL_KIND),                                
                             .b4_rsc_WIDTH(b4_rsc_WIDTH),                                
                             .b4_rsc_RESET_POLARITY(b4_rsc_RESET_POLARITY),                                
                             .b4_rsc_PROTOCOL_KIND(b4_rsc_PROTOCOL_KIND),                                
                             .w6_rsc_0_0_WIDTH(w6_rsc_0_0_WIDTH),                                
                             .w6_rsc_0_0_RESET_POLARITY(w6_rsc_0_0_RESET_POLARITY),                                
                             .w6_rsc_0_0_PROTOCOL_KIND(w6_rsc_0_0_PROTOCOL_KIND),                                
                             .w6_rsc_1_0_WIDTH(w6_rsc_1_0_WIDTH),                                
                             .w6_rsc_1_0_RESET_POLARITY(w6_rsc_1_0_RESET_POLARITY),                                
                             .w6_rsc_1_0_PROTOCOL_KIND(w6_rsc_1_0_PROTOCOL_KIND),                                
                             .w6_rsc_2_0_WIDTH(w6_rsc_2_0_WIDTH),                                
                             .w6_rsc_2_0_RESET_POLARITY(w6_rsc_2_0_RESET_POLARITY),                                
                             .w6_rsc_2_0_PROTOCOL_KIND(w6_rsc_2_0_PROTOCOL_KIND),                                
                             .w6_rsc_3_0_WIDTH(w6_rsc_3_0_WIDTH),                                
                             .w6_rsc_3_0_RESET_POLARITY(w6_rsc_3_0_RESET_POLARITY),                                
                             .w6_rsc_3_0_PROTOCOL_KIND(w6_rsc_3_0_PROTOCOL_KIND),                                
                             .w6_rsc_4_0_WIDTH(w6_rsc_4_0_WIDTH),                                
                             .w6_rsc_4_0_RESET_POLARITY(w6_rsc_4_0_RESET_POLARITY),                                
                             .w6_rsc_4_0_PROTOCOL_KIND(w6_rsc_4_0_PROTOCOL_KIND),                                
                             .w6_rsc_5_0_WIDTH(w6_rsc_5_0_WIDTH),                                
                             .w6_rsc_5_0_RESET_POLARITY(w6_rsc_5_0_RESET_POLARITY),                                
                             .w6_rsc_5_0_PROTOCOL_KIND(w6_rsc_5_0_PROTOCOL_KIND),                                
                             .w6_rsc_6_0_WIDTH(w6_rsc_6_0_WIDTH),                                
                             .w6_rsc_6_0_RESET_POLARITY(w6_rsc_6_0_RESET_POLARITY),                                
                             .w6_rsc_6_0_PROTOCOL_KIND(w6_rsc_6_0_PROTOCOL_KIND),                                
                             .w6_rsc_7_0_WIDTH(w6_rsc_7_0_WIDTH),                                
                             .w6_rsc_7_0_RESET_POLARITY(w6_rsc_7_0_RESET_POLARITY),                                
                             .w6_rsc_7_0_PROTOCOL_KIND(w6_rsc_7_0_PROTOCOL_KIND),                                
                             .w6_rsc_8_0_WIDTH(w6_rsc_8_0_WIDTH),                                
                             .w6_rsc_8_0_RESET_POLARITY(w6_rsc_8_0_RESET_POLARITY),                                
                             .w6_rsc_8_0_PROTOCOL_KIND(w6_rsc_8_0_PROTOCOL_KIND),                                
                             .w6_rsc_9_0_WIDTH(w6_rsc_9_0_WIDTH),                                
                             .w6_rsc_9_0_RESET_POLARITY(w6_rsc_9_0_RESET_POLARITY),                                
                             .w6_rsc_9_0_PROTOCOL_KIND(w6_rsc_9_0_PROTOCOL_KIND),                                
                             .b6_rsc_WIDTH(b6_rsc_WIDTH),                                
                             .b6_rsc_RESET_POLARITY(b6_rsc_RESET_POLARITY),                                
                             .b6_rsc_PROTOCOL_KIND(b6_rsc_PROTOCOL_KIND)                                
                             )  mnist_mlp_pred_t;
  mnist_mlp_pred_t mnist_mlp_pred;

  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))))  output1_rsc_sb_t;
  output1_rsc_sb_t output1_rsc_sb;
  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))))  const_size_in_1_rsc_sb_t;
  const_size_in_1_rsc_sb_t const_size_in_1_rsc_sb;
  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))))  const_size_out_1_rsc_sb_t;
  const_size_out_1_rsc_sb_t const_size_out_1_rsc_sb;


// ****************************************************************************
// FUNCTION : new()
// This function is the standard SystemVerilog constructor.
//
  function new( string name = "", uvm_component parent = null );
    super.new( name, parent );
  endfunction

// ****************************************************************************
// FUNCTION: build_phase()
// This function builds all components within this environment.
//
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    input1_rsc = input1_rsc_agent_t::type_id::create("input1_rsc",this);
    input1_rsc.set_config(configuration.input1_rsc_config);
    output1_rsc = output1_rsc_agent_t::type_id::create("output1_rsc",this);
    output1_rsc.set_config(configuration.output1_rsc_config);
    const_size_in_1_rsc = const_size_in_1_rsc_agent_t::type_id::create("const_size_in_1_rsc",this);
    const_size_in_1_rsc.set_config(configuration.const_size_in_1_rsc_config);
    const_size_out_1_rsc = const_size_out_1_rsc_agent_t::type_id::create("const_size_out_1_rsc",this);
    const_size_out_1_rsc.set_config(configuration.const_size_out_1_rsc_config);
    w2_rsc_0_0 = w2_rsc_0_0_agent_t::type_id::create("w2_rsc_0_0",this);
    w2_rsc_0_0.set_config(configuration.w2_rsc_0_0_config);
    w2_rsc_1_0 = w2_rsc_1_0_agent_t::type_id::create("w2_rsc_1_0",this);
    w2_rsc_1_0.set_config(configuration.w2_rsc_1_0_config);
    w2_rsc_2_0 = w2_rsc_2_0_agent_t::type_id::create("w2_rsc_2_0",this);
    w2_rsc_2_0.set_config(configuration.w2_rsc_2_0_config);
    w2_rsc_3_0 = w2_rsc_3_0_agent_t::type_id::create("w2_rsc_3_0",this);
    w2_rsc_3_0.set_config(configuration.w2_rsc_3_0_config);
    w2_rsc_4_0 = w2_rsc_4_0_agent_t::type_id::create("w2_rsc_4_0",this);
    w2_rsc_4_0.set_config(configuration.w2_rsc_4_0_config);
    w2_rsc_5_0 = w2_rsc_5_0_agent_t::type_id::create("w2_rsc_5_0",this);
    w2_rsc_5_0.set_config(configuration.w2_rsc_5_0_config);
    w2_rsc_6_0 = w2_rsc_6_0_agent_t::type_id::create("w2_rsc_6_0",this);
    w2_rsc_6_0.set_config(configuration.w2_rsc_6_0_config);
    w2_rsc_7_0 = w2_rsc_7_0_agent_t::type_id::create("w2_rsc_7_0",this);
    w2_rsc_7_0.set_config(configuration.w2_rsc_7_0_config);
    w2_rsc_8_0 = w2_rsc_8_0_agent_t::type_id::create("w2_rsc_8_0",this);
    w2_rsc_8_0.set_config(configuration.w2_rsc_8_0_config);
    w2_rsc_9_0 = w2_rsc_9_0_agent_t::type_id::create("w2_rsc_9_0",this);
    w2_rsc_9_0.set_config(configuration.w2_rsc_9_0_config);
    w2_rsc_10_0 = w2_rsc_10_0_agent_t::type_id::create("w2_rsc_10_0",this);
    w2_rsc_10_0.set_config(configuration.w2_rsc_10_0_config);
    w2_rsc_11_0 = w2_rsc_11_0_agent_t::type_id::create("w2_rsc_11_0",this);
    w2_rsc_11_0.set_config(configuration.w2_rsc_11_0_config);
    w2_rsc_12_0 = w2_rsc_12_0_agent_t::type_id::create("w2_rsc_12_0",this);
    w2_rsc_12_0.set_config(configuration.w2_rsc_12_0_config);
    w2_rsc_13_0 = w2_rsc_13_0_agent_t::type_id::create("w2_rsc_13_0",this);
    w2_rsc_13_0.set_config(configuration.w2_rsc_13_0_config);
    w2_rsc_14_0 = w2_rsc_14_0_agent_t::type_id::create("w2_rsc_14_0",this);
    w2_rsc_14_0.set_config(configuration.w2_rsc_14_0_config);
    w2_rsc_15_0 = w2_rsc_15_0_agent_t::type_id::create("w2_rsc_15_0",this);
    w2_rsc_15_0.set_config(configuration.w2_rsc_15_0_config);
    w2_rsc_16_0 = w2_rsc_16_0_agent_t::type_id::create("w2_rsc_16_0",this);
    w2_rsc_16_0.set_config(configuration.w2_rsc_16_0_config);
    w2_rsc_17_0 = w2_rsc_17_0_agent_t::type_id::create("w2_rsc_17_0",this);
    w2_rsc_17_0.set_config(configuration.w2_rsc_17_0_config);
    w2_rsc_18_0 = w2_rsc_18_0_agent_t::type_id::create("w2_rsc_18_0",this);
    w2_rsc_18_0.set_config(configuration.w2_rsc_18_0_config);
    w2_rsc_19_0 = w2_rsc_19_0_agent_t::type_id::create("w2_rsc_19_0",this);
    w2_rsc_19_0.set_config(configuration.w2_rsc_19_0_config);
    w2_rsc_20_0 = w2_rsc_20_0_agent_t::type_id::create("w2_rsc_20_0",this);
    w2_rsc_20_0.set_config(configuration.w2_rsc_20_0_config);
    w2_rsc_21_0 = w2_rsc_21_0_agent_t::type_id::create("w2_rsc_21_0",this);
    w2_rsc_21_0.set_config(configuration.w2_rsc_21_0_config);
    w2_rsc_22_0 = w2_rsc_22_0_agent_t::type_id::create("w2_rsc_22_0",this);
    w2_rsc_22_0.set_config(configuration.w2_rsc_22_0_config);
    w2_rsc_23_0 = w2_rsc_23_0_agent_t::type_id::create("w2_rsc_23_0",this);
    w2_rsc_23_0.set_config(configuration.w2_rsc_23_0_config);
    w2_rsc_24_0 = w2_rsc_24_0_agent_t::type_id::create("w2_rsc_24_0",this);
    w2_rsc_24_0.set_config(configuration.w2_rsc_24_0_config);
    w2_rsc_25_0 = w2_rsc_25_0_agent_t::type_id::create("w2_rsc_25_0",this);
    w2_rsc_25_0.set_config(configuration.w2_rsc_25_0_config);
    w2_rsc_26_0 = w2_rsc_26_0_agent_t::type_id::create("w2_rsc_26_0",this);
    w2_rsc_26_0.set_config(configuration.w2_rsc_26_0_config);
    w2_rsc_27_0 = w2_rsc_27_0_agent_t::type_id::create("w2_rsc_27_0",this);
    w2_rsc_27_0.set_config(configuration.w2_rsc_27_0_config);
    w2_rsc_28_0 = w2_rsc_28_0_agent_t::type_id::create("w2_rsc_28_0",this);
    w2_rsc_28_0.set_config(configuration.w2_rsc_28_0_config);
    w2_rsc_29_0 = w2_rsc_29_0_agent_t::type_id::create("w2_rsc_29_0",this);
    w2_rsc_29_0.set_config(configuration.w2_rsc_29_0_config);
    w2_rsc_30_0 = w2_rsc_30_0_agent_t::type_id::create("w2_rsc_30_0",this);
    w2_rsc_30_0.set_config(configuration.w2_rsc_30_0_config);
    w2_rsc_31_0 = w2_rsc_31_0_agent_t::type_id::create("w2_rsc_31_0",this);
    w2_rsc_31_0.set_config(configuration.w2_rsc_31_0_config);
    w2_rsc_32_0 = w2_rsc_32_0_agent_t::type_id::create("w2_rsc_32_0",this);
    w2_rsc_32_0.set_config(configuration.w2_rsc_32_0_config);
    w2_rsc_33_0 = w2_rsc_33_0_agent_t::type_id::create("w2_rsc_33_0",this);
    w2_rsc_33_0.set_config(configuration.w2_rsc_33_0_config);
    w2_rsc_34_0 = w2_rsc_34_0_agent_t::type_id::create("w2_rsc_34_0",this);
    w2_rsc_34_0.set_config(configuration.w2_rsc_34_0_config);
    w2_rsc_35_0 = w2_rsc_35_0_agent_t::type_id::create("w2_rsc_35_0",this);
    w2_rsc_35_0.set_config(configuration.w2_rsc_35_0_config);
    w2_rsc_36_0 = w2_rsc_36_0_agent_t::type_id::create("w2_rsc_36_0",this);
    w2_rsc_36_0.set_config(configuration.w2_rsc_36_0_config);
    w2_rsc_37_0 = w2_rsc_37_0_agent_t::type_id::create("w2_rsc_37_0",this);
    w2_rsc_37_0.set_config(configuration.w2_rsc_37_0_config);
    w2_rsc_38_0 = w2_rsc_38_0_agent_t::type_id::create("w2_rsc_38_0",this);
    w2_rsc_38_0.set_config(configuration.w2_rsc_38_0_config);
    w2_rsc_39_0 = w2_rsc_39_0_agent_t::type_id::create("w2_rsc_39_0",this);
    w2_rsc_39_0.set_config(configuration.w2_rsc_39_0_config);
    w2_rsc_40_0 = w2_rsc_40_0_agent_t::type_id::create("w2_rsc_40_0",this);
    w2_rsc_40_0.set_config(configuration.w2_rsc_40_0_config);
    w2_rsc_41_0 = w2_rsc_41_0_agent_t::type_id::create("w2_rsc_41_0",this);
    w2_rsc_41_0.set_config(configuration.w2_rsc_41_0_config);
    w2_rsc_42_0 = w2_rsc_42_0_agent_t::type_id::create("w2_rsc_42_0",this);
    w2_rsc_42_0.set_config(configuration.w2_rsc_42_0_config);
    w2_rsc_43_0 = w2_rsc_43_0_agent_t::type_id::create("w2_rsc_43_0",this);
    w2_rsc_43_0.set_config(configuration.w2_rsc_43_0_config);
    w2_rsc_44_0 = w2_rsc_44_0_agent_t::type_id::create("w2_rsc_44_0",this);
    w2_rsc_44_0.set_config(configuration.w2_rsc_44_0_config);
    w2_rsc_45_0 = w2_rsc_45_0_agent_t::type_id::create("w2_rsc_45_0",this);
    w2_rsc_45_0.set_config(configuration.w2_rsc_45_0_config);
    w2_rsc_46_0 = w2_rsc_46_0_agent_t::type_id::create("w2_rsc_46_0",this);
    w2_rsc_46_0.set_config(configuration.w2_rsc_46_0_config);
    w2_rsc_47_0 = w2_rsc_47_0_agent_t::type_id::create("w2_rsc_47_0",this);
    w2_rsc_47_0.set_config(configuration.w2_rsc_47_0_config);
    w2_rsc_48_0 = w2_rsc_48_0_agent_t::type_id::create("w2_rsc_48_0",this);
    w2_rsc_48_0.set_config(configuration.w2_rsc_48_0_config);
    w2_rsc_49_0 = w2_rsc_49_0_agent_t::type_id::create("w2_rsc_49_0",this);
    w2_rsc_49_0.set_config(configuration.w2_rsc_49_0_config);
    w2_rsc_50_0 = w2_rsc_50_0_agent_t::type_id::create("w2_rsc_50_0",this);
    w2_rsc_50_0.set_config(configuration.w2_rsc_50_0_config);
    w2_rsc_51_0 = w2_rsc_51_0_agent_t::type_id::create("w2_rsc_51_0",this);
    w2_rsc_51_0.set_config(configuration.w2_rsc_51_0_config);
    w2_rsc_52_0 = w2_rsc_52_0_agent_t::type_id::create("w2_rsc_52_0",this);
    w2_rsc_52_0.set_config(configuration.w2_rsc_52_0_config);
    w2_rsc_53_0 = w2_rsc_53_0_agent_t::type_id::create("w2_rsc_53_0",this);
    w2_rsc_53_0.set_config(configuration.w2_rsc_53_0_config);
    w2_rsc_54_0 = w2_rsc_54_0_agent_t::type_id::create("w2_rsc_54_0",this);
    w2_rsc_54_0.set_config(configuration.w2_rsc_54_0_config);
    w2_rsc_55_0 = w2_rsc_55_0_agent_t::type_id::create("w2_rsc_55_0",this);
    w2_rsc_55_0.set_config(configuration.w2_rsc_55_0_config);
    w2_rsc_56_0 = w2_rsc_56_0_agent_t::type_id::create("w2_rsc_56_0",this);
    w2_rsc_56_0.set_config(configuration.w2_rsc_56_0_config);
    w2_rsc_57_0 = w2_rsc_57_0_agent_t::type_id::create("w2_rsc_57_0",this);
    w2_rsc_57_0.set_config(configuration.w2_rsc_57_0_config);
    w2_rsc_58_0 = w2_rsc_58_0_agent_t::type_id::create("w2_rsc_58_0",this);
    w2_rsc_58_0.set_config(configuration.w2_rsc_58_0_config);
    w2_rsc_59_0 = w2_rsc_59_0_agent_t::type_id::create("w2_rsc_59_0",this);
    w2_rsc_59_0.set_config(configuration.w2_rsc_59_0_config);
    w2_rsc_60_0 = w2_rsc_60_0_agent_t::type_id::create("w2_rsc_60_0",this);
    w2_rsc_60_0.set_config(configuration.w2_rsc_60_0_config);
    w2_rsc_61_0 = w2_rsc_61_0_agent_t::type_id::create("w2_rsc_61_0",this);
    w2_rsc_61_0.set_config(configuration.w2_rsc_61_0_config);
    w2_rsc_62_0 = w2_rsc_62_0_agent_t::type_id::create("w2_rsc_62_0",this);
    w2_rsc_62_0.set_config(configuration.w2_rsc_62_0_config);
    w2_rsc_63_0 = w2_rsc_63_0_agent_t::type_id::create("w2_rsc_63_0",this);
    w2_rsc_63_0.set_config(configuration.w2_rsc_63_0_config);
    b2_rsc = b2_rsc_agent_t::type_id::create("b2_rsc",this);
    b2_rsc.set_config(configuration.b2_rsc_config);
    w4_rsc_0_0 = w4_rsc_0_0_agent_t::type_id::create("w4_rsc_0_0",this);
    w4_rsc_0_0.set_config(configuration.w4_rsc_0_0_config);
    w4_rsc_1_0 = w4_rsc_1_0_agent_t::type_id::create("w4_rsc_1_0",this);
    w4_rsc_1_0.set_config(configuration.w4_rsc_1_0_config);
    w4_rsc_2_0 = w4_rsc_2_0_agent_t::type_id::create("w4_rsc_2_0",this);
    w4_rsc_2_0.set_config(configuration.w4_rsc_2_0_config);
    w4_rsc_3_0 = w4_rsc_3_0_agent_t::type_id::create("w4_rsc_3_0",this);
    w4_rsc_3_0.set_config(configuration.w4_rsc_3_0_config);
    w4_rsc_4_0 = w4_rsc_4_0_agent_t::type_id::create("w4_rsc_4_0",this);
    w4_rsc_4_0.set_config(configuration.w4_rsc_4_0_config);
    w4_rsc_5_0 = w4_rsc_5_0_agent_t::type_id::create("w4_rsc_5_0",this);
    w4_rsc_5_0.set_config(configuration.w4_rsc_5_0_config);
    w4_rsc_6_0 = w4_rsc_6_0_agent_t::type_id::create("w4_rsc_6_0",this);
    w4_rsc_6_0.set_config(configuration.w4_rsc_6_0_config);
    w4_rsc_7_0 = w4_rsc_7_0_agent_t::type_id::create("w4_rsc_7_0",this);
    w4_rsc_7_0.set_config(configuration.w4_rsc_7_0_config);
    w4_rsc_8_0 = w4_rsc_8_0_agent_t::type_id::create("w4_rsc_8_0",this);
    w4_rsc_8_0.set_config(configuration.w4_rsc_8_0_config);
    w4_rsc_9_0 = w4_rsc_9_0_agent_t::type_id::create("w4_rsc_9_0",this);
    w4_rsc_9_0.set_config(configuration.w4_rsc_9_0_config);
    w4_rsc_10_0 = w4_rsc_10_0_agent_t::type_id::create("w4_rsc_10_0",this);
    w4_rsc_10_0.set_config(configuration.w4_rsc_10_0_config);
    w4_rsc_11_0 = w4_rsc_11_0_agent_t::type_id::create("w4_rsc_11_0",this);
    w4_rsc_11_0.set_config(configuration.w4_rsc_11_0_config);
    w4_rsc_12_0 = w4_rsc_12_0_agent_t::type_id::create("w4_rsc_12_0",this);
    w4_rsc_12_0.set_config(configuration.w4_rsc_12_0_config);
    w4_rsc_13_0 = w4_rsc_13_0_agent_t::type_id::create("w4_rsc_13_0",this);
    w4_rsc_13_0.set_config(configuration.w4_rsc_13_0_config);
    w4_rsc_14_0 = w4_rsc_14_0_agent_t::type_id::create("w4_rsc_14_0",this);
    w4_rsc_14_0.set_config(configuration.w4_rsc_14_0_config);
    w4_rsc_15_0 = w4_rsc_15_0_agent_t::type_id::create("w4_rsc_15_0",this);
    w4_rsc_15_0.set_config(configuration.w4_rsc_15_0_config);
    w4_rsc_16_0 = w4_rsc_16_0_agent_t::type_id::create("w4_rsc_16_0",this);
    w4_rsc_16_0.set_config(configuration.w4_rsc_16_0_config);
    w4_rsc_17_0 = w4_rsc_17_0_agent_t::type_id::create("w4_rsc_17_0",this);
    w4_rsc_17_0.set_config(configuration.w4_rsc_17_0_config);
    w4_rsc_18_0 = w4_rsc_18_0_agent_t::type_id::create("w4_rsc_18_0",this);
    w4_rsc_18_0.set_config(configuration.w4_rsc_18_0_config);
    w4_rsc_19_0 = w4_rsc_19_0_agent_t::type_id::create("w4_rsc_19_0",this);
    w4_rsc_19_0.set_config(configuration.w4_rsc_19_0_config);
    w4_rsc_20_0 = w4_rsc_20_0_agent_t::type_id::create("w4_rsc_20_0",this);
    w4_rsc_20_0.set_config(configuration.w4_rsc_20_0_config);
    w4_rsc_21_0 = w4_rsc_21_0_agent_t::type_id::create("w4_rsc_21_0",this);
    w4_rsc_21_0.set_config(configuration.w4_rsc_21_0_config);
    w4_rsc_22_0 = w4_rsc_22_0_agent_t::type_id::create("w4_rsc_22_0",this);
    w4_rsc_22_0.set_config(configuration.w4_rsc_22_0_config);
    w4_rsc_23_0 = w4_rsc_23_0_agent_t::type_id::create("w4_rsc_23_0",this);
    w4_rsc_23_0.set_config(configuration.w4_rsc_23_0_config);
    w4_rsc_24_0 = w4_rsc_24_0_agent_t::type_id::create("w4_rsc_24_0",this);
    w4_rsc_24_0.set_config(configuration.w4_rsc_24_0_config);
    w4_rsc_25_0 = w4_rsc_25_0_agent_t::type_id::create("w4_rsc_25_0",this);
    w4_rsc_25_0.set_config(configuration.w4_rsc_25_0_config);
    w4_rsc_26_0 = w4_rsc_26_0_agent_t::type_id::create("w4_rsc_26_0",this);
    w4_rsc_26_0.set_config(configuration.w4_rsc_26_0_config);
    w4_rsc_27_0 = w4_rsc_27_0_agent_t::type_id::create("w4_rsc_27_0",this);
    w4_rsc_27_0.set_config(configuration.w4_rsc_27_0_config);
    w4_rsc_28_0 = w4_rsc_28_0_agent_t::type_id::create("w4_rsc_28_0",this);
    w4_rsc_28_0.set_config(configuration.w4_rsc_28_0_config);
    w4_rsc_29_0 = w4_rsc_29_0_agent_t::type_id::create("w4_rsc_29_0",this);
    w4_rsc_29_0.set_config(configuration.w4_rsc_29_0_config);
    w4_rsc_30_0 = w4_rsc_30_0_agent_t::type_id::create("w4_rsc_30_0",this);
    w4_rsc_30_0.set_config(configuration.w4_rsc_30_0_config);
    w4_rsc_31_0 = w4_rsc_31_0_agent_t::type_id::create("w4_rsc_31_0",this);
    w4_rsc_31_0.set_config(configuration.w4_rsc_31_0_config);
    w4_rsc_32_0 = w4_rsc_32_0_agent_t::type_id::create("w4_rsc_32_0",this);
    w4_rsc_32_0.set_config(configuration.w4_rsc_32_0_config);
    w4_rsc_33_0 = w4_rsc_33_0_agent_t::type_id::create("w4_rsc_33_0",this);
    w4_rsc_33_0.set_config(configuration.w4_rsc_33_0_config);
    w4_rsc_34_0 = w4_rsc_34_0_agent_t::type_id::create("w4_rsc_34_0",this);
    w4_rsc_34_0.set_config(configuration.w4_rsc_34_0_config);
    w4_rsc_35_0 = w4_rsc_35_0_agent_t::type_id::create("w4_rsc_35_0",this);
    w4_rsc_35_0.set_config(configuration.w4_rsc_35_0_config);
    w4_rsc_36_0 = w4_rsc_36_0_agent_t::type_id::create("w4_rsc_36_0",this);
    w4_rsc_36_0.set_config(configuration.w4_rsc_36_0_config);
    w4_rsc_37_0 = w4_rsc_37_0_agent_t::type_id::create("w4_rsc_37_0",this);
    w4_rsc_37_0.set_config(configuration.w4_rsc_37_0_config);
    w4_rsc_38_0 = w4_rsc_38_0_agent_t::type_id::create("w4_rsc_38_0",this);
    w4_rsc_38_0.set_config(configuration.w4_rsc_38_0_config);
    w4_rsc_39_0 = w4_rsc_39_0_agent_t::type_id::create("w4_rsc_39_0",this);
    w4_rsc_39_0.set_config(configuration.w4_rsc_39_0_config);
    w4_rsc_40_0 = w4_rsc_40_0_agent_t::type_id::create("w4_rsc_40_0",this);
    w4_rsc_40_0.set_config(configuration.w4_rsc_40_0_config);
    w4_rsc_41_0 = w4_rsc_41_0_agent_t::type_id::create("w4_rsc_41_0",this);
    w4_rsc_41_0.set_config(configuration.w4_rsc_41_0_config);
    w4_rsc_42_0 = w4_rsc_42_0_agent_t::type_id::create("w4_rsc_42_0",this);
    w4_rsc_42_0.set_config(configuration.w4_rsc_42_0_config);
    w4_rsc_43_0 = w4_rsc_43_0_agent_t::type_id::create("w4_rsc_43_0",this);
    w4_rsc_43_0.set_config(configuration.w4_rsc_43_0_config);
    w4_rsc_44_0 = w4_rsc_44_0_agent_t::type_id::create("w4_rsc_44_0",this);
    w4_rsc_44_0.set_config(configuration.w4_rsc_44_0_config);
    w4_rsc_45_0 = w4_rsc_45_0_agent_t::type_id::create("w4_rsc_45_0",this);
    w4_rsc_45_0.set_config(configuration.w4_rsc_45_0_config);
    w4_rsc_46_0 = w4_rsc_46_0_agent_t::type_id::create("w4_rsc_46_0",this);
    w4_rsc_46_0.set_config(configuration.w4_rsc_46_0_config);
    w4_rsc_47_0 = w4_rsc_47_0_agent_t::type_id::create("w4_rsc_47_0",this);
    w4_rsc_47_0.set_config(configuration.w4_rsc_47_0_config);
    w4_rsc_48_0 = w4_rsc_48_0_agent_t::type_id::create("w4_rsc_48_0",this);
    w4_rsc_48_0.set_config(configuration.w4_rsc_48_0_config);
    w4_rsc_49_0 = w4_rsc_49_0_agent_t::type_id::create("w4_rsc_49_0",this);
    w4_rsc_49_0.set_config(configuration.w4_rsc_49_0_config);
    w4_rsc_50_0 = w4_rsc_50_0_agent_t::type_id::create("w4_rsc_50_0",this);
    w4_rsc_50_0.set_config(configuration.w4_rsc_50_0_config);
    w4_rsc_51_0 = w4_rsc_51_0_agent_t::type_id::create("w4_rsc_51_0",this);
    w4_rsc_51_0.set_config(configuration.w4_rsc_51_0_config);
    w4_rsc_52_0 = w4_rsc_52_0_agent_t::type_id::create("w4_rsc_52_0",this);
    w4_rsc_52_0.set_config(configuration.w4_rsc_52_0_config);
    w4_rsc_53_0 = w4_rsc_53_0_agent_t::type_id::create("w4_rsc_53_0",this);
    w4_rsc_53_0.set_config(configuration.w4_rsc_53_0_config);
    w4_rsc_54_0 = w4_rsc_54_0_agent_t::type_id::create("w4_rsc_54_0",this);
    w4_rsc_54_0.set_config(configuration.w4_rsc_54_0_config);
    w4_rsc_55_0 = w4_rsc_55_0_agent_t::type_id::create("w4_rsc_55_0",this);
    w4_rsc_55_0.set_config(configuration.w4_rsc_55_0_config);
    w4_rsc_56_0 = w4_rsc_56_0_agent_t::type_id::create("w4_rsc_56_0",this);
    w4_rsc_56_0.set_config(configuration.w4_rsc_56_0_config);
    w4_rsc_57_0 = w4_rsc_57_0_agent_t::type_id::create("w4_rsc_57_0",this);
    w4_rsc_57_0.set_config(configuration.w4_rsc_57_0_config);
    w4_rsc_58_0 = w4_rsc_58_0_agent_t::type_id::create("w4_rsc_58_0",this);
    w4_rsc_58_0.set_config(configuration.w4_rsc_58_0_config);
    w4_rsc_59_0 = w4_rsc_59_0_agent_t::type_id::create("w4_rsc_59_0",this);
    w4_rsc_59_0.set_config(configuration.w4_rsc_59_0_config);
    w4_rsc_60_0 = w4_rsc_60_0_agent_t::type_id::create("w4_rsc_60_0",this);
    w4_rsc_60_0.set_config(configuration.w4_rsc_60_0_config);
    w4_rsc_61_0 = w4_rsc_61_0_agent_t::type_id::create("w4_rsc_61_0",this);
    w4_rsc_61_0.set_config(configuration.w4_rsc_61_0_config);
    w4_rsc_62_0 = w4_rsc_62_0_agent_t::type_id::create("w4_rsc_62_0",this);
    w4_rsc_62_0.set_config(configuration.w4_rsc_62_0_config);
    w4_rsc_63_0 = w4_rsc_63_0_agent_t::type_id::create("w4_rsc_63_0",this);
    w4_rsc_63_0.set_config(configuration.w4_rsc_63_0_config);
    b4_rsc = b4_rsc_agent_t::type_id::create("b4_rsc",this);
    b4_rsc.set_config(configuration.b4_rsc_config);
    w6_rsc_0_0 = w6_rsc_0_0_agent_t::type_id::create("w6_rsc_0_0",this);
    w6_rsc_0_0.set_config(configuration.w6_rsc_0_0_config);
    w6_rsc_1_0 = w6_rsc_1_0_agent_t::type_id::create("w6_rsc_1_0",this);
    w6_rsc_1_0.set_config(configuration.w6_rsc_1_0_config);
    w6_rsc_2_0 = w6_rsc_2_0_agent_t::type_id::create("w6_rsc_2_0",this);
    w6_rsc_2_0.set_config(configuration.w6_rsc_2_0_config);
    w6_rsc_3_0 = w6_rsc_3_0_agent_t::type_id::create("w6_rsc_3_0",this);
    w6_rsc_3_0.set_config(configuration.w6_rsc_3_0_config);
    w6_rsc_4_0 = w6_rsc_4_0_agent_t::type_id::create("w6_rsc_4_0",this);
    w6_rsc_4_0.set_config(configuration.w6_rsc_4_0_config);
    w6_rsc_5_0 = w6_rsc_5_0_agent_t::type_id::create("w6_rsc_5_0",this);
    w6_rsc_5_0.set_config(configuration.w6_rsc_5_0_config);
    w6_rsc_6_0 = w6_rsc_6_0_agent_t::type_id::create("w6_rsc_6_0",this);
    w6_rsc_6_0.set_config(configuration.w6_rsc_6_0_config);
    w6_rsc_7_0 = w6_rsc_7_0_agent_t::type_id::create("w6_rsc_7_0",this);
    w6_rsc_7_0.set_config(configuration.w6_rsc_7_0_config);
    w6_rsc_8_0 = w6_rsc_8_0_agent_t::type_id::create("w6_rsc_8_0",this);
    w6_rsc_8_0.set_config(configuration.w6_rsc_8_0_config);
    w6_rsc_9_0 = w6_rsc_9_0_agent_t::type_id::create("w6_rsc_9_0",this);
    w6_rsc_9_0.set_config(configuration.w6_rsc_9_0_config);
    b6_rsc = b6_rsc_agent_t::type_id::create("b6_rsc",this);
    b6_rsc.set_config(configuration.b6_rsc_config);
    mnist_mlp_pred = mnist_mlp_pred_t::type_id::create("mnist_mlp_pred",this);
    mnist_mlp_pred.configuration = configuration;
    output1_rsc_sb = output1_rsc_sb_t::type_id::create("output1_rsc_sb",this);
    const_size_in_1_rsc_sb = const_size_in_1_rsc_sb_t::type_id::create("const_size_in_1_rsc_sb",this);
    const_size_out_1_rsc_sb = const_size_out_1_rsc_sb_t::type_id::create("const_size_out_1_rsc_sb",this);
  endfunction

// ****************************************************************************
// FUNCTION: connect_phase()
// This function makes all connections within this environment.  Connections
// typically inclue agent to predictor, predictor to scoreboard and scoreboard
// to agent.
//
  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    input1_rsc.monitored_ap.connect(mnist_mlp_pred.input1_rsc_ae);
    w2_rsc_0_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_0_0_ae);
    w2_rsc_1_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_1_0_ae);
    w2_rsc_2_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_2_0_ae);
    w2_rsc_3_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_3_0_ae);
    w2_rsc_4_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_4_0_ae);
    w2_rsc_5_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_5_0_ae);
    w2_rsc_6_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_6_0_ae);
    w2_rsc_7_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_7_0_ae);
    w2_rsc_8_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_8_0_ae);
    w2_rsc_9_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_9_0_ae);
    w2_rsc_10_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_10_0_ae);
    w2_rsc_11_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_11_0_ae);
    w2_rsc_12_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_12_0_ae);
    w2_rsc_13_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_13_0_ae);
    w2_rsc_14_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_14_0_ae);
    w2_rsc_15_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_15_0_ae);
    w2_rsc_16_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_16_0_ae);
    w2_rsc_17_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_17_0_ae);
    w2_rsc_18_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_18_0_ae);
    w2_rsc_19_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_19_0_ae);
    w2_rsc_20_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_20_0_ae);
    w2_rsc_21_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_21_0_ae);
    w2_rsc_22_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_22_0_ae);
    w2_rsc_23_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_23_0_ae);
    w2_rsc_24_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_24_0_ae);
    w2_rsc_25_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_25_0_ae);
    w2_rsc_26_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_26_0_ae);
    w2_rsc_27_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_27_0_ae);
    w2_rsc_28_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_28_0_ae);
    w2_rsc_29_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_29_0_ae);
    w2_rsc_30_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_30_0_ae);
    w2_rsc_31_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_31_0_ae);
    w2_rsc_32_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_32_0_ae);
    w2_rsc_33_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_33_0_ae);
    w2_rsc_34_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_34_0_ae);
    w2_rsc_35_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_35_0_ae);
    w2_rsc_36_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_36_0_ae);
    w2_rsc_37_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_37_0_ae);
    w2_rsc_38_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_38_0_ae);
    w2_rsc_39_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_39_0_ae);
    w2_rsc_40_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_40_0_ae);
    w2_rsc_41_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_41_0_ae);
    w2_rsc_42_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_42_0_ae);
    w2_rsc_43_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_43_0_ae);
    w2_rsc_44_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_44_0_ae);
    w2_rsc_45_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_45_0_ae);
    w2_rsc_46_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_46_0_ae);
    w2_rsc_47_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_47_0_ae);
    w2_rsc_48_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_48_0_ae);
    w2_rsc_49_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_49_0_ae);
    w2_rsc_50_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_50_0_ae);
    w2_rsc_51_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_51_0_ae);
    w2_rsc_52_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_52_0_ae);
    w2_rsc_53_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_53_0_ae);
    w2_rsc_54_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_54_0_ae);
    w2_rsc_55_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_55_0_ae);
    w2_rsc_56_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_56_0_ae);
    w2_rsc_57_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_57_0_ae);
    w2_rsc_58_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_58_0_ae);
    w2_rsc_59_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_59_0_ae);
    w2_rsc_60_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_60_0_ae);
    w2_rsc_61_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_61_0_ae);
    w2_rsc_62_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_62_0_ae);
    w2_rsc_63_0.monitored_ap.connect(mnist_mlp_pred.w2_rsc_63_0_ae);
    b2_rsc.monitored_ap.connect(mnist_mlp_pred.b2_rsc_ae);
    w4_rsc_0_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_0_0_ae);
    w4_rsc_1_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_1_0_ae);
    w4_rsc_2_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_2_0_ae);
    w4_rsc_3_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_3_0_ae);
    w4_rsc_4_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_4_0_ae);
    w4_rsc_5_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_5_0_ae);
    w4_rsc_6_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_6_0_ae);
    w4_rsc_7_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_7_0_ae);
    w4_rsc_8_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_8_0_ae);
    w4_rsc_9_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_9_0_ae);
    w4_rsc_10_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_10_0_ae);
    w4_rsc_11_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_11_0_ae);
    w4_rsc_12_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_12_0_ae);
    w4_rsc_13_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_13_0_ae);
    w4_rsc_14_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_14_0_ae);
    w4_rsc_15_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_15_0_ae);
    w4_rsc_16_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_16_0_ae);
    w4_rsc_17_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_17_0_ae);
    w4_rsc_18_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_18_0_ae);
    w4_rsc_19_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_19_0_ae);
    w4_rsc_20_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_20_0_ae);
    w4_rsc_21_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_21_0_ae);
    w4_rsc_22_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_22_0_ae);
    w4_rsc_23_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_23_0_ae);
    w4_rsc_24_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_24_0_ae);
    w4_rsc_25_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_25_0_ae);
    w4_rsc_26_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_26_0_ae);
    w4_rsc_27_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_27_0_ae);
    w4_rsc_28_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_28_0_ae);
    w4_rsc_29_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_29_0_ae);
    w4_rsc_30_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_30_0_ae);
    w4_rsc_31_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_31_0_ae);
    w4_rsc_32_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_32_0_ae);
    w4_rsc_33_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_33_0_ae);
    w4_rsc_34_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_34_0_ae);
    w4_rsc_35_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_35_0_ae);
    w4_rsc_36_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_36_0_ae);
    w4_rsc_37_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_37_0_ae);
    w4_rsc_38_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_38_0_ae);
    w4_rsc_39_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_39_0_ae);
    w4_rsc_40_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_40_0_ae);
    w4_rsc_41_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_41_0_ae);
    w4_rsc_42_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_42_0_ae);
    w4_rsc_43_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_43_0_ae);
    w4_rsc_44_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_44_0_ae);
    w4_rsc_45_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_45_0_ae);
    w4_rsc_46_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_46_0_ae);
    w4_rsc_47_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_47_0_ae);
    w4_rsc_48_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_48_0_ae);
    w4_rsc_49_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_49_0_ae);
    w4_rsc_50_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_50_0_ae);
    w4_rsc_51_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_51_0_ae);
    w4_rsc_52_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_52_0_ae);
    w4_rsc_53_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_53_0_ae);
    w4_rsc_54_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_54_0_ae);
    w4_rsc_55_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_55_0_ae);
    w4_rsc_56_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_56_0_ae);
    w4_rsc_57_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_57_0_ae);
    w4_rsc_58_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_58_0_ae);
    w4_rsc_59_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_59_0_ae);
    w4_rsc_60_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_60_0_ae);
    w4_rsc_61_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_61_0_ae);
    w4_rsc_62_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_62_0_ae);
    w4_rsc_63_0.monitored_ap.connect(mnist_mlp_pred.w4_rsc_63_0_ae);
    b4_rsc.monitored_ap.connect(mnist_mlp_pred.b4_rsc_ae);
    w6_rsc_0_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_0_0_ae);
    w6_rsc_1_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_1_0_ae);
    w6_rsc_2_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_2_0_ae);
    w6_rsc_3_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_3_0_ae);
    w6_rsc_4_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_4_0_ae);
    w6_rsc_5_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_5_0_ae);
    w6_rsc_6_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_6_0_ae);
    w6_rsc_7_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_7_0_ae);
    w6_rsc_8_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_8_0_ae);
    w6_rsc_9_0.monitored_ap.connect(mnist_mlp_pred.w6_rsc_9_0_ae);
    b6_rsc.monitored_ap.connect(mnist_mlp_pred.b6_rsc_ae);
    output1_rsc.monitored_ap.connect(output1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.output1_rsc_ap.connect(output1_rsc_sb.expected_analysis_export);
    const_size_in_1_rsc.monitored_ap.connect(const_size_in_1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.const_size_in_1_rsc_ap.connect(const_size_in_1_rsc_sb.expected_analysis_export);
    const_size_out_1_rsc.monitored_ap.connect(const_size_out_1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.const_size_out_1_rsc_ap.connect(const_size_out_1_rsc_sb.expected_analysis_export);
  endfunction

endclass

