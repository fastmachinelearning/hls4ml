//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp Environment 
// Unit            : Environment configuration
// File            : mnist_mlp_env_configuration.svh
//----------------------------------------------------------------------
//                                          
// DESCRIPTION: THis is the configuration for the mnist_mlp environment.
//  it contains configuration classes for each agent.  It also contains
//  environment level configuration variables.
//
//
//
//----------------------------------------------------------------------
//
class mnist_mlp_env_configuration 
            #(
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
             bit[1:0] b6_rsc_PROTOCOL_KIND = 0                                
             )
extends uvmf_environment_configuration_base;

  `uvm_object_param_utils( mnist_mlp_env_configuration #(
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


//Constraints for the configuration variables:


  covergroup mnist_mlp_configuration_cg;
    option.auto_bin_max=1024;
  endgroup


    typedef ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1)) input1_rsc_config_t;
    input1_rsc_config_t input1_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1)) output1_rsc_config_t;
    output1_rsc_config_t output1_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_in_1_rsc_config_t;
    const_size_in_1_rsc_config_t const_size_in_1_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_out_1_rsc_config_t;
    const_size_out_1_rsc_config_t const_size_out_1_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_0_0_config_t;
    w2_rsc_0_0_config_t w2_rsc_0_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_1_0_config_t;
    w2_rsc_1_0_config_t w2_rsc_1_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_2_0_config_t;
    w2_rsc_2_0_config_t w2_rsc_2_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_3_0_config_t;
    w2_rsc_3_0_config_t w2_rsc_3_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_4_0_config_t;
    w2_rsc_4_0_config_t w2_rsc_4_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_5_0_config_t;
    w2_rsc_5_0_config_t w2_rsc_5_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_6_0_config_t;
    w2_rsc_6_0_config_t w2_rsc_6_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_7_0_config_t;
    w2_rsc_7_0_config_t w2_rsc_7_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_8_0_config_t;
    w2_rsc_8_0_config_t w2_rsc_8_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_9_0_config_t;
    w2_rsc_9_0_config_t w2_rsc_9_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_10_0_config_t;
    w2_rsc_10_0_config_t w2_rsc_10_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_11_0_config_t;
    w2_rsc_11_0_config_t w2_rsc_11_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_12_0_config_t;
    w2_rsc_12_0_config_t w2_rsc_12_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_13_0_config_t;
    w2_rsc_13_0_config_t w2_rsc_13_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_14_0_config_t;
    w2_rsc_14_0_config_t w2_rsc_14_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_15_0_config_t;
    w2_rsc_15_0_config_t w2_rsc_15_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_16_0_config_t;
    w2_rsc_16_0_config_t w2_rsc_16_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_17_0_config_t;
    w2_rsc_17_0_config_t w2_rsc_17_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_18_0_config_t;
    w2_rsc_18_0_config_t w2_rsc_18_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_19_0_config_t;
    w2_rsc_19_0_config_t w2_rsc_19_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_20_0_config_t;
    w2_rsc_20_0_config_t w2_rsc_20_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_21_0_config_t;
    w2_rsc_21_0_config_t w2_rsc_21_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_22_0_config_t;
    w2_rsc_22_0_config_t w2_rsc_22_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_23_0_config_t;
    w2_rsc_23_0_config_t w2_rsc_23_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_24_0_config_t;
    w2_rsc_24_0_config_t w2_rsc_24_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_25_0_config_t;
    w2_rsc_25_0_config_t w2_rsc_25_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_26_0_config_t;
    w2_rsc_26_0_config_t w2_rsc_26_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_27_0_config_t;
    w2_rsc_27_0_config_t w2_rsc_27_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_28_0_config_t;
    w2_rsc_28_0_config_t w2_rsc_28_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_29_0_config_t;
    w2_rsc_29_0_config_t w2_rsc_29_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_30_0_config_t;
    w2_rsc_30_0_config_t w2_rsc_30_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_31_0_config_t;
    w2_rsc_31_0_config_t w2_rsc_31_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_32_0_config_t;
    w2_rsc_32_0_config_t w2_rsc_32_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_33_0_config_t;
    w2_rsc_33_0_config_t w2_rsc_33_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_34_0_config_t;
    w2_rsc_34_0_config_t w2_rsc_34_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_35_0_config_t;
    w2_rsc_35_0_config_t w2_rsc_35_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_36_0_config_t;
    w2_rsc_36_0_config_t w2_rsc_36_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_37_0_config_t;
    w2_rsc_37_0_config_t w2_rsc_37_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_38_0_config_t;
    w2_rsc_38_0_config_t w2_rsc_38_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_39_0_config_t;
    w2_rsc_39_0_config_t w2_rsc_39_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_40_0_config_t;
    w2_rsc_40_0_config_t w2_rsc_40_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_41_0_config_t;
    w2_rsc_41_0_config_t w2_rsc_41_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_42_0_config_t;
    w2_rsc_42_0_config_t w2_rsc_42_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_43_0_config_t;
    w2_rsc_43_0_config_t w2_rsc_43_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_44_0_config_t;
    w2_rsc_44_0_config_t w2_rsc_44_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_45_0_config_t;
    w2_rsc_45_0_config_t w2_rsc_45_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_46_0_config_t;
    w2_rsc_46_0_config_t w2_rsc_46_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_47_0_config_t;
    w2_rsc_47_0_config_t w2_rsc_47_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_48_0_config_t;
    w2_rsc_48_0_config_t w2_rsc_48_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_49_0_config_t;
    w2_rsc_49_0_config_t w2_rsc_49_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_50_0_config_t;
    w2_rsc_50_0_config_t w2_rsc_50_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_51_0_config_t;
    w2_rsc_51_0_config_t w2_rsc_51_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_52_0_config_t;
    w2_rsc_52_0_config_t w2_rsc_52_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_53_0_config_t;
    w2_rsc_53_0_config_t w2_rsc_53_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_54_0_config_t;
    w2_rsc_54_0_config_t w2_rsc_54_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_55_0_config_t;
    w2_rsc_55_0_config_t w2_rsc_55_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_56_0_config_t;
    w2_rsc_56_0_config_t w2_rsc_56_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_57_0_config_t;
    w2_rsc_57_0_config_t w2_rsc_57_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_58_0_config_t;
    w2_rsc_58_0_config_t w2_rsc_58_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_59_0_config_t;
    w2_rsc_59_0_config_t w2_rsc_59_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_60_0_config_t;
    w2_rsc_60_0_config_t w2_rsc_60_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_61_0_config_t;
    w2_rsc_61_0_config_t w2_rsc_61_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_62_0_config_t;
    w2_rsc_62_0_config_t w2_rsc_62_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(14112),.RESET_POLARITY(1)) w2_rsc_63_0_config_t;
    w2_rsc_63_0_config_t w2_rsc_63_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) b2_rsc_config_t;
    b2_rsc_config_t b2_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_0_0_config_t;
    w4_rsc_0_0_config_t w4_rsc_0_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_1_0_config_t;
    w4_rsc_1_0_config_t w4_rsc_1_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_2_0_config_t;
    w4_rsc_2_0_config_t w4_rsc_2_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_3_0_config_t;
    w4_rsc_3_0_config_t w4_rsc_3_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_4_0_config_t;
    w4_rsc_4_0_config_t w4_rsc_4_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_5_0_config_t;
    w4_rsc_5_0_config_t w4_rsc_5_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_6_0_config_t;
    w4_rsc_6_0_config_t w4_rsc_6_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_7_0_config_t;
    w4_rsc_7_0_config_t w4_rsc_7_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_8_0_config_t;
    w4_rsc_8_0_config_t w4_rsc_8_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_9_0_config_t;
    w4_rsc_9_0_config_t w4_rsc_9_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_10_0_config_t;
    w4_rsc_10_0_config_t w4_rsc_10_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_11_0_config_t;
    w4_rsc_11_0_config_t w4_rsc_11_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_12_0_config_t;
    w4_rsc_12_0_config_t w4_rsc_12_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_13_0_config_t;
    w4_rsc_13_0_config_t w4_rsc_13_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_14_0_config_t;
    w4_rsc_14_0_config_t w4_rsc_14_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_15_0_config_t;
    w4_rsc_15_0_config_t w4_rsc_15_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_16_0_config_t;
    w4_rsc_16_0_config_t w4_rsc_16_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_17_0_config_t;
    w4_rsc_17_0_config_t w4_rsc_17_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_18_0_config_t;
    w4_rsc_18_0_config_t w4_rsc_18_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_19_0_config_t;
    w4_rsc_19_0_config_t w4_rsc_19_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_20_0_config_t;
    w4_rsc_20_0_config_t w4_rsc_20_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_21_0_config_t;
    w4_rsc_21_0_config_t w4_rsc_21_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_22_0_config_t;
    w4_rsc_22_0_config_t w4_rsc_22_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_23_0_config_t;
    w4_rsc_23_0_config_t w4_rsc_23_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_24_0_config_t;
    w4_rsc_24_0_config_t w4_rsc_24_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_25_0_config_t;
    w4_rsc_25_0_config_t w4_rsc_25_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_26_0_config_t;
    w4_rsc_26_0_config_t w4_rsc_26_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_27_0_config_t;
    w4_rsc_27_0_config_t w4_rsc_27_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_28_0_config_t;
    w4_rsc_28_0_config_t w4_rsc_28_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_29_0_config_t;
    w4_rsc_29_0_config_t w4_rsc_29_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_30_0_config_t;
    w4_rsc_30_0_config_t w4_rsc_30_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_31_0_config_t;
    w4_rsc_31_0_config_t w4_rsc_31_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_32_0_config_t;
    w4_rsc_32_0_config_t w4_rsc_32_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_33_0_config_t;
    w4_rsc_33_0_config_t w4_rsc_33_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_34_0_config_t;
    w4_rsc_34_0_config_t w4_rsc_34_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_35_0_config_t;
    w4_rsc_35_0_config_t w4_rsc_35_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_36_0_config_t;
    w4_rsc_36_0_config_t w4_rsc_36_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_37_0_config_t;
    w4_rsc_37_0_config_t w4_rsc_37_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_38_0_config_t;
    w4_rsc_38_0_config_t w4_rsc_38_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_39_0_config_t;
    w4_rsc_39_0_config_t w4_rsc_39_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_40_0_config_t;
    w4_rsc_40_0_config_t w4_rsc_40_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_41_0_config_t;
    w4_rsc_41_0_config_t w4_rsc_41_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_42_0_config_t;
    w4_rsc_42_0_config_t w4_rsc_42_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_43_0_config_t;
    w4_rsc_43_0_config_t w4_rsc_43_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_44_0_config_t;
    w4_rsc_44_0_config_t w4_rsc_44_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_45_0_config_t;
    w4_rsc_45_0_config_t w4_rsc_45_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_46_0_config_t;
    w4_rsc_46_0_config_t w4_rsc_46_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_47_0_config_t;
    w4_rsc_47_0_config_t w4_rsc_47_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_48_0_config_t;
    w4_rsc_48_0_config_t w4_rsc_48_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_49_0_config_t;
    w4_rsc_49_0_config_t w4_rsc_49_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_50_0_config_t;
    w4_rsc_50_0_config_t w4_rsc_50_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_51_0_config_t;
    w4_rsc_51_0_config_t w4_rsc_51_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_52_0_config_t;
    w4_rsc_52_0_config_t w4_rsc_52_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_53_0_config_t;
    w4_rsc_53_0_config_t w4_rsc_53_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_54_0_config_t;
    w4_rsc_54_0_config_t w4_rsc_54_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_55_0_config_t;
    w4_rsc_55_0_config_t w4_rsc_55_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_56_0_config_t;
    w4_rsc_56_0_config_t w4_rsc_56_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_57_0_config_t;
    w4_rsc_57_0_config_t w4_rsc_57_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_58_0_config_t;
    w4_rsc_58_0_config_t w4_rsc_58_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_59_0_config_t;
    w4_rsc_59_0_config_t w4_rsc_59_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_60_0_config_t;
    w4_rsc_60_0_config_t w4_rsc_60_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_61_0_config_t;
    w4_rsc_61_0_config_t w4_rsc_61_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_62_0_config_t;
    w4_rsc_62_0_config_t w4_rsc_62_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w4_rsc_63_0_config_t;
    w4_rsc_63_0_config_t w4_rsc_63_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) b4_rsc_config_t;
    b4_rsc_config_t b4_rsc_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_0_0_config_t;
    w6_rsc_0_0_config_t w6_rsc_0_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_1_0_config_t;
    w6_rsc_1_0_config_t w6_rsc_1_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_2_0_config_t;
    w6_rsc_2_0_config_t w6_rsc_2_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_3_0_config_t;
    w6_rsc_3_0_config_t w6_rsc_3_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_4_0_config_t;
    w6_rsc_4_0_config_t w6_rsc_4_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_5_0_config_t;
    w6_rsc_5_0_config_t w6_rsc_5_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_6_0_config_t;
    w6_rsc_6_0_config_t w6_rsc_6_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_7_0_config_t;
    w6_rsc_7_0_config_t w6_rsc_7_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_8_0_config_t;
    w6_rsc_8_0_config_t w6_rsc_8_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(1152),.RESET_POLARITY(1)) w6_rsc_9_0_config_t;
    w6_rsc_9_0_config_t w6_rsc_9_0_config;

    typedef ccs_configuration #(.PROTOCOL_KIND(0),.WIDTH(180),.RESET_POLARITY(1)) b6_rsc_config_t;
    b6_rsc_config_t b6_rsc_config;





// ****************************************************************************
// FUNCTION : new()
// This function is the standard SystemVerilog constructor.
// This function constructs the configuration object for each agent in the environment.
//
  function new( string name = "" );
    super.new( name );


    input1_rsc_config = input1_rsc_config_t::type_id::create("input1_rsc_config");
    output1_rsc_config = output1_rsc_config_t::type_id::create("output1_rsc_config");
    const_size_in_1_rsc_config = const_size_in_1_rsc_config_t::type_id::create("const_size_in_1_rsc_config");
    const_size_out_1_rsc_config = const_size_out_1_rsc_config_t::type_id::create("const_size_out_1_rsc_config");
    w2_rsc_0_0_config = w2_rsc_0_0_config_t::type_id::create("w2_rsc_0_0_config");
    w2_rsc_1_0_config = w2_rsc_1_0_config_t::type_id::create("w2_rsc_1_0_config");
    w2_rsc_2_0_config = w2_rsc_2_0_config_t::type_id::create("w2_rsc_2_0_config");
    w2_rsc_3_0_config = w2_rsc_3_0_config_t::type_id::create("w2_rsc_3_0_config");
    w2_rsc_4_0_config = w2_rsc_4_0_config_t::type_id::create("w2_rsc_4_0_config");
    w2_rsc_5_0_config = w2_rsc_5_0_config_t::type_id::create("w2_rsc_5_0_config");
    w2_rsc_6_0_config = w2_rsc_6_0_config_t::type_id::create("w2_rsc_6_0_config");
    w2_rsc_7_0_config = w2_rsc_7_0_config_t::type_id::create("w2_rsc_7_0_config");
    w2_rsc_8_0_config = w2_rsc_8_0_config_t::type_id::create("w2_rsc_8_0_config");
    w2_rsc_9_0_config = w2_rsc_9_0_config_t::type_id::create("w2_rsc_9_0_config");
    w2_rsc_10_0_config = w2_rsc_10_0_config_t::type_id::create("w2_rsc_10_0_config");
    w2_rsc_11_0_config = w2_rsc_11_0_config_t::type_id::create("w2_rsc_11_0_config");
    w2_rsc_12_0_config = w2_rsc_12_0_config_t::type_id::create("w2_rsc_12_0_config");
    w2_rsc_13_0_config = w2_rsc_13_0_config_t::type_id::create("w2_rsc_13_0_config");
    w2_rsc_14_0_config = w2_rsc_14_0_config_t::type_id::create("w2_rsc_14_0_config");
    w2_rsc_15_0_config = w2_rsc_15_0_config_t::type_id::create("w2_rsc_15_0_config");
    w2_rsc_16_0_config = w2_rsc_16_0_config_t::type_id::create("w2_rsc_16_0_config");
    w2_rsc_17_0_config = w2_rsc_17_0_config_t::type_id::create("w2_rsc_17_0_config");
    w2_rsc_18_0_config = w2_rsc_18_0_config_t::type_id::create("w2_rsc_18_0_config");
    w2_rsc_19_0_config = w2_rsc_19_0_config_t::type_id::create("w2_rsc_19_0_config");
    w2_rsc_20_0_config = w2_rsc_20_0_config_t::type_id::create("w2_rsc_20_0_config");
    w2_rsc_21_0_config = w2_rsc_21_0_config_t::type_id::create("w2_rsc_21_0_config");
    w2_rsc_22_0_config = w2_rsc_22_0_config_t::type_id::create("w2_rsc_22_0_config");
    w2_rsc_23_0_config = w2_rsc_23_0_config_t::type_id::create("w2_rsc_23_0_config");
    w2_rsc_24_0_config = w2_rsc_24_0_config_t::type_id::create("w2_rsc_24_0_config");
    w2_rsc_25_0_config = w2_rsc_25_0_config_t::type_id::create("w2_rsc_25_0_config");
    w2_rsc_26_0_config = w2_rsc_26_0_config_t::type_id::create("w2_rsc_26_0_config");
    w2_rsc_27_0_config = w2_rsc_27_0_config_t::type_id::create("w2_rsc_27_0_config");
    w2_rsc_28_0_config = w2_rsc_28_0_config_t::type_id::create("w2_rsc_28_0_config");
    w2_rsc_29_0_config = w2_rsc_29_0_config_t::type_id::create("w2_rsc_29_0_config");
    w2_rsc_30_0_config = w2_rsc_30_0_config_t::type_id::create("w2_rsc_30_0_config");
    w2_rsc_31_0_config = w2_rsc_31_0_config_t::type_id::create("w2_rsc_31_0_config");
    w2_rsc_32_0_config = w2_rsc_32_0_config_t::type_id::create("w2_rsc_32_0_config");
    w2_rsc_33_0_config = w2_rsc_33_0_config_t::type_id::create("w2_rsc_33_0_config");
    w2_rsc_34_0_config = w2_rsc_34_0_config_t::type_id::create("w2_rsc_34_0_config");
    w2_rsc_35_0_config = w2_rsc_35_0_config_t::type_id::create("w2_rsc_35_0_config");
    w2_rsc_36_0_config = w2_rsc_36_0_config_t::type_id::create("w2_rsc_36_0_config");
    w2_rsc_37_0_config = w2_rsc_37_0_config_t::type_id::create("w2_rsc_37_0_config");
    w2_rsc_38_0_config = w2_rsc_38_0_config_t::type_id::create("w2_rsc_38_0_config");
    w2_rsc_39_0_config = w2_rsc_39_0_config_t::type_id::create("w2_rsc_39_0_config");
    w2_rsc_40_0_config = w2_rsc_40_0_config_t::type_id::create("w2_rsc_40_0_config");
    w2_rsc_41_0_config = w2_rsc_41_0_config_t::type_id::create("w2_rsc_41_0_config");
    w2_rsc_42_0_config = w2_rsc_42_0_config_t::type_id::create("w2_rsc_42_0_config");
    w2_rsc_43_0_config = w2_rsc_43_0_config_t::type_id::create("w2_rsc_43_0_config");
    w2_rsc_44_0_config = w2_rsc_44_0_config_t::type_id::create("w2_rsc_44_0_config");
    w2_rsc_45_0_config = w2_rsc_45_0_config_t::type_id::create("w2_rsc_45_0_config");
    w2_rsc_46_0_config = w2_rsc_46_0_config_t::type_id::create("w2_rsc_46_0_config");
    w2_rsc_47_0_config = w2_rsc_47_0_config_t::type_id::create("w2_rsc_47_0_config");
    w2_rsc_48_0_config = w2_rsc_48_0_config_t::type_id::create("w2_rsc_48_0_config");
    w2_rsc_49_0_config = w2_rsc_49_0_config_t::type_id::create("w2_rsc_49_0_config");
    w2_rsc_50_0_config = w2_rsc_50_0_config_t::type_id::create("w2_rsc_50_0_config");
    w2_rsc_51_0_config = w2_rsc_51_0_config_t::type_id::create("w2_rsc_51_0_config");
    w2_rsc_52_0_config = w2_rsc_52_0_config_t::type_id::create("w2_rsc_52_0_config");
    w2_rsc_53_0_config = w2_rsc_53_0_config_t::type_id::create("w2_rsc_53_0_config");
    w2_rsc_54_0_config = w2_rsc_54_0_config_t::type_id::create("w2_rsc_54_0_config");
    w2_rsc_55_0_config = w2_rsc_55_0_config_t::type_id::create("w2_rsc_55_0_config");
    w2_rsc_56_0_config = w2_rsc_56_0_config_t::type_id::create("w2_rsc_56_0_config");
    w2_rsc_57_0_config = w2_rsc_57_0_config_t::type_id::create("w2_rsc_57_0_config");
    w2_rsc_58_0_config = w2_rsc_58_0_config_t::type_id::create("w2_rsc_58_0_config");
    w2_rsc_59_0_config = w2_rsc_59_0_config_t::type_id::create("w2_rsc_59_0_config");
    w2_rsc_60_0_config = w2_rsc_60_0_config_t::type_id::create("w2_rsc_60_0_config");
    w2_rsc_61_0_config = w2_rsc_61_0_config_t::type_id::create("w2_rsc_61_0_config");
    w2_rsc_62_0_config = w2_rsc_62_0_config_t::type_id::create("w2_rsc_62_0_config");
    w2_rsc_63_0_config = w2_rsc_63_0_config_t::type_id::create("w2_rsc_63_0_config");
    b2_rsc_config = b2_rsc_config_t::type_id::create("b2_rsc_config");
    w4_rsc_0_0_config = w4_rsc_0_0_config_t::type_id::create("w4_rsc_0_0_config");
    w4_rsc_1_0_config = w4_rsc_1_0_config_t::type_id::create("w4_rsc_1_0_config");
    w4_rsc_2_0_config = w4_rsc_2_0_config_t::type_id::create("w4_rsc_2_0_config");
    w4_rsc_3_0_config = w4_rsc_3_0_config_t::type_id::create("w4_rsc_3_0_config");
    w4_rsc_4_0_config = w4_rsc_4_0_config_t::type_id::create("w4_rsc_4_0_config");
    w4_rsc_5_0_config = w4_rsc_5_0_config_t::type_id::create("w4_rsc_5_0_config");
    w4_rsc_6_0_config = w4_rsc_6_0_config_t::type_id::create("w4_rsc_6_0_config");
    w4_rsc_7_0_config = w4_rsc_7_0_config_t::type_id::create("w4_rsc_7_0_config");
    w4_rsc_8_0_config = w4_rsc_8_0_config_t::type_id::create("w4_rsc_8_0_config");
    w4_rsc_9_0_config = w4_rsc_9_0_config_t::type_id::create("w4_rsc_9_0_config");
    w4_rsc_10_0_config = w4_rsc_10_0_config_t::type_id::create("w4_rsc_10_0_config");
    w4_rsc_11_0_config = w4_rsc_11_0_config_t::type_id::create("w4_rsc_11_0_config");
    w4_rsc_12_0_config = w4_rsc_12_0_config_t::type_id::create("w4_rsc_12_0_config");
    w4_rsc_13_0_config = w4_rsc_13_0_config_t::type_id::create("w4_rsc_13_0_config");
    w4_rsc_14_0_config = w4_rsc_14_0_config_t::type_id::create("w4_rsc_14_0_config");
    w4_rsc_15_0_config = w4_rsc_15_0_config_t::type_id::create("w4_rsc_15_0_config");
    w4_rsc_16_0_config = w4_rsc_16_0_config_t::type_id::create("w4_rsc_16_0_config");
    w4_rsc_17_0_config = w4_rsc_17_0_config_t::type_id::create("w4_rsc_17_0_config");
    w4_rsc_18_0_config = w4_rsc_18_0_config_t::type_id::create("w4_rsc_18_0_config");
    w4_rsc_19_0_config = w4_rsc_19_0_config_t::type_id::create("w4_rsc_19_0_config");
    w4_rsc_20_0_config = w4_rsc_20_0_config_t::type_id::create("w4_rsc_20_0_config");
    w4_rsc_21_0_config = w4_rsc_21_0_config_t::type_id::create("w4_rsc_21_0_config");
    w4_rsc_22_0_config = w4_rsc_22_0_config_t::type_id::create("w4_rsc_22_0_config");
    w4_rsc_23_0_config = w4_rsc_23_0_config_t::type_id::create("w4_rsc_23_0_config");
    w4_rsc_24_0_config = w4_rsc_24_0_config_t::type_id::create("w4_rsc_24_0_config");
    w4_rsc_25_0_config = w4_rsc_25_0_config_t::type_id::create("w4_rsc_25_0_config");
    w4_rsc_26_0_config = w4_rsc_26_0_config_t::type_id::create("w4_rsc_26_0_config");
    w4_rsc_27_0_config = w4_rsc_27_0_config_t::type_id::create("w4_rsc_27_0_config");
    w4_rsc_28_0_config = w4_rsc_28_0_config_t::type_id::create("w4_rsc_28_0_config");
    w4_rsc_29_0_config = w4_rsc_29_0_config_t::type_id::create("w4_rsc_29_0_config");
    w4_rsc_30_0_config = w4_rsc_30_0_config_t::type_id::create("w4_rsc_30_0_config");
    w4_rsc_31_0_config = w4_rsc_31_0_config_t::type_id::create("w4_rsc_31_0_config");
    w4_rsc_32_0_config = w4_rsc_32_0_config_t::type_id::create("w4_rsc_32_0_config");
    w4_rsc_33_0_config = w4_rsc_33_0_config_t::type_id::create("w4_rsc_33_0_config");
    w4_rsc_34_0_config = w4_rsc_34_0_config_t::type_id::create("w4_rsc_34_0_config");
    w4_rsc_35_0_config = w4_rsc_35_0_config_t::type_id::create("w4_rsc_35_0_config");
    w4_rsc_36_0_config = w4_rsc_36_0_config_t::type_id::create("w4_rsc_36_0_config");
    w4_rsc_37_0_config = w4_rsc_37_0_config_t::type_id::create("w4_rsc_37_0_config");
    w4_rsc_38_0_config = w4_rsc_38_0_config_t::type_id::create("w4_rsc_38_0_config");
    w4_rsc_39_0_config = w4_rsc_39_0_config_t::type_id::create("w4_rsc_39_0_config");
    w4_rsc_40_0_config = w4_rsc_40_0_config_t::type_id::create("w4_rsc_40_0_config");
    w4_rsc_41_0_config = w4_rsc_41_0_config_t::type_id::create("w4_rsc_41_0_config");
    w4_rsc_42_0_config = w4_rsc_42_0_config_t::type_id::create("w4_rsc_42_0_config");
    w4_rsc_43_0_config = w4_rsc_43_0_config_t::type_id::create("w4_rsc_43_0_config");
    w4_rsc_44_0_config = w4_rsc_44_0_config_t::type_id::create("w4_rsc_44_0_config");
    w4_rsc_45_0_config = w4_rsc_45_0_config_t::type_id::create("w4_rsc_45_0_config");
    w4_rsc_46_0_config = w4_rsc_46_0_config_t::type_id::create("w4_rsc_46_0_config");
    w4_rsc_47_0_config = w4_rsc_47_0_config_t::type_id::create("w4_rsc_47_0_config");
    w4_rsc_48_0_config = w4_rsc_48_0_config_t::type_id::create("w4_rsc_48_0_config");
    w4_rsc_49_0_config = w4_rsc_49_0_config_t::type_id::create("w4_rsc_49_0_config");
    w4_rsc_50_0_config = w4_rsc_50_0_config_t::type_id::create("w4_rsc_50_0_config");
    w4_rsc_51_0_config = w4_rsc_51_0_config_t::type_id::create("w4_rsc_51_0_config");
    w4_rsc_52_0_config = w4_rsc_52_0_config_t::type_id::create("w4_rsc_52_0_config");
    w4_rsc_53_0_config = w4_rsc_53_0_config_t::type_id::create("w4_rsc_53_0_config");
    w4_rsc_54_0_config = w4_rsc_54_0_config_t::type_id::create("w4_rsc_54_0_config");
    w4_rsc_55_0_config = w4_rsc_55_0_config_t::type_id::create("w4_rsc_55_0_config");
    w4_rsc_56_0_config = w4_rsc_56_0_config_t::type_id::create("w4_rsc_56_0_config");
    w4_rsc_57_0_config = w4_rsc_57_0_config_t::type_id::create("w4_rsc_57_0_config");
    w4_rsc_58_0_config = w4_rsc_58_0_config_t::type_id::create("w4_rsc_58_0_config");
    w4_rsc_59_0_config = w4_rsc_59_0_config_t::type_id::create("w4_rsc_59_0_config");
    w4_rsc_60_0_config = w4_rsc_60_0_config_t::type_id::create("w4_rsc_60_0_config");
    w4_rsc_61_0_config = w4_rsc_61_0_config_t::type_id::create("w4_rsc_61_0_config");
    w4_rsc_62_0_config = w4_rsc_62_0_config_t::type_id::create("w4_rsc_62_0_config");
    w4_rsc_63_0_config = w4_rsc_63_0_config_t::type_id::create("w4_rsc_63_0_config");
    b4_rsc_config = b4_rsc_config_t::type_id::create("b4_rsc_config");
    w6_rsc_0_0_config = w6_rsc_0_0_config_t::type_id::create("w6_rsc_0_0_config");
    w6_rsc_1_0_config = w6_rsc_1_0_config_t::type_id::create("w6_rsc_1_0_config");
    w6_rsc_2_0_config = w6_rsc_2_0_config_t::type_id::create("w6_rsc_2_0_config");
    w6_rsc_3_0_config = w6_rsc_3_0_config_t::type_id::create("w6_rsc_3_0_config");
    w6_rsc_4_0_config = w6_rsc_4_0_config_t::type_id::create("w6_rsc_4_0_config");
    w6_rsc_5_0_config = w6_rsc_5_0_config_t::type_id::create("w6_rsc_5_0_config");
    w6_rsc_6_0_config = w6_rsc_6_0_config_t::type_id::create("w6_rsc_6_0_config");
    w6_rsc_7_0_config = w6_rsc_7_0_config_t::type_id::create("w6_rsc_7_0_config");
    w6_rsc_8_0_config = w6_rsc_8_0_config_t::type_id::create("w6_rsc_8_0_config");
    w6_rsc_9_0_config = w6_rsc_9_0_config_t::type_id::create("w6_rsc_9_0_config");
    b6_rsc_config = b6_rsc_config_t::type_id::create("b6_rsc_config");


  endfunction

// ****************************************************************************
// FUNCTION: post_randomize()
// This function is automatically called after the randomize() function 
// is executed.
//
  function void post_randomize();
    super.post_randomize();


    if(!input1_rsc_config.randomize()) `uvm_fatal("RAND","input1_rsc randomization failed");
    if(!output1_rsc_config.randomize()) `uvm_fatal("RAND","output1_rsc randomization failed");
    if(!const_size_in_1_rsc_config.randomize()) `uvm_fatal("RAND","const_size_in_1_rsc randomization failed");
    if(!const_size_out_1_rsc_config.randomize()) `uvm_fatal("RAND","const_size_out_1_rsc randomization failed");
    if(!w2_rsc_0_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_0_0 randomization failed");
    if(!w2_rsc_1_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_1_0 randomization failed");
    if(!w2_rsc_2_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_2_0 randomization failed");
    if(!w2_rsc_3_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_3_0 randomization failed");
    if(!w2_rsc_4_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_4_0 randomization failed");
    if(!w2_rsc_5_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_5_0 randomization failed");
    if(!w2_rsc_6_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_6_0 randomization failed");
    if(!w2_rsc_7_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_7_0 randomization failed");
    if(!w2_rsc_8_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_8_0 randomization failed");
    if(!w2_rsc_9_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_9_0 randomization failed");
    if(!w2_rsc_10_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_10_0 randomization failed");
    if(!w2_rsc_11_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_11_0 randomization failed");
    if(!w2_rsc_12_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_12_0 randomization failed");
    if(!w2_rsc_13_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_13_0 randomization failed");
    if(!w2_rsc_14_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_14_0 randomization failed");
    if(!w2_rsc_15_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_15_0 randomization failed");
    if(!w2_rsc_16_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_16_0 randomization failed");
    if(!w2_rsc_17_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_17_0 randomization failed");
    if(!w2_rsc_18_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_18_0 randomization failed");
    if(!w2_rsc_19_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_19_0 randomization failed");
    if(!w2_rsc_20_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_20_0 randomization failed");
    if(!w2_rsc_21_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_21_0 randomization failed");
    if(!w2_rsc_22_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_22_0 randomization failed");
    if(!w2_rsc_23_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_23_0 randomization failed");
    if(!w2_rsc_24_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_24_0 randomization failed");
    if(!w2_rsc_25_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_25_0 randomization failed");
    if(!w2_rsc_26_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_26_0 randomization failed");
    if(!w2_rsc_27_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_27_0 randomization failed");
    if(!w2_rsc_28_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_28_0 randomization failed");
    if(!w2_rsc_29_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_29_0 randomization failed");
    if(!w2_rsc_30_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_30_0 randomization failed");
    if(!w2_rsc_31_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_31_0 randomization failed");
    if(!w2_rsc_32_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_32_0 randomization failed");
    if(!w2_rsc_33_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_33_0 randomization failed");
    if(!w2_rsc_34_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_34_0 randomization failed");
    if(!w2_rsc_35_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_35_0 randomization failed");
    if(!w2_rsc_36_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_36_0 randomization failed");
    if(!w2_rsc_37_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_37_0 randomization failed");
    if(!w2_rsc_38_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_38_0 randomization failed");
    if(!w2_rsc_39_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_39_0 randomization failed");
    if(!w2_rsc_40_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_40_0 randomization failed");
    if(!w2_rsc_41_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_41_0 randomization failed");
    if(!w2_rsc_42_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_42_0 randomization failed");
    if(!w2_rsc_43_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_43_0 randomization failed");
    if(!w2_rsc_44_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_44_0 randomization failed");
    if(!w2_rsc_45_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_45_0 randomization failed");
    if(!w2_rsc_46_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_46_0 randomization failed");
    if(!w2_rsc_47_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_47_0 randomization failed");
    if(!w2_rsc_48_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_48_0 randomization failed");
    if(!w2_rsc_49_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_49_0 randomization failed");
    if(!w2_rsc_50_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_50_0 randomization failed");
    if(!w2_rsc_51_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_51_0 randomization failed");
    if(!w2_rsc_52_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_52_0 randomization failed");
    if(!w2_rsc_53_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_53_0 randomization failed");
    if(!w2_rsc_54_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_54_0 randomization failed");
    if(!w2_rsc_55_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_55_0 randomization failed");
    if(!w2_rsc_56_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_56_0 randomization failed");
    if(!w2_rsc_57_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_57_0 randomization failed");
    if(!w2_rsc_58_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_58_0 randomization failed");
    if(!w2_rsc_59_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_59_0 randomization failed");
    if(!w2_rsc_60_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_60_0 randomization failed");
    if(!w2_rsc_61_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_61_0 randomization failed");
    if(!w2_rsc_62_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_62_0 randomization failed");
    if(!w2_rsc_63_0_config.randomize()) `uvm_fatal("RAND","w2_rsc_63_0 randomization failed");
    if(!b2_rsc_config.randomize()) `uvm_fatal("RAND","b2_rsc randomization failed");
    if(!w4_rsc_0_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_0_0 randomization failed");
    if(!w4_rsc_1_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_1_0 randomization failed");
    if(!w4_rsc_2_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_2_0 randomization failed");
    if(!w4_rsc_3_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_3_0 randomization failed");
    if(!w4_rsc_4_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_4_0 randomization failed");
    if(!w4_rsc_5_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_5_0 randomization failed");
    if(!w4_rsc_6_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_6_0 randomization failed");
    if(!w4_rsc_7_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_7_0 randomization failed");
    if(!w4_rsc_8_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_8_0 randomization failed");
    if(!w4_rsc_9_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_9_0 randomization failed");
    if(!w4_rsc_10_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_10_0 randomization failed");
    if(!w4_rsc_11_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_11_0 randomization failed");
    if(!w4_rsc_12_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_12_0 randomization failed");
    if(!w4_rsc_13_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_13_0 randomization failed");
    if(!w4_rsc_14_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_14_0 randomization failed");
    if(!w4_rsc_15_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_15_0 randomization failed");
    if(!w4_rsc_16_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_16_0 randomization failed");
    if(!w4_rsc_17_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_17_0 randomization failed");
    if(!w4_rsc_18_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_18_0 randomization failed");
    if(!w4_rsc_19_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_19_0 randomization failed");
    if(!w4_rsc_20_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_20_0 randomization failed");
    if(!w4_rsc_21_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_21_0 randomization failed");
    if(!w4_rsc_22_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_22_0 randomization failed");
    if(!w4_rsc_23_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_23_0 randomization failed");
    if(!w4_rsc_24_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_24_0 randomization failed");
    if(!w4_rsc_25_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_25_0 randomization failed");
    if(!w4_rsc_26_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_26_0 randomization failed");
    if(!w4_rsc_27_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_27_0 randomization failed");
    if(!w4_rsc_28_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_28_0 randomization failed");
    if(!w4_rsc_29_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_29_0 randomization failed");
    if(!w4_rsc_30_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_30_0 randomization failed");
    if(!w4_rsc_31_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_31_0 randomization failed");
    if(!w4_rsc_32_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_32_0 randomization failed");
    if(!w4_rsc_33_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_33_0 randomization failed");
    if(!w4_rsc_34_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_34_0 randomization failed");
    if(!w4_rsc_35_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_35_0 randomization failed");
    if(!w4_rsc_36_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_36_0 randomization failed");
    if(!w4_rsc_37_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_37_0 randomization failed");
    if(!w4_rsc_38_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_38_0 randomization failed");
    if(!w4_rsc_39_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_39_0 randomization failed");
    if(!w4_rsc_40_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_40_0 randomization failed");
    if(!w4_rsc_41_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_41_0 randomization failed");
    if(!w4_rsc_42_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_42_0 randomization failed");
    if(!w4_rsc_43_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_43_0 randomization failed");
    if(!w4_rsc_44_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_44_0 randomization failed");
    if(!w4_rsc_45_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_45_0 randomization failed");
    if(!w4_rsc_46_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_46_0 randomization failed");
    if(!w4_rsc_47_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_47_0 randomization failed");
    if(!w4_rsc_48_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_48_0 randomization failed");
    if(!w4_rsc_49_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_49_0 randomization failed");
    if(!w4_rsc_50_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_50_0 randomization failed");
    if(!w4_rsc_51_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_51_0 randomization failed");
    if(!w4_rsc_52_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_52_0 randomization failed");
    if(!w4_rsc_53_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_53_0 randomization failed");
    if(!w4_rsc_54_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_54_0 randomization failed");
    if(!w4_rsc_55_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_55_0 randomization failed");
    if(!w4_rsc_56_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_56_0 randomization failed");
    if(!w4_rsc_57_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_57_0 randomization failed");
    if(!w4_rsc_58_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_58_0 randomization failed");
    if(!w4_rsc_59_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_59_0 randomization failed");
    if(!w4_rsc_60_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_60_0 randomization failed");
    if(!w4_rsc_61_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_61_0 randomization failed");
    if(!w4_rsc_62_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_62_0 randomization failed");
    if(!w4_rsc_63_0_config.randomize()) `uvm_fatal("RAND","w4_rsc_63_0 randomization failed");
    if(!b4_rsc_config.randomize()) `uvm_fatal("RAND","b4_rsc randomization failed");
    if(!w6_rsc_0_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_0_0 randomization failed");
    if(!w6_rsc_1_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_1_0 randomization failed");
    if(!w6_rsc_2_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_2_0 randomization failed");
    if(!w6_rsc_3_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_3_0 randomization failed");
    if(!w6_rsc_4_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_4_0 randomization failed");
    if(!w6_rsc_5_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_5_0 randomization failed");
    if(!w6_rsc_6_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_6_0 randomization failed");
    if(!w6_rsc_7_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_7_0 randomization failed");
    if(!w6_rsc_8_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_8_0 randomization failed");
    if(!w6_rsc_9_0_config.randomize()) `uvm_fatal("RAND","w6_rsc_9_0 randomization failed");
    if(!b6_rsc_config.randomize()) `uvm_fatal("RAND","b6_rsc randomization failed");

  endfunction
  
// ****************************************************************************
// FUNCTION: convert2string()
// This function converts all variables in this class to a single string for
// logfile reporting. This function concatenates the convert2string result for
// each agent configuration in this configuration class.
//
  virtual function string convert2string();
    return {
     
     "\n", input1_rsc_config.convert2string,
     "\n", output1_rsc_config.convert2string,
     "\n", const_size_in_1_rsc_config.convert2string,
     "\n", const_size_out_1_rsc_config.convert2string,
     "\n", w2_rsc_0_0_config.convert2string,
     "\n", w2_rsc_1_0_config.convert2string,
     "\n", w2_rsc_2_0_config.convert2string,
     "\n", w2_rsc_3_0_config.convert2string,
     "\n", w2_rsc_4_0_config.convert2string,
     "\n", w2_rsc_5_0_config.convert2string,
     "\n", w2_rsc_6_0_config.convert2string,
     "\n", w2_rsc_7_0_config.convert2string,
     "\n", w2_rsc_8_0_config.convert2string,
     "\n", w2_rsc_9_0_config.convert2string,
     "\n", w2_rsc_10_0_config.convert2string,
     "\n", w2_rsc_11_0_config.convert2string,
     "\n", w2_rsc_12_0_config.convert2string,
     "\n", w2_rsc_13_0_config.convert2string,
     "\n", w2_rsc_14_0_config.convert2string,
     "\n", w2_rsc_15_0_config.convert2string,
     "\n", w2_rsc_16_0_config.convert2string,
     "\n", w2_rsc_17_0_config.convert2string,
     "\n", w2_rsc_18_0_config.convert2string,
     "\n", w2_rsc_19_0_config.convert2string,
     "\n", w2_rsc_20_0_config.convert2string,
     "\n", w2_rsc_21_0_config.convert2string,
     "\n", w2_rsc_22_0_config.convert2string,
     "\n", w2_rsc_23_0_config.convert2string,
     "\n", w2_rsc_24_0_config.convert2string,
     "\n", w2_rsc_25_0_config.convert2string,
     "\n", w2_rsc_26_0_config.convert2string,
     "\n", w2_rsc_27_0_config.convert2string,
     "\n", w2_rsc_28_0_config.convert2string,
     "\n", w2_rsc_29_0_config.convert2string,
     "\n", w2_rsc_30_0_config.convert2string,
     "\n", w2_rsc_31_0_config.convert2string,
     "\n", w2_rsc_32_0_config.convert2string,
     "\n", w2_rsc_33_0_config.convert2string,
     "\n", w2_rsc_34_0_config.convert2string,
     "\n", w2_rsc_35_0_config.convert2string,
     "\n", w2_rsc_36_0_config.convert2string,
     "\n", w2_rsc_37_0_config.convert2string,
     "\n", w2_rsc_38_0_config.convert2string,
     "\n", w2_rsc_39_0_config.convert2string,
     "\n", w2_rsc_40_0_config.convert2string,
     "\n", w2_rsc_41_0_config.convert2string,
     "\n", w2_rsc_42_0_config.convert2string,
     "\n", w2_rsc_43_0_config.convert2string,
     "\n", w2_rsc_44_0_config.convert2string,
     "\n", w2_rsc_45_0_config.convert2string,
     "\n", w2_rsc_46_0_config.convert2string,
     "\n", w2_rsc_47_0_config.convert2string,
     "\n", w2_rsc_48_0_config.convert2string,
     "\n", w2_rsc_49_0_config.convert2string,
     "\n", w2_rsc_50_0_config.convert2string,
     "\n", w2_rsc_51_0_config.convert2string,
     "\n", w2_rsc_52_0_config.convert2string,
     "\n", w2_rsc_53_0_config.convert2string,
     "\n", w2_rsc_54_0_config.convert2string,
     "\n", w2_rsc_55_0_config.convert2string,
     "\n", w2_rsc_56_0_config.convert2string,
     "\n", w2_rsc_57_0_config.convert2string,
     "\n", w2_rsc_58_0_config.convert2string,
     "\n", w2_rsc_59_0_config.convert2string,
     "\n", w2_rsc_60_0_config.convert2string,
     "\n", w2_rsc_61_0_config.convert2string,
     "\n", w2_rsc_62_0_config.convert2string,
     "\n", w2_rsc_63_0_config.convert2string,
     "\n", b2_rsc_config.convert2string,
     "\n", w4_rsc_0_0_config.convert2string,
     "\n", w4_rsc_1_0_config.convert2string,
     "\n", w4_rsc_2_0_config.convert2string,
     "\n", w4_rsc_3_0_config.convert2string,
     "\n", w4_rsc_4_0_config.convert2string,
     "\n", w4_rsc_5_0_config.convert2string,
     "\n", w4_rsc_6_0_config.convert2string,
     "\n", w4_rsc_7_0_config.convert2string,
     "\n", w4_rsc_8_0_config.convert2string,
     "\n", w4_rsc_9_0_config.convert2string,
     "\n", w4_rsc_10_0_config.convert2string,
     "\n", w4_rsc_11_0_config.convert2string,
     "\n", w4_rsc_12_0_config.convert2string,
     "\n", w4_rsc_13_0_config.convert2string,
     "\n", w4_rsc_14_0_config.convert2string,
     "\n", w4_rsc_15_0_config.convert2string,
     "\n", w4_rsc_16_0_config.convert2string,
     "\n", w4_rsc_17_0_config.convert2string,
     "\n", w4_rsc_18_0_config.convert2string,
     "\n", w4_rsc_19_0_config.convert2string,
     "\n", w4_rsc_20_0_config.convert2string,
     "\n", w4_rsc_21_0_config.convert2string,
     "\n", w4_rsc_22_0_config.convert2string,
     "\n", w4_rsc_23_0_config.convert2string,
     "\n", w4_rsc_24_0_config.convert2string,
     "\n", w4_rsc_25_0_config.convert2string,
     "\n", w4_rsc_26_0_config.convert2string,
     "\n", w4_rsc_27_0_config.convert2string,
     "\n", w4_rsc_28_0_config.convert2string,
     "\n", w4_rsc_29_0_config.convert2string,
     "\n", w4_rsc_30_0_config.convert2string,
     "\n", w4_rsc_31_0_config.convert2string,
     "\n", w4_rsc_32_0_config.convert2string,
     "\n", w4_rsc_33_0_config.convert2string,
     "\n", w4_rsc_34_0_config.convert2string,
     "\n", w4_rsc_35_0_config.convert2string,
     "\n", w4_rsc_36_0_config.convert2string,
     "\n", w4_rsc_37_0_config.convert2string,
     "\n", w4_rsc_38_0_config.convert2string,
     "\n", w4_rsc_39_0_config.convert2string,
     "\n", w4_rsc_40_0_config.convert2string,
     "\n", w4_rsc_41_0_config.convert2string,
     "\n", w4_rsc_42_0_config.convert2string,
     "\n", w4_rsc_43_0_config.convert2string,
     "\n", w4_rsc_44_0_config.convert2string,
     "\n", w4_rsc_45_0_config.convert2string,
     "\n", w4_rsc_46_0_config.convert2string,
     "\n", w4_rsc_47_0_config.convert2string,
     "\n", w4_rsc_48_0_config.convert2string,
     "\n", w4_rsc_49_0_config.convert2string,
     "\n", w4_rsc_50_0_config.convert2string,
     "\n", w4_rsc_51_0_config.convert2string,
     "\n", w4_rsc_52_0_config.convert2string,
     "\n", w4_rsc_53_0_config.convert2string,
     "\n", w4_rsc_54_0_config.convert2string,
     "\n", w4_rsc_55_0_config.convert2string,
     "\n", w4_rsc_56_0_config.convert2string,
     "\n", w4_rsc_57_0_config.convert2string,
     "\n", w4_rsc_58_0_config.convert2string,
     "\n", w4_rsc_59_0_config.convert2string,
     "\n", w4_rsc_60_0_config.convert2string,
     "\n", w4_rsc_61_0_config.convert2string,
     "\n", w4_rsc_62_0_config.convert2string,
     "\n", w4_rsc_63_0_config.convert2string,
     "\n", b4_rsc_config.convert2string,
     "\n", w6_rsc_0_0_config.convert2string,
     "\n", w6_rsc_1_0_config.convert2string,
     "\n", w6_rsc_2_0_config.convert2string,
     "\n", w6_rsc_3_0_config.convert2string,
     "\n", w6_rsc_4_0_config.convert2string,
     "\n", w6_rsc_5_0_config.convert2string,
     "\n", w6_rsc_6_0_config.convert2string,
     "\n", w6_rsc_7_0_config.convert2string,
     "\n", w6_rsc_8_0_config.convert2string,
     "\n", w6_rsc_9_0_config.convert2string,
     "\n", b6_rsc_config.convert2string


       };

  endfunction
// ****************************************************************************
// FUNCTION: initialize();
// This function configures each interface agents configuration class.  The 
// sim level determines the active/passive state of the agent.  The environment_path
// identifies the hierarchy down to and including the instantiation name of the
// environment for this configuration class.  Each instance of the environment 
// has its own configuration class.  The string interface names are used by 
// the agent configurations to identify the virtual interface handle to pull from
// the uvm_config_db.  
//
  function void initialize(uvmf_sim_level_t sim_level, 
                                      string environment_path,
                                      string interface_names[],
                                      uvm_reg_block register_model = null,
                                      uvmf_active_passive_t interface_activity[] = null
                                     );

    super.initialize(sim_level, environment_path, interface_names, register_model, interface_activity);



  // Interface initialization for local agents
     input1_rsc_config.initialize( interface_activity[0], {environment_path,".input1_rsc"}, interface_names[0]);
     input1_rsc_config.initiator_responder = INITIATOR;
     output1_rsc_config.initialize( interface_activity[1], {environment_path,".output1_rsc"}, interface_names[1]);
     output1_rsc_config.initiator_responder = RESPONDER;
     const_size_in_1_rsc_config.initialize( interface_activity[2], {environment_path,".const_size_in_1_rsc"}, interface_names[2]);
     const_size_in_1_rsc_config.initiator_responder = RESPONDER;
     const_size_out_1_rsc_config.initialize( interface_activity[3], {environment_path,".const_size_out_1_rsc"}, interface_names[3]);
     const_size_out_1_rsc_config.initiator_responder = RESPONDER;
     w2_rsc_0_0_config.initialize( interface_activity[4], {environment_path,".w2_rsc_0_0"}, interface_names[4]);
     w2_rsc_0_0_config.initiator_responder = INITIATOR;
     w2_rsc_1_0_config.initialize( interface_activity[5], {environment_path,".w2_rsc_1_0"}, interface_names[5]);
     w2_rsc_1_0_config.initiator_responder = INITIATOR;
     w2_rsc_2_0_config.initialize( interface_activity[6], {environment_path,".w2_rsc_2_0"}, interface_names[6]);
     w2_rsc_2_0_config.initiator_responder = INITIATOR;
     w2_rsc_3_0_config.initialize( interface_activity[7], {environment_path,".w2_rsc_3_0"}, interface_names[7]);
     w2_rsc_3_0_config.initiator_responder = INITIATOR;
     w2_rsc_4_0_config.initialize( interface_activity[8], {environment_path,".w2_rsc_4_0"}, interface_names[8]);
     w2_rsc_4_0_config.initiator_responder = INITIATOR;
     w2_rsc_5_0_config.initialize( interface_activity[9], {environment_path,".w2_rsc_5_0"}, interface_names[9]);
     w2_rsc_5_0_config.initiator_responder = INITIATOR;
     w2_rsc_6_0_config.initialize( interface_activity[10], {environment_path,".w2_rsc_6_0"}, interface_names[10]);
     w2_rsc_6_0_config.initiator_responder = INITIATOR;
     w2_rsc_7_0_config.initialize( interface_activity[11], {environment_path,".w2_rsc_7_0"}, interface_names[11]);
     w2_rsc_7_0_config.initiator_responder = INITIATOR;
     w2_rsc_8_0_config.initialize( interface_activity[12], {environment_path,".w2_rsc_8_0"}, interface_names[12]);
     w2_rsc_8_0_config.initiator_responder = INITIATOR;
     w2_rsc_9_0_config.initialize( interface_activity[13], {environment_path,".w2_rsc_9_0"}, interface_names[13]);
     w2_rsc_9_0_config.initiator_responder = INITIATOR;
     w2_rsc_10_0_config.initialize( interface_activity[14], {environment_path,".w2_rsc_10_0"}, interface_names[14]);
     w2_rsc_10_0_config.initiator_responder = INITIATOR;
     w2_rsc_11_0_config.initialize( interface_activity[15], {environment_path,".w2_rsc_11_0"}, interface_names[15]);
     w2_rsc_11_0_config.initiator_responder = INITIATOR;
     w2_rsc_12_0_config.initialize( interface_activity[16], {environment_path,".w2_rsc_12_0"}, interface_names[16]);
     w2_rsc_12_0_config.initiator_responder = INITIATOR;
     w2_rsc_13_0_config.initialize( interface_activity[17], {environment_path,".w2_rsc_13_0"}, interface_names[17]);
     w2_rsc_13_0_config.initiator_responder = INITIATOR;
     w2_rsc_14_0_config.initialize( interface_activity[18], {environment_path,".w2_rsc_14_0"}, interface_names[18]);
     w2_rsc_14_0_config.initiator_responder = INITIATOR;
     w2_rsc_15_0_config.initialize( interface_activity[19], {environment_path,".w2_rsc_15_0"}, interface_names[19]);
     w2_rsc_15_0_config.initiator_responder = INITIATOR;
     w2_rsc_16_0_config.initialize( interface_activity[20], {environment_path,".w2_rsc_16_0"}, interface_names[20]);
     w2_rsc_16_0_config.initiator_responder = INITIATOR;
     w2_rsc_17_0_config.initialize( interface_activity[21], {environment_path,".w2_rsc_17_0"}, interface_names[21]);
     w2_rsc_17_0_config.initiator_responder = INITIATOR;
     w2_rsc_18_0_config.initialize( interface_activity[22], {environment_path,".w2_rsc_18_0"}, interface_names[22]);
     w2_rsc_18_0_config.initiator_responder = INITIATOR;
     w2_rsc_19_0_config.initialize( interface_activity[23], {environment_path,".w2_rsc_19_0"}, interface_names[23]);
     w2_rsc_19_0_config.initiator_responder = INITIATOR;
     w2_rsc_20_0_config.initialize( interface_activity[24], {environment_path,".w2_rsc_20_0"}, interface_names[24]);
     w2_rsc_20_0_config.initiator_responder = INITIATOR;
     w2_rsc_21_0_config.initialize( interface_activity[25], {environment_path,".w2_rsc_21_0"}, interface_names[25]);
     w2_rsc_21_0_config.initiator_responder = INITIATOR;
     w2_rsc_22_0_config.initialize( interface_activity[26], {environment_path,".w2_rsc_22_0"}, interface_names[26]);
     w2_rsc_22_0_config.initiator_responder = INITIATOR;
     w2_rsc_23_0_config.initialize( interface_activity[27], {environment_path,".w2_rsc_23_0"}, interface_names[27]);
     w2_rsc_23_0_config.initiator_responder = INITIATOR;
     w2_rsc_24_0_config.initialize( interface_activity[28], {environment_path,".w2_rsc_24_0"}, interface_names[28]);
     w2_rsc_24_0_config.initiator_responder = INITIATOR;
     w2_rsc_25_0_config.initialize( interface_activity[29], {environment_path,".w2_rsc_25_0"}, interface_names[29]);
     w2_rsc_25_0_config.initiator_responder = INITIATOR;
     w2_rsc_26_0_config.initialize( interface_activity[30], {environment_path,".w2_rsc_26_0"}, interface_names[30]);
     w2_rsc_26_0_config.initiator_responder = INITIATOR;
     w2_rsc_27_0_config.initialize( interface_activity[31], {environment_path,".w2_rsc_27_0"}, interface_names[31]);
     w2_rsc_27_0_config.initiator_responder = INITIATOR;
     w2_rsc_28_0_config.initialize( interface_activity[32], {environment_path,".w2_rsc_28_0"}, interface_names[32]);
     w2_rsc_28_0_config.initiator_responder = INITIATOR;
     w2_rsc_29_0_config.initialize( interface_activity[33], {environment_path,".w2_rsc_29_0"}, interface_names[33]);
     w2_rsc_29_0_config.initiator_responder = INITIATOR;
     w2_rsc_30_0_config.initialize( interface_activity[34], {environment_path,".w2_rsc_30_0"}, interface_names[34]);
     w2_rsc_30_0_config.initiator_responder = INITIATOR;
     w2_rsc_31_0_config.initialize( interface_activity[35], {environment_path,".w2_rsc_31_0"}, interface_names[35]);
     w2_rsc_31_0_config.initiator_responder = INITIATOR;
     w2_rsc_32_0_config.initialize( interface_activity[36], {environment_path,".w2_rsc_32_0"}, interface_names[36]);
     w2_rsc_32_0_config.initiator_responder = INITIATOR;
     w2_rsc_33_0_config.initialize( interface_activity[37], {environment_path,".w2_rsc_33_0"}, interface_names[37]);
     w2_rsc_33_0_config.initiator_responder = INITIATOR;
     w2_rsc_34_0_config.initialize( interface_activity[38], {environment_path,".w2_rsc_34_0"}, interface_names[38]);
     w2_rsc_34_0_config.initiator_responder = INITIATOR;
     w2_rsc_35_0_config.initialize( interface_activity[39], {environment_path,".w2_rsc_35_0"}, interface_names[39]);
     w2_rsc_35_0_config.initiator_responder = INITIATOR;
     w2_rsc_36_0_config.initialize( interface_activity[40], {environment_path,".w2_rsc_36_0"}, interface_names[40]);
     w2_rsc_36_0_config.initiator_responder = INITIATOR;
     w2_rsc_37_0_config.initialize( interface_activity[41], {environment_path,".w2_rsc_37_0"}, interface_names[41]);
     w2_rsc_37_0_config.initiator_responder = INITIATOR;
     w2_rsc_38_0_config.initialize( interface_activity[42], {environment_path,".w2_rsc_38_0"}, interface_names[42]);
     w2_rsc_38_0_config.initiator_responder = INITIATOR;
     w2_rsc_39_0_config.initialize( interface_activity[43], {environment_path,".w2_rsc_39_0"}, interface_names[43]);
     w2_rsc_39_0_config.initiator_responder = INITIATOR;
     w2_rsc_40_0_config.initialize( interface_activity[44], {environment_path,".w2_rsc_40_0"}, interface_names[44]);
     w2_rsc_40_0_config.initiator_responder = INITIATOR;
     w2_rsc_41_0_config.initialize( interface_activity[45], {environment_path,".w2_rsc_41_0"}, interface_names[45]);
     w2_rsc_41_0_config.initiator_responder = INITIATOR;
     w2_rsc_42_0_config.initialize( interface_activity[46], {environment_path,".w2_rsc_42_0"}, interface_names[46]);
     w2_rsc_42_0_config.initiator_responder = INITIATOR;
     w2_rsc_43_0_config.initialize( interface_activity[47], {environment_path,".w2_rsc_43_0"}, interface_names[47]);
     w2_rsc_43_0_config.initiator_responder = INITIATOR;
     w2_rsc_44_0_config.initialize( interface_activity[48], {environment_path,".w2_rsc_44_0"}, interface_names[48]);
     w2_rsc_44_0_config.initiator_responder = INITIATOR;
     w2_rsc_45_0_config.initialize( interface_activity[49], {environment_path,".w2_rsc_45_0"}, interface_names[49]);
     w2_rsc_45_0_config.initiator_responder = INITIATOR;
     w2_rsc_46_0_config.initialize( interface_activity[50], {environment_path,".w2_rsc_46_0"}, interface_names[50]);
     w2_rsc_46_0_config.initiator_responder = INITIATOR;
     w2_rsc_47_0_config.initialize( interface_activity[51], {environment_path,".w2_rsc_47_0"}, interface_names[51]);
     w2_rsc_47_0_config.initiator_responder = INITIATOR;
     w2_rsc_48_0_config.initialize( interface_activity[52], {environment_path,".w2_rsc_48_0"}, interface_names[52]);
     w2_rsc_48_0_config.initiator_responder = INITIATOR;
     w2_rsc_49_0_config.initialize( interface_activity[53], {environment_path,".w2_rsc_49_0"}, interface_names[53]);
     w2_rsc_49_0_config.initiator_responder = INITIATOR;
     w2_rsc_50_0_config.initialize( interface_activity[54], {environment_path,".w2_rsc_50_0"}, interface_names[54]);
     w2_rsc_50_0_config.initiator_responder = INITIATOR;
     w2_rsc_51_0_config.initialize( interface_activity[55], {environment_path,".w2_rsc_51_0"}, interface_names[55]);
     w2_rsc_51_0_config.initiator_responder = INITIATOR;
     w2_rsc_52_0_config.initialize( interface_activity[56], {environment_path,".w2_rsc_52_0"}, interface_names[56]);
     w2_rsc_52_0_config.initiator_responder = INITIATOR;
     w2_rsc_53_0_config.initialize( interface_activity[57], {environment_path,".w2_rsc_53_0"}, interface_names[57]);
     w2_rsc_53_0_config.initiator_responder = INITIATOR;
     w2_rsc_54_0_config.initialize( interface_activity[58], {environment_path,".w2_rsc_54_0"}, interface_names[58]);
     w2_rsc_54_0_config.initiator_responder = INITIATOR;
     w2_rsc_55_0_config.initialize( interface_activity[59], {environment_path,".w2_rsc_55_0"}, interface_names[59]);
     w2_rsc_55_0_config.initiator_responder = INITIATOR;
     w2_rsc_56_0_config.initialize( interface_activity[60], {environment_path,".w2_rsc_56_0"}, interface_names[60]);
     w2_rsc_56_0_config.initiator_responder = INITIATOR;
     w2_rsc_57_0_config.initialize( interface_activity[61], {environment_path,".w2_rsc_57_0"}, interface_names[61]);
     w2_rsc_57_0_config.initiator_responder = INITIATOR;
     w2_rsc_58_0_config.initialize( interface_activity[62], {environment_path,".w2_rsc_58_0"}, interface_names[62]);
     w2_rsc_58_0_config.initiator_responder = INITIATOR;
     w2_rsc_59_0_config.initialize( interface_activity[63], {environment_path,".w2_rsc_59_0"}, interface_names[63]);
     w2_rsc_59_0_config.initiator_responder = INITIATOR;
     w2_rsc_60_0_config.initialize( interface_activity[64], {environment_path,".w2_rsc_60_0"}, interface_names[64]);
     w2_rsc_60_0_config.initiator_responder = INITIATOR;
     w2_rsc_61_0_config.initialize( interface_activity[65], {environment_path,".w2_rsc_61_0"}, interface_names[65]);
     w2_rsc_61_0_config.initiator_responder = INITIATOR;
     w2_rsc_62_0_config.initialize( interface_activity[66], {environment_path,".w2_rsc_62_0"}, interface_names[66]);
     w2_rsc_62_0_config.initiator_responder = INITIATOR;
     w2_rsc_63_0_config.initialize( interface_activity[67], {environment_path,".w2_rsc_63_0"}, interface_names[67]);
     w2_rsc_63_0_config.initiator_responder = INITIATOR;
     b2_rsc_config.initialize( interface_activity[68], {environment_path,".b2_rsc"}, interface_names[68]);
     b2_rsc_config.initiator_responder = INITIATOR;
     w4_rsc_0_0_config.initialize( interface_activity[69], {environment_path,".w4_rsc_0_0"}, interface_names[69]);
     w4_rsc_0_0_config.initiator_responder = INITIATOR;
     w4_rsc_1_0_config.initialize( interface_activity[70], {environment_path,".w4_rsc_1_0"}, interface_names[70]);
     w4_rsc_1_0_config.initiator_responder = INITIATOR;
     w4_rsc_2_0_config.initialize( interface_activity[71], {environment_path,".w4_rsc_2_0"}, interface_names[71]);
     w4_rsc_2_0_config.initiator_responder = INITIATOR;
     w4_rsc_3_0_config.initialize( interface_activity[72], {environment_path,".w4_rsc_3_0"}, interface_names[72]);
     w4_rsc_3_0_config.initiator_responder = INITIATOR;
     w4_rsc_4_0_config.initialize( interface_activity[73], {environment_path,".w4_rsc_4_0"}, interface_names[73]);
     w4_rsc_4_0_config.initiator_responder = INITIATOR;
     w4_rsc_5_0_config.initialize( interface_activity[74], {environment_path,".w4_rsc_5_0"}, interface_names[74]);
     w4_rsc_5_0_config.initiator_responder = INITIATOR;
     w4_rsc_6_0_config.initialize( interface_activity[75], {environment_path,".w4_rsc_6_0"}, interface_names[75]);
     w4_rsc_6_0_config.initiator_responder = INITIATOR;
     w4_rsc_7_0_config.initialize( interface_activity[76], {environment_path,".w4_rsc_7_0"}, interface_names[76]);
     w4_rsc_7_0_config.initiator_responder = INITIATOR;
     w4_rsc_8_0_config.initialize( interface_activity[77], {environment_path,".w4_rsc_8_0"}, interface_names[77]);
     w4_rsc_8_0_config.initiator_responder = INITIATOR;
     w4_rsc_9_0_config.initialize( interface_activity[78], {environment_path,".w4_rsc_9_0"}, interface_names[78]);
     w4_rsc_9_0_config.initiator_responder = INITIATOR;
     w4_rsc_10_0_config.initialize( interface_activity[79], {environment_path,".w4_rsc_10_0"}, interface_names[79]);
     w4_rsc_10_0_config.initiator_responder = INITIATOR;
     w4_rsc_11_0_config.initialize( interface_activity[80], {environment_path,".w4_rsc_11_0"}, interface_names[80]);
     w4_rsc_11_0_config.initiator_responder = INITIATOR;
     w4_rsc_12_0_config.initialize( interface_activity[81], {environment_path,".w4_rsc_12_0"}, interface_names[81]);
     w4_rsc_12_0_config.initiator_responder = INITIATOR;
     w4_rsc_13_0_config.initialize( interface_activity[82], {environment_path,".w4_rsc_13_0"}, interface_names[82]);
     w4_rsc_13_0_config.initiator_responder = INITIATOR;
     w4_rsc_14_0_config.initialize( interface_activity[83], {environment_path,".w4_rsc_14_0"}, interface_names[83]);
     w4_rsc_14_0_config.initiator_responder = INITIATOR;
     w4_rsc_15_0_config.initialize( interface_activity[84], {environment_path,".w4_rsc_15_0"}, interface_names[84]);
     w4_rsc_15_0_config.initiator_responder = INITIATOR;
     w4_rsc_16_0_config.initialize( interface_activity[85], {environment_path,".w4_rsc_16_0"}, interface_names[85]);
     w4_rsc_16_0_config.initiator_responder = INITIATOR;
     w4_rsc_17_0_config.initialize( interface_activity[86], {environment_path,".w4_rsc_17_0"}, interface_names[86]);
     w4_rsc_17_0_config.initiator_responder = INITIATOR;
     w4_rsc_18_0_config.initialize( interface_activity[87], {environment_path,".w4_rsc_18_0"}, interface_names[87]);
     w4_rsc_18_0_config.initiator_responder = INITIATOR;
     w4_rsc_19_0_config.initialize( interface_activity[88], {environment_path,".w4_rsc_19_0"}, interface_names[88]);
     w4_rsc_19_0_config.initiator_responder = INITIATOR;
     w4_rsc_20_0_config.initialize( interface_activity[89], {environment_path,".w4_rsc_20_0"}, interface_names[89]);
     w4_rsc_20_0_config.initiator_responder = INITIATOR;
     w4_rsc_21_0_config.initialize( interface_activity[90], {environment_path,".w4_rsc_21_0"}, interface_names[90]);
     w4_rsc_21_0_config.initiator_responder = INITIATOR;
     w4_rsc_22_0_config.initialize( interface_activity[91], {environment_path,".w4_rsc_22_0"}, interface_names[91]);
     w4_rsc_22_0_config.initiator_responder = INITIATOR;
     w4_rsc_23_0_config.initialize( interface_activity[92], {environment_path,".w4_rsc_23_0"}, interface_names[92]);
     w4_rsc_23_0_config.initiator_responder = INITIATOR;
     w4_rsc_24_0_config.initialize( interface_activity[93], {environment_path,".w4_rsc_24_0"}, interface_names[93]);
     w4_rsc_24_0_config.initiator_responder = INITIATOR;
     w4_rsc_25_0_config.initialize( interface_activity[94], {environment_path,".w4_rsc_25_0"}, interface_names[94]);
     w4_rsc_25_0_config.initiator_responder = INITIATOR;
     w4_rsc_26_0_config.initialize( interface_activity[95], {environment_path,".w4_rsc_26_0"}, interface_names[95]);
     w4_rsc_26_0_config.initiator_responder = INITIATOR;
     w4_rsc_27_0_config.initialize( interface_activity[96], {environment_path,".w4_rsc_27_0"}, interface_names[96]);
     w4_rsc_27_0_config.initiator_responder = INITIATOR;
     w4_rsc_28_0_config.initialize( interface_activity[97], {environment_path,".w4_rsc_28_0"}, interface_names[97]);
     w4_rsc_28_0_config.initiator_responder = INITIATOR;
     w4_rsc_29_0_config.initialize( interface_activity[98], {environment_path,".w4_rsc_29_0"}, interface_names[98]);
     w4_rsc_29_0_config.initiator_responder = INITIATOR;
     w4_rsc_30_0_config.initialize( interface_activity[99], {environment_path,".w4_rsc_30_0"}, interface_names[99]);
     w4_rsc_30_0_config.initiator_responder = INITIATOR;
     w4_rsc_31_0_config.initialize( interface_activity[100], {environment_path,".w4_rsc_31_0"}, interface_names[100]);
     w4_rsc_31_0_config.initiator_responder = INITIATOR;
     w4_rsc_32_0_config.initialize( interface_activity[101], {environment_path,".w4_rsc_32_0"}, interface_names[101]);
     w4_rsc_32_0_config.initiator_responder = INITIATOR;
     w4_rsc_33_0_config.initialize( interface_activity[102], {environment_path,".w4_rsc_33_0"}, interface_names[102]);
     w4_rsc_33_0_config.initiator_responder = INITIATOR;
     w4_rsc_34_0_config.initialize( interface_activity[103], {environment_path,".w4_rsc_34_0"}, interface_names[103]);
     w4_rsc_34_0_config.initiator_responder = INITIATOR;
     w4_rsc_35_0_config.initialize( interface_activity[104], {environment_path,".w4_rsc_35_0"}, interface_names[104]);
     w4_rsc_35_0_config.initiator_responder = INITIATOR;
     w4_rsc_36_0_config.initialize( interface_activity[105], {environment_path,".w4_rsc_36_0"}, interface_names[105]);
     w4_rsc_36_0_config.initiator_responder = INITIATOR;
     w4_rsc_37_0_config.initialize( interface_activity[106], {environment_path,".w4_rsc_37_0"}, interface_names[106]);
     w4_rsc_37_0_config.initiator_responder = INITIATOR;
     w4_rsc_38_0_config.initialize( interface_activity[107], {environment_path,".w4_rsc_38_0"}, interface_names[107]);
     w4_rsc_38_0_config.initiator_responder = INITIATOR;
     w4_rsc_39_0_config.initialize( interface_activity[108], {environment_path,".w4_rsc_39_0"}, interface_names[108]);
     w4_rsc_39_0_config.initiator_responder = INITIATOR;
     w4_rsc_40_0_config.initialize( interface_activity[109], {environment_path,".w4_rsc_40_0"}, interface_names[109]);
     w4_rsc_40_0_config.initiator_responder = INITIATOR;
     w4_rsc_41_0_config.initialize( interface_activity[110], {environment_path,".w4_rsc_41_0"}, interface_names[110]);
     w4_rsc_41_0_config.initiator_responder = INITIATOR;
     w4_rsc_42_0_config.initialize( interface_activity[111], {environment_path,".w4_rsc_42_0"}, interface_names[111]);
     w4_rsc_42_0_config.initiator_responder = INITIATOR;
     w4_rsc_43_0_config.initialize( interface_activity[112], {environment_path,".w4_rsc_43_0"}, interface_names[112]);
     w4_rsc_43_0_config.initiator_responder = INITIATOR;
     w4_rsc_44_0_config.initialize( interface_activity[113], {environment_path,".w4_rsc_44_0"}, interface_names[113]);
     w4_rsc_44_0_config.initiator_responder = INITIATOR;
     w4_rsc_45_0_config.initialize( interface_activity[114], {environment_path,".w4_rsc_45_0"}, interface_names[114]);
     w4_rsc_45_0_config.initiator_responder = INITIATOR;
     w4_rsc_46_0_config.initialize( interface_activity[115], {environment_path,".w4_rsc_46_0"}, interface_names[115]);
     w4_rsc_46_0_config.initiator_responder = INITIATOR;
     w4_rsc_47_0_config.initialize( interface_activity[116], {environment_path,".w4_rsc_47_0"}, interface_names[116]);
     w4_rsc_47_0_config.initiator_responder = INITIATOR;
     w4_rsc_48_0_config.initialize( interface_activity[117], {environment_path,".w4_rsc_48_0"}, interface_names[117]);
     w4_rsc_48_0_config.initiator_responder = INITIATOR;
     w4_rsc_49_0_config.initialize( interface_activity[118], {environment_path,".w4_rsc_49_0"}, interface_names[118]);
     w4_rsc_49_0_config.initiator_responder = INITIATOR;
     w4_rsc_50_0_config.initialize( interface_activity[119], {environment_path,".w4_rsc_50_0"}, interface_names[119]);
     w4_rsc_50_0_config.initiator_responder = INITIATOR;
     w4_rsc_51_0_config.initialize( interface_activity[120], {environment_path,".w4_rsc_51_0"}, interface_names[120]);
     w4_rsc_51_0_config.initiator_responder = INITIATOR;
     w4_rsc_52_0_config.initialize( interface_activity[121], {environment_path,".w4_rsc_52_0"}, interface_names[121]);
     w4_rsc_52_0_config.initiator_responder = INITIATOR;
     w4_rsc_53_0_config.initialize( interface_activity[122], {environment_path,".w4_rsc_53_0"}, interface_names[122]);
     w4_rsc_53_0_config.initiator_responder = INITIATOR;
     w4_rsc_54_0_config.initialize( interface_activity[123], {environment_path,".w4_rsc_54_0"}, interface_names[123]);
     w4_rsc_54_0_config.initiator_responder = INITIATOR;
     w4_rsc_55_0_config.initialize( interface_activity[124], {environment_path,".w4_rsc_55_0"}, interface_names[124]);
     w4_rsc_55_0_config.initiator_responder = INITIATOR;
     w4_rsc_56_0_config.initialize( interface_activity[125], {environment_path,".w4_rsc_56_0"}, interface_names[125]);
     w4_rsc_56_0_config.initiator_responder = INITIATOR;
     w4_rsc_57_0_config.initialize( interface_activity[126], {environment_path,".w4_rsc_57_0"}, interface_names[126]);
     w4_rsc_57_0_config.initiator_responder = INITIATOR;
     w4_rsc_58_0_config.initialize( interface_activity[127], {environment_path,".w4_rsc_58_0"}, interface_names[127]);
     w4_rsc_58_0_config.initiator_responder = INITIATOR;
     w4_rsc_59_0_config.initialize( interface_activity[128], {environment_path,".w4_rsc_59_0"}, interface_names[128]);
     w4_rsc_59_0_config.initiator_responder = INITIATOR;
     w4_rsc_60_0_config.initialize( interface_activity[129], {environment_path,".w4_rsc_60_0"}, interface_names[129]);
     w4_rsc_60_0_config.initiator_responder = INITIATOR;
     w4_rsc_61_0_config.initialize( interface_activity[130], {environment_path,".w4_rsc_61_0"}, interface_names[130]);
     w4_rsc_61_0_config.initiator_responder = INITIATOR;
     w4_rsc_62_0_config.initialize( interface_activity[131], {environment_path,".w4_rsc_62_0"}, interface_names[131]);
     w4_rsc_62_0_config.initiator_responder = INITIATOR;
     w4_rsc_63_0_config.initialize( interface_activity[132], {environment_path,".w4_rsc_63_0"}, interface_names[132]);
     w4_rsc_63_0_config.initiator_responder = INITIATOR;
     b4_rsc_config.initialize( interface_activity[133], {environment_path,".b4_rsc"}, interface_names[133]);
     b4_rsc_config.initiator_responder = INITIATOR;
     w6_rsc_0_0_config.initialize( interface_activity[134], {environment_path,".w6_rsc_0_0"}, interface_names[134]);
     w6_rsc_0_0_config.initiator_responder = INITIATOR;
     w6_rsc_1_0_config.initialize( interface_activity[135], {environment_path,".w6_rsc_1_0"}, interface_names[135]);
     w6_rsc_1_0_config.initiator_responder = INITIATOR;
     w6_rsc_2_0_config.initialize( interface_activity[136], {environment_path,".w6_rsc_2_0"}, interface_names[136]);
     w6_rsc_2_0_config.initiator_responder = INITIATOR;
     w6_rsc_3_0_config.initialize( interface_activity[137], {environment_path,".w6_rsc_3_0"}, interface_names[137]);
     w6_rsc_3_0_config.initiator_responder = INITIATOR;
     w6_rsc_4_0_config.initialize( interface_activity[138], {environment_path,".w6_rsc_4_0"}, interface_names[138]);
     w6_rsc_4_0_config.initiator_responder = INITIATOR;
     w6_rsc_5_0_config.initialize( interface_activity[139], {environment_path,".w6_rsc_5_0"}, interface_names[139]);
     w6_rsc_5_0_config.initiator_responder = INITIATOR;
     w6_rsc_6_0_config.initialize( interface_activity[140], {environment_path,".w6_rsc_6_0"}, interface_names[140]);
     w6_rsc_6_0_config.initiator_responder = INITIATOR;
     w6_rsc_7_0_config.initialize( interface_activity[141], {environment_path,".w6_rsc_7_0"}, interface_names[141]);
     w6_rsc_7_0_config.initiator_responder = INITIATOR;
     w6_rsc_8_0_config.initialize( interface_activity[142], {environment_path,".w6_rsc_8_0"}, interface_names[142]);
     w6_rsc_8_0_config.initiator_responder = INITIATOR;
     w6_rsc_9_0_config.initialize( interface_activity[143], {environment_path,".w6_rsc_9_0"}, interface_names[143]);
     w6_rsc_9_0_config.initiator_responder = INITIATOR;
     b6_rsc_config.initialize( interface_activity[144], {environment_path,".b6_rsc"}, interface_names[144]);
     b6_rsc_config.initiator_responder = INITIATOR;





  endfunction

endclass

