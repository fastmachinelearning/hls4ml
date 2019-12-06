//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Bench level parameters package
// File            : mnist_mlp_bench_parameters_pkg.sv
//----------------------------------------------------------------------
// 
//                                         
//----------------------------------------------------------------------
//


package mnist_mlp_bench_parameters_pkg;

  import uvmf_base_pkg_hdl::*;


  // These parameters are used to uniquely identify each interface.  The monitor_bfm and
  // driver_bfm are placed into and retrieved from the uvm_config_db using these string 
  // names as the field_name. The parameter is also used to enable transaction viewing 
  // from the command line for selected interfaces using the UVM command line processing.
  parameter string input1_rsc_BFM  = "input1_rsc_BFM"; /* [0] */
  parameter string output1_rsc_BFM  = "output1_rsc_BFM"; /* [1] */
  parameter string const_size_in_1_rsc_BFM  = "const_size_in_1_rsc_BFM"; /* [2] */
  parameter string const_size_out_1_rsc_BFM  = "const_size_out_1_rsc_BFM"; /* [3] */
  parameter string w2_rsc_0_0_BFM  = "w2_rsc_0_0_BFM"; /* [4] */
  parameter string w2_rsc_1_0_BFM  = "w2_rsc_1_0_BFM"; /* [5] */
  parameter string w2_rsc_2_0_BFM  = "w2_rsc_2_0_BFM"; /* [6] */
  parameter string w2_rsc_3_0_BFM  = "w2_rsc_3_0_BFM"; /* [7] */
  parameter string w2_rsc_4_0_BFM  = "w2_rsc_4_0_BFM"; /* [8] */
  parameter string w2_rsc_5_0_BFM  = "w2_rsc_5_0_BFM"; /* [9] */
  parameter string w2_rsc_6_0_BFM  = "w2_rsc_6_0_BFM"; /* [10] */
  parameter string w2_rsc_7_0_BFM  = "w2_rsc_7_0_BFM"; /* [11] */
  parameter string w2_rsc_8_0_BFM  = "w2_rsc_8_0_BFM"; /* [12] */
  parameter string w2_rsc_9_0_BFM  = "w2_rsc_9_0_BFM"; /* [13] */
  parameter string w2_rsc_10_0_BFM  = "w2_rsc_10_0_BFM"; /* [14] */
  parameter string w2_rsc_11_0_BFM  = "w2_rsc_11_0_BFM"; /* [15] */
  parameter string w2_rsc_12_0_BFM  = "w2_rsc_12_0_BFM"; /* [16] */
  parameter string w2_rsc_13_0_BFM  = "w2_rsc_13_0_BFM"; /* [17] */
  parameter string w2_rsc_14_0_BFM  = "w2_rsc_14_0_BFM"; /* [18] */
  parameter string w2_rsc_15_0_BFM  = "w2_rsc_15_0_BFM"; /* [19] */
  parameter string w2_rsc_16_0_BFM  = "w2_rsc_16_0_BFM"; /* [20] */
  parameter string w2_rsc_17_0_BFM  = "w2_rsc_17_0_BFM"; /* [21] */
  parameter string w2_rsc_18_0_BFM  = "w2_rsc_18_0_BFM"; /* [22] */
  parameter string w2_rsc_19_0_BFM  = "w2_rsc_19_0_BFM"; /* [23] */
  parameter string w2_rsc_20_0_BFM  = "w2_rsc_20_0_BFM"; /* [24] */
  parameter string w2_rsc_21_0_BFM  = "w2_rsc_21_0_BFM"; /* [25] */
  parameter string w2_rsc_22_0_BFM  = "w2_rsc_22_0_BFM"; /* [26] */
  parameter string w2_rsc_23_0_BFM  = "w2_rsc_23_0_BFM"; /* [27] */
  parameter string w2_rsc_24_0_BFM  = "w2_rsc_24_0_BFM"; /* [28] */
  parameter string w2_rsc_25_0_BFM  = "w2_rsc_25_0_BFM"; /* [29] */
  parameter string w2_rsc_26_0_BFM  = "w2_rsc_26_0_BFM"; /* [30] */
  parameter string w2_rsc_27_0_BFM  = "w2_rsc_27_0_BFM"; /* [31] */
  parameter string w2_rsc_28_0_BFM  = "w2_rsc_28_0_BFM"; /* [32] */
  parameter string w2_rsc_29_0_BFM  = "w2_rsc_29_0_BFM"; /* [33] */
  parameter string w2_rsc_30_0_BFM  = "w2_rsc_30_0_BFM"; /* [34] */
  parameter string w2_rsc_31_0_BFM  = "w2_rsc_31_0_BFM"; /* [35] */
  parameter string w2_rsc_32_0_BFM  = "w2_rsc_32_0_BFM"; /* [36] */
  parameter string w2_rsc_33_0_BFM  = "w2_rsc_33_0_BFM"; /* [37] */
  parameter string w2_rsc_34_0_BFM  = "w2_rsc_34_0_BFM"; /* [38] */
  parameter string w2_rsc_35_0_BFM  = "w2_rsc_35_0_BFM"; /* [39] */
  parameter string w2_rsc_36_0_BFM  = "w2_rsc_36_0_BFM"; /* [40] */
  parameter string w2_rsc_37_0_BFM  = "w2_rsc_37_0_BFM"; /* [41] */
  parameter string w2_rsc_38_0_BFM  = "w2_rsc_38_0_BFM"; /* [42] */
  parameter string w2_rsc_39_0_BFM  = "w2_rsc_39_0_BFM"; /* [43] */
  parameter string w2_rsc_40_0_BFM  = "w2_rsc_40_0_BFM"; /* [44] */
  parameter string w2_rsc_41_0_BFM  = "w2_rsc_41_0_BFM"; /* [45] */
  parameter string w2_rsc_42_0_BFM  = "w2_rsc_42_0_BFM"; /* [46] */
  parameter string w2_rsc_43_0_BFM  = "w2_rsc_43_0_BFM"; /* [47] */
  parameter string w2_rsc_44_0_BFM  = "w2_rsc_44_0_BFM"; /* [48] */
  parameter string w2_rsc_45_0_BFM  = "w2_rsc_45_0_BFM"; /* [49] */
  parameter string w2_rsc_46_0_BFM  = "w2_rsc_46_0_BFM"; /* [50] */
  parameter string w2_rsc_47_0_BFM  = "w2_rsc_47_0_BFM"; /* [51] */
  parameter string w2_rsc_48_0_BFM  = "w2_rsc_48_0_BFM"; /* [52] */
  parameter string w2_rsc_49_0_BFM  = "w2_rsc_49_0_BFM"; /* [53] */
  parameter string w2_rsc_50_0_BFM  = "w2_rsc_50_0_BFM"; /* [54] */
  parameter string w2_rsc_51_0_BFM  = "w2_rsc_51_0_BFM"; /* [55] */
  parameter string w2_rsc_52_0_BFM  = "w2_rsc_52_0_BFM"; /* [56] */
  parameter string w2_rsc_53_0_BFM  = "w2_rsc_53_0_BFM"; /* [57] */
  parameter string w2_rsc_54_0_BFM  = "w2_rsc_54_0_BFM"; /* [58] */
  parameter string w2_rsc_55_0_BFM  = "w2_rsc_55_0_BFM"; /* [59] */
  parameter string w2_rsc_56_0_BFM  = "w2_rsc_56_0_BFM"; /* [60] */
  parameter string w2_rsc_57_0_BFM  = "w2_rsc_57_0_BFM"; /* [61] */
  parameter string w2_rsc_58_0_BFM  = "w2_rsc_58_0_BFM"; /* [62] */
  parameter string w2_rsc_59_0_BFM  = "w2_rsc_59_0_BFM"; /* [63] */
  parameter string w2_rsc_60_0_BFM  = "w2_rsc_60_0_BFM"; /* [64] */
  parameter string w2_rsc_61_0_BFM  = "w2_rsc_61_0_BFM"; /* [65] */
  parameter string w2_rsc_62_0_BFM  = "w2_rsc_62_0_BFM"; /* [66] */
  parameter string w2_rsc_63_0_BFM  = "w2_rsc_63_0_BFM"; /* [67] */
  parameter string b2_rsc_BFM  = "b2_rsc_BFM"; /* [68] */
  parameter string w4_rsc_0_0_BFM  = "w4_rsc_0_0_BFM"; /* [69] */
  parameter string w4_rsc_1_0_BFM  = "w4_rsc_1_0_BFM"; /* [70] */
  parameter string w4_rsc_2_0_BFM  = "w4_rsc_2_0_BFM"; /* [71] */
  parameter string w4_rsc_3_0_BFM  = "w4_rsc_3_0_BFM"; /* [72] */
  parameter string w4_rsc_4_0_BFM  = "w4_rsc_4_0_BFM"; /* [73] */
  parameter string w4_rsc_5_0_BFM  = "w4_rsc_5_0_BFM"; /* [74] */
  parameter string w4_rsc_6_0_BFM  = "w4_rsc_6_0_BFM"; /* [75] */
  parameter string w4_rsc_7_0_BFM  = "w4_rsc_7_0_BFM"; /* [76] */
  parameter string w4_rsc_8_0_BFM  = "w4_rsc_8_0_BFM"; /* [77] */
  parameter string w4_rsc_9_0_BFM  = "w4_rsc_9_0_BFM"; /* [78] */
  parameter string w4_rsc_10_0_BFM  = "w4_rsc_10_0_BFM"; /* [79] */
  parameter string w4_rsc_11_0_BFM  = "w4_rsc_11_0_BFM"; /* [80] */
  parameter string w4_rsc_12_0_BFM  = "w4_rsc_12_0_BFM"; /* [81] */
  parameter string w4_rsc_13_0_BFM  = "w4_rsc_13_0_BFM"; /* [82] */
  parameter string w4_rsc_14_0_BFM  = "w4_rsc_14_0_BFM"; /* [83] */
  parameter string w4_rsc_15_0_BFM  = "w4_rsc_15_0_BFM"; /* [84] */
  parameter string w4_rsc_16_0_BFM  = "w4_rsc_16_0_BFM"; /* [85] */
  parameter string w4_rsc_17_0_BFM  = "w4_rsc_17_0_BFM"; /* [86] */
  parameter string w4_rsc_18_0_BFM  = "w4_rsc_18_0_BFM"; /* [87] */
  parameter string w4_rsc_19_0_BFM  = "w4_rsc_19_0_BFM"; /* [88] */
  parameter string w4_rsc_20_0_BFM  = "w4_rsc_20_0_BFM"; /* [89] */
  parameter string w4_rsc_21_0_BFM  = "w4_rsc_21_0_BFM"; /* [90] */
  parameter string w4_rsc_22_0_BFM  = "w4_rsc_22_0_BFM"; /* [91] */
  parameter string w4_rsc_23_0_BFM  = "w4_rsc_23_0_BFM"; /* [92] */
  parameter string w4_rsc_24_0_BFM  = "w4_rsc_24_0_BFM"; /* [93] */
  parameter string w4_rsc_25_0_BFM  = "w4_rsc_25_0_BFM"; /* [94] */
  parameter string w4_rsc_26_0_BFM  = "w4_rsc_26_0_BFM"; /* [95] */
  parameter string w4_rsc_27_0_BFM  = "w4_rsc_27_0_BFM"; /* [96] */
  parameter string w4_rsc_28_0_BFM  = "w4_rsc_28_0_BFM"; /* [97] */
  parameter string w4_rsc_29_0_BFM  = "w4_rsc_29_0_BFM"; /* [98] */
  parameter string w4_rsc_30_0_BFM  = "w4_rsc_30_0_BFM"; /* [99] */
  parameter string w4_rsc_31_0_BFM  = "w4_rsc_31_0_BFM"; /* [100] */
  parameter string w4_rsc_32_0_BFM  = "w4_rsc_32_0_BFM"; /* [101] */
  parameter string w4_rsc_33_0_BFM  = "w4_rsc_33_0_BFM"; /* [102] */
  parameter string w4_rsc_34_0_BFM  = "w4_rsc_34_0_BFM"; /* [103] */
  parameter string w4_rsc_35_0_BFM  = "w4_rsc_35_0_BFM"; /* [104] */
  parameter string w4_rsc_36_0_BFM  = "w4_rsc_36_0_BFM"; /* [105] */
  parameter string w4_rsc_37_0_BFM  = "w4_rsc_37_0_BFM"; /* [106] */
  parameter string w4_rsc_38_0_BFM  = "w4_rsc_38_0_BFM"; /* [107] */
  parameter string w4_rsc_39_0_BFM  = "w4_rsc_39_0_BFM"; /* [108] */
  parameter string w4_rsc_40_0_BFM  = "w4_rsc_40_0_BFM"; /* [109] */
  parameter string w4_rsc_41_0_BFM  = "w4_rsc_41_0_BFM"; /* [110] */
  parameter string w4_rsc_42_0_BFM  = "w4_rsc_42_0_BFM"; /* [111] */
  parameter string w4_rsc_43_0_BFM  = "w4_rsc_43_0_BFM"; /* [112] */
  parameter string w4_rsc_44_0_BFM  = "w4_rsc_44_0_BFM"; /* [113] */
  parameter string w4_rsc_45_0_BFM  = "w4_rsc_45_0_BFM"; /* [114] */
  parameter string w4_rsc_46_0_BFM  = "w4_rsc_46_0_BFM"; /* [115] */
  parameter string w4_rsc_47_0_BFM  = "w4_rsc_47_0_BFM"; /* [116] */
  parameter string w4_rsc_48_0_BFM  = "w4_rsc_48_0_BFM"; /* [117] */
  parameter string w4_rsc_49_0_BFM  = "w4_rsc_49_0_BFM"; /* [118] */
  parameter string w4_rsc_50_0_BFM  = "w4_rsc_50_0_BFM"; /* [119] */
  parameter string w4_rsc_51_0_BFM  = "w4_rsc_51_0_BFM"; /* [120] */
  parameter string w4_rsc_52_0_BFM  = "w4_rsc_52_0_BFM"; /* [121] */
  parameter string w4_rsc_53_0_BFM  = "w4_rsc_53_0_BFM"; /* [122] */
  parameter string w4_rsc_54_0_BFM  = "w4_rsc_54_0_BFM"; /* [123] */
  parameter string w4_rsc_55_0_BFM  = "w4_rsc_55_0_BFM"; /* [124] */
  parameter string w4_rsc_56_0_BFM  = "w4_rsc_56_0_BFM"; /* [125] */
  parameter string w4_rsc_57_0_BFM  = "w4_rsc_57_0_BFM"; /* [126] */
  parameter string w4_rsc_58_0_BFM  = "w4_rsc_58_0_BFM"; /* [127] */
  parameter string w4_rsc_59_0_BFM  = "w4_rsc_59_0_BFM"; /* [128] */
  parameter string w4_rsc_60_0_BFM  = "w4_rsc_60_0_BFM"; /* [129] */
  parameter string w4_rsc_61_0_BFM  = "w4_rsc_61_0_BFM"; /* [130] */
  parameter string w4_rsc_62_0_BFM  = "w4_rsc_62_0_BFM"; /* [131] */
  parameter string w4_rsc_63_0_BFM  = "w4_rsc_63_0_BFM"; /* [132] */
  parameter string b4_rsc_BFM  = "b4_rsc_BFM"; /* [133] */
  parameter string w6_rsc_0_0_BFM  = "w6_rsc_0_0_BFM"; /* [134] */
  parameter string w6_rsc_1_0_BFM  = "w6_rsc_1_0_BFM"; /* [135] */
  parameter string w6_rsc_2_0_BFM  = "w6_rsc_2_0_BFM"; /* [136] */
  parameter string w6_rsc_3_0_BFM  = "w6_rsc_3_0_BFM"; /* [137] */
  parameter string w6_rsc_4_0_BFM  = "w6_rsc_4_0_BFM"; /* [138] */
  parameter string w6_rsc_5_0_BFM  = "w6_rsc_5_0_BFM"; /* [139] */
  parameter string w6_rsc_6_0_BFM  = "w6_rsc_6_0_BFM"; /* [140] */
  parameter string w6_rsc_7_0_BFM  = "w6_rsc_7_0_BFM"; /* [141] */
  parameter string w6_rsc_8_0_BFM  = "w6_rsc_8_0_BFM"; /* [142] */
  parameter string w6_rsc_9_0_BFM  = "w6_rsc_9_0_BFM"; /* [143] */
  parameter string b6_rsc_BFM  = "b6_rsc_BFM"; /* [144] */
endpackage

