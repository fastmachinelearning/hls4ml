//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp environment agent
// Unit            : Environment HVL package
// File            : mnist_mlp_pkg.sv
//----------------------------------------------------------------------
//     
// PACKAGE: This file defines all of the files contained in the
//    environment package that will run on the host simulator.
//
// CONTAINS:
//     - <mnist_mlp_configuration.svh>
//     - <mnist_mlp_environment.svh>
//     - <mnist_mlp_env_sequence_base.svh>
//     - <mnist_mlp_predictor.svh>
//
// ****************************************************************************
// ****************************************************************************
//----------------------------------------------------------------------
//
package mnist_mlp_env_pkg;

  import uvm_pkg::*;
  `include "uvm_macros.svh"
  import uvmf_base_pkg::*;
  import ccs_pkg::*;
  import ccs_pkg_hdl::*;
 
  `uvm_analysis_imp_decl(_w2_rsc_6_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_56_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_52_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_15_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_47_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_62_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_20_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_20_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_24_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_26_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_39_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_14_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_18_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_7_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_48_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_8_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_3_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_58_0_ae)
  `uvm_analysis_imp_decl(_b2_rsc_ae)
  `uvm_analysis_imp_decl(_w4_rsc_7_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_27_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_13_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_23_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_0_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_5_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_16_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_5_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_9_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_8_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_62_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_36_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_60_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_22_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_32_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_40_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_6_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_37_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_63_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_9_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_60_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_54_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_39_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_5_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_37_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_43_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_33_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_10_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_33_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_31_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_7_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_61_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_4_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_50_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_22_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_61_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_2_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_40_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_21_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_11_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_58_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_51_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_25_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_30_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_1_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_12_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_27_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_44_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_63_0_ae)
  `uvm_analysis_imp_decl(_b6_rsc_ae)
  `uvm_analysis_imp_decl(_w4_rsc_10_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_19_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_23_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_3_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_3_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_45_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_31_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_4_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_42_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_17_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_14_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_50_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_46_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_16_0_ae)
  `uvm_analysis_imp_decl(_b4_rsc_ae)
  `uvm_analysis_imp_decl(_w2_rsc_53_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_15_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_29_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_46_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_52_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_35_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_42_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_54_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_32_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_28_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_28_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_36_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_11_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_45_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_34_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_57_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_38_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_29_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_30_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_34_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_53_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_12_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_8_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_48_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_18_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_44_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_21_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_49_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_26_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_6_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_49_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_41_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_17_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_4_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_19_0_ae)
  `uvm_analysis_imp_decl(_input1_rsc_ae)
  `uvm_analysis_imp_decl(_w2_rsc_59_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_2_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_9_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_59_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_55_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_51_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_38_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_47_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_2_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_43_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_1_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_41_0_ae)
  `uvm_analysis_imp_decl(_w6_rsc_1_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_13_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_0_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_57_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_0_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_55_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_25_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_35_0_ae)
  `uvm_analysis_imp_decl(_w2_rsc_24_0_ae)
  `uvm_analysis_imp_decl(_w4_rsc_56_0_ae)


  // Parameters defined as HVL parameters

  `include "src/mnist_mlp_env_typedefs.svh"
  `include "src/mnist_mlp_env_configuration.svh"
  `include "src/mnist_mlp_predictor.svh"
  `include "src/mnist_mlp_environment.svh"
  `include "src/mnist_mlp_env_sequence_base.svh"

// UVMF_CHANGE_ME : When adding new environment level sequences to the src directory
//    be sure to add the sequence file here so that it will be
//    compiled as part of the environment package.  Be sure to place
//    the new sequence after any base sequence of the new sequence.

endpackage

