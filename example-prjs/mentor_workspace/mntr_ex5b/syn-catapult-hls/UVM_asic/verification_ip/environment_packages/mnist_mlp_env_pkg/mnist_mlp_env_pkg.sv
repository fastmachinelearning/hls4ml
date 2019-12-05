//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
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
 
  `uvm_analysis_imp_decl(_input1_rsc_ae)


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

