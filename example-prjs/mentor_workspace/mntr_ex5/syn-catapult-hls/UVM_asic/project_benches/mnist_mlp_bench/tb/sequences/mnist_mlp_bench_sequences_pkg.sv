//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Sequences Package
// File            : mnist_mlp_bench_sequences_pkg.sv
//----------------------------------------------------------------------
//
// DESCRIPTION: This package includes all high level sequence classes used 
//     in the environment.  These include utility sequences and top
//     level sequences.
//
// CONTAINS:
//     -<mnist_mlp_bench_sequence_base>
//     -<example_derived_test_sequence>
//
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
//

package mnist_mlp_bench_sequences_pkg;
  import uvm_pkg::*;
  import uvmf_base_pkg::*;
  import ccs_pkg::*;
  import ccs_pkg_hdl::*;
  import mnist_mlp_bench_parameters_pkg::*;
  import mnist_mlp_env_pkg::*;
  `include "uvm_macros.svh"
  `include "src/mnist_mlp_bench_bench_sequence_base.svh"
  `include "src/example_derived_test_sequence.svh"
endpackage


// UVMF_CHANGE_ME : When adding new sequences to the src directory
//    be sure to add the sequence file here so that it will be
//    compiled as part of the sequence package.  Be sure to place
//    the new sequence after any base sequences of the new sequence.

