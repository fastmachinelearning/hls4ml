//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Test package
// File            : mnist_mlp_bench_test_pkg.sv
//----------------------------------------------------------------------
//
// DESCRIPTION: This package contains all tests currently written for
//     the simulation project.  Once compiled, any test can be selected
//     from the vsim command line using +UVM_TESTNAME=yourTestNameHere
//
// CONTAINS:
//     -<test_top>
//     -<example_derived_test>
//
//----------------------------------------------------------------------
//

package mnist_mlp_bench_test_pkg;

   import uvm_pkg::*;
`ifdef QUESTA
   import questa_uvm_pkg::*;
`endif
   import uvmf_base_pkg::*;
   import mnist_mlp_bench_parameters_pkg::*;
   import mnist_mlp_env_pkg::*;
   import mnist_mlp_bench_sequences_pkg::*;


   `include "uvm_macros.svh"

   `include "src/test_top.svh"
   `include "src/example_derived_test.svh"

endpackage

