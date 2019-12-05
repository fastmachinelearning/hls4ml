//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Test package
// File            : mnist_mlp_bench_tests_pkg.sv
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

package mnist_mlp_bench_tests_pkg;

   import uvm_pkg::*;
   import uvmf_base_pkg::*;
   import mnist_mlp_bench_parameters_pkg::*;
   import mnist_mlp_env_pkg::*;
   import mnist_mlp_bench_sequences_pkg::*;
  import ccs_pkg::*;
  import ccs_pkg_hdl::*;


   `include "uvm_macros.svh"

   `include "src/test_top.svh"
   `include "src/example_derived_test.svh"

// UVMF_CHANGE_ME : When adding new tests to the src directory
//    be sure to add the test file here so that it will be
//    compiled as part of the test package.  Be sure to place
//    the new test after any base tests of the new test.

endpackage

