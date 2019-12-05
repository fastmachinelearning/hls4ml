//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
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
endpackage

