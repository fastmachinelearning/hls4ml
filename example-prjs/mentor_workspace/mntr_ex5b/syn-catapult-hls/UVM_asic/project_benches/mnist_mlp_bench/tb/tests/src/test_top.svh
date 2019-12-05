//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Top level UVM test
// File            : test_top.svh
//----------------------------------------------------------------------
// Description: This top level UVM test is the base class for all
//     future tests created for this project.
//
//     This test class contains:
//          Configuration:  The top level configuration for the project.
//          Environment:    The top level environment for the project.
//          Top_level_sequence:  The top level sequence for the project.
//                                       f   
//----------------------------------------------------------------------
//

typedef mnist_mlp_env_configuration mnist_mlp_env_configuration_t;
typedef mnist_mlp_environment mnist_mlp_environment_t;

class test_top extends uvmf_test_base #(.CONFIG_T(mnist_mlp_env_configuration_t), 
                                        .ENV_T(mnist_mlp_environment_t), 
                                        .TOP_LEVEL_SEQ_T(mnist_mlp_bench_bench_sequence_base));

  `uvm_component_utils( test_top );


  string interface_names[] = {
    input1_rsc_BFM /* input1_rsc     [0] */ , 
    output1_rsc_BFM /* output1_rsc     [1] */ , 
    const_size_in_1_rsc_BFM /* const_size_in_1_rsc     [2] */ , 
    const_size_out_1_rsc_BFM /* const_size_out_1_rsc     [3] */ 
};

uvmf_active_passive_t interface_activities[] = { 
    ACTIVE /* input1_rsc     [0] */ , 
    ACTIVE /* output1_rsc     [1] */ , 
    ACTIVE /* const_size_in_1_rsc     [2] */ , 
    ACTIVE /* const_size_out_1_rsc     [3] */   };

  // ****************************************************************************
  // FUNCTION: new()
  // This is the standard system verilog constructor.  All components are 
  // constructed in the build_phase to allow factory overriding.
  //
  function new( string name = "", uvm_component parent = null );
     super.new( name ,parent );
  endfunction



  // ****************************************************************************
  // FUNCTION: build_phase()
  // The construction of the configuration and environment classes is done in
  // the build_phase of uvmf_test_base.  Once the configuraton and environment
  // classes are built then the initialize call is made to perform the
  // following: 
  //     Monitor and driver BFM virtual interface handle passing into agents
  //     Set the active/passive state for each agent
  // Once this build_phase completes, the build_phase of the environment is
  // executed which builds the agents.
  //
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    configuration.initialize(BLOCK, "uvm_test_top.environment", interface_names, null, interface_activities);
  endfunction

endclass
