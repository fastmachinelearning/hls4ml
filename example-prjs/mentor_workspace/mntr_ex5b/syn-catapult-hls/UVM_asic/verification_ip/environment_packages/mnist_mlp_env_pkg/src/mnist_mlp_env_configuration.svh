//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
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
             bit[1:0] const_size_out_1_rsc_PROTOCOL_KIND = 2                                
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
                           const_size_out_1_rsc_PROTOCOL_KIND
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
     "\n", const_size_out_1_rsc_config.convert2string


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





  endfunction

endclass

