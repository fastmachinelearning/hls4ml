//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp Environment 
// Unit            : mnist_mlp Environment
// File            : mnist_mlp_environment.svh
//----------------------------------------------------------------------
//                                          
// DESCRIPTION: This environment contains all agents, predictors and
// scoreboards required for the block level design.
//----------------------------------------------------------------------
//



class mnist_mlp_environment #(
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
  bit[1:0] const_size_out_1_rsc_PROTOCOL_KIND = 2)
  extends uvmf_environment_base #(
    .CONFIG_T( mnist_mlp_env_configuration
    #(
     .input1_rsc_WIDTH(input1_rsc_WIDTH),                                
     .input1_rsc_RESET_POLARITY(input1_rsc_RESET_POLARITY),                                
     .input1_rsc_PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND),                                
     .output1_rsc_WIDTH(output1_rsc_WIDTH),                                
     .output1_rsc_RESET_POLARITY(output1_rsc_RESET_POLARITY),                                
     .output1_rsc_PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND),                                
     .const_size_in_1_rsc_WIDTH(const_size_in_1_rsc_WIDTH),                                
     .const_size_in_1_rsc_RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),                                
     .const_size_in_1_rsc_PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND),                                
     .const_size_out_1_rsc_WIDTH(const_size_out_1_rsc_WIDTH),                                
     .const_size_out_1_rsc_RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),                                
     .const_size_out_1_rsc_PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)                                
    )
  ));
  `uvm_component_param_utils( mnist_mlp_environment #(
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





  typedef ccs_agent #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1)) input1_rsc_agent_t;
  input1_rsc_agent_t input1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1)) output1_rsc_agent_t;
  output1_rsc_agent_t output1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_in_1_rsc_agent_t;
  const_size_in_1_rsc_agent_t const_size_in_1_rsc;

  typedef ccs_agent #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) const_size_out_1_rsc_agent_t;
  const_size_out_1_rsc_agent_t const_size_out_1_rsc;




  typedef mnist_mlp_predictor  #(
                             .input1_rsc_WIDTH(input1_rsc_WIDTH),                                
                             .input1_rsc_RESET_POLARITY(input1_rsc_RESET_POLARITY),                                
                             .input1_rsc_PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND),                                
                             .output1_rsc_WIDTH(output1_rsc_WIDTH),                                
                             .output1_rsc_RESET_POLARITY(output1_rsc_RESET_POLARITY),                                
                             .output1_rsc_PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND),                                
                             .const_size_in_1_rsc_WIDTH(const_size_in_1_rsc_WIDTH),                                
                             .const_size_in_1_rsc_RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),                                
                             .const_size_in_1_rsc_PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND),                                
                             .const_size_out_1_rsc_WIDTH(const_size_out_1_rsc_WIDTH),                                
                             .const_size_out_1_rsc_RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),                                
                             .const_size_out_1_rsc_PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)                                
                             )  mnist_mlp_pred_t;
  mnist_mlp_pred_t mnist_mlp_pred;

  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))))  output1_rsc_sb_t;
  output1_rsc_sb_t output1_rsc_sb;
  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))))  const_size_in_1_rsc_sb_t;
  const_size_in_1_rsc_sb_t const_size_in_1_rsc_sb;
  typedef uvmf_catapult_scoreboard #(.T(ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))))  const_size_out_1_rsc_sb_t;
  const_size_out_1_rsc_sb_t const_size_out_1_rsc_sb;


// ****************************************************************************
// FUNCTION : new()
// This function is the standard SystemVerilog constructor.
//
  function new( string name = "", uvm_component parent = null );
    super.new( name, parent );
  endfunction

// ****************************************************************************
// FUNCTION: build_phase()
// This function builds all components within this environment.
//
  virtual function void build_phase(uvm_phase phase);
    super.build_phase(phase);
    input1_rsc = input1_rsc_agent_t::type_id::create("input1_rsc",this);
    input1_rsc.set_config(configuration.input1_rsc_config);
    output1_rsc = output1_rsc_agent_t::type_id::create("output1_rsc",this);
    output1_rsc.set_config(configuration.output1_rsc_config);
    const_size_in_1_rsc = const_size_in_1_rsc_agent_t::type_id::create("const_size_in_1_rsc",this);
    const_size_in_1_rsc.set_config(configuration.const_size_in_1_rsc_config);
    const_size_out_1_rsc = const_size_out_1_rsc_agent_t::type_id::create("const_size_out_1_rsc",this);
    const_size_out_1_rsc.set_config(configuration.const_size_out_1_rsc_config);
    mnist_mlp_pred = mnist_mlp_pred_t::type_id::create("mnist_mlp_pred",this);
    mnist_mlp_pred.configuration = configuration;
    output1_rsc_sb = output1_rsc_sb_t::type_id::create("output1_rsc_sb",this);
    const_size_in_1_rsc_sb = const_size_in_1_rsc_sb_t::type_id::create("const_size_in_1_rsc_sb",this);
    const_size_out_1_rsc_sb = const_size_out_1_rsc_sb_t::type_id::create("const_size_out_1_rsc_sb",this);
  endfunction

// ****************************************************************************
// FUNCTION: connect_phase()
// This function makes all connections within this environment.  Connections
// typically inclue agent to predictor, predictor to scoreboard and scoreboard
// to agent.
//
  virtual function void connect_phase(uvm_phase phase);
    super.connect_phase(phase);
    input1_rsc.monitored_ap.connect(mnist_mlp_pred.input1_rsc_ae);
    output1_rsc.monitored_ap.connect(output1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.output1_rsc_ap.connect(output1_rsc_sb.expected_analysis_export);
    const_size_in_1_rsc.monitored_ap.connect(const_size_in_1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.const_size_in_1_rsc_ap.connect(const_size_in_1_rsc_sb.expected_analysis_export);
    const_size_out_1_rsc.monitored_ap.connect(const_size_out_1_rsc_sb.actual_analysis_export);
    mnist_mlp_pred.const_size_out_1_rsc_ap.connect(const_size_out_1_rsc_sb.expected_analysis_export);
  endfunction

endclass

