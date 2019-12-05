//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : Bench Sequence Base
// File            : mnist_mlp_bench_bench_sequence_base.svh
//----------------------------------------------------------------------
//
// Description: This file contains the top level and utility sequences
//     used by test_top. It can be extended to create derivative top
//     level sequences.
//
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
//

class mnist_mlp_bench_bench_sequence_base extends uvmf_sequence_base #(uvm_sequence_item);

  `uvm_object_utils( mnist_mlp_bench_bench_sequence_base );

  // UVMF_CHANGE_ME : Instantiate, construct, and start sequences as needed to create stimulus scenarios.

  // Instantiate sequences here
  typedef ccs_random_sequence #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))    input1_rsc_random_seq_t;
  input1_rsc_random_seq_t input1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))    output1_rsc_random_seq_t;
  output1_rsc_random_seq_t output1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))    const_size_in_1_rsc_random_seq_t;
  const_size_in_1_rsc_random_seq_t const_size_in_1_rsc_random_seq;
  typedef ccs_random_sequence #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))    const_size_out_1_rsc_random_seq_t;
  const_size_out_1_rsc_random_seq_t const_size_out_1_rsc_random_seq;

  // Sequencer handles for each active interface in the environment
  typedef ccs_transaction #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_transaction_t;
  uvm_sequencer #(input1_rsc_transaction_t)  input1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_transaction_t;
  uvm_sequencer #(output1_rsc_transaction_t)  output1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_transaction_t;
  uvm_sequencer #(const_size_in_1_rsc_transaction_t)  const_size_in_1_rsc_sequencer; 
  typedef ccs_transaction #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_transaction_t;
  uvm_sequencer #(const_size_out_1_rsc_transaction_t)  const_size_out_1_rsc_sequencer; 


  // Top level environment configuration handle
  typedef mnist_mlp_env_configuration mnist_mlp_env_configuration_t;
  mnist_mlp_env_configuration_t top_configuration;

  // Configuration handles to access interface BFM's
  ccs_configuration  #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_config;
  ccs_configuration  #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_config;

  // ****************************************************************************
  function new( string name = "" );
    super.new( name );
    // Retrieve the configuration handles from the uvm_config_db

    // Retrieve top level configuration handle
    if ( !uvm_config_db#(mnist_mlp_env_configuration_t)::get(null,UVMF_CONFIGURATIONS, "TOP_ENV_CONFIG",top_configuration) ) begin
      `uvm_fatal("CFG", "uvm_config_db#(mnist_mlp_env_configuration_t)::get cannot find resource TOP_ENV_CONFIG");
    end

    // Retrieve config handles for all agents
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , input1_rsc_BFM , input1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource input1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , output1_rsc_BFM , output1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource output1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , const_size_in_1_rsc_BFM , const_size_in_1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource const_size_in_1_rsc_BFM" )
    if( !uvm_config_db #( ccs_configuration #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1)) )::get( null , UVMF_CONFIGURATIONS , const_size_out_1_rsc_BFM , const_size_out_1_rsc_config ) ) 
      `uvm_fatal("CFG" , "uvm_config_db #( ccs_configuration )::get cannot find resource const_size_out_1_rsc_BFM" )

    // Assign the sequencer handles from the handles within agent configurations
    input1_rsc_sequencer = input1_rsc_config.get_sequencer();
    output1_rsc_sequencer = output1_rsc_config.get_sequencer();
    const_size_in_1_rsc_sequencer = const_size_in_1_rsc_config.get_sequencer();
    const_size_out_1_rsc_sequencer = const_size_out_1_rsc_config.get_sequencer();


  endfunction

  // ****************************************************************************
  virtual task body();
    // Construct sequences here
    input1_rsc_random_seq     = input1_rsc_random_seq_t::type_id::create("input1_rsc_random_seq");
    output1_rsc_random_seq     = output1_rsc_random_seq_t::type_id::create("output1_rsc_random_seq");
    const_size_in_1_rsc_random_seq     = const_size_in_1_rsc_random_seq_t::type_id::create("const_size_in_1_rsc_random_seq");
    const_size_out_1_rsc_random_seq     = const_size_out_1_rsc_random_seq_t::type_id::create("const_size_out_1_rsc_random_seq");
    fork
      input1_rsc_config.wait_for_reset();
      output1_rsc_config.wait_for_reset();
      const_size_in_1_rsc_config.wait_for_reset();
      const_size_out_1_rsc_config.wait_for_reset();
    join
    // Start RESPONDER sequences here
    fork
    join_none
    // Start INITIATOR sequences here
    fork
      repeat (25) input1_rsc_random_seq.start(input1_rsc_sequencer);
      repeat (25) output1_rsc_random_seq.start(output1_rsc_sequencer);
      repeat (25) const_size_in_1_rsc_random_seq.start(const_size_in_1_rsc_sequencer);
      repeat (25) const_size_out_1_rsc_random_seq.start(const_size_out_1_rsc_sequencer);
    join
    // UVMF_CHANGE_ME : Extend the simulation XXX number of clocks after 
    // the last sequence to allow for the last sequence item to flow 
    // through the design.
    fork
      input1_rsc_config.wait_for_num_clocks(400);
      output1_rsc_config.wait_for_num_clocks(400);
      const_size_in_1_rsc_config.wait_for_num_clocks(400);
      const_size_out_1_rsc_config.wait_for_num_clocks(400);
    join
  endtask

endclass

