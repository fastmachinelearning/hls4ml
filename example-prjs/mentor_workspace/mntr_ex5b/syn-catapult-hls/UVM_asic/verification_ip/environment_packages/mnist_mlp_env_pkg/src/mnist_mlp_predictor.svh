//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_predictor 
// Unit            : mnist_mlp_predictor 
// File            : mnist_mlp_predictor.svh
//----------------------------------------------------------------------
//----------------------------------------------------------------------
//
//
// DESCRIPTION: This analysis component contains analysis_exports for receiving
//   data and analysis_ports for sending data.
// 
//   This analysis component has the following analysis_exports that receive the 
//   listed transaction type.
//   
//     input1_rsc_ae receives transactions of type  ccs_transaction#(.WIDTH(input1_rsc_WIDTH),.RESET_POLARITY(input1_rsc_RESET_POLARITY),.PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND))  
//
//   This analysis component has the following analysis_ports that can broadcast 
//   the listed transaction type.
//
//     const_size_in_1_rsc_ap broadcasts transactions of type ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND)) 
//     const_size_out_1_rsc_ap broadcasts transactions of type ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)) 
//     output1_rsc_ap broadcasts transactions of type ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND)) 
//

import uvmc_pkg::*;

// ===========================================================================
// use UVM "imp" (implementation) macros to define unique target socket classname
// `uvm_tlm_b_target_socket_decl(_XXX)   Will implement "b_transport_XXX"  for the user
// `uvm_tlm_nb_target_socket_decl(_XXX)  Will implement "nb_transport_XXX" for the user
// `include "uvm_tlm_target_socket_decl.svh"
// =-=-=-=-=-=-=-=-=-=-=-=-=-= BEGIN INLINED INCLUDE
//-------------------------------------------------------------------------------------
//
// Macro to create a tlm target socket with a user-defined b_transport(), nb_transport_fw callback name
//
//-------------------------------------------------------------------------------------


////////////////////////////////////////////
/// b_transport_XXX IMP
////////////////////////////////////////////

`define UVM_TLM_B_TRANSPORT_IMP_DECL(SFX,imp, T, t, delay)     \
  task b_transport(T t, uvm_tlm_time delay);                   \
    if (delay == null) begin                                   \
       `uvm_error("PRED",                         \
                  {get_full_name(),                            \
                   ".b_transport() called with 'null' delay"}) \
       return;                                                 \
    end                                                        \
    imp.b_transport``SFX(t, delay);                            \
  endtask


`define uvm_tlm_b_target_socket_decl(SFX)                               \
class uvm_tlm_b_target_socket``SFX #(type IMP=int,                      \
                                     type T=uvm_tlm_generic_payload)    \
  extends uvm_tlm_b_target_socket_base #(T);                            \
                                                                        \
  local IMP m_imp;                                                      \
                                                                        \
  function new (string name, uvm_component parent, IMP imp = null);     \
    super.new (name, parent);                                           \
    if (imp == null) $cast(m_imp, parent);                              \
    else m_imp = imp;                                                   \
    if (m_imp == null)                                                  \
       `uvm_error("PRED", {"b_target socket ", name,          \
                                     " has no implementation"});        \
  endfunction                                                           \
                                                                        \
  function void connect(this_type provider);                            \
                                                                        \
    uvm_component c;                                                    \
                                                                        \
    super.connect(provider);                                            \
                                                                        \
    c = get_comp();                                                     \
    `uvm_error_context("PRED",                                 \
       "You cannot call connect() on a target termination socket", c)   \
  endfunction                                                           \
                                                                        \
  `UVM_TLM_B_TRANSPORT_IMP_DECL(SFX,m_imp, T, t, delay)                 \
endclass


////////////////////////////////////////////
/// nb_transport_fw_XXX IMP
////////////////////////////////////////////

`define UVM_TLM_NB_TRANSPORT_FW_IMP_DECL(SFX,imp, T, P, t, p, delay)                \
  function uvm_tlm_sync_e nb_transport_fw(T t, ref P p, input uvm_tlm_time delay);  \
    if (delay == null) begin                                                        \
       `uvm_error("PRED",                                              \
                  {get_full_name(),                                                 \
                   ".nb_transport_fw() called with 'null' delay"})                  \
       return UVM_TLM_COMPLETED;                                                    \
    end                                                                             \
    return imp.nb_transport_fw``SFX(t, p, delay);                                   \
  endfunction


`define uvm_tlm_nb_target_socket_decl(SFX)                              \
class uvm_tlm_nb_target_socket``SFX #(type IMP=int                      \
                                     ,type T=uvm_tlm_generic_payload    \
                                     ,type P=uvm_tlm_phase_e)           \
  extends uvm_tlm_nb_target_socket_base #(T,P);                         \
                                                                        \
  local IMP m_imp;                                                      \
                                                                        \
  function new (string name, uvm_component parent, IMP imp = null);     \
    super.new (name, parent);                                           \
    if (imp == null) $cast(m_imp, parent);                              \
    else m_imp = imp;                                                   \
    bw_port = new("bw_port", get_comp());                               \
    if (m_imp == null)                                                  \
       `uvm_error("PRED", {"nb_target socket ", name,         \
                                     " has no implementation"});        \
  endfunction                                                           \
                                                                        \
  function void connect(this_type provider);                            \
                                                                        \
    uvm_component c;                                                    \
                                                                        \
    super.connect(provider);                                            \
                                                                        \
    c = get_comp();                                                     \
    `uvm_error_context("PRED",                                 \
       "You cannot call connect() on a target termination socket", c)   \
  endfunction                                                           \
                                                                        \
  `UVM_TLM_NB_TRANSPORT_FW_IMP_DECL(SFX,m_imp, T, P, t, p, delay)       \
endclass
// =-=-=-=-=-=-=-=-=-=-=-=-=-= END INLINED INCLUDE



// Create target socket callbacks (b_transport, nb_transport_fw) with an appended name for uniqueness.
`uvm_tlm_nb_target_socket_decl(_gp_const_size_in_1_rsc_ap)
// Create target socket callbacks (b_transport, nb_transport_fw) with an appended name for uniqueness.
`uvm_tlm_nb_target_socket_decl(_gp_const_size_out_1_rsc_ap)
// Create target socket callbacks (b_transport, nb_transport_fw) with an appended name for uniqueness.
`uvm_tlm_nb_target_socket_decl(_gp_output1_rsc_ap)
// ===========================================================================

class mnist_mlp_predictor 
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
extends uvm_component;

  // Factory registration of this class
  `uvm_component_param_utils( mnist_mlp_predictor #(
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

  // Instantiate a handle to the configuration of the environment in which this component resides
  mnist_mlp_env_configuration   #(
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
             )  configuration;

  // Instantiate the analysis exports

  // Transaction sequence items from agent monitor for Catapult resource arrive on this export
  uvm_analysis_imp_input1_rsc_ae #(ccs_transaction#(.WIDTH(input1_rsc_WIDTH),.RESET_POLARITY(input1_rsc_RESET_POLARITY),.PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND)), mnist_mlp_predictor  #(
                              .input1_rsc_WIDTH(input1_rsc_WIDTH),                              .input1_rsc_RESET_POLARITY(input1_rsc_RESET_POLARITY),                              .input1_rsc_PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND),                              .output1_rsc_WIDTH(output1_rsc_WIDTH),                              .output1_rsc_RESET_POLARITY(output1_rsc_RESET_POLARITY),                              .output1_rsc_PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND),                              .const_size_in_1_rsc_WIDTH(const_size_in_1_rsc_WIDTH),                              .const_size_in_1_rsc_RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),                              .const_size_in_1_rsc_PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND),                              .const_size_out_1_rsc_WIDTH(const_size_out_1_rsc_WIDTH),                              .const_size_out_1_rsc_RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),                              .const_size_out_1_rsc_PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)                            ) ) input1_rsc_ae;
  // Transaction sequence items converted to GP are sent out the nb port to SysC
  uvm_tlm_nb_transport_fw_port #(uvm_tlm_generic_payload) gp_input1_rsc_ae;



  // Instantiate the analysis ports

  // GP items from analysis port of Catapult C++ Predictor arrive on this port (blocking)
  uvm_tlm_nb_target_socket_gp_const_size_in_1_rsc_ap #(mnist_mlp_predictor,uvm_tlm_generic_payload,uvm_tlm_phase_e)  gp_const_size_in_1_rsc_ap;
  // GP items are converted to transactions and sent out this analysis port to scoreboard
  uvm_analysis_port #(ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))) const_size_in_1_rsc_ap;


  // GP items from analysis port of Catapult C++ Predictor arrive on this port (blocking)
  uvm_tlm_nb_target_socket_gp_const_size_out_1_rsc_ap #(mnist_mlp_predictor,uvm_tlm_generic_payload,uvm_tlm_phase_e)  gp_const_size_out_1_rsc_ap;
  // GP items are converted to transactions and sent out this analysis port to scoreboard
  uvm_analysis_port #(ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))) const_size_out_1_rsc_ap;


  // GP items from analysis port of Catapult C++ Predictor arrive on this port (blocking)
  uvm_tlm_nb_target_socket_gp_output1_rsc_ap #(mnist_mlp_predictor,uvm_tlm_generic_payload,uvm_tlm_phase_e)  gp_output1_rsc_ap;
  // GP items are converted to transactions and sent out this analysis port to scoreboard
  uvm_analysis_port #(ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))) output1_rsc_ap;


  // Transaction variable for predicted values to be sent out const_size_in_1_rsc_ap
  // ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND)) const_size_in_1_rsc_ap_output_transaction;
  // Code for sending output transaction out through const_size_in_1_rsc_ap
  // const_size_in_1_rsc_ap.write(const_size_in_1_rsc_ap_output_transaction);

  // Transaction variable for predicted values to be sent out const_size_out_1_rsc_ap
  // ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)) const_size_out_1_rsc_ap_output_transaction;
  // Code for sending output transaction out through const_size_out_1_rsc_ap
  // const_size_out_1_rsc_ap.write(const_size_out_1_rsc_ap_output_transaction);

  // Transaction variable for predicted values to be sent out output1_rsc_ap
  // ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND)) output1_rsc_ap_output_transaction;
  // Code for sending output transaction out through output1_rsc_ap
  // output1_rsc_ap.write(output1_rsc_ap_output_transaction);


  // FUNCTION: new
  function new(string name, uvm_component parent);
     super.new(name,parent);
  endfunction

  // FUNCTION: build_phase
  virtual function void build_phase (uvm_phase phase);
    super.build_phase(phase);

    // Ports/Exports for transactions coming into predictor
    input1_rsc_ae = new("input1_rsc_ae", this);
    gp_input1_rsc_ae = new("gp_input1_rsc_ae", this);

    // Ports for transactions coming out of predictor
    gp_const_size_in_1_rsc_ap =new("gp_const_size_in_1_rsc_ap", this );
    const_size_in_1_rsc_ap =new("const_size_in_1_rsc_ap", this );
    gp_const_size_out_1_rsc_ap =new("gp_const_size_out_1_rsc_ap", this );
    const_size_out_1_rsc_ap =new("const_size_out_1_rsc_ap", this );
    gp_output1_rsc_ap =new("gp_output1_rsc_ap", this );
    output1_rsc_ap =new("output1_rsc_ap", this );

  endfunction

  // FUNCTION: connect_phase
  virtual function void connect_phase (uvm_phase phase);
    super.connect_phase(phase);
    // Connect tlm2 GP port to uvmc
    uvmc_tlm #(uvm_tlm_generic_payload)::connect(this.gp_input1_rsc_ae,"gp_input1_rsc_ae");
    uvmc_tlm #(uvm_tlm_generic_payload)::connect(this.gp_const_size_in_1_rsc_ap,"gp_const_size_in_1_rsc_ap");
    uvmc_tlm #(uvm_tlm_generic_payload)::connect(this.gp_const_size_out_1_rsc_ap,"gp_const_size_out_1_rsc_ap");
    uvmc_tlm #(uvm_tlm_generic_payload)::connect(this.gp_output1_rsc_ap,"gp_output1_rsc_ap");
  endfunction

  // FUNCTION: write_input1_rsc_ae
  // Transactions received through input1_rsc_ae initiate the execution of this function.
  // Convert to TLM Generic Payload and write out to port (non-blocking)
  //   phase always BEGIN_REQ (ignored by SystemC TLM code), status should always
  //   be returned as UVM_TLM_COMPLETED (guaranteed by SystemC TLM code).
  //
  virtual function void write_input1_rsc_ae(ccs_transaction#(.WIDTH(input1_rsc_WIDTH),.RESET_POLARITY(input1_rsc_RESET_POLARITY),.PROTOCOL_KIND(input1_rsc_PROTOCOL_KIND)) t);
    uvm_tlm_sync_e status;
    uvm_tlm_phase_e phase;
    uvm_tlm_time delay = new("del", 1e-12);
    `uvm_info("PRED",           "Transaction Receivied through input1_rsc_ae", UVM_HIGH)
    `uvm_info("PRED", $psprintf("            Data: %s",t.convert2string()), UVM_FULL)
    status = gp_input1_rsc_ae.nb_transport_fw(t.to_gp(),phase,delay);
    if ( status != UVM_TLM_COMPLETED )
      `uvm_error("PRED", "SystemC TLM did not return UVM_TLM_COMPLETED")
  endfunction



  //============================================================================================
  // FUNCTION: nb_transport_fw_gp_const_size_in_1_rsc_ap
  // Non-Blocking nb_transport_fw callback for gp_const_size_in_1_rsc_ap connection
  //  - executed when SystemC TLM2 wrapper writes to output analysis port "gp_const_size_in_1_rsc_ap_ap"
  //
  virtual function uvm_tlm_sync_e nb_transport_fw_gp_const_size_in_1_rsc_ap(uvm_tlm_generic_payload gp, ref uvm_tlm_phase_e phase, input uvm_tlm_time delay);
    int  unsigned  i;
    bit  [63:0]    addr       = gp.get_address();   // actually 64 bit address
    int  unsigned  size       = gp.get_data_length;
    byte unsigned  gpData[]   = new[size];
    ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND)) const_size_in_1_rsc_ap_output_transaction;
	 if (gp.is_write()) begin
      gp.get_data(gpData);
      // create new transaction object
      const_size_in_1_rsc_ap_output_transaction = ccs_transaction#(.WIDTH(const_size_in_1_rsc_WIDTH),.RESET_POLARITY(const_size_in_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_in_1_rsc_PROTOCOL_KIND))::type_id::create("const_size_in_1_rsc_ap_output_transaction");
      const_size_in_1_rsc_ap_output_transaction.from_gp(gp); // transfer GP data into transaction
      `uvm_info("PRED",           "Transaction Receivied through const_size_in_1_rsc_ap", UVM_MEDIUM)
      `uvm_info("PRED", $psprintf("            Data: %s",const_size_in_1_rsc_ap_output_transaction.convert2string()), UVM_HIGH);
      const_size_in_1_rsc_ap.write(const_size_in_1_rsc_ap_output_transaction); // write to analysis port
    end
    else begin
      $write("Debug nb_transport_fw_gp_const_size_in_1_rsc_ap: @ %t: READ - NYI", $time);
      $display("");
    end
    phase = END_RESP;
  endfunction: nb_transport_fw_gp_const_size_in_1_rsc_ap


  //============================================================================================
  // FUNCTION: nb_transport_fw_gp_const_size_out_1_rsc_ap
  // Non-Blocking nb_transport_fw callback for gp_const_size_out_1_rsc_ap connection
  //  - executed when SystemC TLM2 wrapper writes to output analysis port "gp_const_size_out_1_rsc_ap_ap"
  //
  virtual function uvm_tlm_sync_e nb_transport_fw_gp_const_size_out_1_rsc_ap(uvm_tlm_generic_payload gp, ref uvm_tlm_phase_e phase, input uvm_tlm_time delay);
    int  unsigned  i;
    bit  [63:0]    addr       = gp.get_address();   // actually 64 bit address
    int  unsigned  size       = gp.get_data_length;
    byte unsigned  gpData[]   = new[size];
    ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND)) const_size_out_1_rsc_ap_output_transaction;
	 if (gp.is_write()) begin
      gp.get_data(gpData);
      // create new transaction object
      const_size_out_1_rsc_ap_output_transaction = ccs_transaction#(.WIDTH(const_size_out_1_rsc_WIDTH),.RESET_POLARITY(const_size_out_1_rsc_RESET_POLARITY),.PROTOCOL_KIND(const_size_out_1_rsc_PROTOCOL_KIND))::type_id::create("const_size_out_1_rsc_ap_output_transaction");
      const_size_out_1_rsc_ap_output_transaction.from_gp(gp); // transfer GP data into transaction
      `uvm_info("PRED",           "Transaction Receivied through const_size_out_1_rsc_ap", UVM_MEDIUM)
      `uvm_info("PRED", $psprintf("            Data: %s",const_size_out_1_rsc_ap_output_transaction.convert2string()), UVM_HIGH);
      const_size_out_1_rsc_ap.write(const_size_out_1_rsc_ap_output_transaction); // write to analysis port
    end
    else begin
      $write("Debug nb_transport_fw_gp_const_size_out_1_rsc_ap: @ %t: READ - NYI", $time);
      $display("");
    end
    phase = END_RESP;
  endfunction: nb_transport_fw_gp_const_size_out_1_rsc_ap


  //============================================================================================
  // FUNCTION: nb_transport_fw_gp_output1_rsc_ap
  // Non-Blocking nb_transport_fw callback for gp_output1_rsc_ap connection
  //  - executed when SystemC TLM2 wrapper writes to output analysis port "gp_output1_rsc_ap_ap"
  //
  virtual function uvm_tlm_sync_e nb_transport_fw_gp_output1_rsc_ap(uvm_tlm_generic_payload gp, ref uvm_tlm_phase_e phase, input uvm_tlm_time delay);
    int  unsigned  i;
    bit  [63:0]    addr       = gp.get_address();   // actually 64 bit address
    int  unsigned  size       = gp.get_data_length;
    byte unsigned  gpData[]   = new[size];
    ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND)) output1_rsc_ap_output_transaction;
	 if (gp.is_write()) begin
      gp.get_data(gpData);
      // create new transaction object
      output1_rsc_ap_output_transaction = ccs_transaction#(.WIDTH(output1_rsc_WIDTH),.RESET_POLARITY(output1_rsc_RESET_POLARITY),.PROTOCOL_KIND(output1_rsc_PROTOCOL_KIND))::type_id::create("output1_rsc_ap_output_transaction");
      output1_rsc_ap_output_transaction.from_gp(gp); // transfer GP data into transaction
      `uvm_info("PRED",           "Transaction Receivied through output1_rsc_ap", UVM_MEDIUM)
      `uvm_info("PRED", $psprintf("            Data: %s",output1_rsc_ap_output_transaction.convert2string()), UVM_HIGH);
      output1_rsc_ap.write(output1_rsc_ap_output_transaction); // write to analysis port
    end
    else begin
      $write("Debug nb_transport_fw_gp_output1_rsc_ap: @ %t: READ - NYI", $time);
      $display("");
    end
    phase = END_RESP;
  endfunction: nb_transport_fw_gp_output1_rsc_ap


endclass 

