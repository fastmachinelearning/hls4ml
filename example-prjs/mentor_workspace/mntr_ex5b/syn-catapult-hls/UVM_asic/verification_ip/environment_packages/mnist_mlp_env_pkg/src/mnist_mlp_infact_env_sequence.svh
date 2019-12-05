//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp Environment
// Unit            : Environment infact sequence
// File            : mnist_mlp_infact_env_sequence.svh
//----------------------------------------------------------------------
//     
// DESCRIPTION: 
// This sequences is a place holder for the infact sequence at the 
// environment level which will generated desired scenarios without redundancy.
//
// ****************************************************************************
// 
class mnist_mlp_infact_env_sequence extends mnist_mlp_env_sequence_base;

  // declaration macros
  `uvm_object_utils(mnist_mlp_infact_env_sequence)

//*****************************************************************
  function new(string name = "");
    super.new(name);
  endfunction: new

endclass: mnist_mlp_infact_env_sequence
//----------------------------------------------------------------------
//
