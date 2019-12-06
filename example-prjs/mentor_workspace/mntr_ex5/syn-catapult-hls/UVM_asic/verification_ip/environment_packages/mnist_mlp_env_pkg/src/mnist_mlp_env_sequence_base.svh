//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 05
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp Environment 
// Unit            : Environment Sequence Base
// File            : mnist_mlp_env_sequence_base.svh
//----------------------------------------------------------------------
//                                          
// DESCRIPTION: This file contains environment level sequences that will
//    be reused from block to top level simulations.
//
//----------------------------------------------------------------------
//
class mnist_mlp_env_sequence_base extends uvmf_sequence_base #(uvm_sequence_item);

  `uvm_object_utils( mnist_mlp_env_sequence_base );

  
  function new(string name = "" );
    super.new(name);
  endfunction

endclass

