//----------------------------------------------------------------------
//----------------------------------------------------------------------
// Created by      : giuseppe
// Creation Date   : 2019 Dec 04
// Created with uvmf_gen version 2019.1
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
// Project         : mnist_mlp_bench Simulation Bench 
// Unit            : HDL top level module
// File            : hdl_top.sv
//----------------------------------------------------------------------
//                                          
// Description: This top level module instantiates all synthesizable
//    static content.  This and tb_top.sv are the two top level modules
//    of the simulation.  
//
//    This module instantiates the following:
//        DUT: The Design Under Test
//        Interfaces:  Signal bundles that contain signals connected to DUT
//        Driver BFM's: BFM's that actively drive interface signals
//        Monitor BFM's: BFM's that passively monitor interface signals
//
//----------------------------------------------------------------------
//
//----------------------------------------------------------------------
//

import mnist_mlp_bench_parameters_pkg::*;
import uvmf_base_pkg_hdl::*;

module hdl_top;
  bit clk;
  // Instantiate a clk driver 
  initial begin
    clk = 0;
    #0ns;
    forever begin
      clk = ~clk;
      #5ns;
    end
  end

  bit rst;
  // Instantiate a rst driver
  initial begin
    rst = 1; 
    #25ns;
    rst =  0; 
  end

  // Instantiate the signal bundle, monitor bfm and driver bfm for each interface.
  // The signal bundle, _if, contains signals to be connected to the DUT.
  // The monitor, monitor_bfm, observes the bus, _if, and captures transactions.
  // The driver, driver_bfm, drives transactions onto the bus, _if.
  ccs_if #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_bus(.clk(clk), .rst(rst));
  ccs_if #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_bus(.clk(clk), .rst(rst));
  ccs_if #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_bus(.clk(clk), .rst(rst));
  ccs_if #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_bus(.clk(clk), .rst(rst));
  ccs_monitor_bfm #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_mon_bfm(input1_rsc_bus);
  ccs_monitor_bfm #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_mon_bfm(output1_rsc_bus);
  ccs_monitor_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_mon_bfm(const_size_in_1_rsc_bus);
  ccs_monitor_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_mon_bfm(const_size_out_1_rsc_bus);
  ccs_driver_bfm #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  input1_rsc_drv_bfm(input1_rsc_bus);
  ccs_driver_bfm #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  output1_rsc_drv_bfm(output1_rsc_bus);
  ccs_driver_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_in_1_rsc_drv_bfm(const_size_in_1_rsc_bus);
  ccs_driver_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  const_size_out_1_rsc_drv_bfm(const_size_out_1_rsc_bus);

  // UVMF_CHANGE_ME : Add DUT and connect to signals in _bus interfaces listed above
// Catapult RTL DUT -
   mnist_mlp mnist_mlp_INST(
      .clk(clk),
      .rst(rst),
      .input1_rsc_rdy(input1_rsc_bus.rdy),
      .input1_rsc_vld(input1_rsc_bus.vld),
      .input1_rsc_dat(input1_rsc_bus.dat),
      .output1_rsc_rdy(output1_rsc_bus.rdy),
      .output1_rsc_vld(output1_rsc_bus.vld),
      .output1_rsc_dat(output1_rsc_bus.dat),
      .const_size_in_1_rsc_vld(const_size_in_1_rsc_bus.vld),
      .const_size_in_1_rsc_dat(const_size_in_1_rsc_bus.dat),
      .const_size_out_1_rsc_vld(const_size_out_1_rsc_bus.vld),
      .const_size_out_1_rsc_dat(const_size_out_1_rsc_bus.dat)
      );

  initial begin      import uvm_pkg::uvm_config_db;
    // The monitor_bfm and driver_bfm for each interface is placed into the uvm_config_db.
    // They are placed into the uvm_config_db using the string names defined in the parameters package.
    // The string names are passed to the agent configurations by test_top through the top level configuration.
    // They are retrieved by the agents configuration class for use by the agent.
    uvm_config_db #( virtual ccs_monitor_bfm #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , input1_rsc_BFM , input1_rsc_mon_bfm ); 
    uvm_config_db #( virtual ccs_monitor_bfm #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , output1_rsc_BFM , output1_rsc_mon_bfm ); 
    uvm_config_db #( virtual ccs_monitor_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , const_size_in_1_rsc_BFM , const_size_in_1_rsc_mon_bfm ); 
    uvm_config_db #( virtual ccs_monitor_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , const_size_out_1_rsc_BFM , const_size_out_1_rsc_mon_bfm ); 
    uvm_config_db #( virtual ccs_driver_bfm #(.PROTOCOL_KIND(3),.WIDTH(14112),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , input1_rsc_BFM , input1_rsc_drv_bfm  );
    uvm_config_db #( virtual ccs_driver_bfm #(.PROTOCOL_KIND(3),.WIDTH(180),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , output1_rsc_BFM , output1_rsc_drv_bfm  );
    uvm_config_db #( virtual ccs_driver_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , const_size_in_1_rsc_BFM , const_size_in_1_rsc_drv_bfm  );
    uvm_config_db #( virtual ccs_driver_bfm #(.PROTOCOL_KIND(2),.WIDTH(16),.RESET_POLARITY(1))  )::set( null , UVMF_VIRTUAL_INTERFACES , const_size_out_1_rsc_BFM , const_size_out_1_rsc_drv_bfm  );
  end

endmodule

