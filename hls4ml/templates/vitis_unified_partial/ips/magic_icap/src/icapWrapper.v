`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/22/2025 06:11:40 PM
// Design Name: 
// Module Name: icapWrapper
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module icapWrap(
   input  CLK,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP avail" *)   output        AVAIL,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP o" *)       output [31:0] O,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP prdone" *)  output        PRDONE,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP prerror" *) output        PRERROR,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP csib" *)    input         CSIB,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP i" *)       input [31:0]  I,
  (* X_INTERFACE_INFO = "xilinx.com:interface:icap:1.0 ICAP rdwrb" *)   input         RDWRB
    );
    
    

// ICAPE3: Internal Configuration Access Port
//         UltraScale
// Xilinx HDL Language Template, version 2023.2

ICAPE3 #(
   .DEVICE_ID(32'h03628093),     // Specifies the pre-programmed Device ID value to be used for simulation
                                 // purposes.
   .ICAP_AUTO_SWITCH("DISABLE"), // Enable switch ICAP using sync word.
   .SIM_CFG_FILE_NAME("NONE")    // Specifies the Raw Bitstream (RBT) file to be parsed by the simulation
                                 // model.
)
ICAPE3_inst (
   .AVAIL(AVAIL),     // 1-bit output: Availability status of ICAP.
   .O(O),             // 32-bit output: Configuration data output bus.
   .PRDONE(PRDONE),   // 1-bit output: Indicates completion of Partial Reconfiguration.
   .PRERROR(PRERROR), // 1-bit output: Indicates error during Partial Reconfiguration.
   .CLK(CLK),         // 1-bit input: Clock input.
   .CSIB(CSIB),       // 1-bit input: Active-Low ICAP enable.
   .I(I),             // 32-bit input: Configuration data input bus.
   .RDWRB(RDWRB)      // 1-bit input: Read/Write Select input.
);

// End of ICAPE3_inst instantiation
    
    
endmodule
