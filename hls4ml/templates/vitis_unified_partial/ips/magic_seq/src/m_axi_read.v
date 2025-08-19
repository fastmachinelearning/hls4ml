module m_axi_read #(
    parameter GLOB_ADDR_WIDTH = 32, // Address width for AXI interface
    parameter GLOB_DATA_WIDTH = 32, // Data width for AXI interface
    
    parameter BANK1_INDEX_WIDTH    =  3, // 2 ^ 2 = 4 slots
    parameter BANK1_SRC_ADDR_WIDTH = 32,
    parameter BANK1_SRC_SIZE_WIDTH = 26,
    parameter BANK1_DST_ADDR_WIDTH = 32,
    parameter BANK1_DST_SIZE_WIDTH = 26,
    parameter BANK1_STATUS_WIDTH   =  2,
    parameter BANK1_PROFILE_WIDTH  = 32,
    parameter BANK1_LD_MSK_WIDTH   =  8,
    parameter BANK1_ST_MSK_WIDTH   =  8,

    parameter BANK0_CONTROL_WIDTH = 4,
    parameter BANK0_STATUS_WIDTH  = 4,
    parameter BANK0_CNT_WIDTH     = BANK1_INDEX_WIDTH, /// the counter for the sequencer

    parameter DMA_INIT_TASK_CNT   = 8, //// (baseAddr0 + size0) + (baseAddr1 + size1)
    parameter DMA_EXEC_TASK_CNT   = 1
)(

input wire clk,
input  wire reset,
/**
------------ AXI4-Lite Master Read Interface 
it is supposed to connect to the DMA
*/

// Read Address Channel
output  wire [GLOB_ADDR_WIDTH-1:0]  M_AXI_ARADDR,
output  wire                        M_AXI_ARVALID,
input   wire                        M_AXI_ARREADY,

// Read Data Channel
input   wire  [GLOB_ADDR_WIDTH-1:0]   M_AXI_RDATA, ////// read data output acctually it is a reg
input   wire  [1:0]                   M_AXI_RRESP,
input   wire                          M_AXI_RVALID,
output  wire                          M_AXI_RREADY
);


assign M_AXI_ARADDR    = 0;
assign M_AXI_ARVALID   = 0;

assign M_AXI_RREADY    = 0;

endmodule