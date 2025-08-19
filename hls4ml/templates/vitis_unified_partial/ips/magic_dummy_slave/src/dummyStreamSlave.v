module DummyStreamSlave #
(
    parameter integer DATA_WIDTH        = 32, 
    parameter integer STORAGE_IDX_WIDTH = 10,     //// 4 Kb
    parameter [0:0]   SINK_MODE         = 0
)
(
    // AXIS Slave Interface   store in terface
    input  wire [DATA_WIDTH-1:0]     S_AXI_TDATA,
//    input  wire [DATA_WIDTH/8-1:0]   S_AXI_TKEEP,  // <= tkeep added
    input  wire                      S_AXI_TVALID,
    output wire                      S_AXI_TREADY,
    input  wire                      S_AXI_TLAST,

    input   wire clk,
    input   wire reset
);

assign S_AXI_TREADY  = SINK_MODE; // Dummy implementation, always not ready

endmodule