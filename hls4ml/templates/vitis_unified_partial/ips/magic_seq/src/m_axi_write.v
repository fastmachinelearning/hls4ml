module m_axi_write #(
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

    parameter DMA_INIT_TASK_CNT   = 8, //// (reset interrupt + startReadChannel + baseAddr0 + size0) + (reset interrupt + startWriteChannel + baseAddr1 + size1)
    parameter DMA_EXEC_TASK_CNT   = 1
)(
    input  wire                   clk,
    input  wire                   reset,

    // AXI Lite Write Address Channel
    output  reg [GLOB_ADDR_WIDTH-1:0]  M_AXI_AWADDR, ///// actually it is wire
    output  wire                       M_AXI_AWVALID,
    input   wire                       M_AXI_AWREADY,

    // AXI Lite Write Data Channel
    output  reg [GLOB_DATA_WIDTH-1:0]  M_AXI_WDATA, ///// actually it is wire
    output  wire[(GLOB_DATA_WIDTH/8)-1:0] M_AXI_WSTRB,
    output  wire                   M_AXI_WVALID,
    input   wire                   M_AXI_WREADY,

    // AXI Lite Write Response Channel
    input  wire [1:0]             M_AXI_BRESP,
    input  wire                   M_AXI_BVALID,
    output wire                   M_AXI_BREADY,

    // dma base addr
    input  wire [GLOB_ADDR_WIDTH-1: 0]   ext_bank0_out_dmaBaseAddr,

    // slave input

    input   wire[DMA_INIT_TASK_CNT-1: 0] slaveInit   , ///// trigger slave dma to do somthing
    output  reg [DMA_INIT_TASK_CNT-1: 0] slaveFinInit,

    input   wire[DMA_EXEC_TASK_CNT-1: 0] slaveStartExec      ,
    output  reg [DMA_EXEC_TASK_CNT-1: 0]  slaveStartExecAccept, ///// the slave dma is ready to start

    input wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_src_addr,      // actually it is a reg
    input wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_src_size,      // actually it is a reg
    input wire [BANK1_DST_ADDR_WIDTH -1:0] slave_bank1_out_des_addr,
    input wire [BANK1_DST_SIZE_WIDTH -1:0] slave_bank1_out_des_size,
    input wire [BANK1_STATUS_WIDTH   -1:0] slave_bank1_out_status  ,      // actually it is a reg
    input wire [BANK1_PROFILE_WIDTH  -1:0] slave_bank1_out_profile        // actually it is a reg

);


/**
* This module supposed to connet to dma
*
*
**/

//////// READ CHANNEL

wire[GLOB_ADDR_WIDTH-1: 0] dmSrcStatusADDR    = ext_bank0_out_dmaBaseAddr + 32'h04;

wire[GLOB_ADDR_WIDTH-1: 0] dmaSrcCtrlADDR     = ext_bank0_out_dmaBaseAddr + 32'h00;
wire[GLOB_ADDR_WIDTH-1: 0] dmaSrcDataAddrADDR = ext_bank0_out_dmaBaseAddr + 32'h18;
wire[GLOB_ADDR_WIDTH-1: 0] dmaSrcDataSizeADDR = ext_bank0_out_dmaBaseAddr + 32'h28;

//////// WRITE CHANNEL

wire[GLOB_ADDR_WIDTH-1: 0] dmDesStatusADDR    = ext_bank0_out_dmaBaseAddr + 32'h34;

wire[GLOB_ADDR_WIDTH-1: 0] dmaDesCtrlADDR     = ext_bank0_out_dmaBaseAddr + 32'h30;
wire[GLOB_ADDR_WIDTH-1: 0] dmaDesDataAddrADDR = ext_bank0_out_dmaBaseAddr + 32'h48;
wire[GLOB_ADDR_WIDTH-1: 0] dmaDesDataSizeADDR = ext_bank0_out_dmaBaseAddr + 32'h58;


localparam STATUS_IDLE   = 4'b0000;
localparam STATUS_WADDR  = 4'b0001;
localparam STATUS_WDATA  = 4'b0010;
localparam STATUS_RESP   = 4'b0100;
localparam STATUS_UNLOCK = 4'b1000;

/**
control main state machine
*/

reg[3:0] state;

always @(posedge clk or negedge reset) begin

    if (~reset)begin
        state = STATUS_IDLE;
    end else begin
        case(state) 
            STATUS_IDLE: begin
                if ( (slaveInit != 0) | (slaveStartExec != 0)) begin state = STATUS_WADDR; end
            end
            STATUS_WADDR: begin
                if (M_AXI_AWREADY) begin state = STATUS_WDATA; end
            end
            STATUS_WDATA: begin
                if (M_AXI_WREADY) begin state = STATUS_RESP; end
            end
            STATUS_RESP: begin
                if (M_AXI_BVALID) begin state = STATUS_UNLOCK; end
            end
            STATUS_UNLOCK: begin
                state = STATUS_IDLE;
            end

            default: begin
                state = STATUS_IDLE;
            end
        endcase
    end 
end

//// address channel
assign M_AXI_AWVALID = (state == STATUS_WADDR);
//// data channel
assign M_AXI_WSTRB   = 4'b1111;
assign M_AXI_WVALID  = (state == STATUS_WDATA);
//// resChannel
assign M_AXI_BREADY  = (state == STATUS_RESP);

///// manage address

always @ (*) begin

    M_AXI_AWADDR = 0;
    M_AXI_WDATA  = 0;

    slaveFinInit = 0;
    slaveStartExecAccept = 0;

    if (slaveInit != 0)begin

        if (state == STATUS_UNLOCK)begin //// STATUS_UNLOCK is one cycle
            slaveFinInit = slaveInit;
        end

        case(slaveInit)

            8'b00000001: begin
                        M_AXI_AWADDR = dmSrcStatusADDR;
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - 13){1'b0}}, 13'b1_0000_0000_0000}; //// start command
            end
            8'b00000010: begin
                        M_AXI_AWADDR = dmDesStatusADDR;
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - 13){1'b0}}, 13'b1_0000_0000_0000}; //// start command
            end
        //////////////////// set READ (RUN/SRCADDR/SRCSIZE)
            8'b00000100: begin
                        M_AXI_AWADDR = dmaSrcCtrlADDR;
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - 13){1'b0}}, 13'b1_0000_0000_0001}; //// start command
                    end
            8'b00001000: begin 
                        M_AXI_AWADDR = dmaSrcDataAddrADDR; 
                        M_AXI_WDATA  = slave_bank1_out_src_addr;
                        
                    end
            8'b00010000: begin 
                        M_AXI_AWADDR = dmaSrcDataSizeADDR; 
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - BANK1_DST_SIZE_WIDTH){1'b0}}, slave_bank1_out_src_size}; 
                    end
            //////////////////// set WRITE (RUN/DESADDR/DESSIZE)
            8'b00100000: begin
                        M_AXI_AWADDR = dmaDesCtrlADDR;
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - 13){1'b0}}, 13'b1_0000_0000_0001}; //// start command
                    end
            8'b01000000: begin 
                        M_AXI_AWADDR = dmaDesDataAddrADDR; 
                        M_AXI_WDATA  = slave_bank1_out_des_addr; 
                    end
            8'b10000000: begin 
                        M_AXI_AWADDR = dmaDesDataSizeADDR; 
                        M_AXI_WDATA  = {{(GLOB_DATA_WIDTH - BANK1_DST_SIZE_WIDTH){1'b0}}, slave_bank1_out_des_size}; 
                    end
            default: begin 
                        M_AXI_AWADDR          = 0;
                        M_AXI_WDATA           = 0;
                        slaveFinInit          = 0;
                        slaveStartExecAccept  = 0;        
                    end
        endcase
    end
    // end else if (slaveStartExec != 0) begin
    //         M_AXI_AWADDR = dmaCtrlAddrADDR;
    //         M_AXI_WDATA  = 1;
    //         if (state == STATUS_UNLOCK)begin
    //         slaveFinInit = slaveInit;
    //         end
    // end
    
end

endmodule