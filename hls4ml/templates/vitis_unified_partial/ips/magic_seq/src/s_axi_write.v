module s_axi_write #(
    parameter GLOB_ADDR_WIDTH = 32, // Address width for AXI interface
    parameter GLOB_DATA_WIDTH = 32, // Data width for AXI interface

    parameter ADDR_WIDTH = 16, // Address width for AXI interface
    parameter DATA_WIDTH = 32, // Data width for AXI interface
    
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
    parameter BANK0_INTR_WIDTH    = 1, /// the interrupt for the sequencer
    parameter BANK0_ROUNDTRIP_WIDTH = 16 /// the round trip counter for the sequencer


)(
    input  wire                   clk,
    input  wire                   reset,

    // AXI Lite Write Address Channel
    input  wire [ADDR_WIDTH-1:0]  S_AXI_AWADDR,
    input  wire                   S_AXI_AWVALID,
    output wire                   S_AXI_AWREADY,

    // AXI Lite Write Data Channel
    input  wire [DATA_WIDTH-1:0]  S_AXI_WDATA,
    input  wire [(DATA_WIDTH/8)-1:0] S_AXI_WSTRB,
    input  wire                   S_AXI_WVALID,
    output wire                   S_AXI_WREADY,

    // AXI Lite Write Response Channel
    output wire [1:0]             S_AXI_BRESP,
    output wire                   S_AXI_BVALID,
    input  wire                   S_AXI_BREADY,

    //// bank1 interconnect
    output wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_inp_index,       // actually it is a wire
    output wire [BANK1_SRC_ADDR_WIDTH -1:0] ext_bank1_inp_src_addr,    // actually it is a wire
    output wire [BANK1_SRC_SIZE_WIDTH -1:0] ext_bank1_inp_src_size,    // actually it is a wire
    output wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_inp_des_addr,    // actually it is a wire
    output wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_inp_des_size,    // actually it is a wire
    output wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_inp_status,      // actually it is a wire
    output wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_inp_profile,     // actually it is a wire
    output wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_inp_ld_mask,
    output wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_mask,
    output wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_inp_st_intr_mask_abs,


    output reg ext_bank1_set_src_addr,             // actually it is wire
    output reg ext_bank1_set_src_size,             // actually it is wire
    output reg ext_bank1_set_des_addr,             // actually it is wire
    output reg ext_bank1_set_des_size,             // actually it is wire
    output reg ext_bank1_set_status,               // actually it is wire
    output reg ext_bank1_set_profile,              // actually it is wire
    output reg ext_bank1_set_ld_mask,          // actually it is wire
    output reg ext_bank1_set_st_mask,          // actually it is wire
    output reg ext_bank1_set_st_intr_mask_abs, // actually it is wire

    //// bank0 interconnect
    output wire [BANK0_CONTROL_WIDTH-1:0] ext_bank0_inp_control, /// set control data
    output reg                            ext_bank0_set_control, /// set control signal
    output wire [BANK0_CNT_WIDTH-1:0]     ext_bank0_inp_endCnt,      ///
    output reg                            ext_bank0_set_endCnt,      ///

    
    output wire [GLOB_ADDR_WIDTH-1: 0]    ext_bank0_inp_dmaBaseAddr,
    output reg                            ext_bank0_set_dmaBaseAddr,
    output wire [GLOB_ADDR_WIDTH-1: 0]    ext_bank0_inp_dfxCtrlAddr,
    output reg                           ext_bank0_set_dfxCtrlAddr,

    
    output wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intrEna, //// input data for the interrupt counter
    output reg                           ext_bank0_set_intrEna, //// set the interrupt counter ONLY when the system is in shutdown state

    output wire [BANK0_INTR_WIDTH-1: 0]  ext_bank0_inp_intr, //// input data for the interrupt counter
    output reg                           ext_bank0_set_intr, //// set the interrupt counter ONLY when the system is in shutdown state
    
    output wire [BANK0_ROUNDTRIP_WIDTH-1: 0]  ext_bank0_inp_roundTrip, /// input data for the round trip counter
    output reg                                ext_bank0_set_roundTrip /// set the round trip counter ONLY when the system is in shutdown state


);


always @(*)begin
    case(S_AXI_WSTRB)
        default: begin end
    endcase
end



localparam ST_IDLE = 3'b000;
localparam ST_DATA = 3'b001;
localparam ST_RESP = 3'b010;

reg [2:0] state;
reg [ADDR_WIDTH-1:0] write_addr;

////////// main control state machine

always @(posedge clk or negedge reset ) begin

    if (~reset) begin
        state <= ST_IDLE;
        write_addr <= 0;
    end else begin
        case (state)
            ST_IDLE: begin
                if (S_AXI_AWVALID) begin
                    write_addr <= S_AXI_AWADDR;
                    state <= ST_DATA;
                end
            end

            ST_DATA: begin
                if (S_AXI_WVALID) begin
                    // Here you would typically write the data to the appropriate register or memory location
                    // For this example, we just move to the response state
                    state <= ST_RESP;
                end
            end

            ST_RESP: begin
                if (S_AXI_BREADY) begin
                    state <= ST_IDLE; // Go back to idle after response is acknowledged
                end
            end

            default: state <= ST_IDLE; // Default case to avoid latches

        endcase
    end
    
end

assign S_AXI_AWREADY = (state == ST_IDLE);
assign S_AXI_WREADY  = (state == ST_DATA);
assign S_AXI_BRESP   = 2'b00; // OKAY response
assign S_AXI_BVALID  = (state == ST_RESP);

/////////// writing to bank1 wiring

/////////// bank1 data wiring 

assign ext_bank1_inp_index             = write_addr[BANK1_INDEX_WIDTH + 6 - 1: 6]; /// the row in slot table
assign ext_bank1_inp_src_addr          = S_AXI_WDATA[BANK1_SRC_ADDR_WIDTH-1  : 0];
assign ext_bank1_inp_src_size          = S_AXI_WDATA[BANK1_SRC_SIZE_WIDTH-1  : 0];
assign ext_bank1_inp_des_addr          = S_AXI_WDATA[BANK1_DST_ADDR_WIDTH-1  : 0];
assign ext_bank1_inp_des_size          = S_AXI_WDATA[BANK1_DST_SIZE_WIDTH-1  : 0];
assign ext_bank1_inp_status            = S_AXI_WDATA[BANK1_STATUS_WIDTH  -1  : 0];
assign ext_bank1_inp_profile           = S_AXI_WDATA[BANK1_PROFILE_WIDTH -1  : 0];
assign ext_bank1_inp_ld_mask           = S_AXI_WDATA[BANK1_LD_MSK_WIDTH  -1  : 0];
assign ext_bank1_inp_st_mask           = S_AXI_WDATA[BANK1_ST_MSK_WIDTH  -1  : 0];
assign ext_bank1_inp_st_intr_mask_abs  = S_AXI_WDATA[BANK1_ST_MSK_WIDTH  -1  : 0];

//////////// bank0 data wiring

assign ext_bank0_inp_control        = S_AXI_WDATA[BANK0_CONTROL_WIDTH-1:0]; /// set control data
assign ext_bank0_inp_endCnt         = S_AXI_WDATA[BANK0_CNT_WIDTH    -1:0]; /// set end count

assign ext_bank0_inp_dmaBaseAddr    = S_AXI_WDATA[GLOB_ADDR_WIDTH    -1:0]; 
assign ext_bank0_inp_dfxCtrlAddr    = S_AXI_WDATA[GLOB_ADDR_WIDTH    -1:0];

assign ext_bank0_inp_intrEna        = S_AXI_WDATA[BANK0_INTR_WIDTH   -1:0]; /// input data for the interrupt Ena counter
assign ext_bank0_inp_intr           = S_AXI_WDATA[BANK0_INTR_WIDTH   -1:0]; /// input data for the interrupt counter
assign ext_bank0_inp_roundTrip      = S_AXI_WDATA[BANK0_ROUNDTRIP_WIDTH-1:0]; /// input data for the round trip counter

/////////// block control write signals

always @(*) begin
    ext_bank1_set_src_addr    = 0; // Default value
    ext_bank1_set_src_size    = 0; // Default value
    ext_bank1_set_des_addr    = 0; // Default value
    ext_bank1_set_des_size    = 0; // Default value
    ext_bank1_set_status      = 0; // Default value
    ext_bank1_set_profile     = 0; // Default value
    ext_bank1_set_ld_mask            = 0;
    ext_bank1_set_st_mask            = 0;
    ext_bank1_set_st_intr_mask_abs   = 0;

    

    ext_bank0_set_control     = 0; // Default value
    ext_bank0_set_endCnt      = 0; // Default value
    ext_bank0_set_dmaBaseAddr = 0; // Default value
    ext_bank0_set_dfxCtrlAddr = 0; // Default value
    ext_bank0_set_intrEna     = 0; // Default value
    ext_bank0_set_intr        = 0; // Default value
    ext_bank0_set_roundTrip   = 0; // Default value
    

    if (state == ST_DATA) begin
        case (write_addr[15:14])
            2'b00: begin
                case (write_addr[13:6]) // Address bits 13 to 6 determine the slot
                    8'h00: begin ext_bank0_set_control     = 1; end
                    /// cannot write to status register
                    8'h03: begin ext_bank0_set_endCnt      = 1; end
                    8'h04: begin ext_bank0_set_dmaBaseAddr = 1; end
                    8'h05: begin ext_bank0_set_dfxCtrlAddr = 1; end
                    8'h06: begin ext_bank0_set_intrEna     = 1; end // set interrupt enable
                    8'h07: begin ext_bank0_set_intr        = 1; end // set interrupt
                    8'h08: begin ext_bank0_set_roundTrip   = 1; end // set round trip counter
                    default: begin end
                endcase
            end

            2'b01: begin

                case (write_addr[5:2]) // Address bits 5 to 2 determine the slot
                    4'b0000: begin ext_bank1_set_src_addr         = 1; end // set source address
                    4'b0001: begin ext_bank1_set_src_size         = 1; end // set source size
                    4'b0010: begin ext_bank1_set_des_addr         = 1; end // set destination address
                    4'b0011: begin ext_bank1_set_des_size         = 1; end // set destination size
                    4'b0100: begin ext_bank1_set_status           = 1; end // set status
                    4'b0101: begin ext_bank1_set_profile          = 1; end // set profile
                    4'b0110: begin ext_bank1_set_ld_mask          = 1; end // set load mask
                    4'b0111: begin ext_bank1_set_st_mask          = 1; end // set store mask
                    4'b1000: begin ext_bank1_set_st_intr_mask_abs = 1; end // set store interrupt mask
                    default: begin end // Default case for unsupported addresses
                endcase
            end

            default: begin end
        endcase
    end

end

endmodule