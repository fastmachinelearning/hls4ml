


module s_axi_read #(

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
    parameter BANK1_LD_MSK_WIDTH    =  8,
    parameter BANK1_ST_MSK_WIDTH    =  8,

    parameter BANK0_CONTROL_WIDTH = 4,
    parameter BANK0_STATUS_WIDTH  = 4,
    parameter BANK0_CNT_WIDTH     = BANK1_INDEX_WIDTH, /// the counter for the sequencer
    parameter BANK0_INTR_WIDTH  = 1,       /// the interrupt for the sequencer
    parameter BANK0_ROUNDTRIP_WIDTH = 16 /// the round trip counter for the sequencer
) (

    input  wire clk,
    input  wire reset,

    // Read Address Channel
    input  wire [ADDR_WIDTH-1:0]  S_AXI_ARADDR,
    input  wire                   S_AXI_ARVALID,
    output wire                   S_AXI_ARREADY,

    // Read Data Channel
    output reg  [DATA_WIDTH-1:0]   S_AXI_RDATA, ////// read data output acctually it is a wire
    output wire [1:0]              S_AXI_RRESP,
    output wire                    S_AXI_RVALID,
    input  wire                    S_AXI_RREADY,

    ////// bank1 interconnect
    output  wire [BANK1_INDEX_WIDTH    -1:0] ext_bank1_out_index,
    output  reg                              ext_bank1_out_req,           // actually it is a wire
    input   wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_src_addr,      // actually it is a wire
    input   wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_src_size,      // actually it is a wire
    input   wire [BANK1_DST_ADDR_WIDTH -1:0] ext_bank1_out_des_addr,
    input   wire [BANK1_DST_SIZE_WIDTH -1:0] ext_bank1_out_des_size,
    input   wire [BANK1_STATUS_WIDTH   -1:0] ext_bank1_out_status  ,      // actually it is a wire
    input   wire [BANK1_PROFILE_WIDTH  -1:0] ext_bank1_out_profile,       // actually it is a wire
    input   wire [BANK1_LD_MSK_WIDTH   -1:0] ext_bank1_out_ld_mask,     // actually it is a reg
    input   wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_mask,
    input   wire [BANK1_ST_MSK_WIDTH   -1:0] ext_bank1_out_st_intr_mask,
    input   wire                             ext_bank1_out_ready,         // actually it is a wire

    ////// bank0 interconnect
    input wire [BANK0_STATUS_WIDTH   -1: 0] ext_bank0_out_status,  /// read only and it is reg
    input wire [BANK0_CNT_WIDTH      -1: 0] ext_bank0_out_mainCnt,     /// read only
    input wire [BANK0_CNT_WIDTH      -1: 0] ext_bank0_out_endCnt,      /// read only
    input wire [GLOB_ADDR_WIDTH      -1: 0] ext_bank0_out_dmaBaseAddr,
    input wire [GLOB_ADDR_WIDTH      -1: 0] ext_bank0_out_dfxCtrlAddr,
    input wire [BANK0_INTR_WIDTH     -1: 0] ext_bank0_out_intrEna, //// output data for the round counter
    input wire [BANK0_INTR_WIDTH     -1: 0] ext_bank0_out_intr,  //// output data for the interrupt counter
    input wire [BANK0_ROUNDTRIP_WIDTH-1: 0] ext_bank0_out_roundTrip


    
);

always @(*)begin
    case(ext_bank1_out_ready)
        default: begin end
    endcase
end

localparam ST_IDLE     = 3'b000;
localparam ST_READDATA = 3'b010;


reg[2:0]            state; // State variable for FSM
reg[ADDR_WIDTH-1:0] read_addr; // Register to hold the read address


///////// main control state machine
always @(posedge clk or negedge reset ) begin
    if (~reset) begin
        state <= ST_IDLE;
    end else begin
        case (state)
            ST_IDLE: begin
                if (S_AXI_ARVALID) begin ///// address is comming response immediately
                    state <= ST_READDATA;
                    read_addr <= S_AXI_ARADDR;
                end
            end
            ST_READDATA: begin
                if (S_AXI_RREADY) begin ///// send data response immediately
                    state <= ST_IDLE;
                end
            end
            default: begin
                state <= ST_IDLE; // Default case to handle unexpected states
            end
            
        endcase
    end
end

////////// main control output wires

/////////////// read address channel
assign S_AXI_ARREADY = (state == ST_IDLE) && S_AXI_ARVALID; // Ready to accept read address when
/////////////// read data channel
assign S_AXI_RRESP   = 2'b00;
assign S_AXI_RVALID  = (state == ST_READDATA);

assign ext_bank1_out_index = read_addr[BANK1_INDEX_WIDTH+6-1:6]; // Extracting index from address bits 5 and 4

always @(*) begin

    ext_bank1_out_req = 0; // Default value
    S_AXI_RDATA       = 0; // Default case for unsupported addresses

    if (state == ST_READDATA)begin
        if (read_addr[15:14] == 2'b00) begin

            case (read_addr[13:6]) // Address bits 13 to 6 determine the slot
                8'h00:   begin S_AXI_RDATA = 0;                                                                         end
                8'h01:   begin S_AXI_RDATA = { {(DATA_WIDTH-BANK0_STATUS_WIDTH){1'b0}}, ext_bank0_out_status };         end // read status register
                8'h02:   begin S_AXI_RDATA = { {(DATA_WIDTH- BANK1_INDEX_WIDTH){1'b0}}, ext_bank0_out_mainCnt};         end// read main counter register
                8'h03:   begin S_AXI_RDATA = { {(DATA_WIDTH- BANK1_INDEX_WIDTH){1'b0}}, ext_bank0_out_endCnt };         end// read end counter register
                8'h04:   begin S_AXI_RDATA = ext_bank0_out_dmaBaseAddr;                                                 end
                8'h05:   begin S_AXI_RDATA = ext_bank0_out_dfxCtrlAddr;                                                 end
                8'h06:   begin S_AXI_RDATA = { {(DATA_WIDTH - BANK0_INTR_WIDTH){1'b0}}, ext_bank0_out_intrEna};         end // read round counter register
                8'h07:   begin S_AXI_RDATA = { {(DATA_WIDTH - BANK0_INTR_WIDTH){1'b0}}, ext_bank0_out_intr };           end // read interrupt register
                8'h08:   begin S_AXI_RDATA = { {(DATA_WIDTH - BANK0_ROUNDTRIP_WIDTH){1'b0}}, ext_bank0_out_roundTrip }; end // read round trip counter register
                default: begin S_AXI_RDATA = 0;                                                                         end// Default case for unsupported addresses
            endcase

        end else if (read_addr[15:14] == 2'b01) begin

            ext_bank1_out_req = 1; // Set request signal for bank1
            case (read_addr[5: 2])
                4'b0000: begin S_AXI_RDATA = ext_bank1_out_src_addr;                                                    end // read index register
                4'b0001: begin S_AXI_RDATA = {{(DATA_WIDTH - BANK1_SRC_SIZE_WIDTH){1'b0}}, ext_bank1_out_src_size};     end // read source address
                4'b0010: begin S_AXI_RDATA = ext_bank1_out_des_addr;                                                    end // read source size
                4'b0011: begin S_AXI_RDATA = {{(DATA_WIDTH - BANK1_DST_SIZE_WIDTH){1'b0}}, ext_bank1_out_des_size};     end // read destination address
                4'b0100: begin S_AXI_RDATA = {30'b0, ext_bank1_out_status };                                            end // read destination size
                4'b0101: begin S_AXI_RDATA = ext_bank1_out_profile;                                                     end // read status register
                4'b0110: begin S_AXI_RDATA = { {(DATA_WIDTH - BANK1_LD_MSK_WIDTH){1'b0}}, ext_bank1_out_ld_mask };      end // read load mask
                4'b0111: begin S_AXI_RDATA = { {(DATA_WIDTH - BANK1_ST_MSK_WIDTH){1'b0}}, ext_bank1_out_st_mask };      end // read store mask
                4'b1000: begin S_AXI_RDATA = { {(DATA_WIDTH - BANK1_ST_MSK_WIDTH){1'b0}}, ext_bank1_out_st_intr_mask }; end // read store interrupt mask
                default: begin S_AXI_RDATA = 0;                                                                         end// Default case for unsupported addresses

            endcase

        end
    end
end

endmodule