module MagicStreammerCore #
(
    parameter integer DATA_WIDTH        = 32, 
    parameter integer STORAGE_IDX_WIDTH = 10,     //// 4 Kb
    parameter integer STATE_BIT_WIDTH   =  4
)
(
    input wire                        clk,
    input wire                        reset,

    // AXIS Slave Interface   store in terface
    input  wire [DATA_WIDTH-1:0]     S_AXI_TDATA,
//    input  wire [DATA_WIDTH/8-1:0]   S_AXI_TKEEP,  // <= tkeep added        //// we disable tkeep
    input  wire                      S_AXI_TVALID,
    output wire                      S_AXI_TREADY,
    input  wire                      S_AXI_TLAST,

    // AXIS Master Interface load interface
    output  reg [DATA_WIDTH-1:0]     M_AXI_TDATA,    // it is supposed to be reg
//    output  wire [DATA_WIDTH/8-1:0]  M_AXI_TKEEP,    // it is supposed to be reg
    output  reg                      M_AXI_TVALID,    // it is supposed to be reg
    input   wire                     M_AXI_TREADY,    // it is supposed to be reg
    output  reg                      M_AXI_TLAST,    // it is supposed to be reg

    // control signal 

    input wire storeReset,
    input wire loadReset,
    input wire storeInit,
    input wire loadInit,

    // store complete connect it to mgsFinExec
    output wire finStore,

    // out put wire for debugging
    output wire [STATE_BIT_WIDTH-1:0]   dbg_state,
    output wire [(STORAGE_IDX_WIDTH+1)-1:0] dbg_amt_store_bytes,
    output wire [(STORAGE_IDX_WIDTH+1)-1:0] dbg_amt_load_bytes

);



///// declare state

localparam STATUS_IDLE       = 4'b0000;
localparam STATUS_STORE      = 4'b0001;
localparam STATUS_LOAD       = 4'b0010;

localparam TRACKER_IDX_WIDTH = STORAGE_IDX_WIDTH + 1; ///// this is for tracker index width

///// meta data 
(* ram_style = "block" *) reg[DATA_WIDTH-1: 0] mainMem [0: ((1 << STORAGE_IDX_WIDTH) - 1)];

reg[STATE_BIT_WIDTH  -1: 0] state;
reg[TRACKER_IDX_WIDTH-1: 0] amt_store_bytes; ///// store to this block
reg[TRACKER_IDX_WIDTH-1: 0] amt_load_bytes;  ///// load to this block
reg storeIntr;

/////////////////////////////////////
////// axi signal assign ////////////
/////////////////////////////////////

///////// store
assign S_AXI_TREADY = (state == STATUS_STORE) && S_AXI_TVALID;
///////// load
////assign M_AXI_TKEEP   = 4'b1111;
///////// interrupt signal
assign finStore = storeIntr;
/////////// debug signal
assign dbg_state           = state;
assign dbg_amt_store_bytes = amt_store_bytes;
assign dbg_amt_load_bytes  = amt_load_bytes;

/////////////////////////////////////
////// control system    ////////////
/////////////////////////////////////
always @(posedge clk, negedge reset) begin
    
    if (~reset) begin
        state           <= STATUS_IDLE;
        amt_store_bytes <= 0;
        amt_load_bytes  <= 0;
        storeIntr       <= 0;
    end else begin
        case (state) 
            STATUS_IDLE    : begin 
                    if (storeReset) begin
                        amt_store_bytes <= 0;
                        storeIntr       <= 0;
                    end else if (loadReset) begin
                        amt_load_bytes <= 0;
                        storeIntr      <= 0;
                    end else if (storeInit) begin
                        state <= STATUS_STORE;
                    end else if (loadInit & (amt_store_bytes > 0)) begin
                        state <= STATUS_LOAD;
                    end
            end
            //////////// case store data to the internal memory
            STATUS_STORE    : begin 
                if (S_AXI_TVALID)begin //// we are sure that ready will send this time
                    amt_store_bytes <= amt_store_bytes + 1;
                    if (S_AXI_TLAST)begin
                        storeIntr <= 1;
                        state     <= STATUS_IDLE;
                    end
                end
            end
            STATUS_LOAD    : begin
                if (M_AXI_TREADY | (amt_load_bytes == 0)) begin //// last sending is success in this cycle/ send next

                    if ( amt_load_bytes == amt_store_bytes )begin
                        /////// no data to send anymore
                        M_AXI_TVALID <= 0;
                        M_AXI_TLAST  <= 0;
                        state <= STATUS_IDLE;
                    end else begin
                        M_AXI_TVALID <= 1;
                        M_AXI_TLAST  <= (amt_load_bytes == (amt_store_bytes-1));
                        amt_load_bytes <= amt_load_bytes + 1;
                    end

                end
                ///// at here do nothing just wait for signal
                
            end


            default: begin end

        endcase
    end 

end



///// M_DATA and MEM management

always @(posedge clk) begin
    if (state == STATUS_STORE)begin
        if (S_AXI_TVALID)begin
            mainMem[amt_store_bytes[STORAGE_IDX_WIDTH-1: 0]] <=  S_AXI_TDATA;
        end
        
    end else if (state == STATUS_LOAD) begin

        if (M_AXI_TREADY | (amt_load_bytes == 0))begin
            if (amt_load_bytes == amt_store_bytes)begin
                M_AXI_TDATA <= 48;
            end else begin
                M_AXI_TDATA <= mainMem[amt_load_bytes[STORAGE_IDX_WIDTH-1: 0]];
            end
        end
        
    end
end


endmodule