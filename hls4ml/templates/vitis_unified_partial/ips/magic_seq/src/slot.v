module Slot #(
    ///////// indexing
    parameter INPUT_IDX_WIDTH =  2,
    ///////// slot meta-data slot
    parameter SRC_ADDR_WIDTH  = 32,
    parameter SRC_SIZE_WIDTH  = 26,
    parameter DST_ADDR_WIDTH  = 32,
    parameter DST_SIZE_WIDTH  = 26,
    parameter STATUS_WIDTH    =  2,
    parameter PROFILE_WIDTH   = 32,
    parameter LD_MSK_WIDTH    =  8,
    parameter ST_MSK_WIDTH    =  8,
    ///////// slot identifier
    parameter CUR_IDX         =  0
) (
    input wire clk,
    input wire reset,
    input wire [INPUT_IDX_WIDTH-1: 0] inputIdx,

    // set Input ports 
    input wire [SRC_ADDR_WIDTH-1:0]   inp_src_addr,
    input wire [SRC_SIZE_WIDTH-1:0]   inp_src_size,
    input wire [DST_ADDR_WIDTH-1:0]   inp_des_addr,
    input wire [DST_SIZE_WIDTH-1:0]   inp_des_size,
    input wire [STATUS_WIDTH  -1:0]   inp_status,
    input wire [PROFILE_WIDTH -1:0]   inp_profile,
    input wire [LD_MSK_WIDTH  -1:0]   inp_ld_mask,
    input wire [ST_MSK_WIDTH  -1:0]   inp_st_mask,
    input wire [ST_MSK_WIDTH  -1:0]   inp_st_intr_mask_ack,
    input wire [ST_MSK_WIDTH  -1:0]   inp_st_intr_mask_abs,
    // set trigger ports
    input wire                        set_src_addr,
    input wire                        set_src_size,
    input wire                        set_des_addr,
    input wire                        set_des_size,
    input wire                        set_status,
    input wire                        set_profile,
    input wire                        set_ld_mask,
    input wire                        set_st_mask,
    input wire                        set_st_intr_mask_ack,
    input wire                        set_st_intr_mask_abs,
    // dest output ports
    output reg [SRC_ADDR_WIDTH-1:0]   out_src_addr,
    output reg [SRC_SIZE_WIDTH-1:0]   out_src_size,
    output reg [DST_ADDR_WIDTH-1:0]   out_des_addr,
    output reg [DST_SIZE_WIDTH-1:0]   out_des_size,
    output reg [STATUS_WIDTH  -1:0]   out_status,
    output reg [PROFILE_WIDTH -1:0]   out_profile,
    output reg [LD_MSK_WIDTH  -1:0]   out_ld_mask,
    output reg [ST_MSK_WIDTH  -1:0]   out_st_mask,
    output reg [ST_MSK_WIDTH  -1:0]   out_st_intr_mask
);


//// set trigger ports

always @( posedge clk or negedge reset ) begin
    
    if (~reset) begin
        out_src_addr     <= 0;
        out_src_size     <= 0;
        out_des_addr     <= 0;
        out_des_size     <= 0;
        out_status       <= 0;
        out_profile      <= 0;
        out_ld_mask      <= 0;
        out_st_mask      <= 0;
        out_st_intr_mask <= 0;
    end else begin
        if (inputIdx == CUR_IDX) begin
            // Only update the output if the input index matches the current index
            // This allows for multiple slots to be used in parallel without interference
            if (set_src_addr)         begin out_src_addr <= inp_src_addr; end
            if (set_src_size)         begin out_src_size <= inp_src_size; end
            if (set_des_addr)         begin out_des_addr <= inp_des_addr; end // Assuming des_addr is same as src_addr
            if (set_des_size)         begin out_des_size <= inp_des_size; end // Assuming des_size is same as src_size
            if (set_status)           begin out_status   <= inp_status;   end
            if (set_profile)          begin out_profile  <= inp_profile;  end
            if (set_ld_mask)          begin out_ld_mask  <= inp_ld_mask;  end
            if (set_st_mask)          begin out_st_mask  <= inp_st_mask;  end
            
            if  (set_st_intr_mask_abs)    begin out_st_intr_mask <= inp_st_intr_mask_abs; end
            else if(set_st_intr_mask_ack) begin out_st_intr_mask <= (out_st_intr_mask |  inp_st_intr_mask_ack); end
        end
    end

end
    
endmodule