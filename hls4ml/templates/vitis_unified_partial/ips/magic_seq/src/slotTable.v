module SlotArr #(

    parameter INDEX_WIDTH = 3, // 2 ^ 2 = 4 slots
    parameter SRC_ADDR_WIDTH = 32,
    parameter SRC_SIZE_WIDTH = 26,
    parameter DST_ADDR_WIDTH = 32,
    parameter DST_SIZE_WIDTH = 26,
    parameter STATUS_WIDTH   = 2,
    parameter PROFILE_WIDTH  = 32,
    parameter LD_MSK_WIDTH    =  8,
    parameter ST_MSK_WIDTH    =  8
)(
    input wire clk,
    input wire reset,
    // Declare an array of slots
    input wire [INDEX_WIDTH   -1:0] inp_index,
    input wire [SRC_ADDR_WIDTH-1:0] inp_src_addr,
    input wire [SRC_SIZE_WIDTH-1:0] inp_src_size,
    input wire [DST_ADDR_WIDTH-1:0] inp_des_addr,
    input wire [DST_SIZE_WIDTH-1:0] inp_des_size,
    input wire [STATUS_WIDTH  -1:0] inp_status,
    input wire [PROFILE_WIDTH -1:0] inp_profile,
    input wire [LD_MSK_WIDTH  -1:0] inp_ld_mask,
    input wire [ST_MSK_WIDTH  -1:0] inp_st_mask,
    input wire [ST_MSK_WIDTH  -1:0] inp_st_intr_mask_ack,
    input wire [ST_MSK_WIDTH  -1:0] inp_st_intr_mask_abs,

    input wire set_src_addr,
    input wire set_src_size,
    input wire set_des_addr,
    input wire set_des_size,
    input wire set_status,
    input wire set_profile,
    input wire set_ld_mask,
    input wire set_st_mask,
    input wire set_st_intr_mask_ack,
    input wire set_st_intr_mask_abs,

    // Output ports0
    input wire [INDEX_WIDTH-1:0]    out_index,
    output reg [DST_ADDR_WIDTH-1:0] out_src_addr,      // actually it is a wire
    output reg [DST_SIZE_WIDTH-1:0] out_src_size,      // actually it is a wire
    output reg [DST_ADDR_WIDTH-1:0] out_des_addr,      // actually it is a wire
    output reg [DST_SIZE_WIDTH-1:0] out_des_size,      // actually it is a wire
    output reg [STATUS_WIDTH-1:0]   out_status  ,      // actually it is a wire
    output reg [PROFILE_WIDTH-1:0]  out_profile,       // actually it is a wire
    output reg [LD_MSK_WIDTH  -1:0] out_ld_mask,     // actually it is a wire
    output reg [ST_MSK_WIDTH  -1:0] out_st_mask,     // actually it is a wire
    output reg [ST_MSK_WIDTH  -1:0] out_st_intr_mask // actually it is a wire
);


    wire [DST_ADDR_WIDTH-1:0] out_src_addr_pool     [0: ((1 << INDEX_WIDTH)-1)];
    wire [DST_SIZE_WIDTH-1:0] out_src_size_pool     [0: ((1 << INDEX_WIDTH)-1)];
    wire [DST_ADDR_WIDTH-1:0] out_des_addr_pool     [0: ((1 << INDEX_WIDTH)-1)];
    wire [DST_SIZE_WIDTH-1:0] out_des_size_pool     [0: ((1 << INDEX_WIDTH)-1)];
    wire [STATUS_WIDTH  -1:0] out_status_pool       [0: ((1 << INDEX_WIDTH)-1)];
    wire [PROFILE_WIDTH -1:0] out_profile_pool      [0: ((1 << INDEX_WIDTH)-1)];
    wire [LD_MSK_WIDTH  -1:0] out_ld_mask_pool      [0: ((1 << INDEX_WIDTH)-1)];
    wire [ST_MSK_WIDTH  -1:0] out_st_mask_pool      [0: ((1 << INDEX_WIDTH)-1)];
    wire [ST_MSK_WIDTH  -1:0] out_st_intr_mask_pool [0: ((1 << INDEX_WIDTH)-1)];

    // Instantiate the Slot module for each slot in the array
    genvar i;
    generate
        for (i = 0; i < (1 << INDEX_WIDTH); i = i + 1) begin : slot_gen
            Slot #(
                .SRC_ADDR_WIDTH  (SRC_ADDR_WIDTH),
                .SRC_SIZE_WIDTH  (SRC_SIZE_WIDTH),
                .DST_ADDR_WIDTH  (DST_ADDR_WIDTH),
                .DST_SIZE_WIDTH  (DST_SIZE_WIDTH),
                .STATUS_WIDTH    (STATUS_WIDTH),
                .PROFILE_WIDTH   (PROFILE_WIDTH),
                .LD_MSK_WIDTH    (LD_MSK_WIDTH),
                .ST_MSK_WIDTH    (ST_MSK_WIDTH),

                .INPUT_IDX_WIDTH (INDEX_WIDTH),
                .CUR_IDX         (i)
            ) slot_inst (
                .clk                  (clk),
                .reset                (reset),
                .inputIdx             (inp_index),
                // set Input ports 
                .inp_src_addr         (inp_src_addr),
                .inp_src_size         (inp_src_size),
                .inp_des_addr         (inp_des_addr),
                .inp_des_size         (inp_des_size),
                .inp_status           (inp_status),
                .inp_profile          (inp_profile),
                .inp_ld_mask          (inp_ld_mask),
                .inp_st_mask          (inp_st_mask),
                .inp_st_intr_mask_ack (inp_st_intr_mask_ack),
                .inp_st_intr_mask_abs (inp_st_intr_mask_abs),
                // set trigger ports
                .set_src_addr        (set_src_addr),
                .set_src_size        (set_src_size),
                .set_des_addr        (set_des_addr),
                .set_des_size        (set_des_size),
                .set_status          (set_status),
                .set_profile         (set_profile),
                .set_ld_mask         (set_ld_mask),
                .set_st_mask         (set_st_mask),
                .set_st_intr_mask_ack(set_st_intr_mask_ack),
                .set_st_intr_mask_abs(set_st_intr_mask_abs),
                // dest output ports
                .out_src_addr     (out_src_addr_pool[i]),
                .out_src_size     (out_src_size_pool[i]),
                .out_des_addr     (out_des_addr_pool[i]),
                .out_des_size     (out_des_size_pool[i]),
                .out_status       (out_status_pool  [i]),
                .out_profile      (out_profile_pool [i]),
                .out_ld_mask      (out_ld_mask_pool[i]),
                .out_st_mask      (out_st_mask_pool[i]),
                .out_st_intr_mask (out_st_intr_mask_pool[i])
            );
        end
    endgenerate



    //// build read mux
    // Use a multiplexer to select the output from the appropriate slot based on the input index
integer muxIdx;
always @(*) begin

                out_src_addr = 0;
                out_src_size = 0;
                out_des_addr = 0;
                out_des_size = 0;
                out_status   = 0;
                out_profile  = 0;

        for (muxIdx = 0; muxIdx < (1 << INDEX_WIDTH); muxIdx = muxIdx + 1) begin
            if (out_index == muxIdx) begin
                // If the index matches, assign the output from the corresponding slot
                out_src_addr     = out_src_addr_pool[muxIdx];
                out_src_size     = out_src_size_pool[muxIdx];
                out_des_addr     = out_des_addr_pool[muxIdx];
                out_des_size     = out_des_size_pool[muxIdx];
                out_status       = out_status_pool[muxIdx];
                out_profile      = out_profile_pool[muxIdx];
                out_ld_mask      = out_ld_mask_pool[muxIdx];
                out_st_mask      = out_st_mask_pool[muxIdx];
                out_st_intr_mask = out_st_intr_mask_pool[muxIdx];
            end
        end

end

endmodule