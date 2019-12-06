module mnist_mlp_tb;

// Signals
reg clk;
reg reset;

reg [14111:0] input_data;
reg input_valid;
wire input_ready;

wire [179:0] output_data;
wire output_valid;
reg output_ready;

wire [15:0] size_in_data;
wire size_in_valid;

wire [15:0] size_out_data;
wire size_out_valid;

reg [179:0] expected_output_data;
reg dut_error;

//wire [15:0] w2_adr_a;
//wire [17:0] w2_d_a;
//wire w2_en_a;
//wire w2_we_a;
//reg [17:0] w2_q_a;
//wire [15:0] w2_adr_b;
//wire [17:0] w2_d_b;
//wire w2_en_b;
//wire w2_we_b;
//reg [17:0] w2_q_b;
//
//reg [1151:0] b2_dat;
//
//wire [11:0] w4_adr_a;
//wire [17:0] w4_d_a;
//wire w4_en_a;
//wire w4_we_a;
//reg [17:0] w4_q_a;
//wire [11:0] w4_adr_b;
//wire [17:0] w4_d_b;
//wire w4_en_b;
//wire w4_we_b;
//reg [17:0] w4_q_b;
//
//reg [1151:0] b4_dat;
//
//wire [9:0] w6_adr_a;
//wire [17:0] w6_d_a;
//wire w6_en_a;
//wire w6_we_a;
//reg [17:0] w6_q_a;
//wire [9:0] w6_adr_b;
//wire [17:0] w6_d_b;
//wire w6_en_b;
//wire w6_we_b;
//reg [17:0] w6_q_b;
//
//reg [179:0] b6_dat;

// Module instance
mnist_mlp mnist_mlp_top (
    .clk (clk),
    .rst (reset),

    .input1_rsc_dat (input_data), // IN
    .input1_rsc_vld (input_valid), // IN
    .input1_rsc_rdy (input_ready), // OUT

    .output1_rsc_dat (output_data), // OUT
    .output1_rsc_vld (output_valid), // OUT
    .output1_rsc_rdy (output_ready), // IN

    .const_size_in_1_rsc_dat(size_in_data), // OUT
    .const_size_in_1_rsc_vld(size_in_valid), // OUT

    .const_size_out_1_rsc_dat(size_out_data), // OUT
    .const_size_out_1_rsc_vld(size_out_valid) // OUT

//    // Weights and biases in memories
//    .w2_rsc_adra(w2_adr_a),
//    .w2_rsc_da(w2_d_a),
//    .w2_rsc_ena(w2_en_a),
//    .w2_rsc_wea(w2_we_a),
//    .w2_rsc_qa(w2_q_a),
//    .w2_rsc_adrb(w2_adr_b),
//    .w2_rsc_db(w2_d_b),
//    .w2_rsc_enb(w2_en_b),
//    .w2_rsc_web(w2_we_b),
//    .w2_rsc_qb(w2_q_b),
//
//    .b2_rsc_dat(b2_dat),
//
//    .w4_rsc_adra(w4_adr_a),
//    .w4_rsc_da(w4_d_a),
//    .w4_rsc_ena(w4_en_a),
//    .w4_rsc_wea(w4_we_a),
//    .w4_rsc_qa(w4_q_a),
//    .w4_rsc_adrb(w4_adr_b),
//    .w4_rsc_db(w4_d_b),
//    .w4_rsc_enb(w4_en_b),
//    .w4_rsc_web(w4_we_b),
//    .w4_rsc_qb(w4_q_b),
//
//    .b4_rsc_dat(b4_dat),
//  
//    .w6_rsc_adra(w6_adr_a),
//    .w6_rsc_da(w6_d_a),
//    .w6_rsc_ena(w6_en_a),
//    .w6_rsc_wea(w6_we_a),
//    .w6_rsc_qa(w6_q_a),
//    .w6_rsc_adrb(w6_adr_b),
//    .w6_rsc_db(w6_d_b),
//    .w6_rsc_enb(w6_en_b),
//    .w6_rsc_web(w6_we_b),
//    .w6_rsc_qb(w6_q_b),
//
//    .b6_rsc_dat(b6_dat)
);

// Trace file setup.
initial
begin
    $dumpfile ("mnist_mlp.vcd");
    $dumpvars;
end


// Clock generator.
initial
begin
    clk = 0;
    forever
        #5 clk = ~ clk;
end

// Initialize signals
initial
begin
    $display ("@%04d INFO: ###################################################", $time);

    input_data = 14112'b0;
    output_ready = 1'b0;
    expected_output_data = 180'b0;
    dut_error = 0;

    //w2_q_a = 18'b0;
    //w2_q_b = 18'b0;

    //b2_dat = 1152'b0;
    //w4_q_a = 18'b0;
    //w4_q_b = 18'b0;
    //b4_dat = 1152'b0;
    //w6_q_a = 18'b0;
    //w6_q_b = 18'b0;
    //b6_dat = 180'b0;
end

// Reset generator.
initial
begin
    reset = 1;
    $display ("@%04d INFO: Reset high", $time);
    
    #20 reset = 0;
    $display ("@%04d INFO: Reset low", $time);

end

// Stimuli.
initial begin
    input_valid = 1'b0;

    output_ready = 1'b1;


    // TODO: In the future the testbench will read from file the inputs and
    // expected outputs.
    
    // 0
    input_data = 14112'h00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000434025a00a38028e00a38028e0062400500000000000000000000000000000000000000000000000000000000000000000000000000000000a8007a4033b00fdc03fb00fec03fb0100003fb00fec033f00140000000000000000000000000000000000000000000000000000000000000140017100cec03e700fec03fb00fec03fb00fec03fb00fec03fb00fec03fb00918000000000000000000000000000000000000000000000000000000011100efc03fb00fec03fb00fec03cb00be801d9004b4008800120015500704036300ba8004c0000000000000000000000000000000000000000000000076403d700fec03fb00efc0246005a400480000000000000000000000000000000001a100fec01a100000000000000000000000000000000000000000000000c3c03fb00fec02ce00370000000000000000000000000000000000000000000000005800a98016100000000000000000000000000000000000000000000000d0c03fb00734000800000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e3c027a000d0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d4c021600000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000484038300684003c0000000000000000000021000d400a78027e00080000000000000000000000000000000000000000000000000000000000000000000000000019d00efc032b00968023200948033f00f8c03fb00fec03fb005e4000000000000000000000000000000000000000000000000000000000000000000000000000000210013100edc03fb00fec03fb00fec03fb00f0c01c9001d0000000000000000000000000000000000000000000000000000000000000000000000000000000000004000a1803fb00fec03fb00fac0272004a4000000000000000000000000170024200c6c010500000000000000000000000000000000000000000000000060030300fec03fb00fac0363005b40000000400038001500054003a0023a00eec03fb00fec033f0000000000000000000000000000000000000000000000067403fb00fec03fb007340044003e002c200c7c03a300fec03fb00fec03fb00fec03fb00fec031b0000000000000000000000000000000000000000000000063403fb00fec03fb00fec03fb00fec03fb00fec03fb00fec03fb00fec03fb00d4c02720060400bc0000000000000000000000000000000000000000000000000002c600fec03fb00fec03fb00fec03e700f8c03df00b280282005f40129001500000000000000000000000000000000000000000000000000000000000000000054004f4018d005740181009480064000e00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000; 
    expected_output_data = 180'h00000000000000000000000000000ff40001000000000;
 
    #35 input_valid = 1'b1;
end

// Validation
always @(posedge clk)
begin
    if (reset == 1'b1)
        dut_error <= 0;
    else
        if (output_valid == 1'b1 && output_data != expected_output_data)
        begin
            $display ("@%04d ERROR: DUT ERROR!", $time);
            $display ("@%04d ERROR: Expected value %b, but got value %b", $time, expected_output_data, output_data);
            dut_error <= 1;
        end
end

initial
begin
    forever
    begin
        #10;
        if (reset == 1'b0)
            if (output_valid == 1'b1)
            begin
                $display ("@%04d INFO: Terminating simulation", $time);
                if (dut_error == 0) begin
                    $display ("@%04d INFO: Validation: PASSED", $time);
                end
                else begin
                    $display ("@%04d INFO: Validation: FAIL", $time);
                end
                $display ("@%04d INFO: ###################################################", $time);

                #250;
                //$stop;
                $finish;
            end                
    end
end

endmodule

