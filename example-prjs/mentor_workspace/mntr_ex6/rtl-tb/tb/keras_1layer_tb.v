module keras1layer_tb;

// Signals
reg clk;
reg reset;

reg [179:0] input_data;
reg input_valid;
wire input_ready;

wire [17:0] output_data;
wire output_valid;
reg output_ready;

wire [15:0] size_in_data;
wire size_in_valid;

wire [15:0] size_out_data;
wire size_out_ready;

reg [17:0] expected_output_data;
reg dut_error;

// Module instance
keras1layer keras1layer_top (
    .clk (clk),
    .rst (reset),

    .input_1_rsc_dat (input_data), // IN
    .input_1_rsc_vld (input_valid), // IN
    .input_1_rsc_rdy (input_ready), // OUT

    .layer5_out_rsc_dat (output_data), // OUT
    .layer5_out_rsc_vld (output_valid), // OUT
    .layer5_out_rsc_rdy (output_ready), // IN

    .const_size_in_1_rsc_dat(size_in_data), // OUT
    .const_size_in_1_rsc_vld(size_in_valid), // OUT

    .const_size_out_1_rsc_dat(size_out_data), // OUT
    .const_size_out_1_rsc_vld(size_out_valid) // OUT
);

// Trace file setup.
initial
begin
    $dumpfile ("keras1layer.vcd");
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

    input_data = 180'b0;
    output_ready = 1'b0;

    // Assert input data as valid.
    input_valid = 1'b1;

    // Assert output data as ready.
    output_ready = 1'b1;

    expected_output_data = 18'b0;

    dut_error = 0;
end

// Reset generator.
initial
begin
    reset = 1;
    $display ("@%04d INFO: Reset high", $time);
    
    #50 reset = 0;
    $display ("@%04d INFO: Reset low", $time);
end

// Stimuli.
initial begin
    // TODO: In the future the testbench will read from file the inputs and
    // expected outputs.
    
    // 0
    input_data = 180'b111111111110110110111111110000001111000000010001100110000000011001001010111111111111111100111111111111111100111111110011100101111111110010000101000000000111111011111111110110011001; 
    expected_output_data = 18'b000000000000001111;

    // 1 
    //input_data = 180'b111111110010100000111111101100100110000000000111010011000000001010001000111111111110100111111111111110100111111111101110111100111111101101010100111111101111101011000000010011010100;
    //expected_output_data = 18'b000000000000000100;

    // 2
    //input_data = 180'b111111111101010011000000000001011001111111110101101100111111110000110001111111111101110101111111111101110101111111111110101100111111111110100010111111110100000001000000001100111111;
    //expected_output_data = 18'b000000000001110000;

    // 3
    //input_data = 180'b111111110101100101111111101100011110000000010101001001000000011001101111000000001001110001000000001001110001111111101111100111111111101101011010000000000101111010000000000000100011;
    //expected_output_data = 18'b000000000000000110;

    // 4
    //input_data = 180'b111111111101010011000000000011001001111111110100011101111111110101100111111111111000110011111111111000110011111111111100110110111111111110110100000000000011000101111111111110101110;
    //expected_output_data =   18'b000000000000001011;
   
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

                #10;
                //$stop;
                $finish;
            end                
    end
end

endmodule

