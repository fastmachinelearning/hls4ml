module keras1layer_tb;

// Channels
reg clk, reset;

reg [179:0] input_data;
reg input_valid;
wire input_ready;

wire [17:0] output_data;
wire output_valid;
wire output_ready;

wire [15:0] size_in_data;
wire size_in_ready;

wire [15:0] size_out_data;
wire size_out_ready;

reg [17:0] expected_output_data;
reg dut_error;

// Module instance
keras1layer keras1layer_top (
    .clk    (clk),
    .rst  (reset),
    .input_1_rsc_dat (input_data),
    .input_1_rsc_vld (input_valid),
    .input_1_rsc_triosy_lz (input_ready),
    .layer5_out_rsc_dat (output_data),
    .layer5_out_rsc_vld (output_valid),
    .layer5_out_rsc_triosy_lz (output_ready),
    .const_size_in_1_rsc_dat(size_in_data),
    .const_size_in_1_rsc_triosy_lz(size_in_ready),
    .const_size_out_1_rsc_dat(size_out_data),
    .const_size_out_1_rsc_triosy_lz(size_out_ready)
);

// Clock generator
always
    #5 clk = !clk;

// Initialize signals
initial
begin
    $display ("@%04d INFO: ###################################################", $time);
    clk = 0;
    reset = 0;
    
    input_data = 180'b0;
    input_valid = 1'b0;
    expected_output_data = 18'b0;

    //enable = 0;
    dut_error = 0;
end

// Trace file setup
initial
begin
    $dumpfile ("keras1layer.vcd");
    $dumpvars;
end

// Monitor simulation
event reset_enable;
event terminate_sim;
initial begin
    #10 -> reset_enable; // Time to reset!

    @ (input_ready == 1);     // Wait for negative-edge of the clock

    input_valid = 1'b1;
    
    input_data = 180'b111111110110011001000000000111111011111111110010000101111111110011100101111111111111111100111111111111111100000000011001001010000000010001100110111111110000001111111111111110110110;
    expected_output_data = 18'b000000000000001111;
   
    //input_data = 180'b111111110110000101000000001000101011000000011011100101000000100010100000000000000110110101000000000110110101000000000101111100000000001000001000000000011110101010000000011110011110;
    //expected_output_data = 18'b000000001010011010;

    @ (output_valid == 1'b1);     // Wait for negative-edge of the clock
    @ (size_in_ready == 1'b1);     // Wait for negative-edge of the clock
    @ (size_out_ready == 1'b1);     // Wait for negative-edge of the clock

    repeat (100)
    begin
        @ (negedge clk); // Wait for 5-negative-edges of the clock
    end

    #5 -> terminate_sim; // Terminate!
end

// Reset generator
event reset_done;
initial
forever begin
    @ (reset_enable); // Wait for reset-enable event
    @ (negedge clk)   // At the first negative-edge of the clock
        $display ("@%04d INFO: Applying reset", $time);
        reset = 1;
    @ (negedge clk)    // At the second negative-edge of the clock
        reset = 0;
        $display ("@%04d INFO: Came out of Reset", $time);
    -> reset_done;
end

// Wrap-up simulation
initial
@ (terminate_sim)  begin
    $display ("@%04d INFO: Terminating simulation", $time);
    if (dut_error == 0) begin
        $display ("@%04d INFO: Simulation Result : PASSED", $time);
    end
    else begin
        $display ("@%04d INFO: Simulation Result : FAILED", $time);
    end
    $display ("@%04d INFO: ###################################################", $time);
    //#1 $finish;
    #1 $stop;
end


always @ (negedge clk)
if (output_valid == 1'b1 && output_data != expected_output_data) begin
    $display ("@%04d ERROR: DUT ERROR!", $time);
    $display ("@%04d ERROR: Expected value %b, Got Value %b", $time, expected_output_data, output_data);
    dut_error = 1;
    #5 -> terminate_sim;
end

endmodule

