module verilog_dut(clk, rst, in_signal, out_signal);

input clk;
input rst;
input in_signal;
output out_signal;

bit out_signal_o;

always @(posedge clk) begin
   if (rst) begin
     out_signal_o <= 0;
     end 
   else begin
     out_signal_o <= ~in_signal;
     end
   end

assign out_signal = out_signal_o;

endmodule
