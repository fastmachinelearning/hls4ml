`timescale 1 ns / 1 ps

// Bank selection for a parameter the stock IP exposes as N_SCALARS parallel scalar
// ports rather than a memory (hls4ml lowers a Dense bias this way, because
// nnet_dense_resource.h fully partitions `biases`).
//
// The select is COMBINATIONAL. A registered mux would present the previous bank
// on the cycle the IP starts, which is wrong whenever a bias can be consumed in
// the first cycles of a transaction -- reachable here, since a pipelined design
// can report a latency of one cycle.
//
// cur_bank_id is committed by bank_select_latch before ap_start rises and held to
// ap_done, so these outputs are stable for the whole transaction even though they
// are unregistered.

`default_nettype none

module scalar_bank_mux #(
    parameter int SCALAR_WIDTH  = 16,
    parameter int N_SCALARS     = 4,
    parameter int N_BANKS       = 2,
    parameter int BANK_ID_WIDTH = 1
) (
    input  wire                     ap_clk,
    input  wire                     ap_rst,
    input  wire [BANK_ID_WIDTH-1:0] cur_bank_id,
    input  wire                     quiescent,

    input  wire                     ld_we,
    input  wire [BANK_ID_WIDTH-1:0] ld_bank,
    input  wire [$clog2(N_SCALARS > 1 ? N_SCALARS : 2)-1:0] ld_idx,
    input  wire [SCALAR_WIDTH-1:0]  ld_data,
    output wire                     ld_accept,
    output wire                     ld_reject,

    output wire [SCALAR_WIDTH*N_SCALARS-1:0] q_flat
);

  reg [SCALAR_WIDTH-1:0] bank_mem [0:N_BANKS-1][0:N_SCALARS-1];

  initial begin
    integer bk, s;
    for (bk = 0; bk < N_BANKS; bk = bk + 1)
      for (s = 0; s < N_SCALARS; s = s + 1) bank_mem[bk][s] = '0;
  end

  // same policy as parameter_bank: writes only while the wrapper is idle.
  // ld_idx is $clog2(N_SCALARS) bits, so for a non-power-of-two bundle it can
  // encode indices past the end of the array; reject those.
  wire safe = quiescent & (32'(ld_bank) < N_BANKS) & (32'(ld_idx) < N_SCALARS);

  assign ld_accept = ld_we &  safe;
  assign ld_reject = ld_we & ~safe;

  always @(posedge ap_clk) begin
    if (ld_accept) bank_mem[ld_bank][ld_idx] <= ld_data;
  end

  genvar i;
  generate
    for (i = 0; i < N_SCALARS; i = i + 1) begin : g_out
      assign q_flat[SCALAR_WIDTH*i +: SCALAR_WIDTH] = bank_mem[cur_bank_id][i];
    end
  endgenerate

endmodule

`default_nettype wire
