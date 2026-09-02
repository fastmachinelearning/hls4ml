`timescale 1 ns / 1 ps

// Address translation for one external hls4ml BRAM parameter port.
//
// Combinational: adds zero latency, so the HLS-facing read latency of the wrapped
// memory is identical to the single-bank memory the stock IP expects.
//
//   physical_word = bank_id * BANK_STRIDE_WORDS + local_word
//   local_word    = hls_byte_addr >> $clog2(WORD_BYTES)
//
// hls4ml BRAM ports carry a BYTE address (the generated RTL shifts the word index
// left by log2(WORD_BYTES)). When BANK_STRIDE_WORDS is a power of two this reduces
// to concatenating the bank bits above the local address bits.

`default_nettype none

module bank_addr_mapper #(
    parameter int HLS_ADDR_WIDTH    = 32,
    parameter int WORD_BYTES        = 32,
    parameter int LOCAL_WORDS       = 2,
    parameter int N_BANKS           = 2,
    parameter int BANK_ID_WIDTH     = 1,
    parameter int BANK_STRIDE_WORDS = 2,
    parameter int PHYS_ADDR_WIDTH   = 2
) (
    input  wire [HLS_ADDR_WIDTH-1:0]  hls_byte_addr,
    input  wire [BANK_ID_WIDTH-1:0]   bank_id,
    output wire [PHYS_ADDR_WIDTH-1:0] phys_word_addr,
    output wire                       addr_in_padding
);

  localparam int WORD_SHIFT = $clog2(WORD_BYTES);

  wire [HLS_ADDR_WIDTH-1:0] local_word = hls_byte_addr >> WORD_SHIFT;

  assign addr_in_padding = (local_word >= LOCAL_WORDS[HLS_ADDR_WIDTH-1:0]);

  wire [PHYS_ADDR_WIDTH:0] phys_sum =
      (bank_id * BANK_STRIDE_WORDS) + local_word[PHYS_ADDR_WIDTH:0];

  assign phys_word_addr = phys_sum[PHYS_ADDR_WIDTH-1:0];

endmodule

`default_nettype wire
