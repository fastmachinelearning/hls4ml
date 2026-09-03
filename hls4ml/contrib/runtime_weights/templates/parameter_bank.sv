`timescale 1 ns / 1 ps

// Depth-stacked replacement for the single-bank memory a stock hls4ml IP expects
// on an external BRAM parameter port.
//
// PORT A faces the HLS IP and reproduces the contract of the memory model Vitis
// generates for co-simulation:
//     always_ff @(posedge clk) if (EN) Dout <= mem[Addr / WORD_BYTES];
// i.e. 1-cycle synchronous read, enable-gated. bank_addr_mapper is combinational,
// so this latency is identical to the unbanked memory.
//
// PORT B is the loader. hls4ml drives only port A of a BRAM parameter interface;
// interface.bram_port_b_is_unused() PROVES from the exported RTL that EN_B/WEN_B/
// Addr_B are tied off before the wrapper borrows port B, rather than assuming it.
// Loads are additionally accepted only while the wrapper is idle.
//
// Banks are stacked in depth: phys_word = bank_id*BANK_STRIDE_WORDS + local_word.
// N_BANKS, BANK_STRIDE_WORDS and the physical depth are build-time constants.

`default_nettype none

module parameter_bank #(
    parameter int DATA_WIDTH        = 256,
    parameter int HLS_ADDR_WIDTH    = 32,
    parameter int WORD_BYTES        = 32,
    parameter int LOCAL_WORDS       = 2,
    parameter int N_BANKS           = 2,
    parameter int BANK_ID_WIDTH     = 1,
    parameter int BANK_STRIDE_WORDS = 2,
    parameter     INIT_HEX          = ""
) (
    input  wire                        ap_clk,
    input  wire                        ap_rst,

    input  wire [BANK_ID_WIDTH-1:0]    cur_bank_id,
    input  wire                        quiescent,

    // facing the stock HLS IP (HLS is the master)
    input  wire [HLS_ADDR_WIDTH-1:0]   hls_Addr_A,
    input  wire                        hls_EN_A,
    input  wire [DATA_WIDTH/8-1:0]     hls_WEN_A,
    input  wire [DATA_WIDTH-1:0]       hls_Din_A,
    output reg  [DATA_WIDTH-1:0]       hls_Dout_A,
    input  wire                        hls_Rst_A,

    // native loader
    input  wire                        ld_req,
    input  wire [BANK_ID_WIDTH-1:0]    ld_bank,
    input  wire [$clog2(LOCAL_WORDS > 1 ? LOCAL_WORDS : 2)-1:0] ld_word,
    input  wire [DATA_WIDTH-1:0]       ld_wdata,
    output wire                        ld_accept,
    output wire                        ld_reject,

    output wire                        addr_padding_violation
);

  localparam int PHYS_WORDS      = N_BANKS * BANK_STRIDE_WORDS;
  localparam int PHYS_ADDR_WIDTH = (PHYS_WORDS <= 1) ? 1 : $clog2(PHYS_WORDS);

  // Provisional: forces block RAM regardless of depth, which is likely wrong for
  // shallow banks (a 4-word memory wastes a tile). Revisit with the 2/4/8-bank
  // post-route characterization before generalizing this into a parameter.
  (* ram_style = "block" *)
  reg [DATA_WIDTH-1:0] mem [0:PHYS_WORDS-1];

  initial begin
    integer i;
    for (i = 0; i < PHYS_WORDS; i = i + 1) mem[i] = '0;
    if (INIT_HEX != "") $readmemh(INIT_HEX, mem);
  end

  // ---------------- port A : HLS read path ----------------
  wire [PHYS_ADDR_WIDTH-1:0] phys_a;
  wire                       pad_a;

  bank_addr_mapper #(
      .HLS_ADDR_WIDTH   (HLS_ADDR_WIDTH),
      .WORD_BYTES       (WORD_BYTES),
      .LOCAL_WORDS      (LOCAL_WORDS),
      .BANK_ID_WIDTH    (BANK_ID_WIDTH),
      .BANK_STRIDE_WORDS(BANK_STRIDE_WORDS),
      .PHYS_ADDR_WIDTH  (PHYS_ADDR_WIDTH)
  ) u_map_a (
      .hls_byte_addr  (hls_Addr_A),
      .bank_id        (cur_bank_id),
      .phys_word_addr (phys_a),
      .addr_in_padding(pad_a)
  );

  // port A: registered read, byte-enabled write. Plain always blocks (not
  // always_ff) because `mem` is shared with the loader port below; always_ff
  // asserts exclusive ownership of the variables it assigns.
  always @(posedge ap_clk or posedge hls_Rst_A) begin
    if (hls_Rst_A) hls_Dout_A <= {DATA_WIDTH{1'b0}};
    else if (hls_EN_A) hls_Dout_A <= mem[phys_a];
  end

  integer bidx;
  always @(posedge ap_clk) begin
    if (hls_EN_A) begin
      for (bidx = 0; bidx < DATA_WIDTH / 8; bidx = bidx + 1)
        if (hls_WEN_A[bidx]) mem[phys_a][bidx*8 +: 8] <= hls_Din_A[bidx*8 +: 8];
    end
  end

  assign addr_padding_violation = hls_EN_A & pad_a;

  // ---------------- port B : loader ----------------
  // Loader writes are accepted only while the wrapper is quiescent; any in-range
  // bank may be written. Nothing can read the memory while idle, so there is no
  // active bank to protect.
  // ld_word is $clog2(LOCAL_WORDS) bits, so when LOCAL_WORDS is not a power of two
  // it can encode words that lie in the inter-bank padding. Reject those rather
  // than acknowledging a write that lands nowhere useful.
  wire bank_ok  = (32'(ld_bank) < N_BANKS);
  wire word_ok  = (32'(ld_word) < LOCAL_WORDS);
  wire safe     = quiescent & bank_ok & word_ok;   // idle => no active bank => any bank loadable
  assign ld_accept = ld_req &  safe;
  assign ld_reject = ld_req & ~safe;

  // Widen before combining: ld_word is only $clog2(LOCAL_WORDS) bits, so a
  // part-select of PHYS_ADDR_WIDTH bits from it can read past its width and
  // yield x, silently sending every loader write to an undefined address.
  wire [31:0] phys_b_full = (32'(ld_bank) * BANK_STRIDE_WORDS) + 32'(ld_word);
  wire [PHYS_ADDR_WIDTH-1:0] phys_b = phys_b_full[PHYS_ADDR_WIDTH-1:0];

  // port B: loader writes, accepted only while the wrapper is idle
  always @(posedge ap_clk) begin
    if (ld_accept) mem[phys_b] <= ld_wdata;
  end

`ifdef RUNTIME_WEIGHTS_ASSERT
  a_no_padding_access   : assert property (@(posedge ap_clk) disable iff (ap_rst)
                            !(hls_EN_A & pad_a));
  a_write_only_when_idle: assert property (@(posedge ap_clk) disable iff (ap_rst)
                            (ld_req & ld_accept) |-> quiescent);
  a_write_in_bounds     : assert property (@(posedge ap_clk) disable iff (ap_rst)
                            (ld_req & ld_accept) |-> (32'(ld_word) < LOCAL_WORDS));
`endif

endmodule

`default_nettype wire
