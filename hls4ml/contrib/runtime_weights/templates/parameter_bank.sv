`timescale 1 ns / 1 ps

// Depth-stacked replacement for the single-bank memory a stock hls4ml IP expects
// on an external BRAM parameter port.
//
// Both ports are 1-cycle enable-gated reads for the IP, matching the memory model
// Vitis generates; bank_addr_mapper is combinational, so the latency is that of the
// unbanked memory. A Dense leaves port B idle, a pointwise convolution reads through
// it, and the loader takes port B only while quiescent -- so nothing here depends on
// how many ports the IP uses. It does depend on the IP never WRITING the memory,
// which interface.bram_is_read_only() proves from the exported RTL.
//
// Banks are stacked in depth: phys_word = bank_id*BANK_STRIDE_WORDS + local_word.
// N_BANKS, BANK_STRIDE_WORDS and the physical depth are build-time constants.

`default_nettype none

module parameter_bank #(
    parameter int DATA_WIDTH        = 256,
    // Width of the RTL port the IP actually drives. Vitis rounds a parameter port
    // up to a power-of-two byte count, so a 96-bit packed word arrives on a 128-bit
    // port. The extra bits carry nothing; they are zero-filled on the read path and
    // zero-filled on the read path, explicitly rather than by implicit extension.
    parameter int PORT_WIDTH        = 256,
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
    output wire [PORT_WIDTH-1:0]       hls_Dout_A,
    input  wire                        hls_Rst_A,

    // facing the stock HLS IP, port B (read only; idle for layers that do not use it)
    input  wire [HLS_ADDR_WIDTH-1:0]   hls_Addr_B,
    input  wire                        hls_EN_B,
    output wire [PORT_WIDTH-1:0]       hls_Dout_B,

    // native loader, granted port B while quiescent
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

  // v1 deliberately forces block RAM so every bank has one predictable
  // implementation; memory-style optimization is out of scope.
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

  // port A: registered read. Plain always blocks (not
  // always_ff) because `mem` is shared with the loader port below; always_ff
  // asserts exclusive ownership of the variables it assigns.
  localparam int PAD_BITS = PORT_WIDTH - DATA_WIDTH;

  reg [DATA_WIDTH-1:0] dout_q;

  always @(posedge ap_clk or posedge hls_Rst_A) begin
    if (hls_Rst_A) dout_q <= {DATA_WIDTH{1'b0}};
    else if (hls_EN_A) dout_q <= mem[phys_a];
  end

  // logical -> physical: the padding bits the IP does not use are driven to zero
  generate
    if (PAD_BITS > 0) assign hls_Dout_A = {{PAD_BITS{1'b0}}, dout_q};
    else              assign hls_Dout_A = dout_q;
  endgenerate

  // There is no IP-side write path: interface.bram_is_read_only() proves both write
  // enables are tied off before packaging, so the loader is the only writer.

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
  wire [31:0] phys_ld_full = (32'(ld_bank) * BANK_STRIDE_WORDS) + 32'(ld_word);
  wire [PHYS_ADDR_WIDTH-1:0] phys_ld = phys_ld_full[PHYS_ADDR_WIDTH-1:0];

  // The IP's own port-B address, translated exactly like port A.
  wire [PHYS_ADDR_WIDTH-1:0] phys_hls_b;
  wire                       pad_b;

  bank_addr_mapper #(
      .HLS_ADDR_WIDTH   (HLS_ADDR_WIDTH),
      .WORD_BYTES       (WORD_BYTES),
      .LOCAL_WORDS      (LOCAL_WORDS),
      .BANK_ID_WIDTH    (BANK_ID_WIDTH),
      .BANK_STRIDE_WORDS(BANK_STRIDE_WORDS),
      .PHYS_ADDR_WIDTH  (PHYS_ADDR_WIDTH)
  ) u_map_b (
      .hls_byte_addr  (hls_Addr_B),
      .bank_id        (cur_bank_id),
      .phys_word_addr (phys_hls_b),
      .addr_in_padding(pad_b)
  );

  // One physical port B, owned by the loader while quiescent and by the IP
  // otherwise. ld_accept already requires quiescent, so the two never collide.
  wire [PHYS_ADDR_WIDTH-1:0] phys_b = quiescent ? phys_ld : phys_hls_b;

  reg [DATA_WIDTH-1:0] dout_b_q;

  always @(posedge ap_clk) begin
    if (ld_accept) mem[phys_b] <= ld_wdata;
    else if (hls_EN_B) dout_b_q <= mem[phys_b];
  end

  generate
    if (PAD_BITS > 0) assign hls_Dout_B = {{PAD_BITS{1'b0}}, dout_b_q};
    else              assign hls_Dout_B = dout_b_q;
  endgenerate

`ifdef RUNTIME_WEIGHTS_ASSERT
  a_no_padding_access   : assert property (@(posedge ap_clk) disable iff (ap_rst)
                            !(hls_EN_A & pad_a));
  a_no_padding_access_b : assert property (@(posedge ap_clk) disable iff (ap_rst)
                            !(hls_EN_B & ~quiescent & pad_b));
  a_write_only_when_idle: assert property (@(posedge ap_clk) disable iff (ap_rst)
                            (ld_req & ld_accept) |-> quiescent);
  a_write_in_bounds     : assert property (@(posedge ap_clk) disable iff (ap_rst)
                            (ld_req & ld_accept) |-> (32'(ld_word) < LOCAL_WORDS));
`endif

endmodule

`default_nettype wire
