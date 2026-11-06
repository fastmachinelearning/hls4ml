`timescale 1 ns / 1 ps

// The banks of one external parameter, stacked in depth: bank_id * BANK_STRIDE_WORDS + word.
// Both ports are 1-cycle reads for the IP, like the memories Vitis generates; the loader
// takes port B while the IP is idle. The IP never writes it (checked by package()).

`default_nettype none

module parameter_bank #(
    parameter int DATA_WIDTH        = 256,
    // Vitis rounds the port up to a power-of-two number of bytes; the extra bits read as zero
    parameter int PORT_WIDTH        = 256,
    parameter int HLS_ADDR_WIDTH    = 32,
    parameter int WORD_BYTES        = 32,
    parameter int LOCAL_WORDS       = 2,
    parameter int N_BANKS           = 2,
    parameter int BANK_ID_WIDTH     = 1,
    parameter int BANK_STRIDE_WORDS = 2,
    parameter     INIT_HEX          = "",
    parameter     RAM_STYLE         = "block",  // or "distributed"
    // 0 when the IP never reads port B: it is then the loader's write port only
    parameter int PORT_B_READ       = 1
) (
    input  wire                        ap_clk,
    input  wire                        ap_rst,

    input  wire [BANK_ID_WIDTH-1:0]    cur_bank_id,
    input  wire                        quiescent,

    // HLS IP, port A
    input  wire [HLS_ADDR_WIDTH-1:0]   hls_Addr_A,
    input  wire                        hls_EN_A,
    output wire [PORT_WIDTH-1:0]       hls_Dout_A,

    // HLS IP, port B (unused by Dense)
    input  wire [HLS_ADDR_WIDTH-1:0]   hls_Addr_B,
    input  wire                        hls_EN_B,
    output wire [PORT_WIDTH-1:0]       hls_Dout_B,

    // loader, granted port B while quiescent
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

  (* ram_style = RAM_STYLE *)
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

  // Registered read without reset, like the memories Vitis generates. Plain always blocks:
  // `mem` is also written by the loader below, which always_ff would not allow.
  localparam int PAD_BITS = PORT_WIDTH - DATA_WIDTH;

  reg [DATA_WIDTH-1:0] dout_q;

  always @(posedge ap_clk) begin
    if (hls_EN_A) dout_q <= mem[phys_a];
  end

  generate
    if (PAD_BITS > 0) assign hls_Dout_A = {{PAD_BITS{1'b0}}, dout_q};
    else              assign hls_Dout_A = dout_q;
  endgenerate

  assign addr_padding_violation = hls_EN_A & pad_a;

  // ---------------- port B : loader ----------------
  // ld_word can encode words in the padding between banks; those writes are rejected
  wire word_ok  = (32'(ld_word) < LOCAL_WORDS);
  wire safe     = quiescent & word_ok;
  assign ld_accept = ld_req &  safe;
  assign ld_reject = ld_req & ~safe;

  // widened first: a part-select of the narrow ld_word could read past it and give x
  wire [31:0] phys_ld_full = (32'(ld_bank) * BANK_STRIDE_WORDS) + 32'(ld_word);
  wire [PHYS_ADDR_WIDTH-1:0] phys_ld = phys_ld_full[PHYS_ADDR_WIDTH-1:0];

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

  wire [PHYS_ADDR_WIDTH-1:0] phys_b = (PORT_B_READ && !quiescent) ? phys_hls_b : phys_ld;

  reg [DATA_WIDTH-1:0] dout_b_q = '0;

  always @(posedge ap_clk) begin
    if (ld_accept) mem[phys_b] <= ld_wdata;
    else if (PORT_B_READ && hls_EN_B) dout_b_q <= mem[phys_b];
  end

  generate
    if (PAD_BITS > 0) assign hls_Dout_B = {{PAD_BITS{1'b0}}, dout_b_q};
    else              assign hls_Dout_B = dout_b_q;
  endgenerate

`ifdef PARAMETER_BANKS_ASSERT
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
