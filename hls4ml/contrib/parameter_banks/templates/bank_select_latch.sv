`timescale 1 ns / 1 ps

// Bank ownership for the banked IP.
//
// A request for the selected bank goes straight to the IP, so transactions
// overlap as they do on the bare IP. A request for another bank waits until the
// IP is idle; cur_bank_id then changes and the request goes through on the next
// cycle, so every transaction reads one bank from its first cycle to its last.
// A request for a bank that does not exist is consumed, flagged on
// ext_bank_id_bad, and never reaches the IP.
//
// The loader may write while the IP is idle and no request is presented.

`default_nettype none

module bank_select_latch #(
    parameter int BANK_ID_WIDTH = 1,
    parameter int N_BANKS       = 2
) (
    input  wire                     ap_clk,
    input  wire                     ap_rst,

    input  wire                     ext_ap_start,
    input  wire [BANK_ID_WIDTH-1:0] ext_bank_id,
    output wire                     ext_ap_ready,
    output wire                     ext_bank_id_bad,

    output wire                     hls_ap_start,
    input  wire                     hls_ap_ready,
    input  wire                     hls_ap_idle,

    output reg  [BANK_ID_WIDTH-1:0] cur_bank_id,
    output wire                     busy,
    output wire                     quiescent
);

  wire bank_ok = (ext_bank_id < N_BANKS[BANK_ID_WIDTH:0]);
  wire same    = (ext_bank_id == cur_bank_id);

  assign hls_ap_start    = ext_ap_start & bank_ok & same;
  assign ext_ap_ready    = bank_ok ? (same & hls_ap_ready) : 1'b1;
  assign ext_bank_id_bad = ext_ap_start & ~bank_ok;

  assign busy      = ~hls_ap_idle;
  assign quiescent = hls_ap_idle & ~ext_ap_start;

  always_ff @(posedge ap_clk) begin
    if (ap_rst) cur_bank_id <= '0;
    else if (ext_ap_start & bank_ok & ~same & hls_ap_idle) cur_bank_id <= ext_bank_id;
  end

`ifdef PARAMETER_BANKS_ASSERT
  a_switch_only_when_idle : assert property (@(posedge ap_clk) disable iff (ap_rst)
                              $changed(cur_bank_id) |-> $past(hls_ap_idle));
  a_bad_bank_not_started  : assert property (@(posedge ap_clk) disable iff (ap_rst)
                              ext_bank_id_bad |-> !hls_ap_start);
`endif

endmodule

`default_nettype wire
