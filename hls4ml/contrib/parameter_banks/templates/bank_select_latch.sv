`timescale 1 ns / 1 ps

// Bank ownership for the banked IP.
//
// A request for the selected bank goes straight to the IP, so transactions
// overlap as they do on the bare IP. A request for another bank waits until the
// IP is idle; cur_bank_id then changes and the request goes through on the next
// cycle, so every transaction reads one bank from its first cycle to its last.
// The bank count is a power of two, so every bank id exists.
//
// The loader may write while the IP is idle and no request is presented.

`default_nettype none

module bank_select_latch #(
    parameter int BANK_ID_WIDTH = 1
) (
    input  wire                     ap_clk,
    input  wire                     ap_rst,

    input  wire                     ext_ap_start,
    input  wire [BANK_ID_WIDTH-1:0] ext_bank_id,
    output wire                     ext_ap_ready,

    output wire                     hls_ap_start,
    input  wire                     hls_ap_ready,
    input  wire                     hls_ap_idle,

    output reg  [BANK_ID_WIDTH-1:0] cur_bank_id,
    output wire                     quiescent
);

  wire same = (ext_bank_id == cur_bank_id);

  assign hls_ap_start = ext_ap_start & same;
  assign ext_ap_ready = same & hls_ap_ready;
  assign quiescent    = hls_ap_idle & ~ext_ap_start;

  always_ff @(posedge ap_clk) begin
    if (ap_rst) cur_bank_id <= '0;
    else if (ext_ap_start & ~same & hls_ap_idle) cur_bank_id <= ext_bank_id;
  end

`ifdef PARAMETER_BANKS_ASSERT
  a_switch_only_when_idle : assert property (@(posedge ap_clk) disable iff (ap_rst)
                              $changed(cur_bank_id) |-> $past(hls_ap_idle));
`endif

endmodule

`default_nettype wire
