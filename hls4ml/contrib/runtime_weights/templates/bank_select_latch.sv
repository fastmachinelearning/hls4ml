`timescale 1 ns / 1 ps

// Idle-time ("non-overlapped") bank ownership for runtime-banked weights.
//
// Acceptance follows the real ap_ctrl_hs handshake: ap_start is HELD until the IP
// asserts ap_ready, and the transaction is accepted on (ap_start && ap_ready).
//
// cur_bank_id is committed when the REQUEST is captured, one cycle before
// ap_start is raised, so the banked memories already present the right bank on
// the cycle the IP starts. This matters for short-latency designs, where the
// first weight read can occur in the accept cycle itself.
//
// An out-of-range bank_id is REJECTED: no request is captured and ext_bank_id_bad
// is raised. While the wrapper is idle no transaction can read, so the loader may
// write ANY bank; once a request is pending or in flight, loads are blocked.
//
// Overlapped per-transaction switching is not implemented: a new start is gated
// away from the IP until the current transaction reports ap_done.

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
    input  wire                     hls_ap_done,

    output reg  [BANK_ID_WIDTH-1:0] cur_bank_id,
    output reg                      busy,
    output wire                     quiescent
);

  reg start_pending;

  wire bank_ok  = (ext_bank_id < N_BANKS[BANK_ID_WIDTH:0]);
  wire can_take = ~busy & ~start_pending;
  wire capture  = ext_ap_start & can_take & bank_ok;

  // held high until the IP accepts, per ap_ctrl_hs
  assign hls_ap_start    = start_pending;
  wire   accept          = start_pending & hls_ap_ready;

  assign ext_ap_ready    = can_take;
  assign ext_bank_id_bad = ext_ap_start & can_take & ~bank_ok;

  // idle: nothing pending and nothing in flight -> loader may write any bank
  assign quiescent = ~busy & ~start_pending & hls_ap_idle;

  always_ff @(posedge ap_clk) begin
    if (ap_rst) begin
      busy          <= 1'b0;
      start_pending <= 1'b0;
      cur_bank_id   <= '0;
    end else begin
      if (capture) begin
        start_pending <= 1'b1;
        cur_bank_id   <= ext_bank_id;   // committed before ap_start rises
      end
      if (accept) begin
        start_pending <= 1'b0;
        busy          <= 1'b1;
      end else if (hls_ap_done) begin
        busy          <= 1'b0;
      end
    end
  end

`ifdef RUNTIME_WEIGHTS_ASSERT
  a_no_overlap        : assert property (@(posedge ap_clk) disable iff (ap_rst)
                          accept |-> !busy);
  a_bank_in_range     : assert property (@(posedge ap_clk) disable iff (ap_rst)
                          (busy | start_pending) |-> cur_bank_id < N_BANKS);
  a_bad_bank_rejected : assert property (@(posedge ap_clk) disable iff (ap_rst)
                          ext_bank_id_bad |=> !start_pending);
  a_reset_clears      : assert property (@(posedge ap_clk) ap_rst |=> !busy && !start_pending);
`endif

endmodule

`default_nettype wire
