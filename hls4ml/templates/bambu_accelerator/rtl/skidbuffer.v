////////////////////////////////////////////////////////////////////////////////
//
// Filename:	rtl/skidbuffer.v
// {{{
// Project:	WB2AXIPSP: bus bridges and other odds and ends
//
// Purpose:	A basic SKID buffer.
// {{{
//	Skid buffers are required for high throughput AXI code, since the AXI
//	specification requires that all outputs be registered.  This means
//	that, if there are any stall conditions calculated, it will take a clock
//	cycle before the stall can be propagated up stream.  This means that
//	the data will need to be buffered for a cycle until the stall signal
//	can make it to the output.
//
//	Handling that buffer is the purpose of this core.
//
//	On one end of this core, you have the i_valid and i_data inputs to
//	connect to your bus interface.  There's also a registered o_ready
//	signal to signal stalls for the bus interface.
//
//	The other end of the core has the same basic interface, but it isn't
//	registered.  This allows you to interact with the bus interfaces
//	as though they were combinatorial logic, by interacting with this half
//	of the core.
//
//	If at any time the incoming !stall signal, i_ready, signals a stall,
//	the incoming data is placed into a buffer.  Internally, that buffer
//	is held in r_data with the r_valid flag used to indicate that valid
//	data is within it.
// }}}
// Parameters:
// {{{
//	DW or data width
//		In order to make this core generic, the width of the data in the
//		skid buffer is parameterized
//
//	OPT_LOWPOWER
//		Forces both o_data and r_data to zero if the respective *VALID
//		signal is also low.  While this costs extra logic, it can also
//		be used to guarantee that any unused values aren't toggling and
//		therefore unnecessarily using power.
//
//		This excess toggling can be particularly problematic if the
//		bus signals have a high fanout rate, or a long signal path
//		across an FPGA.
//
//	OPT_OUTREG
//		Causes the outputs to be registered
//
//	OPT_PASSTHROUGH
//		Turns the skid buffer into a passthrough.  Used for formal
//		verification only.
// }}}
// Creator:	Dan Gisselquist, Ph.D.
//		Gisselquist Technology, LLC
//
////////////////////////////////////////////////////////////////////////////////
// }}}
// Copyright (C) 2019-2025, Gisselquist Technology, LLC
// {{{
// This file is part of the WB2AXIP project.
//
// The WB2AXIP project contains free software and gateware, licensed under the
// Apache License, Version 2.0 (the "License").  You may not use this project,
// or this file, except in compliance with the License.  You may obtain a copy
// of the License at
// }}}
//	http://www.apache.org/licenses/LICENSE-2.0
// {{{
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations
// under the License.
//
////////////////////////////////////////////////////////////////////////////////
//
`default_nettype none
// }}}
module skidbuffer #(
		// {{{
		parameter	[0:0]	OPT_LOWPOWER = 0,
		parameter	[0:0]	OPT_OUTREG = 1,
		//
		parameter	[0:0]	OPT_PASSTHROUGH = 0,
		parameter		DW = 8,
		parameter	[0:0]	OPT_INITIAL = 1'b1
		// }}}
	) (
		// {{{
		input	wire			i_clk, i_reset,
		input	wire			i_valid,
		output	wire			o_ready,
		input	wire	[DW-1:0]	i_data,
		output	wire			o_valid,
		input	wire			i_ready,
		output	reg	[DW-1:0]	o_data
		// }}}
	);

	wire	[DW-1:0]	w_data;

	generate if (OPT_PASSTHROUGH)
	begin : PASSTHROUGH
		// {{{
		assign	{ o_valid, o_ready } = { i_valid, i_ready };

		always @(*)
		if (!i_valid && OPT_LOWPOWER)
			o_data = 0;
		else
			o_data = i_data;

		assign	w_data = 0;

		// Keep Verilator happy
		// Verilator lint_off UNUSED
		// {{{
		wire	unused_passthrough;
		assign	unused_passthrough = &{ 1'b0, i_clk, i_reset };
		// }}}
		// Verilator lint_on  UNUSED
		// }}}
	end else begin : LOGIC
		// We'll start with skid buffer itself
		// {{{
		reg			r_valid;
		reg	[DW-1:0]	r_data;

		// r_valid
		// {{{
		initial if (OPT_INITIAL) r_valid = 0;
		always @(posedge i_clk)
		if (i_reset)
			r_valid <= 0;
		else if ((i_valid && o_ready) && (o_valid && !i_ready))
			// We have incoming data, but the output is stalled
			r_valid <= 1;
		else if (i_ready)
			r_valid <= 0;
		// }}}

		// r_data
		// {{{
		initial if (OPT_INITIAL) r_data = 0;
		always @(posedge i_clk)
		if (OPT_LOWPOWER && i_reset)
			r_data <= 0;
		else if (OPT_LOWPOWER && (!o_valid || i_ready))
			r_data <= 0;
		else if ((!OPT_LOWPOWER || !OPT_OUTREG || i_valid) && o_ready)
			r_data <= i_data;

		assign	w_data = r_data;
		// }}}

		// o_ready
		// {{{
		assign o_ready = !r_valid;
		// }}}

		//
		// And then move on to the output port
		//
		if (!OPT_OUTREG)
		begin : NET_OUTPUT
			// Outputs are combinatorially determined from inputs
			// {{{
			// o_valid
			// {{{
			assign	o_valid = !i_reset && (i_valid || r_valid);
			// }}}

			// o_data
			// {{{
			always @(*)
			if (r_valid)
				o_data = r_data;
			else if (!OPT_LOWPOWER || i_valid)
				o_data = i_data;
			else
				o_data = 0;
			// }}}
			// }}}
		end else begin : REG_OUTPUT
			// Register our outputs
			// {{{
			// o_valid
			// {{{
			reg	ro_valid;

			initial if (OPT_INITIAL) ro_valid = 0;
			always @(posedge i_clk)
			if (i_reset)
				ro_valid <= 0;
			else if (!o_valid || i_ready)
				ro_valid <= (i_valid || r_valid);

			assign	o_valid = ro_valid;
			// }}}

			// o_data
			// {{{
			initial if (OPT_INITIAL) o_data = 0;
			always @(posedge i_clk)
			if (OPT_LOWPOWER && i_reset)
				o_data <= 0;
			else if (!o_valid || i_ready)
			begin

				if (r_valid)
					o_data <= r_data;
				else if (!OPT_LOWPOWER || i_valid)
					o_data <= i_data;
				else
					o_data <= 0;
			end
			// }}}

			// }}}
		end
		// }}}
	end endgenerate

	// Keep Verilator happy
	// {{{
	// verilator coverage_off
	// Verilator lint_off UNUSED
	wire	unused;
	assign	unused = &{ 1'b0, w_data };
	// Verilator lint_on  UNUSED
	// verilator coverage_on
	// }}}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// Formal properties
// {{{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
`ifdef	FORMAL
`ifdef	SKIDBUFFER
`define	ASSUME	assume
`else
`define	ASSUME	assert
`endif

	reg	f_past_valid;

	initial	f_past_valid = 0;
	always @(posedge i_clk)
		f_past_valid <= 1;

	always @(*)
	if (!f_past_valid)
		assume(i_reset);

	////////////////////////////////////////////////////////////////////////
	//
	// Incoming stream properties / assumptions
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	always @(posedge i_clk)
	if (!f_past_valid)
	begin
		`ASSUME(!i_valid || !OPT_INITIAL);
	end else if ($past(i_valid && !o_ready && !i_reset) && !i_reset)
		`ASSUME(i_valid && $stable(i_data));

`ifdef	VERIFIC
`define	FORMAL_VERIFIC
	// Reset properties
	property RESET_CLEARS_IVALID;
		@(posedge i_clk) i_reset |=> !i_valid;
	endproperty

	property IDATA_HELD_WHEN_NOT_READY;
		@(posedge i_clk) disable iff (i_reset)
		i_valid && !o_ready |=> i_valid && $stable(i_data);
	endproperty

`ifdef	SKIDBUFFER
	assume	property (IDATA_HELD_WHEN_NOT_READY);
`else
	assert	property (IDATA_HELD_WHEN_NOT_READY);
`endif
`endif
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Outgoing stream properties / assumptions
	// {{{
	////////////////////////////////////////////////////////////////////////
	//

	generate if (!OPT_PASSTHROUGH)
	begin

		always @(posedge i_clk)
		if (!f_past_valid) // || $past(i_reset))
		begin
			// Following any reset, valid must be deasserted
			assert(!o_valid || !OPT_INITIAL);
		end else if ($past(o_valid && !i_ready && !i_reset) && !i_reset)
			// Following any stall, valid must remain high and
			// data must be preserved
			assert(o_valid && $stable(o_data));

	end endgenerate
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Other properties
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
	generate if (!OPT_PASSTHROUGH)
	begin
		// Rule #1:
		//	If registered, then following any reset we should be
		//	ready for a new request
		// {{{
		always @(posedge i_clk)
		if (f_past_valid && $past(OPT_OUTREG && i_reset))
			assert(o_ready);
		// }}}

		// Rule #2:
		//	All incoming data must either go directly to the
		//	output port, or into the skid buffer
		// {{{
`ifndef	VERIFIC
		always @(posedge i_clk)
		if (f_past_valid && !$past(i_reset) && $past(i_valid && o_ready
			&& (!OPT_OUTREG || o_valid) && !i_ready))
			assert(!o_ready && w_data == $past(i_data));
`else
		assert property (@(posedge i_clk)
			disable iff (i_reset)
			(i_valid && o_ready
				&& (!OPT_OUTREG || o_valid) && !i_ready)
				|=> (!o_ready && w_data == $past(i_data)));
`endif
		// }}}

		// Rule #3:
		//	After the last transaction, o_valid should become idle
		// {{{
		if (!OPT_OUTREG)
		begin
			// {{{
			always @(posedge i_clk)
			if (f_past_valid && !$past(i_reset) && !i_reset
					&& $past(i_ready))
			begin
				assert(o_valid == i_valid);
				assert(!i_valid || (o_data == i_data));
			end
			// }}}
		end else begin
			// {{{
			always @(posedge i_clk)
			if (f_past_valid && !$past(i_reset))
			begin
				if ($past(i_valid && o_ready))
					assert(o_valid);

				if ($past(!i_valid && o_ready && i_ready))
					assert(!o_valid);
			end
			// }}}
		end
		// }}}

		// Rule #4
		//	Same thing, but this time for o_ready
		// {{{
		always @(posedge i_clk)
		if (f_past_valid && $past(!o_ready && i_ready))
			assert(o_ready);
		// }}}

		// If OPT_LOWPOWER is set, o_data and w_data both need to be
		// zero any time !o_valid or !r_valid respectively
		// {{{
		if (OPT_LOWPOWER)
		begin
			always @(*)
			if ((OPT_OUTREG || !i_reset) && !o_valid)
				assert(o_data == 0);

			always @(*)
			if (o_ready)
				assert(w_data == 0);

		end
		// }}}
	end endgenerate
	// }}}

	always @(posedge i_clk)
	if (!OPT_PASSTHROUGH && !i_reset && !o_ready)
		assert(o_valid);

	////////////////////////////////////////////////////////////////////////
	//
	// Cover checks
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
`ifdef	SKIDBUFFER
	generate if (!OPT_PASSTHROUGH)
	begin
		reg	f_changed_data;

		initial	f_changed_data = 0;
		always @(posedge i_clk)
		if (i_reset)
			f_changed_data <= 1;
		else if (i_valid && $past(!i_valid || o_ready))
		begin
			if (i_data != $past(i_data + 1))
				f_changed_data <= 0;
		end else if (!i_valid && i_data != 0)
			f_changed_data <= 0;


`ifndef	VERIFIC
		reg	[3:0]	cvr_steps, cvr_hold;

		always @(posedge i_clk)
		if (i_reset)
		begin
			cvr_steps <= 0;
			cvr_hold  <= 0;
		end else begin
			cvr_steps <= cvr_steps + 1;
			cvr_hold  <= cvr_hold  + 1;
			case(cvr_steps)
			 0: if (o_valid || i_valid)
				cvr_steps <= 0;
			 1: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			 2: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			 3: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			 4: if (!i_valid ||  i_ready)
				cvr_steps <= 0;
			 5: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			 6: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			 7: if (!i_valid ||  i_ready)
				cvr_steps <= 0;
			 8: if (!i_valid ||  i_ready)
				cvr_steps <= 0;
			 9: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			10: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			11: if (!i_valid || !i_ready)
				cvr_steps <= 0;
			12: begin
				cvr_steps <= cvr_steps;
				cover(!o_valid && !i_valid && f_changed_data);
				if (!o_valid || !i_ready)
					cvr_steps <= 0;
				else
					cvr_hold <= cvr_hold + 1;
				end
			default: assert(0);
			endcase
		end

`else
		// Cover test
		cover property (@(posedge i_clk)
			disable iff (i_reset)
			(!o_valid && !i_valid)
			##1 i_valid &&  i_ready [*3]
			##1 i_valid && !i_ready
			##1 i_valid &&  i_ready [*2]
			##1 i_valid && !i_ready [*2]
			##1 i_valid &&  i_ready [*3]
			// Wait for the design to clear
			##1 o_valid && i_ready [*0:5]
			##1 (!o_valid && !i_valid && f_changed_data));
`endif
	end endgenerate
`endif	// SKIDBUFFER
	// }}}
`endif
// }}}
endmodule
`default_nettype wire
