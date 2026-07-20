////////////////////////////////////////////////////////////////////////////////
//
// Filename:	rtl/axi_addr.v
// {{{
// Project:	WB2AXIPSP: bus bridges and other odds and ends
//
// Purpose:	The AXI (full) standard has some rather complicated addressing
//		modes, where the address can either be FIXED, INCRementing, or
//	even where it can WRAP around some boundary.  When in either INCR or
//	WRAP modes, the next address must always be aligned.  In WRAP mode,
//	the next address calculation needs to wrap around a given value, and
//	that value is dependent upon the burst size (i.e. bytes per beat) and
//	length (total numbers of beats).  Since this calculation can be
//	non-trivial, and since it needs to be done multiple times, the logic
//	below captures it for every time it might be needed.
//
// 20200918 - modified to accommodate (potential) AXI3 burst lengths
//
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
module	axi_addr #(
		// {{{
		parameter	AW = 32,
				DW = 32,
		// parameter [0:0]	OPT_AXI3 = 1'b0,
		parameter	LENB = 8
		// }}}
	) (
		// {{{
		input	wire	[AW-1:0]	i_last_addr,
		input	wire	[2:0]		i_size, // 1b, 2b, 4b, 8b, etc
		input	wire	[1:0]		i_burst, // fixed, incr, wrap, reserved
		input	wire	[LENB-1:0]	i_len,
		output	wire	[AW-1:0]	o_next_addr
		// }}}
	);

	// Parameter/register declarations
	// {{{
	localparam		DSZ = $clog2(DW)-3;
	localparam [1:0]	FIXED     = 2'b00;
	// localparam [1:0]	INCREMENT = 2'b01;
	// localparam [1:0]	WRAP      = 2'b10;
	localparam		IN_AW = (AW >= 12) ? 12 : AW;
	localparam [IN_AW-1:0]	ONE = 1;

	reg	[IN_AW-1:0]	wrap_mask, increment;
	reg	[IN_AW-1:0]	crossblk_addr, aligned_addr, unaligned_addr;
	// }}}

	// Address increment
	// {{{
	always @(*)
	if (DSZ == 0)
		increment = 1;
	else if (DSZ == 1)
		increment = (i_size[0]) ? 2 : 1;
	else if (DSZ == 2)
		increment = (i_size[1]) ? 4 : ((i_size[0]) ? 2 : 1);
	else if (DSZ == 3)
		case(i_size[1:0])
		2'b00: increment = 1;
		2'b01: increment = 2;
		2'b10: increment = 4;
		2'b11: increment = 8;
		endcase
	else
		increment = (ONE<<i_size);
	// }}}

	// wrap_mask
	// {{{
	// The wrap_mask is used to determine which bits remain stable across
	// the burst, and which are allowed to change.  It is only used during
	// wrapped addressing.
	always @(*)
	begin
		// Start with the default, minimum mask

		/*
		// Here's the original code.  It works, but it's
		// not economical (uses too many LUTs)
		//
		if (i_len[3:0] == 1)
			wrap_mask = (1<<(i_size+1));
		else if (i_len[3:0] == 3)
			wrap_mask = (1<<(i_size+2));
		else if (i_len[3:0] == 7)
			wrap_mask = (1<<(i_size+3));
		else if (i_len[3:0] == 15)
			wrap_mask = (1<<(i_size+4));
		wrap_mask = wrap_mask - 1;
		*/

		// Here's what we *want*
		//
		// wrap_mask[i_size:0] = -1;
		//
		// On the other hand, since we're already guaranteed that our
		// addresses are aligned, do we really care about
		// wrap_mask[i_size-1:0] ?

		// What we want:
		//
		// wrap_mask[i_size+3:i_size] |= i_len[3:0]
		//
		// We could simplify this to
		//
		// wrap_mask = wrap_mask | (i_len[3:0] << (i_size));
		// Verilator lint_off WIDTH
		if (DSZ < 2)
			wrap_mask = ONE | ({{(IN_AW-4){1'b0}},i_len[3:0]} << (i_size[0]));
		else if (DSZ < 4)
			wrap_mask = ONE | ({{(IN_AW-4){1'b0}},i_len[3:0]} << (i_size[1:0]));
		else
			wrap_mask = ONE | ({{(IN_AW-4){1'b0}},i_len[3:0]} << (i_size));
		// Verilator lint_on  WIDTH
	end
	// }}}

	// unaligned_addr
	always @(*)
		unaligned_addr = i_last_addr[IN_AW-1:0] + increment[IN_AW-1:0];

	// aligned_addr
	// {{{
	always @(*)
	if (i_burst != FIXED)
	begin
		// Align subsequent beats in any burst
		// {{{
		aligned_addr = unaligned_addr;
		// We use the bus size here to simplify the logic
		// required in case the bus is smaller than the
		// maximum.  This depends upon AxSIZE being less than
		// $clog2(DATA_WIDTH/8).
		if (DSZ < 2)
		begin
			// {{{
			// Align any subsequent address
			if (i_size[0])
				aligned_addr[0] = 0;
			// }}}
		end else if (DSZ < 4)
		begin
			// {{{
			// Align any subsequent address
			case(i_size[1:0])
			2'b00:  aligned_addr = unaligned_addr;
			2'b01:  aligned_addr[  0] = 0;
			2'b10:  aligned_addr[(AW-1>1) ? 1 : (AW-1):0]= 0;
			2'b11:  aligned_addr[(AW-1>2) ? 2 : (AW-1):0]= 0;
			endcase
			// }}}
		end else begin
			// {{{
			// Align any subsequent address
			case(i_size)
			3'b001: aligned_addr[  0] = 0;
			3'b010: aligned_addr[(AW-1>1) ? 1 : (AW-1):0]=0;
			3'b011: aligned_addr[(AW-1>2) ? 2 : (AW-1):0]=0;
			3'b100: aligned_addr[(AW-1>3) ? 3 : (AW-1):0]=0;
			3'b101: aligned_addr[(AW-1>4) ? 4 : (AW-1):0]=0;
			3'b110: aligned_addr[(AW-1>5) ? 5 : (AW-1):0]=0;
			3'b111: aligned_addr[(AW-1>6) ? 6 : (AW-1):0]=0;
			default: aligned_addr = unaligned_addr;
			endcase
			// }}}
		end
		// }}}
	end else
		aligned_addr = i_last_addr[IN_AW-1:0];
	// }}}

	// crossblk_addr from aligned_addr, for WRAP addressing
	// {{{
	always @(*)
	if (i_burst[1])
	begin
		// WRAP!
		crossblk_addr[IN_AW-1:0] = (i_last_addr[IN_AW-1:0] & ~wrap_mask)
				| (aligned_addr & wrap_mask);
	end else
		crossblk_addr[IN_AW-1:0] = aligned_addr;
	// }}}

	// o_next_addr: Guarantee only the bottom 12 bits change
	// {{{
	// This is really a logic simplification.  AXI bursts aren't allowed
	// to cross 4kB boundaries.  Given that's the case, we don't have to
	// suffer from the propagation across all AW bits, and can limit any
	// address propagation to just the lower 12 bits
	generate if (AW > 12)
	begin : WIDE_ADDRESS
		assign	o_next_addr = { i_last_addr[AW-1:12],
						crossblk_addr[11:0] };
	end else begin : NARROW_ADDRESS
		assign	o_next_addr = crossblk_addr[AW-1:0];
	end endgenerate
	// }}}

	// Make Verilator happy
	// {{{
	// Verilator lint_off UNUSED
	wire	unused;
	assign	unused = (LENB <= 4) ? &{1'b0, i_len[0] }
				: &{ 1'b0, i_len[LENB-1:4], i_len[0] };
	// Verilator lint_on UNUSED
	// }}}
endmodule
`default_nettype wire
