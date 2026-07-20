////////////////////////////////////////////////////////////////////////////////
//
// Filename: 	sfifo.v
// {{{
// Project:	WB2AXIPSP: bus bridges and other odds and ends
//
// Purpose:	A synchronous data FIFO.
//
// Creator:	Dan Gisselquist, Ph.D.
//		Gisselquist Technology, LLC
//
////////////////////////////////////////////////////////////////////////////////
//
// Written and distributed by Gisselquist Technology, LLC
// }}}
// This design is hereby granted to the public domain.
// {{{
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTIBILITY or
// FITNESS FOR A PARTICULAR PURPOSE.
//
////////////////////////////////////////////////////////////////////////////////
//
`default_nettype	none
// }}}
module sfifo #(
		// {{{
		parameter	BW=8,	// Byte/data width
		parameter 	LGFLEN=4,
		parameter [0:0]	OPT_ASYNC_READ = 1'b1,
		parameter [0:0]	OPT_WRITE_ON_FULL = 1'b0,
		parameter [0:0]	OPT_READ_ON_EMPTY = 1'b0
		// }}}
	) (
		// {{{
		input	wire		i_clk,
		input	wire		i_reset,
		//
		// Write interface
		input	wire		i_wr,
		input	wire [(BW-1):0]	i_data,
		output	wire 		o_full,
		output	reg [LGFLEN:0]	o_fill,
		//
		// Read interface
		input	wire		i_rd,
		output	reg [(BW-1):0]	o_data,
		output	wire		o_empty	// True if FIFO is empty
		// }}}
	);

	// Register/net declarations
	// {{{
	localparam	FLEN=(1<<LGFLEN);
	reg			r_empty;
	wire			r_full;
	reg	[(BW-1):0]	mem[0:(FLEN-1)];
	reg	[LGFLEN:0]	wr_addr, rd_addr;

	wire	w_wr = (i_wr && !o_full);
	wire	w_rd = (i_rd && !o_empty);
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Write half
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	// o_fill
	// {{{
	initial	o_fill = 0;
	always @(posedge i_clk)
	if (i_reset)
		o_fill <= 0;
	else case({ w_wr, w_rd })
	2'b01: o_fill <= o_fill - 1'b1;
	2'b10: o_fill <= o_fill + 1'b1;
	default: begin end // o_fill <= wr_addr - rd_addr;
	endcase
	// }}}

	// r_full, o_full
	// {{{
	/*
	initial	r_full = 0;
	always @(posedge i_clk)
	if (i_reset)
		r_full <= 0;
	else case({ w_wr, w_rd})
	2'b01: r_full <= 1'b0;
	2'b10: r_full <= (o_fill == { 1'b0, {(LGFLEN){1'b1}} });
	default: r_full <= (o_fill == { 1'b1, {(LGFLEN){1'b0}} });
	endcase
	*/
	assign	r_full = o_fill[LGFLEN];

	assign	o_full = (i_rd && OPT_WRITE_ON_FULL) ? 1'b0 : r_full;
	// }}}

	// wr_addr, the write address pointer
	// {{{
	initial	wr_addr = 0;
	always @(posedge i_clk)
	if (i_reset)
		wr_addr <= 0;
	else if (w_wr)
		wr_addr <= wr_addr + 1'b1;
	// }}}

	// Write to memory
	// {{{
	always @(posedge i_clk)
	if (w_wr)
		mem[wr_addr[(LGFLEN-1):0]] <= i_data;
	// }}}
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Read half
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	// rd_addr, the read address pointer
	// {{{
	initial	rd_addr = 0;
	always @(posedge i_clk)
	if (i_reset)
		rd_addr <= 0;
	else if (w_rd)
		rd_addr <= rd_addr + 1'b1;
	// }}}

	// r_empty, o_empty
	// {{{
	initial	r_empty = 1'b1;
	always @(posedge i_clk)
	if (i_reset)
		r_empty <= 1'b1;
	else case ({ w_wr, w_rd })
	2'b01: r_empty <= (o_fill <= 1);
	2'b10: r_empty <= 1'b0;
	default: begin end
	endcase

	assign	o_empty = (OPT_READ_ON_EMPTY && i_wr) ? 1'b0 : r_empty;
	// }}}

	// Read from the FIFO
	// {{{
	generate if (OPT_ASYNC_READ && OPT_READ_ON_EMPTY)
	begin : ASYNCHRONOUS_READ_ON_EMPTY
		// o_data
		// {{{
		always @(*)
		begin
			o_data = mem[rd_addr[LGFLEN-1:0]];
			if (r_empty)
				o_data = i_data;
		end
		// }}}
	end else if (OPT_ASYNC_READ)
	begin : ASYNCHRONOUS_READ
		// o_data
		// {{{
		always @(*)
			o_data = mem[rd_addr[LGFLEN-1:0]];
		// }}}
	end else begin : REGISTERED_READ
		// {{{
		reg		bypass_valid;
		reg [BW-1:0]	bypass_data, rd_data;
		reg [LGFLEN-1:0]	rd_next;

		always @(*)
			rd_next = rd_addr[LGFLEN-1:0] + 1;

		// Memory read, bypassing it if we must
		// {{{
		initial bypass_valid = 0;
		always @(posedge i_clk)
		if (i_reset)
			bypass_valid <= 0;
		else if (r_empty || i_rd)
		begin
			if (!i_wr)
				bypass_valid <= 1'b0;
			else if (r_empty || (i_rd && (o_fill == 1)))
				bypass_valid <= 1'b1;
			else
				bypass_valid <= 1'b0;
		end

		always @(posedge i_clk)
		if (r_empty || i_rd)
			bypass_data <= i_data;

		// NxMap rejects "partial" memory initialisers (the original
		// `initial mem[0] = 0;` caused "Critical: Invalid Init bit"
		// during pre-mapping).  Initialise every entry explicitly so
		// the mapper accepts the pattern on all targets.
		integer _init_idx;
		initial begin
			for (_init_idx = 0; _init_idx < FLEN; _init_idx = _init_idx + 1)
				mem[_init_idx] = 0;
		end
		initial rd_data = 0;
		always @(posedge i_clk)
		if (w_rd)
			rd_data <= mem[rd_next];

		always @(*)
		if (OPT_READ_ON_EMPTY && r_empty)
			o_data = i_data;
		else if (bypass_valid)
			o_data = bypass_data;
		else
			o_data = rd_data;
		// }}}
		// }}}
	end endgenerate
	// }}}
	// }}}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// FORMAL METHODS
// {{{
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// }}}
endmodule
`ifndef	YOSYS
`default_nettype wire
`endif
