////////////////////////////////////////////////////////////////////////////////
//
// Filename:	rtl/demofull.v
// {{{
// Project:	WB2AXIPSP: bus bridges and other odds and ends
//
// Purpose:	Demonstrate a formally verified AXI4 core with a (basic)
//		interface.  This interface is explained below.
//
// Performance: This core has been designed for a total throughput of one beat
//		per clock cycle.  Both read and write channels can achieve
//	this.  The write channel will also introduce two clocks of latency,
//	assuming no other latency from the master.  This means it will take
//	a minimum of 3+AWLEN clock cycles per transaction of (1+AWLEN) beats,
//	including both address and acknowledgment cycles.  The read channel
//	will introduce a single clock of latency, requiring 2+ARLEN cycles
//	per transaction of 1+ARLEN beats.
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
`default_nettype	none
// }}}
module AXISlaveStream #(
		// {{{
		parameter integer C_S_AXI_ID_WIDTH	= 12,
		parameter integer C_S_AXI_DATA_WIDTH	= 128,
		parameter integer C_S_AXI_ADDR_WIDTH	= 40,
		parameter	[0:0]	OPT_LOCK     = 1'b0,
		parameter	[0:0]	OPT_LOCKID   = 1'b1,
		parameter	[0:0]	OPT_LOWPOWER = 1'b0,
		parameter HLS_IN_DATA_W   = 16,
		parameter HLS_OUT_DATA_W  = 16,
		parameter HLS_IN_N_WORDS  = 8,
		parameter HLS_OUT_N_WORDS = 8,
		parameter LGFLEN_IN       = 4,
		parameter LGFLEN_OUT      = 4
		// }}}
	) (
		// {{{
		// User ports
		// {{{
		// A very basic protocol-independent peripheral interface
		// 1. A value will be written any time o_we is true
		// 2. A value will be read any time o_rd is true
		// 3. Such a slave might just as easily be written as:
		//
		//	always @(posedge S_AXI_ACLK)
		//	if (o_we)
		//	begin
		//	    for(k=0; k<C_S_AXI_DATA_WIDTH/8; k=k+1)
		//	    begin
		//		if (o_wstrb[k])
		//		mem[o_waddr[AW-1:LSB]][k*8+:8] <= o_wdata[k*8+:8]
		//	    end
		//	end
		//
		//	always @(posedge S_AXI_ACLK)
		//	if (o_rd)
		//		i_rdata <= mem[o_raddr[AW-1:LSB]];
		//
		// 4. The rule on the input is that i_rdata must be registered,
		//    and that it must only change if o_rd is true.  Violating
		//    this rule will cause this core to violate the AXI
		//    protocol standard, as this value is not registered within
		//    this core
// Internal signals mapped from former ports
// We don't need them in the port list


		//
		// User ports ends
		// }}}
		// Do not modify the ports beyond this line
		// AXI signals
		// {{{
		// Global Clock Signal
		input wire  S_AXI_ACLK,
		// Global Reset Signal. This Signal is Active LOW
		input wire  S_AXI_ARESETN,

		// Write address channel
		// {{{
		// Write Address ID
		input wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_AWID,
		// Write address
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
		// Burst length. The burst length gives the exact number of
		// transfers in a burst
		input wire [7 : 0] S_AXI_AWLEN,
		// Burst size. This signal indicates the size of each transfer
		// in the burst
		input wire [2 : 0] S_AXI_AWSIZE,
		// Burst type. The burst type and the size information,
		// determine how the address for each transfer within the burst
		// is calculated.
		input wire [1 : 0] S_AXI_AWBURST,
		// Lock type. Provides additional information about the
		// atomic characteristics of the transfer.
		input wire  S_AXI_AWLOCK,
		// Memory type. This signal indicates how transactions
		// are required to progress through a system.
		input wire [3 : 0] S_AXI_AWCACHE,
		// Protection type. This signal indicates the privilege
		// and security level of the transaction, and whether
		// the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_AWPROT,
		// Quality of Service, QoS identifier sent for each
		// write transaction.
		input wire [3 : 0] S_AXI_AWQOS,
		// Region identifier. Permits a single physical interface
		// on a slave to be used for multiple logical interfaces.
		// Write address valid. This signal indicates that
		// the channel is signaling valid write address and
		// control information.
		input wire  S_AXI_AWVALID,
		// Write address ready. This signal indicates that
		// the slave is ready to accept an address and associated
		// control signals.
		output wire  S_AXI_AWREADY,
		// }}}
		// Write data channel
		// {{{
		// Write Data
		input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
		// Write strobes. This signal indicates which byte
		// lanes hold valid data. There is one write strobe
		// bit for each eight bits of the write data bus.
		input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
		// Write last. This signal indicates the last transfer
		// in a write burst.
		input wire  S_AXI_WLAST,
		// Optional User-defined signal in the write data channel.
		// Write valid. This signal indicates that valid write
		// data and strobes are available.
		input wire  S_AXI_WVALID,
		// Write ready. This signal indicates that the slave
		// can accept the write data.
		output wire  S_AXI_WREADY,
		// }}}
		// Write response channel
		// {{{
		// Response ID tag. This signal is the ID tag of the
		// write response.
		output wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_BID,
		// Write response. This signal indicates the status
		// of the write transaction.
		output wire [1 : 0] S_AXI_BRESP,
		// Optional User-defined signal in the write response channel.
		// Write response valid. This signal indicates that the
		// channel is signaling a valid write response.
		output wire  S_AXI_BVALID,
		// Response ready. This signal indicates that the master
		// can accept a write response.
		input wire  S_AXI_BREADY,
		// }}}
		// Read address channel
		// {{{
		// Read address ID. This signal is the identification
		// tag for the read address group of signals.
		input wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_ARID,
		// Read address. This signal indicates the initial
		// address of a read burst transaction.
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
		// Burst length. The burst length gives the exact number of
		// transfers in a burst
		input wire [7 : 0] S_AXI_ARLEN,
		// Burst size. This signal indicates the size of each transfer
		// in the burst
		input wire [2 : 0] S_AXI_ARSIZE,
		// Burst type. The burst type and the size information,
		// determine how the address for each transfer within the
		// burst is calculated.
		input wire [1 : 0] S_AXI_ARBURST,
		// Lock type. Provides additional information about the
		// atomic characteristics of the transfer.
		input wire  S_AXI_ARLOCK,
		// Memory type. This signal indicates how transactions
		// are required to progress through a system.
		input wire [3 : 0] S_AXI_ARCACHE,
		// Protection type. This signal indicates the privilege
		// and security level of the transaction, and whether
		// the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_ARPROT,
		// Quality of Service, QoS identifier sent for each
		// read transaction.
		input wire [3 : 0] S_AXI_ARQOS,
		// Region identifier. Permits a single physical interface
		// on a slave to be used for multiple logical interfaces.
		// Optional User-defined signal in the read address channel.
		// Write address valid. This signal indicates that
		// the channel is signaling valid read address and
		// control information.
		input wire  S_AXI_ARVALID,
		// Read address ready. This signal indicates that
		// the slave is ready to accept an address and associated
		// control signals.
		output wire  S_AXI_ARREADY,
		// }}}
		// Read data (return) channel
		// {{{
		// Read ID tag. This signal is the identification tag
		// for the read data group of signals generated by the slave.
		output wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_RID,
		// Read Data
		output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
		// Read response. This signal indicates the status of
		// the read transfer.
		output wire [1 : 0] S_AXI_RRESP,
		// Read last. This signal indicates the last transfer
		// in a read burst.
		output wire  S_AXI_RLAST,
		// Optional User-defined signal in the read address channel.
		// Read valid. This signal indicates that the channel
		// is signaling the required read data.
		output wire  S_AXI_RVALID,
		// Read ready. This signal indicates that the master can
		// accept the read data and response information.
		input wire  S_AXI_RREADY,
		// }}}
		// }}}
		// }}}

        output wire o_interrupt,

        // AXIS input: AXISlaveStream → HLS IP
        output wire [HLS_IN_DATA_W-1:0]  hls_in_tdata,
        output wire                       hls_in_tvalid,
        input  wire                       hls_in_tready,
        // AXIS output: HLS IP → AXISlaveStream
        input  wire [HLS_OUT_DATA_W-1:0] hls_out_tdata,
        input  wire                       hls_out_tvalid,
        output wire                       hls_out_tready,
        // Control
        output reg  start_out,
        input  wire done_in
	);

	// Local declarations
	// {{{
	// More useful shorthand definitions
	localparam	AW = C_S_AXI_ADDR_WIDTH;
	localparam	DW = C_S_AXI_DATA_WIDTH;
	localparam	IW = C_S_AXI_ID_WIDTH;
	localparam	LSB = $clog2(C_S_AXI_DATA_WIDTH)-3;
	// Local declarations for former ports
	reg					o_we;
	reg	[C_S_AXI_ADDR_WIDTH-LSB-1:0]	o_waddr;
	reg	[C_S_AXI_DATA_WIDTH-1:0]	o_wdata;
	reg	[C_S_AXI_DATA_WIDTH/8-1:0]	o_wstrb;
	reg					o_rd;
	reg	[C_S_AXI_ADDR_WIDTH-LSB-1:0]	o_raddr;
	reg	[C_S_AXI_DATA_WIDTH-1:0]	i_rdata;
	// Double buffer the write response channel only
	reg	[IW-1 : 0]	r_bid;
	reg			r_bvalid;
	reg	[IW-1 : 0]	axi_bid;
	reg			axi_bvalid;

	reg			axi_awready, axi_wready;
	reg	[AW-1:0]	waddr;
	wire	[AW-1:0]	next_wr_addr;

	// Vivado will warn about wlen only using 4-bits.  This is
	// to be expected, since the axi_addr module only needs to use
	// the bottom four bits of wlen to determine address increments
	reg	[7:0]		wlen;
	// Vivado will also warn about the top bit of wsize being unused.
	// This is also to be expected for a DATA_WIDTH of 32-bits.
	reg	[2:0]		wsize;
	reg	[1:0]		wburst;

	wire			m_awvalid, m_awlock;
	reg			m_awready;
	wire	[AW-1:0]	m_awaddr;
	wire	[1:0]		m_awburst;
	wire	[2:0]		m_awsize;
	wire	[7:0]		m_awlen;
	wire	[IW-1:0]	m_awid;

	wire	[AW-1:0]	next_rd_addr;

	// Vivado will warn about rlen only using 4-bits.  This is
	// to be expected, since for a DATA_WIDTH of 32-bits, the axi_addr
	// module only uses the bottom four bits of rlen to determine
	// address increments
	reg	[7:0]		rlen;
	// Vivado will also warn about the top bit of wsize being unused.
	// This is also to be expected for a DATA_WIDTH of 32-bits.
	reg	[2:0]		rsize;
	reg	[1:0]		rburst;
	reg	[IW-1:0]	rid;
	reg			rlock;
	reg			axi_arready;
	reg	[8:0]		axi_rlen;
	reg	[AW-1:0]	raddr;

	// Read skid buffer
	reg			rskd_valid, rskd_last, rskd_lock;
	wire			rskd_ready;
	reg	[IW-1:0]	rskd_id;

	// Exclusive address register checking
	reg			exclusive_write, block_write;
	wire			write_lock_valid;
	reg			axi_exclusive_write;

	// Custom flow control
	wire wr_full;
	wire rd_empty;
	wire w_wready = axi_wready && !wr_full;

	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// AW Skid buffer
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//
	skidbuffer #(
		// {{{
		.DW(AW+2+3+1+8+IW),
		.OPT_LOWPOWER(OPT_LOWPOWER),
		.OPT_OUTREG(1'b0)
		// }}}
	) awbuf(
		// {{{
		.i_clk(S_AXI_ACLK), .i_reset(!S_AXI_ARESETN),
		.i_valid(S_AXI_AWVALID), .o_ready(S_AXI_AWREADY),
			.i_data({ S_AXI_AWADDR, S_AXI_AWBURST, S_AXI_AWSIZE,
				S_AXI_AWLOCK, S_AXI_AWLEN, S_AXI_AWID }),
		.o_valid(m_awvalid), .i_ready(m_awready),
			.o_data({ m_awaddr, m_awburst, m_awsize,
				m_awlock, m_awlen, m_awid })
		// }}}
	);
	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Write processing
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	// axi_awready, axi_wready
	// {{{
	initial	axi_awready = 1;
	initial	axi_wready  = 0;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
	begin
		axi_awready  <= 1;
		axi_wready   <= 0;
	end else if (m_awvalid && m_awready)
	begin
		axi_awready <= 0;
		axi_wready  <= 1;
	end else if (S_AXI_WVALID && w_wready)
	begin
		axi_awready <= (S_AXI_WLAST)&&(!S_AXI_BVALID || S_AXI_BREADY);
		axi_wready  <= (!S_AXI_WLAST);
	end else if (!axi_awready)
	begin
		if (axi_wready)
			axi_awready <= 1'b0;
		else if (r_bvalid && !S_AXI_BREADY)
			axi_awready <= 1'b0;
		else
			axi_awready <= 1'b1;
	end
	// }}}

	// Exclusive write calculation
	// {{{
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN || !OPT_LOCK)
	begin
		exclusive_write <= 0;
		block_write <= 0;
	end else if (m_awvalid && m_awready)
	begin
		exclusive_write <= 1'b0;
		block_write     <= 1'b0;
		if (write_lock_valid)
			exclusive_write <= 1'b1;
		else if (m_awlock)
			block_write <= 1'b1;
	end else if (m_awready)
	begin
		exclusive_write <= 1'b0;
		block_write     <= 1'b0;
	end

	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN || !OPT_LOCK)
		axi_exclusive_write <= 0;
	else if (!S_AXI_BVALID || S_AXI_BREADY)
	begin
		axi_exclusive_write <= exclusive_write;
		if (OPT_LOWPOWER && (!S_AXI_WVALID || !w_wready || !S_AXI_WLAST)
				&& !r_bvalid)
			axi_exclusive_write <= 0;
	end
	// }}}

	// Next write address calculation
	// {{{
	always @(posedge S_AXI_ACLK)
	if (m_awready)
	begin
		waddr    <= m_awaddr;
		wburst   <= m_awburst;
		wsize    <= m_awsize;
		wlen     <= m_awlen;
	end else if (S_AXI_WVALID)
		waddr <= next_wr_addr;

	axi_addr #(
		// {{{
		.AW(AW), .DW(DW)
		// }}}
	) get_next_wr_addr(
		// {{{
		waddr, wsize, wburst, wlen,
			next_wr_addr
		// }}}
	);
	// }}}

	// o_w*
	// {{{
	always @(posedge S_AXI_ACLK)
	begin
		o_we    <= (S_AXI_WVALID && w_wready);
		o_waddr <= waddr[AW-1:LSB];
		o_wdata <= S_AXI_WDATA;
		if (block_write)
			o_wstrb <= 0;
		else
			o_wstrb <= S_AXI_WSTRB;

		if (!S_AXI_ARESETN)
			o_we <= 0;
		if (OPT_LOWPOWER && (!S_AXI_ARESETN || !S_AXI_WVALID
					|| !w_wready))
		begin
			o_waddr <= 0;
			o_wdata <= 0;
			o_wstrb <= 0;
		end
	end
	// }}}

	//
	// Write return path
	// {{{
	// r_bvalid
	// {{{
	initial	r_bvalid = 0;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
		r_bvalid <= 1'b0;
	else if (S_AXI_WVALID && w_wready && S_AXI_WLAST
			&&(S_AXI_BVALID && !S_AXI_BREADY))
		r_bvalid <= 1'b1;
	else if (S_AXI_BREADY)
		r_bvalid <= 1'b0;
	// }}}

	// r_bid, axi_bid
	// {{{
	initial	r_bid = 0;
	initial	axi_bid = 0;
	always @(posedge S_AXI_ACLK)
	begin
		if (m_awready && (!OPT_LOWPOWER || m_awvalid))
			r_bid    <= m_awid;

		if (!S_AXI_BVALID || S_AXI_BREADY)
			axi_bid <= r_bid;

		if (OPT_LOWPOWER && !S_AXI_ARESETN)
		begin
			r_bid <= 0;
			axi_bid <= 0;
		end
	end
	// }}}

	// axi_bvalid
	// {{{
	initial	axi_bvalid = 0;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
		axi_bvalid <= 0;
	else if (S_AXI_WVALID && w_wready && S_AXI_WLAST)
		axi_bvalid <= 1;
	else if (S_AXI_BREADY)
		axi_bvalid <= r_bvalid;
	// }}}

	// m_awready
	// {{{
	always @(*)
	begin
		m_awready = axi_awready;
		if (S_AXI_WVALID && w_wready && S_AXI_WLAST
			&& (!S_AXI_BVALID || S_AXI_BREADY))
			m_awready = 1;
	end
	// }}}

	// At one time, axi_awready was the same as S_AXI_AWREADY.  Now, though,
	// with the extra write address skid buffer, this is no longer the case.
	// S_AXI_AWREADY is handled/created/managed by the skid buffer.
	//
	// assign S_AXI_AWREADY = axi_awready;
	//
	// The rest of these signals can be set according to their registered
	// values above.
	assign	S_AXI_WREADY  = w_wready;
	assign	S_AXI_BVALID  = axi_bvalid;
	assign	S_AXI_BID     = axi_bid;
	//
	// This core does not produce any bus errors, nor does it support
	// exclusive access, so 2'b00 will always be the correct response.
	assign	S_AXI_BRESP = { 1'b0, axi_exclusive_write };
	// }}}

	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Read processing
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	// axi_arready
	// {{{
	initial axi_arready = 1;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
		axi_arready <= 1;
	else if (S_AXI_ARVALID && S_AXI_ARREADY)
		axi_arready <= (S_AXI_ARLEN==0)&&(o_rd);
	else if (o_rd)
		axi_arready <= (axi_rlen <= 1);
	// }}}

	// axi_rlen
	// {{{
	initial	axi_rlen = 0;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
		axi_rlen <= 0;
	else if (S_AXI_ARVALID && S_AXI_ARREADY)
		axi_rlen <= {1'b0, S_AXI_ARLEN} + (o_rd ? 9'd0:9'd1);
	else if (o_rd)
		axi_rlen <= axi_rlen - 1'b1;
	// }}}

	// Next read address calculation
	// {{{
	always @(posedge S_AXI_ACLK)
	if (o_rd)
		raddr <= next_rd_addr;
	else if (S_AXI_ARREADY)
	begin
		raddr <= S_AXI_ARADDR;
		if (OPT_LOWPOWER && !S_AXI_ARVALID)
			raddr <= 0;
	end

	// r*
	// {{{
	always @(posedge S_AXI_ACLK)
	if (S_AXI_ARREADY)
	begin
		rburst   <= S_AXI_ARBURST;
		rsize    <= S_AXI_ARSIZE;
		rlen     <= S_AXI_ARLEN;
		rid      <= S_AXI_ARID;
		rlock    <= S_AXI_ARLOCK && S_AXI_ARVALID && OPT_LOCK;

		if (OPT_LOWPOWER && !S_AXI_ARVALID)
		begin
			rburst   <= 0;
			rsize    <= 0;
			rlen     <= 0;
			rid      <= 0;
			rlock    <= 0;
		end
	end
	// }}}

	axi_addr #(
		// {{{
		.AW(AW), .DW(DW)
		// }}}
	) get_next_rd_addr(
		// {{{
		(S_AXI_ARREADY ? S_AXI_ARADDR : raddr),
		(S_AXI_ARREADY  ? S_AXI_ARSIZE : rsize),
		(S_AXI_ARREADY  ? S_AXI_ARBURST: rburst),
		(S_AXI_ARREADY  ? S_AXI_ARLEN  : rlen),
		next_rd_addr
		// }}}
	);
	// }}}

	// o_rd, o_raddr
	// {{{
	always @(*)
	begin
		o_rd = (S_AXI_ARVALID || !S_AXI_ARREADY);
		if (S_AXI_RVALID && !S_AXI_RREADY)
			o_rd = 0;
		if (rskd_valid && !rskd_ready)
			o_rd = 0;
		if (rd_empty)
			o_rd = 0;
		o_raddr = (S_AXI_ARREADY ? S_AXI_ARADDR[AW-1:LSB] : raddr[AW-1:LSB]);
	end
	// }}}

	// rskd_valid
	// {{{
	initial	rskd_valid = 0;
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN)
		rskd_valid <= 0;
	else if (o_rd)
		rskd_valid <= 1;
	else if (rskd_ready)
		rskd_valid <= 0;
	// }}}

	// rskd_id
	// {{{
	always @(posedge S_AXI_ACLK)
	if (!rskd_valid || rskd_ready)
	begin
		if (S_AXI_ARVALID && S_AXI_ARREADY)
			rskd_id <= S_AXI_ARID;
		else
			rskd_id <= rid;
	end
	// }}}

	// rskd_last
	// {{{
	initial rskd_last   = 0;
	always @(posedge S_AXI_ACLK)
	if (!rskd_valid || rskd_ready)
	begin
		rskd_last <= 0;
		if (o_rd && axi_rlen == 1)
			rskd_last <= 1;
		if (S_AXI_ARVALID && S_AXI_ARREADY && S_AXI_ARLEN == 0 && o_rd)
			rskd_last <= 1;
	end
	// }}}

	// rskd_lock
	// {{{
	always @(posedge S_AXI_ACLK)
	if (!S_AXI_ARESETN || !OPT_LOCK)
		rskd_lock <= 1'b0;
	else if (!rskd_valid || rskd_ready)
	begin
		rskd_lock <= 0;
		if (!OPT_LOWPOWER || o_rd)
		begin
			if (S_AXI_ARVALID && S_AXI_ARREADY)
				rskd_lock <= S_AXI_ARLOCK;
			else
				rskd_lock <= rlock;
		end
	end
	// }}}


	// Outgoing read skidbuffer
	// {{{
	skidbuffer #(
		// {{{
		.OPT_LOWPOWER(OPT_LOWPOWER),
		.OPT_OUTREG(1),
		.DW(IW+2+DW)
		// }}}
	) rskid (
		// {{{
		.i_clk(S_AXI_ACLK), .i_reset(!S_AXI_ARESETN),
		.i_valid(rskd_valid), .o_ready(rskd_ready),
		.i_data({ rskd_id, rskd_lock, rskd_last, i_rdata }),
		.o_valid(S_AXI_RVALID), .i_ready(S_AXI_RREADY),
			.o_data({ S_AXI_RID, S_AXI_RRESP[0], S_AXI_RLAST,
					S_AXI_RDATA })
		// }}}
	);
	// }}}

	assign	S_AXI_RRESP[1] = 1'b0;
	assign	S_AXI_ARREADY = axi_arready;

	// }}}
	////////////////////////////////////////////////////////////////////////
	//
	// Exclusive address caching
	// {{{
	////////////////////////////////////////////////////////////////////////
	//
	//

	generate if (OPT_LOCK && !OPT_LOCKID)
	begin : EXCLUSIVE_ACCESS_BLOCK
		// {{{
		// The AXI4 specification requires that we check one address
		// per ID.  This isn't that.  This algorithm checks one ID,
		// whichever the last ID was.  It's designed to be lighter on
		// the logic requirements, and (unnoticably) not (fully) spec
		// compliant.  (The difference, if noticed at all, will be in
		// performance when multiple masters try to perform an exclusive
		// transaction at once.)

		// Local declarations
		// {{{
		reg			w_valid_lock_request, w_cancel_lock,
					w_lock_request,
					lock_valid, returned_lock_valid;
		reg	[AW-LSB-1:0]	lock_start, lock_end;
		reg	[3:0]		lock_len;
		reg	[1:0]		lock_burst;
		reg	[2:0]		lock_size;
		reg	[IW-1:0]	lock_id;
		reg			w_write_lock_valid;
		// }}}

		// w_lock_request
		// {{{
		always @(*)
		begin
			w_lock_request = 0;
			if (S_AXI_ARVALID && S_AXI_ARREADY && S_AXI_ARLOCK)
				w_lock_request = 1;
		end
		// }}}

		// w_valid_lock_request
		// {{{
		always @(*)
		begin
			w_valid_lock_request = 0;
			if (w_lock_request)
				w_valid_lock_request = 1;
			if (o_we && o_waddr == S_AXI_ARADDR[AW-1:LSB])
				w_valid_lock_request = 0;
		end
		// }}}

		// returned_lock_valid
		// {{{
		initial	returned_lock_valid = 0;
		always @(posedge S_AXI_ACLK)
		if (!S_AXI_ARESETN)
			returned_lock_valid <= 0;
		else if (S_AXI_ARVALID && S_AXI_ARREADY
					&& S_AXI_ARLOCK && S_AXI_ARID== lock_id)
			returned_lock_valid <= 0;
		else if (w_cancel_lock)
			returned_lock_valid <= 0;
		else if (rskd_valid && rskd_lock && rskd_ready)
			returned_lock_valid <= lock_valid;
		// }}}

		// w_cancel_lock
		// {{{
		always @(*)
			w_cancel_lock = (lock_valid && w_lock_request)
				|| (lock_valid && o_we
					&& o_waddr >= lock_start
					&& o_waddr <= lock_end
					&& o_wstrb != 0);
		// }}}

		// lock_valid
		// {{{
		initial	lock_valid = 0;
		always @(posedge S_AXI_ACLK)
		if (!S_AXI_ARESETN || !OPT_LOCK)
			lock_valid <= 0;
		else begin
			if (S_AXI_ARVALID && S_AXI_ARREADY
					&& S_AXI_ARLOCK && S_AXI_ARID== lock_id)
				lock_valid <= 0;
			if (w_cancel_lock)
				lock_valid <= 0;
			if (w_valid_lock_request)
				lock_valid <= 1;
		end
		// }}}

		// lock_start, lock_end, lock_len, lock_size, lock_id
		// {{{
		always @(posedge S_AXI_ACLK)
		if (w_valid_lock_request)
		begin
			lock_start <= S_AXI_ARADDR[C_S_AXI_ADDR_WIDTH-1:LSB];
			lock_end <= S_AXI_ARADDR[C_S_AXI_ADDR_WIDTH-1:LSB]
					+ ((S_AXI_ARBURST == 2'b00) ? 0 : S_AXI_ARLEN[3:0]);
			lock_len   <= S_AXI_ARLEN[3:0];
			lock_burst <= S_AXI_ARBURST;
			lock_size  <= S_AXI_ARSIZE;
			lock_id    <= S_AXI_ARID;
		end
		// }}}

		// w_write_lock_valid
		// {{{
		always @(*)
		begin
			w_write_lock_valid = returned_lock_valid;
			if (!m_awvalid || !m_awready || !m_awlock || !lock_valid)
				w_write_lock_valid = 0;
			if (m_awaddr[C_S_AXI_ADDR_WIDTH-1:LSB] != lock_start)
				w_write_lock_valid = 0;
			if (m_awid != lock_id)
				w_write_lock_valid = 0;
			if (m_awlen[3:0] != lock_len)	// MAX transfer size is 16 beats
				w_write_lock_valid = 0;
			if (m_awburst != 2'b01 && lock_len != 0)
				w_write_lock_valid = 0;
			if (m_awsize != lock_size)
				w_write_lock_valid = 0;
		end
		// }}}

		assign	write_lock_valid = w_write_lock_valid;
		// }}}
	end else if (OPT_LOCK) // && OPT_LOCKID
	begin : EXCLUSIVE_ACCESS_PER_ID
		// {{{

		genvar	gk;
		wire	[(1<<IW)-1:0]	write_lock_valid_per_id;

		for(gk=0; gk<(1<<IW); gk=gk+1)
		begin : PER_ID_LOGIC
		// {{{
			// Local declarations
			// {{{
			reg			w_valid_lock_request,
						w_cancel_lock,
						lock_valid, returned_lock_valid;
			reg	[1:0]		lock_burst;
			reg	[2:0]		lock_size;
			reg	[3:0]		lock_len;
			reg	[AW-LSB-1:0]	lock_start, lock_end;
			reg			w_write_lock_valid;
			// }}}

			// valid_lock_request
			// {{{
			always @(*)
			begin
				w_valid_lock_request = 0;
				if (S_AXI_ARVALID && S_AXI_ARREADY
						&& S_AXI_ARID == gk[IW-1:0]
						&& S_AXI_ARLOCK)
					w_valid_lock_request = 1;
				if (o_we && o_waddr == S_AXI_ARADDR[AW-1:LSB])
					w_valid_lock_request = 0;
			end
			// }}}

			// returned_lock_valid
			// {{{
			initial	returned_lock_valid = 0;
			always @(posedge S_AXI_ACLK)
			if (!S_AXI_ARESETN)
				returned_lock_valid <= 0;
			else if (S_AXI_ARVALID && S_AXI_ARREADY
					&&S_AXI_ARLOCK&&S_AXI_ARID== gk[IW-1:0])
				returned_lock_valid <= 0;
			else if (w_cancel_lock)
				returned_lock_valid <= 0;
			else if (rskd_valid && rskd_lock && rskd_ready
					&& rskd_id == gk[IW-1:0])
				returned_lock_valid <= lock_valid;
			// }}}

			// w_cancel_lock
			// {{{
			always @(*)
				w_cancel_lock=(lock_valid&&w_valid_lock_request)
					|| (lock_valid && o_we
						&& o_waddr >= lock_start
						&& o_waddr <= lock_end
						&& o_wstrb != 0);
			// }}}

			// lock_valid
			// {{{
			initial	lock_valid = 0;
			always @(posedge S_AXI_ACLK)
			if (!S_AXI_ARESETN || !OPT_LOCK)
				lock_valid <= 0;
			else begin
				if (S_AXI_ARVALID && S_AXI_ARREADY
						&& S_AXI_ARLOCK
						&& S_AXI_ARID == gk[IW-1:0])
					lock_valid <= 0;
				if (w_cancel_lock)
					lock_valid <= 0;
				if (w_valid_lock_request)
					lock_valid <= 1;
			end
			// }}}

			// lock_start, lock_end, lock_len, lock_size
			// {{{
			always @(posedge S_AXI_ACLK)
			if (w_valid_lock_request)
			begin
				lock_start <= S_AXI_ARADDR[C_S_AXI_ADDR_WIDTH-1:LSB];
				// Verilator lint_off WIDTH
				lock_end <= S_AXI_ARADDR[C_S_AXI_ADDR_WIDTH-1:LSB]
					+ ((S_AXI_ARBURST == 2'b00) ? 4'h0 : S_AXI_ARLEN[3:0]);
				// Verilator lint_on  WIDTH
				lock_len   <= S_AXI_ARLEN[3:0];
				lock_size  <= S_AXI_ARSIZE;
				lock_burst <= S_AXI_ARBURST;
			end
			// }}}

			// w_write_lock_valid
			// {{{
			always @(*)
			begin
				w_write_lock_valid = returned_lock_valid;
				if (!m_awvalid || !m_awready || !m_awlock || !lock_valid)
					w_write_lock_valid = 0;
				if (m_awaddr[C_S_AXI_ADDR_WIDTH-1:LSB] != lock_start)
					w_write_lock_valid = 0;
				if (m_awid[IW-1:0] != gk[IW-1:0])
					w_write_lock_valid = 0;
				if (m_awlen[3:0] != lock_len)	// MAX transfer size is 16 beats
					w_write_lock_valid = 0;
				if (m_awburst != 2'b01 && lock_len != 0)
					w_write_lock_valid = 0;
				if (m_awsize != lock_size)
					w_write_lock_valid = 0;
			end
			// }}}

			assign	write_lock_valid_per_id[gk]= w_write_lock_valid;
		// }}}
		end

		assign	write_lock_valid = |write_lock_valid_per_id;
		// }}}
	end else begin : NO_LOCKING
		// {{{

		assign	write_lock_valid = 1'b0;
		// Verilator lint_off UNUSED
		wire	unused_lock;
		assign	unused_lock = &{ 1'b0, S_AXI_ARLOCK, S_AXI_AWLOCK };
		// Verilator lint_on  UNUSED
		// }}}
	end endgenerate
	// }}}

	// Make Verilator happy
	// {{{
	// Verilator lint_off UNUSED
	wire	unused;
	assign	unused = &{ 1'b0, S_AXI_AWCACHE, S_AXI_AWPROT, S_AXI_AWQOS,
		S_AXI_ARCACHE, S_AXI_ARPROT, S_AXI_ARQOS };
	// Verilator lint_on  UNUSED
	// }}}

    // ==========================================
    // HLS AXIS FIFO INTERFACE + CONTROL FSM
    // ==========================================
    localparam WORDS_PER_IN_BEAT  = C_S_AXI_DATA_WIDTH / HLS_IN_DATA_W;
    localparam WORDS_PER_OUT_BEAT = C_S_AXI_DATA_WIDTH / HLS_OUT_DATA_W;

    // Per-inference beat counts and the valid-word count of the LAST beat.
    // hls4ml io_stream networks often have N_IN/N_OUT that don't align to the
    // AXI beat width (e.g. N_IN=10 with WORDS_PER_IN_BEAT=8 → 2 beats, last
    // beat carries only 2 valid words).  We track the beat index within the
    // current inference to drop padding words on write and flush partial
    // beats on read.
    localparam N_BEATS_IN         = (HLS_IN_N_WORDS  + WORDS_PER_IN_BEAT  - 1) / WORDS_PER_IN_BEAT;
    localparam N_BEATS_OUT        = (HLS_OUT_N_WORDS + WORDS_PER_OUT_BEAT - 1) / WORDS_PER_OUT_BEAT;
    localparam LAST_BEAT_IN_VALID  = HLS_IN_N_WORDS  - (N_BEATS_IN  - 1) * WORDS_PER_IN_BEAT;
    localparam LAST_BEAT_OUT_VALID = HLS_OUT_N_WORDS - (N_BEATS_OUT - 1) * WORDS_PER_OUT_BEAT;

    // FSM
    localparam ST_IDLE      = 2'd0;
    localparam ST_STREAMING = 2'd1;
    localparam ST_DONE      = 2'd2;

    reg [1:0] state;

    // --- Inference cycle counter ---
    reg  [31:0]   cycle_counter;
    reg  [31:0]   inference_cycles;

    // Note on OPT_ASYNC_READ=0 below:
    //   NxMap auto-maps small sfifo memories (16x16 bits) into NX_RFB
    //   register-file primitives, which have a SYNCHRONOUS read port.
    //   The async-read variant of sfifo assumes combinational memory
    //   output, so on silicon it reads mem[rd_addr-1] one cycle late —
    //   the entire input stream is silently corrupted.  The sync-read
    //   path includes a bypass register that matches NX_RFB behaviour
    //   and preserves correctness at the cost of one cycle of read
    //   latency.

    // --- Input FIFO (AXI W → HLS AXIS in) ---
    wire                      in_fifo_i_wr;
    wire [HLS_IN_DATA_W-1:0]  in_fifo_i_data;
    wire                      in_fifo_o_empty;
    wire                      in_fifo_o_full;
    wire [HLS_IN_DATA_W-1:0]  in_fifo_o_data;
    wire                      in_fifo_i_rd;

    sfifo #(.BW(HLS_IN_DATA_W), .LGFLEN(LGFLEN_IN), .OPT_ASYNC_READ(1'b0))
    u_in_fifo (
        .i_clk(S_AXI_ACLK), .i_reset(!S_AXI_ARESETN),
        .i_wr(in_fifo_i_wr), .i_data(in_fifo_i_data),
        .o_full(in_fifo_o_full), .o_fill(),
        .i_rd(in_fifo_i_rd), .o_data(in_fifo_o_data),
        .o_empty(in_fifo_o_empty)
    );

    // --- Output FIFO (HLS AXIS out → AXI R) ---
    wire                      out_fifo_i_wr;
    wire [HLS_OUT_DATA_W-1:0] out_fifo_i_data;
    wire                      out_fifo_o_empty;
    wire                      out_fifo_o_full;
    wire [HLS_OUT_DATA_W-1:0] out_fifo_o_data;
    wire                      out_fifo_i_rd;

    sfifo #(.BW(HLS_OUT_DATA_W), .LGFLEN(LGFLEN_OUT), .OPT_ASYNC_READ(1'b0))
    u_out_fifo (
        .i_clk(S_AXI_ACLK), .i_reset(!S_AXI_ARESETN),
        .i_wr(out_fifo_i_wr), .i_data(out_fifo_i_data),
        .o_full(out_fifo_o_full), .o_fill(),
        .i_rd(out_fifo_i_rd), .o_data(out_fifo_o_data),
        .o_empty(out_fifo_o_empty)
    );

    // --- Write path: unpack one AXI beat into WORDS_PER_IN_BEAT FIFO entries ---
    //
    // A dedicated pipeline register (wr_stage) sits between the skidbuffer
    // output (o_wdata, registered) and the unpacker's wr_hold latch.  The
    // direct o_wdata → wr_hold path is short enough on NG-Ultra that clock
    // skew between distant flops exceeds data-path delay, producing marginal
    // hold-time violations (~57 ps) that NxMap's STA cannot absorb.  Running
    // the data through an explicit register stage turns the critical path
    // into flop-to-flop-to-flop, which always meets hold by construction.
    // The cost is one extra cycle of AXI-write-to-FIFO latency.
    reg                                    wr_unpacking;
    reg                                    wr_stage_valid;
    reg [C_S_AXI_DATA_WIDTH-1:0]           wr_stage;
    reg [$clog2(WORDS_PER_IN_BEAT)-1:0]   wr_word_idx;
    reg [C_S_AXI_DATA_WIDTH-1:0]           wr_hold;

    // Which beat of the current inference we are unpacking.  Wraps back to 0
    // once the last beat of the inference is fully consumed so the next
    // AXI-queued inference starts fresh.
    reg [$clog2(N_BEATS_IN > 1 ? N_BEATS_IN : 2)-1:0] wr_beat_idx;
    wire wr_is_last_beat = (wr_beat_idx == N_BEATS_IN - 1);
    // How many words in THIS beat actually carry payload (the rest are padding
    // zeros the testbench/DMA wrote to fill out the AXI beat).
    wire [$clog2(WORDS_PER_IN_BEAT+1)-1:0] wr_valid_words_this_beat =
        wr_is_last_beat ? LAST_BEAT_IN_VALID[$clog2(WORDS_PER_IN_BEAT+1)-1:0]
                        : WORDS_PER_IN_BEAT[$clog2(WORDS_PER_IN_BEAT+1)-1:0];
    wire wr_word_valid = ({1'b0, wr_word_idx} < wr_valid_words_this_beat);

    // Stage 1: absorb an AXI beat into wr_stage the cycle o_we fires.
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            wr_stage_valid <= 1'b0;
            wr_stage       <= 0;
        end else if (o_we) begin
            wr_stage_valid <= 1'b1;
            wr_stage       <= o_wdata;
        end else if (!wr_unpacking) begin
            // Consumed by stage 2 below; clear only when stage 2 has latched.
            wr_stage_valid <= 1'b0;
        end
    end

    // Stage 2: move wr_stage into wr_hold when the unpacker is idle.
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            wr_unpacking <= 1'b0;
            wr_word_idx  <= 0;
            wr_hold      <= 0;
            wr_beat_idx  <= 0;
        end else if (wr_stage_valid && !wr_unpacking) begin
            wr_hold      <= wr_stage;
            wr_unpacking <= 1'b1;
            wr_word_idx  <= 0;
        end else if (wr_unpacking && !in_fifo_o_full) begin
            if (wr_word_idx == WORDS_PER_IN_BEAT - 1) begin
                wr_unpacking <= 1'b0;
                wr_word_idx  <= 0;
                wr_beat_idx  <= wr_is_last_beat ? {$clog2(N_BEATS_IN > 1 ? N_BEATS_IN : 2){1'b0}}
                                                : (wr_beat_idx + 1'b1);
            end else begin
                wr_word_idx <= wr_word_idx + 1'b1;
            end
        end
    end

    // Only push words that carry real payload.  The last beat of an
    // inference with misaligned N_IN silently drops its padding slots.
    assign in_fifo_i_wr   = wr_unpacking && !in_fifo_o_full && wr_word_valid;
    assign in_fifo_i_data = wr_hold[wr_word_idx * HLS_IN_DATA_W +: HLS_IN_DATA_W];
    // Back-pressure when stage 1 already holds an unconsumed beat, when the
    // narrow FIFO is full, or when a beat is being accepted this cycle.
    // Note: wr_unpacking alone no longer back-pressures — stage 1 (wr_stage)
    // absorbs the next AXI beat concurrently with stage 2's word-by-word push.
    assign wr_full        = wr_stage_valid || in_fifo_o_full || o_we;

    // --- AXIS input wiring (input FIFO → HLS IP) ---
    assign hls_in_tdata  = in_fifo_o_data;
    assign hls_in_tvalid = !in_fifo_o_empty;
    assign in_fifo_i_rd  = hls_in_tready && !in_fifo_o_empty;

    // --- AXIS output wiring (HLS IP → output FIFO) ---
    // NOTE: the per-inference capture counter (out_words_captured) was tried
    // here but the corresponding bitstream reproduced the NG-Ultra config
    // fault.  Reverted for now; if spurious pre/post IP outputs turn out to
    // pollute the FIFO boundary, we'll re-introduce it with a different
    // synthesis seed or rewrite it to share state with the FSM.
    assign out_fifo_i_wr   = hls_out_tvalid && !out_fifo_o_full;
    assign out_fifo_i_data = hls_out_tdata;
    assign hls_out_tready  = !out_fifo_o_full;

    // --- Read path: pack FIFO entries into AXI beats ---
    // For the LAST output beat of an inference (when N_OUT isn't a multiple
    // of WORDS_PER_OUT_BEAT) we flush with fewer than WORDS_PER_OUT_BEAT
    // words so the reader never hangs waiting on padding that will never come.
    reg [$clog2(WORDS_PER_OUT_BEAT)-1:0]  rd_pack_idx;
    reg                                    rd_beat_ready;
    reg [C_S_AXI_DATA_WIDTH-1:0]           rd_data_assembled;

    // Which output beat of the current inference we are assembling.
    reg [$clog2(N_BEATS_OUT > 1 ? N_BEATS_OUT : 2)-1:0] rd_beat_idx;
    wire rd_is_last_beat = (rd_beat_idx == N_BEATS_OUT - 1);
    wire [$clog2(WORDS_PER_OUT_BEAT+1)-1:0] rd_pack_target =
        rd_is_last_beat ? LAST_BEAT_OUT_VALID[$clog2(WORDS_PER_OUT_BEAT+1)-1:0]
                        : WORDS_PER_OUT_BEAT[$clog2(WORDS_PER_OUT_BEAT+1)-1:0];

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            rd_pack_idx       <= 0;
            rd_beat_ready     <= 1'b0;
            rd_data_assembled <= 0;
            rd_beat_idx       <= 0;
        end else if (rd_beat_ready && o_rd) begin
            rd_beat_ready     <= 1'b0;
            rd_pack_idx       <= 0;
            // Clear so padding slots in the next (possibly short) beat read
            // back as zero rather than leaking bits from the prior beat.
            rd_data_assembled <= 0;
            rd_beat_idx       <= rd_is_last_beat
                ? {$clog2(N_BEATS_OUT > 1 ? N_BEATS_OUT : 2){1'b0}}
                : (rd_beat_idx + 1'b1);
        end else if (!rd_beat_ready && !out_fifo_o_empty) begin
            rd_data_assembled[rd_pack_idx * HLS_OUT_DATA_W +: HLS_OUT_DATA_W] <= out_fifo_o_data;
            // Flush when we've filled the target for this beat — either a
            // full beat or the short last-beat of an inference.
            if (({1'b0, rd_pack_idx} + 1'b1) == rd_pack_target) begin
                rd_beat_ready <= 1'b1;
                rd_pack_idx   <= 0;
            end else begin
                rd_pack_idx <= rd_pack_idx + 1'b1;
            end
        end
    end

    assign out_fifo_i_rd = !rd_beat_ready && !out_fifo_o_empty;

    // --- rd_phase: RD_DATA serves output beats; RD_CYCLES serves cycle counter ---
    localparam RD_DATA   = 1'b0;
    localparam RD_CYCLES = 1'b1;
    reg rd_phase;
    reg cycles_served;

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN || state == ST_IDLE) begin
            rd_phase      <= RD_DATA;
            cycles_served <= 1'b0;
        end else begin
            if (rd_phase == RD_DATA && state == ST_DONE
                    && out_fifo_o_empty && !rd_beat_ready)
                rd_phase <= RD_CYCLES;
            else if (rd_phase == RD_CYCLES && o_rd) begin
                rd_phase      <= RD_DATA;
                cycles_served <= 1'b1;
            end
        end
    end

    assign rd_empty = (rd_phase == RD_DATA) ? !rd_beat_ready : 1'b0;

    // --- FSM ---
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            state     <= ST_IDLE;
            start_out <= 1'b0;
        end else begin
            start_out <= 1'b0;
            case (state)
                ST_IDLE: begin
                    // Start when either an AXI write is in flight or the
                    // input FIFO already holds data from a previously
                    // back-pressured inference.
                    if (o_we || !in_fifo_o_empty) begin
                        start_out <= 1'b1;
                        state     <= ST_STREAMING;
                    end
                end
                ST_STREAMING: begin
                    if (done_in) state <= ST_DONE;
                end
                ST_DONE: begin
                    if (cycles_served) state <= ST_IDLE;
                end
            endcase
        end
    end

    // --- Cycle counter: starts on start_out, captured on done_in ---
    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN) begin
            cycle_counter    <= 32'd0;
            inference_cycles <= 32'd0;
        end else if (start_out) begin
            cycle_counter <= 32'd0;
        end else if (state == ST_STREAMING) begin
            cycle_counter <= cycle_counter + 1'b1;
            if (done_in)
                inference_cycles <= cycle_counter;
        end
    end

    // --- i_rdata: registered, only changes on o_rd ---
    wire [C_S_AXI_DATA_WIDTH-1:0] cycles_beat =
        { {(C_S_AXI_DATA_WIDTH-32){1'b0}}, inference_cycles };

    always @(posedge S_AXI_ACLK) begin
        if (!S_AXI_ARESETN)
            i_rdata <= {C_S_AXI_DATA_WIDTH{1'b0}};
        else if (o_rd) begin
            if (rd_phase == RD_CYCLES)
                i_rdata <= cycles_beat;
            else
                i_rdata <= rd_data_assembled;
        end
    end

    assign o_interrupt = (state == ST_DONE);

endmodule
`ifndef	YOSYS
`default_nettype wire
`endif
