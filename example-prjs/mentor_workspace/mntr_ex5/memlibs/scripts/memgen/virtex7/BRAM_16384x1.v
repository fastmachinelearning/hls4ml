`timescale 1 ps / 1 ps
// Copyright (c) 2014-2015, Columbia University
module BRAM_16384x1( CLK, A0, D0, Q0, WE0, WEM0, CE0, A1, D1, Q1, WE1, WEM1, CE1 );
	input CLK;
	input [13:0] A0;
	input D0;
	output Q0;
	input WE0;
	input WEM0;
	input CE0;
	input [13:0] A1;
	input D1;
	output Q1;
	input WE1;
	input WEM1;
	input CE1;

   reg 	      CE0_tmp;
   reg 	      CE1_tmp;
   reg [13:0]  A0_tmp;
   reg [13:0]  A1_tmp;
   reg 	      WE0_tmp;
   reg 	      WE1_tmp;
   reg        D0_tmp;
   reg        D1_tmp;

   always @(*)
     begin
	#5 A0_tmp = A0;
	A1_tmp = A1;
	CE0_tmp = CE0;
	CE1_tmp = CE1;
	WE0_tmp = WE0;
	WE1_tmp = WE1;
	D0_tmp = D0;
	D1_tmp = D1;
     end

	RAMB16_S1_S1 bram (
		.DOA(Q0),
		.DOB(Q1),
		.ADDRA(A0_tmp),
		.ADDRB(A1_tmp),
		.CLKA(CLK),
		.CLKB(CLK),
		.DIA(D0_tmp),
		.DIB(D1_tmp),
		.ENA(CE0_tmp),
		.ENB(CE1_tmp),
		.SSRA(1'b0),
		.SSRB(1'b0),
		.WEA(WE0_tmp),
		.WEB(WE1_tmp));
endmodule
