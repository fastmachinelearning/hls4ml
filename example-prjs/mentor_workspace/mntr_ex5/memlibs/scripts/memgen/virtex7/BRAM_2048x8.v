`timescale 1 ps / 1 ps
// Copyright (c) 2014-2015, Columbia University
module BRAM_2048x8( CLK, A0, D0, Q0, WE0, WEM0, CE0, A1, D1, Q1, WE1, WEM1, CE1 );
	input CLK;
	input [10:0] A0;
	input [7:0] D0;
	output [7:0] Q0;
	input WE0;
	input [7:0] WEM0;
	input CE0;
	input [10:0] A1;
	input [7:0] D1;
	output [7:0] Q1;
	input WE1;
	input [7:0] WEM1;
	input CE1;

   reg 	      CE0_tmp;
   reg 	      CE1_tmp;
   reg [10:0]  A0_tmp;
   reg [10:0]  A1_tmp;
   reg 	      WE0_tmp;
   reg 	      WE1_tmp;
   reg [7:0] D0_tmp;
   reg [7:0] D1_tmp;

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
   
  wire DOPA_float;
  wire DOPB_float;

	RAMB16_S9_S9 bram (
		.DOA(Q0),
		.DOB(Q1),
		.DOPA(DOPA_float),
		.DOPB(DOPB_float),
		.ADDRA(A0_tmp),
		.ADDRB(A1_tmp),
		.CLKA(CLK),
		.CLKB(CLK),
		.DIA(D0_tmp),
		.DIB(D1_tmp),
		.DIPA(1'b0),
		.DIPB(1'b0),
		.ENA(CE0_tmp),
		.ENB(CE1_tmp),
		.SSRA(1'b0),
		.SSRB(1'b0),
		.WEA(WE0_tmp),
		.WEB(WE1_tmp));
endmodule
