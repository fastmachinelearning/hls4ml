
//------> /opt/cad/catapult/pkgs/siflibs/ccs_in_wait_v1.v 
//------------------------------------------------------------------------------
// Catapult Synthesis - Sample I/O Port Library
//
// Copyright (c) 2003-2017 Mentor Graphics Corp.
//       All Rights Reserved
//
// This document may be used and distributed without restriction provided that
// this copyright statement is not removed from the file and that any derivative
// work contains this copyright notice.
//
// The design information contained in this file is intended to be an example
// of the functionality which the end user may study in preparation for creating
// their own custom interfaces. This design does not necessarily present a 
// complete implementation of the named protocol or standard.
//
//------------------------------------------------------------------------------


module ccs_in_wait_v1 (idat, rdy, ivld, dat, irdy, vld);

  parameter integer rscid = 1;
  parameter integer width = 8;

  output [width-1:0] idat;
  output             rdy;
  output             ivld;
  input  [width-1:0] dat;
  input              irdy;
  input              vld;

  wire   [width-1:0] idat;
  wire               rdy;
  wire               ivld;

  assign idat = dat;
  assign rdy = irdy;
  assign ivld = vld;

endmodule


//------> /opt/cad/catapult/pkgs/siflibs/ccs_out_wait_v1.v 
//------------------------------------------------------------------------------
// Catapult Synthesis - Sample I/O Port Library
//
// Copyright (c) 2003-2017 Mentor Graphics Corp.
//       All Rights Reserved
//
// This document may be used and distributed without restriction provided that
// this copyright statement is not removed from the file and that any derivative
// work contains this copyright notice.
//
// The design information contained in this file is intended to be an example
// of the functionality which the end user may study in preparation for creating
// their own custom interfaces. This design does not necessarily present a 
// complete implementation of the named protocol or standard.
//
//------------------------------------------------------------------------------


module ccs_out_wait_v1 (dat, irdy, vld, idat, rdy, ivld);

  parameter integer rscid = 1;
  parameter integer width = 8;

  output [width-1:0] dat;
  output             irdy;
  output             vld;
  input  [width-1:0] idat;
  input              rdy;
  input              ivld;

  wire   [width-1:0] dat;
  wire               irdy;
  wire               vld;

  assign dat = idat;
  assign irdy = rdy;
  assign vld = ivld;

endmodule



//------> /opt/cad/catapult/pkgs/siflibs/ccs_out_vld_v1.v 
//------------------------------------------------------------------------------
// Catapult Synthesis - Sample I/O Port Library
//
// Copyright (c) 2003-2017 Mentor Graphics Corp.
//       All Rights Reserved
//
// This document may be used and distributed without restriction provided that
// this copyright statement is not removed from the file and that any derivative
// work contains this copyright notice.
//
// The design information contained in this file is intended to be an example
// of the functionality which the end user may study in preparation for creating
// their own custom interfaces. This design does not necessarily present a 
// complete implementation of the named protocol or standard.
//
//------------------------------------------------------------------------------


module ccs_out_vld_v1 (dat, vld, idat, ivld);

  parameter integer rscid = 1;
  parameter integer width = 8;

  output [width-1:0] dat;
  output             vld;
  input  [width-1:0] idat;
  input              ivld;

  wire   [width-1:0] dat;
  wire               vld;

  assign dat = idat;
  assign vld = ivld;

endmodule


//------> /opt/cad/catapult/pkgs/siflibs/ccs_in_vld_v1.v 
//------------------------------------------------------------------------------
// Catapult Synthesis - Sample I/O Port Library
//
// Copyright (c) 2003-2017 Mentor Graphics Corp.
//       All Rights Reserved
//
// This doocument may be used and distributed without restriction provided that
// this copyright statement is not removed from the file and that any derivative
// work contains this copyright notice.
//
// The design information contained in this file is intended to be an example
// of the functionality which the end user may study in preparation for creating
// their own custom interfaces. This design does not necessarily present a 
// complete implementation of the named protocol or standard.
//
//------------------------------------------------------------------------------


module ccs_in_vld_v1 (idat, ivld, dat, vld);

  parameter integer rscid = 1;
  parameter integer width = 8;

  output [width-1:0] idat;
  output             ivld;
  input  [width-1:0] dat;
  input              vld;

  wire   [width-1:0] idat;
  wire               ivld;

  assign idat = dat;
  assign ivld = vld;

endmodule


//------> /opt/cad/catapult/pkgs/siflibs/ccs_in_v1.v 
//------------------------------------------------------------------------------
// Catapult Synthesis - Sample I/O Port Library
//
// Copyright (c) 2003-2017 Mentor Graphics Corp.
//       All Rights Reserved
//
// This document may be used and distributed without restriction provided that
// this copyright statement is not removed from the file and that any derivative
// work contains this copyright notice.
//
// The design information contained in this file is intended to be an example
// of the functionality which the end user may study in preparation for creating
// their own custom interfaces. This design does not necessarily present a 
// complete implementation of the named protocol or standard.
//
//------------------------------------------------------------------------------


module ccs_in_v1 (idat, dat);

  parameter integer rscid = 1;
  parameter integer width = 8;

  output [width-1:0] idat;
  input  [width-1:0] dat;

  wire   [width-1:0] idat;

  assign idat = dat;

endmodule


//------> ../td_ccore_solutions/ROM_1i3_1o10_d515ec42b6831339071874e16a9b8d3ab1_0/rtl.v 
// ----------------------------------------------------------------------
//  HLS HDL:        Verilog Netlister
//  HLS Version:    10.4b/841621 Production Release
//  HLS Date:       Thu Oct 24 17:20:07 PDT 2019
// 
//  Generated by:   giuseppe@fastml02
//  Generated date: Mon Dec  9 15:55:40 2019
// ----------------------------------------------------------------------

// 
// ------------------------------------------------------------------
//  Design Unit:    ROM_1i3_1o10_d515ec42b6831339071874e16a9b8d3ab1
// ------------------------------------------------------------------


module ROM_1i3_1o10_d515ec42b6831339071874e16a9b8d3ab1 (
  I_1, O_1
);
  input [2:0] I_1;
  output [9:0] O_1;



  // Interconnect Declarations for Component Instantiations 
  assign O_1 = MUX_v_10_8_2(10'b1111111101, 10'b1100011001, 10'b1001100100, 10'b0111010000,
      10'b0101010100, 10'b0011101011, 10'b0010010001, 10'b0001000100, I_1);

  function automatic [9:0] MUX_v_10_8_2;
    input [9:0] input_0;
    input [9:0] input_1;
    input [9:0] input_2;
    input [9:0] input_3;
    input [9:0] input_4;
    input [9:0] input_5;
    input [9:0] input_6;
    input [9:0] input_7;
    input [2:0] sel;
    reg [9:0] result;
  begin
    case (sel)
      3'b000 : begin
        result = input_0;
      end
      3'b001 : begin
        result = input_1;
      end
      3'b010 : begin
        result = input_2;
      end
      3'b011 : begin
        result = input_3;
      end
      3'b100 : begin
        result = input_4;
      end
      3'b101 : begin
        result = input_5;
      end
      3'b110 : begin
        result = input_6;
      end
      default : begin
        result = input_7;
      end
    endcase
    MUX_v_10_8_2 = result;
  end
  endfunction

endmodule




//------> /opt/cad/catapult/pkgs/siflibs/mgc_shift_br_beh_v5.v 
module mgc_shift_br_v5(a,s,z);
   parameter    width_a = 4;
   parameter    signd_a = 1;
   parameter    width_s = 2;
   parameter    width_z = 8;

   input [width_a-1:0] a;
   input [width_s-1:0] s;
   output [width_z -1:0] z;

   generate
     if (signd_a)
     begin: SGNED
       assign z = fshr_s(a,s,a[width_a-1]);
     end
     else
     begin: UNSGNED
       assign z = fshr_s(a,s,1'b0);
     end
   endgenerate

   //Shift-left - unsigned shift argument one bit more
   function [width_z-1:0] fshl_u_1;
      input [width_a  :0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      parameter olen = width_z;
      parameter ilen = width_a+1;
      parameter len = (ilen >= olen) ? ilen : olen;
      reg [len-1:0] result;
      reg [len-1:0] result_t;
      begin
        result_t = {(len){sbit}};
        result_t[ilen-1:0] = arg1;
        result = result_t <<< arg2;
        fshl_u_1 =  result[olen-1:0];
      end
   endfunction // fshl_u

   //Shift right - unsigned shift argument
   function [width_z-1:0] fshr_u;
      input [width_a-1:0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      parameter olen = width_z;
      parameter ilen = signd_a ? width_a : width_a+1;
      parameter len = (ilen >= olen) ? ilen : olen;
      reg signed [len-1:0] result;
      reg signed [len-1:0] result_t;
      begin
        result_t = $signed( {(len){sbit}} );
        result_t[width_a-1:0] = arg1;
        result = result_t >>> arg2;
        fshr_u =  result[olen-1:0];
      end
   endfunction // fshr_u

   //Shift right - signed shift argument
   function [width_z-1:0] fshr_s;
     input [width_a-1:0] arg1;
     input [width_s-1:0] arg2;
     input sbit;
     begin
       if ( arg2[width_s-1] == 1'b0 )
       begin
         fshr_s = fshr_u(arg1, arg2, sbit);
       end
       else
       begin
         fshr_s = fshl_u_1({arg1, 1'b0},~arg2, sbit);
       end
     end
   endfunction 

endmodule

//------> /opt/cad/catapult/pkgs/siflibs/mgc_shift_bl_beh_v5.v 
module mgc_shift_bl_v5(a,s,z);
   parameter    width_a = 4;
   parameter    signd_a = 1;
   parameter    width_s = 2;
   parameter    width_z = 8;

   input [width_a-1:0] a;
   input [width_s-1:0] s;
   output [width_z -1:0] z;

   generate if ( signd_a )
   begin: SGNED
     assign z = fshl_s(a,s,a[width_a-1]);
   end
   else
   begin: UNSGNED
     assign z = fshl_s(a,s,1'b0);
   end
   endgenerate

   //Shift-left - unsigned shift argument one bit more
   function [width_z-1:0] fshl_u_1;
      input [width_a  :0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      parameter olen = width_z;
      parameter ilen = width_a+1;
      parameter len = (ilen >= olen) ? ilen : olen;
      reg [len-1:0] result;
      reg [len-1:0] result_t;
      begin
        result_t = {(len){sbit}};
        result_t[ilen-1:0] = arg1;
        result = result_t <<< arg2;
        fshl_u_1 =  result[olen-1:0];
      end
   endfunction // fshl_u

   //Shift-left - unsigned shift argument
   function [width_z-1:0] fshl_u;
      input [width_a-1:0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      fshl_u = fshl_u_1({sbit,arg1} ,arg2, sbit);
   endfunction // fshl_u

   //Shift right - unsigned shift argument
   function [width_z-1:0] fshr_u;
      input [width_a-1:0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      parameter olen = width_z;
      parameter ilen = signd_a ? width_a : width_a+1;
      parameter len = (ilen >= olen) ? ilen : olen;
      reg signed [len-1:0] result;
      reg signed [len-1:0] result_t;
      begin
        result_t = $signed( {(len){sbit}} );
        result_t[width_a-1:0] = arg1;
        result = result_t >>> arg2;
        fshr_u =  result[olen-1:0];
      end
   endfunction // fshl_u

   //Shift left - signed shift argument
   function [width_z-1:0] fshl_s;
      input [width_a-1:0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      reg [width_a:0] sbit_arg1;
      begin
        // Ignoring the possibility that arg2[width_s-1] could be X
        // because of customer complaints regarding X'es in simulation results
        if ( arg2[width_s-1] == 1'b0 )
        begin
          sbit_arg1[width_a:0] = {(width_a+1){1'b0}};
          fshl_s = fshl_u(arg1, arg2, sbit);
        end
        else
        begin
          sbit_arg1[width_a] = sbit;
          sbit_arg1[width_a-1:0] = arg1;
          fshl_s = fshr_u(sbit_arg1[width_a:1], ~arg2, sbit);
        end
      end
   endfunction

endmodule

//------> ../td_ccore_solutions/ROM_1i3_1o8_75ee39ff4c2e67ce55133b7c869c3b33b0_0/rtl.v 
// ----------------------------------------------------------------------
//  HLS HDL:        Verilog Netlister
//  HLS Version:    10.4b/841621 Production Release
//  HLS Date:       Thu Oct 24 17:20:07 PDT 2019
// 
//  Generated by:   giuseppe@fastml02
//  Generated date: Mon Dec  9 15:55:37 2019
// ----------------------------------------------------------------------

// 
// ------------------------------------------------------------------
//  Design Unit:    ROM_1i3_1o8_75ee39ff4c2e67ce55133b7c869c3b33b0
// ------------------------------------------------------------------


module ROM_1i3_1o8_75ee39ff4c2e67ce55133b7c869c3b33b0 (
  I_1, O_1
);
  input [2:0] I_1;
  output [7:0] O_1;



  // Interconnect Declarations for Component Instantiations 
  assign O_1 = MUX_v_8_8_2(8'b00011100, 8'b01001011, 8'b01101100, 8'b10000100, 8'b10010111,
      8'b10100110, 8'b10110011, 8'b10111100, I_1);

  function automatic [7:0] MUX_v_8_8_2;
    input [7:0] input_0;
    input [7:0] input_1;
    input [7:0] input_2;
    input [7:0] input_3;
    input [7:0] input_4;
    input [7:0] input_5;
    input [7:0] input_6;
    input [7:0] input_7;
    input [2:0] sel;
    reg [7:0] result;
  begin
    case (sel)
      3'b000 : begin
        result = input_0;
      end
      3'b001 : begin
        result = input_1;
      end
      3'b010 : begin
        result = input_2;
      end
      3'b011 : begin
        result = input_3;
      end
      3'b100 : begin
        result = input_4;
      end
      3'b101 : begin
        result = input_5;
      end
      3'b110 : begin
        result = input_6;
      end
      default : begin
        result = input_7;
      end
    endcase
    MUX_v_8_8_2 = result;
  end
  endfunction

endmodule




//------> /opt/cad/catapult/pkgs/siflibs/mgc_shift_l_beh_v5.v 
module mgc_shift_l_v5(a,s,z);
   parameter    width_a = 4;
   parameter    signd_a = 1;
   parameter    width_s = 2;
   parameter    width_z = 8;

   input [width_a-1:0] a;
   input [width_s-1:0] s;
   output [width_z -1:0] z;

   generate
   if (signd_a)
   begin: SGNED
      assign z = fshl_u(a,s,a[width_a-1]);
   end
   else
   begin: UNSGNED
      assign z = fshl_u(a,s,1'b0);
   end
   endgenerate

   //Shift-left - unsigned shift argument one bit more
   function [width_z-1:0] fshl_u_1;
      input [width_a  :0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      parameter olen = width_z;
      parameter ilen = width_a+1;
      parameter len = (ilen >= olen) ? ilen : olen;
      reg [len-1:0] result;
      reg [len-1:0] result_t;
      begin
        result_t = {(len){sbit}};
        result_t[ilen-1:0] = arg1;
        result = result_t <<< arg2;
        fshl_u_1 =  result[olen-1:0];
      end
   endfunction // fshl_u

   //Shift-left - unsigned shift argument
   function [width_z-1:0] fshl_u;
      input [width_a-1:0] arg1;
      input [width_s-1:0] arg2;
      input sbit;
      fshl_u = fshl_u_1({sbit,arg1} ,arg2, sbit);
   endfunction // fshl_u

endmodule

//------> ../td_ccore_solutions/leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_0/rtl.v 
// ----------------------------------------------------------------------
//  HLS HDL:        Verilog Netlister
//  HLS Version:    10.4b/841621 Production Release
//  HLS Date:       Thu Oct 24 17:20:07 PDT 2019
// 
//  Generated by:   giuseppe@fastml02
//  Generated date: Thu Dec 12 11:44:40 2019
// ----------------------------------------------------------------------

// 
// ------------------------------------------------------------------
//  Design Unit:    leading_sign_71_0
// ------------------------------------------------------------------


module leading_sign_71_0 (
  mantissa, rtn
);
  input [70:0] mantissa;
  output [6:0] rtn;


  // Interconnect Declarations
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_18_3_sdt_3;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_42_4_sdt_4;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_62_3_sdt_3;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_90_5_sdt_5;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_110_3_sdt_3;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_134_4_sdt_4;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_154_3_sdt_3;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_186_6_sdt_6;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_2;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_14_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_34_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_58_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_78_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_106_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_126_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_150_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_170_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_1;
  wire ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_200_2_sdt_1;
  wire c_h_1_2;
  wire c_h_1_5;
  wire c_h_1_6;
  wire c_h_1_9;
  wire c_h_1_12;
  wire c_h_1_13;
  wire c_h_1_14;
  wire c_h_1_17;
  wire c_h_1_20;
  wire c_h_1_21;
  wire c_h_1_24;
  wire c_h_1_27;
  wire c_h_1_28;
  wire c_h_1_29;
  wire c_h_1_30;
  wire c_h_1_33;
  wire c_h_1_34;

  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_1_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_2_nl;
  wire[2:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_or_1_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_293_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_295_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_296_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_281_nl;

  // Interconnect Declarations for Component Instantiations 
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_2
      = ~((mantissa[68:67]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_1
      = ~((mantissa[70:69]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_14_2_sdt_1
      = ~((mantissa[66:65]!=2'b00));
  assign c_h_1_2 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_2;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_18_3_sdt_3
      = (mantissa[64:63]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_14_2_sdt_1;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_2
      = ~((mantissa[60:59]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_1
      = ~((mantissa[62:61]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_34_2_sdt_1
      = ~((mantissa[58:57]!=2'b00));
  assign c_h_1_5 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_2;
  assign c_h_1_6 = c_h_1_2 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_18_3_sdt_3;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_42_4_sdt_4
      = (mantissa[56:55]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_34_2_sdt_1
      & c_h_1_5;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_2
      = ~((mantissa[52:51]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_1
      = ~((mantissa[54:53]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_58_2_sdt_1
      = ~((mantissa[50:49]!=2'b00));
  assign c_h_1_9 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_2;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_62_3_sdt_3
      = (mantissa[48:47]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_58_2_sdt_1;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_2
      = ~((mantissa[44:43]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_1
      = ~((mantissa[46:45]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_78_2_sdt_1
      = ~((mantissa[42:41]!=2'b00));
  assign c_h_1_12 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_2;
  assign c_h_1_13 = c_h_1_9 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_62_3_sdt_3;
  assign c_h_1_14 = c_h_1_6 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_42_4_sdt_4;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_90_5_sdt_5
      = (mantissa[40:39]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_78_2_sdt_1
      & c_h_1_12 & c_h_1_13;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_2
      = ~((mantissa[36:35]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_1
      = ~((mantissa[38:37]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_106_2_sdt_1
      = ~((mantissa[34:33]!=2'b00));
  assign c_h_1_17 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_2;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_110_3_sdt_3
      = (mantissa[32:31]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_106_2_sdt_1;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_2
      = ~((mantissa[28:27]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_1
      = ~((mantissa[30:29]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_126_2_sdt_1
      = ~((mantissa[26:25]!=2'b00));
  assign c_h_1_20 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_2;
  assign c_h_1_21 = c_h_1_17 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_110_3_sdt_3;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_134_4_sdt_4
      = (mantissa[24:23]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_126_2_sdt_1
      & c_h_1_20;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_2
      = ~((mantissa[20:19]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_1
      = ~((mantissa[22:21]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_150_2_sdt_1
      = ~((mantissa[18:17]!=2'b00));
  assign c_h_1_24 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_2;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_154_3_sdt_3
      = (mantissa[16:15]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_150_2_sdt_1;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_2
      = ~((mantissa[12:11]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_1
      = ~((mantissa[14:13]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_170_2_sdt_1
      = ~((mantissa[10:9]!=2'b00));
  assign c_h_1_27 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_2;
  assign c_h_1_28 = c_h_1_24 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_154_3_sdt_3;
  assign c_h_1_29 = c_h_1_21 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_134_4_sdt_4;
  assign c_h_1_30 = c_h_1_14 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_90_5_sdt_5;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_186_6_sdt_6
      = (mantissa[8:7]==2'b00) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_170_2_sdt_1
      & c_h_1_27 & c_h_1_28 & c_h_1_29;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_2
      = ~((mantissa[4:3]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_1
      = ~((mantissa[6:5]!=2'b00));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_200_2_sdt_1
      = ~((mantissa[2:1]!=2'b00));
  assign c_h_1_33 = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_1
      & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_2;
  assign c_h_1_34 = c_h_1_30 & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_186_6_sdt_6;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_nl
      = c_h_1_30 & (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_186_6_sdt_6);
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_1_nl
      = c_h_1_14 & (c_h_1_29 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_90_5_sdt_5))
      & (~ c_h_1_34);
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_2_nl
      = c_h_1_6 & (c_h_1_13 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_42_4_sdt_4))
      & (~((~(c_h_1_21 & (c_h_1_28 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_134_4_sdt_4))))
      & c_h_1_30)) & (~ c_h_1_34);
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_293_nl
      = c_h_1_2 & (c_h_1_5 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_18_3_sdt_3))
      & (~((~(c_h_1_9 & (c_h_1_12 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_62_3_sdt_3))))
      & c_h_1_14)) & (~((~(c_h_1_17 & (c_h_1_20 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_110_3_sdt_3))
      & (~((~(c_h_1_24 & (c_h_1_27 | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_154_3_sdt_3))))
      & c_h_1_29)))) & c_h_1_30)) & (c_h_1_33 | (~ c_h_1_34));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_295_nl
      = ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_14_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_6_2_sdt_2))
      & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_34_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_26_2_sdt_2))))
      & c_h_1_6)) & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_58_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_50_2_sdt_2))
      & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_78_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_70_2_sdt_2))))
      & c_h_1_13)))) & c_h_1_14)) & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_106_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_98_2_sdt_2))
      & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_126_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_118_2_sdt_2))))
      & c_h_1_21)) & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_150_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_142_2_sdt_2))
      & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_170_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_162_2_sdt_2))))
      & c_h_1_28)))) & c_h_1_29)))) & c_h_1_30)) & (~((~(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_1
      & (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_200_2_sdt_1
      | (~ ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_194_2_sdt_2))))
      & c_h_1_34));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_296_nl
      = (~((mantissa[70]) | (~((mantissa[69:68]!=2'b01))))) & (~(((mantissa[66])
      | (~((mantissa[65:64]!=2'b01)))) & c_h_1_2)) & (~((~((~((mantissa[62]) | (~((mantissa[61:60]!=2'b01)))))
      & (~(((mantissa[58]) | (~((mantissa[57:56]!=2'b01)))) & c_h_1_5)))) & c_h_1_6))
      & (~((~((~((mantissa[54]) | (~((mantissa[53:52]!=2'b01))))) & (~(((mantissa[50])
      | (~((mantissa[49:48]!=2'b01)))) & c_h_1_9)) & (~((~((~((mantissa[46]) | (~((mantissa[45:44]!=2'b01)))))
      & (~(((mantissa[42]) | (~((mantissa[41:40]!=2'b01)))) & c_h_1_12)))) & c_h_1_13))))
      & c_h_1_14)) & (~((~((~((mantissa[38]) | (~((mantissa[37:36]!=2'b01))))) &
      (~(((mantissa[34]) | (~((mantissa[33:32]!=2'b01)))) & c_h_1_17)) & (~((~((~((mantissa[30])
      | (~((mantissa[29:28]!=2'b01))))) & (~(((mantissa[26]) | (~((mantissa[25:24]!=2'b01))))
      & c_h_1_20)))) & c_h_1_21)) & (~((~((~((mantissa[22]) | (~((mantissa[21:20]!=2'b01)))))
      & (~(((mantissa[18]) | (~((mantissa[17:16]!=2'b01)))) & c_h_1_24)) & (~((~((~((mantissa[14])
      | (~((mantissa[13:12]!=2'b01))))) & (~(((mantissa[10]) | (~((mantissa[9:8]!=2'b01))))
      & c_h_1_27)))) & c_h_1_28)))) & c_h_1_29)))) & c_h_1_30)) & (~((~((~((mantissa[6])
      | (~((mantissa[5:4]!=2'b01))))) & (~((~((mantissa[2:1]==2'b01))) & c_h_1_33))))
      & c_h_1_34));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_281_nl
      = (~ (mantissa[0])) & ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_wrs_c_200_2_sdt_1
      & c_h_1_33 & c_h_1_34;
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_or_1_nl
      = MUX_v_3_2_2(({(ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_293_nl)
      , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_295_nl)
      , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_296_nl)}),
      3'b111, (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_281_nl));
  assign rtn = {c_h_1_34 , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_nl)
      , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_1_nl)
      , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_and_2_nl)
      , (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_leading_1_leading_sign_71_0_rtn_or_1_nl)};

  function automatic [2:0] MUX_v_3_2_2;
    input [2:0] input_0;
    input [2:0] input_1;
    input [0:0] sel;
    reg [2:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_3_2_2 = result;
  end
  endfunction

endmodule




//------> ./rtl.v 
// ----------------------------------------------------------------------
//  HLS HDL:        Verilog Netlister
//  HLS Version:    10.4b/841621 Production Release
//  HLS Date:       Thu Oct 24 17:20:07 PDT 2019
// 
//  Generated by:   giuseppe@fastml02
//  Generated date: Thu Dec 12 11:47:05 2019
// ----------------------------------------------------------------------

// 
// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_w4_Nangate_RAMS_w4_4096_18_2w2r_rport_7_4096_18_1_gen
// ------------------------------------------------------------------


module mnist_mlp_w4_Nangate_RAMS_w4_4096_18_2w2r_rport_7_4096_18_1_gen (
  Q3, A3, CE3, Q2, A2, CE2, CE2_d, A2_d, Q2_d, port_3_r_ram_ir_internal_RMASK_B_d
);
  input [17:0] Q3;
  output [11:0] A3;
  output CE3;
  input [17:0] Q2;
  output [11:0] A2;
  output CE2;
  input [1:0] CE2_d;
  input [23:0] A2_d;
  output [35:0] Q2_d;
  input [1:0] port_3_r_ram_ir_internal_RMASK_B_d;



  // Interconnect Declarations for Component Instantiations 
  assign Q2_d[35:18] = Q3;
  assign A3 = (A2_d[23:12]);
  assign CE3 = (port_3_r_ram_ir_internal_RMASK_B_d[1]);
  assign Q2_d[17:0] = Q2;
  assign A2 = (A2_d[11:0]);
  assign CE2 = (port_3_r_ram_ir_internal_RMASK_B_d[0]);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_w2_Nangate_RAMS_w2_50176_18_2w2r_rport_5_50176_18_1_gen
// ------------------------------------------------------------------


module mnist_mlp_w2_Nangate_RAMS_w2_50176_18_2w2r_rport_5_50176_18_1_gen (
  Q3, A3, CE3, Q2, A2, CE2, CE2_d, A2_d, Q2_d, port_3_r_ram_ir_internal_RMASK_B_d
);
  input [17:0] Q3;
  output [15:0] A3;
  output CE3;
  input [17:0] Q2;
  output [15:0] A2;
  output CE2;
  input [1:0] CE2_d;
  input [31:0] A2_d;
  output [35:0] Q2_d;
  input [1:0] port_3_r_ram_ir_internal_RMASK_B_d;



  // Interconnect Declarations for Component Instantiations 
  assign Q2_d[35:18] = Q3;
  assign A3 = (A2_d[31:16]);
  assign CE3 = (port_3_r_ram_ir_internal_RMASK_B_d[1]);
  assign Q2_d[17:0] = Q2;
  assign A2 = (A2_d[15:0]);
  assign CE2 = (port_3_r_ram_ir_internal_RMASK_B_d[0]);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_core_fsm
//  FSM Module
// ------------------------------------------------------------------


module mnist_mlp_core_core_fsm (
  clk, rst, core_wen, fsm_output, InitAccum_C_0_tr0, MultLoop_C_1_tr0, ReuseLoop_C_0_tr0,
      ResultLoop_C_0_tr0, MultLoop_1_C_1_tr0, ReuseLoop_1_C_0_tr0, ResultLoop_1_C_0_tr0,
      InitAccum_2_C_0_tr0, MultLoop_2_C_0_tr0, ReuseLoop_2_C_0_tr0, ResultLoop_2_C_1_tr0,
      CALC_SOFTMAX_LOOP_C_1_tr0
);
  input clk;
  input rst;
  input core_wen;
  output [18:0] fsm_output;
  reg [18:0] fsm_output;
  input InitAccum_C_0_tr0;
  input MultLoop_C_1_tr0;
  input ReuseLoop_C_0_tr0;
  input ResultLoop_C_0_tr0;
  input MultLoop_1_C_1_tr0;
  input ReuseLoop_1_C_0_tr0;
  input ResultLoop_1_C_0_tr0;
  input InitAccum_2_C_0_tr0;
  input MultLoop_2_C_0_tr0;
  input ReuseLoop_2_C_0_tr0;
  input ResultLoop_2_C_1_tr0;
  input CALC_SOFTMAX_LOOP_C_1_tr0;


  // FSM State Type Declaration for mnist_mlp_core_core_fsm_1
  parameter
    core_rlp_C_0 = 5'd0,
    main_C_0 = 5'd1,
    InitAccum_C_0 = 5'd2,
    MultLoop_C_0 = 5'd3,
    MultLoop_C_1 = 5'd4,
    ReuseLoop_C_0 = 5'd5,
    ResultLoop_C_0 = 5'd6,
    MultLoop_1_C_0 = 5'd7,
    MultLoop_1_C_1 = 5'd8,
    ReuseLoop_1_C_0 = 5'd9,
    ResultLoop_1_C_0 = 5'd10,
    InitAccum_2_C_0 = 5'd11,
    MultLoop_2_C_0 = 5'd12,
    ReuseLoop_2_C_0 = 5'd13,
    ResultLoop_2_C_0 = 5'd14,
    ResultLoop_2_C_1 = 5'd15,
    main_C_1 = 5'd16,
    CALC_SOFTMAX_LOOP_C_0 = 5'd17,
    CALC_SOFTMAX_LOOP_C_1 = 5'd18;

  reg [4:0] state_var;
  reg [4:0] state_var_NS;


  // Interconnect Declarations for Component Instantiations 
  always @(*)
  begin : mnist_mlp_core_core_fsm_1
    case (state_var)
      main_C_0 : begin
        fsm_output = 19'b0000000000000000010;
        state_var_NS = InitAccum_C_0;
      end
      InitAccum_C_0 : begin
        fsm_output = 19'b0000000000000000100;
        if ( InitAccum_C_0_tr0 ) begin
          state_var_NS = MultLoop_C_0;
        end
        else begin
          state_var_NS = InitAccum_C_0;
        end
      end
      MultLoop_C_0 : begin
        fsm_output = 19'b0000000000000001000;
        state_var_NS = MultLoop_C_1;
      end
      MultLoop_C_1 : begin
        fsm_output = 19'b0000000000000010000;
        if ( MultLoop_C_1_tr0 ) begin
          state_var_NS = ReuseLoop_C_0;
        end
        else begin
          state_var_NS = MultLoop_C_0;
        end
      end
      ReuseLoop_C_0 : begin
        fsm_output = 19'b0000000000000100000;
        if ( ReuseLoop_C_0_tr0 ) begin
          state_var_NS = ResultLoop_C_0;
        end
        else begin
          state_var_NS = MultLoop_C_0;
        end
      end
      ResultLoop_C_0 : begin
        fsm_output = 19'b0000000000001000000;
        if ( ResultLoop_C_0_tr0 ) begin
          state_var_NS = MultLoop_1_C_0;
        end
        else begin
          state_var_NS = ResultLoop_C_0;
        end
      end
      MultLoop_1_C_0 : begin
        fsm_output = 19'b0000000000010000000;
        state_var_NS = MultLoop_1_C_1;
      end
      MultLoop_1_C_1 : begin
        fsm_output = 19'b0000000000100000000;
        if ( MultLoop_1_C_1_tr0 ) begin
          state_var_NS = ReuseLoop_1_C_0;
        end
        else begin
          state_var_NS = MultLoop_1_C_0;
        end
      end
      ReuseLoop_1_C_0 : begin
        fsm_output = 19'b0000000001000000000;
        if ( ReuseLoop_1_C_0_tr0 ) begin
          state_var_NS = ResultLoop_1_C_0;
        end
        else begin
          state_var_NS = MultLoop_1_C_0;
        end
      end
      ResultLoop_1_C_0 : begin
        fsm_output = 19'b0000000010000000000;
        if ( ResultLoop_1_C_0_tr0 ) begin
          state_var_NS = InitAccum_2_C_0;
        end
        else begin
          state_var_NS = ResultLoop_1_C_0;
        end
      end
      InitAccum_2_C_0 : begin
        fsm_output = 19'b0000000100000000000;
        if ( InitAccum_2_C_0_tr0 ) begin
          state_var_NS = MultLoop_2_C_0;
        end
        else begin
          state_var_NS = InitAccum_2_C_0;
        end
      end
      MultLoop_2_C_0 : begin
        fsm_output = 19'b0000001000000000000;
        if ( MultLoop_2_C_0_tr0 ) begin
          state_var_NS = ReuseLoop_2_C_0;
        end
        else begin
          state_var_NS = MultLoop_2_C_0;
        end
      end
      ReuseLoop_2_C_0 : begin
        fsm_output = 19'b0000010000000000000;
        if ( ReuseLoop_2_C_0_tr0 ) begin
          state_var_NS = ResultLoop_2_C_0;
        end
        else begin
          state_var_NS = MultLoop_2_C_0;
        end
      end
      ResultLoop_2_C_0 : begin
        fsm_output = 19'b0000100000000000000;
        state_var_NS = ResultLoop_2_C_1;
      end
      ResultLoop_2_C_1 : begin
        fsm_output = 19'b0001000000000000000;
        if ( ResultLoop_2_C_1_tr0 ) begin
          state_var_NS = main_C_1;
        end
        else begin
          state_var_NS = ResultLoop_2_C_0;
        end
      end
      main_C_1 : begin
        fsm_output = 19'b0010000000000000000;
        state_var_NS = CALC_SOFTMAX_LOOP_C_0;
      end
      CALC_SOFTMAX_LOOP_C_0 : begin
        fsm_output = 19'b0100000000000000000;
        state_var_NS = CALC_SOFTMAX_LOOP_C_1;
      end
      CALC_SOFTMAX_LOOP_C_1 : begin
        fsm_output = 19'b1000000000000000000;
        if ( CALC_SOFTMAX_LOOP_C_1_tr0 ) begin
          state_var_NS = main_C_0;
        end
        else begin
          state_var_NS = CALC_SOFTMAX_LOOP_C_0;
        end
      end
      // core_rlp_C_0
      default : begin
        fsm_output = 19'b0000000000000000001;
        state_var_NS = main_C_0;
      end
    endcase
  end

  always @(posedge clk) begin
    if ( rst ) begin
      state_var <= core_rlp_C_0;
    end
    else if ( core_wen ) begin
      state_var <= state_var_NS;
    end
  end

endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_staller
// ------------------------------------------------------------------


module mnist_mlp_core_staller (
  clk, rst, core_wen, core_wten, input1_rsci_wen_comp, output1_rsci_wen_comp, b2_rsci_wen_comp,
      b4_rsci_wen_comp, b6_rsci_wen_comp
);
  input clk;
  input rst;
  output core_wen;
  output core_wten;
  input input1_rsci_wen_comp;
  input output1_rsci_wen_comp;
  input b2_rsci_wen_comp;
  input b4_rsci_wen_comp;
  input b6_rsci_wen_comp;


  // Interconnect Declarations
  reg core_wten_reg;


  // Interconnect Declarations for Component Instantiations 
  assign core_wen = input1_rsci_wen_comp & output1_rsci_wen_comp & b2_rsci_wen_comp
      & b4_rsci_wen_comp & b6_rsci_wen_comp;
  assign core_wten = core_wten_reg;
  always @(posedge clk) begin
    if ( rst ) begin
      core_wten_reg <= 1'b0;
    end
    else begin
      core_wten_reg <= ~ core_wen;
    end
  end
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_b6_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_b6_rsci (
  b6_rsc_dat, b6_rsc_vld, b6_rsci_oswt, b6_rsci_wen_comp, b6_rsci_idat_mxwt
);
  input [1151:0] b6_rsc_dat;
  input b6_rsc_vld;
  input b6_rsci_oswt;
  output b6_rsci_wen_comp;
  output [179:0] b6_rsci_idat_mxwt;


  // Interconnect Declarations
  wire b6_rsci_ivld;
  wire [1151:0] b6_rsci_idat;


  // Interconnect Declarations for Component Instantiations 
  ccs_in_vld_v1 #(.rscid(32'sd10),
  .width(32'sd1152)) b6_rsci (
      .vld(b6_rsc_vld),
      .dat(b6_rsc_dat),
      .ivld(b6_rsci_ivld),
      .idat(b6_rsci_idat)
    );
  assign b6_rsci_idat_mxwt = b6_rsci_idat[179:0];
  assign b6_rsci_wen_comp = (~ b6_rsci_oswt) | b6_rsci_ivld;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_b4_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_b4_rsci (
  b4_rsc_dat, b4_rsc_vld, b4_rsci_oswt, b4_rsci_wen_comp, b4_rsci_idat_mxwt
);
  input [1151:0] b4_rsc_dat;
  input b4_rsc_vld;
  input b4_rsci_oswt;
  output b4_rsci_wen_comp;
  output [1151:0] b4_rsci_idat_mxwt;


  // Interconnect Declarations
  wire b4_rsci_ivld;
  wire [1151:0] b4_rsci_idat;


  // Interconnect Declarations for Component Instantiations 
  ccs_in_vld_v1 #(.rscid(32'sd8),
  .width(32'sd1152)) b4_rsci (
      .vld(b4_rsc_vld),
      .dat(b4_rsc_dat),
      .ivld(b4_rsci_ivld),
      .idat(b4_rsci_idat)
    );
  assign b4_rsci_idat_mxwt = b4_rsci_idat;
  assign b4_rsci_wen_comp = (~ b4_rsci_oswt) | b4_rsci_ivld;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp
// ------------------------------------------------------------------


module mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp (
  clk, rst, w4_rsci_A2_d, w4_rsci_Q2_d, w4_rsci_A2_d_core, w4_rsci_Q2_d_mxwt, w4_rsci_biwt,
      w4_rsci_bdwt
);
  input clk;
  input rst;
  output [11:0] w4_rsci_A2_d;
  input [35:0] w4_rsci_Q2_d;
  input [23:0] w4_rsci_A2_d_core;
  output [17:0] w4_rsci_Q2_d_mxwt;
  input w4_rsci_biwt;
  input w4_rsci_bdwt;


  // Interconnect Declarations
  reg w4_rsci_bcwt;
  reg [17:0] w4_rsci_Q2_d_bfwt_17_0;
  wire [17:0] w4_rsci_Q2_d_mxwt_opt_17_0;


  // Interconnect Declarations for Component Instantiations 
  assign w4_rsci_Q2_d_mxwt_opt_17_0 = MUX_v_18_2_2((w4_rsci_Q2_d[17:0]), w4_rsci_Q2_d_bfwt_17_0,
      w4_rsci_bcwt);
  assign w4_rsci_Q2_d_mxwt = w4_rsci_Q2_d_mxwt_opt_17_0;
  assign w4_rsci_A2_d = w4_rsci_A2_d_core[11:0];
  always @(posedge clk) begin
    if ( rst ) begin
      w4_rsci_bcwt <= 1'b0;
    end
    else begin
      w4_rsci_bcwt <= ~((~(w4_rsci_bcwt | w4_rsci_biwt)) | w4_rsci_bdwt);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      w4_rsci_Q2_d_bfwt_17_0 <= 18'b000000000000000000;
    end
    else if ( ~ w4_rsci_bcwt ) begin
      w4_rsci_Q2_d_bfwt_17_0 <= w4_rsci_Q2_d_mxwt_opt_17_0;
    end
  end

  function automatic [17:0] MUX_v_18_2_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [0:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_18_2_2 = result;
  end
  endfunction

endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl (
  core_wen, core_wten, w4_rsci_oswt, w4_rsci_CE2_d_core_psct, w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct,
      w4_rsci_biwt, w4_rsci_bdwt, w4_rsci_CE2_d_core_sct, w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct,
      w4_rsci_oswt_pff
);
  input core_wen;
  input core_wten;
  input w4_rsci_oswt;
  input [1:0] w4_rsci_CE2_d_core_psct;
  input [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  output w4_rsci_biwt;
  output w4_rsci_bdwt;
  output [1:0] w4_rsci_CE2_d_core_sct;
  output [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  input w4_rsci_oswt_pff;


  // Interconnect Declarations
  wire w4_rsci_dswt_pff;

  wire[0:0] w4_and_nl;
  wire[0:0] w4_and_2_nl;

  // Interconnect Declarations for Component Instantiations 
  assign w4_rsci_bdwt = w4_rsci_oswt & core_wen;
  assign w4_rsci_biwt = (~ core_wten) & w4_rsci_oswt;
  assign w4_and_nl = (w4_rsci_CE2_d_core_psct[0]) & w4_rsci_dswt_pff;
  assign w4_rsci_CE2_d_core_sct = {1'b0 , (w4_and_nl)};
  assign w4_rsci_dswt_pff = core_wen & w4_rsci_oswt_pff;
  assign w4_and_2_nl = (w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[0])
      & w4_rsci_dswt_pff;
  assign w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct = {1'b0 , (w4_and_2_nl)};
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_b2_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_b2_rsci (
  b2_rsc_dat, b2_rsc_vld, b2_rsci_oswt, b2_rsci_wen_comp, b2_rsci_idat_mxwt
);
  input [1151:0] b2_rsc_dat;
  input b2_rsc_vld;
  input b2_rsci_oswt;
  output b2_rsci_wen_comp;
  output [1151:0] b2_rsci_idat_mxwt;


  // Interconnect Declarations
  wire b2_rsci_ivld;
  wire [1151:0] b2_rsci_idat;


  // Interconnect Declarations for Component Instantiations 
  ccs_in_vld_v1 #(.rscid(32'sd6),
  .width(32'sd1152)) b2_rsci (
      .vld(b2_rsc_vld),
      .dat(b2_rsc_dat),
      .ivld(b2_rsci_ivld),
      .idat(b2_rsci_idat)
    );
  assign b2_rsci_idat_mxwt = b2_rsci_idat;
  assign b2_rsci_wen_comp = (~ b2_rsci_oswt) | b2_rsci_ivld;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp
// ------------------------------------------------------------------


module mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp (
  clk, rst, w2_rsci_A2_d, w2_rsci_Q2_d, w2_rsci_A2_d_core, w2_rsci_Q2_d_mxwt, w2_rsci_biwt,
      w2_rsci_bdwt
);
  input clk;
  input rst;
  output [15:0] w2_rsci_A2_d;
  input [35:0] w2_rsci_Q2_d;
  input [31:0] w2_rsci_A2_d_core;
  output [17:0] w2_rsci_Q2_d_mxwt;
  input w2_rsci_biwt;
  input w2_rsci_bdwt;


  // Interconnect Declarations
  reg w2_rsci_bcwt;
  reg [17:0] w2_rsci_Q2_d_bfwt_17_0;
  wire [17:0] w2_rsci_Q2_d_mxwt_opt_17_0;


  // Interconnect Declarations for Component Instantiations 
  assign w2_rsci_Q2_d_mxwt_opt_17_0 = MUX_v_18_2_2((w2_rsci_Q2_d[17:0]), w2_rsci_Q2_d_bfwt_17_0,
      w2_rsci_bcwt);
  assign w2_rsci_Q2_d_mxwt = w2_rsci_Q2_d_mxwt_opt_17_0;
  assign w2_rsci_A2_d = w2_rsci_A2_d_core[15:0];
  always @(posedge clk) begin
    if ( rst ) begin
      w2_rsci_bcwt <= 1'b0;
    end
    else begin
      w2_rsci_bcwt <= ~((~(w2_rsci_bcwt | w2_rsci_biwt)) | w2_rsci_bdwt);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      w2_rsci_Q2_d_bfwt_17_0 <= 18'b000000000000000000;
    end
    else if ( ~ w2_rsci_bcwt ) begin
      w2_rsci_Q2_d_bfwt_17_0 <= w2_rsci_Q2_d_mxwt_opt_17_0;
    end
  end

  function automatic [17:0] MUX_v_18_2_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [0:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_18_2_2 = result;
  end
  endfunction

endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl (
  core_wen, core_wten, w2_rsci_oswt, w2_rsci_CE2_d_core_psct, w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct,
      w2_rsci_biwt, w2_rsci_bdwt, w2_rsci_CE2_d_core_sct, w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct,
      w2_rsci_oswt_pff
);
  input core_wen;
  input core_wten;
  input w2_rsci_oswt;
  input [1:0] w2_rsci_CE2_d_core_psct;
  input [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  output w2_rsci_biwt;
  output w2_rsci_bdwt;
  output [1:0] w2_rsci_CE2_d_core_sct;
  output [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  input w2_rsci_oswt_pff;


  // Interconnect Declarations
  wire w2_rsci_dswt_pff;

  wire[0:0] w2_and_nl;
  wire[0:0] w2_and_2_nl;

  // Interconnect Declarations for Component Instantiations 
  assign w2_rsci_bdwt = w2_rsci_oswt & core_wen;
  assign w2_rsci_biwt = (~ core_wten) & w2_rsci_oswt;
  assign w2_and_nl = (w2_rsci_CE2_d_core_psct[0]) & w2_rsci_dswt_pff;
  assign w2_rsci_CE2_d_core_sct = {1'b0 , (w2_and_nl)};
  assign w2_rsci_dswt_pff = core_wen & w2_rsci_oswt_pff;
  assign w2_and_2_nl = (w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[0])
      & w2_rsci_dswt_pff;
  assign w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct = {1'b0 , (w2_and_2_nl)};
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_const_size_out_1_rsci_const_size_out_1_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_const_size_out_1_rsci_const_size_out_1_rsc_wait_ctrl (
  core_wten, const_size_out_1_rsci_iswt0, const_size_out_1_rsci_ivld_core_sct
);
  input core_wten;
  input const_size_out_1_rsci_iswt0;
  output const_size_out_1_rsci_ivld_core_sct;



  // Interconnect Declarations for Component Instantiations 
  assign const_size_out_1_rsci_ivld_core_sct = const_size_out_1_rsci_iswt0 & (~ core_wten);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_const_size_in_1_rsci_const_size_in_1_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_const_size_in_1_rsci_const_size_in_1_rsc_wait_ctrl (
  core_wten, const_size_in_1_rsci_iswt0, const_size_in_1_rsci_ivld_core_sct
);
  input core_wten;
  input const_size_in_1_rsci_iswt0;
  output const_size_in_1_rsci_ivld_core_sct;



  // Interconnect Declarations for Component Instantiations 
  assign const_size_in_1_rsci_ivld_core_sct = const_size_in_1_rsci_iswt0 & (~ core_wten);
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_output1_rsci_output1_rsc_wait_dp
// ------------------------------------------------------------------


module mnist_mlp_core_output1_rsci_output1_rsc_wait_dp (
  clk, rst, output1_rsci_oswt, output1_rsci_wen_comp, output1_rsci_biwt, output1_rsci_bdwt,
      output1_rsci_bcwt
);
  input clk;
  input rst;
  input output1_rsci_oswt;
  output output1_rsci_wen_comp;
  input output1_rsci_biwt;
  input output1_rsci_bdwt;
  output output1_rsci_bcwt;
  reg output1_rsci_bcwt;



  // Interconnect Declarations for Component Instantiations 
  assign output1_rsci_wen_comp = (~ output1_rsci_oswt) | output1_rsci_biwt | output1_rsci_bcwt;
  always @(posedge clk) begin
    if ( rst ) begin
      output1_rsci_bcwt <= 1'b0;
    end
    else begin
      output1_rsci_bcwt <= ~((~(output1_rsci_bcwt | output1_rsci_biwt)) | output1_rsci_bdwt);
    end
  end
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_output1_rsci_output1_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_output1_rsci_output1_rsc_wait_ctrl (
  core_wen, output1_rsci_oswt, output1_rsci_irdy, output1_rsci_biwt, output1_rsci_bdwt,
      output1_rsci_bcwt, output1_rsci_ivld_core_sct
);
  input core_wen;
  input output1_rsci_oswt;
  input output1_rsci_irdy;
  output output1_rsci_biwt;
  output output1_rsci_bdwt;
  input output1_rsci_bcwt;
  output output1_rsci_ivld_core_sct;


  // Interconnect Declarations
  wire output1_rsci_ogwt;


  // Interconnect Declarations for Component Instantiations 
  assign output1_rsci_bdwt = output1_rsci_oswt & core_wen;
  assign output1_rsci_biwt = output1_rsci_ogwt & output1_rsci_irdy;
  assign output1_rsci_ogwt = output1_rsci_oswt & (~ output1_rsci_bcwt);
  assign output1_rsci_ivld_core_sct = output1_rsci_ogwt;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_input1_rsci_input1_rsc_wait_dp
// ------------------------------------------------------------------


module mnist_mlp_core_input1_rsci_input1_rsc_wait_dp (
  clk, rst, input1_rsci_oswt, input1_rsci_wen_comp, input1_rsci_idat_mxwt, input1_rsci_biwt,
      input1_rsci_bdwt, input1_rsci_bcwt, input1_rsci_idat
);
  input clk;
  input rst;
  input input1_rsci_oswt;
  output input1_rsci_wen_comp;
  output [14111:0] input1_rsci_idat_mxwt;
  input input1_rsci_biwt;
  input input1_rsci_bdwt;
  output input1_rsci_bcwt;
  reg input1_rsci_bcwt;
  input [14111:0] input1_rsci_idat;


  // Interconnect Declarations
  reg [14111:0] input1_rsci_idat_bfwt;


  // Interconnect Declarations for Component Instantiations 
  assign input1_rsci_wen_comp = (~ input1_rsci_oswt) | input1_rsci_biwt | input1_rsci_bcwt;
  assign input1_rsci_idat_mxwt = MUX_v_14112_2_2(input1_rsci_idat, input1_rsci_idat_bfwt,
      input1_rsci_bcwt);
  always @(posedge clk) begin
    if ( rst ) begin
      input1_rsci_bcwt <= 1'b0;
    end
    else begin
      input1_rsci_bcwt <= ~((~(input1_rsci_bcwt | input1_rsci_biwt)) | input1_rsci_bdwt);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      input1_rsci_idat_bfwt <= {882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
          , 882'b000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000};
    end
    else if ( ~ input1_rsci_bcwt ) begin
      input1_rsci_idat_bfwt <= input1_rsci_idat_mxwt;
    end
  end

  function automatic [14111:0] MUX_v_14112_2_2;
    input [14111:0] input_0;
    input [14111:0] input_1;
    input [0:0] sel;
    reg [14111:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_14112_2_2 = result;
  end
  endfunction

endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_input1_rsci_input1_rsc_wait_ctrl
// ------------------------------------------------------------------


module mnist_mlp_core_input1_rsci_input1_rsc_wait_ctrl (
  core_wen, input1_rsci_oswt, input1_rsci_biwt, input1_rsci_bdwt, input1_rsci_bcwt,
      input1_rsci_irdy_core_sct, input1_rsci_ivld
);
  input core_wen;
  input input1_rsci_oswt;
  output input1_rsci_biwt;
  output input1_rsci_bdwt;
  input input1_rsci_bcwt;
  output input1_rsci_irdy_core_sct;
  input input1_rsci_ivld;


  // Interconnect Declarations
  wire input1_rsci_ogwt;


  // Interconnect Declarations for Component Instantiations 
  assign input1_rsci_bdwt = input1_rsci_oswt & core_wen;
  assign input1_rsci_biwt = input1_rsci_ogwt & input1_rsci_ivld;
  assign input1_rsci_ogwt = input1_rsci_oswt & (~ input1_rsci_bcwt);
  assign input1_rsci_irdy_core_sct = input1_rsci_ogwt;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w4_rsci_1
// ------------------------------------------------------------------


module mnist_mlp_core_w4_rsci_1 (
  clk, rst, w4_rsci_CE2_d, w4_rsci_A2_d, w4_rsci_Q2_d, w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d,
      core_wen, core_wten, w4_rsci_oswt, w4_rsci_CE2_d_core_psct, w4_rsci_A2_d_core,
      w4_rsci_Q2_d_mxwt, w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct, w4_rsci_oswt_pff
);
  input clk;
  input rst;
  output [1:0] w4_rsci_CE2_d;
  output [11:0] w4_rsci_A2_d;
  input [35:0] w4_rsci_Q2_d;
  output [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d;
  input core_wen;
  input core_wten;
  input w4_rsci_oswt;
  input [1:0] w4_rsci_CE2_d_core_psct;
  input [23:0] w4_rsci_A2_d_core;
  output [17:0] w4_rsci_Q2_d_mxwt;
  input [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  input w4_rsci_oswt_pff;


  // Interconnect Declarations
  wire w4_rsci_biwt;
  wire w4_rsci_bdwt;
  wire [1:0] w4_rsci_CE2_d_core_sct;
  wire [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  wire [17:0] w4_rsci_Q2_d_mxwt_pconst;
  wire [11:0] w4_rsci_A2_d_reg;


  // Interconnect Declarations for Component Instantiations 
  wire [1:0] nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_CE2_d_core_psct;
  assign nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_CE2_d_core_psct
      = {1'b0 , (w4_rsci_CE2_d_core_psct[0])};
  wire [1:0] nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  assign nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct
      = {1'b0 , (w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[0])};
  wire [23:0] nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp_inst_w4_rsci_A2_d_core;
  assign nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp_inst_w4_rsci_A2_d_core = {12'b000000000000
      , (w4_rsci_A2_d_core[11:0])};
  mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst
      (
      .core_wen(core_wen),
      .core_wten(core_wten),
      .w4_rsci_oswt(w4_rsci_oswt),
      .w4_rsci_CE2_d_core_psct(nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_CE2_d_core_psct[1:0]),
      .w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct(nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_ctrl_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[1:0]),
      .w4_rsci_biwt(w4_rsci_biwt),
      .w4_rsci_bdwt(w4_rsci_bdwt),
      .w4_rsci_CE2_d_core_sct(w4_rsci_CE2_d_core_sct),
      .w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct(w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct),
      .w4_rsci_oswt_pff(w4_rsci_oswt_pff)
    );
  mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp_inst
      (
      .clk(clk),
      .rst(rst),
      .w4_rsci_A2_d(w4_rsci_A2_d_reg),
      .w4_rsci_Q2_d(w4_rsci_Q2_d),
      .w4_rsci_A2_d_core(nl_mnist_mlp_core_w4_rsci_1_w4_rsc_wait_dp_inst_w4_rsci_A2_d_core[23:0]),
      .w4_rsci_Q2_d_mxwt(w4_rsci_Q2_d_mxwt_pconst),
      .w4_rsci_biwt(w4_rsci_biwt),
      .w4_rsci_bdwt(w4_rsci_bdwt)
    );
  assign w4_rsci_Q2_d_mxwt = w4_rsci_Q2_d_mxwt_pconst;
  assign w4_rsci_CE2_d = w4_rsci_CE2_d_core_sct;
  assign w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d = w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  assign w4_rsci_A2_d = w4_rsci_A2_d_reg;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_w2_rsci_1
// ------------------------------------------------------------------


module mnist_mlp_core_w2_rsci_1 (
  clk, rst, w2_rsci_CE2_d, w2_rsci_A2_d, w2_rsci_Q2_d, w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d,
      core_wen, core_wten, w2_rsci_oswt, w2_rsci_CE2_d_core_psct, w2_rsci_A2_d_core,
      w2_rsci_Q2_d_mxwt, w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct, w2_rsci_oswt_pff
);
  input clk;
  input rst;
  output [1:0] w2_rsci_CE2_d;
  output [15:0] w2_rsci_A2_d;
  input [35:0] w2_rsci_Q2_d;
  output [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d;
  input core_wen;
  input core_wten;
  input w2_rsci_oswt;
  input [1:0] w2_rsci_CE2_d_core_psct;
  input [31:0] w2_rsci_A2_d_core;
  output [17:0] w2_rsci_Q2_d_mxwt;
  input [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  input w2_rsci_oswt_pff;


  // Interconnect Declarations
  wire w2_rsci_biwt;
  wire w2_rsci_bdwt;
  wire [1:0] w2_rsci_CE2_d_core_sct;
  wire [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  wire [17:0] w2_rsci_Q2_d_mxwt_pconst;
  wire [15:0] w2_rsci_A2_d_reg;


  // Interconnect Declarations for Component Instantiations 
  wire [1:0] nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_CE2_d_core_psct;
  assign nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_CE2_d_core_psct
      = {1'b0 , (w2_rsci_CE2_d_core_psct[0])};
  wire [1:0] nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  assign nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct
      = {1'b0 , (w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[0])};
  wire [31:0] nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp_inst_w2_rsci_A2_d_core;
  assign nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp_inst_w2_rsci_A2_d_core = {16'b0000000000000000
      , (w2_rsci_A2_d_core[15:0])};
  mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst
      (
      .core_wen(core_wen),
      .core_wten(core_wten),
      .w2_rsci_oswt(w2_rsci_oswt),
      .w2_rsci_CE2_d_core_psct(nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_CE2_d_core_psct[1:0]),
      .w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct(nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_ctrl_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[1:0]),
      .w2_rsci_biwt(w2_rsci_biwt),
      .w2_rsci_bdwt(w2_rsci_bdwt),
      .w2_rsci_CE2_d_core_sct(w2_rsci_CE2_d_core_sct),
      .w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct(w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct),
      .w2_rsci_oswt_pff(w2_rsci_oswt_pff)
    );
  mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp_inst
      (
      .clk(clk),
      .rst(rst),
      .w2_rsci_A2_d(w2_rsci_A2_d_reg),
      .w2_rsci_Q2_d(w2_rsci_Q2_d),
      .w2_rsci_A2_d_core(nl_mnist_mlp_core_w2_rsci_1_w2_rsc_wait_dp_inst_w2_rsci_A2_d_core[31:0]),
      .w2_rsci_Q2_d_mxwt(w2_rsci_Q2_d_mxwt_pconst),
      .w2_rsci_biwt(w2_rsci_biwt),
      .w2_rsci_bdwt(w2_rsci_bdwt)
    );
  assign w2_rsci_Q2_d_mxwt = w2_rsci_Q2_d_mxwt_pconst;
  assign w2_rsci_CE2_d = w2_rsci_CE2_d_core_sct;
  assign w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d = w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_sct;
  assign w2_rsci_A2_d = w2_rsci_A2_d_reg;
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_const_size_out_1_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_const_size_out_1_rsci (
  const_size_out_1_rsc_dat, const_size_out_1_rsc_vld, core_wten, const_size_out_1_rsci_iswt0
);
  output [15:0] const_size_out_1_rsc_dat;
  output const_size_out_1_rsc_vld;
  input core_wten;
  input const_size_out_1_rsci_iswt0;


  // Interconnect Declarations
  wire const_size_out_1_rsci_ivld_core_sct;


  // Interconnect Declarations for Component Instantiations 
  ccs_out_vld_v1 #(.rscid(32'sd4),
  .width(32'sd16)) const_size_out_1_rsci (
      .ivld(const_size_out_1_rsci_ivld_core_sct),
      .idat(16'b0000000000001010),
      .vld(const_size_out_1_rsc_vld),
      .dat(const_size_out_1_rsc_dat)
    );
  mnist_mlp_core_const_size_out_1_rsci_const_size_out_1_rsc_wait_ctrl mnist_mlp_core_const_size_out_1_rsci_const_size_out_1_rsc_wait_ctrl_inst
      (
      .core_wten(core_wten),
      .const_size_out_1_rsci_iswt0(const_size_out_1_rsci_iswt0),
      .const_size_out_1_rsci_ivld_core_sct(const_size_out_1_rsci_ivld_core_sct)
    );
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_const_size_in_1_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_const_size_in_1_rsci (
  const_size_in_1_rsc_dat, const_size_in_1_rsc_vld, core_wten, const_size_in_1_rsci_iswt0
);
  output [15:0] const_size_in_1_rsc_dat;
  output const_size_in_1_rsc_vld;
  input core_wten;
  input const_size_in_1_rsci_iswt0;


  // Interconnect Declarations
  wire const_size_in_1_rsci_ivld_core_sct;


  // Interconnect Declarations for Component Instantiations 
  ccs_out_vld_v1 #(.rscid(32'sd3),
  .width(32'sd16)) const_size_in_1_rsci (
      .ivld(const_size_in_1_rsci_ivld_core_sct),
      .idat(16'b0000001100010000),
      .vld(const_size_in_1_rsc_vld),
      .dat(const_size_in_1_rsc_dat)
    );
  mnist_mlp_core_const_size_in_1_rsci_const_size_in_1_rsc_wait_ctrl mnist_mlp_core_const_size_in_1_rsci_const_size_in_1_rsc_wait_ctrl_inst
      (
      .core_wten(core_wten),
      .const_size_in_1_rsci_iswt0(const_size_in_1_rsci_iswt0),
      .const_size_in_1_rsci_ivld_core_sct(const_size_in_1_rsci_ivld_core_sct)
    );
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_output1_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_output1_rsci (
  clk, rst, output1_rsc_dat, output1_rsc_vld, output1_rsc_rdy, core_wen, output1_rsci_oswt,
      output1_rsci_wen_comp, output1_rsci_idat
);
  input clk;
  input rst;
  output [179:0] output1_rsc_dat;
  output output1_rsc_vld;
  input output1_rsc_rdy;
  input core_wen;
  input output1_rsci_oswt;
  output output1_rsci_wen_comp;
  input [179:0] output1_rsci_idat;


  // Interconnect Declarations
  wire output1_rsci_irdy;
  wire output1_rsci_biwt;
  wire output1_rsci_bdwt;
  wire output1_rsci_bcwt;
  wire output1_rsci_ivld_core_sct;


  // Interconnect Declarations for Component Instantiations 
  wire [179:0] nl_output1_rsci_idat;
  assign nl_output1_rsci_idat = {6'b000000 , (output1_rsci_idat[173:162]) , 6'b000000
      , (output1_rsci_idat[155:144]) , 6'b000000 , (output1_rsci_idat[137:126]) ,
      6'b000000 , (output1_rsci_idat[119:108]) , 6'b000000 , (output1_rsci_idat[101:90])
      , 6'b000000 , (output1_rsci_idat[83:72]) , 6'b000000 , (output1_rsci_idat[65:54])
      , 6'b000000 , (output1_rsci_idat[47:36]) , 6'b000000 , (output1_rsci_idat[29:18])
      , 6'b000000 , (output1_rsci_idat[11:0])};
  ccs_out_wait_v1 #(.rscid(32'sd2),
  .width(32'sd180)) output1_rsci (
      .irdy(output1_rsci_irdy),
      .ivld(output1_rsci_ivld_core_sct),
      .idat(nl_output1_rsci_idat[179:0]),
      .rdy(output1_rsc_rdy),
      .vld(output1_rsc_vld),
      .dat(output1_rsc_dat)
    );
  mnist_mlp_core_output1_rsci_output1_rsc_wait_ctrl mnist_mlp_core_output1_rsci_output1_rsc_wait_ctrl_inst
      (
      .core_wen(core_wen),
      .output1_rsci_oswt(output1_rsci_oswt),
      .output1_rsci_irdy(output1_rsci_irdy),
      .output1_rsci_biwt(output1_rsci_biwt),
      .output1_rsci_bdwt(output1_rsci_bdwt),
      .output1_rsci_bcwt(output1_rsci_bcwt),
      .output1_rsci_ivld_core_sct(output1_rsci_ivld_core_sct)
    );
  mnist_mlp_core_output1_rsci_output1_rsc_wait_dp mnist_mlp_core_output1_rsci_output1_rsc_wait_dp_inst
      (
      .clk(clk),
      .rst(rst),
      .output1_rsci_oswt(output1_rsci_oswt),
      .output1_rsci_wen_comp(output1_rsci_wen_comp),
      .output1_rsci_biwt(output1_rsci_biwt),
      .output1_rsci_bdwt(output1_rsci_bdwt),
      .output1_rsci_bcwt(output1_rsci_bcwt)
    );
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core_input1_rsci
// ------------------------------------------------------------------


module mnist_mlp_core_input1_rsci (
  clk, rst, input1_rsc_dat, input1_rsc_vld, input1_rsc_rdy, core_wen, input1_rsci_oswt,
      input1_rsci_wen_comp, input1_rsci_idat_mxwt
);
  input clk;
  input rst;
  input [14111:0] input1_rsc_dat;
  input input1_rsc_vld;
  output input1_rsc_rdy;
  input core_wen;
  input input1_rsci_oswt;
  output input1_rsci_wen_comp;
  output [14111:0] input1_rsci_idat_mxwt;


  // Interconnect Declarations
  wire input1_rsci_biwt;
  wire input1_rsci_bdwt;
  wire input1_rsci_bcwt;
  wire input1_rsci_irdy_core_sct;
  wire input1_rsci_ivld;
  wire [14111:0] input1_rsci_idat;


  // Interconnect Declarations for Component Instantiations 
  ccs_in_wait_v1 #(.rscid(32'sd1),
  .width(32'sd14112)) input1_rsci (
      .rdy(input1_rsc_rdy),
      .vld(input1_rsc_vld),
      .dat(input1_rsc_dat),
      .irdy(input1_rsci_irdy_core_sct),
      .ivld(input1_rsci_ivld),
      .idat(input1_rsci_idat)
    );
  mnist_mlp_core_input1_rsci_input1_rsc_wait_ctrl mnist_mlp_core_input1_rsci_input1_rsc_wait_ctrl_inst
      (
      .core_wen(core_wen),
      .input1_rsci_oswt(input1_rsci_oswt),
      .input1_rsci_biwt(input1_rsci_biwt),
      .input1_rsci_bdwt(input1_rsci_bdwt),
      .input1_rsci_bcwt(input1_rsci_bcwt),
      .input1_rsci_irdy_core_sct(input1_rsci_irdy_core_sct),
      .input1_rsci_ivld(input1_rsci_ivld)
    );
  mnist_mlp_core_input1_rsci_input1_rsc_wait_dp mnist_mlp_core_input1_rsci_input1_rsc_wait_dp_inst
      (
      .clk(clk),
      .rst(rst),
      .input1_rsci_oswt(input1_rsci_oswt),
      .input1_rsci_wen_comp(input1_rsci_wen_comp),
      .input1_rsci_idat_mxwt(input1_rsci_idat_mxwt),
      .input1_rsci_biwt(input1_rsci_biwt),
      .input1_rsci_bdwt(input1_rsci_bdwt),
      .input1_rsci_bcwt(input1_rsci_bcwt),
      .input1_rsci_idat(input1_rsci_idat)
    );
endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp_core
// ------------------------------------------------------------------


module mnist_mlp_core (
  clk, rst, input1_rsc_dat, input1_rsc_vld, input1_rsc_rdy, output1_rsc_dat, output1_rsc_vld,
      output1_rsc_rdy, const_size_in_1_rsc_dat, const_size_in_1_rsc_vld, const_size_out_1_rsc_dat,
      const_size_out_1_rsc_vld, b2_rsc_dat, b2_rsc_vld, b4_rsc_dat, b4_rsc_vld, w6_rsc_0_0_dat,
      w6_rsc_1_0_dat, w6_rsc_2_0_dat, w6_rsc_3_0_dat, w6_rsc_4_0_dat, w6_rsc_5_0_dat,
      w6_rsc_6_0_dat, w6_rsc_7_0_dat, w6_rsc_8_0_dat, w6_rsc_9_0_dat, b6_rsc_dat,
      b6_rsc_vld, w2_rsci_CE2_d, w2_rsci_A2_d, w2_rsci_Q2_d, w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d,
      w4_rsci_CE2_d, w4_rsci_A2_d, w4_rsci_Q2_d, w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d
);
  input clk;
  input rst;
  input [14111:0] input1_rsc_dat;
  input input1_rsc_vld;
  output input1_rsc_rdy;
  output [179:0] output1_rsc_dat;
  output output1_rsc_vld;
  input output1_rsc_rdy;
  output [15:0] const_size_in_1_rsc_dat;
  output const_size_in_1_rsc_vld;
  output [15:0] const_size_out_1_rsc_dat;
  output const_size_out_1_rsc_vld;
  input [1151:0] b2_rsc_dat;
  input b2_rsc_vld;
  input [1151:0] b4_rsc_dat;
  input b4_rsc_vld;
  input [1151:0] w6_rsc_0_0_dat;
  input [1151:0] w6_rsc_1_0_dat;
  input [1151:0] w6_rsc_2_0_dat;
  input [1151:0] w6_rsc_3_0_dat;
  input [1151:0] w6_rsc_4_0_dat;
  input [1151:0] w6_rsc_5_0_dat;
  input [1151:0] w6_rsc_6_0_dat;
  input [1151:0] w6_rsc_7_0_dat;
  input [1151:0] w6_rsc_8_0_dat;
  input [1151:0] w6_rsc_9_0_dat;
  input [1151:0] b6_rsc_dat;
  input b6_rsc_vld;
  output [1:0] w2_rsci_CE2_d;
  output [15:0] w2_rsci_A2_d;
  input [35:0] w2_rsci_Q2_d;
  output [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d;
  output [1:0] w4_rsci_CE2_d;
  output [11:0] w4_rsci_A2_d;
  input [35:0] w4_rsci_Q2_d;
  output [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d;


  // Interconnect Declarations
  wire core_wen;
  wire core_wten;
  wire input1_rsci_wen_comp;
  wire [14111:0] input1_rsci_idat_mxwt;
  wire output1_rsci_wen_comp;
  wire [17:0] w2_rsci_Q2_d_mxwt;
  reg b2_rsci_oswt;
  wire b2_rsci_wen_comp;
  wire [1151:0] b2_rsci_idat_mxwt;
  wire [17:0] w4_rsci_Q2_d_mxwt;
  reg b4_rsci_oswt;
  wire b4_rsci_wen_comp;
  wire [1151:0] b4_rsci_idat_mxwt;
  reg b6_rsci_oswt;
  wire b6_rsci_wen_comp;
  wire [179:0] b6_rsci_idat_mxwt;
  wire [1151:0] w6_rsc_0_0_i_idat;
  wire [1151:0] w6_rsc_1_0_i_idat;
  wire [1151:0] w6_rsc_2_0_i_idat;
  wire [1151:0] w6_rsc_3_0_i_idat;
  wire [1151:0] w6_rsc_4_0_i_idat;
  wire [1151:0] w6_rsc_5_0_i_idat;
  wire [1151:0] w6_rsc_6_0_i_idat;
  wire [1151:0] w6_rsc_7_0_i_idat;
  wire [1151:0] w6_rsc_8_0_i_idat;
  wire [1151:0] w6_rsc_9_0_i_idat;
  reg [11:0] output1_rsci_idat_173_162;
  reg [11:0] output1_rsci_idat_155_144;
  reg [11:0] output1_rsci_idat_137_126;
  reg [11:0] output1_rsci_idat_119_108;
  reg [11:0] output1_rsci_idat_101_90;
  reg [11:0] output1_rsci_idat_83_72;
  reg [11:0] output1_rsci_idat_65_54;
  reg [11:0] output1_rsci_idat_47_36;
  reg [11:0] output1_rsci_idat_29_18;
  reg [11:0] output1_rsci_idat_11_0;
  wire [18:0] fsm_output;
  wire OUTPUT_LOOP_or_tmp;
  wire [3:0] MultLoop_2_if_1_acc_tmp;
  wire [4:0] nl_MultLoop_2_if_1_acc_tmp;
  wire ResultLoop_1_and_tmp;
  wire ResultLoop_and_1_tmp;
  wire or_dcpl_105;
  wire or_dcpl_106;
  wire or_dcpl_107;
  wire or_dcpl_117;
  wire or_dcpl_118;
  wire or_dcpl_153;
  wire or_dcpl_154;
  wire or_dcpl_160;
  wire or_dcpl_528;
  wire or_dcpl_532;
  wire or_dcpl_537;
  wire or_dcpl_539;
  wire or_dcpl_547;
  wire or_dcpl_548;
  wire or_dcpl_550;
  wire or_dcpl_551;
  wire or_dcpl_552;
  wire or_dcpl_554;
  wire or_dcpl_556;
  wire or_dcpl_557;
  wire or_dcpl_558;
  wire or_dcpl_559;
  wire or_dcpl_560;
  wire or_dcpl_561;
  wire or_dcpl_566;
  wire or_dcpl_568;
  wire or_dcpl_570;
  wire or_dcpl_571;
  wire or_dcpl_572;
  wire or_dcpl_573;
  wire or_dcpl_574;
  wire or_dcpl_576;
  wire or_dcpl_578;
  wire or_dcpl_579;
  wire or_dcpl_580;
  wire or_dcpl_581;
  wire or_dcpl_582;
  wire or_dcpl_583;
  wire or_dcpl_584;
  wire or_dcpl_585;
  wire or_dcpl_586;
  wire or_dcpl_587;
  wire or_dcpl_588;
  wire or_dcpl_589;
  wire or_dcpl_590;
  wire or_dcpl_591;
  wire or_dcpl_592;
  wire or_dcpl_593;
  wire or_dcpl_594;
  wire or_dcpl_595;
  wire or_dcpl_596;
  wire or_dcpl_597;
  wire or_dcpl_598;
  wire or_dcpl_599;
  wire or_dcpl_600;
  wire or_dcpl_601;
  wire or_dcpl_602;
  wire or_dcpl_603;
  wire or_dcpl_604;
  wire or_dcpl_605;
  wire or_dcpl_606;
  wire or_dcpl_607;
  wire or_dcpl_608;
  wire or_dcpl_609;
  wire or_dcpl_610;
  wire or_dcpl_611;
  wire or_dcpl_612;
  wire or_dcpl_613;
  wire or_dcpl_614;
  wire or_dcpl_615;
  wire or_dcpl_616;
  wire or_dcpl_617;
  wire or_dcpl_618;
  wire or_dcpl_619;
  wire or_dcpl_620;
  wire or_dcpl_621;
  wire or_dcpl_622;
  wire or_dcpl_623;
  wire or_dcpl_624;
  wire or_dcpl_625;
  wire or_dcpl_626;
  wire or_dcpl_627;
  wire or_dcpl_628;
  wire or_dcpl_629;
  wire or_dcpl_630;
  wire or_dcpl_631;
  wire or_dcpl_632;
  wire or_dcpl_633;
  wire or_dcpl_634;
  wire or_dcpl_635;
  wire or_dcpl_636;
  wire or_dcpl_637;
  wire or_dcpl_638;
  wire or_dcpl_639;
  wire or_dcpl_640;
  wire or_dcpl_641;
  wire or_dcpl_642;
  wire or_dcpl_643;
  wire or_dcpl_644;
  wire or_dcpl_645;
  wire or_dcpl_646;
  wire or_dcpl_647;
  wire or_dcpl_648;
  wire or_dcpl_649;
  wire or_dcpl_650;
  wire or_dcpl_651;
  wire or_dcpl_652;
  wire or_dcpl_653;
  wire or_dcpl_654;
  wire or_dcpl_655;
  wire or_dcpl_656;
  wire or_dcpl_657;
  wire or_dcpl_658;
  wire or_dcpl_659;
  wire or_dcpl_660;
  wire or_dcpl_661;
  wire or_dcpl_662;
  wire or_dcpl_663;
  wire or_dcpl_670;
  wire and_dcpl_31;
  wire and_dcpl_32;
  wire and_dcpl_33;
  wire and_dcpl_34;
  wire and_dcpl_37;
  wire and_dcpl_38;
  wire and_dcpl_39;
  wire and_dcpl_40;
  wire or_dcpl_680;
  wire and_dcpl_44;
  wire and_dcpl_48;
  wire and_dcpl_49;
  wire and_dcpl_50;
  wire and_dcpl_51;
  wire and_dcpl_52;
  wire and_dcpl_53;
  wire and_dcpl_55;
  wire and_dcpl_56;
  wire and_dcpl_57;
  wire and_dcpl_58;
  wire and_dcpl_59;
  wire and_dcpl_62;
  wire and_dcpl_64;
  wire and_dcpl_67;
  wire and_dcpl_69;
  wire and_dcpl_70;
  wire and_dcpl_72;
  wire and_dcpl_73;
  wire and_dcpl_76;
  wire and_dcpl_77;
  wire and_dcpl_79;
  wire and_dcpl_82;
  wire and_dcpl_86;
  wire and_dcpl_91;
  wire and_dcpl_92;
  wire and_dcpl_96;
  wire and_dcpl_97;
  wire and_dcpl_107;
  wire and_dcpl_111;
  wire and_dcpl_120;
  wire and_dcpl_124;
  wire and_dcpl_140;
  wire and_dcpl_148;
  wire and_dcpl_156;
  wire and_dcpl_179;
  wire and_dcpl_180;
  wire and_dcpl_182;
  wire and_dcpl_183;
  wire and_dcpl_186;
  wire and_dcpl_187;
  wire and_dcpl_189;
  wire and_dcpl_190;
  wire and_dcpl_193;
  wire and_dcpl_196;
  wire and_dcpl_200;
  wire and_dcpl_240;
  wire and_dcpl_244;
  wire or_dcpl_768;
  wire or_dcpl_771;
  wire and_dcpl_296;
  wire or_tmp_318;
  wire or_tmp_1116;
  wire or_tmp_1141;
  wire or_tmp_1155;
  wire and_585_cse;
  wire and_605_cse;
  wire and_586_cse;
  wire and_593_cse;
  wire and_1829_cse;
  wire and_583_cse;
  wire and_2802_cse;
  reg CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm;
  reg [70:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva;
  wire [66:0] operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_0_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_15_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_0_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_1_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_2_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_3_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_4_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_5_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_6_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_7_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_8_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_9_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_10_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_11_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_12_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_13_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_14_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_0_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_1_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_2_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_3_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_4_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_5_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_6_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_7_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_0_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_1_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_2_sva_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_3_sva_1;
  reg [6:0] MultLoop_1_im_6_0_sva_1;
  reg [3:0] CALC_EXP_LOOP_i_3_0_sva_1;
  reg [5:0] InitAccum_1_iacc_6_0_sva_5_0;
  wire [70:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1;
  wire [71:0] nl_ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1;
  wire [17:0] MultLoop_mux_64_itm_mx0w0;
  reg reg_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse;
  reg reg_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse;
  reg reg_const_size_out_1_rsci_ivld_core_psct_cse;
  reg reg_output1_rsci_ivld_core_psct_cse;
  reg reg_input1_rsci_irdy_core_psct_cse;
  wire layer3_out_and_cse;
  wire layer5_out_and_cse;
  wire nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse;
  wire [1:0] w2_rsci_CE2_d_reg;
  wire [15:0] w2_rsci_A2_d_reg;
  wire [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg;
  wire [1:0] w4_rsci_CE2_d_reg;
  wire [11:0] w4_rsci_A2_d_reg;
  wire [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_66_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_68_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_70_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_72_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_74_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_76_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_78_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_80_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_82_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_84_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_86_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_88_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_90_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_92_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_94_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_96_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_98_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_100_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_102_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_104_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_106_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_108_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_110_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_112_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_114_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_116_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_118_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_120_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_122_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_124_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_126_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_132_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_130_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_136_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_134_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_140_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_138_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_144_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_142_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_148_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_146_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_152_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_150_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_156_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_154_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_160_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_158_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_164_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_162_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_168_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_166_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_172_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_170_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_176_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_174_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_180_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_178_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_184_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_182_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_188_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_186_0;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_128_0;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_122_tmp_1;
  reg [16:0] layer3_out_32_16_0_sva_dfm;
  reg [16:0] layer3_out_1_16_0_sva_dfm;
  reg [16:0] layer3_out_33_16_0_sva_dfm;
  reg [16:0] layer3_out_2_16_0_sva_dfm;
  reg [16:0] layer3_out_34_16_0_sva_dfm;
  reg [16:0] layer3_out_3_16_0_sva_dfm;
  reg [16:0] layer3_out_35_16_0_sva_dfm;
  reg [16:0] layer3_out_4_16_0_sva_dfm;
  reg [16:0] layer3_out_36_16_0_sva_dfm;
  reg [16:0] layer3_out_5_16_0_sva_dfm;
  reg [16:0] layer3_out_37_16_0_sva_dfm;
  reg [16:0] layer3_out_6_16_0_sva_dfm;
  reg [16:0] layer3_out_38_16_0_sva_dfm;
  reg [16:0] layer3_out_7_16_0_sva_dfm;
  reg [16:0] layer3_out_39_16_0_sva_dfm;
  reg [16:0] layer3_out_8_16_0_sva_dfm;
  reg [16:0] layer3_out_40_16_0_sva_dfm;
  reg [16:0] layer3_out_9_16_0_sva_dfm;
  reg [16:0] layer3_out_41_16_0_sva_dfm;
  reg [16:0] layer3_out_10_16_0_sva_dfm;
  reg [16:0] layer3_out_42_16_0_sva_dfm;
  reg [16:0] layer3_out_11_16_0_sva_dfm;
  reg [16:0] layer3_out_43_16_0_sva_dfm;
  reg [16:0] layer3_out_12_16_0_sva_dfm;
  reg [16:0] layer3_out_44_16_0_sva_dfm;
  reg [16:0] layer3_out_13_16_0_sva_dfm;
  reg [16:0] layer3_out_45_16_0_sva_dfm;
  reg [16:0] layer3_out_14_16_0_sva_dfm;
  reg [16:0] layer3_out_46_16_0_sva_dfm;
  reg [16:0] layer3_out_15_16_0_sva_dfm;
  reg [16:0] layer3_out_47_16_0_sva_dfm;
  reg [16:0] layer3_out_16_16_0_sva_dfm;
  reg [16:0] layer3_out_48_16_0_sva_dfm;
  reg [16:0] layer3_out_17_16_0_sva_dfm;
  reg [16:0] layer3_out_49_16_0_sva_dfm;
  reg [16:0] layer3_out_18_16_0_sva_dfm;
  reg [16:0] layer3_out_50_16_0_sva_dfm;
  reg [16:0] layer3_out_19_16_0_sva_dfm;
  reg [16:0] layer3_out_51_16_0_sva_dfm;
  reg [16:0] layer3_out_20_16_0_sva_dfm;
  reg [16:0] layer3_out_52_16_0_sva_dfm;
  reg [16:0] layer3_out_21_16_0_sva_dfm;
  reg [16:0] layer3_out_53_16_0_sva_dfm;
  reg [16:0] layer3_out_22_16_0_sva_dfm;
  reg [16:0] layer3_out_54_16_0_sva_dfm;
  reg [16:0] layer3_out_23_16_0_sva_dfm;
  reg [16:0] layer3_out_55_16_0_sva_dfm;
  reg [16:0] layer3_out_24_16_0_sva_dfm;
  reg [16:0] layer3_out_56_16_0_sva_dfm;
  reg [16:0] layer3_out_25_16_0_sva_dfm;
  reg [16:0] layer3_out_57_16_0_sva_dfm;
  reg [16:0] layer3_out_26_16_0_sva_dfm;
  reg [16:0] layer3_out_58_16_0_sva_dfm;
  reg [16:0] layer3_out_27_16_0_sva_dfm;
  reg [16:0] layer3_out_59_16_0_sva_dfm;
  reg [16:0] layer3_out_28_16_0_sva_dfm;
  reg [16:0] layer3_out_60_16_0_sva_dfm;
  reg [16:0] layer3_out_29_16_0_sva_dfm;
  reg [16:0] layer3_out_61_16_0_sva_dfm;
  reg [16:0] layer3_out_30_16_0_sva_dfm;
  reg [16:0] layer3_out_62_16_0_sva_dfm;
  reg [16:0] layer3_out_31_16_0_sva_dfm;
  reg [16:0] layer5_out_32_16_0_sva_dfm;
  reg [16:0] layer5_out_1_16_0_sva_dfm;
  reg [16:0] layer5_out_33_16_0_sva_dfm;
  reg [16:0] layer5_out_2_16_0_sva_dfm;
  reg [16:0] layer5_out_34_16_0_sva_dfm;
  reg [16:0] layer5_out_3_16_0_sva_dfm;
  reg [16:0] layer5_out_35_16_0_sva_dfm;
  reg [16:0] layer5_out_4_16_0_sva_dfm;
  reg [16:0] layer5_out_36_16_0_sva_dfm;
  reg [16:0] layer5_out_5_16_0_sva_dfm;
  reg [16:0] layer5_out_37_16_0_sva_dfm;
  reg [16:0] layer5_out_6_16_0_sva_dfm;
  reg [16:0] layer5_out_38_16_0_sva_dfm;
  reg [16:0] layer5_out_7_16_0_sva_dfm;
  reg [16:0] layer5_out_39_16_0_sva_dfm;
  reg [16:0] layer5_out_8_16_0_sva_dfm;
  reg [16:0] layer5_out_40_16_0_sva_dfm;
  reg [16:0] layer5_out_9_16_0_sva_dfm;
  reg [16:0] layer5_out_41_16_0_sva_dfm;
  reg [16:0] layer5_out_10_16_0_sva_dfm;
  reg [16:0] layer5_out_42_16_0_sva_dfm;
  reg [16:0] layer5_out_11_16_0_sva_dfm;
  reg [16:0] layer5_out_43_16_0_sva_dfm;
  reg [16:0] layer5_out_12_16_0_sva_dfm;
  reg [16:0] layer5_out_44_16_0_sva_dfm;
  reg [16:0] layer5_out_13_16_0_sva_dfm;
  reg [16:0] layer5_out_45_16_0_sva_dfm;
  reg [16:0] layer5_out_14_16_0_sva_dfm;
  reg [16:0] layer5_out_46_16_0_sva_dfm;
  reg [16:0] layer5_out_15_16_0_sva_dfm;
  reg [16:0] layer5_out_47_16_0_sva_dfm;
  reg [16:0] layer5_out_16_16_0_sva_dfm;
  reg [16:0] layer5_out_48_16_0_sva_dfm;
  reg [16:0] layer5_out_17_16_0_sva_dfm;
  reg [16:0] layer5_out_49_16_0_sva_dfm;
  reg [16:0] layer5_out_18_16_0_sva_dfm;
  reg [16:0] layer5_out_50_16_0_sva_dfm;
  reg [16:0] layer5_out_19_16_0_sva_dfm;
  reg [16:0] layer5_out_51_16_0_sva_dfm;
  reg [16:0] layer5_out_20_16_0_sva_dfm;
  reg [16:0] layer5_out_52_16_0_sva_dfm;
  reg [16:0] layer5_out_21_16_0_sva_dfm;
  reg [16:0] layer5_out_53_16_0_sva_dfm;
  reg [16:0] layer5_out_22_16_0_sva_dfm;
  reg [16:0] layer5_out_54_16_0_sva_dfm;
  reg [16:0] layer5_out_23_16_0_sva_dfm;
  reg [16:0] layer5_out_55_16_0_sva_dfm;
  reg [16:0] layer5_out_24_16_0_sva_dfm;
  reg [16:0] layer5_out_56_16_0_sva_dfm;
  reg [16:0] layer5_out_25_16_0_sva_dfm;
  reg [16:0] layer5_out_57_16_0_sva_dfm;
  reg [16:0] layer5_out_26_16_0_sva_dfm;
  reg [16:0] layer5_out_58_16_0_sva_dfm;
  reg [16:0] layer5_out_27_16_0_sva_dfm;
  reg [16:0] layer5_out_59_16_0_sva_dfm;
  reg [16:0] layer5_out_28_16_0_sva_dfm;
  reg [16:0] layer5_out_60_16_0_sva_dfm;
  reg [16:0] layer5_out_29_16_0_sva_dfm;
  reg [16:0] layer5_out_61_16_0_sva_dfm;
  reg [16:0] layer5_out_30_16_0_sva_dfm;
  reg [16:0] layer5_out_62_16_0_sva_dfm;
  reg [16:0] layer5_out_31_16_0_sva_dfm;
  reg [16:0] layer3_out_0_16_0_sva_dfm;
  wire mux_tmp;
  wire nnet_relu_layer2_t_layer3_t_relu_config3_for_and_132_tmp;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_61_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_120_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_63_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_118_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_65_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_116_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_67_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_114_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_69_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_112_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_71_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_110_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_73_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_108_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_75_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_106_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_77_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_104_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_79_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_102_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_81_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_100_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_83_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_98_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_85_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_96_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_87_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_94_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_89_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_92_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_91_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_90_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_93_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_88_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_95_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_86_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_97_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_84_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_99_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_82_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_101_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_80_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_103_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_78_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_105_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_76_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_107_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_74_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_109_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_72_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_111_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_70_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_113_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_68_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_115_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_66_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_117_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_64_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_119_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_62_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_121_tmp_1;
  wire nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_60_tmp_1;
  wire [90:0] operator_91_21_false_AC_TRN_AC_WRAP_rshift_itm;
  wire [69:0] operator_71_0_false_AC_TRN_AC_WRAP_lshift_itm;
  wire [157:0] z_out;
  wire signed [159:0] nl_z_out;
  wire [17:0] z_out_1;
  wire [3:0] z_out_2;
  wire [4:0] nl_z_out_2;
  wire [9:0] z_out_3;
  wire [10:0] nl_z_out_3;
  wire [18:0] z_out_4;
  wire [19:0] nl_z_out_4;
  wire [6:0] z_out_5;
  wire [7:0] nl_z_out_5;
  wire [6:0] z_out_6;
  wire [7:0] nl_z_out_6;
  wire or_tmp_1201;
  wire [6:0] z_out_7;
  wire [7:0] nl_z_out_7;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_11_0_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_29_18_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_47_36_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_65_54_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_83_72_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_101_90_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_119_108_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_137_126_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_155_144_lpi_2;
  reg [11:0] OUTPUT_LOOP_io_read_output1_rsc_sdt_173_162_lpi_2;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_63_lpi_4;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_31_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_32_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_30_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_33_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_29_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_34_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_28_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_35_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_27_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_36_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_26_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_37_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_25_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_38_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_24_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_39_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_23_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_40_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_22_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_41_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_21_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_42_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_20_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_43_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_19_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_44_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_18_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_45_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_17_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_46_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_16_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_47_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_15_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_48_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_14_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_49_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_13_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_50_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_12_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_51_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_11_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_52_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_10_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_53_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_9_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_54_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_8_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_55_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_7_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_56_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_6_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_57_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_5_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_58_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_4_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_59_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_3_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_60_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_2_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_61_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_1_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_62_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1;
  reg [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1;
  reg [3:0] nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva;
  reg [18:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1;
  reg [66:0] ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1;
  reg [90:0] ac_math_ac_reciprocal_pwl_AC_TRN_71_51_false_AC_TRN_AC_WRAP_91_21_false_AC_TRN_AC_WRAP_output_temp_lpi_1_dfm;
  reg [17:0] MultLoop_1_mux_64_itm;
  reg [4:0] ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_itm;
  reg [2:0] ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_2_itm;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1_mx0w0;
  wire [17:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1_mx0w0;
  wire InitAccum_1_iacc_6_0_sva_5_0_mx0c0;
  wire InitAccum_1_iacc_6_0_sva_5_0_mx0c2;
  wire InitAccum_1_iacc_6_0_sva_5_0_mx0c3;
  wire [17:0] InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1;
  wire [17:0] InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3;
  wire [17:0] InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1;
  wire [3:0] MultLoop_2_im_3_0_sva_1_mx0w1;
  wire [4:0] nl_MultLoop_2_im_3_0_sva_1_mx0w1;
  wire [1151:0] tmp_lpi_3_dfm_1;
  wire [14:0] nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1;
  wire nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1;
  wire [11:0] CALC_SOFTMAX_LOOP_or_psp_sva_1;
  wire [5:0] ReuseLoop_ir_9_0_sva_mx0_tmp_9_4;
  wire [9:0] ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_1;
  wire [7:0] ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_1;
  wire [6:0] libraries_leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_1;
  wire MultLoop_and_rgt;
  wire output1_and_cse;
  wire [16:0] mux_149_cse;
  wire layer3_out_or_1_cse;
  wire [16:0] mux_150_cse;
  wire layer3_out_or_2_cse;
  wire [16:0] mux_151_cse;
  wire layer3_out_or_3_cse;
  wire [16:0] mux_152_cse;
  wire layer3_out_or_4_cse;
  wire [16:0] mux_153_cse;
  wire layer3_out_or_5_cse;
  wire [16:0] mux_154_cse;
  wire layer3_out_or_6_cse;
  wire [16:0] mux_155_cse;
  wire layer3_out_or_7_cse;
  wire [16:0] mux_156_cse;
  wire layer3_out_or_8_cse;
  wire [16:0] mux_157_cse;
  wire layer3_out_or_9_cse;
  wire [16:0] mux_158_cse;
  wire layer3_out_or_10_cse;
  wire [16:0] mux_159_cse;
  wire layer3_out_or_11_cse;
  wire [16:0] mux_160_cse;
  wire layer3_out_or_12_cse;
  wire [16:0] mux_161_cse;
  wire layer3_out_or_13_cse;
  wire [16:0] mux_162_cse;
  wire layer3_out_or_14_cse;
  wire [16:0] mux_163_cse;
  wire layer3_out_or_15_cse;
  wire [16:0] mux_164_cse;
  wire layer3_out_or_16_cse;
  wire [16:0] mux_165_cse;
  wire layer3_out_or_17_cse;
  wire [16:0] mux_166_cse;
  wire layer3_out_or_18_cse;
  wire [16:0] mux_167_cse;
  wire layer3_out_or_19_cse;
  wire [16:0] mux_168_cse;
  wire layer3_out_or_20_cse;
  wire [16:0] mux_169_cse;
  wire layer3_out_or_21_cse;
  wire [16:0] mux_170_cse;
  wire layer3_out_or_22_cse;
  wire [16:0] mux_171_cse;
  wire layer3_out_or_23_cse;
  wire [16:0] mux_172_cse;
  wire layer3_out_or_24_cse;
  wire [16:0] mux_173_cse;
  wire layer3_out_or_25_cse;
  wire [16:0] mux_174_cse;
  wire layer3_out_or_26_cse;
  wire [16:0] mux_175_cse;
  wire layer3_out_or_27_cse;
  wire [16:0] mux_176_cse;
  wire layer3_out_or_28_cse;
  wire [16:0] mux_177_cse;
  wire layer3_out_or_29_cse;
  wire [16:0] mux_178_cse;
  wire layer3_out_or_30_cse;
  wire [16:0] mux_179_cse;
  wire layer3_out_or_31_cse;
  wire [16:0] mux_180_cse;
  wire layer3_out_or_32_cse;
  wire [16:0] mux_181_cse;
  wire layer3_out_or_33_cse;
  wire [16:0] mux_182_cse;
  wire layer3_out_or_34_cse;
  wire [16:0] mux_183_cse;
  wire layer3_out_or_35_cse;
  wire [16:0] mux_184_cse;
  wire layer3_out_or_36_cse;
  wire [16:0] mux_185_cse;
  wire layer3_out_or_37_cse;
  wire [16:0] mux_186_cse;
  wire layer3_out_or_38_cse;
  wire [16:0] mux_187_cse;
  wire layer3_out_or_39_cse;
  wire [16:0] mux_188_cse;
  wire layer3_out_or_40_cse;
  wire [16:0] mux_189_cse;
  wire layer3_out_or_41_cse;
  wire [16:0] mux_190_cse;
  wire layer3_out_or_42_cse;
  wire [16:0] mux_191_cse;
  wire layer3_out_or_43_cse;
  wire [16:0] mux_192_cse;
  wire layer3_out_or_44_cse;
  wire [16:0] mux_193_cse;
  wire layer3_out_or_45_cse;
  wire [16:0] mux_194_cse;
  wire layer3_out_or_46_cse;
  wire [16:0] mux_195_cse;
  wire layer3_out_or_47_cse;
  wire [16:0] mux_196_cse;
  wire layer3_out_or_48_cse;
  wire [16:0] mux_197_cse;
  wire layer3_out_or_49_cse;
  wire [16:0] mux_198_cse;
  wire layer3_out_or_50_cse;
  wire [16:0] mux_199_cse;
  wire layer3_out_or_51_cse;
  wire [16:0] mux_200_cse;
  wire layer3_out_or_52_cse;
  wire [16:0] mux_201_cse;
  wire layer3_out_or_53_cse;
  wire [16:0] mux_202_cse;
  wire layer3_out_or_54_cse;
  wire [16:0] mux_203_cse;
  wire layer3_out_or_55_cse;
  wire [16:0] mux_204_cse;
  wire layer3_out_or_56_cse;
  wire [16:0] mux_205_cse;
  wire layer3_out_or_57_cse;
  wire [16:0] mux_206_cse;
  wire layer3_out_or_58_cse;
  wire [16:0] mux_207_cse;
  wire layer3_out_or_59_cse;
  wire [16:0] mux_208_cse;
  wire layer3_out_or_60_cse;
  wire [16:0] mux_209_cse;
  wire layer3_out_or_61_cse;
  wire [16:0] mux_210_cse;
  wire layer3_out_or_62_cse;
  wire or_tmp_1207;
  wire [5:0] InitAccum_1_iacc_and_2_rgt;
  wire [5:0] InitAccum_1_iacc_and_1_rgt;
  reg [3:0] reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp;
  reg [1:0] reg_ResultLoop_1_ires_6_0_sva_5_0_tmp;
  wire [16:0] layer3_out_layer3_out_mux_rgt;
  reg [1:0] MultLoop_1_if_1_acc_itm_5_4;
  reg [3:0] MultLoop_1_if_1_acc_itm_3_0;
  reg [6:0] layer3_out_63_16_0_lpi_2_dfm_16_10;
  reg [9:0] layer3_out_63_16_0_lpi_2_dfm_9_0;
  wire MultLoop_1_if_1_or_1_ssc;
  reg [1:0] reg_ReuseLoop_1_w_index_11_6_reg;
  reg [3:0] reg_ReuseLoop_1_w_index_11_6_1_reg;
  wire or_1917_ssc;
  wire and_3282_cse;
  wire and_3330_cse;
  wire nor_416_cse;
  wire nor_415_cse;
  wire nor_400_cse;
  wire and_4008_cse;
  wire or_2157_cse;
  wire nor_518_cse;
  wire and_3921_cse;
  wire and_3931_cse;
  wire nor_417_cse;
  wire nand_60_cse;
  wire nand_62_cse;
  wire nor_517_cse;
  wire nand_53_cse;
  wire mux_280_cse;
  wire nand_42_cse;
  wire mux_274_cse;
  wire or_dcpl_1103;
  wire or_2579_tmp;
  wire or_2583_tmp;
  wire [18:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1;
  wire InitAccum_2_acc_1_itm_3;
  wire ReuseLoop_acc_itm_6;
  wire MultLoop_or_3_cse;

  wire[0:0] and_524_nl;
  wire[0:0] and_530_nl;
  wire[0:0] and_536_nl;
  wire[0:0] and_542_nl;
  wire[0:0] and_548_nl;
  wire[0:0] and_554_nl;
  wire[0:0] and_560_nl;
  wire[0:0] and_566_nl;
  wire[0:0] and_572_nl;
  wire[0:0] and_578_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_128_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_140_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_136_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_148_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_144_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_156_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_152_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_164_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_160_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_172_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_168_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_180_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_176_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_188_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_184_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_196_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_192_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_204_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_200_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_212_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_208_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_220_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_216_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_228_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_224_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_236_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_232_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_244_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_240_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_252_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_248_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_260_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_256_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_268_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_264_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_276_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_272_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_284_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_280_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_292_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_288_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_300_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_296_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_308_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_304_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_316_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_312_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_324_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_320_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_332_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_328_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_340_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_336_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_348_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_344_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_356_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_352_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_364_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_360_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_372_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_368_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_and_376_nl;
  wire[5:0] ReuseLoop_in_index_mux1h_nl;
  wire[2:0] ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_1_nl;
  wire[0:0] InitAccum_1_iacc_not_nl;
  wire[0:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_62_nl;
  wire[0:0] and_1844_nl;
  wire[0:0] and_1846_nl;
  wire[0:0] nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_65_nl;
  wire[0:0] or_2584_nl;
  wire[0:0] mux_273_nl;
  wire[0:0] nor_520_nl;
  wire[0:0] or_2147_nl;
  wire[0:0] nor_523_nl;
  wire[0:0] and_1884_nl;
  wire[0:0] and_1886_nl;
  wire[0:0] and_4027_nl;
  wire[0:0] and_1899_nl;
  wire[0:0] and_1901_nl;
  wire[0:0] or_2586_nl;
  wire[0:0] and_1917_nl;
  wire[0:0] and_1919_nl;
  wire[0:0] and_4032_nl;
  wire[0:0] and_1931_nl;
  wire[0:0] and_1933_nl;
  wire[0:0] and_4034_nl;
  wire[0:0] and_1945_nl;
  wire[0:0] and_1947_nl;
  wire[0:0] and_4036_nl;
  wire[0:0] and_1959_nl;
  wire[0:0] and_1961_nl;
  wire[0:0] and_4038_nl;
  wire[0:0] and_1973_nl;
  wire[0:0] and_1975_nl;
  wire[0:0] and_4040_nl;
  wire[0:0] and_1987_nl;
  wire[0:0] and_1989_nl;
  wire[0:0] and_4042_nl;
  wire[0:0] and_2001_nl;
  wire[0:0] and_2003_nl;
  wire[0:0] and_4044_nl;
  wire[0:0] and_2015_nl;
  wire[0:0] and_2017_nl;
  wire[0:0] and_4046_nl;
  wire[0:0] and_2029_nl;
  wire[0:0] and_2031_nl;
  wire[0:0] and_4048_nl;
  wire[0:0] and_2043_nl;
  wire[0:0] and_2045_nl;
  wire[0:0] and_4050_nl;
  wire[0:0] and_2057_nl;
  wire[0:0] and_2059_nl;
  wire[0:0] and_4052_nl;
  wire[0:0] and_2071_nl;
  wire[0:0] and_2073_nl;
  wire[0:0] and_4054_nl;
  wire[0:0] and_2085_nl;
  wire[0:0] and_2087_nl;
  wire[0:0] and_4056_nl;
  wire[0:0] and_2099_nl;
  wire[0:0] and_2101_nl;
  wire[0:0] and_4058_nl;
  wire[0:0] and_2113_nl;
  wire[0:0] and_2115_nl;
  wire[0:0] and_4060_nl;
  wire[0:0] and_2127_nl;
  wire[0:0] and_2129_nl;
  wire[0:0] and_4062_nl;
  wire[0:0] and_2141_nl;
  wire[0:0] and_2143_nl;
  wire[0:0] and_4064_nl;
  wire[0:0] and_2156_nl;
  wire[0:0] and_2158_nl;
  wire[0:0] or_2604_nl;
  wire[0:0] and_2174_nl;
  wire[0:0] and_2176_nl;
  wire[0:0] and_4069_nl;
  wire[0:0] and_2189_nl;
  wire[0:0] and_2191_nl;
  wire[0:0] or_2606_nl;
  wire[0:0] and_2207_nl;
  wire[0:0] and_2209_nl;
  wire[0:0] and_4074_nl;
  wire[0:0] and_2222_nl;
  wire[0:0] and_2224_nl;
  wire[0:0] or_2608_nl;
  wire[0:0] mux_276_nl;
  wire[0:0] or_2280_nl;
  wire[0:0] and_2240_nl;
  wire[0:0] and_2242_nl;
  wire[0:0] and_4079_nl;
  wire[0:0] and_2255_nl;
  wire[0:0] and_2257_nl;
  wire[0:0] or_2610_nl;
  wire[0:0] and_2273_nl;
  wire[0:0] and_2275_nl;
  wire[0:0] and_4084_nl;
  wire[0:0] and_2288_nl;
  wire[0:0] and_2290_nl;
  wire[0:0] or_2612_nl;
  wire[0:0] and_2306_nl;
  wire[0:0] and_2308_nl;
  wire[0:0] and_4089_nl;
  wire[0:0] and_2321_nl;
  wire[0:0] and_2323_nl;
  wire[0:0] or_2614_nl;
  wire[0:0] and_2339_nl;
  wire[0:0] and_2341_nl;
  wire[0:0] and_4094_nl;
  wire[0:0] nor_466_nl;
  wire[0:0] and_2354_nl;
  wire[0:0] and_2356_nl;
  wire[0:0] or_2616_nl;
  wire[0:0] and_2372_nl;
  wire[0:0] and_2374_nl;
  wire[0:0] and_4099_nl;
  wire[0:0] and_2387_nl;
  wire[0:0] and_2389_nl;
  wire[0:0] or_2618_nl;
  wire[0:0] and_2405_nl;
  wire[0:0] and_2407_nl;
  wire[0:0] and_4104_nl;
  wire[0:0] and_2419_nl;
  wire[0:0] and_2421_nl;
  wire[0:0] and_4106_nl;
  wire[0:0] and_2433_nl;
  wire[0:0] and_2435_nl;
  wire[0:0] and_4108_nl;
  wire[0:0] and_2447_nl;
  wire[0:0] and_2449_nl;
  wire[0:0] and_4110_nl;
  wire[0:0] and_2461_nl;
  wire[0:0] and_2463_nl;
  wire[0:0] and_4112_nl;
  wire[0:0] and_2475_nl;
  wire[0:0] and_2477_nl;
  wire[0:0] and_4114_nl;
  wire[0:0] and_2489_nl;
  wire[0:0] and_2491_nl;
  wire[0:0] and_4116_nl;
  wire[0:0] and_2503_nl;
  wire[0:0] and_2505_nl;
  wire[0:0] and_4118_nl;
  wire[0:0] and_2517_nl;
  wire[0:0] and_2519_nl;
  wire[0:0] and_4120_nl;
  wire[0:0] and_2531_nl;
  wire[0:0] and_2533_nl;
  wire[0:0] and_4122_nl;
  wire[0:0] and_2545_nl;
  wire[0:0] and_2547_nl;
  wire[0:0] and_4124_nl;
  wire[0:0] and_2559_nl;
  wire[0:0] and_2561_nl;
  wire[0:0] and_4126_nl;
  wire[0:0] and_2573_nl;
  wire[0:0] and_2575_nl;
  wire[0:0] and_4128_nl;
  wire[0:0] and_2587_nl;
  wire[0:0] and_2589_nl;
  wire[0:0] and_4130_nl;
  wire[0:0] and_2601_nl;
  wire[0:0] and_2603_nl;
  wire[0:0] and_4132_nl;
  wire[0:0] and_2615_nl;
  wire[0:0] and_2617_nl;
  wire[0:0] and_4134_nl;
  wire[0:0] and_2629_nl;
  wire[0:0] and_2631_nl;
  wire[0:0] and_4136_nl;
  wire[0:0] and_2643_nl;
  wire[0:0] and_2645_nl;
  wire[0:0] and_4138_nl;
  wire[0:0] and_2657_nl;
  wire[0:0] and_2659_nl;
  wire[0:0] and_4140_nl;
  wire[0:0] and_2671_nl;
  wire[0:0] and_2673_nl;
  wire[0:0] and_4142_nl;
  wire[0:0] and_2685_nl;
  wire[0:0] and_2687_nl;
  wire[0:0] and_4144_nl;
  wire[0:0] and_2699_nl;
  wire[0:0] and_2701_nl;
  wire[0:0] and_4146_nl;
  wire[0:0] and_2713_nl;
  wire[0:0] and_2715_nl;
  wire[0:0] and_4148_nl;
  wire[0:0] and_2727_nl;
  wire[0:0] and_2729_nl;
  wire[0:0] and_4150_nl;
  wire[0:0] and_2741_nl;
  wire[0:0] and_2743_nl;
  wire[0:0] and_4152_nl;
  wire[0:0] and_2755_nl;
  wire[0:0] and_2757_nl;
  wire[0:0] and_4154_nl;
  wire[0:0] and_2769_nl;
  wire[0:0] and_2771_nl;
  wire[0:0] and_4156_nl;
  wire[0:0] and_2783_nl;
  wire[0:0] and_2785_nl;
  wire[0:0] and_4158_nl;
  wire[3:0] ReuseLoop_ir_ReuseLoop_ir_and_2_nl;
  wire[3:0] InitAccum_1_iacc_mux1h_3_nl;
  wire[0:0] InitAccum_1_iacc_and_5_nl;
  wire[0:0] nor_521_nl;
  wire[0:0] mux_282_nl;
  wire[0:0] or_2649_nl;
  wire[0:0] nand_33_nl;
  wire[1:0] InitAccum_1_iacc_InitAccum_1_iacc_mux_nl;
  wire[0:0] not_1748_nl;
  wire[5:0] InitAccum_1_iacc_mux1h_4_nl;
  wire[3:0] ReuseLoop_2_out_index_ReuseLoop_2_out_index_and_nl;
  wire[0:0] not_286_nl;
  wire[0:0] or_1923_nl;
  wire[0:0] not_629_nl;
  wire[5:0] InitAccum_1_iacc_mux1h_5_nl;
  wire[3:0] OUTPUT_LOOP_i_asn_ReuseLoop_1_w_index_11_6_sva_1_3_ReuseLoop_2_w_index_and_nl;
  wire[0:0] not_285_nl;
  wire[0:0] nor_nl;
  wire[5:0] ReuseLoop_2_in_index_asn_MultLoop_1_im_6_0_sva_2_5_ReuseLoop_in_index_and_nl;
  wire[5:0] ReuseLoop_in_index_mux1h_1_nl;
  wire[0:0] or_1973_nl;
  wire[0:0] or_1936_nl;
  wire[0:0] MultLoop_1_im_and_nl;
  wire[0:0] MultLoop_1_im_and_1_nl;
  wire[0:0] MultLoop_1_im_and_2_nl;
  wire[0:0] MultLoop_1_im_and_3_nl;
  wire[9:0] ReuseLoop_ir_ReuseLoop_ir_and_1_nl;
  wire[0:0] or_1972_nl;
  wire[16:0] nnet_relu_layer4_t_layer5_t_relu_config5_for_nnet_relu_layer4_t_layer5_t_relu_config5_for_and_nl;
  wire[17:0] MultLoop_mux_65_nl;
  wire[16:0] MultLoop_1_MultLoop_1_mux_nl;
  wire[0:0] nor_27_nl;
  wire[0:0] nand_29_nl;
  wire[0:0] or_2036_nl;
  wire[0:0] nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nand_63_nl;
  wire[0:0] layer3_out_and_62_nl;
  wire[0:0] layer3_out_and_63_nl;
  wire[0:0] not_nl;
  wire[0:0] ResultLoop_2_and_nl;
  wire[0:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_nor_nl;
  wire[0:0] CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_nl;
  wire[19:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl;
  wire[20:0] nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl;
  wire[21:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl;
  wire[22:0] nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl;
  wire[17:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl;
  wire[18:0] nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl;
  wire[22:0] ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl;
  wire[24:0] nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl;
  wire[14:0] nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_2_nl;
  wire[0:0] nnet_softmax_layer6_t_result_t_softmax_config7_for_and_nl;
  wire[0:0] CALC_SOFTMAX_LOOP_or_1_nl;
  wire[3:0] InitAccum_2_acc_1_nl;
  wire[4:0] nl_InitAccum_2_acc_1_nl;
  wire[6:0] ReuseLoop_acc_nl;
  wire[7:0] nl_ReuseLoop_acc_nl;
  wire[0:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_2_nl;
  wire[49:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl;
  wire[49:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_4_nl;
  wire[49:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux_1_nl;
  wire[49:0] CALC_SOFTMAX_LOOP_mux_3_nl;
  wire[0:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_nor_3_nl;
  wire[6:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl;
  wire[6:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_5_nl;
  wire[6:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_3_nl;
  wire[6:0] MultLoop_2_MultLoop_2_mux_2_nl;
  wire[6:0] CALC_SOFTMAX_LOOP_mux_4_nl;
  wire[0:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_not_1_nl;
  wire[9:0] mux1h_189_nl;
  wire[9:0] MultLoop_2_MultLoop_2_mux_3_nl;
  wire[9:0] CALC_SOFTMAX_LOOP_mux_5_nl;
  wire[0:0] and_4241_nl;
  wire[0:0] and_4242_nl;
  wire[0:0] and_4243_nl;
  wire[91:0] nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_4_nl;
  wire[17:0] MultLoop_2_mux_14_nl;
  wire[9:0] ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_qif_mux_2_nl;
  wire[17:0] MultLoop_mux1h_4_nl;
  wire[0:0] MultLoop_or_6_nl;
  wire[8:0] MultLoop_and_63_nl;
  wire[8:0] MultLoop_mux1h_5_nl;
  wire[0:0] MultLoop_or_7_nl;
  wire[0:0] MultLoop_not_37_nl;
  wire[0:0] MultLoop_and_64_nl;
  wire[0:0] MultLoop_mux1h_6_nl;
  wire[7:0] MultLoop_mux1h_7_nl;
  wire[1:0] MultLoop_1_if_1_MultLoop_1_if_1_MultLoop_1_if_1_or_1_nl;
  wire[3:0] MultLoop_1_if_1_MultLoop_1_if_1_mux_1_nl;
  wire[2:0] MultLoop_1_if_1_mux1h_9_nl;
  wire[0:0] MultLoop_1_if_1_or_2_nl;
  wire[1:0] MultLoop_1_MultLoop_1_or_1_nl;
  wire[3:0] MultLoop_1_mux_2_nl;
  wire[2:0] MultLoop_1_mux_3_nl;
  wire[5:0] MultLoop_1_mux1h_3_nl;
  wire[0:0] or_2650_nl;
  wire[2:0] MultLoop_1_MultLoop_1_mux_2_nl;
  wire[3:0] MultLoop_2_mux_15_nl;

  // Interconnect Declarations for Component Instantiations 
  wire [2:0] nl_U_ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_rg_I_1;
  assign nl_U_ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_rg_I_1 = operator_71_0_false_AC_TRN_AC_WRAP_lshift_itm[69:67];
  wire [70:0] nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_a;
  assign nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_a = {(z_out_4[10:0]) ,
      (z_out[9:0]) , 50'b00000000000000000000000000000000000000000000000000};
  wire [7:0] nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_s;
  assign nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_s = {(z_out_3[5:0]) , (~
      (libraries_leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_1[1:0]))};
  wire [20:0] nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_a;
  assign nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_a = {(z_out_4[10:0]) ,
      (z_out[9:0])};
  wire [6:0] nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_s;
  assign nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_s = ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva[18:12];
  wire [2:0] nl_U_ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_rg_I_1;
  assign nl_U_ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_rg_I_1 = operator_71_0_false_AC_TRN_AC_WRAP_lshift_itm[69:67];
  wire [69:0] nl_operator_71_0_false_AC_TRN_AC_WRAP_lshift_rg_a;
  assign nl_operator_71_0_false_AC_TRN_AC_WRAP_lshift_rg_a = ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva[69:0];
  wire [179:0] nl_mnist_mlp_core_output1_rsci_inst_output1_rsci_idat;
  assign nl_mnist_mlp_core_output1_rsci_inst_output1_rsci_idat = {6'b000000 , output1_rsci_idat_173_162
      , 6'b000000 , output1_rsci_idat_155_144 , 6'b000000 , output1_rsci_idat_137_126
      , 6'b000000 , output1_rsci_idat_119_108 , 6'b000000 , output1_rsci_idat_101_90
      , 6'b000000 , output1_rsci_idat_83_72 , 6'b000000 , output1_rsci_idat_65_54
      , 6'b000000 , output1_rsci_idat_47_36 , 6'b000000 , output1_rsci_idat_29_18
      , 6'b000000 , output1_rsci_idat_11_0};
  wire [1:0] nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_CE2_d_core_psct;
  assign nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_CE2_d_core_psct = {1'b0 , (fsm_output[3])};
  wire [31:0] nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_A2_d_core;
  assign nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_A2_d_core = {16'b0000000000000000
      , reg_ReuseLoop_1_w_index_11_6_reg , reg_ReuseLoop_1_w_index_11_6_1_reg , InitAccum_1_iacc_6_0_sva_5_0
      , (layer3_out_63_16_0_lpi_2_dfm_9_0[3:0])};
  wire [1:0] nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  assign nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct
      = {1'b0 , (fsm_output[3])};
  wire [0:0] nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_oswt_pff;
  assign nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_oswt_pff = fsm_output[3];
  wire [1:0] nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_CE2_d_core_psct;
  assign nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_CE2_d_core_psct = {1'b0 , (fsm_output[7])};
  wire [23:0] nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_A2_d_core;
  assign nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_A2_d_core = {12'b000000000000 ,
      reg_ReuseLoop_1_w_index_11_6_reg , reg_ReuseLoop_1_w_index_11_6_1_reg , InitAccum_1_iacc_6_0_sva_5_0};
  wire [1:0] nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct;
  assign nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct
      = {1'b0 , (fsm_output[7])};
  wire [0:0] nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_oswt_pff;
  assign nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_oswt_pff = fsm_output[7];
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_InitAccum_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_InitAccum_C_0_tr0 = z_out_7[6];
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_MultLoop_C_1_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_MultLoop_C_1_tr0 = MultLoop_1_im_6_0_sva_1[6];
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_C_0_tr0 = ~ ReuseLoop_acc_itm_6;
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_MultLoop_1_C_1_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_MultLoop_1_C_1_tr0 = MultLoop_1_im_6_0_sva_1[6];
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_1_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_1_C_0_tr0 = z_out_7[6];
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_InitAccum_2_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_InitAccum_2_C_0_tr0 = ~ InitAccum_2_acc_1_itm_3;
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_MultLoop_2_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_MultLoop_2_C_0_tr0 = ~ (z_out_7[3]);
  wire [0:0] nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_2_C_0_tr0;
  assign nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_2_C_0_tr0 = z_out_7[6];
  ccs_in_v1 #(.rscid(32'sd27),
  .width(32'sd1152)) w6_rsc_0_0_i (
      .dat(w6_rsc_0_0_dat),
      .idat(w6_rsc_0_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd28),
  .width(32'sd1152)) w6_rsc_1_0_i (
      .dat(w6_rsc_1_0_dat),
      .idat(w6_rsc_1_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd29),
  .width(32'sd1152)) w6_rsc_2_0_i (
      .dat(w6_rsc_2_0_dat),
      .idat(w6_rsc_2_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd30),
  .width(32'sd1152)) w6_rsc_3_0_i (
      .dat(w6_rsc_3_0_dat),
      .idat(w6_rsc_3_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd31),
  .width(32'sd1152)) w6_rsc_4_0_i (
      .dat(w6_rsc_4_0_dat),
      .idat(w6_rsc_4_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd32),
  .width(32'sd1152)) w6_rsc_5_0_i (
      .dat(w6_rsc_5_0_dat),
      .idat(w6_rsc_5_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd33),
  .width(32'sd1152)) w6_rsc_6_0_i (
      .dat(w6_rsc_6_0_dat),
      .idat(w6_rsc_6_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd34),
  .width(32'sd1152)) w6_rsc_7_0_i (
      .dat(w6_rsc_7_0_dat),
      .idat(w6_rsc_7_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd35),
  .width(32'sd1152)) w6_rsc_8_0_i (
      .dat(w6_rsc_8_0_dat),
      .idat(w6_rsc_8_0_i_idat)
    );
  ccs_in_v1 #(.rscid(32'sd36),
  .width(32'sd1152)) w6_rsc_9_0_i (
      .dat(w6_rsc_9_0_dat),
      .idat(w6_rsc_9_0_i_idat)
    );
  ROM_1i3_1o10_d515ec42b6831339071874e16a9b8d3ab1  U_ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_rg
      (
      .I_1(nl_U_ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_rg_I_1[2:0]),
      .O_1(ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_1)
    );
  mgc_shift_br_v5 #(.width_a(32'sd71),
  .signd_a(32'sd0),
  .width_s(32'sd8),
  .width_z(32'sd91)) operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg (
      .a(nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_a[70:0]),
      .s(nl_operator_91_21_false_AC_TRN_AC_WRAP_rshift_rg_s[7:0]),
      .z(operator_91_21_false_AC_TRN_AC_WRAP_rshift_itm)
    );
  mgc_shift_bl_v5 #(.width_a(32'sd21),
  .signd_a(32'sd0),
  .width_s(32'sd7),
  .width_z(32'sd67)) operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg (
      .a(nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_a[20:0]),
      .s(nl_operator_67_47_false_AC_TRN_AC_WRAP_lshift_rg_s[6:0]),
      .z(operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1)
    );
  ROM_1i3_1o8_75ee39ff4c2e67ce55133b7c869c3b33b0  U_ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_rg
      (
      .I_1(nl_U_ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_rg_I_1[2:0]),
      .O_1(ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_1)
    );
  mgc_shift_l_v5 #(.width_a(32'sd70),
  .signd_a(32'sd0),
  .width_s(32'sd7),
  .width_z(32'sd70)) operator_71_0_false_AC_TRN_AC_WRAP_lshift_rg (
      .a(nl_operator_71_0_false_AC_TRN_AC_WRAP_lshift_rg_a[69:0]),
      .s(libraries_leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_1),
      .z(operator_71_0_false_AC_TRN_AC_WRAP_lshift_itm)
    );
  leading_sign_71_0  leading_sign_71_0_rg (
      .mantissa(ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva),
      .rtn(libraries_leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_1)
    );
  mnist_mlp_core_input1_rsci mnist_mlp_core_input1_rsci_inst (
      .clk(clk),
      .rst(rst),
      .input1_rsc_dat(input1_rsc_dat),
      .input1_rsc_vld(input1_rsc_vld),
      .input1_rsc_rdy(input1_rsc_rdy),
      .core_wen(core_wen),
      .input1_rsci_oswt(reg_input1_rsci_irdy_core_psct_cse),
      .input1_rsci_wen_comp(input1_rsci_wen_comp),
      .input1_rsci_idat_mxwt(input1_rsci_idat_mxwt)
    );
  mnist_mlp_core_output1_rsci mnist_mlp_core_output1_rsci_inst (
      .clk(clk),
      .rst(rst),
      .output1_rsc_dat(output1_rsc_dat),
      .output1_rsc_vld(output1_rsc_vld),
      .output1_rsc_rdy(output1_rsc_rdy),
      .core_wen(core_wen),
      .output1_rsci_oswt(reg_output1_rsci_ivld_core_psct_cse),
      .output1_rsci_wen_comp(output1_rsci_wen_comp),
      .output1_rsci_idat(nl_mnist_mlp_core_output1_rsci_inst_output1_rsci_idat[179:0])
    );
  mnist_mlp_core_const_size_in_1_rsci mnist_mlp_core_const_size_in_1_rsci_inst (
      .const_size_in_1_rsc_dat(const_size_in_1_rsc_dat),
      .const_size_in_1_rsc_vld(const_size_in_1_rsc_vld),
      .core_wten(core_wten),
      .const_size_in_1_rsci_iswt0(reg_const_size_out_1_rsci_ivld_core_psct_cse)
    );
  mnist_mlp_core_const_size_out_1_rsci mnist_mlp_core_const_size_out_1_rsci_inst
      (
      .const_size_out_1_rsc_dat(const_size_out_1_rsc_dat),
      .const_size_out_1_rsc_vld(const_size_out_1_rsc_vld),
      .core_wten(core_wten),
      .const_size_out_1_rsci_iswt0(reg_const_size_out_1_rsci_ivld_core_psct_cse)
    );
  mnist_mlp_core_w2_rsci_1 mnist_mlp_core_w2_rsci_1_inst (
      .clk(clk),
      .rst(rst),
      .w2_rsci_CE2_d(w2_rsci_CE2_d_reg),
      .w2_rsci_A2_d(w2_rsci_A2_d_reg),
      .w2_rsci_Q2_d(w2_rsci_Q2_d),
      .w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d(w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg),
      .core_wen(core_wen),
      .core_wten(core_wten),
      .w2_rsci_oswt(reg_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse),
      .w2_rsci_CE2_d_core_psct(nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_CE2_d_core_psct[1:0]),
      .w2_rsci_A2_d_core(nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_A2_d_core[31:0]),
      .w2_rsci_Q2_d_mxwt(w2_rsci_Q2_d_mxwt),
      .w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct(nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[1:0]),
      .w2_rsci_oswt_pff(nl_mnist_mlp_core_w2_rsci_1_inst_w2_rsci_oswt_pff[0:0])
    );
  mnist_mlp_core_b2_rsci mnist_mlp_core_b2_rsci_inst (
      .b2_rsc_dat(b2_rsc_dat),
      .b2_rsc_vld(b2_rsc_vld),
      .b2_rsci_oswt(b2_rsci_oswt),
      .b2_rsci_wen_comp(b2_rsci_wen_comp),
      .b2_rsci_idat_mxwt(b2_rsci_idat_mxwt)
    );
  mnist_mlp_core_w4_rsci_1 mnist_mlp_core_w4_rsci_1_inst (
      .clk(clk),
      .rst(rst),
      .w4_rsci_CE2_d(w4_rsci_CE2_d_reg),
      .w4_rsci_A2_d(w4_rsci_A2_d_reg),
      .w4_rsci_Q2_d(w4_rsci_Q2_d),
      .w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d(w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg),
      .core_wen(core_wen),
      .core_wten(core_wten),
      .w4_rsci_oswt(reg_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse),
      .w4_rsci_CE2_d_core_psct(nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_CE2_d_core_psct[1:0]),
      .w4_rsci_A2_d_core(nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_A2_d_core[23:0]),
      .w4_rsci_Q2_d_mxwt(w4_rsci_Q2_d_mxwt),
      .w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct(nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct[1:0]),
      .w4_rsci_oswt_pff(nl_mnist_mlp_core_w4_rsci_1_inst_w4_rsci_oswt_pff[0:0])
    );
  mnist_mlp_core_b4_rsci mnist_mlp_core_b4_rsci_inst (
      .b4_rsc_dat(b4_rsc_dat),
      .b4_rsc_vld(b4_rsc_vld),
      .b4_rsci_oswt(b4_rsci_oswt),
      .b4_rsci_wen_comp(b4_rsci_wen_comp),
      .b4_rsci_idat_mxwt(b4_rsci_idat_mxwt)
    );
  mnist_mlp_core_b6_rsci mnist_mlp_core_b6_rsci_inst (
      .b6_rsc_dat(b6_rsc_dat),
      .b6_rsc_vld(b6_rsc_vld),
      .b6_rsci_oswt(b6_rsci_oswt),
      .b6_rsci_wen_comp(b6_rsci_wen_comp),
      .b6_rsci_idat_mxwt(b6_rsci_idat_mxwt)
    );
  mnist_mlp_core_staller mnist_mlp_core_staller_inst (
      .clk(clk),
      .rst(rst),
      .core_wen(core_wen),
      .core_wten(core_wten),
      .input1_rsci_wen_comp(input1_rsci_wen_comp),
      .output1_rsci_wen_comp(output1_rsci_wen_comp),
      .b2_rsci_wen_comp(b2_rsci_wen_comp),
      .b4_rsci_wen_comp(b4_rsci_wen_comp),
      .b6_rsci_wen_comp(b6_rsci_wen_comp)
    );
  mnist_mlp_core_core_fsm mnist_mlp_core_core_fsm_inst (
      .clk(clk),
      .rst(rst),
      .core_wen(core_wen),
      .fsm_output(fsm_output),
      .InitAccum_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_InitAccum_C_0_tr0[0:0]),
      .MultLoop_C_1_tr0(nl_mnist_mlp_core_core_fsm_inst_MultLoop_C_1_tr0[0:0]),
      .ReuseLoop_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_C_0_tr0[0:0]),
      .ResultLoop_C_0_tr0(ResultLoop_and_1_tmp),
      .MultLoop_1_C_1_tr0(nl_mnist_mlp_core_core_fsm_inst_MultLoop_1_C_1_tr0[0:0]),
      .ReuseLoop_1_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_1_C_0_tr0[0:0]),
      .ResultLoop_1_C_0_tr0(ResultLoop_1_and_tmp),
      .InitAccum_2_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_InitAccum_2_C_0_tr0[0:0]),
      .MultLoop_2_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_MultLoop_2_C_0_tr0[0:0]),
      .ReuseLoop_2_C_0_tr0(nl_mnist_mlp_core_core_fsm_inst_ReuseLoop_2_C_0_tr0[0:0]),
      .ResultLoop_2_C_1_tr0(CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm),
      .CALC_SOFTMAX_LOOP_C_1_tr0(CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm)
    );
  assign output1_and_cse = core_wen & (fsm_output[17]);
  assign layer3_out_and_cse = core_wen & (fsm_output[6]);
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_128_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_61_tmp_1
      & (z_out_4[18]);
  assign mux_149_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_66_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_66_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_128_nl);
  assign layer3_out_or_1_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_66_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_61_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_140_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_120_tmp_1
      & (z_out_4[18]);
  assign mux_150_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_68_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_68_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_140_nl);
  assign layer3_out_or_2_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_68_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_120_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_136_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_63_tmp_1
      & (z_out_4[18]);
  assign mux_151_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_70_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_70_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_136_nl);
  assign layer3_out_or_3_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_70_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_63_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_148_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_118_tmp_1
      & (z_out_4[18]);
  assign mux_152_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_72_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_72_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_148_nl);
  assign layer3_out_or_4_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_72_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_118_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_144_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_65_tmp_1
      & (z_out_4[18]);
  assign mux_153_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_74_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_74_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_144_nl);
  assign layer3_out_or_5_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_74_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_65_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_156_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_116_tmp_1
      & (z_out_4[18]);
  assign mux_154_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_76_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_76_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_156_nl);
  assign layer3_out_or_6_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_76_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_116_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_152_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_67_tmp_1
      & (z_out_4[18]);
  assign mux_155_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_78_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_78_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_152_nl);
  assign layer3_out_or_7_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_78_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_67_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_164_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_114_tmp_1
      & (z_out_4[18]);
  assign mux_156_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_80_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_80_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_164_nl);
  assign layer3_out_or_8_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_80_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_114_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_160_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_69_tmp_1
      & (z_out_4[18]);
  assign mux_157_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_82_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_82_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_160_nl);
  assign layer3_out_or_9_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_82_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_69_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_172_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_112_tmp_1
      & (z_out_4[18]);
  assign mux_158_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_84_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_84_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_172_nl);
  assign layer3_out_or_10_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_84_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_112_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_168_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_71_tmp_1
      & (z_out_4[18]);
  assign mux_159_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_86_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_86_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_168_nl);
  assign layer3_out_or_11_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_86_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_71_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_180_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_110_tmp_1
      & (z_out_4[18]);
  assign mux_160_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_88_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_88_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_180_nl);
  assign layer3_out_or_12_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_88_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_110_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_176_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_73_tmp_1
      & (z_out_4[18]);
  assign mux_161_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_90_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_90_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_176_nl);
  assign layer3_out_or_13_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_90_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_73_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_188_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_108_tmp_1
      & (z_out_4[18]);
  assign mux_162_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_92_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_92_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_188_nl);
  assign layer3_out_or_14_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_92_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_108_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_184_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_75_tmp_1
      & (z_out_4[18]);
  assign mux_163_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_94_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_94_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_184_nl);
  assign layer3_out_or_15_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_94_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_75_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_196_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_106_tmp_1
      & (z_out_4[18]);
  assign mux_164_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_96_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_96_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_196_nl);
  assign layer3_out_or_16_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_96_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_106_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_192_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_77_tmp_1
      & (z_out_4[18]);
  assign mux_165_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_98_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_98_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_192_nl);
  assign layer3_out_or_17_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_98_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_77_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_204_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_104_tmp_1
      & (z_out_4[18]);
  assign mux_166_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_100_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_100_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_204_nl);
  assign layer3_out_or_18_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_100_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_104_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_200_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_79_tmp_1
      & (z_out_4[18]);
  assign mux_167_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_102_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_102_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_200_nl);
  assign layer3_out_or_19_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_102_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_79_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_212_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_102_tmp_1
      & (z_out_4[18]);
  assign mux_168_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_104_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_104_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_212_nl);
  assign layer3_out_or_20_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_104_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_102_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_208_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_81_tmp_1
      & (z_out_4[18]);
  assign mux_169_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_106_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_106_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_208_nl);
  assign layer3_out_or_21_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_106_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_81_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_220_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_100_tmp_1
      & (z_out_4[18]);
  assign mux_170_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_108_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_108_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_220_nl);
  assign layer3_out_or_22_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_108_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_100_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_216_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_83_tmp_1
      & (z_out_4[18]);
  assign mux_171_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_110_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_110_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_216_nl);
  assign layer3_out_or_23_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_110_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_83_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_228_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_98_tmp_1
      & (z_out_4[18]);
  assign mux_172_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_112_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_112_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_228_nl);
  assign layer3_out_or_24_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_112_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_98_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_224_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_85_tmp_1
      & (z_out_4[18]);
  assign mux_173_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_114_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_114_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_224_nl);
  assign layer3_out_or_25_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_114_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_85_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_236_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_96_tmp_1
      & (z_out_4[18]);
  assign mux_174_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_116_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_116_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_236_nl);
  assign layer3_out_or_26_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_116_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_96_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_232_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_87_tmp_1
      & (z_out_4[18]);
  assign mux_175_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_118_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_118_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_232_nl);
  assign layer3_out_or_27_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_118_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_87_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_244_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_94_tmp_1
      & (z_out_4[18]);
  assign mux_176_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_120_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_120_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_244_nl);
  assign layer3_out_or_28_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_120_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_94_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_240_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_89_tmp_1
      & (z_out_4[18]);
  assign mux_177_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_122_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_122_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_240_nl);
  assign layer3_out_or_29_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_122_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_89_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_252_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_92_tmp_1
      & (z_out_4[18]);
  assign mux_178_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_124_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_124_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_252_nl);
  assign layer3_out_or_30_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_124_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_92_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_248_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_91_tmp_1
      & (z_out_4[18]);
  assign mux_179_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_126_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_126_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_248_nl);
  assign layer3_out_or_31_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_126_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_91_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_260_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_90_tmp_1
      & (z_out_4[18]);
  assign mux_180_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_132_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_132_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_260_nl);
  assign layer3_out_or_32_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_132_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_90_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_256_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_93_tmp_1
      & (z_out_4[18]);
  assign mux_181_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_130_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_130_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_256_nl);
  assign layer3_out_or_33_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_130_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_93_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_268_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_88_tmp_1
      & (z_out_4[18]);
  assign mux_182_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_136_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_136_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_268_nl);
  assign layer3_out_or_34_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_136_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_88_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_264_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_95_tmp_1
      & (z_out_4[18]);
  assign mux_183_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_134_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_134_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_264_nl);
  assign layer3_out_or_35_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_134_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_95_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_276_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_86_tmp_1
      & (z_out_4[18]);
  assign mux_184_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_140_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_140_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_276_nl);
  assign layer3_out_or_36_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_140_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_86_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_272_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_97_tmp_1
      & (z_out_4[18]);
  assign mux_185_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_138_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_138_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_272_nl);
  assign layer3_out_or_37_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_138_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_97_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_284_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_84_tmp_1
      & (z_out_4[18]);
  assign mux_186_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_144_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_144_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_284_nl);
  assign layer3_out_or_38_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_144_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_84_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_280_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_99_tmp_1
      & (z_out_4[18]);
  assign mux_187_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_142_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_142_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_280_nl);
  assign layer3_out_or_39_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_142_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_99_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_292_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_82_tmp_1
      & (z_out_4[18]);
  assign mux_188_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_148_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_148_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_292_nl);
  assign layer3_out_or_40_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_148_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_82_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_288_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_101_tmp_1
      & (z_out_4[18]);
  assign mux_189_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_146_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_146_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_288_nl);
  assign layer3_out_or_41_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_146_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_101_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_300_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_80_tmp_1
      & (z_out_4[18]);
  assign mux_190_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_152_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_152_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_300_nl);
  assign layer3_out_or_42_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_152_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_80_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_296_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_103_tmp_1
      & (z_out_4[18]);
  assign mux_191_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_150_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_150_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_296_nl);
  assign layer3_out_or_43_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_150_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_103_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_308_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_78_tmp_1
      & (z_out_4[18]);
  assign mux_192_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_156_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_156_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_308_nl);
  assign layer3_out_or_44_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_156_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_78_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_304_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_105_tmp_1
      & (z_out_4[18]);
  assign mux_193_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_154_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_154_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_304_nl);
  assign layer3_out_or_45_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_154_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_105_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_316_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_76_tmp_1
      & (z_out_4[18]);
  assign mux_194_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_160_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_160_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_316_nl);
  assign layer3_out_or_46_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_160_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_76_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_312_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_107_tmp_1
      & (z_out_4[18]);
  assign mux_195_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_158_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_158_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_312_nl);
  assign layer3_out_or_47_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_158_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_107_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_324_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_74_tmp_1
      & (z_out_4[18]);
  assign mux_196_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_164_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_164_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_324_nl);
  assign layer3_out_or_48_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_164_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_74_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_320_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_109_tmp_1
      & (z_out_4[18]);
  assign mux_197_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_162_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_162_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_320_nl);
  assign layer3_out_or_49_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_162_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_109_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_332_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_72_tmp_1
      & (z_out_4[18]);
  assign mux_198_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_168_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_168_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_332_nl);
  assign layer3_out_or_50_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_168_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_72_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_328_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_111_tmp_1
      & (z_out_4[18]);
  assign mux_199_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_166_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_166_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_328_nl);
  assign layer3_out_or_51_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_166_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_111_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_340_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_70_tmp_1
      & (z_out_4[18]);
  assign mux_200_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_172_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_172_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_340_nl);
  assign layer3_out_or_52_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_172_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_70_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_336_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_113_tmp_1
      & (z_out_4[18]);
  assign mux_201_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_170_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_170_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_336_nl);
  assign layer3_out_or_53_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_170_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_113_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_348_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_68_tmp_1
      & (z_out_4[18]);
  assign mux_202_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_176_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_176_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_348_nl);
  assign layer3_out_or_54_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_176_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_68_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_344_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_115_tmp_1
      & (z_out_4[18]);
  assign mux_203_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_174_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_174_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_344_nl);
  assign layer3_out_or_55_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_174_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_115_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_356_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_66_tmp_1
      & (z_out_4[18]);
  assign mux_204_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_180_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_180_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_356_nl);
  assign layer3_out_or_56_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_180_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_66_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_352_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_117_tmp_1
      & (z_out_4[18]);
  assign mux_205_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_178_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_178_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_352_nl);
  assign layer3_out_or_57_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_178_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_117_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_364_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_64_tmp_1
      & (z_out_4[18]);
  assign mux_206_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_184_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_184_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_364_nl);
  assign layer3_out_or_58_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_184_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_64_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_360_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_119_tmp_1
      & (z_out_4[18]);
  assign mux_207_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_182_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_182_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_360_nl);
  assign layer3_out_or_59_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_182_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_119_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_372_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_62_tmp_1
      & (z_out_4[18]);
  assign mux_208_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_188_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_188_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_372_nl);
  assign layer3_out_or_60_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_188_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_62_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_368_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_121_tmp_1
      & (z_out_4[18]);
  assign mux_209_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_186_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_186_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_368_nl);
  assign layer3_out_or_61_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_186_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_121_tmp_1) & (z_out_4[18])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_376_nl = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_60_tmp_1
      & (z_out_4[18]);
  assign mux_210_cse = MUX_v_17_2_2(({{16{nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_128_0}},
      nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_128_0}), (MultLoop_mux_64_itm_mx0w0[16:0]),
      nnet_relu_layer2_t_layer3_t_relu_config3_for_and_376_nl);
  assign layer3_out_or_62_cse = ~(((~ (z_out_4[18])) & nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_128_0)
      | ((~ nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_60_tmp_1) & (z_out_4[18])));
  assign layer5_out_and_cse = core_wen & (fsm_output[10]);
  assign and_3282_cse = (fsm_output[13]) & (~ (z_out_7[6]));
  assign nor_417_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00));
  assign nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse = core_wen
      & ((fsm_output[6]) | or_tmp_318);
  assign and_3330_cse = (fsm_output[5]) & ReuseLoop_acc_itm_6;
  assign nor_416_cse = ~((InitAccum_1_iacc_6_0_sva_5_0[1]) | (InitAccum_1_iacc_6_0_sva_5_0[3]));
  assign nor_415_cse = ~((InitAccum_1_iacc_6_0_sva_5_0[0]) | (InitAccum_1_iacc_6_0_sva_5_0[2]));
  assign nor_400_cse = ~((InitAccum_1_iacc_6_0_sva_5_0[5:4]!=2'b00));
  assign and_4008_cse = (fsm_output[6]) & (z_out_5[6]) & (z_out_6[6]) & (z_out_7[6]);
  assign or_2157_cse = (fsm_output[4]) | (fsm_output[8]);
  assign nand_62_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:1]==2'b11) &
      or_2157_cse);
  assign nor_518_cse = ~((fsm_output[4]) | (fsm_output[8]));
  assign nor_517_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00) | nor_518_cse);
  assign nand_60_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2]) & or_2157_cse);
  assign nand_53_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b111) &
      or_2157_cse);
  assign mux_274_cse = MUX_s_1_2_2((fsm_output[12]), nor_517_cse, reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign nor_466_nl = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | nor_518_cse);
  assign mux_280_cse = MUX_s_1_2_2((nor_466_nl), (fsm_output[12]), reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign nand_42_cse = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp[0]) & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b111)
      & or_2157_cse);
  assign or_1917_ssc = and_586_cse | ((~ ResultLoop_1_and_tmp) & (fsm_output[10]));
  assign not_286_nl = ~ or_tmp_1155;
  assign ReuseLoop_2_out_index_ReuseLoop_2_out_index_and_nl = MUX_v_4_2_2(4'b0000,
      MultLoop_2_im_3_0_sva_1_mx0w1, (not_286_nl));
  assign or_1923_nl = and_586_cse | (fsm_output[10]);
  assign InitAccum_1_iacc_mux1h_4_nl = MUX1HOT_v_6_4_2((z_out_5[5:0]), (MultLoop_1_im_6_0_sva_1[5:0]),
      (z_out_6[5:0]), ({2'b00 , (ReuseLoop_2_out_index_ReuseLoop_2_out_index_and_nl)}),
      {or_dcpl_670 , or_2157_cse , (or_1923_nl) , and_2802_cse});
  assign not_629_nl = ~ or_tmp_1116;
  assign InitAccum_1_iacc_and_2_rgt = MUX_v_6_2_2(6'b000000, (InitAccum_1_iacc_mux1h_4_nl),
      (not_629_nl));
  assign and_3921_cse = (fsm_output[12]) | (fsm_output[13]) | (fsm_output[16]) |
      (fsm_output[14]) | (fsm_output[11]) | (fsm_output[17]);
  assign not_285_nl = ~ or_tmp_1155;
  assign OUTPUT_LOOP_i_asn_ReuseLoop_1_w_index_11_6_sva_1_3_ReuseLoop_2_w_index_and_nl
      = MUX_v_4_2_2(4'b0000, z_out_2, (not_285_nl));
  assign InitAccum_1_iacc_mux1h_5_nl = MUX1HOT_v_6_3_2((layer3_out_0_16_0_sva_dfm[11:6]),
      (z_out_7[5:0]), ({2'b00 , (OUTPUT_LOOP_i_asn_ReuseLoop_1_w_index_11_6_sva_1_3_ReuseLoop_2_w_index_and_nl)}),
      {(fsm_output[4]) , (fsm_output[7]) , and_2802_cse});
  assign nor_nl = ~(or_dcpl_768 | (fsm_output[6]) | (fsm_output[9]));
  assign InitAccum_1_iacc_and_1_rgt = MUX_v_6_2_2(6'b000000, (InitAccum_1_iacc_mux1h_5_nl),
      (nor_nl));
  assign or_1972_nl = (fsm_output[5:3]!=3'b000);
  assign ReuseLoop_ir_ReuseLoop_ir_and_1_nl = MUX_v_10_2_2(10'b0000000000, z_out_3,
      (or_1972_nl));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_nnet_relu_layer4_t_layer5_t_relu_config5_for_and_nl
      = MUX_v_17_2_2(17'b00000000000000000, (MultLoop_mux_64_itm_mx0w0[16:0]), (z_out_4[18]));
  assign layer3_out_layer3_out_mux_rgt = MUX_v_17_2_2(({7'b0000000 , (ReuseLoop_ir_ReuseLoop_ir_and_1_nl)}),
      (nnet_relu_layer4_t_layer5_t_relu_config5_for_nnet_relu_layer4_t_layer5_t_relu_config5_for_and_nl),
      or_tmp_1141);
  assign and_3931_cse = (~((fsm_output[13:11]!=3'b000))) & (~((fsm_output[9:7]!=3'b000)));
  assign MultLoop_and_rgt = (~ or_dcpl_680) & (fsm_output[6]);
  assign nor_27_nl = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_122_tmp_1
      | (~ (z_out_4[18])));
  assign nand_29_nl = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_122_tmp_1
      & (z_out_4[18]));
  assign or_2036_nl = (~ nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1)
      | (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign mux_tmp = MUX_s_1_2_2((nor_27_nl), (nand_29_nl), or_2036_nl);
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_and_132_tmp = nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_122_tmp_1
      & (z_out_4[18]);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1,
      or_dcpl_571);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1,
      or_dcpl_572);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1,
      or_dcpl_568);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1,
      or_dcpl_573);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1,
      or_dcpl_566);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1,
      or_dcpl_574);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1,
      or_dcpl_118);
  assign nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1,
      or_dcpl_576);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1, or_dcpl_582);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1, or_dcpl_587);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1, or_dcpl_589);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1, or_dcpl_591);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1, or_dcpl_594);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1, or_dcpl_597);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1, or_dcpl_598);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1, or_dcpl_599);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1, or_dcpl_601);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1, or_dcpl_603);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1, or_dcpl_604);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1, or_dcpl_605);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1, or_dcpl_607);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1, or_dcpl_609);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1, or_dcpl_610);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1, or_dcpl_611);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1, or_dcpl_612);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1, or_dcpl_613);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1, or_dcpl_614);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1, or_dcpl_615);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1, or_dcpl_616);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1, or_dcpl_617);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1, or_dcpl_618);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1, or_dcpl_619);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1, or_dcpl_620);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1, or_dcpl_621);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1, or_dcpl_622);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1, or_dcpl_623);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1, or_dcpl_624);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1, or_dcpl_625);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1, or_dcpl_626);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1, or_dcpl_627);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1, or_dcpl_630);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1, or_dcpl_633);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1, or_dcpl_635);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1, or_dcpl_637);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1, or_dcpl_638);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1, or_dcpl_639);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1, or_dcpl_640);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1, or_dcpl_641);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1, or_dcpl_642);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1, or_dcpl_643);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1, or_dcpl_644);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1, or_dcpl_645);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1, or_dcpl_646);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1, or_dcpl_647);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1, or_dcpl_648);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1, or_dcpl_649);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1, or_dcpl_650);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1, or_dcpl_651);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1, or_dcpl_652);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1, or_dcpl_653);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1, or_dcpl_654);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1, or_dcpl_655);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1, or_dcpl_656);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1, or_dcpl_657);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1, or_dcpl_658);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1, or_dcpl_659);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1, or_dcpl_660);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1, or_dcpl_661);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1_mx0w0 =
      MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1, or_dcpl_662);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1_mx0w0
      = MUX_v_18_2_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1, or_dcpl_663);
  assign ReuseLoop_ir_9_0_sva_mx0_tmp_9_4 = MUX_v_6_2_2(6'b000000, (z_out_3[9:4]),
      (fsm_output[5]));
  assign InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1
      = MUX_v_18_64_2((b2_rsci_idat_mxwt[17:0]), (b2_rsci_idat_mxwt[35:18]), (b2_rsci_idat_mxwt[53:36]),
      (b2_rsci_idat_mxwt[71:54]), (b2_rsci_idat_mxwt[89:72]), (b2_rsci_idat_mxwt[107:90]),
      (b2_rsci_idat_mxwt[125:108]), (b2_rsci_idat_mxwt[143:126]), (b2_rsci_idat_mxwt[161:144]),
      (b2_rsci_idat_mxwt[179:162]), (b2_rsci_idat_mxwt[197:180]), (b2_rsci_idat_mxwt[215:198]),
      (b2_rsci_idat_mxwt[233:216]), (b2_rsci_idat_mxwt[251:234]), (b2_rsci_idat_mxwt[269:252]),
      (b2_rsci_idat_mxwt[287:270]), (b2_rsci_idat_mxwt[305:288]), (b2_rsci_idat_mxwt[323:306]),
      (b2_rsci_idat_mxwt[341:324]), (b2_rsci_idat_mxwt[359:342]), (b2_rsci_idat_mxwt[377:360]),
      (b2_rsci_idat_mxwt[395:378]), (b2_rsci_idat_mxwt[413:396]), (b2_rsci_idat_mxwt[431:414]),
      (b2_rsci_idat_mxwt[449:432]), (b2_rsci_idat_mxwt[467:450]), (b2_rsci_idat_mxwt[485:468]),
      (b2_rsci_idat_mxwt[503:486]), (b2_rsci_idat_mxwt[521:504]), (b2_rsci_idat_mxwt[539:522]),
      (b2_rsci_idat_mxwt[557:540]), (b2_rsci_idat_mxwt[575:558]), (b2_rsci_idat_mxwt[593:576]),
      (b2_rsci_idat_mxwt[611:594]), (b2_rsci_idat_mxwt[629:612]), (b2_rsci_idat_mxwt[647:630]),
      (b2_rsci_idat_mxwt[665:648]), (b2_rsci_idat_mxwt[683:666]), (b2_rsci_idat_mxwt[701:684]),
      (b2_rsci_idat_mxwt[719:702]), (b2_rsci_idat_mxwt[737:720]), (b2_rsci_idat_mxwt[755:738]),
      (b2_rsci_idat_mxwt[773:756]), (b2_rsci_idat_mxwt[791:774]), (b2_rsci_idat_mxwt[809:792]),
      (b2_rsci_idat_mxwt[827:810]), (b2_rsci_idat_mxwt[845:828]), (b2_rsci_idat_mxwt[863:846]),
      (b2_rsci_idat_mxwt[881:864]), (b2_rsci_idat_mxwt[899:882]), (b2_rsci_idat_mxwt[917:900]),
      (b2_rsci_idat_mxwt[935:918]), (b2_rsci_idat_mxwt[953:936]), (b2_rsci_idat_mxwt[971:954]),
      (b2_rsci_idat_mxwt[989:972]), (b2_rsci_idat_mxwt[1007:990]), (b2_rsci_idat_mxwt[1025:1008]),
      (b2_rsci_idat_mxwt[1043:1026]), (b2_rsci_idat_mxwt[1061:1044]), (b2_rsci_idat_mxwt[1079:1062]),
      (b2_rsci_idat_mxwt[1097:1080]), (b2_rsci_idat_mxwt[1115:1098]), (b2_rsci_idat_mxwt[1133:1116]),
      (b2_rsci_idat_mxwt[1151:1134]), InitAccum_1_iacc_6_0_sva_5_0);
  assign InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3
      = MUX_v_18_64_2((b4_rsci_idat_mxwt[17:0]), (b4_rsci_idat_mxwt[35:18]), (b4_rsci_idat_mxwt[53:36]),
      (b4_rsci_idat_mxwt[71:54]), (b4_rsci_idat_mxwt[89:72]), (b4_rsci_idat_mxwt[107:90]),
      (b4_rsci_idat_mxwt[125:108]), (b4_rsci_idat_mxwt[143:126]), (b4_rsci_idat_mxwt[161:144]),
      (b4_rsci_idat_mxwt[179:162]), (b4_rsci_idat_mxwt[197:180]), (b4_rsci_idat_mxwt[215:198]),
      (b4_rsci_idat_mxwt[233:216]), (b4_rsci_idat_mxwt[251:234]), (b4_rsci_idat_mxwt[269:252]),
      (b4_rsci_idat_mxwt[287:270]), (b4_rsci_idat_mxwt[305:288]), (b4_rsci_idat_mxwt[323:306]),
      (b4_rsci_idat_mxwt[341:324]), (b4_rsci_idat_mxwt[359:342]), (b4_rsci_idat_mxwt[377:360]),
      (b4_rsci_idat_mxwt[395:378]), (b4_rsci_idat_mxwt[413:396]), (b4_rsci_idat_mxwt[431:414]),
      (b4_rsci_idat_mxwt[449:432]), (b4_rsci_idat_mxwt[467:450]), (b4_rsci_idat_mxwt[485:468]),
      (b4_rsci_idat_mxwt[503:486]), (b4_rsci_idat_mxwt[521:504]), (b4_rsci_idat_mxwt[539:522]),
      (b4_rsci_idat_mxwt[557:540]), (b4_rsci_idat_mxwt[575:558]), (b4_rsci_idat_mxwt[593:576]),
      (b4_rsci_idat_mxwt[611:594]), (b4_rsci_idat_mxwt[629:612]), (b4_rsci_idat_mxwt[647:630]),
      (b4_rsci_idat_mxwt[665:648]), (b4_rsci_idat_mxwt[683:666]), (b4_rsci_idat_mxwt[701:684]),
      (b4_rsci_idat_mxwt[719:702]), (b4_rsci_idat_mxwt[737:720]), (b4_rsci_idat_mxwt[755:738]),
      (b4_rsci_idat_mxwt[773:756]), (b4_rsci_idat_mxwt[791:774]), (b4_rsci_idat_mxwt[809:792]),
      (b4_rsci_idat_mxwt[827:810]), (b4_rsci_idat_mxwt[845:828]), (b4_rsci_idat_mxwt[863:846]),
      (b4_rsci_idat_mxwt[881:864]), (b4_rsci_idat_mxwt[899:882]), (b4_rsci_idat_mxwt[917:900]),
      (b4_rsci_idat_mxwt[935:918]), (b4_rsci_idat_mxwt[953:936]), (b4_rsci_idat_mxwt[971:954]),
      (b4_rsci_idat_mxwt[989:972]), (b4_rsci_idat_mxwt[1007:990]), (b4_rsci_idat_mxwt[1025:1008]),
      (b4_rsci_idat_mxwt[1043:1026]), (b4_rsci_idat_mxwt[1061:1044]), (b4_rsci_idat_mxwt[1079:1062]),
      (b4_rsci_idat_mxwt[1097:1080]), (b4_rsci_idat_mxwt[1115:1098]), (b4_rsci_idat_mxwt[1133:1116]),
      (b4_rsci_idat_mxwt[1151:1134]), InitAccum_1_iacc_6_0_sva_5_0);
  assign InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1 = MUX_v_18_10_2((b6_rsci_idat_mxwt[17:0]),
      (b6_rsci_idat_mxwt[35:18]), (b6_rsci_idat_mxwt[53:36]), (b6_rsci_idat_mxwt[71:54]),
      (b6_rsci_idat_mxwt[89:72]), (b6_rsci_idat_mxwt[107:90]), (b6_rsci_idat_mxwt[125:108]),
      (b6_rsci_idat_mxwt[143:126]), (b6_rsci_idat_mxwt[161:144]), (b6_rsci_idat_mxwt[179:162]),
      reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp);
  assign nl_MultLoop_2_if_1_acc_tmp = reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp + 4'b0001;
  assign MultLoop_2_if_1_acc_tmp = nl_MultLoop_2_if_1_acc_tmp[3:0];
  assign MultLoop_mux_64_itm_mx0w0 = MUX_v_18_64_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_0_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_63_lpi_4, {reg_ResultLoop_1_ires_6_0_sva_5_0_tmp
      , reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp});
  assign nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl
      = conv_s2s_22_23({(z_out_1[17]) , nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1
      , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1 , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1
      , 4'b0000}) + conv_s2s_20_23({(z_out_1[17]) , nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1
      , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1 , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1
      , 2'b00}) + conv_s2s_18_23({(z_out_1[17]) , nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1
      , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1 , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1})
      + conv_s2s_16_23({(z_out_1[17]) , nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1});
  assign ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl
      = nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl[22:0];
  assign nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl
      = ({(~ (z_out_1[17])) , (~ nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1)
      , (~ nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1) , (~
      nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1)}) + conv_s2s_15_18(readslicef_23_15_8((ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_3_nl)));
  assign ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl
      = nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl[17:0];
  assign nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl
      = conv_s2u_21_22({(~ (z_out_1[17])) , (~ nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1)
      , (~ nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1) , (~
      nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1) , 3'b001})
      + conv_s2u_18_22(ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_6_nl);
  assign ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl
      = nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl[21:0];
  assign nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl
      = conv_s2u_19_20(readslicef_22_19_3((ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_7_nl)))
      + ({(z_out_1[17]) , nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1
      , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1 , nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1
      , 2'b01});
  assign ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl
      = nl_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl[19:0];
  assign ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1
      = readslicef_20_19_1((ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_nl));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_0_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_14_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_1_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_13_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_2_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_12_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_3_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_11_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_4_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_10_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_5_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_9_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_6_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_8_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_7_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_7_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_8_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_6_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_9_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_5_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_10_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_4_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_11_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_3_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_12_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_2_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_13_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_1_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_14_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_0_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[0]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_15_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[0]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_15_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_7_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_121_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_119_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_117_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_115_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_113_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_111_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_109_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_107_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_105_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_103_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_101_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_99_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_97_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_95_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_93_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_91_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_89_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_87_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_85_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_83_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_81_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_79_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_77_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_75_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_73_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_71_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_69_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_67_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_65_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_63_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_61_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_60_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_15_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4==2'b01);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_62_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_64_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_66_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_68_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_70_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_72_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_74_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_76_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_78_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_80_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_82_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_84_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_86_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_88_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_90_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_92_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_94_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_96_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_98_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_100_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_102_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_104_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_106_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_108_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_110_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_112_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_114_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_116_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_118_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_120_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_if_and_122_tmp_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_0_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_0_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_1_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_1_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_2_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_2_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_3_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_3_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_4_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_4_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_5_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_5_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_6_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_6_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_7_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_7_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[3]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_8_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_0_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_9_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_1_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_10_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_2_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_11_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_3_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_12_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_4_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_13_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_5_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_14_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_6_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[3]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_0_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_0_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[2]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_1_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_1_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[2]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_2_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_2_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[2]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_3_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_3_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_3_0[2]));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_4_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_0_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[2]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_5_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_1_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[2]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_6_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_2_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[2]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_2_7_sva_1 = nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_3_sva_1
      & (MultLoop_1_if_1_acc_itm_3_0[2]);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_0_sva_1 = ~((MultLoop_1_if_1_acc_itm_3_0[1:0]!=2'b00));
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_1_sva_1 = (MultLoop_1_if_1_acc_itm_3_0[1:0]==2'b01);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_2_sva_1 = (MultLoop_1_if_1_acc_itm_3_0[1:0]==2'b10);
  assign nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_1_3_sva_1 = (MultLoop_1_if_1_acc_itm_3_0[1:0]==2'b11);
  assign nl_MultLoop_2_im_3_0_sva_1_mx0w1 = MultLoop_1_if_1_acc_itm_3_0 + 4'b0001;
  assign MultLoop_2_im_3_0_sva_1_mx0w1 = nl_MultLoop_2_im_3_0_sva_1_mx0w1[3:0];
  assign tmp_lpi_3_dfm_1 = MUX_v_1152_10_2(w6_rsc_0_0_i_idat, w6_rsc_1_0_i_idat,
      w6_rsc_2_0_i_idat, w6_rsc_3_0_i_idat, w6_rsc_4_0_i_idat, w6_rsc_5_0_i_idat,
      w6_rsc_6_0_i_idat, w6_rsc_7_0_i_idat, w6_rsc_8_0_i_idat, w6_rsc_9_0_i_idat,
      reg_ReuseLoop_1_w_index_11_6_1_reg);
  assign nl_ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1
      = ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva
      + conv_u2u_67_71(operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1);
  assign ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1
      = nl_ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1[70:0];
  assign nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_2_nl = ~(MUX_v_15_2_2((z_out_1[14:0]),
      15'b111111111111111, nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1));
  assign nnet_softmax_layer6_t_result_t_softmax_config7_for_and_nl = (z_out_1[17])
      & (~((z_out_1[16:15]==2'b11)));
  assign nnet_softmax_layer6_t_result_t_softmax_config7_for_nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_psp_sva_1
      = ~(MUX_v_15_2_2((nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_2_nl),
      15'b111111111111111, (nnet_softmax_layer6_t_result_t_softmax_config7_for_and_nl)));
  assign nnet_softmax_layer6_t_result_t_softmax_config7_for_nor_ovfl_sva_1 = ~((z_out_1[17])
      | (~((z_out_1[16:15]!=2'b00))));
  assign CALC_SOFTMAX_LOOP_or_1_nl = (z_out[157:92]!=66'b000000000000000000000000000000000000000000000000000000000000000000);
  assign CALC_SOFTMAX_LOOP_or_psp_sva_1 = MUX_v_12_2_2((z_out[91:80]), 12'b111111111111,
      (CALC_SOFTMAX_LOOP_or_1_nl));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_66_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_68_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_70_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_1_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_72_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_74_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_2_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_76_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_78_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_3_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_80_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_82_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_4_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_84_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_86_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_5_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_88_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_90_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_6_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_92_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_94_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_7_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_96_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_98_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_8_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_100_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_102_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_9_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_104_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_106_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_10_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_108_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_110_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_11_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_112_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_114_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_12_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_116_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_118_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_13_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_120_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_122_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_14_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_124_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_126_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_15_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_128_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_3_15_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4==2'b01));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_130_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_132_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_16_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_134_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_136_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_17_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_138_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_140_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_18_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_142_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_144_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_19_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_146_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_148_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_20_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_150_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_152_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_21_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_154_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_156_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_22_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_158_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_160_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_23_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_162_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_164_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_24_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_166_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_168_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_25_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_170_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_172_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_26_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_174_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_176_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_27_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_178_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_180_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_28_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_182_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_184_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_29_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_186_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1
      & (MultLoop_1_if_1_acc_itm_5_4[1]));
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_exs_188_0 = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_30_sva_1
      & (~ (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign nl_InitAccum_2_acc_1_nl = conv_u2s_3_4(MultLoop_2_if_1_acc_tmp[3:1]) + 4'b1011;
  assign InitAccum_2_acc_1_nl = nl_InitAccum_2_acc_1_nl[3:0];
  assign InitAccum_2_acc_1_itm_3 = readslicef_4_1_3((InitAccum_2_acc_1_nl));
  assign OUTPUT_LOOP_or_tmp = (reg_ReuseLoop_1_w_index_11_6_1_reg!=4'b0000);
  assign ResultLoop_1_and_tmp = (z_out_5[6]) & (z_out_6[6]);
  assign ResultLoop_and_1_tmp = (z_out_5[6]) & (z_out_6[6]) & (z_out_7[6]);
  assign or_dcpl_105 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00);
  assign or_dcpl_106 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]!=2'b00);
  assign or_dcpl_107 = or_dcpl_106 | or_dcpl_105;
  assign or_dcpl_117 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01);
  assign or_dcpl_118 = or_dcpl_106 | or_dcpl_117;
  assign or_dcpl_153 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b10);
  assign or_dcpl_154 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]!=2'b10);
  assign or_dcpl_160 = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]==2'b11));
  assign or_dcpl_528 = (reg_ReuseLoop_1_w_index_11_6_1_reg[2]) | (~ (reg_ReuseLoop_1_w_index_11_6_1_reg[0]));
  assign or_dcpl_532 = (reg_ReuseLoop_1_w_index_11_6_1_reg[2]) | (reg_ReuseLoop_1_w_index_11_6_1_reg[0]);
  assign or_dcpl_537 = (~ (reg_ReuseLoop_1_w_index_11_6_1_reg[2])) | (reg_ReuseLoop_1_w_index_11_6_1_reg[0]);
  assign or_dcpl_539 = ~((reg_ReuseLoop_1_w_index_11_6_1_reg[2]) & (reg_ReuseLoop_1_w_index_11_6_1_reg[0]));
  assign or_dcpl_547 = (reg_ReuseLoop_1_w_index_11_6_1_reg[1]) | (reg_ReuseLoop_1_w_index_11_6_1_reg[3]);
  assign or_dcpl_548 = or_dcpl_547 | or_dcpl_528;
  assign or_dcpl_550 = (~ (reg_ReuseLoop_1_w_index_11_6_1_reg[1])) | (reg_ReuseLoop_1_w_index_11_6_1_reg[3]);
  assign or_dcpl_551 = or_dcpl_550 | or_dcpl_532;
  assign or_dcpl_552 = or_dcpl_550 | or_dcpl_528;
  assign or_dcpl_554 = or_dcpl_547 | or_dcpl_537;
  assign or_dcpl_556 = or_dcpl_547 | or_dcpl_539;
  assign or_dcpl_557 = or_dcpl_550 | or_dcpl_537;
  assign or_dcpl_558 = or_dcpl_550 | or_dcpl_539;
  assign or_dcpl_559 = (reg_ReuseLoop_1_w_index_11_6_1_reg[1]) | (~ (reg_ReuseLoop_1_w_index_11_6_1_reg[3]));
  assign or_dcpl_560 = or_dcpl_559 | or_dcpl_532;
  assign or_dcpl_561 = or_dcpl_559 | or_dcpl_528;
  assign or_dcpl_566 = or_dcpl_106 | or_dcpl_153;
  assign or_dcpl_568 = or_dcpl_106 | or_dcpl_160;
  assign or_dcpl_570 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]!=2'b01);
  assign or_dcpl_571 = or_dcpl_570 | or_dcpl_105;
  assign or_dcpl_572 = or_dcpl_570 | or_dcpl_117;
  assign or_dcpl_573 = or_dcpl_570 | or_dcpl_153;
  assign or_dcpl_574 = or_dcpl_570 | or_dcpl_160;
  assign or_dcpl_576 = or_dcpl_154 | or_dcpl_105;
  assign or_dcpl_578 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]!=2'b01);
  assign or_dcpl_579 = or_dcpl_578 | (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign or_dcpl_580 = ~((InitAccum_1_iacc_6_0_sva_5_0[2:1]==2'b11));
  assign or_dcpl_581 = or_dcpl_580 | (~ (InitAccum_1_iacc_6_0_sva_5_0[3]));
  assign or_dcpl_582 = or_dcpl_581 | or_dcpl_579;
  assign or_dcpl_583 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]!=2'b10);
  assign or_dcpl_584 = or_dcpl_583 | (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign or_dcpl_585 = (InitAccum_1_iacc_6_0_sva_5_0[2:1]!=2'b00);
  assign or_dcpl_586 = or_dcpl_585 | (InitAccum_1_iacc_6_0_sva_5_0[3]);
  assign or_dcpl_587 = or_dcpl_586 | or_dcpl_584;
  assign or_dcpl_588 = or_dcpl_578 | (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign or_dcpl_589 = or_dcpl_581 | or_dcpl_588;
  assign or_dcpl_590 = or_dcpl_583 | (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign or_dcpl_591 = or_dcpl_586 | or_dcpl_590;
  assign or_dcpl_592 = (InitAccum_1_iacc_6_0_sva_5_0[2:1]!=2'b10);
  assign or_dcpl_593 = or_dcpl_592 | (~ (InitAccum_1_iacc_6_0_sva_5_0[3]));
  assign or_dcpl_594 = or_dcpl_593 | or_dcpl_579;
  assign or_dcpl_595 = (InitAccum_1_iacc_6_0_sva_5_0[2:1]!=2'b01);
  assign or_dcpl_596 = or_dcpl_595 | (InitAccum_1_iacc_6_0_sva_5_0[3]);
  assign or_dcpl_597 = or_dcpl_596 | or_dcpl_584;
  assign or_dcpl_598 = or_dcpl_593 | or_dcpl_588;
  assign or_dcpl_599 = or_dcpl_596 | or_dcpl_590;
  assign or_dcpl_600 = or_dcpl_595 | (~ (InitAccum_1_iacc_6_0_sva_5_0[3]));
  assign or_dcpl_601 = or_dcpl_600 | or_dcpl_579;
  assign or_dcpl_602 = or_dcpl_592 | (InitAccum_1_iacc_6_0_sva_5_0[3]);
  assign or_dcpl_603 = or_dcpl_602 | or_dcpl_584;
  assign or_dcpl_604 = or_dcpl_600 | or_dcpl_588;
  assign or_dcpl_605 = or_dcpl_602 | or_dcpl_590;
  assign or_dcpl_606 = or_dcpl_585 | (~ (InitAccum_1_iacc_6_0_sva_5_0[3]));
  assign or_dcpl_607 = or_dcpl_606 | or_dcpl_579;
  assign or_dcpl_608 = or_dcpl_580 | (InitAccum_1_iacc_6_0_sva_5_0[3]);
  assign or_dcpl_609 = or_dcpl_608 | or_dcpl_584;
  assign or_dcpl_610 = or_dcpl_606 | or_dcpl_588;
  assign or_dcpl_611 = or_dcpl_608 | or_dcpl_590;
  assign or_dcpl_612 = or_dcpl_608 | or_dcpl_579;
  assign or_dcpl_613 = or_dcpl_606 | or_dcpl_584;
  assign or_dcpl_614 = or_dcpl_608 | or_dcpl_588;
  assign or_dcpl_615 = or_dcpl_606 | or_dcpl_590;
  assign or_dcpl_616 = or_dcpl_602 | or_dcpl_579;
  assign or_dcpl_617 = or_dcpl_600 | or_dcpl_584;
  assign or_dcpl_618 = or_dcpl_602 | or_dcpl_588;
  assign or_dcpl_619 = or_dcpl_600 | or_dcpl_590;
  assign or_dcpl_620 = or_dcpl_596 | or_dcpl_579;
  assign or_dcpl_621 = or_dcpl_593 | or_dcpl_584;
  assign or_dcpl_622 = or_dcpl_596 | or_dcpl_588;
  assign or_dcpl_623 = or_dcpl_593 | or_dcpl_590;
  assign or_dcpl_624 = or_dcpl_586 | or_dcpl_579;
  assign or_dcpl_625 = or_dcpl_581 | or_dcpl_584;
  assign or_dcpl_626 = or_dcpl_586 | or_dcpl_588;
  assign or_dcpl_627 = or_dcpl_581 | or_dcpl_590;
  assign or_dcpl_628 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]!=2'b00);
  assign or_dcpl_629 = or_dcpl_628 | (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign or_dcpl_630 = or_dcpl_581 | or_dcpl_629;
  assign or_dcpl_631 = ~((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11));
  assign or_dcpl_632 = or_dcpl_631 | (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign or_dcpl_633 = or_dcpl_586 | or_dcpl_632;
  assign or_dcpl_634 = or_dcpl_628 | (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign or_dcpl_635 = or_dcpl_581 | or_dcpl_634;
  assign or_dcpl_636 = or_dcpl_631 | (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign or_dcpl_637 = or_dcpl_586 | or_dcpl_636;
  assign or_dcpl_638 = or_dcpl_593 | or_dcpl_629;
  assign or_dcpl_639 = or_dcpl_596 | or_dcpl_632;
  assign or_dcpl_640 = or_dcpl_593 | or_dcpl_634;
  assign or_dcpl_641 = or_dcpl_596 | or_dcpl_636;
  assign or_dcpl_642 = or_dcpl_600 | or_dcpl_629;
  assign or_dcpl_643 = or_dcpl_602 | or_dcpl_632;
  assign or_dcpl_644 = or_dcpl_600 | or_dcpl_634;
  assign or_dcpl_645 = or_dcpl_602 | or_dcpl_636;
  assign or_dcpl_646 = or_dcpl_606 | or_dcpl_629;
  assign or_dcpl_647 = or_dcpl_608 | or_dcpl_632;
  assign or_dcpl_648 = or_dcpl_606 | or_dcpl_634;
  assign or_dcpl_649 = or_dcpl_608 | or_dcpl_636;
  assign or_dcpl_650 = or_dcpl_608 | or_dcpl_629;
  assign or_dcpl_651 = or_dcpl_606 | or_dcpl_632;
  assign or_dcpl_652 = or_dcpl_608 | or_dcpl_634;
  assign or_dcpl_653 = or_dcpl_606 | or_dcpl_636;
  assign or_dcpl_654 = or_dcpl_602 | or_dcpl_629;
  assign or_dcpl_655 = or_dcpl_600 | or_dcpl_632;
  assign or_dcpl_656 = or_dcpl_602 | or_dcpl_634;
  assign or_dcpl_657 = or_dcpl_600 | or_dcpl_636;
  assign or_dcpl_658 = or_dcpl_596 | or_dcpl_629;
  assign or_dcpl_659 = or_dcpl_593 | or_dcpl_632;
  assign or_dcpl_660 = or_dcpl_596 | or_dcpl_634;
  assign or_dcpl_661 = or_dcpl_593 | or_dcpl_636;
  assign or_dcpl_662 = or_dcpl_586 | or_dcpl_629;
  assign or_dcpl_663 = or_dcpl_581 | or_dcpl_632;
  assign or_dcpl_670 = (fsm_output[7]) | (fsm_output[3]);
  assign and_dcpl_31 = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:1]!=2'b00));
  assign and_dcpl_32 = and_dcpl_31 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]));
  assign and_dcpl_33 = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00));
  assign and_dcpl_34 = and_dcpl_33 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]));
  assign and_dcpl_37 = nor_400_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign and_dcpl_38 = ~((InitAccum_1_iacc_6_0_sva_5_0[3:2]!=2'b00));
  assign and_dcpl_39 = ResultLoop_and_1_tmp & (~ (InitAccum_1_iacc_6_0_sva_5_0[1]));
  assign and_dcpl_40 = and_dcpl_39 & and_dcpl_38;
  assign or_dcpl_680 = or_dcpl_586 | or_dcpl_634;
  assign and_dcpl_44 = ~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]!=2'b00));
  assign and_dcpl_48 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:1]==2'b11);
  assign and_dcpl_49 = and_dcpl_48 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]);
  assign and_dcpl_50 = (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11);
  assign and_dcpl_51 = and_dcpl_50 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign and_dcpl_52 = and_dcpl_51 & and_dcpl_49;
  assign and_dcpl_53 = and_dcpl_48 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]));
  assign and_dcpl_55 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11);
  assign and_dcpl_56 = and_dcpl_55 & (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign and_dcpl_57 = (InitAccum_1_iacc_6_0_sva_5_0[3:2]==2'b11);
  assign and_dcpl_58 = ResultLoop_and_1_tmp & (InitAccum_1_iacc_6_0_sva_5_0[1]);
  assign and_dcpl_59 = and_dcpl_58 & and_dcpl_57;
  assign and_dcpl_62 = and_dcpl_31 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]);
  assign and_dcpl_64 = nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign and_dcpl_67 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]==2'b01);
  assign and_dcpl_69 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:1]==2'b10);
  assign and_dcpl_70 = and_dcpl_69 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]);
  assign and_dcpl_72 = and_dcpl_55 & (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign and_dcpl_73 = and_dcpl_39 & and_dcpl_57;
  assign and_dcpl_76 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:1]==2'b01);
  assign and_dcpl_77 = and_dcpl_76 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]));
  assign and_dcpl_79 = and_dcpl_58 & and_dcpl_38;
  assign and_dcpl_82 = and_dcpl_69 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]));
  assign and_dcpl_86 = and_dcpl_76 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]);
  assign and_dcpl_91 = (InitAccum_1_iacc_6_0_sva_5_0[3:2]==2'b10);
  assign and_dcpl_92 = and_dcpl_58 & and_dcpl_91;
  assign and_dcpl_96 = (InitAccum_1_iacc_6_0_sva_5_0[3:2]==2'b01);
  assign and_dcpl_97 = and_dcpl_39 & and_dcpl_96;
  assign and_dcpl_107 = and_dcpl_39 & and_dcpl_91;
  assign and_dcpl_111 = and_dcpl_58 & and_dcpl_96;
  assign and_dcpl_120 = and_dcpl_50 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]));
  assign and_dcpl_124 = and_dcpl_33 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign and_dcpl_140 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]==2'b10);
  assign and_dcpl_148 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]==2'b11);
  assign and_dcpl_156 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]==2'b01);
  assign and_dcpl_179 = (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b10);
  assign and_dcpl_180 = and_dcpl_179 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign and_dcpl_182 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b10);
  assign and_dcpl_183 = and_dcpl_182 & (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign and_dcpl_186 = (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b01);
  assign and_dcpl_187 = and_dcpl_186 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]));
  assign and_dcpl_189 = (InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b01);
  assign and_dcpl_190 = and_dcpl_189 & (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign and_dcpl_193 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]==2'b10);
  assign and_dcpl_196 = and_dcpl_182 & (~ (InitAccum_1_iacc_6_0_sva_5_0[0]));
  assign and_dcpl_200 = and_dcpl_189 & (InitAccum_1_iacc_6_0_sva_5_0[0]);
  assign and_dcpl_240 = and_dcpl_179 & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]));
  assign and_dcpl_244 = and_dcpl_186 & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign or_dcpl_768 = (fsm_output[2]) | (fsm_output[5]);
  assign or_dcpl_771 = (fsm_output[13:12]!=2'b00);
  assign and_dcpl_296 = ~((fsm_output[18]) | (fsm_output[17]) | (fsm_output[15]));
  assign and_583_cse = ResultLoop_1_and_tmp & (fsm_output[10]);
  assign and_585_cse = (~ ReuseLoop_acc_itm_6) & (fsm_output[5]);
  assign and_586_cse = (~ ResultLoop_and_1_tmp) & (fsm_output[6]);
  assign and_593_cse = (~ (z_out_7[6])) & (fsm_output[2]);
  assign and_605_cse = (z_out_7[6]) & (fsm_output[2]);
  assign or_tmp_318 = (~ (z_out_7[6])) & (fsm_output[9]);
  assign and_1829_cse = ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_tmp_1116 = (fsm_output[2]) | (fsm_output[9]) | (fsm_output[5]) | and_1829_cse;
  assign and_2802_cse = (fsm_output[18]) | (fsm_output[16]) | (fsm_output[17]) |
      (fsm_output[15]) | (fsm_output[14]) | (fsm_output[11]) | or_dcpl_771;
  assign or_tmp_1141 = (fsm_output[6]) | (fsm_output[10]);
  assign or_tmp_1155 = and_dcpl_296 & (~ (fsm_output[14])) & (~ (fsm_output[12]));
  assign InitAccum_1_iacc_6_0_sva_5_0_mx0c0 = and_3330_cse | and_605_cse;
  assign InitAccum_1_iacc_6_0_sva_5_0_mx0c2 = (fsm_output[9]) | (fsm_output[13])
      | and_586_cse | and_593_cse;
  assign InitAccum_1_iacc_6_0_sva_5_0_mx0c3 = (fsm_output[1]) | (fsm_output[11])
      | and_585_cse | and_1829_cse;
  assign nl_ReuseLoop_acc_nl = conv_u2s_6_7(z_out_3[9:4]) + 7'b1001111;
  assign ReuseLoop_acc_nl = nl_ReuseLoop_acc_nl[6:0];
  assign ReuseLoop_acc_itm_6 = readslicef_7_1_6((ReuseLoop_acc_nl));
  assign w2_rsci_CE2_d = w2_rsci_CE2_d_reg;
  assign w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d = w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg;
  assign w4_rsci_CE2_d = w4_rsci_CE2_d_reg;
  assign w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d = w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_reg;
  assign w2_rsci_A2_d = w2_rsci_A2_d_reg;
  assign w4_rsci_A2_d = w4_rsci_A2_d_reg;
  assign or_tmp_1201 = (fsm_output[14]) | (fsm_output[17]) | (fsm_output[12]);
  assign or_tmp_1207 = (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp!=4'b0000) | (~((fsm_output[12:11]!=2'b00)
      | nor_517_cse));
  assign MultLoop_1_if_1_or_1_ssc = (fsm_output[14]) | (fsm_output[17]);
  assign or_dcpl_1103 = and_583_cse | and_2802_cse;
  assign or_2579_tmp = (and_dcpl_52 & (fsm_output[8])) | (and_dcpl_52 & (fsm_output[4]));
  assign or_2583_tmp = or_2157_cse | (fsm_output[15]);
  assign MultLoop_or_3_cse = (fsm_output[16]) | (fsm_output[8]) | (fsm_output[12])
      | (fsm_output[4]);
  always @(posedge clk) begin
    if ( rst ) begin
      reg_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse <= 1'b0;
      reg_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse <= 1'b0;
      b6_rsci_oswt <= 1'b0;
      b4_rsci_oswt <= 1'b0;
      b2_rsci_oswt <= 1'b0;
      reg_const_size_out_1_rsci_ivld_core_psct_cse <= 1'b0;
      reg_output1_rsci_ivld_core_psct_cse <= 1'b0;
      reg_input1_rsci_irdy_core_psct_cse <= 1'b0;
      CALC_EXP_LOOP_i_3_0_sva_1 <= 4'b0000;
      MultLoop_1_im_6_0_sva_1 <= 7'b0000000;
      ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva
          <= 19'b0000000000000000000;
      ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_2_itm
          <= 3'b000;
      ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_itm
          <= 5'b00000;
      CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm <= 1'b0;
    end
    else if ( core_wen ) begin
      reg_w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse <= fsm_output[7];
      reg_w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d_core_psct_0_cse <= fsm_output[3];
      b6_rsci_oswt <= (InitAccum_2_acc_1_itm_3 & (fsm_output[11])) | and_583_cse;
      b4_rsci_oswt <= and_585_cse | and_586_cse;
      b2_rsci_oswt <= (fsm_output[1]) | and_593_cse;
      reg_const_size_out_1_rsci_ivld_core_psct_cse <= ~((~((fsm_output[18]) | (fsm_output[0])))
          | ((~ CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm) & (fsm_output[18])));
      reg_output1_rsci_ivld_core_psct_cse <= fsm_output[17];
      reg_input1_rsci_irdy_core_psct_cse <= and_3330_cse | and_605_cse | ((~ (MultLoop_1_im_6_0_sva_1[6]))
          & (fsm_output[4]));
      CALC_EXP_LOOP_i_3_0_sva_1 <= MUX1HOT_v_4_3_2((ReuseLoop_ir_ReuseLoop_ir_and_2_nl),
          (layer3_out_63_16_0_lpi_2_dfm_9_0[3:0]), MultLoop_2_if_1_acc_tmp, {or_dcpl_768
          , (fsm_output[4]) , (fsm_output[14])});
      MultLoop_1_im_6_0_sva_1 <= MUX1HOT_v_7_6_2(({1'b0 , (ReuseLoop_2_in_index_asn_MultLoop_1_im_6_0_sva_2_5_ReuseLoop_in_index_and_nl)}),
          z_out_6, 7'b1111110, 7'b1000000, 7'b0100110, 7'b0110111, {(or_1936_nl)
          , or_dcpl_670 , (MultLoop_1_im_and_nl) , (MultLoop_1_im_and_1_nl) , (MultLoop_1_im_and_2_nl)
          , (MultLoop_1_im_and_3_nl)});
      ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva
          <= MUX1HOT_v_19_3_2(({1'b0 , (MultLoop_mux_65_nl)}), ({2'b00 , (MultLoop_1_MultLoop_1_mux_nl)}),
          ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1,
          {(fsm_output[3]) , (fsm_output[7]) , (fsm_output[14])});
      ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_2_itm
          <= MUX_v_3_4_2(3'b011, 3'b100, 3'b101, 3'b110, ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]);
      ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_itm
          <= MUX_v_5_4_2(5'b01100, 5'b01110, 5'b10001, 5'b10100, ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]);
      CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm <= MUX1HOT_s_1_3_2((ResultLoop_2_and_nl),
          (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_nor_nl),
          (CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_nl), {(fsm_output[14]) , (fsm_output[15])
          , (fsm_output[17])});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      output1_rsci_idat_11_0 <= 12'b000000000000;
      output1_rsci_idat_29_18 <= 12'b000000000000;
      output1_rsci_idat_47_36 <= 12'b000000000000;
      output1_rsci_idat_65_54 <= 12'b000000000000;
      output1_rsci_idat_83_72 <= 12'b000000000000;
      output1_rsci_idat_101_90 <= 12'b000000000000;
      output1_rsci_idat_119_108 <= 12'b000000000000;
      output1_rsci_idat_137_126 <= 12'b000000000000;
      output1_rsci_idat_155_144 <= 12'b000000000000;
      output1_rsci_idat_173_162 <= 12'b000000000000;
    end
    else if ( output1_and_cse ) begin
      output1_rsci_idat_11_0 <= MUX_v_12_2_2(OUTPUT_LOOP_io_read_output1_rsc_sdt_11_0_lpi_2,
          CALC_SOFTMAX_LOOP_or_psp_sva_1, and_524_nl);
      output1_rsci_idat_29_18 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_29_18_lpi_2,
          and_530_nl);
      output1_rsci_idat_47_36 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_47_36_lpi_2,
          and_536_nl);
      output1_rsci_idat_65_54 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_65_54_lpi_2,
          and_542_nl);
      output1_rsci_idat_83_72 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_83_72_lpi_2,
          and_548_nl);
      output1_rsci_idat_101_90 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_101_90_lpi_2,
          and_554_nl);
      output1_rsci_idat_119_108 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_119_108_lpi_2,
          and_560_nl);
      output1_rsci_idat_137_126 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_137_126_lpi_2,
          and_566_nl);
      output1_rsci_idat_155_144 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_155_144_lpi_2,
          and_572_nl);
      output1_rsci_idat_173_162 <= MUX_v_12_2_2(CALC_SOFTMAX_LOOP_or_psp_sva_1, OUTPUT_LOOP_io_read_output1_rsc_sdt_173_162_lpi_2,
          and_578_nl);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_32_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_1_cse ) begin
      layer3_out_32_16_0_sva_dfm <= mux_149_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_1_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_2_cse ) begin
      layer3_out_1_16_0_sva_dfm <= mux_150_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_33_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_3_cse ) begin
      layer3_out_33_16_0_sva_dfm <= mux_151_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_2_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_4_cse ) begin
      layer3_out_2_16_0_sva_dfm <= mux_152_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_34_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_5_cse ) begin
      layer3_out_34_16_0_sva_dfm <= mux_153_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_3_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_6_cse ) begin
      layer3_out_3_16_0_sva_dfm <= mux_154_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_35_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_7_cse ) begin
      layer3_out_35_16_0_sva_dfm <= mux_155_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_4_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_8_cse ) begin
      layer3_out_4_16_0_sva_dfm <= mux_156_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_36_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_9_cse ) begin
      layer3_out_36_16_0_sva_dfm <= mux_157_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_5_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_10_cse ) begin
      layer3_out_5_16_0_sva_dfm <= mux_158_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_37_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_11_cse ) begin
      layer3_out_37_16_0_sva_dfm <= mux_159_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_6_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_12_cse ) begin
      layer3_out_6_16_0_sva_dfm <= mux_160_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_38_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_13_cse ) begin
      layer3_out_38_16_0_sva_dfm <= mux_161_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_7_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_14_cse ) begin
      layer3_out_7_16_0_sva_dfm <= mux_162_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_39_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_15_cse ) begin
      layer3_out_39_16_0_sva_dfm <= mux_163_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_8_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_16_cse ) begin
      layer3_out_8_16_0_sva_dfm <= mux_164_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_40_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_17_cse ) begin
      layer3_out_40_16_0_sva_dfm <= mux_165_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_9_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_18_cse ) begin
      layer3_out_9_16_0_sva_dfm <= mux_166_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_41_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_19_cse ) begin
      layer3_out_41_16_0_sva_dfm <= mux_167_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_10_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_20_cse ) begin
      layer3_out_10_16_0_sva_dfm <= mux_168_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_42_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_21_cse ) begin
      layer3_out_42_16_0_sva_dfm <= mux_169_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_11_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_22_cse ) begin
      layer3_out_11_16_0_sva_dfm <= mux_170_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_43_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_23_cse ) begin
      layer3_out_43_16_0_sva_dfm <= mux_171_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_12_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_24_cse ) begin
      layer3_out_12_16_0_sva_dfm <= mux_172_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_44_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_25_cse ) begin
      layer3_out_44_16_0_sva_dfm <= mux_173_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_13_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_26_cse ) begin
      layer3_out_13_16_0_sva_dfm <= mux_174_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_45_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_27_cse ) begin
      layer3_out_45_16_0_sva_dfm <= mux_175_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_14_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_28_cse ) begin
      layer3_out_14_16_0_sva_dfm <= mux_176_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_46_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_29_cse ) begin
      layer3_out_46_16_0_sva_dfm <= mux_177_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_15_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_30_cse ) begin
      layer3_out_15_16_0_sva_dfm <= mux_178_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_47_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_31_cse ) begin
      layer3_out_47_16_0_sva_dfm <= mux_179_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_16_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_32_cse ) begin
      layer3_out_16_16_0_sva_dfm <= mux_180_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_48_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_33_cse ) begin
      layer3_out_48_16_0_sva_dfm <= mux_181_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_17_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_34_cse ) begin
      layer3_out_17_16_0_sva_dfm <= mux_182_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_49_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_35_cse ) begin
      layer3_out_49_16_0_sva_dfm <= mux_183_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_18_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_36_cse ) begin
      layer3_out_18_16_0_sva_dfm <= mux_184_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_50_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_37_cse ) begin
      layer3_out_50_16_0_sva_dfm <= mux_185_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_19_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_38_cse ) begin
      layer3_out_19_16_0_sva_dfm <= mux_186_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_51_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_39_cse ) begin
      layer3_out_51_16_0_sva_dfm <= mux_187_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_20_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_40_cse ) begin
      layer3_out_20_16_0_sva_dfm <= mux_188_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_52_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_41_cse ) begin
      layer3_out_52_16_0_sva_dfm <= mux_189_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_21_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_42_cse ) begin
      layer3_out_21_16_0_sva_dfm <= mux_190_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_53_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_43_cse ) begin
      layer3_out_53_16_0_sva_dfm <= mux_191_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_22_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_44_cse ) begin
      layer3_out_22_16_0_sva_dfm <= mux_192_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_54_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_45_cse ) begin
      layer3_out_54_16_0_sva_dfm <= mux_193_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_23_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_46_cse ) begin
      layer3_out_23_16_0_sva_dfm <= mux_194_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_55_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_47_cse ) begin
      layer3_out_55_16_0_sva_dfm <= mux_195_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_24_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_48_cse ) begin
      layer3_out_24_16_0_sva_dfm <= mux_196_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_56_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_49_cse ) begin
      layer3_out_56_16_0_sva_dfm <= mux_197_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_25_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_50_cse ) begin
      layer3_out_25_16_0_sva_dfm <= mux_198_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_57_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_51_cse ) begin
      layer3_out_57_16_0_sva_dfm <= mux_199_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_26_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_52_cse ) begin
      layer3_out_26_16_0_sva_dfm <= mux_200_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_58_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_53_cse ) begin
      layer3_out_58_16_0_sva_dfm <= mux_201_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_27_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_54_cse ) begin
      layer3_out_27_16_0_sva_dfm <= mux_202_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_59_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_55_cse ) begin
      layer3_out_59_16_0_sva_dfm <= mux_203_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_28_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_56_cse ) begin
      layer3_out_28_16_0_sva_dfm <= mux_204_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_60_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_57_cse ) begin
      layer3_out_60_16_0_sva_dfm <= mux_205_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_29_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_58_cse ) begin
      layer3_out_29_16_0_sva_dfm <= mux_206_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_61_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_59_cse ) begin
      layer3_out_61_16_0_sva_dfm <= mux_207_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_30_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_60_cse ) begin
      layer3_out_30_16_0_sva_dfm <= mux_208_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_62_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_61_cse ) begin
      layer3_out_62_16_0_sva_dfm <= mux_209_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_31_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer3_out_and_cse & layer3_out_or_62_cse ) begin
      layer3_out_31_16_0_sva_dfm <= mux_210_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_32_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_1_cse ) begin
      layer5_out_32_16_0_sva_dfm <= mux_149_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_1_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_2_cse ) begin
      layer5_out_1_16_0_sva_dfm <= mux_150_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_33_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_3_cse ) begin
      layer5_out_33_16_0_sva_dfm <= mux_151_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_2_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_4_cse ) begin
      layer5_out_2_16_0_sva_dfm <= mux_152_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_34_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_5_cse ) begin
      layer5_out_34_16_0_sva_dfm <= mux_153_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_3_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_6_cse ) begin
      layer5_out_3_16_0_sva_dfm <= mux_154_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_35_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_7_cse ) begin
      layer5_out_35_16_0_sva_dfm <= mux_155_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_4_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_8_cse ) begin
      layer5_out_4_16_0_sva_dfm <= mux_156_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_36_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_9_cse ) begin
      layer5_out_36_16_0_sva_dfm <= mux_157_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_5_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_10_cse ) begin
      layer5_out_5_16_0_sva_dfm <= mux_158_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_37_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_11_cse ) begin
      layer5_out_37_16_0_sva_dfm <= mux_159_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_6_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_12_cse ) begin
      layer5_out_6_16_0_sva_dfm <= mux_160_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_38_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_13_cse ) begin
      layer5_out_38_16_0_sva_dfm <= mux_161_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_7_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_14_cse ) begin
      layer5_out_7_16_0_sva_dfm <= mux_162_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_39_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_15_cse ) begin
      layer5_out_39_16_0_sva_dfm <= mux_163_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_8_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_16_cse ) begin
      layer5_out_8_16_0_sva_dfm <= mux_164_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_40_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_17_cse ) begin
      layer5_out_40_16_0_sva_dfm <= mux_165_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_9_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_18_cse ) begin
      layer5_out_9_16_0_sva_dfm <= mux_166_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_41_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_19_cse ) begin
      layer5_out_41_16_0_sva_dfm <= mux_167_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_10_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_20_cse ) begin
      layer5_out_10_16_0_sva_dfm <= mux_168_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_42_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_21_cse ) begin
      layer5_out_42_16_0_sva_dfm <= mux_169_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_11_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_22_cse ) begin
      layer5_out_11_16_0_sva_dfm <= mux_170_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_43_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_23_cse ) begin
      layer5_out_43_16_0_sva_dfm <= mux_171_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_12_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_24_cse ) begin
      layer5_out_12_16_0_sva_dfm <= mux_172_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_44_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_25_cse ) begin
      layer5_out_44_16_0_sva_dfm <= mux_173_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_13_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_26_cse ) begin
      layer5_out_13_16_0_sva_dfm <= mux_174_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_45_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_27_cse ) begin
      layer5_out_45_16_0_sva_dfm <= mux_175_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_14_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_28_cse ) begin
      layer5_out_14_16_0_sva_dfm <= mux_176_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_46_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_29_cse ) begin
      layer5_out_46_16_0_sva_dfm <= mux_177_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_15_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_30_cse ) begin
      layer5_out_15_16_0_sva_dfm <= mux_178_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_47_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_31_cse ) begin
      layer5_out_47_16_0_sva_dfm <= mux_179_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_16_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_32_cse ) begin
      layer5_out_16_16_0_sva_dfm <= mux_180_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_48_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_33_cse ) begin
      layer5_out_48_16_0_sva_dfm <= mux_181_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_17_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_34_cse ) begin
      layer5_out_17_16_0_sva_dfm <= mux_182_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_49_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_35_cse ) begin
      layer5_out_49_16_0_sva_dfm <= mux_183_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_18_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_36_cse ) begin
      layer5_out_18_16_0_sva_dfm <= mux_184_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_50_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_37_cse ) begin
      layer5_out_50_16_0_sva_dfm <= mux_185_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_19_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_38_cse ) begin
      layer5_out_19_16_0_sva_dfm <= mux_186_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_51_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_39_cse ) begin
      layer5_out_51_16_0_sva_dfm <= mux_187_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_20_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_40_cse ) begin
      layer5_out_20_16_0_sva_dfm <= mux_188_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_52_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_41_cse ) begin
      layer5_out_52_16_0_sva_dfm <= mux_189_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_21_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_42_cse ) begin
      layer5_out_21_16_0_sva_dfm <= mux_190_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_53_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_43_cse ) begin
      layer5_out_53_16_0_sva_dfm <= mux_191_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_22_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_44_cse ) begin
      layer5_out_22_16_0_sva_dfm <= mux_192_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_54_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_45_cse ) begin
      layer5_out_54_16_0_sva_dfm <= mux_193_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_23_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_46_cse ) begin
      layer5_out_23_16_0_sva_dfm <= mux_194_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_55_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_47_cse ) begin
      layer5_out_55_16_0_sva_dfm <= mux_195_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_24_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_48_cse ) begin
      layer5_out_24_16_0_sva_dfm <= mux_196_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_56_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_49_cse ) begin
      layer5_out_56_16_0_sva_dfm <= mux_197_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_25_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_50_cse ) begin
      layer5_out_25_16_0_sva_dfm <= mux_198_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_57_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_51_cse ) begin
      layer5_out_57_16_0_sva_dfm <= mux_199_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_26_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_52_cse ) begin
      layer5_out_26_16_0_sva_dfm <= mux_200_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_58_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_53_cse ) begin
      layer5_out_58_16_0_sva_dfm <= mux_201_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_27_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_54_cse ) begin
      layer5_out_27_16_0_sva_dfm <= mux_202_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_59_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_55_cse ) begin
      layer5_out_59_16_0_sva_dfm <= mux_203_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_28_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_56_cse ) begin
      layer5_out_28_16_0_sva_dfm <= mux_204_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_60_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_57_cse ) begin
      layer5_out_60_16_0_sva_dfm <= mux_205_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_29_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_58_cse ) begin
      layer5_out_29_16_0_sva_dfm <= mux_206_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_61_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_59_cse ) begin
      layer5_out_61_16_0_sva_dfm <= mux_207_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_30_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_60_cse ) begin
      layer5_out_30_16_0_sva_dfm <= mux_208_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_62_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_61_cse ) begin
      layer5_out_62_16_0_sva_dfm <= mux_209_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer5_out_31_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( layer5_out_and_cse & layer3_out_or_62_cse ) begin
      layer5_out_31_16_0_sva_dfm <= mux_210_cse;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_118) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_566) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_568) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_571) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_572) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_573) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_574) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_576) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~(or_dcpl_154 | or_dcpl_117)) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0100) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0101) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0011) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0110) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0010) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0111) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b0001) & (fsm_output[11]))
        | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_417_cse & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3:2]==2'b10)
        & (fsm_output[11])) | and_3282_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4, and_3282_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_31_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_582) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_31_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_32_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_587) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_32_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_30_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_589) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_30_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_33_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_591) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_33_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_29_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_594) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_29_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_34_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_597) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_34_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_28_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_598) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_28_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_35_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_599) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_35_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_27_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_601) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_27_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_36_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_603) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_36_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_26_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_604) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_26_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_37_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_605) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_37_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_25_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_607) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_25_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_38_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_609) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_38_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_24_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_610) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_24_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_39_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_611) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_39_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_23_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_612) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_23_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_40_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_613) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_40_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_22_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_614) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_22_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_41_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_615) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_41_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_21_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_616) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_21_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_42_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_617) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_42_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_20_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_618) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_20_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_43_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_619) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_43_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_19_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_620) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_19_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_44_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_621) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_44_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_18_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_622) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_18_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_45_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_623) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_45_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_17_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_624) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_17_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_46_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_625) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_46_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_16_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_626) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_16_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_47_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_627) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_47_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_15_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_630) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_15_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_48_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_633) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_48_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_14_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_635) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_14_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_49_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_637) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_49_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_13_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_638) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_13_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_50_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_639) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_50_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_12_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_640) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_12_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_51_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_641) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_51_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_11_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_642) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_11_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_52_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_643) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_52_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_10_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_644) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_10_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_53_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_645) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_53_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_9_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_646) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_9_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_54_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_647) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_54_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_8_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_648) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_8_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_55_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_649) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_55_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_7_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_650) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_7_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_56_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_651) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_56_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_6_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_652) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_6_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_57_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_653) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_57_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_5_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_654) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_5_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_58_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_655) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_58_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_4_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_656) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_4_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_59_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_657) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_59_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_3_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_658) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_3_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_60_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_659) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_60_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_2_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_660) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_2_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_61_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_661) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_61_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_1_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_662) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_1_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_62_sva_1 <= 18'b000000000000000000;
    end
    else if ( nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_and_cse &
        ((~ or_dcpl_663) | or_tmp_318) ) begin
      nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_62_sva_1 <= MUX_v_18_2_2(InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4, or_tmp_318);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011111) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b10) & nor_415_cse & nor_416_cse
        & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((~ (InitAccum_1_iacc_6_0_sva_5_0[4])) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[2]))
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011101) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b10) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[3])) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011100) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b100011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((~ (InitAccum_1_iacc_6_0_sva_5_0[4])) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[0])) & (InitAccum_1_iacc_6_0_sva_5_0[2])
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b01) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((~ (InitAccum_1_iacc_6_0_sva_5_0[4])) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (InitAccum_1_iacc_6_0_sva_5_0[2]) &
        nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b011001) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b100110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b01) & nor_415_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[1]))
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b100111) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b010111) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b10) & nor_415_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[1]))
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b010110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101001) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[5]))
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (InitAccum_1_iacc_6_0_sva_5_0[2]) &
        nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b10) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[5]))
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[0])) & (InitAccum_1_iacc_6_0_sva_5_0[2])
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b010011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101100) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b01) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[3])) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101101) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[5]))
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[2]))
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b01) & nor_415_cse & nor_416_cse
        & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b101111) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1111) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11) & nor_415_cse & nor_416_cse
        & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1110) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[2]))
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1101) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[3])) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1100) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b110011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1011) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (~ (InitAccum_1_iacc_6_0_sva_5_0[0])) & (InitAccum_1_iacc_6_0_sva_5_0[2])
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1]) & (InitAccum_1_iacc_6_0_sva_5_0[3])
        & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[4]) & (InitAccum_1_iacc_6_0_sva_5_0[5])
        & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (InitAccum_1_iacc_6_0_sva_5_0[2]) &
        nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b1001) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b110110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & nor_415_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[1]))
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b110111) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b0111) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11) & nor_415_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[1]))
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b0110) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b111001) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (InitAccum_1_iacc_6_0_sva_5_0[2])
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0[5:4]==2'b11) & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1])
        & (InitAccum_1_iacc_6_0_sva_5_0[3]) & (fsm_output[2])) | and_3330_cse) &
        core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (~ (InitAccum_1_iacc_6_0_sva_5_0[0])) & (InitAccum_1_iacc_6_0_sva_5_0[2])
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b111011) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[3:0]==4'b0011) & (fsm_output[2]))
        | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b111100) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & nor_415_cse & (InitAccum_1_iacc_6_0_sva_5_0[1]) & (~
        (InitAccum_1_iacc_6_0_sva_5_0[3])) & (fsm_output[2])) | and_3330_cse) & core_wen
        ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b111101) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1 <= 18'b000000000000000000;
    end
    else if ( ((nor_400_cse & (InitAccum_1_iacc_6_0_sva_5_0[0]) & (~ (InitAccum_1_iacc_6_0_sva_5_0[2]))
        & nor_416_cse & (fsm_output[2])) | and_3330_cse) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1 <= 18'b000000000000000000;
    end
    else if ( (((InitAccum_1_iacc_6_0_sva_5_0==6'b111110) & (fsm_output[2])) | and_3330_cse)
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1 <= MUX_v_18_2_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1_mx0w0,
          nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4, and_3330_cse);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_101_90_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_556) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_101_90_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_119_108_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_557) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_119_108_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_11_0_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~((~ (fsm_output[17])) | OUTPUT_LOOP_or_tmp)) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_11_0_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_137_126_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_558) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_137_126_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_155_144_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_560) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_155_144_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_173_162_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_561) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_173_162_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_29_18_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_548) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_29_18_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_47_36_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_551) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_47_36_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_65_54_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_552) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_65_54_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_83_72_lpi_2 <= 12'b000000000000;
    end
    else if ( core_wen & (~ or_dcpl_554) & (fsm_output[17]) ) begin
      OUTPUT_LOOP_io_read_output1_rsc_sdt_83_72_lpi_2 <= CALC_SOFTMAX_LOOP_or_psp_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      InitAccum_1_iacc_6_0_sva_5_0 <= 6'b000000;
    end
    else if ( core_wen & ((fsm_output[4]) | (fsm_output[14]) | InitAccum_1_iacc_6_0_sva_5_0_mx0c0
        | InitAccum_1_iacc_6_0_sva_5_0_mx0c2 | InitAccum_1_iacc_6_0_sva_5_0_mx0c3)
        ) begin
      InitAccum_1_iacc_6_0_sva_5_0 <= MUX_v_6_2_2(6'b000000, (ReuseLoop_in_index_mux1h_nl),
          (InitAccum_1_iacc_not_nl));
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_0_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (~((~((z_out_5[6]) & (z_out_6[6]) & (z_out_7[6]) & (fsm_output[6])))
        & (mux_273_nl))) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_0_lpi_4 <= MUX1HOT_v_18_5_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          MultLoop_1_mux_64_itm, InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1,
          (z_out_4[17:0]), {(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_62_nl)
          , (and_1844_nl) , (and_1846_nl) , (nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_65_nl)
          , (or_2584_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_63_lpi_4 <= 18'b000000000000000000;
    end
    else if ( core_wen & (~(or_dcpl_670 | (fsm_output[10]) | (fsm_output[9]) | (fsm_output[5])
        | and_586_cse | ((~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11) & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp==4'b1111)))
        & or_2157_cse))) ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_63_lpi_4 <= MUX1HOT_v_18_3_2(InitAccum_slc_InitAccum_iacc_slc_InitAccum_iacc_6_0_5_0_1_18_17_0_ctmp_sva_1,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          (z_out_4[17:0]), {(nor_523_nl) , and_1829_cse , or_2579_tmp});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3])
        & (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11) & (~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]))))
        | nand_62_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_62_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_62_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1884_nl) , (and_1886_nl) , (and_4027_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp!=4'b0001)
        | (~((fsm_output[12]) | nor_517_cse)))) | (fsm_output[2]) | (fsm_output[11]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_1_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_1_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_1899_nl) , (and_1901_nl) , (fsm_output[11])
          , (or_2586_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3])
        & (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11) & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]==2'b01)))
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_61_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_61_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1917_nl) , (and_1919_nl) , (and_4032_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_2_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_2_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1931_nl) , (and_1933_nl) , (and_4034_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00)
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_60_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_60_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1945_nl) , (and_1947_nl) , (and_4036_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_3_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_3_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1959_nl) , (and_1961_nl) , (and_4038_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3])
        & (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11) & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b011)))
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_59_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_59_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1973_nl) , (and_1975_nl) , (and_4040_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_4_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_4_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_1987_nl) , (and_1989_nl) , (and_4042_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_58_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_58_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2001_nl) , (and_2003_nl) , (and_4044_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_5_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_5_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2015_nl) , (and_2017_nl) , (and_4046_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_57_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_57_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2029_nl) , (and_2031_nl) , (and_4048_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]) | nand_62_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_6_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_6_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2043_nl) , (and_2045_nl) , (and_4050_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_56_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_56_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2057_nl) , (and_2059_nl) , (and_4052_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00)
        | nand_53_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_7_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_7_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2071_nl) , (and_2073_nl) , (and_4054_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_tmp==2'b11)
        & (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b111) & or_2157_cse))))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_55_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_55_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2085_nl) , (and_2087_nl) , (and_4056_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_8_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_8_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2099_nl) , (and_2101_nl) , (and_4058_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]) | nand_62_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_54_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_54_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2113_nl) , (and_2115_nl) , (and_4060_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_9_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_9_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2127_nl) , (and_2129_nl) , (and_4062_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_53_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_53_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2141_nl) , (and_2143_nl) , (and_4064_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | ((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010)))
        & mux_274_cse) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_10_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_2_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2156_nl) , (and_2158_nl) , (fsm_output[11])
          , (or_2604_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_52_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_52_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2174_nl) , (and_2176_nl) , (and_4069_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | ((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011)))
        & mux_274_cse) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_11_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_3_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2189_nl) , (and_2191_nl) , (fsm_output[11])
          , (or_2606_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_51_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_51_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2207_nl) , (and_2209_nl) , (and_4074_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b100)
        | (mux_276_nl))) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_12_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_4_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2222_nl) , (and_2224_nl) , (fsm_output[11])
          , (or_2608_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_50_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_50_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2240_nl) , (and_2242_nl) , (and_4079_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | ((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b101)
        & mux_274_cse) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_13_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_5_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2255_nl) , (and_2257_nl) , (fsm_output[11])
          , (or_2610_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_49_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_49_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2273_nl) , (and_2275_nl) , (and_4084_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | ((~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b110)))
        & mux_274_cse) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_14_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_6_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2288_nl) , (and_2290_nl) , (fsm_output[11])
          , (or_2612_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b11)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_48_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_48_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2306_nl) , (and_2308_nl) , (and_4089_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | ((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]==3'b111)
        & mux_274_cse) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_15_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_7_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2321_nl) , (and_2323_nl) , (fsm_output[11])
          , (or_2614_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | nand_53_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_47_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_47_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2339_nl) , (and_2341_nl) , (and_4094_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000)
        | (~ mux_280_cse))) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_16_sva_1, nnet_dense_large_rf_leq_nin_layer5_t_layer6_t_config6_acc_8_sva_1_mx0w0,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2354_nl) , (and_2356_nl) , (fsm_output[11])
          , (or_2616_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0])
        | nand_62_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_46_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_46_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2372_nl) , (and_2374_nl) , (and_4099_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001)
        | (~ mux_280_cse))) | (fsm_output[2]) | (fsm_output[11])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4 <= MUX1HOT_v_18_5_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_17_sva_1, InitAccum_2_slc_InitAccum_2_asn_18_17_0_ctmp_sva_1,
          (z_out_4[17:0]), {(fsm_output[2]) , (and_2387_nl) , (and_2389_nl) , (fsm_output[11])
          , (or_2618_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01)
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_45_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_45_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2405_nl) , (and_2407_nl) , (and_4104_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_18_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_18_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2419_nl) , (and_2421_nl) , (and_4106_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00)
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_44_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_44_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2433_nl) , (and_2435_nl) , (and_4108_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_19_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_19_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2447_nl) , (and_2449_nl) , (and_4110_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_43_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_43_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2461_nl) , (and_2463_nl) , (and_4112_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_20_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_20_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2475_nl) , (and_2477_nl) , (and_4114_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_42_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_42_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2489_nl) , (and_2491_nl) , (and_4116_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_21_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_21_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2503_nl) , (and_2505_nl) , (and_4118_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_41_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_41_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2517_nl) , (and_2519_nl) , (and_4120_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]) | nand_62_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_22_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_22_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2531_nl) , (and_2533_nl) , (and_4122_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_40_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_40_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2545_nl) , (and_2547_nl) , (and_4124_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp[1])
        | nand_42_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_23_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_23_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2559_nl) , (and_2561_nl) , (and_4126_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | nand_53_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_39_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_39_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2573_nl) , (and_2575_nl) , (and_4128_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_24_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_24_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2587_nl) , (and_2589_nl) , (and_4130_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0]) | nand_62_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_38_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_38_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2601_nl) , (and_2603_nl) , (and_4132_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_25_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_25_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2615_nl) , (and_2617_nl) , (and_4134_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_37_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_37_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2629_nl) , (and_2631_nl) , (and_4136_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_26_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_26_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2643_nl) , (and_2645_nl) , (and_4138_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00) | nand_60_cse)) |
        (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_36_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_36_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2657_nl) , (and_2659_nl) , (and_4140_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011)
        | nor_518_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_27_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_27_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2671_nl) , (and_2673_nl) , (and_4142_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b011) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_35_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_35_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2685_nl) , (and_2687_nl) , (and_4144_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b00)
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_28_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_28_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2699_nl) , (and_2701_nl) , (and_4146_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b010) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_34_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_34_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2713_nl) , (and_2715_nl) , (and_4148_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[1:0]!=2'b01)
        | nand_60_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_29_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_29_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2727_nl) , (and_2729_nl) , (and_4150_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b001) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_33_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_33_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2741_nl) , (and_2743_nl) , (and_4152_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b01) | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[0])
        | nand_62_cse)) | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_30_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_30_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2755_nl) , (and_2757_nl) , (and_4154_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]) | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b10)
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[2:0]!=3'b000) | nor_518_cse))
        | (fsm_output[2])) & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_32_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_32_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2769_nl) , (and_2771_nl) , (and_4156_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4 <= 18'b000000000000000000;
    end
    else if ( (and_4008_cse | (~((~ (reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]))
        | (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp[1]) | nand_42_cse)) | (fsm_output[2]))
        & core_wen ) begin
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_lpi_4 <= MUX1HOT_v_18_4_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_31_sva_1_mx0w0,
          InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          nnet_dense_large_rf_leq_nin_layer3_t_layer4_t_config4_acc_31_sva_1, (z_out_4[17:0]),
          {(fsm_output[2]) , (and_2783_nl) , (and_2785_nl) , (and_4158_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp <= 4'b0000;
    end
    else if ( ((mux_282_nl) | (fsm_output[17]) | (fsm_output[15]) | (fsm_output[11])
        | (fsm_output[12]) | (fsm_output[4]) | (fsm_output[8]) | (fsm_output[6])
        | (fsm_output[5]) | (fsm_output[9]) | (fsm_output[2])) & core_wen ) begin
      reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp <= MUX_v_4_2_2(4'b0000, (InitAccum_1_iacc_mux1h_3_nl),
          (nor_521_nl));
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      reg_ResultLoop_1_ires_6_0_sva_5_0_tmp <= 2'b00;
    end
    else if ( ((fsm_output[17]) | (fsm_output[15]) | (fsm_output[11]) | (fsm_output[16])
        | (fsm_output[12]) | (fsm_output[13]) | (fsm_output[18]) | (fsm_output[14])
        | (fsm_output[10]) | (fsm_output[4]) | (fsm_output[8]) | (fsm_output[6])
        | (fsm_output[5]) | (fsm_output[9]) | (fsm_output[2])) & core_wen ) begin
      reg_ResultLoop_1_ires_6_0_sva_5_0_tmp <= MUX_v_2_2_2(2'b00, (InitAccum_1_iacc_InitAccum_1_iacc_mux_nl),
          (not_1748_nl));
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      MultLoop_1_if_1_acc_itm_5_4 <= 2'b00;
    end
    else if ( ((fsm_output[18:2]!=17'b00000000000000000)) & core_wen ) begin
      MultLoop_1_if_1_acc_itm_5_4 <= InitAccum_1_iacc_and_2_rgt[5:4];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      MultLoop_1_if_1_acc_itm_3_0 <= 4'b0000;
    end
    else if ( (and_3921_cse | (fsm_output[10:2]!=9'b000000000)) & core_wen ) begin
      MultLoop_1_if_1_acc_itm_3_0 <= InitAccum_1_iacc_and_2_rgt[3:0];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      reg_ReuseLoop_1_w_index_11_6_reg <= 2'b00;
    end
    else if ( ((fsm_output[12]) | (fsm_output[13]) | (fsm_output[16]) | (fsm_output[14])
        | (fsm_output[11]) | (fsm_output[18]) | (fsm_output[15]) | (fsm_output[17])
        | (fsm_output[7]) | (fsm_output[4]) | (fsm_output[9]) | (fsm_output[6]) |
        (fsm_output[5]) | (fsm_output[2])) & core_wen ) begin
      reg_ReuseLoop_1_w_index_11_6_reg <= InitAccum_1_iacc_and_1_rgt[5:4];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      reg_ReuseLoop_1_w_index_11_6_1_reg <= 4'b0000;
    end
    else if ( (and_3921_cse | (fsm_output[7]) | (fsm_output[4]) | (fsm_output[9])
        | (fsm_output[6]) | (fsm_output[5]) | (fsm_output[2])) & core_wen ) begin
      reg_ReuseLoop_1_w_index_11_6_1_reg <= InitAccum_1_iacc_and_1_rgt[3:0];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_63_16_0_lpi_2_dfm_16_10 <= 7'b0000000;
    end
    else if ( and_3931_cse & core_wen ) begin
      layer3_out_63_16_0_lpi_2_dfm_16_10 <= layer3_out_layer3_out_mux_rgt[16:10];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_63_16_0_lpi_2_dfm_9_0 <= 10'b0000000000;
    end
    else if ( and_3931_cse & (fsm_output[4:3]==2'b00) & core_wen ) begin
      layer3_out_63_16_0_lpi_2_dfm_9_0 <= layer3_out_layer3_out_mux_rgt[9:0];
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      MultLoop_1_mux_64_itm <= 18'b000000000000000000;
    end
    else if ( core_wen & ((~ (fsm_output[6])) | MultLoop_and_rgt) ) begin
      MultLoop_1_mux_64_itm <= MUX_v_18_2_2(MultLoop_mux_64_itm_mx0w0, InitAccum_1_slc_InitAccum_1_iacc_slc_InitAccum_1_iacc_6_0_5_0_1_18_17_0_ctmp_sva_mx0w3,
          MultLoop_and_rgt);
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      layer3_out_0_16_0_sva_dfm <= 17'b00000000000000000;
    end
    else if ( core_wen & (~((mux_tmp & or_tmp_1141) | (fsm_output[7]) | (fsm_output[8])
        | (fsm_output[9]) | (fsm_output[11]) | (fsm_output[12]) | (fsm_output[13])))
        ) begin
      layer3_out_0_16_0_sva_dfm <= MUX1HOT_v_17_3_2(({5'b00000 , (z_out_4[11:0])}),
          (signext_17_1(nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nand_63_nl)),
          (MultLoop_mux_64_itm_mx0w0[16:0]), {(fsm_output[3]) , (layer3_out_and_62_nl)
          , (layer3_out_and_63_nl)});
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1
          <= 67'b0000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (~ or_dcpl_107) & (fsm_output[15]) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1
          <= operator_67_47_false_AC_TRN_AC_WRAP_lshift_ncse_sva_1;
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva
          <= 71'b00000000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & ((fsm_output[15]) | (fsm_output[13])) ) begin
      ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva
          <= MUX_v_71_2_2(71'b00000000000000000000000000000000000000000000000000000000000000000000000,
          ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1,
          (fsm_output[15]));
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva <= 4'b0000;
    end
    else if ( (~ (fsm_output[15])) & core_wen ) begin
      nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva <= MUX_v_4_2_2(4'b0000,
          (z_out_4[3:0]), (not_nl));
    end
  end
  always @(posedge clk) begin
    if ( rst ) begin
      ac_math_ac_reciprocal_pwl_AC_TRN_71_51_false_AC_TRN_AC_WRAP_91_21_false_AC_TRN_AC_WRAP_output_temp_lpi_1_dfm
          <= 91'b0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
    end
    else if ( core_wen & (fsm_output[18:17]==2'b00) ) begin
      ac_math_ac_reciprocal_pwl_AC_TRN_71_51_false_AC_TRN_AC_WRAP_91_21_false_AC_TRN_AC_WRAP_output_temp_lpi_1_dfm
          <= MUX_v_91_2_2(operator_91_21_false_AC_TRN_AC_WRAP_rshift_itm, 91'b1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111,
          CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_itm);
    end
  end
  assign ReuseLoop_ir_ReuseLoop_ir_and_2_nl = MUX_v_4_2_2(4'b0000, (z_out_3[3:0]),
      (fsm_output[5]));
  assign ReuseLoop_in_index_mux1h_1_nl = MUX1HOT_v_6_4_2(ReuseLoop_ir_9_0_sva_mx0_tmp_9_4,
      (layer3_out_63_16_0_lpi_2_dfm_9_0[9:4]), InitAccum_1_iacc_6_0_sva_5_0, (z_out_7[5:0]),
      {or_dcpl_768 , (fsm_output[4]) , (fsm_output[12]) , (fsm_output[13])});
  assign or_1973_nl = (fsm_output[4]) | (fsm_output[2]) | (fsm_output[12]) | (fsm_output[13])
      | (fsm_output[5]);
  assign ReuseLoop_2_in_index_asn_MultLoop_1_im_6_0_sva_2_5_ReuseLoop_in_index_and_nl
      = MUX_v_6_2_2(6'b000000, (ReuseLoop_in_index_mux1h_1_nl), (or_1973_nl));
  assign or_1936_nl = (fsm_output[4]) | (fsm_output[2]) | (fsm_output[11]) | or_dcpl_771
      | (fsm_output[5]);
  assign MultLoop_1_im_and_nl = (ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]==2'b00)
      & (fsm_output[14]);
  assign MultLoop_1_im_and_1_nl = (ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]==2'b01)
      & (fsm_output[14]);
  assign MultLoop_1_im_and_2_nl = (ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]==2'b10)
      & (fsm_output[14]);
  assign MultLoop_1_im_and_3_nl = (ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]==2'b11)
      & (fsm_output[14]);
  assign MultLoop_mux_65_nl = MUX_v_18_784_2((input1_rsci_idat_mxwt[17:0]), (input1_rsci_idat_mxwt[35:18]),
      (input1_rsci_idat_mxwt[53:36]), (input1_rsci_idat_mxwt[71:54]), (input1_rsci_idat_mxwt[89:72]),
      (input1_rsci_idat_mxwt[107:90]), (input1_rsci_idat_mxwt[125:108]), (input1_rsci_idat_mxwt[143:126]),
      (input1_rsci_idat_mxwt[161:144]), (input1_rsci_idat_mxwt[179:162]), (input1_rsci_idat_mxwt[197:180]),
      (input1_rsci_idat_mxwt[215:198]), (input1_rsci_idat_mxwt[233:216]), (input1_rsci_idat_mxwt[251:234]),
      (input1_rsci_idat_mxwt[269:252]), (input1_rsci_idat_mxwt[287:270]), (input1_rsci_idat_mxwt[305:288]),
      (input1_rsci_idat_mxwt[323:306]), (input1_rsci_idat_mxwt[341:324]), (input1_rsci_idat_mxwt[359:342]),
      (input1_rsci_idat_mxwt[377:360]), (input1_rsci_idat_mxwt[395:378]), (input1_rsci_idat_mxwt[413:396]),
      (input1_rsci_idat_mxwt[431:414]), (input1_rsci_idat_mxwt[449:432]), (input1_rsci_idat_mxwt[467:450]),
      (input1_rsci_idat_mxwt[485:468]), (input1_rsci_idat_mxwt[503:486]), (input1_rsci_idat_mxwt[521:504]),
      (input1_rsci_idat_mxwt[539:522]), (input1_rsci_idat_mxwt[557:540]), (input1_rsci_idat_mxwt[575:558]),
      (input1_rsci_idat_mxwt[593:576]), (input1_rsci_idat_mxwt[611:594]), (input1_rsci_idat_mxwt[629:612]),
      (input1_rsci_idat_mxwt[647:630]), (input1_rsci_idat_mxwt[665:648]), (input1_rsci_idat_mxwt[683:666]),
      (input1_rsci_idat_mxwt[701:684]), (input1_rsci_idat_mxwt[719:702]), (input1_rsci_idat_mxwt[737:720]),
      (input1_rsci_idat_mxwt[755:738]), (input1_rsci_idat_mxwt[773:756]), (input1_rsci_idat_mxwt[791:774]),
      (input1_rsci_idat_mxwt[809:792]), (input1_rsci_idat_mxwt[827:810]), (input1_rsci_idat_mxwt[845:828]),
      (input1_rsci_idat_mxwt[863:846]), (input1_rsci_idat_mxwt[881:864]), (input1_rsci_idat_mxwt[899:882]),
      (input1_rsci_idat_mxwt[917:900]), (input1_rsci_idat_mxwt[935:918]), (input1_rsci_idat_mxwt[953:936]),
      (input1_rsci_idat_mxwt[971:954]), (input1_rsci_idat_mxwt[989:972]), (input1_rsci_idat_mxwt[1007:990]),
      (input1_rsci_idat_mxwt[1025:1008]), (input1_rsci_idat_mxwt[1043:1026]), (input1_rsci_idat_mxwt[1061:1044]),
      (input1_rsci_idat_mxwt[1079:1062]), (input1_rsci_idat_mxwt[1097:1080]), (input1_rsci_idat_mxwt[1115:1098]),
      (input1_rsci_idat_mxwt[1133:1116]), (input1_rsci_idat_mxwt[1151:1134]), (input1_rsci_idat_mxwt[1169:1152]),
      (input1_rsci_idat_mxwt[1187:1170]), (input1_rsci_idat_mxwt[1205:1188]), (input1_rsci_idat_mxwt[1223:1206]),
      (input1_rsci_idat_mxwt[1241:1224]), (input1_rsci_idat_mxwt[1259:1242]), (input1_rsci_idat_mxwt[1277:1260]),
      (input1_rsci_idat_mxwt[1295:1278]), (input1_rsci_idat_mxwt[1313:1296]), (input1_rsci_idat_mxwt[1331:1314]),
      (input1_rsci_idat_mxwt[1349:1332]), (input1_rsci_idat_mxwt[1367:1350]), (input1_rsci_idat_mxwt[1385:1368]),
      (input1_rsci_idat_mxwt[1403:1386]), (input1_rsci_idat_mxwt[1421:1404]), (input1_rsci_idat_mxwt[1439:1422]),
      (input1_rsci_idat_mxwt[1457:1440]), (input1_rsci_idat_mxwt[1475:1458]), (input1_rsci_idat_mxwt[1493:1476]),
      (input1_rsci_idat_mxwt[1511:1494]), (input1_rsci_idat_mxwt[1529:1512]), (input1_rsci_idat_mxwt[1547:1530]),
      (input1_rsci_idat_mxwt[1565:1548]), (input1_rsci_idat_mxwt[1583:1566]), (input1_rsci_idat_mxwt[1601:1584]),
      (input1_rsci_idat_mxwt[1619:1602]), (input1_rsci_idat_mxwt[1637:1620]), (input1_rsci_idat_mxwt[1655:1638]),
      (input1_rsci_idat_mxwt[1673:1656]), (input1_rsci_idat_mxwt[1691:1674]), (input1_rsci_idat_mxwt[1709:1692]),
      (input1_rsci_idat_mxwt[1727:1710]), (input1_rsci_idat_mxwt[1745:1728]), (input1_rsci_idat_mxwt[1763:1746]),
      (input1_rsci_idat_mxwt[1781:1764]), (input1_rsci_idat_mxwt[1799:1782]), (input1_rsci_idat_mxwt[1817:1800]),
      (input1_rsci_idat_mxwt[1835:1818]), (input1_rsci_idat_mxwt[1853:1836]), (input1_rsci_idat_mxwt[1871:1854]),
      (input1_rsci_idat_mxwt[1889:1872]), (input1_rsci_idat_mxwt[1907:1890]), (input1_rsci_idat_mxwt[1925:1908]),
      (input1_rsci_idat_mxwt[1943:1926]), (input1_rsci_idat_mxwt[1961:1944]), (input1_rsci_idat_mxwt[1979:1962]),
      (input1_rsci_idat_mxwt[1997:1980]), (input1_rsci_idat_mxwt[2015:1998]), (input1_rsci_idat_mxwt[2033:2016]),
      (input1_rsci_idat_mxwt[2051:2034]), (input1_rsci_idat_mxwt[2069:2052]), (input1_rsci_idat_mxwt[2087:2070]),
      (input1_rsci_idat_mxwt[2105:2088]), (input1_rsci_idat_mxwt[2123:2106]), (input1_rsci_idat_mxwt[2141:2124]),
      (input1_rsci_idat_mxwt[2159:2142]), (input1_rsci_idat_mxwt[2177:2160]), (input1_rsci_idat_mxwt[2195:2178]),
      (input1_rsci_idat_mxwt[2213:2196]), (input1_rsci_idat_mxwt[2231:2214]), (input1_rsci_idat_mxwt[2249:2232]),
      (input1_rsci_idat_mxwt[2267:2250]), (input1_rsci_idat_mxwt[2285:2268]), (input1_rsci_idat_mxwt[2303:2286]),
      (input1_rsci_idat_mxwt[2321:2304]), (input1_rsci_idat_mxwt[2339:2322]), (input1_rsci_idat_mxwt[2357:2340]),
      (input1_rsci_idat_mxwt[2375:2358]), (input1_rsci_idat_mxwt[2393:2376]), (input1_rsci_idat_mxwt[2411:2394]),
      (input1_rsci_idat_mxwt[2429:2412]), (input1_rsci_idat_mxwt[2447:2430]), (input1_rsci_idat_mxwt[2465:2448]),
      (input1_rsci_idat_mxwt[2483:2466]), (input1_rsci_idat_mxwt[2501:2484]), (input1_rsci_idat_mxwt[2519:2502]),
      (input1_rsci_idat_mxwt[2537:2520]), (input1_rsci_idat_mxwt[2555:2538]), (input1_rsci_idat_mxwt[2573:2556]),
      (input1_rsci_idat_mxwt[2591:2574]), (input1_rsci_idat_mxwt[2609:2592]), (input1_rsci_idat_mxwt[2627:2610]),
      (input1_rsci_idat_mxwt[2645:2628]), (input1_rsci_idat_mxwt[2663:2646]), (input1_rsci_idat_mxwt[2681:2664]),
      (input1_rsci_idat_mxwt[2699:2682]), (input1_rsci_idat_mxwt[2717:2700]), (input1_rsci_idat_mxwt[2735:2718]),
      (input1_rsci_idat_mxwt[2753:2736]), (input1_rsci_idat_mxwt[2771:2754]), (input1_rsci_idat_mxwt[2789:2772]),
      (input1_rsci_idat_mxwt[2807:2790]), (input1_rsci_idat_mxwt[2825:2808]), (input1_rsci_idat_mxwt[2843:2826]),
      (input1_rsci_idat_mxwt[2861:2844]), (input1_rsci_idat_mxwt[2879:2862]), (input1_rsci_idat_mxwt[2897:2880]),
      (input1_rsci_idat_mxwt[2915:2898]), (input1_rsci_idat_mxwt[2933:2916]), (input1_rsci_idat_mxwt[2951:2934]),
      (input1_rsci_idat_mxwt[2969:2952]), (input1_rsci_idat_mxwt[2987:2970]), (input1_rsci_idat_mxwt[3005:2988]),
      (input1_rsci_idat_mxwt[3023:3006]), (input1_rsci_idat_mxwt[3041:3024]), (input1_rsci_idat_mxwt[3059:3042]),
      (input1_rsci_idat_mxwt[3077:3060]), (input1_rsci_idat_mxwt[3095:3078]), (input1_rsci_idat_mxwt[3113:3096]),
      (input1_rsci_idat_mxwt[3131:3114]), (input1_rsci_idat_mxwt[3149:3132]), (input1_rsci_idat_mxwt[3167:3150]),
      (input1_rsci_idat_mxwt[3185:3168]), (input1_rsci_idat_mxwt[3203:3186]), (input1_rsci_idat_mxwt[3221:3204]),
      (input1_rsci_idat_mxwt[3239:3222]), (input1_rsci_idat_mxwt[3257:3240]), (input1_rsci_idat_mxwt[3275:3258]),
      (input1_rsci_idat_mxwt[3293:3276]), (input1_rsci_idat_mxwt[3311:3294]), (input1_rsci_idat_mxwt[3329:3312]),
      (input1_rsci_idat_mxwt[3347:3330]), (input1_rsci_idat_mxwt[3365:3348]), (input1_rsci_idat_mxwt[3383:3366]),
      (input1_rsci_idat_mxwt[3401:3384]), (input1_rsci_idat_mxwt[3419:3402]), (input1_rsci_idat_mxwt[3437:3420]),
      (input1_rsci_idat_mxwt[3455:3438]), (input1_rsci_idat_mxwt[3473:3456]), (input1_rsci_idat_mxwt[3491:3474]),
      (input1_rsci_idat_mxwt[3509:3492]), (input1_rsci_idat_mxwt[3527:3510]), (input1_rsci_idat_mxwt[3545:3528]),
      (input1_rsci_idat_mxwt[3563:3546]), (input1_rsci_idat_mxwt[3581:3564]), (input1_rsci_idat_mxwt[3599:3582]),
      (input1_rsci_idat_mxwt[3617:3600]), (input1_rsci_idat_mxwt[3635:3618]), (input1_rsci_idat_mxwt[3653:3636]),
      (input1_rsci_idat_mxwt[3671:3654]), (input1_rsci_idat_mxwt[3689:3672]), (input1_rsci_idat_mxwt[3707:3690]),
      (input1_rsci_idat_mxwt[3725:3708]), (input1_rsci_idat_mxwt[3743:3726]), (input1_rsci_idat_mxwt[3761:3744]),
      (input1_rsci_idat_mxwt[3779:3762]), (input1_rsci_idat_mxwt[3797:3780]), (input1_rsci_idat_mxwt[3815:3798]),
      (input1_rsci_idat_mxwt[3833:3816]), (input1_rsci_idat_mxwt[3851:3834]), (input1_rsci_idat_mxwt[3869:3852]),
      (input1_rsci_idat_mxwt[3887:3870]), (input1_rsci_idat_mxwt[3905:3888]), (input1_rsci_idat_mxwt[3923:3906]),
      (input1_rsci_idat_mxwt[3941:3924]), (input1_rsci_idat_mxwt[3959:3942]), (input1_rsci_idat_mxwt[3977:3960]),
      (input1_rsci_idat_mxwt[3995:3978]), (input1_rsci_idat_mxwt[4013:3996]), (input1_rsci_idat_mxwt[4031:4014]),
      (input1_rsci_idat_mxwt[4049:4032]), (input1_rsci_idat_mxwt[4067:4050]), (input1_rsci_idat_mxwt[4085:4068]),
      (input1_rsci_idat_mxwt[4103:4086]), (input1_rsci_idat_mxwt[4121:4104]), (input1_rsci_idat_mxwt[4139:4122]),
      (input1_rsci_idat_mxwt[4157:4140]), (input1_rsci_idat_mxwt[4175:4158]), (input1_rsci_idat_mxwt[4193:4176]),
      (input1_rsci_idat_mxwt[4211:4194]), (input1_rsci_idat_mxwt[4229:4212]), (input1_rsci_idat_mxwt[4247:4230]),
      (input1_rsci_idat_mxwt[4265:4248]), (input1_rsci_idat_mxwt[4283:4266]), (input1_rsci_idat_mxwt[4301:4284]),
      (input1_rsci_idat_mxwt[4319:4302]), (input1_rsci_idat_mxwt[4337:4320]), (input1_rsci_idat_mxwt[4355:4338]),
      (input1_rsci_idat_mxwt[4373:4356]), (input1_rsci_idat_mxwt[4391:4374]), (input1_rsci_idat_mxwt[4409:4392]),
      (input1_rsci_idat_mxwt[4427:4410]), (input1_rsci_idat_mxwt[4445:4428]), (input1_rsci_idat_mxwt[4463:4446]),
      (input1_rsci_idat_mxwt[4481:4464]), (input1_rsci_idat_mxwt[4499:4482]), (input1_rsci_idat_mxwt[4517:4500]),
      (input1_rsci_idat_mxwt[4535:4518]), (input1_rsci_idat_mxwt[4553:4536]), (input1_rsci_idat_mxwt[4571:4554]),
      (input1_rsci_idat_mxwt[4589:4572]), (input1_rsci_idat_mxwt[4607:4590]), (input1_rsci_idat_mxwt[4625:4608]),
      (input1_rsci_idat_mxwt[4643:4626]), (input1_rsci_idat_mxwt[4661:4644]), (input1_rsci_idat_mxwt[4679:4662]),
      (input1_rsci_idat_mxwt[4697:4680]), (input1_rsci_idat_mxwt[4715:4698]), (input1_rsci_idat_mxwt[4733:4716]),
      (input1_rsci_idat_mxwt[4751:4734]), (input1_rsci_idat_mxwt[4769:4752]), (input1_rsci_idat_mxwt[4787:4770]),
      (input1_rsci_idat_mxwt[4805:4788]), (input1_rsci_idat_mxwt[4823:4806]), (input1_rsci_idat_mxwt[4841:4824]),
      (input1_rsci_idat_mxwt[4859:4842]), (input1_rsci_idat_mxwt[4877:4860]), (input1_rsci_idat_mxwt[4895:4878]),
      (input1_rsci_idat_mxwt[4913:4896]), (input1_rsci_idat_mxwt[4931:4914]), (input1_rsci_idat_mxwt[4949:4932]),
      (input1_rsci_idat_mxwt[4967:4950]), (input1_rsci_idat_mxwt[4985:4968]), (input1_rsci_idat_mxwt[5003:4986]),
      (input1_rsci_idat_mxwt[5021:5004]), (input1_rsci_idat_mxwt[5039:5022]), (input1_rsci_idat_mxwt[5057:5040]),
      (input1_rsci_idat_mxwt[5075:5058]), (input1_rsci_idat_mxwt[5093:5076]), (input1_rsci_idat_mxwt[5111:5094]),
      (input1_rsci_idat_mxwt[5129:5112]), (input1_rsci_idat_mxwt[5147:5130]), (input1_rsci_idat_mxwt[5165:5148]),
      (input1_rsci_idat_mxwt[5183:5166]), (input1_rsci_idat_mxwt[5201:5184]), (input1_rsci_idat_mxwt[5219:5202]),
      (input1_rsci_idat_mxwt[5237:5220]), (input1_rsci_idat_mxwt[5255:5238]), (input1_rsci_idat_mxwt[5273:5256]),
      (input1_rsci_idat_mxwt[5291:5274]), (input1_rsci_idat_mxwt[5309:5292]), (input1_rsci_idat_mxwt[5327:5310]),
      (input1_rsci_idat_mxwt[5345:5328]), (input1_rsci_idat_mxwt[5363:5346]), (input1_rsci_idat_mxwt[5381:5364]),
      (input1_rsci_idat_mxwt[5399:5382]), (input1_rsci_idat_mxwt[5417:5400]), (input1_rsci_idat_mxwt[5435:5418]),
      (input1_rsci_idat_mxwt[5453:5436]), (input1_rsci_idat_mxwt[5471:5454]), (input1_rsci_idat_mxwt[5489:5472]),
      (input1_rsci_idat_mxwt[5507:5490]), (input1_rsci_idat_mxwt[5525:5508]), (input1_rsci_idat_mxwt[5543:5526]),
      (input1_rsci_idat_mxwt[5561:5544]), (input1_rsci_idat_mxwt[5579:5562]), (input1_rsci_idat_mxwt[5597:5580]),
      (input1_rsci_idat_mxwt[5615:5598]), (input1_rsci_idat_mxwt[5633:5616]), (input1_rsci_idat_mxwt[5651:5634]),
      (input1_rsci_idat_mxwt[5669:5652]), (input1_rsci_idat_mxwt[5687:5670]), (input1_rsci_idat_mxwt[5705:5688]),
      (input1_rsci_idat_mxwt[5723:5706]), (input1_rsci_idat_mxwt[5741:5724]), (input1_rsci_idat_mxwt[5759:5742]),
      (input1_rsci_idat_mxwt[5777:5760]), (input1_rsci_idat_mxwt[5795:5778]), (input1_rsci_idat_mxwt[5813:5796]),
      (input1_rsci_idat_mxwt[5831:5814]), (input1_rsci_idat_mxwt[5849:5832]), (input1_rsci_idat_mxwt[5867:5850]),
      (input1_rsci_idat_mxwt[5885:5868]), (input1_rsci_idat_mxwt[5903:5886]), (input1_rsci_idat_mxwt[5921:5904]),
      (input1_rsci_idat_mxwt[5939:5922]), (input1_rsci_idat_mxwt[5957:5940]), (input1_rsci_idat_mxwt[5975:5958]),
      (input1_rsci_idat_mxwt[5993:5976]), (input1_rsci_idat_mxwt[6011:5994]), (input1_rsci_idat_mxwt[6029:6012]),
      (input1_rsci_idat_mxwt[6047:6030]), (input1_rsci_idat_mxwt[6065:6048]), (input1_rsci_idat_mxwt[6083:6066]),
      (input1_rsci_idat_mxwt[6101:6084]), (input1_rsci_idat_mxwt[6119:6102]), (input1_rsci_idat_mxwt[6137:6120]),
      (input1_rsci_idat_mxwt[6155:6138]), (input1_rsci_idat_mxwt[6173:6156]), (input1_rsci_idat_mxwt[6191:6174]),
      (input1_rsci_idat_mxwt[6209:6192]), (input1_rsci_idat_mxwt[6227:6210]), (input1_rsci_idat_mxwt[6245:6228]),
      (input1_rsci_idat_mxwt[6263:6246]), (input1_rsci_idat_mxwt[6281:6264]), (input1_rsci_idat_mxwt[6299:6282]),
      (input1_rsci_idat_mxwt[6317:6300]), (input1_rsci_idat_mxwt[6335:6318]), (input1_rsci_idat_mxwt[6353:6336]),
      (input1_rsci_idat_mxwt[6371:6354]), (input1_rsci_idat_mxwt[6389:6372]), (input1_rsci_idat_mxwt[6407:6390]),
      (input1_rsci_idat_mxwt[6425:6408]), (input1_rsci_idat_mxwt[6443:6426]), (input1_rsci_idat_mxwt[6461:6444]),
      (input1_rsci_idat_mxwt[6479:6462]), (input1_rsci_idat_mxwt[6497:6480]), (input1_rsci_idat_mxwt[6515:6498]),
      (input1_rsci_idat_mxwt[6533:6516]), (input1_rsci_idat_mxwt[6551:6534]), (input1_rsci_idat_mxwt[6569:6552]),
      (input1_rsci_idat_mxwt[6587:6570]), (input1_rsci_idat_mxwt[6605:6588]), (input1_rsci_idat_mxwt[6623:6606]),
      (input1_rsci_idat_mxwt[6641:6624]), (input1_rsci_idat_mxwt[6659:6642]), (input1_rsci_idat_mxwt[6677:6660]),
      (input1_rsci_idat_mxwt[6695:6678]), (input1_rsci_idat_mxwt[6713:6696]), (input1_rsci_idat_mxwt[6731:6714]),
      (input1_rsci_idat_mxwt[6749:6732]), (input1_rsci_idat_mxwt[6767:6750]), (input1_rsci_idat_mxwt[6785:6768]),
      (input1_rsci_idat_mxwt[6803:6786]), (input1_rsci_idat_mxwt[6821:6804]), (input1_rsci_idat_mxwt[6839:6822]),
      (input1_rsci_idat_mxwt[6857:6840]), (input1_rsci_idat_mxwt[6875:6858]), (input1_rsci_idat_mxwt[6893:6876]),
      (input1_rsci_idat_mxwt[6911:6894]), (input1_rsci_idat_mxwt[6929:6912]), (input1_rsci_idat_mxwt[6947:6930]),
      (input1_rsci_idat_mxwt[6965:6948]), (input1_rsci_idat_mxwt[6983:6966]), (input1_rsci_idat_mxwt[7001:6984]),
      (input1_rsci_idat_mxwt[7019:7002]), (input1_rsci_idat_mxwt[7037:7020]), (input1_rsci_idat_mxwt[7055:7038]),
      (input1_rsci_idat_mxwt[7073:7056]), (input1_rsci_idat_mxwt[7091:7074]), (input1_rsci_idat_mxwt[7109:7092]),
      (input1_rsci_idat_mxwt[7127:7110]), (input1_rsci_idat_mxwt[7145:7128]), (input1_rsci_idat_mxwt[7163:7146]),
      (input1_rsci_idat_mxwt[7181:7164]), (input1_rsci_idat_mxwt[7199:7182]), (input1_rsci_idat_mxwt[7217:7200]),
      (input1_rsci_idat_mxwt[7235:7218]), (input1_rsci_idat_mxwt[7253:7236]), (input1_rsci_idat_mxwt[7271:7254]),
      (input1_rsci_idat_mxwt[7289:7272]), (input1_rsci_idat_mxwt[7307:7290]), (input1_rsci_idat_mxwt[7325:7308]),
      (input1_rsci_idat_mxwt[7343:7326]), (input1_rsci_idat_mxwt[7361:7344]), (input1_rsci_idat_mxwt[7379:7362]),
      (input1_rsci_idat_mxwt[7397:7380]), (input1_rsci_idat_mxwt[7415:7398]), (input1_rsci_idat_mxwt[7433:7416]),
      (input1_rsci_idat_mxwt[7451:7434]), (input1_rsci_idat_mxwt[7469:7452]), (input1_rsci_idat_mxwt[7487:7470]),
      (input1_rsci_idat_mxwt[7505:7488]), (input1_rsci_idat_mxwt[7523:7506]), (input1_rsci_idat_mxwt[7541:7524]),
      (input1_rsci_idat_mxwt[7559:7542]), (input1_rsci_idat_mxwt[7577:7560]), (input1_rsci_idat_mxwt[7595:7578]),
      (input1_rsci_idat_mxwt[7613:7596]), (input1_rsci_idat_mxwt[7631:7614]), (input1_rsci_idat_mxwt[7649:7632]),
      (input1_rsci_idat_mxwt[7667:7650]), (input1_rsci_idat_mxwt[7685:7668]), (input1_rsci_idat_mxwt[7703:7686]),
      (input1_rsci_idat_mxwt[7721:7704]), (input1_rsci_idat_mxwt[7739:7722]), (input1_rsci_idat_mxwt[7757:7740]),
      (input1_rsci_idat_mxwt[7775:7758]), (input1_rsci_idat_mxwt[7793:7776]), (input1_rsci_idat_mxwt[7811:7794]),
      (input1_rsci_idat_mxwt[7829:7812]), (input1_rsci_idat_mxwt[7847:7830]), (input1_rsci_idat_mxwt[7865:7848]),
      (input1_rsci_idat_mxwt[7883:7866]), (input1_rsci_idat_mxwt[7901:7884]), (input1_rsci_idat_mxwt[7919:7902]),
      (input1_rsci_idat_mxwt[7937:7920]), (input1_rsci_idat_mxwt[7955:7938]), (input1_rsci_idat_mxwt[7973:7956]),
      (input1_rsci_idat_mxwt[7991:7974]), (input1_rsci_idat_mxwt[8009:7992]), (input1_rsci_idat_mxwt[8027:8010]),
      (input1_rsci_idat_mxwt[8045:8028]), (input1_rsci_idat_mxwt[8063:8046]), (input1_rsci_idat_mxwt[8081:8064]),
      (input1_rsci_idat_mxwt[8099:8082]), (input1_rsci_idat_mxwt[8117:8100]), (input1_rsci_idat_mxwt[8135:8118]),
      (input1_rsci_idat_mxwt[8153:8136]), (input1_rsci_idat_mxwt[8171:8154]), (input1_rsci_idat_mxwt[8189:8172]),
      (input1_rsci_idat_mxwt[8207:8190]), (input1_rsci_idat_mxwt[8225:8208]), (input1_rsci_idat_mxwt[8243:8226]),
      (input1_rsci_idat_mxwt[8261:8244]), (input1_rsci_idat_mxwt[8279:8262]), (input1_rsci_idat_mxwt[8297:8280]),
      (input1_rsci_idat_mxwt[8315:8298]), (input1_rsci_idat_mxwt[8333:8316]), (input1_rsci_idat_mxwt[8351:8334]),
      (input1_rsci_idat_mxwt[8369:8352]), (input1_rsci_idat_mxwt[8387:8370]), (input1_rsci_idat_mxwt[8405:8388]),
      (input1_rsci_idat_mxwt[8423:8406]), (input1_rsci_idat_mxwt[8441:8424]), (input1_rsci_idat_mxwt[8459:8442]),
      (input1_rsci_idat_mxwt[8477:8460]), (input1_rsci_idat_mxwt[8495:8478]), (input1_rsci_idat_mxwt[8513:8496]),
      (input1_rsci_idat_mxwt[8531:8514]), (input1_rsci_idat_mxwt[8549:8532]), (input1_rsci_idat_mxwt[8567:8550]),
      (input1_rsci_idat_mxwt[8585:8568]), (input1_rsci_idat_mxwt[8603:8586]), (input1_rsci_idat_mxwt[8621:8604]),
      (input1_rsci_idat_mxwt[8639:8622]), (input1_rsci_idat_mxwt[8657:8640]), (input1_rsci_idat_mxwt[8675:8658]),
      (input1_rsci_idat_mxwt[8693:8676]), (input1_rsci_idat_mxwt[8711:8694]), (input1_rsci_idat_mxwt[8729:8712]),
      (input1_rsci_idat_mxwt[8747:8730]), (input1_rsci_idat_mxwt[8765:8748]), (input1_rsci_idat_mxwt[8783:8766]),
      (input1_rsci_idat_mxwt[8801:8784]), (input1_rsci_idat_mxwt[8819:8802]), (input1_rsci_idat_mxwt[8837:8820]),
      (input1_rsci_idat_mxwt[8855:8838]), (input1_rsci_idat_mxwt[8873:8856]), (input1_rsci_idat_mxwt[8891:8874]),
      (input1_rsci_idat_mxwt[8909:8892]), (input1_rsci_idat_mxwt[8927:8910]), (input1_rsci_idat_mxwt[8945:8928]),
      (input1_rsci_idat_mxwt[8963:8946]), (input1_rsci_idat_mxwt[8981:8964]), (input1_rsci_idat_mxwt[8999:8982]),
      (input1_rsci_idat_mxwt[9017:9000]), (input1_rsci_idat_mxwt[9035:9018]), (input1_rsci_idat_mxwt[9053:9036]),
      (input1_rsci_idat_mxwt[9071:9054]), (input1_rsci_idat_mxwt[9089:9072]), (input1_rsci_idat_mxwt[9107:9090]),
      (input1_rsci_idat_mxwt[9125:9108]), (input1_rsci_idat_mxwt[9143:9126]), (input1_rsci_idat_mxwt[9161:9144]),
      (input1_rsci_idat_mxwt[9179:9162]), (input1_rsci_idat_mxwt[9197:9180]), (input1_rsci_idat_mxwt[9215:9198]),
      (input1_rsci_idat_mxwt[9233:9216]), (input1_rsci_idat_mxwt[9251:9234]), (input1_rsci_idat_mxwt[9269:9252]),
      (input1_rsci_idat_mxwt[9287:9270]), (input1_rsci_idat_mxwt[9305:9288]), (input1_rsci_idat_mxwt[9323:9306]),
      (input1_rsci_idat_mxwt[9341:9324]), (input1_rsci_idat_mxwt[9359:9342]), (input1_rsci_idat_mxwt[9377:9360]),
      (input1_rsci_idat_mxwt[9395:9378]), (input1_rsci_idat_mxwt[9413:9396]), (input1_rsci_idat_mxwt[9431:9414]),
      (input1_rsci_idat_mxwt[9449:9432]), (input1_rsci_idat_mxwt[9467:9450]), (input1_rsci_idat_mxwt[9485:9468]),
      (input1_rsci_idat_mxwt[9503:9486]), (input1_rsci_idat_mxwt[9521:9504]), (input1_rsci_idat_mxwt[9539:9522]),
      (input1_rsci_idat_mxwt[9557:9540]), (input1_rsci_idat_mxwt[9575:9558]), (input1_rsci_idat_mxwt[9593:9576]),
      (input1_rsci_idat_mxwt[9611:9594]), (input1_rsci_idat_mxwt[9629:9612]), (input1_rsci_idat_mxwt[9647:9630]),
      (input1_rsci_idat_mxwt[9665:9648]), (input1_rsci_idat_mxwt[9683:9666]), (input1_rsci_idat_mxwt[9701:9684]),
      (input1_rsci_idat_mxwt[9719:9702]), (input1_rsci_idat_mxwt[9737:9720]), (input1_rsci_idat_mxwt[9755:9738]),
      (input1_rsci_idat_mxwt[9773:9756]), (input1_rsci_idat_mxwt[9791:9774]), (input1_rsci_idat_mxwt[9809:9792]),
      (input1_rsci_idat_mxwt[9827:9810]), (input1_rsci_idat_mxwt[9845:9828]), (input1_rsci_idat_mxwt[9863:9846]),
      (input1_rsci_idat_mxwt[9881:9864]), (input1_rsci_idat_mxwt[9899:9882]), (input1_rsci_idat_mxwt[9917:9900]),
      (input1_rsci_idat_mxwt[9935:9918]), (input1_rsci_idat_mxwt[9953:9936]), (input1_rsci_idat_mxwt[9971:9954]),
      (input1_rsci_idat_mxwt[9989:9972]), (input1_rsci_idat_mxwt[10007:9990]), (input1_rsci_idat_mxwt[10025:10008]),
      (input1_rsci_idat_mxwt[10043:10026]), (input1_rsci_idat_mxwt[10061:10044]),
      (input1_rsci_idat_mxwt[10079:10062]), (input1_rsci_idat_mxwt[10097:10080]),
      (input1_rsci_idat_mxwt[10115:10098]), (input1_rsci_idat_mxwt[10133:10116]),
      (input1_rsci_idat_mxwt[10151:10134]), (input1_rsci_idat_mxwt[10169:10152]),
      (input1_rsci_idat_mxwt[10187:10170]), (input1_rsci_idat_mxwt[10205:10188]),
      (input1_rsci_idat_mxwt[10223:10206]), (input1_rsci_idat_mxwt[10241:10224]),
      (input1_rsci_idat_mxwt[10259:10242]), (input1_rsci_idat_mxwt[10277:10260]),
      (input1_rsci_idat_mxwt[10295:10278]), (input1_rsci_idat_mxwt[10313:10296]),
      (input1_rsci_idat_mxwt[10331:10314]), (input1_rsci_idat_mxwt[10349:10332]),
      (input1_rsci_idat_mxwt[10367:10350]), (input1_rsci_idat_mxwt[10385:10368]),
      (input1_rsci_idat_mxwt[10403:10386]), (input1_rsci_idat_mxwt[10421:10404]),
      (input1_rsci_idat_mxwt[10439:10422]), (input1_rsci_idat_mxwt[10457:10440]),
      (input1_rsci_idat_mxwt[10475:10458]), (input1_rsci_idat_mxwt[10493:10476]),
      (input1_rsci_idat_mxwt[10511:10494]), (input1_rsci_idat_mxwt[10529:10512]),
      (input1_rsci_idat_mxwt[10547:10530]), (input1_rsci_idat_mxwt[10565:10548]),
      (input1_rsci_idat_mxwt[10583:10566]), (input1_rsci_idat_mxwt[10601:10584]),
      (input1_rsci_idat_mxwt[10619:10602]), (input1_rsci_idat_mxwt[10637:10620]),
      (input1_rsci_idat_mxwt[10655:10638]), (input1_rsci_idat_mxwt[10673:10656]),
      (input1_rsci_idat_mxwt[10691:10674]), (input1_rsci_idat_mxwt[10709:10692]),
      (input1_rsci_idat_mxwt[10727:10710]), (input1_rsci_idat_mxwt[10745:10728]),
      (input1_rsci_idat_mxwt[10763:10746]), (input1_rsci_idat_mxwt[10781:10764]),
      (input1_rsci_idat_mxwt[10799:10782]), (input1_rsci_idat_mxwt[10817:10800]),
      (input1_rsci_idat_mxwt[10835:10818]), (input1_rsci_idat_mxwt[10853:10836]),
      (input1_rsci_idat_mxwt[10871:10854]), (input1_rsci_idat_mxwt[10889:10872]),
      (input1_rsci_idat_mxwt[10907:10890]), (input1_rsci_idat_mxwt[10925:10908]),
      (input1_rsci_idat_mxwt[10943:10926]), (input1_rsci_idat_mxwt[10961:10944]),
      (input1_rsci_idat_mxwt[10979:10962]), (input1_rsci_idat_mxwt[10997:10980]),
      (input1_rsci_idat_mxwt[11015:10998]), (input1_rsci_idat_mxwt[11033:11016]),
      (input1_rsci_idat_mxwt[11051:11034]), (input1_rsci_idat_mxwt[11069:11052]),
      (input1_rsci_idat_mxwt[11087:11070]), (input1_rsci_idat_mxwt[11105:11088]),
      (input1_rsci_idat_mxwt[11123:11106]), (input1_rsci_idat_mxwt[11141:11124]),
      (input1_rsci_idat_mxwt[11159:11142]), (input1_rsci_idat_mxwt[11177:11160]),
      (input1_rsci_idat_mxwt[11195:11178]), (input1_rsci_idat_mxwt[11213:11196]),
      (input1_rsci_idat_mxwt[11231:11214]), (input1_rsci_idat_mxwt[11249:11232]),
      (input1_rsci_idat_mxwt[11267:11250]), (input1_rsci_idat_mxwt[11285:11268]),
      (input1_rsci_idat_mxwt[11303:11286]), (input1_rsci_idat_mxwt[11321:11304]),
      (input1_rsci_idat_mxwt[11339:11322]), (input1_rsci_idat_mxwt[11357:11340]),
      (input1_rsci_idat_mxwt[11375:11358]), (input1_rsci_idat_mxwt[11393:11376]),
      (input1_rsci_idat_mxwt[11411:11394]), (input1_rsci_idat_mxwt[11429:11412]),
      (input1_rsci_idat_mxwt[11447:11430]), (input1_rsci_idat_mxwt[11465:11448]),
      (input1_rsci_idat_mxwt[11483:11466]), (input1_rsci_idat_mxwt[11501:11484]),
      (input1_rsci_idat_mxwt[11519:11502]), (input1_rsci_idat_mxwt[11537:11520]),
      (input1_rsci_idat_mxwt[11555:11538]), (input1_rsci_idat_mxwt[11573:11556]),
      (input1_rsci_idat_mxwt[11591:11574]), (input1_rsci_idat_mxwt[11609:11592]),
      (input1_rsci_idat_mxwt[11627:11610]), (input1_rsci_idat_mxwt[11645:11628]),
      (input1_rsci_idat_mxwt[11663:11646]), (input1_rsci_idat_mxwt[11681:11664]),
      (input1_rsci_idat_mxwt[11699:11682]), (input1_rsci_idat_mxwt[11717:11700]),
      (input1_rsci_idat_mxwt[11735:11718]), (input1_rsci_idat_mxwt[11753:11736]),
      (input1_rsci_idat_mxwt[11771:11754]), (input1_rsci_idat_mxwt[11789:11772]),
      (input1_rsci_idat_mxwt[11807:11790]), (input1_rsci_idat_mxwt[11825:11808]),
      (input1_rsci_idat_mxwt[11843:11826]), (input1_rsci_idat_mxwt[11861:11844]),
      (input1_rsci_idat_mxwt[11879:11862]), (input1_rsci_idat_mxwt[11897:11880]),
      (input1_rsci_idat_mxwt[11915:11898]), (input1_rsci_idat_mxwt[11933:11916]),
      (input1_rsci_idat_mxwt[11951:11934]), (input1_rsci_idat_mxwt[11969:11952]),
      (input1_rsci_idat_mxwt[11987:11970]), (input1_rsci_idat_mxwt[12005:11988]),
      (input1_rsci_idat_mxwt[12023:12006]), (input1_rsci_idat_mxwt[12041:12024]),
      (input1_rsci_idat_mxwt[12059:12042]), (input1_rsci_idat_mxwt[12077:12060]),
      (input1_rsci_idat_mxwt[12095:12078]), (input1_rsci_idat_mxwt[12113:12096]),
      (input1_rsci_idat_mxwt[12131:12114]), (input1_rsci_idat_mxwt[12149:12132]),
      (input1_rsci_idat_mxwt[12167:12150]), (input1_rsci_idat_mxwt[12185:12168]),
      (input1_rsci_idat_mxwt[12203:12186]), (input1_rsci_idat_mxwt[12221:12204]),
      (input1_rsci_idat_mxwt[12239:12222]), (input1_rsci_idat_mxwt[12257:12240]),
      (input1_rsci_idat_mxwt[12275:12258]), (input1_rsci_idat_mxwt[12293:12276]),
      (input1_rsci_idat_mxwt[12311:12294]), (input1_rsci_idat_mxwt[12329:12312]),
      (input1_rsci_idat_mxwt[12347:12330]), (input1_rsci_idat_mxwt[12365:12348]),
      (input1_rsci_idat_mxwt[12383:12366]), (input1_rsci_idat_mxwt[12401:12384]),
      (input1_rsci_idat_mxwt[12419:12402]), (input1_rsci_idat_mxwt[12437:12420]),
      (input1_rsci_idat_mxwt[12455:12438]), (input1_rsci_idat_mxwt[12473:12456]),
      (input1_rsci_idat_mxwt[12491:12474]), (input1_rsci_idat_mxwt[12509:12492]),
      (input1_rsci_idat_mxwt[12527:12510]), (input1_rsci_idat_mxwt[12545:12528]),
      (input1_rsci_idat_mxwt[12563:12546]), (input1_rsci_idat_mxwt[12581:12564]),
      (input1_rsci_idat_mxwt[12599:12582]), (input1_rsci_idat_mxwt[12617:12600]),
      (input1_rsci_idat_mxwt[12635:12618]), (input1_rsci_idat_mxwt[12653:12636]),
      (input1_rsci_idat_mxwt[12671:12654]), (input1_rsci_idat_mxwt[12689:12672]),
      (input1_rsci_idat_mxwt[12707:12690]), (input1_rsci_idat_mxwt[12725:12708]),
      (input1_rsci_idat_mxwt[12743:12726]), (input1_rsci_idat_mxwt[12761:12744]),
      (input1_rsci_idat_mxwt[12779:12762]), (input1_rsci_idat_mxwt[12797:12780]),
      (input1_rsci_idat_mxwt[12815:12798]), (input1_rsci_idat_mxwt[12833:12816]),
      (input1_rsci_idat_mxwt[12851:12834]), (input1_rsci_idat_mxwt[12869:12852]),
      (input1_rsci_idat_mxwt[12887:12870]), (input1_rsci_idat_mxwt[12905:12888]),
      (input1_rsci_idat_mxwt[12923:12906]), (input1_rsci_idat_mxwt[12941:12924]),
      (input1_rsci_idat_mxwt[12959:12942]), (input1_rsci_idat_mxwt[12977:12960]),
      (input1_rsci_idat_mxwt[12995:12978]), (input1_rsci_idat_mxwt[13013:12996]),
      (input1_rsci_idat_mxwt[13031:13014]), (input1_rsci_idat_mxwt[13049:13032]),
      (input1_rsci_idat_mxwt[13067:13050]), (input1_rsci_idat_mxwt[13085:13068]),
      (input1_rsci_idat_mxwt[13103:13086]), (input1_rsci_idat_mxwt[13121:13104]),
      (input1_rsci_idat_mxwt[13139:13122]), (input1_rsci_idat_mxwt[13157:13140]),
      (input1_rsci_idat_mxwt[13175:13158]), (input1_rsci_idat_mxwt[13193:13176]),
      (input1_rsci_idat_mxwt[13211:13194]), (input1_rsci_idat_mxwt[13229:13212]),
      (input1_rsci_idat_mxwt[13247:13230]), (input1_rsci_idat_mxwt[13265:13248]),
      (input1_rsci_idat_mxwt[13283:13266]), (input1_rsci_idat_mxwt[13301:13284]),
      (input1_rsci_idat_mxwt[13319:13302]), (input1_rsci_idat_mxwt[13337:13320]),
      (input1_rsci_idat_mxwt[13355:13338]), (input1_rsci_idat_mxwt[13373:13356]),
      (input1_rsci_idat_mxwt[13391:13374]), (input1_rsci_idat_mxwt[13409:13392]),
      (input1_rsci_idat_mxwt[13427:13410]), (input1_rsci_idat_mxwt[13445:13428]),
      (input1_rsci_idat_mxwt[13463:13446]), (input1_rsci_idat_mxwt[13481:13464]),
      (input1_rsci_idat_mxwt[13499:13482]), (input1_rsci_idat_mxwt[13517:13500]),
      (input1_rsci_idat_mxwt[13535:13518]), (input1_rsci_idat_mxwt[13553:13536]),
      (input1_rsci_idat_mxwt[13571:13554]), (input1_rsci_idat_mxwt[13589:13572]),
      (input1_rsci_idat_mxwt[13607:13590]), (input1_rsci_idat_mxwt[13625:13608]),
      (input1_rsci_idat_mxwt[13643:13626]), (input1_rsci_idat_mxwt[13661:13644]),
      (input1_rsci_idat_mxwt[13679:13662]), (input1_rsci_idat_mxwt[13697:13680]),
      (input1_rsci_idat_mxwt[13715:13698]), (input1_rsci_idat_mxwt[13733:13716]),
      (input1_rsci_idat_mxwt[13751:13734]), (input1_rsci_idat_mxwt[13769:13752]),
      (input1_rsci_idat_mxwt[13787:13770]), (input1_rsci_idat_mxwt[13805:13788]),
      (input1_rsci_idat_mxwt[13823:13806]), (input1_rsci_idat_mxwt[13841:13824]),
      (input1_rsci_idat_mxwt[13859:13842]), (input1_rsci_idat_mxwt[13877:13860]),
      (input1_rsci_idat_mxwt[13895:13878]), (input1_rsci_idat_mxwt[13913:13896]),
      (input1_rsci_idat_mxwt[13931:13914]), (input1_rsci_idat_mxwt[13949:13932]),
      (input1_rsci_idat_mxwt[13967:13950]), (input1_rsci_idat_mxwt[13985:13968]),
      (input1_rsci_idat_mxwt[14003:13986]), (input1_rsci_idat_mxwt[14021:14004]),
      (input1_rsci_idat_mxwt[14039:14022]), (input1_rsci_idat_mxwt[14057:14040]),
      (input1_rsci_idat_mxwt[14075:14058]), (input1_rsci_idat_mxwt[14093:14076]),
      (input1_rsci_idat_mxwt[14111:14094]), {(MultLoop_1_im_6_0_sva_1[5:0]) , CALC_EXP_LOOP_i_3_0_sva_1});
  assign MultLoop_1_MultLoop_1_mux_nl = MUX_v_17_64_2(layer3_out_0_16_0_sva_dfm,
      layer3_out_1_16_0_sva_dfm, layer3_out_2_16_0_sva_dfm, layer3_out_3_16_0_sva_dfm,
      layer3_out_4_16_0_sva_dfm, layer3_out_5_16_0_sva_dfm, layer3_out_6_16_0_sva_dfm,
      layer3_out_7_16_0_sva_dfm, layer3_out_8_16_0_sva_dfm, layer3_out_9_16_0_sva_dfm,
      layer3_out_10_16_0_sva_dfm, layer3_out_11_16_0_sva_dfm, layer3_out_12_16_0_sva_dfm,
      layer3_out_13_16_0_sva_dfm, layer3_out_14_16_0_sva_dfm, layer3_out_15_16_0_sva_dfm,
      layer3_out_16_16_0_sva_dfm, layer3_out_17_16_0_sva_dfm, layer3_out_18_16_0_sva_dfm,
      layer3_out_19_16_0_sva_dfm, layer3_out_20_16_0_sva_dfm, layer3_out_21_16_0_sva_dfm,
      layer3_out_22_16_0_sva_dfm, layer3_out_23_16_0_sva_dfm, layer3_out_24_16_0_sva_dfm,
      layer3_out_25_16_0_sva_dfm, layer3_out_26_16_0_sva_dfm, layer3_out_27_16_0_sva_dfm,
      layer3_out_28_16_0_sva_dfm, layer3_out_29_16_0_sva_dfm, layer3_out_30_16_0_sva_dfm,
      layer3_out_31_16_0_sva_dfm, layer3_out_32_16_0_sva_dfm, layer3_out_33_16_0_sva_dfm,
      layer3_out_34_16_0_sva_dfm, layer3_out_35_16_0_sva_dfm, layer3_out_36_16_0_sva_dfm,
      layer3_out_37_16_0_sva_dfm, layer3_out_38_16_0_sva_dfm, layer3_out_39_16_0_sva_dfm,
      layer3_out_40_16_0_sva_dfm, layer3_out_41_16_0_sva_dfm, layer3_out_42_16_0_sva_dfm,
      layer3_out_43_16_0_sva_dfm, layer3_out_44_16_0_sva_dfm, layer3_out_45_16_0_sva_dfm,
      layer3_out_46_16_0_sva_dfm, layer3_out_47_16_0_sva_dfm, layer3_out_48_16_0_sva_dfm,
      layer3_out_49_16_0_sva_dfm, layer3_out_50_16_0_sva_dfm, layer3_out_51_16_0_sva_dfm,
      layer3_out_52_16_0_sva_dfm, layer3_out_53_16_0_sva_dfm, layer3_out_54_16_0_sva_dfm,
      layer3_out_55_16_0_sva_dfm, layer3_out_56_16_0_sva_dfm, layer3_out_57_16_0_sva_dfm,
      layer3_out_58_16_0_sva_dfm, layer3_out_59_16_0_sva_dfm, layer3_out_60_16_0_sva_dfm,
      layer3_out_61_16_0_sva_dfm, layer3_out_62_16_0_sva_dfm, ({layer3_out_63_16_0_lpi_2_dfm_16_10
      , layer3_out_63_16_0_lpi_2_dfm_9_0}), InitAccum_1_iacc_6_0_sva_5_0);
  assign ResultLoop_2_and_nl = (~((z_out_7[3]) | (z_out_5[3]))) & (~(InitAccum_2_acc_1_itm_3
      | (z_out_6[3])));
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_nor_nl
      = ~((ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_sum_exp_sva_1_mx0w1!=71'b00000000000000000000000000000000000000000000000000000000000000000000000));
  assign CALC_SOFTMAX_LOOP_CALC_SOFTMAX_LOOP_nor_nl = ~((z_out_7[3]) | InitAccum_2_acc_1_itm_3
      | (z_out_5[3]));
  assign and_524_nl = (~ OUTPUT_LOOP_or_tmp) & (fsm_output[17]);
  assign and_530_nl = or_dcpl_548 & (fsm_output[17]);
  assign and_536_nl = or_dcpl_551 & (fsm_output[17]);
  assign and_542_nl = or_dcpl_552 & (fsm_output[17]);
  assign and_548_nl = or_dcpl_554 & (fsm_output[17]);
  assign and_554_nl = or_dcpl_556 & (fsm_output[17]);
  assign and_560_nl = or_dcpl_557 & (fsm_output[17]);
  assign and_566_nl = or_dcpl_558 & (fsm_output[17]);
  assign and_572_nl = or_dcpl_560 & (fsm_output[17]);
  assign and_578_nl = or_dcpl_561 & (fsm_output[17]);
  assign ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_1_nl
      = MUX_v_3_4_2(3'b010, 3'b110, 3'b001, 3'b101, ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_acc_itm_19_1[11:10]);
  assign ReuseLoop_in_index_mux1h_nl = MUX1HOT_v_6_4_2(ReuseLoop_ir_9_0_sva_mx0_tmp_9_4,
      (layer3_out_0_16_0_sva_dfm[5:0]), (z_out_7[5:0]), ({3'b000 , (ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_1_nl)}),
      {InitAccum_1_iacc_6_0_sva_5_0_mx0c0 , (fsm_output[4]) , InitAccum_1_iacc_6_0_sva_5_0_mx0c2
      , (fsm_output[14])});
  assign InitAccum_1_iacc_not_nl = ~ InitAccum_1_iacc_6_0_sva_5_0_mx0c3;
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_62_nl = (~
      or_dcpl_680) & (fsm_output[2]);
  assign and_1844_nl = and_dcpl_40 & and_dcpl_37 & (fsm_output[6]);
  assign and_1846_nl = or_dcpl_680 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_and_65_nl = (~
      or_dcpl_107) & (fsm_output[11]);
  assign or_2584_nl = (and_dcpl_44 & nor_417_cse & (fsm_output[12])) | (and_dcpl_34
      & and_dcpl_32 & or_2157_cse);
  assign nor_520_nl = ~((fsm_output[2]) | (~ or_tmp_1207));
  assign or_2147_nl = (InitAccum_1_iacc_6_0_sva_5_0!=6'b000000);
  assign mux_273_nl = MUX_s_1_2_2((nor_520_nl), or_tmp_1207, or_2147_nl);
  assign nor_523_nl = ~(and_1829_cse | or_2579_tmp);
  assign and_1884_nl = and_dcpl_59 & and_dcpl_56 & (fsm_output[6]);
  assign and_1886_nl = or_dcpl_663 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4027_nl = and_dcpl_51 & and_dcpl_53 & or_2157_cse;
  assign and_1899_nl = and_dcpl_40 & and_dcpl_64 & (fsm_output[6]);
  assign and_1901_nl = or_dcpl_662 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2586_nl = (and_dcpl_44 & and_dcpl_67 & (fsm_output[12])) | (and_dcpl_34
      & and_dcpl_62 & or_2157_cse);
  assign and_1917_nl = and_dcpl_73 & and_dcpl_72 & (fsm_output[6]);
  assign and_1919_nl = or_dcpl_661 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4032_nl = and_dcpl_51 & and_dcpl_70 & or_2157_cse;
  assign and_1931_nl = and_dcpl_79 & and_dcpl_37 & (fsm_output[6]);
  assign and_1933_nl = or_dcpl_660 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4034_nl = and_dcpl_34 & and_dcpl_77 & or_2157_cse;
  assign and_1945_nl = and_dcpl_73 & and_dcpl_56 & (fsm_output[6]);
  assign and_1947_nl = or_dcpl_659 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4036_nl = and_dcpl_51 & and_dcpl_82 & or_2157_cse;
  assign and_1959_nl = and_dcpl_79 & and_dcpl_64 & (fsm_output[6]);
  assign and_1961_nl = or_dcpl_658 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4038_nl = and_dcpl_34 & and_dcpl_86 & or_2157_cse;
  assign and_1973_nl = and_dcpl_92 & and_dcpl_72 & (fsm_output[6]);
  assign and_1975_nl = or_dcpl_657 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4040_nl = and_dcpl_51 & and_dcpl_86 & or_2157_cse;
  assign and_1987_nl = and_dcpl_97 & and_dcpl_37 & (fsm_output[6]);
  assign and_1989_nl = or_dcpl_656 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4042_nl = and_dcpl_34 & and_dcpl_82 & or_2157_cse;
  assign and_2001_nl = and_dcpl_92 & and_dcpl_56 & (fsm_output[6]);
  assign and_2003_nl = or_dcpl_655 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4044_nl = and_dcpl_51 & and_dcpl_77 & or_2157_cse;
  assign and_2015_nl = and_dcpl_97 & and_dcpl_64 & (fsm_output[6]);
  assign and_2017_nl = or_dcpl_654 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4046_nl = and_dcpl_34 & and_dcpl_70 & or_2157_cse;
  assign and_2029_nl = and_dcpl_107 & and_dcpl_72 & (fsm_output[6]);
  assign and_2031_nl = or_dcpl_653 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4048_nl = and_dcpl_51 & and_dcpl_62 & or_2157_cse;
  assign and_2043_nl = and_dcpl_111 & and_dcpl_37 & (fsm_output[6]);
  assign and_2045_nl = or_dcpl_652 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4050_nl = and_dcpl_34 & and_dcpl_53 & or_2157_cse;
  assign and_2057_nl = and_dcpl_107 & and_dcpl_56 & (fsm_output[6]);
  assign and_2059_nl = or_dcpl_651 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4052_nl = and_dcpl_51 & and_dcpl_32 & or_2157_cse;
  assign and_2071_nl = and_dcpl_111 & and_dcpl_64 & (fsm_output[6]);
  assign and_2073_nl = or_dcpl_650 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4054_nl = and_dcpl_34 & and_dcpl_49 & or_2157_cse;
  assign and_2085_nl = and_dcpl_111 & and_dcpl_72 & (fsm_output[6]);
  assign and_2087_nl = or_dcpl_649 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4056_nl = and_dcpl_120 & and_dcpl_49 & or_2157_cse;
  assign and_2099_nl = and_dcpl_107 & and_dcpl_37 & (fsm_output[6]);
  assign and_2101_nl = or_dcpl_648 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4058_nl = and_dcpl_124 & and_dcpl_32 & or_2157_cse;
  assign and_2113_nl = and_dcpl_111 & and_dcpl_56 & (fsm_output[6]);
  assign and_2115_nl = or_dcpl_647 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4060_nl = and_dcpl_120 & and_dcpl_53 & or_2157_cse;
  assign and_2127_nl = and_dcpl_107 & and_dcpl_64 & (fsm_output[6]);
  assign and_2129_nl = or_dcpl_646 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4062_nl = and_dcpl_124 & and_dcpl_62 & or_2157_cse;
  assign and_2141_nl = and_dcpl_97 & and_dcpl_72 & (fsm_output[6]);
  assign and_2143_nl = or_dcpl_645 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4064_nl = and_dcpl_120 & and_dcpl_70 & or_2157_cse;
  assign and_2156_nl = and_dcpl_92 & and_dcpl_37 & (fsm_output[6]);
  assign and_2158_nl = or_dcpl_644 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2604_nl = (and_dcpl_44 & and_dcpl_140 & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_77 & or_2157_cse);
  assign and_2174_nl = and_dcpl_97 & and_dcpl_56 & (fsm_output[6]);
  assign and_2176_nl = or_dcpl_643 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4069_nl = and_dcpl_120 & and_dcpl_82 & or_2157_cse;
  assign and_2189_nl = and_dcpl_92 & and_dcpl_64 & (fsm_output[6]);
  assign and_2191_nl = or_dcpl_642 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2606_nl = (and_dcpl_44 & and_dcpl_148 & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_86 & or_2157_cse);
  assign and_2207_nl = and_dcpl_79 & and_dcpl_72 & (fsm_output[6]);
  assign and_2209_nl = or_dcpl_641 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4074_nl = and_dcpl_120 & and_dcpl_86 & or_2157_cse;
  assign and_2222_nl = and_dcpl_73 & and_dcpl_37 & (fsm_output[6]);
  assign and_2224_nl = or_dcpl_640 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2608_nl = (and_dcpl_156 & nor_417_cse & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_82 & or_2157_cse);
  assign or_2280_nl = (reg_ResultLoop_1_ires_6_0_sva_5_0_tmp!=2'b00) | nor_518_cse;
  assign mux_276_nl = MUX_s_1_2_2((~ (fsm_output[12])), (or_2280_nl), reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp[3]);
  assign and_2240_nl = and_dcpl_79 & and_dcpl_56 & (fsm_output[6]);
  assign and_2242_nl = or_dcpl_639 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4079_nl = and_dcpl_120 & and_dcpl_77 & or_2157_cse;
  assign and_2255_nl = and_dcpl_73 & and_dcpl_64 & (fsm_output[6]);
  assign and_2257_nl = or_dcpl_638 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2610_nl = (and_dcpl_156 & and_dcpl_67 & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_70 & or_2157_cse);
  assign and_2273_nl = and_dcpl_40 & and_dcpl_72 & (fsm_output[6]);
  assign and_2275_nl = or_dcpl_637 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4084_nl = and_dcpl_120 & and_dcpl_62 & or_2157_cse;
  assign and_2288_nl = and_dcpl_59 & and_dcpl_37 & (fsm_output[6]);
  assign and_2290_nl = or_dcpl_635 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2612_nl = (and_dcpl_156 & and_dcpl_140 & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_53 & or_2157_cse);
  assign and_2306_nl = and_dcpl_40 & and_dcpl_56 & (fsm_output[6]);
  assign and_2308_nl = or_dcpl_633 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4089_nl = and_dcpl_120 & and_dcpl_32 & or_2157_cse;
  assign and_2321_nl = and_dcpl_59 & and_dcpl_64 & (fsm_output[6]);
  assign and_2323_nl = or_dcpl_630 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2614_nl = (and_dcpl_156 & and_dcpl_148 & (fsm_output[12])) | (and_dcpl_124
      & and_dcpl_49 & or_2157_cse);
  assign and_2339_nl = and_dcpl_59 & and_dcpl_183 & (fsm_output[6]);
  assign and_2341_nl = or_dcpl_627 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4094_nl = and_dcpl_180 & and_dcpl_49 & or_2157_cse;
  assign and_2354_nl = and_dcpl_40 & and_dcpl_190 & (fsm_output[6]);
  assign and_2356_nl = or_dcpl_626 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2616_nl = (and_dcpl_193 & nor_417_cse & (fsm_output[12])) | (and_dcpl_187
      & and_dcpl_32 & or_2157_cse);
  assign and_2372_nl = and_dcpl_59 & and_dcpl_196 & (fsm_output[6]);
  assign and_2374_nl = or_dcpl_625 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4099_nl = and_dcpl_180 & and_dcpl_53 & or_2157_cse;
  assign and_2387_nl = and_dcpl_40 & and_dcpl_200 & (fsm_output[6]);
  assign and_2389_nl = or_dcpl_624 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign or_2618_nl = (and_dcpl_193 & and_dcpl_67 & (fsm_output[12])) | (and_dcpl_187
      & and_dcpl_62 & or_2157_cse);
  assign and_2405_nl = and_dcpl_73 & and_dcpl_183 & (fsm_output[6]);
  assign and_2407_nl = or_dcpl_623 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4104_nl = and_dcpl_180 & and_dcpl_70 & or_2157_cse;
  assign and_2419_nl = and_dcpl_79 & and_dcpl_190 & (fsm_output[6]);
  assign and_2421_nl = or_dcpl_622 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4106_nl = and_dcpl_187 & and_dcpl_77 & or_2157_cse;
  assign and_2433_nl = and_dcpl_73 & and_dcpl_196 & (fsm_output[6]);
  assign and_2435_nl = or_dcpl_621 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4108_nl = and_dcpl_180 & and_dcpl_82 & or_2157_cse;
  assign and_2447_nl = and_dcpl_79 & and_dcpl_200 & (fsm_output[6]);
  assign and_2449_nl = or_dcpl_620 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4110_nl = and_dcpl_187 & and_dcpl_86 & or_2157_cse;
  assign and_2461_nl = and_dcpl_92 & and_dcpl_183 & (fsm_output[6]);
  assign and_2463_nl = or_dcpl_619 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4112_nl = and_dcpl_180 & and_dcpl_86 & or_2157_cse;
  assign and_2475_nl = and_dcpl_97 & and_dcpl_190 & (fsm_output[6]);
  assign and_2477_nl = or_dcpl_618 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4114_nl = and_dcpl_187 & and_dcpl_82 & or_2157_cse;
  assign and_2489_nl = and_dcpl_92 & and_dcpl_196 & (fsm_output[6]);
  assign and_2491_nl = or_dcpl_617 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4116_nl = and_dcpl_180 & and_dcpl_77 & or_2157_cse;
  assign and_2503_nl = and_dcpl_97 & and_dcpl_200 & (fsm_output[6]);
  assign and_2505_nl = or_dcpl_616 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4118_nl = and_dcpl_187 & and_dcpl_70 & or_2157_cse;
  assign and_2517_nl = and_dcpl_107 & and_dcpl_183 & (fsm_output[6]);
  assign and_2519_nl = or_dcpl_615 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4120_nl = and_dcpl_180 & and_dcpl_62 & or_2157_cse;
  assign and_2531_nl = and_dcpl_111 & and_dcpl_190 & (fsm_output[6]);
  assign and_2533_nl = or_dcpl_614 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4122_nl = and_dcpl_187 & and_dcpl_53 & or_2157_cse;
  assign and_2545_nl = and_dcpl_107 & and_dcpl_196 & (fsm_output[6]);
  assign and_2547_nl = or_dcpl_613 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4124_nl = and_dcpl_180 & and_dcpl_32 & or_2157_cse;
  assign and_2559_nl = and_dcpl_111 & and_dcpl_200 & (fsm_output[6]);
  assign and_2561_nl = or_dcpl_612 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4126_nl = and_dcpl_187 & and_dcpl_49 & or_2157_cse;
  assign and_2573_nl = and_dcpl_111 & and_dcpl_183 & (fsm_output[6]);
  assign and_2575_nl = or_dcpl_611 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4128_nl = and_dcpl_240 & and_dcpl_49 & or_2157_cse;
  assign and_2587_nl = and_dcpl_107 & and_dcpl_190 & (fsm_output[6]);
  assign and_2589_nl = or_dcpl_610 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4130_nl = and_dcpl_244 & and_dcpl_32 & or_2157_cse;
  assign and_2601_nl = and_dcpl_111 & and_dcpl_196 & (fsm_output[6]);
  assign and_2603_nl = or_dcpl_609 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4132_nl = and_dcpl_240 & and_dcpl_53 & or_2157_cse;
  assign and_2615_nl = and_dcpl_107 & and_dcpl_200 & (fsm_output[6]);
  assign and_2617_nl = or_dcpl_607 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4134_nl = and_dcpl_244 & and_dcpl_62 & or_2157_cse;
  assign and_2629_nl = and_dcpl_97 & and_dcpl_183 & (fsm_output[6]);
  assign and_2631_nl = or_dcpl_605 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4136_nl = and_dcpl_240 & and_dcpl_70 & or_2157_cse;
  assign and_2643_nl = and_dcpl_92 & and_dcpl_190 & (fsm_output[6]);
  assign and_2645_nl = or_dcpl_604 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4138_nl = and_dcpl_244 & and_dcpl_77 & or_2157_cse;
  assign and_2657_nl = and_dcpl_97 & and_dcpl_196 & (fsm_output[6]);
  assign and_2659_nl = or_dcpl_603 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4140_nl = and_dcpl_240 & and_dcpl_82 & or_2157_cse;
  assign and_2671_nl = and_dcpl_92 & and_dcpl_200 & (fsm_output[6]);
  assign and_2673_nl = or_dcpl_601 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4142_nl = and_dcpl_244 & and_dcpl_86 & or_2157_cse;
  assign and_2685_nl = and_dcpl_79 & and_dcpl_183 & (fsm_output[6]);
  assign and_2687_nl = or_dcpl_599 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4144_nl = and_dcpl_240 & and_dcpl_86 & or_2157_cse;
  assign and_2699_nl = and_dcpl_73 & and_dcpl_190 & (fsm_output[6]);
  assign and_2701_nl = or_dcpl_598 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4146_nl = and_dcpl_244 & and_dcpl_82 & or_2157_cse;
  assign and_2713_nl = and_dcpl_79 & and_dcpl_196 & (fsm_output[6]);
  assign and_2715_nl = or_dcpl_597 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4148_nl = and_dcpl_240 & and_dcpl_77 & or_2157_cse;
  assign and_2727_nl = and_dcpl_73 & and_dcpl_200 & (fsm_output[6]);
  assign and_2729_nl = or_dcpl_594 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4150_nl = and_dcpl_244 & and_dcpl_70 & or_2157_cse;
  assign and_2741_nl = and_dcpl_40 & and_dcpl_183 & (fsm_output[6]);
  assign and_2743_nl = or_dcpl_591 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4152_nl = and_dcpl_240 & and_dcpl_62 & or_2157_cse;
  assign and_2755_nl = and_dcpl_59 & and_dcpl_190 & (fsm_output[6]);
  assign and_2757_nl = or_dcpl_589 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4154_nl = and_dcpl_244 & and_dcpl_53 & or_2157_cse;
  assign and_2769_nl = and_dcpl_40 & and_dcpl_196 & (fsm_output[6]);
  assign and_2771_nl = or_dcpl_587 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4156_nl = and_dcpl_240 & and_dcpl_32 & or_2157_cse;
  assign and_2783_nl = and_dcpl_59 & and_dcpl_200 & (fsm_output[6]);
  assign and_2785_nl = or_dcpl_582 & ResultLoop_and_1_tmp & (fsm_output[6]);
  assign and_4158_nl = and_dcpl_244 & and_dcpl_49 & or_2157_cse;
  assign InitAccum_1_iacc_and_5_nl = (~ (fsm_output[15])) & or_dcpl_1103;
  assign InitAccum_1_iacc_mux1h_3_nl = MUX1HOT_v_4_4_2(MultLoop_1_if_1_acc_itm_3_0,
      (z_out_5[3:0]), MultLoop_2_if_1_acc_tmp, CALC_EXP_LOOP_i_3_0_sva_1, {or_2157_cse
      , or_1917_ssc , (InitAccum_1_iacc_and_5_nl) , (fsm_output[15])});
  assign nor_521_nl = ~((or_dcpl_1103 & and_dcpl_296 & (~((fsm_output[14]) | (fsm_output[11])
      | (fsm_output[12])))) | ((~ InitAccum_2_acc_1_itm_3) & (fsm_output[11])) |
      or_tmp_1116);
  assign or_2649_nl = (fsm_output[13]) | (fsm_output[16]);
  assign nand_33_nl = ~((z_out_5[6]) & (z_out_6[6]) & ((fsm_output[14]) | (fsm_output[18])));
  assign mux_282_nl = MUX_s_1_2_2((or_2649_nl), (nand_33_nl), fsm_output[10]);
  assign InitAccum_1_iacc_InitAccum_1_iacc_mux_nl = MUX_v_2_2_2(MultLoop_1_if_1_acc_itm_5_4,
      (z_out_5[5:4]), or_1917_ssc);
  assign not_1748_nl = ~ or_tmp_1116;
  assign nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nnet_relu_layer2_t_layer3_t_relu_config3_for_else_nand_63_nl
      = ~(nnet_relu_layer4_t_layer5_t_relu_config5_for_else_and_stg_4_0_sva_1 & (~
      (MultLoop_1_if_1_acc_itm_5_4[1])));
  assign layer3_out_and_62_nl = (~(nnet_relu_layer2_t_layer3_t_relu_config3_for_and_132_tmp
      | mux_tmp)) & or_tmp_1141;
  assign layer3_out_and_63_nl = nnet_relu_layer2_t_layer3_t_relu_config3_for_and_132_tmp
      & or_tmp_1141;
  assign not_nl = ~ (fsm_output[13]);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_2_nl
      = ((ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva[17])
      & (~((fsm_output[8]) | (fsm_output[12]) | (fsm_output[15]) | (fsm_output[17]))))
      | (fsm_output[16]);
  assign CALC_SOFTMAX_LOOP_mux_3_nl = MUX_v_50_10_2((ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1[66:17]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1[66:17]),
      MultLoop_1_if_1_acc_itm_3_0);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux_1_nl = MUX_v_50_2_2((signext_50_1(ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva[17])),
      (CALC_SOFTMAX_LOOP_mux_3_nl), fsm_output[17]);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_nor_3_nl = ~((fsm_output[8])
      | (fsm_output[12]) | (fsm_output[15]));
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_4_nl = MUX_v_50_2_2(50'b00000000000000000000000000000000000000000000000000,
      (nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux_1_nl), (nnet_product_input_t_config2_weight_t_config2_accum_t_1_nor_3_nl));
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl
      = MUX_v_50_2_2((nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_4_nl),
      50'b11111111111111111111111111111111111111111111111111, (fsm_output[16]));
  assign MultLoop_2_MultLoop_2_mux_2_nl = MUX_v_7_64_2((layer3_out_0_16_0_sva_dfm[16:10]),
      (layer5_out_1_16_0_sva_dfm[16:10]), (layer5_out_2_16_0_sva_dfm[16:10]), (layer5_out_3_16_0_sva_dfm[16:10]),
      (layer5_out_4_16_0_sva_dfm[16:10]), (layer5_out_5_16_0_sva_dfm[16:10]), (layer5_out_6_16_0_sva_dfm[16:10]),
      (layer5_out_7_16_0_sva_dfm[16:10]), (layer5_out_8_16_0_sva_dfm[16:10]), (layer5_out_9_16_0_sva_dfm[16:10]),
      (layer5_out_10_16_0_sva_dfm[16:10]), (layer5_out_11_16_0_sva_dfm[16:10]), (layer5_out_12_16_0_sva_dfm[16:10]),
      (layer5_out_13_16_0_sva_dfm[16:10]), (layer5_out_14_16_0_sva_dfm[16:10]), (layer5_out_15_16_0_sva_dfm[16:10]),
      (layer5_out_16_16_0_sva_dfm[16:10]), (layer5_out_17_16_0_sva_dfm[16:10]), (layer5_out_18_16_0_sva_dfm[16:10]),
      (layer5_out_19_16_0_sva_dfm[16:10]), (layer5_out_20_16_0_sva_dfm[16:10]), (layer5_out_21_16_0_sva_dfm[16:10]),
      (layer5_out_22_16_0_sva_dfm[16:10]), (layer5_out_23_16_0_sva_dfm[16:10]), (layer5_out_24_16_0_sva_dfm[16:10]),
      (layer5_out_25_16_0_sva_dfm[16:10]), (layer5_out_26_16_0_sva_dfm[16:10]), (layer5_out_27_16_0_sva_dfm[16:10]),
      (layer5_out_28_16_0_sva_dfm[16:10]), (layer5_out_29_16_0_sva_dfm[16:10]), (layer5_out_30_16_0_sva_dfm[16:10]),
      (layer5_out_31_16_0_sva_dfm[16:10]), (layer5_out_32_16_0_sva_dfm[16:10]), (layer5_out_33_16_0_sva_dfm[16:10]),
      (layer5_out_34_16_0_sva_dfm[16:10]), (layer5_out_35_16_0_sva_dfm[16:10]), (layer5_out_36_16_0_sva_dfm[16:10]),
      (layer5_out_37_16_0_sva_dfm[16:10]), (layer5_out_38_16_0_sva_dfm[16:10]), (layer5_out_39_16_0_sva_dfm[16:10]),
      (layer5_out_40_16_0_sva_dfm[16:10]), (layer5_out_41_16_0_sva_dfm[16:10]), (layer5_out_42_16_0_sva_dfm[16:10]),
      (layer5_out_43_16_0_sva_dfm[16:10]), (layer5_out_44_16_0_sva_dfm[16:10]), (layer5_out_45_16_0_sva_dfm[16:10]),
      (layer5_out_46_16_0_sva_dfm[16:10]), (layer5_out_47_16_0_sva_dfm[16:10]), (layer5_out_48_16_0_sva_dfm[16:10]),
      (layer5_out_49_16_0_sva_dfm[16:10]), (layer5_out_50_16_0_sva_dfm[16:10]), (layer5_out_51_16_0_sva_dfm[16:10]),
      (layer5_out_52_16_0_sva_dfm[16:10]), (layer5_out_53_16_0_sva_dfm[16:10]), (layer5_out_54_16_0_sva_dfm[16:10]),
      (layer5_out_55_16_0_sva_dfm[16:10]), (layer5_out_56_16_0_sva_dfm[16:10]), (layer5_out_57_16_0_sva_dfm[16:10]),
      (layer5_out_58_16_0_sva_dfm[16:10]), (layer5_out_59_16_0_sva_dfm[16:10]), (layer5_out_60_16_0_sva_dfm[16:10]),
      (layer5_out_61_16_0_sva_dfm[16:10]), (layer5_out_62_16_0_sva_dfm[16:10]), layer3_out_63_16_0_lpi_2_dfm_16_10,
      MultLoop_1_im_6_0_sva_1[5:0]);
  assign CALC_SOFTMAX_LOOP_mux_4_nl = MUX_v_7_10_2((ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1[16:10]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1[16:10]),
      MultLoop_1_if_1_acc_itm_3_0);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_3_nl = MUX1HOT_v_7_3_2((ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva[16:10]),
      (MultLoop_2_MultLoop_2_mux_2_nl), (CALC_SOFTMAX_LOOP_mux_4_nl), {or_2157_cse
      , (fsm_output[12]) , (fsm_output[17])});
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_not_1_nl = ~ (fsm_output[15]);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_5_nl = MUX_v_7_2_2(7'b0000000,
      (nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_3_nl), (nnet_product_input_t_config2_weight_t_config2_accum_t_1_not_1_nl));
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl = MUX_v_7_2_2((nnet_product_input_t_config2_weight_t_config2_accum_t_1_and_5_nl),
      7'b1111111, (fsm_output[16]));
  assign MultLoop_2_MultLoop_2_mux_3_nl = MUX_v_10_64_2((layer3_out_0_16_0_sva_dfm[9:0]),
      (layer5_out_1_16_0_sva_dfm[9:0]), (layer5_out_2_16_0_sva_dfm[9:0]), (layer5_out_3_16_0_sva_dfm[9:0]),
      (layer5_out_4_16_0_sva_dfm[9:0]), (layer5_out_5_16_0_sva_dfm[9:0]), (layer5_out_6_16_0_sva_dfm[9:0]),
      (layer5_out_7_16_0_sva_dfm[9:0]), (layer5_out_8_16_0_sva_dfm[9:0]), (layer5_out_9_16_0_sva_dfm[9:0]),
      (layer5_out_10_16_0_sva_dfm[9:0]), (layer5_out_11_16_0_sva_dfm[9:0]), (layer5_out_12_16_0_sva_dfm[9:0]),
      (layer5_out_13_16_0_sva_dfm[9:0]), (layer5_out_14_16_0_sva_dfm[9:0]), (layer5_out_15_16_0_sva_dfm[9:0]),
      (layer5_out_16_16_0_sva_dfm[9:0]), (layer5_out_17_16_0_sva_dfm[9:0]), (layer5_out_18_16_0_sva_dfm[9:0]),
      (layer5_out_19_16_0_sva_dfm[9:0]), (layer5_out_20_16_0_sva_dfm[9:0]), (layer5_out_21_16_0_sva_dfm[9:0]),
      (layer5_out_22_16_0_sva_dfm[9:0]), (layer5_out_23_16_0_sva_dfm[9:0]), (layer5_out_24_16_0_sva_dfm[9:0]),
      (layer5_out_25_16_0_sva_dfm[9:0]), (layer5_out_26_16_0_sva_dfm[9:0]), (layer5_out_27_16_0_sva_dfm[9:0]),
      (layer5_out_28_16_0_sva_dfm[9:0]), (layer5_out_29_16_0_sva_dfm[9:0]), (layer5_out_30_16_0_sva_dfm[9:0]),
      (layer5_out_31_16_0_sva_dfm[9:0]), (layer5_out_32_16_0_sva_dfm[9:0]), (layer5_out_33_16_0_sva_dfm[9:0]),
      (layer5_out_34_16_0_sva_dfm[9:0]), (layer5_out_35_16_0_sva_dfm[9:0]), (layer5_out_36_16_0_sva_dfm[9:0]),
      (layer5_out_37_16_0_sva_dfm[9:0]), (layer5_out_38_16_0_sva_dfm[9:0]), (layer5_out_39_16_0_sva_dfm[9:0]),
      (layer5_out_40_16_0_sva_dfm[9:0]), (layer5_out_41_16_0_sva_dfm[9:0]), (layer5_out_42_16_0_sva_dfm[9:0]),
      (layer5_out_43_16_0_sva_dfm[9:0]), (layer5_out_44_16_0_sva_dfm[9:0]), (layer5_out_45_16_0_sva_dfm[9:0]),
      (layer5_out_46_16_0_sva_dfm[9:0]), (layer5_out_47_16_0_sva_dfm[9:0]), (layer5_out_48_16_0_sva_dfm[9:0]),
      (layer5_out_49_16_0_sva_dfm[9:0]), (layer5_out_50_16_0_sva_dfm[9:0]), (layer5_out_51_16_0_sva_dfm[9:0]),
      (layer5_out_52_16_0_sva_dfm[9:0]), (layer5_out_53_16_0_sva_dfm[9:0]), (layer5_out_54_16_0_sva_dfm[9:0]),
      (layer5_out_55_16_0_sva_dfm[9:0]), (layer5_out_56_16_0_sva_dfm[9:0]), (layer5_out_57_16_0_sva_dfm[9:0]),
      (layer5_out_58_16_0_sva_dfm[9:0]), (layer5_out_59_16_0_sva_dfm[9:0]), (layer5_out_60_16_0_sva_dfm[9:0]),
      (layer5_out_61_16_0_sva_dfm[9:0]), (layer5_out_62_16_0_sva_dfm[9:0]), layer3_out_63_16_0_lpi_2_dfm_9_0,
      MultLoop_1_im_6_0_sva_1[5:0]);
  assign CALC_SOFTMAX_LOOP_mux_5_nl = MUX_v_10_10_2((ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_0_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_1_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_2_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_3_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_4_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_5_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_6_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_7_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_8_sva_1[9:0]),
      (ac_math_ac_softmax_pwl_AC_TRN_false_0_0_AC_TRN_AC_WRAP_false_0_0_AC_TRN_AC_WRAP_10_18_6_true_AC_TRN_AC_SAT_18_2_AC_TRN_AC_SAT_exp_arr_9_sva_1[9:0]),
      MultLoop_1_if_1_acc_itm_3_0);
  assign and_4241_nl = (fsm_output[12]) & (~ or_2583_tmp);
  assign and_4242_nl = (fsm_output[16]) & (~ or_2583_tmp);
  assign and_4243_nl = (fsm_output[17]) & (~ or_2583_tmp);
  assign mux1h_189_nl = MUX1HOT_v_10_4_2((MultLoop_2_MultLoop_2_mux_3_nl), ({2'b11
      , ROM_1i3_1o8_bdb5a3eca137308489a677a1241b230a2e_1}), (CALC_SOFTMAX_LOOP_mux_5_nl),
      (ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_slc_ac_math_ac_exp_pwl_0_AC_TRN_18_6_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_mul_32_14_psp_sva[9:0]),
      {(and_4241_nl) , (and_4242_nl) , (and_4243_nl) , or_2583_tmp});
  assign MultLoop_2_mux_14_nl = MUX_v_18_64_2((tmp_lpi_3_dfm_1[17:0]), (tmp_lpi_3_dfm_1[35:18]),
      (tmp_lpi_3_dfm_1[53:36]), (tmp_lpi_3_dfm_1[71:54]), (tmp_lpi_3_dfm_1[89:72]),
      (tmp_lpi_3_dfm_1[107:90]), (tmp_lpi_3_dfm_1[125:108]), (tmp_lpi_3_dfm_1[143:126]),
      (tmp_lpi_3_dfm_1[161:144]), (tmp_lpi_3_dfm_1[179:162]), (tmp_lpi_3_dfm_1[197:180]),
      (tmp_lpi_3_dfm_1[215:198]), (tmp_lpi_3_dfm_1[233:216]), (tmp_lpi_3_dfm_1[251:234]),
      (tmp_lpi_3_dfm_1[269:252]), (tmp_lpi_3_dfm_1[287:270]), (tmp_lpi_3_dfm_1[305:288]),
      (tmp_lpi_3_dfm_1[323:306]), (tmp_lpi_3_dfm_1[341:324]), (tmp_lpi_3_dfm_1[359:342]),
      (tmp_lpi_3_dfm_1[377:360]), (tmp_lpi_3_dfm_1[395:378]), (tmp_lpi_3_dfm_1[413:396]),
      (tmp_lpi_3_dfm_1[431:414]), (tmp_lpi_3_dfm_1[449:432]), (tmp_lpi_3_dfm_1[467:450]),
      (tmp_lpi_3_dfm_1[485:468]), (tmp_lpi_3_dfm_1[503:486]), (tmp_lpi_3_dfm_1[521:504]),
      (tmp_lpi_3_dfm_1[539:522]), (tmp_lpi_3_dfm_1[557:540]), (tmp_lpi_3_dfm_1[575:558]),
      (tmp_lpi_3_dfm_1[593:576]), (tmp_lpi_3_dfm_1[611:594]), (tmp_lpi_3_dfm_1[629:612]),
      (tmp_lpi_3_dfm_1[647:630]), (tmp_lpi_3_dfm_1[665:648]), (tmp_lpi_3_dfm_1[683:666]),
      (tmp_lpi_3_dfm_1[701:684]), (tmp_lpi_3_dfm_1[719:702]), (tmp_lpi_3_dfm_1[737:720]),
      (tmp_lpi_3_dfm_1[755:738]), (tmp_lpi_3_dfm_1[773:756]), (tmp_lpi_3_dfm_1[791:774]),
      (tmp_lpi_3_dfm_1[809:792]), (tmp_lpi_3_dfm_1[827:810]), (tmp_lpi_3_dfm_1[845:828]),
      (tmp_lpi_3_dfm_1[863:846]), (tmp_lpi_3_dfm_1[881:864]), (tmp_lpi_3_dfm_1[899:882]),
      (tmp_lpi_3_dfm_1[917:900]), (tmp_lpi_3_dfm_1[935:918]), (tmp_lpi_3_dfm_1[953:936]),
      (tmp_lpi_3_dfm_1[971:954]), (tmp_lpi_3_dfm_1[989:972]), (tmp_lpi_3_dfm_1[1007:990]),
      (tmp_lpi_3_dfm_1[1025:1008]), (tmp_lpi_3_dfm_1[1043:1026]), (tmp_lpi_3_dfm_1[1061:1044]),
      (tmp_lpi_3_dfm_1[1079:1062]), (tmp_lpi_3_dfm_1[1097:1080]), (tmp_lpi_3_dfm_1[1115:1098]),
      (tmp_lpi_3_dfm_1[1133:1116]), (tmp_lpi_3_dfm_1[1151:1134]), InitAccum_1_iacc_6_0_sva_5_0);
  assign nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_4_nl = MUX1HOT_v_92_6_2(({{74{w4_rsci_Q2_d_mxwt[17]}},
      w4_rsci_Q2_d_mxwt}), (signext_92_18(MultLoop_2_mux_14_nl)), ({{74{w2_rsci_Q2_d_mxwt[17]}},
      w2_rsci_Q2_d_mxwt}), ({83'b00000000000000000000000000000000000000000000000000000000000000000000000000000000000
      , ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_itm
      , 1'b0 , (InitAccum_1_iacc_6_0_sva_5_0[2:0])}), ({82'b0000000000000000000000000000000000000000000000000000000000000000000000000000000000
      , (operator_71_0_false_AC_TRN_AC_WRAP_lshift_itm[66:57])}), ({1'b0 , ac_math_ac_reciprocal_pwl_AC_TRN_71_51_false_AC_TRN_AC_WRAP_91_21_false_AC_TRN_AC_WRAP_output_temp_lpi_1_dfm}),
      {(fsm_output[8]) , (fsm_output[12]) , (fsm_output[4]) , (fsm_output[15]) ,
      (fsm_output[16]) , (fsm_output[17])});
  assign nl_z_out = $signed(({(nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_2_nl)
      , (nnet_product_input_t_config2_weight_t_config2_accum_t_1_nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl)
      , (nnet_product_input_t_config2_weight_t_config2_accum_t_1_or_3_nl) , (mux1h_189_nl)}))
      * $signed((nnet_product_input_t_config2_weight_t_config2_accum_t_1_mux1h_4_nl));
  assign z_out = nl_z_out[157:0];
  assign nl_z_out_2 = reg_ReuseLoop_1_w_index_11_6_1_reg + 4'b0001;
  assign z_out_2 = nl_z_out_2[3:0];
  assign ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_qif_mux_2_nl = MUX_v_10_2_2(({5'b00001
      , (~ (libraries_leading_sign_71_0_e45508726cf228b35de6d4ea83b9e993ba11_1[6:2]))}),
      layer3_out_63_16_0_lpi_2_dfm_9_0, fsm_output[5]);
  assign nl_z_out_3 = (ac_math_ac_normalize_71_51_false_AC_TRN_AC_WRAP_expret_qif_mux_2_nl)
      + conv_u2u_4_10(signext_4_3({(~ (fsm_output[5])) , 2'b01}));
  assign z_out_3 = nl_z_out_3[9:0];
  assign MultLoop_or_6_nl = (fsm_output[14]) | or_tmp_1141;
  assign MultLoop_mux1h_4_nl = MUX1HOT_v_18_6_2((signext_18_12({reg_ReuseLoop_1_w_index_11_6_reg
      , reg_ReuseLoop_1_w_index_11_6_1_reg , InitAccum_1_iacc_6_0_sva_5_0})), ({8'b00000001
      , ROM_1i3_1o10_bb905e8578f158e8f5b59add1dc96bdb2f_1}), MultLoop_1_mux_64_itm,
      z_out_1, 18'b000000000000000001, ({9'b000000000 , (z_out[18:10])}), {(fsm_output[3])
      , (fsm_output[16]) , or_2157_cse , (fsm_output[12]) , (MultLoop_or_6_nl) ,
      (fsm_output[15])});
  assign MultLoop_or_7_nl = (fsm_output[8]) | (fsm_output[12]) | (fsm_output[4]);
  assign MultLoop_mux1h_5_nl = MUX1HOT_v_9_5_2((signext_9_1(z_out[18])), (z_out[27:19]),
      (signext_9_1(nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva[3])),
      ({7'b0000000 , (ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_2_itm[2:1])}),
      (~ (MultLoop_mux_64_itm_mx0w0[17:9])), {(fsm_output[16]) , (MultLoop_or_7_nl)
      , (fsm_output[14]) , (fsm_output[15]) , or_tmp_1141});
  assign MultLoop_not_37_nl = ~ (fsm_output[3]);
  assign MultLoop_and_63_nl = MUX_v_9_2_2(9'b000000000, (MultLoop_mux1h_5_nl), (MultLoop_not_37_nl));
  assign MultLoop_mux1h_6_nl = MUX1HOT_s_1_4_2((z_out[18]), (nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva[3]),
      (ac_math_ac_pow2_pwl_AC_TRN_19_7_true_AC_TRN_AC_SAT_67_47_AC_TRN_AC_WRAP_output_pwl_mux_2_itm[0]),
      (~ (MultLoop_mux_64_itm_mx0w0[8])), {MultLoop_or_3_cse , (fsm_output[14]) ,
      (fsm_output[15]) , or_tmp_1141});
  assign MultLoop_and_64_nl = (MultLoop_mux1h_6_nl) & (~ (fsm_output[3]));
  assign MultLoop_mux1h_7_nl = MUX1HOT_v_8_5_2(8'b00110001, (z_out[17:10]), ({{4{nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva[3]}},
      nnet_softmax_layer6_t_result_t_softmax_config7_for_i_3_0_sva}), ({1'b1 , MultLoop_1_im_6_0_sva_1}),
      (~ (MultLoop_mux_64_itm_mx0w0[7:0])), {(fsm_output[3]) , MultLoop_or_3_cse
      , (fsm_output[14]) , (fsm_output[15]) , or_tmp_1141});
  assign nl_z_out_4 = conv_u2u_18_19(MultLoop_mux1h_4_nl) + conv_s2u_18_19({(MultLoop_and_63_nl)
      , (MultLoop_and_64_nl) , (MultLoop_mux1h_7_nl)});
  assign z_out_4 = nl_z_out_4[18:0];
  assign MultLoop_1_if_1_MultLoop_1_if_1_MultLoop_1_if_1_or_1_nl = MUX_v_2_2_2(reg_ResultLoop_1_ires_6_0_sva_5_0_tmp,
      2'b11, MultLoop_1_if_1_or_1_ssc);
  assign MultLoop_1_if_1_MultLoop_1_if_1_mux_1_nl = MUX_v_4_2_2(reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp,
      4'b1011, MultLoop_1_if_1_or_1_ssc);
  assign MultLoop_1_if_1_or_2_nl = or_dcpl_670 | or_tmp_1141;
  assign MultLoop_1_if_1_mux1h_9_nl = MUX1HOT_v_3_3_2(3'b001, (z_out_4[3:1]), (z_out_2[3:1]),
      {(MultLoop_1_if_1_or_2_nl) , (fsm_output[14]) , (fsm_output[17])});
  assign nl_z_out_5 = conv_u2u_6_7({(MultLoop_1_if_1_MultLoop_1_if_1_MultLoop_1_if_1_or_1_nl)
      , (MultLoop_1_if_1_MultLoop_1_if_1_mux_1_nl)}) + conv_u2u_3_7(MultLoop_1_if_1_mux1h_9_nl);
  assign z_out_5 = nl_z_out_5[6:0];
  assign MultLoop_1_MultLoop_1_or_1_nl = MUX_v_2_2_2(MultLoop_1_if_1_acc_itm_5_4,
      2'b11, (fsm_output[14]));
  assign MultLoop_1_mux_2_nl = MUX_v_4_2_2(MultLoop_1_if_1_acc_itm_3_0, 4'b1011,
      fsm_output[14]);
  assign MultLoop_1_mux_3_nl = MUX_v_3_2_2(3'b001, (z_out_2[3:1]), fsm_output[14]);
  assign nl_z_out_6 = conv_u2u_6_7({(MultLoop_1_MultLoop_1_or_1_nl) , (MultLoop_1_mux_2_nl)})
      + conv_u2u_3_7(MultLoop_1_mux_3_nl);
  assign z_out_6 = nl_z_out_6[6:0];
  assign or_2650_nl = (fsm_output[2]) | (fsm_output[9]) | (fsm_output[13]) | (fsm_output[6]);
  assign MultLoop_1_mux1h_3_nl = MUX1HOT_v_6_3_2(({reg_ReuseLoop_1_w_index_11_6_reg
      , reg_ReuseLoop_1_w_index_11_6_1_reg}), InitAccum_1_iacc_6_0_sva_5_0, 6'b111011,
      {(fsm_output[7]) , (or_2650_nl) , or_tmp_1201});
  assign MultLoop_1_MultLoop_1_mux_2_nl = MUX_v_3_2_2(3'b001, (MultLoop_2_im_3_0_sva_1_mx0w1[3:1]),
      or_tmp_1201);
  assign nl_z_out_7 = conv_u2u_6_7(MultLoop_1_mux1h_3_nl) + conv_u2u_3_7(MultLoop_1_MultLoop_1_mux_2_nl);
  assign z_out_7 = nl_z_out_7[6:0];
  assign MultLoop_2_mux_15_nl = MUX_v_4_2_2(reg_ResultLoop_1_ires_6_0_sva_5_0_1_tmp,
      MultLoop_1_if_1_acc_itm_3_0, fsm_output[14]);
  assign z_out_1 = MUX_v_18_10_2(nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_0_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_1_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_10_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_11_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_12_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_13_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_14_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_15_lpi_4, nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_16_lpi_4,
      nnet_dense_large_rf_leq_nin_input_t_layer2_t_config2_acc_17_lpi_4, MultLoop_2_mux_15_nl);

  function automatic [0:0] MUX1HOT_s_1_3_2;
    input [0:0] input_2;
    input [0:0] input_1;
    input [0:0] input_0;
    input [2:0] sel;
    reg [0:0] result;
  begin
    result = input_0 & {1{sel[0]}};
    result = result | ( input_1 & {1{sel[1]}});
    result = result | ( input_2 & {1{sel[2]}});
    MUX1HOT_s_1_3_2 = result;
  end
  endfunction


  function automatic [0:0] MUX1HOT_s_1_4_2;
    input [0:0] input_3;
    input [0:0] input_2;
    input [0:0] input_1;
    input [0:0] input_0;
    input [3:0] sel;
    reg [0:0] result;
  begin
    result = input_0 & {1{sel[0]}};
    result = result | ( input_1 & {1{sel[1]}});
    result = result | ( input_2 & {1{sel[2]}});
    result = result | ( input_3 & {1{sel[3]}});
    MUX1HOT_s_1_4_2 = result;
  end
  endfunction


  function automatic [9:0] MUX1HOT_v_10_4_2;
    input [9:0] input_3;
    input [9:0] input_2;
    input [9:0] input_1;
    input [9:0] input_0;
    input [3:0] sel;
    reg [9:0] result;
  begin
    result = input_0 & {10{sel[0]}};
    result = result | ( input_1 & {10{sel[1]}});
    result = result | ( input_2 & {10{sel[2]}});
    result = result | ( input_3 & {10{sel[3]}});
    MUX1HOT_v_10_4_2 = result;
  end
  endfunction


  function automatic [16:0] MUX1HOT_v_17_3_2;
    input [16:0] input_2;
    input [16:0] input_1;
    input [16:0] input_0;
    input [2:0] sel;
    reg [16:0] result;
  begin
    result = input_0 & {17{sel[0]}};
    result = result | ( input_1 & {17{sel[1]}});
    result = result | ( input_2 & {17{sel[2]}});
    MUX1HOT_v_17_3_2 = result;
  end
  endfunction


  function automatic [17:0] MUX1HOT_v_18_3_2;
    input [17:0] input_2;
    input [17:0] input_1;
    input [17:0] input_0;
    input [2:0] sel;
    reg [17:0] result;
  begin
    result = input_0 & {18{sel[0]}};
    result = result | ( input_1 & {18{sel[1]}});
    result = result | ( input_2 & {18{sel[2]}});
    MUX1HOT_v_18_3_2 = result;
  end
  endfunction


  function automatic [17:0] MUX1HOT_v_18_4_2;
    input [17:0] input_3;
    input [17:0] input_2;
    input [17:0] input_1;
    input [17:0] input_0;
    input [3:0] sel;
    reg [17:0] result;
  begin
    result = input_0 & {18{sel[0]}};
    result = result | ( input_1 & {18{sel[1]}});
    result = result | ( input_2 & {18{sel[2]}});
    result = result | ( input_3 & {18{sel[3]}});
    MUX1HOT_v_18_4_2 = result;
  end
  endfunction


  function automatic [17:0] MUX1HOT_v_18_5_2;
    input [17:0] input_4;
    input [17:0] input_3;
    input [17:0] input_2;
    input [17:0] input_1;
    input [17:0] input_0;
    input [4:0] sel;
    reg [17:0] result;
  begin
    result = input_0 & {18{sel[0]}};
    result = result | ( input_1 & {18{sel[1]}});
    result = result | ( input_2 & {18{sel[2]}});
    result = result | ( input_3 & {18{sel[3]}});
    result = result | ( input_4 & {18{sel[4]}});
    MUX1HOT_v_18_5_2 = result;
  end
  endfunction


  function automatic [17:0] MUX1HOT_v_18_6_2;
    input [17:0] input_5;
    input [17:0] input_4;
    input [17:0] input_3;
    input [17:0] input_2;
    input [17:0] input_1;
    input [17:0] input_0;
    input [5:0] sel;
    reg [17:0] result;
  begin
    result = input_0 & {18{sel[0]}};
    result = result | ( input_1 & {18{sel[1]}});
    result = result | ( input_2 & {18{sel[2]}});
    result = result | ( input_3 & {18{sel[3]}});
    result = result | ( input_4 & {18{sel[4]}});
    result = result | ( input_5 & {18{sel[5]}});
    MUX1HOT_v_18_6_2 = result;
  end
  endfunction


  function automatic [18:0] MUX1HOT_v_19_3_2;
    input [18:0] input_2;
    input [18:0] input_1;
    input [18:0] input_0;
    input [2:0] sel;
    reg [18:0] result;
  begin
    result = input_0 & {19{sel[0]}};
    result = result | ( input_1 & {19{sel[1]}});
    result = result | ( input_2 & {19{sel[2]}});
    MUX1HOT_v_19_3_2 = result;
  end
  endfunction


  function automatic [2:0] MUX1HOT_v_3_3_2;
    input [2:0] input_2;
    input [2:0] input_1;
    input [2:0] input_0;
    input [2:0] sel;
    reg [2:0] result;
  begin
    result = input_0 & {3{sel[0]}};
    result = result | ( input_1 & {3{sel[1]}});
    result = result | ( input_2 & {3{sel[2]}});
    MUX1HOT_v_3_3_2 = result;
  end
  endfunction


  function automatic [3:0] MUX1HOT_v_4_3_2;
    input [3:0] input_2;
    input [3:0] input_1;
    input [3:0] input_0;
    input [2:0] sel;
    reg [3:0] result;
  begin
    result = input_0 & {4{sel[0]}};
    result = result | ( input_1 & {4{sel[1]}});
    result = result | ( input_2 & {4{sel[2]}});
    MUX1HOT_v_4_3_2 = result;
  end
  endfunction


  function automatic [3:0] MUX1HOT_v_4_4_2;
    input [3:0] input_3;
    input [3:0] input_2;
    input [3:0] input_1;
    input [3:0] input_0;
    input [3:0] sel;
    reg [3:0] result;
  begin
    result = input_0 & {4{sel[0]}};
    result = result | ( input_1 & {4{sel[1]}});
    result = result | ( input_2 & {4{sel[2]}});
    result = result | ( input_3 & {4{sel[3]}});
    MUX1HOT_v_4_4_2 = result;
  end
  endfunction


  function automatic [5:0] MUX1HOT_v_6_3_2;
    input [5:0] input_2;
    input [5:0] input_1;
    input [5:0] input_0;
    input [2:0] sel;
    reg [5:0] result;
  begin
    result = input_0 & {6{sel[0]}};
    result = result | ( input_1 & {6{sel[1]}});
    result = result | ( input_2 & {6{sel[2]}});
    MUX1HOT_v_6_3_2 = result;
  end
  endfunction


  function automatic [5:0] MUX1HOT_v_6_4_2;
    input [5:0] input_3;
    input [5:0] input_2;
    input [5:0] input_1;
    input [5:0] input_0;
    input [3:0] sel;
    reg [5:0] result;
  begin
    result = input_0 & {6{sel[0]}};
    result = result | ( input_1 & {6{sel[1]}});
    result = result | ( input_2 & {6{sel[2]}});
    result = result | ( input_3 & {6{sel[3]}});
    MUX1HOT_v_6_4_2 = result;
  end
  endfunction


  function automatic [6:0] MUX1HOT_v_7_3_2;
    input [6:0] input_2;
    input [6:0] input_1;
    input [6:0] input_0;
    input [2:0] sel;
    reg [6:0] result;
  begin
    result = input_0 & {7{sel[0]}};
    result = result | ( input_1 & {7{sel[1]}});
    result = result | ( input_2 & {7{sel[2]}});
    MUX1HOT_v_7_3_2 = result;
  end
  endfunction


  function automatic [6:0] MUX1HOT_v_7_6_2;
    input [6:0] input_5;
    input [6:0] input_4;
    input [6:0] input_3;
    input [6:0] input_2;
    input [6:0] input_1;
    input [6:0] input_0;
    input [5:0] sel;
    reg [6:0] result;
  begin
    result = input_0 & {7{sel[0]}};
    result = result | ( input_1 & {7{sel[1]}});
    result = result | ( input_2 & {7{sel[2]}});
    result = result | ( input_3 & {7{sel[3]}});
    result = result | ( input_4 & {7{sel[4]}});
    result = result | ( input_5 & {7{sel[5]}});
    MUX1HOT_v_7_6_2 = result;
  end
  endfunction


  function automatic [7:0] MUX1HOT_v_8_5_2;
    input [7:0] input_4;
    input [7:0] input_3;
    input [7:0] input_2;
    input [7:0] input_1;
    input [7:0] input_0;
    input [4:0] sel;
    reg [7:0] result;
  begin
    result = input_0 & {8{sel[0]}};
    result = result | ( input_1 & {8{sel[1]}});
    result = result | ( input_2 & {8{sel[2]}});
    result = result | ( input_3 & {8{sel[3]}});
    result = result | ( input_4 & {8{sel[4]}});
    MUX1HOT_v_8_5_2 = result;
  end
  endfunction


  function automatic [91:0] MUX1HOT_v_92_6_2;
    input [91:0] input_5;
    input [91:0] input_4;
    input [91:0] input_3;
    input [91:0] input_2;
    input [91:0] input_1;
    input [91:0] input_0;
    input [5:0] sel;
    reg [91:0] result;
  begin
    result = input_0 & {92{sel[0]}};
    result = result | ( input_1 & {92{sel[1]}});
    result = result | ( input_2 & {92{sel[2]}});
    result = result | ( input_3 & {92{sel[3]}});
    result = result | ( input_4 & {92{sel[4]}});
    result = result | ( input_5 & {92{sel[5]}});
    MUX1HOT_v_92_6_2 = result;
  end
  endfunction


  function automatic [8:0] MUX1HOT_v_9_5_2;
    input [8:0] input_4;
    input [8:0] input_3;
    input [8:0] input_2;
    input [8:0] input_1;
    input [8:0] input_0;
    input [4:0] sel;
    reg [8:0] result;
  begin
    result = input_0 & {9{sel[0]}};
    result = result | ( input_1 & {9{sel[1]}});
    result = result | ( input_2 & {9{sel[2]}});
    result = result | ( input_3 & {9{sel[3]}});
    result = result | ( input_4 & {9{sel[4]}});
    MUX1HOT_v_9_5_2 = result;
  end
  endfunction


  function automatic [0:0] MUX_s_1_2_2;
    input [0:0] input_0;
    input [0:0] input_1;
    input [0:0] sel;
    reg [0:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_s_1_2_2 = result;
  end
  endfunction


  function automatic [9:0] MUX_v_10_10_2;
    input [9:0] input_0;
    input [9:0] input_1;
    input [9:0] input_2;
    input [9:0] input_3;
    input [9:0] input_4;
    input [9:0] input_5;
    input [9:0] input_6;
    input [9:0] input_7;
    input [9:0] input_8;
    input [9:0] input_9;
    input [3:0] sel;
    reg [9:0] result;
  begin
    case (sel)
      4'b0000 : begin
        result = input_0;
      end
      4'b0001 : begin
        result = input_1;
      end
      4'b0010 : begin
        result = input_2;
      end
      4'b0011 : begin
        result = input_3;
      end
      4'b0100 : begin
        result = input_4;
      end
      4'b0101 : begin
        result = input_5;
      end
      4'b0110 : begin
        result = input_6;
      end
      4'b0111 : begin
        result = input_7;
      end
      4'b1000 : begin
        result = input_8;
      end
      default : begin
        result = input_9;
      end
    endcase
    MUX_v_10_10_2 = result;
  end
  endfunction


  function automatic [9:0] MUX_v_10_2_2;
    input [9:0] input_0;
    input [9:0] input_1;
    input [0:0] sel;
    reg [9:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_10_2_2 = result;
  end
  endfunction


  function automatic [9:0] MUX_v_10_64_2;
    input [9:0] input_0;
    input [9:0] input_1;
    input [9:0] input_2;
    input [9:0] input_3;
    input [9:0] input_4;
    input [9:0] input_5;
    input [9:0] input_6;
    input [9:0] input_7;
    input [9:0] input_8;
    input [9:0] input_9;
    input [9:0] input_10;
    input [9:0] input_11;
    input [9:0] input_12;
    input [9:0] input_13;
    input [9:0] input_14;
    input [9:0] input_15;
    input [9:0] input_16;
    input [9:0] input_17;
    input [9:0] input_18;
    input [9:0] input_19;
    input [9:0] input_20;
    input [9:0] input_21;
    input [9:0] input_22;
    input [9:0] input_23;
    input [9:0] input_24;
    input [9:0] input_25;
    input [9:0] input_26;
    input [9:0] input_27;
    input [9:0] input_28;
    input [9:0] input_29;
    input [9:0] input_30;
    input [9:0] input_31;
    input [9:0] input_32;
    input [9:0] input_33;
    input [9:0] input_34;
    input [9:0] input_35;
    input [9:0] input_36;
    input [9:0] input_37;
    input [9:0] input_38;
    input [9:0] input_39;
    input [9:0] input_40;
    input [9:0] input_41;
    input [9:0] input_42;
    input [9:0] input_43;
    input [9:0] input_44;
    input [9:0] input_45;
    input [9:0] input_46;
    input [9:0] input_47;
    input [9:0] input_48;
    input [9:0] input_49;
    input [9:0] input_50;
    input [9:0] input_51;
    input [9:0] input_52;
    input [9:0] input_53;
    input [9:0] input_54;
    input [9:0] input_55;
    input [9:0] input_56;
    input [9:0] input_57;
    input [9:0] input_58;
    input [9:0] input_59;
    input [9:0] input_60;
    input [9:0] input_61;
    input [9:0] input_62;
    input [9:0] input_63;
    input [5:0] sel;
    reg [9:0] result;
  begin
    case (sel)
      6'b000000 : begin
        result = input_0;
      end
      6'b000001 : begin
        result = input_1;
      end
      6'b000010 : begin
        result = input_2;
      end
      6'b000011 : begin
        result = input_3;
      end
      6'b000100 : begin
        result = input_4;
      end
      6'b000101 : begin
        result = input_5;
      end
      6'b000110 : begin
        result = input_6;
      end
      6'b000111 : begin
        result = input_7;
      end
      6'b001000 : begin
        result = input_8;
      end
      6'b001001 : begin
        result = input_9;
      end
      6'b001010 : begin
        result = input_10;
      end
      6'b001011 : begin
        result = input_11;
      end
      6'b001100 : begin
        result = input_12;
      end
      6'b001101 : begin
        result = input_13;
      end
      6'b001110 : begin
        result = input_14;
      end
      6'b001111 : begin
        result = input_15;
      end
      6'b010000 : begin
        result = input_16;
      end
      6'b010001 : begin
        result = input_17;
      end
      6'b010010 : begin
        result = input_18;
      end
      6'b010011 : begin
        result = input_19;
      end
      6'b010100 : begin
        result = input_20;
      end
      6'b010101 : begin
        result = input_21;
      end
      6'b010110 : begin
        result = input_22;
      end
      6'b010111 : begin
        result = input_23;
      end
      6'b011000 : begin
        result = input_24;
      end
      6'b011001 : begin
        result = input_25;
      end
      6'b011010 : begin
        result = input_26;
      end
      6'b011011 : begin
        result = input_27;
      end
      6'b011100 : begin
        result = input_28;
      end
      6'b011101 : begin
        result = input_29;
      end
      6'b011110 : begin
        result = input_30;
      end
      6'b011111 : begin
        result = input_31;
      end
      6'b100000 : begin
        result = input_32;
      end
      6'b100001 : begin
        result = input_33;
      end
      6'b100010 : begin
        result = input_34;
      end
      6'b100011 : begin
        result = input_35;
      end
      6'b100100 : begin
        result = input_36;
      end
      6'b100101 : begin
        result = input_37;
      end
      6'b100110 : begin
        result = input_38;
      end
      6'b100111 : begin
        result = input_39;
      end
      6'b101000 : begin
        result = input_40;
      end
      6'b101001 : begin
        result = input_41;
      end
      6'b101010 : begin
        result = input_42;
      end
      6'b101011 : begin
        result = input_43;
      end
      6'b101100 : begin
        result = input_44;
      end
      6'b101101 : begin
        result = input_45;
      end
      6'b101110 : begin
        result = input_46;
      end
      6'b101111 : begin
        result = input_47;
      end
      6'b110000 : begin
        result = input_48;
      end
      6'b110001 : begin
        result = input_49;
      end
      6'b110010 : begin
        result = input_50;
      end
      6'b110011 : begin
        result = input_51;
      end
      6'b110100 : begin
        result = input_52;
      end
      6'b110101 : begin
        result = input_53;
      end
      6'b110110 : begin
        result = input_54;
      end
      6'b110111 : begin
        result = input_55;
      end
      6'b111000 : begin
        result = input_56;
      end
      6'b111001 : begin
        result = input_57;
      end
      6'b111010 : begin
        result = input_58;
      end
      6'b111011 : begin
        result = input_59;
      end
      6'b111100 : begin
        result = input_60;
      end
      6'b111101 : begin
        result = input_61;
      end
      6'b111110 : begin
        result = input_62;
      end
      default : begin
        result = input_63;
      end
    endcase
    MUX_v_10_64_2 = result;
  end
  endfunction


  function automatic [1151:0] MUX_v_1152_10_2;
    input [1151:0] input_0;
    input [1151:0] input_1;
    input [1151:0] input_2;
    input [1151:0] input_3;
    input [1151:0] input_4;
    input [1151:0] input_5;
    input [1151:0] input_6;
    input [1151:0] input_7;
    input [1151:0] input_8;
    input [1151:0] input_9;
    input [3:0] sel;
    reg [1151:0] result;
  begin
    case (sel)
      4'b0000 : begin
        result = input_0;
      end
      4'b0001 : begin
        result = input_1;
      end
      4'b0010 : begin
        result = input_2;
      end
      4'b0011 : begin
        result = input_3;
      end
      4'b0100 : begin
        result = input_4;
      end
      4'b0101 : begin
        result = input_5;
      end
      4'b0110 : begin
        result = input_6;
      end
      4'b0111 : begin
        result = input_7;
      end
      4'b1000 : begin
        result = input_8;
      end
      default : begin
        result = input_9;
      end
    endcase
    MUX_v_1152_10_2 = result;
  end
  endfunction


  function automatic [11:0] MUX_v_12_2_2;
    input [11:0] input_0;
    input [11:0] input_1;
    input [0:0] sel;
    reg [11:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_12_2_2 = result;
  end
  endfunction


  function automatic [14:0] MUX_v_15_2_2;
    input [14:0] input_0;
    input [14:0] input_1;
    input [0:0] sel;
    reg [14:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_15_2_2 = result;
  end
  endfunction


  function automatic [16:0] MUX_v_17_2_2;
    input [16:0] input_0;
    input [16:0] input_1;
    input [0:0] sel;
    reg [16:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_17_2_2 = result;
  end
  endfunction


  function automatic [16:0] MUX_v_17_64_2;
    input [16:0] input_0;
    input [16:0] input_1;
    input [16:0] input_2;
    input [16:0] input_3;
    input [16:0] input_4;
    input [16:0] input_5;
    input [16:0] input_6;
    input [16:0] input_7;
    input [16:0] input_8;
    input [16:0] input_9;
    input [16:0] input_10;
    input [16:0] input_11;
    input [16:0] input_12;
    input [16:0] input_13;
    input [16:0] input_14;
    input [16:0] input_15;
    input [16:0] input_16;
    input [16:0] input_17;
    input [16:0] input_18;
    input [16:0] input_19;
    input [16:0] input_20;
    input [16:0] input_21;
    input [16:0] input_22;
    input [16:0] input_23;
    input [16:0] input_24;
    input [16:0] input_25;
    input [16:0] input_26;
    input [16:0] input_27;
    input [16:0] input_28;
    input [16:0] input_29;
    input [16:0] input_30;
    input [16:0] input_31;
    input [16:0] input_32;
    input [16:0] input_33;
    input [16:0] input_34;
    input [16:0] input_35;
    input [16:0] input_36;
    input [16:0] input_37;
    input [16:0] input_38;
    input [16:0] input_39;
    input [16:0] input_40;
    input [16:0] input_41;
    input [16:0] input_42;
    input [16:0] input_43;
    input [16:0] input_44;
    input [16:0] input_45;
    input [16:0] input_46;
    input [16:0] input_47;
    input [16:0] input_48;
    input [16:0] input_49;
    input [16:0] input_50;
    input [16:0] input_51;
    input [16:0] input_52;
    input [16:0] input_53;
    input [16:0] input_54;
    input [16:0] input_55;
    input [16:0] input_56;
    input [16:0] input_57;
    input [16:0] input_58;
    input [16:0] input_59;
    input [16:0] input_60;
    input [16:0] input_61;
    input [16:0] input_62;
    input [16:0] input_63;
    input [5:0] sel;
    reg [16:0] result;
  begin
    case (sel)
      6'b000000 : begin
        result = input_0;
      end
      6'b000001 : begin
        result = input_1;
      end
      6'b000010 : begin
        result = input_2;
      end
      6'b000011 : begin
        result = input_3;
      end
      6'b000100 : begin
        result = input_4;
      end
      6'b000101 : begin
        result = input_5;
      end
      6'b000110 : begin
        result = input_6;
      end
      6'b000111 : begin
        result = input_7;
      end
      6'b001000 : begin
        result = input_8;
      end
      6'b001001 : begin
        result = input_9;
      end
      6'b001010 : begin
        result = input_10;
      end
      6'b001011 : begin
        result = input_11;
      end
      6'b001100 : begin
        result = input_12;
      end
      6'b001101 : begin
        result = input_13;
      end
      6'b001110 : begin
        result = input_14;
      end
      6'b001111 : begin
        result = input_15;
      end
      6'b010000 : begin
        result = input_16;
      end
      6'b010001 : begin
        result = input_17;
      end
      6'b010010 : begin
        result = input_18;
      end
      6'b010011 : begin
        result = input_19;
      end
      6'b010100 : begin
        result = input_20;
      end
      6'b010101 : begin
        result = input_21;
      end
      6'b010110 : begin
        result = input_22;
      end
      6'b010111 : begin
        result = input_23;
      end
      6'b011000 : begin
        result = input_24;
      end
      6'b011001 : begin
        result = input_25;
      end
      6'b011010 : begin
        result = input_26;
      end
      6'b011011 : begin
        result = input_27;
      end
      6'b011100 : begin
        result = input_28;
      end
      6'b011101 : begin
        result = input_29;
      end
      6'b011110 : begin
        result = input_30;
      end
      6'b011111 : begin
        result = input_31;
      end
      6'b100000 : begin
        result = input_32;
      end
      6'b100001 : begin
        result = input_33;
      end
      6'b100010 : begin
        result = input_34;
      end
      6'b100011 : begin
        result = input_35;
      end
      6'b100100 : begin
        result = input_36;
      end
      6'b100101 : begin
        result = input_37;
      end
      6'b100110 : begin
        result = input_38;
      end
      6'b100111 : begin
        result = input_39;
      end
      6'b101000 : begin
        result = input_40;
      end
      6'b101001 : begin
        result = input_41;
      end
      6'b101010 : begin
        result = input_42;
      end
      6'b101011 : begin
        result = input_43;
      end
      6'b101100 : begin
        result = input_44;
      end
      6'b101101 : begin
        result = input_45;
      end
      6'b101110 : begin
        result = input_46;
      end
      6'b101111 : begin
        result = input_47;
      end
      6'b110000 : begin
        result = input_48;
      end
      6'b110001 : begin
        result = input_49;
      end
      6'b110010 : begin
        result = input_50;
      end
      6'b110011 : begin
        result = input_51;
      end
      6'b110100 : begin
        result = input_52;
      end
      6'b110101 : begin
        result = input_53;
      end
      6'b110110 : begin
        result = input_54;
      end
      6'b110111 : begin
        result = input_55;
      end
      6'b111000 : begin
        result = input_56;
      end
      6'b111001 : begin
        result = input_57;
      end
      6'b111010 : begin
        result = input_58;
      end
      6'b111011 : begin
        result = input_59;
      end
      6'b111100 : begin
        result = input_60;
      end
      6'b111101 : begin
        result = input_61;
      end
      6'b111110 : begin
        result = input_62;
      end
      default : begin
        result = input_63;
      end
    endcase
    MUX_v_17_64_2 = result;
  end
  endfunction


  function automatic [17:0] MUX_v_18_10_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [17:0] input_2;
    input [17:0] input_3;
    input [17:0] input_4;
    input [17:0] input_5;
    input [17:0] input_6;
    input [17:0] input_7;
    input [17:0] input_8;
    input [17:0] input_9;
    input [3:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      4'b0000 : begin
        result = input_0;
      end
      4'b0001 : begin
        result = input_1;
      end
      4'b0010 : begin
        result = input_2;
      end
      4'b0011 : begin
        result = input_3;
      end
      4'b0100 : begin
        result = input_4;
      end
      4'b0101 : begin
        result = input_5;
      end
      4'b0110 : begin
        result = input_6;
      end
      4'b0111 : begin
        result = input_7;
      end
      4'b1000 : begin
        result = input_8;
      end
      default : begin
        result = input_9;
      end
    endcase
    MUX_v_18_10_2 = result;
  end
  endfunction


  function automatic [17:0] MUX_v_18_2_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [0:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_18_2_2 = result;
  end
  endfunction


  function automatic [17:0] MUX_v_18_64_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [17:0] input_2;
    input [17:0] input_3;
    input [17:0] input_4;
    input [17:0] input_5;
    input [17:0] input_6;
    input [17:0] input_7;
    input [17:0] input_8;
    input [17:0] input_9;
    input [17:0] input_10;
    input [17:0] input_11;
    input [17:0] input_12;
    input [17:0] input_13;
    input [17:0] input_14;
    input [17:0] input_15;
    input [17:0] input_16;
    input [17:0] input_17;
    input [17:0] input_18;
    input [17:0] input_19;
    input [17:0] input_20;
    input [17:0] input_21;
    input [17:0] input_22;
    input [17:0] input_23;
    input [17:0] input_24;
    input [17:0] input_25;
    input [17:0] input_26;
    input [17:0] input_27;
    input [17:0] input_28;
    input [17:0] input_29;
    input [17:0] input_30;
    input [17:0] input_31;
    input [17:0] input_32;
    input [17:0] input_33;
    input [17:0] input_34;
    input [17:0] input_35;
    input [17:0] input_36;
    input [17:0] input_37;
    input [17:0] input_38;
    input [17:0] input_39;
    input [17:0] input_40;
    input [17:0] input_41;
    input [17:0] input_42;
    input [17:0] input_43;
    input [17:0] input_44;
    input [17:0] input_45;
    input [17:0] input_46;
    input [17:0] input_47;
    input [17:0] input_48;
    input [17:0] input_49;
    input [17:0] input_50;
    input [17:0] input_51;
    input [17:0] input_52;
    input [17:0] input_53;
    input [17:0] input_54;
    input [17:0] input_55;
    input [17:0] input_56;
    input [17:0] input_57;
    input [17:0] input_58;
    input [17:0] input_59;
    input [17:0] input_60;
    input [17:0] input_61;
    input [17:0] input_62;
    input [17:0] input_63;
    input [5:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      6'b000000 : begin
        result = input_0;
      end
      6'b000001 : begin
        result = input_1;
      end
      6'b000010 : begin
        result = input_2;
      end
      6'b000011 : begin
        result = input_3;
      end
      6'b000100 : begin
        result = input_4;
      end
      6'b000101 : begin
        result = input_5;
      end
      6'b000110 : begin
        result = input_6;
      end
      6'b000111 : begin
        result = input_7;
      end
      6'b001000 : begin
        result = input_8;
      end
      6'b001001 : begin
        result = input_9;
      end
      6'b001010 : begin
        result = input_10;
      end
      6'b001011 : begin
        result = input_11;
      end
      6'b001100 : begin
        result = input_12;
      end
      6'b001101 : begin
        result = input_13;
      end
      6'b001110 : begin
        result = input_14;
      end
      6'b001111 : begin
        result = input_15;
      end
      6'b010000 : begin
        result = input_16;
      end
      6'b010001 : begin
        result = input_17;
      end
      6'b010010 : begin
        result = input_18;
      end
      6'b010011 : begin
        result = input_19;
      end
      6'b010100 : begin
        result = input_20;
      end
      6'b010101 : begin
        result = input_21;
      end
      6'b010110 : begin
        result = input_22;
      end
      6'b010111 : begin
        result = input_23;
      end
      6'b011000 : begin
        result = input_24;
      end
      6'b011001 : begin
        result = input_25;
      end
      6'b011010 : begin
        result = input_26;
      end
      6'b011011 : begin
        result = input_27;
      end
      6'b011100 : begin
        result = input_28;
      end
      6'b011101 : begin
        result = input_29;
      end
      6'b011110 : begin
        result = input_30;
      end
      6'b011111 : begin
        result = input_31;
      end
      6'b100000 : begin
        result = input_32;
      end
      6'b100001 : begin
        result = input_33;
      end
      6'b100010 : begin
        result = input_34;
      end
      6'b100011 : begin
        result = input_35;
      end
      6'b100100 : begin
        result = input_36;
      end
      6'b100101 : begin
        result = input_37;
      end
      6'b100110 : begin
        result = input_38;
      end
      6'b100111 : begin
        result = input_39;
      end
      6'b101000 : begin
        result = input_40;
      end
      6'b101001 : begin
        result = input_41;
      end
      6'b101010 : begin
        result = input_42;
      end
      6'b101011 : begin
        result = input_43;
      end
      6'b101100 : begin
        result = input_44;
      end
      6'b101101 : begin
        result = input_45;
      end
      6'b101110 : begin
        result = input_46;
      end
      6'b101111 : begin
        result = input_47;
      end
      6'b110000 : begin
        result = input_48;
      end
      6'b110001 : begin
        result = input_49;
      end
      6'b110010 : begin
        result = input_50;
      end
      6'b110011 : begin
        result = input_51;
      end
      6'b110100 : begin
        result = input_52;
      end
      6'b110101 : begin
        result = input_53;
      end
      6'b110110 : begin
        result = input_54;
      end
      6'b110111 : begin
        result = input_55;
      end
      6'b111000 : begin
        result = input_56;
      end
      6'b111001 : begin
        result = input_57;
      end
      6'b111010 : begin
        result = input_58;
      end
      6'b111011 : begin
        result = input_59;
      end
      6'b111100 : begin
        result = input_60;
      end
      6'b111101 : begin
        result = input_61;
      end
      6'b111110 : begin
        result = input_62;
      end
      default : begin
        result = input_63;
      end
    endcase
    MUX_v_18_64_2 = result;
  end
  endfunction


  function automatic [17:0] MUX_v_18_784_2;
    input [17:0] input_0;
    input [17:0] input_1;
    input [17:0] input_2;
    input [17:0] input_3;
    input [17:0] input_4;
    input [17:0] input_5;
    input [17:0] input_6;
    input [17:0] input_7;
    input [17:0] input_8;
    input [17:0] input_9;
    input [17:0] input_10;
    input [17:0] input_11;
    input [17:0] input_12;
    input [17:0] input_13;
    input [17:0] input_14;
    input [17:0] input_15;
    input [17:0] input_16;
    input [17:0] input_17;
    input [17:0] input_18;
    input [17:0] input_19;
    input [17:0] input_20;
    input [17:0] input_21;
    input [17:0] input_22;
    input [17:0] input_23;
    input [17:0] input_24;
    input [17:0] input_25;
    input [17:0] input_26;
    input [17:0] input_27;
    input [17:0] input_28;
    input [17:0] input_29;
    input [17:0] input_30;
    input [17:0] input_31;
    input [17:0] input_32;
    input [17:0] input_33;
    input [17:0] input_34;
    input [17:0] input_35;
    input [17:0] input_36;
    input [17:0] input_37;
    input [17:0] input_38;
    input [17:0] input_39;
    input [17:0] input_40;
    input [17:0] input_41;
    input [17:0] input_42;
    input [17:0] input_43;
    input [17:0] input_44;
    input [17:0] input_45;
    input [17:0] input_46;
    input [17:0] input_47;
    input [17:0] input_48;
    input [17:0] input_49;
    input [17:0] input_50;
    input [17:0] input_51;
    input [17:0] input_52;
    input [17:0] input_53;
    input [17:0] input_54;
    input [17:0] input_55;
    input [17:0] input_56;
    input [17:0] input_57;
    input [17:0] input_58;
    input [17:0] input_59;
    input [17:0] input_60;
    input [17:0] input_61;
    input [17:0] input_62;
    input [17:0] input_63;
    input [17:0] input_64;
    input [17:0] input_65;
    input [17:0] input_66;
    input [17:0] input_67;
    input [17:0] input_68;
    input [17:0] input_69;
    input [17:0] input_70;
    input [17:0] input_71;
    input [17:0] input_72;
    input [17:0] input_73;
    input [17:0] input_74;
    input [17:0] input_75;
    input [17:0] input_76;
    input [17:0] input_77;
    input [17:0] input_78;
    input [17:0] input_79;
    input [17:0] input_80;
    input [17:0] input_81;
    input [17:0] input_82;
    input [17:0] input_83;
    input [17:0] input_84;
    input [17:0] input_85;
    input [17:0] input_86;
    input [17:0] input_87;
    input [17:0] input_88;
    input [17:0] input_89;
    input [17:0] input_90;
    input [17:0] input_91;
    input [17:0] input_92;
    input [17:0] input_93;
    input [17:0] input_94;
    input [17:0] input_95;
    input [17:0] input_96;
    input [17:0] input_97;
    input [17:0] input_98;
    input [17:0] input_99;
    input [17:0] input_100;
    input [17:0] input_101;
    input [17:0] input_102;
    input [17:0] input_103;
    input [17:0] input_104;
    input [17:0] input_105;
    input [17:0] input_106;
    input [17:0] input_107;
    input [17:0] input_108;
    input [17:0] input_109;
    input [17:0] input_110;
    input [17:0] input_111;
    input [17:0] input_112;
    input [17:0] input_113;
    input [17:0] input_114;
    input [17:0] input_115;
    input [17:0] input_116;
    input [17:0] input_117;
    input [17:0] input_118;
    input [17:0] input_119;
    input [17:0] input_120;
    input [17:0] input_121;
    input [17:0] input_122;
    input [17:0] input_123;
    input [17:0] input_124;
    input [17:0] input_125;
    input [17:0] input_126;
    input [17:0] input_127;
    input [17:0] input_128;
    input [17:0] input_129;
    input [17:0] input_130;
    input [17:0] input_131;
    input [17:0] input_132;
    input [17:0] input_133;
    input [17:0] input_134;
    input [17:0] input_135;
    input [17:0] input_136;
    input [17:0] input_137;
    input [17:0] input_138;
    input [17:0] input_139;
    input [17:0] input_140;
    input [17:0] input_141;
    input [17:0] input_142;
    input [17:0] input_143;
    input [17:0] input_144;
    input [17:0] input_145;
    input [17:0] input_146;
    input [17:0] input_147;
    input [17:0] input_148;
    input [17:0] input_149;
    input [17:0] input_150;
    input [17:0] input_151;
    input [17:0] input_152;
    input [17:0] input_153;
    input [17:0] input_154;
    input [17:0] input_155;
    input [17:0] input_156;
    input [17:0] input_157;
    input [17:0] input_158;
    input [17:0] input_159;
    input [17:0] input_160;
    input [17:0] input_161;
    input [17:0] input_162;
    input [17:0] input_163;
    input [17:0] input_164;
    input [17:0] input_165;
    input [17:0] input_166;
    input [17:0] input_167;
    input [17:0] input_168;
    input [17:0] input_169;
    input [17:0] input_170;
    input [17:0] input_171;
    input [17:0] input_172;
    input [17:0] input_173;
    input [17:0] input_174;
    input [17:0] input_175;
    input [17:0] input_176;
    input [17:0] input_177;
    input [17:0] input_178;
    input [17:0] input_179;
    input [17:0] input_180;
    input [17:0] input_181;
    input [17:0] input_182;
    input [17:0] input_183;
    input [17:0] input_184;
    input [17:0] input_185;
    input [17:0] input_186;
    input [17:0] input_187;
    input [17:0] input_188;
    input [17:0] input_189;
    input [17:0] input_190;
    input [17:0] input_191;
    input [17:0] input_192;
    input [17:0] input_193;
    input [17:0] input_194;
    input [17:0] input_195;
    input [17:0] input_196;
    input [17:0] input_197;
    input [17:0] input_198;
    input [17:0] input_199;
    input [17:0] input_200;
    input [17:0] input_201;
    input [17:0] input_202;
    input [17:0] input_203;
    input [17:0] input_204;
    input [17:0] input_205;
    input [17:0] input_206;
    input [17:0] input_207;
    input [17:0] input_208;
    input [17:0] input_209;
    input [17:0] input_210;
    input [17:0] input_211;
    input [17:0] input_212;
    input [17:0] input_213;
    input [17:0] input_214;
    input [17:0] input_215;
    input [17:0] input_216;
    input [17:0] input_217;
    input [17:0] input_218;
    input [17:0] input_219;
    input [17:0] input_220;
    input [17:0] input_221;
    input [17:0] input_222;
    input [17:0] input_223;
    input [17:0] input_224;
    input [17:0] input_225;
    input [17:0] input_226;
    input [17:0] input_227;
    input [17:0] input_228;
    input [17:0] input_229;
    input [17:0] input_230;
    input [17:0] input_231;
    input [17:0] input_232;
    input [17:0] input_233;
    input [17:0] input_234;
    input [17:0] input_235;
    input [17:0] input_236;
    input [17:0] input_237;
    input [17:0] input_238;
    input [17:0] input_239;
    input [17:0] input_240;
    input [17:0] input_241;
    input [17:0] input_242;
    input [17:0] input_243;
    input [17:0] input_244;
    input [17:0] input_245;
    input [17:0] input_246;
    input [17:0] input_247;
    input [17:0] input_248;
    input [17:0] input_249;
    input [17:0] input_250;
    input [17:0] input_251;
    input [17:0] input_252;
    input [17:0] input_253;
    input [17:0] input_254;
    input [17:0] input_255;
    input [17:0] input_256;
    input [17:0] input_257;
    input [17:0] input_258;
    input [17:0] input_259;
    input [17:0] input_260;
    input [17:0] input_261;
    input [17:0] input_262;
    input [17:0] input_263;
    input [17:0] input_264;
    input [17:0] input_265;
    input [17:0] input_266;
    input [17:0] input_267;
    input [17:0] input_268;
    input [17:0] input_269;
    input [17:0] input_270;
    input [17:0] input_271;
    input [17:0] input_272;
    input [17:0] input_273;
    input [17:0] input_274;
    input [17:0] input_275;
    input [17:0] input_276;
    input [17:0] input_277;
    input [17:0] input_278;
    input [17:0] input_279;
    input [17:0] input_280;
    input [17:0] input_281;
    input [17:0] input_282;
    input [17:0] input_283;
    input [17:0] input_284;
    input [17:0] input_285;
    input [17:0] input_286;
    input [17:0] input_287;
    input [17:0] input_288;
    input [17:0] input_289;
    input [17:0] input_290;
    input [17:0] input_291;
    input [17:0] input_292;
    input [17:0] input_293;
    input [17:0] input_294;
    input [17:0] input_295;
    input [17:0] input_296;
    input [17:0] input_297;
    input [17:0] input_298;
    input [17:0] input_299;
    input [17:0] input_300;
    input [17:0] input_301;
    input [17:0] input_302;
    input [17:0] input_303;
    input [17:0] input_304;
    input [17:0] input_305;
    input [17:0] input_306;
    input [17:0] input_307;
    input [17:0] input_308;
    input [17:0] input_309;
    input [17:0] input_310;
    input [17:0] input_311;
    input [17:0] input_312;
    input [17:0] input_313;
    input [17:0] input_314;
    input [17:0] input_315;
    input [17:0] input_316;
    input [17:0] input_317;
    input [17:0] input_318;
    input [17:0] input_319;
    input [17:0] input_320;
    input [17:0] input_321;
    input [17:0] input_322;
    input [17:0] input_323;
    input [17:0] input_324;
    input [17:0] input_325;
    input [17:0] input_326;
    input [17:0] input_327;
    input [17:0] input_328;
    input [17:0] input_329;
    input [17:0] input_330;
    input [17:0] input_331;
    input [17:0] input_332;
    input [17:0] input_333;
    input [17:0] input_334;
    input [17:0] input_335;
    input [17:0] input_336;
    input [17:0] input_337;
    input [17:0] input_338;
    input [17:0] input_339;
    input [17:0] input_340;
    input [17:0] input_341;
    input [17:0] input_342;
    input [17:0] input_343;
    input [17:0] input_344;
    input [17:0] input_345;
    input [17:0] input_346;
    input [17:0] input_347;
    input [17:0] input_348;
    input [17:0] input_349;
    input [17:0] input_350;
    input [17:0] input_351;
    input [17:0] input_352;
    input [17:0] input_353;
    input [17:0] input_354;
    input [17:0] input_355;
    input [17:0] input_356;
    input [17:0] input_357;
    input [17:0] input_358;
    input [17:0] input_359;
    input [17:0] input_360;
    input [17:0] input_361;
    input [17:0] input_362;
    input [17:0] input_363;
    input [17:0] input_364;
    input [17:0] input_365;
    input [17:0] input_366;
    input [17:0] input_367;
    input [17:0] input_368;
    input [17:0] input_369;
    input [17:0] input_370;
    input [17:0] input_371;
    input [17:0] input_372;
    input [17:0] input_373;
    input [17:0] input_374;
    input [17:0] input_375;
    input [17:0] input_376;
    input [17:0] input_377;
    input [17:0] input_378;
    input [17:0] input_379;
    input [17:0] input_380;
    input [17:0] input_381;
    input [17:0] input_382;
    input [17:0] input_383;
    input [17:0] input_384;
    input [17:0] input_385;
    input [17:0] input_386;
    input [17:0] input_387;
    input [17:0] input_388;
    input [17:0] input_389;
    input [17:0] input_390;
    input [17:0] input_391;
    input [17:0] input_392;
    input [17:0] input_393;
    input [17:0] input_394;
    input [17:0] input_395;
    input [17:0] input_396;
    input [17:0] input_397;
    input [17:0] input_398;
    input [17:0] input_399;
    input [17:0] input_400;
    input [17:0] input_401;
    input [17:0] input_402;
    input [17:0] input_403;
    input [17:0] input_404;
    input [17:0] input_405;
    input [17:0] input_406;
    input [17:0] input_407;
    input [17:0] input_408;
    input [17:0] input_409;
    input [17:0] input_410;
    input [17:0] input_411;
    input [17:0] input_412;
    input [17:0] input_413;
    input [17:0] input_414;
    input [17:0] input_415;
    input [17:0] input_416;
    input [17:0] input_417;
    input [17:0] input_418;
    input [17:0] input_419;
    input [17:0] input_420;
    input [17:0] input_421;
    input [17:0] input_422;
    input [17:0] input_423;
    input [17:0] input_424;
    input [17:0] input_425;
    input [17:0] input_426;
    input [17:0] input_427;
    input [17:0] input_428;
    input [17:0] input_429;
    input [17:0] input_430;
    input [17:0] input_431;
    input [17:0] input_432;
    input [17:0] input_433;
    input [17:0] input_434;
    input [17:0] input_435;
    input [17:0] input_436;
    input [17:0] input_437;
    input [17:0] input_438;
    input [17:0] input_439;
    input [17:0] input_440;
    input [17:0] input_441;
    input [17:0] input_442;
    input [17:0] input_443;
    input [17:0] input_444;
    input [17:0] input_445;
    input [17:0] input_446;
    input [17:0] input_447;
    input [17:0] input_448;
    input [17:0] input_449;
    input [17:0] input_450;
    input [17:0] input_451;
    input [17:0] input_452;
    input [17:0] input_453;
    input [17:0] input_454;
    input [17:0] input_455;
    input [17:0] input_456;
    input [17:0] input_457;
    input [17:0] input_458;
    input [17:0] input_459;
    input [17:0] input_460;
    input [17:0] input_461;
    input [17:0] input_462;
    input [17:0] input_463;
    input [17:0] input_464;
    input [17:0] input_465;
    input [17:0] input_466;
    input [17:0] input_467;
    input [17:0] input_468;
    input [17:0] input_469;
    input [17:0] input_470;
    input [17:0] input_471;
    input [17:0] input_472;
    input [17:0] input_473;
    input [17:0] input_474;
    input [17:0] input_475;
    input [17:0] input_476;
    input [17:0] input_477;
    input [17:0] input_478;
    input [17:0] input_479;
    input [17:0] input_480;
    input [17:0] input_481;
    input [17:0] input_482;
    input [17:0] input_483;
    input [17:0] input_484;
    input [17:0] input_485;
    input [17:0] input_486;
    input [17:0] input_487;
    input [17:0] input_488;
    input [17:0] input_489;
    input [17:0] input_490;
    input [17:0] input_491;
    input [17:0] input_492;
    input [17:0] input_493;
    input [17:0] input_494;
    input [17:0] input_495;
    input [17:0] input_496;
    input [17:0] input_497;
    input [17:0] input_498;
    input [17:0] input_499;
    input [17:0] input_500;
    input [17:0] input_501;
    input [17:0] input_502;
    input [17:0] input_503;
    input [17:0] input_504;
    input [17:0] input_505;
    input [17:0] input_506;
    input [17:0] input_507;
    input [17:0] input_508;
    input [17:0] input_509;
    input [17:0] input_510;
    input [17:0] input_511;
    input [17:0] input_512;
    input [17:0] input_513;
    input [17:0] input_514;
    input [17:0] input_515;
    input [17:0] input_516;
    input [17:0] input_517;
    input [17:0] input_518;
    input [17:0] input_519;
    input [17:0] input_520;
    input [17:0] input_521;
    input [17:0] input_522;
    input [17:0] input_523;
    input [17:0] input_524;
    input [17:0] input_525;
    input [17:0] input_526;
    input [17:0] input_527;
    input [17:0] input_528;
    input [17:0] input_529;
    input [17:0] input_530;
    input [17:0] input_531;
    input [17:0] input_532;
    input [17:0] input_533;
    input [17:0] input_534;
    input [17:0] input_535;
    input [17:0] input_536;
    input [17:0] input_537;
    input [17:0] input_538;
    input [17:0] input_539;
    input [17:0] input_540;
    input [17:0] input_541;
    input [17:0] input_542;
    input [17:0] input_543;
    input [17:0] input_544;
    input [17:0] input_545;
    input [17:0] input_546;
    input [17:0] input_547;
    input [17:0] input_548;
    input [17:0] input_549;
    input [17:0] input_550;
    input [17:0] input_551;
    input [17:0] input_552;
    input [17:0] input_553;
    input [17:0] input_554;
    input [17:0] input_555;
    input [17:0] input_556;
    input [17:0] input_557;
    input [17:0] input_558;
    input [17:0] input_559;
    input [17:0] input_560;
    input [17:0] input_561;
    input [17:0] input_562;
    input [17:0] input_563;
    input [17:0] input_564;
    input [17:0] input_565;
    input [17:0] input_566;
    input [17:0] input_567;
    input [17:0] input_568;
    input [17:0] input_569;
    input [17:0] input_570;
    input [17:0] input_571;
    input [17:0] input_572;
    input [17:0] input_573;
    input [17:0] input_574;
    input [17:0] input_575;
    input [17:0] input_576;
    input [17:0] input_577;
    input [17:0] input_578;
    input [17:0] input_579;
    input [17:0] input_580;
    input [17:0] input_581;
    input [17:0] input_582;
    input [17:0] input_583;
    input [17:0] input_584;
    input [17:0] input_585;
    input [17:0] input_586;
    input [17:0] input_587;
    input [17:0] input_588;
    input [17:0] input_589;
    input [17:0] input_590;
    input [17:0] input_591;
    input [17:0] input_592;
    input [17:0] input_593;
    input [17:0] input_594;
    input [17:0] input_595;
    input [17:0] input_596;
    input [17:0] input_597;
    input [17:0] input_598;
    input [17:0] input_599;
    input [17:0] input_600;
    input [17:0] input_601;
    input [17:0] input_602;
    input [17:0] input_603;
    input [17:0] input_604;
    input [17:0] input_605;
    input [17:0] input_606;
    input [17:0] input_607;
    input [17:0] input_608;
    input [17:0] input_609;
    input [17:0] input_610;
    input [17:0] input_611;
    input [17:0] input_612;
    input [17:0] input_613;
    input [17:0] input_614;
    input [17:0] input_615;
    input [17:0] input_616;
    input [17:0] input_617;
    input [17:0] input_618;
    input [17:0] input_619;
    input [17:0] input_620;
    input [17:0] input_621;
    input [17:0] input_622;
    input [17:0] input_623;
    input [17:0] input_624;
    input [17:0] input_625;
    input [17:0] input_626;
    input [17:0] input_627;
    input [17:0] input_628;
    input [17:0] input_629;
    input [17:0] input_630;
    input [17:0] input_631;
    input [17:0] input_632;
    input [17:0] input_633;
    input [17:0] input_634;
    input [17:0] input_635;
    input [17:0] input_636;
    input [17:0] input_637;
    input [17:0] input_638;
    input [17:0] input_639;
    input [17:0] input_640;
    input [17:0] input_641;
    input [17:0] input_642;
    input [17:0] input_643;
    input [17:0] input_644;
    input [17:0] input_645;
    input [17:0] input_646;
    input [17:0] input_647;
    input [17:0] input_648;
    input [17:0] input_649;
    input [17:0] input_650;
    input [17:0] input_651;
    input [17:0] input_652;
    input [17:0] input_653;
    input [17:0] input_654;
    input [17:0] input_655;
    input [17:0] input_656;
    input [17:0] input_657;
    input [17:0] input_658;
    input [17:0] input_659;
    input [17:0] input_660;
    input [17:0] input_661;
    input [17:0] input_662;
    input [17:0] input_663;
    input [17:0] input_664;
    input [17:0] input_665;
    input [17:0] input_666;
    input [17:0] input_667;
    input [17:0] input_668;
    input [17:0] input_669;
    input [17:0] input_670;
    input [17:0] input_671;
    input [17:0] input_672;
    input [17:0] input_673;
    input [17:0] input_674;
    input [17:0] input_675;
    input [17:0] input_676;
    input [17:0] input_677;
    input [17:0] input_678;
    input [17:0] input_679;
    input [17:0] input_680;
    input [17:0] input_681;
    input [17:0] input_682;
    input [17:0] input_683;
    input [17:0] input_684;
    input [17:0] input_685;
    input [17:0] input_686;
    input [17:0] input_687;
    input [17:0] input_688;
    input [17:0] input_689;
    input [17:0] input_690;
    input [17:0] input_691;
    input [17:0] input_692;
    input [17:0] input_693;
    input [17:0] input_694;
    input [17:0] input_695;
    input [17:0] input_696;
    input [17:0] input_697;
    input [17:0] input_698;
    input [17:0] input_699;
    input [17:0] input_700;
    input [17:0] input_701;
    input [17:0] input_702;
    input [17:0] input_703;
    input [17:0] input_704;
    input [17:0] input_705;
    input [17:0] input_706;
    input [17:0] input_707;
    input [17:0] input_708;
    input [17:0] input_709;
    input [17:0] input_710;
    input [17:0] input_711;
    input [17:0] input_712;
    input [17:0] input_713;
    input [17:0] input_714;
    input [17:0] input_715;
    input [17:0] input_716;
    input [17:0] input_717;
    input [17:0] input_718;
    input [17:0] input_719;
    input [17:0] input_720;
    input [17:0] input_721;
    input [17:0] input_722;
    input [17:0] input_723;
    input [17:0] input_724;
    input [17:0] input_725;
    input [17:0] input_726;
    input [17:0] input_727;
    input [17:0] input_728;
    input [17:0] input_729;
    input [17:0] input_730;
    input [17:0] input_731;
    input [17:0] input_732;
    input [17:0] input_733;
    input [17:0] input_734;
    input [17:0] input_735;
    input [17:0] input_736;
    input [17:0] input_737;
    input [17:0] input_738;
    input [17:0] input_739;
    input [17:0] input_740;
    input [17:0] input_741;
    input [17:0] input_742;
    input [17:0] input_743;
    input [17:0] input_744;
    input [17:0] input_745;
    input [17:0] input_746;
    input [17:0] input_747;
    input [17:0] input_748;
    input [17:0] input_749;
    input [17:0] input_750;
    input [17:0] input_751;
    input [17:0] input_752;
    input [17:0] input_753;
    input [17:0] input_754;
    input [17:0] input_755;
    input [17:0] input_756;
    input [17:0] input_757;
    input [17:0] input_758;
    input [17:0] input_759;
    input [17:0] input_760;
    input [17:0] input_761;
    input [17:0] input_762;
    input [17:0] input_763;
    input [17:0] input_764;
    input [17:0] input_765;
    input [17:0] input_766;
    input [17:0] input_767;
    input [17:0] input_768;
    input [17:0] input_769;
    input [17:0] input_770;
    input [17:0] input_771;
    input [17:0] input_772;
    input [17:0] input_773;
    input [17:0] input_774;
    input [17:0] input_775;
    input [17:0] input_776;
    input [17:0] input_777;
    input [17:0] input_778;
    input [17:0] input_779;
    input [17:0] input_780;
    input [17:0] input_781;
    input [17:0] input_782;
    input [17:0] input_783;
    input [9:0] sel;
    reg [17:0] result;
  begin
    case (sel)
      10'b0000000000 : begin
        result = input_0;
      end
      10'b0000000001 : begin
        result = input_1;
      end
      10'b0000000010 : begin
        result = input_2;
      end
      10'b0000000011 : begin
        result = input_3;
      end
      10'b0000000100 : begin
        result = input_4;
      end
      10'b0000000101 : begin
        result = input_5;
      end
      10'b0000000110 : begin
        result = input_6;
      end
      10'b0000000111 : begin
        result = input_7;
      end
      10'b0000001000 : begin
        result = input_8;
      end
      10'b0000001001 : begin
        result = input_9;
      end
      10'b0000001010 : begin
        result = input_10;
      end
      10'b0000001011 : begin
        result = input_11;
      end
      10'b0000001100 : begin
        result = input_12;
      end
      10'b0000001101 : begin
        result = input_13;
      end
      10'b0000001110 : begin
        result = input_14;
      end
      10'b0000001111 : begin
        result = input_15;
      end
      10'b0000010000 : begin
        result = input_16;
      end
      10'b0000010001 : begin
        result = input_17;
      end
      10'b0000010010 : begin
        result = input_18;
      end
      10'b0000010011 : begin
        result = input_19;
      end
      10'b0000010100 : begin
        result = input_20;
      end
      10'b0000010101 : begin
        result = input_21;
      end
      10'b0000010110 : begin
        result = input_22;
      end
      10'b0000010111 : begin
        result = input_23;
      end
      10'b0000011000 : begin
        result = input_24;
      end
      10'b0000011001 : begin
        result = input_25;
      end
      10'b0000011010 : begin
        result = input_26;
      end
      10'b0000011011 : begin
        result = input_27;
      end
      10'b0000011100 : begin
        result = input_28;
      end
      10'b0000011101 : begin
        result = input_29;
      end
      10'b0000011110 : begin
        result = input_30;
      end
      10'b0000011111 : begin
        result = input_31;
      end
      10'b0000100000 : begin
        result = input_32;
      end
      10'b0000100001 : begin
        result = input_33;
      end
      10'b0000100010 : begin
        result = input_34;
      end
      10'b0000100011 : begin
        result = input_35;
      end
      10'b0000100100 : begin
        result = input_36;
      end
      10'b0000100101 : begin
        result = input_37;
      end
      10'b0000100110 : begin
        result = input_38;
      end
      10'b0000100111 : begin
        result = input_39;
      end
      10'b0000101000 : begin
        result = input_40;
      end
      10'b0000101001 : begin
        result = input_41;
      end
      10'b0000101010 : begin
        result = input_42;
      end
      10'b0000101011 : begin
        result = input_43;
      end
      10'b0000101100 : begin
        result = input_44;
      end
      10'b0000101101 : begin
        result = input_45;
      end
      10'b0000101110 : begin
        result = input_46;
      end
      10'b0000101111 : begin
        result = input_47;
      end
      10'b0000110000 : begin
        result = input_48;
      end
      10'b0000110001 : begin
        result = input_49;
      end
      10'b0000110010 : begin
        result = input_50;
      end
      10'b0000110011 : begin
        result = input_51;
      end
      10'b0000110100 : begin
        result = input_52;
      end
      10'b0000110101 : begin
        result = input_53;
      end
      10'b0000110110 : begin
        result = input_54;
      end
      10'b0000110111 : begin
        result = input_55;
      end
      10'b0000111000 : begin
        result = input_56;
      end
      10'b0000111001 : begin
        result = input_57;
      end
      10'b0000111010 : begin
        result = input_58;
      end
      10'b0000111011 : begin
        result = input_59;
      end
      10'b0000111100 : begin
        result = input_60;
      end
      10'b0000111101 : begin
        result = input_61;
      end
      10'b0000111110 : begin
        result = input_62;
      end
      10'b0000111111 : begin
        result = input_63;
      end
      10'b0001000000 : begin
        result = input_64;
      end
      10'b0001000001 : begin
        result = input_65;
      end
      10'b0001000010 : begin
        result = input_66;
      end
      10'b0001000011 : begin
        result = input_67;
      end
      10'b0001000100 : begin
        result = input_68;
      end
      10'b0001000101 : begin
        result = input_69;
      end
      10'b0001000110 : begin
        result = input_70;
      end
      10'b0001000111 : begin
        result = input_71;
      end
      10'b0001001000 : begin
        result = input_72;
      end
      10'b0001001001 : begin
        result = input_73;
      end
      10'b0001001010 : begin
        result = input_74;
      end
      10'b0001001011 : begin
        result = input_75;
      end
      10'b0001001100 : begin
        result = input_76;
      end
      10'b0001001101 : begin
        result = input_77;
      end
      10'b0001001110 : begin
        result = input_78;
      end
      10'b0001001111 : begin
        result = input_79;
      end
      10'b0001010000 : begin
        result = input_80;
      end
      10'b0001010001 : begin
        result = input_81;
      end
      10'b0001010010 : begin
        result = input_82;
      end
      10'b0001010011 : begin
        result = input_83;
      end
      10'b0001010100 : begin
        result = input_84;
      end
      10'b0001010101 : begin
        result = input_85;
      end
      10'b0001010110 : begin
        result = input_86;
      end
      10'b0001010111 : begin
        result = input_87;
      end
      10'b0001011000 : begin
        result = input_88;
      end
      10'b0001011001 : begin
        result = input_89;
      end
      10'b0001011010 : begin
        result = input_90;
      end
      10'b0001011011 : begin
        result = input_91;
      end
      10'b0001011100 : begin
        result = input_92;
      end
      10'b0001011101 : begin
        result = input_93;
      end
      10'b0001011110 : begin
        result = input_94;
      end
      10'b0001011111 : begin
        result = input_95;
      end
      10'b0001100000 : begin
        result = input_96;
      end
      10'b0001100001 : begin
        result = input_97;
      end
      10'b0001100010 : begin
        result = input_98;
      end
      10'b0001100011 : begin
        result = input_99;
      end
      10'b0001100100 : begin
        result = input_100;
      end
      10'b0001100101 : begin
        result = input_101;
      end
      10'b0001100110 : begin
        result = input_102;
      end
      10'b0001100111 : begin
        result = input_103;
      end
      10'b0001101000 : begin
        result = input_104;
      end
      10'b0001101001 : begin
        result = input_105;
      end
      10'b0001101010 : begin
        result = input_106;
      end
      10'b0001101011 : begin
        result = input_107;
      end
      10'b0001101100 : begin
        result = input_108;
      end
      10'b0001101101 : begin
        result = input_109;
      end
      10'b0001101110 : begin
        result = input_110;
      end
      10'b0001101111 : begin
        result = input_111;
      end
      10'b0001110000 : begin
        result = input_112;
      end
      10'b0001110001 : begin
        result = input_113;
      end
      10'b0001110010 : begin
        result = input_114;
      end
      10'b0001110011 : begin
        result = input_115;
      end
      10'b0001110100 : begin
        result = input_116;
      end
      10'b0001110101 : begin
        result = input_117;
      end
      10'b0001110110 : begin
        result = input_118;
      end
      10'b0001110111 : begin
        result = input_119;
      end
      10'b0001111000 : begin
        result = input_120;
      end
      10'b0001111001 : begin
        result = input_121;
      end
      10'b0001111010 : begin
        result = input_122;
      end
      10'b0001111011 : begin
        result = input_123;
      end
      10'b0001111100 : begin
        result = input_124;
      end
      10'b0001111101 : begin
        result = input_125;
      end
      10'b0001111110 : begin
        result = input_126;
      end
      10'b0001111111 : begin
        result = input_127;
      end
      10'b0010000000 : begin
        result = input_128;
      end
      10'b0010000001 : begin
        result = input_129;
      end
      10'b0010000010 : begin
        result = input_130;
      end
      10'b0010000011 : begin
        result = input_131;
      end
      10'b0010000100 : begin
        result = input_132;
      end
      10'b0010000101 : begin
        result = input_133;
      end
      10'b0010000110 : begin
        result = input_134;
      end
      10'b0010000111 : begin
        result = input_135;
      end
      10'b0010001000 : begin
        result = input_136;
      end
      10'b0010001001 : begin
        result = input_137;
      end
      10'b0010001010 : begin
        result = input_138;
      end
      10'b0010001011 : begin
        result = input_139;
      end
      10'b0010001100 : begin
        result = input_140;
      end
      10'b0010001101 : begin
        result = input_141;
      end
      10'b0010001110 : begin
        result = input_142;
      end
      10'b0010001111 : begin
        result = input_143;
      end
      10'b0010010000 : begin
        result = input_144;
      end
      10'b0010010001 : begin
        result = input_145;
      end
      10'b0010010010 : begin
        result = input_146;
      end
      10'b0010010011 : begin
        result = input_147;
      end
      10'b0010010100 : begin
        result = input_148;
      end
      10'b0010010101 : begin
        result = input_149;
      end
      10'b0010010110 : begin
        result = input_150;
      end
      10'b0010010111 : begin
        result = input_151;
      end
      10'b0010011000 : begin
        result = input_152;
      end
      10'b0010011001 : begin
        result = input_153;
      end
      10'b0010011010 : begin
        result = input_154;
      end
      10'b0010011011 : begin
        result = input_155;
      end
      10'b0010011100 : begin
        result = input_156;
      end
      10'b0010011101 : begin
        result = input_157;
      end
      10'b0010011110 : begin
        result = input_158;
      end
      10'b0010011111 : begin
        result = input_159;
      end
      10'b0010100000 : begin
        result = input_160;
      end
      10'b0010100001 : begin
        result = input_161;
      end
      10'b0010100010 : begin
        result = input_162;
      end
      10'b0010100011 : begin
        result = input_163;
      end
      10'b0010100100 : begin
        result = input_164;
      end
      10'b0010100101 : begin
        result = input_165;
      end
      10'b0010100110 : begin
        result = input_166;
      end
      10'b0010100111 : begin
        result = input_167;
      end
      10'b0010101000 : begin
        result = input_168;
      end
      10'b0010101001 : begin
        result = input_169;
      end
      10'b0010101010 : begin
        result = input_170;
      end
      10'b0010101011 : begin
        result = input_171;
      end
      10'b0010101100 : begin
        result = input_172;
      end
      10'b0010101101 : begin
        result = input_173;
      end
      10'b0010101110 : begin
        result = input_174;
      end
      10'b0010101111 : begin
        result = input_175;
      end
      10'b0010110000 : begin
        result = input_176;
      end
      10'b0010110001 : begin
        result = input_177;
      end
      10'b0010110010 : begin
        result = input_178;
      end
      10'b0010110011 : begin
        result = input_179;
      end
      10'b0010110100 : begin
        result = input_180;
      end
      10'b0010110101 : begin
        result = input_181;
      end
      10'b0010110110 : begin
        result = input_182;
      end
      10'b0010110111 : begin
        result = input_183;
      end
      10'b0010111000 : begin
        result = input_184;
      end
      10'b0010111001 : begin
        result = input_185;
      end
      10'b0010111010 : begin
        result = input_186;
      end
      10'b0010111011 : begin
        result = input_187;
      end
      10'b0010111100 : begin
        result = input_188;
      end
      10'b0010111101 : begin
        result = input_189;
      end
      10'b0010111110 : begin
        result = input_190;
      end
      10'b0010111111 : begin
        result = input_191;
      end
      10'b0011000000 : begin
        result = input_192;
      end
      10'b0011000001 : begin
        result = input_193;
      end
      10'b0011000010 : begin
        result = input_194;
      end
      10'b0011000011 : begin
        result = input_195;
      end
      10'b0011000100 : begin
        result = input_196;
      end
      10'b0011000101 : begin
        result = input_197;
      end
      10'b0011000110 : begin
        result = input_198;
      end
      10'b0011000111 : begin
        result = input_199;
      end
      10'b0011001000 : begin
        result = input_200;
      end
      10'b0011001001 : begin
        result = input_201;
      end
      10'b0011001010 : begin
        result = input_202;
      end
      10'b0011001011 : begin
        result = input_203;
      end
      10'b0011001100 : begin
        result = input_204;
      end
      10'b0011001101 : begin
        result = input_205;
      end
      10'b0011001110 : begin
        result = input_206;
      end
      10'b0011001111 : begin
        result = input_207;
      end
      10'b0011010000 : begin
        result = input_208;
      end
      10'b0011010001 : begin
        result = input_209;
      end
      10'b0011010010 : begin
        result = input_210;
      end
      10'b0011010011 : begin
        result = input_211;
      end
      10'b0011010100 : begin
        result = input_212;
      end
      10'b0011010101 : begin
        result = input_213;
      end
      10'b0011010110 : begin
        result = input_214;
      end
      10'b0011010111 : begin
        result = input_215;
      end
      10'b0011011000 : begin
        result = input_216;
      end
      10'b0011011001 : begin
        result = input_217;
      end
      10'b0011011010 : begin
        result = input_218;
      end
      10'b0011011011 : begin
        result = input_219;
      end
      10'b0011011100 : begin
        result = input_220;
      end
      10'b0011011101 : begin
        result = input_221;
      end
      10'b0011011110 : begin
        result = input_222;
      end
      10'b0011011111 : begin
        result = input_223;
      end
      10'b0011100000 : begin
        result = input_224;
      end
      10'b0011100001 : begin
        result = input_225;
      end
      10'b0011100010 : begin
        result = input_226;
      end
      10'b0011100011 : begin
        result = input_227;
      end
      10'b0011100100 : begin
        result = input_228;
      end
      10'b0011100101 : begin
        result = input_229;
      end
      10'b0011100110 : begin
        result = input_230;
      end
      10'b0011100111 : begin
        result = input_231;
      end
      10'b0011101000 : begin
        result = input_232;
      end
      10'b0011101001 : begin
        result = input_233;
      end
      10'b0011101010 : begin
        result = input_234;
      end
      10'b0011101011 : begin
        result = input_235;
      end
      10'b0011101100 : begin
        result = input_236;
      end
      10'b0011101101 : begin
        result = input_237;
      end
      10'b0011101110 : begin
        result = input_238;
      end
      10'b0011101111 : begin
        result = input_239;
      end
      10'b0011110000 : begin
        result = input_240;
      end
      10'b0011110001 : begin
        result = input_241;
      end
      10'b0011110010 : begin
        result = input_242;
      end
      10'b0011110011 : begin
        result = input_243;
      end
      10'b0011110100 : begin
        result = input_244;
      end
      10'b0011110101 : begin
        result = input_245;
      end
      10'b0011110110 : begin
        result = input_246;
      end
      10'b0011110111 : begin
        result = input_247;
      end
      10'b0011111000 : begin
        result = input_248;
      end
      10'b0011111001 : begin
        result = input_249;
      end
      10'b0011111010 : begin
        result = input_250;
      end
      10'b0011111011 : begin
        result = input_251;
      end
      10'b0011111100 : begin
        result = input_252;
      end
      10'b0011111101 : begin
        result = input_253;
      end
      10'b0011111110 : begin
        result = input_254;
      end
      10'b0011111111 : begin
        result = input_255;
      end
      10'b0100000000 : begin
        result = input_256;
      end
      10'b0100000001 : begin
        result = input_257;
      end
      10'b0100000010 : begin
        result = input_258;
      end
      10'b0100000011 : begin
        result = input_259;
      end
      10'b0100000100 : begin
        result = input_260;
      end
      10'b0100000101 : begin
        result = input_261;
      end
      10'b0100000110 : begin
        result = input_262;
      end
      10'b0100000111 : begin
        result = input_263;
      end
      10'b0100001000 : begin
        result = input_264;
      end
      10'b0100001001 : begin
        result = input_265;
      end
      10'b0100001010 : begin
        result = input_266;
      end
      10'b0100001011 : begin
        result = input_267;
      end
      10'b0100001100 : begin
        result = input_268;
      end
      10'b0100001101 : begin
        result = input_269;
      end
      10'b0100001110 : begin
        result = input_270;
      end
      10'b0100001111 : begin
        result = input_271;
      end
      10'b0100010000 : begin
        result = input_272;
      end
      10'b0100010001 : begin
        result = input_273;
      end
      10'b0100010010 : begin
        result = input_274;
      end
      10'b0100010011 : begin
        result = input_275;
      end
      10'b0100010100 : begin
        result = input_276;
      end
      10'b0100010101 : begin
        result = input_277;
      end
      10'b0100010110 : begin
        result = input_278;
      end
      10'b0100010111 : begin
        result = input_279;
      end
      10'b0100011000 : begin
        result = input_280;
      end
      10'b0100011001 : begin
        result = input_281;
      end
      10'b0100011010 : begin
        result = input_282;
      end
      10'b0100011011 : begin
        result = input_283;
      end
      10'b0100011100 : begin
        result = input_284;
      end
      10'b0100011101 : begin
        result = input_285;
      end
      10'b0100011110 : begin
        result = input_286;
      end
      10'b0100011111 : begin
        result = input_287;
      end
      10'b0100100000 : begin
        result = input_288;
      end
      10'b0100100001 : begin
        result = input_289;
      end
      10'b0100100010 : begin
        result = input_290;
      end
      10'b0100100011 : begin
        result = input_291;
      end
      10'b0100100100 : begin
        result = input_292;
      end
      10'b0100100101 : begin
        result = input_293;
      end
      10'b0100100110 : begin
        result = input_294;
      end
      10'b0100100111 : begin
        result = input_295;
      end
      10'b0100101000 : begin
        result = input_296;
      end
      10'b0100101001 : begin
        result = input_297;
      end
      10'b0100101010 : begin
        result = input_298;
      end
      10'b0100101011 : begin
        result = input_299;
      end
      10'b0100101100 : begin
        result = input_300;
      end
      10'b0100101101 : begin
        result = input_301;
      end
      10'b0100101110 : begin
        result = input_302;
      end
      10'b0100101111 : begin
        result = input_303;
      end
      10'b0100110000 : begin
        result = input_304;
      end
      10'b0100110001 : begin
        result = input_305;
      end
      10'b0100110010 : begin
        result = input_306;
      end
      10'b0100110011 : begin
        result = input_307;
      end
      10'b0100110100 : begin
        result = input_308;
      end
      10'b0100110101 : begin
        result = input_309;
      end
      10'b0100110110 : begin
        result = input_310;
      end
      10'b0100110111 : begin
        result = input_311;
      end
      10'b0100111000 : begin
        result = input_312;
      end
      10'b0100111001 : begin
        result = input_313;
      end
      10'b0100111010 : begin
        result = input_314;
      end
      10'b0100111011 : begin
        result = input_315;
      end
      10'b0100111100 : begin
        result = input_316;
      end
      10'b0100111101 : begin
        result = input_317;
      end
      10'b0100111110 : begin
        result = input_318;
      end
      10'b0100111111 : begin
        result = input_319;
      end
      10'b0101000000 : begin
        result = input_320;
      end
      10'b0101000001 : begin
        result = input_321;
      end
      10'b0101000010 : begin
        result = input_322;
      end
      10'b0101000011 : begin
        result = input_323;
      end
      10'b0101000100 : begin
        result = input_324;
      end
      10'b0101000101 : begin
        result = input_325;
      end
      10'b0101000110 : begin
        result = input_326;
      end
      10'b0101000111 : begin
        result = input_327;
      end
      10'b0101001000 : begin
        result = input_328;
      end
      10'b0101001001 : begin
        result = input_329;
      end
      10'b0101001010 : begin
        result = input_330;
      end
      10'b0101001011 : begin
        result = input_331;
      end
      10'b0101001100 : begin
        result = input_332;
      end
      10'b0101001101 : begin
        result = input_333;
      end
      10'b0101001110 : begin
        result = input_334;
      end
      10'b0101001111 : begin
        result = input_335;
      end
      10'b0101010000 : begin
        result = input_336;
      end
      10'b0101010001 : begin
        result = input_337;
      end
      10'b0101010010 : begin
        result = input_338;
      end
      10'b0101010011 : begin
        result = input_339;
      end
      10'b0101010100 : begin
        result = input_340;
      end
      10'b0101010101 : begin
        result = input_341;
      end
      10'b0101010110 : begin
        result = input_342;
      end
      10'b0101010111 : begin
        result = input_343;
      end
      10'b0101011000 : begin
        result = input_344;
      end
      10'b0101011001 : begin
        result = input_345;
      end
      10'b0101011010 : begin
        result = input_346;
      end
      10'b0101011011 : begin
        result = input_347;
      end
      10'b0101011100 : begin
        result = input_348;
      end
      10'b0101011101 : begin
        result = input_349;
      end
      10'b0101011110 : begin
        result = input_350;
      end
      10'b0101011111 : begin
        result = input_351;
      end
      10'b0101100000 : begin
        result = input_352;
      end
      10'b0101100001 : begin
        result = input_353;
      end
      10'b0101100010 : begin
        result = input_354;
      end
      10'b0101100011 : begin
        result = input_355;
      end
      10'b0101100100 : begin
        result = input_356;
      end
      10'b0101100101 : begin
        result = input_357;
      end
      10'b0101100110 : begin
        result = input_358;
      end
      10'b0101100111 : begin
        result = input_359;
      end
      10'b0101101000 : begin
        result = input_360;
      end
      10'b0101101001 : begin
        result = input_361;
      end
      10'b0101101010 : begin
        result = input_362;
      end
      10'b0101101011 : begin
        result = input_363;
      end
      10'b0101101100 : begin
        result = input_364;
      end
      10'b0101101101 : begin
        result = input_365;
      end
      10'b0101101110 : begin
        result = input_366;
      end
      10'b0101101111 : begin
        result = input_367;
      end
      10'b0101110000 : begin
        result = input_368;
      end
      10'b0101110001 : begin
        result = input_369;
      end
      10'b0101110010 : begin
        result = input_370;
      end
      10'b0101110011 : begin
        result = input_371;
      end
      10'b0101110100 : begin
        result = input_372;
      end
      10'b0101110101 : begin
        result = input_373;
      end
      10'b0101110110 : begin
        result = input_374;
      end
      10'b0101110111 : begin
        result = input_375;
      end
      10'b0101111000 : begin
        result = input_376;
      end
      10'b0101111001 : begin
        result = input_377;
      end
      10'b0101111010 : begin
        result = input_378;
      end
      10'b0101111011 : begin
        result = input_379;
      end
      10'b0101111100 : begin
        result = input_380;
      end
      10'b0101111101 : begin
        result = input_381;
      end
      10'b0101111110 : begin
        result = input_382;
      end
      10'b0101111111 : begin
        result = input_383;
      end
      10'b0110000000 : begin
        result = input_384;
      end
      10'b0110000001 : begin
        result = input_385;
      end
      10'b0110000010 : begin
        result = input_386;
      end
      10'b0110000011 : begin
        result = input_387;
      end
      10'b0110000100 : begin
        result = input_388;
      end
      10'b0110000101 : begin
        result = input_389;
      end
      10'b0110000110 : begin
        result = input_390;
      end
      10'b0110000111 : begin
        result = input_391;
      end
      10'b0110001000 : begin
        result = input_392;
      end
      10'b0110001001 : begin
        result = input_393;
      end
      10'b0110001010 : begin
        result = input_394;
      end
      10'b0110001011 : begin
        result = input_395;
      end
      10'b0110001100 : begin
        result = input_396;
      end
      10'b0110001101 : begin
        result = input_397;
      end
      10'b0110001110 : begin
        result = input_398;
      end
      10'b0110001111 : begin
        result = input_399;
      end
      10'b0110010000 : begin
        result = input_400;
      end
      10'b0110010001 : begin
        result = input_401;
      end
      10'b0110010010 : begin
        result = input_402;
      end
      10'b0110010011 : begin
        result = input_403;
      end
      10'b0110010100 : begin
        result = input_404;
      end
      10'b0110010101 : begin
        result = input_405;
      end
      10'b0110010110 : begin
        result = input_406;
      end
      10'b0110010111 : begin
        result = input_407;
      end
      10'b0110011000 : begin
        result = input_408;
      end
      10'b0110011001 : begin
        result = input_409;
      end
      10'b0110011010 : begin
        result = input_410;
      end
      10'b0110011011 : begin
        result = input_411;
      end
      10'b0110011100 : begin
        result = input_412;
      end
      10'b0110011101 : begin
        result = input_413;
      end
      10'b0110011110 : begin
        result = input_414;
      end
      10'b0110011111 : begin
        result = input_415;
      end
      10'b0110100000 : begin
        result = input_416;
      end
      10'b0110100001 : begin
        result = input_417;
      end
      10'b0110100010 : begin
        result = input_418;
      end
      10'b0110100011 : begin
        result = input_419;
      end
      10'b0110100100 : begin
        result = input_420;
      end
      10'b0110100101 : begin
        result = input_421;
      end
      10'b0110100110 : begin
        result = input_422;
      end
      10'b0110100111 : begin
        result = input_423;
      end
      10'b0110101000 : begin
        result = input_424;
      end
      10'b0110101001 : begin
        result = input_425;
      end
      10'b0110101010 : begin
        result = input_426;
      end
      10'b0110101011 : begin
        result = input_427;
      end
      10'b0110101100 : begin
        result = input_428;
      end
      10'b0110101101 : begin
        result = input_429;
      end
      10'b0110101110 : begin
        result = input_430;
      end
      10'b0110101111 : begin
        result = input_431;
      end
      10'b0110110000 : begin
        result = input_432;
      end
      10'b0110110001 : begin
        result = input_433;
      end
      10'b0110110010 : begin
        result = input_434;
      end
      10'b0110110011 : begin
        result = input_435;
      end
      10'b0110110100 : begin
        result = input_436;
      end
      10'b0110110101 : begin
        result = input_437;
      end
      10'b0110110110 : begin
        result = input_438;
      end
      10'b0110110111 : begin
        result = input_439;
      end
      10'b0110111000 : begin
        result = input_440;
      end
      10'b0110111001 : begin
        result = input_441;
      end
      10'b0110111010 : begin
        result = input_442;
      end
      10'b0110111011 : begin
        result = input_443;
      end
      10'b0110111100 : begin
        result = input_444;
      end
      10'b0110111101 : begin
        result = input_445;
      end
      10'b0110111110 : begin
        result = input_446;
      end
      10'b0110111111 : begin
        result = input_447;
      end
      10'b0111000000 : begin
        result = input_448;
      end
      10'b0111000001 : begin
        result = input_449;
      end
      10'b0111000010 : begin
        result = input_450;
      end
      10'b0111000011 : begin
        result = input_451;
      end
      10'b0111000100 : begin
        result = input_452;
      end
      10'b0111000101 : begin
        result = input_453;
      end
      10'b0111000110 : begin
        result = input_454;
      end
      10'b0111000111 : begin
        result = input_455;
      end
      10'b0111001000 : begin
        result = input_456;
      end
      10'b0111001001 : begin
        result = input_457;
      end
      10'b0111001010 : begin
        result = input_458;
      end
      10'b0111001011 : begin
        result = input_459;
      end
      10'b0111001100 : begin
        result = input_460;
      end
      10'b0111001101 : begin
        result = input_461;
      end
      10'b0111001110 : begin
        result = input_462;
      end
      10'b0111001111 : begin
        result = input_463;
      end
      10'b0111010000 : begin
        result = input_464;
      end
      10'b0111010001 : begin
        result = input_465;
      end
      10'b0111010010 : begin
        result = input_466;
      end
      10'b0111010011 : begin
        result = input_467;
      end
      10'b0111010100 : begin
        result = input_468;
      end
      10'b0111010101 : begin
        result = input_469;
      end
      10'b0111010110 : begin
        result = input_470;
      end
      10'b0111010111 : begin
        result = input_471;
      end
      10'b0111011000 : begin
        result = input_472;
      end
      10'b0111011001 : begin
        result = input_473;
      end
      10'b0111011010 : begin
        result = input_474;
      end
      10'b0111011011 : begin
        result = input_475;
      end
      10'b0111011100 : begin
        result = input_476;
      end
      10'b0111011101 : begin
        result = input_477;
      end
      10'b0111011110 : begin
        result = input_478;
      end
      10'b0111011111 : begin
        result = input_479;
      end
      10'b0111100000 : begin
        result = input_480;
      end
      10'b0111100001 : begin
        result = input_481;
      end
      10'b0111100010 : begin
        result = input_482;
      end
      10'b0111100011 : begin
        result = input_483;
      end
      10'b0111100100 : begin
        result = input_484;
      end
      10'b0111100101 : begin
        result = input_485;
      end
      10'b0111100110 : begin
        result = input_486;
      end
      10'b0111100111 : begin
        result = input_487;
      end
      10'b0111101000 : begin
        result = input_488;
      end
      10'b0111101001 : begin
        result = input_489;
      end
      10'b0111101010 : begin
        result = input_490;
      end
      10'b0111101011 : begin
        result = input_491;
      end
      10'b0111101100 : begin
        result = input_492;
      end
      10'b0111101101 : begin
        result = input_493;
      end
      10'b0111101110 : begin
        result = input_494;
      end
      10'b0111101111 : begin
        result = input_495;
      end
      10'b0111110000 : begin
        result = input_496;
      end
      10'b0111110001 : begin
        result = input_497;
      end
      10'b0111110010 : begin
        result = input_498;
      end
      10'b0111110011 : begin
        result = input_499;
      end
      10'b0111110100 : begin
        result = input_500;
      end
      10'b0111110101 : begin
        result = input_501;
      end
      10'b0111110110 : begin
        result = input_502;
      end
      10'b0111110111 : begin
        result = input_503;
      end
      10'b0111111000 : begin
        result = input_504;
      end
      10'b0111111001 : begin
        result = input_505;
      end
      10'b0111111010 : begin
        result = input_506;
      end
      10'b0111111011 : begin
        result = input_507;
      end
      10'b0111111100 : begin
        result = input_508;
      end
      10'b0111111101 : begin
        result = input_509;
      end
      10'b0111111110 : begin
        result = input_510;
      end
      10'b0111111111 : begin
        result = input_511;
      end
      10'b1000000000 : begin
        result = input_512;
      end
      10'b1000000001 : begin
        result = input_513;
      end
      10'b1000000010 : begin
        result = input_514;
      end
      10'b1000000011 : begin
        result = input_515;
      end
      10'b1000000100 : begin
        result = input_516;
      end
      10'b1000000101 : begin
        result = input_517;
      end
      10'b1000000110 : begin
        result = input_518;
      end
      10'b1000000111 : begin
        result = input_519;
      end
      10'b1000001000 : begin
        result = input_520;
      end
      10'b1000001001 : begin
        result = input_521;
      end
      10'b1000001010 : begin
        result = input_522;
      end
      10'b1000001011 : begin
        result = input_523;
      end
      10'b1000001100 : begin
        result = input_524;
      end
      10'b1000001101 : begin
        result = input_525;
      end
      10'b1000001110 : begin
        result = input_526;
      end
      10'b1000001111 : begin
        result = input_527;
      end
      10'b1000010000 : begin
        result = input_528;
      end
      10'b1000010001 : begin
        result = input_529;
      end
      10'b1000010010 : begin
        result = input_530;
      end
      10'b1000010011 : begin
        result = input_531;
      end
      10'b1000010100 : begin
        result = input_532;
      end
      10'b1000010101 : begin
        result = input_533;
      end
      10'b1000010110 : begin
        result = input_534;
      end
      10'b1000010111 : begin
        result = input_535;
      end
      10'b1000011000 : begin
        result = input_536;
      end
      10'b1000011001 : begin
        result = input_537;
      end
      10'b1000011010 : begin
        result = input_538;
      end
      10'b1000011011 : begin
        result = input_539;
      end
      10'b1000011100 : begin
        result = input_540;
      end
      10'b1000011101 : begin
        result = input_541;
      end
      10'b1000011110 : begin
        result = input_542;
      end
      10'b1000011111 : begin
        result = input_543;
      end
      10'b1000100000 : begin
        result = input_544;
      end
      10'b1000100001 : begin
        result = input_545;
      end
      10'b1000100010 : begin
        result = input_546;
      end
      10'b1000100011 : begin
        result = input_547;
      end
      10'b1000100100 : begin
        result = input_548;
      end
      10'b1000100101 : begin
        result = input_549;
      end
      10'b1000100110 : begin
        result = input_550;
      end
      10'b1000100111 : begin
        result = input_551;
      end
      10'b1000101000 : begin
        result = input_552;
      end
      10'b1000101001 : begin
        result = input_553;
      end
      10'b1000101010 : begin
        result = input_554;
      end
      10'b1000101011 : begin
        result = input_555;
      end
      10'b1000101100 : begin
        result = input_556;
      end
      10'b1000101101 : begin
        result = input_557;
      end
      10'b1000101110 : begin
        result = input_558;
      end
      10'b1000101111 : begin
        result = input_559;
      end
      10'b1000110000 : begin
        result = input_560;
      end
      10'b1000110001 : begin
        result = input_561;
      end
      10'b1000110010 : begin
        result = input_562;
      end
      10'b1000110011 : begin
        result = input_563;
      end
      10'b1000110100 : begin
        result = input_564;
      end
      10'b1000110101 : begin
        result = input_565;
      end
      10'b1000110110 : begin
        result = input_566;
      end
      10'b1000110111 : begin
        result = input_567;
      end
      10'b1000111000 : begin
        result = input_568;
      end
      10'b1000111001 : begin
        result = input_569;
      end
      10'b1000111010 : begin
        result = input_570;
      end
      10'b1000111011 : begin
        result = input_571;
      end
      10'b1000111100 : begin
        result = input_572;
      end
      10'b1000111101 : begin
        result = input_573;
      end
      10'b1000111110 : begin
        result = input_574;
      end
      10'b1000111111 : begin
        result = input_575;
      end
      10'b1001000000 : begin
        result = input_576;
      end
      10'b1001000001 : begin
        result = input_577;
      end
      10'b1001000010 : begin
        result = input_578;
      end
      10'b1001000011 : begin
        result = input_579;
      end
      10'b1001000100 : begin
        result = input_580;
      end
      10'b1001000101 : begin
        result = input_581;
      end
      10'b1001000110 : begin
        result = input_582;
      end
      10'b1001000111 : begin
        result = input_583;
      end
      10'b1001001000 : begin
        result = input_584;
      end
      10'b1001001001 : begin
        result = input_585;
      end
      10'b1001001010 : begin
        result = input_586;
      end
      10'b1001001011 : begin
        result = input_587;
      end
      10'b1001001100 : begin
        result = input_588;
      end
      10'b1001001101 : begin
        result = input_589;
      end
      10'b1001001110 : begin
        result = input_590;
      end
      10'b1001001111 : begin
        result = input_591;
      end
      10'b1001010000 : begin
        result = input_592;
      end
      10'b1001010001 : begin
        result = input_593;
      end
      10'b1001010010 : begin
        result = input_594;
      end
      10'b1001010011 : begin
        result = input_595;
      end
      10'b1001010100 : begin
        result = input_596;
      end
      10'b1001010101 : begin
        result = input_597;
      end
      10'b1001010110 : begin
        result = input_598;
      end
      10'b1001010111 : begin
        result = input_599;
      end
      10'b1001011000 : begin
        result = input_600;
      end
      10'b1001011001 : begin
        result = input_601;
      end
      10'b1001011010 : begin
        result = input_602;
      end
      10'b1001011011 : begin
        result = input_603;
      end
      10'b1001011100 : begin
        result = input_604;
      end
      10'b1001011101 : begin
        result = input_605;
      end
      10'b1001011110 : begin
        result = input_606;
      end
      10'b1001011111 : begin
        result = input_607;
      end
      10'b1001100000 : begin
        result = input_608;
      end
      10'b1001100001 : begin
        result = input_609;
      end
      10'b1001100010 : begin
        result = input_610;
      end
      10'b1001100011 : begin
        result = input_611;
      end
      10'b1001100100 : begin
        result = input_612;
      end
      10'b1001100101 : begin
        result = input_613;
      end
      10'b1001100110 : begin
        result = input_614;
      end
      10'b1001100111 : begin
        result = input_615;
      end
      10'b1001101000 : begin
        result = input_616;
      end
      10'b1001101001 : begin
        result = input_617;
      end
      10'b1001101010 : begin
        result = input_618;
      end
      10'b1001101011 : begin
        result = input_619;
      end
      10'b1001101100 : begin
        result = input_620;
      end
      10'b1001101101 : begin
        result = input_621;
      end
      10'b1001101110 : begin
        result = input_622;
      end
      10'b1001101111 : begin
        result = input_623;
      end
      10'b1001110000 : begin
        result = input_624;
      end
      10'b1001110001 : begin
        result = input_625;
      end
      10'b1001110010 : begin
        result = input_626;
      end
      10'b1001110011 : begin
        result = input_627;
      end
      10'b1001110100 : begin
        result = input_628;
      end
      10'b1001110101 : begin
        result = input_629;
      end
      10'b1001110110 : begin
        result = input_630;
      end
      10'b1001110111 : begin
        result = input_631;
      end
      10'b1001111000 : begin
        result = input_632;
      end
      10'b1001111001 : begin
        result = input_633;
      end
      10'b1001111010 : begin
        result = input_634;
      end
      10'b1001111011 : begin
        result = input_635;
      end
      10'b1001111100 : begin
        result = input_636;
      end
      10'b1001111101 : begin
        result = input_637;
      end
      10'b1001111110 : begin
        result = input_638;
      end
      10'b1001111111 : begin
        result = input_639;
      end
      10'b1010000000 : begin
        result = input_640;
      end
      10'b1010000001 : begin
        result = input_641;
      end
      10'b1010000010 : begin
        result = input_642;
      end
      10'b1010000011 : begin
        result = input_643;
      end
      10'b1010000100 : begin
        result = input_644;
      end
      10'b1010000101 : begin
        result = input_645;
      end
      10'b1010000110 : begin
        result = input_646;
      end
      10'b1010000111 : begin
        result = input_647;
      end
      10'b1010001000 : begin
        result = input_648;
      end
      10'b1010001001 : begin
        result = input_649;
      end
      10'b1010001010 : begin
        result = input_650;
      end
      10'b1010001011 : begin
        result = input_651;
      end
      10'b1010001100 : begin
        result = input_652;
      end
      10'b1010001101 : begin
        result = input_653;
      end
      10'b1010001110 : begin
        result = input_654;
      end
      10'b1010001111 : begin
        result = input_655;
      end
      10'b1010010000 : begin
        result = input_656;
      end
      10'b1010010001 : begin
        result = input_657;
      end
      10'b1010010010 : begin
        result = input_658;
      end
      10'b1010010011 : begin
        result = input_659;
      end
      10'b1010010100 : begin
        result = input_660;
      end
      10'b1010010101 : begin
        result = input_661;
      end
      10'b1010010110 : begin
        result = input_662;
      end
      10'b1010010111 : begin
        result = input_663;
      end
      10'b1010011000 : begin
        result = input_664;
      end
      10'b1010011001 : begin
        result = input_665;
      end
      10'b1010011010 : begin
        result = input_666;
      end
      10'b1010011011 : begin
        result = input_667;
      end
      10'b1010011100 : begin
        result = input_668;
      end
      10'b1010011101 : begin
        result = input_669;
      end
      10'b1010011110 : begin
        result = input_670;
      end
      10'b1010011111 : begin
        result = input_671;
      end
      10'b1010100000 : begin
        result = input_672;
      end
      10'b1010100001 : begin
        result = input_673;
      end
      10'b1010100010 : begin
        result = input_674;
      end
      10'b1010100011 : begin
        result = input_675;
      end
      10'b1010100100 : begin
        result = input_676;
      end
      10'b1010100101 : begin
        result = input_677;
      end
      10'b1010100110 : begin
        result = input_678;
      end
      10'b1010100111 : begin
        result = input_679;
      end
      10'b1010101000 : begin
        result = input_680;
      end
      10'b1010101001 : begin
        result = input_681;
      end
      10'b1010101010 : begin
        result = input_682;
      end
      10'b1010101011 : begin
        result = input_683;
      end
      10'b1010101100 : begin
        result = input_684;
      end
      10'b1010101101 : begin
        result = input_685;
      end
      10'b1010101110 : begin
        result = input_686;
      end
      10'b1010101111 : begin
        result = input_687;
      end
      10'b1010110000 : begin
        result = input_688;
      end
      10'b1010110001 : begin
        result = input_689;
      end
      10'b1010110010 : begin
        result = input_690;
      end
      10'b1010110011 : begin
        result = input_691;
      end
      10'b1010110100 : begin
        result = input_692;
      end
      10'b1010110101 : begin
        result = input_693;
      end
      10'b1010110110 : begin
        result = input_694;
      end
      10'b1010110111 : begin
        result = input_695;
      end
      10'b1010111000 : begin
        result = input_696;
      end
      10'b1010111001 : begin
        result = input_697;
      end
      10'b1010111010 : begin
        result = input_698;
      end
      10'b1010111011 : begin
        result = input_699;
      end
      10'b1010111100 : begin
        result = input_700;
      end
      10'b1010111101 : begin
        result = input_701;
      end
      10'b1010111110 : begin
        result = input_702;
      end
      10'b1010111111 : begin
        result = input_703;
      end
      10'b1011000000 : begin
        result = input_704;
      end
      10'b1011000001 : begin
        result = input_705;
      end
      10'b1011000010 : begin
        result = input_706;
      end
      10'b1011000011 : begin
        result = input_707;
      end
      10'b1011000100 : begin
        result = input_708;
      end
      10'b1011000101 : begin
        result = input_709;
      end
      10'b1011000110 : begin
        result = input_710;
      end
      10'b1011000111 : begin
        result = input_711;
      end
      10'b1011001000 : begin
        result = input_712;
      end
      10'b1011001001 : begin
        result = input_713;
      end
      10'b1011001010 : begin
        result = input_714;
      end
      10'b1011001011 : begin
        result = input_715;
      end
      10'b1011001100 : begin
        result = input_716;
      end
      10'b1011001101 : begin
        result = input_717;
      end
      10'b1011001110 : begin
        result = input_718;
      end
      10'b1011001111 : begin
        result = input_719;
      end
      10'b1011010000 : begin
        result = input_720;
      end
      10'b1011010001 : begin
        result = input_721;
      end
      10'b1011010010 : begin
        result = input_722;
      end
      10'b1011010011 : begin
        result = input_723;
      end
      10'b1011010100 : begin
        result = input_724;
      end
      10'b1011010101 : begin
        result = input_725;
      end
      10'b1011010110 : begin
        result = input_726;
      end
      10'b1011010111 : begin
        result = input_727;
      end
      10'b1011011000 : begin
        result = input_728;
      end
      10'b1011011001 : begin
        result = input_729;
      end
      10'b1011011010 : begin
        result = input_730;
      end
      10'b1011011011 : begin
        result = input_731;
      end
      10'b1011011100 : begin
        result = input_732;
      end
      10'b1011011101 : begin
        result = input_733;
      end
      10'b1011011110 : begin
        result = input_734;
      end
      10'b1011011111 : begin
        result = input_735;
      end
      10'b1011100000 : begin
        result = input_736;
      end
      10'b1011100001 : begin
        result = input_737;
      end
      10'b1011100010 : begin
        result = input_738;
      end
      10'b1011100011 : begin
        result = input_739;
      end
      10'b1011100100 : begin
        result = input_740;
      end
      10'b1011100101 : begin
        result = input_741;
      end
      10'b1011100110 : begin
        result = input_742;
      end
      10'b1011100111 : begin
        result = input_743;
      end
      10'b1011101000 : begin
        result = input_744;
      end
      10'b1011101001 : begin
        result = input_745;
      end
      10'b1011101010 : begin
        result = input_746;
      end
      10'b1011101011 : begin
        result = input_747;
      end
      10'b1011101100 : begin
        result = input_748;
      end
      10'b1011101101 : begin
        result = input_749;
      end
      10'b1011101110 : begin
        result = input_750;
      end
      10'b1011101111 : begin
        result = input_751;
      end
      10'b1011110000 : begin
        result = input_752;
      end
      10'b1011110001 : begin
        result = input_753;
      end
      10'b1011110010 : begin
        result = input_754;
      end
      10'b1011110011 : begin
        result = input_755;
      end
      10'b1011110100 : begin
        result = input_756;
      end
      10'b1011110101 : begin
        result = input_757;
      end
      10'b1011110110 : begin
        result = input_758;
      end
      10'b1011110111 : begin
        result = input_759;
      end
      10'b1011111000 : begin
        result = input_760;
      end
      10'b1011111001 : begin
        result = input_761;
      end
      10'b1011111010 : begin
        result = input_762;
      end
      10'b1011111011 : begin
        result = input_763;
      end
      10'b1011111100 : begin
        result = input_764;
      end
      10'b1011111101 : begin
        result = input_765;
      end
      10'b1011111110 : begin
        result = input_766;
      end
      10'b1011111111 : begin
        result = input_767;
      end
      10'b1100000000 : begin
        result = input_768;
      end
      10'b1100000001 : begin
        result = input_769;
      end
      10'b1100000010 : begin
        result = input_770;
      end
      10'b1100000011 : begin
        result = input_771;
      end
      10'b1100000100 : begin
        result = input_772;
      end
      10'b1100000101 : begin
        result = input_773;
      end
      10'b1100000110 : begin
        result = input_774;
      end
      10'b1100000111 : begin
        result = input_775;
      end
      10'b1100001000 : begin
        result = input_776;
      end
      10'b1100001001 : begin
        result = input_777;
      end
      10'b1100001010 : begin
        result = input_778;
      end
      10'b1100001011 : begin
        result = input_779;
      end
      10'b1100001100 : begin
        result = input_780;
      end
      10'b1100001101 : begin
        result = input_781;
      end
      10'b1100001110 : begin
        result = input_782;
      end
      default : begin
        result = input_783;
      end
    endcase
    MUX_v_18_784_2 = result;
  end
  endfunction


  function automatic [1:0] MUX_v_2_2_2;
    input [1:0] input_0;
    input [1:0] input_1;
    input [0:0] sel;
    reg [1:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_2_2_2 = result;
  end
  endfunction


  function automatic [2:0] MUX_v_3_2_2;
    input [2:0] input_0;
    input [2:0] input_1;
    input [0:0] sel;
    reg [2:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_3_2_2 = result;
  end
  endfunction


  function automatic [2:0] MUX_v_3_4_2;
    input [2:0] input_0;
    input [2:0] input_1;
    input [2:0] input_2;
    input [2:0] input_3;
    input [1:0] sel;
    reg [2:0] result;
  begin
    case (sel)
      2'b00 : begin
        result = input_0;
      end
      2'b01 : begin
        result = input_1;
      end
      2'b10 : begin
        result = input_2;
      end
      default : begin
        result = input_3;
      end
    endcase
    MUX_v_3_4_2 = result;
  end
  endfunction


  function automatic [3:0] MUX_v_4_2_2;
    input [3:0] input_0;
    input [3:0] input_1;
    input [0:0] sel;
    reg [3:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_4_2_2 = result;
  end
  endfunction


  function automatic [49:0] MUX_v_50_10_2;
    input [49:0] input_0;
    input [49:0] input_1;
    input [49:0] input_2;
    input [49:0] input_3;
    input [49:0] input_4;
    input [49:0] input_5;
    input [49:0] input_6;
    input [49:0] input_7;
    input [49:0] input_8;
    input [49:0] input_9;
    input [3:0] sel;
    reg [49:0] result;
  begin
    case (sel)
      4'b0000 : begin
        result = input_0;
      end
      4'b0001 : begin
        result = input_1;
      end
      4'b0010 : begin
        result = input_2;
      end
      4'b0011 : begin
        result = input_3;
      end
      4'b0100 : begin
        result = input_4;
      end
      4'b0101 : begin
        result = input_5;
      end
      4'b0110 : begin
        result = input_6;
      end
      4'b0111 : begin
        result = input_7;
      end
      4'b1000 : begin
        result = input_8;
      end
      default : begin
        result = input_9;
      end
    endcase
    MUX_v_50_10_2 = result;
  end
  endfunction


  function automatic [49:0] MUX_v_50_2_2;
    input [49:0] input_0;
    input [49:0] input_1;
    input [0:0] sel;
    reg [49:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_50_2_2 = result;
  end
  endfunction


  function automatic [4:0] MUX_v_5_4_2;
    input [4:0] input_0;
    input [4:0] input_1;
    input [4:0] input_2;
    input [4:0] input_3;
    input [1:0] sel;
    reg [4:0] result;
  begin
    case (sel)
      2'b00 : begin
        result = input_0;
      end
      2'b01 : begin
        result = input_1;
      end
      2'b10 : begin
        result = input_2;
      end
      default : begin
        result = input_3;
      end
    endcase
    MUX_v_5_4_2 = result;
  end
  endfunction


  function automatic [5:0] MUX_v_6_2_2;
    input [5:0] input_0;
    input [5:0] input_1;
    input [0:0] sel;
    reg [5:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_6_2_2 = result;
  end
  endfunction


  function automatic [70:0] MUX_v_71_2_2;
    input [70:0] input_0;
    input [70:0] input_1;
    input [0:0] sel;
    reg [70:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_71_2_2 = result;
  end
  endfunction


  function automatic [6:0] MUX_v_7_10_2;
    input [6:0] input_0;
    input [6:0] input_1;
    input [6:0] input_2;
    input [6:0] input_3;
    input [6:0] input_4;
    input [6:0] input_5;
    input [6:0] input_6;
    input [6:0] input_7;
    input [6:0] input_8;
    input [6:0] input_9;
    input [3:0] sel;
    reg [6:0] result;
  begin
    case (sel)
      4'b0000 : begin
        result = input_0;
      end
      4'b0001 : begin
        result = input_1;
      end
      4'b0010 : begin
        result = input_2;
      end
      4'b0011 : begin
        result = input_3;
      end
      4'b0100 : begin
        result = input_4;
      end
      4'b0101 : begin
        result = input_5;
      end
      4'b0110 : begin
        result = input_6;
      end
      4'b0111 : begin
        result = input_7;
      end
      4'b1000 : begin
        result = input_8;
      end
      default : begin
        result = input_9;
      end
    endcase
    MUX_v_7_10_2 = result;
  end
  endfunction


  function automatic [6:0] MUX_v_7_2_2;
    input [6:0] input_0;
    input [6:0] input_1;
    input [0:0] sel;
    reg [6:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_7_2_2 = result;
  end
  endfunction


  function automatic [6:0] MUX_v_7_64_2;
    input [6:0] input_0;
    input [6:0] input_1;
    input [6:0] input_2;
    input [6:0] input_3;
    input [6:0] input_4;
    input [6:0] input_5;
    input [6:0] input_6;
    input [6:0] input_7;
    input [6:0] input_8;
    input [6:0] input_9;
    input [6:0] input_10;
    input [6:0] input_11;
    input [6:0] input_12;
    input [6:0] input_13;
    input [6:0] input_14;
    input [6:0] input_15;
    input [6:0] input_16;
    input [6:0] input_17;
    input [6:0] input_18;
    input [6:0] input_19;
    input [6:0] input_20;
    input [6:0] input_21;
    input [6:0] input_22;
    input [6:0] input_23;
    input [6:0] input_24;
    input [6:0] input_25;
    input [6:0] input_26;
    input [6:0] input_27;
    input [6:0] input_28;
    input [6:0] input_29;
    input [6:0] input_30;
    input [6:0] input_31;
    input [6:0] input_32;
    input [6:0] input_33;
    input [6:0] input_34;
    input [6:0] input_35;
    input [6:0] input_36;
    input [6:0] input_37;
    input [6:0] input_38;
    input [6:0] input_39;
    input [6:0] input_40;
    input [6:0] input_41;
    input [6:0] input_42;
    input [6:0] input_43;
    input [6:0] input_44;
    input [6:0] input_45;
    input [6:0] input_46;
    input [6:0] input_47;
    input [6:0] input_48;
    input [6:0] input_49;
    input [6:0] input_50;
    input [6:0] input_51;
    input [6:0] input_52;
    input [6:0] input_53;
    input [6:0] input_54;
    input [6:0] input_55;
    input [6:0] input_56;
    input [6:0] input_57;
    input [6:0] input_58;
    input [6:0] input_59;
    input [6:0] input_60;
    input [6:0] input_61;
    input [6:0] input_62;
    input [6:0] input_63;
    input [5:0] sel;
    reg [6:0] result;
  begin
    case (sel)
      6'b000000 : begin
        result = input_0;
      end
      6'b000001 : begin
        result = input_1;
      end
      6'b000010 : begin
        result = input_2;
      end
      6'b000011 : begin
        result = input_3;
      end
      6'b000100 : begin
        result = input_4;
      end
      6'b000101 : begin
        result = input_5;
      end
      6'b000110 : begin
        result = input_6;
      end
      6'b000111 : begin
        result = input_7;
      end
      6'b001000 : begin
        result = input_8;
      end
      6'b001001 : begin
        result = input_9;
      end
      6'b001010 : begin
        result = input_10;
      end
      6'b001011 : begin
        result = input_11;
      end
      6'b001100 : begin
        result = input_12;
      end
      6'b001101 : begin
        result = input_13;
      end
      6'b001110 : begin
        result = input_14;
      end
      6'b001111 : begin
        result = input_15;
      end
      6'b010000 : begin
        result = input_16;
      end
      6'b010001 : begin
        result = input_17;
      end
      6'b010010 : begin
        result = input_18;
      end
      6'b010011 : begin
        result = input_19;
      end
      6'b010100 : begin
        result = input_20;
      end
      6'b010101 : begin
        result = input_21;
      end
      6'b010110 : begin
        result = input_22;
      end
      6'b010111 : begin
        result = input_23;
      end
      6'b011000 : begin
        result = input_24;
      end
      6'b011001 : begin
        result = input_25;
      end
      6'b011010 : begin
        result = input_26;
      end
      6'b011011 : begin
        result = input_27;
      end
      6'b011100 : begin
        result = input_28;
      end
      6'b011101 : begin
        result = input_29;
      end
      6'b011110 : begin
        result = input_30;
      end
      6'b011111 : begin
        result = input_31;
      end
      6'b100000 : begin
        result = input_32;
      end
      6'b100001 : begin
        result = input_33;
      end
      6'b100010 : begin
        result = input_34;
      end
      6'b100011 : begin
        result = input_35;
      end
      6'b100100 : begin
        result = input_36;
      end
      6'b100101 : begin
        result = input_37;
      end
      6'b100110 : begin
        result = input_38;
      end
      6'b100111 : begin
        result = input_39;
      end
      6'b101000 : begin
        result = input_40;
      end
      6'b101001 : begin
        result = input_41;
      end
      6'b101010 : begin
        result = input_42;
      end
      6'b101011 : begin
        result = input_43;
      end
      6'b101100 : begin
        result = input_44;
      end
      6'b101101 : begin
        result = input_45;
      end
      6'b101110 : begin
        result = input_46;
      end
      6'b101111 : begin
        result = input_47;
      end
      6'b110000 : begin
        result = input_48;
      end
      6'b110001 : begin
        result = input_49;
      end
      6'b110010 : begin
        result = input_50;
      end
      6'b110011 : begin
        result = input_51;
      end
      6'b110100 : begin
        result = input_52;
      end
      6'b110101 : begin
        result = input_53;
      end
      6'b110110 : begin
        result = input_54;
      end
      6'b110111 : begin
        result = input_55;
      end
      6'b111000 : begin
        result = input_56;
      end
      6'b111001 : begin
        result = input_57;
      end
      6'b111010 : begin
        result = input_58;
      end
      6'b111011 : begin
        result = input_59;
      end
      6'b111100 : begin
        result = input_60;
      end
      6'b111101 : begin
        result = input_61;
      end
      6'b111110 : begin
        result = input_62;
      end
      default : begin
        result = input_63;
      end
    endcase
    MUX_v_7_64_2 = result;
  end
  endfunction


  function automatic [90:0] MUX_v_91_2_2;
    input [90:0] input_0;
    input [90:0] input_1;
    input [0:0] sel;
    reg [90:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_91_2_2 = result;
  end
  endfunction


  function automatic [8:0] MUX_v_9_2_2;
    input [8:0] input_0;
    input [8:0] input_1;
    input [0:0] sel;
    reg [8:0] result;
  begin
    case (sel)
      1'b0 : begin
        result = input_0;
      end
      default : begin
        result = input_1;
      end
    endcase
    MUX_v_9_2_2 = result;
  end
  endfunction


  function automatic [18:0] readslicef_20_19_1;
    input [19:0] vector;
    reg [19:0] tmp;
  begin
    tmp = vector >> 1;
    readslicef_20_19_1 = tmp[18:0];
  end
  endfunction


  function automatic [18:0] readslicef_22_19_3;
    input [21:0] vector;
    reg [21:0] tmp;
  begin
    tmp = vector >> 3;
    readslicef_22_19_3 = tmp[18:0];
  end
  endfunction


  function automatic [14:0] readslicef_23_15_8;
    input [22:0] vector;
    reg [22:0] tmp;
  begin
    tmp = vector >> 8;
    readslicef_23_15_8 = tmp[14:0];
  end
  endfunction


  function automatic [0:0] readslicef_4_1_3;
    input [3:0] vector;
    reg [3:0] tmp;
  begin
    tmp = vector >> 3;
    readslicef_4_1_3 = tmp[0:0];
  end
  endfunction


  function automatic [0:0] readslicef_7_1_6;
    input [6:0] vector;
    reg [6:0] tmp;
  begin
    tmp = vector >> 6;
    readslicef_7_1_6 = tmp[0:0];
  end
  endfunction


  function automatic [16:0] signext_17_1;
    input [0:0] vector;
  begin
    signext_17_1= {{16{vector[0]}}, vector};
  end
  endfunction


  function automatic [17:0] signext_18_12;
    input [11:0] vector;
  begin
    signext_18_12= {{6{vector[11]}}, vector};
  end
  endfunction


  function automatic [3:0] signext_4_3;
    input [2:0] vector;
  begin
    signext_4_3= {{1{vector[2]}}, vector};
  end
  endfunction


  function automatic [49:0] signext_50_1;
    input [0:0] vector;
  begin
    signext_50_1= {{49{vector[0]}}, vector};
  end
  endfunction


  function automatic [91:0] signext_92_18;
    input [17:0] vector;
  begin
    signext_92_18= {{74{vector[17]}}, vector};
  end
  endfunction


  function automatic [8:0] signext_9_1;
    input [0:0] vector;
  begin
    signext_9_1= {{8{vector[0]}}, vector};
  end
  endfunction


  function automatic [17:0] conv_s2s_15_18 ;
    input [14:0]  vector ;
  begin
    conv_s2s_15_18 = {{3{vector[14]}}, vector};
  end
  endfunction


  function automatic [22:0] conv_s2s_16_23 ;
    input [15:0]  vector ;
  begin
    conv_s2s_16_23 = {{7{vector[15]}}, vector};
  end
  endfunction


  function automatic [22:0] conv_s2s_18_23 ;
    input [17:0]  vector ;
  begin
    conv_s2s_18_23 = {{5{vector[17]}}, vector};
  end
  endfunction


  function automatic [22:0] conv_s2s_20_23 ;
    input [19:0]  vector ;
  begin
    conv_s2s_20_23 = {{3{vector[19]}}, vector};
  end
  endfunction


  function automatic [22:0] conv_s2s_22_23 ;
    input [21:0]  vector ;
  begin
    conv_s2s_22_23 = {vector[21], vector};
  end
  endfunction


  function automatic [18:0] conv_s2u_18_19 ;
    input [17:0]  vector ;
  begin
    conv_s2u_18_19 = {vector[17], vector};
  end
  endfunction


  function automatic [21:0] conv_s2u_18_22 ;
    input [17:0]  vector ;
  begin
    conv_s2u_18_22 = {{4{vector[17]}}, vector};
  end
  endfunction


  function automatic [19:0] conv_s2u_19_20 ;
    input [18:0]  vector ;
  begin
    conv_s2u_19_20 = {vector[18], vector};
  end
  endfunction


  function automatic [21:0] conv_s2u_21_22 ;
    input [20:0]  vector ;
  begin
    conv_s2u_21_22 = {vector[20], vector};
  end
  endfunction


  function automatic [3:0] conv_u2s_3_4 ;
    input [2:0]  vector ;
  begin
    conv_u2s_3_4 =  {1'b0, vector};
  end
  endfunction


  function automatic [6:0] conv_u2s_6_7 ;
    input [5:0]  vector ;
  begin
    conv_u2s_6_7 =  {1'b0, vector};
  end
  endfunction


  function automatic [6:0] conv_u2u_3_7 ;
    input [2:0]  vector ;
  begin
    conv_u2u_3_7 = {{4{1'b0}}, vector};
  end
  endfunction


  function automatic [9:0] conv_u2u_4_10 ;
    input [3:0]  vector ;
  begin
    conv_u2u_4_10 = {{6{1'b0}}, vector};
  end
  endfunction


  function automatic [6:0] conv_u2u_6_7 ;
    input [5:0]  vector ;
  begin
    conv_u2u_6_7 = {1'b0, vector};
  end
  endfunction


  function automatic [18:0] conv_u2u_18_19 ;
    input [17:0]  vector ;
  begin
    conv_u2u_18_19 = {1'b0, vector};
  end
  endfunction


  function automatic [70:0] conv_u2u_67_71 ;
    input [66:0]  vector ;
  begin
    conv_u2u_67_71 = {{4{1'b0}}, vector};
  end
  endfunction

endmodule

// ------------------------------------------------------------------
//  Design Unit:    mnist_mlp
// ------------------------------------------------------------------


module mnist_mlp (
  clk, rst, input1_rsc_dat, input1_rsc_vld, input1_rsc_rdy, output1_rsc_dat, output1_rsc_vld,
      output1_rsc_rdy, const_size_in_1_rsc_dat, const_size_in_1_rsc_vld, const_size_out_1_rsc_dat,
      const_size_out_1_rsc_vld, w2_rsc_CE2, w2_rsc_A2, w2_rsc_Q2, w2_rsc_CE3, w2_rsc_A3,
      w2_rsc_Q3, b2_rsc_dat, b2_rsc_vld, w4_rsc_CE2, w4_rsc_A2, w4_rsc_Q2, w4_rsc_CE3,
      w4_rsc_A3, w4_rsc_Q3, b4_rsc_dat, b4_rsc_vld, w6_rsc_0_0_dat, w6_rsc_1_0_dat,
      w6_rsc_2_0_dat, w6_rsc_3_0_dat, w6_rsc_4_0_dat, w6_rsc_5_0_dat, w6_rsc_6_0_dat,
      w6_rsc_7_0_dat, w6_rsc_8_0_dat, w6_rsc_9_0_dat, b6_rsc_dat, b6_rsc_vld
);
  input clk;
  input rst;
  input [14111:0] input1_rsc_dat;
  input input1_rsc_vld;
  output input1_rsc_rdy;
  output [179:0] output1_rsc_dat;
  output output1_rsc_vld;
  input output1_rsc_rdy;
  output [15:0] const_size_in_1_rsc_dat;
  output const_size_in_1_rsc_vld;
  output [15:0] const_size_out_1_rsc_dat;
  output const_size_out_1_rsc_vld;
  output w2_rsc_CE2;
  output [15:0] w2_rsc_A2;
  input [17:0] w2_rsc_Q2;
  output w2_rsc_CE3;
  output [15:0] w2_rsc_A3;
  input [17:0] w2_rsc_Q3;
  input [1151:0] b2_rsc_dat;
  input b2_rsc_vld;
  output w4_rsc_CE2;
  output [11:0] w4_rsc_A2;
  input [17:0] w4_rsc_Q2;
  output w4_rsc_CE3;
  output [11:0] w4_rsc_A3;
  input [17:0] w4_rsc_Q3;
  input [1151:0] b4_rsc_dat;
  input b4_rsc_vld;
  input [1151:0] w6_rsc_0_0_dat;
  input [1151:0] w6_rsc_1_0_dat;
  input [1151:0] w6_rsc_2_0_dat;
  input [1151:0] w6_rsc_3_0_dat;
  input [1151:0] w6_rsc_4_0_dat;
  input [1151:0] w6_rsc_5_0_dat;
  input [1151:0] w6_rsc_6_0_dat;
  input [1151:0] w6_rsc_7_0_dat;
  input [1151:0] w6_rsc_8_0_dat;
  input [1151:0] w6_rsc_9_0_dat;
  input [1151:0] b6_rsc_dat;
  input b6_rsc_vld;


  // Interconnect Declarations
  wire [1:0] w2_rsci_CE2_d;
  wire [15:0] w2_rsci_A2_d;
  wire [35:0] w2_rsci_Q2_d;
  wire [1:0] w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d;
  wire [1:0] w4_rsci_CE2_d;
  wire [11:0] w4_rsci_A2_d;
  wire [35:0] w4_rsci_Q2_d;
  wire [1:0] w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d;


  // Interconnect Declarations for Component Instantiations 
  wire [31:0] nl_w2_rsci_A2_d;
  assign nl_w2_rsci_A2_d = {16'b0000000000000000 , w2_rsci_A2_d};
  wire [23:0] nl_w4_rsci_A2_d;
  assign nl_w4_rsci_A2_d = {12'b000000000000 , w4_rsci_A2_d};
  mnist_mlp_w2_Nangate_RAMS_w2_50176_18_2w2r_rport_5_50176_18_1_gen w2_rsci (
      .Q3(w2_rsc_Q3),
      .A3(w2_rsc_A3),
      .CE3(w2_rsc_CE3),
      .Q2(w2_rsc_Q2),
      .A2(w2_rsc_A2),
      .CE2(w2_rsc_CE2),
      .CE2_d(w2_rsci_CE2_d),
      .A2_d(nl_w2_rsci_A2_d[31:0]),
      .Q2_d(w2_rsci_Q2_d),
      .port_3_r_ram_ir_internal_RMASK_B_d(w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d)
    );
  mnist_mlp_w4_Nangate_RAMS_w4_4096_18_2w2r_rport_7_4096_18_1_gen w4_rsci (
      .Q3(w4_rsc_Q3),
      .A3(w4_rsc_A3),
      .CE3(w4_rsc_CE3),
      .Q2(w4_rsc_Q2),
      .A2(w4_rsc_A2),
      .CE2(w4_rsc_CE2),
      .CE2_d(w4_rsci_CE2_d),
      .A2_d(nl_w4_rsci_A2_d[23:0]),
      .Q2_d(w4_rsci_Q2_d),
      .port_3_r_ram_ir_internal_RMASK_B_d(w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d)
    );
  mnist_mlp_core mnist_mlp_core_inst (
      .clk(clk),
      .rst(rst),
      .input1_rsc_dat(input1_rsc_dat),
      .input1_rsc_vld(input1_rsc_vld),
      .input1_rsc_rdy(input1_rsc_rdy),
      .output1_rsc_dat(output1_rsc_dat),
      .output1_rsc_vld(output1_rsc_vld),
      .output1_rsc_rdy(output1_rsc_rdy),
      .const_size_in_1_rsc_dat(const_size_in_1_rsc_dat),
      .const_size_in_1_rsc_vld(const_size_in_1_rsc_vld),
      .const_size_out_1_rsc_dat(const_size_out_1_rsc_dat),
      .const_size_out_1_rsc_vld(const_size_out_1_rsc_vld),
      .b2_rsc_dat(b2_rsc_dat),
      .b2_rsc_vld(b2_rsc_vld),
      .b4_rsc_dat(b4_rsc_dat),
      .b4_rsc_vld(b4_rsc_vld),
      .w6_rsc_0_0_dat(w6_rsc_0_0_dat),
      .w6_rsc_1_0_dat(w6_rsc_1_0_dat),
      .w6_rsc_2_0_dat(w6_rsc_2_0_dat),
      .w6_rsc_3_0_dat(w6_rsc_3_0_dat),
      .w6_rsc_4_0_dat(w6_rsc_4_0_dat),
      .w6_rsc_5_0_dat(w6_rsc_5_0_dat),
      .w6_rsc_6_0_dat(w6_rsc_6_0_dat),
      .w6_rsc_7_0_dat(w6_rsc_7_0_dat),
      .w6_rsc_8_0_dat(w6_rsc_8_0_dat),
      .w6_rsc_9_0_dat(w6_rsc_9_0_dat),
      .b6_rsc_dat(b6_rsc_dat),
      .b6_rsc_vld(b6_rsc_vld),
      .w2_rsci_CE2_d(w2_rsci_CE2_d),
      .w2_rsci_A2_d(w2_rsci_A2_d),
      .w2_rsci_Q2_d(w2_rsci_Q2_d),
      .w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d(w2_rsci_port_3_r_ram_ir_internal_RMASK_B_d),
      .w4_rsci_CE2_d(w4_rsci_CE2_d),
      .w4_rsci_A2_d(w4_rsci_A2_d),
      .w4_rsci_Q2_d(w4_rsci_Q2_d),
      .w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d(w4_rsci_port_3_r_ram_ir_internal_RMASK_B_d)
    );
endmodule



