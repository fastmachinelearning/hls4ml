//
// Created with the ESP Memory Generator
//
// Copyright (c) 2011-2019 Columbia University, System Level Design Group
// SPDX-License-Identifier: Apache-2.0
//
// @author Paolo Mantovani <paolo@cs.columbia.edu>
//

`timescale  1 ps / 1 ps

module w2_data56448_1w16r(
    CLK,
    CE0,
    A0,
    D0,
    WE0,
    WEM0,
    CE1,
    A1,
    Q1,
    CE2,
    A2,
    Q2,
    CE3,
    A3,
    Q3,
    CE4,
    A4,
    Q4,
    CE5,
    A5,
    Q5,
    CE6,
    A6,
    Q6,
    CE7,
    A7,
    Q7,
    CE8,
    A8,
    Q8,
    CE9,
    A9,
    Q9,
    CE10,
    A10,
    Q10,
    CE11,
    A11,
    Q11,
    CE12,
    A12,
    Q12,
    CE13,
    A13,
    Q13,
    CE14,
    A14,
    Q14,
    CE15,
    A15,
    Q15,
    CE16,
    A16,
    Q16
  );
  input CLK;
  input CE0;
  input [3:0] A0;
  input [56447:0] D0;
  input WE0;
  input [56447:0] WEM0;
  input CE1;
  input [3:0] A1;
  output [56447:0] Q1;
  input CE2;
  input [3:0] A2;
  output [56447:0] Q2;
  input CE3;
  input [3:0] A3;
  output [56447:0] Q3;
  input CE4;
  input [3:0] A4;
  output [56447:0] Q4;
  input CE5;
  input [3:0] A5;
  output [56447:0] Q5;
  input CE6;
  input [3:0] A6;
  output [56447:0] Q6;
  input CE7;
  input [3:0] A7;
  output [56447:0] Q7;
  input CE8;
  input [3:0] A8;
  output [56447:0] Q8;
  input CE9;
  input [3:0] A9;
  output [56447:0] Q9;
  input CE10;
  input [3:0] A10;
  output [56447:0] Q10;
  input CE11;
  input [3:0] A11;
  output [56447:0] Q11;
  input CE12;
  input [3:0] A12;
  output [56447:0] Q12;
  input CE13;
  input [3:0] A13;
  output [56447:0] Q13;
  input CE14;
  input [3:0] A14;
  output [56447:0] Q14;
  input CE15;
  input [3:0] A15;
  output [56447:0] Q15;
  input CE16;
  input [3:0] A16;
  output [56447:0] Q16;
  genvar d, h, v, hh;

  reg               bank_CE  [0:0][7:0][0:0][1763:0][1:0];
  reg         [8:0] bank_A   [0:0][7:0][0:0][1763:0][1:0];
  reg        [31:0] bank_D   [0:0][7:0][0:0][1763:0][1:0];
  reg               bank_WE  [0:0][7:0][0:0][1763:0][1:0];
  reg        [31:0] bank_WEM [0:0][7:0][0:0][1763:0][1:0];
  wire       [31:0] bank_Q   [0:0][7:0][0:0][1763:0][1:0];
  wire        [0:0] ctrld    [16:1];
  wire        [2:0] ctrlh    [16:0];
  wire        [0:0] ctrlv    [16:0];
  reg         [0:0] seld     [16:1];
  reg         [2:0] selh     [16:1];
  reg         [0:0] selv     [16:1];
// synthesis translate_off
// synopsys translate_off
  integer check_bank_access [0:0][7:0][0:0][1763:0][1:0];

  task check_access;
    input integer iface;
    input integer d;
    input integer h;
    input integer v;
    input integer hh;
    input integer p;
  begin
    if ((check_bank_access[d][h][v][hh][p] != -1) &&
        (check_bank_access[d][h][v][hh][p] != iface)) begin
      $display("ASSERTION FAILED in %m: port conflict on bank", h, "h", v, "v", hh, "hh", " for port", p, " involving interfaces", check_bank_access[d][h][v][hh][p], iface);
      $finish;
    end
    else begin
      check_bank_access[d][h][v][hh][p] = iface;
    end
  end
  endtask
// synopsys translate_on
// synthesis translate_on

  assign ctrld[1] = 0;
  assign ctrld[2] = 0;
  assign ctrld[3] = 0;
  assign ctrld[4] = 0;
  assign ctrld[5] = 0;
  assign ctrld[6] = 0;
  assign ctrld[7] = 0;
  assign ctrld[8] = 0;
  assign ctrld[9] = 0;
  assign ctrld[10] = 0;
  assign ctrld[11] = 0;
  assign ctrld[12] = 0;
  assign ctrld[13] = 0;
  assign ctrld[14] = 0;
  assign ctrld[15] = 0;
  assign ctrld[16] = 0;
  assign ctrlh[0] = A0[2:0];
  assign ctrlh[1] = A1[2:0];
  assign ctrlh[2] = A2[2:0];
  assign ctrlh[3] = A3[2:0];
  assign ctrlh[4] = A4[2:0];
  assign ctrlh[5] = A5[2:0];
  assign ctrlh[6] = A6[2:0];
  assign ctrlh[7] = A7[2:0];
  assign ctrlh[8] = A8[2:0];
  assign ctrlh[9] = A9[2:0];
  assign ctrlh[10] = A10[2:0];
  assign ctrlh[11] = A11[2:0];
  assign ctrlh[12] = A12[2:0];
  assign ctrlh[13] = A13[2:0];
  assign ctrlh[14] = A14[2:0];
  assign ctrlh[15] = A15[2:0];
  assign ctrlh[16] = A16[2:0];
  assign ctrlv[0] = 0;
  assign ctrlv[1] = 0;
  assign ctrlv[2] = 0;
  assign ctrlv[3] = 0;
  assign ctrlv[4] = 0;
  assign ctrlv[5] = 0;
  assign ctrlv[6] = 0;
  assign ctrlv[7] = 0;
  assign ctrlv[8] = 0;
  assign ctrlv[9] = 0;
  assign ctrlv[10] = 0;
  assign ctrlv[11] = 0;
  assign ctrlv[12] = 0;
  assign ctrlv[13] = 0;
  assign ctrlv[14] = 0;
  assign ctrlv[15] = 0;
  assign ctrlv[16] = 0;

  always @(posedge CLK) begin
    seld[1] <= ctrld[1];
    selh[1] <= ctrlh[1];
    selv[1] <= ctrlv[1];
    seld[2] <= ctrld[2];
    selh[2] <= ctrlh[2];
    selv[2] <= ctrlv[2];
    seld[3] <= ctrld[3];
    selh[3] <= ctrlh[3];
    selv[3] <= ctrlv[3];
    seld[4] <= ctrld[4];
    selh[4] <= ctrlh[4];
    selv[4] <= ctrlv[4];
    seld[5] <= ctrld[5];
    selh[5] <= ctrlh[5];
    selv[5] <= ctrlv[5];
    seld[6] <= ctrld[6];
    selh[6] <= ctrlh[6];
    selv[6] <= ctrlv[6];
    seld[7] <= ctrld[7];
    selh[7] <= ctrlh[7];
    selv[7] <= ctrlv[7];
    seld[8] <= ctrld[8];
    selh[8] <= ctrlh[8];
    selv[8] <= ctrlv[8];
    seld[9] <= ctrld[9];
    selh[9] <= ctrlh[9];
    selv[9] <= ctrlv[9];
    seld[10] <= ctrld[10];
    selh[10] <= ctrlh[10];
    selv[10] <= ctrlv[10];
    seld[11] <= ctrld[11];
    selh[11] <= ctrlh[11];
    selv[11] <= ctrlv[11];
    seld[12] <= ctrld[12];
    selh[12] <= ctrlh[12];
    selv[12] <= ctrlv[12];
    seld[13] <= ctrld[13];
    selh[13] <= ctrlh[13];
    selv[13] <= ctrlv[13];
    seld[14] <= ctrld[14];
    selh[14] <= ctrlh[14];
    selv[14] <= ctrlv[14];
    seld[15] <= ctrld[15];
    selh[15] <= ctrlh[15];
    selv[15] <= ctrlv[15];
    seld[16] <= ctrld[16];
    selh[16] <= ctrlh[16];
    selv[16] <= ctrlv[16];
  end

  generate
  for (h = 0; h < 8; h = h + 1) begin : gen_ctrl_hbanks
    for (v = 0; v < 1; v = v + 1) begin : gen_ctrl_vbanks
      for (hh = 0; hh < 1764; hh = hh + 1) begin : gen_ctrl_hhbanks

        always @(*) begin : handle_ops

// synthesis translate_off
// synopsys translate_off
          // Prevent assertions to trigger with false positive
          # 1
// synopsys translate_on
// synthesis translate_on

          /** Default **/
// synthesis translate_off
// synopsys translate_off
          check_bank_access[0][h][v][hh][0] = -1;
// synopsys translate_on
// synthesis translate_on
          bank_CE[0][h][v][hh][0]  = 0;
          bank_A[0][h][v][hh][0]   = 0;
          bank_D[0][h][v][hh][0]   = 0;
          bank_WE[0][h][v][hh][0]  = 0;
          bank_WEM[0][h][v][hh][0] = 0;
// synthesis translate_off
// synopsys translate_off
          check_bank_access[0][h][v][hh][1] = -1;
// synopsys translate_on
// synthesis translate_on
          bank_CE[0][h][v][hh][1]  = 0;
          bank_A[0][h][v][hh][1]   = 0;
          bank_D[0][h][v][hh][1]   = 0;
          bank_WE[0][h][v][hh][1]  = 0;
          bank_WEM[0][h][v][hh][1] = 0;

          /** Handle 1w:0r **/
          // Duplicated bank set 0
            if (ctrlh[0] == h && ctrlv[0] == v && CE0 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(0, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE0;
                bank_A[0][h][v][hh][0]   = A0[3:3];
              if (hh != 1763) begin
                bank_D[0][h][v][hh][0]   = D0[32 * (hh + 1) - 1:32 * hh];
                bank_WEM[0][h][v][hh][0] = WEM0[32 * (hh + 1) - 1:32 * hh];
              end
              else begin
                bank_D[0][h][v][hh][0]   = D0[31 + 32 * hh:32 * hh];
                bank_WEM[0][h][v][hh][0] = WEM0[31 + 32 * hh:32 * hh];
              end
                bank_WE[0][h][v][hh][0]  = WE0;
            end

          /** Handle 0w:16r **/
          // Always choose duplicated bank set 0
            if (ctrlh[1] == h && ctrlv[1] == v && CE1 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(1, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE1;
                bank_A[0][h][v][hh][0]   = A1[3:3];
            end
            if (ctrlh[2] == h && ctrlv[2] == v && CE2 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(2, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE2;
                bank_A[0][h][v][hh][1]   = A2[3:3];
            end
            if (ctrlh[3] == h && ctrlv[3] == v && CE3 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(3, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE3;
                bank_A[0][h][v][hh][0]   = A3[3:3];
            end
            if (ctrlh[4] == h && ctrlv[4] == v && CE4 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(4, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE4;
                bank_A[0][h][v][hh][1]   = A4[3:3];
            end
            if (ctrlh[5] == h && ctrlv[5] == v && CE5 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(5, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE5;
                bank_A[0][h][v][hh][0]   = A5[3:3];
            end
            if (ctrlh[6] == h && ctrlv[6] == v && CE6 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(6, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE6;
                bank_A[0][h][v][hh][1]   = A6[3:3];
            end
            if (ctrlh[7] == h && ctrlv[7] == v && CE7 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(7, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE7;
                bank_A[0][h][v][hh][0]   = A7[3:3];
            end
            if (ctrlh[8] == h && ctrlv[8] == v && CE8 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(8, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE8;
                bank_A[0][h][v][hh][1]   = A8[3:3];
            end
            if (ctrlh[9] == h && ctrlv[9] == v && CE9 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(9, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE9;
                bank_A[0][h][v][hh][1]   = A9[3:3];
            end
            if (ctrlh[10] == h && ctrlv[10] == v && CE10 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(10, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE10;
                bank_A[0][h][v][hh][0]   = A10[3:3];
            end
            if (ctrlh[11] == h && ctrlv[11] == v && CE11 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(11, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE11;
                bank_A[0][h][v][hh][1]   = A11[3:3];
            end
            if (ctrlh[12] == h && ctrlv[12] == v && CE12 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(12, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE12;
                bank_A[0][h][v][hh][0]   = A12[3:3];
            end
            if (ctrlh[13] == h && ctrlv[13] == v && CE13 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(13, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE13;
                bank_A[0][h][v][hh][1]   = A13[3:3];
            end
            if (ctrlh[14] == h && ctrlv[14] == v && CE14 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(14, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE14;
                bank_A[0][h][v][hh][0]   = A14[3:3];
            end
            if (ctrlh[15] == h && ctrlv[15] == v && CE15 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(15, 0, h, v, hh, 1);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][1]  = CE15;
                bank_A[0][h][v][hh][1]   = A15[3:3];
            end
            if (ctrlh[16] == h && ctrlv[16] == v && CE16 == 1'b1) begin
// synthesis translate_off
// synopsys translate_off
              check_access(16, 0, h, v, hh, 0);
// synopsys translate_on
// synthesis translate_on
                bank_CE[0][h][v][hh][0]  = CE16;
                bank_A[0][h][v][hh][0]   = A16[3:3];
            end

        end

      end
    end
  end
  endgenerate

  generate
  for (hh = 0; hh < 1764; hh = hh + 1) begin : gen_q_assign_hhbanks
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_1 
       assign Q1[56447:32 * hh] = bank_Q[seld[1]][selh[1]][selv[1]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_1 
      assign Q1[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[1]][selh[1]][selv[1]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_2 
       assign Q2[56447:32 * hh] = bank_Q[seld[2]][selh[2]][selv[2]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_2 
      assign Q2[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[2]][selh[2]][selv[2]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_3 
       assign Q3[56447:32 * hh] = bank_Q[seld[3]][selh[3]][selv[3]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_3 
      assign Q3[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[3]][selh[3]][selv[3]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_4 
       assign Q4[56447:32 * hh] = bank_Q[seld[4]][selh[4]][selv[4]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_4 
      assign Q4[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[4]][selh[4]][selv[4]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_5 
       assign Q5[56447:32 * hh] = bank_Q[seld[5]][selh[5]][selv[5]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_5 
      assign Q5[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[5]][selh[5]][selv[5]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_6 
       assign Q6[56447:32 * hh] = bank_Q[seld[6]][selh[6]][selv[6]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_6 
      assign Q6[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[6]][selh[6]][selv[6]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_7 
       assign Q7[56447:32 * hh] = bank_Q[seld[7]][selh[7]][selv[7]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_7 
      assign Q7[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[7]][selh[7]][selv[7]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_8 
       assign Q8[56447:32 * hh] = bank_Q[seld[8]][selh[8]][selv[8]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_8 
      assign Q8[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[8]][selh[8]][selv[8]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_9 
       assign Q9[56447:32 * hh] = bank_Q[seld[9]][selh[9]][selv[9]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_9 
      assign Q9[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[9]][selh[9]][selv[9]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_10 
       assign Q10[56447:32 * hh] = bank_Q[seld[10]][selh[10]][selv[10]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_10 
      assign Q10[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[10]][selh[10]][selv[10]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_11 
       assign Q11[56447:32 * hh] = bank_Q[seld[11]][selh[11]][selv[11]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_11 
      assign Q11[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[11]][selh[11]][selv[11]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_12 
       assign Q12[56447:32 * hh] = bank_Q[seld[12]][selh[12]][selv[12]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_12 
      assign Q12[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[12]][selh[12]][selv[12]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_13 
       assign Q13[56447:32 * hh] = bank_Q[seld[13]][selh[13]][selv[13]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_13 
      assign Q13[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[13]][selh[13]][selv[13]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_14 
       assign Q14[56447:32 * hh] = bank_Q[seld[14]][selh[14]][selv[14]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_14 
      assign Q14[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[14]][selh[14]][selv[14]][hh][0];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_15 
       assign Q15[56447:32 * hh] = bank_Q[seld[15]][selh[15]][selv[15]][hh][1][31:0];
    end else begin : gen_q_assign_hhbanks_others_15 
      assign Q15[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[15]][selh[15]][selv[15]][hh][1];
    end
    if (hh == 1763 && (hh + 1) * 32 > 56448) begin : gen_q_assign_hhbanks_last_16 
       assign Q16[56447:32 * hh] = bank_Q[seld[16]][selh[16]][selv[16]][hh][0][31:0];
    end else begin : gen_q_assign_hhbanks_others_16 
      assign Q16[32 * (hh + 1) - 1:32 * hh] = bank_Q[seld[16]][selh[16]][selv[16]][hh][0];
    end
  end
  endgenerate

  generate
  for (d = 0; d < 1; d = d + 1) begin : gen_wires_dbanks
    for (h = 0; h < 8; h = h + 1) begin : gen_wires_hbanks
      for (v = 0; v < 1; v = v + 1) begin : gen_wires_vbanks
        for (hh = 0; hh < 1764; hh = hh + 1) begin : gen_wires_hhbanks

          BRAM_512x32 bank_i(
              .CLK(CLK),
              .CE0(bank_CE[d][h][v][hh][0]),
              .A0(bank_A[d][h][v][hh][0]),
              .D0(bank_D[d][h][v][hh][0]),
              .WE0(bank_WE[d][h][v][hh][0]),
              .WEM0(bank_WEM[d][h][v][hh][0]),
              .Q0(bank_Q[d][h][v][hh][0]),
              .CE1(bank_CE[d][h][v][hh][1]),
              .A1(bank_A[d][h][v][hh][1]),
              .D1(bank_D[d][h][v][hh][1]),
              .WE1(bank_WE[d][h][v][hh][1]),
              .WEM1(bank_WEM[d][h][v][hh][1]),
              .Q1(bank_Q[d][h][v][hh][1])
            );

// synthesis translate_off
// synopsys translate_off
            always @(posedge CLK) begin
              if ((bank_CE[d][h][v][hh][0] & bank_CE[d][h][v][hh][1]) &&
                  (bank_WE[d][h][v][hh][0] | bank_WE[d][h][v][hh][1]) &&
                  (bank_A[d][h][v][hh][0] == bank_A[d][h][v][hh][1])) begin
                $display("ASSERTION FAILED in %m: address conflict on bank", h, "h", v, "v", hh, "hh");
                $finish;
              end
            end
// synopsys translate_on
// synthesis translate_on

        end
      end
    end
  end
  endgenerate

endmodule
