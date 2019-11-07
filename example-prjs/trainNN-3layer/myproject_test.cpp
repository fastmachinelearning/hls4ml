//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"
#include "nnet_helpers.h"


int main(int argc, char **argv)
{
  //hls-fpga-machine-learning insert data
  input_t input1[N_INPUT_1_1] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
  result_t layer7_out[N_LAYER_5] = {0};

  //hls-fpga-machine-learning insert top-level-function
  unsigned short size_in1,size_out1;
  myproject(input1,layer7_out,size_in1,size_out1);

  //hls-fpga-machine-learning insert output
  for(int i = 0; i < N_LAYER_5; i++) {
    std::cout << layer7_out[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
