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
  input_t  X_str[N_NODES][N_FEATURES] = {0.0319,  0.4396,  0.0276, 
					 0.0713,  0.4137,  0.0631, 
					 0.0315,  0.7407,  0.0110, 
					 0.0717,  0.7647,  0.0284};

  ap_uint<1> Ri_str[N_NODES][N_EDGES] =   {0,  0,  0,  0, 
					 1,  0,  1,  0, 
					 0,  0,  0,  0, 
					 0,  1,  0,  1};

  ap_uint<1> Ro_str[N_NODES][N_EDGES] =   {1,  1,  0,  0, 
					 0,  0,  0,  0, 
					 0,  0,  1,  1, 
					 0,  0,  0,  0};

  result_t e_str[N_EDGES][1];
  for(int i=0; i<N_EDGES; i++){
    e_str[i][0]=0;
  }
  

  unsigned short size_in, size_out;
  myproject(X_str, Ri_str, Ro_str, e_str, size_in, size_out);
    
  std::cout << "e = " << std::endl;
  for(int i=0; i<N_EDGES; i++){
    std::cout << e_str[i][0] << " ";
  }
  std::cout << std::endl;
  
  return 0;
}

