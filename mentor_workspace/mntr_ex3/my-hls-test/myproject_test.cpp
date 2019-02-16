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
//#include <fstream>
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

//#include "firmware/parameters.h"
#include "firmware/myproject.h"
//#include "nnet_helpers.h"

// SCVerify verification MACROs
#include "mc_scverify.h"

#ifdef MNTR_CATAPULT_HLS
CCS_MAIN (int argc, char *argv[])
{
  std::cout << "Mentor Graphics Catapult HLS" << std::endl;
#else
int main(int argc, char **argv)
{
  std::cout << "Xilinx Vivado HLS" << std::endl;
#endif

  //hls-fpga-machine-learning insert data
  input_t  data_str[N_INPUTS] = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0};


  result_t res_str[N_OUTPUTS] = {-1.0};
  unsigned short size_in, size_out;
  CCS_DESIGN(myproject)(data_str, res_str, size_in, size_out);
 
  for(int i=0; i<N_OUTPUTS; i++){
    std::cout << res_str[i] << " ";
  }
  std::cout << std::endl;

#ifdef MNTR_CATAPULT_HLS
  CCS_RETURN(0);
#endif
}
