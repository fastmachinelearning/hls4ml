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
#include <iostream>

#include "parameters.h"
#include "myproject.h"


#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_recursive.h"

//hls-fpga-machine-learning insert weights
mytype w_F[N_STATE][N_STATE+N_INPUTS] = {1, 2, 3, 4, 5, 6, 7, 8};
mytype w_I[N_STATE][N_STATE+N_INPUTS] = {1, 2, 3, 4, 5, 6, 7, 8};
mytype w_O[N_STATE][N_STATE+N_INPUTS] = {1, 2, 3, 4, 5, 6, 7, 8};
mytype w_G[N_STATE][N_STATE+N_INPUTS] = {1, 2, 3, 4, 5, 6, 7, 8};

mytype b_F[N_STATE] = {1, 2};
mytype b_I[N_STATE] = {1, 2};
mytype b_O[N_STATE] = {1, 2};
mytype b_G[N_STATE] = {1, 2};
void myproject(
	       mytype data[N_LOOP][N_INPUTS],
	       mytype res [N_LOOP][N_OUTPUTS],
	       unsigned short &const_size_in,
	       unsigned short &const_size_out)
{

    mytype s_state[N_STATE]            = {0,0};
    mytype tmp_state[N_STATE]          = {0,0};
    #pragma HLS ARRAY_PARTITION variable=s_state   dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=tmp_state dim=0 complete
    #pragma HLS ARRAY_PARTITION variable=data      dim=0 complete 
    #pragma HLS ARRAY_PARTITION variable=res       dim=0 complete 
    #pragma HLS INTERFACE ap_vld port=data,res
    #pragma HLS PIPELINE
    const_size_in   = 2;
    const_size_out  = 2;

   // ****************************************
   // NETWORK INSTANTIATION
   // ****************************************
   //hls-fpga-machine-learning insert layers
   //First layer we pass a temporary state for H as 0s
   std::cout << "Data: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << data[0][ii] << " "; std::cout << "]" << std::endl;
   nnet::lstm<mytype, mytype, config1, config1_activ>(data[0], res[0], s_state, tmp_state,  w_F, w_I, w_G, w_O, b_F, b_I, b_G, b_O);
   std::cout << "Res: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << res[0][ii] << " "; std::cout << "]" << std::endl;
   for(int iloop = 1; iloop < N_LOOP; iloop++) {
        std::cout << std::endl << "********* Loop " << iloop << " ************" << std::endl;
        std::cout << "Data: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << data[iloop][ii] << " "; std::cout << "]" << std::endl;
	nnet::lstm<mytype, mytype, config1, config1_activ>(data[iloop], res[iloop], s_state, res[iloop-1],  w_F, w_I, w_G, w_O, b_F, b_I, b_G, b_O);
        std::cout << "Res: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << res[iloop][ii] << " "; std::cout << "]" << std::endl;
    }
}
