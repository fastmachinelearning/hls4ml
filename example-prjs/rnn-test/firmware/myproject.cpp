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
float w_U[N_INPUTS][N_STATE]  = {1, 2, 0, 4, 0, 2};
float w_W[N_STATE][N_STATE]   = {2, 3, 4, 0, -1, -2, -1, 0, 0};
float w_V[N_STATE][N_OUTPUTS] = {0, 2, 1, 0,-2, 3};

void myproject(
		  float data[N_LOOP][N_INPUTS],
		  float res[N_LOOP][N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0
    #pragma HLS INTERFACE ap_vld port=data,res
    // #pragma HLS PIPELINE
    const_size_in   = 2;
    const_size_out  = 2;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers
    for(int iloop = 0; iloop < N_LOOP; iloop++) {
        std::cout << std::endl << "********* Loop " << iloop << " ************" << std::endl;
        std::cout << "Data: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << data[iloop][ii] << " "; std::cout << "]" << std::endl;

        nnet::simple_rnn<float, float, config1, config1_activ>(data[iloop], res[iloop], w_U, w_W, w_V);

        std::cout << "Res: [ "; for (int ii = 0; ii < N_INPUTS; ii++) std::cout << res[iloop][ii] << " "; std::cout << "]" << std::endl;
    }

}
