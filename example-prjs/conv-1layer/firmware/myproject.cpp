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

#include "parameters.h"
#include "myproject.h"

#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"

void myproject(
		  input_t data[Y_INPUTS][N_CHAN],
		  result_t res[Y_INPUTS][N_CHAN],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_PARTITION variable=data complete 
    #pragma HLS ARRAY_PARTITION variable=res complete 

    #pragma HLS PIPELINE

    const_size_in   = Y_INPUTS*N_CHAN;
    const_size_out  = Y_INPUTS*N_CHAN;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    //#pragma HLS ARRAY_PARTITION variable=conv1_out complete
    //conv1_t logits1[N_LAYER_1];
    //#pragma HLS ARRAY_PARTITION variable=logits1 complete
    //nnet::conv_1d<input_t, layer1_t, config1>(data, logits1_out, w1, b1);
    //nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, conv1_out);
    nnet::conv_1d<input_t, result_t, config1>(data, res, w1, b1);

}
