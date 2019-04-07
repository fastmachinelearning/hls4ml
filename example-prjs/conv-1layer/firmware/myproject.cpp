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


#include "nnet_dense.h"
#include "nnet_conv.h"
#include "nnet_activation.h"



//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2.h"
#include "weights/b2.h"

void myproject(
		  input_t data[Y_INPUTS][N_CHAN],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0
    #pragma HLS INTERFACE ap_vld port=data,res

    #pragma HLS PIPELINE

    const_size_in   = Y_INPUTS*N_CHAN;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers
    
    //Conv1d
    input_t layer1_out[Y_OUTPUTS][N_FILT];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    nnet::conv_1d<input_t, input_t, config1>(data, layer1_out, w1, b1);

    //Flatten
    input_t layer1_out_flat[Y_OUTPUTS*N_FILT];
    #pragma HLS ARRAY_PARTITION variable=layer1_out_flat complete dim=0
    nnet::flatten<input_t, Y_OUTPUTS, N_FILT>(layer1_out, layer1_out_flat);
    //TODO:: CHANGE flatten to use a struct

    //Relu
    input_t layer1_out_flat_relu[Y_OUTPUTS*N_FILT];
    #pragma HLS ARRAY_PARTITION variable=layer1_out_flat_relu complete dim=0
    nnet::relu<input_t, input_t, relu_config1>(layer1_out_flat, layer1_out_flat_relu); 

    //Dense
    result_t logits2[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::dense<input_t, result_t, config2>(layer1_out_flat_relu, logits2, w2, b2);
    
    //Softmax
    nnet::softmax<result_t, result_t, softmax_config2>(logits2, res);

}
