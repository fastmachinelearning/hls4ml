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
#include "nnet_conv2d.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2_0.h"
#include "weights/b2_0.h"
#include "weights/w2_1.h"
#include "weights/b2_1.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w4.h"
#include "weights/b4.h"

void myproject(
		  input_t data[N_INPUTS],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = N_INPUTS;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[N_LAYER_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    layer1_t logits1[N_LAYER_1];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::dense<input_t, layer1_t, config1>(data, logits1, w1, b1);
    nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    layer2_t logits2[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    dense2(layer1_out, logits2);
    nnet::relu<layer2_t, layer2_t, relu_config2>(logits2, layer2_out);

    layer3_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    layer3_t logits3[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=logits3 complete dim=0
    nnet::dense<layer2_t, layer3_t, config3>(layer2_out, logits3, w3, b3);
    nnet::relu<layer3_t, layer3_t, relu_config3>(logits3, layer3_out);

    result_t logits4[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits4 complete dim=0
    nnet::dense<layer3_t, result_t, config4>(layer3_out, logits4, w4, b4);
    nnet::softmax<result_t, result_t, softmax_config4>(logits4, res);


}

void dense2(layer1_t layer1_out[N_LAYER_1], layer2_t logits2[N_LAYER_2]) {
    layer2_t logits2_0[16];
    #pragma HLS ARRAY_PARTITION variable=logits2_0 complete dim=0
    layer2_t logits2_1[16];
    #pragma HLS ARRAY_PARTITION variable=logits2_1 complete dim=0
    nnet::dense<layer1_t, layer2_t, config2_0>(layer1_out, logits2_0, w2_0, b2_0);
    nnet::dense<layer1_t, layer2_t, config2_1>(layer1_out, logits2_1, w2_1, b2_1);
    nnet::merge<layer2_t, 16, 16>(logits2_0, logits2_1, logits2);
}

