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

#include "mnist_mlp.h"

////hls-fpga-machine-learning insert weights
//#include "weights/w2.h"
//#include "weights/b2.h"
//#include "weights/w4.h"
//#include "weights/b4.h"
//#include "weights/w6.h"
//#include "weights/b6.h"


#include <string>
#include <limits.h>
#include <unistd.h>

std::string getexepath()
{
  char result[ PATH_MAX ];
   ssize_t count = readlink( "/proc/self/exe", result, PATH_MAX );
    return std::string( result, (count > 0) ? count : 0 );
}

void mnist_mlp(
    input_t input1[N_INPUT_1_1],
    result_t output1[N_LAYER_6],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1,
    model_default_t w2[50176],
    model_default_t b2[64],
    model_default_t w4[4096],
    model_default_t b4[64],
    model_default_t w6[640],
    model_default_t b6[10]
) {

    //hls-fpga-machine-learning insert IO
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=output1 complete dim=0 
    #pragma HLS INTERFACE ap_vld port=input1,output1
    #pragma HLS DATAFLOW 
#endif
    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_6;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
#endif
    nnet::dense_large<input_t, layer2_t, config2>(input1, layer2_out, w2, b2);

    layer3_t layer3_out[N_LAYER_2];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
#endif
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out);

    layer4_t layer4_out[N_LAYER_4];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
#endif
    nnet::dense_large<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

    layer5_t layer5_out[N_LAYER_4];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
#endif
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out);

    layer6_t layer6_out[N_LAYER_6];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
#endif
    nnet::dense_large<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer6_t layer7_out[N_LAYER_6];
#ifdef XLNX_VIVADO_HLS
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
#endif
    nnet::softmax<layer6_t, result_t, softmax_config7>(layer6_out, layer7_out);

OUTPUT_LOOP:
    for (unsigned i = 0; i < N_LAYER_6; i++) {
#ifdef XLNX_VIVADO_HLS
    #pragma HLS UNROLL
#endif
        output1[i] = layer7_out[i]; 
    }
}
