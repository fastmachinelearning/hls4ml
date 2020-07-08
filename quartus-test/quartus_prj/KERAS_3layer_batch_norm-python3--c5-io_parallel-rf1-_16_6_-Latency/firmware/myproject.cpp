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

#include "myproject.h"

//hls-fpga-machine-learning insert cpragmas
component outputdat myproject(
    inputdat input_1
) {
    //hls-fpga-machine-learning insert weights
    #include "weights/w2.h"
    #include "weights/b2.h"
    #include "weights/w6.h"
    #include "weights/b6.h"
    #include "weights/w10.h"
    #include "weights/b10.h"
    #include "weights/w14.h"
    #include "weights/b14.h"

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    nnet::dense<input_t, layer2_t, config2>(input_1.data, layer2_out, w2, b2);

    layer5_t layer5_out[N_LAYER_2];
    nnet::relu<layer2_t, layer5_t, relu_config5>(layer2_out, layer5_out);

    layer6_t layer6_out[N_LAYER_6];
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer9_t layer9_out[N_LAYER_6];
    nnet::relu<layer6_t, layer9_t, relu_config9>(layer6_out, layer9_out);

    layer10_t layer10_out[N_LAYER_10];
    nnet::dense<layer9_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10);

    layer13_t layer13_out[N_LAYER_10];
    nnet::relu<layer10_t, layer13_t, relu_config13>(layer10_out, layer13_out);

    layer14_t layer14_out[N_LAYER_14];
    nnet::dense<layer13_t, layer14_t, config14>(layer13_out, layer14_out, w14, b14);

    hls_register outputdat layer17_out;
    nnet::softmax<layer14_t, result_t, softmax_config17>(layer14_out, layer17_out.data);

    return layer17_out;
}
