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
    #include "weights/w4.h"
    #include "weights/b4.h"

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    nnet::dense<input_t, layer2_t, config2>(input_1.data, layer2_out, w2, b2);

    layer3_t layer3_out[N_LAYER_2];
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out);

    layer4_t layer4_out[N_LAYER_4];
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

    hls_register outputdat layer5_out;
    nnet::sigmoid<layer4_t, result_t, sigmoid_config5>(layer4_out, layer5_out.data);

    return layer5_out;
}
