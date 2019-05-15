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

//#include "nnet_layer.h"
#include "nnet_large_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_batchnorm.h"
#include "nnet_activation.h"
#include "nnet_pooling.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2.h"
#include "weights/b2.h"

#ifndef __SYNTHESIS__
#include <fstream>
#define xstr(a) str(a)
#define str(a) #a
template<class T, size_t SIZE>
void load_txt_file(T *w, const char* fname) {

    std::string full_path = std::string(xstr(WEIGHTS_DIR)) + "/" + std::string(fname);
    std::ifstream infile(full_path.c_str(), std::ios::binary);

    if (infile.fail()) {
        std::cerr << "ERROR: file " << std::string(fname);
        std::cerr << " does not exist" << std::endl;
        exit(1);
    }

    size_t i = 0;
    size_t size;
    std::string line;

    // The first line of the input file contains the total number of values.
    if (std::getline(infile, line)) {
         std::istringstream iss(line);
         iss >> size;
         if (size != SIZE) {
            std::cerr << "ERROR: file " << std::string(fname);
            std::cerr << " contains an unexpected number of elements (";
            std::cerr << size << " rather than  "<< SIZE << ")" << std::endl;
            exit(1);
        }
    };

    // The second line of the input file contains all of the values.
    if (std::getline(infile, line)) {
        std::istringstream iss(line);
        for (size_t i = 0; i < size; i++) {
            double fdata;
            iss >> fdata;
            w[i] = T(fdata);
        }
    }
}
#endif

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
#pragma HLS DATAFLOW

#ifndef __SYNTHESIS__
    load_txt_file< weight_default_t, config1::n_in * config1::n_out >(w1, "w1.txt");
    load_txt_file< weight_default_t, config2::n_in * config2::n_out >(w2, "w2.txt");
    load_txt_file< bias_default_t, config1::n_out >(b1, "b1.txt");
    load_txt_file< bias_default_t, config2::n_out >(b2, "b2.txt");
#endif

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
    nnet::compute_large_layer<input_t, layer1_t, config1>(data, logits1, w1, b1);
    nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

    result_t logits2[N_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::compute_large_layer<layer1_t, result_t, config2>(layer1_out, logits2, w2, b2);
    nnet::softmax<result_t, result_t, softmax_config2>(logits2, res);


}
