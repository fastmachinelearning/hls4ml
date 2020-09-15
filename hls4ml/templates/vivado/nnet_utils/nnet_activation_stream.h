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

#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include <cmath>
#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_types.h"
#include "nnet_stream.h"
#include "nnet_activation.h"

namespace nnet {

// *************************************************
//       LINEAR Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void linear(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    hls::stream<res_T> data_repack("linear_data_repack");
    constexpr unsigned repack_depth = CONFIG_T::n_in / res_T::size;
    #pragma HLS STREAM variable=data_repack depth=repack_depth
    repack_stream<data_T, res_T, CONFIG_T::n_in>(data, data_repack);
    
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        res_T in_data = data_repack.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}


// *************************************************
//       RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void relu(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    hls::stream<res_T> data_repack("relu_data_repack");
    constexpr unsigned repack_depth = CONFIG_T::n_in / res_T::size;
    #pragma HLS STREAM variable=data_repack depth=repack_depth
    repack_stream<data_T, res_T, CONFIG_T::n_in>(data, data_repack);
    
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        res_T in_data = data_repack.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax(hls::stream<data_T> &data, hls::stream<res_T> &res){
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<typename data_T::value_type, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    hls::stream<res_T> data_repack("softmax_data_repack");
    constexpr unsigned repack_depth = CONFIG_T::n_in / res_T::size;
    #pragma HLS STREAM variable=data_repack depth=repack_depth
    repack_stream<data_T, res_T, CONFIG_T::n_in>(data, data_repack);

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for(unsigned i = 0; i < CONFIG_T::n_in / res_T::size; i++){
        #pragma HLS PIPELINE
        res_T in_pack = data_repack.read();
        for(unsigned j = 0; j < res_T::size; j++){
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename res_T::value_type, CONFIG_T>(in_pack[j]);
            exp_res[i * res_T::size + j] = exp_table[x];
        }
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
    for(unsigned i = 0; i < CONFIG_T::n_in / res_T::size; i++){
        #pragma HLS PIPELINE
        res_T out_pack;
        for(unsigned j = 0; j < res_T::size; j++){
            #pragma HLS UNROLL
            out_pack[i * res_T::size + j] = exp_res[i * res_T::size + j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

}

#endif
