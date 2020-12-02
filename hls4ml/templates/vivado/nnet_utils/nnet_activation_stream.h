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
    LinearActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        LinearPackLoop: for (int j = 0; j < res_T::size; j++) {
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
    ReLUActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        ReLUPackLoop: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************

template<class data_T, class res_T, typename CONFIG_T>
void sigmoid(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t sigmoid_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_sigmoid_table<CONFIG_T, CONFIG_T::table_size>(sigmoid_table);
        initialized = true;
    }

    SigmoidActLoop: for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        data_T in_data = data.read();
        res_T out_data;
        #pragma HLS DATA_PACK variable=out_data

        SigmoidPackLoop: for (int j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            int data_round = in_data[j]*CONFIG_T::table_size/16;
            int index = data_round + 8*CONFIG_T::table_size/16;
            if (index < 0)   index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = sigmoid_table[index];
        }

        res.write(out_data);
    }
}


// *************************************************
//       Softmax Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(hls::stream<data_T> &data, hls::stream<res_T> &res){
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

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    constexpr unsigned ii = CONFIG_T::n_in / multiplier_limit;

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma HLS array_partition variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    SoftmaxExpLoop: for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++){
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T in_pack = data.read();
        SoftmaxExpPackLoop: for(unsigned j = 0; j < data_T::size; j++){
            #pragma HLS UNROLL
            unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j]);
            exp_res[i * data_T::size + j] = exp_table[x];
        }
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
    SoftmaxInvLoop: for(unsigned i = 0; i < CONFIG_T::n_in / res_T::size; i++){
        #pragma HLS PIPELINE II=ii

        res_T out_pack;
        #pragma HLS DATA_PACK variable=out_pack
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++){
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[i * res_T::size + j] = exp_res[i * res_T::size + j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(hls::stream<data_T> &data, hls::stream<res_T> &res){
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

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in, CONFIG_T::reuse_factor);
    constexpr unsigned ii = CONFIG_T::n_in / multiplier_limit;

    typename data_T::value_type data_array[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data_array complete
    SoftmaxArrayLoop: for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++){
        if (CONFIG_T::n_in / data_T::size > 1) {
            #pragma HLS PIPELINE
        }
        data_T in_pack = data.read();
        SoftmaxArrayPackLoop: for(unsigned j = 0; j < data_T::size; j++){
            #pragma HLS UNROLL
            data_array[i * data_T::size + j] = in_pack[j];
        }
    }

    // Find the max and compute all delta(x_i, x_max)
    Op_max<typename data_T::value_type> op_max;
    typename data_T::value_type x_max = reduce<typename data_T::value_type, CONFIG_T::n_in, Op_max<typename data_T::value_type>>(data_array, op_max);

    // For the diffs, use the same type as the input but force rounding and saturation
    ap_fixed<data_T::value_type::width, data_T::value_type::iwidth,AP_RND,AP_SAT> d_xi_xmax[CONFIG_T::n_in];
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS UNROLL
        d_xi_xmax[i] = data_array[i] - x_max;
    }

    // Calculate all the e^x's
    typename CONFIG_T::exp_table_t exp_res[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=exp_res complete
    typename CONFIG_T::exp_table_t exp_sum(0);
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS UNROLL
        unsigned x = softmax_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[i]);
        exp_res[i] = exp_table[x];
    }

    // Explicitly sum the results with an adder tree.
    // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
    Op_add<typename CONFIG_T::exp_table_t> op_add;
    exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

    typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
    SoftmaxInvLoop: for(unsigned i = 0; i < CONFIG_T::n_in / res_T::size; i++){
        #pragma HLS PIPELINE II=ii

        res_T out_pack;
        #pragma HLS DATA_PACK variable=out_pack
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++){
            #pragma HLS UNROLL
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[i * res_T::size + j] = exp_res[i * res_T::size + j] * inv_exp_sum;
        }
        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(hls::stream<data_T> &data, hls::stream<res_T> &res) {
    // Initialize the lookup table
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::table_t invert_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_invert_table_legacy<CONFIG_T, CONFIG_T::table_size>(invert_table);
        initialized = true;
    }

    // Index into the lookup table based on data for exponentials
    typename CONFIG_T::table_t exp_res[CONFIG_T::n_in];
    typename CONFIG_T::table_t exp_diff_res;
    typename data_T::value_type data_cache[CONFIG_T::n_in];

    SoftmaxInitLoop: for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        #pragma HLS PIPELINE
        data_T in_pack = data.read();
        SoftmaxInitPackLoop: for(unsigned j = 0; j < data_T::size; j++) {
            #pragma HLS UNROLL
            data_cache[i * data_T::size + j] = in_pack[j];
            exp_res[i * data_T::size + j] = 0;
        }
    }

    SoftmaxExpLoop: for (int i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS PIPELINE
        SoftmaxExpInner: for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            
            if (i == j) {
                exp_diff_res = 1;
            } else {
                int data_round = (data_cache[j] - data_cache[i]) * CONFIG_T::table_size / 16;
                int index = data_round + 8 * CONFIG_T::table_size / 16;
                if (index < 0) index = 0;
                if (index > CONFIG_T::table_size - 1) index = CONFIG_T::table_size - 1;
                exp_diff_res = exp_table[index];
            }
            
            exp_res[i] += exp_diff_res;
        }
    }

    SoftmaxInvLoop: for(unsigned i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        #pragma HLS PIPELINE

        res_T out_pack;
        #pragma HLS DATA_PACK variable=out_pack
        SoftmaxInvPackLoop: for(unsigned j = 0; j < res_T::size; j++) {
            #pragma HLS UNROLL
            
            int exp_res_index = exp_res[i * res_T::size + j] * CONFIG_T::table_size / 64;
            if (exp_res_index < 0) exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1) exp_res_index = CONFIG_T::table_size - 1;
            
            out_pack[i * res_T::size + j] = (typename res_T::value_type) invert_table[exp_res_index];
        }
        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax(hls::stream<data_T> &data, hls::stream<res_T> &res){
    switch(CONFIG_T::implementation){
    case softmax_implementation::latency:
        softmax_latency<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::stable:
        softmax_stable<data_T, res_T, CONFIG_T>(data, res);
        break;
    case softmax_implementation::legacy:
        softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
        break;
    }    
}

}

#endif
