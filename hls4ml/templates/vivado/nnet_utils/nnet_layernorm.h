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

#ifndef NNET_BATCHNORM_H_
#define NNET_BATCHNORM_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "hls_stream.h"
#include <math.h>
#include <iostream>

namespace nnet {

struct layernorm_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float scale_t;

    // Layer Sizes
    static const unsigned n_in = 20;
    static const unsigned seq_len = 4;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

template<typename CONFIG_T, int N_TABLE>
void init_invert_sqr_table(typename CONFIG_T::table_t table_out[N_TABLE])
{
    float inv_range = 0.01;
    // Inversion function:
    //   result = 1/sqrt(x)
    for (int ii = 0; ii < N_TABLE; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range 0 to +2)
        float in_val = inv_range*ii/float(N_TABLE);
        // Next, compute lookup table function
        if (in_val > 0.0) table_out[ii] = 1.0/sqrt(in_val);
        else table_out[ii] = 0.0;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void layernorm_1d(
    data_T    data[CONFIG_T::n_in/CONFIG_T::seq_len],
    res_T     res[CONFIG_T::n_in/CONFIG_T::seq_len],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_in/CONFIG_T::seq_len],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_in/CONFIG_T::seq_len]
)
{   

int inv_range_inv = (int) 1/0.01;
typename CONFIG_T::table_t deno_inver = 0;
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t invert_sqr_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_invert_sqr_table<CONFIG_T, CONFIG_T::table_size>(invert_sqr_table);
        initialized = true;
    }

    static const unsigned dim = CONFIG_T::n_in/CONFIG_T::seq_len;
    data_T sum_cache = 0;
    data_T sum_cache2 = 0; 
    data_T var, mean, diff;
    data_T data_diff[dim];
    data_T data_norm[dim];
    
    const data_T k_inv = 1.0/dim;
    for (int i = 0; i < dim; ++i){
        sum_cache += data[i];
    }
    mean = CONFIG_T::template product<data_T, data_T>::product(sum_cache, k_inv);
    // std::cout << "mean: " << std::endl;
    // std::cout << mean << std::endl;
    
    for (int i = 0; i < dim; ++i){
        data_diff[i] = data[i] - mean;
        diff = data_diff[i]*data_diff[i];
        sum_cache2 += diff;
        // std::cout << "data_diff: " << std::endl;
        // std::cout << data_diff[i] << std::endl;
        // std::cout << " " << std::endl;
    }
    var = CONFIG_T::template product<data_T, data_T>::product(sum_cache2, k_inv);
    // std::cout << "var: " << std::endl;
    // std::cout << var << std::endl;
    // std::cout << " " << std::endl;

    int index = var*(CONFIG_T::table_size)*inv_range_inv;
	if (index < 0)   index = 0;
	if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
	deno_inver = (typename CONFIG_T::table_t) invert_sqr_table[index];
    // std::cout << "deno_inver: " << std::endl;
    // std::cout << deno_inver << std::endl;
    // std::cout << " " << std::endl;


    for (int i = 0; i < dim; ++i){
        res[i] = data_diff[i] * deno_inver * scale[i] + bias[i];
    }

}


template<class data_T, class res_T, typename CONFIG_T>
void layernormalize(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_in],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_scale_bias],
    typename CONFIG_T::bias_t   bias[CONFIG_T::n_scale_bias]
)
{
    data_T cache;
    static const unsigned dim = CONFIG_T::n_in/CONFIG_T::seq_len;

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,bias

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    #pragma HLS ARRAY_PARTITION variable=scale complete
    #pragma HLS ARRAY_PARTITION variable=bias complete

    for (int j=0; j <CONFIG_T::seq_len; ++j){
        layernorm_1d<data_T, res_T, CONFIG_T>(data+(dim*j), res+(dim*j), scale, bias);
    }


}

}

#endif
