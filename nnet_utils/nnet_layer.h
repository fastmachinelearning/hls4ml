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

#ifndef NNET_LAYER_H_
#define NNET_LAYER_H_

#include "nnet_default.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct layer_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Resource reuse info
    static const bool full_parallel = true;
    static const unsigned roll_factor_in = 1;
    static const unsigned roll_factor_out = 1;
    static const bool store_weights_in_bram = false;
    // partitioning arrays cyclically to go with roll factors?
};

template<class res_t, typename CONFIG_T>
void accumulator(typename CONFIG_T::acc_t mult[CONFIG_T::n_in][CONFIG_T::n_out],
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out],
                 res_t res[CONFIG_T::n_out]){

    typename CONFIG_T::acc_t acc[CONFIG_T::n_out];

    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=acc complete
    if (CONFIG_T::full_parallel){
        #pragma HLS PIPELINE
    }
    else {
        #pragma HLS PIPELINE II=2
    }
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::acc_t) biases[iacc];
    }

    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            acc[jj] += mult[ii][jj];
        }
    }

    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        res[ires] = (res_t) (acc[ires]);// + (typename CONFIG_T::acc_t) biases[ires]);
    }    
}

template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in][CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{

    data_T cache[CONFIG_T::n_in];
    typename CONFIG_T::acc_t mult[CONFIG_T::n_in][CONFIG_T::n_out];
    typename CONFIG_T::acc_t acc[CONFIG_T::n_out];

    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::full_parallel){
        #pragma HLS ARRAY_PARTITION variable=weights complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
        #pragma HLS PIPELINE
    }
    else {
        int multiplier_limit  = ceil(CONFIG_T::n_in*CONFIG_T::n_out / 4);
        #pragma HLS ARRAY_PARTITION variable=weights complete dim=0
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
        #pragma HLS ARRAY_PARTITION variable=cache complete
        #pragma HLS DATAFLOW
        // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
    }

    int unroll_factor_in  = CONFIG_T::n_in / 2;
    int unroll_factor_out = CONFIG_T::n_out / 2;
    //int unroll_factor_in  = CONFIG_T::n_in;
    //int unroll_factor_out = CONFIG_T::n_out;

    Input: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS UNROLL
        cache[ii] = data[ii];
    }

    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (!CONFIG_T::full_parallel){
            #pragma HLS UNROLL factor=unroll_factor_in
            #pragma HLS PIPELINE rewind
        }
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            mult[ii][jj] = cache[ii] * weights[ii][jj];
        }
    }

    accumulator<res_T, CONFIG_T>(mult, biases, res);

}

}

#endif
