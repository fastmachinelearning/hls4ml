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

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

// This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)

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
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // For parallel inputs:
    //   - completely partition arrays
    //   - if we have a reuse factor, limit number of multipliers
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Pipelining force all the loops being unrolled
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault
    int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Workaround the above restriction.
    #pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit
    #pragma HLS ARRAY_RESHAPE variable=mult block factor=multiplier_limit
    // #pragma HLS ARRAY_PARTITION variable=weights block factor=multiplier_limit
    // #pragma HLS ARRAY_PARTITION variable=mult    block factor=multiplier_limit

    // Do the matrix-multiply
    // Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {

    //     cache = data[ii];
    //     Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
    //         if (CONFIG_T::io_type == io_serial) {
    //             int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
    //             #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
    //         }
	   //  int index = ii*CONFIG_T::n_out+jj;
	   //  mult[index] = cache * weights[index];
    //     }
    // }

    int rufactor=CONFIG_T::reuse_factor;
    for (int ir = 0; ir < rufactor; ir++){
        #pragma HLS PIPELINE ii=1 rewind

        ///
        for (int im = 0; im < multiplier_limit; im++){
            #pragma UNROLL
            int w_index  = ir * multiplier_limit + im;
            int in_index = w_index / CONFIG_T::n_out;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue;
            printf("w_index = %i, in_index = %i \n", w_index, in_index);
            mult[w_index] = data[in_index] * weights[w_index];
        }
        ///

    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS UNROLL 
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            #pragma HLS UNROLL
	    	int index = ii*CONFIG_T::n_out+jj;
	    	acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL
        res[ires] = (res_T) (acc[ires]);
    }    
}

}

#endif
