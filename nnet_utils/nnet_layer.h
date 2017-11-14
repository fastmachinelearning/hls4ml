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
    // partitioning arrays cyclically to go with roll factors?
};

template<class res_t, typename CONFIG_T>
void accumulator2D(typename CONFIG_T::acc_t mult[CONFIG_T::n_in][CONFIG_T::n_out],
                 typename CONFIG_T::bias_t biases[CONFIG_T::n_out],
                 res_t res[CONFIG_T::n_out]){

    typename CONFIG_T::acc_t acc[CONFIG_T::n_out];

    if (CONFIG_T::io_type == io_parallel){
        #pragma HLS PIPELINE
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
    }

    // Initialize with the biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::acc_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            acc[jj] += mult[ii][jj];
        }
    }

    // Cast to res_t
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
    data_T cache;
    typename CONFIG_T::acc_t mult[CONFIG_T::n_in][CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE
        #pragma HLS ARRAY_PARTITION variable=weights complete
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        if (CONFIG_T::reuse_factor > 1) {
            int multiplier_limit  = ceil(CONFIG_T::n_in*CONFIG_T::n_out / CONFIG_T::reuse_factor);
            #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
        }
    } else if (CONFIG_T::io_type == io_serial){
        // TODO: Fill out the directives for serial input
    }

    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            mult[ii][jj] = cache * weights[ii][jj];
        }
    }

    accumulator2D<res_T, CONFIG_T>(mult, biases, res);
}

}

#endif
