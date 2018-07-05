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
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct batchnorm_config
{
    // Internal data type definitions
    typedef float beta_t;
    typedef float scale_t;
    typedef float mean_t;
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
void normalize(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::scale_t  scale[CONFIG_T::n_in],
    typename CONFIG_T::beta_t   beta[CONFIG_T::n_in],
    typename CONFIG_T::mean_t   mean[CONFIG_T::n_in])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_in];
    typename CONFIG_T::accum_t shift[CONFIG_T::n_in];

    
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=scale,beta,mean

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
	#pragma HLS ARRAY_PARTITION variable=scale complete
        #pragma HLS ARRAY_PARTITION variable=beta complete
	#pragma HLS ARRAY_PARTITION variable=mean complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS ARRAY_PARTITION variable=shift complete

        int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS ARRAY_RESHAPE variable=scale complete dim=2
        #pragma HLS ARRAY_RESHAPE variable=beta complete dim=2
        #pragma HLS ARRAY_RESHAPE variable=mean complete dim=2
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=2
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        #pragma HLS ARRAY_PARTITION variable=shift complete dim=1
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
    }

    // Shift input data by the mean
    ShiftInputs: for(int i = 0; i < CONFIG_T::n_in; i++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        shift[i] = (typename CONFIG_T::accum_t) (data[i]-mean[i]);
    }
        
    // Do the multiplication
    Product: for(int i = 0; i < CONFIG_T::n_in; i++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        mult[i] = scale[i] * shift[i];
    }    
    
    // Initialize accumulator with beta values
    ResetAccum: for(int i = 0; i < CONFIG_T::n_out; i++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[i] = (typename CONFIG_T::accum_t) beta[i];
    }

    // Accumulate multiplication result
    Accum: for(int i = 0; i < CONFIG_T::n_in; i++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        acc[i] += mult[i];
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }              
       
}

}

#endif
