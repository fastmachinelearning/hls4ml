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

// Pre-processor time computation. This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)

// Pre-processor time computation.
#define MULTIPLIER_LIMIT(in,out,zeros,reuse) ((((in*out)+reuse-1)/reuse) - (zeros/reuse))

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
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // Replace expression with a macro.
        int multiplier_limit = MULTIPLIER_LIMIT(CONFIG_T::n_in, CONFIG_T::n_out,  CONFIG_T::n_zeros, CONFIG_T::reuse_factor);
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
#if 0
        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
#else
#pragma HLS ARRAY_RESHAPE variable=biases complete
#pragma HLS ARRAY_RESHAPE variable=acc complete

#pragma HLS ARRAY_RESHAPE variable=weights block factor=32
#pragma HLS ARRAY_RESHAPE variable=mult block factor=32
#endif
    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        //int cycle_factor = CONFIG_T::n_out;
        //float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        //if (reused_cycle == ceil(reused_cycle)){
        //    // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
        //    cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        //}
        // Remove previous workaround.
        // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault.
        int cycle_factor = DIV_ROUNDUP(CONFIG_T::n_out, CONFIG_T::reuse_factor);
        #pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
        #pragma HLS ARRAY_PARTITION variable=acc complete
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram){
            #pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
        }
    }
    
    // Do the matrix-multiply
    Product11: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product21: for(int jj = 0; jj < CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product12: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product22: for(int jj = CONFIG_T::n_out/8; jj < 2*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product13: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product23: for(int jj = 2*CONFIG_T::n_out/8; jj < 3*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product14: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product24: for(int jj = 3*CONFIG_T::n_out/8; jj < 4*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product15: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product25: for(int jj = 4*CONFIG_T::n_out/8; jj < 5*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product16: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product26: for(int jj = 5*CONFIG_T::n_out/8; jj < 6*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product17: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product27: for(int jj = 6*CONFIG_T::n_out/8; jj < 7*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Do the matrix-multiply
    Product18: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        data_T cache = data[ii];
        Product28: for(int jj = 7*CONFIG_T::n_out/8; jj < 8*CONFIG_T::n_out/8; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int index = ii*CONFIG_T::n_out+jj;
	    mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    Accum11: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum21: for(int jj = 0; jj < CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum12: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum22: for(int jj = CONFIG_T::n_out/8; jj < 2*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum13: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum23: for(int jj = 2*CONFIG_T::n_out/8; jj < 3*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum14: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum24: for(int jj = 3*CONFIG_T::n_out/8; jj < 4*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum15: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum25: for(int jj = 4*CONFIG_T::n_out/8; jj < 5*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum16: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum26: for(int jj = 5*CONFIG_T::n_out/8; jj < 6*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum17: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum27: for(int jj = 6*CONFIG_T::n_out/8; jj < 7*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Accumulate multiplication result
    Accum18: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum28: for(int jj = 7*CONFIG_T::n_out/8; jj < 8*CONFIG_T::n_out/8; jj++) {
	    int index = ii*CONFIG_T::n_out+jj;
	    acc[jj] += mult[index];
        }
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
