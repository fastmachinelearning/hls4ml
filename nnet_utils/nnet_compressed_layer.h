//
//    hls4ml: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2018 Giuseppe Di Guglielmo
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

#ifndef NNET_COMPRESSED_LAYER_H_
#define NNET_COMPRESSED_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

// This is a substitute for "ceil(n/(float)d)".
#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void compute_compressed_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    // Create a separate module for each layer
#pragma HLS inline off

    // Explicitly optimize unchanging weights/biases
#pragma HLS function_instantiate variable=weights,biases

    // Intermediate computational buffers
    typename CONFIG_T::compressed_weight_t mult[CONFIG_T::n_nonzeros];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    printf("INFO: compute_compressed_layer\n");
    printf("INFO:   n_in: %u\n", CONFIG_T::n_in);
    printf("INFO:   n_out: %u\n", CONFIG_T::n_out);
    printf("INFO:   reuse_factor: %u\n", CONFIG_T::reuse_factor);
    printf("INFO:   n_zeros: %u\n", CONFIG_T::n_zeros);
    printf("INFO:   n_nonzeros: %u\n", CONFIG_T::n_nonzeros);
    printf("INFO:   store_weights_in_bram: %u\n", CONFIG_T::store_weights_in_bram);

    // Parallel vs. Serial IO
    if (CONFIG_T::io_type == io_parallel) {
        // For parallel inputs:
        //   - completely partition arrays
        //   - if we have a reuse factor, limit number of multipliers
#pragma HLS ARRAY_PARTITION variable=biases complete
#pragma HLS ARRAY_PARTITION variable=acc complete

        // Pipelining force all the loops being unrolled
#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault
        int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

        // BRAMs vs. registers
        if (CONFIG_T::store_weights_in_bram) {
            // Reshape the arrays as memory with a larger bit-width
#pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit
#pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
#pragma HLS ARRAY_RESHAPE variable=mult block factor=multiplier_limit
#pragma HLS RESOURCE variable=mult core=RAM_2P_BRAM

            // Pack the row_index, col_index, and weight in a single 32-bit memory element
//#pragma HLS data_pack variable=weights struct_level
            // TODO: Packing mult provides worst results (disable it)
//#pragma HLS data_pack variable=mult struct_level
         } else {
             // Completely partition the remaining arrays
#pragma HLS ARRAY_PARTITION variable=weights complete
#pragma HLS ARRAY_PARTITION variable=mult complete
         }

    } else if (CONFIG_T::io_type == io_serial) {
#pragma HLS ARRAY_PARTITION variable=biases complete
#pragma HLS ARRAY_PARTITION variable=acc complete

        // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault
        int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

        if (CONFIG_T::store_weights_in_bram) {
#pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
#pragma HLS RESOURCE variable=mul core=RAM_2P_BRAM

#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=multiplier_limit
#pragma HLS ARRAY_PARTITION variable=mult cyclic factor=multiplier_limit

            // Pack the row_index, col_index, and weight in a single 32-bit memory element
#pragma HLS data_pack variable=weights struct_level
            // TODO: packing the structs in mult array provide worst results.
//#pragma HLS data_pack variable=mult struct_level
        } else {
            // TODO: non tested yet!
#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=multiplier_limit
#pragma HLS ARRAY_PARTITION variable=mult cyclic factor=multiplier_limit
        }
        // TODO: it generates a segfault
#pragma HLS DATAFLOW
    }

    // Do the compressed matrix-multiply
COMPRESSED_MAT_MULT_L:
    for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
        if (CONFIG_T::io_type == io_serial){
#pragma HLS PIPELINE
        }

        unsigned j = weights[i].row_index;
        mult[i].row_index = weights[i].row_index;
        mult[i].col_index = weights[i].col_index;
        mult[i].weight = weights[i].weight * data[j];
    }

    // Initialize accumulator with input biases
ACCUMULATOR_INIT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        if (CONFIG_T::io_type == io_serial){
#pragma HLS UNROLL
        }

        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }

    // Accumulate over the columns
COMPRESSED_ACCUMULATOR_L:
    for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
        if (CONFIG_T::io_type == io_serial){
#pragma HLS PIPELINE
        }

        unsigned j = mult[i].col_index;
        acc[j] += mult[i].weight;
    }

    // Cast to "res_t" type
RESULT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        if (CONFIG_T::io_type == io_serial){
#pragma HLS UNROLL
        }
        res[i] = (res_T) (acc[i]);
    }
}

}

#endif
