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
#pragma HLS function_instantiate variable=weights,biases

    // Pack the weight/index in a single element
//#pragma HLS data_pack variable=weights struct_level

    typename CONFIG_T::compressed_weight_t mult[CONFIG_T::n_nonzeros];
//#pragma HLS data_pack variable=mult struct_level

    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases

    printf("INFO: compute_compressed_layer\n");
    printf("INFO:   n_in: %u\n", CONFIG_T::n_in);
    printf("INFO:   n_out: %u\n", CONFIG_T::n_out);
    printf("INFO:   reuse_factor: %u\n", CONFIG_T::reuse_factor);
    printf("INFO:   n_zeros: %u\n", CONFIG_T::n_zeros);
    printf("INFO:   n_nonzeros: %u\n", CONFIG_T::n_nonzeros);
    printf("INFO:   store_weights_in_bram: %u\n", CONFIG_T::store_weights_in_bram);

    if (CONFIG_T::io_type == io_parallel) {
    	printf("INFO:   io_type: io_parallel\n");

        // For parallel inputs:
        //   - completely partition arrays
        //   - if we have a reuse factor, limit number of multipliers
#pragma HLS ARRAY_PARTITION variable=biases complete
#pragma HLS ARRAY_PARTITION variable=mult complete
#pragma HLS ARRAY_PARTITION variable=acc complete

#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        unsigned multiplier_limit  = ceil(float(CONFIG_T::n_nonzeros) / float(CONFIG_T::reuse_factor));
#pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

         if (CONFIG_T::store_weights_in_bram) {
             unsigned block_size = CONFIG_T::n_nonzeros / CONFIG_T::reuse_factor;
#pragma HLS ARRAY_RESHAPE variable=weights block factor=block_size
#pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
         } else {
#pragma HLS ARRAY_PARTITION variable=weights complete
         }

    } else if (CONFIG_T::io_type == io_serial) {
    	printf("INFO:   io_type: io_serial\n");

//        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
//        // Otherwise, HLS wont be happy
//        int cycle_factor = CONFIG_T::n_out;
//        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
//        if (reused_cycle == ceil(reused_cycle)){
//            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
//            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
//        }

#pragma HLS ARRAY_PARTITION variable=biases complete
#pragma HLS ARRAY_PARTITION variable=acc complete


//#pragma HLS ARRAY_PARTITION variable=weights cyclic factor=cycle_factor
////#pragma HLS ARRAY_PARTITION variable=mult cyclic factor=cycle_factor
//#pragma HLS ARRAY_PARTITION variable=biases complete
//#pragma HLS ARRAY_PARTITION variable=acc complete
////#pragma HLS STREAM variable=mult depth=1
////#pragma HLS STREAM variable=acc depth=1
        if (CONFIG_T::store_weights_in_bram) {
#pragma HLS RESOURCE variable=weights core=ROM_2P_BRAM
#pragma HLS RESOURCE variable=mul core=ROM_2P_BRAM
        }
////#pragma HLS DATAFLOW
    }

    // Do the compressed matrix-multiply
COMPRESSED_MAT_MULT_L:
	for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
        if (CONFIG_T::io_type == io_serial){
#pragma HLS PIPELINE
        }

        // TODO: remove this division
		unsigned j = weights[i].index / CONFIG_T::n_out;
        mult[i].index = weights[i].index;
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

COMPRESSED_ACCUMULATOR_L:
    for(unsigned i = 0; i < CONFIG_T::n_nonzeros; i++) {
        if (CONFIG_T::io_type == io_serial){
#pragma HLS PIPELINE
        }

        unsigned j = mult[i].index % CONFIG_T::n_out;
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
