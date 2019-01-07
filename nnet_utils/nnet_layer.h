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

    const int cycle_factor_loop = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    typename CONFIG_T::accum_t tmpmult[cycle_factor_loop];
    #pragma HLS ARRAY_PARTITION variable=tmpmult complete

    if (CONFIG_T::io_type == io_parallel){

        /*
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=acc complete

        // Pipelining force all the loops being unrolled
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault
        int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

        // Workaround the above restriction.
        #pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit
        #pragma HLS ARRAY_RESHAPE variable=mult block factor=multiplier_limit 
        */

        // the same for all io_parallel and io_serial
        // typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
        // #pragma HLS function_instantiate variable=weights,biases
        
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind
        // const int cycle_factor_loop = (CONFIG_T::n_in*CONFIG_T::n_out)/CONFIG_T::reuse_factor;
        
        //#pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
        #pragma HLS RESOURCE variable=weights core=ROM_1P_LUTRAM
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=acc    complete
        #pragma HLS ARRAY_RESHAPE variable=weights block factor=cycle_factor_loop
        //float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        //#pragma HLS ARRAY_RESHAPE variable=biases block factor=reuse_cycle
        //#pragma HLS ARRAY_RESHAPE variable=acc    block factor=reuse_cycle
        
        // phil says you don't need this
        // int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
        int multiplier_limit = cycle_factor_loop;
        // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
        
    } else if (CONFIG_T::io_type == io_serial){
        // Only reduce cycle_factor if n_out is evenly divisible by reuse_factor
        // Otherwise, HLS wont be happy
        int cycle_factor = CONFIG_T::n_out;
        float reused_cycle = CONFIG_T::n_out / CONFIG_T::reuse_factor;
        if (reused_cycle == ceil(reused_cycle)){
            // Dont use "ceil" here; as of 2018.2, HLS crashes mysteriously
            cycle_factor = cycle_factor / CONFIG_T::reuse_factor;
        }
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
    
    // // Do the matrix-multiply
    // Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
    //     if (CONFIG_T::io_type == io_serial){
    //         #pragma HLS PIPELINE
    //     }
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

    // // Initialize accumulator with input biases
    // ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
    //     if (CONFIG_T::io_type == io_serial){
    //         #pragma HLS UNROLL
    //     }
    //     acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    // }

    // // Accumulate multiplication result
    // Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
    //     if (CONFIG_T::io_type == io_serial){
    //         #pragma HLS PIPELINE
    //     }
    //     Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
	   //  int index = ii*CONFIG_T::n_out+jj;
	   //  acc[jj] += mult[index];
    //     }
    // }

    // // Cast to "res_t" type
    // Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
    //     if (CONFIG_T::io_type == io_serial){
    //         #pragma HLS UNROLL
    //     }
    //     res[ires] = (res_T) (acc[ires]);
    // }    

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }
    ResetMult: for(int iacc = 0; iacc < cycle_factor_loop; iacc++) {
        #pragma HLS UNROLL
        tmpmult[iacc] = 0;
    }
    const int rufactor=CONFIG_T::reuse_factor;
    /* => One alternative to deal with zero supp
    Setup1: for(int ii = 0; ii < rufactor; ii++) {
    #pragma HLS PIPELINE
    Setup0: for(int jj = 0; jj < cycle_factor; jj++) {
     #pragma HLS UNROLL
     int windex = ii+jj*rufactor;
     int index   = windex/CONFIG_T::n_out;    
     tmpmult[windex] = data[index];
     }
    }*/
    //#pragma HLS LOOP_FLATTEN off
    Product1: for(int ii = 0; ii < rufactor; ii++) {
        #pragma HLS PIPELINE II=1
        Product0: for(int jj = 0; jj < cycle_factor_loop; jj++) {
            #pragma HLS UNROLL
            int windex = ii+jj*rufactor;
            int index   = windex/CONFIG_T::n_out;
            tmpmult[jj] += data[index]*weights[windex];
            //tmpmult[windex] = data[index]*weights[windex]; //See above
      }
    }
    
    for(int ii = 0; ii < cycle_factor_loop; ii++) {  
        #pragma HLS UNROLL
       int aindex  = (ii*rufactor)/CONFIG_T::n_in;
       acc[aindex] += tmpmult[ii];
    }
    /* Accumulate multiplication result See above alternative
    Accum1: for(int ii = 0; ii < rufactor; ii++) {
      #pragma HLS PIPELINE 
    Accum0: for(int jj = 0; jj < cycle_factor_loop; jj++) { 
     #pragma HLS UNROLL
        //int windex=jj*rufactor+ii;
    int index=windex/CONFIG_T::n_in;
    acc[index] += tmpmult[windex];
      }    
    }
    */
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL
        res[ires] = (res_T) (acc[ires]);
    }    


}

}

#endif
