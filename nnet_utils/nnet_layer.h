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
// #define ADD_LAT 5

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

// begin function declarations
// template<class data_T, typename CONFIG_T>
// void matvec_op(
//     data_T                       data[CONFIG_T::n_in],
//     typename CONFIG_T::accum_t   acc[CONFIG_T::n_out],
//     typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
//     int                          reuse_index);
// template<class data_T, typename CONFIG_T>
// void matvec_op(
//     data_T                       *data,
//     typename CONFIG_T::accum_t   *acc,
//     typename CONFIG_T::weight_t  *weights,
//     int                          reuse_index);
// begin function declarations

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{

    // Pipelining force all the loops being unrolled
    // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // Replace ceil function with home-made macro prevents Vivado 2018.2 segfault
    const int totals_multipliers = CONFIG_T::n_in*CONFIG_T::n_out;
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Workaround the above restriction.
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    // #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS RESOURCE        variable=weights core=ROM_nP_LUTRAM
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=multiplier_limit
    #pragma HLS ARRAY_PARTITION variable=biases complete
    
    // typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    // #pragma HLS ARRAY_RESHAPE variable=mult    block factor=multiplier_limit
    #pragma HLS ARRAY_PARTITION variable=acc complete
    // #pragma HLS DEPENDENCE variable=acc,weights,biases inter false
    #pragma HLS DEPENDENCE variable=acc inter false

    // typename CONFIG_T::accum_t acc_tmp[CONFIG_T::n_out];
    // #pragma HLS ARRAY_PARTITION variable=acc_tmp complete
    // #pragma HLS DEPENDENCE variable=acc_tmp inter false

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // core functionality
    int rufactor=CONFIG_T::reuse_factor;
    const int ADD_LAT = DIV_ROUNDUP(multiplier_limit,CONFIG_T::n_out);
    ReuseLoop: for (int ir = 0; ir < rufactor; ir++){

        #pragma HLS PIPELINE II=1 rewind
        ///////// --------------------------------------
        // a tmp mult for each reuse loop iteration
        typename CONFIG_T::accum_t mult[multiplier_limit];
        #pragma HLS ARRAY_PARTITION variable=mult complete

        MultLoop: 
        for (int im = 0; im < multiplier_limit; im++){
            int w_index   = ir + rufactor * im;
            int in_index  = w_index / CONFIG_T::n_out;
            int out_index = w_index % CONFIG_T::n_out;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
            mult[im] = data[in_index] * weights[w_index];
        }

        // special loop for accumulation
        typename CONFIG_T::accum_t acc_lat[CONFIG_T::n_out][ADD_LAT];
        #pragma HLS ARRAY_PARTITION variable=acc_lat complete dim=0
        #pragma HLS DEPENDENCE variable=acc_lat inter false

        AddLatencyInit: 
        for (int ii = 0; ii < CONFIG_T::n_out; ii++){
            for (int ij= 0; ij < ADD_LAT; ij++){
                #pragma UNROLL
                acc_lat[ii][ij] = 0;
            }
        }

        AccumLoop:
        for (int im = 0; im < multiplier_limit; im += ADD_LAT){
            #pragma UNROLL
            for (int il = 0; il < ADD_LAT; il++){
                int w_index   = ir + rufactor * (im+il);
                int out_index = w_index % CONFIG_T::n_out;
                if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
                acc_lat[out_index][il] += mult[im+il];
            }
        }
        // printf("\n");

        FullAccum: 
        for (int ii = 0; ii < CONFIG_T::n_out; ii++){
            for (int ij= 0; ij < ADD_LAT; ij++){
                #pragma UNROLL
                acc[ii] += acc_lat[ii][ij];
            }
        }    
        ///////// --------------------------------------

    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        #pragma HLS UNROLL
        printf("acc[%i] = %0.4f ", ires, (float) acc[ires]);
        res[ires] = (res_T) (acc[ires]);
    }    
    printf("\n");

}

// template<class data_T, typename CONFIG_T>
// void matvec_op(
//     data_T                       data[CONFIG_T::n_in],
//     typename CONFIG_T::accum_t   acc[CONFIG_T::n_out],
//     typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
//     int                          reuse_index){
// // template<class data_T, typename CONFIG_T>
// // void matvec_op(
// //     data_T                       *data,
// //     typename CONFIG_T::accum_t   *acc,
// //     typename CONFIG_T::weight_t  *weights,
// //     int                          reuse_index){
//     // #pragma HLS PIPELINE II=1 
//     // #pragma HLS ALLOCATION instances=FIFO_BRAM limit=0 core
//     // #pragma HLS ALLOCATION instances=RAM_1P_BRAM limit=0 core
//     // #pragma HLS ALLOCATION instances=RAM_2P_BRAM limit=0 core

//     int rufactor=CONFIG_T::reuse_factor;
//     const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
//     typename CONFIG_T::accum_t mult[multiplier_limit];
//     #pragma HLS ARRAY_PARTITION variable=mult complete

//     MultLoop: 
//     for (int im = 0; im < multiplier_limit; im++){
//         // #pragma UNROLL
        
//         int w_index   = reuse_index + rufactor * im;
//         int in_index  = w_index / CONFIG_T::n_out;
//         int out_index = w_index % CONFIG_T::n_out;

//         if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
//         printf("ir = %i, im = %i, w_index = %i, in_index = %i, out_index = %i, data = %0.4f, weight = %0.4f \n", reuse_index, im, w_index, in_index, out_index, (float) data[in_index], (float) weights[w_index]);
//         mult[im] = data[in_index] * weights[w_index];
//         // acc[out_index] += mult[im];
//         // acc[out_index] += data[in_index] * weights[w_index];
//     }

//     // special loop for accumulation
//     const int ADD_LAT = DIV_ROUNDUP(multiplier_limit,CONFIG_T::n_out);
//     typename CONFIG_T::accum_t acc_lat[CONFIG_T::n_out][ADD_LAT];
//     #pragma HLS ARRAY_PARTITION variable=acc_lat complete


//     AddLatencyInit: 
//     for (int ii = 0; ii < CONFIG_T::n_out; ii++){
//         for (int ij= 0; ij < ADD_LAT; ij++){
//             #pragma UNROLL
//             acc_lat[ii][ij] = 0;
//         }
//     }

//     AccumLoop:
//     for (int im = 0; im < multiplier_limit; im += ADD_LAT){
//         #pragma UNROLL
        
//         for (int il = 0; il < ADD_LAT; il++){
//             int w_index   = reuse_index + rufactor * (im+il);
//             int out_index = w_index % CONFIG_T::n_out;
//             if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds
//             // printf("ir = %i, im = %i, w_index = %i, in_index = %i, out_index = %i \n", ir, im, w_index, in_index, out_index);
//             // printf("acc[%i] = %0.4f ", out_index, (float) acc[out_index]);
//             acc_lat[out_index][il] += mult[im+il];
//         }
//     }
//     // printf("\n");

//     FullAccum: 
//     for (int ii = 0; ii < CONFIG_T::n_out; ii++){
//         for (int ij= 0; ij < ADD_LAT; ij++){
//             #pragma UNROLL
//             acc[ii] += acc_lat[ii][ij];
//         }
//     }    

// }

}

#endif
