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
#define MATVECSIZE 64

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

// function declarations
template<class data_T, class res_T, typename CONFIG_T>
void matvec_op(
    data_T    data[CONFIG_T::n_in],
    res_T     acc_tmp[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    int       index_offset);

 template<class data_T, class res_T, typename CONFIG_T>
void compute_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    
    // Pipelining force all the loops being unrolled
    // #pragma HLS DATAFLOW
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor 

    // typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=biases
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // break up the matrix-vector operation into small units
    // const int MatVecSize   = 256;
    const int NMatVecUnits = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out,MATVECSIZE);
    typename CONFIG_T::weight_t weight_units [NMatVecUnits][MATVECSIZE];
    typename CONFIG_T::accum_t  acc_units    [NMatVecUnits][CONFIG_T::n_out];
    // #pragma HLS ARRAY_PARTITION variable=acc_units complete dim=1

    // build weight units
    RemapWeights: for (int imv = 0; imv < NMatVecUnits; imv++){
        // break up weight units
        for (int iu = 0; iu < MATVECSIZE; iu++){
            int w_index = imv * NMatVecUnits + iu;
            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out){
                weight_units[imv][iu] = 0.;
            } else{
                weight_units[imv][iu] = weights[w_index];
            }
        }
        // initialize acc_units
        for (int io = 0; io < CONFIG_T::n_out; io++){
            acc_units[imv][io] = 0.;
        }   
    }

    // #pragma HLS DEPENDENCE variable=weight_units inter false
    // #pragma HLS DEPENDENCE variable=acc_units inter false
    MatVecOp: for (int imv = 0; imv < NMatVecUnits; imv++){
        matvec_op<data_T,res_T,CONFIG_T>(data,acc_units[imv],weight_units[imv],imv*MATVECSIZE);
    }

    #pragma HLS DEPENDENCE variable=acc inter false
    AccumAccum: for (int imv = 0; imv < NMatVecUnits; imv++){
        for (int io = 0; io < CONFIG_T::n_out; io++){
            printf("acc_unit[%i] = %0.4f ", io, (float) acc_units[imv][io]);
            acc[io] += acc_units[imv][io];
        }   
    }
    printf("\n");

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        // #pragma HLS UNROLL
        // printf("acc[%i] = %0.4f ", ires, (float) acc[ires]);
        res[ires] = (res_T) (acc[ires]);
    }    
    // printf("\n");
}

////////////////////////////////////////////////////////////////////////
template<class data_T, class res_T, typename CONFIG_T>
void matvec_op(
    data_T    data[CONFIG_T::n_in],
    res_T     acc_tmp[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    int       index_offset)
{
    // printf("in matvec unit func \n");
	// #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // ------- accumulated outside --------
    // int rufactor=CONFIG_T::reuse_factor;
    const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor);
    // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    typename CONFIG_T::accum_t mult[MATVECSIZE];
    #pragma HLS ARRAY_RESHAPE variable=mult complete // block factor=multiplier_limit

    #pragma HLS function_instantiate variable=weights
    #pragma HLS ARRAY_RESHAPE variable=weights complete // block factor=multiplier_limit
    #pragma HLS function_instantiate variable=acc_tmp
    #pragma HLS ARRAY_RESHAPE variable=acc_tmp complete
    // #pragma HLS DEPENDENCE variable=acc_tmp inter false

    // multiply
    MatVecOpMain: for (int ii = 0; ii < MATVECSIZE; ii++){
        int glob_index = ii + index_offset;
        int in_index  = glob_index / CONFIG_T::n_out;
        int out_index = glob_index % CONFIG_T::n_out;   
       
        if (glob_index >= CONFIG_T::n_in*CONFIG_T::n_out) break; // check out of bounds 
        mult[ii] = data[in_index] * weights[ii];
        printf("glob_index = %i, in_index = %i, out_index = %i, mult = %0.4f, data = %0.4f, weights = %0.4f \n", glob_index, in_index, out_index, (float) mult[ii], (float) data[in_index], (float) weights[ii]);
    }
    printf("\n");

    // accumulate
    MatVecOpAccum: for (int ii = 0; ii < MATVECSIZE; ii++){
        int glob_index = ii + index_offset;
        int out_index = glob_index % CONFIG_T::n_out;   
        if (glob_index >= CONFIG_T::n_in*CONFIG_T::n_out) break; // check out of bounds 
        acc_tmp[out_index] += mult[ii];
    }    

    // for (int ii = 0; ii < MATVECSIZE; ii++){
    //     int glob_index = ii + index_offset;
    //     int in_index  = glob_index / CONFIG_T::n_out;
    //     int out_index = glob_index % CONFIG_T::n_out;   
       
    //     if (glob_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds 
    //     mult[ii] = data[in_index] * weights[ii];
    // }
	// #pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit
	// #pragma HLS ARRAY_RESHAPE variable=mult block factor=multiplier_limit
 //    #pragma HLS ARRAY_PARTITION variable=acc complete

 //    ReuseLoop: for (int ir = 0; ir < rufactor; ir++){
 //        //#pragma HLS PIPELINE ii=1 rewind
	//     // #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
 //    	#pragma UNROLL
 //        ///
 //        // printf("on the clock tick \n");
 //        MultLoop1: for (int im = 0; im < multiplier_limit; im++){
 //            #pragma UNROLL
            
 //            int w_index   = ir + rufactor * im;
 //            int in_index  = w_index / CONFIG_T::n_out;
 //            int out_index = w_index % CONFIG_T::n_out;

 //            if (w_index >= CONFIG_T::n_in*CONFIG_T::n_out) continue; // check out of bounds

 //            // printf("ir = %i, im = %i, w_index = %i, in_index = %i, out_index = %i \n", ir, im, w_index, in_index, out_index);
 //            mult[w_index] = data[in_index] * weights[w_index];
 //        }
 //        ///
 //    }

 //    // Accumulate multiplication result
 //    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
	// 	// #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
 //        #pragma HLS UNROLL 
 //        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
 //            // #pragma HLS UNROLL
	//     	int index = ii*CONFIG_T::n_out+jj;
	//     	acc[jj] += mult[index];
 //        }
    // }

}
////////////////////////////////////////////////////////////////////////

}

#endif
