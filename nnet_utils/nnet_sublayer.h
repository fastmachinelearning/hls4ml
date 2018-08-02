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

#ifndef NNET_SUBLAYER_H_
#define NNET_SUBLAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct sublayer_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 10;

    // Number of sublayer partitions
    static const unsigned n_part = 2;
    static const unsigned i_part = 0;

    // Number of outputs; starting index (inclusive)
    static const unsigned n_sub_out = 5;
    static const unsigned i_sub_out = 0;
  
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const unsigned n_zeros = 0;
    // partitioning arrays cyclically to go with roll factors?
};

//Computes multiplier limit
//This function should not be synthesized into firmware
template<typename CONFIG_T>
int compute_multiplier_limit_sublayer(
			     typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out]
)
{
    int n_mult = 0;
    // Do the matrix-multiply                                                                                                                       
    for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
      for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
	int weight_index = ii*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
	//need to tune this cut?
	if( weights[weight_index] > 1e-20 || weights[weight_index] < -1e-20 ){
	  n_mult++;
	}//end if nonzero weight
      }//end output loop
    }//end input loop
    
    return ceil( float(n_mult) / float(CONFIG_T::reuse_factor) );
   
}//end compute_n_mult

 template<class data_T, class res_T, typename CONFIG_T>
void compute_sublayer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_sub_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_sub_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_sub_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=biases complete
        #pragma HLS ARRAY_PARTITION variable=mult complete
        #pragma HLS ARRAY_PARTITION variable=acc complete
  
	const int multiplier_limit = compute_multiplier_limit_sublayer<CONFIG_T>(weights);
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS ARRAY_RESHAPE variable=weights complete dim=1
        #pragma HLS ARRAY_PARTITION variable=mult complete dim=1
        #pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        #pragma HLS DATAFLOW
        #pragma HLS STREAM variable=mult depth=1
        #pragma HLS STREAM variable=acc depth=1
    }
    
    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        cache = data[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
            if (CONFIG_T::io_type == io_serial) {
                int multiplier_limit  = ceil(float(CONFIG_T::n_sub_out) / float(CONFIG_T::reuse_factor));
                #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            }
	    int weight_index = ii*CONFIG_T::n_out+jj+CONFIG_T::i_sub_out;
	    int mult_index   = ii*CONFIG_T::n_sub_out+jj;
	    mult[mult_index] = cache * weights[weight_index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_sub_out; iacc++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
	int bias_index = iacc+CONFIG_T::i_sub_out;
        acc[iacc] = (typename CONFIG_T::accum_t) biases[bias_index];
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS PIPELINE
        }
        Accum2: for(int jj = 0; jj < CONFIG_T::n_sub_out; jj++) {
	    int index = ii*CONFIG_T::n_sub_out+jj;
	    acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_sub_out; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (acc[ires]);
    }    
}

template<class data_T, int NIN1, int NIN2>
    void merge(
        data_T data1[NIN1], 
	data_T data2[NIN2],
        data_T res[NIN1+NIN2])
{
    for(int ii=0; ii<NIN1; ii++){
        res[ii] = data1[ii];
    }
    for(int ii=0; ii<NIN2; ii++){
        res[NIN1+ii] = data2[ii];
    }
}




}//end namespace
  
  
#endif
