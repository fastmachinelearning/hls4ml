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

 template<typename CONFIG_T>
 void fillMult(int aindex,
	       typename CONFIG_T::accum_t* acc,
	       typename CONFIG_T::accum_t weight);

template<class data_T, class res_T, typename CONFIG_T>
void compute_compressed_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
  //#pragma HLS DATAFLOW
    static const int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
    static const int nmult            = DIV_ROUNDUP(multiplier_limit, CONFIG_T::n_out); 
    static const int nmults           = nmult*CONFIG_T::n_out;

    // Intermediate computational buffers
    typename CONFIG_T::accum_t mult[nmults];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
    if (CONFIG_T::store_weights_in_bram) {
     #pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
    }
    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=acc    complete
    #pragma HLS ARRAY_PARTITION variable=mult   complete
    #pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit
    #pragma HLS DEPENDENCE variable=weights inter false
    #pragma HLS DEPENDENCE variable=data    inter false
    #pragma HLS DEPENDENCE variable=acc     inter false
    #pragma HLS DEPENDENCE variable=mult    inter false
    #pragma HLS data_pack variable=weights struct_level

    // Initialize accumulator with input biases
    ACCUMULATOR_INIT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }
    ResetMult: for(int imult = 0; imult < nmults; imult++) {
        #pragma HLS UNROLL
        mult[imult] = 0;
    }
    int rufactor=CONFIG_T::reuse_factor;
    // Do the compressed matrix-multiply
    COMPRESSED_MAT_MULT_L:
    for(unsigned ru = 0; ru < rufactor; ru++) { 
      #pragma HLS PIPELINE II=1 rewind
      for(unsigned i = 0; i < multiplier_limit; i++) {
        #pragma HLS UNROLL
	unsigned w    = i*rufactor+ru;
	if (w >= CONFIG_T::n_in*CONFIG_T::n_out) continue;
        unsigned j    = weights[w].row_index;
	int aindex    = (weights[w].col_index)*nmult+float(j*nmult)/float(CONFIG_T::n_in);
        typename CONFIG_T::accum_t tmpweight = weights[w].weight * data[j];
	fillMult<CONFIG_T>(aindex,mult,tmpweight);
	//fillMult<CONFIG_T>(aindex,acc,tmpweight);
      }
    }
    // Accumulate over the columns
    COMPRESSED_ACCUMULATOR_L:
    for(unsigned m = 0; m < nmults; m++) {
       #pragma HLS UNROLL
       unsigned index = m/nmult;
       acc[index] += mult[m];
       }
    // Cast to "res_t" type
    RESULT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        #pragma HLS UNROLL
        res[i] = (res_T) (acc[i]);
    }
}
 template<typename CONFIG_T>
 void fillMult(int aindex,
	typename CONFIG_T::accum_t *mult,
	typename CONFIG_T::accum_t weight) { 
  int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
  int nmults           = DIV_ROUNDUP(multiplier_limit, CONFIG_T::n_out)*CONFIG_T::n_out; 
  for(int k = 0; k < nmults; k++) { 
   #pragma HLS UNROLL
    if(k==aindex) mult[k] += weight;
  }
 }
}

#endif
