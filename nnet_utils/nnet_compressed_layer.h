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
 template<class data_T, typename CONFIG_T>
   void fillacc(
	       data_T              data[CONFIG_T::n_in],
	       typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
	       typename CONFIG_T::accum_t acc[CONFIG_T::n_out],
	       unsigned ru);

template<class data_T, class res_T, typename CONFIG_T>
void compute_compressed_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
   static const unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
   #pragma HLS function_instantiate variable=weights,biases
   #pragma HLS ARRAY_PARTITION variable=biases complete
   #pragma HLS ARRAY_RESHAPE variable=weights block factor=multiplier_limit   
   
   typename CONFIG_T::accum_t acc[CONFIG_T::n_out];
   #pragma HLS ARRAY_PARTITION variable=acc    complete
   #pragma HLS DEPENDENCE variable=data    inter false
   #pragma HLS DEPENDENCE variable=acc     inter false
   #pragma HLS DEPENDENCE variable=weights inter false

    // Initialize accumulator with input biases
ACCUMULATOR_INIT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
        #pragma HLS UNROLL
        acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }
    int rufactor=CONFIG_T::reuse_factor;
    // Do the compressed matrix-multiply
COMPRESSED_MAT_MULT_L:
    for(unsigned ru = 0; ru < rufactor; ru++) { 
      #pragma HLS PIPELINE ii=1 rewind
     fillacc<data_T,CONFIG_T>(data,weights,acc,ru);
     //PH: Note to self putting fillacc in the loop still doesn't work
    }
RESULT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        #pragma HLS UNROLL
        res[i] = (res_T) (acc[i]);
    }
}

  template<class data_T, typename CONFIG_T>
  void fillacc(
	       data_T              data[CONFIG_T::n_in],
	       typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
	       typename CONFIG_T::accum_t acc[CONFIG_T::n_out],
	       unsigned ru) { 
    #pragma HLS PIPELINE
    const int multlimit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
    int rufactor=CONFIG_T::reuse_factor;
    typename CONFIG_T::accum_t mult[multlimit];
    #pragma HLS ARRAY_PARTITION variable=mult complete
    for(int i = 0; i < multlimit; i++) { 
     #pragma HLS UNROLL
     unsigned w = i*rufactor+ru;
     unsigned j = weights[w].row_index;
     unsigned c = weights[w].col_index;
     typename CONFIG_T::weight_t cache_weight = weights[w].weight;
     typename CONFIG_T::weight_t data_cache = data[j];
     mult[i]  = data_cache * cache_weight;
     acc[c]  += mult[i];
    }
  }

}

#endif
