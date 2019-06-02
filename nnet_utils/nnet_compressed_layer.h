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
    void fillMult(typename CONFIG_T::index_t aindex,
		  typename CONFIG_T::accum_t *acc,
		  typename CONFIG_T::accum_t weight);
template<class data_T, class res_T, typename CONFIG_T>
void compute_compressed_layer(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::compressed_weight_t  weights[CONFIG_T::n_nonzeros],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{

    int multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);
    static const int multiplier_limits = DIV_ROUNDUP(CONFIG_T::n_nonzeros, CONFIG_T::reuse_factor);

    typename CONFIG_T::accum_t acc [CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=acc    complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=multiplier_limit
    //if (CONFIG_T::store_weights_in_bram){
    //#pragma HLS RESOURCE variable=weights core=ROM_1P_BRAM
    #pragma HLS data_pack variable=weights struct_level 
    //}
    ACCUMULATOR_INIT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++) {
      #pragma HLS UNROLL
      acc[i] = (typename CONFIG_T::accum_t) (biases[i]);
    }
    // Do the compressed matrix-multiply
    const int rufactor = CONFIG_T::reuse_factor;
    COMPRESSED_MAT_MULT_L:
    for(unsigned ru = 0; ru < rufactor; ru++) {
      #pragma HLS PIPELINE  II=1 rewind
      typename CONFIG_T::accum_t  tmpmult[multiplier_limits];
      #pragma HLS ARRAY_PARTITION variable=tmpmult   complete

      typename CONFIG_T::index_t  tmpindx[multiplier_limits];
      #pragma HLS ARRAY_PARTITION variable=tmpindx   complete

      for(unsigned i = 0; i < multiplier_limit; i++) { 
        #pragma HLS UNROLL 
        unsigned w = i*rufactor + ru; //PH CHeck me
	typename CONFIG_T::index_t  j = weights[w].row_index;
	typename CONFIG_T::index_t  c = weights[w].col_index;
	typename CONFIG_T::weight_t cache = weights[w].weight;
	typename CONFIG_T::accum_t  data_cache = data[j];
	tmpmult[i]            = (cache)*data_cache;
	tmpindx[i]            = c;
      }
      typename CONFIG_T::accum_t mult[CONFIG_T::n_out];
      #pragma HLS ARRAY_PARTITION variable=mult complete
      ResetMult: for(int imult = 0; imult < CONFIG_T::n_out; imult++) {
            #pragma HLS UNROLL
            mult[imult] = 0;                                                                                                                                                                                                                                                 
      }
      for(unsigned ifm = 0; ifm < multiplier_limit; ifm++) { 
        #pragma HLS UNROLL 
        typename CONFIG_T::weight_t cache = tmpmult[ifm];
        typename CONFIG_T::index_t a      = tmpindx[ifm];
	//mult[a] += cache;
	fillMult<CONFIG_T>(a,mult,cache);
      }
      for (int im = 0; im < CONFIG_T::n_out; im++){
       acc[im] += mult[im];
      }
    }
    // Cast to "res_t" type
    RESULT_L:
    for(unsigned i = 0; i < CONFIG_T::n_out; i++){
        #pragma HLS UNROLL
        res[i] = (res_T) (acc[i]);
    }
}
template<typename CONFIG_T>
void fillMult(typename CONFIG_T::index_t aindex,
    typename CONFIG_T::accum_t *mult,
    typename CONFIG_T::accum_t weight) { 
  for(unsigned  k = 0; k < CONFIG_T::n_out; k++) { 
   #pragma HLS UNROLL
    if(k==aindex) mult[k] += weight;
  }
 }

}

#endif
