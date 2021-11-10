#ifndef NNET_EMBED_H_
#define NNET_EMBED_H_

#include "nnet_common.h"
#include "nnet_helpers.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct embed_config
{
    // Internal data type definitions
    typedef float weight_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 16;
    static const unsigned vocab_size = 50;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

 template<class data_T, class res_T, typename CONFIG_T>
   void embedding(
	      data_T    data[CONFIG_T::n_in],
	      res_T     res[CONFIG_T::n_in*CONFIG_T::n_out],
	      typename CONFIG_T::weight_t  weights[CONFIG_T::vocab_size*CONFIG_T::n_out])
 {
   // copy over the corresponding row in the weights lookup table
   for (int j = 0; j < CONFIG_T::n_in; j++) {
     for (int i = 0; i < CONFIG_T::n_out; i++) {
       #pragma HLS UNROLL
       res[j * CONFIG_T::n_out + i] = (res_T) weights[data[j] * CONFIG_T::n_out + i];
     }
   }
 }

}

#endif
