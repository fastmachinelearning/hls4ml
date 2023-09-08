
#ifndef NNET_DENSE_LATENCY_H_
#define NNET_DENSE_LATENCY_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_helpers.h"
#include "ac_channel.h"
#include <math.h>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void dense_latency(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
  // For Catapult, add an extra scope so that we can apply the pipeline pragma as if it applied to the function
  constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;
  #pragma hls_pipeline_init_interval ce_reuse_factor
  #pragma hls_preserve_loop yes
  #pragma hls_unroll //yet to finalize on this 

  do {
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    //#pragma HLS function_instantiate variable=weights,biases

    // For parallel inputs:
    //   - completely partition arrays -- target fabric
    //   - if we have an unroll factor, limit number of multipliers
    //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    // //#pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
    //#pragma HLS ARRAY_PARTITION variable=biases complete
    //#pragma HLS ARRAY_PARTITION variable=mult complete
    //#pragma HLS ARRAY_PARTITION variable=acc complete

    //int multiplier_limit  = ceil(float(CONFIG_T::n_in*CONFIG_T::n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::n_zeros) / float(CONFIG_T::reuse_factor));
    constexpr int multiplier_limit  = ((CONFIG_T::n_in*CONFIG_T::n_out)/CONFIG_T::reuse_factor) - CONFIG_T::n_zeros / CONFIG_T::reuse_factor;
    CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::limit(multiplier_limit);

    // Do the matrix-multiply
    #pragma hls_unroll
    Product1: for (unsigned int ii = 0; ii < CONFIG_T::n_in; ii++) {
        cache = data[ii];
        #pragma hls_unroll
        Product2: for (unsigned int jj = 0; jj < CONFIG_T::n_out; jj++) {
        int index = ii*CONFIG_T::n_out+jj;
        mult[index] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(cache, weights[index]);
        }
    }

    // Initialize accumulator with input biases
    #pragma hls_unroll
    ResetAccum: for (unsigned int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        acc[iacc] = (typename CONFIG_T::accum_t) biases[iacc];
    }

    // Accumulate multiplication result
    #pragma hls_unroll
    Accum1: for (unsigned int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma hls_unroll
        Accum2: for (unsigned int jj = 0; jj < CONFIG_T::n_out; jj++) {
        int index = ii*CONFIG_T::n_out+jj;
        acc[jj] += mult[index];
        }
    }

    // Cast to "res_t" type
    #pragma hls_unroll
    Result: for (unsigned int ires = 0; ires < CONFIG_T::n_out; ires++){
        //res[ires] = (res_T) (acc[ires]);
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
  } while (false); // one iteration loop
}

} // end namespace

#endif
