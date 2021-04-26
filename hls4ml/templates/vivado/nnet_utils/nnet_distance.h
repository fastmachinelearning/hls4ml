#ifndef NNET_DISTANCE_H_
#define NNET_DISTANCE_H_

#include "nnet_common.h"
#include "nnet_activation.h"
#include <cstdlib>

namespace nnet {

struct distance_config
{
    // IO size
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    
    // Internal data type definitions
    typedef float accum_t;
    typedef float sum_t;
    typedef ap_fixed<18,8> exp_table_t;

    // Internal info
    static const unsigned table_size = 1024;
};

template<class data1_T, class data2_T, class res_T, typename CONFIG_T>
void klloss(
    data1_T mean[CONFIG_T::n_in],
    data2_T log_var[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out]
) {
    #pragma HLS PIPELINE
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_exp_table<data2_T, CONFIG_T>(exp_table);
        initialized = true;
    }

    typename CONFIG_T::accum_t kl[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=kl complete

    typename CONFIG_T::accum_t mean_sq[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=mean_sq complete

    typename CONFIG_T::sum_t kl_sum(0);

    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
    	mean_sq[i] = mean[i] * mean[i];
    }

    for (unsigned i = 0; i < CONFIG_T::n_in; i++) {
        #pragma HLS UNROLL
        unsigned x = softmax_idx_from_real_val<data2_T, CONFIG_T>(log_var[i]);
        kl[i] = data2_T(1.) + log_var[i] - mean_sq[i] - exp_table[x];
    }

    Op_add<typename CONFIG_T::accum_t> op_add;
    kl_sum = reduce<typename CONFIG_T::accum_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::accum_t>>(kl, op_add);

    res[0] = res_T(-0.5) * kl_sum;
}

template<class data_T, class res_T, typename CONFIG_T>
void radius(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_out]
) {
    #pragma HLS PIPELINE
    // Initialize the lookup tables
#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::table_size];
    static typename CONFIG_T::inv_table_t invert_table[CONFIG_T::table_size];

#endif
    if (!initialized) {
        // Note we are exponentiating the inputs, which have type data_T
        init_exp_table<data_T, CONFIG_T>(exp_table);
        // Note we are inverting the exponentials, which have type exp_table_t
        init_invert_table<typename CONFIG_T::exp_table_t, CONFIG_T>(invert_table);
        initialized = true;
    }

    // Not implemented
}

}//end namespace

#endif
