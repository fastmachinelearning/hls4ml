#ifndef NNET_DISTANCE_H_
#define NNET_DISTANCE_H_

#include "nnet_common.h"
#include "nnet_activation.h"
#include <cstdlib>
#include <cmath>

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

struct mse_config{
    // IO size
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    
    // Internal data type definitions
    typedef float error_t;
    typedef float squared_error_t;
    typedef float accum_t;
};

template<class data1_T, class data2_T, class res_T, typename CONFIG_T>
void mse(data1_T a[CONFIG_T::n_in],
         data2_T b[CONFIG_T::n_in],
         res_T   res[CONFIG_T::n_out]){
    #pragma HLS pipeline
    typename CONFIG_T::error_t error[CONFIG_T::n_in];
    typename CONFIG_T::squared_error_t error_squared[CONFIG_T::n_in];
    #pragma HLS array_partition variable=delta complete
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        #pragma HLS unroll
        error[i] = a[i] - b[i];
        error_squared[i] = error[i] * error[i];
    }
    // TODO: check the cast here is okay
    Op_add<typename CONFIG_T::squared_error_t> op_add;
    res[0] = reduce<typename CONFIG_T::squared_error_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::squared_error_t>>(error_squared, op_add) * typename CONFIG_T::squared_error_t(1. / CONFIG_T::n_in);
}

struct custom_mse_config
{
    // IO size
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned NMET = 1;
    static const unsigned NEGAMMAS = 4;
    static const unsigned NMUONS = 4;
    static const unsigned NJETS = 10;
    static const unsigned NPARTS = NMET + NEGAMMAS + NMUONS + NJETS;
    
    // Internal data type definitions
    typedef ap_fixed<16,3> table_t; // type for tanh table data
    typedef ap_fixed<16,6,AP_RND_CONV,AP_SAT> accum_t; // type for squared error

};

template<class data1_T, class data2_T, class res_T, typename CONFIG_T>
void custom_mse(data1_T a[CONFIG_T::n_in],
                data2_T b[CONFIG_T::n_in],
                res_T  res[CONFIG_T::n_out]){

    #pragma HLS pipeline II=1
    static const unsigned N_PARTS = CONFIG_T::NPARTS;
    // Extract separately the pt, eta, phi
    data1_T true_pt[N_PARTS];
    data1_T true_phi[N_PARTS];
    data1_T true_eta[N_PARTS];
    data2_T pred_pt[N_PARTS];
    data2_T pred_phi[N_PARTS];
    data2_T pred_eta[N_PARTS];
    #pragma HLS array_partition variable=pred_pt complete
    #pragma HLS array_partition variable=pred_eta complete
    #pragma HLS array_partition variable=pred_phi complete
    #pragma HLS array_partition variable=true_pt complete
    #pragma HLS array_partition variable=true_eta complete
    #pragma HLS array_partition variable=true_phi complete
    for(unsigned i = 0; i < N_PARTS; i++){
        #pragma HLS unroll
        true_pt[i]  = a[3*i + 0];
        true_eta[i] = a[3*i + 1];
        true_phi[i] = a[3*i + 2];
        pred_pt[i]  = b[3*i + 0];
        pred_eta[i] = b[3*i + 1];
        pred_phi[i] = b[3*i + 2];
    }

    // Compute the tanh of predicted {eta, phi} and do the scaling (pi, 3.0, 2.1 or 4.0)
    // Concat predicted tanh {eta, phi} in a single array to reduce together
    typename CONFIG_T::table_t tanh_pred[N_PARTS];
    typename CONFIG_T::table_t tanh_pred_eta[N_PARTS];
    typename CONFIG_T::table_t tanh_pred_phi[N_PARTS];
    data1_T true_eta_phi[2*N_PARTS];
    #pragma HLS array_partition variable=tanh_pred_eta complete
    #pragma HLS array_partition variable=tanh_pred_phi complete
    tanh<data2_T, typename CONFIG_T::table_t, typename CONFIG_T::tanh_config>(pred_eta, tanh_pred_eta);
    tanh<data2_T, typename CONFIG_T::table_t, typename CONFIG_T::tanh_config>(pred_phi, tanh_pred_phi);
    typename CONFIG_T::table_t mults[N_PARTS] = {1, 3, 3, 3, 3, 2.1, 2.1, 2.1, 2.1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
    for(unsigned i = 0; i < N_PARTS; i++){
        #pragma HLS unroll
        tanh_pred[3*i + 0] = pred_pt[i];
        tanh_pred[3*i + 1] = tanh_pred_eta[i] * mults[i];
        tanh_pred[3*i + 2] = tanh_pred_phi[i] * typename CONFIG_T::table_t(M_PI);
    }

    // apply the mask: for any index in true which contains zero, set that index in pred to 0
    for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        tanh_pred[i] = a[i] == 0 ? typename CONFIG_T::table_t(0) : tanh_pred[i];
    }
    // Reduce the {eta, phi} and {pt} parts of the MSE separately
    typename CONFIG_T::accum_t mse_acc[1];
    mse<data1_T, typename CONFIG_T::table_t, typename CONFIG_T::accum_t, typename CONFIG_T::mse_config>(a, tanh_pred, mse_acc);
    res[0] = mse_acc[0];

}

}//end namespace

#endif
