#ifndef NNET_RECURRENT_H_
#define NNET_RECURRENT_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recurrent_activation.h"

namespace nnet {

struct gru_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in =  1;
    static const unsigned n_out = 1;
    static const unsigned n_units = 1;
    static const unsigned n_timesteps = 1;
    static const unsigned n_outputs = 1;
    static const bool return_sequences = false;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    
    template<class x_T, class y_T, class config_T>
    using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    
    template<class x_T, class y_T, class config_T>
    using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void gru_cell(
    data_T x[CONFIG_T::n_in],
    res_T  h[CONFIG_T::n_units],
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) { 
    static constexpr int recurrent_unroll_factor = CONFIG_T::n_units / CONFIG_T::reuse_factor;
    // A matrix containing the values of matrix product between input (x) and weights (weights), for update, reset and candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_x_w[3 * CONFIG_T::n_units];
    nnet::dense_resource<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_x>(x, mat_mul_x_w, weights, bias);

    // A matrix containing the values of matrix product between previou state (h) and recurrent weights (recurrent_weights), for update, reset and candidate state gates, for each of the units
    hls_register typename CONFIG_T::accum_t mat_mul_h_wr[3 * CONFIG_T::n_units];
    nnet::dense_resource<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config_h>(h, mat_mul_h_wr, recurrent_weights, recurrent_bias);

    // A vector containing both the values of z(t) and r(t) for every state 
    hls_register typename CONFIG_T::accum_t z_r [2 * CONFIG_T::n_units]; 
    
    // Add the individual vectors from the multiplication of mat_mul_x_w = Wx*x(t) and mat_mul_h_wr = Wh*h(t-1)
    // Unrolled fully, no DSPs used
    #pragma unroll      
    for(int i = 0; i < (2 * CONFIG_T::n_units); i++) {
        z_r[i] = mat_mul_x_w[i] + mat_mul_h_wr[i];
    }

    // Activation on z(t) and r(t)
    hls_register typename CONFIG_T::accum_t z_r_act [2*CONFIG_T::n_units]; 
    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_RECURRENT_T>::activation(z_r, z_r_act);

    // A matrix containing the values of Hadamard product between r(t) = z_r_act[n_units:2*n_units] and h(t-1) = h
    hls_register typename CONFIG_T::accum_t hadamard_r_h[CONFIG_T::n_units];
    #pragma unroll recurrent_unroll_factor
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        hadamard_r_h[i] = z_r_act[i + CONFIG_T::n_units] * mat_mul_h_wr[i + 2 * CONFIG_T::n_units];
    }

    // The candidate state; X * W_{hx} + hadmard(r(t), h_(t-1)) * W_{hh} + b_{h}
    typename CONFIG_T::accum_t h_cand[CONFIG_T::n_units];
    // Addition - can unroll fully; no DSPs used here
    #pragma unroll      
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        h_cand[i] =  mat_mul_x_w[i + 2 * CONFIG_T::n_units] + hadamard_r_h[i];
    }

    // Activation on candidate state
    hls_register typename CONFIG_T::accum_t h_cand_act[CONFIG_T::n_units]; 
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::weight_t, typename CONFIG_T::ACT_CONFIG_T>::activation(h_cand, h_cand_act);

    // Update state
    #pragma unroll recurrent_unroll_factor
    for(int i = 0; i < (CONFIG_T::n_units); i++) {
        h[i] = static_cast<res_T>(h_cand_act[i] * (1 - z_r_act[i]) + h[i] * z_r_act[i]);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void gru(
    data_T data[CONFIG_T::n_in],
    res_T  res[CONFIG_T::n_outputs * CONFIG_T::n_units],      
    const typename CONFIG_T::weight_t weights[3 * CONFIG_T::n_units * CONFIG_T::n_in],
    const typename CONFIG_T::weight_t recurrent_weights[3 * CONFIG_T::n_units * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t bias[3 * CONFIG_T::n_units],
    const typename CONFIG_T::bias_t recurrent_bias[3 * CONFIG_T::n_units]
) { 

    hls_register data_T x[CONFIG_T::n_in];
    hls_register res_T h[CONFIG_T::n_units];
    
    #pragma unroll
    for(int i = 0; i < CONFIG_T::n_units; i++) {
        h[i] = 0;
    }

    // Loop depedency - cannot pipeline
    #pragma disable_loop_pipelining
    for(int t = 0; t < CONFIG_T::n_timesteps; t++) {
        // Get data at current time step
        #pragma unroll
        for(int j = 0; j < CONFIG_T::n_in; j++) {
            x[j] = data[j + t * CONFIG_T::n_in];
        }
      
        nnet::gru_cell<data_T, res_T, CONFIG_T>(x, h, weights, recurrent_weights, bias, recurrent_bias);

        if (CONFIG_T::return_sequences) {
            #pragma unroll
            for(int i = 0 ; i < CONFIG_T::n_units ; i++) {
                res[CONFIG_T::n_units * t + i] = h[i];
            }
        }
    }
    
    if (!CONFIG_T::return_sequences) {
        #pragma unroll
        for(int i = 0; i < (CONFIG_T::n_units); i++) {
            res[i] = h[i];
        }
    }
}

}

#endif
