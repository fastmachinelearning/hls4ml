#ifndef NNET_MHT_H_
#define NNET_MHT_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "nnet_dense.h"
#include "nnet_activation.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>

namespace nnet {

struct multiheadattention_config
{
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned num_heads = 10;
    static const unsigned head_dim_key = 10;
    static const unsigned head_dim_value = 10;
    static const unsigned feature_dim = 20;
    static const unsigned seq_len = 500;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned strategy = latency; 
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

template<class data_T, class res_T, typename CONFIG_T>
void matrixmul_transpose(
    data_T  Q[CONFIG_T::seq_len][CONFIG_T::head_dim_key], 
    data_T  K[CONFIG_T::seq_len][CONFIG_T::head_dim_key], 
    res_T  QK[CONFIG_T::seq_len][CONFIG_T::seq_len]) // seq_Q, seq_K
{
    const data_T dk = 1.0/sqrt(CONFIG_T::head_dim_key);
    data_T QKij;
    data_T Product[CONFIG_T::seq_len];
#pragma HLS ARRAY_RESHAPE variable=K cyclic factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=Q cyclic factor=2 dim=1
#pragma HLS ARRAY_RESHAPE variable=K complete dim=2
#pragma HLS ARRAY_RESHAPE variable=Q complete dim=2
#pragma HLS ARRAY_PARTITION variable=Product complete

#pragma HLS ARRAY_Partition variable=QK complete dim=2

    // for each row and column of AB
    row: for(int i = 0; i < CONFIG_T::seq_len; ++i) {
        col: for(int j = 0; j < CONFIG_T::seq_len; ++j) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=4
            // compute (QK)i,j
            QKij = 0;
            product: for(int k = 0; k < CONFIG_T::head_dim_key; ++k) {
                QKij += Q[i][k] * K[j][k];
            }
            Product[j] = QKij * dk;
//            QK[i][j] = QKij * dk;
        }
        // std::cout << "input to softmax: " << std::endl;
        // nnet::print_result<result_t, CONFIG_T::seq_len>(Product, std::cout);

        softmax<data_T, res_T, typename CONFIG_T::softmax_config1>(Product, QK[i]);

        // std::cout << "output from softmax: " << std::endl;
        // nnet::print_result<result_t, CONFIG_T::seq_len>( QK[i], std::cout);
    }
}


template<class data_T, class res_T, typename CONFIG_T, class T>
void matrixmul(
    data_T QK[CONFIG_T::seq_len][CONFIG_T::seq_len], 
    data_T  V[CONFIG_T::seq_len][CONFIG_T::head_dim_value], 
    res_T   S[CONFIG_T::seq_len][CONFIG_T::num_heads * CONFIG_T::head_dim_value],
    T       head) // S: attention score
{
	#pragma HLS ARRAY_Partition variable=QK complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=V complete dim=1
    // for each row and column of AB
    data_T Sij;
    row: for(int i = 0; i < CONFIG_T::seq_len; ++i) {
        col: for(int j = 0; j < CONFIG_T::head_dim_value; ++j) {
#pragma HLS PIPELINE
            // compute (S)i,j
            Sij = 0;
            product: for(int k = 0; k < CONFIG_T::seq_len; ++k) {
                Sij += QK[i][k] * V[k][j];
            }
            S[i][CONFIG_T::head_dim_value*head+j] = Sij;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void multiheadattention(
    data_T    data_q[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    data_T    data_vk[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    res_T     res[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t  attention_output_weight[CONFIG_T::num_heads * CONFIG_T::head_dim_value * CONFIG_T::feature_dim],  // num_heads,head_size_v,dim
    typename CONFIG_T::bias_t    attention_output_bias[CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t  key_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key],  // n_head,dim,head_dim
    typename CONFIG_T::bias_t    key_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t  query_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_key], //same shape as key
    typename CONFIG_T::bias_t    query_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t  value_weight[CONFIG_T::feature_dim * CONFIG_T::num_heads * CONFIG_T::head_dim_value],
    typename CONFIG_T::bias_t    value_bias[CONFIG_T::num_heads * CONFIG_T::head_dim_value])
{

    data_T q_proj[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::head_dim_key];
    data_T v_proj[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::head_dim_value];
    data_T k_proj[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::head_dim_key];
    data_T qk_mul[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::seq_len];
    data_T qk_mul_sm[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::seq_len];

    data_T dense_in[CONFIG_T::seq_len][CONFIG_T::num_heads * CONFIG_T::head_dim_value];
#pragma HLS ARRAY_PARTITION variable=dense_in complete dim=0

#pragma HLS ARRAY_PARTITION variable=q_proj complete dim=1  // partition the 1-dim
#pragma HLS ARRAY_PARTITION variable=v_proj complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_proj complete dim=1
//#pragma HLS ARRAY_PARTITION variable=qk_mul complete dim=1



    // linear projection
    dense_for_each_head: for (int i=0; i < CONFIG_T::num_heads; ++i){
#pragma HLS DATAFLOW
// or #pragma HLS unroll // less BRAM slower

        seq1: for (int j=0; j <CONFIG_T::seq_len; ++j){
            // nnet::print_result<result_t, CONFIG_T::feature_dim>(data_q +(CONFIG_T::feature_dim*j), std::cout);
            dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_q +(CONFIG_T::feature_dim*j), q_proj[i][j], query_weight+(CONFIG_T::head_dim_key  *CONFIG_T::feature_dim*i), query_bias+(CONFIG_T::head_dim_key*i));
        }
        
        seq2: for (int j=0; j <CONFIG_T::seq_len; ++j){
            dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_vk+(CONFIG_T::feature_dim*j), k_proj[i][j], key_weight  +(CONFIG_T::head_dim_key  *CONFIG_T::feature_dim*i), key_bias  +(CONFIG_T::head_dim_key*i));
        }

        nnet::matrixmul_transpose<data_T, res_T, CONFIG_T>(q_proj[i], k_proj[i], qk_mul[i]);

        seq3: for (int j=0; j <CONFIG_T::seq_len; ++j){
            dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_vk+(CONFIG_T::feature_dim*j), v_proj[i][j], value_weight+(CONFIG_T::head_dim_value*CONFIG_T::feature_dim*i), value_bias+(CONFIG_T::head_dim_value*i));
        }

        nnet::matrixmul<data_T, res_T, CONFIG_T, int>(qk_mul[i], v_proj[i], dense_in, i);
    }

    output_dense: for (int j=0; j <CONFIG_T::seq_len; ++j){ 
        // nnet::print_result<result_t, CONFIG_T::num_heads*CONFIG_T::head_dim_value>(dense_in[j], std::cout);
        dense<data_T, res_T, typename CONFIG_T::config_mult2>(dense_in[j], res+(CONFIG_T::feature_dim*j), attention_output_weight, attention_output_bias);
    }
}
}

#endif
