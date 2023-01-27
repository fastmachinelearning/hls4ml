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
	hls::stream<data_T>  Q[CONFIG_T::head_dim_key],
	data_T  K[CONFIG_T::seq_len][CONFIG_T::head_dim_key],
    res_T  QK[CONFIG_T::seq_len][CONFIG_T::seq_len]) // seq_Q, seq_K
{
    const data_T dk = 1.0/sqrt(CONFIG_T::head_dim_key);
    data_T QKij, QK_1;
    data_T Qi[CONFIG_T::head_dim_key];
    data_T Product[CONFIG_T::seq_len];
#pragma HLS ARRAY_RESHAPE variable=K cyclic factor=4 dim=1
#pragma HLS ARRAY_RESHAPE variable=K complete dim=2
#pragma HLS ARRAY_RESHAPE variable=Qi complete dim=0
#pragma HLS ARRAY_PARTITION variable=Product complete

#pragma HLS ARRAY_PARTITION variable=QK complete dim=2

    // for each row and column of AB
    row: for(int i = 0; i < CONFIG_T::seq_len; ++i) {
    	q: for(int q_i = 0; q_i < CONFIG_T::head_dim_key; ++q_i){
		#pragma HLS UNROLL
    		Qi[q_i]=Q[q_i].read();
    	}
        col: for(int j = 0; j < CONFIG_T::seq_len; ++j) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=8
            // compute (QK)i,j
            QKij = 0;
            product: for(int k = 0; k < CONFIG_T::head_dim_key; ++k) {
            	QK_1 = Qi[k] * K[j][k];
                QKij += QK_1;
            }
            Product[j] = QKij * dk;
        }
        softmax<data_T, res_T, typename CONFIG_T::softmax_config1>(Product, QK[i]);
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void matrixmul(
    data_T QK[CONFIG_T::seq_len][CONFIG_T::seq_len], 
    data_T  V[CONFIG_T::seq_len][CONFIG_T::head_dim_value], 
	hls::stream<data_T>   S[CONFIG_T::head_dim_value]) // S: attention score
{
//	data_T Product[CONFIG_T::head_dim_value];
	#pragma HLS ARRAY_Partition variable=QK complete dim=2
	#pragma HLS ARRAY_Partition variable=S complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=V complete dim=1
//	#pragma HLS ARRAY_PARTITION variable=Product complete
//	#pragma HLS ARRAY_Partition variable=V cyclic factor=2 dim=2 //
    // for each row and column of AB
    data_T Sij;
    row: for(int i = 0; i < CONFIG_T::seq_len; ++i) {
        col: for(int j = 0; j < CONFIG_T::head_dim_value; ++j) {
#pragma HLS PIPELINE
#pragma HLS UNROLL factor=2
            // compute (S)i,j
            Sij = 0;
            product: for(int k = 0; k < CONFIG_T::seq_len; ++k) {
                Sij += QK[i][k] * V[k][j];
            }
            S[j].write(Sij);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_value(
    data_T    data_v[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    data_T    v_proj [CONFIG_T::seq_len][CONFIG_T::head_dim_value],
    typename CONFIG_T::weight_t  value_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_value],
    typename CONFIG_T::bias_t    value_bias[CONFIG_T::head_dim_value])
{
    data_T  v_row[CONFIG_T::head_dim_value];
    #pragma HLS ARRAY_PARTITION variable=v_row complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=v_proj complete dim=1
//    #pragma HLS function_instantiate variable=value_weight,value_bias
    v_h: for (int j=0; j <CONFIG_T::seq_len; ++j){
#pragma HLS DATAFLOW
        dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_v+(CONFIG_T::feature_dim*j), v_row, value_weight, value_bias);
        for (int k=0; k <CONFIG_T::head_dim_value; ++k){
            v_proj[j][k]=v_row[k];
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_query(
    data_T    data_q[CONFIG_T::seq_len * CONFIG_T::feature_dim],
	hls::stream<data_T>    q_proj[CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t  query_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_key],
    typename CONFIG_T::bias_t    query_bias[CONFIG_T::head_dim_key])
{
//    #pragma HLS function_instantiate variable=query_weight,query_bias
	data_T proj[CONFIG_T::head_dim_key];
	#pragma HLS ARRAY_PARTITION variable=proj complete dim=0
	#pragma HLS ARRAY_PARTITION variable=q_proj complete dim=0
    q_h: for (int j=0; j <CONFIG_T::seq_len; ++j){
#pragma HLS DATAFLOW
        dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_q +(CONFIG_T::feature_dim*j), proj, query_weight, query_bias);
        update_proj: for (int i=0; i<CONFIG_T::head_dim_key; ++i){
		#pragma HLS UNROLL
        	q_proj[i].write(proj[i]);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void dense_key(
    data_T    data_k[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    data_T    k_proj[CONFIG_T::seq_len][CONFIG_T::head_dim_key],
    typename CONFIG_T::weight_t  key_weight[CONFIG_T::feature_dim * CONFIG_T::head_dim_key],
    typename CONFIG_T::bias_t    key_bias[CONFIG_T::head_dim_key])
{
	data_T  k_row[CONFIG_T::head_dim_key];
#pragma HLS ARRAY_PARTITION variable=k_row complete dim=0
//    #pragma HLS function_instantiate variable=key_weight,key_bias
    k_h: for (int j=0; j <CONFIG_T::seq_len; ++j){
#pragma HLS DATAFLOW
        dense<data_T, res_T, typename CONFIG_T::config_mult1>(data_k+(CONFIG_T::feature_dim*j), k_row, key_weight, key_bias);
        //k_proj[j] = k_row;
        for (int k=0; k <CONFIG_T::head_dim_key; ++k){
#pragma HLS UNROLL
        	k_proj[j][k]=k_row[k];
		}
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void dense_out(
	hls::stream<data_T> data_in[CONFIG_T::num_heads][CONFIG_T::head_dim_value],
	res_T     res[CONFIG_T::seq_len * CONFIG_T::feature_dim],
    typename CONFIG_T::weight_t  attention_output_weight[CONFIG_T::num_heads * CONFIG_T::head_dim_value * CONFIG_T::feature_dim],
    typename CONFIG_T::bias_t    attention_output_bias[CONFIG_T::feature_dim])
{
//    #pragma HLS function_instantiate variable=query_weight,query_bias
	data_T mat_res_con[CONFIG_T::num_heads*CONFIG_T::head_dim_value];
	#pragma HLS ARRAY_PARTITION variable=mat_res_con complete dim=0
	output_dense: for (int k=0; k <CONFIG_T::seq_len; ++k){
		for (int i=0;i<CONFIG_T::num_heads; ++i){
			for (int j=0;j<CONFIG_T::head_dim_value; ++j){
			#pragma HLS UNROLL
				mat_res_con[CONFIG_T::head_dim_value*i+j]=data_in[i][j].read();
			}
		}
		dense<data_T, res_T, typename CONFIG_T::config_mult2>(mat_res_con, res+(CONFIG_T::feature_dim*k), attention_output_weight, attention_output_bias);
		// nnet::print_result<result_t, CONFIG_T::feature_dim>( res+(CONFIG_T::feature_dim*j), std::cout);
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

	hls::stream<data_T> q_proj[CONFIG_T::num_heads][CONFIG_T::head_dim_key];
    data_T v_proj[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::head_dim_value];
    data_T k_proj[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::head_dim_key];
    data_T qk_mul[CONFIG_T::num_heads][CONFIG_T::seq_len][CONFIG_T::seq_len];
//    data_T mat_res_con[CONFIG_T::num_heads*CONFIG_T::head_dim_value];
//    data_T dense_in[CONFIG_T::seq_len][CONFIG_T::num_heads * CONFIG_T::head_dim_value];
    hls::stream<data_T> matr_out[CONFIG_T::num_heads][CONFIG_T::head_dim_value];

#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=v_proj complete dim=1
#pragma HLS ARRAY_PARTITION variable=q_proj complete dim=1
#pragma HLS ARRAY_PARTITION variable=k_proj complete dim=1
#pragma HLS ARRAY_PARTITION variable=qk_mul complete dim=1
#pragma HLS ARRAY_PARTITION variable=matr_out complete dim=1
    // std::cout << "input to MHA: " << std::endl;
    // nnet::print_result<result_t, CONFIG_T::seq_len * CONFIG_T::feature_dim>(data_q, std::cout);
    // std::cout << " " << std::endl;

    // linear projection
    for (int i=0;i<CONFIG_T::num_heads; ++i){
#pragma HLS UNROLL
    	dense_value<data_T, res_T, CONFIG_T>(data_vk, v_proj[i], value_weight+(CONFIG_T::head_dim_value*CONFIG_T::feature_dim*i), value_bias+(CONFIG_T::head_dim_value*i));
    }
    for (int i=0;i<CONFIG_T::num_heads; ++i){
#pragma HLS UNROLL
    	dense_query<data_T, res_T, CONFIG_T>(data_q, q_proj[i], query_weight+(CONFIG_T::head_dim_key*CONFIG_T::feature_dim*i), query_bias+(CONFIG_T::head_dim_key*i));
    }
    for (int i=0;i<CONFIG_T::num_heads; ++i){
#pragma HLS UNROLL
    dense_key<data_T, res_T, CONFIG_T>(data_vk, k_proj[i], key_weight+(CONFIG_T::head_dim_key*CONFIG_T::feature_dim*i), key_bias+(CONFIG_T::head_dim_key*i));
    }
    
    nnet::matrixmul_transpose<data_T, res_T, CONFIG_T>(q_proj[0], k_proj[0], qk_mul[0]);
    nnet::matrixmul_transpose<data_T, res_T, CONFIG_T>(q_proj[1], k_proj[1], qk_mul[1]);
    nnet::matrixmul_transpose<data_T, res_T, CONFIG_T>(q_proj[2], k_proj[2], qk_mul[2]);


    maxtrixmul2: for (int i=0; i < CONFIG_T::num_heads; ++i){
#pragma HLS UNROLL
    	nnet::matrixmul<data_T, res_T, CONFIG_T>(qk_mul[i], v_proj[i], matr_out[i]);//stream
    }

    dense_out<data_T, res_T, CONFIG_T>(matr_out, res, attention_output_weight, attention_output_bias);
    // std::cout << "output from MHA: " << std::endl;
    // std::cout << " " << std::endl;
}
}

#endif
