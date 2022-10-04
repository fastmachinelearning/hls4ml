#ifndef NNET_GRAPH_H_
#define NNET_GRAPH_H_

#include "nnet_common.h"
#include "nnet_merge.h"
// #include "nnet_dense.h"
#include "nnet_dense_resource.h"
#include "nnet_activation.h"
#include "nnet_array.h"
#include <math.h>
#include "utils/x_hls_utils.h"
#include "nnet_activation.h" // for softmax_idx_from_real_val, init_exp_table, init_invert_table, relu_0D

namespace nnet {
  enum flow {source_to_target=0, target_to_source=1};
  enum aggr {aggr_sum=0, aggr_mean=1, aggr_max=2, aggr_softmax=3};
  enum activation {
    //linear_act=0=default,
    relu_act=1,
    sigmoid_act=2,
    selu_act=3,
    tanh_act=4,
    softplus_act=5,
    softsign_act=6,
    hard_sigmoid_act=7,
    binary_tanh_act=8,
    ternary_tanh_act=9
  };
  
  struct graph_config
  {
    // Internal data type definitions
    typedef float bias_t;
    typedef float weight_t;
    typedef float table_t;
    
    // Layer Sizes
    static const unsigned n_node = 10;
    static const unsigned n_edge = 20;
    static const unsigned n_features = 3;
    static const unsigned e_features = 4;
    static const unsigned n_out = 4;
    static const unsigned n_layers = 3;

    // message-passing parameters
    static const unsigned aggr = aggr_sum;
    static const unsigned flow = source_to_target;
    
    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;

    // Final activation info
    static const bool activate_final = false;
    static const bool gnn_resource_limit = false;
    static const unsigned par_factor = 16;
  };

  struct edge_aggregate_config
  {
     typedef float table_t;
     static const unsigned n_node = 10;
     static const unsigned n_edge = 20;
     static const unsigned edge_dim = 4;
     static const unsigned aggr = aggr_sum;
     static const unsigned flow = source_to_target;
     static const unsigned io_type = io_parallel;
     static const unsigned reuse_factor = 1;
     static const bool io_stream = false;
     static const bool activate_final = false;
     static const bool gnn_resource_limit = false;
     static const unsigned par_factor = 16;
  };

  struct residual_config
  {
     static const unsigned n_elem = 7;
     static const bool gnn_resource_limit = false;
  };

  struct block_activation_config
  {
    // IO size
    static const unsigned n_in = 10;

    // Activation type (sigmoid, relu, etc.)
    static const unsigned activation=0;

    // Internal info
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;

    // Internal data type definitions
    typedef ap_fixed<52,20> table_t; // ap_fixed<18,8>
  };

  // LUT-division for mean-aggregation
  inline float division(float input){
    return 1.0/input;
  }
  template<typename CONFIG_T, int N_TABLE>
  void init_div_table(typename CONFIG_T::table_t table_out[N_TABLE]){
    int j = 0;
    typename CONFIG_T::table_t k = 1;
    table_out[j] = k;
    for(int i=1; i<N_TABLE; i++){
      float in_val = float(i);
      typename CONFIG_T::table_t reciprocal = nnet::division(in_val);
      table_out[i] = reciprocal;
    }
  }
  template<class data_T, class index_T, class res_T, typename CONFIG_T>
  void edge_divide(data_T edge_sum_i, index_T n_edges_i, res_T &edge_mean_i){
    // initialize LUT
  #ifdef __HLS_SYN__
      bool initialized=false;
      typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #else
      static bool initialized=false;
      static typename CONFIG_T::table_t div_table[CONFIG_T::n_edge];
  #endif

      if(!initialized){
        nnet::init_div_table<CONFIG_T, CONFIG_T::n_edge>(div_table);
        initialized=true;
      }

      if(CONFIG_T::io_type==io_parallel){
        #pragma HLS PIPELINE
      }

      data_T reciprocal;
      reciprocal = div_table[n_edges_i];
      edge_mean_i = edge_sum_i*reciprocal;
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_1lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out])
  {
    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
    // nnet::dense<data_T, res_T, typename CONFIG_T::dense_config1>(data, res, weights0, biases0);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_2lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
       typename CONFIG_T::norm_config1::scale_t   norm_scales0[CONFIG_T::norm_config1::n_in],
       typename CONFIG_T::norm_config1::bias_t   norm_biases0[CONFIG_T::norm_config1::n_in]
       )
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T norm_data0[CONFIG_T::norm_config1::n_in];
    #pragma HLS ARRAY_PARTITION variable=norm_data0 complete dim=0
    nnet::normalize<data_T, data_T, typename CONFIG_T::norm_config1>(data0_logits, norm_data0, norm_scales0, norm_biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    // nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);
    nnet::leaky_relu<data_T, data_T, typename CONFIG_T::relu_config1>(norm_data0, data0);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
    // nnet::dense<data_T, res_T, typename CONFIG_T::dense_config2>(data0, res, weights1, biases1);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_3lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
    // nnet::dense<data_T, res_T, typename CONFIG_T::dense_config3>(data1, res, weights2, biases2);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void dense_mult_4lyr(
			 data_T data[CONFIG_T::dense_config1::n_in],
			 res_T res[CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config1::weight_t weights0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config1::bias_t   biases0[CONFIG_T::dense_config1::n_out],
			 typename CONFIG_T::dense_config2::weight_t weights1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config2::bias_t   biases1[CONFIG_T::dense_config2::n_out],
			 typename CONFIG_T::dense_config3::weight_t weights2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config3::bias_t   biases2[CONFIG_T::dense_config3::n_out],
			 typename CONFIG_T::dense_config4::weight_t weights3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			 typename CONFIG_T::dense_config4::bias_t   biases3[CONFIG_T::dense_config4::n_out])
  {
    data_T data0_logits[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config1>(data, data0_logits, weights0, biases0);
    data_T data0[CONFIG_T::dense_config1::n_out];
    #pragma HLS ARRAY_PARTITION variable=data0 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config1>(data0_logits, data0);

    data_T data1_logits[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config2>(data0, data1_logits, weights1, biases1);
    data_T data1[CONFIG_T::dense_config2::n_out];
    #pragma HLS ARRAY_PARTITION variable=data1 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config2>(data1_logits, data1);

    data_T data2_logits[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2_logits complete dim=0
    nnet::dense_resource<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    // nnet::dense<data_T, data_T, typename CONFIG_T::dense_config3>(data1, data2_logits, weights2, biases2);
    data_T data2[CONFIG_T::dense_config3::n_out];
    #pragma HLS ARRAY_PARTITION variable=data2 complete dim=0
    nnet::relu<data_T, data_T, typename CONFIG_T::relu_config3>(data2_logits, data2);

    nnet::dense_resource<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
    // nnet::dense<data_T, res_T, typename CONFIG_T::dense_config4>(data2, res, weights3, biases3);
  }

  //////////////////////////////dataflow/////////////////////////////////////
  template<class data_T, int row, int col>
    void replicate(
    data_T     IN  [row*col],
    data_T     OUT1[row*col],
    data_T     OUT2[row*col]
  )
  {
    for(int i=0; i<row; i++){
      #pragma HLS UNROLL
      for(int j=0; j<col; j++){
        #pragma HLS UNROLL
        OUT1[i*col+j] =  IN[i*col+j];
        OUT2[i*col+j] =  IN[i*col+j];
      }
    }
  }

    // generalized block-activation function for dataflow
    /*
    data_T data[CONFIG_T::n_in] is the input,
    res_T res[CONFIG_T::n_in] is the output
    */
  template<class data_T, class res_T, typename CONFIG_T>
  void  block_activation(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]){
    if (CONFIG_T::activation==relu_act){
      nnet::relu<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==sigmoid_act){
      nnet::sigmoid<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==selu_act){
      nnet::selu<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==tanh_act){
      nnet::tanh<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==softplus_act){
      nnet::softplus<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==softsign_act){
      nnet::softsign<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==hard_sigmoid_act){
      nnet::hard_sigmoid<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==binary_tanh_act){
      nnet::binary_tanh<data_T, res_T, CONFIG_T>(data, res);
    }
    else if (CONFIG_T::activation==ternary_tanh_act){
      nnet::ternary_tanh<data_T, res_T, CONFIG_T>(data, res);
    }
    else {
      nnet::linear<data_T, res_T, CONFIG_T>(data, res);
    }
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edge_aggregate_dataflow(
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  {
    #pragma HLS INLINE
    res_T edge_attr_aggr[CONFIG_T::par_factor][CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=1
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr cyclic factor=CONFIG_T::par_factor dim=2
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=3

    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      receiver_col = 1;
    }
    else{
      receiver_col = 0;
    }

    if(CONFIG_T::aggr==aggr_max){
      ap_uint<1> edge_aggr_mask_mat[CONFIG_T::par_factor][CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask_mat complete dim=0
      ap_uint<1> edge_aggr_mask[CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask complete dim=0

      mask_reset:for(int i=0;i<CONFIG_T::n_node;i++){
        #pragma HLS UNROLL
        edge_aggr_mask[i]=0;
        for(int k=0; k < CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          edge_aggr_mask_mat[k][i]=0;
        }
      }

      res_T most_negative_num = -hls::numeric_limits<res_T>::max();
      edge_attr_aggr_reset:for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        for(int k=0; k < CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for(int j=0; j<CONFIG_T::edge_dim; j++){
            #pragma HLS UNROLL
            edge_attr_aggr[k][i][j] = most_negative_num;
          }
        }
      }

      compute:for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        edge_aggr_mask_mat[i%CONFIG_T::par_factor][r]=1;
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i%CONFIG_T::par_factor][r][j] = edge_attr_1D[i*CONFIG_T::edge_dim+j] > edge_attr_aggr[i%CONFIG_T::par_factor][r][j] ? edge_attr_1D[i*CONFIG_T::edge_dim+j] : edge_attr_aggr[i%CONFIG_T::par_factor][r][j];
        }
      }

      combine:for (int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for (int k=0; k<CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for (int j=0; j<CONFIG_T::edge_dim; j++){
            #pragma HLS UNROLL
            if(k==0){
              edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j]=edge_attr_aggr[k][i][j];
            }
            else{
              edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j] = edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j] > edge_attr_aggr[k][i][j] ? edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j] : edge_attr_aggr[k][i][j];
            }
          }
          if(edge_aggr_mask[i]==0){
            edge_aggr_mask[i]=edge_aggr_mask_mat[k][i];
          }
        }
      }

      masking:for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for(int j=0; j<CONFIG_T::edge_dim; j++){
            #pragma HLS UNROLL
            edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j] = edge_aggr_mask[i]*edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j];
        }
      }

    }

    if(CONFIG_T::aggr==aggr_mean){
      index_T num_edge_per_node_mat[CONFIG_T::par_factor][CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=num_edge_per_node_mat complete dim=0
      index_T num_edge_per_node[CONFIG_T::n_node];
      #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0

      for(int i=0;i<CONFIG_T::n_node;i++){
        #pragma HLS UNROLL
        for(int k=0; k < CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          num_edge_per_node_mat[k][i]=0;
        }
      }

      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        for(int k=0; k < CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for(int j=0; j<CONFIG_T::edge_dim; j++){
            #pragma HLS UNROLL
            edge_attr_aggr[k][i][j] = 0;
          }
        }
      }

      for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        num_edge_per_node_mat[i%CONFIG_T::par_factor][r]+=1;
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i%CONFIG_T::par_factor][r][j] += edge_attr_1D[i*CONFIG_T::edge_dim+j];
        }
      }

      for (int r=0; r < CONFIG_T::n_node; r++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for (int k=0; k<CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for (int c=0; c<CONFIG_T::edge_dim; c++){
            #pragma HLS UNROLL
            if(k==0){
              edge_attr_aggr_1D[r*CONFIG_T::edge_dim+c]=edge_attr_aggr[k][r][c];
            }
            else{
              edge_attr_aggr_1D[r*CONFIG_T::edge_dim+c]+=edge_attr_aggr[k][r][c];
            }
          }
          if(k==0){
            num_edge_per_node[r]=num_edge_per_node_mat[k][r];
          }
          else{
            num_edge_per_node[r]+=num_edge_per_node_mat[k][r];
          }
        }
      }

      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for (int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j], num_edge_per_node[i], edge_mean_j);
          edge_attr_aggr_1D[i*CONFIG_T::edge_dim+j] = edge_mean_j;
        }
      }

    }

    if(CONFIG_T::aggr==aggr_sum){
      for(int i=0; i < CONFIG_T::n_node; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        for(int k=0; k < CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for(int j=0; j<CONFIG_T::edge_dim; j++){
             #pragma HLS UNROLL
             edge_attr_aggr[k][i][j] = 0;
          }
        }
      }

      for(int i=0; i < CONFIG_T::n_edge; i++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        index_T r = edge_index_1D[i*2+receiver_col];
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i%CONFIG_T::par_factor][r][j] += edge_attr_1D[i*CONFIG_T::edge_dim+j];
        }
      }

      //output array --> output vec
      for (int r=0; r < CONFIG_T::n_node; r++){
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        for (int k=0; k<CONFIG_T::par_factor; k++){
          #pragma HLS UNROLL
          for (int c=0; c<CONFIG_T::edge_dim; c++){
            #pragma HLS UNROLL
            if(k==0){
              edge_attr_aggr_1D[r*CONFIG_T::edge_dim+c]=edge_attr_aggr[k][r][c];
            }
            else{
              edge_attr_aggr_1D[r*CONFIG_T::edge_dim+c]+=edge_attr_aggr[k][r][c];
            }
          }
        }
      }

    }
    
  }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock_dataflow(
      data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
      data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
      index_T   edge_index_1D[CONFIG_T::n_edge*2],
      res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
      typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      sender_col = 0;
      receiver_col = 1;
    }
    else{
      sender_col = 1;
      receiver_col = 0;
    }

    data_T node_attr_1D_mat[CONFIG_T::par_factor][CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr_1D_mat complete  dim=1
    #pragma HLS ARRAY_RESHAPE variable=node_attr_1D_mat cyclic factor=CONFIG_T::par_factor dim=2
    #pragma HLS ARRAY_PARTITION variable=node_attr_1D_mat complete dim=3
    replicate_loop:for(int j=0;j<CONFIG_T::n_node;j=j+1)
    {
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
      #pragma HLS UNROLL factor=CONFIG_T::par_factor
      for(int i=0;i<CONFIG_T::par_factor;i++)
      {
        #pragma HLS UNROLL
        for (int c=0; c < CONFIG_T::node_dim; c++){
          #pragma HLS UNROLL
          node_attr_1D_mat[i][j][c] = node_attr_1D[j*CONFIG_T::node_dim+c];
        }
      }
    }

    edge_loop_1: for(int i = 0; i < CONFIG_T::n_edge; i+=1) { //for each edge
      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
      #pragma HLS UNROLL factor=CONFIG_T::par_factor
      data_T edge_attr[CONFIG_T::edge_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
      trans_loop_1: for (int c=0; c < CONFIG_T::edge_dim; c++){
        #pragma HLS UNROLL
        edge_attr[c] = edge_attr_1D[i*CONFIG_T::edge_dim+c];
      }

      index_T s = edge_index_1D[i*2+sender_col];
      index_T r = edge_index_1D[i*2+receiver_col];
      data_T node_attr_r[CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_attr_r complete dim=0

      data_T node_attr_s[CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=node_attr_s complete dim=0
      trans_loop_3: for (int c=0; c < CONFIG_T::node_dim; c++){
        #pragma HLS UNROLL
        node_attr_s[c] = node_attr_1D_mat[i%CONFIG_T::par_factor][s][c];
        node_attr_r[c] = node_attr_1D_mat[i%CONFIG_T::par_factor][r][c];
      }

      data_T phi_input[CONFIG_T::edge_dim + 2*CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      for(int j=0; j<CONFIG_T::node_dim; j++){
        #pragma HLS UNROLL
        phi_input[j] = node_attr_r[j];
        phi_input[CONFIG_T::node_dim+j] = node_attr_s[j];
      }
      for(int k=0; k<CONFIG_T::edge_dim; k++){
        #pragma HLS UNROLL
        phi_input[2*CONFIG_T::node_dim+k] = edge_attr[k];
      }
      res_T edge_update[CONFIG_T::out_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_update complete dim=0
      // send it through NN
      if(CONFIG_T::n_layers == 1){
        nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0);
      }
      else if(CONFIG_T::n_layers == 2){
        nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
      }
      else if(CONFIG_T::n_layers == 3){
        nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
      }
      else if(CONFIG_T::n_layers == 4){
        nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, edge_update, core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
      }

      data_T edge_update_act [CONFIG_T::out_dim];
      if(CONFIG_T::activate_final){
        #pragma HLS ARRAY_PARTITION variable=edge_update_act dim=0
        nnet::block_activation<data_T, res_T, typename CONFIG_T::activation_config>(edge_update, edge_update_act);
      }
      trans_loop_5: for (int c=0; c < CONFIG_T::out_dim; c++){
        #pragma HLS UNROLL
        if(CONFIG_T::activate_final){
          edge_update_1D[i*CONFIG_T::out_dim+c] = edge_update_act[c];
        }
        else{
          edge_update_1D[i*CONFIG_T::out_dim+c] = edge_update[c];
        }
      }
    }
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock_dataflow(
      data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
      data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
      res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
      typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
      typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
      typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
      typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  {
    #pragma HLS INLINE
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node

      #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        #pragma HLS UNROLL factor=CONFIG_T::par_factor
       data_T node_attr[CONFIG_T::node_dim];
       #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
      trans_loop_1: for (int c=0; c < CONFIG_T::node_dim; c++){
        #pragma HLS UNROLL
        node_attr[c] = node_attr_1D[i*CONFIG_T::node_dim+c];
      }
      data_T edge_attr_aggr[CONFIG_T::edge_dim];
      #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
      trans_loop_2: for (int c=0; c < CONFIG_T::edge_dim; c++){
        #pragma HLS UNROLL
        edge_attr_aggr[c] = edge_attr_aggr_1D[i*CONFIG_T::edge_dim+c];
      }
      data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr, edge_attr_aggr, phi_input);
       res_T node_update[CONFIG_T::out_dim];
       #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0
        if(CONFIG_T::n_layers == 1){
          nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0);
        }
        else if(CONFIG_T::n_layers == 2){
          nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1);
        }
        else if(CONFIG_T::n_layers == 3){
          nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
        }
        else { // CONFIG_T::n_layers == 4
          nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update, core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
        }

        data_T node_update_act [CONFIG_T::out_dim];
        if (CONFIG_T::activate_final){
          #pragma HLS ARRAY_PARTITION variable=node_update_act dim=0
          nnet::block_activation<data_T, res_T, typename CONFIG_T::activation_config>(node_update, node_update_act);
        }
        trans_loop_3: for (int c=0; c < CONFIG_T::out_dim; c++){
        #pragma HLS UNROLL
        if (CONFIG_T::activate_final){
          node_update_1D[i*CONFIG_T::out_dim+c] = node_update_act[c];
        }
        else{
          node_update_1D[i*CONFIG_T::out_dim+c] = node_update[c];
        }
      }
    }
  }

  ////////////////////////////////////pipeline////////////////////////////////////
  // this one seems to be throughput oriented version since there is no
  // par_factor (parallization factor)
  /*
    data_T data[CONFIG_T::n_in] is the input,
    res_T res[CONFIG_T::n_in] is the output
    */
  // template<class data_T, class index_T, class res_T, typename CONFIG_T>
  //   void edge_aggregate_pipeline(
  //           data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
  //           index_T   edge_index_1D[CONFIG_T::n_edge*2],
  //           res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  // {
  //   //initialize arrays
  //   std::cout << "CONFIG_T::aggr: " << CONFIG_T::aggr << "\n";
  //   // 1. edge_attr (input)
  //   data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
  //   #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
  //   nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

  //   //2. num_edge_per_node (intermediate), 3. edge_aggr_mask (intermediate)
  //   index_T num_edge_per_node[CONFIG_T::n_node]; // Think this is the list of degrees for each node
  //   #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
  //   ap_uint<1> edge_aggr_mask[CONFIG_T::n_node];
  //   #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask complete dim=0
  //   for(int i=0; i<CONFIG_T::n_node; i++){
  //     #pragma HLS UNROLL
  //     num_edge_per_node[i] = 0;
  //     if(CONFIG_T::aggr==aggr_max){
  //       edge_aggr_mask[i] = 0;
  //     }
  //   }

  //   //4. edge_attr_aggr (output)
  //   res_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
  //   #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
  //   if((CONFIG_T::aggr==aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
  //     for(int i=0; i < CONFIG_T::n_node; i++){
  //       for(int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         edge_attr_aggr[i][j] = 0;
  //       }
  //     }
  //   }
  //   else{ //CONFIG_T:aggr==aggr_max, we want to initialize this with the most negative number we can represent ->why?
  //     res_T most_negative_num = -hls::numeric_limits<res_T>::max();
  //     for(int i=0; i < CONFIG_T::n_node; i++){
  //       for(int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         edge_attr_aggr[i][j] = most_negative_num;
  //       }
  //     }
  //   }

  //   int receiver_col;
  //   if(CONFIG_T::flow == source_to_target){
  //     receiver_col = 1;
  //   }
  //   else{
  //     receiver_col = 0;
  //   }

  //   #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
  //   for(int i=0; i<CONFIG_T::n_edge; i++){
  //     #pragma HLS UNROLL
  //     index_T r = edge_index_1D[2*i+receiver_col];
  //     num_edge_per_node[r] += 1;
  //     edge_aggr_mask[r] = 1;

  //     // if sum or mean
  //     if((CONFIG_T::aggr == aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
  //       for(int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         edge_attr_aggr[r][j] += edge_attr[i][j];
  //       }
  //     }
  //     else{ //CONFIG_T::aggr==aggr_max
  //       for(int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         edge_attr_aggr[r][j] = edge_attr[i][j] > edge_attr_aggr[r][j] ? edge_attr[i][j] : edge_attr_aggr[r][j];
  //       }
  //     }
  //   }

  //   // sum --> mean
  //   if(CONFIG_T::aggr == aggr_mean){
  //     for(int i=0; i < CONFIG_T::n_node; i++){
  //       for (int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         res_T edge_mean_j;
  //         nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], num_edge_per_node[i], edge_mean_j);
  //         edge_attr_aggr[i][j] = edge_mean_j;
  //       }
  //     }
  //   }

  //   // None --> max
  //   if(CONFIG_T::aggr == aggr_max){ //note: the edge_attr_aggr array has been initialized but IS NOT ZEROS
  //     for(int i=0; i < CONFIG_T::n_node; i++){
  //       for(int j=0; j<CONFIG_T::edge_dim; j++){
  //         #pragma HLS UNROLL
  //         edge_attr_aggr[i][j] = edge_aggr_mask[i]*edge_attr_aggr[i][j];
  //       }
  //     }
  //   }

  //   //output array --> output vec
  //   nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr, edge_attr_aggr_1D);
  // }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock_pipeline(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
    //initialize arrays
    // 1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    // 2. edge_attr (input)
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // 3. phi_input (intermediate)
    int sender_col;
    int receiver_col;
    if(CONFIG_T::flow == source_to_target){
      sender_col = 0;
      receiver_col = 1;
    }
    else{
      sender_col = 1;
      receiver_col = 0;
    }
    data_T phi_input[CONFIG_T::n_edge][2*CONFIG_T::node_dim+CONFIG_T::n_edge];
    #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
    for(int i=0; i<CONFIG_T::n_edge; i++){
      index_T s = edge_index_1D[2*i+sender_col];
      index_T r = edge_index_1D[2*i+receiver_col];

      // phi_input_i = <receiver_i, sender_i, edge_i>
      for(int j=0; j<CONFIG_T::node_dim; j++){
        #pragma HLS UNROLL
        phi_input[i][j] = node_attr[r][j];
        phi_input[i][CONFIG_T::node_dim+j] = node_attr[s][j];
      }
      for(int k=0; k<CONFIG_T::edge_dim; k++){
        #pragma HLS UNROLL
        phi_input[i][2*CONFIG_T::node_dim+k] = edge_attr[i][k];
      }
    }

    // 4. edge_update (output)
    res_T edge_update[CONFIG_T::n_edge][CONFIG_T::out_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_update complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    edge_loop: for(int i = 0; i < CONFIG_T::n_edge; i++) { //for each edge
      #pragma HLS UNROLL

      // send phi_input[i] through NN to edge_update[i]
        if(CONFIG_T::n_layers == 1){
	      nnet::dense_mult_1lyr<data_T, data_T, CONFIG_T>(phi_input[i], edge_update[i], core_edge_w0, core_edge_b0);
          }
        else if(CONFIG_T::n_layers == 2){
	      nnet::dense_mult_2lyr<data_T, data_T, CONFIG_T>(phi_input[i], edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1);
        }
        else if(CONFIG_T::n_layers == 3){
	      nnet::dense_mult_3lyr<data_T, data_T, CONFIG_T>(phi_input[i], edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2);
        }
        else if(CONFIG_T::n_layers == 4){
	      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input[i], edge_update[i], core_edge_w0, core_edge_b0, core_edge_w1, core_edge_b1, core_edge_w2, core_edge_b2, core_edge_w3, core_edge_b3);
        }
    }

    //output arrays --> output vectors
    // 1. edge_update_1D
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_update_config>(edge_update, edge_update_1D);
  }

  // template<class data_T, class res_T, typename CONFIG_T>
  //   void nodeblock_pipeline(
	// 		data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
	// 		data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
	// 		res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
	// 		typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
	// 		typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  // {
  //   //initialize arrays
  //   //1. node_attr (input)
  //   data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
  //   #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
  //   nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

  //   //2. edge_attr_aggr (input)
  //   data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
  //   #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
  //   nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

  //   // 3. node_update (output)
  //   res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];
  //   #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0

  //   #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
  //   node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
  //     #pragma HLS UNROLL

  //     // construct NN input: <node, edge_attr_aggr>
  //     data_T phi_input[CONFIG_T::edge_dim + CONFIG_T::node_dim];
  //     #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
  //     nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);

  //     // send it through NN
  //       if(CONFIG_T::n_layers == 1){
	//       nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
  //       }
  //       else if(CONFIG_T::n_layers == 2){
	//       nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
  //       }
  //       else if(CONFIG_T::n_layers == 3){
	//       nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  //       }
  //       else { // CONFIG_T::n_layers == 4
	//       nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
  //       }
  //   }

  //   // output array --> output vector
  //   nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update, node_update_1D);

  // }

  ////////////////////////////////////top-level//////////////////////////////////////
  // template<class data_T, class index_T, class res_T, typename CONFIG_T>
  //   void edge_aggregate(
  //           data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
  //           index_T   edge_index_1D[CONFIG_T::n_edge*2],
  //           res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  // {
  //   std::cout << "edge_aggregate \n";
  //     if(CONFIG_T::gnn_resource_limit){
  //       std::cout << "edge_aggregate_dataflow \n";
  //       edge_aggregate_dataflow<data_T,index_T,res_T,CONFIG_T>(edge_attr_1D,edge_index_1D,edge_attr_aggr_1D);
  //     }
  //     else{
  //       std::cout << "edge_aggregate_pipeline \n";
  //       edge_aggregate_pipeline<data_T,index_T,res_T,CONFIG_T>(edge_attr_1D,edge_index_1D,edge_attr_aggr_1D);
  //     }
  // }

  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edgeblock(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
			index_T   edge_index_1D[CONFIG_T::n_edge*2],
			res_T     edge_update_1D[CONFIG_T::n_edge*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_edge_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_edge_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_edge_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_edge_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_edge_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_edge_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_edge_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_edge_b3[CONFIG_T::dense_config4::n_out])
  {
      if(CONFIG_T::gnn_resource_limit){
        edgeblock_dataflow<data_T,index_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_1D,edge_index_1D,edge_update_1D,core_edge_w0,core_edge_b0,core_edge_w1,core_edge_b1,core_edge_w2,core_edge_b2,core_edge_w3,core_edge_b3);
      }
      else{
        edgeblock_pipeline<data_T,index_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_1D,edge_index_1D,edge_update_1D,core_edge_w0,core_edge_b0,core_edge_w1,core_edge_b1,core_edge_w2,core_edge_b2,core_edge_w3,core_edge_b3);
      }
  }

  // template<class data_T, class res_T, typename CONFIG_T>
  //   void nodeblock(
	// 		data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
	// 		data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
	// 		res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
	// 		typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
	// 		typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  // {
  //     if(CONFIG_T::gnn_resource_limit){
  //       nodeblock_dataflow<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3);
  //     }
  //     else{
  //       nodeblock_pipeline<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3);
  //     }
  // }



  /*
  Hyeon-Seo Code
  */
  template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
    void residualBlock(
      input1_T data1[CONFIG_T::n_elem],
      input2_T data2[CONFIG_T::n_elem],
      res_T res[CONFIG_T::n_elem])
    /*
    
    */
  {
    // if(CONFIG_T::gnn_resource_limit){
    //     //#pragma DATAFLOW
    //   }
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
      // if(CONFIG_T::gnn_resource_limit){
      //   #pragma HLS UNROLL
      //   //#pragma HLS PIPELINE II=1
      // }
      res[ii] = data1[ii] + data2[ii];
      // std::cout << "Residual output index: " << ii << ", output: "<< res[ii]<< ", input1: "<< data1[ii] <<", input2: "<< data2[ii]<<"\n";
    }
  }

  

  // template<class data_T, class res_T, typename CONFIG_T>
  //   void nodeblock_pipeline(
	// 		data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
	// 		data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
	// 		res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
	// 		typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
	// 		typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
	// 		typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
	// 		typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
	// 		typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out])
  // /*
  // This is the pipeline version. dataflow is not yet supported
  // */
  // {
  //   //initialize arrays
  //   //1. node_attr (input)
  //   data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
  //   #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
  //   nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

  //   //2. edge_attr_aggr (input)
  //   data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
  //   #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
  //   nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

  //   // 3. node_update (output)
  //   res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];
  //   #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0

  //   #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
  //   node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
  //     #pragma HLS UNROLL

  //     // construct NN input: <node, edge_attr_aggr>
  //     // data_T phi_input[CONFIG_T::common_dim];
  //     data_T phi_input[CONFIG_T::node_dim];
  //     #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
  //     std::cout << "c = x + aggr_out\n";
  //     // nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
  //     nnet::residualBlock<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
  //     // nnet::add<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
  //     // send it through NN
  //       if(CONFIG_T::n_layers == 1){
	//       nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
  //       }
  //       else if(CONFIG_T::n_layers == 2){
	//       nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
  //       }
  //       else if(CONFIG_T::n_layers == 3){
	//       nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  //       }
  //       else { // CONFIG_T::n_layers == 4
	//       nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
  //       }

  //     // // send it through NN
  //     //   if(CONFIG_T::n_layers == 1){
	//     //   nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0);
  //     //   }
  //     //   else if(CONFIG_T::n_layers == 2){
	//     //   nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1);
  //     //   }
  //     //   else if(CONFIG_T::n_layers == 3){
	//     //   nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
  //     //   }
  //     //   else { // CONFIG_T::n_layers == 4
	//     //   nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
  //     //   }





  //   }

  //   // output array --> output vector
  //   nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update, node_update_1D);

  // }

  template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock_residual(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::norm_config1::scale_t   norm_s0[CONFIG_T::norm_config1::n_in],
      typename CONFIG_T::norm_config1::bias_t   norm_b0[CONFIG_T::norm_config1::n_in])
  /*
  This is the pipeline version. dataflow is not yet supported
  */
  {
    //initialize arrays
    //1. node_attr (input)
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    //2. edge_attr_aggr (input)
    data_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr_1D, edge_attr_aggr);

    // //debugging
    // for (unsigned i = 0; i < CONFIG_T::n_node; i++) {
    //   for (unsigned j = 0; j < CONFIG_T::edge_dim; j++) {
    //     std::cout << "nodeblock index i:" << i << ", j:" << j << ", edge_attr_aggr:" << edge_attr_aggr[i][j] << "\n";
    //     // std::cout << "nodeblock index i:" << i << ", j:" << j << ", node_attr:" << node_attr[i][j] << "\n";
        
    //   }
    // }


    // 3. node_update (output)
    res_T node_update[CONFIG_T::n_node][CONFIG_T::out_dim];
    #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0

    // 3. node_update_update (output)
    res_T node_update_update[CONFIG_T::n_node][CONFIG_T::out_dim];
    #pragma HLS ARRAY_PARTITION variable=node_update_update complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    node_loop: for(int i = 0; i < CONFIG_T::n_node; i++){ //for each node
      // std::cout << "nodeblock index i: " << i << "\n";
      #pragma HLS UNROLL

      // construct NN input: <node, edge_attr_aggr>
      // data_T phi_input[CONFIG_T::common_dim];
      data_T phi_input[CONFIG_T::node_dim];
      #pragma HLS ARRAY_PARTITION variable=phi_input complete dim=0
      // std::cout << "c = x + aggr_out\n";
      // nnet::concatenate1d<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
      nnet::residualBlock<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
      // nnet::add<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], edge_attr_aggr[i], phi_input);
      // send it through NN
      // std::cout << "n_layers: " <<  CONFIG_T::n_layers<< "\n";


      // debugging
      // for (unsigned j = 0; j < CONFIG_T::edge_dim; j++) {
      //   std::cout << "nodeblock index i:" << i << ", j:" << j << ", phi_input:" << phi_input[j] << "\n";
      // }

      if(CONFIG_T::n_layers == 1){
      nnet::dense_mult_1lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0);
      }
      else if(CONFIG_T::n_layers == 2){
      nnet::dense_mult_2lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, norm_s0, norm_b0);
      }
      else if(CONFIG_T::n_layers == 3){
      nnet::dense_mult_3lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2);
      }
      else { // CONFIG_T::n_layers == 4
      nnet::dense_mult_4lyr<data_T, res_T, CONFIG_T>(phi_input, node_update[i], core_node_w0, core_node_b0, core_node_w1, core_node_b1, core_node_w2, core_node_b2, core_node_w3, core_node_b3);
      }

      // std::cout << "nodeblock index i: " << i << "\n";
      // for (int j=0; j<CONFIG_T::node_update_config::n_cols; j++){
      //     std::cout << "j: " << j << ", output: "<< node_update[i][j]<<"\n";
      // }

      // std::cout << "ResidualBlock row: " << i <<"\n";
      nnet::residualBlock<data_T, data_T, data_T, typename CONFIG_T::merge_config1>(node_attr[i], node_update[i], node_update_update[i]);
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::node_update_config>(node_update_update, node_update_1D);

  }

  template<class data_T, class res_T, typename CONFIG_T>
    void nodeblock(
			data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
			data_T    edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim],
			res_T     node_update_1D[CONFIG_T::n_node*CONFIG_T::out_dim],
			typename CONFIG_T::dense_config1::weight_t  core_node_w0[CONFIG_T::dense_config1::n_in*CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config1::bias_t    core_node_b0[CONFIG_T::dense_config1::n_out],
			typename CONFIG_T::dense_config2::weight_t  core_node_w1[CONFIG_T::dense_config2::n_in*CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config2::bias_t    core_node_b1[CONFIG_T::dense_config2::n_out],
			typename CONFIG_T::dense_config3::weight_t  core_node_w2[CONFIG_T::dense_config3::n_in*CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config3::bias_t    core_node_b2[CONFIG_T::dense_config3::n_out],
			typename CONFIG_T::dense_config4::weight_t  core_node_w3[CONFIG_T::dense_config4::n_in*CONFIG_T::dense_config4::n_out],
			typename CONFIG_T::dense_config4::bias_t    core_node_b3[CONFIG_T::dense_config4::n_out],
      typename CONFIG_T::norm_config1::scale_t   norm_s0[CONFIG_T::norm_config1::n_in],
      typename CONFIG_T::norm_config1::bias_t   norm_b0[CONFIG_T::norm_config1::n_in])
  {
    // std::cout << "Nodeblock start \n";
    // nodeblock_pipeline<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3);
    nodeblock_residual<data_T,res_T,CONFIG_T>(node_attr_1D,edge_attr_aggr_1D,node_update_1D,core_node_w0,core_node_b0,core_node_w1,core_node_b1,core_node_w2,core_node_b2,core_node_w3,core_node_b3, norm_s0, norm_b0);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void encoder(
      data_T    node_attr_1D[CONFIG_T::n_rows*CONFIG_T::n_in],
			res_T     node_update_1D[CONFIG_T::n_rows*CONFIG_T::n_out],
      typename CONFIG_T::weight_t  core_node_w0[CONFIG_T::n_in*CONFIG_T::n_out],
			typename CONFIG_T::bias_t    core_node_b0[CONFIG_T::n_out]
    )
  {
    // std::cout << "NodeEncoder CONFIG_T::n_out : " << CONFIG_T::n_out<<"\n";
    // std::cout << "NodeEncoder CONFIG_T::n_rows : " << CONFIG_T::n_rows<<"\n";
    // std::cout << "Encoder starting with n_in: "  << CONFIG_T::n_in<<" \n";

    //initialize arrays
    //1. node_attr (input)
    data_T node_attr[CONFIG_T::n_rows][CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::input_config>(node_attr_1D, node_attr);

    // 2. node_update (output)
    res_T node_update[CONFIG_T::n_rows][CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=node_update complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    node_loop: for(int i = 0; i < CONFIG_T::n_rows; i++){ //for each node
      #pragma HLS UNROLL

      // construct NN input: <node, edge_attr_aggr>
      // send it through NN
      // nnet::dense_resource<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0);
      nnet::dense<data_T, res_T, CONFIG_T>(node_attr[i], node_update[i], core_node_w0, core_node_b0);
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::output_config>(node_update, node_update_1D);

  }


  template<class data_T, class index_T, class res_T, typename CONFIG_T>
    void edge_aggregate(
            data_T    node_attr_1D[CONFIG_T::n_node*CONFIG_T::node_dim],
            data_T    edge_attr_1D[CONFIG_T::n_edge*CONFIG_T::edge_dim],
            index_T   edge_index_1D[CONFIG_T::n_edge*2],
            res_T     edge_attr_aggr_1D[CONFIG_T::n_node*CONFIG_T::edge_dim])
  {
    //initialize arrays
    // std::cout << "CONFIG_T::aggr: " << CONFIG_T::aggr << "\n";

    // // assign CONFIG_T::aggr to softmax bc I can't be bothered to change the code up in the chain
    // CONFIG_T::aggr = 0;
    // std::cout << "CONFIG_T::table_size: " << CONFIG_T::table_size << "\n";
    // std::cout << "CONFIG_T::aggr: " << CONFIG_T::aggr << "\n";

    // // just printing out stuff
    // for(int i=0; i<CONFIG_T::n_edge*2; i++){
    //   std::cout << "edge_index_1D index: " << i << ", value: " << edge_index_1D[i] <<"\n";
    // }

    // 1. edge_attr (input), node_attr
    data_T edge_attr[CONFIG_T::n_edge][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::edge_attr_config>(edge_attr_1D, edge_attr);

    // std::cout << "CONFIG_T::node_dim:" << CONFIG_T::node_dim << "\n";
    data_T node_attr[CONFIG_T::n_node][CONFIG_T::node_dim];
    #pragma HLS ARRAY_PARTITION variable=node_attr complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::node_attr_config>(node_attr_1D, node_attr);

    /*2. num_edge_per_node (intermediate) -> for aggr_mean, 
    edge_aggr_mask (intermediate) -> for aggr_max,
    normalization_value (intermediate) -> for aggr_softmax,
    */
    index_T num_edge_per_node[CONFIG_T::n_node]; // Think this is the list of degrees for each node
    #pragma HLS ARRAY_PARTITION variable=num_edge_per_node complete dim=0
    ap_uint<1> edge_aggr_mask[CONFIG_T::n_node];
    #pragma HLS ARRAY_PARTITION variable=edge_aggr_mask complete dim=0
    data_T normalization_value[CONFIG_T::n_node][CONFIG_T::node_dim]; // Normalization value for softmax
    #pragma HLS ARRAY_PARTITION variable=normalization_value complete dim=0
    for(int i=0; i<CONFIG_T::n_node; i++){
      #pragma HLS UNROLL
      num_edge_per_node[i] = 0;
      if(CONFIG_T::aggr==aggr_max){
        edge_aggr_mask[i] = 0;
      }
      for(int j=0; j<CONFIG_T::node_dim; j++){ // node_dim == edge_dim
        normalization_value[i][j] = 0; //initialize normalization for softmax aggr
      }
    }
    

    // Initialize the lookup tables for softmax aggr
    #ifdef __HLS_SYN__
        bool initialized = false;
        data_T exp_table[CONFIG_T::table_size];
        data_T invert_table[CONFIG_T::table_size];
    #else
        bool initialized = false;
        data_T exp_table[CONFIG_T::table_size];
        data_T invert_table[CONFIG_T::table_size];

    #endif
        if (!initialized) {
            // std::cout << "not initialized, idk what that means tho \n";
            // Note we are exponentiating the inputs, which have type data_T
            init_exp_table<data_T, CONFIG_T>(exp_table);
            // Note we are inverting the exponentials, which have type exp_table_t
            init_invert_table<data_T, CONFIG_T>(invert_table);
            initialized = true;
        }


    //3. edge_attr_aggr (output)
    res_T edge_attr_aggr[CONFIG_T::n_node][CONFIG_T::edge_dim];
    #pragma HLS ARRAY_PARTITION variable=edge_attr_aggr complete dim=0
    if((CONFIG_T::aggr==aggr_sum)||(CONFIG_T::aggr==aggr_mean)||(CONFIG_T::aggr==aggr_softmax)){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = 0;
        }
      }
    }
    else{ //CONFIG_T:aggr==aggr_max, we want to initialize this with the most negative number we can represent ->why?
      res_T most_negative_num = -hls::numeric_limits<res_T>::max();
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = most_negative_num;
        }
      }
    }

    int receiver_col;
    int sender_col;
    if(CONFIG_T::flow == source_to_target){
      receiver_col = 1;
      sender_col = 0;
    }
    else{
      receiver_col = 0;
      sender_col = 1;
    }

    
    // for(int i=0; i<CONFIG_T::n_node; i++){
    //   for(int j = 0; j < CONFIG_T::node_dim; j++){
    //     std::cout << "node_attr index i:" << i << ", j:" << j << ", node_attr:" << node_attr[i][j] << "\n";
    //   }
    // }

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    for(int i=0; i<CONFIG_T::n_edge; i++){
      #pragma HLS UNROLL
      index_T r = edge_index_1D[2*i+receiver_col];
      num_edge_per_node[r] += 1;
      edge_aggr_mask[r] = 1;
      index_T s = edge_index_1D[2*i+sender_col];
      // std::cout << "index r: " << r << "\n";
      // std::cout << "n_edge: " << CONFIG_T::n_edge << "\n";
      
      // std::cout << "index s: " << s << "\n";

      // if sum or mean
      if((CONFIG_T::aggr == aggr_sum)||(CONFIG_T::aggr==aggr_mean)){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] += edge_attr[i][j];
        }
      }
      else if ((CONFIG_T::aggr==aggr_softmax)){
        // Calculate all the e^x's
        data_T exp_sum(0);
        data_T beta = CONFIG_T::Beta;
        
        for(unsigned j = 0; j < CONFIG_T::edge_dim; j++){
            #pragma HLS unroll
            // unsigned x = softmax_idx_from_real_val<data_T, CONFIG_T>(
            //   edge_attr[i][j] * beta // may have to convert Beta to data_T first
            // );
            // data_T exp_x = exp_table[x];
            // data_T msg;
            // #pragma HLS ARRAY_PARTITION variable=msg complete dim=0
            // nnet::relu<data_T, data_T, typename CONFIG_T>(edge_attr[i][j] + node_attr[s][j], data0);
            data_T eps = CONFIG_T::eps;
            data_T msg =  relu_0D(edge_attr[i][j] + node_attr[s][j]) + eps;
            data_T exp_x = exp_fcn_float(msg * beta);
            edge_attr_aggr[r][j] += exp_x*msg;
            normalization_value[r][j] += exp_x;
            // std::cout << "index s: " << s << "\n";
            // std::cout << "index r: " << r << "\n";

            // std::cout << "aggregate index i:" << i << ", j:" << j << ", edge_attr: " << edge_attr[i][j] << ", node_attr:" << node_attr[s][j] << "\n";
            // std::cout << "aggregate index i:" << i << ", j:" << j << ", msg b4: " << edge_attr[i][j] + node_attr[s][j] << ", msg after: " << msg << "\n";
            // std::cout << "aggregate index i:" << i << ", j:" << j << ", exp_x:" << exp_x << ", exp_x*msg: " << exp_x*msg << "\n";
            // std::cout << "current edge_attr_aggr index: "<< r << ", val: " << edge_attr_aggr[r][j] << "\n";

            
        }
        // // debugging
        // for(unsigned j = 0; j < CONFIG_T::edge_dim; j++){
        //   std::cout << "aggregate index i:" << i << ", j:" << j << ", edge_attr: " << edge_attr[i][j] << "\n";
        // }
        // // Explicitly sum the results with an adder tree.
        // // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        // Op_add<data_T> op_add;
        // exp_sum = reduce<data_T, CONFIG_T::edge_dim, Op_add<data_T>>(edge_attr_aggr[r], op_add);

        // typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
        // for(unsigned i = 0; i < CONFIG_T::n_in; i++){
        //     #pragma HLS unroll
        //     res[i] = exp_res[i] * inv_exp_sum;
        // }  
      }
      else{ //CONFIG_T::aggr==aggr_max
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[r][j] = edge_attr[i][j] > edge_attr_aggr[r][j] ? edge_attr[i][j] : edge_attr_aggr[r][j];
        }
      }
    }

    // sum --> mean
    if(CONFIG_T::aggr == aggr_mean){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for (int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          res_T edge_mean_j;
          nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], num_edge_per_node[i], edge_mean_j);
          edge_attr_aggr[i][j] = edge_mean_j;
        }
      }
    }
    else if(CONFIG_T::aggr == aggr_softmax){
      for(int i=0; i < CONFIG_T::n_node; i++){
        for (int j=0; j<CONFIG_T::edge_dim; j++){
          // #pragma HLS UNROLL
          // res_T normalized_val_j;
          // nnet::edge_divide<res_T, index_T, res_T, CONFIG_T>(edge_attr_aggr[i][j], normalization_value[i], normalized_val_j);
          // edge_attr_aggr[i][j] = normalized_val_j;


          // if (normalization_value[i][j] != 0){
          //   edge_attr_aggr[i][j] = edge_attr_aggr[i][j]/ normalization_value[i][j];
          // }
          edge_attr_aggr[i][j] = edge_attr_aggr[i][j]/ normalization_value[i][j];


          
          // std::cout << "final aggregate index i:" << i << ", j:" << j << ", normalization_value: "<<normalization_value[i][j]<<", attr output: " << edge_attr_aggr[i][j] << "\n";
          
          // std::cout << "final aggregate index i:" << i << ", j:" << j << ", normalization_value: "<<normalization_value[i][j]<<", attr output: " << edge_attr_aggr[i][j] << "\n";
        }
      }
    }

    // None --> max
    if(CONFIG_T::aggr == aggr_max){ //note: the edge_attr_aggr array has been initialized but IS NOT ZEROS
      for(int i=0; i < CONFIG_T::n_node; i++){
        for(int j=0; j<CONFIG_T::edge_dim; j++){
          #pragma HLS UNROLL
          edge_attr_aggr[i][j] = edge_aggr_mask[i]*edge_attr_aggr[i][j];
        }
      }
    }

    //output array --> output vec
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::edge_attr_aggr_config>(edge_attr_aggr, edge_attr_aggr_1D);
  }

  template<class data_T, class res_T, typename CONFIG_T>
    void normalize_2D(
      data_T    data_1D[CONFIG_T::n_rows*CONFIG_T::n_in],
			res_T     res_1D[CONFIG_T::n_rows*CONFIG_T::n_in],
      typename CONFIG_T::scale_t  scale[CONFIG_T::n_in],
      typename CONFIG_T::bias_t   bias[CONFIG_T::n_in]
    )
    /*
    normalize, but for 2 dimensional data like graphs
    */
  {

    //initialize arrays
    //1. data (input)
    data_T data[CONFIG_T::n_rows][CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=data complete dim=0
    nnet::vec_to_mat<data_T, data_T, typename CONFIG_T::matrix_op_config>(data_1D, data);

    // 2. res (output)
    res_T res[CONFIG_T::n_rows][CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=res complete dim=0

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    //for each node/edge, we normalize
    for(int i = 0; i < CONFIG_T::n_rows; i++){ 
      #pragma HLS UNROLL

      // construct NN input: <node, edge_attr_aggr>
      // send it through NN
      // std::cout << "normalize row: " << i << "\n";
      nnet::normalize<data_T, res_T, CONFIG_T>(data[i], res[i], scale, bias);
    }

    // output array --> output vector
    nnet::mat_to_vec<res_T, res_T, typename CONFIG_T::matrix_op_config>(res, res_1D);

  }

}
#endif
