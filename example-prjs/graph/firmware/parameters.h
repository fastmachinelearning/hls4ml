#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_sublayer.h"
#include "nnet_graph.h"
#include "nnet_common.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<16,6> accum_default_t;
typedef ap_fixed<16,6> weight_default_t;
typedef ap_fixed<16,6> bias_default_t;
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
//typedef float accum_default_t;
//typedef float weight_default_t;
//typedef float bias_default_t;
//typedef float input_t;
//typedef float result_t;
#define N_FEATURES 3
#define N_HIDDEN_FEATURES 4
#define N_NODES 4
#define N_EDGES 4

//hls-fpga-machine-learning insert layer-config
struct graph_config1 : nnet::graph_config {
  static const unsigned n_node = N_NODES;
  static const unsigned n_edge = N_EDGES;
  static const unsigned n_input_dim = N_FEATURES+N_HIDDEN_FEATURES;
  static const unsigned n_hidden_dim = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};
                                                                                              
struct layer_config1 : nnet::layer_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_FEATURES;
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config1 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

struct layer_config2 : nnet::layer_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = 2*(N_FEATURES+N_HIDDEN_FEATURES);
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config2 : nnet::activ_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

struct layer_config3 : nnet::layer_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned n_out = 1;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct sigmoid_config1 : nnet::activ_config {
  static const unsigned n_batch = N_EDGES;
  static const unsigned n_in = 1;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

struct layer_config4 : nnet::layer_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = 3*(N_FEATURES+N_HIDDEN_FEATURES);
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config3 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

struct tanh_config4 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

struct layer_config5 : nnet::layer_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned n_out = N_HIDDEN_FEATURES;
  static const unsigned io_type = nnet::io_parallel;
  static const unsigned reuse_factor = 1;
  static const unsigned n_zeros = 0;
  static const bool store_weights_in_bram = false;
  typedef accum_default_t accum_t;
  typedef bias_default_t bias_t;
  typedef weight_default_t weight_t;
};

struct tanh_config5 : nnet::activ_config {
  static const unsigned n_batch = N_NODES;
  static const unsigned n_in = N_HIDDEN_FEATURES;
  static const unsigned table_size = 4096;
  static const unsigned io_type = nnet::io_parallel;
};

#endif 
