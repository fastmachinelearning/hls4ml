#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 8
#define N_LAYER_2 4
#define N_LAYER_2 4
#define N_LAYER_5 2
#define N_LAYER_5 2

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<16,6,true>, 8*1> input_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef nnet::array<ac_fixed<36,16,true>, 4*1> fc1_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 32*1> w2_t;
typedef nnet::array<ac_fixed<16,6,true>, 4*1> b2_t;
typedef ac_int<1, false> layer2_index;
typedef nnet::array<ac_fixed<16,6,true>, 4*1> layer4_t;
typedef ac_fixed<18,8,true> relu1_table_t;
typedef nnet::array<ac_fixed<35,15,true>, 2*1> fc2_result_t;
typedef nnet::array<ac_fixed<16,6,true>, 8*1> w5_t;
typedef nnet::array<ac_fixed<16,6,true>, 2*1> b5_t;
typedef ac_int<1, false> layer5_index;
typedef nnet::array<ac_fixed<16,6,true>, 2*1> result_t;
typedef ac_fixed<18,8,true> relu2_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
