#ifndef DEFINES_H_
#define DEFINES_H_

#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/sycl.hpp>

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 32
#define N_INPUT_2_1 3
#define N_OUTPUTS_2 32
#define N_FILT_2 16
#define N_OUTPUTS_2 32
#define N_FILT_2 16
#define N_OUT_5 16

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ac_fixed<16,6,true>, 96*1> input_t;
typedef ac_fixed<16,6,true> model_default_t;
typedef nnet::array<ac_fixed<16,6,true>, 512*1> layer2_t;
typedef nnet::array<ac_fixed<16,6,true>, 144*1> w2_t;
typedef nnet::array<ac_fixed<16,6,true>, 16*1> b2_t;
typedef nnet::array<ac_fixed<16,6,true>, 512*1> layer4_t;
typedef ac_fixed<18,8,true> relu1_table_t;
typedef nnet::array<ac_fixed<16,6,true>, 16*1> result_t;
typedef nnet::array<ac_fixed<16,6,true>, 768*1> w5_t;
typedef nnet::array<ac_fixed<16,6,true>, 768*1> wr5_t;
typedef nnet::array<ac_fixed<16,6,true>, 48*1> b5_t;
typedef nnet::array<ac_fixed<16,6,true>, 48*1> br5_t;
typedef ac_fixed<18,8,true> gru_table_t;

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
