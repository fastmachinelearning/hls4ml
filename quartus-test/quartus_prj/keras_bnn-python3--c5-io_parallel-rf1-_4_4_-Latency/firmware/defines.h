#ifndef DEFINES_H_
#define DEFINES_H_

#include <complex>
#ifndef __INTELFPGA_COMPILER__
#include "ref/ac_int.h"
#include "ref/ac_fixed.h"
#else
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_6 32
#define N_LAYER_10 32
#define N_LAYER_14 5


//hls-fpga-machine-learning insert layer-precision
typedef <4,4> model_default_t;
typedef <4,4> input_t;
typedef <4,4> layer2_t;
typedef ac_int<2, true> weight2_t;
typedef ac_int<1, false> bias2_t;
typedef ac_int<1, false> layer17_t;
typedef <4,4> threshold17_t;
typedef ac_int<8, true> layer6_t;
typedef ac_int<1, false> weight6_t;
typedef ac_int<1, false> bias6_t;
typedef ac_int<1, false> layer18_t;
typedef ac_int<8, true> threshold18_t;
typedef ac_int<7, true> layer10_t;
typedef ac_int<1, false> weight10_t;
typedef ac_int<1, false> bias10_t;
typedef ac_int<1, false> layer19_t;
typedef ac_int<7, true> threshold19_t;
typedef ac_int<7, true> layer14_t;
typedef ac_int<1, false> weight14_t;
typedef ac_int<1, false> bias14_t;
typedef <4,4> result_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
