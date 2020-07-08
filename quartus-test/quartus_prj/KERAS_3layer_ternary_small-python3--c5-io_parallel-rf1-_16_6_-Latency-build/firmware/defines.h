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
typedef <16,6> model_default_t;
typedef <16,6> input_t;
typedef <16,6> layer2_t;
typedef ac_int<2, true> weight2_t;
typedef ac_int<1, false> bias2_t;
typedef ac_int<2, true> layer17_t;
typedef <16,6> threshold_hi_17_t;
typedef <16,6> threshold_lo_17_t;
typedef ac_int<8, true> layer6_t;
typedef ac_int<2, true> weight6_t;
typedef ac_int<2, true> bias6_t;
typedef ac_int<2, true> layer18_t;
typedef ac_int<8, true> threshold_hi_18_t;
typedef ac_int<8, true> threshold_lo_18_t;
typedef ac_int<7, true> layer10_t;
typedef ac_int<2, true> weight10_t;
typedef ac_int<2, true> bias10_t;
typedef ac_int<2, true> layer19_t;
typedef ac_int<7, true> threshold_hi_19_t;
typedef ac_int<7, true> threshold_lo_19_t;
typedef ac_int<7, true> layer14_t;
typedef ac_int<2, true> weight14_t;
typedef ac_int<2, true> bias14_t;
typedef <16,6> result_t;


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
