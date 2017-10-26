#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers

typedef ap_fixed<32,10> accum_t;
typedef ap_fixed<32,8> weight_t;
typedef ap_fixed<32,8> bias_t;
typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> layer1_t;
typedef ap_fixed<32,8> layer2_t;
typedef ap_fixed<32,8> layer3_t;
typedef ap_fixed<32,8> result_t;

#endif 
