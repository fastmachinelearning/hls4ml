#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#ifdef __INTELFPGA_COMPILER__
#include "HLS/hls.h"
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif

//hls-fpga-machine-learning insert numbers


//hls-fpga-machine-learning insert layer-precision


#define DIV_ROUNDUP(n,d) ((n + d - 1) / d)
#define MIN(n,d) (n > d ? d : n)

#endif
