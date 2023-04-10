#ifndef DEFINES_H_
#define DEFINES_H_

/*
 * Intel HLS makes use of three streaming interfaces:
 *   (1) stream_in - used as the main input to a component
 *   (2) stream_out - used as the main output of a component
 *   (3) stream - allows both reading and writing; used for inter-component connections
 * ihc::stream has a implicitly deleted constructor and therefore, cannot be used as the output of a function/component
 * Therefore, variables of type 'stream' are always passed by reference
 */

#ifndef __INTELFPGA_COMPILER__

#include "ac_fixed.h"
#include "ac_int.h"
#define hls_register

#include "stream.h"
template <typename T> using stream = nnet::stream<T>;
template <typename T> using stream_in = nnet::stream<T>;
template <typename T> using stream_out = nnet::stream<T>;

#else

#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"

template <typename T> using stream = ihc::stream<T>;
template <typename T> using stream_in = ihc::stream_in<T>;
template <typename T> using stream_out = ihc::stream_out<T>;

#endif

// Include nnet::array - a custom array-like struct, mainly used with io_stream
#include "nnet_utils/nnet_types.h"

// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision

#define DIV_ROUNDUP(n, d) ((n + d - 1) / d)
#define MIN(n, d) (n > d ? d : n)
#define MAX(n, d) (n < d ? d : n)

#endif
