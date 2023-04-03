#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#ifndef __INTELFPGA_COMPILER__
#include "ac_fixed.h"
#include "ac_int.h"
#define hls_register
#else
#include "HLS/ac_fixed.h"
#include "HLS/ac_int.h"
#include "HLS/hls.h"
#endif

// Streams are explicitly defined in defines.h, which are included for parameters.h
// Defining them again in this file will cause compile-time errors
#include "defines.h"

// If using io_parallel, inputs and output need to be initialised before calling the top-level function
// If using io_stream, no inputs/outputs are initialised, as they are passed by reference to the top-level function
// hls-fpga-machine-learning insert inputs
// hls-fpga-machine-learning insert outputs

#ifndef __INTELFPGA_COMPILER__
/*
* The top-level function used during GCC compilation / hls4ml.predic(...) goes here
* An important distinction is made between io_stream and io_parallel:
*     (1) io_parallel:
               - Top-level function takes a struct containing an array as function argument
               - Returns a struct containing an array - the prediction
      (2) io_stream:
               - Top-level function is 'void' - no return value
               - Instead, both the input and output are passed by reference
               - This is due the HLS Streaming Interfaces; stream cannot be copied (implicitly deleted copy constructor)
* This distinction is handled in quartus_writer.py
*/
// hls-fpga-machine-learning instantiate GCC top-level
#else
// Maximum initiation interval, concurrency and frequency for HLS syntheis are defined here
// hls-fpga-machine-learning insert cpragmas

/*
 * The top-level function used during HLS Synthesis goes here
 * In a similar manner to GCC, there is a distinction between io_stream & io_parallel
 */
// hls-fpga-machine-learning instantiate HLS top-level
#endif

#endif
