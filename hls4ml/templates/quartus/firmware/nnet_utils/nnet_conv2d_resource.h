#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_helpers.h"

namespace nnet {

// ****************************************************************
//      im2col - General-purpose 2D Convolution algorithm
// ****************************************************************

template<class data_T, typename CONFIG_T>
void im2col_2d_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
    const int row,
    const int col
) {
    // im2col can be unrolled fully, since number of parallel executions = filt_h x filt_w x n_chann ~ O(100) and very little DSP usage

    hls_register int index = 0;

    FiltHeightLoop:    
    #pragma unroll
    for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
        hls_register int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
        
        FiltWidthLoop:
        #pragma unroll
        for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
            hls_register int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
            
            ChannelLoop:
            #pragma unroll
            for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
                if (input_row >= 0 && input_row < CONFIG_T::in_height && input_col >= 0 && input_col < CONFIG_T::in_width) {
                    data_col[index++] = data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
                } else {
                    data_col[index++] = 0;
                }
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_im2col_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {  
    // Unroll factor for loop traversing input image, derived from parallelisation_factor
    // Outer loop only gets unrolled after inner loop is fully unrolled
    static constexpr int pfc = MIN(CONFIG_T::parallelisation_factor, CONFIG_T::out_width);
    static constexpr int pfr = MIN((CONFIG_T::parallelisation_factor / pfc), CONFIG_T::out_height);

    HeightLoop: 
    #pragma unroll pfr
    for (int i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop: 
        #pragma unroll pfc
        #pragma ii CONFIG_T::reuse_factor
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            // Loop variables should always be declared in the deepest scope available
            // See Intel's HLS - Loop Best Practices https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html 
            
            hls_register data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
            im2col_2d_cl<data_T, CONFIG_T>(data, data_col, i, j);
            
            hls_register res_T res_col[CONFIG_T::n_filt];
            dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            
            // Unroll fully, since
            // (1) n_filt is usually low in io_parallel (< 32)
            // (2) no complex operations handled in loop, this loop performs a simple register writing operation
            FiltLoop: 
            #pragma unroll
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

// ****************************************************************
//      Top-level function - handles different implementations
// ****************************************************************

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    conv_2d_im2col_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

}

#endif