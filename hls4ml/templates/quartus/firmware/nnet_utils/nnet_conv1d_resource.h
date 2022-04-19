#ifndef NNET_CONV1D_RESOURCE_H_
#define NNET_CONV1D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

// ****************************************************************
//      im2col - General-purpose 2D Convolution algorithm
// ****************************************************************

template<class data_T, typename CONFIG_T>
void im2col_1d_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan], data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan], const int col) {
    // im2col can be unrolled fully, since number of parallel executions = filt_w x n_chann ~ O(100) and very little DSP usage

    hls_register int index = 0;
    
    KernelLoop:
    #pragma unroll
    for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
        ChannelLoop:
        #pragma unroll
        for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
            hls_register int index_data = (col * CONFIG_T::stride_width + kernel_col - CONFIG_T::pad_left) * CONFIG_T::n_chan + channel;
            if (index_data >= 0 && index_data < CONFIG_T::in_width * CONFIG_T::n_chan) {
                data_col[index++] = data[index_data];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_im2col_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    static constexpr int pf = MIN(CONFIG_T::parallelisation_factor, CONFIG_T::out_width);

    ColLoop:
    #pragma unroll pf
    #pragma ii CONFIG_T::reuse_factor
    for (int i = 0; i < CONFIG_T::out_width; i++) {
        // Loop variables should always be declared in the deepest scope available
        // See Intel's HLS - Loop Best Practices https://www.intel.com/content/www/us/en/docs/programmable/683152/22-2/declare-variables-in-the-deepest-scope.html 
            
        hls_register data_T data_col[CONFIG_T::filt_width * CONFIG_T::n_chan];
        im2col_1d_cl<data_T, CONFIG_T>(data, data_col, i);
        
        hls_register res_T res_col[CONFIG_T::n_filt];
        dense_resource<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
        
        // Unroll fully, since
        // (1) n_filt is usually low in io_parallel (< 32)
        // (2) no complex operations handled in loop, this loop performs a simple register writing operation    
        FiltLoop:
        #pragma unroll
        for (int j = 0; j < CONFIG_T::n_filt; j++) {
            res[i * CONFIG_T::n_filt + j] = res_col[j];
        }
    }
}

// ****************************************************************
//      Top-level function - handles different implementations
// ****************************************************************

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_resource_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    const typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    const typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
) {
    conv_1d_im2col_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
}

}
#endif
