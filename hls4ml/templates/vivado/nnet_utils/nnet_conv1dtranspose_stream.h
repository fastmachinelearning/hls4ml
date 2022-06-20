#ifndef NNET_CONV1DTRANSPOSE_STREAM_H
#define NNET_CONV1DTRANSPOSE_STREAM_H

#include "nnet_common.h"
#include "nnet_conv_stream.h"
#include "hls_stream.h"

namespace nnet {

template <class data_T, typename CONFIG_T>
void kernel_shift_tr_1d(
    const data_T& in_elem,
    typename data_T::value_type kernel_window[CONFIG_T::trfilt_width * CONFIG_T::n_chan]
) {
    #pragma HLS INLINE
    
    // Shift kernel_window by one step to the left (manual shift operation)
    static const int filt_width = CONFIG_T::trfilt_width - 1;
    KernelShiftWidth: for (int i_iw = 0; i_iw < filt_width; i_iw++) {
        #pragma HLS PIPELINE II = 1
        KernelShiftChannel: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
            #pragma HLS UNROLL
            // Shift every element in kernel_window to the left
            kernel_window[i_iw * CONFIG_T::n_chan + i_ic] = kernel_window[(i_iw + 1) * CONFIG_T::n_chan + i_ic];
        }
    }

    // Insert shift_buffer column into right-most column of kernel
    static const int lastheight = (CONFIG_T::trfilt_width - 1) * CONFIG_T::n_chan;
    KernelPushChannel: for (int i_ic = 0; i_ic < CONFIG_T::n_chan; i_ic++) {
        #pragma HLS UNROLL
        kernel_window[lastheight + i_ic] = in_elem[i_ic];
    }
}

// Conv 1D transpose compute output
template<class data_T, class res_T, typename CONFIG_T>
void compute_output_buffer_tr_1d(
    const data_T& in_elem,
    hls::stream<res_T> &res_stream,
    typename CONFIG_T::weight_t weights[CONFIG_T::stride_width][
        CONFIG_T::trfilt_width * CONFIG_T::n_filt * CONFIG_T::n_chan
    ],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]
) 
{
    #pragma HLS INLINE

    // Thresholds
    const static int lShiftX = CONFIG_T::trfilt_width - 1;

    // Counters
    static int pX = 0; // pixel counter
    static int oX = 0; // output counter (deals with 'padding')

    static typename data_T::value_type kernel_data[CONFIG_T::trfilt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete

    typename res_T::value_type res_out[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res_out complete dim = 0

    res_T res_pack;
    #pragma HLS DATA_PACK variable=res_pack

    // Add pixel to buffer
    nnet::kernel_shift_tr_1d<data_T, CONFIG_T>(in_elem, kernel_data);

    //always do stride number of multiplications
    StrideLoop: for (int idx = 0; idx < CONFIG_T::stride_width; idx++) {
        #pragma HLS UNROLL
        #pragma HLS INLINE region
        // Dense multiply
        if (CONFIG_T::strategy == nnet::latency) {
            dense_latency<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
                kernel_data, res_out, weights[idx], biases);
        } else {
            dense_resource<typename data_T::value_type, typename res_T::value_type, typename CONFIG_T::mult_config>(
                kernel_data, res_out, weights[idx], biases);
        }

        // Pack output
        if (oX >= CONFIG_T::pad_left && oX < CONFIG_T::pad_left + CONFIG_T::out_width) {
            CastLoop: for (unsigned i_ic = 0; i_ic < CONFIG_T::n_filt; i_ic++) {
                #pragma HLS UNROLL
                res_pack[i_ic] = res_out[i_ic];
            }
            res_stream.write(res_pack);
        }
        // Write output to stream when output ready
        oX++;
    }

    // static var housekeeping
    if (pX + 1 == CONFIG_T::in_width)  // done with all of the inputs
    {
        pX = 0;
        oX = 0;
    } else {
        pX = pX + 1;
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_buffer_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T>  &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::stride_width][
        CONFIG_T::trfilt_width * CONFIG_T::n_filt * CONFIG_T::n_chan
    ],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    ReadInputWidth: for (unsigned i_iw = 0; i_iw < CONFIG_T::in_width; i_iw++) {
        #pragma HLS LOOP_FLATTEN
        // if (CONFIG_T::strategy == nnet::latency) {
        //     #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
        // }
        compute_output_buffer_tr_1d<data_T, res_T, CONFIG_T>(data.read(), res, weights, biases);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_transpose_cl(
    hls::stream<data_T> &data,
    hls::stream<res_T> &res,
    typename CONFIG_T::weight_t weights[CONFIG_T::stride_width][
        CONFIG_T::trfilt_width * CONFIG_T::n_filt * CONFIG_T::n_chan
    ],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    switch(CONFIG_T::implementation) {
        #pragma HLS inline region
        case conv_implementation::linebuffer:
            conv_1d_transpose_buffer_cl<data_T, res_T, CONFIG_T>(data, res, weights, biases);
            break;
    }
}

}
#endif
//NEED TO PAD INPUT OR CLEAR KERNEL
