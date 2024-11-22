#ifndef NNET_CONV1D_LATENCY_H_
#define NNET_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include <cstdlib>

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void conv_1d_latency_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                        res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                        typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
                        typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    constexpr unsigned mult_n_in = CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

    typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete

    typename CONFIG_T::accum_t acc[mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    // Limit multipliers to control parallelization
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit

PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor rewind

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

    PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            data_T cache;

        // Do the matrix-multiply
        Product1:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
                cache = data_buf[i_pxl][i_in];
            Product2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    mult[i_in * mult_n_out + i_out] =
                        CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(
                            cache, weights[i_in * mult_n_out + i_out]);
                }
            }

        // Initialize accumulator with input biases
        ResetAccum:
            for (int i_acc = 0; i_acc < mult_n_out; i_acc++) {
                #pragma HLS UNROLL
                acc[i_acc] = (typename CONFIG_T::accum_t)biases[i_acc];
            }

        // Accumulate multiplication result
        Accum1:
            for (int i_in = 0; i_in < mult_n_in; i_in++) {
                #pragma HLS UNROLL
            Accum2:
                for (int i_out = 0; i_out < mult_n_out; i_out++) {
                    #pragma HLS UNROLL
                    acc[i_out] += mult[i_in * mult_n_out + i_out];
                }
            }

        // Cast to "res_t" type
        Result:
            for (int i_res = 0; i_res < mult_n_out; i_res++) {
                #pragma HLS UNROLL
                res[i_part * CONFIG_T::n_pixels * mult_n_out + i_pxl * mult_n_out + i_res] =
                    cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_res]);
            }
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor],
                                  res_T res[CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor],
                                  typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                                  typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    assert(CONFIG_T::filt_width == 1);

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width / CONFIG_T::reuse_factor][CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=0
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::mult_config::multiplier_limit

// Convolve, saving all multiplication results to accumulate later
ConvOut:
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
    ConvFilt:
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        ConvChan:
            for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                #pragma HLS UNROLL
                int index_mult = ii * CONFIG_T::n_filt * CONFIG_T::n_chan + ff * CONFIG_T::n_chan + cc;
                int index_weight = cc * CONFIG_T::n_filt + ff;
                int index_data = (ii * CONFIG_T::stride_width - CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                if ((ii * CONFIG_T::stride_width) < CONFIG_T::pad_left ||
                    (ii * CONFIG_T::stride_width) >= (CONFIG_T::pad_left + CONFIG_T::in_width)) {
                    mult[index_mult] = 0;
                } else {
                    mult[index_mult] = CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::weight_t>::product(
                        data[index_data], weights[index_weight]);
                }
            } // end channel loop
        }     // end filter loop
    }         // end output loop

    // Initialize accumulator with input biases
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            #pragma HLS UNROLL
            acc[ii][ff] = biases[ff];
        }
    }

// Accumulate multiplication result
AccumOut:
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
    AccumFilt:
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        // Do "dot product" sum within filter and sum over channels
        AccumChan:
            for (int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult = ii * CONFIG_T::n_filt * CONFIG_T::n_chan + ff * CONFIG_T::n_chan + cc;
                acc[ii][ff] += mult[index_mult];
            } // end channel loop
        }     // end filter loop
    }         // end output loop

    // Cast to "res_t" type
    for (int ii = 0; ii < CONFIG_T::out_width / CONFIG_T::reuse_factor; ii++) {
        for (int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            #pragma HLS UNROLL
            res[ii * CONFIG_T::n_filt + ff] = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[ii][ff]);
        }
    }
}

} // namespace nnet
#endif
