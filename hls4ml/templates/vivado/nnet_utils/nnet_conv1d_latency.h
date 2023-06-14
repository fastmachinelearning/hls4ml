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
                *(res++) = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_res]);
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
    int multiplier_limit =
        ceil((float(CONFIG_T::out_width) / float(CONFIG_T::reuse_factor) * CONFIG_T::n_filt * CONFIG_T::n_chan) /
             float(CONFIG_T::reuse_factor));
#pragma HLS ALLOCATION operation instances=mul limit=multiplier_limit

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
                    mult[index_mult] = data[index_data] * weights[index_weight];
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
            res[ii * CONFIG_T::n_filt + ff] = (res_T)(acc[ii][ff]);
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl_split_by_rf(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                                              res_T res[CONFIG_T::out_width * CONFIG_T::n_filt],
                                              typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                                              typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {

    data_T data_tmp[CONFIG_T::reuse_factor][CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=data_tmp complete dim=0
    res_T res_tmp[CONFIG_T::reuse_factor][CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=res_tmp complete dim=0

RFInputLoop:
    for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
    #pragma HLS UNROLL
    InnerInputLoop:
        for (int ii = 0; ii < CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor; ii++) {
            #pragma HLS UNROLL
            data_tmp[jj][ii] = data[jj * CONFIG_T::in_width * CONFIG_T::n_chan / CONFIG_T::reuse_factor + ii];
        }
    }

    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[0], res_tmp[0], weights, biases);
    pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[1], res_tmp[1], weights, biases);
    if (CONFIG_T::reuse_factor > 2)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[2], res_tmp[2], weights, biases);
    if (CONFIG_T::reuse_factor > 3)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[3], res_tmp[3], weights, biases);
    if (CONFIG_T::reuse_factor > 4)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[4], res_tmp[4], weights, biases);
    if (CONFIG_T::reuse_factor > 5)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[5], res_tmp[5], weights, biases);
    if (CONFIG_T::reuse_factor > 6)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[6], res_tmp[6], weights, biases);
    if (CONFIG_T::reuse_factor > 7)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[7], res_tmp[7], weights, biases);
    if (CONFIG_T::reuse_factor > 8)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[8], res_tmp[8], weights, biases);
    if (CONFIG_T::reuse_factor > 9)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[9], res_tmp[9], weights, biases);
    if (CONFIG_T::reuse_factor > 10)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[10], res_tmp[10], weights, biases);
    if (CONFIG_T::reuse_factor > 11)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[11], res_tmp[11], weights, biases);
    if (CONFIG_T::reuse_factor > 12)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[12], res_tmp[12], weights, biases);
    if (CONFIG_T::reuse_factor > 13)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[13], res_tmp[13], weights, biases);
    if (CONFIG_T::reuse_factor > 14)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[14], res_tmp[14], weights, biases);
    if (CONFIG_T::reuse_factor > 15)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[15], res_tmp[15], weights, biases);
    if (CONFIG_T::reuse_factor > 16)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[16], res_tmp[16], weights, biases);
    if (CONFIG_T::reuse_factor > 17)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[17], res_tmp[17], weights, biases);
    if (CONFIG_T::reuse_factor > 18)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[18], res_tmp[18], weights, biases);
    if (CONFIG_T::reuse_factor > 19)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[19], res_tmp[19], weights, biases);
    if (CONFIG_T::reuse_factor > 20)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[20], res_tmp[20], weights, biases);
    if (CONFIG_T::reuse_factor > 21)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[21], res_tmp[21], weights, biases);
    if (CONFIG_T::reuse_factor > 22)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[22], res_tmp[22], weights, biases);
    if (CONFIG_T::reuse_factor > 23)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[23], res_tmp[23], weights, biases);
    if (CONFIG_T::reuse_factor > 24)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[24], res_tmp[24], weights, biases);
    if (CONFIG_T::reuse_factor > 25)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[25], res_tmp[25], weights, biases);
    if (CONFIG_T::reuse_factor > 26)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[26], res_tmp[26], weights, biases);
    if (CONFIG_T::reuse_factor > 27)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[27], res_tmp[27], weights, biases);
    if (CONFIG_T::reuse_factor > 28)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[28], res_tmp[28], weights, biases);
    if (CONFIG_T::reuse_factor > 29)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[29], res_tmp[29], weights, biases);
    if (CONFIG_T::reuse_factor > 30)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[30], res_tmp[30], weights, biases);
    if (CONFIG_T::reuse_factor > 31)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[31], res_tmp[31], weights, biases);
    if (CONFIG_T::reuse_factor > 32)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[32], res_tmp[32], weights, biases);
    if (CONFIG_T::reuse_factor > 33)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[33], res_tmp[33], weights, biases);
    if (CONFIG_T::reuse_factor > 34)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[34], res_tmp[34], weights, biases);
    if (CONFIG_T::reuse_factor > 35)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[35], res_tmp[35], weights, biases);
    if (CONFIG_T::reuse_factor > 36)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[36], res_tmp[36], weights, biases);
    if (CONFIG_T::reuse_factor > 37)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[37], res_tmp[37], weights, biases);
    if (CONFIG_T::reuse_factor > 38)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[38], res_tmp[38], weights, biases);
    if (CONFIG_T::reuse_factor > 39)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[39], res_tmp[39], weights, biases);
    if (CONFIG_T::reuse_factor > 40)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[40], res_tmp[40], weights, biases);
    if (CONFIG_T::reuse_factor > 41)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[41], res_tmp[41], weights, biases);
    if (CONFIG_T::reuse_factor > 42)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[42], res_tmp[42], weights, biases);
    if (CONFIG_T::reuse_factor > 43)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[43], res_tmp[43], weights, biases);
    if (CONFIG_T::reuse_factor > 44)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[44], res_tmp[44], weights, biases);
    if (CONFIG_T::reuse_factor > 45)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[45], res_tmp[45], weights, biases);
    if (CONFIG_T::reuse_factor > 46)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[46], res_tmp[45], weights, biases);
    if (CONFIG_T::reuse_factor > 47)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[47], res_tmp[47], weights, biases);
    if (CONFIG_T::reuse_factor > 48)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[48], res_tmp[48], weights, biases);
    if (CONFIG_T::reuse_factor > 49)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[49], res_tmp[49], weights, biases);
    if (CONFIG_T::reuse_factor > 50)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[50], res_tmp[50], weights, biases);
    if (CONFIG_T::reuse_factor > 51)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[51], res_tmp[51], weights, biases);
    if (CONFIG_T::reuse_factor > 52)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[52], res_tmp[52], weights, biases);
    if (CONFIG_T::reuse_factor > 53)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[53], res_tmp[53], weights, biases);
    if (CONFIG_T::reuse_factor > 54)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[54], res_tmp[54], weights, biases);
    if (CONFIG_T::reuse_factor > 55)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[55], res_tmp[55], weights, biases);
    if (CONFIG_T::reuse_factor > 56)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[56], res_tmp[55], weights, biases);
    if (CONFIG_T::reuse_factor > 57)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[57], res_tmp[57], weights, biases);
    if (CONFIG_T::reuse_factor > 58)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[58], res_tmp[58], weights, biases);
    if (CONFIG_T::reuse_factor > 59)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[59], res_tmp[59], weights, biases);
    if (CONFIG_T::reuse_factor > 60)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[60], res_tmp[60], weights, biases);
    if (CONFIG_T::reuse_factor > 61)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[61], res_tmp[61], weights, biases);
    if (CONFIG_T::reuse_factor > 62)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[62], res_tmp[62], weights, biases);
    if (CONFIG_T::reuse_factor > 63)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[63], res_tmp[63], weights, biases);
    if (CONFIG_T::reuse_factor > 64)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[64], res_tmp[64], weights, biases);
    if (CONFIG_T::reuse_factor > 65)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[65], res_tmp[65], weights, biases);
    if (CONFIG_T::reuse_factor > 66)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[66], res_tmp[66], weights, biases);
    if (CONFIG_T::reuse_factor > 67)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[67], res_tmp[67], weights, biases);
    if (CONFIG_T::reuse_factor > 68)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[68], res_tmp[68], weights, biases);
    if (CONFIG_T::reuse_factor > 69)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[69], res_tmp[69], weights, biases);
    if (CONFIG_T::reuse_factor > 70)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[70], res_tmp[70], weights, biases);
    if (CONFIG_T::reuse_factor > 71)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[71], res_tmp[71], weights, biases);
    if (CONFIG_T::reuse_factor > 72)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[72], res_tmp[72], weights, biases);
    if (CONFIG_T::reuse_factor > 73)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[73], res_tmp[73], weights, biases);
    if (CONFIG_T::reuse_factor > 74)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[74], res_tmp[74], weights, biases);
    if (CONFIG_T::reuse_factor > 75)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[75], res_tmp[75], weights, biases);
    if (CONFIG_T::reuse_factor > 76)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[76], res_tmp[76], weights, biases);
    if (CONFIG_T::reuse_factor > 77)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[77], res_tmp[77], weights, biases);
    if (CONFIG_T::reuse_factor > 78)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[78], res_tmp[78], weights, biases);
    if (CONFIG_T::reuse_factor > 79)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[79], res_tmp[79], weights, biases);
    if (CONFIG_T::reuse_factor > 80)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[80], res_tmp[80], weights, biases);
    if (CONFIG_T::reuse_factor > 81)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[81], res_tmp[81], weights, biases);
    if (CONFIG_T::reuse_factor > 82)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[82], res_tmp[82], weights, biases);
    if (CONFIG_T::reuse_factor > 83)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[83], res_tmp[83], weights, biases);
    if (CONFIG_T::reuse_factor > 84)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[84], res_tmp[84], weights, biases);
    if (CONFIG_T::reuse_factor > 85)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[85], res_tmp[85], weights, biases);
    if (CONFIG_T::reuse_factor > 86)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[86], res_tmp[86], weights, biases);
    if (CONFIG_T::reuse_factor > 87)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[87], res_tmp[87], weights, biases);
    if (CONFIG_T::reuse_factor > 88)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[88], res_tmp[88], weights, biases);
    if (CONFIG_T::reuse_factor > 89)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[89], res_tmp[89], weights, biases);
    if (CONFIG_T::reuse_factor > 90)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[90], res_tmp[90], weights, biases);
    if (CONFIG_T::reuse_factor > 91)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[91], res_tmp[91], weights, biases);
    if (CONFIG_T::reuse_factor > 92)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[92], res_tmp[92], weights, biases);
    if (CONFIG_T::reuse_factor > 93)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[93], res_tmp[93], weights, biases);
    if (CONFIG_T::reuse_factor > 94)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[94], res_tmp[94], weights, biases);
    if (CONFIG_T::reuse_factor > 95)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[95], res_tmp[95], weights, biases);
    if (CONFIG_T::reuse_factor > 96)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[96], res_tmp[96], weights, biases);
    if (CONFIG_T::reuse_factor > 97)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[97], res_tmp[97], weights, biases);
    if (CONFIG_T::reuse_factor > 98)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[98], res_tmp[98], weights, biases);
    if (CONFIG_T::reuse_factor > 99)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[99], res_tmp[99], weights, biases);
    if (CONFIG_T::reuse_factor > 100)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[100], res_tmp[100], weights, biases);
    if (CONFIG_T::reuse_factor > 101)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[101], res_tmp[101], weights, biases);
    if (CONFIG_T::reuse_factor > 102)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[102], res_tmp[102], weights, biases);
    if (CONFIG_T::reuse_factor > 103)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[103], res_tmp[103], weights, biases);
    if (CONFIG_T::reuse_factor > 104)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[104], res_tmp[104], weights, biases);
    if (CONFIG_T::reuse_factor > 105)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[105], res_tmp[105], weights, biases);
    if (CONFIG_T::reuse_factor > 106)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[106], res_tmp[106], weights, biases);
    if (CONFIG_T::reuse_factor > 107)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[107], res_tmp[107], weights, biases);
    if (CONFIG_T::reuse_factor > 108)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[108], res_tmp[108], weights, biases);
    if (CONFIG_T::reuse_factor > 109)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[109], res_tmp[109], weights, biases);
    if (CONFIG_T::reuse_factor > 110)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[110], res_tmp[110], weights, biases);
    if (CONFIG_T::reuse_factor > 111)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[111], res_tmp[111], weights, biases);
    if (CONFIG_T::reuse_factor > 112)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[112], res_tmp[112], weights, biases);
    if (CONFIG_T::reuse_factor > 113)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[113], res_tmp[113], weights, biases);
    if (CONFIG_T::reuse_factor > 114)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[114], res_tmp[114], weights, biases);
    if (CONFIG_T::reuse_factor > 115)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[115], res_tmp[115], weights, biases);
    if (CONFIG_T::reuse_factor > 116)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[116], res_tmp[116], weights, biases);
    if (CONFIG_T::reuse_factor > 117)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[117], res_tmp[117], weights, biases);
    if (CONFIG_T::reuse_factor > 118)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[118], res_tmp[118], weights, biases);
    if (CONFIG_T::reuse_factor > 119)
        pointwise_conv_1d_latency_cl<data_T, res_T, CONFIG_T>(data_tmp[119], res_tmp[119], weights, biases);

RFOutputLoop:
    for (int jj = 0; jj < CONFIG_T::reuse_factor; jj++) {
    #pragma HLS UNROLL
    InnerOutputLoop:
        for (int ii = 0; ii < CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor; ii++) {
            #pragma HLS UNROLL
            res[jj * CONFIG_T::out_width * CONFIG_T::n_filt / CONFIG_T::reuse_factor + ii] = res_tmp[jj][ii];
        }
    }
}

} // namespace nnet
#endif
