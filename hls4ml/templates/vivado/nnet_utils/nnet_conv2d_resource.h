#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_instr_gen.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void im2col_2d_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    data_T data_col[CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width],
    const int row,
    const int col)
{
    const int channel_size = CONFIG_T::in_height * CONFIG_T::in_width;
    int index = 0;
    for (int channel = CONFIG_T::n_chan; channel--; data += channel_size) {
        #pragma HLS UNROLL
        for (int kernel_row = 0; kernel_row < CONFIG_T::filt_height; kernel_row++) {
            int input_row = -CONFIG_T::pad_top + kernel_row * CONFIG_T::dilation_height + row * CONFIG_T::stride_height;
            for (int kernel_col = 0; kernel_col < CONFIG_T::filt_width; kernel_col++) {
                if (input_row < 0 || input_row > CONFIG_T::in_height) {
                    data_col[index++] = 0;
                } else {
                    int input_col = -CONFIG_T::pad_left + kernel_col * CONFIG_T::dilation_width + col * CONFIG_T::stride_width;
                    if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                        //*(data_col++) = data[input_row * CONFIG_T::in_width + input_col];
                        data_col[index++] = data[input_row * CONFIG_T::in_width + input_col];
                    } else {
                        //*(data_col++) = 0;
                        data_col[index++] = 0;
                    }
                    input_col += CONFIG_T::stride_width;
                }
            }
            input_row += CONFIG_T::stride_height;
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cf(
    data_T data[CONFIG_T::n_chan * CONFIG_T::in_height * CONFIG_T::in_width],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    const int nin = CONFIG_T::n_chan * CONFIG_T::filt_width;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    //#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete
    
    HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE
            im2col_2d_cf<data_T, CONFIG_T>(data, data_col, i, j);
            dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                //res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
                res[k * CONFIG_T::out_height * CONFIG_T::out_width + i * CONFIG_T::out_width + j] = res_col[k]; // Transposed order
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T, int P>
void mult_line_buffer(
    data_T    data[P][CONFIG_T::n_in],
    res_T     res[P][CONFIG_T::n_out],
    typename CONFIG_T::weight_t  weights[CONFIG_T::n_in*CONFIG_T::n_out],
    typename CONFIG_T::bias_t    biases[CONFIG_T::n_out])
{
    #pragma HLS INLINE

    data_T cache[P];
    typename CONFIG_T::accum_t mult[P][CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[P][CONFIG_T::n_out];

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete
    #pragma HLS ARRAY_PARTITION variable=cache complete
    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(CONFIG_T::n_in*CONFIG_T::n_out, CONFIG_T::reuse_factor) - (CONFIG_T::n_zeros / CONFIG_T::reuse_factor);
    #pragma HLS ALLOCATION instances=product limit=multiplier_limit function

    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        for(int p = 0; p < P; p++) {
            #pragma HLS UNROLL
            cache[p] = data[p][ii];
        }
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii*CONFIG_T::n_out+jj;
            for(int p = 0; p < P; p++) {
                #pragma HLS UNROLL
                cache[p] = data[p][ii];
                mult[p][index] = product<data_T, typename CONFIG_T::weight_t, typename CONFIG_T::accum_t>(cache[p], weights[index]);
            }
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        for(int p = 0; p < P; p++) {
            #pragma HLS UNROLL
            acc[p][iacc] = (typename CONFIG_T::accum_t) biases[iacc];
        }
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii*CONFIG_T::n_out+jj;
            for(int p = 0; p < P; p++) {
                #pragma HLS UNROLL
                acc[p][jj] += mult[p][index];
            }
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_out; ires++){
        for(int p = 0; p < P; p++) {
            #pragma HLS UNROLL
            res[p][ires] = cast<data_T, res_T, CONFIG_T>(acc[p][ires]);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    constexpr unsigned P = CONFIG_T::out_height * CONFIG_T::out_width / CONFIG_T::reuse_factor;

    data_T data_line[P][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data_line complete dim=0

    res_T res_line[P][CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res_line complete dim=0

    ReadInput: for (unsigned i = 0; i < CONFIG_T::out_height * CONFIG_T::out_width; i += P) {
        #pragma HLS PIPELINE
        FillLine: for(unsigned p = 0; p < P; p++) {
            #pragma HLS UNROLL
            CONFIG_T::template fill_line<data_T, CONFIG_T>::fill_line(data, data_line[p], i + p);
        }

        mult_line_buffer<data_T, res_T, typename CONFIG_T::mult_config, P>(data_line, res_line, weights, biases);

        CopyRes: for(int p = 0; p < P; p++) {
            #pragma HLS UNROLL
            CopyResLine: for (int k = 0; k < CONFIG_T::n_filt; k++) {
                #pragma HLS UNROLL
                *(res++) = res_line[p][k];
            }
        }
    }
}

}
#endif
