#ifndef NNET_CONV2D_RESOURCE_H_
#define NNET_CONV2D_RESOURCE_H_

#include "nnet_common.h"
#include "nnet_dense.h"

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;
    constexpr unsigned block_factor = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(mult_n_in * mult_n_out, CONFIG_T::reuse_factor);
    constexpr unsigned multscale = multiplier_limit / mult_n_out;

    assert((multiplier_limit % mult_n_out == 0 || CONFIG_T::reuse_factor >= mult_n_in) && "The current Reuse Factor is not allowed");
    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= FILT_HEIGHT * FILT_WIDTH * N_CHAN");

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[CONFIG_T::n_pixels][mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    PartitionLoop:
    for (unsigned i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        //#pragma HLS UNROLL // We don't want this loop unrolled

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

        PixelInitAccumLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            InitAccumLoop:
            for (unsigned i_acc = 0; i_acc < mult_n_out; i_acc++) {
                #pragma HLS UNROLL
                acc[i_pxl][i_acc] = (typename CONFIG_T::accum_t) biases[i_acc];
            }
        }

        ReuseLoop:
        for (unsigned i_rf = 0; i_rf < CONFIG_T::reuse_factor; i_rf++) {
            #pragma HLS PIPELINE II=1 rewind

            unsigned i_w = i_rf;
            unsigned i_in = i_rf;
            unsigned i_out = 0;
            unsigned i_acc = 0;

            MultLoop:
            for (unsigned i_blk = 0; i_blk < block_factor; i_blk++) {
                #pragma HLS UNROLL

                PixelMultLoop:
                for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
                    #pragma HLS UNROLL

                    acc[i_pxl][i_out] += static_cast<typename CONFIG_T::accum_t>(
                            CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(data_buf[i_pxl][i_in], weights[i_w]));
                }

                // Increment i_w
                i_w += CONFIG_T::reuse_factor;
                // Increment i_in
                i_in += CONFIG_T::reuse_factor;
                if (i_in >= mult_n_in) {
                    i_in = i_rf;
                }
                // Increment i_out
                if (i_acc + 1 >= multscale) {
                    i_acc = 0;
                    i_out++;
                } else {
                    i_acc++;
                }
            }
        }

        PixelResultLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL
            // Cast to "res_t" type
            ResultLoop:
            for (unsigned i_res = 0; i_res < mult_n_out; i_res++) {
                #pragma HLS UNROLL
                *(res++) = cast<data_T, res_T, typename CONFIG_T::mult_config>(acc[i_pxl][i_res]);
            }
        }
    }
}

template<class data_T, typename CONFIG_T>
void im2col_2d_pointwise_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    data_T data_col[CONFIG_T::n_chan],
    const int row,
    const int col)
{
    int index = 0;
    int input_row = -CONFIG_T::pad_top + row * CONFIG_T::stride_height;

    ChannelLoop:
    for (int channel = 0; channel < CONFIG_T::n_chan; channel++) {
        #pragma HLS UNROLL
        if (input_row < 0 || input_row >= CONFIG_T::in_height) {
            data_col[index++] = 0;
        } else {
            int input_col = -CONFIG_T::pad_left + col * CONFIG_T::stride_width;
            if (input_col >= 0 && input_col < CONFIG_T::in_width) {
                data_col[index++] = data[input_row * CONFIG_T::in_width * CONFIG_T::n_chan + input_col * CONFIG_T::n_chan + channel];
            } else {
                data_col[index++] = 0;
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_resource_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{
    assert(CONFIG_T::filt_height == 1 && CONFIG_T::filt_width == 1);

    const int nin = CONFIG_T::n_chan;
    const int nout = CONFIG_T::n_filt;
    const int rufactor = CONFIG_T::reuse_factor;
    const int block_factor = DIV_ROUNDUP(nin*nout, rufactor);

    //#pragma HLS function_instantiate variable=weights,biases
    //#pragma HLS RESOURCE         variable=weights core=RAM_2P_BRAM Commenting out the deisgnation HLS seems to choose correctly
    //#pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    //#pragma HLS ARRAY_PARTITION variable=biases complete

    data_T data_col[CONFIG_T::n_chan];
    res_T res_col[CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=data_col complete
    #pragma HLS ARRAY_PARTITION variable=res_col complete

    HeightLoop:
    for (int i = 0; i < CONFIG_T::out_height; i++) {
        WidthLoop:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            #pragma HLS PIPELINE
            im2col_2d_pointwise_cl<data_T, CONFIG_T>(data, data_col, i, j);
            dense<data_T, res_T, typename CONFIG_T::mult_config>(data_col, res_col, weights, biases);
            FiltLoop:
            for (int k = 0; k < CONFIG_T::n_filt; k++) {
                res[i * CONFIG_T::out_width * CONFIG_T::n_filt + j * CONFIG_T::n_filt + k] = res_col[k];
            }
        }
    }
}

}
#endif
