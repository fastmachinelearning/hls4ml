#ifndef NNET_CONV2D_LATENCY_H_
#define NNET_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include <cstdlib>

namespace nnet {

//Computes multiplier limit
//This function should not be synthesized into firmware
template<typename CONFIG_T>
    int compute_multiplier_limit_conv2d(
    typename CONFIG_T::weight_t  weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt]
)
{
    int n_mult = 0;

    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
                for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                                int index_weight = fh*CONFIG_T::filt_width*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + fw*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + cc*CONFIG_T::n_filt
                                                  + ff;

                                if ((oh*CONFIG_T::stride_height+fh) < CONFIG_T::pad_top
                                || (oh*CONFIG_T::stride_height+fh) >= (CONFIG_T::pad_top+CONFIG_T::in_height)
                                || (ow*CONFIG_T::stride_width+fw) < CONFIG_T::pad_left
                                || (ow*CONFIG_T::stride_width+fw) >= (CONFIG_T::pad_left+CONFIG_T::in_width)) {
                                    //padded - do nothing
                                    continue;
                                } else {
                                    if (weights[index_weight] > 1e-20 || weights[index_weight] < -1e-20) {
                                          n_mult++;
                                    }
                                }

                        }//end mult loop
                    }//end channel loop
                }//end filter width loop
            }//end filter height loop
        }//end output width loop
    }//end output height loop

    return ceil( float(n_mult) / float(CONFIG_T::reuse_factor) );

}//end compute_n_mult

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_latency_cf(
    data_T data[CONFIG_T::in_height*CONFIG_T::in_width*CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height*CONFIG_T::out_width*CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    typename CONFIG_T::accum_t mult[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_height * CONFIG_T::filt_width];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    const int multiplier_limit = compute_multiplier_limit_conv2d<CONFIG_T>(weights);
    #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation

    // Convolve, saving all multiplication results to accumulate later
    ConvOutHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        ConvOutWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
                ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    ConvFiltHeight: for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        ConvFiltWidth: for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                            int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ow*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ff*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + cc*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + fh*CONFIG_T::filt_width
                                           + fw;

                                int index_weight = fh*CONFIG_T::filt_width*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + fw*CONFIG_T::n_chan*CONFIG_T::n_filt
                                                 + cc*CONFIG_T::n_filt
                                                 + ff;

                                if ((oh*CONFIG_T::stride_height+fh) < CONFIG_T::pad_top
                                || (oh*CONFIG_T::stride_height+fh) >= (CONFIG_T::pad_top+CONFIG_T::in_height)
                                || (ow*CONFIG_T::stride_width+fw) < CONFIG_T::pad_left
                                || (ow*CONFIG_T::stride_width+fw) >= (CONFIG_T::pad_left+CONFIG_T::in_width)) {
                                    mult[index_mult] = 0;
                                } else {
                                    int index_data = cc*CONFIG_T::in_height*CONFIG_T::in_width
                                                   + (oh*CONFIG_T::stride_height+fh-CONFIG_T::pad_top)*CONFIG_T::in_width
                                                   + (ow*CONFIG_T::stride_width+fw-CONFIG_T::pad_left);
                                    mult[index_mult] = data[index_data] * weights[index_weight];
                                }

                        }//end mult loop
                    }//end channel loop
                  }//end filter width loop
            }//end filter height loop
        }//end output width loop
    }//end output height loop


    // Initialize accumulator with input biases
    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                acc[oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff]=biases[ff];
            }
        }
    }


    // Accumulate multiplication result
    AccumOutHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        AccumOutWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                //Do "dot product" sum within filter and sum over channels
                AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                    AccumDotHeight: for(int fh = 0; fh < CONFIG_T::filt_height; fh++){
                        AccumDotWidth: for(int fw = 0; fw < CONFIG_T::filt_width; fw++){

                            int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ow*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + ff*CONFIG_T::n_chan*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + cc*CONFIG_T::filt_height*CONFIG_T::filt_width
                                           + fh*CONFIG_T::filt_width
                                           + fw;
                            int index_acc = oh*CONFIG_T::out_width*CONFIG_T::n_filt
                                          + ow*CONFIG_T::n_filt
                                          + ff;

                            acc[index_acc] += mult[index_mult];

                        }//end dot product filter width loop
                    }//end dot product filter height loop
                }//end n channel loop
            }//end n filter loop
        }//end output width loop
    }//end output height loop

    // Cast to "res_t" type
    for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
        for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
            for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
                int res_index = ff*CONFIG_T::out_height*CONFIG_T::out_width + oh*CONFIG_T::out_width + ow;
                int acc_index = oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff;
                res[res_index] = acc[acc_index];
            }
        }
    }

}//end conv2d

template<class data_T, class res_T, typename CONFIG_T>
void conv_2d_latency_cl(
    data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    constexpr unsigned mult_n_in = CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan;
    constexpr unsigned mult_n_out = CONFIG_T::n_filt;

    data_T data_buf[CONFIG_T::n_pixels][mult_n_in];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=0

    typename CONFIG_T::accum_t mult[mult_n_in * mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=mult complete

    typename CONFIG_T::accum_t acc[mult_n_out];
    #pragma HLS ARRAY_PARTITION variable=acc complete

    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=biases complete

    int multiplier_limit  = CONFIG_T::n_pixels * (ceil(float(mult_n_in * mult_n_out) / float(CONFIG_T::reuse_factor)) - floor(float(CONFIG_T::mult_config::n_zeros) / float(CONFIG_T::reuse_factor)));
    CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::limit(multiplier_limit);

    PartitionLoop:
    for (int i_part = 0; i_part < CONFIG_T::n_partitions; i_part++) {
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        CONFIG_T::template fill_buffer<data_T, CONFIG_T>::fill_buffer(data, data_buf, i_part);

        PixelLoop:
        for (unsigned i_pxl = 0; i_pxl < CONFIG_T::n_pixels; i_pxl++) {
            #pragma HLS UNROLL

            data_T cache;

            // Do the matrix-multiply
            Product1: for(int i_in = 0; i_in < mult_n_in; i_in++) {
                cache = data_buf[i_pxl][i_in];
                Product2: for(int i_out = 0; i_out < mult_n_out; i_out++) {
                    mult[i_in * mult_n_out + i_out] = CONFIG_T::mult_config::template product<data_T, typename CONFIG_T::mult_config::weight_t>::product(cache, weights[i_in * mult_n_out + i_out]);
                }
            }

            // Initialize accumulator with input biases
            ResetAccum: for(int i_acc = 0; i_acc < mult_n_out; i_acc++) {
                acc[i_acc] = (typename CONFIG_T::accum_t) biases[i_acc];
            }

            // Accumulate multiplication result
            Accum1: for(int i_in = 0; i_in < mult_n_in; i_in++) {
                Accum2: for(int i_out = 0; i_out < mult_n_out; i_out++) {
                    acc[i_out] += mult[i_in * mult_n_out + i_out];
                }
            }

            // Cast to "res_t" type
            Result: for(int i_res = 0; i_res < mult_n_out; i_res++){
                *(res++) = cast<data_T, res_T, CONFIG_T>(acc[i_res]);
            }

        }
    }

}

}
#endif