#ifndef NNET_CONV2D_LATENCY_H_
#define NNET_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

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
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

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
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

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
                                    int index_data = (oh*CONFIG_T::stride_height+fh-CONFIG_T::pad_top)*CONFIG_T::in_width*CONFIG_T::n_chan
                                                   + (ow*CONFIG_T::stride_width+fw-CONFIG_T::pad_left)*CONFIG_T::n_chan
                                                   + cc;
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
    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
              for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                int index = oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff;
                res[index] = (res_T)(acc[index]);
            }
        }
    }

}//end conv2d

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_2d_latency_cl(
    data_T data[CONFIG_T::in_height*CONFIG_T::in_width*CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_height*CONFIG_T::out_width*CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{

    typename CONFIG_T::accum_t mult[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_height * CONFIG_T::out_width * CONFIG_T::n_filt];

    #pragma HLS ARRAY_PARTITION variable=mult complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc complete dim=0

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights,biases

    // Parallel mode
    #pragma HLS PIPELINE
    #pragma HLS ARRAY_PARTITION variable=biases complete dim=0

    // Limit multipliers to control parallelization
    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

    // Convolve, saving all multiplication results to accumulate later
    ConvOutHeight: for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        ConvOutWidth: for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
            ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {

                    int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan
                                   + ow*CONFIG_T::n_filt*CONFIG_T::n_chan
                                   + ff*CONFIG_T::n_chan
                                   + cc;

                    int index_weight = cc*CONFIG_T::n_filt + ff;

                    if ((oh*CONFIG_T::stride_height) < CONFIG_T::pad_top
                    || (oh*CONFIG_T::stride_height) >= (CONFIG_T::pad_top+CONFIG_T::in_height)
                    || (ow*CONFIG_T::stride_width) < CONFIG_T::pad_left
                    || (ow*CONFIG_T::stride_width) >= (CONFIG_T::pad_left+CONFIG_T::in_width)) {
                        mult[index_mult] = 0;
                    } else {
                        int index_data = (oh*CONFIG_T::stride_height-CONFIG_T::pad_top)*CONFIG_T::in_width*CONFIG_T::n_chan
                                       + (ow*CONFIG_T::stride_width-CONFIG_T::pad_left)*CONFIG_T::n_chan
                                       + cc;
                        mult[index_mult] = data[index_data] * weights[index_weight];
                    }

                }
            }
        }
    }


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
                AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {

                    int index_mult = oh*CONFIG_T::out_width*CONFIG_T::n_filt*CONFIG_T::n_chan
                                    + ow*CONFIG_T::n_filt*CONFIG_T::n_chan
                                    + ff*CONFIG_T::n_chan
                                    + cc;
                    int index_acc = oh*CONFIG_T::out_width*CONFIG_T::n_filt
                                    + ow*CONFIG_T::n_filt
                                    + ff;

                    acc[index_acc] += mult[index_mult];

                }
            }
        }
    }

    // Cast to "res_t" type
    for(int oh = 0; oh < CONFIG_T::out_height; oh++) {
        for(int ow = 0; ow < CONFIG_T::out_width; ow++) {
              for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
                int index = oh*CONFIG_T::out_width*CONFIG_T::n_filt + ow*CONFIG_T::n_filt + ff;
                res[index] = (res_T)(acc[index]);
            }
        }
    }

}//end conv2d

}
#endif