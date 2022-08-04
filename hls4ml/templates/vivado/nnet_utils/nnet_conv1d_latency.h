#ifndef NNET_CONV1D_LATENCY_H_
#define NNET_CONV1D_LATENCY_H_

#include "nnet_common.h"
#include <cstdlib>

namespace nnet {

template<class data_T, class res_T, typename CONFIG_T>
void conv_1d_latency_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::filt_width * CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt]
)
{

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan * CONFIG_T::filt_width];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width][CONFIG_T::n_filt];

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
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
            ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                ConvMult: for(int jj = 0; jj < CONFIG_T::filt_width; jj++){

                    int index_mult   = ii*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_width + ff*CONFIG_T::n_chan*CONFIG_T::filt_width + cc*CONFIG_T::filt_width + jj;
                    int index_weight = jj*CONFIG_T::n_chan*CONFIG_T::n_filt + cc*CONFIG_T::n_filt + ff;
                    int index_data   = (ii*CONFIG_T::stride_width+jj-CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                    if((ii*CONFIG_T::stride_width+jj) < CONFIG_T::pad_left || (ii*CONFIG_T::stride_width+jj) >= (CONFIG_T::pad_left + CONFIG_T::in_width)){
                        mult[index_mult] = 0;
                    }
                    else {
                        mult[index_mult] = data[index_data] * weights[index_weight];
                    }
                }
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Initialize accumulator with input biases
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            acc[ii][ff]=biases[ff];
        }
    }


    // Accumulate multiplication result
    AccumOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            //Do "dot product" sum within filter and sum over channels
            AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++){
                AccumDot: for(int jj = 0; jj < CONFIG_T::filt_width; jj++){
                    int index_mult = ii*CONFIG_T::n_filt*CONFIG_T::n_chan*CONFIG_T::filt_width + ff*CONFIG_T::n_chan*CONFIG_T::filt_width + cc*CONFIG_T::filt_width + jj;
                    acc[ii][ff] += mult[index_mult];
                }//end dot product loop
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Cast to "res_t" type
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            res[ii * CONFIG_T::n_filt + ff] = (res_T)(acc[ii][ff]);
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void pointwise_conv_1d_latency_cl(
    data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
    res_T  res[CONFIG_T::out_width * CONFIG_T::n_filt],
    typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
    typename CONFIG_T::bias_t   biases[CONFIG_T::n_filt])
{
    assert(CONFIG_T::filt_width == 1);

    typename CONFIG_T::accum_t mult[CONFIG_T::out_width * CONFIG_T::n_filt * CONFIG_T::n_chan];
    typename CONFIG_T::accum_t acc[CONFIG_T::out_width][CONFIG_T::n_filt];

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
    ConvOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        ConvFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            ConvChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult   = ii*CONFIG_T::n_filt*CONFIG_T::n_chan + ff*CONFIG_T::n_chan + cc;
                int index_weight = cc*CONFIG_T::n_filt + ff;
                int index_data   = (ii*CONFIG_T::stride_width-CONFIG_T::pad_left) * CONFIG_T::n_chan + cc;

                if((ii*CONFIG_T::stride_width) < CONFIG_T::pad_left || (ii*CONFIG_T::stride_width) >= (CONFIG_T::pad_left + CONFIG_T::in_width)){
                    mult[index_mult] = 0;
                }
                else {
                    mult[index_mult] = data[index_data] * weights[index_weight];
                }
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Initialize accumulator with input biases
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            acc[ii][ff]=biases[ff];
        }
    }


    // Accumulate multiplication result
    AccumOut: for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        AccumFilt: for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            //Do "dot product" sum within filter and sum over channels
            AccumChan: for(int cc = 0; cc < CONFIG_T::n_chan; cc++) {
                int index_mult = ii*CONFIG_T::n_filt*CONFIG_T::n_chan + ff*CONFIG_T::n_chan + cc;
                acc[ii][ff] += mult[index_mult];
            }//end channel loop
        }//end filter loop
    }//end output loop


    // Cast to "res_t" type
    for(int ii = 0; ii < CONFIG_T::out_width; ii++) {
        for(int ff = 0; ff < CONFIG_T::n_filt; ff++) {
            res[ii * CONFIG_T::n_filt + ff] = (res_T)(acc[ii][ff]);
        }
    }
}

}
#endif
