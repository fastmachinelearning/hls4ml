#ifndef NNET_SEPARABLE_CONV2D_LATENCY_H_
#define NNET_SEPARABLE_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_conv2d_resource.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_resource_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
                                  const typename CONFIG_T::bias_t &biases) {


    int depth_multiplier = CONFIG_T::n_filt/CONFIG_T::n_chan;

    
    //#pragma unroll
    for (int dm = 0; dm < depth_multiplier; dm++) {

        //#pragma unroll
        for (int c = 0; c < CONFIG_T::n_chan; c++) {

            //#pragma unroll
            for (int h = 0; h < CONFIG_T::out_height; h++) {

                //#pragma unroll
                for (int w = 0; w < CONFIG_T::out_width; w++) {

                    
                    //res[(h * CONFIG_T::out_width * CONFIG_T::n_chan) + (w * CONFIG_T::n_chan) + c] = 0;

                    int res_idx = (h * CONFIG_T::out_width * CONFIG_T::n_filt) + 
                                    (w * CONFIG_T::n_filt) + 
                                    (c * depth_multiplier) + dm;

                    //res[res_idx] = 0;
                    //res[res_idx] = biases[(dm * depth_multiplier) + c]; 

                    res[res_idx] = biases[c * depth_multiplier + dm];

                    
                    //#pragma unroll
                    for (int kh = 0; kh < CONFIG_T::filt_height; kh++) {

                        //#pragma unroll
                        for (int kw = 0; kw < CONFIG_T::filt_width; kw++)  {

                            int h_in = h * CONFIG_T::stride_height + kh - CONFIG_T::pad_top;
                            int w_in = w * CONFIG_T::stride_width + kw - CONFIG_T::pad_left;

                            if ((h_in >= 0) && (h_in < CONFIG_T::in_height) 
                            && (w_in >= 0) && (w_in < CONFIG_T::in_width)) {

                                res[res_idx] = 
                                res[res_idx] +
                                data[(h_in)*CONFIG_T::in_width * CONFIG_T::n_chan + (w_in)*CONFIG_T::n_chan+c] 
                                * weights[(dm * CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan) + 
                                          (kh * CONFIG_T::filt_width * CONFIG_T::n_chan) +
                                          (kw * CONFIG_T::n_chan) + c];

                            }
                        }


                    }

                    //res[res_idx] = res[res_idx] + biases[c * depth_multiplier + dm];


                }
            }
        }

    
    }
     
}
} // namespace nnet
#endif