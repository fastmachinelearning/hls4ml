#ifndef NNET_SEPARABLE_CONV2D_LATENCY_H_
#define NNET_SEPARABLE_CONV2D_LATENCY_H_

#include "nnet_common.h"
#include "nnet_conv2d_resource.h"
#include "nnet_mult.h"

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_conv_2d_latency_cl(const data_T &data, res_T &res, const typename CONFIG_T::weight_t &weights,
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

                            if ((h+kh-CONFIG_T::pad_top >= 0) && (h+kh-CONFIG_T::pad_top < CONFIG_T::in_height) 
                            && (w+kw-CONFIG_T::pad_left >= 0) && (w+kw-CONFIG_T::pad_left < CONFIG_T::in_width)) {

                                res[res_idx] = 
                                res[res_idx] +
                                data[(h+kh-CONFIG_T::pad_top)*CONFIG_T::in_width * CONFIG_T::n_chan+ (w+kw-CONFIG_T::pad_left)*CONFIG_T::n_chan+c] 
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