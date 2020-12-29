#ifndef NNET_IMAGE_H_
#define NNET_IMAGE_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct resize_config {
    static const unsigned height = 10;
    static const unsigned width = 10;
    static const unsigned n_chan = 10;
    static const unsigned new_height = 10;
    static const unsigned new_width = 10;
};

template<class data_T, typename CONFIG_T>
void resize_nearest(
    data_T image[CONFIG_T::height * CONFIG_T::width * CONFIG_T::n_chan],
    data_T resized[CONFIG_T::new_height * CONFIG_T::new_width * CONFIG_T::n_chan]
) {
    assert(CONFIG_T::new_height % CONFIG_T::height == 0);
    assert(CONFIG_T::new_width % CONFIG_T::width == 0);
    constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
    constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;

    #pragma HLS PIPELINE

    ResizeImage: for (unsigned i = 0; i < CONFIG_T::height * CONFIG_T::width; i++) {
        ResizeNew: for (unsigned j = 0; j < ratio_height * ratio_width; j++) {
            #pragma HLS UNROLL
            ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                #pragma HLS UNROLL
                *(resized++) = image[i * CONFIG_T::n_chan + k];
            }
        }
    }
}

}

#endif
