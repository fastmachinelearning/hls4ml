#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "nnet_common.h"
#include "hls_stream.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void resize_nearest(
    hls::stream<data_T> &image,
    hls::stream<data_T> &resized
) {
    assert(CONFIG_T::new_height % CONFIG_T::height == 0);
    assert(CONFIG_T::new_width % CONFIG_T::width == 0);
    constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
    constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;
    constexpr unsigned ii = ratio_height * ratio_width;

    ResizeImage: for (unsigned i = 0; i < CONFIG_T::height * CONFIG_T::width; i++) {
        #pragma HLS PIPELINE II=ii
        
        data_T  in_data = image.read();

        ResizeNew: for (unsigned j = 0; j < ratio_height * ratio_width; j++) {
            #pragma HLS UNROLL

            data_T out_data;
            #pragma HLS DATA_PACK variable=out_data
            
            ResizeChan: for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                #pragma HLS UNROLL
                out_data[k] = in_data[k];
            }

            resized.write(out_data);
        }
    }
}

}

#endif
