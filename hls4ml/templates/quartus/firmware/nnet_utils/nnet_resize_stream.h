#ifndef NNET_IMAGE_STREAM_H_
#define NNET_IMAGE_STREAM_H_

#include "nnet_common.h"

namespace nnet {

template<class data_T, typename CONFIG_T>
void resize_nearest(stream<data_T> &image, stream<data_T> &resized) {
    assert(CONFIG_T::new_height % CONFIG_T::height == 0);
    assert(CONFIG_T::new_width % CONFIG_T::width == 0);

    constexpr unsigned ratio_height = CONFIG_T::new_height / CONFIG_T::height;
    constexpr unsigned ratio_width = CONFIG_T::new_width / CONFIG_T::width;

    ImageHeight:
    for (unsigned h = 0; h < CONFIG_T::height; h++) {
        hls_register data_T data_in_row[CONFIG_T::width];

        ImageWidth:
        for (unsigned i = 0; i < CONFIG_T::width; i++) {
            hls_register data_T  in_data = image.read();

            ImageChan:
            #pragma unroll
            for (unsigned j = 0; j < CONFIG_T::n_chan; j++) {
                data_in_row[i][j] = in_data[j];
            }
        }

        ResizeHeight:
        for (unsigned i = 0; i < ratio_height; i++) {

            ImageWidth2:
            for (unsigned l = 0; l < CONFIG_T::width; l++) {

                ResizeWidth:
                for (unsigned j = 0; j < ratio_width; j++) {

                    hls_register data_T out_data;

                    ResizeChan:
                    #pragma unroll
                    for (unsigned k = 0; k < CONFIG_T::n_chan; k++) {
                        out_data[k] = data_in_row[l][k];
                    }

                    resized.write(out_data);
                }
            }
        }
    }
}

}

#endif
