#ifndef NNET_TIME_DISTRIBUTED_H_
#define NNET_TIME_DISTRIBUTED_H_

#include <math.h>

namespace nnet {

struct time_distributed_config {
    static const unsigned dim = 2;

    static const unsigned n_time_steps = 10;
    static const unsigned in_height = 10;
    static const unsigned in_width = 10;
    static const unsigned n_chan = 10;
};

template <class data_T, typename CONFIG_T>
void read_time_step_2d(unsigned time_step, data_T data[CONFIG_T::n_time_steps * CONFIG_T::n_chan],
                       data_T res[CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

ChannelLoop:
    for (unsigned i = 0; i < CONFIG_T::n_chan; i++) {
        res[i] = data[time_step * CONFIG_T::n_chan + i];
    }
}

template <class data_T, typename CONFIG_T>
void read_time_step_3d(unsigned time_step, data_T data[CONFIG_T::n_time_steps * CONFIG_T::in_width * CONFIG_T::n_chan],
                       data_T res[CONFIG_T::in_width * CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

WidthLoop:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
    ChannelLoop:
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            res[i * CONFIG_T::n_chan + j] =
                data[time_step * CONFIG_T::in_width * CONFIG_T::n_chan + i * CONFIG_T::n_chan + j];
        }
    }
}

template <class data_T, typename CONFIG_T>
void read_time_step_4d(unsigned time_step,
                       data_T data[CONFIG_T::n_time_steps * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                       data_T res[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

HeightLoop:
    for (int i = 0; i < CONFIG_T::in_height; i++) {
    WidthLoop:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
        ChannelLoop:
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                res[i * CONFIG_T::in_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k] =
                    data[time_step * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan +
                         i * CONFIG_T::in_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k];
            }
        }
    }
}

template <class data_T, typename CONFIG_T>
void write_time_step_2d(unsigned time_step, data_T data[CONFIG_T::n_chan],
                        data_T res[CONFIG_T::n_time_steps * CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

ChannelLoop:
    for (unsigned i = 0; i < CONFIG_T::n_chan; i++) {
        res[time_step * CONFIG_T::n_chan + i] = data[i];
    }
}

template <class data_T, typename CONFIG_T>
void write_time_step_3d(unsigned time_step, data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                        data_T res[CONFIG_T::n_time_steps * CONFIG_T::in_width * CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

WidthLoop:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
    ChannelLoop:
        for (int j = 0; j < CONFIG_T::n_chan; j++) {
            res[time_step * CONFIG_T::in_width * CONFIG_T::n_chan + i * CONFIG_T::n_chan + j] =
                data[i * CONFIG_T::n_chan + j];
        }
    }
}

template <class data_T, typename CONFIG_T>
void write_time_step_4d(unsigned time_step, data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                        data_T res[CONFIG_T::n_time_steps * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan]) {
    #pragma HLS PIPELINE

HeightLoop:
    for (int i = 0; i < CONFIG_T::in_height; i++) {
    WidthLoop:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
        ChannelLoop:
            for (int k = 0; k < CONFIG_T::n_chan; k++) {
                res[time_step * CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan +
                    i * CONFIG_T::in_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k] =
                    data[i * CONFIG_T::in_width * CONFIG_T::n_chan + j * CONFIG_T::n_chan + k];
            }
        }
    }
}

} // namespace nnet

#endif
