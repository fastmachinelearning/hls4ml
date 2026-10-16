#ifndef NNET_PADDING_STREAM_H_
#define NNET_PADDING_STREAM_H_

namespace nnet {

template <class res_pipe, typename CONFIG_T> inline void fill_zero() {
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res_part;
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
        res_part[i] = 0;
    }
    res_pipe::write(res_part);
}

template <class data_pipe, class res_pipe, typename CONFIG_T> inline void fill_data() {
    [[intel::fpga_register]] auto data_part = data_pipe::read();
    [[intel::fpga_register]] typename ExtractPipeType<res_pipe>::value_type res_part;
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_chan; i++) {
        res_part[i] = data_part[i];
    }
    res_pipe::write(res_part);
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void zeropad1d_cl_stream() {
PadLeft:
    for (int i = 0; i < CONFIG_T::pad_left; i++) {
        fill_zero<res_pipe, CONFIG_T>();
    }

CopyMain:
    for (int i = 0; i < CONFIG_T::in_width; i++) {
        fill_data<data_pipe, res_pipe, CONFIG_T>();
    }

PadRight:
    for (int i = 0; i < CONFIG_T::pad_right; i++) {
        fill_zero<res_pipe, CONFIG_T>();
    }
}

template <class data_pipe, class res_pipe, typename CONFIG_T> void zeropad2d_cl_stream() {
PadTop:
    [[intel::loop_coalesce(2)]] for (int i = 0; i < CONFIG_T::pad_top; i++) {
    PadTopWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_pipe, CONFIG_T>();
        }
    }

PadMain:
    [[intel::loop_coalesce(2)]] for (int i = 0; i < CONFIG_T::in_height; i++) {

    PadLeft:
        for (int j = 0; j < CONFIG_T::pad_left; j++) {
            fill_zero<res_pipe, CONFIG_T>();
        }

    CopyMain:
        for (int j = 0; j < CONFIG_T::in_width; j++) {
            fill_data<data_pipe, res_pipe, CONFIG_T>();
        }

    PadRight:
        for (int j = 0; j < CONFIG_T::pad_right; j++) {
            fill_zero<res_pipe, CONFIG_T>();
        }
    }

PadBottom:
    for (int i = 0; i < CONFIG_T::pad_bottom; i++) {
    PadBottomWidth:
        for (int j = 0; j < CONFIG_T::out_width; j++) {
            fill_zero<res_pipe, CONFIG_T>();
        }
    }
}

} // namespace nnet

#endif
