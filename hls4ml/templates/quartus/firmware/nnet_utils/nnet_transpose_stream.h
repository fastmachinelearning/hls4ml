#ifndef NNET_TRANSPOSE_STREAM_H_
#define NNET_TRANSPOSE_STREAM_H_

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T> void transpose_2d(stream<data_T> &data, stream<res_T> &res) {
    hls_register typename data_T::value_type data_array[CONFIG_T::height * CONFIG_T::width];

    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / data_T::size; i++) {
        hls_register data_T in_data = data.read();

        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / res_T::size; i++) {
        hls_register res_T out_data;

        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = typename res_T::value_type(data_array[j * data_T::size + i]);
        }

        res.write(out_data);
    }
}

} // namespace nnet

#endif
