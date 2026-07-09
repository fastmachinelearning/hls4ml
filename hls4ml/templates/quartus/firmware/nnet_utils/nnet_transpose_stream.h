#ifndef NNET_TRANSPOSE_STREAM_H_
#define NNET_TRANSPOSE_STREAM_H_

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T> void transpose_2d(stream<data_T> &data, stream<res_T> &res) {
    if (CONFIG_T::perm[1] == 1 && CONFIG_T::perm[2] == 2) {
        for (int i = 0; i < CONFIG_T::height * CONFIG_T::width / data_T::size; i++) {
            hls_register data_T in_data = data.read();
            hls_register res_T out_data;

            #pragma unroll
            for (int j = 0; j < data_T::size; j++) {
                out_data[j] = typename res_T::value_type(in_data[j]);
            }

            res.write(out_data);
        }

        return;
    }

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

template <typename CONFIG_T> unsigned transpose_3d_stream_idx(unsigned index) {
    static constexpr unsigned dims[3] = {CONFIG_T::depth, CONFIG_T::height, CONFIG_T::width};
    static constexpr unsigned dims_t[3] = {dims[CONFIG_T::perm[0]], dims[CONFIG_T::perm[1]], dims[CONFIG_T::perm[2]]};

    unsigned index_res[3];
    index_res[2] = index % dims_t[2];
    index /= dims_t[2];
    index_res[1] = index % dims_t[1];
    index /= dims_t[1];
    index_res[0] = index;

    unsigned index_data[3];
    index_data[CONFIG_T::perm[0]] = index_res[0];
    index_data[CONFIG_T::perm[1]] = index_res[1];
    index_data[CONFIG_T::perm[2]] = index_res[2];

    return index_data[0] * dims[1] * dims[2] + index_data[1] * dims[2] + index_data[2];
}

template <class data_T, class res_T, typename CONFIG_T> void transpose_3d(stream<data_T> &data, stream<res_T> &res) {
    hls_register typename data_T::value_type data_array[CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width];

    for (int i = 0; i < CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width / data_T::size; i++) {
        hls_register data_T in_data = data.read();

        #pragma unroll
        for (int j = 0; j < data_T::size; j++) {
            data_array[i * data_T::size + j] = typename data_T::value_type(in_data[j]);
        }
    }

    for (int i = 0; i < CONFIG_T::depth * CONFIG_T::height * CONFIG_T::width / res_T::size; i++) {
        hls_register res_T out_data;

        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = typename res_T::value_type(data_array[transpose_3d_stream_idx<CONFIG_T>(i * res_T::size + j)]);
        }

        res.write(out_data);
    }
}

} // namespace nnet

#endif
