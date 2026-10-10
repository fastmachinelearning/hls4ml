#ifndef NNET_EMBED_STREAM_H_
#define NNET_EMBED_STREAM_H_

namespace nnet {

template <class data_pipe, class res_pipe, typename CONFIG_T>
void embedding_stream(typename CONFIG_T::embeddings_t embeddings) {

    using res_T = typename ExtractPipeType<res_pipe>::value_type;
    constexpr auto datasize = std::tuple_size<typename ExtractPipeType<data_pipe>::value_type>{};

    auto in_data = data_pipe::read();

InputSequence:
    [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < datasize; j++) {

        res_T res_pack;

    DenseEmbedding:
        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res_pack[i] = embeddings[in_data[j] * CONFIG_T::n_out + i];
        }

        res_pipe::write(res_pack);
    }
}

} // namespace nnet

#endif
