#ifndef NNET_EMBED_H_
#define NNET_EMBED_H_

#include "nnet_common.h"
#include "nnet_helpers.h"

namespace nnet {

struct embed_config {
    // Internal data type definitions
    typedef float embeddings_t;

    // (Default layer sizes, overwritten form the backend
    static const unsigned n_in = 10;
    static const unsigned n_out = 16;
    static const unsigned vocab_size = 50;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

template <class data_T, class res_T, typename CONFIG_T>
void embedding(const data_T &data, res_T &res, const typename CONFIG_T::embeddings_t &embeddings) {

    /*
     * Can store embeddings[] in a register, but a large multiiplexer
     * is created due to a non-constant access pattern
     */

InputSequence:
    #pragma unroll
    [[intel::initiation_interval(CONFIG_T::reuse_factor)]] for (int j = 0; j < CONFIG_T::n_in; j++) {
    DenseEmbedding:
        #pragma unroll
        for (int i = 0; i < CONFIG_T::n_out; i++) {
            res[j * CONFIG_T::n_out + i] = embeddings[data[j].to_uint() * CONFIG_T::n_out + i];
        }
    }
}

} // namespace nnet
#endif
