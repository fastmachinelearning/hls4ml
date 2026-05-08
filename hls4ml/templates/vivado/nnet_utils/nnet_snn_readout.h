#ifndef NNET_SNN_READOUT_H_
#define NNET_SNN_READOUT_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_snn_common.h"

namespace nnet {

struct snn_readout_config {
    static const unsigned n_classes = 2;
    static const unsigned io_type = io_parallel;
    static const unsigned window_size = 1;
    static const unsigned class_threshold = 1;
    static constexpr float beta = 1.0;
    static const snn_readout_mode output_mode = snn_readout_mode::spike;
    static const snn_decision_rule decision_rule = snn_decision_rule::argmax_spike_count;
    typedef float membrane_t;
};

template <class data_T, class res_T, typename CONFIG_T> void snn_readout(data_T data[CONFIG_T::n_classes], res_T res[1]) {
    #pragma HLS PIPELINE II=1

    // Counts and membrane values persist across calls within one readout window.
    static unsigned counts[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=counts complete
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    if (CONFIG_T::output_mode == snn_readout_mode::membrane) {
        unsigned best = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            typename CONFIG_T::membrane_t v =
                (typename CONFIG_T::membrane_t)((typename CONFIG_T::membrane_t)CONFIG_T::beta * mem[i]) +
                (typename CONFIG_T::membrane_t)data[i];
            mem[i] = v;
            if (i == 0 || v > mem[best]) {
                best = i;
            }
        }

        if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
            res[0] = (typename res_T::value_type)(mem[1] - mem[0]);
        } else {
            res[0] = (typename res_T::value_type)best;
        }

        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                mem[i] = 0;
            }
        }
        return;
    }

    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            counts[i] += (data[i] != 0) ? 1 : 0;
        }
        // Fast BCE-like score path for binary classifiers: logit = count_1 - count_0.
        res[0] = (typename res_T::value_type)((int)counts[1] - (int)counts[0]);
        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                counts[i] = 0;
            }
        }
        return;
    }

    // Fast argmax path: update counters and argmax in one pass.
    if (CONFIG_T::decision_rule == snn_decision_rule::argmax_spike_count) {
        unsigned best = 0;
        unsigned best_count = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            unsigned c = counts[i] + ((data[i] != 0) ? 1 : 0);
            counts[i] = c;
            if (i == 0 || c > best_count) {
                best = i;
                best_count = c;
            }
        }
        res[0] = (typename res_T::value_type)best;

        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                counts[i] = 0;
            }
        }
        return;
    }

    // Generic path for threshold-based decision rules.
    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (data[i] != 0) ? 1 : 0;
    }

    unsigned best = 0;
    for (unsigned i = 1; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        if (counts[i] > counts[best]) {
            best = i;
        }
    }

    if (CONFIG_T::decision_rule == snn_decision_rule::first_to_threshold) {
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            if (counts[i] >= CONFIG_T::class_threshold) {
                best = i;
                break;
            }
        }
    } else if (CONFIG_T::decision_rule == snn_decision_rule::threshold_then_argmax) {
        bool any_reached = false;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            any_reached |= (counts[i] >= CONFIG_T::class_threshold);
        }
        if (any_reached) {
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                if (counts[i] >= CONFIG_T::class_threshold) {
                    best = i;
                    break;
                }
            }
        }
    }

    res[0] = best;

    ts++;
    if (ts >= CONFIG_T::window_size) {
        ts = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            counts[i] = 0;
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void snn_readout(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS PIPELINE II=1

    // Counts and membrane values persist across calls within one readout window.
    static unsigned counts[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=counts complete
    static typename CONFIG_T::membrane_t mem[CONFIG_T::n_classes];
    #pragma HLS ARRAY_PARTITION variable=mem complete
    static unsigned ts = 0;

    data_T in_pack = data_stream.read();

    if (CONFIG_T::output_mode == snn_readout_mode::membrane) {
        unsigned best = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            typename CONFIG_T::membrane_t v =
                (typename CONFIG_T::membrane_t)((typename CONFIG_T::membrane_t)CONFIG_T::beta * mem[i]) +
                (typename CONFIG_T::membrane_t)in_pack[i];
            mem[i] = v;
            if (i == 0 || v > mem[best]) {
                best = i;
            }
        }

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)
        for (unsigned i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            out_pack[i] = 0;
        }
        if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
            out_pack[0] = (typename res_T::value_type)(mem[1] - mem[0]);
        } else {
            out_pack[0] = (typename res_T::value_type)best;
        }
        res_stream.write(out_pack);

        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                mem[i] = 0;
            }
        }
        return;
    }

    if (CONFIG_T::decision_rule == snn_decision_rule::binary_logit) {
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            counts[i] += (in_pack[i] != 0) ? 1 : 0;
        }
        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)
        for (unsigned i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            out_pack[i] = 0;
        }
        out_pack[0] = (typename res_T::value_type)((int)counts[1] - (int)counts[0]);
        res_stream.write(out_pack);

        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                counts[i] = 0;
            }
        }
        return;
    }

    if (CONFIG_T::decision_rule == snn_decision_rule::argmax_spike_count) {
        unsigned best = 0;
        unsigned best_count = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            unsigned c = counts[i] + ((in_pack[i] != 0) ? 1 : 0);
            counts[i] = c;
            if (i == 0 || c > best_count) {
                best = i;
                best_count = c;
            }
        }

        res_T out_pack;
        PRAGMA_DATA_PACK(out_pack)
        for (unsigned i = 0; i < res_T::size; i++) {
            #pragma HLS UNROLL
            out_pack[i] = 0;
        }
        out_pack[0] = (typename res_T::value_type)best;
        res_stream.write(out_pack);

        ts++;
        if (ts >= CONFIG_T::window_size) {
            ts = 0;
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                counts[i] = 0;
            }
        }
        return;
    }

    for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        counts[i] += (in_pack[i] != 0) ? 1 : 0;
    }

    unsigned best = 0;
    for (unsigned i = 1; i < CONFIG_T::n_classes; i++) {
        #pragma HLS UNROLL
        if (counts[i] > counts[best]) {
            best = i;
        }
    }

    if (CONFIG_T::decision_rule == snn_decision_rule::first_to_threshold) {
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            if (counts[i] >= CONFIG_T::class_threshold) {
                best = i;
                break;
            }
        }
    } else if (CONFIG_T::decision_rule == snn_decision_rule::threshold_then_argmax) {
        bool any_reached = false;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            any_reached |= (counts[i] >= CONFIG_T::class_threshold);
        }
        if (any_reached) {
            for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
                #pragma HLS UNROLL
                if (counts[i] >= CONFIG_T::class_threshold) {
                    best = i;
                    break;
                }
            }
        }
    }

    res_T out_pack;
    PRAGMA_DATA_PACK(out_pack)
    for (unsigned i = 0; i < res_T::size; i++) {
        #pragma HLS UNROLL
        out_pack[i] = 0;
    }
    out_pack[0] = (typename res_T::value_type)best;
    res_stream.write(out_pack);

    ts++;
    if (ts >= CONFIG_T::window_size) {
        ts = 0;
        for (unsigned i = 0; i < CONFIG_T::n_classes; i++) {
            #pragma HLS UNROLL
            counts[i] = 0;
        }
    }
}

} // namespace nnet

#endif
