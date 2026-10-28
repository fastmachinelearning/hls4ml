#ifndef NNET_RECURSIVE_H_
#define NNET_RECURSIVE_H_

#include "hls_stream.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_dense.h"
#include "nnet_recr_activations.h"

namespace nnet {

// Struct for the LSTM template

struct lstm_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float recurrent_weight_t;
    typedef float bias_t;
    typedef float recurrent_bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_parts = 20;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;

    // Used by the cells of the Resource strategy, described below
    static const unsigned strategy = latency;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 1;
    static const unsigned n_lanes_state = 1;
    static const unsigned n_weight = 16;
    static const unsigned n_recurrent_weight = 16;

    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

// Long Short term Memory NN (LSTM)
// Resources:
// https://github.com/nicodjimenez/lstm/blob/master/lstm.py
// https://github.com/llSourcell/LSTM_Networks/blob/master/LSTM%20Demo.ipynb
// https://en.wikipedia.org/wiki/Long_short-term_memory
// Notes:
//  - LSTM naming conventions adopted from the above links
//      - s_newstate = activation(U*input + W*state)
//      - h_output   = activation(U*input + W*state)*activation(s_newstate)
//  - If softmax is needed on output, perform *outside* this operations
//  Originally had a version that allows for the state in each layer to be saved, moved this to above (this requires a LARGE
//  dense network at the end)
template <class data_T, class res_T, typename CONFIG_T>
void lstm(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
          res_T s_newstate[CONFIG_T::n_state], typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
          typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
          typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
          typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
    // Initialize the state variable -- will maintain state between function calls

    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
    typename CONFIG_T::accum_t s_actstate[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
    #pragma HLS ARRAY_PARTITION variable=s_actstate   complete

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_newstate, tmpres_state, param_r,
                                                                                    param_br);

    for (int iacc = 0; iacc < (3 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        if (iacc > 2 * CONFIG_T::n_state - 1)
            index = iacc + CONFIG_T::n_state;
        inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
    }
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
    }

    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                       typename CONFIG_T::ACT_CONFIG_LSTM>::activation(inputacc_ifo, tmpres_ifo);

    // Now for the confusion matrix
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                  typename CONFIG_T::ACT_CONFIG_T>::activation(inputacc_c, tmpres_c);

    // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        s_newstate[iacc] = tmpres_c[iacc] * tmpres_ifo[iacc] + s_newstate[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
    }
    // Operation: h=act(s)*o
    CONFIG_T::template activation<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        s_newstate, s_actstate);

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        h_newstate[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
    }
}

template <class data_T, class res_T, typename CONFIG_T, bool backward = false>
void lstm_static(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
                 res_T s_newstate[CONFIG_T::n_state],
                 typename CONFIG_T::weight_t param[CONFIG_T::n_state * 4 * CONFIG_T::n_in],
                 typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_state * 4 * CONFIG_T::n_state],
                 typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                 typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
    static res_T h_state[CONFIG_T::n_state];
    static res_T s_state[CONFIG_T::n_state];
    // Initialize the state variable -- will maintain state between function calls
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state * 4];
    typename CONFIG_T::accum_t tmpres_ifo[CONFIG_T::n_state * 3];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_c[CONFIG_T::n_state];         // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_ifo[CONFIG_T::n_state * 3]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_c[CONFIG_T::n_state];       // c-matrix (keras notation)
    typename CONFIG_T::accum_t s_actstate[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=h_state      complete
    #pragma HLS ARRAY_PARTITION variable=s_state      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_ifo   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_c     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_ifo complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_c   complete
    #pragma HLS ARRAY_PARTITION variable=s_actstate   complete

    if (reset_state) {
        for (int i_state = 0; i_state < (CONFIG_T::n_state); i_state++) {
            #pragma HLS UNROLL
            s_state[i_state] = 0;
            h_state[i_state] = 0;
        }
    }
    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_state, tmpres_state, param_r,
                                                                                    param_br);

    for (int iacc = 0; iacc < (3 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        if (iacc > 2 * CONFIG_T::n_state - 1)
            index = iacc + CONFIG_T::n_state;
        inputacc_ifo[iacc] = tmpres[index] + tmpres_state[index];
    }
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_c[iacc] = tmpres[index] + tmpres_state[index];
    }

    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                       typename CONFIG_T::ACT_CONFIG_LSTM>::activation(inputacc_ifo, tmpres_ifo);

    // Now for the confusion matrix
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                  typename CONFIG_T::ACT_CONFIG_T>::activation(inputacc_c, tmpres_c);

    // Operation: s=g*i+sold*f (update state with buffer to avoid timing issues)
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        s_state[iacc] = tmpres_c[iacc] * tmpres_ifo[iacc] + s_state[iacc] * tmpres_ifo[iacc + (CONFIG_T::n_state)];
        s_newstate[iacc] = s_state[iacc];
    }
    // Operation: h=act(s)*o
    CONFIG_T::template activation<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::ACT_CONFIG_T>::activation(
        s_state, s_actstate);

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        h_state[iacc] = tmpres_ifo[iacc + 2 * (CONFIG_T::n_state)] * s_actstate[iacc];
        h_newstate[iacc] = h_state[iacc];
    }
}

// *************************************************
//       Cells of the Resource strategy
// *************************************************
// A step computes the units n_unit_par at a time, spending n_cycles cycles on each group, and updates the state of each
// unit as soon as its gates are done. The input and the state are multiplied by separate groups of multipliers,
// n_lanes_in and n_lanes_state for each gate of each unit, so a step takes n_state / n_unit_par * n_cycles cycles,
// the reuse factor of the layer. Of the ways to get a reuse factor, the backend uses the one with the fewest units at a
// time, since each of them needs its own activations and state update. It lays out the weights as
// [group][cycle][unit][gate][lane], with zeros past the end of the input or the state, so the arrays are n_weight and
// n_recurrent_weight long, and the biases as [unit][gate], with the gates in Keras order. The results are those of the
// cells above.

// The sums of one cycle: each lane adds the product of one input or state value and one weight. step is the iteration of
// the loop over all groups and cycles, so the weights of a lane are read from consecutive addresses of its own memory.
template <class data_T, class res_T, typename CONFIG_T, unsigned n_gates>
void recurrent_resource_mac(unsigned step, unsigned cycle, data_T data[CONFIG_T::n_in], res_T state[CONFIG_T::n_state],
                            typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                            typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                            typename CONFIG_T::accum_t sum_in[CONFIG_T::n_unit_par * n_gates * CONFIG_T::n_lanes_in],
                            typename CONFIG_T::accum_t sum_state[CONFIG_T::n_unit_par * n_gates * CONFIG_T::n_lanes_state]) {
    #pragma HLS INLINE
    const unsigned lanes_in = CONFIG_T::n_unit_par * n_gates * CONFIG_T::n_lanes_in;
    const unsigned lanes_state = CONFIG_T::n_unit_par * n_gates * CONFIG_T::n_lanes_state;

    for (unsigned l = 0; l < lanes_in; l++) {
        #pragma HLS UNROLL
        unsigned i = cycle * CONFIG_T::n_lanes_in + l % CONFIG_T::n_lanes_in;
        typename CONFIG_T::accum_t term = 0;
        if (i < CONFIG_T::n_in)
            term = CONFIG_T::mult_config1::template product<data_T, typename CONFIG_T::weight_t>::product(
                data[i], param[step * lanes_in + l]);
        sum_in[l] = cycle == 0 ? term : (typename CONFIG_T::accum_t)(sum_in[l] + term);
    }
    for (unsigned l = 0; l < lanes_state; l++) {
        #pragma HLS UNROLL
        unsigned i = cycle * CONFIG_T::n_lanes_state + l % CONFIG_T::n_lanes_state;
        typename CONFIG_T::accum_t term = 0;
        if (i < CONFIG_T::n_state)
            term = CONFIG_T::mult_config2::template product<res_T, typename CONFIG_T::recurrent_weight_t>::product(
                state[i], param_r[step * lanes_state + l]);
        sum_state[l] = cycle == 0 ? term : (typename CONFIG_T::accum_t)(sum_state[l] + term);
    }
}

// The sum of the lanes of one gate of one unit, started from the bias
template <typename CONFIG_T, unsigned n_gates, unsigned n_lanes>
typename CONFIG_T::accum_t recurrent_resource_total(typename CONFIG_T::accum_t sum[CONFIG_T::n_unit_par * n_gates * n_lanes],
                                                    unsigned unit, unsigned gate, typename CONFIG_T::accum_t bias) {
    #pragma HLS INLINE
    typename CONFIG_T::accum_t total = bias;
    for (unsigned p = 0; p < n_lanes; p++) {
        #pragma HLS UNROLL
        total += sum[(unit * n_gates + gate) * n_lanes + p];
    }
    return total;
}

// Lookup tables of the activations, one for each value computed in the same cycle. They are filled once in C
// simulation, as in nnet_activation.h, and on every call in synthesis, where they become ROMs.
#ifdef __HLS_SYN__
#define NNET_RECURRENT_TABLE
#else
#define NNET_RECURRENT_TABLE static
#endif

template <class data_T, class res_T, typename CONFIG_T>
void lstm_resource(data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state], res_T s_state[CONFIG_T::n_state],
                   typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                   typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                   typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                   typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::ACT_CONFIG_LSTM recr_config;
    typedef typename CONFIG_T::ACT_CONFIG_T act_config;
    typedef typename CONFIG_T::template activation_recr<accum_t, accum_t, recr_config> recr_act;
    typedef typename CONFIG_T::template activation<accum_t, accum_t, act_config> act;
    typedef typename CONFIG_T::template activation<res_T, accum_t, act_config> state_act;

    const unsigned n_par = CONFIG_T::n_unit_par;
    const unsigned lanes_in = n_par * 4 * CONFIG_T::n_lanes_in;
    const unsigned lanes_state = n_par * 4 * CONFIG_T::n_lanes_state;
    const unsigned bias_banks = n_par * 4;
    #pragma HLS ARRAY_PARTITION variable=param cyclic factor=lanes_in
    #pragma HLS ARRAY_PARTITION variable=param_r cyclic factor=lanes_state
    #pragma HLS ARRAY_PARTITION variable=param_b cyclic factor=bias_banks
    #pragma HLS ARRAY_PARTITION variable=param_br cyclic factor=bias_banks
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=s_state complete

    NNET_RECURRENT_TABLE bool initialized = false;
    NNET_RECURRENT_TABLE typename recr_config::table_t recr_table[n_par][3][recr_config::table_size];
    NNET_RECURRENT_TABLE typename act_config::table_t act_table[n_par][2][act_config::table_size];
    #pragma HLS ARRAY_PARTITION variable=recr_table complete dim=1
    #pragma HLS ARRAY_PARTITION variable=recr_table complete dim=2
    #pragma HLS ARRAY_PARTITION variable=act_table complete dim=1
    #pragma HLS ARRAY_PARTITION variable=act_table complete dim=2
    if (!initialized) {
        for (unsigned u = 0; u < n_par; u++) {
            #pragma HLS UNROLL
            for (unsigned k = 0; k < 3; k++) {
                #pragma HLS UNROLL
                recr_act::init_table(recr_table[u][k]);
            }
            for (unsigned k = 0; k < 2; k++) {
                #pragma HLS UNROLL
                act::init_table(act_table[u][k]);
            }
        }
        initialized = true;
    }

    accum_t sum_in[lanes_in];
    accum_t sum_state[lanes_state];
    res_T h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=sum_in complete
    #pragma HLS ARRAY_PARTITION variable=sum_state complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete

    unsigned group = 0, cycle = 0;
LSTMUnits:
    for (unsigned it = 0; it < CONFIG_T::n_state / n_par * CONFIG_T::n_cycles; it++) {
        #pragma HLS PIPELINE II=1
        recurrent_resource_mac<data_T, res_T, CONFIG_T, 4>(it, cycle, data, h_state, param, param_r, sum_in, sum_state);
        if (cycle == CONFIG_T::n_cycles - 1) {
            for (unsigned up = 0; up < n_par; up++) {
                #pragma HLS UNROLL
                unsigned u = group * n_par + up;
                accum_t gate[4];
                #pragma HLS ARRAY_PARTITION variable=gate complete
                for (unsigned g = 0; g < 4; g++) {
                    #pragma HLS UNROLL
                    accum_t from_in =
                        recurrent_resource_total<CONFIG_T, 4, CONFIG_T::n_lanes_in>(sum_in, up, g, param_b[u * 4 + g]);
                    accum_t from_state = recurrent_resource_total<CONFIG_T, 4, CONFIG_T::n_lanes_state>(sum_state, up, g,
                                                                                                        param_br[u * 4 + g]);
                    gate[g] = from_in + from_state;
                }
                accum_t gate_i = recr_act::single(gate[0], recr_table[up][0]);
                accum_t gate_f = recr_act::single(gate[1], recr_table[up][1]);
                accum_t gate_o = recr_act::single(gate[3], recr_table[up][2]);
                accum_t gate_c = act::single(gate[2], act_table[up][0]);
                s_state[u] = gate_c * gate_i + s_state[u] * gate_f;
                h_newstate[u] = gate_o * state_act::single(s_state[u], act_table[up][1]);
            }
        }
        if (++cycle == CONFIG_T::n_cycles) {
            cycle = 0;
            group++;
        }
    }

    for (unsigned u = 0; u < CONFIG_T::n_state; u++) {
        #pragma HLS UNROLL
        h_state[u] = h_newstate[u];
    }
}

// One step of the LSTM: the cell of the Resource strategy, or the static or the non-static cell built from Dense calls
template <class data_T, class res_T, typename CONFIG_T, bool use_static,
          bool resource = CONFIG_T::strategy == nnet::resource>
struct lstm_cell {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      res_T s_state[CONFIG_T::n_state], typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
        #pragma HLS INLINE
        if (use_static)
            nnet::lstm_static<data_T, res_T, CONFIG_T>(reset_state, data, h_state, s_state, param, param_r, param_b,
                                                       param_br);
        else
            nnet::lstm<data_T, res_T, CONFIG_T>(reset_state, data, h_state, s_state, param, param_r, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T, bool use_static>
struct lstm_cell<data_T, res_T, CONFIG_T, use_static, true> {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      res_T s_state[CONFIG_T::n_state], typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
        #pragma HLS INLINE
        nnet::lstm_resource<data_T, res_T, CONFIG_T>(data, h_state, s_state, param, param_r, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T, bool backward = false> class lstm_class {
  public:
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_total[2 * CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
        res_T h_newstate[CONFIG_T::n_state];
        res_T s_newstate[CONFIG_T::n_state];
        #pragma HLS ARRAY_PARTITION variable=h_newstate complete
        #pragma HLS ARRAY_PARTITION variable=s_newstate complete
        for (int i_state = 0; i_state < CONFIG_T::n_state; i_state++) {
            #pragma HLS UNROLL
            h_newstate[i_state] = h_total[i_state];
            s_newstate[i_state] = h_total[i_state + CONFIG_T::n_state];
        }
        nnet::lstm_cell<data_T, res_T, CONFIG_T, false>::apply(reset_state, data, h_newstate, s_newstate, param, param_r,
                                                               param_b, param_br);
        for (int i_state = 0; i_state < CONFIG_T::n_state; i_state++) {
            #pragma HLS UNROLL
            h_total[i_state] = h_newstate[i_state];
            h_total[i_state + CONFIG_T::n_state] = s_newstate[i_state];
        }
    };
};

template <class data_T, class res_T, typename CONFIG_T, bool backward = false> class lstm_class_static {
  public:
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_total[2 * CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {
        res_T h_newstate[CONFIG_T::n_state];
        res_T s_newstate[CONFIG_T::n_state];
        #pragma HLS ARRAY_PARTITION variable=h_newstate complete
        #pragma HLS ARRAY_PARTITION variable=s_newstate complete
        nnet::lstm_static<data_T, res_T, CONFIG_T, backward>(reset_state, data, h_newstate, s_newstate, param, param_r,
                                                             param_b, param_br);
        for (int i_state = 0; i_state < CONFIG_T::n_state; i_state++) {
            #pragma HLS UNROLL
            h_total[i_state] = h_newstate[i_state];
            h_total[i_state + CONFIG_T::n_state] = s_newstate[i_state];
        }
    };
};

template <class data_T, class res_T, typename CONFIG_T>
void lstm_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
                typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {

    res_T h_newstate[CONFIG_T::n_state];
    res_T s_newstate[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
        s_newstate[ii] = 0;
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        nnet::lstm_cell<data_T, res_T, CONFIG_T, CONFIG_T::use_static>::apply(reset_state, data_in, h_newstate, s_newstate,
                                                                              param, param_r, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_newstate[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_newstate[i];
        }
}

template <class data_T, class h_T, class s_T, class res_T, typename CONFIG_T>
void lstm_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], h_T h_init[CONFIG_T::n_state],
                s_T s_init[CONFIG_T::n_state], res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
                typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {

    res_T h_newstate[CONFIG_T::n_state];
    res_T s_newstate[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = false;

    #pragma HLS ARRAY_PARTITION variable=h_init complete
    #pragma HLS ARRAY_PARTITION variable=s_init complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = h_init[ii];
        s_newstate[ii] = s_init[ii];
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }

        nnet::lstm_cell<data_T, res_T, CONFIG_T, false>::apply(reset_state, data_in, h_newstate, s_newstate, param, param_r,
                                                               param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_newstate[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_newstate[i];
        }
}

template <class data_T, class res_T, typename CONFIG_T>
void lstm_stack(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 4],
                typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 4]) {

    typename res_T::value_type h_newstate[CONFIG_T::n_state];
    typename res_T::value_type s_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=s_newstate complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
        s_newstate[ii] = 0;
    }

    typename data_T::value_type data_in[CONFIG_T::n_in];
    bool reset_state = true;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size > 1) {
            // #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data_in[i_pack] = data_pack[i_pack];
        }
        // The time steps run one after another, so one cell is shared whether or not it is static.
        nnet::lstm_cell<typename data_T::value_type, typename res_T::value_type, CONFIG_T, false>::apply(
            reset_state, data_in, h_newstate, s_newstate, param, param_r, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack_sequences:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = h_newstate[i_pack];
            }
            res_stream.write(res_pack);
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = h_newstate[i_pack];
        }
        res_stream.write(res_pack);
    }
}

// Struct for the GRU template

struct gru_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float recurrent_weight_t;
    typedef float bias_t;
    typedef float recurrent_bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_sequence = 2;
    static const unsigned n_4state = 8;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const bool pytorch_order = false;
    static const bool reset_after = true;
    static const unsigned n_zeros = 0;

    static const unsigned strategy = latency;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 1;
    static const unsigned n_lanes_state = 1;
    static const unsigned n_weight = 12;
    static const unsigned n_recurrent_weight = 12;

    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

template <class res_T, typename CONFIG_T>
void gru_candidate_recurrent(res_T h_state[CONFIG_T::n_state], typename CONFIG_T::accum_t tmpres_zr[CONFIG_T::n_state * 2],
                             typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state * 3],
                             typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
                             typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3],
                             typename CONFIG_T::accum_t tmpres_state_h[CONFIG_T::n_state]) {
    #pragma HLS INLINE

    if (CONFIG_T::reset_after) {
        for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
            #pragma HLS UNROLL
            if (CONFIG_T::pytorch_order)
                tmpres_state_h[iacc] = tmpres_zr[iacc] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
            else
                tmpres_state_h[iacc] =
                    tmpres_zr[iacc + (CONFIG_T::n_state)] * tmpres_state_zr[iacc + (2 * CONFIG_T::n_state)];
        }
    } else {
        res_T h_reset[CONFIG_T::n_state];
        typename CONFIG_T::accum_t tmpres_state_reset[CONFIG_T::n_state * 3];
        #pragma HLS ARRAY_PARTITION variable=h_reset            complete
        #pragma HLS ARRAY_PARTITION variable=tmpres_state_reset complete

        for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
            #pragma HLS UNROLL
            h_reset[iacc] = (res_T)(tmpres_zr[iacc + (CONFIG_T::n_state)] * h_state[iacc]);
        }

        nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_reset, tmpres_state_reset,
                                                                                        param_zr, param_br);

        // Keep the h-matrix part (keras notation)
        for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
            #pragma HLS UNROLL
            tmpres_state_h[iacc] = tmpres_state_reset[iacc + (2 * CONFIG_T::n_state)];
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
         typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in], // TODO - Check the layout of the param
                                                                                    // weights - refer page in copy!!
         typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
         typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
         typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
    // Initialize the state variable -- will maintain state between function calls
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_h[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr[CONFIG_T::n_state * 2];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h[CONFIG_T::n_state];        // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr[CONFIG_T::n_state * 2]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h[CONFIG_T::n_state];      // c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_newstate, tmpres_state_zr, param_zr,
                                                                                    param_br);
    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres
    // initialized with biases -- DONE
    for (int iacc = 0; iacc < (2 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                       typename CONFIG_T::ACT_CONFIG_GRU>::activation(inputacc_zr, tmpres_zr);

    // Activation function Sub layer -- END

    nnet::gru_candidate_recurrent<res_T, CONFIG_T>(h_newstate, tmpres_zr, tmpres_state_zr, param_zr, param_br,
                                                   tmpres_state_h);

    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_h[iacc] = tmpres[index] + tmpres_state_h[iacc];
    }

    // Now run the activation on this guy
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                  typename CONFIG_T::ACT_CONFIG_T>::activation(inputacc_h, tmpres_h);

    // Mix the stat with the previous state
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            h_newstate[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc + (CONFIG_T::n_state)]) +
                                       h_newstate[iacc] * tmpres_zr[iacc + (CONFIG_T::n_state)]);
        else
            h_newstate[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc]) + h_newstate[iacc] * tmpres_zr[iacc]);
    }
}

template <class data_T, class res_T, typename CONFIG_T, bool backward = false>
void gru_static(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
                typename CONFIG_T::weight_t param[CONFIG_T::n_state * 3 * CONFIG_T::n_in],
                typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_state * 3 * CONFIG_T::n_state],
                typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
    static res_T h_state[CONFIG_T::n_state];
    // Initialize the state variable -- will maintain state between function calls
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_zr[CONFIG_T::n_state * 3];
    typename CONFIG_T::accum_t tmpres_state_h[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_zr[CONFIG_T::n_state * 2];   // activated i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t tmpres_h[CONFIG_T::n_state];        // activated c-matrix (keras notation)
    typename CONFIG_T::accum_t inputacc_zr[CONFIG_T::n_state * 2]; // i,f,o matrices (keras notation)
    typename CONFIG_T::accum_t inputacc_h[CONFIG_T::n_state];      // c-matrix (keras notation)

    #pragma HLS ARRAY_PARTITION variable=h_state         complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres          complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_zr complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state_h  complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_zr       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_h        complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_zr     complete
    #pragma HLS ARRAY_PARTITION variable=inputacc_h      complete

    if (reset_state) {
        for (int i_h_state = 0; i_h_state < (CONFIG_T::n_state); i_h_state++) {
            #pragma HLS UNROLL
            h_state[i_h_state] = 0;
        }
    }

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_state, tmpres_state_zr, param_zr,
                                                                                    param_br);

    // Adding the individual vectors from the multiplication of tmpres = Wx*x(t); tmpres_state_zr = Wh*h(t-1); tmpres
    // initialized with biases -- DONE
    for (int iacc = 0; iacc < (2 * CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc;
        inputacc_zr[iacc] = tmpres[index] + tmpres_state_zr[index];
    }

    // Activation function Sub layer -- START
    CONFIG_T::template activation_recr<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                       typename CONFIG_T::ACT_CONFIG_GRU>::activation(inputacc_zr, tmpres_zr);

    // Activation function Sub layer -- END

    nnet::gru_candidate_recurrent<res_T, CONFIG_T>(h_state, tmpres_zr, tmpres_state_zr, param_zr, param_br, tmpres_state_h);

    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        int index = iacc + CONFIG_T::n_state * 2;
        inputacc_h[iacc] = tmpres[index] + tmpres_state_h[iacc];
    }

    // Now run the activation on this guy
    CONFIG_T::template activation<typename CONFIG_T::accum_t, typename CONFIG_T::accum_t,
                                  typename CONFIG_T::ACT_CONFIG_T>::activation(inputacc_h, tmpres_h);

    // Mix the stat with the previous state
    for (int iacc = 0; iacc < (CONFIG_T::n_state); iacc++) {
        #pragma HLS UNROLL
        if (CONFIG_T::pytorch_order)
            h_state[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc + (CONFIG_T::n_state)]) +
                                    h_state[iacc] * tmpres_zr[iacc + (CONFIG_T::n_state)]);
        else
            h_state[iacc] = (res_T)(tmpres_h[iacc] * (1 - tmpres_zr[iacc]) + h_state[iacc] * tmpres_zr[iacc]);
        h_newstate[iacc] = h_state[iacc];
    }
}

// The GRU cell of the Resource strategy (see lstm_resource). When the reset gate is applied before the recurrent
// multiplication (reset_after false), the candidate needs the reset gate of every unit, so a step takes two passes over
// the units: one for the update and reset gates, one for the candidate.
template <class data_T, class res_T, typename CONFIG_T>
void gru_resource(data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                  typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                  typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
                  typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                  typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::ACT_CONFIG_GRU recr_config;
    typedef typename CONFIG_T::ACT_CONFIG_T act_config;
    typedef typename CONFIG_T::template activation_recr<accum_t, accum_t, recr_config> recr_act;
    typedef typename CONFIG_T::template activation<accum_t, accum_t, act_config> act;

    const unsigned n_par = CONFIG_T::n_unit_par;
    const unsigned lanes_in = n_par * 3 * CONFIG_T::n_lanes_in;
    const unsigned lanes_state = n_par * 3 * CONFIG_T::n_lanes_state;
    const unsigned bias_banks = n_par * 3;
    const unsigned n_iter = CONFIG_T::n_state / n_par * CONFIG_T::n_cycles;
    #pragma HLS ARRAY_PARTITION variable=param cyclic factor=lanes_in
    #pragma HLS ARRAY_PARTITION variable=param_zr cyclic factor=lanes_state
    #pragma HLS ARRAY_PARTITION variable=param_b cyclic factor=bias_banks
    #pragma HLS ARRAY_PARTITION variable=param_br cyclic factor=bias_banks
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=h_state complete

    NNET_RECURRENT_TABLE bool initialized = false;
    NNET_RECURRENT_TABLE typename recr_config::table_t recr_table[n_par][2][recr_config::table_size];
    NNET_RECURRENT_TABLE typename act_config::table_t act_table[n_par][act_config::table_size];
    #pragma HLS ARRAY_PARTITION variable=recr_table complete dim=1
    #pragma HLS ARRAY_PARTITION variable=recr_table complete dim=2
    #pragma HLS ARRAY_PARTITION variable=act_table complete dim=1
    if (!initialized) {
        for (unsigned u = 0; u < n_par; u++) {
            #pragma HLS UNROLL
            for (unsigned k = 0; k < 2; k++) {
                #pragma HLS UNROLL
                recr_act::init_table(recr_table[u][k]);
            }
            act::init_table(act_table[u]);
        }
        initialized = true;
    }

    accum_t sum_in[lanes_in];
    accum_t sum_state[lanes_state];
    res_T h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=sum_in complete
    #pragma HLS ARRAY_PARTITION variable=sum_state complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete

    if (CONFIG_T::reset_after) {
        unsigned group = 0, cycle = 0;
    GRUUnits:
        for (unsigned it = 0; it < n_iter; it++) {
            #pragma HLS PIPELINE II=1
            recurrent_resource_mac<data_T, res_T, CONFIG_T, 3>(it, cycle, data, h_state, param, param_zr, sum_in, sum_state);
            if (cycle == CONFIG_T::n_cycles - 1) {
                for (unsigned up = 0; up < n_par; up++) {
                    #pragma HLS UNROLL
                    unsigned u = group * n_par + up;
                    accum_t from_in[3], from_state[3];
                    #pragma HLS ARRAY_PARTITION variable=from_in complete
                    #pragma HLS ARRAY_PARTITION variable=from_state complete
                    for (unsigned g = 0; g < 3; g++) {
                        #pragma HLS UNROLL
                        from_in[g] =
                            recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_in>(sum_in, up, g, param_b[u * 3 + g]);
                        from_state[g] = recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_state>(sum_state, up, g,
                                                                                                       param_br[u * 3 + g]);
                    }
                    accum_t sum_z = from_in[0] + from_state[0];
                    accum_t sum_r = from_in[1] + from_state[1];
                    accum_t gate_z = recr_act::single(sum_z, recr_table[up][0]);
                    accum_t gate_r = recr_act::single(sum_r, recr_table[up][1]);
                    accum_t reset_h = gate_r * from_state[2];
                    accum_t sum_h = from_in[2] + reset_h;
                    accum_t gate_h = act::single(sum_h, act_table[up]);
                    h_newstate[u] = (res_T)(gate_h * (1 - gate_z) + h_state[u] * gate_z);
                }
            }
            if (++cycle == CONFIG_T::n_cycles) {
                cycle = 0;
                group++;
            }
        }
    } else {
        accum_t gate_z[CONFIG_T::n_state], cand_in[CONFIG_T::n_state];
        res_T h_reset[CONFIG_T::n_state];
        #pragma HLS ARRAY_PARTITION variable=gate_z complete
        #pragma HLS ARRAY_PARTITION variable=cand_in complete
        #pragma HLS ARRAY_PARTITION variable=h_reset complete

        unsigned group = 0, cycle = 0;
    GRUGates:
        for (unsigned it = 0; it < n_iter; it++) {
            #pragma HLS PIPELINE II=1
            recurrent_resource_mac<data_T, res_T, CONFIG_T, 3>(it, cycle, data, h_state, param, param_zr, sum_in, sum_state);
            if (cycle == CONFIG_T::n_cycles - 1) {
                for (unsigned up = 0; up < n_par; up++) {
                    #pragma HLS UNROLL
                    unsigned u = group * n_par + up;
                    accum_t sum_z =
                        recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_in>(sum_in, up, 0, param_b[u * 3]) +
                        recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_state>(sum_state, up, 0, param_br[u * 3]);
                    accum_t sum_r =
                        recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_in>(sum_in, up, 1, param_b[u * 3 + 1]) +
                        recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_state>(sum_state, up, 1,
                                                                                       param_br[u * 3 + 1]);
                    gate_z[u] = recr_act::single(sum_z, recr_table[up][0]);
                    accum_t gate_r = recr_act::single(sum_r, recr_table[up][1]);
                    cand_in[u] =
                        recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_in>(sum_in, up, 2, param_b[u * 3 + 2]);
                    h_reset[u] = (res_T)(gate_r * h_state[u]);
                }
            }
            if (++cycle == CONFIG_T::n_cycles) {
                cycle = 0;
                group++;
            }
        }

        group = 0;
        cycle = 0;
    GRUCandidate:
        for (unsigned it = 0; it < n_iter; it++) {
            #pragma HLS PIPELINE II=1
            recurrent_resource_mac<data_T, res_T, CONFIG_T, 3>(it, cycle, data, h_reset, param, param_zr, sum_in, sum_state);
            if (cycle == CONFIG_T::n_cycles - 1) {
                for (unsigned up = 0; up < n_par; up++) {
                    #pragma HLS UNROLL
                    unsigned u = group * n_par + up;
                    accum_t sum_h = cand_in[u] + recurrent_resource_total<CONFIG_T, 3, CONFIG_T::n_lanes_state>(
                                                     sum_state, up, 2, param_br[u * 3 + 2]);
                    accum_t gate_h = act::single(sum_h, act_table[up]);
                    h_newstate[u] = (res_T)(gate_h * (1 - gate_z[u]) + h_state[u] * gate_z[u]);
                }
            }
            if (++cycle == CONFIG_T::n_cycles) {
                cycle = 0;
                group++;
            }
        }
    }

    for (unsigned u = 0; u < CONFIG_T::n_state; u++) {
        #pragma HLS UNROLL
        h_state[u] = h_newstate[u];
    }
}

// One step of the GRU, as lstm_cell
template <class data_T, class res_T, typename CONFIG_T, bool use_static,
          bool resource = CONFIG_T::strategy == nnet::resource>
struct gru_cell {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
        #pragma HLS INLINE
        if (use_static)
            nnet::gru_static<data_T, res_T, CONFIG_T>(reset_state, data, h_state, param, param_zr, param_b, param_br);
        else
            nnet::gru<data_T, res_T, CONFIG_T>(reset_state, data, h_state, param, param_zr, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T, bool use_static>
struct gru_cell<data_T, res_T, CONFIG_T, use_static, true> {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
        #pragma HLS INLINE
        nnet::gru_resource<data_T, res_T, CONFIG_T>(data, h_state, param, param_zr, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T, bool backward = false> struct gru_class {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
        nnet::gru_cell<data_T, res_T, CONFIG_T, false>::apply(reset_state, data, h_state, param, param_zr, param_b,
                                                              param_br);
    };
};

template <class data_T, class res_T, typename CONFIG_T, bool backward = false> struct gru_class_static {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {
        nnet::gru_static<data_T, res_T, CONFIG_T, backward>(reset_state, data, h_state, param, param_zr, param_b, param_br);
    };
};

template <class data_T, class res_T, typename CONFIG_T>
void gru_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
               typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
               typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
               typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
               typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {

    res_T h_state[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=data_in complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_state[ii] = 0;
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        nnet::gru_cell<data_T, res_T, CONFIG_T, CONFIG_T::use_static>::apply(reset_state, data_in, h_state, param, param_zr,
                                                                             param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_state[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_state[i];
        }
}

template <class data_T, class h_T, class res_T, typename CONFIG_T>
void gru_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], h_T h_init[CONFIG_T::n_state],
               res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
               typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
               typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
               typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
               typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {

    res_T h_state[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = false;

    #pragma HLS ARRAY_PARTITION variable=h_init complete
    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=data_in complete
    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_state[ii] = h_init[ii];
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        nnet::gru_cell<data_T, res_T, CONFIG_T, false>::apply(reset_state, data_in, h_state, param, param_zr, param_b,
                                                              param_br);

        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_state[j];
            }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_state[i];
        }
}

template <class data_T, class res_T, typename CONFIG_T>
void gru_stack(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
               typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
               typename CONFIG_T::recurrent_weight_t param_zr[CONFIG_T::n_recurrent_weight],
               typename CONFIG_T::bias_t param_b[CONFIG_T::n_state * 3],
               typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state * 3]) {

    typename res_T::value_type h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
    }

    typename data_T::value_type data_in[CONFIG_T::n_in];
    bool reset_state = true;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size; i_in++) {
        if (CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size > 1) {
            // #pragma HLS PIPELINE
        }
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data_in[i_pack] = data_pack[i_pack];
        }
        // The time steps run one after another, so one cell is shared whether or not it is static. The static
        // cell is not used here: with it, Vivado HLS outputs zeros when only the last time step is returned.
        nnet::gru_cell<typename data_T::value_type, typename res_T::value_type, CONFIG_T, false>::apply(
            reset_state, data_in, h_newstate, param, param_zr, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack_sequences:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = h_newstate[i_pack];
            }
            res_stream.write(res_pack);
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = h_newstate[i_pack];
        }
        res_stream.write(res_pack);
    }
}

// Struct for the SimpleRNN template

struct simple_rnn_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float recurrent_weight_t;
    typedef float bias_t;
    typedef float recurrent_bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned n_sequence = 2;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;
    static const unsigned n_zeros = 0;

    static const unsigned strategy = latency;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 1;
    static const unsigned n_lanes_state = 1;
    static const unsigned n_weight = 4;
    static const unsigned n_recurrent_weight = 4;

    template <class x_T, class y_T, class config_T> using activation = nnet::activation::tanh<x_T, y_T, config_T>;
};

// Simple recurrent cell (SimpleRNN in Keras, RNN in PyTorch):
//  - h_newstate = activation(W*input + b + U*h_state + br)
//  - Keras has no recurrent bias, so br is zero for Keras models
template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
                typename CONFIG_T::weight_t param[CONFIG_T::n_state * CONFIG_T::n_in],
                typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_state * CONFIG_T::n_state],
                typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state];
    typename CONFIG_T::accum_t inputacc[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=inputacc     complete

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_newstate, tmpres_state, param_r,
                                                                                    param_br);

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        inputacc[iacc] = tmpres[iacc] + tmpres_state[iacc];
    }

    CONFIG_T::template activation<typename CONFIG_T::accum_t, res_T, typename CONFIG_T::ACT_CONFIG_T>::activation(
        inputacc, h_newstate);
}

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_static(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_newstate[CONFIG_T::n_state],
                       typename CONFIG_T::weight_t param[CONFIG_T::n_state * CONFIG_T::n_in],
                       typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_state * CONFIG_T::n_state],
                       typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                       typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {
    static res_T h_state[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres[CONFIG_T::n_state];
    typename CONFIG_T::accum_t tmpres_state[CONFIG_T::n_state];
    typename CONFIG_T::accum_t inputacc[CONFIG_T::n_state];

    #pragma HLS ARRAY_PARTITION variable=h_newstate   complete
    #pragma HLS ARRAY_PARTITION variable=h_state      complete
    #pragma HLS ARRAY_PARTITION variable=tmpres       complete
    #pragma HLS ARRAY_PARTITION variable=tmpres_state complete
    #pragma HLS ARRAY_PARTITION variable=inputacc     complete

    if (reset_state) {
        for (int i_state = 0; i_state < (CONFIG_T::n_state); i_state++) {
            #pragma HLS UNROLL
            h_state[i_state] = 0;
        }
    }

    nnet::dense<data_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config1>(data, tmpres, param, param_b);
    nnet::dense<res_T, typename CONFIG_T::accum_t, typename CONFIG_T::mult_config2>(h_state, tmpres_state, param_r,
                                                                                    param_br);

    for (int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        #pragma HLS UNROLL
        inputacc[iacc] = tmpres[iacc] + tmpres_state[iacc];
    }

    CONFIG_T::template activation<typename CONFIG_T::accum_t, res_T, typename CONFIG_T::ACT_CONFIG_T>::activation(inputacc,
                                                                                                                  h_state);

    for (int i_state = 0; i_state < (CONFIG_T::n_state); i_state++) {
        #pragma HLS UNROLL
        h_newstate[i_state] = h_state[i_state];
    }
}

// The SimpleRNN cell of the Resource strategy (see lstm_resource)
template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_resource(data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                         typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                         typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                         typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                         typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {
    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::ACT_CONFIG_T act_config;
    typedef typename CONFIG_T::template activation<accum_t, res_T, act_config> act;

    const unsigned n_par = CONFIG_T::n_unit_par;
    const unsigned lanes_in = n_par * CONFIG_T::n_lanes_in;
    const unsigned lanes_state = n_par * CONFIG_T::n_lanes_state;
    #pragma HLS ARRAY_PARTITION variable=param cyclic factor=lanes_in
    #pragma HLS ARRAY_PARTITION variable=param_r cyclic factor=lanes_state
    #pragma HLS ARRAY_PARTITION variable=param_b cyclic factor=n_par
    #pragma HLS ARRAY_PARTITION variable=param_br cyclic factor=n_par
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=h_state complete

    NNET_RECURRENT_TABLE bool initialized = false;
    NNET_RECURRENT_TABLE typename act_config::table_t act_table[n_par][act_config::table_size];
    #pragma HLS ARRAY_PARTITION variable=act_table complete dim=1
    if (!initialized) {
        for (unsigned u = 0; u < n_par; u++) {
            #pragma HLS UNROLL
            act::init_table(act_table[u]);
        }
        initialized = true;
    }

    accum_t sum_in[lanes_in];
    accum_t sum_state[lanes_state];
    res_T h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=sum_in complete
    #pragma HLS ARRAY_PARTITION variable=sum_state complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete

    unsigned group = 0, cycle = 0;
SimpleRNNUnits:
    for (unsigned it = 0; it < CONFIG_T::n_state / n_par * CONFIG_T::n_cycles; it++) {
        #pragma HLS PIPELINE II=1
        recurrent_resource_mac<data_T, res_T, CONFIG_T, 1>(it, cycle, data, h_state, param, param_r, sum_in, sum_state);
        if (cycle == CONFIG_T::n_cycles - 1) {
            for (unsigned up = 0; up < n_par; up++) {
                #pragma HLS UNROLL
                unsigned u = group * n_par + up;
                accum_t sum = recurrent_resource_total<CONFIG_T, 1, CONFIG_T::n_lanes_in>(sum_in, up, 0, param_b[u]) +
                              recurrent_resource_total<CONFIG_T, 1, CONFIG_T::n_lanes_state>(sum_state, up, 0, param_br[u]);
                h_newstate[u] = act::single(sum, act_table[up]);
            }
        }
        if (++cycle == CONFIG_T::n_cycles) {
            cycle = 0;
            group++;
        }
    }

    for (unsigned u = 0; u < CONFIG_T::n_state; u++) {
        #pragma HLS UNROLL
        h_state[u] = h_newstate[u];
    }
}

#undef NNET_RECURRENT_TABLE

// One step of the SimpleRNN, as lstm_cell
template <class data_T, class res_T, typename CONFIG_T, bool use_static,
          bool resource = CONFIG_T::strategy == nnet::resource>
struct simple_rnn_cell {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {
        #pragma HLS INLINE
        if (use_static)
            nnet::simple_rnn_static<data_T, res_T, CONFIG_T>(reset_state, data, h_state, param, param_r, param_b, param_br);
        else
            nnet::simple_rnn<data_T, res_T, CONFIG_T>(reset_state, data, h_state, param, param_r, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T, bool use_static>
struct simple_rnn_cell<data_T, res_T, CONFIG_T, use_static, true> {
    static void apply(bool reset_state, data_T data[CONFIG_T::n_in], res_T h_state[CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {
        #pragma HLS INLINE
        nnet::simple_rnn_resource<data_T, res_T, CONFIG_T>(data, h_state, param, param_r, param_b, param_br);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in],
                      res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {

    res_T h_state[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=data_in complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_state[ii] = 0;
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        nnet::simple_rnn_cell<data_T, res_T, CONFIG_T, CONFIG_T::use_static>::apply(reset_state, data_in, h_state, param,
                                                                                    param_r, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_state[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_state[i];
        }
}

template <class data_T, class h_T, class res_T, typename CONFIG_T>
void simple_rnn_stack(data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], h_T h_init[CONFIG_T::n_state],
                      res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_state],
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {

    res_T h_state[CONFIG_T::n_state];
    data_T data_in[CONFIG_T::n_in];
    bool reset_state = false;

    #pragma HLS ARRAY_PARTITION variable=h_init complete
    #pragma HLS ARRAY_PARTITION variable=h_state complete
    #pragma HLS ARRAY_PARTITION variable=data_in complete

    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_state[ii] = h_init[ii];
    }
    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
        }
        nnet::simple_rnn_cell<data_T, res_T, CONFIG_T, false>::apply(reset_state, data_in, h_state, param, param_r, param_b,
                                                                     param_br);
        if (CONFIG_T::n_sequence_out > 1)
            for (int i = CONFIG_T::n_state * iloop, j = 0; i < (CONFIG_T::n_state * (iloop + 1)); i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_state[j];
            }
        reset_state = false;
    }
    if (CONFIG_T::n_sequence_out == 1)
        for (int i = 0; i < (CONFIG_T::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_state[i];
        }
}

template <class data_T, class res_T, typename CONFIG_T>
void simple_rnn_stack(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream,
                      typename CONFIG_T::weight_t param[CONFIG_T::n_weight],
                      typename CONFIG_T::recurrent_weight_t param_r[CONFIG_T::n_recurrent_weight],
                      typename CONFIG_T::bias_t param_b[CONFIG_T::n_state],
                      typename CONFIG_T::recurrent_bias_t param_br[CONFIG_T::n_state]) {

    typename res_T::value_type h_newstate[CONFIG_T::n_state];
    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    for (int ii = 0; ii < CONFIG_T::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
    }

    typename data_T::value_type data_in[CONFIG_T::n_in];
    bool reset_state = true;

DataPropagation:
    for (int i_in = 0; i_in < CONFIG_T::n_sequence * CONFIG_T::n_in / data_T::size; i_in++) {
        data_T data_pack = data_stream.read();
    DataPack:
        for (int i_pack = 0; i_pack < data_T::size; i_pack++) {
            #pragma HLS UNROLL
            data_in[i_pack] = data_pack[i_pack];
        }
        nnet::simple_rnn_cell<typename data_T::value_type, typename res_T::value_type, CONFIG_T, false>::apply(
            reset_state, data_in, h_newstate, param, param_r, param_b, param_br);
        if (CONFIG_T::n_sequence_out > 1) {
            res_T res_pack;
            PRAGMA_DATA_PACK(res_pack)
        ResPack_sequences:
            for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
                #pragma HLS UNROLL
                res_pack[i_pack] = h_newstate[i_pack];
            }
            res_stream.write(res_pack);
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        res_T res_pack;
        PRAGMA_DATA_PACK(res_pack)
    ResPack:
        for (int i_pack = 0; i_pack < res_T::size; i_pack++) {
            #pragma HLS UNROLL
            res_pack[i_pack] = h_newstate[i_pack];
        }
        res_stream.write(res_pack);
    }
}

// Struct for the Bidirectional template

struct single_layer_config {
    // Internal data type definitions
    typedef float weight_t;
    typedef float recurrent_weight_t;
    typedef float bias_t;
    typedef float recurrent_bias_t;
    typedef float accum_t;

    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_state = 2;
    static const unsigned n_mult = 3;
    static const unsigned table_size = 1024;
    static const bool reset_after = true;

    static const unsigned strategy = latency;
    static const unsigned n_unit_par = 1;
    static const unsigned n_cycles = 1;
    static const unsigned n_lanes_in = 1;
    static const unsigned n_lanes_state = 1;
    static const unsigned n_weight = 12;
    static const unsigned n_recurrent_weight = 12;

    template <class x_T, class y_T, class config_T> using activation_recr = nnet::activation::relu<x_T, y_T, config_T>;
    template <class x_T, class y_T, class config_T> using activation = nnet::activation::relu<x_T, y_T, config_T>;
};

struct bidirectional_config {
    // Layer Sizes
    static const unsigned n_in = 2;
    static const unsigned n_parts = 20;
    static const unsigned n_out = 2;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const bool use_static = true;

    // Layers info

    template <class x_T, class y_T, typename config_T, bool backward>
    using RNNfunc_forward = nnet::lstm_class<x_T, y_T, config_T>;
    template <class x_T, class y_T, typename config_T, bool backward>
    using RNNfunc_backward = nnet::lstm_class<x_T, y_T, config_T>;
};

template <class data_T, class res_T, typename CONFIG_T>
void bidirectional_stack(
    data_T data[CONFIG_T::n_sequence * CONFIG_T::n_in], res_T res[CONFIG_T::n_sequence_out * CONFIG_T::n_out],
    typename CONFIG_T::FORWARD_CONFIG::weight_t param[CONFIG_T::FORWARD_CONFIG::n_weight],
    typename CONFIG_T::FORWARD_CONFIG::recurrent_weight_t param_r[CONFIG_T::FORWARD_CONFIG::n_recurrent_weight],
    typename CONFIG_T::FORWARD_CONFIG::bias_t param_b[CONFIG_T::FORWARD_CONFIG::n_state * CONFIG_T::FORWARD_CONFIG::n_mult],
    typename CONFIG_T::FORWARD_CONFIG::recurrent_bias_t
        param_br[CONFIG_T::FORWARD_CONFIG::n_state * CONFIG_T::FORWARD_CONFIG::n_mult],
    typename CONFIG_T::BACKWARD_CONFIG::weight_t param_back[CONFIG_T::BACKWARD_CONFIG::n_weight],
    typename CONFIG_T::BACKWARD_CONFIG::recurrent_weight_t param_r_back[CONFIG_T::BACKWARD_CONFIG::n_recurrent_weight],
    typename CONFIG_T::BACKWARD_CONFIG::bias_t
        param_b_back[CONFIG_T::BACKWARD_CONFIG::n_state * CONFIG_T::BACKWARD_CONFIG::n_mult],
    typename CONFIG_T::BACKWARD_CONFIG::recurrent_bias_t
        param_br_back[CONFIG_T::BACKWARD_CONFIG::n_state * CONFIG_T::BACKWARD_CONFIG::n_mult]) {

    res_T h_newstate[(CONFIG_T::FORWARD_CONFIG::n_mult - 2) * CONFIG_T::FORWARD_CONFIG::n_state];
    res_T h_newstate_back[(CONFIG_T::BACKWARD_CONFIG::n_mult - 2) * CONFIG_T::BACKWARD_CONFIG::n_state];
    data_T data_in[CONFIG_T::n_in];
    data_T data_in_back[CONFIG_T::n_in];
    bool reset_state = true;

    #pragma HLS ARRAY_PARTITION variable=h_newstate complete
    #pragma HLS ARRAY_PARTITION variable=h_newstate_back complete

    for (int ii = 0; ii < (CONFIG_T::FORWARD_CONFIG::n_mult - 2) * CONFIG_T::FORWARD_CONFIG::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate[ii] = 0;
    }
    for (int ii = 0; ii < (CONFIG_T::BACKWARD_CONFIG::n_mult - 2) * CONFIG_T::BACKWARD_CONFIG::n_state; ii++) {
        #pragma HLS UNROLL
        h_newstate_back[ii] = 0;
    }

    for (int iloop = 0; iloop < CONFIG_T::n_sequence; iloop++) {
        for (int j = 0; j < CONFIG_T::n_in; j++) {
            #pragma HLS UNROLL
            data_in[j] = data[j + iloop * CONFIG_T::n_in];
            data_in_back[j] = data[j + (CONFIG_T::n_sequence - iloop - 1) * CONFIG_T::n_in];
        }

        CONFIG_T::template RNNfunc_forward<data_T, res_T, typename CONFIG_T::FORWARD_CONFIG, 0>::apply(
            reset_state, data_in, h_newstate, param, param_r, param_b, param_br);
        CONFIG_T::template RNNfunc_backward<data_T, res_T, typename CONFIG_T::BACKWARD_CONFIG, 1>::apply(
            reset_state, data_in_back, h_newstate_back, param_back, param_r_back, param_b_back, param_br_back);

        if (CONFIG_T::n_sequence_out > 1) {
            for (int i = (CONFIG_T::FORWARD_CONFIG::n_state + CONFIG_T::BACKWARD_CONFIG::n_state) * iloop, j = 0;
                 i < (CONFIG_T::FORWARD_CONFIG::n_state + CONFIG_T::BACKWARD_CONFIG::n_state) * iloop +
                         CONFIG_T::FORWARD_CONFIG::n_state;
                 i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_newstate[j];
            }
            for (int i = (CONFIG_T::FORWARD_CONFIG::n_state + CONFIG_T::BACKWARD_CONFIG::n_state) *
                             (CONFIG_T::n_sequence - iloop) -
                         CONFIG_T::BACKWARD_CONFIG::n_state,
                     j = 0;
                 i <
                 (CONFIG_T::FORWARD_CONFIG::n_state + CONFIG_T::BACKWARD_CONFIG::n_state) * (CONFIG_T::n_sequence - iloop);
                 i++, j++) {
                #pragma HLS UNROLL
                res[i] = h_newstate_back[j];
            }
        }
        reset_state = false;
    }

    if (CONFIG_T::n_sequence_out == 1) {
        for (int i = 0; i < (CONFIG_T::FORWARD_CONFIG::n_state); i++) {
            #pragma HLS UNROLL
            res[i] = h_newstate[i];
        }
        for (int i = 0; i < (CONFIG_T::BACKWARD_CONFIG::n_state); i++) {
            #pragma HLS UNROLL
            res[i + CONFIG_T::FORWARD_CONFIG::n_state] = h_newstate_back[i];
        }
    }
}

} // namespace nnet

#endif
