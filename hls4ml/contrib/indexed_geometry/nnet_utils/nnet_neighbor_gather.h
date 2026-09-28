#ifndef NNET_NEIGHBOR_GATHER_H_
#define NNET_NEIGHBOR_GATHER_H_

#include "nnet_common.h"

namespace nnet {

// Configuration struct for neighbor_gather_2d.
//
// NOTE: indices[] is NOT declared in the base struct.
// Each instantiation (config{index}) declares its own
//   static const int indices[n_pixels * n_neighbors]
// with an out-of-class initializer. This avoids C++ member-hiding
// ambiguity in Vitis HLS when multiple gather layers are present.

struct neighbor_gather_config {
    static const unsigned n_pixels = 1;
    static const unsigned n_neighbors = 1;
    static const unsigned n_features = 1;
    // If true, the io_stream overload emits one wide packet of
    // n_neighbors*n_features per pixel (all neighbors read with
    // UNROLL, no extra cycles) instead of n_neighbors separate
    // packets of n_features. Faster (n_pixels cycles instead of
    // n_pixels*n_neighbors) but the output is only directly
    // consumable by this library's own "packed" conv/pool overloads,
    // not by generic hls4ml layers. Ignored by io_parallel. Default
    // false preserves compatibility with any standard hls4ml layer
    // placed directly after NeighborGatherLayer.
    static const bool pack_neighbors = false;
};

// neighbor_gather_2d
//
// Gathers neighbor features for each pixel according to a fixed index map.
//
// hls4ml passes tensors as flat 1D arrays in row-major order:
//   input  : [n_pixels * n_features]
//   output : [n_pixels * n_neighbors * n_features]
//
// input_T and output_T are separate template parameters to allow
// hls4ml to assign independent fixed-point precisions to the input
// and output of this layer.
//
// Neighbor indices are embedded in CONFIG_T::indices as
//   int[n_pixels * n_neighbors], layout: indices[p * K + k].
// Slots with index == -1 correspond to camera border pixels and are
// zero-masked in the output.
//
// The outer loop over pixels is pipelined (II=1) rather than fully
// unrolled. For large cameras (e.g. n_pixels=163), a fully combinational
// implementation (UNROLL on all three nested loops) creates on the
// order of n_pixels * n_neighbors * n_features flattened operations,
// which can exhaust memory during Vitis HLS scheduling/binding. This
// version trades a small amount of latency (n_pixels cycles) for a
// design Vitis HLS can actually synthesize.

template <class input_T, class output_T, typename CONFIG_T>
void neighbor_gather_2d(input_T input[CONFIG_T::n_pixels * CONFIG_T::n_features],
                        output_T output[CONFIG_T::n_pixels * CONFIG_T::n_neighbors * CONFIG_T::n_features]) {
    #pragma HLS INLINE

Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {
    #pragma HLS PIPELINE II=1
    Loop_Neighbors:
        for (unsigned k = 0; k < CONFIG_T::n_neighbors; k++) {
            #pragma HLS UNROLL
            const int idx = CONFIG_T::indices[p * CONFIG_T::n_neighbors + k];
        Loop_Features:
            for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
                #pragma HLS UNROLL
                const unsigned out_i = p * (CONFIG_T::n_neighbors * CONFIG_T::n_features) + k * CONFIG_T::n_features + f;
                output[out_i] = (idx != -1)
                                    ? static_cast<output_T>(input[static_cast<unsigned>(idx) * CONFIG_T::n_features + f])
                                    : static_cast<output_T>(0);
            }
        }
    }
}

// ============================================================
// io_stream variant
// ============================================================
//
// neighbor_gather_2d (io_stream overload)
//
// io_stream counterpart of neighbor_gather_2d, overloaded on parameter
// type (hls::stream<T>& vs. flat array) following hls4ml's own
// convention (see e.g. nnet::conv_1d_cl in nnet_conv1d.h /
// nnet_conv1d_stream.h). Consumes one
// nnet::array<data_T, n_features>-wide packet per pixel (input order
// p = 0..n_pixels-1) and produces one
// nnet::array<res_T, n_neighbors * n_features> packet per pixel,
// containing the K gathered neighbor feature-vectors flattened as
// [k * n_features + f] — the same flat layout already used by the
// io_parallel kernel and by nnet_indexed_conv.h / nnet_indexed_pool.h.
//
// Two-stage dataflow, mirroring the manual streaming pattern validated:
// (1) buffer all pixels, replicated n_neighbors times to
// allow parallel random-access reads; (2) for each output pixel, fetch
// its n_neighbors neighbors via CONFIG_T::indices uniformly (no
// special-casing of any slot) and emit the gathered packet. Border
// slots (index == -1) are zero-masked, matching io_parallel semantics
// exactly.

template <class data_T, typename CONFIG_T>
void store_gather_input_stream(
    hls::stream<data_T> &data_stream,
    typename data_T::value_type local_buffer[CONFIG_T::n_neighbors][CONFIG_T::n_pixels][CONFIG_T::n_features]) {
    #pragma HLS INLINE off

Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {
        #pragma HLS PIPELINE II=1
        data_T in_pack = data_stream.read();
    Loop_Replicate:
        for (unsigned k = 0; k < CONFIG_T::n_neighbors; k++) {
        #pragma HLS UNROLL
        Loop_Features:
            for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
                #pragma HLS UNROLL
                local_buffer[k][p][f] = in_pack[f];
            }
        }
    }
}

template <class res_T, typename CONFIG_T>
void fetch_gather_output_stream(
    typename res_T::value_type local_buffer[CONFIG_T::n_neighbors][CONFIG_T::n_pixels][CONFIG_T::n_features],
    hls::stream<res_T> &res_stream) {
#pragma HLS INLINE off

// hls4ml packs the neighbor dimension as successive stream writes
// (one res_T packet of width n_features per neighbor), not as a
// single wide packet of n_neighbors*n_features -- confirmed against
// the generated layer2_t = nnet::array<T, n_features> in defines.h.
Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {
    Loop_Neighbors:
        for (unsigned k = 0; k < CONFIG_T::n_neighbors; k++) {
            #pragma HLS PIPELINE II=1
            const int idx = CONFIG_T::indices[p * CONFIG_T::n_neighbors + k];
            res_T out_pack;
        Loop_Features:
            for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
                #pragma HLS UNROLL
                out_pack[f] = (idx != -1) ? static_cast<typename res_T::value_type>(local_buffer[k][idx][f])
                                          : static_cast<typename res_T::value_type>(0);
            }
            res_stream.write(out_pack);
        }
    }
}

template <class res_T, typename CONFIG_T>
void fetch_gather_output_stream_packed(
    typename res_T::value_type local_buffer[CONFIG_T::n_neighbors][CONFIG_T::n_pixels][CONFIG_T::n_features],
    hls::stream<res_T> &res_stream) {
#pragma HLS INLINE off

// Packed variant: one res_T packet of width n_neighbors*n_features
// per pixel, all K neighbors gathered in the same cycle via
// UNROLL (mirrors the manual fetch_gather_output pattern).
Loop_Pixels:
    for (unsigned p = 0; p < CONFIG_T::n_pixels; p++) {
        #pragma HLS PIPELINE II=1
        res_T out_pack;
    Loop_Neighbors:
        for (unsigned k = 0; k < CONFIG_T::n_neighbors; k++) {
            #pragma HLS UNROLL
            const int idx = CONFIG_T::indices[p * CONFIG_T::n_neighbors + k];
        Loop_Features:
            for (unsigned f = 0; f < CONFIG_T::n_features; f++) {
                #pragma HLS UNROLL
                const unsigned out_i = k * CONFIG_T::n_features + f;
                out_pack[out_i] = (idx != -1) ? static_cast<typename res_T::value_type>(local_buffer[k][idx][f])
                                              : static_cast<typename res_T::value_type>(0);
            }
        }
        res_stream.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void neighbor_gather_2d(hls::stream<data_T> &data_stream, hls::stream<res_T> &res_stream) {
    #pragma HLS INLINE off
    #pragma HLS DATAFLOW

    typename data_T::value_type local_buffer[CONFIG_T::n_neighbors][CONFIG_T::n_pixels][CONFIG_T::n_features];
    #pragma HLS ARRAY_PARTITION variable=local_buffer complete dim=1
    #pragma HLS ARRAY_PARTITION variable=local_buffer complete dim=3
    #pragma HLS STREAM variable=local_buffer type=pipo

    store_gather_input_stream<data_T, CONFIG_T>(data_stream, local_buffer);
    if (CONFIG_T::pack_neighbors) {
        fetch_gather_output_stream_packed<res_T, CONFIG_T>(local_buffer, res_stream);
    } else {
        fetch_gather_output_stream<res_T, CONFIG_T>(local_buffer, res_stream);
    }
}

} // namespace nnet

#endif // NNET_NEIGHBOR_GATHER_H_
