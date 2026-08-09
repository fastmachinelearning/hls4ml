#ifndef NNET_SPARSEPIXELS_H_
#define NNET_SPARSEPIXELS_H_

#include "ap_fixed.h"
#include "ap_int.h"

constexpr int _sp_floorlog2(int x) { return (x < 2) ? 0 : 1 + _sp_floorlog2(x / 2); }
constexpr int _sp_pow2(int x) { return x == 0 ? 1 : 2 * _sp_pow2(x - 1); }
// ceil(log2(x)): bits needed to encode values 0..x-1
constexpr int _sp_ceillog2(int x) { return (x <= 1) ? 1 : _sp_floorlog2(x - 1) + 1; }

template <typename T, int INDEX_BITS> struct value_idx_pair {
    T value;
    ap_uint<INDEX_BITS> index;
};

template <class T, class t> class Op_active {
  public:
    T operator()(T a, T b, t threshold) {
        if (a.value > threshold)
            return a;
        else if (b.value > threshold)
            return b;
        else {
            T none;
            none.value = 0;
            none.index = 0;
            return none;
        }
    }
};

template <class T, int N, class Op, class t> T find_active(T *x, Op op, t threshold) {
    #pragma HLS INLINE
    static constexpr int leftN = _sp_pow2(_sp_floorlog2(N - 1)) > 0 ? _sp_pow2(_sp_floorlog2(N - 1)) : 0;
    static constexpr int rightN = N - leftN > 0 ? N - leftN : 0;

    if (N == 1) {
        return x[0];
    }
    if (N == 2) {
        return op(x[0], x[1], threshold);
    }
    return op(find_active<T, leftN, Op, t>(x, op, threshold), find_active<T, rightN, Op, t>(x + leftN, op, threshold),
              threshold);
}

// Input-reduce (find-max tree): selects the first N_sparse active pixels (first input channel
// > threshold) in raster order and emits their features (all channels) and 1-based (h, w) hashes.
// A combinational find-active reduction is reused across N_sparse pipelined extractions -- low
// latency, high LUT.
template <class data_T, class res_T, class hash_T, int N_h, int N_w, int N_c, int N_sparse>
void sparse_input_reduce(data_T input_arr[N_h * N_w * N_c], data_T threshold, res_T sparse_arr_feat[N_sparse * N_c],
                         hash_T sparse_arr_hash[N_sparse * 2]) {

    static constexpr int IDX_BITS = _sp_ceillog2(N_h * N_w);
    typedef value_idx_pair<data_T, IDX_BITS> pair_t;

    pair_t pair_arr[N_h * N_w];
    int j_h_arr[N_h * N_w];
    int j_w_arr[N_h * N_w];
    #pragma HLS ARRAY_PARTITION variable = j_h_arr type = complete dim = 0
    #pragma HLS ARRAY_PARTITION variable = j_w_arr type = complete dim = 0
    #pragma HLS ARRAY_PARTITION variable = pair_arr type = complete dim = 0

DataPrepareLoop:
    for (int j = 0; j < N_h * N_w; j++) {
        #pragma HLS UNROLL
        pair_arr[j].value = input_arr[N_c * j];
        pair_arr[j].index = j;

        int remainder = j % (N_h * N_w);
        int j_h = remainder / N_w + 1;
        int j_w = remainder % N_w + 1;

        j_h_arr[j] = j_h;
        j_w_arr[j] = j_w;
    }

    Op_active<pair_t, data_T> op_active;
MaxPixelsLoop:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS PIPELINE
        pair_t pair = find_active<pair_t, N_h * N_w, Op_active<pair_t, data_T>, data_T>(pair_arr, op_active, threshold);
        sparse_arr_feat[N_c * i] = (res_T)pair.value;
        for (int j = 1; j < N_c; j++) {
            #pragma HLS UNROLL
            sparse_arr_feat[N_c * i + j] = (res_T)input_arr[N_c * pair.index + j];
        }

        sparse_arr_hash[2 * i] = j_h_arr[pair.index];
        sparse_arr_hash[2 * i + 1] = j_w_arr[pair.index];

        pair_arr[pair.index].value = 0;
    }
}

// Input-reduce (streaming): same selection as the tree, via a one-pixel-per-cycle raster scan --
// minimal LUT, latency ~N_h*N_w. Unused output slots (fewer than N_sparse active pixels) are zeroed.
template <class data_T, class res_T, class hash_T, int N_h, int N_w, int N_c, int N_sparse>
void sparse_input_reduce_stream(data_T input_arr[N_h * N_w * N_c], data_T threshold, res_T sparse_arr_feat[N_sparse * N_c],
                                hash_T sparse_arr_hash[N_sparse * 2]) {
    constexpr int NP = N_h * N_w;

InitOut:
    for (int s = 0; s < N_sparse; s++) {
        #pragma HLS UNROLL
        for (int c = 0; c < N_c; c++) {
            #pragma HLS UNROLL
            sparse_arr_feat[N_c * s + c] = 0;
        }
        sparse_arr_hash[2 * s] = 0;
        sparse_arr_hash[2 * s + 1] = 0;
    }

    int cnt = 0;
ScanLoop:
    for (int j = 0; j < NP; j++) {
        #pragma HLS PIPELINE
        if (cnt < N_sparse && input_arr[N_c * j] > threshold) {
            sparse_arr_feat[N_c * cnt] = (res_T)input_arr[N_c * j];
            for (int c = 1; c < N_c; c++) {
                #pragma HLS UNROLL
                sparse_arr_feat[N_c * cnt + c] = (res_T)input_arr[N_c * j + c];
            }
            sparse_arr_hash[2 * cnt] = j / N_w + 1;
            sparse_arr_hash[2 * cnt + 1] = j % N_w + 1;
            cnt++;
        }
    }
}

template <class data_T, class accum_T, class w_T, int n_chan, int n_filt, int N_sparse, int ker_size>
accum_T mult_for_sparse_conv_kernel(int offset_h, int offset_w, data_T sparse_arr_feat_in[n_chan * N_sparse],
                                    w_T filt_w[ker_size * ker_size * n_chan * n_filt], int i_filt, int i_pixel_in) {
    #pragma HLS INLINE
    constexpr int R = (ker_size - 1) / 2;
    if ((unsigned)(offset_h + R) >= ker_size || (unsigned)(offset_w + R) >= ker_size) {
        return (accum_T)0;
    }
    // Smallest functional widths for the given ker_size (compile-time):
    //   row, col in [0, ker_size-1]          -> ceil(log2(ker_size)) bits
    //   pos     in [0, ker_size*ker_size-1]  -> ceil(log2(ker_size*ker_size)) bits
    static constexpr int ROW_BITS = _sp_ceillog2(ker_size);
    static constexpr int POS_BITS = _sp_ceillog2(ker_size * ker_size);
    ap_uint<ROW_BITS> row = R - offset_h;
    ap_uint<ROW_BITS> col = R - offset_w;
    ap_uint<POS_BITS> pos = row * ker_size + col;

    accum_T acc = 0;
MultLoopPerFilter:
    for (int i_chan = 0; i_chan < n_chan; i_chan++) {
        #pragma HLS UNROLL
        int w_idx = n_filt * n_chan * pos + n_filt * i_chan + i_filt;
        acc += filt_w[w_idx] * sparse_arr_feat_in[n_chan * i_pixel_in + i_chan];
    }
    return acc;
}

// Sparse convolution on the active pixels. Two independent parallelization knobs trade LUT for
// latency without changing the output:
//   pixel_parallel_factor : output pixels (N_sparse axis) computed per cycle. Default = N_sparse.
//   filt_parallel_factor  : output filters (n_filt axis) computed per cycle. Default = n_filt.
// Both loops use UNROLL factor (no PIPELINE: pipelining the outer loop would force-unroll the filter
// loop and ignore filt_parallel_factor); inter-layer throughput comes from the top-level DATAFLOW.
// accum_T accumulates the MACs; a single cast to res_T is applied at the store.
template <class data_T, class res_T, class hash_T, class w_T, class b_T, class accum_T, int N_sparse, int n_chan, int n_filt,
          int ker_size, int pixel_parallel_factor = N_sparse, int filt_parallel_factor = n_filt>
void sparse_conv(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_filt],
                 hash_T sparse_arr_hash[N_sparse * 2], w_T w[ker_size * ker_size * n_chan * n_filt], b_T b[n_filt]) {

OutputPixelLoop:
    for (int i_pixel_out = 0; i_pixel_out < N_sparse; i_pixel_out++) {
        #pragma HLS UNROLL factor = pixel_parallel_factor

        bool nonzero = false;
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            nonzero |= (sparse_arr_feat_in[i_pixel_out * n_chan + i_chan] != (data_T)0);
        }

    OutputFilterLoop:
        for (int i_filt = 0; i_filt < n_filt; i_filt++) {
            #pragma HLS UNROLL factor = filt_parallel_factor
            accum_T acc = 0;

        InputPixelLoop:
            for (int i_pixel_in = 0; i_pixel_in < N_sparse; i_pixel_in++) {
                #pragma HLS UNROLL
                int offset_h = sparse_arr_hash[2 * i_pixel_out] - sparse_arr_hash[2 * i_pixel_in];
                int offset_w = sparse_arr_hash[2 * i_pixel_out + 1] - sparse_arr_hash[2 * i_pixel_in + 1];

                acc += mult_for_sparse_conv_kernel<data_T, accum_T, w_T, n_chan, n_filt, N_sparse, ker_size>(
                    offset_h, offset_w, sparse_arr_feat_in, w, i_filt, i_pixel_in);
            }

            if (acc != 0) {
                acc += b[i_filt];
            }
            if (nonzero == false) {
                acc = 0;
            }
            sparse_arr_feat_out[n_filt * i_pixel_out + i_filt] = (res_T)acc;
        }
    }
}

template <class data_T, class res_T, int N_sparse, int n_chan>
void sparse_relu(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_chan]) {
    #pragma HLS PIPELINE
    data_T data;
    for (int i = 0; i < N_sparse * n_chan; i++) {
        data = sparse_arr_feat_in[i];
        if (data > 0) {
            sparse_arr_feat_out[i] = data;
        } else {
            sparse_arr_feat_out[i] = 0;
        }
    }
}

// Sparse average pooling. Each pooled cell is emitted once -- by the lowest-indexed output pixel
// mapping to it (the is_first test); duplicate pixels of the same cell emit 0. The averaging reads
// only the input array (no scratch mutation), so it is safe to partially unroll. Two independent
// knobs: pixel_parallel_factor (N_sparse axis) and chan_parallel_factor (n_chan axis).
// Pool height and width are independent (asymmetric pooling). A pooled coordinate falling outside
// the valid output grid (partial window of an odd input dimension under 'valid' pooling) has its
// features zeroed, matching the dense layer dropping that window; zero-feature pixels are inert in
// every downstream sparse kernel. The averaging divides by the full pool area as one reciprocal
// multiply per axis (skipped for a unit axis, whose reciprocal 1.0 the fixed-point type cannot
// hold); for square pools this reproduces the previous two-multiply arithmetic exactly.
template <class data_T, class res_T, class hash_T, class accum_T, int N_sparse, int n_chan, int in_height, int in_width,
          int pool_height, int pool_width, int pixel_parallel_factor = N_sparse, int chan_parallel_factor = n_chan>
void sparse_pooling_avg(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_chan],
                        hash_T sparse_arr_hash_in[N_sparse * 2], hash_T sparse_arr_hash_out[N_sparse * 2]) {

    constexpr int out_height = in_height / pool_height;
    constexpr int out_width = in_width / pool_width;
    // Unsigned reciprocals: 1/2 = 0.5 needs the unsigned [0, 1) range (signed ap_fixed<10,0> tops
    // out just below 0.5 and would wrap). Truncation at 10 fractional bits matches the previous
    // signed type for every value below 0.5, so square-pool results are unchanged.
    const ap_ufixed<10, 0> pool_h_recip = 1.0 / double(pool_height); // only used when pool_height > 1
    const ap_ufixed<10, 0> pool_w_recip = 1.0 / double(pool_width);  // only used when pool_width > 1

    int hash_tmp[N_sparse * 2];
    #pragma HLS ARRAY_PARTITION variable = hash_tmp type = complete dim = 0
ComputePooledLoc:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        hash_tmp[2 * i] = (sparse_arr_hash_in[2 * i] - 1) / pool_height + 1;
        hash_tmp[2 * i + 1] = (sparse_arr_hash_in[2 * i + 1] - 1) / pool_width + 1;
    }

HashOutLoop:
    for (int i_pixel = 0; i_pixel < N_sparse; i_pixel++) {
        #pragma HLS UNROLL factor = pixel_parallel_factor
        int h_out = hash_tmp[2 * i_pixel];
        int w_out = hash_tmp[2 * i_pixel + 1];
        bool valid = (h_out <= out_height) && (w_out <= out_width);

        bool is_first = true;
    FirstCheck:
        for (int k = 0; k < N_sparse; k++) {
            #pragma HLS UNROLL
            if (k < i_pixel && hash_tmp[2 * k] == h_out && hash_tmp[2 * k + 1] == w_out) {
                is_first = false;
            }
        }

    ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL factor = chan_parallel_factor
            accum_T acc = 0;

        HashInLoop:
            for (int j_pixel = 0; j_pixel < N_sparse; j_pixel++) {
                #pragma HLS UNROLL
                int h_in = hash_tmp[2 * j_pixel];
                int w_in = hash_tmp[2 * j_pixel + 1];

                if ((h_out == h_in) && (w_out == w_in)) {
                    acc += sparse_arr_feat_in[n_chan * j_pixel + i_chan];
                }
            }
            res_T avg;
            if (pool_height > 1 && pool_width > 1) {
                avg = (res_T)(acc * pool_h_recip * pool_w_recip);
            } else if (pool_height > 1) {
                avg = (res_T)(acc * pool_h_recip);
            } else if (pool_width > 1) {
                avg = (res_T)(acc * pool_w_recip);
            } else {
                avg = (res_T)acc;
            }
            sparse_arr_feat_out[n_chan * i_pixel + i_chan] = (is_first && valid) ? avg : (res_T)0;
        }
        sparse_arr_hash_out[2 * i_pixel] = h_out;
        sparse_arr_hash_out[2 * i_pixel + 1] = w_out;
    }
}

// Sparse max pooling. Same structure as the average version (one emission per pooled cell via the
// is_first test), but takes the per-channel maximum of the active pixels in the cell, floored at 0
// to match dense max pooling over the zero-masked window. Two independent knobs:
// pixel_parallel_factor (N_sparse axis) and chan_parallel_factor (n_chan axis).
// Pool height and width are independent (asymmetric pooling); out-of-range pooled coordinates are
// zeroed as in the average version.
template <class data_T, class res_T, class hash_T, int N_sparse, int n_chan, int in_height, int in_width, int pool_height,
          int pool_width, int pixel_parallel_factor = N_sparse, int chan_parallel_factor = n_chan>
void sparse_pooling_max(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_chan],
                        hash_T sparse_arr_hash_in[N_sparse * 2], hash_T sparse_arr_hash_out[N_sparse * 2]) {

    constexpr int out_height = in_height / pool_height;
    constexpr int out_width = in_width / pool_width;

    int hash_tmp[N_sparse * 2];
    #pragma HLS ARRAY_PARTITION variable = hash_tmp type = complete dim = 0
ComputePooledLoc:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        hash_tmp[2 * i] = (sparse_arr_hash_in[2 * i] - 1) / pool_height + 1;
        hash_tmp[2 * i + 1] = (sparse_arr_hash_in[2 * i + 1] - 1) / pool_width + 1;
    }

HashOutLoop:
    for (int i_pixel = 0; i_pixel < N_sparse; i_pixel++) {
        #pragma HLS UNROLL factor = pixel_parallel_factor
        int h_out = hash_tmp[2 * i_pixel];
        int w_out = hash_tmp[2 * i_pixel + 1];
        bool valid = (h_out <= out_height) && (w_out <= out_width);

        bool is_first = true;
    FirstCheck:
        for (int k = 0; k < N_sparse; k++) {
            #pragma HLS UNROLL
            if (k < i_pixel && hash_tmp[2 * k] == h_out && hash_tmp[2 * k + 1] == w_out) {
                is_first = false;
            }
        }

    ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL factor = chan_parallel_factor
            data_T vmax = 0;

        HashInLoop:
            for (int j_pixel = 0; j_pixel < N_sparse; j_pixel++) {
                #pragma HLS UNROLL
                int h_in = hash_tmp[2 * j_pixel];
                int w_in = hash_tmp[2 * j_pixel + 1];

                data_T v = sparse_arr_feat_in[n_chan * j_pixel + i_chan];
                if ((h_out == h_in) && (w_out == w_in) && (v > vmax)) {
                    vmax = v;
                }
            }
            sparse_arr_feat_out[n_chan * i_pixel + i_chan] = (is_first && valid) ? (res_T)vmax : (res_T)0;
        }
        sparse_arr_hash_out[2 * i_pixel] = h_out;
        sparse_arr_hash_out[2 * i_pixel + 1] = w_out;
    }
}

// Scatters the sparse pixels back to a dense n_height * n_width * n_chan grid (the sparse->dense
// transition before Dense layers). Implemented as a gather: each dense location is written exactly
// once by scanning the sparse pixels for the one mapping to it (no data-dependent writes), so it is
// safe to fully or partially unroll. parallel_factor = dense locations produced per cycle.
template <class data_T, class res_T, class hash_T, int n_height, int n_width, int n_chan, int N_sparse,
          int parallel_factor = n_height *n_width>
void sparse_flatten(data_T sparse_arr_feat[N_sparse * n_chan], hash_T sparse_arr_hash[N_sparse * 2],
                    res_T flat_arr[n_height * n_width * n_chan]) {

    int pix_idx[N_sparse];
    #pragma HLS ARRAY_PARTITION variable = pix_idx type = complete dim = 0
PixIdxLoop:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        pix_idx[i] = (sparse_arr_hash[2 * i] - 1) * n_width + (sparse_arr_hash[2 * i + 1] - 1);
    }

GatherLoop:
    for (int p = 0; p < n_height * n_width; p++) {
        #pragma HLS UNROLL factor = parallel_factor

    ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            res_T val = 0;

        ScanLoop:
            for (int i = 0; i < N_sparse; i++) {
                #pragma HLS UNROLL
                data_T data = sparse_arr_feat[n_chan * i + i_chan];
                if (pix_idx[i] == p && data != 0) {
                    val = (res_T)data;
                }
            }
            flat_arr[n_chan * p + i_chan] = val;
        }
    }
}

#endif // NNET_SPARSEPIXELS_H_
