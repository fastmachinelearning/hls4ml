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

template <class data_T, class res_T, class hash_T, int N_h, int N_w, int N_c, int N_sparse>
void sparse_input_reduce(data_T input_arr[N_h * N_w * N_c], data_T threshold, res_T sparse_arr_feat[N_sparse * N_c],
                         hash_T sparse_arr_hash[N_sparse * 2]) {

    // Flat pixel index ranges over 0..N_h*N_w-1 -> auto-sized to minimum bits
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

template <class data_T, class accum_T, class w_T, int n_chan, int n_filt, int N_sparse, int ker_size>
accum_T mult_for_sparse_conv_kernel(int offset_h, int offset_w, data_T sparse_arr_feat_in[n_chan * N_sparse],
                                    w_T filt_w[ker_size * ker_size * n_chan * n_filt], int i_filt, int i_pixel_in) {
    #pragma HLS INLINE
    constexpr int R = (ker_size - 1) / 2;
    if ((unsigned)(offset_h + R) >= ker_size || (unsigned)(offset_w + R) >= ker_size) {
        return (accum_T)0;
    }
    ap_uint<4> row = R - offset_h;
    ap_uint<4> col = R - offset_w;
    ap_uint<7> pos = row * ker_size + col;

    accum_T acc = 0;
MultLoopPerFilter:
    for (int i_chan = 0; i_chan < n_chan; i_chan++) {
        #pragma HLS UNROLL
        int w_idx = n_filt * n_chan * pos + n_filt * i_chan + i_filt;
        acc += filt_w[w_idx] * sparse_arr_feat_in[n_chan * i_pixel_in + i_chan];
    }
    return acc;
}

template <class data_T, class res_T, class hash_T, class w_T, class b_T, class accum_T, int N_sparse, int n_chan, int n_filt,
          int ker_size>
void sparse_conv(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_filt],
                 hash_T sparse_arr_hash[N_sparse * 2], w_T w[ker_size * ker_size * n_chan * n_filt], b_T b[n_filt]) {

OutputPixelLoop:
    for (int i_pixel_out = 0; i_pixel_out < N_sparse; i_pixel_out++) {
        #pragma HLS UNROLL

        bool nonzero = false;
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            nonzero |= (sparse_arr_feat_in[i_pixel_out * n_chan + i_chan] != (data_T)0);
        }

    OutputFilterLoop:
        for (int i_filt = 0; i_filt < n_filt; i_filt++) {
            #pragma HLS UNROLL
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

template <class data_T, class res_T, class hash_T, class accum_T, int N_sparse, int n_chan, int pool_size>
void sparse_pooling_avg(data_T sparse_arr_feat_in[N_sparse * n_chan], res_T sparse_arr_feat_out[N_sparse * n_chan],
                        hash_T sparse_arr_hash_in[N_sparse * 2], hash_T sparse_arr_hash_out[N_sparse * 2]) {

    constexpr double _pool_size_recip_d = 1.0 / double(pool_size);
    const ap_fixed<10, 0> pool_size_recip = _pool_size_recip_d;

    int hash_tmp[N_sparse * 2];
#pragma HLS ARRAY_PARTITION variable = hash_tmp type = complete dim = 0
ComputePooledLoc:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL
        hash_tmp[2 * i] = (sparse_arr_hash_in[2 * i] - 1) / pool_size + 1;
        hash_tmp[2 * i + 1] = (sparse_arr_hash_in[2 * i + 1] - 1) / pool_size + 1;
    }

    data_T sparse_arr_feat_in_copy[N_sparse * n_chan];
    #pragma HLS ARRAY_PARTITION variable = sparse_arr_feat_in_copy type = complete dim = 0
    for (int i = 0; i < N_sparse * n_chan; i++) {
        #pragma HLS UNROLL
        sparse_arr_feat_in_copy[i] = sparse_arr_feat_in[i];
    }

HashOutLoop:
    for (int i_pixel = 0; i_pixel < N_sparse; i_pixel++) {
        #pragma HLS UNROLL
        int h_out = hash_tmp[2 * i_pixel];
        int w_out = hash_tmp[2 * i_pixel + 1];

    ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            accum_T acc = 0;

        HashInLoop:
            for (int j_pixel = 0; j_pixel < N_sparse; j_pixel++) {
                #pragma HLS UNROLL
                int h_in = hash_tmp[2 * j_pixel];
                int w_in = hash_tmp[2 * j_pixel + 1];

                data_T data = sparse_arr_feat_in_copy[n_chan * j_pixel + i_chan];
                if ((h_out == h_in) && (w_out == w_in)) {
                    acc += data;
                    sparse_arr_feat_in_copy[n_chan * j_pixel + i_chan] = 0;
                }
            }
            sparse_arr_feat_out[n_chan * i_pixel + i_chan] = (res_T)(acc * pool_size_recip * pool_size_recip);
        }
        sparse_arr_hash_out[2 * i_pixel] = h_out;
        sparse_arr_hash_out[2 * i_pixel + 1] = w_out;
    }
}

template <class data_T, class res_T, class hash_T, int n_height, int n_width, int n_chan, int N_sparse>
void sparse_flatten(data_T sparse_arr_feat[N_sparse * n_chan], hash_T sparse_arr_hash[N_sparse * 2],
                    res_T flat_arr[n_height * n_width * n_chan]) {

InitFlatArr:
    for (int i = 0; i < n_height * n_width * n_chan; i++) {
        #pragma HLS UNROLL
        flat_arr[i] = 0;
    }

FillFlatArr:
    for (int i = 0; i < N_sparse; i++) {
        #pragma HLS UNROLL factor = 4
        int i_h = sparse_arr_hash[2 * i];
        int i_w = sparse_arr_hash[2 * i + 1];
        int pixel_idx = (i_h - 1) * n_width + (i_w - 1);

    ChannelLoop:
        for (int i_chan = 0; i_chan < n_chan; i_chan++) {
            #pragma HLS UNROLL
            data_T data = sparse_arr_feat[n_chan * i + i_chan];

            if (data != 0) {
                flat_arr[n_chan * pixel_idx + i_chan] = (res_T)data;
            }
        }
    }
}

#endif // NNET_SPARSEPIXELS_H_
