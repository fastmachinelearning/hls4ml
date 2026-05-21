#ifndef NNET_HEPT_H_
#define NNET_HEPT_H_

#include "ap_int.h"
#include "hls_stream.h"

#include "nnet_helpers.h"
#include "nnet_layernorm_stream.h"
#include "nnet_qkv_stream.h"
#include "nnet_e2lsh_stream.h"
#include "nnet_stream.h"

namespace nnet {

struct hept_config {
    static const unsigned seq_len = 1;
    static const unsigned embed_dim = 1;
    static const unsigned num_heads = 1;
    static const unsigned dim_per_head = 1;
    static const unsigned coords_dim = 1;
    static const unsigned num_hashes = 1;
    static const unsigned num_w_per_dist = 1;
    static const unsigned block_size = 1;
    static const unsigned inter_reuse_factor = 1;
    static const unsigned qkv_reuse_factor = 1;
    static const unsigned par_factor = 1;
    static const unsigned par_factor_sort = 1;

    static const unsigned norm_inv_sqrt_table_size = 1024;
    static const unsigned norm_inv_sqrt_range = 16;
    static const unsigned qkv_exp_table_size = 1024;
    static const unsigned qkv_exp_range = 8;
    static const unsigned qkv_sqrt_table_size = 1024;
    static const unsigned qkv_sqrt_range = 8;
    static const unsigned e2lsh_exp_table_size = 1024;
    static const unsigned e2lsh_exp_range = 8;

    typedef float input_t;
    typedef float coords_t;
    typedef float shift_t;
    typedef float output_t;
    typedef float weight_t;
    typedef float accum_t;

    typedef float norm_accum_t;
    typedef float norm_sum_t;
    typedef float norm_sum_sqr_t;
    typedef float norm_mean_t;
    typedef float norm_var_table_t;

    typedef float qkv_exp_table_t;
    typedef float qkv_sqrt_table_t;
    typedef float e2lsh_exp_table_t;
    typedef float hash_mul_accum_t;

    typedef float norm_out_t;
    typedef float prepare_qk_t;
    typedef float combined_shift_out_t;
    typedef ap_uint<1> sorted_index_t;
    typedef float batched_index_select_out_t;
    typedef float qkv_res_out_denom_t;
    typedef float qkv_res_out_so_t;
    typedef float batched_index_select_unsort_out_o_t;
    typedef float batched_index_select_unsort_out_logits_t;
    typedef float o_divide_logits_out_t;
};

template <typename CONFIG_T>
struct hept_layernorm_config : nnet::layernorm_config {
    static const unsigned seq_len = CONFIG_T::seq_len;
    static const unsigned embed_dim = CONFIG_T::embed_dim;
    static const unsigned inv_sqrt_table_size = CONFIG_T::norm_inv_sqrt_table_size;
    static const unsigned inv_sqrt_range = CONFIG_T::norm_inv_sqrt_range;
    typedef typename CONFIG_T::norm_sum_sqr_t sum_sqr_t;
    typedef typename CONFIG_T::norm_mean_t mean_t;
    typedef typename CONFIG_T::norm_sum_t sum_t;
    typedef typename CONFIG_T::weight_t bias_t;
    typedef typename CONFIG_T::weight_t scale_t;
    typedef typename CONFIG_T::norm_accum_t accum_t;
    static const unsigned par_factor = CONFIG_T::par_factor;
};

template <typename CONFIG_T>
struct hept_qkv_config : nnet::qkv_config {
    static const unsigned num_head = CONFIG_T::num_heads;
    static const unsigned dim_per_head = CONFIG_T::dim_per_head;
    static const unsigned seq_len = CONFIG_T::seq_len;
    static const unsigned num_w_per_dist = CONFIG_T::num_w_per_dist;
    typedef typename CONFIG_T::weight_t weight_t;
    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::qkv_exp_table_t exp_table_t;
    typedef typename CONFIG_T::qkv_sqrt_table_t sqrt_table_t;
    static const unsigned exp_table_size = CONFIG_T::qkv_exp_table_size;
    static const unsigned exp_range = CONFIG_T::qkv_exp_range;
    static const unsigned sqrt_table_size = CONFIG_T::qkv_sqrt_table_size;
    static const unsigned sqrt_range = CONFIG_T::qkv_sqrt_range;
    static const unsigned coords_dim = CONFIG_T::coords_dim;
    static const unsigned par_factor = CONFIG_T::par_factor;
};

template <typename CONFIG_T>
struct hept_e2lsh_config : nnet::e2lsh_config {
    static const unsigned num_head = CONFIG_T::num_heads;
    static const unsigned dim_per_head = CONFIG_T::dim_per_head;
    static const unsigned num_hashes = CONFIG_T::num_hashes;
    static const unsigned seq_len = CONFIG_T::seq_len;
    static const unsigned num_w_per_dist = CONFIG_T::num_w_per_dist;
    typedef typename CONFIG_T::weight_t weight_t;
    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::e2lsh_exp_table_t exp_table_t;
    static const unsigned exp_table_size = CONFIG_T::e2lsh_exp_table_size;
    static const unsigned exp_range = CONFIG_T::e2lsh_exp_range;
    typedef typename CONFIG_T::hash_mul_accum_t hash_mul_accum_t;
    static const unsigned coords_dim = CONFIG_T::coords_dim;
    static const unsigned block_size = CONFIG_T::block_size;
    static const unsigned inter_reuse_factor = CONFIG_T::inter_reuse_factor;
    static const unsigned qkv_reuse_factor = CONFIG_T::qkv_reuse_factor;
    static const unsigned par_factor = CONFIG_T::par_factor;
    static const unsigned par_factor_sort = CONFIG_T::par_factor_sort;

    static const unsigned num_tiles = num_head;
    static const unsigned qk_tile_size = dim_per_head + coords_dim;
    static const unsigned v_tile_size = dim_per_head;
    static const unsigned elem_bits = CONFIG_T::prepare_qk_t::width;
    typedef ap_uint<elem_bits> elem_bits_t;
    typedef ap_uint<qk_tile_size * elem_bits> qk_tile_wide_t;
    typedef ap_uint<v_tile_size * elem_bits> v_tile_wide_t;

    static const unsigned unsort_o_tile_size = dim_per_head;
    static const unsigned unsort_o_elem_bits = CONFIG_T::batched_index_select_unsort_out_o_t::width;
    typedef ap_uint<unsort_o_elem_bits> unsort_o_elem_bits_t;
    typedef ap_uint<unsort_o_tile_size * unsort_o_elem_bits> unsort_o_tile_wide_t;

    static const unsigned unsort_logits_tile_size = num_hashes * num_head;
    static const unsigned unsort_logits_elem_bits = CONFIG_T::batched_index_select_unsort_out_logits_t::width;
    typedef ap_uint<unsort_logits_elem_bits> unsort_logits_elem_bits_t;
    typedef ap_uint<unsort_logits_tile_size * unsort_logits_elem_bits> unsort_logits_tile_wide_t;
};

template <class packed_T, class scalar_T, unsigned CHANNELS, unsigned PAR_FACTOR, unsigned SEQ_LEN>
void packed_stream_to_scalar_lanes(
    hls::stream<packed_T> &src,
    hls::stream<scalar_T> dst[PAR_FACTOR * CHANNELS]
) {
    #pragma HLS INLINE OFF
    for (unsigned i = 0; i < SEQ_LEN / PAR_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1 rewind
        for (unsigned pf = 0; pf < PAR_FACTOR; ++pf) {
            #pragma HLS UNROLL
            packed_T token = src.read();
            for (unsigned c = 0; c < CHANNELS; ++c) {
                #pragma HLS UNROLL
                scalar_T scalar = static_cast<scalar_T>(token[c]);
                dst[pf * CHANNELS + c].write(scalar);
            }
        }
    }
}

template <class scalar_T, class packed_T, unsigned CHANNELS, unsigned PAR_FACTOR, unsigned SEQ_LEN>
void scalar_lanes_to_packed_stream(
    hls::stream<scalar_T> src[PAR_FACTOR * CHANNELS],
    hls::stream<packed_T> &dst
) {
    #pragma HLS INLINE OFF
    for (unsigned i = 0; i < SEQ_LEN / PAR_FACTOR; ++i) {
        #pragma HLS PIPELINE II=1 rewind
        for (unsigned pf = 0; pf < PAR_FACTOR; ++pf) {
            #pragma HLS UNROLL
            packed_T token;
            for (unsigned c = 0; c < CHANNELS; ++c) {
                #pragma HLS UNROLL
                token[c] = src[pf * CHANNELS + c].read();
            }
            dst.write(token);
        }
    }
}

template <class input_T, class coords_T, class shift_T, class output_T, typename CONFIG_T>
void hept_attention(
    hls::stream<input_T> &x,
    hls::stream<coords_T> &coords,
    hls::stream<shift_T> &combined_shifts,
    hls::stream<output_T> &out,
    typename CONFIG_T::weight_t layer_norm_weight[CONFIG_T::embed_dim],
    typename CONFIG_T::weight_t layer_norm_bias[CONFIG_T::embed_dim],
    typename CONFIG_T::weight_t q_weight[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_heads],
    typename CONFIG_T::weight_t k_weight[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_heads],
    typename CONFIG_T::weight_t v_weight[CONFIG_T::dim_per_head * CONFIG_T::dim_per_head * CONFIG_T::num_heads],
    typename CONFIG_T::weight_t sqrt_w[CONFIG_T::num_heads * CONFIG_T::coords_dim],
    typename CONFIG_T::weight_t alpha[(CONFIG_T::num_w_per_dist + 2 * CONFIG_T::coords_dim) * CONFIG_T::num_hashes * CONFIG_T::num_heads],
    typename CONFIG_T::weight_t out_linear_weight[CONFIG_T::num_heads * CONFIG_T::dim_per_head * CONFIG_T::dim_per_head],
    typename CONFIG_T::weight_t out_linear_bias[CONFIG_T::dim_per_head]
) {
    #pragma HLS DATAFLOW

    static_assert(CONFIG_T::embed_dim == CONFIG_T::dim_per_head,
                  "The imported HEPT helper kernels currently require embed_dim == dim_per_head.");
    static_assert(CONFIG_T::seq_len % CONFIG_T::par_factor == 0,
                  "seq_len must be divisible by par_factor.");
    static_assert(CONFIG_T::seq_len % CONFIG_T::par_factor_sort == 0,
                  "seq_len must be divisible by par_factor_sort.");
    static_assert(CONFIG_T::par_factor_sort % CONFIG_T::block_size == 0,
                  "par_factor_sort must be divisible by block_size.");
    //static_assert(input_T::size == CONFIG_T::embed_dim,
    //              "Packed input token width must match embed_dim.");
    //static_assert(coords_T::size == CONFIG_T::coords_dim,
    //              "Packed coords token width must match coords_dim.");
    //static_assert(shift_T::size == CONFIG_T::num_heads * CONFIG_T::num_hashes,
    //              "Packed combined_shifts token width must match num_heads * num_hashes.");
    //static_assert(output_T::size == CONFIG_T::embed_dim,
    //              "Packed output token width must match embed_dim.");

    typedef hept_layernorm_config<CONFIG_T> layernorm_cfg;
    typedef hept_qkv_config<CONFIG_T> qkv_cfg;
    typedef hept_e2lsh_config<CONFIG_T> e2lsh_cfg;

    enum {
        FEATURE_LANES = CONFIG_T::par_factor * CONFIG_T::embed_dim,
        COORD_LANES = CONFIG_T::par_factor * CONFIG_T::coords_dim,
        SHIFT_LANES = CONFIG_T::par_factor_sort * CONFIG_T::num_heads * CONFIG_T::num_hashes,
        QK_LANES = CONFIG_T::par_factor * CONFIG_T::num_heads * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim),
        V_LANES = CONFIG_T::par_factor * CONFIG_T::num_heads * CONFIG_T::dim_per_head,
        HEAD_HASH = CONFIG_T::num_heads * CONFIG_T::num_hashes,
        BATCHED_QK_LANES = CONFIG_T::par_factor_sort * (CONFIG_T::dim_per_head + CONFIG_T::coords_dim),
        BATCHED_V_LANES = CONFIG_T::par_factor_sort * CONFIG_T::dim_per_head,
        O_LANES = CONFIG_T::num_heads * CONFIG_T::dim_per_head * CONFIG_T::par_factor,
        LOGIT_LANES = CONFIG_T::num_heads * CONFIG_T::par_factor
    };

    hls::stream<typename CONFIG_T::input_t> x_scalar[FEATURE_LANES];
    hls::stream<typename CONFIG_T::coords_t> coords_scalar[COORD_LANES];
    hls::stream<typename CONFIG_T::shift_t> shifts_scalar[SHIFT_LANES];
    hls::stream<typename CONFIG_T::norm_out_t> layernorm_out[FEATURE_LANES];

    hls::stream<typename CONFIG_T::prepare_qk_t> q_hat[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> q_hat_1[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> q_hat_2[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> k_hat[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> k_hat_1[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> k_hat_2[QK_LANES];
    hls::stream<typename CONFIG_T::prepare_qk_t> v[V_LANES];

    hls::stream<typename CONFIG_T::combined_shift_out_t> q_hashed_combined[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::combined_shift_out_t> k_hashed_combined[HEAD_HASH][CONFIG_T::par_factor_sort];

    hls::stream<typename CONFIG_T::sorted_index_t> q_hashed_index[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> q_hashed_index_1[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> q_hashed_index_2[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> k_hashed_index[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> k_hashed_index_1[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> k_hashed_index_2[HEAD_HASH][CONFIG_T::par_factor_sort];

    hls::stream<typename CONFIG_T::sorted_index_t> rev_q_hashed_index[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> rev_q_hashed_index_1[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::sorted_index_t> rev_q_hashed_index_2[HEAD_HASH][CONFIG_T::par_factor_sort];

    hls::stream<typename CONFIG_T::batched_index_select_out_t> batched_q[HEAD_HASH][BATCHED_QK_LANES];
    hls::stream<typename CONFIG_T::batched_index_select_out_t> batched_k[HEAD_HASH][BATCHED_QK_LANES];
    hls::stream<typename CONFIG_T::batched_index_select_out_t> batched_v[HEAD_HASH][BATCHED_V_LANES];

    hls::stream<typename CONFIG_T::qkv_res_out_denom_t> denom[HEAD_HASH][CONFIG_T::par_factor_sort];
    hls::stream<typename CONFIG_T::qkv_res_out_so_t> so[HEAD_HASH][BATCHED_V_LANES];

    hls::stream<typename CONFIG_T::batched_index_select_unsort_out_o_t> o[O_LANES];
    hls::stream<typename CONFIG_T::batched_index_select_unsort_out_logits_t> logits[LOGIT_LANES];
    hls::stream<typename CONFIG_T::o_divide_logits_out_t> o_div_logits[O_LANES];
    hls::stream<typename CONFIG_T::output_t> out_scalar[FEATURE_LANES];

    packed_stream_to_scalar_lanes<input_T, typename CONFIG_T::input_t,
                                  CONFIG_T::embed_dim, CONFIG_T::par_factor, CONFIG_T::seq_len>(x, x_scalar);
    packed_stream_to_scalar_lanes<coords_T, typename CONFIG_T::coords_t,
                                  CONFIG_T::coords_dim, CONFIG_T::par_factor, CONFIG_T::seq_len>(coords, coords_scalar);
    packed_stream_to_scalar_lanes<shift_T, typename CONFIG_T::shift_t,
                                  CONFIG_T::num_heads * CONFIG_T::num_hashes,
                                  CONFIG_T::par_factor_sort, CONFIG_T::seq_len>(combined_shifts, shifts_scalar);

    nnet::LayerNormalize<typename CONFIG_T::input_t, typename CONFIG_T::norm_out_t, layernorm_cfg>(
        x_scalar, layernorm_out,
        layer_norm_weight,
        layer_norm_bias
    );

    nnet::compute_qkv_pre_qk_merged<typename CONFIG_T::norm_out_t,
                                    typename CONFIG_T::coords_t,
                                    typename CONFIG_T::prepare_qk_t,
                                    qkv_cfg>(
        layernorm_out,
        coords_scalar,
        q_hat,
        k_hat,
        v,
        q_weight,
        k_weight,
        v_weight,
        sqrt_w
    );

    nnet::clone_stream<typename CONFIG_T::prepare_qk_t, typename CONFIG_T::prepare_qk_t,
                       (qkv_cfg::dim_per_head + qkv_cfg::coords_dim) * qkv_cfg::num_head * qkv_cfg::seq_len,
                       (qkv_cfg::dim_per_head + qkv_cfg::coords_dim) * qkv_cfg::num_head * qkv_cfg::par_factor>(
        q_hat, q_hat_1, q_hat_2
    );
    nnet::clone_stream<typename CONFIG_T::prepare_qk_t, typename CONFIG_T::prepare_qk_t,
                       (qkv_cfg::dim_per_head + qkv_cfg::coords_dim) * qkv_cfg::num_head * qkv_cfg::seq_len,
                       (qkv_cfg::dim_per_head + qkv_cfg::coords_dim) * qkv_cfg::num_head * qkv_cfg::par_factor>(
        k_hat, k_hat_1, k_hat_2
    );

    nnet::lsh_mapping_combined_shift<typename CONFIG_T::prepare_qk_t,
                                     typename CONFIG_T::shift_t,
                                     typename CONFIG_T::combined_shift_out_t,
                                     e2lsh_cfg>(
        q_hat_1,
        k_hat_1,
        shifts_scalar,
        q_hashed_combined,
        k_hashed_combined,
        alpha
    );

    SORT_HEAD_HASH:
    for (int i = 0; i < HEAD_HASH; ++i) {
        #pragma HLS UNROLL
        nnet::bucket_sort_single<typename CONFIG_T::combined_shift_out_t,
                                 typename CONFIG_T::sorted_index_t,
                                 e2lsh_cfg>(q_hashed_combined[i], q_hashed_index[i]);
        nnet::bucket_sort_single<typename CONFIG_T::combined_shift_out_t,
                                 typename CONFIG_T::sorted_index_t,
                                 e2lsh_cfg>(k_hashed_combined[i], k_hashed_index[i]);

        nnet::clone_stream<typename CONFIG_T::sorted_index_t, typename CONFIG_T::sorted_index_t,
                           e2lsh_cfg::seq_len, e2lsh_cfg::par_factor_sort>(
            q_hashed_index[i], q_hashed_index_1[i], q_hashed_index_2[i]
        );
        nnet::clone_stream<typename CONFIG_T::sorted_index_t, typename CONFIG_T::sorted_index_t,
                           e2lsh_cfg::seq_len, e2lsh_cfg::par_factor_sort>(
            k_hashed_index[i], k_hashed_index_1[i], k_hashed_index_2[i]
        );

        nnet::invert_permutation_single<typename CONFIG_T::sorted_index_t,
                                        typename CONFIG_T::sorted_index_t,
                                        e2lsh_cfg>(q_hashed_index_1[i], rev_q_hashed_index[i]);

        nnet::clone_stream<typename CONFIG_T::sorted_index_t, typename CONFIG_T::sorted_index_t,
                           e2lsh_cfg::seq_len, e2lsh_cfg::par_factor_sort>(
            rev_q_hashed_index[i], rev_q_hashed_index_1[i], rev_q_hashed_index_2[i]
        );
    }

    nnet::batched_index_select<typename CONFIG_T::sorted_index_t,
                               typename CONFIG_T::prepare_qk_t,
                               typename CONFIG_T::batched_index_select_out_t,
                               e2lsh_cfg>(q_hashed_index_2, q_hat_2, batched_q);
    nnet::batched_index_select<typename CONFIG_T::sorted_index_t,
                               typename CONFIG_T::prepare_qk_t,
                               typename CONFIG_T::batched_index_select_out_t,
                               e2lsh_cfg>(k_hashed_index_1, k_hat_2, batched_k);
    nnet::batched_index_select_2<typename CONFIG_T::sorted_index_t,
                                 typename CONFIG_T::prepare_qk_t,
                                 typename CONFIG_T::batched_index_select_out_t,
                                 e2lsh_cfg>(k_hashed_index_2, v, batched_v);

    nnet::qkv_res<typename CONFIG_T::batched_index_select_out_t,
                  typename CONFIG_T::qkv_res_out_denom_t,
                  typename CONFIG_T::qkv_res_out_so_t,
                  e2lsh_cfg>(batched_q, batched_k, batched_v, denom, so);

    UNSORT_HEAD:
    for (int h = 0; h < CONFIG_T::num_heads; ++h) {
        #pragma HLS UNROLL
        const int hash_offset = h * CONFIG_T::num_hashes;
        const int o_offset = h * CONFIG_T::dim_per_head * CONFIG_T::par_factor;
        const int logits_offset = h * CONFIG_T::par_factor;

        nnet::batched_index_select_unsort_o_num_hash<typename CONFIG_T::sorted_index_t,
                                                     typename CONFIG_T::qkv_res_out_so_t,
                                                     typename CONFIG_T::batched_index_select_unsort_out_o_t,
                                                     e2lsh_cfg>(
            &rev_q_hashed_index_1[hash_offset],
            &so[hash_offset],
            &o[o_offset]
        );
        nnet::batched_index_select_unsort_logits_num_hash<typename CONFIG_T::sorted_index_t,
                                                          typename CONFIG_T::qkv_res_out_denom_t,
                                                          typename CONFIG_T::batched_index_select_unsort_out_logits_t,
                                                          e2lsh_cfg>(
            &rev_q_hashed_index_2[hash_offset],
            &denom[hash_offset],
            &logits[logits_offset]
        );
    }

    nnet::o_divide_logits<typename CONFIG_T::batched_index_select_unsort_out_o_t,
                          typename CONFIG_T::batched_index_select_unsort_out_logits_t,
                          typename CONFIG_T::o_divide_logits_out_t,
                          e2lsh_cfg>(o, logits, o_div_logits);

    nnet::out_linear<typename CONFIG_T::o_divide_logits_out_t,
                     typename CONFIG_T::output_t,
                     e2lsh_cfg>(
        o_div_logits,
        out_scalar,
        out_linear_weight,
        out_linear_bias
    );

    scalar_lanes_to_packed_stream<typename CONFIG_T::output_t, output_T,
                                  CONFIG_T::embed_dim, CONFIG_T::par_factor, CONFIG_T::seq_len>(out_scalar, out);
}

} // namespace nnet

#endif
