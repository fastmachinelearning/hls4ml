// -----------------------------------------------------------------------------
// Vendored helper header for the HEPT hls4ml extension.
//
// Upstream source repository:
//   Howard1011/HEPT_HLS
//   https://github.com/Howard1011/HEPT_HLS/tree/main/HEPT/firmware/nnet_utils
// Upstream file URL at retrieval time:
//   https://raw.githubusercontent.com/Howard1011/HEPT_HLS/main/HEPT/firmware/nnet_utils/nnet_e2lsh_stream.h
// Retrieved on: 2026-03-30
//
// Keep this file in sync with upstream only after reviewing provenance and
// licensing for your intended use.
// -----------------------------------------------------------------------------

#ifndef NNET_E2LSH_SS_H_
#define NNET_E2LSH_SS_H_

#include "nnet_common.h"
#include "nnet_mult.h"
#include "hls_stream.h"
#include <iostream>
#include <math.h>
#include "nnet_helpers.h"
#include "hls_streamofblocks.h"
//#include "nnet_activation.h"

namespace nnet {

struct e2lsh_config {
    static const unsigned num_head = 2;
    static const unsigned dim_per_head = 16;
    static const unsigned num_hashes = 5;
    static const unsigned seq_len = 30;
    static const unsigned num_w_per_dist = 10;
    static const unsigned coords_dim = 2;
};

template<class data_T_qk, class data_T_combined_shifts, class res_T_qk, typename CONFIG_T>
void lsh_mapping_combined_shift(
    hls::stream<data_T_qk>    q_hat[CONFIG_T::num_head*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::par_factor],
    hls::stream<data_T_qk>    k_hat[CONFIG_T::num_head*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::par_factor],
    hls::stream<data_T_combined_shifts>     combined_shifts[CONFIG_T::par_factor_sort*CONFIG_T::num_head*CONFIG_T::num_hashes],
    hls::stream<res_T_qk>     q_hashed_combined[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<res_T_qk>     k_hashed_combined[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    typename CONFIG_T::weight_t               weight[(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::num_head*CONFIG_T::num_hashes])
{
    #pragma HLS INLINE
    #pragma HLS DATAFLOW

    data_T_qk data_pack_q;
    data_T_qk data_pack_k;
    data_T_combined_shifts data_pack_combined_shifts;

    res_T_qk res_pack_q;
    res_T_qk res_pack_k;
    typename CONFIG_T::accum_t row_buffer_q[CONFIG_T::num_head][CONFIG_T::num_hashes]; // 2 5
    typename CONFIG_T::accum_t row_buffer_k[CONFIG_T::num_head][CONFIG_T::num_hashes];
    #pragma HLS ARRAY_PARTITION variable=row_buffer_q complete
    #pragma HLS ARRAY_PARTITION variable=row_buffer_k complete

    
    typename CONFIG_T::hash_mul_accum_t row_buffer_q_tmp[CONFIG_T::num_head][(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)][CONFIG_T::num_hashes]; // 2 18 5
    typename CONFIG_T::hash_mul_accum_t row_buffer_k_tmp[CONFIG_T::num_head][(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)][CONFIG_T::num_hashes];
    #pragma HLS ARRAY_PARTITION variable=row_buffer_q_tmp complete
    #pragma HLS ARRAY_PARTITION variable=row_buffer_k_tmp complete
    
    typename CONFIG_T::accum_t col_buffer_q[CONFIG_T::num_head][CONFIG_T::num_hashes][CONFIG_T::seq_len]; // 2 5 30
    typename CONFIG_T::accum_t col_buffer_k[CONFIG_T::num_head][CONFIG_T::num_hashes][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=col_buffer_q complete
    #pragma HLS ARRAY_PARTITION variable=col_buffer_k complete
    
    typename CONFIG_T::accum_t row_max[CONFIG_T::num_head][CONFIG_T::num_hashes][CONFIG_T::par_factor];
    typename CONFIG_T::accum_t row_min[CONFIG_T::num_head][CONFIG_T::num_hashes][CONFIG_T::par_factor];
    #pragma HLS ARRAY_PARTITION variable=row_max complete
    #pragma HLS ARRAY_PARTITION variable=row_min complete

    
    typename CONFIG_T::accum_t real_row_max[CONFIG_T::num_head][CONFIG_T::num_hashes];
    typename CONFIG_T::accum_t real_row_min[CONFIG_T::num_head][CONFIG_T::num_hashes];
    #pragma HLS ARRAY_PARTITION variable=real_row_max complete
    #pragma HLS ARRAY_PARTITION variable=real_row_min complete

    #pragma HLS ARRAY_PARTITION variable=weight complete
    
    // 30
    LSH_MAPPING_COMBINED_SHIFT:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int pf = 0; pf < CONFIG_T::par_factor; pf++) {
            #pragma HLS UNROLL
            INIT_ROW_BUFFER:
            for(int k = 0; k < CONFIG_T::num_head; k++) {
                for(int l = 0; l < CONFIG_T::num_hashes; l++) {
                    #pragma HLS UNROLL
                    row_buffer_q[k][l] = 0;
                    row_buffer_k[k][l] = 0;
                }
            }
            // 18
            READ_AND_MULTIPLY:
            for (int j = 0; j < CONFIG_T::num_head; j++) {
                for (int k = 0; k < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); k++) {
                    int flat_idx = j * (CONFIG_T::dim_per_head+CONFIG_T::coords_dim) + k;
                
                    data_pack_q = q_hat[pf*CONFIG_T::num_head*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)+flat_idx].read();
                    data_pack_k = k_hat[pf*CONFIG_T::num_head*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)+flat_idx].read();
                
                    for (int h = 0; h < CONFIG_T::num_hashes; h++) {
                        #pragma HLS UNROLL
                        typename CONFIG_T::hash_mul_accum_t tmp_q = data_pack_q * weight[CONFIG_T::num_hashes * flat_idx + h];
                        typename CONFIG_T::hash_mul_accum_t tmp_k = data_pack_k * weight[CONFIG_T::num_hashes * flat_idx + h];
                    
                        row_buffer_q_tmp[j][k][h] = tmp_q;
                        row_buffer_k_tmp[j][k][h] = tmp_k;
                    }
                }
            }

            SUM:
            for(int j = 0; j < CONFIG_T::num_head; j++) {
                for (int k = 0; k < CONFIG_T::num_hashes; k++) {
                    #pragma HLS UNROLL
                    row_buffer_q[j][k] = 0;
                    row_buffer_k[j][k] = 0;
                    for (int h = 0; h < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); h++) {
                        #pragma HLS UNROLL
                        row_buffer_q[j][k] += row_buffer_q_tmp[j][h][k];
                        row_buffer_k[j][k] += row_buffer_k_tmp[j][h][k];
                    }
                }
            }

            MAX_MIN:
            for(int j = 0; j < CONFIG_T::num_head; j++) {
                #pragma HLS UNROLL
                for(int k = 0; k < CONFIG_T::num_hashes; k++) {
                    #pragma HLS UNROLL
                    int seq = i * CONFIG_T::par_factor + pf;
                    col_buffer_q[j][k][seq] = row_buffer_q[j][k];
                    col_buffer_k[j][k][seq] = row_buffer_k[j][k];
                }
                if(i == 0) {
                    for(int k = 0; k < CONFIG_T::num_hashes; k++) {
                        #pragma HLS UNROLL
                        row_max[j][k][pf] = (row_buffer_q[j][k] > row_buffer_k[j][k]) ? row_buffer_q[j][k] : row_buffer_k[k+5*j];
                        row_min[j][k][pf] = (row_buffer_q[j][k] < row_buffer_k[j][k]) ? row_buffer_q[j][k] : row_buffer_k[k+5*j];
                    }
                }  else {
                    for(int k = 0; k < CONFIG_T::num_hashes; k++) {
                        #pragma HLS UNROLL
                        if(row_buffer_q[j][k] > row_max[j][k][pf])
                            row_max[j][k][pf] = row_buffer_q[j][k];
                        if(row_buffer_q[j][k] < row_min[j][k][pf])
                            row_min[j][k][pf] = row_buffer_q[j][k];
                        if(row_buffer_k[j][k] > row_max[j][k][pf])
                            row_max[j][k][pf] = row_buffer_k[j][k];
                        if(row_buffer_k[j][k] < row_min[j][k][pf])
                            row_min[j][k][pf] = row_buffer_k[j][k];
                    }
                }
            }
            for(int j = 0; j < CONFIG_T::num_head; j++) {
                #pragma HLS UNROLL
                for(int k = 0; k < CONFIG_T::num_hashes; k++) {
                    #pragma HLS UNROLL
                    if(pf == 0) {
                        real_row_max[j][k] = row_max[j][k][pf];
                        real_row_min[j][k] = row_min[j][k][pf];
                    } else {
                        if(row_max[j][k][pf] > real_row_max[j][k])
                            real_row_max[j][k] = row_max[j][k][pf];
                        if(row_min[j][k][pf] < real_row_min[j][k])
                            real_row_min[j][k] = row_min[j][k][pf];
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------------------------------------------------------------------------

    LSH_MAPPING_WRITE_RESULT:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL
        int current_hash = n / CONFIG_T::num_head;
        int current_head = n % CONFIG_T::num_head;
        typename CONFIG_T::accum_t hashed_shift;
        hashed_shift = real_row_max[current_head][current_hash] - real_row_min[current_head][current_hash];
        LSH_MAPPING_WRITE_RESULT_INNER:
        for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
            #pragma HLS PIPELINE II=1 rewind
            for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                data_pack_combined_shifts = combined_shifts[n*CONFIG_T::par_factor_sort+par].read();
                int seq = i * CONFIG_T::par_factor_sort + par;
                typename CONFIG_T::accum_t tmp_q = col_buffer_q[current_head][current_hash][seq] + hashed_shift*data_pack_combined_shifts;
                typename CONFIG_T::accum_t tmp_k = col_buffer_k[current_head][current_hash][seq] + hashed_shift*data_pack_combined_shifts;
                res_pack_q = tmp_q;
                res_pack_k = tmp_k;
                q_hashed_combined[n][par].write(res_pack_q);
                k_hashed_combined[n][par].write(res_pack_k);
            }
        }
    }
}
/*
template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort_single(
    hls::stream<data_T>     data[CONFIG_T::par_factor],
    hls::stream<res_T>     index[CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF

    ap_uint<32> buckets[CONFIG_T::par_factor];
    #pragma HLS ARRAY_RESHAPE variable=buckets complete dim=1
    res_T sum[32];
    #pragma HLS ARRAY_RESHAPE variable=sum complete dim=1
    #pragma HLS BIND_STORAGE variable=sum type=ram_2p impl=bram
    res_T data_buf[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];

    BUCKKET_SORT_INIT:
    for(int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        sum[b] = 0;
    }
    BUCKKET_SORT_INNER:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            data_T data_pack;
            data_pack = data[par].read();
            if(data_pack > 31)
                data_pack = 31;
            else if(data_pack < 0)
                data_pack = 0;
            res_T rounded_data = (res_T)(data_pack);
            //std::cout << "data pack: " << data_pack << ", rounded data: " << rounded_data << std::endl;
            for(int b = 0; b < 32; b++) {
                if(b <= rounded_data)
                    buckets[par][b] = 0;
                else
                    buckets[par][b] = 1;
            }
            data_buf[i][par] = (res_T) data_pack;
        }
        for(int b = 0; b < 32; b++) {
            #pragma HLS UNROLL
            for(int par = 0; par < CONFIG_T::par_factor; par++) {
                #pragma HLS UNROLL
                if(buckets[par][b] == 1)
                    sum[b] += 1;
                else
                    sum[b] += 0;
            }
        }
    }
    WRITE_SORTED_INDEX_INNER:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            data_T data_pack = data_buf[i][par];
            res_T res_pack = sum[data_pack];
            index[par].write(res_pack);
            sum[data_pack] += 1;
        }
    }
}
*/

template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort_single(
    hls::stream<data_T> data[CONFIG_T::par_factor_sort],
    hls::stream<res_T>  index[CONFIG_T::par_factor_sort])
{
    #pragma HLS INLINE OFF

    // --------- Storage ----------
    // Keep input buffering as you had
    res_T data_buf[CONFIG_T::seq_len / CONFIG_T::par_factor_sort][CONFIG_T::par_factor_sort];

    // Pass 1: per-lane histograms
    res_T hist_lane[CONFIG_T::par_factor_sort][32];
    #pragma HLS ARRAY_PARTITION variable=hist_lane complete dim=1

    // Pass 2: totals/base/lane offsets
    res_T total[32];
    res_T base[32];
    res_T lane_off[CONFIG_T::par_factor_sort][32];
    #pragma HLS ARRAY_PARTITION variable=lane_off complete dim=1

    // Pass 3: per-lane cursors (no conflicts)
    res_T lane_cur[CONFIG_T::par_factor_sort][32];
    #pragma HLS ARRAY_PARTITION variable=lane_cur complete dim=1

    // --------- Init ---------
    // Zero histograms
    BUCKET_SORT_INIT_HIST:
    for (int p = 0; p < CONFIG_T::par_factor_sort; p++) {
        #pragma HLS UNROLL
        for (int b = 0; b < 32; b++) {
            #pragma HLS UNROLL
            hist_lane[p][b] = 0;
        }
    }

    
    // Zero lane cursors
    BUCKET_SORT_INIT_CUR:
    for (int p = 0; p < CONFIG_T::par_factor_sort; p++) {
        #pragma HLS UNROLL
        for (int b = 0; b < 32; b++) {
            #pragma HLS UNROLL
            lane_cur[p][b] = 0;
        }
    }

    // --------- Pass 1: read & build per-lane histograms ---------
    BUCKET_SORT_PASS1:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
            #pragma HLS UNROLL
            data_T raw = data[par].read();

            // Clamp to [0,31]
            if (raw > 31) raw = 31;
            else if (raw < 0) raw = 0;

            res_T k = (res_T)raw;
            data_buf[i][par] = k;

            // Lane-private histogram update -> no cross-lane conflict
            hist_lane[par][k] = hist_lane[par][k] + (res_T)1;
        }
    }

    // --------- Pass 2: totals -> base (exclusive prefix) ---------
    // totals per key
    BUCKET_SORT_TOTALS:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        res_T s = 0;
        for (int p = 0; p < CONFIG_T::par_factor_sort; p++) {
            #pragma HLS UNROLL
            s = s + hist_lane[p][b];
        }
        total[b] = s;
    }

    // base via exclusive prefix over keys
    res_T run = 0;
    BUCKET_SORT_BASE:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        base[b] = run;
        run = run + total[b];
    }

    // lane_off[par][b] = sum_{q < par} hist_lane[q][b]
    BUCKET_SORT_LANE_OFF:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        res_T s = 0;
        for (int p = 0; p < CONFIG_T::par_factor_sort; p++) {
            #pragma HLS UNROLL
            lane_off[p][b] = s;
            s = s + hist_lane[p][b];
        }
    }

    // --------- Pass 3: emit ranks with lane-local cursors ---------

    WRITE_SORTED_INDEX_INNER:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
            #pragma HLS UNROLL
            res_T k = data_buf[i][par];

            // rank = base[k] + lane_off[par][k] + lane_cur[par][k]
            res_T res_pack = base[k] + lane_off[par][k] + lane_cur[par][k];
            index[par].write(res_pack);

            // lane-private increment -> no hazard
            lane_cur[par][k] = lane_cur[par][k] + (res_T)1;
        }
    }
}


template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort_invert_single(
    hls::stream<data_T> data[CONFIG_T::par_factor],
    hls::stream<res_T>  index[CONFIG_T::par_factor],
    hls::stream<res_T>  revert_indices[CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF

    // --------- Storage ----------
    // Keep input buffering as you had
    res_T data_buf[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];

    // Pass 1: per-lane histograms
    res_T hist_lane[CONFIG_T::par_factor][32];
    #pragma HLS ARRAY_PARTITION variable=hist_lane complete dim=1

    // Pass 2: totals/base/lane offsets
    res_T total[32];
    res_T base[32];
    res_T lane_off[CONFIG_T::par_factor][32];
    #pragma HLS ARRAY_PARTITION variable=lane_off complete dim=1

    // Pass 3: per-lane cursors (no conflicts)
    res_T lane_cur[CONFIG_T::par_factor][32];
    #pragma HLS ARRAY_PARTITION variable=lane_cur complete dim=1

    res_T revert_index[CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=revert_index complete dim=1
    #pragma HLS BIND_STORAGE variable=revert_index type=ram_1p impl=register

    // --------- Init ---------
    // Zero histograms
    BUCKET_SORT_INIT_HIST:
    for (int p = 0; p < CONFIG_T::par_factor; p++) {
        #pragma HLS UNROLL
        for (int b = 0; b < 32; b++) {
            #pragma HLS UNROLL
            hist_lane[p][b] = 0;
        }
    }

    
    // Zero lane cursors
    BUCKET_SORT_INIT_CUR:
    for (int p = 0; p < CONFIG_T::par_factor; p++) {
        #pragma HLS UNROLL
        for (int b = 0; b < 32; b++) {
            #pragma HLS UNROLL
            lane_cur[p][b] = 0;
        }
    }

    // --------- Pass 1: read & build per-lane histograms ---------
    BUCKET_SORT_PASS1:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            data_T raw = data[par].read();

            // Clamp to [0,31]
            if (raw > 31) raw = 31;
            else if (raw < 0) raw = 0;

            res_T k = (res_T)raw;
            data_buf[i][par] = k;

            // Lane-private histogram update -> no cross-lane conflict
            hist_lane[par][k] = hist_lane[par][k] + (res_T)1;
        }
    }

    // --------- Pass 2: totals -> base (exclusive prefix) ---------
    // totals per key
    BUCKET_SORT_TOTALS:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        res_T s = 0;
        for (int p = 0; p < CONFIG_T::par_factor; p++) {
            #pragma HLS UNROLL
            s = s + hist_lane[p][b];
        }
        total[b] = s;
    }

    // base via exclusive prefix over keys
    res_T run = 0;
    BUCKET_SORT_BASE:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        base[b] = run;
        run = run + total[b];
    }

    // lane_off[par][b] = sum_{q < par} hist_lane[q][b]
    BUCKET_SORT_LANE_OFF:
    for (int b = 0; b < 32; b++) {
        #pragma HLS UNROLL
        res_T s = 0;
        for (int p = 0; p < CONFIG_T::par_factor; p++) {
            #pragma HLS UNROLL
            lane_off[p][b] = s;
            s = s + hist_lane[p][b];
        }
    }

    // --------- Pass 3: emit ranks with lane-local cursors ---------

    WRITE_SORTED_INDEX_INNER:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            res_T k = data_buf[i][par];

            // rank = base[k] + lane_off[par][k] + lane_cur[par][k]
            res_T res_pack = base[k] + lane_off[par][k] + lane_cur[par][k];
            index[par].write(res_pack);

            revert_index[res_pack] = (res_T)(i * CONFIG_T::par_factor + par);

            // lane-private increment -> no hazard
            lane_cur[par][k] = lane_cur[par][k] + (res_T)1;
        }
    }

    // --------- Write revert indices ---------
    WRITE_INVERT_INDEX_INNER:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            res_T res_pack = revert_index[i * CONFIG_T::par_factor + par];
            revert_indices[par].write(res_pack);
        }
    }
}

/*
template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort_invert_single(
    hls::stream<data_T>     data[CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>     index[CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>     revert_indices[CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor])
{
    #pragma HLS INLINE OFF
    ap_uint<32> buckets[CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    #pragma HLS ARRAY_RESHAPE variable=buckets complete dim=1
    res_T sum[32];
    #pragma HLS ARRAY_RESHAPE variable=sum complete dim=1
    res_T data_buf[CONFIG_T::inter_reuse_factor][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    res_T revert_index[CONFIG_T::seq_len];
    #pragma HLS ARRAY_RESHAPE variable=revert_index complete dim=1

    BUCKKET_SORT_INVERT_INIT:
    for(int j = 0; j < 32; j++) {
        #pragma HLS UNROLL
        sum[j] = 0;
    }
    BUCKKET_SORT_INVERT_INNER:
    for(int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
            data_T data_pack;
            data_pack = data[i].read();
            if(data_pack > 31)
                data_pack = 31;
            else if(data_pack < 0)
                data_pack = 0;
            res_T rounded_data = (res_T)(data_pack);
            //std::cout << "data pack: " << data_pack << ", rounded data: " << rounded_data << std::endl;
            for(int j = 0; j < 32; j++) {
                if(j <= rounded_data)
                    buckets[i][j] = 0;
                else
                    buckets[i][j] = 1;
            }
            data_buf[f][i] = (res_T) data_pack;
        }
        for(int j = 0; j < 32; j++) {
            #pragma HLS UNROLL
            for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
                #pragma HLS UNROLL
                if(buckets[i][j] == 1)
                    sum[j] += 1;
                else
                    sum[j] += 0;
            }
        }
    }
    
    WRITE_SORTED_INVERT_INDEX_INNER:
    for(int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
            data_T data_pack = data_buf[f][i];
            res_T res_pack = sum[data_pack];
            index[i%(CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)].write(res_pack);
            revert_index[res_pack] = (res_T)i;
            sum[data_pack] += 1;
        }
    } 
    WRITE_INVERT_INDEX_INNER:
    for(int i = 0; i < CONFIG_T::inter_reuse_factor; i++) {
        #pragma HLS PIPELINE II=1
        for(int j = 0; j < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; j++) {
            revert_indices[j].write(revert_index[i]);
        }
    }
    
}
*/
/*
template<class data_T, class res_T, typename CONFIG_T>
void invert_permutation_single(
    hls::stream<data_T>     indices[CONFIG_T::par_factor_sort],
    hls::stream<res_T>      revert_indices[CONFIG_T::par_factor_sort])
{
    #pragma HLS INLINE OFF
    data_T reverted_index[CONFIG_T::seq_len];
    #pragma HLS ARRAY_RESHAPE variable=reverted_index type=block factor=CONFIG_T::par_factor_sort dim=1
    #pragma HLS BIND_STORAGE variable=reverted_index type=ram_1p impl=register
    INVERT_PERMUTATION_READ_CAL_LOOP:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        #pragma HLS DEPENDENCE variable=reverted_index inter false
        for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
            #pragma HLS UNROLL
            data_T data_pack = indices[par].read();
            int global_j = i * CONFIG_T::par_factor_sort + par;
            reverted_index[data_pack] = (data_T)global_j;
        }
    }

    INVERT_PERMUTATION_WRITE_LOOP:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
            #pragma HLS UNROLL
            int seq = i * CONFIG_T::par_factor_sort + par;
            res_T res_pack = (res_T)reverted_index[seq];
            revert_indices[par].write(res_pack);
        }
    }
}
*/

// Invert a permutation of size N=CONFIG_T::seq_len (here keys 0..599), PAR lanes.
// No scatter into a single array; all writes are lane-private.
// Guaranteed II=1 at ingest and emit.

template<class data_T, class res_T, typename CONFIG_T>
void invert_permutation_single(
    hls::stream<data_T> indices[CONFIG_T::par_factor_sort],
    hls::stream<res_T>  revert_indices[CONFIG_T::par_factor_sort])
{
    #pragma HLS INLINE OFF

    const int PAR = CONFIG_T::par_factor_sort;
    const int N   = CONFIG_T::seq_len;

    // lane-private tables
    res_T value_at_lane[CONFIG_T::par_factor_sort][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=value_at_lane complete dim=1
    #pragma HLS BIND_STORAGE  variable=value_at_lane type=ram_2p impl=bram

    ap_uint<1> present[CONFIG_T::par_factor_sort][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=present complete dim=1
    #pragma HLS BIND_STORAGE  variable=present type=ram_2p impl=bram

    INIT_PRESENT:
    for (int p = 0; p < PAR; p++) {
        #pragma HLS UNROLL
        for (int k = 0; k < N; k++) {
            #pragma HLS UNROLL
            present[p][k] = 0;
        }
    }

    // ---- Pass 1: ingest (PAR inputs/cycle, II=1) ----
    INGEST:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; ++i) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor_sort; ++par) {
            #pragma HLS UNROLL
            data_T k = indices[par].read();          // 0..N-1, unique
            res_T  j = (res_T)(i * CONFIG_T::par_factor_sort + par);       // original position
            value_at_lane[par][k] = j;
            present[par][k]       = 1;
        }
    }

    // ---- Pass 2: emit (PAR outputs/cycle, II=1) ----
    EMIT:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; ++i) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor_sort; ++par) {
            #pragma HLS UNROLL
            int k = i * CONFIG_T::par_factor_sort + par;
            res_T out = 0;
            // small PAR-to-1 select (tiny)
            for (int p = 0; p < CONFIG_T::par_factor_sort; ++p) {
                #pragma HLS UNROLL
                if (present[p][k]) out = value_at_lane[p][k];
            }
            revert_indices[par].write(out);
        }
    }
}



template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort(
    hls::stream<data_T>     data[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>     index[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor])
{
    #pragma HLS INLINE OFF

    ap_uint<512> buckets[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=buckets complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buckets complete dim=2
    res_T sum[CONFIG_T::num_head*CONFIG_T::num_hashes][512];
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=2
    res_T data_buf[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::inter_reuse_factor][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=2

    BUCKKET_SORT_OUTER:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL

        BUCKKET_SORT_INIT:
        for(int j = 0; j < 512; j++) {
            #pragma HLS UNROLL
            sum[n][j] = 0;
        }
        BUCKKET_SORT_INNER:
        for(int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
            #pragma HLS PIPELINE II=1 rewind
            for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
                data_T data_pack;
                data_pack = data[n][i].read();
                if(data_pack > 511)
                    data_pack = 511;
                else if(data_pack < 0)
                    data_pack = 0;
                res_T rounded_data = (res_T)(data_pack);
                //std::cout << "data pack: " << data_pack << ", rounded data: " << rounded_data << std::endl;
                for(int j = 0; j < 512; j++) {
                    if(j <= rounded_data)
                        buckets[n][i][j] = 0;
                    else
                        buckets[n][i][j] = 1;
                }
                data_buf[n][f][i] = (res_T) data_pack;
            }
            for(int j = 0; j < 512; j++) {
                #pragma HLS UNROLL
                for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
                    #pragma HLS UNROLL
                    if(buckets[n][i][j] == 1)
                        sum[n][j] += 1;
                    else
                        sum[n][j] += 0;
                }
            }
        }
    }
    WRITE_SORTED_INDEX_OUTER:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL
        WRITE_SORTED_INDEX_INNER:
        for(int i = 0; i < CONFIG_T::seq_len; i++) {
            #pragma HLS PIPELINE II=1 rewind
            data_T data_pack = data_buf[i / (CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)][i % (CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)];
            res_T res_pack = sum[data_pack];
            index[n][i%(CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)].write(res_pack);
            sum[n][data_pack] += 1;
        }
        
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void bucket_sort_invert(
    hls::stream<data_T>     data[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>     index[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>     revert_indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor])
{
    #pragma HLS INLINE OFF
    ap_uint<512> buckets[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=buckets complete dim=1
    #pragma HLS ARRAY_PARTITION variable=buckets complete dim=2
    res_T sum[CONFIG_T::num_head*CONFIG_T::num_hashes][512];
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sum complete dim=2
    res_T data_buf[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::inter_reuse_factor][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor];
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=2
    res_T revert_index[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=revert_index complete dim=1
    #pragma HLS ARRAY_PARTITION variable=revert_index complete dim=2

    BUCKKET_SORT_INVERT_OUTER:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL
        for(int j = 0; j < 512; j++) {
            #pragma HLS UNROLL
            sum[n][j] = 0;
        }
        for(int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
            #pragma HLS PIPELINE II=1 rewind
            for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
                data_T data_pack;
                data_pack = data[n][i].read();
                if(data_pack > 511)
                    data_pack = 511;
                else if(data_pack < 0)
                    data_pack = 0;
                res_T rounded_data = (res_T)(data_pack);
                //std::cout << "data pack: " << data_pack << ", rounded data: " << rounded_data << std::endl;
                for(int j = 0; j < 512; j++) {
                    if(j <= rounded_data)
                        buckets[n][i][j] = 0;
                    else
                        buckets[n][i][j] = 1;
                }
                data_buf[n][f][i] = (res_T) data_pack;
            }
            for(int j = 0; j < 512; j++) {
                #pragma HLS UNROLL
                for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; i++) {
                    #pragma HLS UNROLL
                    if(buckets[n][i][j] == 1)
                        sum[n][j] += 1;
                    else
                        sum[n][j] += 0;
                }
            }
        }
    }
    WRITE_SORTED_INVERT_INDEX_OUTER:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL
        WRITE_SORTED_INVERT_INDEX_INNER:
        for(int i = 0; i < CONFIG_T::seq_len; i++) {
            #pragma HLS PIPELINE II=1
            data_T data_pack = data_buf[n][i / (CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)][i % (CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)];
            res_T res_pack = sum[n][data_pack];
            index[n][i%(CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor)].write(res_pack);
            revert_index[n][res_pack] = (res_T)i;
            sum[n][data_pack] += 1;
        }
    } 
    WRITE_INVERT_INDEX_OUTER:
    for(int n = 0; n < CONFIG_T::num_hashes*CONFIG_T::num_head; n++) {
        #pragma HLS UNROLL
        WRITE_INVERT_INDEX_INNER:
        for(int i = 0; i < CONFIG_T::inter_reuse_factor; i++) {
            #pragma HLS PIPELINE II=1
            for(int j = 0; j < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; j++) {
                revert_indices[n][j].write(revert_index[n][i]);
            }
        }
    }
}


template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T> 
void batched_index_select(
    hls::stream<data_T_indices> indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values [CONFIG_T::num_head*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)*CONFIG_T::par_factor],
    hls::stream<res_T>          res_data[CONFIG_T::num_head*CONFIG_T::num_hashes][(CONFIG_T::dim_per_head+CONFIG_T::coords_dim) * CONFIG_T::par_factor_sort])
{
    #pragma HLS INLINE OFF

    // Store only [seq][head] (hash-independent)
    typedef typename CONFIG_T::qk_tile_wide_t wide_t;

    wide_t data_buf[CONFIG_T::seq_len][CONFIG_T::num_tiles];   // num_tiles == num_head
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=data_buf complete dim=2
    #pragma HLS BIND_STORAGE  variable=data_buf type=ram_1p impl=bram
    // Consider impl=uram if capacity/timing requires.

    // -------- Stage 1: Read all values -> BRAM --------
    BATCHED_INDEX_SELECT_READ_VALUE_LOOP:
    for (int seq = 0; seq < CONFIG_T::seq_len / CONFIG_T::par_factor; seq++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            const int global_seq = seq * CONFIG_T::par_factor + par;

            // Read values for all heads
            for (int head = 0; head < CONFIG_T::num_head; head++) {
                wide_t data_buf_row;
                for (int d = 0; d < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); d++) {
                    #pragma HLS UNROLL
                    const int row = head * (CONFIG_T::dim_per_head+CONFIG_T::coords_dim) + d;
                    const int lo  = d * CONFIG_T::elem_bits;
                    const int hi  = lo + CONFIG_T::elem_bits - 1;

                    data_T_values data_pack_values =
                        values[row + par * CONFIG_T::num_head * (CONFIG_T::dim_per_head+CONFIG_T::coords_dim)].read();

                    // --- reverted to reinterpret_cast packing ---
                    typename CONFIG_T::elem_bits_t bits =
                        *reinterpret_cast<typename CONFIG_T::elem_bits_t*>(&data_pack_values);

                    data_buf_row.range(hi, lo) = bits;
                }
                // Write once per (seq, head) — no redundant hash writes
                data_buf[global_seq][head] = data_buf_row;
            }
        }
    }

    // -------- Stage 2: Index select -> stream out --------
    BATCHED_INDEX_SELECT_WRITE_RES_INNER:
    for (int seq = 0; seq < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; seq++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int i = 0; i < CONFIG_T::num_hashes*CONFIG_T::num_head; i++) {
            #pragma HLS UNROLL
            const int current_head = i % CONFIG_T::num_head;
            for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                #pragma HLS UNROLL
                data_T_indices data_pack_indices = indices[i][par].read();
                wide_t data_buf_row = data_buf[(int)data_pack_indices][current_head];

                // --- reverted to reinterpret_cast unpacking ---
                WRITE_ROW_TO_STREAM:
                for (int y = 0; y < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); y++) {
                    const int lo = y * CONFIG_T::elem_bits;
                    const int hi = lo + CONFIG_T::elem_bits - 1;

                    typename CONFIG_T::elem_bits_t bits = data_buf_row.range(hi, lo);
                    res_T res_pack = *reinterpret_cast<res_T*>(&bits);

                    res_data[i][par*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim) + y].write(res_pack);
                }
            }
        }
    }
}




template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_2(
    hls::stream<data_T_indices> indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values [CONFIG_T::num_head * CONFIG_T::dim_per_head * CONFIG_T::par_factor],
    hls::stream<res_T>          res_data[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort * CONFIG_T::dim_per_head])
{
    #pragma HLS INLINE OFF

    // Store only [seq][head] (hash-independent)
    typedef typename CONFIG_T::v_tile_wide_t wide_t;

    wide_t data_buf[CONFIG_T::seq_len][CONFIG_T::num_tiles]; // num_tiles == num_head
    #pragma HLS ARRAY_PARTITION variable=data_buf complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=data_buf complete dim=2
    #pragma HLS BIND_STORAGE  variable=data_buf type=ram_1p impl=bram
    // Switch to impl=uram if capacity/timing requires.

    // -------- Stage 1: Read all values -> BRAM --------
    BATCHED_INDEX_SELECT_2_READ_VALUE_LOOP:
    for (int seq = 0; seq < CONFIG_T::seq_len / CONFIG_T::par_factor; seq++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            const int global_seq = seq * CONFIG_T::par_factor + par;

            for (int head = 0; head < CONFIG_T::num_head; head++) {
                wide_t data_buf_row;

                for (int d = 0; d < CONFIG_T::dim_per_head; d++) {
                    #pragma HLS UNROLL
                    const int row = head * CONFIG_T::dim_per_head + d;
                    const int lo  = d * CONFIG_T::elem_bits;
                    const int hi  = lo + CONFIG_T::elem_bits - 1;

                    data_T_values data_pack_values =
                        values[row + par * CONFIG_T::num_head * CONFIG_T::dim_per_head].read();

                    // pack element into tile slice (reverted to reinterpret_cast)
                    typename CONFIG_T::elem_bits_t bits =
                        *reinterpret_cast<typename CONFIG_T::elem_bits_t*>(&data_pack_values);
                    data_buf_row.range(hi, lo) = bits;
                }

                // Write once per (seq, head) — no redundant hash writes
                data_buf[global_seq][head] = data_buf_row;
            }
        }
    }

    // -------- Stage 2: Index select -> stream out --------
    //BATCHED_INDEX_SELECT_2_WRITE_RES_OUTER:
        // const int current_hash = i / CONFIG_T::num_head; // kept if needed for bookkeeping

    BATCHED_INDEX_SELECT_2_WRITE_RES_INNER:
    for (int seq = 0; seq < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; seq++) {
        #pragma HLS PIPELINE II=1 rewind
        for (int i = 0; i < CONFIG_T::num_hashes * CONFIG_T::num_head; i++) {
            #pragma HLS UNROLL
            const int current_head = i % CONFIG_T::num_head;
            for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                #pragma HLS UNROLL

                data_T_indices idx = indices[i][par].read();
                wide_t data_buf_row = data_buf[(int)idx][current_head];

                // Directly read row from BRAM and stream out
                WRITE_ROW_TO_STREAM:
                for (int y = 0; y < CONFIG_T::dim_per_head; y++) {
                    const int lo = y * CONFIG_T::elem_bits;
                    const int hi = lo + CONFIG_T::elem_bits - 1;

                    typename CONFIG_T::elem_bits_t bits = data_buf_row.range(hi, lo);
                    res_T res_pack = *reinterpret_cast<res_T*>(&bits);
                    res_data[i][par * CONFIG_T::dim_per_head + y].write(res_pack);
                }
            }
        }
    }
}





template <typename CONFIG_T> 
void init_exp_table_attn(typename CONFIG_T::exp_table_t table_out[CONFIG_T::exp_table_size])
{
    for (int ii = 0; ii < CONFIG_T::exp_table_size; ii++) {
        // First, convert from table index to X-value (signed 8-bit, range -8 to +8)
        double in_val = 2 * double(CONFIG_T::exp_range) * (ii - double(CONFIG_T::exp_table_size) / 2.0) / double(CONFIG_T::exp_table_size);
        // Next, compute lookup table function
        typename CONFIG_T::exp_table_t real_val = std::exp(in_val);
        // std::cout << "Lookup table In Value: " << in_val << " Result: " << real_val << std::endl;
        table_out[ii] = real_val;
    }
}

template<typename CONFIG_T>
typename CONFIG_T::exp_table_t lookup_exp_attn(
    typename CONFIG_T::accum_t data)
{
    #ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #else
    static bool initialized = false;
    static typename CONFIG_T::exp_table_t exp_table[CONFIG_T::exp_table_size];
    #endif

    if (!initialized) {
        //init_exp_table_legacy<CONFIG_T, CONFIG_T::table_size>(exp_table);
        init_exp_table_attn<CONFIG_T>(exp_table);
        initialized = true;
    }
    //std::cout << "fixed point data before: " << data << std::endl;
    int data_round = int(data*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2)));
    //std::cout << "data_round: " << data_round << std::endl;
    //std::cout << "fixed point data: " << static_cast<typename CONFIG_T::accum_t>(data)*(CONFIG_T::exp_range*2)/CONFIG_T::exp_table_size << std::endl;
    int index = data_round + CONFIG_T::exp_range*(CONFIG_T::exp_table_size/(CONFIG_T::exp_range*2));
    //print index
    // if (index > CONFIG_T::exp_table_size-1)
    // std::cout << "index out of range: " << index << std::endl;
    // else if (index < 0)
    // std::cout << "index out of range: " << index << std::endl;
    if (index < 0)   index = 0;
    if (index > CONFIG_T::exp_table_size-1) index = CONFIG_T::exp_table_size-1;
    // std::cout << "index: " << index << " value:" << exp_table[index] <<std::endl;
    return exp_table[index];
}

template<class data_T, class res_T_denom, class res_T_so, typename CONFIG_T>
void qkv_res(
    hls::stream<data_T>     Q[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)],
    hls::stream<data_T>     K[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)],
    hls::stream<data_T>     V[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort*CONFIG_T::dim_per_head],
    hls::stream<res_T_denom>      denom[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<res_T_so>     so[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort*CONFIG_T::dim_per_head])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    QKV_RES_OUTER:
    for(int i = 0; i < CONFIG_T::num_hashes*CONFIG_T::num_head; i++) {
        #pragma HLS UNROLL factor=CONFIG_T::num_head
        QKV_RES_INNER:
        for(int f = 0; f < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; f++) {
            #pragma HLS PIPELINE II=1 rewind
            for(int p = 0; p < CONFIG_T::par_factor_sort / CONFIG_T::block_size; p++) {
                SQUARE_MUL_TWO:
                data_T buffer_k[CONFIG_T::block_size][CONFIG_T::dim_per_head+CONFIG_T::coords_dim];
                typename CONFIG_T::accum_t q_sq[CONFIG_T::block_size];
                typename CONFIG_T::accum_t k_sq[CONFIG_T::block_size];
                typename CONFIG_T::accum_t denom_row_buffer[CONFIG_T::block_size];
                typename CONFIG_T::accum_t so_row_buffer[CONFIG_T::dim_per_head*CONFIG_T::block_size];
                typename CONFIG_T::accum_t qk_buffer[CONFIG_T::block_size][CONFIG_T::block_size];
                for(int j = 0; j < CONFIG_T::block_size; j++) {
                    k_sq[j] = 0;
                    for(int k = 0; k < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); k++) {
                        data_T data_pack_k = K[i][(p*CONFIG_T::block_size+j)*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)+k].read();
                        // std::cout << "K[" << p << "][" << j << "][" << k << "] = " << data_pack_k << std::endl;
                        k_sq[j] -= data_pack_k*data_pack_k;
                        buffer_k[j][k] = data_pack_k;
                    }
                    k_sq[j] = k_sq[j] / 2;
                    // std::cout << "k_sq[" << j << "] = " << k_sq[j] << std::endl;
                }
                CAL_DENOM_OUTER:
                for(int j = 0; j < CONFIG_T::block_size; j++) {
                    q_sq[j] = 0;
                    INIT_DENOM_ROW_BUF:
                    for(int k = 0; k < CONFIG_T::block_size; k++) {
                        #pragma HLS UNROLL
                        denom_row_buffer[k] = 0;
                    }
                    CAL_DENOM:
                    for(int k = 0; k < (CONFIG_T::dim_per_head+CONFIG_T::coords_dim); k++) {
                        #pragma HLS UNROLL
                        data_T data_pack_q = Q[i][(p*CONFIG_T::block_size+j)*(CONFIG_T::dim_per_head+CONFIG_T::coords_dim)+k].read();
                        q_sq[j] -= data_pack_q*data_pack_q;
                        for(int x = 0; x < CONFIG_T::block_size; x++) {
                            denom_row_buffer[x] += data_pack_q*buffer_k[x][k];
                            //std::cout << "denom_row_buffer[" << x << "] += " << data_pack_q << "*" << buffer_k[x][k] << std::endl;
                        }
                    }
                    q_sq[j] = q_sq[j] / 2;
                    typename CONFIG_T::accum_t res_denom_temp = 0;
                    // std::cout << "q_sq[" << j << "] = " << q_sq[j] << std::endl;
                    SUM_EXP:
                    for(int k = 0; k < CONFIG_T::block_size; k++) {
                        #pragma HLS UNROLL
                        // std::cout << "denom_row_buffer_before[" << k << "] = " << denom_row_buffer[k] << " q_sq[" << j << "] = " << q_sq[j] << " k_sq[" << k << "] = " << k_sq[k] << std::endl;
                        denom_row_buffer[k] = denom_row_buffer[k] + q_sq[j] + k_sq[k];
                        if(denom_row_buffer[k] > 0)
                            denom_row_buffer[k] = 0;
                        // std::cout << "denom_row_buffer[" << k << "] = " << denom_row_buffer[k] << std::endl;
                        denom_row_buffer[k] = (typename CONFIG_T::accum_t)lookup_exp_attn<CONFIG_T>(denom_row_buffer[k]);
                        // std::cout << "denom_row_buffer exp[" << k << "] = " << denom_row_buffer[k] << std::endl;
                    }
                    SUM_DENOM:
                    for(int k = 0; k < CONFIG_T::block_size; k++) {
                        res_denom_temp += denom_row_buffer[k];
                        qk_buffer[j][k] = denom_row_buffer[k];
                    }
                    // std::cout << "res_denom_temp: " << res_denom_temp << std::endl;
                    denom[i][p*CONFIG_T::block_size+j].write((res_T_denom)res_denom_temp);
                }
                INIT_SO_ROW_BUF:
                for(int j = 0; j < CONFIG_T::block_size*CONFIG_T::dim_per_head; j++) {
                    #pragma HLS UNROLL
                    so_row_buffer[j] = 0;
                }
                CAL_SO:
                for(int j = 0; j < CONFIG_T::block_size; j++) {
                    for(int k = 0; k < CONFIG_T::dim_per_head; k++) {
                        data_T data_pack_v = V[i][(p*CONFIG_T::block_size+j)*CONFIG_T::dim_per_head+k].read();
                        for(int l = 0; l < CONFIG_T::block_size; l++) {
                            #pragma HLS UNROLL
                            so_row_buffer[l*CONFIG_T::dim_per_head+k] += qk_buffer[l][j]*data_pack_v;
                        }
                    }
                }
                WRITE_SO:
                for(int j = 0; j < CONFIG_T::block_size; j++) {
                    for(int k = 0; k < CONFIG_T::dim_per_head; k++) {
                        #pragma HLS UNROLL
                        so[i][(p*CONFIG_T::block_size+j)*CONFIG_T::dim_per_head+k].write((res_T_so)so_row_buffer[j*CONFIG_T::dim_per_head+k]);
                    }
                }
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void invert_permutation(
    hls::stream<data_T>     indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor],
    hls::stream<res_T>      revert_indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor])
{
    #pragma HLS INLINE OFF
    INVERT_PERMUTATION:
    for (int i = 0; i < CONFIG_T::num_hashes * CONFIG_T::num_head; i++) {
        #pragma HLS UNROLL
        // Big buffer in URAM
        data_T reverted_index[CONFIG_T::seq_len];
        //#pragma HLS BIND_STORAGE variable=reverted_index type=ram_1p impl=lutram
        #pragma HLS ARRAY_PARTITION variable=reverted_index complete dim=1
        READ_CAL_LOOP:
        for (int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
            #pragma HLS PIPELINE II=1 rewind
            // -------- Stage 1: Write indices into URAM --------
            for (int j = 0; j < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; j++) {
                #pragma HLS DEPENDENCE variable=reverted_index inter false
                #pragma HLS UNROLL
                data_T data_pack = indices[i][j].read();
                int global_j = f * CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor + j;
                reverted_index[data_pack] = (data_T)global_j;
            }
        }

        WRITE_LOOP:
        for (int f = 0; f < CONFIG_T::inter_reuse_factor; f++) {
            // -------- Stage 3: Stream results from BRAM --------
            #pragma HLS PIPELINE II=1 rewind
            for (int j = 0; j < CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor; j++) {
                #pragma HLS UNROLL
                int global_j = f * CONFIG_T::seq_len / CONFIG_T::inter_reuse_factor + j;
                res_T res_pack = (res_T)reverted_index[global_j];
                revert_indices[i][j].write(res_pack);
            }
        }
    }
}

template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_unsort_o_num_hash(
    hls::stream<data_T_indices> indices[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort * CONFIG_T::dim_per_head],
    hls::stream<res_T> res_data[CONFIG_T::dim_per_head * CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // Local buffers in BRAM

    typename CONFIG_T::unsort_o_tile_wide_t data_buffer[CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];
    #pragma HLS BIND_STORAGE variable=data_buffer  type=ram_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=3


    typename CONFIG_T::unsort_o_tile_wide_t data_buffer_row[CONFIG_T::num_hashes][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=data_buffer_row complete dim=2
    #pragma HLS BIND_STORAGE variable=data_buffer_row  type=ram_1p impl=bram
    // Main computation
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
            // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    #pragma HLS UNROLL
                    int lo = l * CONFIG_T::unsort_o_elem_bits;
                    int hi = lo + CONFIG_T::unsort_o_elem_bits - 1;
                    data_T_values data_pack_values = values[n][par*CONFIG_T::dim_per_head+l].read();
                    data_buffer_row[n][seq].range(hi, lo) = *reinterpret_cast<typename CONFIG_T::unsort_o_elem_bits_t*>(&data_pack_values);
                }
            }
        }
    }
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER_2:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                data_T_indices data_pack_indices = indices[n][par].read();
                data_buffer[n][i][par] = data_buffer_row[n][data_pack_indices];
            }
        }
    }
    
    // --- Stage 3: Write final results ---
    typename CONFIG_T::accum_t data_accum[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor][CONFIG_T::dim_per_head];
    #pragma HLS ARRAY_PARTITION variable=data_accum complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=data_accum complete dim=3
    BATCHED_INDEX_SELECT_UNSORT_O_WRITE:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1
            for(int par = 0; par < CONFIG_T::par_factor; par++) {
                #pragma HLS UNROLL
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    int lo = l * CONFIG_T::unsort_o_elem_bits;
                    int hi = lo + CONFIG_T::unsort_o_elem_bits - 1;
                    typename CONFIG_T::unsort_logits_elem_bits_t bits = data_buffer[n][i][par].range(hi, lo);
                    data_T_values data_value = *reinterpret_cast<data_T_values*>(&bits);
                    if(n == 0) {
                        data_accum[i][par][l] = data_value;
                    } else if (n == CONFIG_T::num_hashes - 1) {
                        res_T res_pack = (res_T)(data_accum[i][par][l] + data_value);
                        res_data[l * CONFIG_T::par_factor + par].write(res_pack);
                    } else {
                        data_accum[i][par][l] += data_value;
                    }
                }
            }
        }
    }
}

template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_unsort_o_num_hash_uram(
    hls::stream<data_T_indices> indices[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort * CONFIG_T::dim_per_head],
    hls::stream<res_T> res_data[CONFIG_T::dim_per_head * CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // Local buffers in BRAM

    typename CONFIG_T::unsort_o_tile_wide_t data_buffer[CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];
    #pragma HLS BIND_STORAGE variable=data_buffer  type=ram_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=3


    typename CONFIG_T::unsort_o_tile_wide_t data_buffer_row[CONFIG_T::num_hashes][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=data_buffer_row complete dim=2
    #pragma HLS BIND_STORAGE variable=data_buffer_row  type=ram_1p impl=uram
    // Main computation
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
            // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    #pragma HLS UNROLL
                    int lo = l * CONFIG_T::unsort_o_elem_bits;
                    int hi = lo + CONFIG_T::unsort_o_elem_bits - 1;
                    data_T_values data_pack_values = values[n][par*CONFIG_T::dim_per_head+l].read();
                    data_buffer_row[n][seq].range(hi, lo) = *reinterpret_cast<typename CONFIG_T::unsort_o_elem_bits_t*>(&data_pack_values);
                }
            }
        }
    }
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER_2:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                data_T_indices data_pack_indices = indices[n][par].read();
                data_buffer[n][i][par] = data_buffer_row[n][data_pack_indices];
            }
        }
    }
    
    // --- Stage 3: Write final results ---
    typename CONFIG_T::accum_t data_accum[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor][CONFIG_T::dim_per_head];
    #pragma HLS ARRAY_PARTITION variable=data_accum complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=data_accum complete dim=3
    BATCHED_INDEX_SELECT_UNSORT_O_WRITE:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1
            for(int par = 0; par < CONFIG_T::par_factor; par++) {
                #pragma HLS UNROLL
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    int lo = l * CONFIG_T::unsort_o_elem_bits;
                    int hi = lo + CONFIG_T::unsort_o_elem_bits - 1;
                    typename CONFIG_T::unsort_logits_elem_bits_t bits = data_buffer[n][i][par].range(hi, lo);
                    data_T_values data_value = *reinterpret_cast<data_T_values*>(&bits);
                    if(n == 0) {
                        data_accum[i][par][l] = data_value;
                    } else if (n == CONFIG_T::num_hashes - 1) {
                        res_T res_pack = (res_T)(data_accum[i][par][l] + data_value);
                        res_data[l * CONFIG_T::par_factor + par].write(res_pack);
                    } else {
                        data_accum[i][par][l] += data_value;
                    }
                }
            }
        }
    }
}

template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_unsort_o(
    hls::stream<data_T_indices> indices[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values[CONFIG_T::num_head*CONFIG_T::num_hashes][CONFIG_T::par_factor_sort * CONFIG_T::dim_per_head],
    hls::stream<res_T> res_data[CONFIG_T::dim_per_head * CONFIG_T::num_head* CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // Local buffers in BRAM

    typename CONFIG_T::unsort_logits_tile_wide_t data_buffer[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor][CONFIG_T::dim_per_head];
    #pragma HLS BIND_STORAGE variable=data_buffer  type=ram_1p impl=bram
    #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=data_buffer complete dim=3


    typename CONFIG_T::unsort_o_tile_wide_t data_buffer_row[CONFIG_T::seq_len][CONFIG_T::num_hashes * CONFIG_T::num_head];
    #pragma HLS ARRAY_PARTITION variable=data_buffer_row complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=data_buffer_row complete dim=2
    #pragma HLS BIND_STORAGE variable=data_buffer_row  type=ram_1p impl=bram
    // Main computation
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1
        for(int n = 0; n < CONFIG_T::num_hashes * CONFIG_T::num_head; n++) {
            #pragma HLS UNROLL factor=CONFIG_T::num_head
            // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    #pragma HLS UNROLL
                    int lo = n * CONFIG_T::unsort_o_elem_bits;
                    int hi = lo + CONFIG_T::unsort_o_elem_bits - 1;
                    data_T_values data_pack_values = values[n][par*CONFIG_T::dim_per_head+l].read();
                    data_buffer_row[seq][CONFIG_T::num_hashes * CONFIG_T::num_head].range(hi, lo) = *reinterpret_cast<typename CONFIG_T::unsort_o_elem_bits_t*>(&data_pack_values);
                }
            }
        }
    }
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER_2:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1
        // --- Stage 1: Read values into buffer ---
        for(int n = 0; n < CONFIG_T::num_hashes * CONFIG_T::num_head; n++) {
            #pragma HLS UNROLL factor=CONFIG_T::num_head
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;
                data_T_indices data_pack_indices = indices[n][par].read();
                int lo = n * CONFIG_T::unsort_logits_elem_bits;
                int hi = lo + CONFIG_T::unsort_logits_elem_bits - 1;
                for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                    #pragma HLS UNROLL
                    int lo_small = l * CONFIG_T::unsort_o_elem_bits;
                    int hi_small = lo_small + CONFIG_T::unsort_o_elem_bits - 1;
                    data_buffer[i][par][l].range(hi, lo) = data_buffer_row[data_pack_indices][CONFIG_T::num_hashes * CONFIG_T::num_head].range(hi_small, lo_small);
                }
            }
        }
    }
    
    // --- Stage 3: Write final results ---
    BATCHED_INDEX_SELECT_UNSORT_O_WRITE:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            int seq = i * CONFIG_T::par_factor + par;
            for(int l = 0; l < CONFIG_T::dim_per_head; l++) {
                typename CONFIG_T::unsort_logits_tile_wide_t data_row = data_buffer[i][par][l];
                for(int h = 0; h < CONFIG_T::num_head; h++) {
                    typename CONFIG_T::accum_t data_accum = 0;
                    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
                        int lo = (n * CONFIG_T::num_head + h) * CONFIG_T::unsort_logits_elem_bits;
                        int hi = lo + CONFIG_T::unsort_logits_elem_bits - 1;
                        typename CONFIG_T::unsort_logits_elem_bits_t bits = data_row.range(hi, lo);
                        data_accum += *reinterpret_cast<data_T_values*>(&bits);
                    }
                    res_T res_pack = (res_T)data_accum;
                    res_data[par *CONFIG_T::dim_per_head * CONFIG_T::num_head + h*CONFIG_T::dim_per_head+l].write(res_pack);
                }
            }
        }
    }
}

template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_unsort_logits_num_hash(
    hls::stream<data_T_indices> indices[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values[CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<res_T> res_data[ CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // Local buffers in BRAM

    data_T_values data_buffer[CONFIG_T::num_hashes][CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];
    //#pragma HLS BIND_STORAGE variable=data_buffer  type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=3


    data_T_values data_buffer_row[CONFIG_T::num_hashes][CONFIG_T::seq_len];
    #pragma HLS ARRAY_PARTITION variable=data_buffer_row complete dim=2
    //#pragma HLS BIND_STORAGE variable=data_buffer_row  type=ram_1p impl=bram
    // Main computation
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
            // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                #pragma HLS UNROLL
                int seq = i * CONFIG_T::par_factor_sort + par;
                data_T_values data_pack_values = values[n][par].read();
                data_buffer_row[n][seq] = data_pack_values;
            }
        }
    }
    BATCHED_INDEX_SELECT_UNSORT_O_READ_INNER_2:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1 rewind
        // --- Stage 1: Read values into buffer ---
            for(int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                data_T_indices data_pack_indices = indices[n][par].read();
                data_buffer[n][i][par] = data_buffer_row[n][data_pack_indices];
            }
        }
    }
    
    // --- Stage 3: Write final results ---
    typename CONFIG_T::accum_t data_accum[CONFIG_T::seq_len / CONFIG_T::par_factor][CONFIG_T::par_factor];
    #pragma HLS ARRAY_PARTITION variable=data_accum complete dim=2
    BATCHED_INDEX_SELECT_UNSORT_O_WRITE:
    for(int n = 0; n < CONFIG_T::num_hashes; n++) {
        for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1
            for(int par = 0; par < CONFIG_T::par_factor; par++) {
                #pragma HLS UNROLL
                if(n == 0) {
                    data_accum[i][par] = data_buffer[n][i][par];
                } else if (n == CONFIG_T::num_hashes - 1) {
                    res_T res_pack = (res_T)(data_accum[i][par] + data_buffer[n][i][par]);
                    res_data[par].write(res_pack);
                } else {
                    data_accum[i][par] += data_buffer[n][i][par];
                }
            }
        }
    }
}

template<class data_T_indices, class data_T_values, class res_T, typename CONFIG_T>
void batched_index_select_unsort_logits(
    hls::stream<data_T_indices> indices[CONFIG_T::num_head * CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<data_T_values>  values [CONFIG_T::num_head * CONFIG_T::num_hashes][CONFIG_T::par_factor_sort],
    hls::stream<res_T>          res_data[CONFIG_T::num_head * CONFIG_T::par_factor]
) {
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW

    // One wide tile per sequence element, holding (num_hashes*num_head) packed logits
    typename CONFIG_T::unsort_logits_tile_wide_t data_buffer[CONFIG_T::seq_len/CONFIG_T::par_factor][CONFIG_T::par_factor];
    //#pragma HLS BIND_STORAGE variable=data_buffer type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=2
        // Row buffer for this (hash,head)
        typename CONFIG_T::unsort_logits_tile_wide_t data_buffer_row[CONFIG_T::seq_len][CONFIG_T::num_hashes * CONFIG_T::num_head];
        //#pragma HLS BIND_STORAGE variable=data_buffer_row type=ram_1p impl=bram
        #pragma HLS ARRAY_PARTITION variable=data_buffer_row complete dim=1
    #pragma HLS ARRAY_RESHAPE variable=data_buffer_row complete dim=2

    // ==========================
    // Stage 1+2: Read values (pack per seq) -> Scatter by indices into global buffer
    // ==========================
BATCHED_INDEX_SELECT_UNSORT_LOGITS_READ_OUTER:


    BATCHED_INDEX_SELECT_UNSORT_LOGITS_READ_INNER:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1
        for (int nh = 0; nh < CONFIG_T::num_hashes * CONFIG_T::num_head; nh++) {
            #pragma HLS UNROLL factor=CONFIG_T::num_head
            for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;
            
                const int lo = nh * CONFIG_T::unsort_logits_elem_bits;
                const int hi = lo + CONFIG_T::unsort_logits_elem_bits - 1;
            
                data_T_values v = values[nh][par].read();
                data_buffer_row[seq][nh].range(hi, lo)
                    = *reinterpret_cast<typename CONFIG_T::unsort_logits_elem_bits_t *>(&v);
            }
        }
    }

    BATCHED_INDEX_SELECT_UNSORT_LOGITS_READ_INNER_2:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor_sort; i++) {
        #pragma HLS PIPELINE II=1
        for (int nh = 0; nh < CONFIG_T::num_hashes * CONFIG_T::num_head; nh++) {
            #pragma HLS UNROLL factor=CONFIG_T::num_head
            for (int par = 0; par < CONFIG_T::par_factor_sort; par++) {
                int seq = i * CONFIG_T::par_factor_sort + par;

                data_T_indices idx = indices[nh][par].read();

                const int lo_big = nh * CONFIG_T::unsort_logits_elem_bits;
                const int hi_big = lo_big + CONFIG_T::unsort_logits_elem_bits - 1;

                // Scatter: place the (hash,head) slice for the *indexed* row into output row[seq]
                data_buffer[i][par].range(hi_big, lo_big) =
                data_buffer_row[idx][nh].range(hi_big, lo_big);
            }
    }
    }

    // ==========================
    // Stage 3: Reduce across hashes for each head, emit per (par, head)
    // ==========================
BATCHED_INDEX_SELECT_UNSORT_LOGITS_WRITE:
    for (int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1
        for (int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            const int seq = i * CONFIG_T::par_factor + par;

            typename CONFIG_T::unsort_logits_tile_wide_t row = data_buffer[i][par];

            for (int h = 0; h < CONFIG_T::num_head; h++) {
                typename CONFIG_T::accum_t acc = 0;

                // Sum over hashes for fixed head h
                for (int n = 0; n < CONFIG_T::num_hashes; n++) {
                    const int lo = (n * CONFIG_T::num_head + h) * CONFIG_T::unsort_logits_elem_bits;
                    const int hi = lo + CONFIG_T::unsort_logits_elem_bits - 1;

                    typename CONFIG_T::unsort_logits_elem_bits_t bits = row.range(hi, lo);
                    acc += *reinterpret_cast<data_T_values *>(&bits);
                }

                res_T out = (res_T)acc;
                const int out_idx = par * CONFIG_T::num_head + h; // [par][head]
                res_data[out_idx].write(out);
            }
        }
    }
}



template<class data_T_o, class data_T_logits, class res_T, typename CONFIG_T>
void o_divide_logits(
    hls::stream<data_T_o>     o[CONFIG_T::num_head*CONFIG_T::dim_per_head*CONFIG_T::par_factor],
    hls::stream<data_T_logits>     logits[CONFIG_T::num_head*CONFIG_T::par_factor],
    hls::stream<res_T>     res_data[CONFIG_T::num_head*CONFIG_T::dim_per_head*CONFIG_T::par_factor])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    data_T_o data_pack_o;
    data_T_logits data_pack_logits;
    res_T res_pack;

    O_DIVIDE_LOGITS:
    for(int seq = 0; seq < CONFIG_T::seq_len / CONFIG_T::par_factor; seq++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            for(int h = 0; h < CONFIG_T::num_head; h++) {
                data_pack_logits = logits[par*CONFIG_T::num_head+h].read();
                if(data_pack_logits==0) //{
                    data_pack_logits = 0.1;
                    //std::cout << "Warning: logits is zero, set to 0.001 to avoid NaN" << std::endl;
                    //std::cout << data_pack_logits << std::endl;
                //}
                for(int k = 0; k < CONFIG_T::dim_per_head; k++) {
                    data_pack_o = o[par*CONFIG_T::num_head*CONFIG_T::dim_per_head+h*CONFIG_T::dim_per_head+k].read();
                    res_pack = (res_T)(data_pack_o / data_pack_logits);
                    res_data[par*CONFIG_T::num_head*CONFIG_T::dim_per_head+h*CONFIG_T::dim_per_head+k].write(res_pack);
                }
            }
        }
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void out_linear(
    hls::stream<data_T>     data[CONFIG_T::num_head*CONFIG_T::dim_per_head*CONFIG_T::par_factor],
    hls::stream<res_T>     res_data[CONFIG_T::dim_per_head*CONFIG_T::par_factor],
    typename CONFIG_T::weight_t               weight[CONFIG_T::num_head*CONFIG_T::dim_per_head*CONFIG_T::dim_per_head],
    typename CONFIG_T::weight_t                 bias[CONFIG_T::dim_per_head])
{
    #pragma HLS INLINE OFF
    #pragma HLS DATAFLOW
    data_T data_pack;
    res_T res_pack;

    OUT_LINEAR_CAL:
    for(int i = 0; i < CONFIG_T::seq_len / CONFIG_T::par_factor; i++) {
        #pragma HLS PIPELINE II=1 rewind
        for(int par = 0; par < CONFIG_T::par_factor; par++) {
            #pragma HLS UNROLL
            data_T data_buffer[CONFIG_T::dim_per_head*CONFIG_T::num_head];
            #pragma HLS ARRAY_PARTITION variable=data_buffer complete dim=1
            typename CONFIG_T::accum_t row_buffer[CONFIG_T::dim_per_head];
            #pragma HLS ARRAY_PARTITION variable=row_buffer complete dim=1
            for(int j = 0; j < CONFIG_T::dim_per_head; j++) {
                row_buffer[j] = bias[j];
            }
            for(int j = 0; j < CONFIG_T::dim_per_head*CONFIG_T::num_head; j++) {
                data_pack = data[par * CONFIG_T::num_head*CONFIG_T::dim_per_head + j].read();
                data_buffer[j] = data_pack;
            }
            for(int j = 0; j < CONFIG_T::dim_per_head; j++) {
                for(int k = 0; k < CONFIG_T::dim_per_head*CONFIG_T::num_head; k++) {
                    //if(i==0)
                        // std::cout << "data_buffer[" << i << "][" << k << "] = " << data_buffer[i][k] << " weight[" << j*CONFIG_T::num_head*CONFIG_T::dim_per_head+k << "] = " << weight[j*CONFIG_T::num_head*CONFIG_T::dim_per_head+k] << std::endl;
                    row_buffer[j] += data_buffer[k]*weight[j*CONFIG_T::num_head*CONFIG_T::dim_per_head+k];
                    //if(i==0)
                        // std::cout << "row_buffer[" << j << "] = " << row_buffer[j] << std::endl;
                }
            }
            for(int j = 0; j < CONFIG_T::dim_per_head; j++) {
                res_pack = (res_T)row_buffer[j];
                res_data[par*CONFIG_T::dim_per_head + j].write(res_pack);
            }
        }
    }
}

}

#endif
