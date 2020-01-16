//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_GARNET_H_
#define NNET_GARNET_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include "hls_math.h"

namespace nnet {

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::accum_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_bitwidth> index_t;
  typedef ap_fixed<CONFIG_T::distance_bitwidth, CONFIG_T::distance_bitwidth / 2, AP_RND, AP_SAT> rdistance_t;

  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);

  index_t index;
  rdistance_t rdist;
  typename CONFIG_T::accum_t distance;
  
  for (unsigned iw = 0; iw < table_size; ++iw) {
    index = iw;
    rdist.range(CONFIG_T::distance_bitwidth - 1, 0) = index.range(CONFIG_T::distance_bitwidth - 1, 0);
    distance = rdist;
    edge_weights_table[iw] = hls::pow(2., -distance);
  }
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::accum_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);
  double const step = 64. / table_size;

  typename CONFIG_T::accum_t v = -32.;
  for (unsigned iw = 0; iw < table_size; ++iw) {
    edge_weights_table[iw] = std::pow(2., -v);
    v += step;
  }
}

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::accum_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::accum_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_bitwidth> index_t;
  typedef ap_fixed<CONFIG_T::distance_bitwidth, CONFIG_T::distance_bitwidth / 2, AP_RND, AP_SAT> rdistance_t;

  index_t index;
  rdistance_t rdist = distance;

  index.range(CONFIG_T::distance_bitwidth - 1, 0) = rdist.range(CONFIG_T::distance_bitwidth - 1, 0);

  return edge_weights_table[index];
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::accum_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::accum_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_bitwidth);
  double const step = 64. / table_size;
  
  int index = (distance + 32.) / step;
  if (index < 0)
    index = 0;
  else if (index >= table_size)
    index = table_size - 1;

  return edge_weights_table[index];
}

template<class CONFIG_T>
inline typename CONFIG_T::edge_weight_t
compute_garnet_edge_weight(typename CONFIG_T::accum_t distance)
{
  #pragma HLS PIPELINE

#ifdef __SYNTHESIS__
  typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_bitwidth];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * CONFIG_T::n_in_features * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights_table cyclic factor=reshape_factor dim=1
  bool initialized = false;
#else
  static typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_bitwidth];
  static bool initialized = false;
#endif
  if (!initialized) {
    initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
    initialized = true;
  }

  return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
}

template<class nvtx_T, class CONFIG_T>
inline void
set_igraph_single(nvtx_T nvtx, typename CONFIG_T::ngrph_t igraph[CONFIG_T::n_vertices])
{
 Vertices:
  for (nvtx_T iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE
    
    if (iv < nvtx)
      igraph[iv] = 0;
    else
      igraph[iv] = -1;
  }
}

template<class CONFIG_T>
inline void
initialize_nvtx(typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices], typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs])
{
 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
    #pragma HLS UNROLL
    nvtx[ic] = 0;
  }
}

template<class CONFIG_T>
inline void
initialize_edge_weight_sums(typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators])
{
  #pragma HLS PIPELINE
 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
      edge_weight_sums[ic * CONFIG_T::n_aggregators + ia] = 0.;
  }
}

template<class CONFIG_T>
inline void
normalize_sums(
  typename CONFIG_T::nvtx_t const nvtx[CONFIG_T::n_graphs],
  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators]
)
{
  #pragma HLS PIPELINE
 Graphs:
  for (unsigned ie = 0; ie < CONFIG_T::n_graphs; ++ie) {
    typename CONFIG_T::accum_t const nvtx_norm = 1. / nvtx[ie];

   Aggregators2:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      edge_weight_sums[ie * CONFIG_T::n_aggregators + ia] *= nvtx_norm;

     InFeatures3:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        weighted_features_sums[(ie * CONFIG_T::n_in_features + ix) * CONFIG_T::n_aggregators + ia] *= nvtx_norm;
      }
    }
  }
}

template<class data_T, class CONFIG_T>
void
compute_edges_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs];
  #pragma HLS ARRAY_RESHAPE variable=nvtx complete

  initialize_nvtx<CONFIG_T>(nvtx);
  initialize_edge_weight_sums<CONFIG_T>(edge_weight_sums);
  initialize_weighted_features_sums<CONFIG_T>(weighted_features_sums);

 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t igr = igraph[iv];
    
    if (igr == -1)
      continue;

    nvtx[igr] += 1;
    
    typename CONFIG_T::index_t const data_offset = iv * CONFIG_T::n_in_features;
    typename CONFIG_T::index_t const weights_offset = iv * CONFIG_T::n_aggregators;
    typename CONFIG_T::index_t const edge_weight_sum_offset = igr * CONFIG_T::n_aggregators;
    typename CONFIG_T::index_t const features_sum_offset = igr * CONFIG_T::n_in_features * CONFIG_T::n_aggregators;

    // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first

   Aggregators1:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::accum_t distance = aggregator_distance_biases[ia];
     InFeatures1:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        distance += data[data_offset + ix] * aggregator_distance_weights[ix * CONFIG_T::n_aggregators + ia];

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      edge_weights[weights_offset + ia] = edge_weight;
      edge_weight_sums[edge_weight_sum_offset + ia] += edge_weight;

     InFeatures2:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        weighted_features_sums[features_sum_offset + ix * CONFIG_T::n_aggregators + ia] += data[data_offset + ix] * edge_weight;
    }
  }

  normalize_sums<CONFIG_T>(nvtx, edge_weight_sums, weighted_features_sums);
}

template<class data_T, class CONFIG_T>
void
compute_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs];
  #pragma HLS ARRAY_RESHAPE variable=nvtx complete

  initialize_nvtx<CONFIG_T>(nvtx);
  initialize_edge_weight_sums<CONFIG_T>(edge_weight_sums);
  initialize_weighted_features_sums<CONFIG_T>(weighted_features_sums);

 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t igr = igraph[iv];
    
    if (igr == -1)
      continue;

    nvtx[igr] += 1;
    
    typename CONFIG_T::index_t const data_offset = iv * CONFIG_T::n_in_features;
    typename CONFIG_T::index_t const weights_offset = iv * CONFIG_T::n_aggregators;
    typename CONFIG_T::index_t const edge_weight_sum_offset = igr * CONFIG_T::n_aggregators;
    typename CONFIG_T::index_t const features_sum_offset = igr * CONFIG_T::n_in_features * CONFIG_T::n_aggregators;

    // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first

   Aggregators1:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::accum_t distance = aggregator_distance_biases[ia];
     InFeatures1:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        distance += data[data_offset + ix] * aggregator_distance_weights[ix * CONFIG_T::n_aggregators + ia];

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      edge_weight_sums[edge_weight_sum_offset + ia] += edge_weight;

     InFeatures2:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        weighted_features_sums[features_sum_offset + ix * CONFIG_T::n_aggregators + ia] += data[data_offset + ix] * edge_weight;
    }
  }

  normalize_sums<CONFIG_T>(nvtx, edge_weight_sums, weighted_features_sums);
}

template<class CONFIG_T>
void
transform_aggregated(
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ie = 0; ie < CONFIG_T::n_graphs; ++ie) {
   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const bias_index = ie * CONFIG_T::n_out_features + io;

      aggregated_biases[bias_index] = output_transform_biases[io];

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const weight_index = ia * CONFIG_T::n_out_features + io;

        aggregated_biases[bias_index] += input_transform_biases[weight_index] * edge_weight_sums[ie * CONFIG_T::n_aggregators + ia];
        aggregated_weights[weight_index] = 0.;

       InFeatures:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          aggregated_weights[weight_index] +=
            input_transform_weights[(ia * CONFIG_T::n_in_features + ix) * CONFIG_T::n_out_features + io] *
            weighted_features_sums[(ie * CONFIG_T::n_in_features + ix) * CONFIG_T::n_aggregators + ia];
        }
      }
    }
  }
}

template<class data_T, class CONFIG_T>
void
process_input_vertices(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  compute_edges_aggregates(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weights,
    edge_weight_sums,
    weighted_features_sums
  );

  transform_aggregated<CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    aggregated_weights,
    aggregated_biases
  );
}

template<class data_T, class CONFIG_T>
void
process_input_vertices_collapse(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  // This is a near-identical copy of process_input_vertices, just without the per-vertex weights

  #pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  compute_aggregates(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weight_sums,
    weighted_features_sums
  );

  transform_aggregated<CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    aggregated_weights,
    aggregated_biases
  );
}


template<class res_T, class CONFIG_T>
void
set_output_vertices(
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
)
{
 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t igr = igraph[iv];
    
    if (igr == -1)
      break;

    typename CONFIG_T::index_t const weights_offset = igr * CONFIG_T::n_aggregators * CONFIG_T::n_out_features;
    typename CONFIG_T::index_t const bias_offset = igr * CONFIG_T::n_out_features;

   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      res_T acc = aggregated_biases[bias_offset + io];

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
        acc += edge_weights[iv * CONFIG_T::n_aggregators + ia] * aggregated_weights[weights_offset + ia * CONFIG_T::n_out_features + io];

      res[iv * CONFIG_T::n_out_features + io] = acc;
    }
  }
}

template<class res_T, class CONFIG_T>
void
set_output_mean(
  typename CONFIG_T::nvtx_t const nvtx[CONFIG_T::n_graphs],
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ie = 0; ie < CONFIG_T::n_graphs; ++ie) {
    typename CONFIG_T::accum_t nvtx_norm = 1. / nvtx[ie];

   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      res_T acc = aggregated_biases[ie * CONFIG_T::n_out_features + io];

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
        acc += edge_weight_sums[ie * CONFIG_T::n_aggregators + ia] * aggregated_weights[(ie * CONFIG_T::n_aggregators + ia) * CONFIG_T::n_out_features + io];
      acc *= nvtx_norm;

      res[ie * CONFIG_T::n_out_features + io] = acc;
    }
  }
}

template<class res_T, class CONFIG_T>
void
set_output_sum(
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ie = 0; ie < CONFIG_T::n_graphs; ++ie) {

   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      res_T acc = aggregated_biases[ie * CONFIG_T::n_out_features + io];

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
        acc += edge_weight_sums[ie * CONFIG_T::n_aggregators + ia] * aggregated_weights[(ie * CONFIG_T::n_aggregators + ia) * CONFIG_T::n_out_features + io];
  
      res[ie * CONFIG_T::n_out_features + io] = acc;
    }
  }
}

struct garnet_config
{
  // Internal data type definitions
  typedef float input_transform_weights_t;
  typedef float input_transform_biases_t;
  //typedef float output_transform_weights_t;
  typedef float output_transform_biases_t;
  typedef float aggregator_distance_weights_t;
  typedef float aggregator_distance_biases_t;

  typedef float accum_t;
  typedef ap_ufixed<64, 32> edge_weight_t;
  typedef ap_fixed<64, 24> aggr_t;

  typedef unsigned short index_t;

  // Type wide enough to accommodate n_vertices - can be determined programmatically in principle
  typedef ap_int<8> nvtx_t;
  // Type wide enough to accommodate n_graphs - can be determined programmatically in principle
  typedef ap_int<4> ngrph_t;

  // Layer specs
  static const unsigned n_vertices = 256;
  static const unsigned n_graphs = 16; // maximum number of events that may be packed in the input array
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_out_features = 4;
  static const unsigned distance_bitwidth = 10;

  enum CollapseType {
    collapse_mean,
    collapse_sum,
    no_collapse
  };
  static const unsigned collapse_type = no_collapse;

  // Optimization specs
  static const unsigned reuse_factor = 64;
};

template<class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type == CONFIG_T::no_collapse>::type
garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights cyclic factor=reshape_factor dim=1

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_biases complete

  process_input_vertices<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weights,
    aggregated_weights,
    aggregated_biases
  );

  set_output_vertices<res_T, CONFIG_T>(
    igraph,
    edge_weights,
    aggregated_weights,
    aggregated_biases,
    res    
  );
}

template<class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type != CONFIG_T::no_collapse>::type
garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
    res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete dim=1

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_biases complete

  process_input_vertices_collapse<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weight_sums,
    aggregated_weights,
    aggregated_biases
  );

  switch (CONFIG_T::collapse_type) {
  case CONFIG_T::collapse_mean:
    set_output_mean<res_T, CONFIG_T>(
      edge_weight_sums,
      aggregated_weights,
      aggregated_biases,
      res
    );
    break;

  /* case CONFIG_T::collapse_sum: */
  /*   set_output_sum<res_T, CONFIG_T>( */
  /*     edge_weight_sums, */
  /*     aggregated_weights, */
  /*     aggregated_biases, */
  /*     res */
  /*   ); */
  /*   break; */

  default:
    break;
  }
}

// TODO can just set res to res_T* and use the same function?
template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type == CONFIG_T::no_collapse>::type
garnet_single(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx_sample[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::ngrph_t igraph[CONFIG_T::n_vertices];
  unsigned const reshape_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
  #pragma HLS ARRAY_RESHAPE variable=igraph cyclic factor=reshape_factor

  set_igraph_single<nvtx_T, CONFIG_T>(nvtx_sample[0], igraph);

  garnet<data_T, res_T, CONFIG_T>(
    data,
    igraph,
    res,
    input_transform_weights,
    input_transform_biases,
    aggregator_distance_weights,
    aggregator_distance_biases,
    output_transform_biases
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type != CONFIG_T::no_collapse>::type
garnet_single(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx_sample[1],
    res_T res[CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::ngrph_t igraph[CONFIG_T::n_vertices];
  unsigned const reshape_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
  #pragma HLS ARRAY_RESHAPE variable=igraph cyclic factor=reshape_factor

  set_igraph_single<nvtx_T, CONFIG_T>(nvtx_sample[0], igraph);

  garnet<data_T, res_T, CONFIG_T>(
    data,
    igraph,
    res,
    input_transform_weights,
    input_transform_biases,
    aggregator_distance_weights,
    aggregator_distance_biases,
    output_transform_biases
  );
}


}

#endif
