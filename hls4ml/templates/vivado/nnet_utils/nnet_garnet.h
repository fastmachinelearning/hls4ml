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

template<class data_T, class nvtx_T, class CONFIG_T, unsigned COLLAPSE>
void
process_input_vertices(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T nvtx,
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t* edge_weights, //[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
)
{
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    #pragma HLS UNROLL
    edge_weight_sums[ia] = 0.;
  }

 VerticesFirstPass:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE
    
    if (iv >= nvtx)
      break;
    
    typename CONFIG_T::index_t const data_offset = iv * CONFIG_T::n_in_features;
    typename CONFIG_T::index_t const weights_offset = iv * CONFIG_T::n_aggregators;

    // keras Dense applies weights as K.dot(inputs, kernel) -> kernel is channels first

  Accum:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::accum_t distance = aggregator_distance_biases[ia];
    AccumMatMul:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        distance += data[data_offset + ix] * aggregator_distance_weights[ix * CONFIG_T::n_aggregators + ia];

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      if (COLLAPSE == CONFIG_T::no_collapse)
        edge_weights[weights_offset + ia] = edge_weight;

      edge_weight_sums[ia] += edge_weight;

    WeightedFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        weighted_features_sums[ix * CONFIG_T::n_aggregators + ia] += data[data_offset + ix] * edge_weight;
    }
  }
}

template<class nvtx_T, class CONFIG_T>
void
transform_aggregated(
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const weighted_features_sums[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  nvtx_T nvtx,
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE
  
  typename CONFIG_T::accum_t nvtx_norm = 1. / nvtx;
  
  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {

    aggregated_biases[io] = 0.;

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const weight_index = ia * CONFIG_T::n_out_features + io;

      aggregated_biases[io] += input_transform_biases[weight_index] * edge_weight_sums[ia];

      aggregated_weights[weight_index] = 0.;

      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix)
        aggregated_weights[weight_index] +=
          input_transform_weights[(ia * CONFIG_T::n_in_features + ix) * CONFIG_T::n_out_features + io] *
          weighted_features_sums[ix * CONFIG_T::n_aggregators + ia];

      aggregated_weights[weight_index] *= nvtx_norm;
    }

    aggregated_biases[io] *= nvtx_norm;
    aggregated_biases[io] += output_transform_biases[io];
  }
}

template<class nvtx_T, class res_T, class CONFIG_T>
void
set_output_vertices(
  nvtx_T nvtx,
  typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
)
{
 VerticesSecondPass:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE
    
    if (iv >= nvtx)
      break;

    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const res_index = iv * CONFIG_T::n_out_features + io;

      res[res_index] = aggregated_biases[io];

      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
        res[res_index] += edge_weights[iv * CONFIG_T::n_aggregators + ia] * aggregated_weights[ia * CONFIG_T::n_out_features + io];
    }
  }
}

template<class nvtx_T, class res_T, class CONFIG_T>
void
set_output_mean(
  nvtx_T nvtx,
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

  typename CONFIG_T::accum_t nvtx_norm = 1. / nvtx;  

 Collapse:
  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    res[io] = aggregated_biases[io];

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
      res[io] += edge_weight_sums[ia] * aggregated_weights[ia * CONFIG_T::n_out_features + io];

    res[io] *= nvtx_norm;
  }
}

template<class res_T, class CONFIG_T>
void
set_output_sum(
  typename CONFIG_T::edge_weight_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const aggregated_biases[CONFIG_T::n_out_features],
  res_T res[CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 Collapse:
  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    res[io] = aggregated_biases[io];

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia)
      res[io] += edge_weight_sums[ia] * aggregated_weights[ia * CONFIG_T::n_out_features + io];
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

  // Layer specs
  static const unsigned n_vertices = 256;
  static const unsigned n_in_features = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_out_features = 4;
  static const unsigned distance_bitwidth = 10;

  // Optimization specs
  static const unsigned reuse_factor = 64;

  enum CollapseType {
    collapse_mean,
    collapse_sum,
    no_collapse
  };
};

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
void garnet_passthrough(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
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

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_in_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  nvtx_T nvtx_local = nvtx[0];

  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete dim=1

  process_input_vertices<data_T, nvtx_T, CONFIG_T, CONFIG_T::no_collapse>(
    data,
    nvtx[0],
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weights,
    edge_weight_sums,
    weighted_features_sums
  );

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_biases complete

  transform_aggregated<nvtx_T, CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    output_transform_biases,
    nvtx_local,
    aggregated_weights,
    aggregated_biases
  );

  set_output_vertices<nvtx_T, res_T, CONFIG_T>(
    nvtx_local,
    edge_weights,
    aggregated_weights,
    aggregated_biases,
    res    
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T, unsigned COLLAPSE>
void garnet_collapse(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_aggregators * CONFIG_T::n_out_features],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_in_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_in_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  nvtx_T nvtx_local = nvtx[0];

  typename CONFIG_T::edge_weight_t edge_weight_sums[CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete dim=1

  process_input_vertices<data_T, nvtx_T, CONFIG_T, COLLAPSE>(
    data,
    nvtx[0],
    aggregator_distance_weights,
    aggregator_distance_biases,
    nullptr,
    edge_weight_sums,
    weighted_features_sums
  );

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_aggregators * CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete
  typename CONFIG_T::aggr_t aggregated_biases[CONFIG_T::n_out_features];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_biases complete

  transform_aggregated<nvtx_T, CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    output_transform_biases,
    nvtx_local,
    aggregated_weights,
    aggregated_biases
  );

  switch (COLLAPSE) {
  case CONFIG_T::collapse_mean:
    set_output_mean<nvtx_T, res_T, CONFIG_T>(
      nvtx_local,
      edge_weight_sums,
      aggregated_weights,
      aggregated_biases,
      res
    );
    break;

  case CONFIG_T::collapse_sum:
}

}

#endif
