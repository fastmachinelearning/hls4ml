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

//#define GARNET_NVERT_MEAN 1

namespace nnet {

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_width> index_t;

  unsigned const table_size = (1 << CONFIG_T::distance_width);

  index_t index;
  typename CONFIG_T::distance_t distance;

  for (unsigned iw = 0; iw < table_size; ++iw) {
    index = iw;
    distance.range(CONFIG_T::distance_width - 1, 0) = index.range(CONFIG_T::distance_width - 1, 0);
    edge_weights_table[iw] = hls::pow(2., -distance * distance);
  }
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value>::type
initialize_edge_weights_table(typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_width);
  double const step = 64. / table_size;

  typename CONFIG_T::distance_t v = -32.;
  for (unsigned iw = 0; iw < table_size; ++iw) {
    edge_weights_table[iw] = std::pow(2., -v * v);
    v += step;
  }
}

template<class CONFIG_T>
inline typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::distance_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  typedef ap_uint<CONFIG_T::distance_width> index_t;

  index_t index(distance.range(CONFIG_T::distance_width - 1, 0));

  return edge_weights_table[index];
}

template<class CONFIG_T>
inline typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
get_edge_weight(typename CONFIG_T::distance_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
{
  unsigned const table_size = (1 << CONFIG_T::distance_width);
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
compute_garnet_edge_weight(typename CONFIG_T::distance_t distance)
{
  #pragma HLS PIPELINE

#ifdef __SYNTHESIS__
  typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
  // unsigned const reshape_factor = CONFIG_T::n_aggregators * CONFIG_T::n_in_features * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  // #pragma HLS ARRAY_RESHAPE variable=edge_weights_table cyclic factor=reshape_factor dim=1
  bool initialized = false;
#else
  static typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
  static bool initialized = false;
#endif
  if (!initialized) {
    initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
    initialized = true;
  }

  return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
}

template<class CONFIG_T>
inline void
initialize_sums(
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs],
#endif
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
#ifdef GARNET_NVERT_MEAN
    nvtx[ic] = 0;
#endif

   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

      edge_weight_sums[ica] = 0.;

     InFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;

        weighted_features_sums[icax] = 0.;
      }
    }
  }
}

template<class CONFIG_T>
inline void
initialize_sums_single(
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
  #pragma HLS PIPELINE

 Aggregators:
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    edge_weight_sums[ia] = 0.;

   InFeatures:
    for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
      typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

      weighted_features_sums[iax] = 0.;
    }
  }
}

template<class data_T, class CONFIG_T>
void
do_compute_edges_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::ngrph_t igraph_copy[CONFIG_T::n_vertices],
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs],
#endif
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t ic = igraph[iv];
    igraph_copy[iv] = ic;

#ifdef GARNET_NVERT_MEAN
    if (ic != -1)
      nvtx[ic] += 1;
#endif

   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;
      typename CONFIG_T::edge_weight_t edge_weight;

      if (ic == -1) {
        edge_weight = 0.;
      }
      else {
        typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];

      InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
          typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

          distance += data[ivx] * aggregator_distance_weights[iax];
        }

        edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

        typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

        edge_weight_sums[ica] += edge_weight;

      InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;
          typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;

          weighted_features_sums[icax] += data[ivx] * edge_weight;
        }
      }

      edge_weights[iva] = edge_weight;
    }
  }
}

template<class data_T, class CONFIG_T>
void
do_compute_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs],
#endif
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t ic = igraph[iv];

    if (ic == -1)
      continue;

#ifdef GARNET_NVERT_MEAN
    nvtx[ic] += 1;
#endif

   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

      typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];

     InFeatures1:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        distance += data[ivx] * aggregator_distance_weights[iax];
      }

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

      edge_weight_sums[ica] += edge_weight;

     InFeatures2:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;

        weighted_features_sums[icax] += data[ivx] * edge_weight;
      }
    }
  }
}

template<class CONFIG_T>
inline void
normalize_sums(
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t const nvtx[CONFIG_T::n_graphs],
#endif
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
  #pragma HLS PIPELINE
 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
#ifdef GARNET_NVERT_MEAN
    typename CONFIG_T::nvtx_t nv = nvtx[ic];

    if (nv == 0) {
     Aggregators1:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

        edge_weight_sums[ica] = 0.;

       InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;

          weighted_features_sums[icax] = 0.;
        }
      }
    }
    else {
      typename CONFIG_T::norm_t const nvtx_norm = 1. / nv;

     Aggregators2:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

        edge_weight_sums[ica] *= nvtx_norm;

       InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;

          weighted_features_sums[icax] *= nvtx_norm;
        }
      }
    }
#else

   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;

      edge_weight_sums[ica] >>= CONFIG_T::n_vertices_width;

     InFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;

        weighted_features_sums[icax] >>= CONFIG_T::n_vertices_width;
      }
    }
#endif
  }
}

#ifdef GARNET_NVERT_MEAN
template<class nvtx_T, class CONFIG_T>
#else
template<class CONFIG_T>
#endif
inline void
normalize_sums_single(
#ifdef GARNET_NVERT_MEAN
  nvtx_T const nvtx,
#endif
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
  #pragma HLS PIPELINE

#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::norm_t const nvtx_norm = 1. / nvtx;
#endif

 Aggregators:
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
#ifdef GARNET_NVERT_MEAN
    edge_weight_sums[ia] *= nvtx_norm;
#else
    edge_weight_sums[ia] >>= CONFIG_T::n_vertices_width;
#endif

   InFeatures:
    for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
      typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

#ifdef GARNET_NVERT_MEAN
      weighted_features_sums[iax] *= nvtx_norm;
#else
      weighted_features_sums[iax] >>= CONFIG_T::n_vertices_width;
#endif
    }
  }
}

template<class data_T, class CONFIG_T>
void
compute_edges_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::ngrph_t igraph_copy[CONFIG_T::n_vertices],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
)
{
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs];
  #pragma HLS ARRAY_RESHAPE variable=nvtx complete
#endif

  initialize_sums<CONFIG_T>(
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weight_sums,
    weighted_features_sums
  );

  do_compute_edges_aggregates<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    igraph_copy,
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weights,
    edge_weight_sums,
    weighted_features_sums
  );

  normalize_sums<CONFIG_T>(
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weight_sums,
    weighted_features_sums
  );
}

template<class data_T, class nvtx_T, class CONFIG_T>
void
compute_edges_aggregates_single(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx,
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  nvtx_T& nvtx_copy
)
{
  //  initialize_sums_single<CONFIG_T>(edge_weight_sums, weighted_features_sums, );

 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE II=4

    if (iv == nvtx)
      break;

   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;

      typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];

     InFeatures1:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        distance += data[ivx] * aggregator_distance_weights[iax];
      }

      typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

#ifdef GARNET_NVERT_MEAN
      edge_weight_sums[ia] += edge_weight;
#else
      edge_weight_sums[ia] += (edge_weight >> CONFIG_T::n_vertices_width);
#endif

     InFeatures2:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;

#ifdef GARNET_NVERT_MEAN
        weighted_features_sums[iax] += data[ivx] * edge_weight;
#else
        weighted_features_sums[iax] += data[ivx] * (edge_weight >> CONFIG_T::n_vertices_width);
#endif
      }

      edge_weights[iva] = edge_weight;
    }
  }

#ifdef GARNET_NVERT_MEAN
  normalize_sums_single<nvtx_T, CONFIG_T>(nvtx, edge_weight_sums, weighted_features_sums);
#else
  normalize_sums_single<CONFIG_T>(edge_weight_sums, weighted_features_sums);
#endif

  nvtx_copy = nvtx;
}

template<class data_T, class CONFIG_T>
void
compute_aggregates(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_in_features * CONFIG_T::n_aggregators]
)
{
#ifdef GARNET_NVERT_MEAN
  typename CONFIG_T::nvtx_t nvtx[CONFIG_T::n_graphs];
  #pragma HLS ARRAY_RESHAPE variable=nvtx complete
#endif

  initialize_sums<CONFIG_T>(
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weight_sums,
    weighted_features_sums
  );

  do_compute_aggregates<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weight_sums,
    weighted_features_sums
  );

  normalize_sums<CONFIG_T>(
#ifdef GARNET_NVERT_MEAN
    nvtx,
#endif
    edge_weight_sums,
    weighted_features_sums
  );
}

template<class data_T, class nvtx_T, class CONFIG_T>
void
compute_aggregates_single(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx,
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_in_features * CONFIG_T::n_aggregators]
)
{
  initialize_sums_single<CONFIG_T>(edge_weight_sums, weighted_features_sums);

  unsigned unroll_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
  
 Vertices:
  for (unsigned ivv = 0; ivv < unroll_factor; ++ivv) {
    #pragma HLS UNROLL skip_exit_check
    #pragma HLS PIPELINE II=8

    typename CONFIG_T::aggr_t edge_weight_sums_local[CONFIG_T::n_aggregators];
    typename CONFIG_T::aggr_t weighted_features_sums_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      edge_weight_sums_local[ia] = 0.;
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;
        weighted_features_sums_local[iax] = 0.;
      }
    }
    
    for (unsigned ir = 0; ir < CONFIG_T::reuse_factor; ++ir) {
      //#pragma HLS UNROLL factor=unroll_factor skip_exit_check
    // II will depend on the precision of data types - revisit
      unsigned iv = ivv * CONFIG_T::reuse_factor + ir;


      if (iv >= nvtx)
        break;

    Aggregators1:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        //typename CONFIG_T::aggr_t distance = aggregator_distance_biases[ia];
        typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];

      InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
          typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

          distance += data[ivx] * aggregator_distance_weights[iax];
        }

        typename CONFIG_T::edge_weight_t edge_weight = compute_garnet_edge_weight<CONFIG_T>(distance);

        edge_weight_sums_local[ia] += edge_weight;

      InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;
          typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;

          weighted_features_sums_local[iax] += data[ivx] * edge_weight;
        }
      }
    }

    #ifndef GARNET_NVERT_MEAN
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      edge_weight_sums[ia] += edge_weight_sums_local[ia] / CONFIG_T::reuse_factor;
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;
        weighted_features_sums[iax] += weighted_features_sums_local[iax] / CONFIG_T::reuse_factor;
      }
    }
    #endif
  }

#ifdef GARNET_NVERT_MEAN
  normalize_sums_single<nvtx_T, CONFIG_T>(nvtx, edge_weight_sums, weighted_features_sums);
#else
  //normalize_sums_single<CONFIG_T>(edge_weight_sums, weighted_features_sums);
  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    edge_weight_sums[ia] /= unroll_factor;
    for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
      typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;
      weighted_features_sums[iax] /= unroll_factor;
    }
  }
#endif
}

template<class CONFIG_T>
void
transform_aggregates(
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ico = ic * CONFIG_T::n_out_features + io;
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const icoa = ico * CONFIG_T::n_aggregators + ia;

        aggregated_weights[icoa] = edge_weight_sums[ica] * input_transform_biases[ioa];

       InFeatures:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          typename CONFIG_T::index_t const ioax = ioa * CONFIG_T::n_in_features + ix;
          typename CONFIG_T::index_t const icax = ica * CONFIG_T::n_in_features + ix;

          aggregated_weights[icoa] += weighted_features_sums[icax] * input_transform_weights[ioax];
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
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::ngrph_t igraph_copy[CONFIG_T::n_vertices],
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators]
)
{
  //#pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  compute_edges_aggregates<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    igraph_copy,
    edge_weights,
    edge_weight_sums,
    weighted_features_sums
  );

  transform_aggregates<CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    aggregated_weights
  );
}

template<class data_T, class CONFIG_T>
void
process_input_vertices_collapse(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators]
)
{
  // This is a near-identical copy of process_input_vertices, just without the per-vertex weights

  //#pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  compute_aggregates<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weight_sums,
    weighted_features_sums
  );

  transform_aggregates<CONFIG_T>(
    input_transform_weights,
    input_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    aggregated_weights
  );
}

template<class res_T, class CONFIG_T>
void
set_output_vertices(
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
)
{
 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    typename CONFIG_T::ngrph_t ic = igraph[iv];

    // copy the edge weights for the vertex to registers so that we can
    // read from all io in parallel
    typename CONFIG_T::edge_weight_t vertex_edge_weights[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=vertex_edge_weights complete

   Aggregators1:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;

      vertex_edge_weights[ia] = edge_weights[iva];
    }

   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ivo = iv * CONFIG_T::n_out_features + io;
      res_T acc;

      if (ic == -1) {
        acc = 0.;
      }
      else {
        typename CONFIG_T::index_t const ico = ic * CONFIG_T::n_out_features + io;

        acc = output_transform_biases[io];

       Aggregators2:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          typename CONFIG_T::index_t const icoa = ico * CONFIG_T::n_aggregators + ia;

          acc += vertex_edge_weights[ia] * aggregated_weights[icoa];
        }
      }

      res[ivo] = acc;
    }
  }
}

template<class nvtx_T, class res_T, class CONFIG_T>
void
set_vertex_output_single(
  nvtx_T const nvtx,
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
)
{
  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

 OutFeatures1:
  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    #pragma HLS UNROLL
   Aggregators1:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      #pragma HLS UNROLL
      typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;

      aggregated_weights[ioa] = edge_weight_sums[ia] * input_transform_biases[ioa];

     InFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        #pragma HLS UNROLL
        typename CONFIG_T::index_t const ioax = ioa * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        aggregated_weights[ioa] += weighted_features_sums[iax] * input_transform_weights[ioax];
      }
    }
  }
  
 Vertices:
  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    unsigned cycle_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
    #pragma HLS UNROLL factor=cycle_factor skip_exit_check
    #pragma HLS PIPELINE

    if (iv == nvtx)
      break;

   //  // copy the edge weights for the vertex to registers so that we can
   //  // read from all io in parallel
   //  typename CONFIG_T::edge_weight_t vertex_edge_weights[CONFIG_T::n_aggregators];
   //  #pragma HLS ARRAY_RESHAPE variable=vertex_edge_weights complete

   // Aggregators1:
   //  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
   //    typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;

   //    vertex_edge_weights[ia] = edge_weights[iva];
   //  }

   OutFeatures2:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ivo = iv * CONFIG_T::n_out_features + io;

      res_T acc = output_transform_biases[io];

    Aggregators2:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;

        //acc += vertex_edge_weights[ia] * aggregated_weights[ioa];
        acc += edge_weights[iva] * aggregated_weights[ioa];
      }

      res[ivo] = acc;
    }
  }
}

template<class res_T, class CONFIG_T>
void
set_output_mean(
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 Graphs:
  for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
    // For unused graphs, aggregated_biases, edge_weight_sums, and aggregated_weights are all 0
    // The logic is pipelined so the circuits will be written and run; there is nothing to gain from continuing based on nvtx being 0 etc.

   OutFeatures:
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ico = ic * CONFIG_T::n_out_features + io;

      res_T acc = output_transform_biases[io];

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const ica = ic * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const icoa = ico * CONFIG_T::n_aggregators + ia;

        acc += edge_weight_sums[ica] * aggregated_weights[icoa];
      }

      res[ico] = acc;
    }
  }
}

#ifdef GARNET_NVERT_MEAN
template<class res_T, class CONFIG_T>
#else
template<class nvtx_T, class res_T, class CONFIG_T>
#endif
void
set_aggregate_output_single(
#ifndef GARNET_NVERT_MEAN
  nvtx_T const nvtx,
#endif
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
  typename CONFIG_T::aggr_t const edge_weight_sums[CONFIG_T::n_aggregators],
  typename CONFIG_T::aggr_t const weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  res_T res[CONFIG_T::n_out_features]
)
{
  #pragma HLS PIPELINE

 OutFeatures:
  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    typename CONFIG_T::aggr_t acc = output_transform_biases[io];
#ifndef GARNET_NVERT_MEAN
    acc *= nvtx;
    acc >>= CONFIG_T::n_vertices_width;
#endif
    
   Aggregators:
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;

      typename CONFIG_T::aggr_t aggregated_weight = edge_weight_sums[ia] * input_transform_biases[ioa];

     InFeatures:
      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ioax = ioa * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        aggregated_weight += weighted_features_sums[iax] * input_transform_weights[ioax];
      }

      acc += edge_weight_sums[ia] * aggregated_weight;
    }

    res[io] = acc;
  }
}


struct garnet_config
{
  // Layer specs
  static const unsigned n_vertices_width = 8;
  static const unsigned n_vertices = (1 << n_vertices_width);
  static const unsigned n_graphs = 4; // maximum number of events that may be packed in the input array
  static const unsigned n_in_features = 4;
  static const unsigned n_propagate = 4;
  static const unsigned n_aggregators = 4;
  static const unsigned n_out_features = 4;
  static const unsigned distance_width = 12;

  // Internal data type definitions
  typedef float input_transform_weights_t;
  typedef float input_transform_biases_t;
  typedef float output_transform_weights_t;
  typedef float output_transform_biases_t;
  typedef float aggregator_distance_weights_t;
  typedef float aggregator_distance_biases_t;

  typedef float norm_t;
  typedef float distance_t;
  typedef float edge_weight_t;
  typedef float aggr_t;

  typedef unsigned short index_t;

  // Type wide enough to accommodate n_vertices - can be determined programmatically in principle
  typedef ap_int<8> nvtx_t;
  // Type wide enough to accommodate n_graphs - can be determined programmatically in principle
  typedef ap_int<4> ngrph_t;

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
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  //#pragma HLS DATAFLOW

  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights cyclic factor=reshape_factor dim=1

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

  // Copy the igraph array to enable dataflow optimization
  typename CONFIG_T::ngrph_t igraph_copy[CONFIG_T::n_vertices];
  unsigned const igraph_reshape_factor = CONFIG_T::n_vertices / CONFIG_T::reuse_factor;
  #pragma HLS ARRAY_RESHAPE variable=igraph_copy cyclic factor=igraph_reshape_factor

  process_input_vertices<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    input_transform_weights,
    input_transform_biases,
    igraph_copy,
    edge_weights,
    aggregated_weights
  );

  set_output_vertices<res_T, CONFIG_T>(
    igraph_copy,
    output_transform_biases,
    edge_weights,
    aggregated_weights,
    res
  );
}

template<class data_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type != CONFIG_T::no_collapse>::type
garnet(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  typename CONFIG_T::ngrph_t const igraph[CONFIG_T::n_vertices],
  res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  //#pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete dim=1

  typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

  process_input_vertices_collapse<data_T, CONFIG_T>(
    data,
    igraph,
    aggregator_distance_weights,
    aggregator_distance_biases,
    input_transform_weights,
    input_transform_biases,
    edge_weight_sums,
    aggregated_weights
  );

  set_output_mean<res_T, CONFIG_T>(
    output_transform_biases,
    edge_weight_sums,
    aggregated_weights,
    res
  );
}

// TODO can just set res to res_T* and use garnet?
template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type == CONFIG_T::no_collapse>::type
garnet_single(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx_sample[1],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
  unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
  #pragma HLS ARRAY_RESHAPE variable=edge_weights cyclic factor=reshape_factor dim=1

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  nvtx_T nvtx_copy;

  compute_edges_aggregates_single<data_T, nvtx_T, CONFIG_T>(
    data,
    nvtx_sample[0],
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weights,
    edge_weight_sums,
    weighted_features_sums,
    nvtx_copy
  );

  set_vertex_output_single<nvtx_T, res_T, CONFIG_T>(
    nvtx_copy,
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weight_sums,
    edge_weights,
    weighted_features_sums,
    res
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type != CONFIG_T::no_collapse>::type
garnet_single(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx_sample[1],
  res_T res[CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  #pragma HLS DATAFLOW

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators];
  #pragma HLS ARRAY_RESHAPE variable=edge_weight_sums complete

  typename CONFIG_T::aggr_t weighted_features_sums[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
  #pragma HLS ARRAY_RESHAPE variable=weighted_features_sums complete

  compute_aggregates_single<data_T, nvtx_T, CONFIG_T>(
    data,
    nvtx_sample[0],
    aggregator_distance_weights,
    aggregator_distance_biases,
    edge_weight_sums,
    weighted_features_sums
  );

#ifndef GARNET_NVERT_MEAN
  set_aggregate_output_single<nvtx_T, res_T, CONFIG_T>(
    nvtx_sample[0],
#else
  set_aggregate_output_single<res_T, CONFIG_T>(
#endif
    input_transform_weights,
    input_transform_biases,
    output_transform_biases,
    edge_weight_sums,
    weighted_features_sums,
    res
  );
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type == CONFIG_T::no_collapse>::type
garnet_ref(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx[1],
  res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_propagate * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_propagate],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];

  typename CONFIG_T::aggr_t aggregated_features[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];

  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;

      aggregated_features[iap] = 0.;
    }
  }

  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv == nvtx[0])
      break;

    typename CONFIG_T::aggr_t propagated_features[CONFIG_T::n_propagate];

    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      propagated_features[ip] = input_transform_biases[ip];

      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const ipx = ip * CONFIG_T::n_in_features + ix;

        propagated_features[ip] += data[ivx] * input_transform_weights[ipx];
      }
    }

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;

      typename CONFIG_T::aggr_t distance = aggregator_distance_biases[ia];

      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        distance += data[ivx] * aggregator_distance_weights[iax];
      }

      edge_weights[iva] = compute_garnet_edge_weight<CONFIG_T>(distance);

      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;

        aggregated_features[iap] += edge_weights[iva] * propagated_features[ip];
      }
    }
  }

  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;

#ifdef GARNET_NVERT_MEAN
      aggregated_features[iap] /= nvtx[0];
#else
      aggregated_features[iap] >>= CONFIG_T::n_vertices_width;
#endif
    }
  }

  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv == nvtx[0])
      break;

    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ivo = iv * CONFIG_T::n_out_features + io;
      res_T acc = output_transform_biases[io];

      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;

        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
          typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;
          typename CONFIG_T::index_t const ioap = ioa * CONFIG_T::n_propagate + ip;

          acc += output_transform_weights[ioap] * edge_weights[iva] * aggregated_features[iap];
        }
      }

      res[ivo] = acc;
    }
  }
}

template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
typename std::enable_if<CONFIG_T::collapse_type != CONFIG_T::no_collapse>::type
garnet_ref(
  data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
  nvtx_T const nvtx[1],
  res_T res[CONFIG_T::n_out_features],
  typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_propagate * CONFIG_T::n_in_features],
  typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_propagate],
  typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
  typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
  typename CONFIG_T::output_transform_weights_t const output_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_propagate],
  typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
)
{
  typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
  typename CONFIG_T::aggr_t propagated_features[CONFIG_T::n_vertices * CONFIG_T::n_propagate];

  typename CONFIG_T::aggr_t edge_weight_sums[CONFIG_T::n_aggregators]{};

  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv == nvtx[0])
      break;

    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      typename CONFIG_T::index_t const ivp = iv * CONFIG_T::n_propagate + ip;

      propagated_features[ivp] = input_transform_biases[ip];

      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const ipx = ip * CONFIG_T::n_in_features + ix;

        propagated_features[ivp] += data[ivx] * input_transform_weights[ipx];
      }
    }

    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;

      typename CONFIG_T::aggr_t distance = aggregator_distance_biases[ia];

      for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
        typename CONFIG_T::index_t const ivx = iv * CONFIG_T::n_in_features + ix;
        typename CONFIG_T::index_t const iax = ia * CONFIG_T::n_in_features + ix;

        distance += data[ivx] * aggregator_distance_weights[iax];
      }

      edge_weights[iva] = compute_garnet_edge_weight<CONFIG_T>(distance);
      edge_weight_sums[ia] += edge_weights[iva];
    }
  }

  typename CONFIG_T::aggr_t aggregated_features[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];

  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;

      aggregated_features[iap] = 0.;

      for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
        if (iv == nvtx[0])
          break;

        typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const ivp = iv * CONFIG_T::n_propagate + ip;

        aggregated_features[iap] += edge_weights[iva] * propagated_features[ivp];
      }
    }
  }

  for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
      typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;

#ifdef GARNET_NVERT_MEAN
      aggregated_features[iap] /= nvtx[0];
#else
      aggregated_features[iap] >>= CONFIG_T::n_vertices_width;
#endif
    }
  }

  res_T vertex_res[CONFIG_T::n_vertices * CONFIG_T::n_out_features];

  for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
    if (iv == nvtx[0])
      break;

    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::index_t const ivo = iv * CONFIG_T::n_out_features + io;

      typename CONFIG_T::aggr_t acc = output_transform_biases[io];

      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        typename CONFIG_T::index_t const iva = iv * CONFIG_T::n_aggregators + ia;
        typename CONFIG_T::index_t const ioa = io * CONFIG_T::n_aggregators + ia;

        typename CONFIG_T::aggr_t aggr = 0.;

        for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
          typename CONFIG_T::index_t const iap = ia * CONFIG_T::n_propagate + ip;
          typename CONFIG_T::index_t const ioap = ioa * CONFIG_T::n_propagate + ip;

          aggr += output_transform_weights[ioap] * aggregated_features[iap];
        }

        acc += edge_weights[iva] * aggr;
      }

      vertex_res[ivo] = acc;
    }
  }

  for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
    typename CONFIG_T::aggr_t acc = 0.;

    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
      if (iv == nvtx[0])
        break;

      typename CONFIG_T::index_t const ivo = iv * CONFIG_T::n_out_features + io;

      acc += vertex_res[ivo];
    }

#ifdef GARNET_NVERT_MEAN
    acc /= nvtx[0];
#else
    acc >>= CONFIG_T::n_vertices_width;
#endif

    res[io] = acc;
  }
}

}

#endif
