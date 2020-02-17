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
  namespace garnet_utils {

    template<class CONFIG_T>
    inline
    typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value>::type
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
    inline
    typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value>::type
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
    inline
    typename std::enable_if<std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
    get_edge_weight(typename CONFIG_T::distance_t distance, typename CONFIG_T::edge_weight_t edge_weights_table[])
    {
      typedef ap_uint<CONFIG_T::distance_width> index_t;

      index_t index(distance.range(CONFIG_T::distance_width - 1, 0));

      return edge_weights_table[index];
    }

    template<class CONFIG_T>
    inline
    typename std::enable_if<not std::is_class<typename CONFIG_T::distance_t>::value, typename CONFIG_T::edge_weight_t>::type
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
    inline
    typename CONFIG_T::edge_weight_t
    compute_edge_weight(typename CONFIG_T::distance_t distance)
    {
#ifdef __SYNTHESIS__
      typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
      // unsigned const reshape_factor = CONFIG_T::n_aggregators * CONFIG_T::n_in_features * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
      // #pragma HLS ARRAY_RESHAPE variable=edge_weights_table cyclic factor=reshape_factor dim=1
      bool initialized = false;
#else
      static typename CONFIG_T::edge_weight_t edge_weights_table[1 << CONFIG_T::distance_width];
      static bool initialized = false;
#endif
      if (not initialized) {
        initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
        initialized = true;
      }

      return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
    }

    template<unsigned n_aggregators, unsigned n_in_features, class edge_weight_T = float, class feature_T = float>
    inline
    void
    initialize_sums(
      edge_weight_T edge_weight_mean[n_aggregators],
      feature_T weighted_feature_mean[n_aggregators * n_in_features]
    )
    {
     Aggregators:
      for (unsigned ia = 0; ia < n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = 0.;
    
       InFeatures1:
        for (unsigned ix = 0; ix < n_in_features; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * n_in_features + ix;

          weighted_feature_mean[iax] = 0.;
        }
      }
    }

    template<class dividend_T, class exponent_T>
    inline
    typename std::enable_if<std::is_class<dividend_T>::value, dividend_T>::type
    normalize_log2(dividend_T dividend, exponent_T exponent)
    {
      return dividend >> exponent;
    }

    template<class dividend_T, class exponent_T>
    inline
    typename std::enable_if<not std::is_class<dividend_T>::value, dividend_T>::type
    normalize_log2(dividend_T dividend, exponent_T exponent)
    {
      return dividend / std::pow(2., exponent);
    }    

    template<class CONFIG_T, class nvtx_T = unsigned>
    inline
    void
    normalize_output_biases(
      nvtx_T const nvtx,
      typename CONFIG_T::output_transform_biases_t const original[CONFIG_T::n_out_features],
      typename CONFIG_T::output_transform_biases_t normalized[CONFIG_T::n_out_features]
    )
    {
     OutFeatures:
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        #pragma HLS UNROLL
        typename CONFIG_T::aggr_t bias = original[io];
        bias *= nvtx;
        bias = normalize_log2(bias, CONFIG_T::n_vertices_width);
        normalized[io] = bias;
      }
    }

    // normalization by nvtx
    template<class CONFIG_T, unsigned n_aggregators, unsigned n_in_features, class nvtx_T = unsigned, class feature_T = float>
    inline
    void
    normalize_sums(
      nvtx_T const nvtx,
      typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[n_aggregators],
      typename CONFIG_T::aggr_t const weighted_feature_accum[n_aggregators * n_in_features],
      typename CONFIG_T::edge_weight_t edge_weight_mean[n_aggregators],
      feature_T weighted_feature_mean[n_aggregators * n_in_features]
    )
    {
      // accum comes divided by unroll factor
      typename CONFIG_T::norm_t nvtx_norm = (CONFIG_T::n_vertices / CONFIG_T::reuse_factor) / nvtx;

     Aggregators:
      for (unsigned ia = 0; ia < n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = edge_weight_accum[ia] * nvtx_norm;

       InFeatures1:
        for (unsigned ix = 0; ix < n_in_features; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * n_in_features + ix;

          weighted_feature_mean[iax] = weighted_feature_accum[iax] * nvtx_norm;
        }
      }
    }

    // normalization by CONFIG_T::n_vertices (constant)
    template<class CONFIG_T, unsigned n_aggregators, unsigned n_in_features, class feature_T = float, class ufeature_T = float>
    inline void
    normalize_sums(
      typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[n_aggregators],
      typename CONFIG_T::aggr_t const weighted_feature_accum[n_aggregators * n_in_features],
      typename CONFIG_T::edge_weight_t edge_weight_mean[n_aggregators],
      feature_T weighted_feature_mean[n_aggregators * n_in_features]
    )
    {
     Aggregators:
      for (unsigned ia = 0; ia < n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = normalize_log2(edge_weight_accum[ia], CONFIG_T::log2_reuse_factor);

       InFeatures1:
        for (unsigned ix = 0; ix < n_in_features; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * n_in_features + ix;

          weighted_feature_mean[iax] = normalize_log2(weighted_feature_accum[iax], CONFIG_T::log2_reuse_factor);
        }
      }
    }

    template<class CONFIG_T, class data_T = float, class nvtx_T = unsigned>
    void
    compute_edges_aggregates(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    )
    {
      garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_mean, weighted_feature_mean);
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete      
      typename CONFIG_T::aggr_t weighted_feature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
      #pragma HLS ARRAY_RESHAPE variable=weighted_feature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_accum, weighted_feature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
        #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_mean_local, weighted_feature_mean_local);

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
    
              typename CONFIG_T::distance_t incr = data[ivx] * aggregator_distance_weights[iax];
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

            unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

            edge_weights[iva] = edge_weight;
          }
        }
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;
            weighted_feature_accum[iax] += normalize_log2(weighted_feature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(nvtx, edge_weight_accum, weighted_feature_accum, edge_weight_mean, weighted_feature_mean);
      else
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_accum, weighted_feature_accum, edge_weight_mean, weighted_feature_mean);
    }

    template<class CONFIG_T, class data_T = float, class nvtx_T = unsigned>
    void
    compute_aggregates(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    )
    {
      garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_mean, weighted_feature_mean);
    
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete
      typename CONFIG_T::aggr_t weighted_feature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
      #pragma HLS ARRAY_RESHAPE variable=weighted_feature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_accum, weighted_feature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
        #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_mean_local, weighted_feature_mean_local);
    
       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
    
              typename CONFIG_T::distance_t incr = data[ivx] * aggregator_distance_weights[iax];
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
    
              data_T incr = data[ivx] * edge_weight;

              weighted_feature_mean_local[iax] += incr;
            }
          }
        }
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;
            weighted_feature_accum[iax] += normalize_log2(weighted_feature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(nvtx, edge_weight_accum, weighted_feature_accum, edge_weight_mean, weighted_feature_mean);
      else
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators, CONFIG_T::n_in_features>(edge_weight_accum, weighted_feature_accum, edge_weight_mean, weighted_feature_mean);
    }

    template<class CONFIG_T, class data_T = float, class nvtx_T = unsigned, class res_T = float>
    void
    set_vertex_output(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
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
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;

          typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ia] * input_transform_biases[ioa];
    
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            #pragma HLS UNROLL
            unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
          }

          aggregated_weights[ioa] = aggregated_weight;
        }
      }

      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;

     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;

        OutFeatures2:
          for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
            unsigned const ivo = iv * CONFIG_T::n_out_features + io;
    
            res_T acc = output_transform_biases[io];
    
           Aggregators2:
            for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
              unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
              unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
    
               acc += edge_weights[iva] * aggregated_weights[ioa];
            }
    
            res[ivo] = acc;
          }
        }
      }
    }
    
    template<class CONFIG_T, class data_T = float, class res_T = float>
    void
    set_aggregate_output(
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      res_T res[CONFIG_T::n_out_features]
    )
    {
      #pragma HLS PIPELINE
    
     OutFeatures:
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        res_T acc = output_transform_biases[io];

       Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
    
          typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ia] * input_transform_biases[ioa];
    
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
          }

          acc += edge_weight_mean[ia] * aggregated_weight;
        }
    
        res[io] = acc;
      }
    }

    template<class CONFIG_T, unsigned isubl, class data_T = float, class nvtx_T = unsigned>
    void
    aggregates_stack_propagate(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators[isubl - 1]],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators[isubl]],
      typename CONFIG_T::edge_weight_t edge_weights_next[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[isubl]],
      typename CONFIG_T::edge_weight_t edge_weight_mean_next[CONFIG_T::n_aggregators[isubl]],
      data_T weighted_feature_mean_next[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]]
    )
    {
      unsigned const iprev = isubl - 1;
     
      typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_out_features[iprev] * CONFIG_T::n_aggregators[iprev]];
      #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete
    
     OutFeatures1:
      for (unsigned io = 0; io < CONFIG_T::n_out_features[iprev]; ++io) {
        #pragma HLS UNROLL
       Aggregators1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[iprev]; ++ia) {
          #pragma HLS UNROLL
          unsigned const ioa = io * CONFIG_T::n_aggregators[iprev] + ia;

          typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ia] * input_transform_biases[ioa];
    
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features[iprev]; ++ix) {
            #pragma HLS UNROLL
            unsigned const ioax = ioa * CONFIG_T::n_in_features[iprev] + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features[iprev] + ix;

            aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
          }

          aggregated_weights[ioa] = aggregated_weight;
        }
      }
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete      
      typename CONFIG_T::aggr_t weighted_feature_accum[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=weighted_feature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_accum, weighted_feature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators[isubl]];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]];
        #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_mean_local, weighted_feature_mean_local);

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;

          data_T data[CONFIG_T::n_out_features[iprev]];
          #pragma HLS ARRAY_RESHAPE variable=data complete

        OutFeatures2:
          for (unsigned io = 0; io < CONFIG_T::n_out_features[iprev]; ++io) {
            unsigned const ivo = iv * CONFIG_T::n_out_features[iprev] + io;
    
            res_T acc = output_transform_biases[io];
    
           Aggregators2:
            for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[iprev]; ++ia) {
              unsigned const iva = iv * CONFIG_T::n_aggregators[iprev] + ia;
              unsigned const ioa = io * CONFIG_T::n_aggregators[iprev] + ia;
    
               acc += edge_weights[iva] * aggregated_weights[ioa];
            }
    
            data[io] = acc;
          }
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[isubl]; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features[isubl] + ix;
    
              typename CONFIG_T::distance_t incr = data[ix] * aggregator_distance_weights[iax];
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features[isubl] + ix;
    
              data_T incr = data[ix] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

            unsigned const iva = iv * CONFIG_T::n_aggregators[isubl] + ia;

            edge_weights_next[iva] = edge_weight;
          }
        }
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[isubl]; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
            weighted_feature_accum[iax] += normalize_log2(weighted_feature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(nvtx, edge_weight_accum, weighted_feature_accum, edge_weight_mean_next, weighted_feature_mean_next);
      else
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_accum, weighted_feature_accum, edge_weight_mean_next, weighted_feature_mean_next);
    }

    template<class CONFIG_T, unsigned isubl, class data_T = float, class nvtx_T = unsigned>
    typename std::enable_if<isubl == CONFIG_T::n_sublayers - 1>::type
    compute_aggregates_stack(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators[isubl - 1]],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators[isubl]],
      typename CONFIG_T::edge_weight_t edge_weight_mean_out[CONFIG_T::n_aggregators[CONFIG_T::n_sublayers - 1]],
      data_T weighted_feature_mean_out[CONFIG_T::n_aggregators[CONFIG_T::n_sublayers - 1] * CONFIG_T::n_in_features[CONFIG_T::n_sublayers - 1]]
    )
    {
      garnet_utils::initialize_sums<CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_mean_out, weighted_feature_mean_out);

      unsigned const iprev = isubl - 1;

      typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_out_features[iprev] * CONFIG_T::n_aggregators[iprev]];
      #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete
    
     OutFeatures1:
      for (unsigned io = 0; io < CONFIG_T::n_out_features[iprev]; ++io) {
        #pragma HLS UNROLL
       Aggregators1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[iprev]; ++ia) {
          #pragma HLS UNROLL
          unsigned const ioa = io * CONFIG_T::n_aggregators[iprev] + ia;

          typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ia] * input_transform_biases[ioa];
    
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features[iprev]; ++ix) {
            #pragma HLS UNROLL
            unsigned const ioax = ioa * CONFIG_T::n_in_features[iprev] + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features[iprev] + ix;

            aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
          }

          aggregated_weights[ioa] = aggregated_weight;
        }
      }
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete      
      typename CONFIG_T::aggr_t weighted_feature_accum[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=weighted_feature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_accum, weighted_feature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators[isubl]];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]];
        #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_mean_local, weighted_feature_mean_local);

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;

          data_T data[CONFIG_T::n_out_features[iprev]];
          #pragma HLS ARRAY_RESHAPE variable=data complete

        OutFeatures2:
          for (unsigned io = 0; io < CONFIG_T::n_out_features[iprev]; ++io) {
            unsigned const ivo = iv * CONFIG_T::n_out_features[iprev] + io;
    
            res_T acc = output_transform_biases[io];
    
           Aggregators2:
            for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[iprev]; ++ia) {
              unsigned const iva = iv * CONFIG_T::n_aggregators[iprev] + ia;
              unsigned const ioa = io * CONFIG_T::n_aggregators[iprev] + ia;
    
               acc += edge_weights[iva] * aggregated_weights[ioa];
            }
    
            data[io] = acc;
          }
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[isubl]; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features[isubl] + ix;
    
              typename CONFIG_T::distance_t incr = data[ix] * aggregator_distance_weights[iax];
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_features[isubl] + ix;
    
              data_T incr = data[ix] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }
          }
        }
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators[isubl]; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features[isubl]; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_features[isubl] + ix;
            weighted_feature_accum[iax] += normalize_log2(weighted_feature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(nvtx, edge_weight_accum, weighted_feature_accum, edge_weight_mean_out, weighted_feature_mean_out);
      else
        normalize_sums<CONFIG_T, CONFIG_T::n_aggregators[isubl], CONFIG_T::n_in_features[isubl]>(edge_weight_accum, weighted_feature_accum, edge_weight_mean_out, weighted_feature_mean_out);
    }
      
    template<class CONFIG_T, unsigned isubl, class data_T = float, class nvtx_T = unsigned>
    typename std::enable_if<isubl != CONFIG_T::n_sublayers - 1>::type
    compute_aggregates_stack(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features[isubl - 1] * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[isubl - 1]],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators[isubl - 1]],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators[isubl - 1] * CONFIG_T::n_in_features[isubl - 1]],
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators[isubl]],
      typename CONFIG_T::edge_weight_t edge_weight_mean_out[CONFIG_T::n_aggregators[CONFIG_T::n_sublayers - 1]],
      data_T weighted_feature_mean_out[CONFIG_T::n_aggregators[CONFIG_T::n_sublayers - 1] * CONFIG_T::n_in_features[CONFIG_T::n_sublayers - 1]]
    )
    {
      #pragma HLS INLINE

      unsigned const iprev = isubl - 1;

      typename CONFIG_T::edge_weight_t edge_weights_next[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[isubl]];
      unsigned const reshape_factor = CONFIG_T::n_aggregators[isubl] * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
      #pragma HLS ARRAY_RESHAPE variable=edge_weights_next cyclic factor=reshape_factor dim=1
      typename CONFIG_T::edge_weight_t edge_weight_mean_next[CONFIG_T::n_aggregators[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_next complete
      data_T weighted_feature_mean_next[CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]];
      #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_next complete

      aggregates_stack_propagate<CONFIG_T, isubl>(
        nvtx,
        input_transform_weights,
        input_transform_biases,
        output_transform_biases,
        edge_weights,
        edge_weight_mean,
        weighted_feature_mean,
        aggregator_distance_weights,
        aggregator_distance_biases,
        edge_weights_next,
        edge_weight_mean_next,
        weighted_features_mean_next
      );

      compute_aggregates_stack<CONFIG_T, isubl + 1>(
        nvtx,
        input_transform_weights + (CONFIG_T::n_out_features[iprev] * CONFIG_T::n_aggregators[iprev] * CONFIG_T::n_in_features[iprev]),
        input_transform_biases + (CONFIG_T::n_out_features[iprev] * CONFIG_T::n_aggregators[iprev]),
        output_transform_biases + (CONFIG_T::n_out_features[iprev]),
        edge_weights,
        edge_weight_mean_next,
        weighted_feature_mean_next,
        aggregator_distance_weights + (CONFIG_T::n_aggregators[isubl] * CONFIG_T::n_in_features[isubl]),
        aggregator_distance_biases + (CONFIG_T::n_aggregators[isubl]),
        edge_weight_mean_out,
        weighted_feature_mean_out
      );
    }

  }
 
  struct garnet_config
  {
    // Layer specs
    static const unsigned n_vertices_width = 8;
    static const unsigned n_vertices = (1 << n_vertices_width);
    static const unsigned n_in_features = 4;
    static const unsigned n_in_ufeatures = 0; // number of unsigned features within in_features
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
    typedef float edge_weight_aggr_t;
    typedef float aggr_t;
    typedef float uaggr_t;
  
    enum OutputCollapse {
      no_collapse,
      collapse_mean,
      collapse_max
    };
  
    static const unsigned output_collapse = no_collapse;

    static const bool mean_by_nvert = false;
 
    // Optimization specs
    static const unsigned reuse_factor = 64;
    static const unsigned log2_reuse_factor = 6;

    static constexpr unsigned tmp[5] = {0, 1, 2, 3, 4};
  };

  // vertices -> vertices
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
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
  
    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    garnet_utils::compute_edges_aggregates<CONFIG_T>(
      data,
      nvtx[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weights,
      edge_weight_mean,
      weighted_feature_mean
    );
  
    garnet_utils::set_vertex_output<CONFIG_T>(
      nvtx[0],
      input_transform_weights,
      input_transform_biases,
      output_transform_biases,
      edge_weights,
      edge_weight_mean,
      weighted_feature_mean,
      res
    );
  }

  // vertices -> out features
  template<class data_T, class nvtx_T, class res_T, class CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW
  
    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    garnet_utils::compute_aggregates<CONFIG_T>(
      data,
      nvtx[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weight_mean,
      weighted_feature_mean
    );

    typename CONFIG_T::output_transform_biases_t normalized_output_biases[CONFIG_T::n_out_features];
    typename CONFIG_T::output_transform_biases_t const* output_biases;
    if (CONFIG_T::mean_by_nvert)
      output_biases = output_transform_biases;
    else {
      garnet_utils::normalize_output_biases<CONFIG_T>(nvtx[0], output_transform_biases, normalized_output_biases);
      output_biases = normalized_output_biases;
    }

    garnet_utils::set_aggregate_output<CONFIG_T>(
      input_transform_weights,
      input_transform_biases,
      output_biases,
      edge_weight_mean,
      weighted_feature_mean,
      res
    );
  }

  // vertices -> out features
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet_stack(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_input_transform_weights],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_input_transform_biases],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregator_distance_weights],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregator_distance_biases],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_output_transform_biases]
  )
  {
    #pragma HLS DATAFLOW

    typename CONFIG_T::edge_weight_t edge_weights_first[CONFIG_T::n_vertices * CONFIG_T::n_aggregators[0]];
    unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
    #pragma HLS ARRAY_RESHAPE variable=edge_weights_first cyclic factor=reshape_factor dim=1

    typename CONFIG_T::edge_weight_t edge_weight_mean_first[CONFIG_T::n_aggregators[0]];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_first complete
    
    data_T weighted_feature_mean_first[CONFIG_T::n_aggregators[0] * CONFIG_T::n_in_features[0]];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_first complete

    unsigned const ilast = CONFIG_T::n_sublayers - 1;

    typename CONFIG_T::edge_weight_t edge_weight_mean_last[CONFIG_T::n_aggregators[ilast]];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_ladt complete
    
    data_T weighted_feature_mean_last[CONFIG_T::n_aggregators[ilast] * CONFIG_T::n_in_features[ilast]];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean_last complete

    garnet_utils::compute_edges_aggregates<CONFIG_T>(
      data,
      nvtx[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weights_first,
      edge_weight_mean_first,
      weighted_feature_mean_first
    );

    garnet_utils::compute_aggregates_stack<CONFIG_T, 1>(
      nvtx[0],
      input_transform_weights,
      input_transform_biases,
      output_transform_biases,
      edge_weights_first,
      edge_weight_mean_first,
      weighted_feature_mean_first,
      aggregator_distance_weights + (CONFIG_T::n_aggregators[0] * CONFIG_T::n_in_features[0]),
      aggregator_distance_biases + (CONFIG_T::n_aggregators[0]),
      edge_weight_mean_last,
      weighted_feature_mean_last
    );

    typename CONFIG_T::output_transform_biases_t normalized_output_biases[CONFIG_T::n_out_features[ilast]];
    typename CONFIG_T::output_transform_biases_t const* output_biases;
    if (CONFIG_T::mean_by_nvert)
      output_biases = output_transform_biases + (CONFIG_T::n_output_transform_biases - CONFIG_T::n_out_features[ilast]);
    else {
      garnet_utils::normalize_output_biases<CONFIG_T>(nvtx[0], output_transform_biases + (CONFIG_T::n_output_transform_biases - CONFIG_T::n_out_features[ilast]);, normalized_output_biases);
      output_biases = normalized_output_biases;
    }

    garnet_utils::set_aggregate_output<CONFIG_T>(
      input_transform_weights + (CONFIG_T::n_input_transform_weights - CONFIG_T::n_out_features[ilast] * CONFIG_T::n_aggregators[ilast] * CONFIG_T::n_in_features[ilast]),
      input_transform_biases + (CONFIG_T::n_input_transform_biases - CONFIG_T::n_out_features[ilast] * CONFIG_T::n_aggregators[ilast]),
      output_biases,
      edge_weight_mean_last,
      weighted_feature_mean_last,
      res
    );
  }

  /* Reference (dumb) implementation returning (Vertices, Features) */
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
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
    typename CONFIG_T::aggr_t propagated_features[CONFIG_T::n_vertices * CONFIG_T::n_propagate];
  
    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
      if (iv == nvtx[0])
        break;
  
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        unsigned const ivp = iv * CONFIG_T::n_propagate + ip;
  
        propagated_features[ivp] = input_transform_biases[ip];
  
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
          unsigned const ipx = ip * CONFIG_T::n_in_features + ix;
  
          propagated_features[ivp] += data[ivx] * input_transform_weights[ipx];
        }
      }
  
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
  
        typename CONFIG_T::aggr_t distance = aggregator_distance_biases[ia];
  
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
  
          distance += data[ivx] * aggregator_distance_weights[iax];
        }
  
        edge_weights[iva] = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
      }
    }
  
    typename CONFIG_T::aggr_t aggregated_features[CONFIG_T::n_aggregators * CONFIG_T::n_propagate];
  
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        unsigned const iap = ia * CONFIG_T::n_propagate + ip;
  
        aggregated_features[iap] = 0.;
  
        for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
          if (iv == nvtx[0])
            break;
  
          unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
          unsigned const ivp = iv * CONFIG_T::n_propagate + ip;
  
          aggregated_features[iap] += edge_weights[iva] * propagated_features[ivp];
        }
      }
    }
  
    for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        unsigned const iap = ia * CONFIG_T::n_propagate + ip;
  
        if (CONFIG_T::mean_by_nvert)
          aggregated_features[iap] /= nvtx[0];
        else {
          // Not using right shift in case aggr_t is float or double
          aggregated_features[iap] /= CONFIG_T::n_vertices;
        }
      }
    }
  
    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
      if (iv == nvtx[0])
        break;
  
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        unsigned const ivo = iv * CONFIG_T::n_out_features + io;
  
        typename CONFIG_T::aggr_t acc = output_transform_biases[io];
  
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
  
          typename CONFIG_T::aggr_t aggr = 0.;
  
          for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            unsigned const iap = ia * CONFIG_T::n_propagate + ip;
            unsigned const ioap = ioa * CONFIG_T::n_propagate + ip;
  
            aggr += output_transform_weights[ioap] * aggregated_features[iap];
          }
  
          acc += edge_weights[iva] * aggr;
        }
  
        res[ivo] = acc;
      }
    }
  }

  /* Reference (dumb) implementation returning (Features) - output averaged over vertices already */
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
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
    typename CONFIG_T::aggr_t vertex_res[CONFIG_T::n_vertices * CONFIG_T::n_out_features];

    garnet_ref<CONFIG_T>(
      data,
      nvtx,
      vertex_res,
      input_transform_weights,
      input_transform_biases,
      aggregator_distance_weights,
      aggregator_distance_biases,
      output_transform_weights,
      output_transform_biases
    );
  
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::aggr_t acc = 0.;
  
      for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
        if (iv == nvtx[0])
          break;
  
        unsigned const ivo = iv * CONFIG_T::n_out_features + io;
  
        acc += vertex_res[ivo];
      }
  
      if (CONFIG_T::mean_by_nvert)
        acc /= nvtx[0];
      else {
        // Not using right shift in case aggr_t is float or double
        acc /= CONFIG_T::n_vertices;
      }
  
      res[io] = acc;
    }
  }

}

#endif
