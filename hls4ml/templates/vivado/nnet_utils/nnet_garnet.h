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
  namespace garnet {

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
      if (!initialized) {
        initialize_edge_weights_table<CONFIG_T>(edge_weights_table);
        initialized = true;
      }

      return get_edge_weight<CONFIG_T>(distance, edge_weights_table);
    }

    template<class CONFIG_T, class T1 = float, class T2 = float, class T3 = float>
    inline void
    initialize_sums_single(
      T1 edge_weight_mean[CONFIG_T::n_aggregators],
      T2 weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      T3 weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = 0.;
    
       InFeatures:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          #pragma HLS UNROLL

          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
    
          if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)
            weighted_feature_mean[iax] = 0.;
          else
            weighted_ufeature_mean[iax] = 0.;
        }
      }
    }

    template<class CONFIG_T, class T1 = float, class T2 = float, class T3 = float>
    inline void
    initialize_sums(
      T1 edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      T2 weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      T3 weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
        #pragma HLS UNROLL

       Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL

          unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
  
          edge_weight_mean[ica] = 0.;
      
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            #pragma HLS UNROLL
  
            unsigned const icax = ica * CONFIG_T::n_in_features + ix;
      
            if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)
              weighted_feature_mean[icax] = 0.;
            else
              weighted_ufeature_mean[icax] = 0.;
          }
        }
      }
    }

    template<class nvtx_T, class CONFIG_T>
    inline void
    normalize_output_biases(
      nvtx_T const nvtx[CONFIG_T::n_graphs],
      typename CONFIG_T::output_transform_biases_t const original[CONFIG_T::n_out_features],
      typename CONFIG_T::output_transform_biases_t normalized[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
    )
    {
     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
       #pragma HLS UNROLL

        nvtx_T nv = nvtx[ic];

       OutFeatures:
        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
          #pragma HLS UNROLL
          unsigned const ico = ic * CONFIG_T::n_out_features + io;

          typename CONFIG_T::aggr_t bias = original[io];
          bias >>= CONFIG_T::n_vertices_width;
          bias *= nv;
          normalized[ico] = bias;
        }
      }
    }

    template<class CONFIG_T>
    inline void
    copy_output_biases(
      typename CONFIG_T::output_transform_biases_t const original[CONFIG_T::n_out_features],
      typename CONFIG_T::output_transform_biases_t copied[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
    )
    {
     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
       #pragma HLS UNROLL
       OutFeatures:
        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
          #pragma HLS UNROLL
          unsigned const ico = ic * CONFIG_T::n_out_features + io;

          copied[ico] = original[io];
        }
      }
    }

    template<class nvtx_T, class CONFIG_T>
    inline void
    normalize_sums_single(
      nvtx_T const nvtx,
      typename CONFIG_T::aggr_t edge_weight_mean[CONFIG_T::n_aggregators],
      typename CONFIG_T::aggr_t weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    )
    {
      #pragma HLS PIPELINE
    
      typename CONFIG_T::norm_t const nvtx_norm = 1. / nvtx;
    
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        edge_weight_mean[ia] *= nvtx_norm;
    
       InFeatures:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
    
          weighted_feature_mean[iax] *= nvtx_norm;
        }
      }
    }

    template<class nvtx_T, class CONFIG_T>
    inline void
    normalize_sums(
      nvtx_T const nvtx[CONFIG_T::n_graphs],
      typename CONFIG_T::aggr_t edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      typename CONFIG_T::aggr_t weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    )
    {
      #pragma HLS PIPELINE
     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
        nvtx_T nv = nvtx[ic];
    
        if (nv == 0) {
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
    
            edge_weight_mean[ica] = 0.;
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const icax = ica * CONFIG_T::n_in_features + ix;
    
              weighted_feature_mean[icax] = 0.;
            }
          }
        }
        else {
          typename CONFIG_T::norm_t const nvtx_norm = 1. / nv;
    
         Aggregators2:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
    
            edge_weight_mean[ica] *= nvtx_norm;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const icax = ica * CONFIG_T::n_in_features + ix;
    
              weighted_feature_mean[icax] *= nvtx_norm;
            }
          }
        }
      }
    }

    // TODO investigate consolidating with compute_aggregates_single
    template<class data_T, class nvtx_T, class CONFIG_T>
    void
    compute_edges_aggregates_single(
      data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean, weighted_feature_mean, weighted_ufeature_mean);
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        typename CONFIG_T::weighted_feature_aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
        typename CONFIG_T::weighted_ufeature_aggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];

        garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean_local, weighted_feature_mean_local, weighted_ufeature_mean_local);

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
    
              typename CONFIG_T::distance_t incr;
    
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
              unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

           InFeatures3:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
    
              udata_T incr = udata[ivx] * edge_weight;
    
              weighted_ufeature_mean_local[iax] += incr;
            }

            unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

            edge_weights[iva] = edge_weight;
          }
        }
    
#ifndef GARNET_NVERT_MEAN
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_mean[ia] += (edge_weight_mean_local[ia] >> CONFIG_T::n_vertices_width);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
            weighted_feature_mean[iax] += (weighted_feature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_mean[iax] += (weighted_ufeature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
        }
#endif
      }
    
#ifdef GARNET_NVERT_MEAN
      normalize_sums_single<nvtx_T, CONFIG_T>(nvtx, edge_weight_mean, weighted_feature_mean);
#endif
    }

    template<class data_T, class udata_T, class nvtx_T, class CONFIG_T>
    void
    compute_aggregates_single(
      data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean, weighted_feature_mean, weighted_ufeature_mean);
    
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        typename CONFIG_T::weighted_feature_aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
        typename CONFIG_T::weighted_ufeature_aggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];

        garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean_local, weighted_feature_mean_local, weighted_ufeature_mean_local);
    
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
    
              typename CONFIG_T::distance_t incr;
    
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
              unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

           InFeatures3:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
    
              udata_T incr = udata[ivx] * edge_weight;
    
              weighted_ufeature_mean_local[iax] += incr;
            }
          }
        }
    
#ifndef GARNET_NVERT_MEAN
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_mean[ia] += (edge_weight_mean_local[ia] >> CONFIG_T::n_vertices_width);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
            weighted_feature_mean[iax] += (weighted_feature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_mean[iax] += (weighted_ufeature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
        }
#endif
      }
    
#ifdef GARNET_NVERT_MEAN
      normalize_sums_single<nvtx_T, CONFIG_T>(nvtx, edge_weight_mean, weighted_feature_mean);
#endif
    }

    template<class data_T, class udata_T, class nvtx_T, class res_T, class CONFIG_T>
    void
    set_vertex_output_single(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
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

            if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
            }
            else {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
    
              aggregated_weight += weighted_ufeature_mean[iax] * input_transform_weights[ioax];
            }
          }

          aggregated_weights[ioa] = aggregated_weight;
        }
      }

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

    template<class data_T, class udata_T, class res_T, class CONFIG_T>
    void
    set_aggregate_output_single(
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
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

            if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              aggregated_weight += weighted_feature_mean[iax] * input_transform_weights[ioax];
            }
            else {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;

              aggregated_weight += weighted_ufeature_mean[iax] * input_transform_weights[ioax];
            }
          }
    
          acc += edge_weight_mean[ia] * aggregated_weight;
        }
    
        res[io] = acc;
      }
    }

    template<class data_T, class nvtx_T, class CONFIG_T>
    void
    compute_edges_aggregates(
      data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      typename CONFIG_T::graph_index_t const igraph[CONFIG_T::reuse_factor],
      nvtx_T const nvtx[CONFIG_T::n_graphs],
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_graphs * CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      data_T weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet::initialize_sums<CONFIG_T>(edge_weight_mean, weighted_feature_mean, weighted_ufeature_mean);
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::graph_index_t current_ic = -1;
      nvtx_T iv_max = 0;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        typename CONFIG_T::graph_index_t ic = igraph[ivv];

        if (ic == -1)
          break;

        if (ic != current_ic) {
          current_ic = ic;
          iv_max = ivv * unroll_factor + nvtx[ic];
        }

        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        typename CONFIG_T::weighted_feature_aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
        typename CONFIG_T::weighted_ufeature_aggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];

        garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean_local, weighted_feature_mean_local, weighted_ufeature_mean_local);

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;

          if (iv >= iv_max)
            break;    
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
    
              typename CONFIG_T::distance_t incr;
    
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
              unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

           InFeatures3:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
    
              udata_T incr = udata[ivx] * edge_weight;
    
              weighted_ufeature_mean_local[iax] += incr;
            }

            unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

            edge_weights[iva] = edge_weight;
          }
        }
    
#ifndef GARNET_NVERT_MEAN
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_mean[ia] += (edge_weight_mean_local[ia] >> CONFIG_T::n_vertices_width);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
            weighted_feature_mean[iax] += (weighted_feature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_mean[iax] += (weighted_ufeature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
        }
#endif
      }
    
#ifdef GARNET_NVERT_MEAN
      normalize_sums_single<nvtx_T, CONFIG_T>(nvtx, edge_weight_mean, weighted_feature_mean);
#endif
    }
    
    template<class data_T, class udata_T, class nvtx_T, class CONFIG_T>
    void
    compute_aggregates(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      typename CONFIG_T::graph_index_t const igraph[CONFIG_T::reuse_factor],
      nvtx_T const nvtx[CONFIG_T::n_graphs],
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      typename CONFIG_T::data_t weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      typename CONFIG_T::udata_t weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet::initialize_sums<CONFIG_T>(edge_weight_mean, weighted_feature_mean, weighted_ufeature_mean);
    
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::graph_index_t current_ic = -1;
      nvtx_T iv_max = 0;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        typename CONFIG_T::graph_index_t ic = igraph[ivv];

        if (ic == -1)
          break;
        
        if (ic != current_ic) {
          current_ic = ic;
          iv_max = ivv * unroll_factor + nvtx[ic];
        }

        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        typename CONFIG_T::weighted_feature_aggr_t weighted_feature_mean_local[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
        typename CONFIG_T::weighted_ufeature_aggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];

        garnet::initialize_sums_single<CONFIG_T>(edge_weight_mean_local, weighted_feature_mean_local, weighted_ufeature_mean_local);
    
       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= iv_max)
            break;
    
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            typename CONFIG_T::distance_t distance = aggregator_distance_biases[ia];
    
           InFeatures1:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_features + ix;
    
              typename CONFIG_T::distance_t incr;
    
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
              unsigned const ivx = iv * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_feature_mean_local[iax] += incr;
            }

           InFeatures3:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix;
    
              udata_T incr = udata[ivx] * edge_weight;
    
              weighted_ufeature_mean_local[iax] += incr;
            }

            unsigned const iva = iv * CONFIG_T::n_aggregators + ia;

            edge_weights[iva] = edge_weight;
          }
        }
    
#ifndef GARNET_NVERT_MEAN
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
          
          edge_weight_mean[ica] += (edge_weight_mean_local[ia] >> CONFIG_T::n_vertices_width);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
            unsigned const icax = ica * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
            weighted_feature_mean[icax] += (weighted_feature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            unsigned const icax = ica * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_mean[icax] += (weighted_ufeature_mean_local[iax] >> CONFIG_T::n_vertices_width);
          }
        }
#endif
      }
    
#ifdef GARNET_NVERT_MEAN
      normalize_sums<nvtx_T, CONFIG_T>(nvtx, edge_weight_mean, weighted_feature_mean);
#endif
    }

    template<class data_T, class udata_T, class res_T, class CONFIG_T>
    void
    set_vertex_output(
      typename CONFIG_T::graph_index_t const igraph[CONFIG_T::reuse_factor],
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
      res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
    )
    {
      typename CONFIG_T::aggr_t aggregated_weights[CONFIG_T::n_graphs * CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=aggregated_weights complete

     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
        #pragma HLS UNROLL
       OutFeatures1:
        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
          #pragma HLS UNROLL
         Aggregators1:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            #pragma HLS UNROLL
            unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
            unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
  
            typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ica] * input_transform_biases[ioa];
      
           InFeatures:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              #pragma HLS UNROLL
              unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
  
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const icax = ica * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
      
                aggregated_weight += weighted_feature_mean[icax] * input_transform_weights[ioax];
              }
              else {
                unsigned const icax = ica * CONFIG_T::n_in_ufeatures + ix;
      
                aggregated_weight += weighted_ufeature_mean[icax] * input_transform_weights[ioax];
              }
            }

            unsigned const icoa = (ic * CONFIG_T::n_out_features + io) * CONFIG_T::n_aggregators + ia;
  
            aggregated_weights[icoa] = aggregated_weight;
          }
        }
      }

     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        #pragma HLS PIPELINE

        typename CONFIG_T::graph_index_t ic = igraph[ivv];

        if (ic == -1)
          break;

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
        OutFeatures2:
          for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
            unsigned const ico = ic * CONFIG_T::n_out_features + io;
    
            res_T acc = output_transform_biases[ico];
    
           Aggregators2:
            for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
              unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
              unsigned const icoa = ico * CONFIG_T::n_aggregators + ia;
    
               acc += edge_weights[iva] * aggregated_weights[icoa];
            }

            unsigned const ivo = iv * CONFIG_T::n_out_features + io;
    
            res[ivo] = acc;
          }
        }
      }
    }

    template<class data_T, class udata_T, class res_T, class CONFIG_T>
    void
    set_aggregate_output(
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators],
      data_T const weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
      udata_T const weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
      res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features]
    )
    {
      #pragma HLS PIPELINE

     Graphs:
      for (unsigned ic = 0; ic < CONFIG_T::n_graphs; ++ic) {
       OutFeatures:
        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
          unsigned const ico = ic * CONFIG_T::n_out_features + io;

          res_T acc = output_transform_biases[ico];
          
         Aggregators:
          for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
            unsigned const ica = ic * CONFIG_T::n_aggregators + ia;
            unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
      
            typename CONFIG_T::aggr_t aggregated_weight = edge_weight_mean[ica] * input_transform_biases[ioa];
      
           InFeatures:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
              unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
  
              if (ix < CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) {
                unsigned const icax = ica * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures) + ix;
      
                aggregated_weight += weighted_feature_mean[icax] * input_transform_weights[ioax];
              }
              else {
                unsigned const icax = ica * CONFIG_T::n_in_ufeatures + ix;
  
                aggregated_weight += weighted_ufeature_mean[icax] * input_transform_weights[ioax];
              }
            }
      
            acc += edge_weight_mean[ica] * aggregated_weight;
          }
      
          res[ico] = acc;
        }
      }
    }
    
  }
 
  struct garnet_config
  {
    // Layer specs
    static const unsigned n_vertices_width = 8;
    static const unsigned n_vertices = (1 << n_vertices_width);
    static const unsigned n_graphs = 4; // maximum number of events that may be packed in the input array
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
    typedef float weighted_feature_aggr_t;
    typedef float weighted_ufeature_aggr_t;
    typedef float aggr_t;
  
    // Type wide enough to accommodate n_graphs - can be determined programmatically in principle
    typedef ap_int<4> graph_index_t;

    enum OutputCollapse {
      no_collapse,
      collapse_mean,
      collapse_max
    };
  
    static const unsigned output_collapse = no_collapse;
 
    // Optimization specs
    static const unsigned reuse_factor = 64;
    static const unsigned log2_reuse_factor = 6;
  };
  
  /* Multi-graph inference with unsigned input features returning (Vertices, Features) */
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
    typename CONFIG_T::graph_index_t const igraph[CONFIG_T::reuse_factor],
    nvtx_T const nvtx[CONFIG_T::n_graphs],
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

    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete

    garnet::compute_edges_aggregates<data_T, udata_T, nvtx_T, CONFIG_T>(
      data,
      udata,
      igraph,
      nvtx,
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weights,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean
    );
  
    garnet::set_vertex_output<data_T, udata_T, res_T, CONFIG_T>(
      igraph,
      input_transform_weights,
      input_transform_biases,
      output_transform_biases,
      edge_weight_mean,
      edge_weights,
      weighted_feature_mean,
      weighted_ufeature_mean,
      res
    );
  }

  /* Multi-graph inference with unsigned input features returning (Graphs, Features) - output averaged over vertices already */  
  template<class data_T, class udata_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
    typename CONFIG_T::graph_index_t const igraph[CONFIG_T::reuse_factor],
    nvtx_T const nvtx[CONFIG_T::n_graphs],
    res_T res[CONFIG_T::n_graphs * CONFIG_T::n_out_features],
    typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
    typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
    typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW

    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_graphs * CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete
  
    garnet::compute_aggregates<data_T, udata_T, nvtx_T, CONFIG_T>(
      data,
      udata,
      igraph,
      nvtx,
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean
    );

    typename CONFIG_T::output_transform_biases_t output_biases[CONFIG_T::n_graphs * CONFIG_T::n_out_features];
#ifndef GARNET_NVERT_MEAN
    garnet::normalize_output_biases<nvtx_T, CONFIG_T>(nvtx, output_transform_biases, output_biases);
#else
    garnet::copy_output_biases<CONFIG_T>(output_transform_biases, output_biases);
#endif

    garnet::set_aggregate_output<data_T, udata_T, res_T, CONFIG_T>(
      input_transform_weights,
      input_transform_biases,
      output_biases,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean,
      res
    );
  }
  
  /* Single-graph inference with unsigned input features returning (Vertices, Features) */
  // TODO can just set res to res_T* and use garnet?
  template<class data_T, class udata_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet_single(
    data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
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
  
    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete
  
    garnet::compute_edges_aggregates_single<data_T, udata_T, nvtx_T, CONFIG_T>(
      data,
      udata,
      nvtx_sample[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weights,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean
    );
  
    garnet::set_vertex_output_single<data_T, udata_T, nvtx_T, res_T, CONFIG_T>(
      nvtx_sample[0],
      input_transform_weights,
      input_transform_biases,
      output_transform_biases,
      edge_weight_mean,
      edge_weights,
      weighted_feature_mean,
      weighted_ufeature_mean,
      res
    );
  }

  /* Single-graph inference with unsigned input features returning (Features) - output averaged over vertices already */
  template<class data_T, class udata_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet_single(
    data_T const data[CONFIG_T::n_vertices * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
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
  
    typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
    #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
  
    data_T weighted_feature_mean[CONFIG_T::n_aggregators * (CONFIG_T::n_in_features - CONFIG_T::n_in_ufeatures)];
    #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete
  
    garnet::compute_aggregates_single<data_T, udata_T, nvtx_T, CONFIG_T>(
      data,
      udata,
      nvtx_sample[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean
    );
  
#ifndef GARNET_NVERT_MEAN
    typename CONFIG_T::output_transform_biases_t output_biases[CONFIG_T::n_out_features];
    garnet::normalize_output_biases<nvtx_T, CONFIG_T>(nvtx_sample, output_transform_biases, output_biases);
#else
    typename CONFIG_T::output_transform_biases_t const* output_biases = output_transform_biases;
#endif

    garnet::set_aggregate_output_single<data_T, udata_T, res_T, CONFIG_T>(
      input_transform_weights,
      input_transform_biases,
      output_biases,
      edge_weight_mean,
      weighted_feature_mean,
      weighted_ufeature_mean,
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
  
        edge_weights[iva] = garnet::compute_edge_weight<CONFIG_T>(distance);
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
  
  #ifdef GARNET_NVERT_MEAN
        aggregated_features[iap] /= nvtx[0];
  #else
        // Not using right shift in case aggr_t is float or double
        aggregated_features[iap] /= CONFIG_T::n_vertices;
  #endif
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

    garnet_ref(
      data,
      nvtx,
      vertex_res,
      input_transform_weights,
      input_transform_biases,
      aggregator_distance_weights,
      aggregator_distance_biases,
      output_transform_weights,
      output_transform_biases,
    );
  
    for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
      typename CONFIG_T::aggr_t acc = 0.;
  
      for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
        if (iv == nvtx[0])
          break;
  
        unsigned const ivo = iv * CONFIG_T::n_out_features + io;
  
        acc += vertex_res[ivo];
      }
  
  #ifdef GARNET_NVERT_MEAN
      acc /= nvtx[0];
  #else
      // Not using right shift in case aggr_t is float or double
      acc /= CONFIG_T::n_vertices;
  #endif
  
      res[io] = acc;
    }
  }

}

#endif
