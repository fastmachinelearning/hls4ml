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

#ifndef NNET_GARNET_UNSIGNED_H_
#define NNET_GARNET_UNSIGNED_H_

// Precision-optimizing GarNet by separating signed and unsigned features

#include "nnet_garnet.h"

namespace nnet {
  namespace garnet_utils {

    template<class CONFIG_T, class edge_weight_T = float, class sfeature_T = float, class ufeature_T = float>
    inline
    void
    initialize_sums(
      edge_weight_T edge_weight_mean[CONFIG_T::n_aggregators],
      sfeature_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      ufeature_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = 0.;
    
       InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;

          weighted_sfeature_mean[iax] = 0.;
        }
       InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;

          weighted_ufeature_mean[iax] = 0.;
        }
      }
    }

    // normalization by nvtx
    template<class CONFIG_T, class nvtx_T = unsigned, class sfeature_T = float, class ufeature_T = float>
    inline
    void
    normalize_sums(
      nvtx_T const nvtx,
      typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[CONFIG_T::n_aggregators],
      typename CONFIG_T::aggr_t const weighted_sfeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      typename CONFIG_T::uaggr_t const weighted_ufeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      sfeature_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      ufeature_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      // accum comes divided by unroll factor
      typename CONFIG_T::norm_t nvtx_norm = (CONFIG_T::n_vertices / CONFIG_T::reuse_factor) / nvtx;

     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = edge_weight_accum[ia] * nvtx_norm;

       InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;

          weighted_sfeature_mean[iax] = weighted_sfeature_accum[iax] * nvtx_norm;
        }

       InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;

          weighted_ufeature_mean[iax] = weighted_ufeature_accum[iax] * nvtx_norm;
        }
      }
    }

    // normalization by CONFIG_T::n_vertices (constant)
    template<class CONFIG_T, class sfeature_T = float, class ufeature_T = float>
    inline void
    normalize_sums(
      typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[CONFIG_T::n_aggregators],
      typename CONFIG_T::aggr_t const weighted_sfeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      typename CONFIG_T::uaggr_t const weighted_ufeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      sfeature_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      ufeature_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        #pragma HLS UNROLL

        edge_weight_mean[ia] = normalize_log2(edge_weight_accum[ia], CONFIG_T::log2_reuse_factor);

       InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;

          weighted_sfeature_mean[iax] = normalize_log2(weighted_sfeature_accum[iax], CONFIG_T::log2_reuse_factor);
        }
       InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
          #pragma HLS UNROLL
          unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;

          weighted_ufeature_mean[iax] = normalize_log2(weighted_ufeature_accum[iax], CONFIG_T::log2_reuse_factor);
        }
      }
    }

    template<class CONFIG_T, class data_T = float, class udata_T = float, class nvtx_T = unsigned>
    void
    compute_edges_aggregates(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_sfeatures],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet_utils::initialize_sums<CONFIG_T>(edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
  
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete      
      typename CONFIG_T::aggr_t weighted_sfeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
      #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_accum complete 
      typename CONFIG_T::uaggr_t weighted_ufeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
      #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T>(edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_sfeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
        #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_mean_local complete
        typename CONFIG_T::uaggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
        #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T>(edge_weight_mean_local, weighted_sfeature_mean_local, weighted_ufeature_mean_local);

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
    
              if (ix < CONFIG_T::n_in_sfeatures) {
                unsigned const ivx = iv * CONFIG_T::n_in_sfeatures + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix - CONFIG_T::n_in_sfeatures;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_sfeatures + ix;
    
              data_T incr = data[ivx] * edge_weight;
    
              weighted_sfeature_mean_local[iax] += incr;
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
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
            weighted_sfeature_accum[iax] += normalize_log2(weighted_sfeature_mean_local[iax], log2_unroll_factor);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_accum[iax] += normalize_log2(weighted_ufeature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T>(nvtx, edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_accum, edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
      else
        normalize_sums<CONFIG_T>(edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_mean, edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
    }

    template<class CONFIG_T, class data_T = float, class udata_T = float, class nvtx_T = unsigned>
    void
    compute_aggregates(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_sfeatures],
      udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
      nvtx_T const nvtx,
      typename CONFIG_T::aggregator_distance_weights_t const aggregator_distance_weights[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::aggregator_distance_biases_t const aggregator_distance_biases[CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
      data_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures]
    )
    {
      garnet_utils::initialize_sums<CONFIG_T>(edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
    
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;
      unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;

      typename CONFIG_T::edge_weight_aggr_t edge_weight_accum[CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=edge_weight_accum complete
      typename CONFIG_T::aggr_t weighted_sfeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
      #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_accum complete 
      typename CONFIG_T::uaggr_t weighted_ufeature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
      #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_accum complete 

      garnet_utils::initialize_sums<CONFIG_T>(edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_accum);
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;
    
        typename CONFIG_T::edge_weight_aggr_t edge_weight_mean_local[CONFIG_T::n_aggregators];
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean_local complete
        typename CONFIG_T::aggr_t weighted_sfeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
        #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_mean_local complete
        typename CONFIG_T::uaggr_t weighted_ufeature_mean_local[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
        #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean_local complete

        garnet_utils::initialize_sums<CONFIG_T>(edge_weight_mean_local, weighted_sfeature_mean_local, weighted_ufeature_mean_local);
    
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
    
              if (ix < CONFIG_T::n_in_sfeatures) {
                unsigned const ivx = iv * CONFIG_T::n_in_sfeatures + ix;
                incr = data[ivx] * aggregator_distance_weights[iax];
              }
              else {
                unsigned const ivx = iv * CONFIG_T::n_in_ufeatures + ix - CONFIG_T::n_in_sfeatures;
                incr = udata[ivx] * aggregator_distance_weights[iax];
              }
    
              distance += incr;
            }
    
            typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<CONFIG_T>(distance);
    
            edge_weight_mean_local[ia] += edge_weight;
    
           InFeatures2:
            for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
              unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
              unsigned const ivx = iv * CONFIG_T::n_in_sfeatures + ix;
    
              data_T incr = data[ivx] * edge_weight;

              weighted_sfeature_mean_local[iax] += incr;
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
    
       Normalize1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          edge_weight_accum[ia] += normalize_log2(edge_weight_mean_local[ia], log2_unroll_factor);
         Normalize2:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_sfeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
            weighted_sfeature_accum[iax] += normalize_log2(weighted_sfeature_mean_local[iax], log2_unroll_factor);
          }
         Normalize3:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_ufeatures; ++ix) {
            unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix;
            weighted_ufeature_accum[iax] += normalize_log2(weighted_ufeature_mean_local[iax], log2_unroll_factor);
          }
        }
      }

      if (CONFIG_T::mean_by_nvert)
        normalize_sums<CONFIG_T>(nvtx, edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_accum, edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
      else
        normalize_sums<CONFIG_T>(edge_weight_accum, weighted_sfeature_accum, weighted_ufeature_accum, edge_weight_mean, weighted_sfeature_mean, weighted_ufeature_mean);
    }

    template<class CONFIG_T, class data_T = float, class udata_T = float, class nvtx_T = unsigned, class res_T = float>
    void
    set_vertex_output(
      nvtx_T const nvtx,
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      data_T const weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
      udata_T const weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures],
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

            if (ix < CONFIG_T::n_in_sfeatures) {
              unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
    
              aggregated_weight += weighted_sfeature_mean[iax] * input_transform_weights[ioax];
            }
            else {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix - CONFIG_T::n_in_sfeatures;
    
              aggregated_weight += weighted_ufeature_mean[iax] * input_transform_weights[ioax];
            }
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

    template<class CONFIG_T, class data_T = float, class udata_T = float, class res_T = float>
    void
    set_aggregate_output(
      typename CONFIG_T::input_transform_weights_t const input_transform_weights[CONFIG_T::n_out_features * CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
      typename CONFIG_T::input_transform_biases_t const input_transform_biases[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      typename CONFIG_T::output_transform_biases_t const output_transform_biases[CONFIG_T::n_out_features],
      typename CONFIG_T::edge_weight_t const edge_weight_mean[CONFIG_T::n_aggregators],
      data_T const weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures],
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

            if (ix < CONFIG_T::n_in_sfeatures) {
              unsigned const iax = ia * CONFIG_T::n_in_sfeatures + ix;
    
              aggregated_weight += weighted_sfeature_mean[iax] * input_transform_weights[ioax];
            }
            else {
              unsigned const iax = ia * CONFIG_T::n_in_ufeatures + ix - CONFIG_T::n_in_sfeatures;

              aggregated_weight += weighted_ufeature_mean[iax] * input_transform_weights[ioax];
            }
          }

          acc += edge_weight_mean[ia] * aggregated_weight;
        }
    
        res[io] = acc;
      }
    }

  }

  // vertices -> vertices
  template<class data_T, class udata_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_sfeatures],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
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
  
    data_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete
  
    garnet_utils::compute_edges_aggregates<CONFIG_T>(
      data,
      udata,
      nvtx[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weights,
      edge_weight_mean,
      weighted_sfeature_mean,
      weighted_ufeature_mean
    );
  
    garnet_utils::set_vertex_output<CONFIG_T>(
      nvtx[0],
      input_transform_weights,
      input_transform_biases,
      output_transform_biases,
      edge_weights,
      edge_weight_mean,
      weighted_sfeature_mean,
      weighted_ufeature_mean,
      res
    );
  }

  // vertices -> out features
  template<class data_T, class udata_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_sfeatures],
    udata_T const udata[CONFIG_T::n_vertices * CONFIG_T::n_in_ufeatures],
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
  
    data_T weighted_sfeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_sfeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_sfeature_mean complete

    udata_T weighted_ufeature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_ufeatures];
    #pragma HLS ARRAY_RESHAPE variable=weighted_ufeature_mean complete
  
    garnet_utils::compute_aggregates<CONFIG_T>(
      data,
      udata,
      nvtx[0],
      aggregator_distance_weights,
      aggregator_distance_biases,
      edge_weight_mean,
      weighted_sfeature_mean,
      weighted_ufeature_mean
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
      weighted_sfeature_mean,
      weighted_ufeature_mean,
      res
    );
  }
  
}

#endif
