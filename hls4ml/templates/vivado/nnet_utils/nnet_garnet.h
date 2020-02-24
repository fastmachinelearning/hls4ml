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

    // template<unsigned n_aggregators, unsigned n_in_features, class edge_weight_T = float, class feature_T = float>
    // inline
    // void
    // initialize_sums(
    //   edge_weight_T edge_weight_mean[n_aggregators],
    //   feature_T weighted_feature_mean[n_aggregators * n_in_features]
    // )
    // {
    //  Aggregators:
    //   for (unsigned ia = 0; ia < n_aggregators; ++ia) {
    //     #pragma HLS UNROLL

    //     edge_weight_mean[ia] = 0.;
    
    //    InFeatures1:
    //     for (unsigned ix = 0; ix < n_in_features; ++ix) {
    //       #pragma HLS UNROLL
    //       unsigned const iax = ia * n_in_features + ix;

    //       weighted_feature_mean[iax] = 0.;
    //     }
    //   }
    // }

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

    /* template<class CONFIG_T, class nvtx_T = unsigned> */
    /* inline */
    /* typename std::enable_if<CONFIG_T::mean_by_nvert>::type */
    /* normalize_output_biases( */
    /*   nvtx_T const nvtx, */
    /*   typename CONFIG_T::output_transform_biases_t normalized[CONFIG_T::n_out_features], */
    /*   typename CONFIG_T::output_transform_biases_t const*& output_biases */
    /* ) */
    /* { */
    /*   #pragma HLS INLINE */

    /*  OutFeatures: */
    /*   for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) { */
    /*     #pragma HLS UNROLL */
    /*     typename CONFIG_T::aggr_t bias = CONFIG_T::output_transform_biases[io]; */
    /*     bias *= nvtx; */
    /*     bias = normalize_log2(bias, CONFIG_T::n_vertices_width); */
    /*     normalized[io] = bias; */
    /*   } */
    /*   output_biases = normalized; */
    /* } */

    // normalization by nvtx
    // template<class CONFIG_T, class nvtx_T = unsigned, class feature_T = float>
    // inline
    // void
    // normalize_sums(
    //   nvtx_T const nvtx,
    //   typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[CONFIG_T::n_aggregators],
    //   typename CONFIG_T::aggr_t const weighted_feature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    //   typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
    //   feature_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    // )
    // {
    //   // accum comes divided by unroll factor
    //   typename CONFIG_T::norm_t nvtx_norm = (CONFIG_T::n_vertices / CONFIG_T::reuse_factor) / nvtx;

    //  Aggregators:
    //   for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    //     #pragma HLS UNROLL

    //     edge_weight_mean[ia] = edge_weight_accum[ia] * nvtx_norm;

    //    InFeatures1:
    //     for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
    //       #pragma HLS UNROLL
    //       unsigned const iax = ia * CONFIG_T:n_in_features + ix;

    //       weighted_feature_mean[iax] = weighted_feature_accum[iax] * nvtx_norm;
    //     }
    //   }
    // }

    // // normalization by CONFIG_T::n_vertices (constant)
    // template<class CONFIG_T, class feature_T = float, class ufeature_T = float>
    // inline void
    // normalize_sums(
    //   typename CONFIG_T::edge_weight_aggr_t const edge_weight_accum[CONFIG_T::n_aggregators],
    //   typename CONFIG_T::aggr_t const weighted_feature_accum[CONFIG_T::n_aggregators * CONFIG_T::n_in_features],
    //   typename CONFIG_T::edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators],
    //   feature_T weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features]
    // )
    // {
    //  Aggregators:
    //   for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
    //     #pragma HLS UNROLL

    //     edge_weight_mean[ia] = normalize_log2(edge_weight_accum[ia], CONFIG_T::log2_reuse_factor);

    //    InFeatures1:
    //     for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
    //       #pragma HLS UNROLL
    //       unsigned const iax = ia * CONFIG_T::n_in_features + ix;

    //       weighted_feature_mean[iax] = normalize_log2(weighted_feature_accum[iax], CONFIG_T::log2_reuse_factor);
    //     }
    //   }
    // }

    template<class CONFIG_T, class F, class E = typename CONFIG_T::edge_weight_t>
    struct Means {
      typedef F feature_t;
      typedef E edge_weight_t;
      
      edge_weight_t edge_weight_mean[CONFIG_T::n_aggregators];
      feature_t weighted_feature_mean[CONFIG_T::n_aggregators * CONFIG_T::n_in_features];

      Means() {
        #pragma HLS INLINE
        #pragma HLS ARRAY_RESHAPE variable=edge_weight_mean complete
        #pragma HLS ARRAY_RESHAPE variable=weighted_feature_mean complete

       Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL

          edge_weight_mean[ia] = 0.;

         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            #pragma HLS UNROLL

            unsigned const iax = ia * CONFIG_T::n_in_features + ix;
            weighted_feature_mean[iax] = 0.;
          }
        }
      }

      void set_weight(unsigned, edge_weight_t) {
        #pragma HLS INLINE
      }

      void add_means_normalized(Means<CONFIG_T, feature_t, edge_weight_t> const& local) {
        #pragma HLS INLINE

        unsigned const log2_unroll_factor = CONFIG_T::n_vertices_width - CONFIG_T::log2_reuse_factor;
        
       Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL
          
          edge_weight_mean[ia] += normalize_log2(local.edge_weight_mean[ia], log2_unroll_factor);

         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            #pragma HLS UNROLL
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;
            weighted_feature_mean[iax] += normalize_log2(local.weighted_feature_mean[iax], log2_unroll_factor);
          }
        }
      }

      template<class nvtx_T, class arrays_T, class T = CONFIG_T>
      typename std::enable_if<T::mean_by_nvert>::type set_means_normalized(nvtx_T const nvtx, arrays_T const& accum) {
        #pragma HLS INLINE

        // accum comes divided by unroll factor
        typename T::norm_t nvtx_norm = (T::n_vertices / T::reuse_factor) / nvtx;

       Aggregators:
        for (unsigned ia = 0; ia < T::n_aggregators; ++ia) {
          #pragma HLS UNROLL

          edge_weight_mean[ia] = accum.edge_weight_mean[ia] * nvtx_norm;

         InFeatures1:
          for (unsigned ix = 0; ix < T::n_in_features; ++ix) {
            #pragma HLS UNROLL
            unsigned const iax = ia * T::n_in_features + ix;

            weighted_feature_mean[iax] = accum.weighted_feature_mean[iax] * nvtx_norm;
          }
        }
      }

      template<class nvtx_T, class arrays_T, class T = CONFIG_T>
      typename std::enable_if<not T::mean_by_nvert>::type set_means_normalized(nvtx_T const nvtx, arrays_T const& accum) {
        #pragma HLS INLINE

       Aggregators:
        for (unsigned ia = 0; ia < T::n_aggregators; ++ia) {
          #pragma HLS UNROLL
  
          edge_weight_mean[ia] = normalize_log2(accum.edge_weight_mean[ia], T::log2_reuse_factor);
  
         InFeatures1:
          for (unsigned ix = 0; ix < T::n_in_features; ++ix) {
            #pragma HLS UNROLL
            unsigned const iax = ia * T::n_in_features + ix;
  
            weighted_feature_mean[iax] = normalize_log2(accum.weighted_feature_mean[iax], T::log2_reuse_factor);
          }
        }
      }
    };

    template<class CONFIG_T, class F, class E = typename CONFIG_T::edge_weight_t>
    struct WeightsAndMeans : public Means<CONFIG_T, F, E> {
      typedef E edge_weight_t;
      
      edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];

      WeightsAndMeans() : Means<CONFIG_T, F, E>() {
        #pragma HLS INLINE
        unsigned const reshape_factor = CONFIG_T::n_aggregators * (CONFIG_T::n_vertices / CONFIG_T::reuse_factor);
        #pragma HLS ARRAY_RESHAPE variable=edge_weights cyclic factor=reshape_factor dim=1
      }

      void set_weight(unsigned iva, edge_weight_t weight) {
        #pragma HLS INLINE
        edge_weights[iva] = weight;
      }
    };


    template<class CONFIG_T, class nvtx_T, class Enable = void>
    struct OutputBiasNormalizer;

    template<class CONFIG_T, class nvtx_T>
    struct OutputBiasNormalizer<CONFIG_T, nvtx_T, typename std::enable_if<CONFIG_T::mean_by_nvert>::type> {
      typedef typename CONFIG_T::output_transform_biases_t biases_t;

      biases_t const (&output_biases)[CONFIG_T::n_out_features];

      OutputBiasNormalizer(nvtx_T const) : output_biases{CONFIG_T::output_transform_biases} {
        #pragma HLS INLINE
      }
    };

    template<class CONFIG_T, class nvtx_T>
    struct OutputBiasNormalizer<CONFIG_T, nvtx_T, typename std::enable_if<not CONFIG_T::mean_by_nvert>::type> {
      typedef typename CONFIG_T::output_transform_biases_t biases_t;

      biases_t normalized[CONFIG_T::n_out_features];
      biases_t const (&output_biases)[CONFIG_T::n_out_features];
      
      OutputBiasNormalizer(nvtx_T const nvtx) : output_biases{normalized} {
        #pragma HLS ARRAY_RESHAPE variable=normalized complete

        for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
          #pragma HLS UNROLL
          typename CONFIG_T::aggr_t bias = CONFIG_T::output_transform_biases[io];
          bias *= nvtx;
          bias = normalize_log2(bias, CONFIG_T::n_vertices_width);
          normalized[io] = bias;
        }
      }
    };

    template<class CONFIG_T, class data_T, class arrays_local_T, class arrays_T>
    inline
    void
    compute_weights_aggregates(
      data_T const* data,
      arrays_local_T& arrays_local,
      arrays_T& arrays,
      unsigned iv = 0
    )
    {
      #pragma HLS INLINE
      
     Aggregators:
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        #pragma HLS UNROLL
        
        typename CONFIG_T::distance_t distance = CONFIG_T::aggregator_distance_biases[ia];

       InFeatures1:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          #pragma HLS UNROLL
          
          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;

          typename CONFIG_T::distance_t incr = data[ivx] * CONFIG_T::aggregator_distance_weights[iax];

          distance += incr;
        }

        typename CONFIG_T::edge_weight_t edge_weight = garnet_utils::compute_edge_weight<typename CONFIG_T::base_t>(distance);

        arrays_local.edge_weight_mean[ia] += edge_weight;

       InFeatures2:
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          #pragma HLS UNROLL
          
          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;

          data_T incr = data[ivx] * edge_weight;

          arrays_local.weighted_feature_mean[iax] += incr;
        }

        unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
        
        arrays.set_weight(iva, edge_weight);
      }
    }

    template<class CONFIG_T, class data_T, class nvtx_T, class arrays_T>
    void
    aggregate(
      data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
      nvtx_T const nvtx,
      arrays_T& arrays
    )
    {
      unsigned const unroll_factor = CONFIG_T::n_vertices >> CONFIG_T::log2_reuse_factor;

      Means<CONFIG_T, typename CONFIG_T::aggr_t, typename CONFIG_T::edge_weight_aggr_t> means_accum;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < CONFIG_T::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;

        Means<CONFIG_T, typename CONFIG_T::aggr_t, typename CONFIG_T::edge_weight_aggr_t> means_local;

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;
    
          compute_weights_aggregates<CONFIG_T>(data, means_local, arrays, iv);
        }

        means_accum.add_means_normalized(means_local);
      }

      arrays.set_means_normalized(nvtx, means_accum);
    }

    template<class CONFIG_T, class arrays_T>
    inline
    void
    compute_output_base(
      arrays_T const& arrays,
      typename CONFIG_T::aggr_t output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators]
    )
    {
      #pragma HLS INLINE
      
     OutFeatures1:
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        #pragma HLS UNROLL
        
       Aggregators1:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL
          
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;

          typename CONFIG_T::aggr_t aggr = arrays.edge_weight_mean[ia] * CONFIG_T::input_transform_biases[ioa];
    
         InFeatures1:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            #pragma HLS UNROLL
            
            unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            aggr += arrays.weighted_feature_mean[iax] * CONFIG_T::input_transform_weights[ioax];
          }

          output_base[ioa] = aggr;
        }
      }
    }

    template<class CONFIG_T, class arrays_T, class res_T>
    inline
    void
    compute_vertex_output(
      arrays_T const& arrays,
      typename CONFIG_T::aggr_t const output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators],
      res_T* res,
      unsigned iv = 0
    )
    {
      #pragma HLS INLINE
      
     OutFeatures2:
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        #pragma HLS UNROLL
        
        unsigned const ivo = iv * CONFIG_T::n_out_features + io;
    
        res_T acc = CONFIG_T::output_transform_biases[io];
    
       Aggregators2:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          #pragma HLS UNROLL
          
          unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
    
          acc += arrays.edge_weights[iva] * output_base[ioa];
        }
    
        res[ivo] = acc;
      }
    }

    template<class CONFIG_T, class nvtx_T, class arrays_T, class res_T>
    void
    disperse(
      nvtx_T const nvtx,
      arrays_T const& arrays,
      res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
    )
    {
      typename CONFIG_T::aggr_t output_base[CONFIG_T::n_out_features * CONFIG_T::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=output_base complete      

      compute_output_base<CONFIG_T>(arrays, output_base);

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

          compute_vertex_output<CONFIG_T>(arrays, output_base, res, iv);
        }
      }
    }
    
    template<class CONFIG_T, class output_biases_T, class arrays_T, class res_T>
    void
    set_output(
      output_biases_T const& output_transform_biases,
      arrays_T const& arrays,
      res_T res[CONFIG_T::n_out_features]
    )
    {
      #pragma HLS PIPELINE
    
     OutFeatures:
      for (unsigned io = 0; io < CONFIG_T::n_out_features; ++io) {
        res_T acc = output_transform_biases.output_biases[io];

       Aggregators:
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
    
          typename CONFIG_T::aggr_t aggregated_weight = arrays.edge_weight_mean[ia] * CONFIG_T::input_transform_biases[ioa];
    
         InFeatures:
          for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
            unsigned const ioax = ioa * CONFIG_T::n_in_features + ix;
            unsigned const iax = ia * CONFIG_T::n_in_features + ix;

            aggregated_weight += arrays.weighted_feature_mean[iax] * CONFIG_T::input_transform_weights[ioax];
          }

          acc += arrays.edge_weight_mean[ia] * aggregated_weight;
        }
    
        res[io] = acc;
      }
    }

    template<class prev_layer_t, class current_layer_t, class nvtx_T, class prev_arrays_T, class current_arrays_T>
    void
    disperse_aggregate(
      nvtx_T const nvtx,
      prev_arrays_T const& prev_arrays,
      current_arrays_T& current_arrays
    )
    {
      typename prev_layer_t::aggr_t prev_output_base[prev_layer_t::n_out_features * prev_layer_t::n_aggregators];
      #pragma HLS ARRAY_RESHAPE variable=prev_output_base complete

      compute_output_base<prev_layer_t>(prev_arrays, prev_output_base);
  
      unsigned const unroll_factor = current_layer_t::n_vertices >> current_layer_t::log2_reuse_factor;

      Means<current_layer_t, typename current_layer_t::aggr_t, typename current_layer_t::edge_weight_aggr_t> means_accum;
      
     VerticesOuter:
      for (unsigned ivv = 0; ivv < current_layer_t::reuse_factor; ++ivv) {
        // II will depend on the precision of data types - revisit
        #pragma HLS PIPELINE

        if (ivv * unroll_factor >= nvtx)
          break;

        Means<current_layer_t, typename current_layer_t::aggr_t, typename current_layer_t::edge_weight_aggr_t> means_local;

       VerticesInner:
        for (unsigned ir = 0; ir < unroll_factor; ++ir) {
          unsigned iv = ivv * unroll_factor + ir;
    
          if (iv >= nvtx)
            break;

          typename prev_arrays_T::feature_t data[prev_layer_t::n_out_features];
          #pragma HLS ARRAY_RESHAPE variable=data complete

          compute_vertex_output<prev_layer_t>(prev_arrays, prev_output_base, data);

          compute_weights_aggregates<current_layer_t>(data, means_local, current_arrays);
        }

        means_accum.add_means_normalized(means_local);
      }

      current_arrays.set_means_normalized(nvtx, means_accum);
    }

    template<class prev_layer_t, class current_layer_t, class last_layer_t, class nvtx_T, class prev_arrays_T, class last_arrays_T>
    inline
    typename std::enable_if<std::is_same<current_layer_t, last_layer_t>::value>::type
    sublayer(
      nvtx_T const nvtx,
      prev_arrays_T const& prev_arrays,
      last_arrays_T& last_arrays
    )
    {
      #pragma HLS INLINE

      disperse_aggregate<prev_layer_t, current_layer_t>(nvtx, prev_arrays, last_arrays);
    }
      
    template<class prev_layer_t, class current_layer_t, class last_layer_t, class nvtx_T, class prev_arrays_T, class last_arrays_T>
    inline
    typename std::enable_if<not std::is_same<current_layer_t, last_layer_t>::value>::type
    sublayer(
      nvtx_T const nvtx,
      prev_arrays_T const& prev_arrays,
      last_arrays_T& last_arrays
    )
    {
      #pragma HLS INLINE

      WeightsAndMeans<current_layer_t, typename prev_arrays_T::feature_t> current_arrays;

      disperse_aggregate<prev_layer_t, current_layer_t>(nvtx, prev_arrays, current_arrays);

      sublayer<current_layer_t, typename current_layer_t::next_layer_t, last_layer_t>(nvtx, current_arrays, last_arrays);
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

    /* static const input_transform_weights_t (&input_transform_weights)[n_out_features * n_aggregators * n_in_features]; */
    /* static const input_transform_biases_t (&input_transform_biases)[n_out_features * n_aggregators]; */
    /* static const aggregator_distance_weights_t (&aggregator_distance_weights)[n_aggregators * n_in_features]; */
    /* static const aggregator_distance_biases_t (&aggregator_distance_biases)[n_aggregators]; */
    /* static const output_transform_biases_t (&output_transform_biases)[n_out_features]; */

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
  };

  // vertices -> vertices
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW

    garnet_utils::WeightsAndMeans<CONFIG_T, data_T> arrays;

    garnet_utils::aggregate<CONFIG_T>(
      data,
      nvtx[0],
      arrays
    );
  
    garnet_utils::disperse<CONFIG_T>(
      nvtx[0],
      arrays,
      res
    );
  }

  // vertices -> out features
  template<class data_T, class nvtx_T, class res_T, class CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW

    garnet_utils::Means<CONFIG_T, data_T> arrays;
  
    garnet_utils::aggregate<CONFIG_T>(
      data,
      nvtx[0],
      arrays
    );

    garnet_utils::OutputBiasNormalizer<CONFIG_T, nvtx_T> normalize_bias(nvtx[0]);

    garnet_utils::set_output<CONFIG_T>(
      normalize_bias,
      arrays,
      res
    );
  }

  // vertices -> vertices
  template<class data_T, class nvtx_T, class res_T, class CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet_stack(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW

    typedef typename CONFIG_T::template sublayer<0> first_layer_t;
    unsigned const ilast = CONFIG_T::n_sublayers - 1;
    typedef typename CONFIG_T::template sublayer<ilast> last_layer_t;

    garnet_utils::WeightsAndMeans<first_layer_t, data_T> arrays_first;
    garnet_utils::Means<last_layer_t, data_T> arrays_last;

    garnet_utils::aggregate<first_layer_t>(
      data,
      nvtx[0],
      arrays_first
    );

    garnet_utils::sublayer<first_layer_t, typename first_layer_t::next_layer_t, last_layer_t>(
      nvtx[0],
      arrays_first,
      arrays_last
    );

    garnet_utils::disperse<last_layer_t>(
      nvtx[0],
      arrays_last,
      res
    );
  }

  // vertices -> out features
  template<class data_T, class nvtx_T, class res_T, class CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::collapse_mean>::type
  garnet_stack(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_out_features]
  )
  {
    #pragma HLS DATAFLOW

    typedef typename CONFIG_T::template sublayer<0> first_layer_t;
    unsigned const ilast = CONFIG_T::n_sublayers - 1;
    typedef typename CONFIG_T::template sublayer<ilast> last_layer_t;

    garnet_utils::WeightsAndMeans<first_layer_t, data_T> arrays_first;
    garnet_utils::Means<last_layer_t, data_T> arrays_last;

    garnet_utils::aggregate<first_layer_t>(
      data,
      nvtx[0],
      arrays_first
    );

    garnet_utils::sublayer<first_layer_t, typename first_layer_t::next_layer_t, last_layer_t>(
      nvtx[0],
      arrays_first,
      arrays_last
    );

    garnet_utils::OutputBiasNormalizer<last_layer_t, nvtx_T> normalize_bias(nvtx[0]);

    garnet_utils::set_output<last_layer_t>(
      normalize_bias,
      arrays_last,
      res
    );
  }

  /* Reference (dumb) implementation returning (Vertices, Features) */
  template<class data_T, class nvtx_T, class res_T, typename CONFIG_T>
  typename std::enable_if<CONFIG_T::output_collapse == CONFIG_T::no_collapse>::type
  garnet_ref(
    data_T const data[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    nvtx_T const nvtx[1],
    res_T res[CONFIG_T::n_vertices * CONFIG_T::n_out_features]
  )
  {
    typename CONFIG_T::edge_weight_t edge_weights[CONFIG_T::n_vertices * CONFIG_T::n_aggregators];
    typename CONFIG_T::aggr_t propagated_features[CONFIG_T::n_vertices * CONFIG_T::n_propagate];
  
    for (unsigned iv = 0; iv < CONFIG_T::n_vertices; ++iv) {
      if (iv == nvtx[0])
        break;
  
      for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
        unsigned const ivp = iv * CONFIG_T::n_propagate + ip;
  
        propagated_features[ivp] = CONFIG_T::input_transform_biases[ip];
  
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
          unsigned const ipx = ip * CONFIG_T::n_in_features + ix;
  
          propagated_features[ivp] += data[ivx] * CONFIG_T::input_transform_weights[ipx];
        }
      }
  
      for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
        unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
  
        typename CONFIG_T::aggr_t distance = CONFIG_T::aggregator_distance_biases[ia];
  
        for (unsigned ix = 0; ix < CONFIG_T::n_in_features; ++ix) {
          unsigned const ivx = iv * CONFIG_T::n_in_features + ix;
          unsigned const iax = ia * CONFIG_T::n_in_features + ix;
  
          distance += data[ivx] * CONFIG_T::aggregator_distance_weights[iax];
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
  
        typename CONFIG_T::aggr_t acc = CONFIG_T::output_transform_biases[io];
  
        for (unsigned ia = 0; ia < CONFIG_T::n_aggregators; ++ia) {
          unsigned const iva = iv * CONFIG_T::n_aggregators + ia;
          unsigned const ioa = io * CONFIG_T::n_aggregators + ia;
  
          typename CONFIG_T::aggr_t aggr = 0.;
  
          for (unsigned ip = 0; ip < CONFIG_T::n_propagate; ++ip) {
            unsigned const iap = ia * CONFIG_T::n_propagate + ip;
            unsigned const ioap = ioa * CONFIG_T::n_propagate + ip;
  
            aggr += CONFIG_T::output_transform_weights[ioap] * aggregated_features[iap];
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
    res_T res[CONFIG_T::n_out_features]
  )
  {
    typename CONFIG_T::aggr_t vertex_res[CONFIG_T::n_vertices * CONFIG_T::n_out_features];

    garnet_ref<CONFIG_T>(
      data,
      nvtx,
      vertex_res
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
