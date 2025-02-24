#ifndef NNET_DEPTHWISE_PRODUCT_H_
#define NNET_DEPTHWISE_PRODUCT_H_

namespace nnet {

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_latency(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                               typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                               typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
    #pragma HLS INLINE

    typename CONFIG_T::accum_t mult[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_out];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=weights

    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

    #pragma HLS ARRAY_PARTITION variable=mult complete

    #pragma HLS ALLOCATION operation instances=mul limit=CONFIG_T::multiplier_limit

// Do the matrix-multiply
Product:
    for (int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS UNROLL
        mult[ii] = CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[ii], weights[ii]);
    }

// Initialize accumulator with input biases
ResetAccum:
    for (int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

// Accumulate multiplication result
Accum1:
    for (int ii = 0; ii < CONFIG_T::n_in / CONFIG_T::n_out; ii++) {
    Accum2:
        for (int jj = 0; jj < CONFIG_T::n_out; jj++) {
            int index = ii * CONFIG_T::n_out + jj;
            acc[jj] += mult[index];
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < CONFIG_T::n_out; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_rf_leq_nout(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                            typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                            typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;
    const int rufactor = CONFIG_T::reuse_factor;
    const int multfactor = MIN(CONFIG_T::n_in, rufactor);
    const int multiplier_limit = DIV_ROUNDUP(nin, multfactor);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);

    assert((multiplier_limit == block_factor) && "This function is correct only for RF <= N_CHAN");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        int out_index = ir;

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[in_index]));

            in_index += rufactor;
            out_index += rufactor;

            if (out_index >= nout) {
                out_index -= nout;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_rf_gt_nout_rem0(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                                typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                                typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;
    const int rufactor = MIN(CONFIG_T::reuse_factor, nin);
    const int multfactor = MIN(nin, rufactor);
    const int multiplier_limit = DIV_ROUNDUP(nin, multfactor);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);

    assert((rufactor >= nout && rufactor % nout == 0) &&
           "This function is correct only for RF >= N_CHAN && RF % N_CHAN == 0");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    int outidx[rufactor];
    int outstep = 0;
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        outstep++;
        if (outstep == nout) {
            outstep = 0;
        }
    }

    int out_index = 0;

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        out_index = outidx[ir];

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[in_index]));

            in_index += rufactor;
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void depthwise_product_resource_gt_nout(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                                        typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                                        typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {

    const int nin = CONFIG_T::n_in;
    const int nout = CONFIG_T::n_out;
    const int rufactor = MIN(CONFIG_T::reuse_factor, nin);
    const int block_factor = DIV_ROUNDUP(nin, rufactor);
    assert((rufactor > nout) && "This function is correct only for RF > N_CHAN");

    #pragma HLS function_instantiate variable=weights,biases
    #pragma HLS ARRAY_RESHAPE   variable=weights block factor=block_factor
    #pragma HLS ARRAY_RESHAPE   variable=data block factor=block_factor

    #pragma HLS ARRAY_PARTITION variable=biases complete

    typename CONFIG_T::accum_t acc[nout];
    #pragma HLS ARRAY_PARTITION variable=acc complete

InitAccum:
    for (int iacc = 0; iacc < nout; iacc++) {
        #pragma HLS UNROLL
        acc[iacc] = (typename CONFIG_T::accum_t)biases[iacc];
    }

    const int remainder = CONFIG_T::reuse_factor % nout;

    int outidx[rufactor];
    int outstep = 0;
IndexLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        outidx[ir] = outstep;
        outstep++;
        if (outstep == nout) {
            outstep = 0;
        }
    }

ReuseLoop:
    for (int ir = 0; ir < rufactor; ir++) {
        #pragma HLS PIPELINE II=1 rewind

        int in_index = ir;
        int out_index = outidx[ir];

    MultLoop:
        for (int im = 0; im < block_factor; im++) {
            #pragma HLS UNROLL

            // out_index = in_index % nout;
            acc[out_index] += static_cast<typename CONFIG_T::accum_t>(
                CONFIG_T::template product<data_T, typename CONFIG_T::weight_t>::product(data[in_index], weights[in_index]));

            in_index += rufactor;
            out_index += remainder;
            if (out_index >= nout) {
                out_index -= nout;
            }
        }
    }

// Cast to "res_t" type
Result:
    for (int ires = 0; ires < nout; ires++) {
        #pragma HLS UNROLL
        res[ires] = cast<data_T, res_T, CONFIG_T>(acc[ires]);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
class DepthwiseDenseLatency : public DepthwiseDenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        #pragma HLS INLINE
        depthwise_product_latency<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
class DepthwiseDenseResource_rf_leq_nout : public DepthwiseDenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        #pragma HLS INLINE
        depthwise_product_resource_rf_leq_nout<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
class DepthwiseDenseResource_rf_gt_nout_rem0 : public DepthwiseDenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        #pragma HLS INLINE
        depthwise_product_resource_rf_gt_nout_rem0<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

template <class data_T, class res_T, typename CONFIG_T>
class DepthwiseDenseResource_rf_gt_nout : public DepthwiseDenseKernel<data_T, res_T, CONFIG_T> {
  public:
    static void dense(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_out],
                      typename CONFIG_T::weight_t weights[CONFIG_T::n_in * CONFIG_T::n_out],
                      typename CONFIG_T::bias_t biases[CONFIG_T::n_out]) {
        #pragma HLS INLINE
        depthwise_product_resource_gt_nout<data_T, res_T, CONFIG_T>(data, res, weights, biases);
    }
};

} // namespace nnet
#endif
