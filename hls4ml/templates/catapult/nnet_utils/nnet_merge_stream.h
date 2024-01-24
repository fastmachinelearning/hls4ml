
#ifndef NNET_MERGE_STREAM_H_
#define NNET_MERGE_STREAM_H_

#include "nnet_common.h"
#include "ac_channel.h"
#include <math.h>

namespace nnet {

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    AddLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        AddPack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = in_data1[j] + in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    SubtractLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        SubtractPack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = in_data1[j] - in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    MultiplyLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        MultiplyPack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = in_data1[j] * in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    AverageLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        AveragePack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = (in_data1[j] + in_data2[j]) / (typename res_T::value_type) 2;
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    MaximumLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        MaximumPack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = (in_data1[j] > in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    assert(input1_T::size == input2_T::size && input1_T::size == res_T::size);

    constexpr int ce_reuse_factor = CONFIG_T::reuse_factor; (void)ce_reuse_factor;
    #pragma hls_pipeline_init_interval ce_reuse_factor
    MinimumLoop: for (int i = 0; i < CONFIG_T::n_elem / input1_T::size; i++) {
        //#pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        MinimumPack: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
            out_data[j] = (in_data1[j] < in_data2[j]) ? in_data1[j] : in_data2[j];
        }

        res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    ConcatLoopHeight1: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma hls_pipeline_init_interval 1
        ConcatLoopWidth1: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            //#pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            //#pragma HLS DATA_PACK variable=out_data

            #pragma hls_unroll
            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
    }
    ConcatLoopHeight2: for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        #pragma hls_pipeline_init_interval 1
        ConcatLoopWidth2: for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            //#pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            //#pragma HLS DATA_PACK variable=out_data

            #pragma hls_unroll
            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    ConcatLoopHeight: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma hls_pipeline_init_interval 1
        ConcatLoopWidth1: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            //#pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            res_T out_data;
            //#pragma HLS DATA_PACK variable=out_data

            #pragma hls_unroll
            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }

            res.write(out_data);
        }
        #pragma hls_pipeline_init_interval 1
        ConcatLoopWidth2: for (int j = 0; j < CONFIG_T::n_elem2_1; j++) {
            //#pragma HLS PIPELINE II=1

            input2_T in_data2 = data2.read();
            res_T out_data;
            //#pragma HLS DATA_PACK variable=out_data

            #pragma hls_unroll
            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

#pragma hls_pipeline_init_interval 1
template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    ConcatLoopHeight: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        #pragma hls_pipeline_init_interval 1
        ConcatLoopWidth: for (int j = 0; j < CONFIG_T::n_elem1_1; j++) {
            //#pragma HLS PIPELINE II=1

            input1_T in_data1 = data1.read();
            input2_T in_data2 = data2.read();
            res_T out_data;
            //#pragma HLS DATA_PACK variable=out_data

            #pragma hls_unroll
            ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[k] = in_data1[k];
            }
            
            #pragma hls_unroll
            ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
                // #pragma HLS UNROLL
                out_data[input1_T::size + k] = in_data2[k];
            }

            res.write(out_data);
        }
    }
}

#pragma hls_design block
#pragma hls_pipeline_init_interval 1
template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    if (CONFIG_T::axis == 3 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 2 || CONFIG_T::axis == -2) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

#pragma hls_design block
template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    #pragma hls_pipeline_init_interval 1
    ConcatLoopHeight1: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        //pragma HLS PIPELINE II=1

        input1_T in_data1 = data1.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
            // #pragma HLS UNROLL
            out_data[k] = in_data1[k];
        }

        res.write(out_data);
    }
    #pragma hls_pipeline_init_interval 1
    ConcatLoopHeight2: for (int i = 0; i < CONFIG_T::n_elem2_0; i++) {
        //#pragma HLS PIPELINE II=1

        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
            // #pragma HLS UNROLL
            out_data[k] = in_data2[k];
	}

	res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    #pragma hls_pipeline_init_interval 1
    ConcatLoopHeight: for (int i = 0; i < CONFIG_T::n_elem1_0; i++) {
        //#pragma HLS PIPELINE II=1

        input1_T in_data1 = data1.read();
        input2_T in_data2 = data2.read();
        res_T out_data;
        //#pragma HLS DATA_PACK variable=out_data

        #pragma hls_unroll
        ConcatPackInput1: for (int k = 0; k < input1_T::size; k++) {
            // #pragma HLS UNROLL
            out_data[k] = in_data1[k];
	}
            
   #pragma hls_unroll
	ConcatPackInput2: for (int k = 0; k < input2_T::size; k++) {
            // #pragma HLS UNROLL
            out_data[input1_T::size + k] = in_data2[k];
	}

	res.write(out_data);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(
    ac_channel<input1_T> &data1,
    ac_channel<input2_T> &data2,
    ac_channel<res_T> &res)
{
    res_T out_data;
    //#pragma HLS DATA_PACK variable=out_data
    ConcatLoop1: for (int i = 0; i < CONFIG_T::n_elem1_0 / input1_T::size; i++) {
        //#pragma HLS PIPELINE
        input1_T in_data1 = data1.read();
        #pragma hls_unroll
        ConcatPack1: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
	    out_data[j] = in_data1[j];
        }
        res.write(out_data);
    }
    ConcatLoop2: for (int i = 0; i < CONFIG_T::n_elem2_0 / input2_T::size; i++) {
        //#pragma HLS PIPELINE
        input2_T in_data2 = data2.read();
        #pragma hls_unroll
        ConcatPack2: for (int j = 0; j < res_T::size; j++) {
            // #pragma HLS UNROLL
	    out_data[j] = in_data2[j];
        }
        res.write(out_data);
    }
}
}

#endif
