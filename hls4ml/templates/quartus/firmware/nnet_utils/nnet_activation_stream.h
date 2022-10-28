#ifndef NNET_ACTIVATION_STREAM_H_
#define NNET_ACTIVATION_STREAM_H_

#include "nnet_common.h"
#include "nnet_types.h"

namespace nnet{

// *************************************************
//       Linear Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void linear(stream<data_T> &data, stream<res_T> &res) {
    LinearActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        LinearPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            out_data[j] = in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       ReLU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void relu(stream<data_T> &data, stream<res_T> &res) {
    ReLUActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        ReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Leaky RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void leaky_relu(stream<data_T> &data, const typename data_T::value_type alpha, stream<res_T> &res) {
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;
    
    LeakyReLUActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        LeakyReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = alpha * in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Thresholded RELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void thresholded_relu(stream<data_T> &data, const typename data_T::value_type theta, stream<res_T> &res) {
    ThresholdedReLUActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        ThresholdedReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > theta) out_data[j] = in_data[j];
            else out_data[j] = 0;
        }

        res.write(out_data);
    }
}

// *************************************************
//       ELU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void elu(stream<data_T> &data, const typename data_T::value_type alpha, stream<res_T> &res) {
    #include "activation_tables/elu_table.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    EluActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        EluPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = datareg;
            } else {
                int index = (datareg*CONFIG_T::table_size/-8).to_int();
                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                out_data[j] = alpha * elu_table[index];
            }
        }

        res.write(out_data);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void elu(stream<data_T> &data, stream<res_T> &res) {
    elu<data_T, res_T, CONFIG_T>(data, 1.0, res);
}

// *************************************************
//       SeLU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void selu(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/selu_table.tb"

    SeluActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        SeluPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type datareg = in_data[j];
            if (datareg >= 0) {
                out_data[j] = typename data_T::value_type (1.0507009873554804934193349852946) * datareg;
            } else {
                int index = (datareg*CONFIG_T::table_size/-8).to_int();
                if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
                out_data[j] = selu_table[index];
            }
        }

        res.write(out_data);
    }
}

// *************************************************
//       PReLU Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void prelu(stream<data_T> &data, const typename data_T::value_type alpha[CONFIG_T::n_in], stream<res_T> &res) {
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;
    
    PReLUActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        PReLUPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if (in_data[j] > 0) out_data[j] = in_data[j];
            else out_data[j] = alpha[i*res_T::size+j] * in_data[j];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softplus Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void softplus(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/softplus_table.tb"

    SoftplusActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        SoftplusPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register int data_round = (in_data[j]*CONFIG_T::table_size/16).to_int();
            hls_register int index = data_round + 8*CONFIG_T::table_size/16;
            if (index < 0) index = 0;
            else if (index > CONFIG_T::table_size-1) index = CONFIG_T::table_size-1;
            out_data[j] = softplus_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softsign Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void softsign(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/softsign_table.tb"

    static const int MAX_VALUE = 8;

    SoftsignActLoop:
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        SoftsignPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type absValue;;
            if(in_data[j] < 0){
                absValue = -in_data[j];
            }
            else{
                absValue = in_data[j];
            }
            ac_int<16> index = (absValue * CONFIG_T::table_size / MAX_VALUE).to_int();
            if (absValue > MAX_VALUE) index = CONFIG_T::table_size - 1;
            if(in_data[j] < 0) {
                out_data[j] = -(typename res_T::value_type) softsign_table[index];
            }
            else {
                out_data[j] = (typename res_T::value_type) softsign_table[index];
            }
        }

        res.write(out_data);
    }
}

// *************************************************
//       Softmax Activation
// *************************************************

template <class data_T, class res_T, typename CONFIG_T>
void softmax_stable(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/exp_table.tb"
    #include "activation_tables/invert_table.tb"

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    hls_register typename data_T::value_type data_array[data_T::size];
    
    SoftmaxArrayLoop: 
    #pragma ii pipeline
    for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {    
        data_T in_pack = data.read();
        
        SoftmaxArrayPackLoop: 
        #pragma unroll 
        for(unsigned j = 0; j < data_T::size; j++) {
            data_array[j] = in_pack[j];
        }

        // Find the max and compute all delta(x_i, x_max)
        Op_max<typename data_T::value_type> op_max;
        hls_register typename data_T::value_type x_max = reduce<typename data_T::value_type, data_T::size, Op_max<typename data_T::value_type>>(data_array, op_max);

        // For the diffs, use the same type as the input but force rounding and saturation
        hls_register ac_fixed<data_T::value_type::width, data_T::value_type::i_width, true, AC_RND, AC_SAT> d_xi_xmax[data_T::size];
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++){
            d_xi_xmax[j] = data_array[j] - x_max;
        }

        // Calculate all the e^x's
        hls_register typename CONFIG_T::exp_table_t exp_res[data_T::size];
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            exp_res[j] = exp_table[softmax_stable_idx_from_real_val<typename data_T::value_type, CONFIG_T>(d_xi_xmax[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        hls_register typename CONFIG_T::exp_table_t exp_sum = reduce<typename CONFIG_T::exp_table_t, data_T::size, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        hls_register typename CONFIG_T::inv_table_t inv_exp_sum = invert_table[softmax_stable_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];
        res_T out_pack;
        
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++){
            
            // TODO - Find Quartus-equivalent pragma
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }
        
        res.write(out_pack);
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void softmax_latency(stream<data_T> &data, stream<res_T> &res){
    #include "activation_tables/exp_table_latency.tb"
    #include "activation_tables/invert_table_latency.tb"
    
    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    // Calculate all the e^x's
    hls_register typename CONFIG_T::exp_table_t exp_res[data_T::size];
    
    SoftmaxExpLoop: 
    #pragma ii pipeline
    for(unsigned i = 0; i < CONFIG_T::n_in / data_T::size; i++) {
        data_T in_pack = data.read();
        
        SoftmaxExpPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            exp_res[j] = exp_table_latency[softmax_latency_idx_from_real_val<typename data_T::value_type, CONFIG_T>(in_pack[j])];
        }

        // Explicitly sum the results with an adder tree.
        // Rounding & Saturation mode, which improve accuracy, prevent Vivado from expression balancing
        Op_add<typename CONFIG_T::exp_table_t> op_add;
        hls_register typename CONFIG_T::exp_table_t exp_sum = reduce<typename CONFIG_T::exp_table_t, CONFIG_T::n_in, Op_add<typename CONFIG_T::exp_table_t>>(exp_res, op_add);

        // Multiply previously calculated exponetials with the reciprocal of the sum
        hls_register typename CONFIG_T::inv_table_t inv_exp_sum = invert_table_latency[softmax_latency_idx_from_real_val<typename CONFIG_T::exp_table_t,CONFIG_T>(exp_sum)];

        res_T out_pack;
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++){
            // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
            out_pack[j] = exp_res[j] * inv_exp_sum;
        }

        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax_legacy(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/exp_table_legacy.tb"
    #include "activation_tables/invert_table_legacy.tb"
    
    // Index into the lookup table based on data for exponentials
    hls_register typename CONFIG_T::table_t exp_res[data_T::size];
    hls_register typename CONFIG_T::table_t exp_diff_res;
    hls_register typename data_T::value_type data_cache[data_T::size];

    SoftmaxInitLoop: 
    #pragma ii 1
    for(unsigned s = 0; s < CONFIG_T::n_in / data_T::size; s++) {
        data_T in_pack = data.read();
        
        SoftmaxInitPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < data_T::size; j++) {
            data_cache[j] = in_pack[j];
            exp_res[j] = 0;
        }

        SoftmaxExpLoop: 
        #pragma unroll
        for (int i = 0; i < data_T::size; i++) {
            SoftmaxExpInner: 
            #pragma unroll
            for (int j = 0; j < data_T::size; j++) {
                if (i == j) {
                    exp_diff_res = 1;
                } else {
                    int data_round = ((data_cache[j] - data_cache[i])*CONFIG_T::table_size/16).to_int();
                    int index = data_round + 8 * CONFIG_T::table_size / 16;
                    if (index < 0) index = 0;
                    if (index > CONFIG_T::table_size - 1) index = CONFIG_T::table_size - 1;
                    exp_diff_res = exp_table_legacy[index];
                }
                exp_res[i] += exp_diff_res;
            }
        }

        res_T out_pack;
        SoftmaxInvPackLoop: 
        #pragma unroll
        for(unsigned j = 0; j < res_T::size; j++) {
            int exp_res_index = (exp_res[j]*CONFIG_T::table_size/64).to_int();
            if (exp_res_index < 0) exp_res_index = 0;
            if (exp_res_index > CONFIG_T::table_size - 1) exp_res_index = CONFIG_T::table_size - 1;
            out_pack[j] = (typename res_T::value_type) invert_table_legacy[exp_res_index];
        }

        res.write(out_pack);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax_argmax(stream<data_T> &data, stream<res_T> &res) {
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        #pragma unroll
        for (int i = 0; i < res_T::size; i++) {
            out_data[i] = (typename res_T::value_type) 0;
        }

        hls_register typename data_T::value_type maximum = in_data[0];
        hls_register int idx = 0; 

        #pragma ii 1
        for (int i = 1; i < res_T::size; i++) {
            if (in_data[i] > maximum) {
                maximum = in_data[i];
                idx = i;
            }
        }

        out_data[idx] = (typename res_T::value_type) 1;
        res.write(out_data);
    }
}

template<class data_T, class res_T, typename CONFIG_T>
void softmax(stream<data_T> &data, stream<res_T> &res) {
    switch(CONFIG_T::implementation) {
        case softmax_implementation::latency:
            softmax_latency<data_T, res_T, CONFIG_T>(data, res);
            break;
        case softmax_implementation::stable:
            softmax_stable<data_T, res_T, CONFIG_T>(data, res);
            break;
        case softmax_implementation::legacy:
            softmax_legacy<data_T, res_T, CONFIG_T>(data, res);
            break;
        case softmax_implementation::argmax:
            softmax_argmax<data_T, res_T, CONFIG_T>(data, res);
            break;
        default:
            softmax_stable<data_T, res_T, CONFIG_T>(data, res);
            break;
    }    
}

// *************************************************
//       TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void dense_tanh(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/tanh_table.tb"
    static const int MAX_VALUE=4;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    TanHActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        TanHPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type absoluteValue;

            if(in_data[j] < 0) absoluteValue = (-1)*in_data[j];
            else absoluteValue = in_data[j];

            hls_register int index;
            if (absoluteValue <= MAX_VALUE) index = (absoluteValue*(CONFIG_T::table_size/MAX_VALUE)).to_int();
            else index = CONFIG_T::table_size-1;

            if(in_data[j] > 0) out_data[j] = tanh_table[index];
            else out_data[j] = -tanh_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Sigmoid Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void sigmoid(stream<data_T> &data, stream<res_T> &res) {
    #include "activation_tables/sigmoid_table.tb"
    static const int MAX_VALUE=8;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    SigmoidActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        data_T in_data = data.read();
        res_T out_data;

        SigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type absoluteValue;

            if(in_data[j] < 0) absoluteValue = (-1)*in_data[j];
            else absoluteValue = in_data[j];

            hls_register int index;
            if (absoluteValue <= MAX_VALUE) index = (absoluteValue*(CONFIG_T::table_size/MAX_VALUE)).to_int();
            else index = CONFIG_T::table_size-1;

            if(in_data[j] > 0) out_data[j] = sigmoid_table[index];
            else out_data[j] = 1 - sigmoid_table[index];
        }

        res.write(out_data);
    }
}

// *************************************************
//       Hard sigmoid Activation
// *************************************************
// Note - Theano and Tensorflow might have different definitions for hard sigmoid; could provide two implementations
template<class data_T, class res_T, typename CONFIG_T>
void hard_sigmoid(stream<data_T> &data, stream<res_T> &res) {
    static const typename data_T::value_type slope = (typename data_T::value_type) 0.2;
    static const typename data_T::value_type shift = (typename data_T::value_type) 0.5;

    constexpr unsigned multiplier_limit = DIV_ROUNDUP(data_T::size, CONFIG_T::reuse_factor);
    constexpr unsigned pipeline = data_T::size / multiplier_limit;

    HardSigmoidActLoop:
    #pragma ii pipeline
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {

        data_T in_data = data.read();
        res_T out_data;

        HardSigmoidPackLoop:
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            hls_register typename data_T::value_type datareg = slope * in_data[j] + shift;
            if (datareg > 1) datareg = 1;
            else if (datareg < 0) datareg = 0;
            out_data[j] = datareg;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Binary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void binary_tanh(stream<data_T> &data, stream<res_T> &res) {
    BinaryTanHActLoop: 
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        
        hls_register data_T in_data = data.read();
        hls_register res_T out_data;

        BinaryTanHPackLoop: 
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if(in_data[j] > 0) out_data[j] = (typename res_T::value_type) 1;
            else out_data[j] = (typename res_T::value_type) -1;
        }

        res.write(out_data);
    }
}

// *************************************************
//       Ternary TanH Activation
// *************************************************
template<class data_T, class res_T, typename CONFIG_T>
void ternary_tanh(stream<data_T> &data, stream<res_T> &res) {
  TernaryTanHActLoop: 
    #pragma ii 1
    for (int i = 0; i < CONFIG_T::n_in / res_T::size; i++) {
        
        hls_register data_T in_data = data.read();
        hls_register res_T out_data;

        TernaryTanHPackLoop: 
        #pragma unroll
        for (int j = 0; j < res_T::size; j++) {
            if(in_data[j] > 1) out_data[j] = (typename res_T::value_type) 1;
            else if (in_data[j] <=-1) out_data[j] = (typename res_T::value_type) -1;
            else out_data[j] = (typename res_T::value_type) 0;
        }

        res.write(out_data);
    }
  
}

}

#endif