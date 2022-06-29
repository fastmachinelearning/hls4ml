#ifndef NNET_SIMPLERNN_H_
#define NNET_SIMPLERNN_H_

#include "nnet_activation.h"
#include "nnet_common.h"

#ifndef SIMULATION_TIMES
  #define SIMULATION_TIMES 1
#endif
#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif

namespace nnet {

struct simple_rnn_activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    typedef ac_fixed<16,8> table_t;
};

struct simpleRNN_config {
  static const unsigned n_in=1;
  static const unsigned n_out=8;
  static const unsigned n_timestamp=5;
  static const unsigned sliding_window = false;
  static const unsigned return_sequences = false;
  typedef ac_fixed<16,6,true> weight_t;
  typedef ac_fixed<23,3,true> fixed_p_internal_t;
  typedef simple_rnn_activ_config activ_config;

};


//----------------------
// COMUM CODE
//----------------------

template<class data_T, typename res_T, typename CONFIG_T, class WEIGHT_T>
void multiply_W(data_T input,
                res_T *out,
                const WEIGHT_T *kernel) {
    MULTIPLY_W_LOOP:
    #pragma unroll
    for (int j = 0; j < CONFIG_T::n_out; j++) {
      out[j] = input * kernel[j];
    }
}

template<class data_T, typename res_T, typename CONFIG_T, class WEIGHT_T>
void multiply_U(data_T *inputs,
                res_T out[],
                const WEIGHT_T *recurrent_kernel) {
  MULTIPLY_U_LOOP_I:
  for (int i = 0 ; i <  CONFIG_T::n_out ; i++){
    out[i] = 0;
    MULTIPLY_U_LOOP_J:
    #pragma unroll
    for (int j=0;j< CONFIG_T::n_out; j++){
      out[i] += inputs[j] * recurrent_kernel[j*CONFIG_T::n_out +i];
    }
  }
}

template<typename res_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(res_T *inputs,
              res_T *out,
              const WEIGHT_T *bias) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = inputs[i] + bias[i];

    }
}

template<class data_T, class res_T,typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T out[] ) {
    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = (res_T)(in1[i] * in2[i]);
    }
}

template<typename res_T,typename CONFIG_T>
void add_vectors(res_T *in1, res_T *in2, res_T out[] ) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = (res_T) in1[i] + in2[i];

    }
}

//----------------------
// SimpleRNN CODE
//----------------------


template<class data_T, typename CONFIG_T, typename WEIGHT_T>
void simpleRNN_cell(
          data_T *hidden_state,
          data_T *hidden_state_o,
          data_T inputs,
          const WEIGHT_T *kernel,
          const WEIGHT_T *rec_kernel,
          const WEIGHT_T *bias){

        //----------------------
        //Internals definitions
        //----------------------

        // Gate outputs

        //Weight multiplication
        typename simpleRNN_config::fixed_p_internal_t afterW[CONFIG_T::n_out] hls_register;
        multiply_W<data_T,simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(inputs, afterW, kernel);

        //Bias addition
        typename simpleRNN_config::fixed_p_internal_t afterBias[CONFIG_T::n_out] hls_register;
        add_bias<simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(afterW,afterBias, bias);

        //hidden
        typename simpleRNN_config::fixed_p_internal_t hiddenCand[CONFIG_T::n_out] hls_register;
        multiply_U<data_T,simpleRNN_config::fixed_p_internal_t,CONFIG_T,WEIGHT_T>(hidden_state, hiddenCand, rec_kernel);

        typename simpleRNN_config::fixed_p_internal_t afterAdd[CONFIG_T::n_out];
        add_vectors<simpleRNN_config::fixed_p_internal_t, CONFIG_T>(afterBias, hiddenCand, afterAdd);

        data_T h[CONFIG_T::n_out];

        //Activation
        //hls_fpga insert activation

       OUTPUT_WRITE_LOOP:
        #pragma unroll
        for (int x = 0; x < CONFIG_T::n_out; x++) {
          hidden_state_o[x]=h[x];
        }
        return;
}

template<class data_T, class res_T, typename CONFIG_T, class WEIGHT_T>
  void simple_rnn_network(data_T input0[CONFIG_T::n_timestamp*CONFIG_T::n_in], res_T res[CONFIG_T::n_timestamp*CONFIG_T::n_out],
  const WEIGHT_T *kernel, const WEIGHT_T *rec_kernel, const WEIGHT_T *bias){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_out]  ;
    data_T h[CONFIG_T::n_out] hls_register    ;

    static data_T inputs[CONFIG_T::n_timestamp*CONFIG_T::n_in] hls_register;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
 
    //Input dimention

      for (int j=0; j<CONFIG_T::n_timestamp; j++){
        for (int z=0; z<CONFIG_T::n_in; z++){
          inputs[z* CONFIG_T::n_in + j] = input0[z * CONFIG_T::n_in + j];
        }
      }

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
      }

      for (int j=0; j<CONFIG_T::n_in; j++){
        simpleRNN_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,inputs[i], kernel, rec_kernel, bias);
      }

      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false 
      #pragma unroll           
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }

template<class data_T, class res_T, typename CONFIG_T, class WEIGHT_T>
  void simple_rnn_network(data_T input0, res_T *res,
  const WEIGHT_T *kernel, const WEIGHT_T *rec_kernel, const WEIGHT_T *bias){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_out]  ;
    data_T h[CONFIG_T::n_out] hls_register    ;

    static data_T inputs[CONFIG_T::n_timestamp] hls_register;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep

    for (int j=1;j<CONFIG_T::n_timestamp; j++){
      inputs[j-1] = inputs[j];
    }
    inputs[CONFIG_T::n_timestamp-1]=input0;

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
      }

      simpleRNN_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,inputs[i], kernel, rec_kernel, bias);
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
      }
    }


    if(CONFIG_T::return_sequences == 0){
      //Output when return_sequences is false
      #pragma unroll            
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
      }
    }
    else{
      //Output when return_sequences is true
      #pragma unroll
      for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
        for(int h = 0; h < CONFIG_T::n_out; h++){
            res[x + h * CONFIG_T::n_out ] = hidden_state[h][x+1];
        }
      }
    }
  }

}
#endif