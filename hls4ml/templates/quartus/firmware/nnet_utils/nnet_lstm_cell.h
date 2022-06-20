#ifndef NNET_LSTM_H_
#define NNET_LSTM_H_

#include "nnet_activation.h"
#include "nnet_common.h"
namespace nnet {

struct lstm_activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    typedef ac_fixed<16,8> table_t;
};

struct lstm_config {
  static const unsigned n_in=1;
  static const unsigned n_out=10;
  static const unsigned sliding_window = false;
  static const unsigned return_sequences = false;
  typedef ac_fixed<16,6,true> weight_t;
  typedef lstm_activ_config activ_config;

};

#ifndef HLS_SYNTHESIS
  #include <iostream>
  #include <fstream>
#endif
#ifndef TIMESTAMP_UNROLLING
  #define TIMESTAMP_UNROLLING
#endif

//----------------------
// COMUM CODE
//----------------------

template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_W(data_T input, res_T *out, const WEIGHT_T *weight) {

    MULTIPLY_W_LOOP:
    if(input != 1){
      #pragma unroll
      for (int j = 0; j < CONFIG_T::n_out; j++) {
        out[j] = input * weight[j];
      }
    }

    else{
      #pragma unroll
      for (int i = 0; i < CONFIG_T::n_out ; i++) {
          out[i] = 0;
          #pragma unroll
           for (int j = 0; j < CONFIG_T::n_out; j++) {
              out[i] += input[j] * weight[j*CONFIG_T::n_out +i];
          }
      }
    }
}
template<class data_T, class res_T,typename CONFIG_T,class WEIGHT_T>
void multiply_U(data_T *inputs, res_T out[], const WEIGHT_T *weight) {

    MULTIPLY_U_LOOP_I:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out ; i++) {
        out[i] = 0;
        MULTIPLY_U_LOOP_J:
        #pragma unroll
         for (int j = 0; j < CONFIG_T::n_out; j++) {
            out[i] += (data_T) inputs[j] * weight[j*CONFIG_T::n_out +i];

        }
    }
}

template<class data_T,class res_T, typename CONFIG_T, class WEIGHT_T>
void add_bias(data_T *inputs,res_T *out,const WEIGHT_T *bias) {

    ADD_BIAS_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = inputs[i] + bias[i];
    }

}
template<class data_T, class res_T, typename CONFIG_T>
void multiply_vectors(data_T *in1, data_T *in2, res_T *out) {

    MULTIPLY_VECT_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = in1[i] * in2[i];

    }
}
template<class data_T, class res_T,typename CONFIG_T>
void add_vectors(data_T *in1,data_T *in2,res_T *out) {

    ADD_VECTOR_LOOP:
    #pragma unroll
    for (int i = 0; i < CONFIG_T::n_out; i++) {
        out[i] = in1[i] + in2[i];

    }
}

//----------------------
// LSTM CODE
//----------------------

template<class data_T, typename CONFIG_T, typename WEIGHT_T>
  void lstm_cell(
        data_T *hidden_state,
        data_T *hidden_state_o,
        data_T *cell_state,
        data_T *cell_state_o,
        data_T inputs,
        const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
        const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
        const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    //----------------------
    //Internals definitions
    //----------------------

    data_T i_afterW   [lstm_config::n_out] ;
    data_T i_afterBias[lstm_config::n_out] ;
    data_T c_afterW   [lstm_config::n_out] ;
    data_T c_afterBias[lstm_config::n_out] ;
    data_T o_afterW   [lstm_config::n_out] ;
    data_T o_afterBias[lstm_config::n_out] ;
    data_T f_afterW   [lstm_config::n_out] ;
    data_T f_afterBias[lstm_config::n_out] ;

    // Hidden state Gate candidates, intermediate variables
    data_T i_hiddenCand[lstm_config::n_out] ;
    data_T f_hiddenCand[lstm_config::n_out] ;
    data_T c_hiddenCand[lstm_config::n_out] ;
    data_T o_hiddenCand[lstm_config::n_out] ;

    // AfterAddition, intermediate variables
    data_T i_afterAdd[lstm_config::n_out] ;
    data_T f_afterAdd[lstm_config::n_out] ;
    data_T c_afterAdd[lstm_config::n_out] ;
    data_T o_afterAdd[lstm_config::n_out] ;

    // Gate outputs
    data_T gate_i[lstm_config::n_out] ;
    data_T gate_f[lstm_config::n_out] ;
    data_T gate_c[lstm_config::n_out] ;
    data_T gate_o[lstm_config::n_out] ;
    data_T gate_ic[lstm_config::n_out] ;
    data_T gate_forget[lstm_config::n_out] ;

    data_T h[lstm_config::n_out] ;


    //intermediate variable cell calculation
    data_T cell_act_multp[lstm_config::n_out] ;
    data_T cell_act_add[lstm_config::n_out] ;


    //-----------Gate I Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, i_afterW , WI);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(i_afterW, i_afterBias, BI);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, i_hiddenCand, RWI);
    add_vectors<data_T,data_T,CONFIG_T>(i_afterBias, i_hiddenCand, i_afterAdd);
    //Activation
    //hls_fpga insert recurrent_activation --- Gate I


    //-----------Gate F Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, f_afterW, WF);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(f_afterW, f_afterBias, BF);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, f_hiddenCand, RWF);
    add_vectors<data_T,data_T,CONFIG_T>(f_afterBias, f_hiddenCand, f_afterAdd);
    //Activation
    //hls_fpga insert recurrent_activation --- Gate F


    //-----------Gate C Calculations
    //Weight multiplication
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, c_afterW, WC);
    //Bias addition
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(c_afterW, c_afterBias, BC);
    //Hidden Candidate
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, c_hiddenCand, RWC);
    add_vectors<data_T,data_T,CONFIG_T>(c_afterBias, c_hiddenCand, c_afterAdd);
    //Activation
    //hls_fpga insert activation  --- Gate C


    //-----------gate I and C multiply
    multiply_vectors<data_T,data_T,CONFIG_T>(gate_i, gate_c, gate_ic);

    //-----------Gate O Calculations
    multiply_W<data_T,data_T,CONFIG_T,WEIGHT_T>(inputs, o_afterW, WO);
    add_bias<data_T,data_T,CONFIG_T,WEIGHT_T>(o_afterW, o_afterBias, BO);
    multiply_U<data_T,data_T,CONFIG_T,WEIGHT_T>(hidden_state, o_hiddenCand, RWO);
    add_vectors<data_T,data_T,CONFIG_T>(o_afterBias, o_hiddenCand, o_afterAdd);
    //hls_fpga insert recurrent_activation  --- Gate O


    //-----------Cell State Calculation
    multiply_vectors<data_T,data_T,CONFIG_T>(gate_f, cell_state, cell_act_multp);
    add_vectors<data_T,data_T,CONFIG_T>(gate_ic, cell_act_multp, cell_act_add);

    //-----------Forget gate Calculation
    //hls_fpga insert activation  --- Forget Gate

    multiply_vectors<data_T,data_T,CONFIG_T>(gate_o, gate_forget, h);


    OUTPUT_WRITE_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state_o[x]=h[x];
      cell_state_o[x]=cell_act_add[x];
    }
    return;
  }


template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network( data_T *input0, res_T *res,
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_out]    ;
    data_T cell_state_temp  [CONFIG_T::n_out]    ;
    data_T h[CONFIG_T::n_out]    ;
    data_T c[CONFIG_T::n_out]    ;

    //Multi inputs
    if (CONFIG_T::return_sequences == false ) {
      static data_T inputs[CONFIG_T::n_timestamp] ;
    }
    else{
      static data_T inputs[CONFIG_T::n_timestamp][CONFIG_T::n_out] ;
    }

    static data_T inputs[CONFIG_T::n_timestamp] ;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
    }

    #pragma unroll
    #pragma ivdep
    //Write input dimention

    //Single input dimention
    for (int j=0; j<CONFIG_T::n_timestamp; j++){
      inputs[j] = input0[j];
    }

    //Multi input dimention
    //for (int j=0; j<CONFIG_T::n_timestamp; j++){
    //  for (int z=0; z<CONFIG_T::n_out; z++){
    //  inputs[j][z] = input0[z * CONFIG_T::n_out + j];
      //}
    //}

    #pragma unroll TIMESTAMP_UNROLLING
    for (int i=0; i < CONFIG_T::n_timestamp; i++){
      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state_temp[x] = hidden_state[x][i];
        cell_state_temp[x]   = cell_state[x][i];
      }

      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);

      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }

    #pragma unroll
    //Output when return_sequences

    //Output when return_sequences is false            
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }

    //Output when return_sequences is true
    //for(int x = 0; x < CONFIG_T::n_timestamp; x++){ 
    //  for(int h = 0; h < CONFIG_T::n_out; h++){
    //      res[x * CONFIG_T::n_out + h] = hidden_state[h][x+1];
    //  }
    //}
  }

  template<class data_T, class res_T,class CONFIG_T ,class WEIGHT_T>
  void lstm_network(data_T input0,res_T *res,
            const WEIGHT_T *WI   , const WEIGHT_T *WF   , const WEIGHT_T *WC   , const WEIGHT_T *WO  ,
            const WEIGHT_T *RWI  , const WEIGHT_T *RWF  , const WEIGHT_T *RWC  , const WEIGHT_T *RWO ,
            const WEIGHT_T *BI   , const WEIGHT_T *BF   , const WEIGHT_T *BC   , const WEIGHT_T *BO){

    data_T hidden_state[CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T cell_state  [CONFIG_T::n_out][CONFIG_T::n_timestamp + 1]     ;
    data_T hidden_state_temp[CONFIG_T::n_out]     ;
    data_T cell_state_temp  [CONFIG_T::n_out]     ;
    data_T h[CONFIG_T::n_out]     ;
    data_T c[CONFIG_T::n_out]     ;

    static data_T inputs[CONFIG_T::n_timestamp] ;

    INIT_LOOP:
    #pragma unroll
    for (int x = 0; x < CONFIG_T::n_out; x++) {
      hidden_state[x][0]=0;
      cell_state[x][0]=0;
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
        cell_state_temp[x]   = cell_state[x][i];
      }

      lstm_cell<data_T,CONFIG_T,WEIGHT_T>(hidden_state_temp,h,cell_state_temp,c,inputs[i],WI,WF,WC,WO,RWI,RWF,RWC,RWO,BI,BF,BC,BO);

      #pragma unroll
      for (int x = 0; x < CONFIG_T::n_out; x++) {
        hidden_state[x][i+1]=h[x];
        cell_state[x][i+1]=c[x];
      }
    }

    #pragma unroll
    //Output when return_sequences

    //Output for Sliding Window when return_sequences is false
    for (int x = 0; x < CONFIG_T::n_out; x++) {
        res[x]= hidden_state[x][CONFIG_T::n_timestamp];
    }

    //Output for Sliding Window when return_sequences is true
    //for(int x = 0; x < CONFIG_T::n_timestamp; x++){
    //  for(int h = 0; h < CONFIG_T::n_out; h++){
    //    res[x][h]= hidden_state[h][x+1];
    //  }
    //}
  }
}
#endif