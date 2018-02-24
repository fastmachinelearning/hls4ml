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

#ifndef NNET_RECURSIVE_H_
#define NNET_RECURSIVE_H_

#include "nnet_common.h"
#include "nnet_activation.h"

namespace nnet {

struct rnn_config
{
    // Internal data type definitions
    typedef float state_t;
    typedef float U_t;  // State x Input
    typedef float W_t;  // State x State
    typedef float V_t;  // Output x State

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 2;
    static const unsigned n_state = 2;
    static const unsigned activation_type = activ_relu;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

// Recusive Neural Network (RNN)
// Resources: 
//  - http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
//  - https://github.com/pangolulu/rnn-from-scratch
// Notes:
//  - RNN naming conventions adopted from the above links
//      - newstate = activation(U*input + W*state)
//      - output   = V*newstate
//  - If softmax is needed on output, perform *outside* this operations
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void simple_rnn(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::state_t newstate[CONFIG_T::n_state],
    typename CONFIG_T::U_t     param_U[CONFIG_T::n_in]   [CONFIG_T::n_state],
    typename CONFIG_T::W_t     param_W[CONFIG_T::n_state][CONFIG_T::n_state],
    typename CONFIG_T::V_t     param_V[CONFIG_T::n_state][CONFIG_T::n_out])
{

    // Initialize the state variable -- will maintain state between function calls
    //static typename CONFIG_T::state_t newstate[CONFIG_T::n_state];
    std::cout << "Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: U*input
    data_T inputcache;
    typename CONFIG_T::state_t inputmult[CONFIG_T::n_in][CONFIG_T::n_state];
    typename CONFIG_T::state_t inputacc[CONFIG_T::n_state];
    InputProd1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        inputcache = data[ii];
        InputProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputmult[ii][jj] = inputcache * param_U[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        inputacc[iacc] = 0;
    }
    InputAccum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        InputAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputacc[jj] += inputmult[ii][jj];
        }
    }

    // Operation: W*state
    data_T statecache;
    typename CONFIG_T::state_t statemult[CONFIG_T::n_state][CONFIG_T::n_state];
    typename CONFIG_T::state_t stateacc[CONFIG_T::n_state];
    StateProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        statecache = newstate[ii];
        StateProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            statemult[ii][jj] = statecache * param_W[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        stateacc[iacc] = 0;
    }
    StateAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        StateAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            stateacc[jj] += statemult[ii][jj];
        }
    }

    // Operation: U*input + W*state
    typename CONFIG_T::state_t rawstate[CONFIG_T::n_state];
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        rawstate[iacc] = inputacc[iacc] + stateacc[iacc];
    }

    std::cout << "Post-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << rawstate[ii] << " "; std::cout << "]" << std::endl;

    // Run activation function

    if (CONFIG_T::activation_type == activ_relu){
        relu<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_sigmoid){
        sigmoid<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_tanh){
        tanh<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }

    std::cout << "Activated State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: output = V*state
    data_T outputcache;
    typename CONFIG_T::state_t outputmult[CONFIG_T::n_state][CONFIG_T::n_out];
    OutputProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        outputcache = newstate[ii];
        OutputProd2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            outputmult[ii][jj] = outputcache * param_V[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        res[iacc] = 0;
    }
    OutputAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        OutputAccum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            res[jj] += outputmult[ii][jj];
        }
    }

}
//Version of the RNN where Pipeline is impossible but resource usage is significantly smaller. Here the states are contained in a fixed variable that is maintained during recursive calls
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void simple_rnn_static(
    data_T    data[CONFIG_T::n_in],
    res_T     res[CONFIG_T::n_out],
    typename CONFIG_T::U_t     param_U[CONFIG_T::n_in]   [CONFIG_T::n_state],
    typename CONFIG_T::W_t     param_W[CONFIG_T::n_state][CONFIG_T::n_state],
    typename CONFIG_T::V_t     param_V[CONFIG_T::n_state][CONFIG_T::n_out])
{

    // Initialize the state variable -- will maintain state between function calls
    static typename CONFIG_T::state_t newstate[CONFIG_T::n_state];
    std::cout << "Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: U*input
    data_T inputcache;
    typename CONFIG_T::state_t inputmult[CONFIG_T::n_in][CONFIG_T::n_state];
    typename CONFIG_T::state_t inputacc[CONFIG_T::n_state];
    InputProd1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        inputcache = data[ii];
        InputProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputmult[ii][jj] = inputcache * param_U[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        inputacc[iacc] = 0;
    }
    InputAccum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        InputAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            inputacc[jj] += inputmult[ii][jj];
        }
    }

    // Operation: W*state
    data_T statecache;
    typename CONFIG_T::state_t statemult[CONFIG_T::n_state][CONFIG_T::n_state];
    typename CONFIG_T::state_t stateacc[CONFIG_T::n_state];
    StateProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        statecache = newstate[ii];
        StateProd2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            statemult[ii][jj] = statecache * param_W[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        stateacc[iacc] = 0;
    }
    StateAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        StateAccum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
            stateacc[jj] += statemult[ii][jj];
        }
    }

    // Operation: U*input + W*state
    typename CONFIG_T::state_t rawstate[CONFIG_T::n_state];
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
        rawstate[iacc] = inputacc[iacc] + stateacc[iacc];
    }

    std::cout << "Post-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << rawstate[ii] << " "; std::cout << "]" << std::endl;

    // Run activation function

    if (CONFIG_T::activation_type == activ_relu){
        relu<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_sigmoid){
        sigmoid<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }
    else if (CONFIG_T::activation_type == activ_tanh){
        tanh<typename CONFIG_T::state_t, typename CONFIG_T::state_t, ACT_CONFIG_T>(rawstate, newstate);
    }

    std::cout << "Activated State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << newstate[ii] << " "; std::cout << "]" << std::endl;

    // Operation: output = V*state
    data_T outputcache;
    typename CONFIG_T::state_t outputmult[CONFIG_T::n_state][CONFIG_T::n_out];
    OutputProd1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        outputcache = newstate[ii];
        OutputProd2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            outputmult[ii][jj] = outputcache * param_V[ii][jj];
        }
    }
    for(int iacc = 0; iacc < CONFIG_T::n_out; iacc++) {
        res[iacc] = 0;
    }
    OutputAccum1: for(int ii = 0; ii < CONFIG_T::n_state; ii++) {
        OutputAccum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
            res[jj] += outputmult[ii][jj];
        }
    }

}
struct lstm_config
{
    // Internal data type definitions
    typedef float W_t;
    typedef float b_t;

    // Layer Sizes
    static const unsigned n_in = 10;
    static const unsigned n_out = 2;
    static const unsigned n_tot = 4;
    static const unsigned n_state = 2;
    static const unsigned activation_type = activ_relu;
    static const unsigned table_size = 1024;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
};

template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void matrixmult(
		data_T tmpdata[CONFIG_T::n_tot],
		res_T  tmpres [CONFIG_T::n_state],
		typename CONFIG_T::W_t     param_W[CONFIG_T::n_state][CONFIG_T::n_tot],
		typename CONFIG_T::b_t     param_b[CONFIG_T::n_state]
		) 
{
  // Operation: U*input
  data_T inputcache;
  typename CONFIG_T::W_t inputmult[CONFIG_T::n_in][CONFIG_T::n_state];
  typename CONFIG_T::W_t inputacc [CONFIG_T::n_state];

  const unsigned n_tot = CONFIG_T::n_in+CONFIG_T::n_state;
  Prod1: for(int ii = 0; ii < n_tot; ii++) {
    inputcache = tmpdata[ii];
  Prod2: for(int jj = 0; jj < n_tot; jj++) {
      inputmult[ii][jj] = inputcache * param_W[ii][jj];
    }
  }
  for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
    inputacc[iacc] = 0;
  }
 Accum1: for(int ii = 0; ii < n_tot; ii++) {
  Accum2: for(int jj = 0; jj < CONFIG_T::n_state; jj++) {
      inputacc[jj] += inputmult[ii][jj] + param_b[jj];
    }
  }
  //std::cout << "Param B: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << param_b[ii] << " "; std::cout << "]" << std::endl;  
    
  // Run activation function
  if (CONFIG_T::activation_type == activ_relu){
    relu<res_T, typename CONFIG_T::W_t, ACT_CONFIG_T>(inputacc, tmpres);
  }
  else if (CONFIG_T::activation_type == activ_sigmoid){
    sigmoid<res_T, typename CONFIG_T::W_t, ACT_CONFIG_T>(inputacc, tmpres);
  }
  else if (CONFIG_T::activation_type == activ_tanh){
    tanh<res_T, typename CONFIG_T::W_t, ACT_CONFIG_T>(inputacc, tmpres);
  }
  std::cout << " Mul Activated State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << tmpres[ii] << " "; std::cout << "]" << std::endl;
}

// Long Short term Memory NN (LSTM)
// Resources: 
// https://github.com/nicodjimenez/lstm/blob/master/lstm.py
// https://github.com/llSourcell/LSTM_Networks/blob/master/LSTM%20Demo.ipynb
// https://en.wikipedia.org/wiki/Long_short-term_memory
// Notes:
//  - LSTM naming conventions adopted from the above links
//      - newstate = activation(U*input + W*state)
//      - output   = V*newstate
//  - If softmax is needed on output, perform *outside* this operations
template<class data_T, class res_T, typename CONFIG_T, typename ACT_CONFIG_T>
void lstm(
	  data_T    data[CONFIG_T::n_in],
	  res_T     h_newstate[CONFIG_T::n_state],
	  res_T     s_newstate[CONFIG_T::n_state],
	  res_T     h_oldstate[CONFIG_T::n_state],
	  typename CONFIG_T::W_t     param_F [CONFIG_T::n_state][CONFIG_T::n_tot],
	  typename CONFIG_T::W_t     param_I [CONFIG_T::n_state][CONFIG_T::n_tot],
	  typename CONFIG_T::W_t     param_G [CONFIG_T::n_state][CONFIG_T::n_tot],
	  typename CONFIG_T::W_t     param_O [CONFIG_T::n_state][CONFIG_T::n_tot],
	  /* For the future to split the matrices into in and out
	  typename CONFIG_T::W_t     param_F [CONFIG_T::n_state][CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::W_t     param_I [CONFIG_T::n_state][CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::W_t     param_G [CONFIG_T::n_state][CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::W_t     param_O [CONFIG_T::n_state][CONFIG_T::n_state+CONFIG_T::n_in],
	  */
	  typename CONFIG_T::b_t     param_bF[CONFIG_T::n_state],
	  typename CONFIG_T::b_t     param_bI[CONFIG_T::n_state],
	  typename CONFIG_T::b_t     param_bG[CONFIG_T::n_state],
	  typename CONFIG_T::b_t     param_bO[CONFIG_T::n_state]
	  /* For the future optimizations
	  typename CONFIG_T::b_t     param_bG[CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::b_t     param_bI[CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::b_t     param_bF[CONFIG_T::n_state+CONFIG_T::n_in],
	  typename CONFIG_T::b_t     param_bO[CONFIG_T::n_state+CONFIG_T::n_in]
	  */
	  ) {

    // Initialize the state variable -- will maintain state between function calls
    std::cout << "S Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
    std::cout << "H Pre-State: [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_oldstate[ii] << " "; std::cout << "]" << std::endl;

    //To not have to double the number of matrices we compactify the state and input into one matrix. This however is a BAD use of resources.
    data_T tmpdata[ CONFIG_T::n_in+CONFIG_T::n_state];  
    for (int ii = 0;              ii < (CONFIG_T::n_in); ii++)  tmpdata[ii] = data[ii];
    for (int ii = CONFIG_T::n_in; ii < (CONFIG_T::n_tot); ii++) tmpdata[ii] = h_oldstate[ii-CONFIG_T::n_in];
    //Do all the different matrix multiplications (ideally in parallel)
    std::cout << "Data [ "; for (int ii = 0; ii < CONFIG_T::n_tot; ii++) std::cout << tmpdata[ii] << " "; std::cout << "]" << std::endl;

    //Forget
    res_T tmpres_F[CONFIG_T::n_state];
    matrixmult<data_T,res_T,CONFIG_T,ACT_CONFIG_T>(tmpdata,tmpres_F,param_F,param_bF);

    //Input
    res_T tmpres_I[CONFIG_T::n_state];
    matrixmult<data_T,res_T,CONFIG_T,ACT_CONFIG_T>(tmpdata,tmpres_I,param_I,param_bI);

    //State
    res_T tmpres_G[CONFIG_T::n_state];
    matrixmult<data_T,res_T,CONFIG_T,ACT_CONFIG_T>(tmpdata,tmpres_G,param_G,param_bG);
    
    //Output
    res_T tmpres_O[CONFIG_T::n_state];
    matrixmult<data_T,res_T,CONFIG_T,ACT_CONFIG_T>(tmpdata,tmpres_O,param_O,param_bO);

    // Operation: s=g*i+sold*f (update state)
    res_T rawstate[CONFIG_T::n_state]; //=> Not sure this is needed!!!!! Test Me
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
      rawstate[iacc] = tmpres_G[iacc]*tmpres_I[iacc] + s_newstate[iacc]*tmpres_F[iacc];
      //s_newstate[iacc] = tmpres_G[iacc]*tmpres_I[iacc] + s_newstate[iacc]*tmpres_F[iacc];
    }
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
      s_newstate[iacc] = rawstate[iacc];
    }
    // Operation: h=s*o
    for(int iacc = 0; iacc < CONFIG_T::n_state; iacc++) {
      h_newstate[iacc] = s_newstate[iacc]*tmpres_O[iacc];
    }
    std::cout << "Post-State: s [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << s_newstate[ii] << " "; std::cout << "]" << std::endl;
    std::cout << "Post-State: h [ "; for (int ii = 0; ii < CONFIG_T::n_state; ii++) std::cout << h_newstate[ii] << " "; std::cout << "]" << std::endl;
}



}//end namespace

#endif
