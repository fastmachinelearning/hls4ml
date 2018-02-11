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
    typename CONFIG_T::U_t  param_U[CONFIG_T::n_in][CONFIG_T::n_state],
    typename CONFIG_T::W_t  param_W[CONFIG_T::n_state][CONFIG_T::n_state],
    typename CONFIG_T::V_t  param_V[CONFIG_T::n_state][CONFIG_T::n_out])
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
    else if (CONFIG_T::activation_type == activ_tanh){
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

}//end namespace

#endif
