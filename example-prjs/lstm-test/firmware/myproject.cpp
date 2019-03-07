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
#include <iostream>

#include "parameters.h"
#include "myproject.h"

#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_recursive.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/wr1.h"
#include "weights/w2.h"
#include "weights/b2.h"

void myproject(
		  input_t data[N_LOOP][N_INPUTS],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = N_INPUTS*N_LOOP;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[N_STATE_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    layer1_t s_state1_out[N_STATE_1];
    #pragma HLS ARRAY_PARTITION variable=s_state1_out complete dim=0
    for(int ii = 0; ii < N_STATE_1; ii++) layer1_out[ii] = 0;
    for(int ii = 0; ii < N_STATE_1; ii++) s_layer1_out[ii] = 0;
    for(int iloop = 0; iloop < N_LOOP; iloop++) { 
       nnet::lstm<input_t, input_t, config1,tanh_config1,tanh_config1_lstm>(data[iloop],layer1_out, s_state1_out,w1,wr1,b1);
    }

    result_t logits2[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::compute_layer<layer1_t, result_t, config2>(layer1_out, logits2, w2, b2);
    nnet::softmax<result_t, result_t, softmax_config2>(logits2, res);


}

void lstm_matrixmult( 
          input_t  data              [N_INPUTS],
          input_t  data_recurr       [N_STATE_1],
          layer1_t logits1     [N_STATE_1*4],
          layer1_t logitsnob1  [N_STATE_1*4],
          weight_default_t W1   [N_INPUTS*N_STATE_1*4],
          weight_default_t Wr1  [N_STATE_1*N_STATE_1*4],
          weight_default_t b1   [N_STATE_1*4]) {
    #pragma HLS PIPELINE 
  //#pragma HLS ARRAY_PARTITION variable=data complete dim=0
  //#pragma HLS ARRAY_PARTITION variable=data_recurr complete dim=0
    #pragma HLS ARRAY_PARTITION variable=logits1     complete
    #pragma HLS ARRAY_PARTITION variable=logitsnob1  complete 
    #pragma HLS ARRAY_PARTITION variable=W1          complete 
    #pragma HLS ARRAY_PARTITION variable=b1          complete 
    #pragma HLS ARRAY_PARTITION variable=Wr1         complete 
    nnet::matrixmult_Wb<input_t, layer1_t, 6,64, config1>(data       ,logits1   , W1,b1); 
    nnet::matrixmult_W <input_t, layer1_t,16,64, config1>(data_recurr,logitsnob1, Wr1); 
    //layer1_t logitsnob1_0[32];
    //#pragma HLS ARRAY_PARTITION variable=logitsnob1_0 complete dim=0
    //layer1_t logitsnob1_1[32];
    //#pragma HLS ARRAY_PARTITION variable=logitsnob1_1 complete dim=0
    //nnet::matrixmultsub_W< input_t, input_t, 16, 64, 32, 0, config1>(data_recurr, logitsnob1_0, Wr1);    
    //nnet::matrixmultsub_W< input_t, input_t, 16, 64, 32, 32, config1>(data_recurr, logitsnob1_1, Wr1);    
    //layer1_t logits1_0to0[32];
    //nnet::merge<input_t, 32, 32>(logitsnob1_0, logitsnob1_1, logitsnob1);
}

