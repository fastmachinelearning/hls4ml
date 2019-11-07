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

#include "myproject.h"

//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"

void myproject(
    input_t input1[N_INPUT_1_1],
    result_t layer7_out[N_LAYER_5],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input1 complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=layer7_out complete dim=0 
    #pragma HLS INTERFACE ap_vld port=input1,layer7_out 
    #pragma HLS PIPELINE 

    const_size_in_1 = N_INPUT_1_1;
    const_size_out_1 = N_LAYER_5;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(input1, layer2_out, w2, b2);

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out);

    layer4_t layer4_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::sigmoid<layer3_t, layer4_t, sigmoid_config4>(layer3_out, layer4_out);

    layer4_t layer4_deout[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_deout complete dim=0
    nnet::deriv_sigmoid<layer3_t, layer4_t, sigmoid_config4>(layer3_out, layer4_deout);


    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::dense<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5);

    layer6_t layer6_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::linear<layer5_t, layer6_t, linear_config6>(layer5_out, layer6_out);

    result_t layer7_tempout[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_tempout complete dim=0
    nnet::relu<layer6_t, result_t, relu_config7>(layer6_out, layer7_tempout);

    result_t layer7_deout[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_reout complete dim=0
    nnet::deriv_relu<layer6_t, result_t, relu_config7>(layer6_out, layer7_deout);

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start the NN Training ~~~~~
    result_t layer7_Loss[N_LAYER_5];
    input_t FakeTrue[N_LAYER_5] = {10};
    #pragma HLS ARRAY_PARTITION variable=layer7_Loss complete dim=0
    #pragma HLS ARRAY_PARTITION variable=FakeTrue complete dim=0
    deriv_MSELoss<input_t, layer6_t, result_t, config5>(FakeTrue, layer7_tempout, layer7_Loss);


    result_t layer7_LossAct[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer7_LossAct complete dim=0
    CalLossAct<result_t, result_t, result_t, config5>(layer7_Loss, layer7_deout, layer7_LossAct );

    result_t w5_grad[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=w5_grad complete dim=0
    CalGradient<result_t, layer3_t, config5> (layer7_LossAct, layer3_out, w5_grad);
    //for (int i = 0; i < N_LAYER_2; ++i)
    //{
    //#pragma HLS PIPELINE
      //layer7_out[0] += w5_grad[i];
    //}

////~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gradient for Layer 2 ~~~~~
    result_t w2_proLoss[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=w2_proLoss complete dim=0
    PropogateLoss<result_t, config5> (layer7_LossAct, w5, w2_proLoss);
    //for (int i = 0; i < N_LAYER_2; ++i)
    //{
//#pragma HLS PIPELINE
      //layer7_out[0] += w2_proLoss[i];
    //}

    result_t layer4_LossAct[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer4_LossAct complete dim=0
    CalLossAct<result_t, result_t, result_t, config2>(w2_proLoss, layer4_deout, layer4_LossAct );
    //for (int i = 0; i < N_LAYER_2; ++i)
    //{
//#pragma HLS PIPELINE
      //layer7_out[0] += layer4_LossAct[i];
    //}

    result_t w2_grad[N_INPUT_1_1*N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=w2_grad complete dim=0
    CalGradient<result_t, input_t, config2> (layer4_LossAct, input1, w2_grad);
    //for (int i = 0; i < N_INPUT_1_1*N_LAYER_2; ++i)
    //{
//#pragma HLS PIPELINE
      //layer7_out[0] += w2_grad[i];
    //}
   


    //UpdateWeight<model_default_t, config5>(w5, w5_grad);
    model_default_t neww5[N_LAYER_2];
#pragma HLS ARRAY_PARTITION variable=neww5 complete dim=0
    NewWeight<model_default_t, config5>(w5, w5_grad, neww5);
    //for (int i = 0; i < N_LAYER_2; ++i)
    //{
//#pragma HLS PIPELINE
      //layer7_out[0] += neww5[i];
    //}

    model_default_t neww2[N_INPUT_1_1*N_LAYER_2];
#pragma HLS ARRAY_PARTITION variable=neww2 complete dim=0
    NewWeight<model_default_t, config2>(w2, w2_grad, neww2);
    for (int i = 0; i < N_LAYER_2; ++i)
    {
#pragma HLS PIPELINE
      layer7_out[0] += neww2[i] + neww5[i];
    }

    //UpdateWeight<model_default_t, config2>(w2, w2_grad);
    //layer7_out[0] = w5[2] +w2[10];
    //for (int i = 0; i < N_LAYER_2; ++i)
    //{
//#pragma HLS PIPELINE
    //}
    
    //layer7_out[0] = w2[0] * w5[0];
}
