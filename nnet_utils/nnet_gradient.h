// ===========================================================================
// 
//       Filename:  nnet_gradient.h
// 
//    Description:  function to calculate gradient for NN Training
// 
//        Version:  1.0
//        Created:  11/06/2019 11:39:39 AM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Zhenbin Wu (benwu), zhenbin.wu@gmail.com
//        Company:  UIC, CMS@LPC, CDF@FNAL
// 
// ===========================================================================

#ifndef  MY_NNET_GRADIENT_INC
#define  MY_NNET_GRADIENT_INC

  template<class data_T, class out_T, typename CONFIG_T>
void CalGradient(data_T    LossAct[CONFIG_T::n_out], out_T output[CONFIG_T::n_in],
    typename CONFIG_T::weight_t  w_grad[CONFIG_T::n_in*CONFIG_T::n_out])
{
    out_T cache;
    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS PIPELINE
        cache = output[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
          int index = ii*CONFIG_T::n_out+jj;
          w_grad[index] = LossAct[jj] * output[jj];
        }
    }

}


  template<class data_T, typename CONFIG_T>
void PropogateLoss(data_T    LossAct[CONFIG_T::n_out], typename CONFIG_T::weight_t  weights[CONFIG_T::n_in* CONFIG_T::n_out], 
    data_T w_proLoss[CONFIG_T::n_in])
{
    data_T cache;
    typename CONFIG_T::accum_t mult[CONFIG_T::n_in*CONFIG_T::n_out];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_in];
    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS ARRAY_PARTITION variable=mult complete
    #pragma HLS ARRAY_PARTITION variable=weights complete
    #pragma HLS ARRAY_PARTITION variable=LossAct complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    // Do the matrix-multiply
    Product1: for(int ii = 0; ii < CONFIG_T::n_out; ii++) {
        #pragma HLS PIPELINE
        cache = LossAct[ii];
        Product2: for(int jj = 0; jj < CONFIG_T::n_in; jj++) {
          int index = ii*CONFIG_T::n_in+jj;
          mult[index] = cache * weights[index];
        }
    }

    // Initialize accumulator with input biases
    ResetAccum: for(int iacc = 0; iacc < CONFIG_T::n_in; iacc++) {
            #pragma HLS UNROLL
        acc[iacc] = 0;
    }

    // Accumulate multiplication result
    Accum1: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
#pragma HLS PIPELINE
        Accum2: for(int jj = 0; jj < CONFIG_T::n_out; jj++) {
        int index = ii*CONFIG_T::n_out+jj;
        acc[ii] += mult[index];
        }
    }

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n_in; ires++){
          #pragma HLS UNROLL
        w_proLoss[ires] = (data_T) (acc[ires]);
    }

}

 template<class data_T, class res_T, class res2_T, typename CONFIG_T>
void CalLossAct( data_T    Loss[CONFIG_T::n_out], res_T   DeAct[CONFIG_T::n_out], res2_T     LossAct[CONFIG_T::n_out])
{
  // Do the matrix-multiply
Product1: for(int ii = 0; ii < CONFIG_T::n_out; ii++) {
#pragma HLS PIPELINE
            LossAct[ii] = Loss[ii] * DeAct[ii];
          }

}


template<class data_T, typename CONFIG_T>
void UpdateWeight(data_T weights[CONFIG_T::n_in * CONFIG_T::n_out], data_T gradient[CONFIG_T::n_in * CONFIG_T::n_out])
{
  // Do the matrix-multiply
Product1: for(int ii = 0; ii < CONFIG_T::n_in*CONFIG_T::n_out; ii++) {
#pragma HLS PIPELINE
            weights[ii] -= gradient[ii];
          }

}

template<class data_T, typename CONFIG_T>
void NewWeight(data_T weights[CONFIG_T::n_in * CONFIG_T::n_out], data_T gradient[CONFIG_T::n_in * CONFIG_T::n_out],
data_T newweights[CONFIG_T::n_in * CONFIG_T::n_out])
{
  // Do the matrix-multiply
Product1: for(int ii = 0; ii < CONFIG_T::n_in*CONFIG_T::n_out; ii++) {
#pragma HLS PIPELINE
            newweights[ii] = weights[ii]-gradient[ii];
          }

}

#endif   // ----- #ifndef MY_NNET_GRADIENT_INC  -----

