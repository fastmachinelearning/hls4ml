// ===========================================================================
// 
//       Filename:  nnet_loss.h
// 
//    Description:  A code for loss function
// 
//        Version:  1.0
//        Created:  10/19/2019 02:42:09 PM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  Zhenbin Wu (benwu), zhenbin.wu@gmail.com
//        Company:  UIC, CMS@LPC, CDF@FNAL
// 
// ===========================================================================

#ifndef  MY_NNET_LOSS_INC
#define  MY_NNET_LOSS_INC

 template<class data_T, class res_T, class res2_T, typename CONFIG_T>
void L1Loss( data_T    data[CONFIG_T::n_in], res_T     res[CONFIG_T::n_in], res2_T     res2[CONFIG_T::n_out])
{
    typename CONFIG_T::accum_t diff[CONFIG_T::n_in];
    typename CONFIG_T::accum_t acc[CONFIG_T::n_in];
    #pragma HLS ARRAY_PARTITION variable=diff complete
    #pragma HLS ARRAY_PARTITION variable=acc complete

    Diff_Loss: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
        #pragma HLS PIPELINE
        diff[ii] = res[ii] - data[ii];
    }

    data_T cache;
    // Accumulate multiplication result
    Accum1_Loss: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
          #pragma HLS PIPELINE
              if (diff[ii] > 0 )
                cache = diff[ii];
              else
                cache = -1 *diff[ii];
              acc[ii] = cache;
    }

    data_T cache2 = 0;
    // Accumulate multiplication result
    Accum2_Loss: for(int ii = 0; ii < CONFIG_T::n_in; ii++) {
          #pragma HLS PIPELINE
                   cache2 += acc[ii];
    }

    res2[0] = cache2;
}

 template<class data_T, class res_T, class res2_T, typename CONFIG_T>
void deriv_MSELoss( data_T    Expect[CONFIG_T::n_out], res_T     Prediction[CONFIG_T::n_out], res2_T     Loss[CONFIG_T::n_out])
{
    typename CONFIG_T::accum_t diff[CONFIG_T::n_out];
    #pragma HLS ARRAY_PARTITION variable=diff complete

    Diff_Loss: for(int ii = 0; ii < CONFIG_T::n_out; ii++) {
        #pragma HLS PIPELINE
        diff[ii] = Prediction[ii] - Expect[ii];
    }

    // Accumulate multiplication result
    Accum2_Loss: for(int ii = 0; ii < CONFIG_T::n_out; ii++) {
          #pragma HLS PIPELINE
                   Loss[ii] =  2 * diff[ii];
    }

}


 template<class data_T, class res_T, typename CONFIG_T>
void RateStabilizer(data_T AEout[CONFIG_T::n_in], res_T fire[CONFIG_T::n_in])
{

  // Two bucket for counting events 
  // 2^30= 1073M ~ 1 LS @ 40MHz
#define N_Bucketsize 15
#define N_Maxbucket 1000000

  static ap_uint<N_Bucketsize> bucket1 = 0;
  static ap_uint<N_Bucketsize> bucket2 = 0;
  // Weight bucket for rate estimation
  static ap_fixed<16, 6, AP_TRN, AP_SAT> weightbucket = 0;
  // Init threshol for firing 
  static ap_fixed<16, 6, AP_TRN, AP_SAT> threshold = 10;

  //// Weight bucket for rate estimation
  //static ap_fixed<16, 6> weightbucket = 0;
  //// Init threshol for firing 
  //static ap_fixed<16, 6> threshold = 10;

  // fire vs nofire weight, the rate is the expected rate/40MHz
  // 0.0001 ~ 4kHz
  static const ap_fixed<16, 6> fireweight=1;
  static const ap_fixed<16, 6> nofireweight=-0.0001;

  bucket1 += 1;

  if (AEout[0] > threshold)
    weightbucket += fireweight;
  else
    weightbucket += nofireweight;

  if (bucket1 == N_Maxbucket)
  {
    bucket2 +=1;
    bucket1 = 0;
  }

  if (bucket2 == N_Maxbucket)
  {
    threshold = threshold  + weightbucket * 2;
    bucket2 = 0;
    weightbucket = 0;
  }

  fire[0] = (res_T)( AEout[0] > threshold);
}


#endif   // ----- #ifndef MY_NNET_LOSS_INC  -----
