#ifndef NNET_POOLING_H_
#define NNET_POOLING_H_

#include <iostream>

namespace nnet{

// Return the maximum value from an array
template<typename T, int N>
T max(T x[N]){
  T y = x[0];
  for(int i = 1; i < N; i++){
    y = x[i] > y ? x[i] : y;
  }
  return y;
}

// Return the mean value of an array
template<typename T, int N>
T avg(T x[N]){
  T y = 0;
  for(int i = 0; i < N; i++){
    y += x[i];
  }
  y /= N;
  return y;
}

/*template<typename T, int N>
T l2norm(T x[N]){
	T y = 0;
	for(int i = 0; i < N; i++){
		y += x[i] * x[i];
	}
	y = sqrt(y);
	return y;
}*/

// Enumeration for pooling operation (max, avg, l2norm pooling)
enum Pool_Op { Max, Average }; // L2Norm };
template<typename T, int N, Pool_Op op>
T pool_op(T x[N]){
	switch(op){
	case Max: return max<T, N>(x);
	case Average: return avg<T, N>(x);
	// case L2Norm: return l2norm<T, N>(x);
	}
}

template<typename T, Pool_Op op>
T pad_val(){
  /*---
  *- In Tensorflow, pooling ignores the value in the padded cells
  *- For Avg pooling, return 0 (the divisior is modified to the
  *- area overlapping the unpadded image.
  *- For max pooling, return the most negative value for the type.
  *- TODO this is not really generic, it assumes fixed point or integer T
  ---*/
  switch(op){
    case Max:{ 
      T x = 0;
      x[x.width - 1] = 1;
      return x;
      break;}
    case Average: return 0;
  }
}

struct pooling1d_config{
  // IO size
  static const unsigned n_in = 10;
  static const unsigned pool_size = 2;
  static const unsigned n_out = n_in / pool_size;
  static const unsigned pad_left = 0;
  static const unsigned pad_right = 0;
  // Pooling function
  static const Pool_Op pool_op = Max;
};

template<class data_T, typename CONFIG_T>
void pooling1d(data_T data[CONFIG_T::n_in], data_T res[CONFIG_T::n_out]){
  for(int ii = 0; ii < CONFIG_T::n_out; ii ++){
    data_T pool[CONFIG_T::pool_size];
    for(int jj = 0; jj < CONFIG_T::pool_size; jj++){
      pool[jj] = data[ii * CONFIG_T::pool_size + jj]; 
    }
    res[ii] = pool_op<data_T, CONFIG_T::pool_size, CONFIG_T::pool_op>(pool);
  }
}

struct pooling2d_config{
  // IO size
  static const unsigned in_height = 10;
  static const unsigned in_width = 10;
  static const unsigned n_filt = 4;
  static const unsigned stride_height = 2;
  static const unsigned stride_width = 2;
  static const unsigned pool_height = 2;
  static const unsigned pool_width = 2;
  static const unsigned out_height = (in_height - pool_height) / stride_height + 1;
  static const unsigned out_width = (in_width - pool_width) / stride_width + 1;
  // Padding
  static const unsigned pad_top = 0;
  static const unsigned pad_bottom = 0;
  static const unsigned pad_left = 0;
  static const unsigned pad_right = 0;
  // Pooling function
  static const Pool_Op pool_op = Max;
};

template<class data_T, typename CONFIG_T>
void pooling2d(data_T data[CONFIG_T::in_height][CONFIG_T::in_width][CONFIG_T::n_filt],
               data_T res[CONFIG_T::out_height][CONFIG_T::out_width][CONFIG_T::n_filt]){

  // Add any necessary padding
  const unsigned padded_height = CONFIG_T::in_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom;
  const unsigned padded_width = CONFIG_T::in_width + CONFIG_T::pad_left + CONFIG_T::pad_right;

  for(int ff = 0; ff < CONFIG_T::n_filt; ff++){
	  // Loop over input image y in steps of stride
	  for(int ii = 0; ii < padded_height; ii += CONFIG_T::stride_height){
		  // Loop over input image x in steps of stride
		  for(int jj = 0; jj < padded_width; jj += CONFIG_T::stride_width){
			  data_T pool[CONFIG_T::pool_height * CONFIG_T::pool_width];
        // Keep track of number of pixels in image vs pading region
        unsigned img_overlap = 0;
			  // Loop over pool window y
			  for(int kk = 0; kk < CONFIG_T::stride_height; kk++){
				  // Loop over pool window x
				  for(int ll = 0; ll < CONFIG_T::stride_width; ll++){
            if(ii+kk < CONFIG_T::pad_top || ii+kk >= (padded_height - CONFIG_T::pad_bottom) || jj+ll < CONFIG_T::pad_left || jj+ll >= (padded_width - CONFIG_T::pad_right)){
              // Add padding
              pool[kk * CONFIG_T::stride_width + ll] = pad_val<data_T, CONFIG_T::pool_op>();
            }else{
  					  pool[kk * CONFIG_T::stride_width + ll] = data[ii + kk][jj + ll][ff];
              img_overlap++;
            }
				  }
			  }
			  // do the pooling
        // TODO in the case of average pooling, need to reduce height * width to area of pool window
        // not overlapping padding region
			  res[ii/CONFIG_T::stride_height][jj/CONFIG_T::stride_width][ff] =
					  pool_op<data_T, CONFIG_T::pool_height*CONFIG_T::pool_width, CONFIG_T::pool_op>(pool);
        // If the pool op is Average, the zero-padding needs to be removed from the results
        if(CONFIG_T::pool_op == Pool_Op::Average){
          res[ii/CONFIG_T::stride_height][jj/CONFIG_T::stride_width][ff] *= CONFIG_T::pool_height * CONFIG_T::pool_width / img_overlap;
        }
		  }
	  }
  }
}

}

#endif
