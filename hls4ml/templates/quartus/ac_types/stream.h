#ifndef NNET_STREAM_H
#define NNET_STREAM_H

#include <deque>

namespace nnet {

/*
* A struct with the same high-level functionality as Intel's HLS ihc::stream
* This struct is used during GCC compilation / hls4ml model.predict(...)
* This is because GCC does not have access to HLS source files (ihc::stream)
* Software-wise, this struct behaves like a first-in, first-out (FIFO) buffer
* However, it cannot be used for HLS synthesis, since it uses dynamic memory allocation (deque)
*/
template<typename T>
struct stream {
  private:
    std::deque<T> _data;

  public:
    stream() {}

    T read() {
        T element = _data.front();
        _data.pop_front();
        return element; 
    }

    void write(const T& element) { 
        _data.push_back(element);
    }   
};

}
 
#endif