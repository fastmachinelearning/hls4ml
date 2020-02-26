/*
#-  (c) Copyright 2011-2018 Xilinx, Inc. All rights reserved.
#-
#-  This file contains confidential and proprietary information
#-  of Xilinx, Inc. and is protected under U.S. and
#-  international copyright and other intellectual property
#-  laws.
#-
#-  DISCLAIMER
#-  This disclaimer is not a license and does not grant any
#-  rights to the materials distributed herewith. Except as
#-  otherwise provided in a valid license issued to you by
#-  Xilinx, and to the maximum extent permitted by applicable
#-  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
#-  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
#-  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
#-  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
#-  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
#-  (2) Xilinx shall not be liable (whether in contract or tort,
#-  including negligence, or under any other theory of
#-  liability) for any loss or damage of any kind or nature
#-  related to, arising under or in connection with these
#-  materials, including for any direct, or any indirect,
#-  special, incidental, or consequential loss or damage
#-  (including loss of data, profits, goodwill, or any type of
#-  loss or damage suffered as a result of any action brought
#-  by a third party) even if such damage or loss was
#-  reasonably foreseeable or Xilinx had been advised of the
#-  possibility of the same.
#-
#-  CRITICAL APPLICATIONS
#-  Xilinx products are not designed or intended to be fail-
#-  safe, or for use in any application requiring fail-safe
#-  performance, such as life-support or safety devices or
#-  systems, Class III medical devices, nuclear facilities,
#-  applications related to the deployment of airbags, or any
#-  other applications that could lead to death, personal
#-  injury, or severe property or environmental damage
#-  (individually and collectively, "Critical
#-  Applications"). Customer assumes the sole risk and
#-  liability of any use of Xilinx products in Critical
#-  Applications, subject only to applicable laws and
#-  regulations governing limitations on product liability.
#-
#-  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
#-  PART OF THIS FILE AT ALL TIMES. 
#- ************************************************************************


   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef X_HLS_STREAM_SIM_H
#define X_HLS_STREAM_SIM_H

/*
 * This file contains a C++ model of hls::stream.
 * It defines C simulation model.
 */
#ifndef __cplusplus

#error C++ is required to include this header file

#else

//////////////////////////////////////////////
// C level simulation models for hls::stream
//////////////////////////////////////////////
#include <queue>
#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>

#ifdef HLS_STREAM_THREAD_SAFE
#include <mutex>
#include <condition_variable>
#endif

#ifndef _MSC_VER
#include <cxxabi.h>
#include <stdlib.h>
#endif

namespace hls {

template<typename __STREAM_T__>
class stream
{
  protected:
    std::string _name;
    std::deque<__STREAM_T__> _data; // container for the elements
#ifdef HLS_STREAM_THREAD_SAFE
    std::mutex _mutex;
    std::condition_variable _condition_var;
#endif    

  public:
    /// Constructors
    // Keep consistent with the synthesis model's constructors
    stream() {
        static unsigned _counter = 1;
        std::stringstream ss;
#ifndef _MSC_VER
        char* _demangle_name = abi::__cxa_demangle(typeid(*this).name(), 0, 0, 0);
        if (_demangle_name) {
            _name = _demangle_name;
            free(_demangle_name);
        }
        else {
            _name = "hls_stream";
        }
#else
        _name = typeid(*this).name();
#endif

        ss << _counter++;
        _name += "." + ss.str();
    }

    stream(const std::string name) {
    // default constructor,
    // capacity set to predefined maximum
        _name = name;
    }

  /// Make copy constructor and assignment operator private
  private:
    stream(const stream< __STREAM_T__ >& chn):
        _name(chn._name), _data(chn._data) {
    }

    stream& operator = (const stream< __STREAM_T__ >& chn) {
        _name = chn._name;
        _data = chn._data;
        return *this;
    }

  public:
    /// Overload >> and << operators to implement read() and write()
    void operator >> (__STREAM_T__& rdata) {
        read(rdata);
    }

    void operator << (const __STREAM_T__& wdata) {
        write(wdata);
    }


  public:
    /// Destructor
    /// Check status of the queue
    virtual ~stream() {
        if (!_data.empty())
        {
            std::cout << "WARNING: Hls::stream '" 
                      << _name 
                      << "' contains leftover data,"
                      << " which may result in RTL simulation hanging."
                      << std::endl;
        }
    }

    /// Status of the queue
    bool empty() {
#ifdef HLS_STREAM_THREAD_SAFE
        std::lock_guard<std::mutex> lg(_mutex);
#endif
        return _data.empty();
    }    

    bool full() const { return false; }

    /// Blocking read
    void read(__STREAM_T__& head) {
        head = read();
    }

#ifdef HLS_STREAM_THREAD_SAFE
    __STREAM_T__ read() {
        std::unique_lock<std::mutex> ul(_mutex);
        while (_data.empty()) {
            _condition_var.wait(ul);
        }

        __STREAM_T__ elem;
        elem = _data.front();
        _data.pop_front();
        return elem;
    }
#else
    __STREAM_T__ read() {
        __STREAM_T__ elem;
        if (_data.empty()) {
            std::cout << "WARNING: Hls::stream '"
                      << _name 
                      << "' is read while empty,"
                      << " which may result in RTL simulation hanging."
                      << std::endl;
            elem = __STREAM_T__();
        } else {
            elem = _data.front();
            _data.pop_front();
        }
        return elem;
    }
#endif

    /// Blocking write
    void write(const __STREAM_T__& tail) { 
#ifdef HLS_STREAM_THREAD_SAFE
        std::unique_lock<std::mutex> ul(_mutex);
#endif
        _data.push_back(tail);
#ifdef HLS_STREAM_THREAD_SAFE
        _condition_var.notify_one();
#endif
    }

    /// Nonblocking read
    bool read_nb(__STREAM_T__& head) {
#ifdef HLS_STREAM_THREAD_SAFE
        std::lock_guard<std::mutex> lg(_mutex);
#endif    
        bool is_empty = _data.empty();
        if (is_empty) {
            head = __STREAM_T__();
        } else {
            __STREAM_T__ elem(_data.front());
            _data.pop_front();
            head = elem;
        }
        return !is_empty;
    }

    /// Nonblocking write
    bool write_nb(const __STREAM_T__& tail) {
        bool is_full = full();
        write(tail);
        return !is_full;
    }

    /// Fifo size
    size_t size() {
        return _data.size();
    }
};

} // namespace hls

#endif // __cplusplus
#endif  // X_HLS_STREAM_SIM_H

