// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

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
#include <unordered_map>
#include <cstring>
#include <array>
#include <limits>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <condition_variable>

#ifndef _MSC_VER
#include <cxxabi.h>
#include <stdlib.h>
#endif

namespace hls {
#if !defined(__HLS_COSIM__) && defined(__VITIS_HLS__)
// We are in bcsim mode, where reads must be non-blocking
#define ALLOW_EMPTY_HLS_STREAM_READS
#ifdef X_HLS_TASK_H
#error "bcsim is not supported with hls::tasks"
#endif
#endif

template<size_t SIZE>
class stream_delegate {
public:
  virtual bool read(void *elem) = 0;
  virtual void write(const void *elem) = 0;
  virtual bool read_nb(void *elem) = 0;
  virtual size_t size() = 0;
};

class stream_globals {
public:
  static void print_max_size() {
#ifndef DISABLE_MAX_HLS_STREAM_DEPTH_PRINT
    std::cout << "INFO [HLS SIM]: The maximum depth reached by any hls::stream() instance in the design is " << get_max_size() << std::endl;
#endif
  }

  static void incr_blocked_counter() {
    get_blocked_counter()++;
  }

  static void decr_blocked_counter() {
    get_blocked_counter()--;
  }

  static void incr_task_counter() {
    get_task_counter()++;
  }

  static void decr_task_counter() {
    get_task_counter()--;
  }

  static void start_threads() {
    // These initializations must be in ONE static function that is called elsewhere
#if defined(__HLS_COSIM__) 
    static std::thread t(deadlock_thread);
#endif
    static std::atomic_flag init_done = ATOMIC_FLAG_INIT;
    if (!init_done.test_and_set()) {
      // Perform global initialization actions once
      // Register function executed at exit
      std::atexit(print_max_size);

#if defined(__HLS_COSIM__) 
      // Detach the thread to avoid error at end with unwaited thread
      t.detach();
#endif
    }
  }

  static std::atomic<int> &get_max_size() {
    static std::atomic<int> max_size(0);

    return max_size;
  }

private:
  static bool check_deadlock() {
  // Check that it is larger than, because the testbench main thread is not counted.
    return get_blocked_counter() > get_task_counter();
  }

#ifndef HLS_STREAM_THREAD_UNSAFE
  static std::mutex &get_mutex() {
      static std::mutex mutex;

      return mutex;
  }
#endif

  static void deadlock_thread() {
    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      if (stream_globals::check_deadlock()) {
        if (get_task_counter()) {
            std::cout << "ERROR [HLS SIM]: deadlock detected when simulating hls::tasks." 
                    << std::endl;
            std::cout << "Execute C simulation in debug mode in the GUI and examine the"
                    << " source code location of all the blocked hls::stream::read()"
                    << " calls to debug." << std::endl;
        } else {
            std::cout << "ERROR [HLS SIM]: an hls::stream is read while empty,"
                    << " which may result in RTL simulation hanging." << std::endl;
            std::cout << "If this is not expected, execute C simulation in debug mode in"
                    << " the GUI and examine the source code location of the blocked"
                    << " hls::stream::read() call to debug." << std::endl;
            std::cout << "If this is expected, add -DALLOW_EMPTY_HLS_STREAM_READS"
                    << " to -cflags to turn this error into a warning and allow empty"
                    << " hls::stream reads to return the default value for the data type."
                    << std::endl;
        }
        abort();
      }
    }
  }

  static std::atomic<int> &get_task_counter() {
      static std::atomic<int> task_counter(0);

      return task_counter;
  }

  static std::atomic<int> &get_blocked_counter() {
      static std::atomic<int> blocked_counter(0);

      return blocked_counter;
  }
};

template<size_t SIZE>
class stream_entity {
public:
#ifdef HLS_STREAM_THREAD_UNSAFE
  stream_entity() : d(0) {}
#else
  stream_entity() : d(0), invalid(false) {}
  ~stream_entity() {
    std::unique_lock<std::mutex> ul(mutex);
    invalid = true;
    condition_var.notify_all();
  }
#endif

  bool read(void *elem) {
    if (d)
      return d->read(elem);

#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    // needed to start the deadlock detector and size reporter
    stream_globals::start_threads();

    if (data.empty()) { 
#ifdef ALLOW_EMPTY_HLS_STREAM_READS
      std::cout << "WARNING [HLS SIM]: hls::stream '"
                << name
                << "' is read while empty,"
                << " which may result in RTL simulation hanging."
                << std::endl;
      return false;
#else
        stream_globals::incr_blocked_counter();
        while (data.empty()) {
#ifndef HLS_STREAM_THREAD_UNSAFE
          while (invalid) { 
            std::this_thread::sleep_for(std::chrono::seconds(1));
          }
          condition_var.wait(ul);
#endif
        }
        stream_globals::decr_blocked_counter();
#endif
    }
    std::array<char, SIZE> &elem_data = data.front();
    memcpy(elem, elem_data.data(), SIZE);
    data.pop_front();
    return true;
  }

  void write(const void *elem) {
    if (d) {
      d->write(elem);
      return;
    }

    std::array<char, SIZE> elem_data;
    memcpy(elem_data.data(), elem, SIZE);

#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    data.push_back(elem_data);
    
    // needed to start the deadlock detector and size reporter
    stream_globals::start_threads();
    
    if (stream_globals::get_max_size() < data.size()) 
        stream_globals::get_max_size() = data.size();
#ifndef HLS_STREAM_THREAD_UNSAFE
    condition_var.notify_one();
#endif
  }

  /// Nonblocking read
  bool read_nb(void *elem) {
    if (d)
      return d->read_nb(elem);

#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(mutex);
#endif
    bool is_empty = data.empty();
    if (!is_empty) {
      std::array<char, SIZE> &elem_data = data.front();
      memcpy(elem, elem_data.data(), SIZE);
      data.pop_front();
    }
    return !is_empty; 
  }

  /// Fifo size
  size_t size() {
    if (d)
      return d->size();

#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(mutex);
#endif
    return data.size();
  }

  /// Set name for c-sim debugging.
  void set_name(const char *n) {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(mutex);
#endif
    name = n;
  }

  stream_delegate<SIZE> *d;
  std::string name;
  std::deque<std::array<char, SIZE> > data;
#ifndef HLS_STREAM_THREAD_UNSAFE
  std::mutex mutex;
  std::condition_variable condition_var;
  bool invalid;
#endif
};

template<size_t SIZE>
class stream_map {
public:
  static size_t count(void *p) {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(get_mutex());
#endif
    return get_map().count(p);
  }

  static void insert(void *p) {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(get_mutex());
#endif
    auto &map = get_map();
    map.erase(p);
    map[p];
  }

  static stream_entity<SIZE> &get_entity(void *p) {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::lock_guard<std::mutex> lg(get_mutex());
#endif
    return get_map()[p];
  }

private:
#ifndef HLS_STREAM_THREAD_UNSAFE
  static std::mutex &get_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
  }
#endif
  static std::unordered_map<void*, stream_entity<SIZE> > &get_map() {
    static std::unordered_map<void*, stream_entity<SIZE> > *map = 
        new std::unordered_map<void*, stream_entity<SIZE> >();
    return *map;
  }
};

template<typename __STREAM_T__, int DEPTH=0>
class stream;
template<typename __STREAM_T__>
class stream<__STREAM_T__, 0> 
{
  public:
    using value_type = __STREAM_T__;

  private:
  typedef stream_map<sizeof(__STREAM_T__)> map_t;

  protected:
#if defined(__VITIS_HLS__)
    __STREAM_T__ _data;
#else
    stream_entity<sizeof(__STREAM_T__)> _data;
#endif

  protected:
  public:
    /// Constructors
    // Keep consistent with the synthesis model's constructors
    stream() {
      std::stringstream ss;
#ifndef _MSC_VER
      char* _demangle_name = abi::__cxa_demangle(typeid(*this).name(), 0, 0, 0);
      if (_demangle_name) {
        ss << _demangle_name;
        free(_demangle_name);
      }
      else {
        ss << "hls_stream";
      }
#else
      ss << typeid(*this).name();
#endif

#ifdef HLS_STREAM_THREAD_UNSAFE
      static unsigned counter = 0;
#else
      static std::atomic<unsigned> counter(0);
#endif

#if defined(__VITIS_HLS__)
      map_t::insert(&_data);
#endif
      ss << counter++;
      get_entity().set_name(ss.str().c_str());
    }

    stream(const char *name) {
    // default constructor,
    // capacity set to predefined maximum
#if defined(__VITIS_HLS__)
      map_t::insert(&_data);
#endif
      get_entity().set_name(name);
    }

  /// Make copy constructor and assignment operator private
  /// They should not be called.
  private:
    stream(const stream< __STREAM_T__ >& chn):
        _data(chn._data) {
    }

    stream& operator = (const stream< __STREAM_T__ >& chn) {
        return *this;
    }

    stream_entity<sizeof(__STREAM_T__)> &get_entity() {
#if defined(__VITIS_HLS__)
      return map_t::get_entity(&_data);
#else
      return _data;
#endif
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
    ~stream() {
      if (!empty())
      {
          std::cout << "WARNING [HLS SIM]: hls::stream '" 
                    << get_entity().name
                    << "' contains leftover data,"
                    << " which may result in RTL simulation hanging."
                    << std::endl;
      }
    }

#if defined(__VITIS_HLS__)
    bool exist() {
      return map_t::count(&_data);
    }
#endif

    /// Status of the queue
    bool empty() {
      return size() == 0;
    }    

    bool full() const { return false; }

    /// Blocking read
    void read(__STREAM_T__& head) {
        head = read();
    }

    /// Blocking read
    bool read_dep(__STREAM_T__& head, volatile bool flag) {
        head = read();
        return flag;
    }

    __STREAM_T__ read() {
      __STREAM_T__ elem;
      auto &entity = get_entity();
      if (!entity.read(&elem)) 
        elem = __STREAM_T__();
      return elem;
    }

    /// Blocking write
    void write(const __STREAM_T__& tail) { 
      get_entity().write(&tail);
    }

    /// Blocking write
    bool write_dep(const __STREAM_T__& tail, volatile bool flag) { 
      write(tail);
      return flag;
    }

    /// Nonblocking read
    bool read_nb(__STREAM_T__& head) {
      __STREAM_T__ elem;
      auto &entity = get_entity();
      bool not_empty = entity.read_nb(&elem);
      if (not_empty)
        head = elem;
      return not_empty;
    }

    /// Nonblocking write
    bool write_nb(const __STREAM_T__& tail) {
        bool is_full = full();
        write(tail);
        return !is_full;
    }

    /// Fifo size
    size_t size() {
      return get_entity().size();
    }

    /// Fifo capacity
    size_t capacity() {
        // actually no limit on simulation model
        return std::numeric_limits<std::size_t>::max();
    }

    /// Set name for c-sim debugging.
    void set_name(const char *name) { 
      get_entity().set_name(name);
    }

    void set_delegate(stream_delegate<sizeof(__STREAM_T__)> *d) {
      get_entity().d = d;
    }
};

template<typename __STREAM_T__, int DEPTH>
class stream : public stream<__STREAM_T__, 0> {
public:
  stream() {}
  stream(const char* name) : stream<__STREAM_T__, 0>(name) {}
};

} // namespace hls

#endif // __cplusplus
#endif  // X_HLS_STREAM_H


