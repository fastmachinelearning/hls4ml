// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689

#ifndef X_HLS_STREAMOFBLOCKS_H
#define X_HLS_STREAMOFBLOCKS_H

#ifndef __cplusplus
#error C++ is required to include this header file
#else

#include <queue>
#include <iostream>
#include <typeinfo>
#include <string>
#include <sstream>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "hls_stream.h"

#define ALWAYS_INLINE inline 
#define UNUSED_ARG 

#ifndef __has_attribute
#define __has_attribute(x) 0  // Compatibility with non-clang compilers.
#endif

#if __has_attribute(no_ctor)
#define NO_CTOR __attribute__((no_ctor))
#else
#define NO_CTOR
#endif

namespace hls {
template <typename __STREAM_T__>
class stream_buf {
  std::string name;
  std::deque<__STREAM_T__*> data;
  int readLocks;
  int writeLocks;
 
 public:
  ALWAYS_INLINE stream_buf(int depth, const char *n)
    : name(n ? n : "stream_of_blocks"), readLocks(0), 
      writeLocks(0), invalid(false){}

  ~stream_buf() {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
    invalid = true;
    condition_var.notify_all();
#endif
  }

  void write_check() {
    if (writeLocks <= 0) {
        std::cerr << "ERROR: writing " << name << " before acquiring." << std::endl;
        abort();
    }
  }
  void read_check() {
    if (readLocks <= 0) {
        std::cerr << "ERROR: reading " << name << " before acquiring." << std::endl;
        abort();
    }
  }

  ALWAYS_INLINE __STREAM_T__& read_acquire() {
    // needed to start the deadlock detector and size reporter
    stream_globals::start_threads();

    if (readLocks > 0) {
        std::cerr << "ERROR: acquiring " << name << " for reading more than once before releasing. Use braces to limit the lifetime of the lock object." << std::endl;
        abort();
    }
    readLocks++;
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    if (!data.size()) { 
#ifdef ALLOW_EMPTY_HLS_STREAM_READS
      std::cout << "WARNING [HLS SIM]: hls::stream_of_blocks '"
                << name
                << "' is read while empty,"
                << " which may result in RTL simulation hanging."
                << std::endl;
        return *new __STREAM_T__[1];
#else
        stream_globals::incr_blocked_counter();
        while (!data.size()) {
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
    return *data.front();
  }
 
  ALWAYS_INLINE void read_release() {
    readLocks--;
    if (readLocks != 0) {
        std::cerr << "INTERNAL ERROR: releasing " << name << " for reading too many times." << std::endl;
        abort();
    }
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    data.pop_front();
  }
 
  ALWAYS_INLINE __STREAM_T__& write_acquire() {
    // needed to start the deadlock detector and size reporter
    stream_globals::start_threads();

    if (writeLocks > 0) {
        std::cerr << "ERROR: acquiring " << name << " for writing more than once before releasing. Use braces to limit the lifetime of the lock object." << std::endl;
        abort();
    }
    writeLocks++;

#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    data.push_back(new __STREAM_T__[1]);
#ifndef HLS_STREAM_THREAD_UNSAFE
    condition_var.notify_one();
#endif
    return *data.back();
  }

  ALWAYS_INLINE void write_release() {
    writeLocks--;
    if (writeLocks != 0) {
        std::cerr << "INTERNAL ERROR: releasing " << name << " for writing too many times." << std::endl;
        abort();
    }
  }
 
  ALWAYS_INLINE bool empty() {
#ifndef HLS_STREAM_THREAD_UNSAFE
    std::unique_lock<std::mutex> ul(mutex);
#endif
    return !data.size();
  }

  ALWAYS_INLINE bool full() {
    return false;
  }

#ifdef EXPLICIT_ACQUIRE_RELEASE
  template <typename>
  friend class read_buf;
  template <typename>
  friend class write_buf;
#endif

  template <typename>
  friend class read_lock;
  template <typename>
  friend class write_lock;
#ifndef HLS_STREAM_THREAD_UNSAFE
  std::mutex mutex;
  std::condition_variable condition_var;
  bool invalid;
#endif
};
 
template<typename __STREAM_T__, int DEPTH=2>
class stream_of_blocks;

#ifdef EXPLICIT_ACQUIRE_RELEASE
template <typename __STREAM_T__>
class read_buf {
  stream_of_blocks<__STREAM_T__>& res;
  __STREAM_T__* buf;
 
 public:
  ALWAYS_INLINE read_buf(stream_of_blocks<__STREAM_T__>& s) : res(s) {
  }

  ALWAYS_INLINE void acquire() {
    buf = &res.read_acquire();
  }

  ALWAYS_INLINE void release() {
    res.read_release();
  }

  ALWAYS_INLINE ~read_buf() { }

  ALWAYS_INLINE operator __STREAM_T__&() { 
    res.buf.read_check();
    return *buf;
  }

  ALWAYS_INLINE __STREAM_T__& operator=(const __STREAM_T__& val) { 
    res.buf.read_check();
    buf = &val; 
    return *buf; 
  }
};
 
template <typename __STREAM_T__>
class write_buf {
  stream_of_blocks<__STREAM_T__>& res;
  __STREAM_T__* buf;
 
 public:
  ALWAYS_INLINE write_buf(stream_of_blocks<__STREAM_T__>& s) : res(s) {
  }

  ALWAYS_INLINE void acquire() {
    buf = &res.write_acquire();
  }

  ALWAYS_INLINE void release() {
    res.write_release();
  }

  ALWAYS_INLINE ~write_buf() { }

  ALWAYS_INLINE operator __STREAM_T__&() { 
    res.buf.write_check();
    return *buf; 
  }

  ALWAYS_INLINE __STREAM_T__& operator=(const __STREAM_T__& val) {
    res.buf.write_check();
    buf = &val; 
    return *buf; 
  }
};
#endif

template <typename __STREAM_T__>
class read_lock {
  stream_of_blocks<__STREAM_T__>& res;
  __STREAM_T__& buf;
 
 public:
  ALWAYS_INLINE read_lock(stream_of_blocks<__STREAM_T__>& s) : res(s), buf(res.read_acquire()) { }

  ALWAYS_INLINE ~read_lock() { res.read_release(); }

  ALWAYS_INLINE operator __STREAM_T__&() { return buf; }

  ALWAYS_INLINE __STREAM_T__& operator=(const __STREAM_T__& val) { return buf = val; }
};
 
template <typename __STREAM_T__>
class write_lock {
  stream_of_blocks<__STREAM_T__>& res;
  __STREAM_T__& buf;
 
 public:
  ALWAYS_INLINE write_lock(stream_of_blocks<__STREAM_T__>& s) : res(s), buf(res.write_acquire()) { }

  ALWAYS_INLINE ~write_lock() { res.write_release(); }

  ALWAYS_INLINE operator __STREAM_T__&() { return buf; }

  ALWAYS_INLINE __STREAM_T__& operator=(const __STREAM_T__& val) { return buf = val; }
};
 
template <typename __STREAM_T__>
class stream_of_blocks<__STREAM_T__, 2> {
#ifdef EXPLICIT_ACQUIRE_RELEASE
  friend class read_buf<__STREAM_T__>; 
  friend class write_buf<__STREAM_T__>; 
#endif
  friend class read_lock<__STREAM_T__>; 
  friend class write_lock<__STREAM_T__>; 

  stream_buf<__STREAM_T__> buf;
 
 public:
  ALWAYS_INLINE stream_of_blocks(int depth=2, UNUSED_ARG char *name=0): buf(depth, name) { }

  ALWAYS_INLINE bool full() { return buf.full(); }

  ALWAYS_INLINE bool empty() { return buf.empty(); }

 private:
  ALWAYS_INLINE __STREAM_T__& read_acquire() { return buf.read_acquire(); }

  ALWAYS_INLINE void read_release() { buf.read_release(); }
 
  ALWAYS_INLINE __STREAM_T__& write_acquire() { return buf.write_acquire(); }

  ALWAYS_INLINE void write_release() { buf.write_release(); }

  //__STREAM_T__& read(); TBD
  //void write(const __STREAM_T__&); TBD
};

template <typename __STREAM_T__, int DEPTH>
class stream_of_blocks: public stream_of_blocks<__STREAM_T__, 2> {
#ifdef EXPLICIT_ACQUIRE_RELEASE
  friend class read_buf<__STREAM_T__>; 
  friend class write_buf<__STREAM_T__>; 
#endif
  friend class read_lock<__STREAM_T__>; 
  friend class write_lock<__STREAM_T__>; 

 public:
  ALWAYS_INLINE stream_of_blocks(): stream_of_blocks<__STREAM_T__, 2>(DEPTH) {}
};

} // end of namespace hls
#endif
#endif
