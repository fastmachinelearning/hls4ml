#ifndef HLS4ML_SYCL_H_
#define HLS4ML_SYCL_H_

#include <sycl/sycl.hpp>

#ifdef HLS4ML_ONEAPI
#include <sycl/ext/intel/ac_types/ac_fixed.hpp>
#include <sycl/ext/intel/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

#if __INTEL_CLANG_COMPILER < 20250000
#include <sycl/ext/intel/prototype/interfaces.hpp>
#endif

namespace hls4ml_sycl_ext = sycl::ext::intel;
#else
#include <sycl/ext/altera/ac_types/ac_fixed.hpp>
#include <sycl/ext/altera/ac_types/ac_fixed_math.hpp>
#include <sycl/ext/altera/ac_types/ac_int.hpp>
#include <sycl/ext/altera/experimental/task_sequence.hpp>
#include <sycl/ext/altera/fpga_extensions.hpp>

namespace hls4ml_sycl_ext = sycl::ext::altera;
#endif

#endif
