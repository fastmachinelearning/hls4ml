#ifndef NNET_PRINTF_H_
#define NNET_PRINTF_H_

namespace nnet {

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

using namespace sycl;

#define PRINTF(format, ...)                                                                                                 \
    {                                                                                                                       \
        static const CL_CONSTANT char _format[] = format;                                                                   \
        ext::oneapi::experimental::printf(_format, ##__VA_ARGS__);                                                          \
    }

} // namespace nnet
#endif
