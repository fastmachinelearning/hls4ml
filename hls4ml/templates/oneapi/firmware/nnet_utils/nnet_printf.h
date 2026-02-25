#ifndef NNET_PRINTF_H_
#define NNET_PRINTF_H_

#ifdef __SYCL_DEVICE_ONLY__
#define CL_CONSTANT __attribute__((opencl_constant))
#else
#define CL_CONSTANT
#endif

#define PRINTF(format, ...)                                                                                                 \
    {                                                                                                                       \
        static const CL_CONSTANT char _format[] = format;                                                                   \
        sycl::ext::oneapi::experimental::printf(_format, ##__VA_ARGS__);                                                    \
    }

#endif
