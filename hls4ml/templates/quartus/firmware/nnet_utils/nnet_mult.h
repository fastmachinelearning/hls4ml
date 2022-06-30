#ifndef NNET_MULT_H_
#define NNET_MULT_H_

#include "nnet_helpers.h"
#include "nnet_common.h"
#include <math.h>

namespace nnet {

    //  Different methods to perform the product of input and weight, depending on their types.
    namespace product{

        class Product{
            public:
            static void limit(unsigned multiplier_limit) {}
        };

        template<class x_T, class w_T>
        class both_binary : public Product{
            public:
            inline static x_T product(x_T a, w_T w){
                // specialisation for 1-bit weights and incoming data
                return a & w;
            }
        };

        template<class x_T, class w_T>
        class weight_binary : public Product{
            public:
            inline static auto product(x_T a, w_T w) -> decltype(-a)
            {
                // Specialisation for 1-bit weights, arbitrary data
                if (w == 0) return -a;
                else return a;
            }
        };

        template<class x_T, class w_T>
        class data_binary : public Product{
            public:
            inline static auto product(x_T a, w_T w) -> decltype(-w)
            {
                // Specialisation for 1-bit data, arbitrary weight
                if (a == 0) return -w;
                else return w;
            }
        };

        template<class x_T, class w_T>
        class weight_ternary : public Product{
            public:
            inline static auto product(x_T a, w_T w) -> decltype(-a)
            {
                // Specialisation for 2-bit weights, arbitrary data
                if (w == 0) return 0;
                else if(w == -1) return -a;
                else return a; // if(w == 1)
            }
        };

        template<class x_T, class w_T>
        class mult : public Product{
            public:
            inline static auto product(x_T a, w_T w) -> decltype(a*w)
            {
                // 'Normal' product
                return a * w;
            }
            static void limit(unsigned multiplier_limit){
                // TODO: Implement for Quartus
                // #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation > Vivado-only, replace with Intel HLS pragma
            }
        };

        template<class x_T, class w_T>
        class weight_exponential : public Product{
            public:
            // Construct the return type from the multiplication equivalent to the largest shifts
            // ac_int<pow2(decltype(w_T::weight)::width-1)-1, true> is the type if the multiplicand equivalent to the largest lshift <<
            // ac_fixed<pow2(decltype(w_T::weight)::width-1)-1,0, true> is the type of the multiplicand equivalent to the largest rshift >>
            using r_T = decltype(x_T(0) * (ac_int<pow2(decltype(w_T::weight)::width-1)-1, true>(1)+ac_fixed<pow2(decltype(w_T::weight)::width-1)-1,0, true>(1)));
            inline static r_T product(x_T a, w_T w){
                // Shift product for exponential weights
                // shift by the exponent. Negative weights shift right
                r_T y = static_cast<r_T>(a) << w.weight;
                // negate or not depending on weight sign
                return w.sign == 1 ? y : static_cast<r_T>(-y);
            }
        };
    }   // namespace product_type

    template<class data_T, class res_T, typename CONFIG_T>
    inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value
            && std::is_same<typename CONFIG_T::weight_t, ac_int<1, false>>::value, ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>>::type
    cast(typename CONFIG_T::accum_t x)
    {
        return (ac_int<nnet::ceillog2(CONFIG_T::n_in) + 2, true>) (x - CONFIG_T::n_in / 2) * 2;
    }

    template<class data_T, class res_T, typename CONFIG_T>
    inline typename std::enable_if<std::is_same<data_T, ac_int<1, false>>::value
        && ! std::is_same<typename CONFIG_T::weight_t, ac_int<1, false>>::value, res_T>::type
    cast(typename CONFIG_T::accum_t x)
    {
        return (res_T) x;
    }

    template<class data_T, class res_T, typename CONFIG_T>
    inline typename std::enable_if<(! std::is_same<data_T, ac_int<1, false>>::value), res_T>::type
    cast(typename CONFIG_T::accum_t x)
    {
        return (res_T) x;
    }

}

#endif
