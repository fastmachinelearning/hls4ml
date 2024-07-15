/*
#-  (c) Copyright 2011-2019 Xilinx, Inc. All rights reserved.
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

#ifndef __SIM_AP_SHIFT_REG_H__
#define __SIM_AP_SHIFT_REG_H__

/*
 * This file contains a C++ model of shift register.
 * It defines C level simulation model.
 */
#ifndef __cplusplus
#error C++ is required to include this header file
#else

#ifndef __SYNTHESIS__
#include <cassert>
#endif

//////////////////////////////////////////////
// C level simulation model for ap_shift_reg
//////////////////////////////////////////////
template <typename __SHIFT_T__, unsigned int __SHIFT_DEPTH__ = 32> class ap_shift_reg {
  public:
    /// Constructors
    ap_shift_reg() {
        for (unsigned int i = 0; i < __SHIFT_DEPTH__; i++) {
            __SHIFT_T__ dummy;
            Array[i] = dummy; // uninitialize so Catapult does not add a reset
        }
    }
    ap_shift_reg(const char *name) {}
    /// Destructor
    virtual ~ap_shift_reg() {}

  private:
    /// Make copy constructor and assignment operator private
    ap_shift_reg(const ap_shift_reg<__SHIFT_T__, __SHIFT_DEPTH__> &shreg) {
        for (unsigned i = 0; i < __SHIFT_DEPTH__; ++i)
            Array[i] = shreg.Array[i];
    }

    ap_shift_reg &operator=(const ap_shift_reg<__SHIFT_T__, __SHIFT_DEPTH__> &shreg) {
        for (unsigned i = 0; i < __SHIFT_DEPTH__; ++i)
            Array[i] = shreg.Array[i];
        return *this;
    }

  public:
    // Shift the queue, push to back and read from a given address.
    __SHIFT_T__ shift(__SHIFT_T__ DataIn, unsigned int Addr = __SHIFT_DEPTH__ - 1, bool Enable = true) {
#ifndef __SYNTHESIS__
        assert(Addr < __SHIFT_DEPTH__ && "Out-of-bound shift is found in ap_shift_reg.");
#endif
        __SHIFT_T__ ret = Array[Addr];
        if (Enable) {
            for (unsigned int i = __SHIFT_DEPTH__ - 1; i > 0; --i)
                Array[i] = Array[i - 1];
            Array[0] = DataIn;
        }
        return ret;
    }

    // Read from a given address.
    __SHIFT_T__ read(unsigned int Addr = __SHIFT_DEPTH__ - 1) const {
#ifndef __SYNTHESIS__
        assert(Addr < __SHIFT_DEPTH__ && "Out-of-bound read is found in ap_shift_reg.");
#endif
        return Array[Addr];
    }

  protected:
    __SHIFT_T__ Array[__SHIFT_DEPTH__];
};

#endif //__cplusplus

#endif //__SIM_AP_SHIFT_REG_H__
