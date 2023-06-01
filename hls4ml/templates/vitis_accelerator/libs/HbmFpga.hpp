#pragma once

#include "FpgaObj.hpp"

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {
    PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),  PC_NAME(7),
    PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13), PC_NAME(14), PC_NAME(15),
    PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20), PC_NAME(21), PC_NAME(22), PC_NAME(23),
    PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27), PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

template <class V, class W>
class HbmFpga : public FpgaObj<V, W> {
 public:
    HbmFpga(int kernInputSize, int kernOutputSize, int numCU, int numThreads, int numEpochs)
        : FpgaObj<V, W>(kernInputSize, kernOutputSize, numCU, numThreads, numEpochs) {
    }

    void allocateHostMemory(int chan_per_port) {
        // Create Pointer objects for the ports for each virtual compute unit
        // Assigning Pointers to specific HBM PC's using cl_mem_ext_ptr_t type and corresponding PC flags
        for (int ib = 0; ib < this->_numThreads; ib++) {
            for (int ik = 0; ik < this->_numCU; ik++) {
                cl_mem_ext_ptr_t buf_in_ext_tmp;
                buf_in_ext_tmp.obj = this->source_in.data() + ((ib*this->_numCU + ik) * this->_kernInputSize);
                buf_in_ext_tmp.param = 0;
                int in_flags = 0;
                for (int i = 0; i < chan_per_port; i++) {
                    in_flags |= pc[(ik * 2 * 4) + i];
                }
                buf_in_ext_tmp.flags = in_flags;
                
                this->buf_in_ext.push_back(buf_in_ext_tmp);

                cl_mem_ext_ptr_t buf_out_ext_tmp;
                buf_out_ext_tmp.obj = this->source_hw_results.data() + ((ib*this->_numCU + ik) * this->_kernOutputSize);
                buf_out_ext_tmp.param = 0;
                int out_flags = 0;
                for (int i = 0; i < chan_per_port; i++) {
                    out_flags |= pc[(ik * 2 * 4) + chan_per_port + i];
                }
                buf_out_ext_tmp.flags = out_flags;
                
                this->buf_out_ext.push_back(buf_out_ext_tmp);
            }
        }

        // Creating Buffer objects in Host memory
        /* ***NOTE*** When creating a Buffer with user pointer (CL_MEM_USE_HOST_PTR), under the hood, user pointer
        is used if it is properly aligned. when not aligned, runtime has no choice but to create
        its own host side Buffer. So it is recommended to use this allocator if user wishes to
        create Buffer using CL_MEM_USE_HOST_PTR to align user buffer to page boundary. It will 
        ensure that user buffer is used when user creates Buffer/Mem object with CL_MEM_USE_HOST_PTR */
        size_t vector_size_in_bytes = sizeof(V) * this->_kernInputSize;
        size_t vector_size_out_bytes = sizeof(W) * this->_kernOutputSize;
        for (int ib = 0; ib < this->_numThreads; ib++) {
            for (int ik = 0; ik < this->_numCU; ik++) {
                cl::Buffer buffer_in_tmp (this->context, 
                        CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY,
                        vector_size_in_bytes,
                        &(this->buf_in_ext[ib*this->_numCU + ik]));
                cl::Buffer buffer_out_tmp(this->context,
                        CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY,
                        vector_size_out_bytes,
                        &(this->buf_out_ext[ib*this->_numCU + ik]));
                this->buffer_in.push_back(buffer_in_tmp);
                this->buffer_out.push_back(buffer_out_tmp);
                this->krnl_xil[ib*this->_numCU + ik].setArg(0, this->buffer_in[ib*this->_numCU + ik]);
                this->krnl_xil[ib*this->_numCU + ik].setArg(1, this->buffer_out[ib*this->_numCU + ik]);
            }
        }
    }
};
