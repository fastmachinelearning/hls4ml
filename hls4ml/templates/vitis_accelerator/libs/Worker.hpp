#pragma once

#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <vector>

#include "Types.hpp"
#include "xcl2.hpp"

// HBM Pseudo-channel(PC) requirements
#define MAX_HBM_PC_COUNT 32
#define PC_NAME(n) n | XCL_MEM_TOPOLOGY
const int pc[MAX_HBM_PC_COUNT] = {PC_NAME(0),  PC_NAME(1),  PC_NAME(2),  PC_NAME(3),  PC_NAME(4),  PC_NAME(5),  PC_NAME(6),
                                  PC_NAME(7),  PC_NAME(8),  PC_NAME(9),  PC_NAME(10), PC_NAME(11), PC_NAME(12), PC_NAME(13),
                                  PC_NAME(14), PC_NAME(15), PC_NAME(16), PC_NAME(17), PC_NAME(18), PC_NAME(19), PC_NAME(20),
                                  PC_NAME(21), PC_NAME(22), PC_NAME(23), PC_NAME(24), PC_NAME(25), PC_NAME(26), PC_NAME(27),
                                  PC_NAME(28), PC_NAME(29), PC_NAME(30), PC_NAME(31)};

template <class T, class U> class Worker {
  public:
    /**
     * \brief Constructor
     * \param batchsize Number of samples
     * \param sampleInputSize Flattened length of a single input to the model
     * \param sampleOutputSize Flattened length of a single output from the model
     * \param commandQueue cl::CommandQueue that the worker will enqueue operations to
     * \param queueMutex Mutex protecting the CommandQueue (potentially shared with other workers)
     */
    Worker(int batchsize, int sampleInputSize, int sampleOutputSize, cl::CommandQueue& queue,
            std::mutex& queueMutex) :
        _batchsize(batchsize),
        _sampleInputSize(sampleInputSize),
        _sampleOutputSize(sampleOutputSize),
        _queue(queue),
        _queueMutex(queueMutex)
    {
        memmap_in.resize(_batchsize * _sampleInputSize, T(0.0f));
        memmap_out.resize(_batchsize * _sampleOutputSize, U(0.0f));
    }

    /**
     * \brief Initializes all resources the Worker needs to drive a compute unit.
     * \param context cl::Context of the FPGA.
     * \param program cl:Program of the FPGA.
     * \param computeUnit The number of the physical compute unit this worker will use.
     * \param workerId Worker's ID number.
     * \param fpga Type of memory resource used by the FPGA.
     * \param firstHBMChannel Start index of this Worker's memory channels this worker uses. Only for HBM.
     * \param numHBMChannels Number of channels per port this worker uses. Only for HBM.
     */
    void initialize(
        cl::Context& context,
        cl::Program& program,
        int computeUnit,
        int workerId,
        FPGAType fpga,
        int firstHBMChannel = 0,  // Only used for if fpga == FPGAType::HBM
        int numHBMChannels = 0  // Only used for if fpga == FPGAType::HBM
    ) {
        if (fpga == FPGAType::HBM) {
            allocateHBMMemory(context, firstHBMChannel, numHBMChannels);
        } else if (fpga == FPGAType::DDR) {
            allocateDDRMemory(context);
        }

        // Creating Kernel object
        std::string krnl_name =
            "kernel_wrapper:{kernel_wrapper_" + std::to_string(computeUnit) +
            "}"; // This is Xilinx's format for specifying the Compute Unit a kernel object uses
        krnl = cl::Kernel(program, krnl_name.c_str(), &err);
        krnl.setArg(0, input_buffer);
        krnl.setArg(1, output_buffer);

        std::cout << "Initialized Worker " << workerId
            << ", which will use Compute Unit " << computeUnit << std::endl;
    }

    /**
     * \brief Evaluates the single batch currently in input_buffer and writes to output_buffer.
     */
    void evaluate() {
        std::lock_guard<std::mutex> lock(_queueMutex);
        // Transfer inputs
        OCL_CHECK(err, err = _queue.enqueueMigrateMemObjects({input_buffer},
                                                             0, // 0 means from host
                                                             NULL, // No dependencies
                                                             &write_event));
        // Execute program
        writeCompleteEvents.push_back(write_event);
        OCL_CHECK(err, err = _queue.enqueueNDRangeKernel(krnl,
                                                         0,
                                                         1,
                                                         1,
                                                         &writeCompleteEvents,
                                                         &kernExe_event));
        writeCompleteEvents.pop_back();
        // Transfer outputs
        kernExeCompleteEvents.push_back(kernExe_event);
        OCL_CHECK(err, err = _queue.enqueueMigrateMemObjects({output_buffer},
                                                              CL_MIGRATE_MEM_OBJECT_HOST,
                                                              &kernExeCompleteEvents,
                                                              &read_event));
        kernExeCompleteEvents.pop_back();
        OCL_CHECK(err, err = read_event.wait());
    }

    /**
     * \brief Evaluates each batch of data provided via dataTracker. Uses float datatype
     * \param dataTracker Vector of input locations to read from and output locations to write to
     */
    void evalLoop(std::list<Batch<T, U>>& dataTracker) {
        while (!dataTracker.empty()) {
            // Copy inputs into memory-mapped buffer
            const T* dataLoc = dataTracker.front().dataIn;
            memcpy(&memmap_in[0], dataLoc, _batchsize * _sampleInputSize * sizeof(T));

            // Evaluate
            evaluate();

            // Copy outputs into persistent results vector
            U* resLoc = dataTracker.front().dataOut;
            memcpy(resLoc, &memmap_out[0], _batchsize * _sampleOutputSize * sizeof(U));

            dataTracker.pop_front();
        }
    }

  private:
    int _batchsize;
    int _sampleInputSize;
    int _sampleOutputSize;

    /// @brief Reference to the OpenCL command queue
    const cl::CommandQueue& _queue;
    /// @brief Mutex for thread-safe access to the command queue
    std::mutex& _queueMutex;

    /// @brief Vector mapped to FPGA input buffer
    std::vector<T, aligned_allocator<T>> memmap_in;
    /// @brief Vector mapped to FPGA output buffer
    std::vector<U, aligned_allocator<U>> memmap_out;

    /// @brief OpenCL buffer object for input
    cl::Buffer input_buffer;
    /// @brief OpenCL buffer object for output
    cl::Buffer output_buffer;
    /// @brief OpenCL kernel object
    cl::Kernel krnl;

    /// @brief Pointer mapping host input buffer to FPGA memory pseudo-channels (HBM only)
    cl_mem_ext_ptr_t hbm_in_ptr;
    /// @brief Pointer mapping host output buffer to FPGA memory pseudo-channels (HBM only)
    cl_mem_ext_ptr_t hbm_out_ptr;

    /// @brief Event for signaling input transfer completion
    cl::Event write_event;
    /// @brief Event for signaling kernel execution completion
    cl::Event kernExe_event;
    /// @brief Event for signaling output transfer completion
    cl::Event read_event;
    /// @brief Vector tracking write events. Required by OpenCL queue functions.
    std::vector<cl::Event> writeCompleteEvents;
    /// @brief Vector tracking kernel execution events. Required by OpenCL queue functions.
    std::vector<cl::Event> kernExeCompleteEvents;

    /// @brief Error code storage
    cl_int err;
    bool firstLoop = true;

    void allocateHBMMemory(cl::Context& context, int firstHBMChannel, int numHBMChannels) {
        // Create Pointer objects for the in/out ports for each worker
        // Assigning Pointers to specific HBM PC's using cl_mem_ext_ptr_t type and corresponding PC flags
        hbm_in_ptr.obj = memmap_in.data();
        hbm_in_ptr.param = 0;
        int in_flags = 0;
        for (int i = 0; i < numHBMChannels; i++) {
            in_flags |= pc[firstHBMChannel + i];
        }
        hbm_in_ptr.flags = in_flags;

        hbm_out_ptr.obj = memmap_out.data();
        hbm_out_ptr.param = 0;
        int out_flags = 0;
        for (int i = 0; i < numHBMChannels; i++) {
            out_flags |= pc[firstHBMChannel + numHBMChannels + i];
        }
        hbm_out_ptr.flags = out_flags;

        // Creating Buffer objects in Host memory
        uint64_t vector_in_size_bytes = sizeof(T) * _batchsize * _sampleInputSize;
        uint64_t vector_out_size_bytes = sizeof(U) * _batchsize * _sampleOutputSize;
        input_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY,
                                    vector_in_size_bytes, &hbm_in_ptr);
        output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_WRITE_ONLY,
                                    vector_out_size_bytes, &hbm_out_ptr);
    }

    void allocateDDRMemory(cl::Context& context) {
        uint64_t vector_in_size_bytes = sizeof(T) * _batchsize * _sampleInputSize;
        uint64_t vector_out_size_bytes = sizeof(U) * _batchsize * _sampleOutputSize;
        input_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, vector_in_size_bytes,
                                    memmap_in.data());
        output_buffer = cl::Buffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, vector_out_size_bytes,
                                    memmap_out.data());
    }
};
