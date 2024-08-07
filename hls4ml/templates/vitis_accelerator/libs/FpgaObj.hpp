#pragma once

#include <chrono>
#include <cstdint>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "DataBatcher.hpp"
#include "Types.hpp"
#include "Worker.hpp"
#include "xcl2.hpp"

template <class T, class U> class FpgaObj {
  public:
    /**
     * \brief Constructor
     * \param batchsize Number of samples
     * \param sampleInputSize Flattened length of a single input to the model
     * \param sampleOutputSize Flattened length of a single output from the model
     * \param numCU Number of compute units synthesized on the FPGA
     * \param xclbinFilename String containing path of synthesized xclbin
     */
    FpgaObj(int batchsize, int sampleInputSize, int sampleOutputSize, int numCU,
            std::string xclbinFilename)
        : _batchsize(batchsize), _sampleInputSize(sampleInputSize), _sampleOutputSize(sampleOutputSize),
          _numCU(numCU), _xclbinFilename(xclbinFilename) {

        // Finds Xilinx device
        devices = xcl::get_xil_devices();
        device = devices[0];
        std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
        std::cout << "Found Device: " << deviceName << std::endl;

        // Load xclbin
        fileBuf = xcl::read_binary_file(_xclbinFilename);
        bins = cl::Program::Binaries({{fileBuf.data(), fileBuf.size()}});

        // Create OpenCL context
        context = cl::Context(device);

        // Create OpenCL program from binary file
        program = cl::Program(context, devices, bins);

        // Create OpenCL command queues
        comQueues.reserve(_numCU);
        for (int i = 0; i < _numCU; i++) {
            comQueues.emplace_back(context,
                                   device,
                                   CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
        }

        // Create mutexes for each command queue
        std::vector<std::mutex> temp(_numCU);
        comQueueMtxi.swap(temp);
    }

    /**
     * \brief Creates worker objects for each compute unit
     * \param workersPerCU Number of worker objects that will drive each compute unit
     * \param fpga Type of memory resource used by the FPGA.
     * \param numHBMChannels Number of channels per port each Worker uses. Only for HBM.
     */
    void createWorkers(int workersPerCU, FPGAType fpga, int numHBMChannels = 0) {
        _workersPerCU = workersPerCU;

        // Construct workers
        workers.reserve(_numCU * _workersPerCU);
        for (int i = 0; i < _numCU; i++) {
            for (int j = 0; j < workersPerCU; j++) {
                workers.emplace_back(_batchsize,
                                     _sampleInputSize,
                                     _sampleOutputSize,
                                     comQueues[i],
                                     comQueueMtxi[i]);
            }
        }

        // Initialize workers
        int currHBMChannel = 0;  // Only used if FPGAType is HBM
        for (int i = 0; i < _numCU; i++) {
            for (int j = 0; j < _workersPerCU; j++) {
                workers[i * _workersPerCU + j].initialize(context,
                                                          program,
                                                          i + 1,
                                                          i * _workersPerCU + j,
                                                          fpga,
                                                          currHBMChannel,
                                                          numHBMChannels);

            }
            currHBMChannel += 2 * numHBMChannels;
        }
    }

    /**
     * \brief Loads data from a file into batches and distribute evenly amongst Workers.
     * \param fin Filename
     * \param s Input type. VitisAccelerator Backend currently uses text input. However,
     * the code also supports binary input in the format produced by NumPy's toFile().
     * \param profiling If true, the given data will be iterated over multiple times,
     * for more accurate throughput testing.
     * \param profilingDataRepeat Only used if profiling is set to True. Additional number of
     * times the given data is iterated over.
     */
    void loadData(const std::string& fin, bool profiling = false, int profilingDataRepeat = 0) {
        // Set-up containers for each Worker's batches/workload
        batchedData.reserve(_numCU * _workersPerCU);
        for (int i = 0; i < _numCU * _workersPerCU; i++) {
            batchedData.emplace_back();
        }

        // Batch and distribute data
        db = new DataBatcher<T, U>(_batchsize,
                                   _sampleInputSize,
                                   _sampleOutputSize,
                                   _numCU * _workersPerCU,
                                   profiling,
                                   profilingDataRepeat);
        db->read(fin);
        db->createResultBuffers();
        db->batch(batchedData);
    }

    /**
     * \brief Workers evaluate all loaded data. Each worker uses a separate thread.
     */
    void evaluateAll() {
        // Check that data has been loaded and batched
        if (batchedData.size() == 0 || db == nullptr) {
            throw std::runtime_error("No data loaded");
        }

        std::cout << "\nStarting FPGA run" << std::endl;

        auto ts_start = std::chrono::system_clock::now();
        std::vector<std::thread> accelThreads;
        accelThreads.reserve(_numCU * _workersPerCU);
        for (int i = 0; i < _numCU * _workersPerCU; i++) {
            accelThreads.emplace_back([this, i]() {
                this->workers[i].evalLoop(this->batchedData[i]);
            });
        }
        for (int i = 0; i < _numCU * _workersPerCU; i++) {
            accelThreads[i].join();
        }
        for (int i = 0; i < _numCU; i++) {
            OCL_CHECK(err, err = comQueues[i].finish());
        }
        auto ts_end = std::chrono::system_clock::now();

        uint64_t ns_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(ts_end - ts_start).count();
        if (db->isProfilingMode()) {
            double profilingThroughput = 1.0e9 * static_cast<double>(db->getProfilingSampleCount()) / ns_elapsed;
            std::cout << "\nProfiling throughput: " << profilingThroughput << " predictions/second" << std::endl;
        } else {
            double throughput = 1.0e9 * static_cast<double>(db->getSampleCount()) / ns_elapsed;
            double maxThroughput = 1.0e9 * static_cast<double>(db->getPaddedSampleCount()) / ns_elapsed;
            std::cout << "\nUtilized throughput: " << throughput << " predictions/second" << std::endl;
            std::cout << "Max possible throughput: " << maxThroughput << " predictions/second" << std::endl;
        }
    }

    /**
     * \brief Writes results, in text format, to provided file. Releases resources
     * \param fout Filename. If file already exists, it will be overwritten with current results.
     */
    void saveResults(const std::string& fout) {
        if (db == nullptr) {
            throw std::runtime_error("No data loaded");
        }
        db->write(fout);
        db->closeFile();
    }

  private:
    int _batchsize;
    int _sampleInputSize;
    int _sampleOutputSize;
    int _numCU;
    std::string _xclbinFilename;

    /// @brief A list of connected Xilinx devices
    std::vector<cl::Device> devices;
    /// @brief The identified FPGA
    cl::Device device;
    /// @brief Container that xclbin file is read into
    std::vector<unsigned char> fileBuf;
    /// @brief OpenCL object constructed from xclbin
    cl::Program::Binaries bins;
    /// @brief OpenCL Program that each compute unit executes
    cl::Program program;
    /// @brief OpenCL Device Context
    cl::Context context;
    /// @brief OpenCL Command Queues for each compute unit
    std::vector<cl::CommandQueue> comQueues;
    /// @brief Mutexes for each Command Queue
    mutable std::vector<std::mutex> comQueueMtxi;
    /// @brief Error code storage
    cl_int err;

    int _workersPerCU = 0;
    /// @brief Workers, indexed by (i_CU * _workersPerCU + i_worker)
    std::vector<Worker<T, U>> workers;
    /// @brief Data Batcher
    DataBatcher<T, U>* db = nullptr;
    /// @brief A vector containing each Worker's batches/workload
    std::vector<std::list<Batch<T, U>>> batchedData;
};
