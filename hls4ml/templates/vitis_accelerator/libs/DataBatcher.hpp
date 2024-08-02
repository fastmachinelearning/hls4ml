#pragma once

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <list>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "Types.hpp"

template <class T, class U> class DataBatcher {
  public:
    /**
     * \brief Constructor
     * \param batchsize Number of samples
     * \param sampleInputSize Flattened length of a single input to the model
     * \param sampleOutputSize Flattened length of a single output from the model
     * \param numWorkers Total number of workers
     * \param profiling If true, the given data will be iterated over multiple times,
     * for more accurate throughput testing.
     * \param profilingDataRepeat Only used if profiling is set to True. Additional number of
     * times the given data is iterated over.
     */
    DataBatcher(int batchsize, int sampleInputSize, int sampleOutputSize, int numWorkers,
                bool profiling, int profilingDataRepeat)
        : _batchsize(batchsize), _sampleInputSize(sampleInputSize), _sampleOutputSize(sampleOutputSize),
          _numWorkers(numWorkers), _profiling(profiling), _profilingDataRepeat(profilingDataRepeat) {}

    /**
     * \brief Read in data to a buffer. Allocate space for results.
     * \param filename Filename.
     * \param s Type of input, currently supports text files used by VitisAccelerator backend, and
     * binary files produced by NumPy's toFile() function
     */
    void read(const std::string& filename) {
        std::cout << "\nReading data from text file " << filename << std::endl;

        // Read in text file
        std::ifstream fin(filename);
        if (!fin.is_open()) {
            throw std::runtime_error("Error opening file " + filename);
        }

        std::string line;
        while (std::getline(fin, line)) {
            originalSampleCount++;
            std::istringstream parser(line);
            T val;
            while (parser >> val) {
                inputData.push_back(val);
            }
            if (!parser.eof()) {
                throw std::runtime_error("Failed to parse value on line " + std::to_string(originalSampleCount));
            }
        }
        std::cout << "Read in " << originalSampleCount << " lines" << std::endl;
        fin.close();

        // Zero-pad
        numBatches = std::ceil(static_cast<double>(originalSampleCount) / _batchsize);
        if (numBatches * _batchsize > originalSampleCount) {
            inputData.resize(numBatches * _batchsize * _sampleInputSize, (T)0);
        }
    }

    /**
     * \brief Allocate space for writing results to.
     */
    void createResultBuffers() {
        storedEvalResults.resize(numBatches * _batchsize * _sampleOutputSize, (U)0);

        // Allocate space to dump the extra arbitrary data used during profiling
        if (_profiling) {
            profilingResultsDump.resize(_numWorkers * _batchsize * _sampleOutputSize, (U)0);
        }
    }

    /**
     * \brief Splits data into batches and distributes batches evenly amongst Workers.
     * \param batchedData A vector of containers for each Worker's batches/workload.
     * Size must be equal to _numWorkers.
     */
    void batch(std::vector<std::list<Batch<T, U>>>& batchedData) {
        if (inputData.size() == 0 || originalSampleCount == 0) {
            throw std::runtime_error("No data to batch");
        }
        if (storedEvalResults.size() == 0) {
            throw std::runtime_error("Create result buffers first");
        }

        batchedData.reserve(_numWorkers);
        for (int i = 0; i < _numWorkers; i++) {
            batchedData.emplace_back();
        }

        uint64_t batchIndex = 0;
        while (batchIndex < numBatches) {
            int worker = batchIndex % _numWorkers;
            uint64_t inputLocation = batchIndex * _batchsize * _sampleInputSize;
            uint64_t outputLocation = batchIndex * _batchsize * _sampleOutputSize;

            const T* in = &inputData[inputLocation];
            U* out = &storedEvalResults[outputLocation];
            Batch<T, U> newBatch = {in, out};

            batchedData[worker].push_back(newBatch);
            batchIndex++;
        }

        if (_profiling) {
            std::cout << "Creating profiling batches" << std::endl;
            profilingBatchCount = numBatches * (_profilingDataRepeat + 1);
            while (batchIndex < profilingBatchCount) {
                int worker = batchIndex % _numWorkers;
                uint64_t inputLocation = (batchIndex % numBatches) * _batchsize * _sampleInputSize;
                uint64_t outputLocation = worker * _batchsize * _sampleOutputSize;

                const T* in = &inputData[inputLocation];
                U* out = &profilingResultsDump[outputLocation];
                Batch<T, U> newBatch = {in, out};

                batchedData[worker].push_back(newBatch);
                batchIndex++;
            }
        }
    }

    /**
     * \brief Releases resources used when reading from input files. Note: Data from those files
     * will be cleared and will no longer be accessible.
     */
    void closeFile() {
        inputData.clear();

        originalSampleCount = 0;
        numBatches = 0;
        profilingBatchCount = 0;
    }

    void write(const std::string& filename) {
        std::cout << "\nWriting HW results to file " << filename << std::endl;
        std::ofstream fout;
        fout.open(filename, std::ios::trunc);

        if (fout.is_open()) {
            for (uint64_t i = 0; i < originalSampleCount; i++) {
                std::stringstream line;
                for (int n = 0; n < _sampleOutputSize; n++) {
                    line << (float)storedEvalResults[(i * _sampleOutputSize) + n] << " ";
                }
                fout << line.str() << "\n";
            }
            fout.close();
        } else {
            throw std::runtime_error("Error writing to file " + filename);
        }

        storedEvalResults.clear();
        profilingResultsDump.clear();
    }

    uint64_t getSampleCount() {
        return originalSampleCount;
    }

    uint64_t getPaddedSampleCount() {
        return numBatches * _batchsize;
    }

    uint64_t getProfilingSampleCount() {
        return profilingBatchCount * _batchsize;
    }

    bool isProfilingMode() {
        return _profiling;
    }

  private:
    int _batchsize;
    int _sampleInputSize;
    int _sampleOutputSize;
    int _numWorkers;
    bool _profiling;
    int _profilingDataRepeat;

    /// @brief Number of floats read in. (Not including padding).
    uint64_t originalSampleCount = 0;
    /// @brief Number of batches of data. (After padding).
    uint64_t numBatches = 0;
    /// @brief Effective number of batches of data being evaluted.
    uint64_t profilingBatchCount = 0;
    /// @brief Vector with values.
    std::vector<T> inputData;
    /// @brief Vector to store evaluation results.
    std::vector<U> storedEvalResults;
    /// @brief Vector for dumping results from extra arbitrary data used during profiling.
    std::vector<U> profilingResultsDump;
};
