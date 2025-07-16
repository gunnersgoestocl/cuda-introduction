#ifndef TRANSFORMER_COMMON_H
#define TRANSFORMER_COMMON_H

#include <vector>
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>

const int INPUT_DIM = 256;
const int HIDDEN_DIM = 768;
const int BATCH_SIZE = 128;

void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(err) (checkCudaError(err, __FILE__, __LINE__))

#endif // TRANSFORMER_COMMON_H