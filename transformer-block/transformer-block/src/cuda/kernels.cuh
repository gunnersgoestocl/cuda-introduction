#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K);
__global__ void softmax(const float* scores, float* A, int N);
__global__ void layer_norm(const float* input, float* output, float* gamma, float* beta, int N);
__global__ void feed_forward(const float* input, const float* weights1, const float* weights2, float* output, int N, int D_ff);
__global__ void relu(const float* input, float* output, int N);

#endif // KERNELS_CUH