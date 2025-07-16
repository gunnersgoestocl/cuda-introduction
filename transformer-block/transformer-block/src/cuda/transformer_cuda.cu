#include <cuda_runtime.h>
#include <iostream>
#include "kernels.cuh"
#include "transformer_common.h"

__global__ void matmul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void softmax_kernel(float* scores, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float maxScore = scores[0];
        for (int i = 1; i < N; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
            }
        }

        float sum = 0.0;
        for (int i = 0; i < N; i++) {
            output[i] = exp(scores[i] - maxScore);
            sum += output[i];
        }

        for (int i = 0; i < N; i++) {
            output[i] /= sum;
        }
    }
}

__global__ void layernorm_kernel(float* input, float* output, float* gamma, float* beta, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.0;
        float variance = 0.0;

        for (int i = 0; i < N; i++) {
            mean += input[i];
        }
        mean /= N;

        for (int i = 0; i < N; i++) {
            variance += (input[i] - mean) * (input[i] - mean);
        }
        variance /= N;

        for (int i = 0; i < N; i++) {
            output[i] = gamma[i] * ((input[i] - mean) / sqrt(variance + 1e-6)) + beta[i];
        }
    }
}

void transformer_block_cuda(float* input_embeddings, float* weights_q, float* weights_k, float* weights_v, 
                            float* ff_weights1, float* ff_weights2, float* output, int batch_size, int seq_length) {
    // Allocate device memory
    float *d_input, *d_weights_q, *d_weights_k, *d_weights_v, *d_ff_weights1, *d_ff_weights2, *d_output;
    cudaMalloc(&d_input, batch_size * seq_length * sizeof(float));
    cudaMalloc(&d_weights_q, seq_length * seq_length * sizeof(float));
    cudaMalloc(&d_weights_k, seq_length * seq_length * sizeof(float));
    cudaMalloc(&d_weights_v, seq_length * seq_length * sizeof(float));
    cudaMalloc(&d_ff_weights1, seq_length * seq_length * sizeof(float));
    cudaMalloc(&d_ff_weights2, seq_length * seq_length * sizeof(float));
    cudaMalloc(&d_output, batch_size * seq_length * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, input_embeddings, batch_size * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_q, weights_q, seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_k, weights_k, seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_v, weights_v, seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_weights1, ff_weights1, seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_weights2, ff_weights2, seq_length * seq_length * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels for each operation
    dim3 blockSize(16, 16);
    dim3 gridSize((seq_length + blockSize.x - 1) / blockSize.x, (seq_length + blockSize.y - 1) / blockSize.y);
    
    matmul_kernel<<<gridSize, blockSize>>>(d_input, d_weights_q, d_output, seq_length);
    // Additional kernels for attention, layer norm, and feed-forward would be called here

    // Copy output back to host
    cudaMemcpy(output, d_output, batch_size * seq_length * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights_q);
    cudaFree(d_weights_k);
    cudaFree(d_weights_v);
    cudaFree(d_ff_weights1);
    cudaFree(d_ff_weights2);
    cudaFree(d_output);
}