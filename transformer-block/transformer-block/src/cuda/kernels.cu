#include <cuda_runtime.h>
#include "kernels.cuh"

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

__global__ void softmax(const float* scores, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float max_val = scores[0];
        for (int i = 1; i < N; ++i) {
            if (scores[i] > max_val) {
                max_val = scores[i];
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            output[i] = exp(scores[i] - max_val);
            sum += output[i];
        }

        for (int i = 0; i < N; ++i) {
            output[i] /= sum;
        }
    }
}

__global__ void layer_norm(const float* input, float* output, float epsilon, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float mean = 0.0f;
        float variance = 0.0f;

        for (int i = 0; i < N; ++i) {
            mean += input[i];
        }
        mean /= N;

        for (int i = 0; i < N; ++i) {
            variance += (input[i] - mean) * (input[i] - mean);
        }
        variance /= N;

        for (int i = 0; i < N; ++i) {
            output[i] = (input[i] - mean) / sqrt(variance + epsilon);
        }
    }
}

__global__ void feed_forward(const float* input, const float* weights, float* output, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float value = 0.0f;
        for (int k = 0; k < N; ++k) {
            value += input[k] * weights[k * M + col];
        }
        output[row * M + col] = value;
    }
}

extern "C" void launch_kernels(const float* input_embeddings, const float* weights_q, const float* weights_k, const float* weights_v, 
                               const float* ff_weights1, const float* ff_weights2, float* output, int N, int M) {
    // Define grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Allocate device memory and copy data
    float *d_input, *d_weights_q, *d_weights_k, *d_weights_v, *d_ff_weights1, *d_ff_weights2, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_weights_q, N * sizeof(float));
    cudaMalloc(&d_weights_k, N * sizeof(float));
    cudaMalloc(&d_weights_v, N * sizeof(float));
    cudaMalloc(&d_ff_weights1, N * M * sizeof(float));
    cudaMalloc(&d_ff_weights2, N * M * sizeof(float));
    cudaMalloc(&d_output, M * sizeof(float));

    cudaMemcpy(d_input, input_embeddings, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_q, weights_q, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_k, weights_k, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_v, weights_v, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_weights1, ff_weights1, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ff_weights2, ff_weights2, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernels
    matmul<<<gridSize, blockSize>>>(d_input, d_weights_q, d_output, N);
    softmax<<<(N + 255) / 256, 256>>>(d_output, d_output, N);
    layer_norm<<<(N + 255) / 256, 256>>>(d_output, d_output, 1e-5, N);
    feed_forward<<<gridSize, blockSize>>>(d_output, d_ff_weights1, d_output, N, M);

    // Copy output back to host
    cudaMemcpy(output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_weights_q);
    cudaFree(d_weights_k);
    cudaFree(d_weights_v);
    cudaFree(d_ff_weights1);
    cudaFree(d_ff_weights2);
    cudaFree(d_output);
}