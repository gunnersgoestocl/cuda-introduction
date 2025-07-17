#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "kernels.cuh"
#include "../common/utils.h"

#define CHECK_CUDA_ERROR(err) \
    do { \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void transformer_block_cuda_main(float* input_embeddings, float* weights_q, float* weights_k, float* weights_v, 
                                 float* ff_weights1, float* ff_weights2, float* output, 
                                 int batch_size, int seq_length, int d_model, int d_ff) {
    
    // Calculate sizes
    size_t input_size = batch_size * seq_length * d_model * sizeof(float);
    size_t weight_size = d_model * d_model * sizeof(float);
    size_t ff1_size = d_model * d_ff * sizeof(float);
    size_t ff2_size = d_ff * d_model * sizeof(float);
    
    // Create CUDA streams for overlap
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    
    // Allocate device memory
    float *d_input, *d_weights_q, *d_weights_k, *d_weights_v;
    float *d_ff_weights1, *d_ff_weights2, *d_output;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, input_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights_q, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights_k, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_weights_v, weight_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ff_weights1, ff1_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ff_weights2, ff2_size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, input_size));

    // 非同期データ転送でオーバーラップを実現
    std::cout << "Transferring data to GPU..." << std::endl;
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input_embeddings, input_size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_weights_q, weights_q, weight_size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_weights_k, weights_k, weight_size, cudaMemcpyHostToDevice, stream2));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_weights_v, weights_v, weight_size, cudaMemcpyHostToDevice, stream2));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_ff_weights1, ff_weights1, ff1_size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_ff_weights2, ff_weights2, ff2_size, cudaMemcpyHostToDevice, stream2));

    // 転送完了を待機
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    
    std::cout << "Executing optimized transformer block kernel..." << std::endl;
    
    // **核心部分**: 統合カーネルで全処理をGPU上で完結
    transformer_block_cuda_optimized(
        d_input, d_weights_q, d_weights_k, d_weights_v,
        d_ff_weights1, d_ff_weights2, d_output,
        batch_size, seq_length, d_model, d_ff,
        stream1  // ストリームを使用して非同期実行
    );
    
    std::cout << "Transferring results back to CPU..." << std::endl;
    
    // 結果を非同期で転送
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, d_output, input_size, cudaMemcpyDeviceToHost, stream1));
    cudaStreamSynchronize(stream1);

    // Cleanup
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    
    cudaFree(d_input); cudaFree(d_weights_q); cudaFree(d_weights_k); cudaFree(d_weights_v);
    cudaFree(d_ff_weights1); cudaFree(d_ff_weights2); cudaFree(d_output);
}

std::vector<float> load_dummy_data_cuda(size_t size) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    return data;
}

int main() {
    srand(static_cast<unsigned int>(time(nullptr)));
    
    std::cout << "=== Optimized Transformer Block CUDA Implementation ===" << std::endl;
    
    int batch_size = 32;
    int seq_length = 64;
    int d_model = 256;
    int d_ff = 1024;
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    std::cout << "  Model dimension: " << d_model << std::endl;
    std::cout << "  Feed forward dimension: " << d_ff << std::endl;
    
    // Generate dummy data
    std::cout << "Generating dummy data..." << std::endl;
    
    auto input_vec = load_dummy_data_cuda(batch_size * seq_length * d_model);
    auto weights_q_vec = load_dummy_data_cuda(d_model * d_model);
    auto weights_k_vec = load_dummy_data_cuda(d_model * d_model);
    auto weights_v_vec = load_dummy_data_cuda(d_model * d_model);
    auto ff_weights1_vec = load_dummy_data_cuda(d_model * d_ff);
    auto ff_weights2_vec = load_dummy_data_cuda(d_ff * d_model);
    
    std::vector<float> output_vec(batch_size * seq_length * d_model);
    
    std::cout << "Executing optimized transformer block..." << std::endl;
    
    // Measure execution time
    double execution_time = measureExecutionTime([&]() {
        transformer_block_cuda_main(input_vec.data(), weights_q_vec.data(), weights_k_vec.data(), 
                                   weights_v_vec.data(), ff_weights1_vec.data(), ff_weights2_vec.data(),
                                   output_vec.data(), batch_size, seq_length, d_model, d_ff);
    });
    
    std::cout << "Transformer block execution completed!" << std::endl;
    std::cout << "Execution time: " << execution_time << " seconds" << std::endl;
    
    // Save output
    saveOutputData("result/cuda_output.bin", output_vec.data(), output_vec.size());
    
    std::cout << "Output saved to result/cuda_output.bin" << std::endl;
    std::cout << "Optimizations applied:" << std::endl;
    std::cout << "  ✓ Fused kernel execution (single GPU launch)" << std::endl;
    std::cout << "  ✓ Device functions for modularity" << std::endl;
    std::cout << "  ✓ Asynchronous memory transfers" << std::endl;
    std::cout << "  ✓ Stream-based overlap" << std::endl;
    std::cout << "  ✓ Minimized host-device synchronization" << std::endl;
    
    return 0;
}