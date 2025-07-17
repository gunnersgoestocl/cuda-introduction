#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Device関数の宣言
__device__ float compute_attention_score(const float* Q, const float* K, int seq_idx1, int seq_idx2, 
                                        int batch_idx, int seq_length, int d_model);

__device__ void compute_qkv_for_position(const float* input, const float* weights_q, const float* weights_k, const float* weights_v,
                                        float* Q, float* K, float* V, int batch_idx, int seq_idx, 
                                        int dim_idx, int seq_length, int d_model);

__device__ void apply_softmax_row(float* attention_scores, int batch_idx, int seq_idx, 
                                 int seq_length, int total_seq_length);

__device__ void compute_attention_output(const float* attention_weights, const float* V, float* output,
                                        int batch_idx, int seq_idx, int dim_idx, 
                                        int seq_length, int d_model);

__device__ void apply_layer_norm(const float* input, float* output, int batch_idx, int seq_idx,
                                int seq_length, int d_model, float eps = 1e-6f);

__device__ void feed_forward_layer(const float* input, const float* weights1, const float* weights2,
                                  float* output, float* temp_buffer, int batch_idx, int seq_idx,
                                  int d_model, int d_ff, int seq_length);

__device__ void add_residual_connection(const float* input1, const float* input2, float* output,
                                       int batch_idx, int seq_idx, int d_model, int seq_length);

// メインのTransformerブロックカーネル
__global__ void transformer_block_kernel(
    const float* input_embeddings,
    const float* weights_q, const float* weights_k, const float* weights_v,
    const float* ff_weights1, const float* ff_weights2,
    float* output,
    float* workspace,
    int batch_size, int seq_length, int d_model, int d_ff
);

// ホスト関数
void transformer_block_cuda_optimized(
    const float* input_embeddings,
    const float* weights_q, const float* weights_k, const float* weights_v,
    const float* ff_weights1, const float* ff_weights2,
    float* output,
    int batch_size, int seq_length, int d_model, int d_ff,
    cudaStream_t stream = 0
);

#endif // KERNELS_CUH