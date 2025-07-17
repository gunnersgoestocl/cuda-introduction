#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cmath>

// Matrix multiplication kernel
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float value = 0.0f;
        for (int k = 0; k < K; ++k) {
            value += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// Softmax kernel (row-wise softmax for attention scores)
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int seq_length) {
    int batch_idx = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx < batch_size && row < seq_length) {
        int offset = batch_idx * seq_length * seq_length + row * seq_length;
        
        // Find max value for numerical stability
        float max_val = input[offset];
        for (int i = 1; i < seq_length; ++i) {
            max_val = fmaxf(max_val, input[offset + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_length; ++i) {
            output[offset + i] = expf(input[offset + i] - max_val);
            sum += output[offset + i];
        }
        
        // Normalize
        for (int i = 0; i < seq_length; ++i) {
            output[offset + i] /= sum;
        }
    }
}

// Layer normalization kernel
__global__ void layer_norm_kernel(const float* input, float* output, const float* gamma, const float* beta, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < N; ++i) {
            mean += input[i];
        }
        mean /= N;
        
        // Compute variance
        float variance = 0.0f;
        for (int i = 0; i < N; ++i) {
            float diff = input[i] - mean;
            variance += diff * diff;
        }
        variance /= N;
        
        // Apply normalization
        float inv_std = rsqrtf(variance + 1e-6f);
        for (int i = 0; i < N; ++i) {
            output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
        }
    }
}

// ReLU activation kernel
__global__ void relu_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Host wrapper functions
void cuda_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    matmul_kernel<<<gridSize, blockSize>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}

void cuda_softmax(const float* input, float* output, int batch_size, int seq_length) {
    dim3 blockSize(1, 16);
    dim3 gridSize(batch_size, (seq_length + blockSize.y - 1) / blockSize.y);
    
    softmax_kernel<<<gridSize, blockSize>>>(input, output, batch_size, seq_length);
    cudaDeviceSynchronize();
}

void cuda_layer_norm(const float* input, float* output, const float* gamma, const float* beta, int N) {
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    layer_norm_kernel<<<gridSize, blockSize>>>(input, output, gamma, beta, N);
    cudaDeviceSynchronize();
}

void cuda_relu(const float* input, float* output, int N) {
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    relu_kernel<<<gridSize, blockSize>>>(input, output, N);
    cudaDeviceSynchronize();
}

__device__ void compute_qkv_for_position(const float* input, const float* weights_q, const float* weights_k, const float* weights_v,
                                        float* Q, float* K, float* V, int batch_idx, int seq_idx, 
                                        int dim_idx, int seq_length, int d_model) {
    int input_base = batch_idx * seq_length * d_model + seq_idx * d_model;
    int output_idx = batch_idx * seq_length * d_model + seq_idx * d_model + dim_idx;
    
    float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
    
    for (int k = 0; k < d_model; ++k) {
        float input_val = input[input_base + k];
        q_val += input_val * weights_q[k * d_model + dim_idx];
        k_val += input_val * weights_k[k * d_model + dim_idx];
        v_val += input_val * weights_v[k * d_model + dim_idx];
    }
    
    Q[output_idx] = q_val;
    K[output_idx] = k_val;
    V[output_idx] = v_val;
}

// Attention scoreの計算
__device__ float compute_attention_score(const float* Q, const float* K, int seq_idx1, int seq_idx2, 
                                        int batch_idx, int seq_length, int d_model) {
    float score = 0.0f;
    int q_base = batch_idx * seq_length * d_model + seq_idx1 * d_model;
    int k_base = batch_idx * seq_length * d_model + seq_idx2 * d_model;
    
    for (int d = 0; d < d_model; ++d) {
        score += Q[q_base + d] * K[k_base + d];
    }
    
    return score / sqrtf((float)d_model);
}

// 行単位でのSoftmax適用
__device__ void apply_softmax_row(float* attention_scores, int batch_idx, int seq_idx, 
                                 int seq_length, int total_seq_length) {
    int row_base = batch_idx * seq_length * seq_length + seq_idx * seq_length;
    
    // 最大値を見つける（数値安定性のため）
    float max_val = attention_scores[row_base];
    for (int i = 1; i < seq_length; ++i) {
        max_val = fmaxf(max_val, attention_scores[row_base + i]);
    }
    
    // 指数関数を適用し、合計を計算
    float sum = 0.0f;
    for (int i = 0; i < seq_length; ++i) {
        attention_scores[row_base + i] = expf(attention_scores[row_base + i] - max_val);
        sum += attention_scores[row_base + i];
    }
    
    // 正規化
    for (int i = 0; i < seq_length; ++i) {
        attention_scores[row_base + i] /= sum;
    }
}

// Attention出力の計算
__device__ void compute_attention_output(const float* attention_weights, const float* V, float* output,
                                        int batch_idx, int seq_idx, int dim_idx, 
                                        int seq_length, int d_model) {
    int output_idx = batch_idx * seq_length * d_model + seq_idx * d_model + dim_idx;
    int weight_base = batch_idx * seq_length * seq_length + seq_idx * seq_length;
    
    float result = 0.0f;
    for (int i = 0; i < seq_length; ++i) {
        int v_idx = batch_idx * seq_length * d_model + i * d_model + dim_idx;
        result += attention_weights[weight_base + i] * V[v_idx];
    }
    
    output[output_idx] = result;
}

// Layer Normalizationの適用
__device__ void apply_layer_norm(const float* input, float* output, int batch_idx, int seq_idx,
                                int seq_length, int d_model, float eps) {
    int base_idx = batch_idx * seq_length * d_model + seq_idx * d_model;
    
    // 平均の計算
    float mean = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        mean += input[base_idx + i];
    }
    mean /= d_model;
    
    // 分散の計算
    float variance = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        float diff = input[base_idx + i] - mean;
        variance += diff * diff;
    }
    variance /= d_model;
    
    // 正規化
    float inv_std = rsqrtf(variance + eps);
    for (int i = 0; i < d_model; ++i) {
        output[base_idx + i] = (input[base_idx + i] - mean) * inv_std;
    }
}

// Feed Forward Networkの実行
__device__ void feed_forward_layer(const float* input, const float* weights1, const float* weights2,
                                  float* output, float* temp_buffer, int batch_idx, int seq_idx,
                                  int d_model, int d_ff, int seq_length) {
    int input_base = batch_idx * seq_length * d_model + seq_idx * d_model;
    int hidden_base = batch_idx * seq_length * d_ff + seq_idx * d_ff;
    int output_base = batch_idx * seq_length * d_model + seq_idx * d_model;
    
    // 第1層（input -> hidden with ReLU）
    for (int h = 0; h < d_ff; ++h) {
        float hidden_val = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            hidden_val += input[input_base + i] * weights1[i * d_ff + h];
        }
        temp_buffer[hidden_base + h] = fmaxf(0.0f, hidden_val); // ReLU
    }
    
    // 第2層（hidden -> output）
    for (int o = 0; o < d_model; ++o) {
        float output_val = 0.0f;
        for (int h = 0; h < d_ff; ++h) {
            output_val += temp_buffer[hidden_base + h] * weights2[h * d_model + o];
        }
        output[output_base + o] = output_val;
    }
}

// Residual connectionの追加
__device__ void add_residual_connection(const float* input1, const float* input2, float* output,
                                       int batch_idx, int seq_idx, int d_model, int seq_length) {
    int base_idx = batch_idx * seq_length * d_model + seq_idx * d_model;
    
    for (int i = 0; i < d_model; ++i) {
        output[base_idx + i] = input1[base_idx + i] + input2[base_idx + i];
    }
}

// メインのTransformerブロックカーネル
__global__ void transformer_block_kernel(
    const float* input_embeddings,
    const float* weights_q, const float* weights_k, const float* weights_v,
    const float* ff_weights1, const float* ff_weights2,
    float* output,
    float* workspace,
    int batch_size, int seq_length, int d_model, int d_ff
) {
    // スレッドインデックス
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= d_model) return;
    
    // ワークスペースの分割
    int total_elements = batch_size * seq_length * d_model;
    // int total_ff_elements = batch_size * seq_length * d_ff;
    int attention_elements = batch_size * seq_length * seq_length;
    
    float* Q = workspace;
    float* K = workspace + total_elements;
    float* V = workspace + 2 * total_elements;
    float* attention_scores = workspace + 3 * total_elements;
    float* attention_output = workspace + 3 * total_elements + attention_elements;
    float* norm_output = workspace + 4 * total_elements + attention_elements;
    float* ff_output = workspace + 5 * total_elements + attention_elements;
    float* ff_temp = workspace + 6 * total_elements + attention_elements;
    
    // **Phase 1: Q, K, V計算**
    compute_qkv_for_position(input_embeddings, weights_q, weights_k, weights_v,
                            Q, K, V, batch_idx, seq_idx, dim_idx, seq_length, d_model);
    
    __syncthreads();
    
    // **Phase 2: Attention scores計算**（一部のスレッドのみ実行）
    if (dim_idx == 0) { // 各(batch, seq)ペアで1つのスレッドのみ
        for (int target_seq = 0; target_seq < seq_length; ++target_seq) {
            float score = compute_attention_score(Q, K, seq_idx, target_seq, 
                                                 batch_idx, seq_length, d_model);
            int score_idx = batch_idx * seq_length * seq_length + seq_idx * seq_length + target_seq;
            attention_scores[score_idx] = score;
        }
        
        // Softmaxを適用
        apply_softmax_row(attention_scores, batch_idx, seq_idx, seq_length, seq_length);
    }
    
    __syncthreads();
    
    // **Phase 3: Attention出力計算**
    compute_attention_output(attention_scores, V, attention_output,
                            batch_idx, seq_idx, dim_idx, seq_length, d_model);
    
    __syncthreads();
    
    // **Phase 4: Residual connection + Layer norm**
    int global_idx = batch_idx * seq_length * d_model + seq_idx * d_model + dim_idx;
    float residual_val = input_embeddings[global_idx] + attention_output[global_idx];
    
    if (dim_idx == 0) { // Layer normは各sequenceで1回
        apply_layer_norm(&residual_val, norm_output, batch_idx, seq_idx, seq_length, d_model);
    }
    
    __syncthreads();
    
    // **Phase 5: Feed Forward Network**
    if (dim_idx == 0) { // FFNも各sequenceで1回
        feed_forward_layer(norm_output, ff_weights1, ff_weights2, ff_output, ff_temp,
                          batch_idx, seq_idx, d_model, d_ff, seq_length);
    }
    
    __syncthreads();
    
    // **Phase 6: Final residual connection**
    add_residual_connection(norm_output, ff_output, output, batch_idx, seq_idx, d_model, seq_length);
}

// ホスト関数
void transformer_block_cuda_optimized(
    const float* input_embeddings,
    const float* weights_q, const float* weights_k, const float* weights_v,
    const float* ff_weights1, const float* ff_weights2,
    float* output,
    int batch_size, int seq_length, int d_model, int d_ff,
    cudaStream_t stream
) {
    // ワークスペース計算
    size_t workspace_size = (
        6 * batch_size * seq_length * d_model +  // Q,K,V,attention_output,norm_output,ff_output
        batch_size * seq_length * seq_length +    // attention_scores
        batch_size * seq_length * d_ff           // ff_temp
    ) * sizeof(float);
    
    float* d_workspace;
    cudaMalloc(&d_workspace, workspace_size);
    
    // グリッド・ブロック設定
    dim3 blockSize(16, 8, 1);
    dim3 gridSize(
        (d_model + blockSize.x - 1) / blockSize.x,
        (seq_length + blockSize.y - 1) / blockSize.y,
        batch_size
    );
    
    // 単一カーネル実行（内部で__device__関数を適切に呼び分け）
    transformer_block_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_embeddings, weights_q, weights_k, weights_v,
        ff_weights1, ff_weights2, output, d_workspace,
        batch_size, seq_length, d_model, d_ff
    );
    
    if (stream != 0) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    
    cudaFree(d_workspace);
}