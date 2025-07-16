#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <chrono>

// CUDA関連のヘッダー
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_profiler_api.h>
#include <iostream>

// TensorCoreのブロックサイズ
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 行列サイズ - TensorCoreに最適化された16の倍数
#define M 1024
#define N 1024
#define K 1024

// CUDA計測用マクロ
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 詳細な計測結果を格納する構造体
typedef struct {
    float total_kernel_time;     // カーネル全体の実行時間
    float fragment_declare_time; // フラグメント宣言時間
    float fragment_init_time;    // フラグメント初期化時間
    float shared_declare_time;   // 共有メモリ宣言時間
    float global_to_shared_time; // グローバル→共有メモリ時間
    float global_to_frag_time;   // グローバル→フラグメント時間
    float shared_to_frag_time;   // 共有メモリ→フラグメント時間
    float mma_sync_time;         // MMA演算時間
    float frag_to_global_time;   // フラグメント→グローバル時間
    float compute_with_memory_time; // メモリアクセス込み計算時間（CUDA Core用）
    // 正規化用の情報
    int tiles_per_result;        // 結果1つあたりのタイル数
    int elements_per_tile;       // タイル1つあたりの要素数
} DetailedTimingInfo;

// 時間計測結果を格納する構造体（デバイス用）
typedef struct {
    uint64_t fragment_declare_start;
    uint64_t fragment_declare_end;
    uint64_t fragment_init_start;
    uint64_t fragment_init_end;
    uint64_t shared_declare_start;
    uint64_t shared_declare_end;
    uint64_t global_to_shared_start;
    uint64_t global_to_shared_end;
    uint64_t global_to_frag_start;
    uint64_t global_to_frag_end;
    uint64_t shared_to_frag_start;
    uint64_t shared_to_frag_end;
    uint64_t mma_sync_start;
    uint64_t mma_sync_end;
    uint64_t frag_to_global_start;
    uint64_t frag_to_global_end;
    uint64_t compute_with_memory_start;
    uint64_t compute_with_memory_end;
    uint64_t kernel_start;
    uint64_t kernel_end;
} DeviceTimingInfo;

// 行列初期化
void init_matrices(half *A, half *B, float *C_tc, float *C_tc_shared, 
                  float *C_cuda_global, float *C_cuda_shared, float *C_ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = __float2half((rand() % 10) / 10.0f);
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = __float2half((rand() % 10) / 10.0f);
        }
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_tc[i * N + j] = 0.0f;
            C_tc_shared[i * N + j] = 0.0f;
            C_cuda_global[i * N + j] = 0.0f;
            C_cuda_shared[i * N + j] = 0.0f;
            C_ref[i * N + j] = 0.0f;
        }
    }
}

// TensorCore グローバルメモリ版カーネル
__global__ void tensor_core_global_memory_kernel(half *A, half *B, float *C, 
                                               int m, int n, int k,
                                               DeviceTimingInfo *timing_info) {
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int row_start = block_row * WMMA_M;
    const int col_start = block_col * WMMA_N;
    
    if (row_start >= m || col_start >= n) return;
    
    bool is_measuring_thread = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    
    if (is_measuring_thread) {
        timing_info->kernel_start = clock64();
        timing_info->fragment_declare_start = clock64();
    }
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    if (is_measuring_thread) {
        timing_info->fragment_declare_end = clock64();
        timing_info->fragment_init_start = clock64();
    }
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    if (is_measuring_thread) {
        timing_info->fragment_init_end = clock64();
    }
    
    for (int i = 0; i < k; i += WMMA_K) {
        if (i + WMMA_K > k) break;
        
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_frag_start = clock64();
        }
        
        nvcuda::wmma::load_matrix_sync(a_frag, A + (row_start * k + i), k);
        nvcuda::wmma::load_matrix_sync(b_frag, B + (i * n + col_start), n);
        
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_frag_end = clock64();
            timing_info->mma_sync_start = clock64();
        }
        
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        if (is_measuring_thread && i == 0) {
            timing_info->mma_sync_end = clock64();
        }
    }
    
    if (is_measuring_thread) {
        timing_info->frag_to_global_start = clock64();
    }
    
    nvcuda::wmma::store_matrix_sync(C + (row_start * n + col_start), c_frag, n, nvcuda::wmma::mem_row_major);
    
    if (is_measuring_thread) {
        timing_info->frag_to_global_end = clock64();
        timing_info->kernel_end = clock64();
    }
}

// TensorCore 共有メモリ版カーネル
__global__ void tensor_core_shared_memory_kernel(half *A, half *B, float *C, 
                                               int m, int n, int k,
                                               DeviceTimingInfo *timing_info) {
    const int padded_k = (WMMA_K + 8) & ~0xF;
    const int padded_n = (WMMA_N + 8) & ~0xF;
    
    bool is_measuring_thread = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    
    if (is_measuring_thread) {
        timing_info->kernel_start = clock64();
        timing_info->shared_declare_start = clock64();
    }
    
    __shared__ half shared_A[WMMA_M][padded_k];
    __shared__ half shared_B[WMMA_K][padded_n];
    
    if (is_measuring_thread) {
        timing_info->shared_declare_end = clock64();
    }
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int row_start = block_row * WMMA_M;
    const int col_start = block_col * WMMA_N;
    
    if (row_start >= m || col_start >= n) return;
    
    const int tid = threadIdx.x;
    
    if (is_measuring_thread) {
        timing_info->fragment_declare_start = clock64();
    }
    
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    if (is_measuring_thread) {
        timing_info->fragment_declare_end = clock64();
        timing_info->fragment_init_start = clock64();
    }
    
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    if (is_measuring_thread) {
        timing_info->fragment_init_end = clock64();
    }
    
    for (int i = 0; i < k; i += WMMA_K) {
        if (i + WMMA_K > k) break;
        
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_shared_start = clock64();
        }
        
        // グローバルメモリから共有メモリへのロード
        for (int j = tid; j < WMMA_M * WMMA_K; j += blockDim.x) {
            int row = j / WMMA_K;
            int col = j % WMMA_K;
            if (row_start + row < m && i + col < k) {
                shared_A[row][col] = A[(row_start + row) * k + (i + col)];
            }
        }
        
        for (int j = tid; j < WMMA_K * WMMA_N; j += blockDim.x) {
            int row = j / WMMA_N;
            int col = j % WMMA_N;
            if (i + row < k && col_start + col < n) {
                shared_B[row][col] = B[(i + row) * n + (col_start + col)];
            }
        }
        
        __syncthreads();
        
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_shared_end = clock64();
            timing_info->shared_to_frag_start = clock64();
        }
        
        nvcuda::wmma::load_matrix_sync(a_frag, &shared_A[0][0], padded_k);
        nvcuda::wmma::load_matrix_sync(b_frag, &shared_B[0][0], padded_n);
        
        if (is_measuring_thread && i == 0) {
            timing_info->shared_to_frag_end = clock64();
            timing_info->mma_sync_start = clock64();
        }
        
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        if (is_measuring_thread && i == 0) {
            timing_info->mma_sync_end = clock64();
        }
        
        __syncthreads();
    }
    
    if (is_measuring_thread) {
        timing_info->frag_to_global_start = clock64();
    }
    
    nvcuda::wmma::store_matrix_sync(C + (row_start * n + col_start), c_frag, n, nvcuda::wmma::mem_row_major);
    
    if (is_measuring_thread) {
        timing_info->frag_to_global_end = clock64();
        timing_info->kernel_end = clock64();
    }
}

// CUDA Core グローバルメモリ版カーネル
__global__ void cuda_core_kernel(half *A, half *B, float *C, 
                                int m, int n, int k,
                                DeviceTimingInfo *timing_info) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= m || col >= n) return;
    
    bool is_measuring_thread = (threadIdx.x == 0 && threadIdx.y == 0 && 
                               blockIdx.x == 0 && blockIdx.y == 0);
    
    if (is_measuring_thread) {
        timing_info->kernel_start = clock64();
        timing_info->compute_with_memory_start = clock64();
    }
    
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += __half2float(A[row * k + i]) * __half2float(B[i * n + col]);
    }
    C[row * n + col] = sum;
    
    if (is_measuring_thread) {
        timing_info->compute_with_memory_end = clock64();
        timing_info->kernel_end = clock64();
    }
}

// CUDA Core 共有メモリ版カーネル
__global__ void cuda_core_shared_kernel(half *A, half *B, float *C, 
                                       int m, int n, int k,
                                       DeviceTimingInfo *timing_info) {
    const int TILE_SIZE = 16;
    
    bool is_measuring_thread = (threadIdx.x == 0 && threadIdx.y == 0 && 
                               blockIdx.x == 0 && blockIdx.y == 0);
    
    if (is_measuring_thread) {
        timing_info->kernel_start = clock64();
        timing_info->shared_declare_start = clock64();
    }
    
    __shared__ half shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ half shared_B[TILE_SIZE][TILE_SIZE];
    
    if (is_measuring_thread) {
        timing_info->shared_declare_end = clock64();
    }
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (k + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (is_measuring_thread && tile == 0) {
            timing_info->global_to_shared_start = clock64();
        }
        
        int a_row = row;
        int a_col = tile * TILE_SIZE + tx;
        int b_row = tile * TILE_SIZE + ty;
        int b_col = col;
        
        if (a_row < m && a_col < k) {
            shared_A[ty][tx] = A[a_row * k + a_col];
        } else {
            shared_A[ty][tx] = __float2half(0.0f);
        }
        
        if (b_row < k && b_col < n) {
            shared_B[ty][tx] = B[b_row * n + b_col];
        } else {
            shared_B[ty][tx] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        if (is_measuring_thread && tile == 0) {
            timing_info->global_to_shared_end = clock64();
            timing_info->compute_with_memory_start = clock64();
        }
        
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += __half2float(shared_A[ty][i]) * __half2float(shared_B[i][tx]);
        }
        
        if (is_measuring_thread && tile == 0) {
            timing_info->compute_with_memory_end = clock64();
        }
        
        __syncthreads();
    }
    
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
    
    if (is_measuring_thread) {
        timing_info->kernel_end = clock64();
    }
}

// プロファイリング関数群
DetailedTimingInfo profile_tensor_core_global(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    DetailedTimingInfo timing = {0};
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    tensor_core_global_memory_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    timing.total_kernel_time = kernel_time_ms / 1000.0f;
    
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f;
    
    timing.fragment_declare_time = (float)(h_timing.fragment_declare_end - h_timing.fragment_declare_start) / clock_rate;
    timing.fragment_init_time = (float)(h_timing.fragment_init_end - h_timing.fragment_init_start) / clock_rate;
    timing.global_to_frag_time = (float)(h_timing.global_to_frag_end - h_timing.global_to_frag_start) / clock_rate;
    timing.mma_sync_time = (float)(h_timing.mma_sync_end - h_timing.mma_sync_start) / clock_rate;
    timing.frag_to_global_time = (float)(h_timing.frag_to_global_end - h_timing.frag_to_global_start) / clock_rate;
    
    timing.tiles_per_result = K / WMMA_K;
    timing.elements_per_tile = WMMA_M * WMMA_N * WMMA_K;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return timing;
}

DetailedTimingInfo profile_tensor_core_shared(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    DetailedTimingInfo timing = {0};
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    tensor_core_shared_memory_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    timing.total_kernel_time = kernel_time_ms / 1000.0f;
    
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f;
    
    timing.fragment_declare_time = (float)(h_timing.fragment_declare_end - h_timing.fragment_declare_start) / clock_rate;
    timing.fragment_init_time = (float)(h_timing.fragment_init_end - h_timing.fragment_init_start) / clock_rate;
    timing.shared_declare_time = (float)(h_timing.shared_declare_end - h_timing.shared_declare_start) / clock_rate;
    timing.global_to_shared_time = (float)(h_timing.global_to_shared_end - h_timing.global_to_shared_start) / clock_rate;
    timing.shared_to_frag_time = (float)(h_timing.shared_to_frag_end - h_timing.shared_to_frag_start) / clock_rate;
    timing.mma_sync_time = (float)(h_timing.mma_sync_end - h_timing.mma_sync_start) / clock_rate;
    timing.frag_to_global_time = (float)(h_timing.frag_to_global_end - h_timing.frag_to_global_start) / clock_rate;
    
    timing.tiles_per_result = K / WMMA_K;
    timing.elements_per_tile = WMMA_M * WMMA_N * WMMA_K;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return timing;
}

DetailedTimingInfo profile_cuda_core_global(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    DetailedTimingInfo timing = {0};
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cuda_core_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    timing.total_kernel_time = kernel_time_ms / 1000.0f;
    
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f;
    
    timing.compute_with_memory_time = (float)(h_timing.compute_with_memory_end - h_timing.compute_with_memory_start) / clock_rate;
    
    // CUDA Coreの場合、1要素につき2*K回の演算
    timing.elements_per_tile = M * N;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return timing;
}

DetailedTimingInfo profile_cuda_core_shared(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    DetailedTimingInfo timing = {0};
    
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    cuda_core_shared_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float kernel_time_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, start, stop));
    timing.total_kernel_time = kernel_time_ms / 1000.0f;
    
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f;
    
    timing.shared_declare_time = (float)(h_timing.shared_declare_end - h_timing.shared_declare_start) / clock_rate;
    timing.global_to_shared_time = (float)(h_timing.global_to_shared_end - h_timing.global_to_shared_start) / clock_rate;
    timing.compute_with_memory_time = (float)(h_timing.compute_with_memory_end - h_timing.compute_with_memory_start) / clock_rate;
    
    timing.elements_per_tile = M * N;
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return timing;
}

// 結果検証
bool validate_results(float *C, float *C_ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C[i * N + j] - C_ref[i * N + j]) > 1e-1) {
                return false;
            }
        }
    }
    return true;
}

// CPU上での行列乗算（リファレンス実装）
double cpu_matmul(half *A, half *B, float *C) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += __half2float(A[i * K + k]) * __half2float(B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// 可視化用データ出力
void output_visualization_data(const DetailedTimingInfo& tc_global, const DetailedTimingInfo& tc_shared,
                              const DetailedTimingInfo& cuda_global, const DetailedTimingInfo& cuda_shared) {
    std::cout << "\n=== VISUALIZATION_DATA_START ===\n";
    
    // 1. 総計算時間の棒グラフ用データ
    std::cout << "TOTAL_COMPUTE_TIME,CUDA_Core_Global," << cuda_global.total_kernel_time * 1000000 << "\n";  // μs
    std::cout << "TOTAL_COMPUTE_TIME,CUDA_Core_Shared," << cuda_shared.total_kernel_time * 1000000 << "\n";
    std::cout << "TOTAL_COMPUTE_TIME,TensorCore_Global," << tc_global.total_kernel_time * 1000000 << "\n";
    std::cout << "TOTAL_COMPUTE_TIME,TensorCore_Shared," << tc_shared.total_kernel_time * 1000000 << "\n";
    
    // 2. 各操作の時間（16x16x16タイル基準で正規化）
    float scale_factor_cuda = (float)(WMMA_M * WMMA_N * WMMA_K) / cuda_global.elements_per_tile;
    
    std::cout << "OPERATION_TIME,CUDA_Core_Memory_Compute," << cuda_global.compute_with_memory_time * 1000000 * scale_factor_cuda << "\n";
    std::cout << "OPERATION_TIME,CUDA_Core_Shared_Declare," << cuda_shared.shared_declare_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,CUDA_Core_Global_to_Shared," << cuda_shared.global_to_shared_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,CUDA_Core_Shared_Compute," << cuda_shared.compute_with_memory_time * 1000000 << "\n";
    
    std::cout << "OPERATION_TIME,TensorCore_Fragment_Declare," << tc_global.fragment_declare_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_Fragment_Init," << tc_global.fragment_init_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_Global_to_Frag," << tc_global.global_to_frag_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_Frag_to_Global," << tc_global.frag_to_global_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_Global_to_Shared," << tc_shared.global_to_shared_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_Shared_to_Frag," << tc_shared.shared_to_frag_time * 1000000 << "\n";
    std::cout << "OPERATION_TIME,TensorCore_MMA_Sync," << tc_global.mma_sync_time * 1000000 << "\n";
    
    std::cout << "=== VISUALIZATION_DATA_END ===\n";
}

int main() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }
    
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using GPU: %s (Compute Capability %d.%d)\n", 
           deviceProp.name, deviceProp.major, deviceProp.minor);
    
    bool hasTensorCore = (deviceProp.major >= 7);
    if (!hasTensorCore) {
        printf("Warning: This GPU does not support Tensor Cores (requires SM 7.0+)\n");
    }
    
    half *A = (half*)malloc(M * K * sizeof(half));
    half *B = (half*)malloc(K * N * sizeof(half));
    float *C_tc = (float*)malloc(M * N * sizeof(float));
    float *C_tc_shared = (float*)malloc(M * N * sizeof(float));
    float *C_cuda_global = (float*)malloc(M * N * sizeof(float));
    float *C_cuda_shared = (float*)malloc(M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    
    printf("Initializing matrices...\n");
    init_matrices(A, B, C_tc, C_tc_shared, C_cuda_global, C_cuda_shared, C_ref);
    
    printf("Computing reference solution on CPU...\n");
    double cpu_time = cpu_matmul(A, B, C_ref);
    
    const int NUM_RUNS = 5;
    DetailedTimingInfo tc_global_avg = {0};
    DetailedTimingInfo tc_shared_avg = {0};
    DetailedTimingInfo cuda_global_avg = {0};
    DetailedTimingInfo cuda_shared_avg = {0};
    
    printf("Running benchmarks (%d iterations)...\n", NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++) {
        printf("  Run %d/%d\n", i+1, NUM_RUNS);
        
        DetailedTimingInfo tc_global = profile_tensor_core_global(A, B, C_tc);
        DetailedTimingInfo tc_shared = profile_tensor_core_shared(A, B, C_tc_shared);
        DetailedTimingInfo cuda_global = profile_cuda_core_global(A, B, C_cuda_global);
        DetailedTimingInfo cuda_shared = profile_cuda_core_shared(A, B, C_cuda_shared);
        
        // 平均計算用に累積
        tc_global_avg.total_kernel_time += tc_global.total_kernel_time;
        tc_global_avg.fragment_declare_time += tc_global.fragment_declare_time;
        tc_global_avg.fragment_init_time += tc_global.fragment_init_time;
        tc_global_avg.global_to_frag_time += tc_global.global_to_frag_time;
        tc_global_avg.mma_sync_time += tc_global.mma_sync_time;
        tc_global_avg.frag_to_global_time += tc_global.frag_to_global_time;
        
        tc_shared_avg.total_kernel_time += tc_shared.total_kernel_time;
        tc_shared_avg.fragment_declare_time += tc_shared.fragment_declare_time;
        tc_shared_avg.fragment_init_time += tc_shared.fragment_init_time;
        tc_shared_avg.shared_declare_time += tc_shared.shared_declare_time;
        tc_shared_avg.global_to_shared_time += tc_shared.global_to_shared_time;
        tc_shared_avg.shared_to_frag_time += tc_shared.shared_to_frag_time;
        tc_shared_avg.mma_sync_time += tc_shared.mma_sync_time;
        tc_shared_avg.frag_to_global_time += tc_shared.frag_to_global_time;
        
        cuda_global_avg.total_kernel_time += cuda_global.total_kernel_time;
        cuda_global_avg.compute_with_memory_time += cuda_global.compute_with_memory_time;
        cuda_global_avg.elements_per_tile = cuda_global.elements_per_tile;
        
        cuda_shared_avg.total_kernel_time += cuda_shared.total_kernel_time;
        cuda_shared_avg.shared_declare_time += cuda_shared.shared_declare_time;
        cuda_shared_avg.global_to_shared_time += cuda_shared.global_to_shared_time;
        cuda_shared_avg.compute_with_memory_time += cuda_shared.compute_with_memory_time;
        cuda_shared_avg.elements_per_tile = cuda_shared.elements_per_tile;
        
        sleep(1);
    }
    
    // 平均計算
    tc_global_avg.total_kernel_time /= NUM_RUNS;
    tc_global_avg.fragment_declare_time /= NUM_RUNS;
    tc_global_avg.fragment_init_time /= NUM_RUNS;
    tc_global_avg.global_to_frag_time /= NUM_RUNS;
    tc_global_avg.mma_sync_time /= NUM_RUNS;
    tc_global_avg.frag_to_global_time /= NUM_RUNS;
    
    tc_shared_avg.total_kernel_time /= NUM_RUNS;
    tc_shared_avg.fragment_declare_time /= NUM_RUNS;
    tc_shared_avg.fragment_init_time /= NUM_RUNS;
    tc_shared_avg.shared_declare_time /= NUM_RUNS;
    tc_shared_avg.global_to_shared_time /= NUM_RUNS;
    tc_shared_avg.shared_to_frag_time /= NUM_RUNS;
    tc_shared_avg.mma_sync_time /= NUM_RUNS;
    tc_shared_avg.frag_to_global_time /= NUM_RUNS;
    
    cuda_global_avg.total_kernel_time /= NUM_RUNS;
    cuda_global_avg.compute_with_memory_time /= NUM_RUNS;
    
    cuda_shared_avg.total_kernel_time /= NUM_RUNS;
    cuda_shared_avg.shared_declare_time /= NUM_RUNS;
    cuda_shared_avg.global_to_shared_time /= NUM_RUNS;
    cuda_shared_avg.compute_with_memory_time /= NUM_RUNS;
    
    // 結果検証
    printf("Validating results...\n");
    bool tc_passed = validate_results(C_tc, C_ref);
    bool tc_shared_passed = validate_results(C_tc_shared, C_ref);
    bool cuda_global_passed = validate_results(C_cuda_global, C_ref);
    bool cuda_shared_passed = validate_results(C_cuda_shared, C_ref);
    
    printf("\n===== Performance Results =====\n");
    printf("Matrix Size: %d x %d x %d\n", M, N, K);
    printf("CPU Time: %.6f seconds\n\n", cpu_time);
    
    printf("TensorCore Global %s: %.6f seconds (%.2fx speedup)\n", 
           tc_passed ? "(PASSED)" : "(FAILED)",
           tc_global_avg.total_kernel_time, 
           cpu_time / tc_global_avg.total_kernel_time);
    
    printf("TensorCore Shared %s: %.6f seconds (%.2fx speedup)\n", 
           tc_shared_passed ? "(PASSED)" : "(FAILED)",
           tc_shared_avg.total_kernel_time, 
           cpu_time / tc_shared_avg.total_kernel_time);
    
    printf("CUDA Core Global %s: %.6f seconds (%.2fx speedup)\n", 
           cuda_global_passed ? "(PASSED)" : "(FAILED)",
           cuda_global_avg.total_kernel_time, 
           cpu_time / cuda_global_avg.total_kernel_time);
    
    printf("CUDA Core Shared %s: %.6f seconds (%.2fx speedup)\n", 
           cuda_shared_passed ? "(PASSED)" : "(FAILED)",
           cuda_shared_avg.total_kernel_time, 
           cpu_time / cuda_shared_avg.total_kernel_time);
    
    // 可視化用データ出力
    output_visualization_data(tc_global_avg, tc_shared_avg, cuda_global_avg, cuda_shared_avg);
    
    free(A);
    free(B);
    free(C_tc);
    free(C_tc_shared);
    free(C_cuda_global);
    free(C_cuda_shared);
    free(C_ref);
    
    return 0;
}