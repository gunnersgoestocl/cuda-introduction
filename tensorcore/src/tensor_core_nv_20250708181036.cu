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
    float total_time;            // 全体の実行時間
    float copy_in_time;          // ホスト→デバイスのコピー時間
    float kernel_time;           // カーネル実行時間
    float global_to_frag_time;   // グローバルメモリ→フラグメント時間
    float global_to_shared_time; // グローバルメモリ→共有メモリ時間
    float shared_to_frag_time;   // 共有メモリ→フラグメント時間
    float compute_time;          // 計算時間
    float store_time;            // 結果の書き戻し時間
    float copy_out_time;         // デバイス→ホストのコピー時間
    float sync_overhead;         // 同期オーバーヘッド
} TimingInfo;

// 時間計測結果を格納する構造体（デバイス用）
typedef struct {
    uint64_t global_to_frag_start;
    uint64_t global_to_frag_end;
    uint64_t global_to_shared_start;
    uint64_t global_to_shared_end;
    uint64_t shared_to_frag_start;
    uint64_t shared_to_frag_end;
    uint64_t compute_start;
    uint64_t compute_end;
    uint64_t store_start;
    uint64_t store_end;
    uint64_t sync1_start;
    uint64_t sync1_end;
    uint64_t sync2_start;
    uint64_t sync2_end;
} DeviceTimingInfo;

// 行列初期化
void init_matrices(half *A, half *B, float *C) {
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
            C[i * N + j] = 0.0f;
        }
    }
}

// 結果検証
bool validate_results(float *C, float *C_ref) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C[i * N + j] - C_ref[i * N + j]) > 1e-1) {
                printf("Mismatch at (%d, %d): GPU=%f, CPU=%f\n", 
                       i, j, C[i * N + j], C_ref[i * N + j]);
                return false;
            }
        }
    }
    return true;
}

// グローバルメモリからフラグメントへ直接ロードする方式のカーネル（clock64で計測）
__global__ void tensor_core_global_memory_kernel(half *A, half *B, float *C, 
                                               int m, int n, int k,
                                               DeviceTimingInfo *timing_info) {
    // ブロックの位置を計算
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    // グローバルメモリ上での開始位置を計算
    const int row_start = block_row * WMMA_M;
    const int col_start = block_col * WMMA_N;
    
    // このブロックが担当する行列の範囲がサイズ内にあるか確認
    if (row_start >= m || col_start >= n) return;
    
    // 時間計測用フラグ（スレッド0のみが計測）
    bool is_measuring_thread = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    
    // フラグメント宣言
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 結果フラグメントを0で初期化
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // K次元の各ステップでデータを計算
    for (int i = 0; i < k; i += WMMA_K) {
        if (i + WMMA_K > k) break;
        
        // グローバルメモリからフラグメントへのロード時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_frag_start = clock64();
        }
        
        // Aフラグメントをグローバルメモリから直接ロード
        nvcuda::wmma::load_matrix_sync(a_frag, A + (row_start * k + i), k);
        
        // Bフラグメントをグローバルメモリから直接ロード
        nvcuda::wmma::load_matrix_sync(b_frag, B + (i * n + col_start), n);
        
        // グローバルメモリからフラグメントへのロード時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_frag_end = clock64();
        }
        
        // 計算時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->compute_start = clock64();
        }
        
        // 行列乗算を実行（TensorCore命令を使用）
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // 計算時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->compute_end = clock64();
        }
    }
    
    // 結果の書き戻し時間測定開始
    if (is_measuring_thread) {
        timing_info->store_start = clock64();
    }
    
    // 結果をグローバルメモリに書き戻す
    nvcuda::wmma::store_matrix_sync(C + (row_start * n + col_start), c_frag, n, nvcuda::wmma::mem_row_major);
    
    // 結果の書き戻し時間測定終了
    if (is_measuring_thread) {
        timing_info->store_end = clock64();
    }
}

// 共有メモリを使用したTensorCoreカーネル（正しいパディング実装）
__global__ void tensor_core_shared_memory_kernel(half *A, half *B, float *C, 
                                               int m, int n, int k,
                                               DeviceTimingInfo *timing_info) {
    // 共有メモリのパディング（16バイトアライメントを維持）
    // パディングサイズをWMMA_Kの倍数に調整
    const int padded_k = (WMMA_K + 8) & ~0xF; // 16の倍数になるようにパディング
    const int padded_n = (WMMA_N + 8) & ~0xF; // 16の倍数になるようにパディング
    
    // アライメントを保ったパディング付き共有メモリ
    __shared__ half shared_A[WMMA_M][padded_k];
    __shared__ half shared_B[WMMA_K][padded_n];
    
    // ブロックの位置を計算
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    // グローバルメモリ上での開始位置を計算
    const int row_start = block_row * WMMA_M;
    const int col_start = block_col * WMMA_N;
    
    // このブロックが担当する行列の範囲がサイズ内にあるか確認
    if (row_start >= m || col_start >= n) return;
    
    // 時間計測用フラグ（スレッド0のみが計測）
    bool is_measuring_thread = (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0);
    
    // スレッドID
    const int tid = threadIdx.x;
    
    // フラグメント宣言
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // 結果フラグメントを0で初期化
    nvcuda::wmma::fill_fragment(c_frag, 0.0f);
    
    // K次元の各ステップでデータを計算
    for (int i = 0; i < k; i += WMMA_K) {
        if (i + WMMA_K > k) break;
        
        // 共有メモリの初期化（オプション - デバッグに役立つ）
        for (int j = tid; j < WMMA_M * padded_k; j += blockDim.x) {
            int row = j / padded_k;
            int col = j % padded_k;
            shared_A[row][col] = __float2half(0.0f);
        }
        
        for (int j = tid; j < WMMA_K * padded_n; j += blockDim.x) {
            int row = j / padded_n;
            int col = j % padded_n;
            shared_B[row][col] = __float2half(0.0f);
        }

        // グローバルメモリから共有メモリへのロード時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_shared_start = clock64();
        }
        
        // 各スレッドが複数の要素をロード
        for (int j = tid; j < WMMA_M * WMMA_K; j += blockDim.x) {
            int row = j / WMMA_K;
            int col = j % WMMA_K;
            
            // A行列をグローバルメモリから共有メモリにロード
            if (row_start + row < m && i + col < k) {
                shared_A[row][col] = A[(row_start + row) * k + (i + col)];
            }
        }
        
        for (int j = tid; j < WMMA_K * WMMA_N; j += blockDim.x) {
            int row = j / WMMA_N;
            int col = j % WMMA_N;
            
            // B行列をグローバルメモリから共有メモリにロード
            if (i + row < k && col_start + col < n) {
                shared_B[row][col] = B[(i + row) * n + (col_start + col)];
            }
        }
        
        // グローバルメモリから共有メモリへのロード時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->global_to_shared_end = clock64();
        }
        
        // 同期1時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->sync1_start = clock64();
        }
        
        // 同期 - すべてのスレッドがデータをロードするまで待機
        __syncthreads();
        
        // 同期1時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->sync1_end = clock64();
        }
        
        // 共有メモリからフラグメントへのロード時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->shared_to_frag_start = clock64();
        }
        
        // 共有メモリからフラグメントにロード - ストライドに正しいパディング値を使用
        nvcuda::wmma::load_matrix_sync(a_frag, &shared_A[0][0], padded_k);
        nvcuda::wmma::load_matrix_sync(b_frag, &shared_B[0][0], padded_n);
        
        // 共有メモリからフラグメントへのロード時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->shared_to_frag_end = clock64();
        }
        
        // 計算時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->compute_start = clock64();
        }
        
        // 行列乗算を実行（TensorCore命令を使用）
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        // 計算時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->compute_end = clock64();
        }
        
        // 同期2時間測定開始
        if (is_measuring_thread && i == 0) {
            timing_info->sync2_start = clock64();
        }
        
        // 同期 - 次のタイルをロードする前にすべての計算が完了するまで待機
        __syncthreads();
        
        // 同期2時間測定終了
        if (is_measuring_thread && i == 0) {
            timing_info->sync2_end = clock64();
        }
    }
    
    // 結果の書き戻し時間測定開始
    if (is_measuring_thread) {
        timing_info->store_start = clock64();
    }
    
    // 結果をグローバルメモリに書き戻す
    nvcuda::wmma::store_matrix_sync(C + (row_start * n + col_start), c_frag, n, nvcuda::wmma::mem_row_major);
    
    // 結果の書き戻し時間測定終了
    if (is_measuring_thread) {
        timing_info->store_end = clock64();
    }
}

// グローバルメモリ版プロファイリング関数
TimingInfo profile_tensor_core_global(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    TimingInfo timing = {0};
    
    // イベント作成
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_copy_in, stop_copy_in;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_copy_out, stop_copy_out;
    
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_copy_in));
    CUDA_CHECK(cudaEventCreate(&stop_copy_in));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_copy_out));
    CUDA_CHECK(cudaEventCreate(&stop_copy_out));
    
    // 全体の時間計測開始
    CUDA_CHECK(cudaEventRecord(start_total));
    
    // GPUメモリ確保
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // 時間計測用構造体のメモリ確保
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    // データ転送時間計測開始
    CUDA_CHECK(cudaEventRecord(start_copy_in));
    
    // データをGPUにコピー
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // データ転送時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_copy_in));
    CUDA_CHECK(cudaEventSynchronize(stop_copy_in));
    
    // カーネル設定
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);  // TensorCoreはワープ単位で動作するため32スレッド
    
    // カーネル実行時間計測開始
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    // カーネル実行
    tensor_core_global_memory_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    
    // カーネル実行時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    // 時間計測データをホストに転送
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    
    // 結果転送時間計測開始
    CUDA_CHECK(cudaEventRecord(start_copy_out));
    
    // 結果をホストにコピー
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 結果転送時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_copy_out));
    CUDA_CHECK(cudaEventSynchronize(stop_copy_out));
    
    // 全体の時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    
    // 時間を計算
    CUDA_CHECK(cudaEventElapsedTime(&timing.total_time, start_total, stop_total));
    CUDA_CHECK(cudaEventElapsedTime(&timing.copy_in_time, start_copy_in, stop_copy_in));
    CUDA_CHECK(cudaEventElapsedTime(&timing.kernel_time, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&timing.copy_out_time, start_copy_out, stop_copy_out));
    
    // GPUのクロックレートを取得（近似値としてプロパティから取得）
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f; // Hz単位
    
    // clock64()の時間をミリ秒に変換
    timing.global_to_frag_time = 
        (float)(h_timing.global_to_frag_end - h_timing.global_to_frag_start) / clock_rate * 1000.0f;
    timing.compute_time = 
        (float)(h_timing.compute_end - h_timing.compute_start) / clock_rate * 1000.0f;
    timing.store_time = 
        (float)(h_timing.store_end - h_timing.store_start) / clock_rate * 1000.0f;
    
    // リソース解放
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_copy_in));
    CUDA_CHECK(cudaEventDestroy(stop_copy_in));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_copy_out));
    CUDA_CHECK(cudaEventDestroy(stop_copy_out));
    
    // ミリ秒から秒に変換
    timing.total_time /= 1000.0f;
    timing.copy_in_time /= 1000.0f;
    timing.kernel_time /= 1000.0f;
    timing.copy_out_time /= 1000.0f;
    timing.global_to_frag_time /= 1000.0f;
    timing.compute_time /= 1000.0f;
    timing.store_time /= 1000.0f;
    
    return timing;
}

// 共有メモリ版プロファイリング関数
TimingInfo profile_tensor_core_shared(half *h_A, half *h_B, float *h_C) {
    half *d_A, *d_B;
    float *d_C;
    TimingInfo timing = {0};
    
    // イベント作成
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_copy_in, stop_copy_in;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_copy_out, stop_copy_out;
    
    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_copy_in));
    CUDA_CHECK(cudaEventCreate(&stop_copy_in));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_copy_out));
    CUDA_CHECK(cudaEventCreate(&stop_copy_out));
    
    // 全体の時間計測開始
    CUDA_CHECK(cudaEventRecord(start_total));
    
    // GPUメモリ確保
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    
    // 時間計測用構造体のメモリ確保
    DeviceTimingInfo *d_timing;
    DeviceTimingInfo h_timing;
    CUDA_CHECK(cudaMalloc(&d_timing, sizeof(DeviceTimingInfo)));
    
    // データ転送時間計測開始
    CUDA_CHECK(cudaEventRecord(start_copy_in));
    
    // データをGPUにコピー
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    
    // データ転送時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_copy_in));
    CUDA_CHECK(cudaEventSynchronize(stop_copy_in));
    
    // カーネル設定
    dim3 grid((N + WMMA_N - 1) / WMMA_N, (M + WMMA_M - 1) / WMMA_M);
    dim3 block(32);  // TensorCoreはワープ単位で動作するため32スレッド
    
    // カーネル実行時間計測開始
    CUDA_CHECK(cudaEventRecord(start_kernel));
    
    // カーネル実行
    tensor_core_shared_memory_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, d_timing);
    
    // カーネル実行時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_kernel));
    CUDA_CHECK(cudaEventSynchronize(stop_kernel));
    
    // 時間計測データをホストに転送
    CUDA_CHECK(cudaMemcpy(&h_timing, d_timing, sizeof(DeviceTimingInfo), cudaMemcpyDeviceToHost));
    
    // 結果転送時間計測開始
    CUDA_CHECK(cudaEventRecord(start_copy_out));
    
    // 結果をホストにコピー
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 結果転送時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_copy_out));
    CUDA_CHECK(cudaEventSynchronize(stop_copy_out));
    
    // 全体の時間計測終了
    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));
    
    // 時間を計算
    CUDA_CHECK(cudaEventElapsedTime(&timing.total_time, start_total, stop_total));
    CUDA_CHECK(cudaEventElapsedTime(&timing.copy_in_time, start_copy_in, stop_copy_in));
    CUDA_CHECK(cudaEventElapsedTime(&timing.kernel_time, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&timing.copy_out_time, start_copy_out, stop_copy_out));
    
    // GPUのクロックレートを取得（近似値としてプロパティから取得）
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    float clock_rate = prop.clockRate * 1000.0f; // Hz単位
    
    // clock64()の時間をミリ秒に変換
    timing.global_to_shared_time = 
        (float)(h_timing.global_to_shared_end - h_timing.global_to_shared_start) / clock_rate * 1000.0f;
    timing.shared_to_frag_time = 
        (float)(h_timing.shared_to_frag_end - h_timing.shared_to_frag_start) / clock_rate * 1000.0f;
    timing.compute_time = 
        (float)(h_timing.compute_end - h_timing.compute_start) / clock_rate * 1000.0f;
    timing.store_time = 
        (float)(h_timing.store_end - h_timing.store_start) / clock_rate * 1000.0f;
    timing.sync_overhead = 
        (float)((h_timing.sync1_end - h_timing.sync1_start) + 
                (h_timing.sync2_end - h_timing.sync2_start)) / clock_rate * 1000.0f;
    
    // リソース解放
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_timing));
    
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_copy_in));
    CUDA_CHECK(cudaEventDestroy(stop_copy_in));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_copy_out));
    CUDA_CHECK(cudaEventDestroy(stop_copy_out));
    
    // ミリ秒から秒に変換
    timing.total_time /= 1000.0f;
    timing.copy_in_time /= 1000.0f;
    timing.kernel_time /= 1000.0f;
    timing.copy_out_time /= 1000.0f;
    timing.global_to_shared_time /= 1000.0f;
    timing.shared_to_frag_time /= 1000.0f;
    timing.compute_time /= 1000.0f;
    timing.store_time /= 1000.0f;
    timing.sync_overhead /= 1000.0f;
    
    return timing;
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

// タイムラインデータ出力関数を修正
void output_timeline_data(const TimingInfo& global_timing, const TimingInfo& shared_timing) {
    std::cout << "\n=== TIMELINE_DATA_START ===\n";
    
    // グローバルメモリ版のタイムライン（開始時刻を0とする）
    float global_start = 0.0f;
    float global_copy_in_end = global_timing.copy_in_time * 1000;
    float global_kernel_start = global_copy_in_end;
    float global_global_to_frag_end = global_kernel_start + global_timing.global_to_frag_time * 1000;
    float global_compute_end = global_global_to_frag_end + global_timing.compute_time * 1000;
    float global_store_end = global_compute_end + global_timing.store_time * 1000;
    float global_kernel_end = global_kernel_start + global_timing.kernel_time * 1000;
    float global_copy_out_end = global_kernel_end + global_timing.copy_out_time * 1000;
    
    std::cout << "PHASE,Global,H2D_Copy," << global_start << "," << global_copy_in_end << "\n";
    std::cout << "PHASE,Global,Global_to_Frag," << global_kernel_start << "," << global_global_to_frag_end << "\n";
    std::cout << "PHASE,Global,Compute," << global_global_to_frag_end << "," << global_compute_end << "\n";
    std::cout << "PHASE,Global,Store," << global_compute_end << "," << global_store_end << "\n";
    std::cout << "PHASE,Global,D2H_Copy," << global_kernel_end << "," << global_copy_out_end << "\n";
    
    // 共有メモリ版のタイムライン（開始時刻を0とする）
    float shared_start = 0.0f;
    float shared_copy_in_end = shared_timing.copy_in_time * 1000;
    float shared_kernel_start = shared_copy_in_end;
    float shared_global_to_shared_end = shared_kernel_start + shared_timing.global_to_shared_time * 1000;
    float shared_shared_to_frag_end = shared_global_to_shared_end + shared_timing.shared_to_frag_time * 1000;
    float shared_compute_end = shared_shared_to_frag_end + shared_timing.compute_time * 1000;
    float shared_store_end = shared_compute_end + shared_timing.store_time * 1000;
    float shared_kernel_end = shared_kernel_start + shared_timing.kernel_time * 1000;
    float shared_copy_out_end = shared_kernel_end + shared_timing.copy_out_time * 1000;
    
    std::cout << "PHASE,Shared,H2D_Copy," << shared_start << "," << shared_copy_in_end << "\n";
    std::cout << "PHASE,Shared,Global_to_Shared," << shared_kernel_start << "," << shared_global_to_shared_end << "\n";
    std::cout << "PHASE,Shared,Shared_to_Frag," << shared_global_to_shared_end << "," << shared_shared_to_frag_end << "\n";
    std::cout << "PHASE,Shared,Compute," << shared_shared_to_frag_end << "," << shared_compute_end << "\n";
    std::cout << "PHASE,Shared,Store," << shared_compute_end << "," << shared_store_end << "\n";
    std::cout << "PHASE,Shared,D2H_Copy," << shared_kernel_end << "," << shared_copy_out_end << "\n";
    
    std::cout << "=== TIMELINE_DATA_END ===\n";
}

int main() {
    // CUDA対応チェック
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }
    
    // デバイス情報の取得
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    printf("Using GPU: %s (Compute Capability %d.%d)\n", 
           deviceProp.name, deviceProp.major, deviceProp.minor);
    
    // GPUクロック周波数の取得
    printf("GPU Clock Rate: %.2f GHz\n", deviceProp.clockRate / 1000000.0f);
    
    // TensorCoreサポートの確認
    bool hasTensorCore = (deviceProp.major >= 7);
    if (!hasTensorCore) {
        printf("Warning: This GPU does not support Tensor Cores (requires SM 7.0+)\n");
    } else {
        printf("Tensor Core support detected!\n");
    }
    
    // メモリ確保
    half *A = (half*)malloc(M * K * sizeof(half));
    half *B = (half*)malloc(K * N * sizeof(half));
    float *C_tc = (float*)malloc(M * N * sizeof(float));
    float *C_tc_shared = (float*)malloc(M * N * sizeof(float));
    float *C_ref = (float*)malloc(M * N * sizeof(float));
    
    // 行列の初期化
    printf("Initializing matrices...\n");
    init_matrices(A, B, C_ref);
    
    // CPU上での計算（リファレンス）
    printf("Computing reference solution on CPU...\n");
    double cpu_time = cpu_matmul(A, B, C_ref);
    
    // GPUのウォームアップ
    printf("Warming up GPU...\n");
    TimingInfo warmup1 = profile_tensor_core_global(A, B, C_tc);
    TimingInfo warmup2 = profile_tensor_core_shared(A, B, C_tc_shared);
    
    // ベンチマーク実行回数
    const int NUM_RUNS = 5;
    TimingInfo global_timing = {0};
    TimingInfo shared_timing = {0};
    
    // CUDAカーネルを複数回実行して平均を取る
    printf("Running benchmarks (%d iterations)...\n", NUM_RUNS);
    for (int i = 0; i < NUM_RUNS; i++) {
        printf("  Run %d/%d\n", i+1, NUM_RUNS);
        
        // Tensor Core (グローバルメモリ)
        TimingInfo timing_global = profile_tensor_core_global(A, B, C_tc);
        global_timing.total_time += timing_global.total_time;
        global_timing.copy_in_time += timing_global.copy_in_time;
        global_timing.kernel_time += timing_global.kernel_time;
        global_timing.global_to_frag_time += timing_global.global_to_frag_time;
        global_timing.compute_time += timing_global.compute_time;
        global_timing.store_time += timing_global.store_time;
        global_timing.copy_out_time += timing_global.copy_out_time;
        
        sleep(1);
        
        // Tensor Core (共有メモリ)
        TimingInfo timing_shared = profile_tensor_core_shared(A, B, C_tc_shared);
        shared_timing.total_time += timing_shared.total_time;
        shared_timing.copy_in_time += timing_shared.copy_in_time;
        shared_timing.kernel_time += timing_shared.kernel_time;
        shared_timing.global_to_shared_time += timing_shared.global_to_shared_time;
        shared_timing.shared_to_frag_time += timing_shared.shared_to_frag_time;
        shared_timing.compute_time += timing_shared.compute_time;
        shared_timing.store_time += timing_shared.store_time;
        shared_timing.copy_out_time += timing_shared.copy_out_time;
        shared_timing.sync_overhead += timing_shared.sync_overhead;
        
        sleep(1);
    }
    
    // 平均計算
    global_timing.total_time /= NUM_RUNS;
    global_timing.copy_in_time /= NUM_RUNS;
    global_timing.kernel_time /= NUM_RUNS;
    global_timing.global_to_frag_time /= NUM_RUNS;
    global_timing.compute_time /= NUM_RUNS;
    global_timing.store_time /= NUM_RUNS;
    global_timing.copy_out_time /= NUM_RUNS;
    
    shared_timing.total_time /= NUM_RUNS;
    shared_timing.copy_in_time /= NUM_RUNS;
    shared_timing.kernel_time /= NUM_RUNS;
    shared_timing.global_to_shared_time /= NUM_RUNS;
    shared_timing.shared_to_frag_time /= NUM_RUNS;
    shared_timing.compute_time /= NUM_RUNS;
    shared_timing.store_time /= NUM_RUNS;
    shared_timing.copy_out_time /= NUM_RUNS;
    shared_timing.sync_overhead /= NUM_RUNS;
    
    // 結果の検証
    printf("Validating results...\n");
    bool tc_passed = validate_results(C_tc, C_ref);
    bool tc_shared_passed = validate_results(C_tc_shared, C_ref);
    
    // 結果表示
    printf("\n===== Detailed Performance Results (clock64()を使用) =====\n");
    printf("Matrix Size: %d x %d x %d\n", M, N, K);
    printf("CPU Time: %.6f seconds\n\n", cpu_time);
    
    printf("--- Tensor Core (Global Memory) %s ---\n", tc_passed ? "(PASSED)" : "(FAILED)");
    printf("Total Time: %.6f seconds\n", global_timing.total_time);
    printf("  Copy H->D:     %.6f seconds (%.1f%%)\n", global_timing.copy_in_time, 
           (global_timing.copy_in_time / global_timing.total_time) * 100);
    printf("  Kernel Total:  %.6f seconds (%.1f%%)\n", global_timing.kernel_time, 
           (global_timing.kernel_time / global_timing.total_time) * 100);
    printf("    Global->Frag:  %.9f seconds\n", global_timing.global_to_frag_time);
    printf("    Compute:       %.9f seconds\n", global_timing.compute_time);
    printf("    Store:         %.9f seconds\n", global_timing.store_time);
    printf("  Copy D->H:     %.6f seconds (%.1f%%)\n", global_timing.copy_out_time, 
           (global_timing.copy_out_time / global_timing.total_time) * 100);
    
    printf("\n--- Tensor Core (Shared Memory) %s ---\n", tc_shared_passed ? "(PASSED)" : "(FAILED)");
    printf("Total Time: %.6f seconds\n", shared_timing.total_time);
    printf("  Copy H->D:     %.6f seconds (%.1f%%)\n", shared_timing.copy_in_time, 
           (shared_timing.copy_in_time / shared_timing.total_time) * 100);
    printf("  Kernel Total:  %.6f seconds (%.1f%%)\n", shared_timing.kernel_time, 
           (shared_timing.kernel_time / shared_timing.total_time) * 100);
    printf("    Global->Shared:  %.9f seconds\n", shared_timing.global_to_shared_time);
    printf("    Shared->Frag:    %.9f seconds\n", shared_timing.shared_to_frag_time);
    printf("    Compute:         %.9f seconds\n", shared_timing.compute_time);
    printf("    Store:           %.9f seconds\n", shared_timing.store_time);
    printf("    Sync Overhead:   %.9f seconds\n", shared_timing.sync_overhead);
    printf("  Copy D->H:     %.6f seconds (%.1f%%)\n", shared_timing.copy_out_time, 
           (shared_timing.copy_out_time / shared_timing.total_time) * 100);
    
    // フェーズごとの比較
    printf("\n===== Phase Comparison (Global vs Shared) =====\n");
    printf("Total Kernel Time: %.2fx slower with shared memory\n", 
           shared_timing.kernel_time / global_timing.kernel_time);
    
    // データロード比較（Global->Fragと、Global->Shared+Shared->Fragの比較）
    float global_load_time = global_timing.global_to_frag_time;
    float shared_load_time = shared_timing.global_to_shared_time + shared_timing.shared_to_frag_time;
    printf("Data Loading:      %.2fx slower with shared memory\n", 
           shared_load_time / global_load_time);
    
    printf("Computation:       %.2fx different with shared memory\n", 
           shared_timing.compute_time / global_timing.compute_time);
    printf("Result Storage:    %.2fx different with shared memory\n", 
           shared_timing.store_time / global_timing.store_time);
    
    // 詳細な性能分析
    printf("\n===== Detailed Analysis =====\n");
    
    // 1. 共有メモリオーバーヘッド分析
    float shared_overhead = shared_load_time - global_load_time;
    printf("Shared Memory Overhead: %.9f seconds\n", shared_overhead);
    printf("  - Global->Shared:     %.9f seconds\n", shared_timing.global_to_shared_time);
    printf("  - Shared->Frag:       %.9f seconds\n", shared_timing.shared_to_frag_time);
    printf("  - Direct Global->Frag: %.9f seconds\n", global_timing.global_to_frag_time);
    
    // 2. 同期オーバーヘッド分析
    printf("Synchronization Overhead: %.9f seconds (%.1f%% of shared kernel time)\n", 
           shared_timing.sync_overhead, 
           (shared_timing.sync_overhead / shared_timing.kernel_time) * 100);
    
    // 3. 理論的なTeraFLOPS計算（FP16入力、FP32蓄積のTensor Core演算）
    float operations = 2.0f * M * N * K; // 行列乗算の演算数（乗算+加算）
    float teraflops_global = operations / (global_timing.compute_time * 1e12);
    float teraflops_shared = operations / (shared_timing.compute_time * 1e12);
    printf("Compute Performance:\n");
    printf("  - Global Memory: %.2f TFLOPS\n", teraflops_global);
    printf("  - Shared Memory: %.2f TFLOPS\n", teraflops_shared);

    // タイムラインデータの出力を追加
    output_timeline_data(global_timing, shared_timing);
    
    // リソース解放
    free(A);
    free(B);
    free(C_tc);
    free(C_tc_shared);
    free(C_ref);
    
    return 0;
}