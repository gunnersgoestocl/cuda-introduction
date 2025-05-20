#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <time.h>

// CUDA エラーチェックマクロ
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

// 時間計測用関数
double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

// 行列を定数倍するカーネル関数
__global__ void matrixMultiplyByScalar(float *matrix, int rows, int cols, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        matrix[index] = matrix[index] * scalar;
    }
}

// 単一GPUでの処理関数
void singleGPU_process(float *matrix, int rows, int cols, float scalar) {
    // デバイス選択
    cudaSetDevice(0);
    
    // サイズ計算
    size_t bytes = rows * cols * sizeof(float);
    
    // デバイスメモリ確保
    float *d_matrix;
    cudaMalloc(&d_matrix, bytes);
    cudaCheckError();
    
    // ホストからデバイスへデータ転送
    cudaMemcpy(d_matrix, matrix, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // カーネル設定
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // カーネル実行
    matrixMultiplyByScalar<<<gridSize, blockSize>>>(d_matrix, rows, cols, scalar);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // デバイスからホストへデータ転送
    cudaMemcpy(matrix, d_matrix, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // デバイスメモリの解放
    cudaFree(d_matrix);
}

// 各GPUで実行する関数（マルチGPU用）
void process_on_device(int gpu, float *matrix, int startRow, int numRows, int totalCols, float scalar) {
    // GPUの設定
    cudaSetDevice(gpu);
    
    // 処理する行列部分のサイズを計算
    size_t bytes = numRows * totalCols * sizeof(float);
    
    // デバイスメモリの確保
    float *d_matrix;
    cudaMalloc(&d_matrix, bytes);
    cudaCheckError();
    
    // ホストからデバイスへデータ転送
    cudaMemcpy(d_matrix, matrix + startRow * totalCols, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // カーネル設定
    dim3 blockSize(16, 16);
    dim3 gridSize((totalCols + blockSize.x - 1) / blockSize.x, 
                  (numRows + blockSize.y - 1) / blockSize.y);
    
    // カーネル実行
    matrixMultiplyByScalar<<<gridSize, blockSize>>>(d_matrix, numRows, totalCols, scalar);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // デバイスからホストへデータ転送
    cudaMemcpy(matrix + startRow * totalCols, d_matrix, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // デバイスメモリの解放
    cudaFree(d_matrix);
}

// 行列の初期化
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<float>(i * cols + j + 1);
        }
    }
}

// 指定GPUの情報表示関数
void printGPUInfo(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("GPU %d: %s\n", deviceId, prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

// 行列の一部を表示
void printMatrixSample(const char* name, float *matrix, int rows, int cols) {
    std::cout << name << " (" << rows << "x" << cols << ") の一部:" << std::endl;
    
    if (rows * cols <= 36) {
        // 小さい行列は全て表示
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        // 大きい行列は一部だけ表示
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << matrix[i * cols + j] << " ";
            }
            std::cout << " ... ";
            for (int j = cols - 2; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << " ... " << std::endl;
        for (int i = rows - 2; i < rows; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << matrix[i * cols + j] << " ";
            }
            std::cout << " ... ";
            for (int j = cols - 2; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(1) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// 結果の検証
bool verifyResults(float *result1, float *result2, int rows, int cols, float tolerance) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if (fabs(result1[idx] - result2[idx]) > tolerance) {
                printf("結果が一致しません！位置 [%d, %d]: %.4f != %.4f\n", 
                       i, j, result1[idx], result2[idx]);
                return false;
            }
        }
    }
    return true;
}

// メイン関数
int main(int argc, char **argv) {
    // コマンドライン引数の確認
    if (argc < 3) {
        fprintf(stderr, "使用方法: %s <行列サイズ> <使用GPU数>\n", argv[0]);
        return -1;
    }
    
    // コマンドラインから引数を取得
    int matrixSize = atoi(argv[1]);   // 行列サイズ
    int numGPUs = atoi(argv[2]);      // 使用GPU数
    
    // 利用可能なGPU数を確認
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("利用可能なGPUデバイス数: %d\n", deviceCount);
    
    // 要求されたGPU数が利用可能かチェック
    if (numGPUs > deviceCount) {
        printf("警告: 要求された%dGPUは利用可能数(%d)を超えています。利用可能なGPU数に調整します。\n", 
               numGPUs, deviceCount);
        numGPUs = deviceCount;
    }
    
    // 各GPUの情報を表示
    for (int i = 0; i < numGPUs; i++) {
        printGPUInfo(i);
    }
    
    // 定数設定
    const int ROWS = matrixSize;
    const int COLS = matrixSize;
    const float SCALAR = 2.0f;
    
    printf("\n===== パラメータ設定 =====\n");
    printf("行列サイズ: %d x %d\n", ROWS, COLS);
    printf("スカラー値: %.1f\n", SCALAR);
    printf("使用GPU数: %d\n\n", numGPUs);
    
    // メモリ確保
    float *originalMatrix = new float[ROWS * COLS];
    float *singleGPUResult = new float[ROWS * COLS];
    float *multiGPUResult = new float[ROWS * COLS];
    
    // 行列の初期化
    printf("行列の初期化中...\n");
    initializeMatrix(originalMatrix, ROWS, COLS);
    
    // 結果用配列にコピー
    memcpy(singleGPUResult, originalMatrix, ROWS * COLS * sizeof(float));
    memcpy(multiGPUResult, originalMatrix, ROWS * COLS * sizeof(float));
    
    // 処理前の行列サンプル表示
    printMatrixSample("元の行列", originalMatrix, ROWS, COLS);
    
    struct timespec start_time, end_time;
    double singleGPU_time, multiGPU_time;
    
    // ===== シングルGPU処理 =====
    printf("\n===== シングルGPU処理開始 =====\n");
    clock_gettime(CLOCK_REALTIME, &start_time);
    
    singleGPU_process(singleGPUResult, ROWS, COLS, SCALAR);
    
    clock_gettime(CLOCK_REALTIME, &end_time);
    singleGPU_time = calculateElapsedTime(start_time, end_time);
    printf("シングルGPU処理完了: %.6f 秒\n", singleGPU_time);
    
    // ===== マルチGPU処理 =====
    printf("\n===== マルチGPU処理開始 (%d GPUs) =====\n", numGPUs);
    clock_gettime(CLOCK_REALTIME, &start_time);
    
    // GPU間で行数を分割
    int rowsPerGPU = ROWS / numGPUs;
    std::thread *threads = new std::thread[numGPUs];
    
    // 各GPUでの処理を開始
    for (int i = 0; i < numGPUs; i++) {
        int startRow = i * rowsPerGPU;
        int numRows = (i == numGPUs - 1) ? (ROWS - startRow) : rowsPerGPU;
        threads[i] = std::thread(process_on_device, i, multiGPUResult, startRow, numRows, COLS, SCALAR);
    }
    
    // 全スレッドの終了を待機
    for (int i = 0; i < numGPUs; i++) {
        threads[i].join();
    }
    
    clock_gettime(CLOCK_REALTIME, &end_time);
    multiGPU_time = calculateElapsedTime(start_time, end_time);
    printf("マルチGPU処理完了: %.6f 秒\n", multiGPU_time);
    
    // ===== 結果の表示と検証 =====
    printf("\n===== 性能比較 =====\n");
    printf("シングルGPU実行時間: %.6f 秒\n", singleGPU_time);
    printf("マルチGPU実行時間(%d GPUs): %.6f 秒\n", numGPUs, multiGPU_time);
    double speedup = singleGPU_time / multiGPU_time;
    printf("高速化率: %.2f倍\n", speedup);
    printf("効率: %.2f%%\n", (speedup / numGPUs) * 100.0);
    
    // 結果の検証
    printf("\n===== 結果検証 =====\n");
    bool resultsMatch = verifyResults(singleGPUResult, multiGPUResult, ROWS, COLS, 1e-5);
    if (resultsMatch) {
        printf("シングルGPUとマルチGPUの結果は一致しています。\n");
    } else {
        printf("警告: シングルGPUとマルチGPUの結果に不一致があります！\n");
    }
    
    // 処理後の行列サンプル表示
    printMatrixSample("シングルGPUでの処理後行列", singleGPUResult, ROWS, COLS);
    printMatrixSample("マルチGPUでの処理後行列", multiGPUResult, ROWS, COLS);
    
    // メモリ解放
    delete[] originalMatrix;
    delete[] singleGPUResult;
    delete[] multiGPUResult;
    delete[] threads;
    
    return 0;
}