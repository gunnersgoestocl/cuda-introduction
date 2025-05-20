#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <mpi.h>
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
double calculateElapsedTime(struct timespec start_time, struct timespec end_time) {
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

// 行列の初期化
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<float>(i * cols + j + 1);
        }
    }
}

// 指定GPUの情報を文字列に格納する関数
std::string getGPUInfoString(int deviceId) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
             "GPU %d: %s\n"
             "  Compute capability: %d.%d\n"
             "  Total global memory: %.2f GB\n"
             "  Multiprocessors: %d\n"
             "  Max threads per block: %d",
             deviceId, prop.name,
             prop.major, prop.minor,
             (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f),
             prop.multiProcessorCount,
             prop.maxThreadsPerBlock);
    
    return std::string(buffer);
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

// 時間計測用の構造体
struct Timings {
    double dataDistribution;
    double kernelExecution;
    double dataCollection;
    double deviceToHost;
    double hostToDevice;
};

// メイン関数
int main(int argc, char **argv) {
    int rank, size;
    
    // MPI初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // コマンドライン引数の確認
    if (argc < 2 && rank == 0) {
        fprintf(stderr, "使用方法: %s <行列サイズ(2のべき乗)>\n", argv[0]);
        MPI_Finalize();
        return -1;
    }
    
    // コマンドラインから引数を取得
    int matrixSize = 1 << atoi(argv[1]);   // 行列サイズ
    int numGPUs = size;  // MPIプロセス数 = GPU数
    
    // 各ノードで1つのGPUを使用
    cudaSetDevice(0);
    
    // 全ノードのGPU情報を収集
    std::string gpuInfo = getGPUInfoString(0);
    int infoLength = gpuInfo.length() + 1;
    
    // GPUの情報の長さを集める（可変長データの収集のため）
    int* infoLengths = NULL;
    if (rank == 0) {
        infoLengths = new int[size];
    }
    
    MPI_Gather(&infoLength, 1, MPI_INT, infoLengths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // マスターノードでGPU情報を集める
    char* allGpuInfo = NULL;
    int* displacements = NULL;
    
    if (rank == 0) {
        // 各プロセスからの情報を格納する配列を確保
        displacements = new int[size];
        int totalLength = 0;
        for (int i = 0; i < size; i++) {
            displacements[i] = totalLength;
            totalLength += infoLengths[i];
        }
        allGpuInfo = new char[totalLength];
    }
    
    // 全GPUの情報を集める
    char* myInfo = new char[infoLength];
    strcpy(myInfo, gpuInfo.c_str());
    
    MPI_Gatherv(myInfo, infoLength, MPI_CHAR, 
               allGpuInfo, infoLengths, displacements, MPI_CHAR,
               0, MPI_COMM_WORLD);
    
    // マスターノードでの情報表示
    if (rank == 0) {
        printf("利用可能なMPIプロセス数（GPU数）: %d\n", size);
        printf("\n===== 各ノードのGPU情報 =====\n");
        
        for (int i = 0; i < size; i++) {
            printf("ランク %d の GPU情報:\n%s\n\n", i, &allGpuInfo[displacements[i]]);
        }
        
        printf("\n===== パラメータ設定 =====\n");
        printf("行列サイズ: %d x %d\n", matrixSize, matrixSize);
        printf("スカラー値: %.1f\n", 2.0f);
        printf("使用GPU数: %d\n\n", numGPUs);
    }
    
    // 情報表示が終わるまで同期
    MPI_Barrier(MPI_COMM_WORLD);
    
    const int ROWS = matrixSize;
    const int COLS = matrixSize;
    const float SCALAR = 2.0f;
    
    float *originalMatrix = NULL;
    float *singleGPUResult = NULL;
    float *multiGPUResult = NULL;
    
    // マスターノードのみ完全な行列を確保
    if (rank == 0) {
        // メモリ確保
        originalMatrix = new float[ROWS * COLS];
        singleGPUResult = new float[ROWS * COLS];
        multiGPUResult = new float[ROWS * COLS];
        
        // 行列の初期化
        printf("行列の初期化中...\n");
        initializeMatrix(originalMatrix, ROWS, COLS);
        
        // 結果用配列にコピー
        memcpy(singleGPUResult, originalMatrix, ROWS * COLS * sizeof(float));
        memcpy(multiGPUResult, originalMatrix, ROWS * COLS * sizeof(float));
        
        // 処理前の行列サンプル表示
        printMatrixSample("元の行列", originalMatrix, ROWS, COLS);
    }
    
    struct timespec start_time, end_time;
    double singleGPU_time = 0.0, multiGPU_time = 0.0;
    
    // ===== シングルGPU処理（マスターノードのみ） =====
    if (rank == 0) {
        printf("\n===== シングルGPU処理開始 =====\n");
        clock_gettime(CLOCK_REALTIME, &start_time);
        
        singleGPU_process(singleGPUResult, ROWS, COLS, SCALAR);
        
        clock_gettime(CLOCK_REALTIME, &end_time);
        singleGPU_time = calculateElapsedTime(start_time, end_time);
        printf("シングルGPU処理完了: %.6f 秒\n", singleGPU_time);
    }
    
    // 全ノードでの同期
    MPI_Barrier(MPI_COMM_WORLD);
    
    // ===== マルチGPU処理（全ノード） =====
    if (rank == 0) {
        printf("\n===== マルチGPU処理開始 (%d GPUs) =====\n", numGPUs);
    }
    
    // 全ノードで時間計測開始
    clock_gettime(CLOCK_REALTIME, &start_time);
    
    // 各ノードの処理範囲を計算
    int rowsPerGPU = ROWS / numGPUs;
    int startRow = rank * rowsPerGPU;
    int numRows = (rank == numGPUs - 1) ? (ROWS - startRow) : rowsPerGPU;
    
    // このノードが処理する部分行列のメモリ確保
    float *localMatrix = new float[numRows * COLS];
    
    struct timespec phase_start, phase_end;
    Timings timings = {0.0, 0.0, 0.0, 0.0, 0.0};
    Timings avgTimings = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    // データ分配フェーズの時間計測
    clock_gettime(CLOCK_REALTIME, &phase_start);
    
    // データ分配
    if (rank == 0) {
        // マスターノードは自分の部分を保持
        memcpy(localMatrix, multiGPUResult, numRows * COLS * sizeof(float));
        
        // 他のノードにそれぞれの部分を送信
        for (int i = 1; i < numGPUs; i++) {
            int nodeStartRow = i * rowsPerGPU;
            int nodeNumRows = (i == numGPUs - 1) ? (ROWS - nodeStartRow) : rowsPerGPU;
            MPI_Send(&multiGPUResult[nodeStartRow * COLS], nodeNumRows * COLS, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // 他のノードはマスターから自分の部分を受信
        MPI_Recv(localMatrix, numRows * COLS, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    clock_gettime(CLOCK_REALTIME, &phase_end);
    timings.dataDistribution = calculateElapsedTime(phase_start, phase_end);

    // 各ノードで自分の担当部分を処理
    size_t bytes = numRows * COLS * sizeof(float);
    
    // ホスト→デバイス転送時間計測
    clock_gettime(CLOCK_REALTIME, &phase_start);
    
    // デバイスメモリ確保とデータ転送
    float *d_matrix;
    cudaMalloc(&d_matrix, bytes);
    cudaMemcpy(d_matrix, localMatrix, bytes, cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_REALTIME, &phase_end);
    timings.hostToDevice = calculateElapsedTime(phase_start, phase_end);
    
    // カーネル実行時間計測
    clock_gettime(CLOCK_REALTIME, &phase_start);
    
    // カーネル設定と実行
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, 
                 (numRows + blockSize.y - 1) / blockSize.y);
    matrixMultiplyByScalar<<<gridSize, blockSize>>>(d_matrix, numRows, COLS, SCALAR);
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_REALTIME, &phase_end);
    timings.kernelExecution = calculateElapsedTime(phase_start, phase_end);
    
    // デバイス→ホスト転送時間計測
    clock_gettime(CLOCK_REALTIME, &phase_start);
    
    // デバイスからホストへデータ転送
    cudaMemcpy(localMatrix, d_matrix, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    
    clock_gettime(CLOCK_REALTIME, &phase_end);
    timings.deviceToHost = calculateElapsedTime(phase_start, phase_end);
    
    // 結果収集の時間計測
    clock_gettime(CLOCK_REALTIME, &phase_start);
    
    // 結果の収集
    if (rank == 0) {
        // マスターノードは自分の部分をmultiGPUResultに戻す
        memcpy(multiGPUResult, localMatrix, numRows * COLS * sizeof(float));
        
        // 他のノードからそれぞれの部分を受信
        for (int i = 1; i < numGPUs; i++) {
            int nodeStartRow = i * rowsPerGPU;
            int nodeNumRows = (i == numGPUs - 1) ? (ROWS - nodeStartRow) : rowsPerGPU;
            MPI_Recv(&multiGPUResult[nodeStartRow * COLS], nodeNumRows * COLS, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        // 他のノードはマスターに結果を送信
        MPI_Send(localMatrix, numRows * COLS, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    
    clock_gettime(CLOCK_REALTIME, &phase_end);
    timings.dataCollection = calculateElapsedTime(phase_start, phase_end);

    // 全体の処理時間終了計測
    clock_gettime(CLOCK_REALTIME, &end_time);
    double local_time = calculateElapsedTime(start_time, end_time);
    
    // 全ノードの時間データを収集
    MPI_Reduce(&timings.dataDistribution, &avgTimings.dataDistribution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.kernelExecution, &avgTimings.kernelExecution, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.dataCollection, &avgTimings.dataCollection, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.hostToDevice, &avgTimings.hostToDevice, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&timings.deviceToHost, &avgTimings.deviceToHost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &multiGPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // マスターノードで平均時間と割合を表示
    if (rank == 0) {
        avgTimings.dataDistribution /= numGPUs;
        avgTimings.kernelExecution /= numGPUs;
        avgTimings.dataCollection /= numGPUs;
        avgTimings.hostToDevice /= numGPUs;
        avgTimings.deviceToHost /= numGPUs;
        
        double totalTime = multiGPU_time;
        
        printf("\n===== 詳細なタイミング分析 =====\n");
        printf("1. データ分配時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgTimings.dataDistribution, (avgTimings.dataDistribution/totalTime)*100);
        printf("2. ホスト→デバイス転送: %.6f 秒 (全体の %.1f%%)\n", 
               avgTimings.hostToDevice, (avgTimings.hostToDevice/totalTime)*100);
        printf("3. カーネル実行時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgTimings.kernelExecution, (avgTimings.kernelExecution/totalTime)*100);
        printf("4. デバイス→ホスト転送: %.6f 秒 (全体の %.1f%%)\n", 
               avgTimings.deviceToHost, (avgTimings.deviceToHost/totalTime)*100);
        printf("5. 結果収集時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgTimings.dataCollection, (avgTimings.dataCollection/totalTime)*100);
        
        // 最も時間がかかっている部分を特定
        double maxTime = avgTimings.dataDistribution;
        const char* bottleneck = "データ分配";
        
        if (avgTimings.kernelExecution > maxTime) {
            maxTime = avgTimings.kernelExecution;
            bottleneck = "カーネル実行";
        }
        if (avgTimings.dataCollection > maxTime) {
            maxTime = avgTimings.dataCollection;
            bottleneck = "結果収集";
        }
        if (avgTimings.hostToDevice > maxTime) {
            maxTime = avgTimings.hostToDevice;
            bottleneck = "ホスト→デバイス転送";
        }
        if (avgTimings.deviceToHost > maxTime) {
            maxTime = avgTimings.deviceToHost;
            bottleneck = "デバイス→ホスト転送";
        }
        
        printf("\n主なボトルネック: %s (%.1f%% of total time)\n", 
               bottleneck, (maxTime/totalTime)*100);
    }
    
    // ===== 結果の表示と検証（マスターノードのみ） =====
    if (rank == 0) {
        printf("マルチGPU処理完了: %.6f 秒\n", multiGPU_time);
        
        printf("\n===== 性能比較 =====\n");
        printf("シングルGPU実行時間: %.6f 秒\n", singleGPU_time);
        printf("マルチGPU実行時間(%d GPUs): %.6f 秒\n", numGPUs, multiGPU_time);
        double speedup = singleGPU_time / multiGPU_time;
        printf("高速化率: %.2f倍\n", speedup);
        printf("効率: %.2f%%\n", (speedup / numGPUs) * 100.0);
        
        // 結果の検証
        printf("\n===== 結果検証 =====\n");
        bool resultsMatch = true;
        for (int i = 0; i < ROWS && resultsMatch; i++) {
            for (int j = 0; j < COLS && resultsMatch; j++) {
                int idx = i * COLS + j;
                if (fabs(singleGPUResult[idx] - multiGPUResult[idx]) > 1e-5) {
                    printf("結果が一致しません！位置 [%d, %d]: %.4f != %.4f\n", 
                          i, j, singleGPUResult[idx], multiGPUResult[idx]);
                    resultsMatch = false;
                    break;
                }
            }
        }
        
        if (resultsMatch) {
            printf("シングルGPUとマルチGPUの結果は一致しています。\n");
        } else {
            printf("警告: シングルGPUとマルチGPUの結果に不一致があります！\n");
        }
        
        // 処理後の行列サンプル表示
        printMatrixSample("シングルGPUでの処理後行列", singleGPUResult, ROWS, COLS);
        printMatrixSample("マルチGPUでの処理後行列", multiGPUResult, ROWS, COLS);
        
        // メモリ解放（マスターノードのみ）
        delete[] originalMatrix;
        delete[] singleGPUResult;
        delete[] multiGPUResult;
        delete[] infoLengths;
        delete[] displacements;
        delete[] allGpuInfo;
    }
    
    // 全ノードで共通のメモリ解放
    delete[] localMatrix;
    delete[] myInfo;
    
    MPI_Finalize();
    return 0;
}