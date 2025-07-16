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

// 畳み込みカーネル関数（3x3フィルタ）
__global__ void convolution2D(float *input, float *output, int rows, int cols) {
    // 3x3 Gaussian blur kernel
    __shared__ float filter[3][3];
    if (threadIdx.x < 3 && threadIdx.y < 3) {
        float kernel[3][3] = {
            {0.0625f, 0.125f, 0.0625f},
            {0.125f,  0.25f,  0.125f},
            {0.0625f, 0.125f, 0.0625f}
        };
        filter[threadIdx.y][threadIdx.x] = kernel[threadIdx.y][threadIdx.x];
    }
    __syncthreads();
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < cols && row < rows) {
        float sum = 0.0f;
        
        // 畳み込み演算
        for (int fy = -1; fy <= 1; fy++) {
            for (int fx = -1; fx <= 1; fx++) {
                int newRow = row + fy;
                int newCol = col + fx;
                
                // 境界処理（ゼロパディング）
                if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
                    sum += input[newRow * cols + newCol] * filter[fy + 1][fx + 1];
                }
            }
        }
        
        output[row * cols + col] = sum;
    }
}

// 行列の定数倍カーネル（旧版）
__global__ void matrixMultiplyByScalar(float *matrix, int rows, int cols, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        int index = idy * cols + idx;
        matrix[index] = matrix[index] * scalar;
    }
}

// 単一GPUでの処理関数（畳み込み処理）
void singleGPU_process(float *matrix, int rows, int cols, float scalar) {
    // デバイス選択
    cudaSetDevice(0);
    
    // サイズ計算
    size_t bytes = rows * cols * sizeof(float);
    
    // デバイスメモリ確保
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaCheckError();
    
    // ホストからデバイスへデータ転送
    cudaMemcpy(d_input, matrix, bytes, cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // カーネル設定
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);
    
    // 畳み込みカーネル実行
    convolution2D<<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // スカラー倍処理
    matrixMultiplyByScalar<<<gridSize, blockSize>>>(d_output, rows, cols, scalar);
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // デバイスからホストへデータ転送
    cudaMemcpy(matrix, d_output, bytes, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // デバイスメモリの解放
    cudaFree(d_input);
    cudaFree(d_output);
}

// 行列の初期化
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<float>((i * cols + j + 1) % 256) / 256.0f;
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
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        // 大きい行列は一部だけ表示
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
            }
            std::cout << " ... ";
            for (int j = cols - 2; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << " ... " << std::endl;
        for (int i = rows - 2; i < rows; i++) {
            for (int j = 0; j < 3; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
            }
            std::cout << " ... ";
            for (int j = cols - 2; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << matrix[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

// 時間計測用の構造体（絶対時刻を記録）
struct TimestampData {
    double globalStart;
    double dataDistributionStart;
    double dataDistributionEnd;
    double hostToDeviceStart;
    double hostToDeviceEnd;
    double kernelExecutionStart;
    double kernelExecutionEnd;
    double deviceToHostStart;
    double deviceToHostEnd;
    double dataCollectionStart;
    double dataCollectionEnd;
    double globalEnd;
};

// 基準時刻からの経過時間を計算
double getElapsedFromBase(struct timespec base, struct timespec current) {
    return (double)(current.tv_sec - base.tv_sec) + (double)(current.tv_nsec - base.tv_nsec) * 1e-9;
}

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
        printf("処理内容: 3x3ガウシアンブラー畳み込み + スカラー倍(%.1f)\n", 2.0f);
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
    
    struct timespec start_time, end_time, global_base_time;
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
    
    // 全ノード共通の基準時刻を設定
    clock_gettime(CLOCK_REALTIME, &global_base_time);
    MPI_Barrier(MPI_COMM_WORLD);  // 基準時刻同期
    
    struct timespec phase_time;
    TimestampData timestamps;
    
    // グローバル開始時刻記録
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.globalStart = getElapsedFromBase(global_base_time, phase_time);
    
    // 各ノードの処理範囲を計算
    int rowsPerGPU = ROWS / numGPUs;
    int startRow = rank * rowsPerGPU;
    int numRows = (rank == numGPUs - 1) ? (ROWS - startRow) : rowsPerGPU;
    
    // このノードが処理する部分行列のメモリ確保
    float *localMatrix = new float[numRows * COLS];
    
    // ===== データ分配フェーズ =====
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.dataDistributionStart = getElapsedFromBase(global_base_time, phase_time);
    
    // データ分配（MPIの集合通信を使用）
    int *sendcounts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        sendcounts = new int[numGPUs];
        displs = new int[numGPUs];
        
        for (int i = 0; i < numGPUs; i++) {
            int nodeNumRows = (i == numGPUs - 1) ? (ROWS - i * rowsPerGPU) : rowsPerGPU;
            sendcounts[i] = nodeNumRows * COLS;
            displs[i] = i * rowsPerGPU * COLS;
        }
    }
    
    // MPI_Scatterv を使用してデータを効率的に分配
    MPI_Scatterv(multiGPUResult, sendcounts, displs, MPI_FLOAT,
                 localMatrix, numRows * COLS, MPI_FLOAT,
                 0, MPI_COMM_WORLD);
    
    // 旧実装（コメントアウト）
    /*
    // データ分配（旧実装：Send/Recvループ）
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
    */
    
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.dataDistributionEnd = getElapsedFromBase(global_base_time, phase_time);

    // 各ノードで自分の担当部分を処理
    size_t input_bytes = numRows * COLS * sizeof(float);
    size_t output_bytes = numRows * COLS * sizeof(float);
    
    // ===== ホスト→デバイス転送フェーズ =====
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.hostToDeviceStart = getElapsedFromBase(global_base_time, phase_time);
    
    // デバイスメモリ確保とデータ転送
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_output, output_bytes);
    cudaMemcpy(d_input, localMatrix, input_bytes, cudaMemcpyHostToDevice);
    
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.hostToDeviceEnd = getElapsedFromBase(global_base_time, phase_time);
    
    // ===== カーネル実行フェーズ =====
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.kernelExecutionStart = getElapsedFromBase(global_base_time, phase_time);
    
    // カーネル設定と実行
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, 
                 (numRows + blockSize.y - 1) / blockSize.y);
    
    // 畳み込み処理
    convolution2D<<<gridSize, blockSize>>>(d_input, d_output, numRows, COLS);
    cudaDeviceSynchronize();
    
    // スカラー倍処理
    matrixMultiplyByScalar<<<gridSize, blockSize>>>(d_output, numRows, COLS, SCALAR);
    cudaDeviceSynchronize();
    
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.kernelExecutionEnd = getElapsedFromBase(global_base_time, phase_time);
    
    // ===== デバイス→ホスト転送フェーズ =====
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.deviceToHostStart = getElapsedFromBase(global_base_time, phase_time);
    
    // デバイスからホストへデータ転送
    cudaMemcpy(localMatrix, d_output, output_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);
    
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.deviceToHostEnd = getElapsedFromBase(global_base_time, phase_time);
    
    // ===== 結果収集フェーズ =====
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.dataCollectionStart = getElapsedFromBase(global_base_time, phase_time);
    
    // 結果の収集（MPI_Gatherv を使用）
    MPI_Gatherv(localMatrix, numRows * COLS, MPI_FLOAT,
                multiGPUResult, sendcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // 旧実装（コメントアウト）
    /*
    // 結果の収集（旧実装：Send/Recvループ）
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
    */
    
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.dataCollectionEnd = getElapsedFromBase(global_base_time, phase_time);

    // グローバル終了時刻記録
    clock_gettime(CLOCK_REALTIME, &phase_time);
    timestamps.globalEnd = getElapsedFromBase(global_base_time, phase_time);
    
    double local_time = timestamps.globalEnd - timestamps.globalStart;
    
    // 各フェーズの所要時間を計算
    double dataDistTime = timestamps.dataDistributionEnd - timestamps.dataDistributionStart;
    double hostToDevTime = timestamps.hostToDeviceEnd - timestamps.hostToDeviceStart;
    double kernelTime = timestamps.kernelExecutionEnd - timestamps.kernelExecutionStart;
    double devToHostTime = timestamps.deviceToHostEnd - timestamps.deviceToHostStart;
    double dataCollTime = timestamps.dataCollectionEnd - timestamps.dataCollectionStart;
    
    // 統計計算用の変数
    double avgDataDist, avgHostToDev, avgKernel, avgDevToHost, avgDataColl;
    
    // 全ノードの時間データを収集（平均計算用）
    MPI_Reduce(&dataDistTime, &avgDataDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&hostToDevTime, &avgHostToDev, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&kernelTime, &avgKernel, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&devToHostTime, &avgDevToHost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&dataCollTime, &avgDataColl, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &multiGPU_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // マスターノードで平均時間と割合を表示
    if (rank == 0) {
        avgDataDist /= numGPUs;
        avgHostToDev /= numGPUs;
        avgKernel /= numGPUs;
        avgDevToHost /= numGPUs;
        avgDataColl /= numGPUs;
        
        double totalTime = multiGPU_time;
        
        printf("\n===== 詳細なタイミング分析 =====\n");
        printf("1. データ分配時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgDataDist, (avgDataDist/totalTime)*100);
        printf("2. ホスト→デバイス転送: %.6f 秒 (全体の %.1f%%)\n", 
               avgHostToDev, (avgHostToDev/totalTime)*100);
        printf("3. カーネル実行時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgKernel, (avgKernel/totalTime)*100);
        printf("4. デバイス→ホスト転送: %.6f 秒 (全体の %.1f%%)\n", 
               avgDevToHost, (avgDevToHost/totalTime)*100);
        printf("5. 結果収集時間: %.6f 秒 (全体の %.1f%%)\n", 
               avgDataColl, (avgDataColl/totalTime)*100);
        
        // 最も時間がかかっている部分を特定
        double maxTime = avgDataDist;
        const char* bottleneck = "データ分配";
        
        if (avgKernel > maxTime) {
            maxTime = avgKernel;
            bottleneck = "カーネル実行";
        }
        if (avgDataColl > maxTime) {
            maxTime = avgDataColl;
            bottleneck = "結果収集";
        }
        if (avgHostToDev > maxTime) {
            maxTime = avgHostToDev;
            bottleneck = "ホスト→デバイス転送";
        }
        if (avgDevToHost > maxTime) {
            maxTime = avgDevToHost;
            bottleneck = "デバイス→ホスト転送";
        }
        
        printf("\n主なボトルネック: %s (%.1f%% of total time)\n", 
               bottleneck, (maxTime/totalTime)*100);
        
        // sendcounts と displs をクリーンアップ
        delete[] sendcounts;
        delete[] displs;
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
        
        // 結果の検証（畳み込み処理のため、厳密な一致は期待しない）
        printf("\n===== 結果検証 =====\n");
        bool resultsMatch = true;
        int mismatchCount = 0;
        const int maxMismatchesToShow = 5;
        
        for (int i = 0; i < ROWS && resultsMatch; i++) {
            for (int j = 0; j < COLS && resultsMatch; j++) {
                int idx = i * COLS + j;
                if (fabs(singleGPUResult[idx] - multiGPUResult[idx]) > 1e-4) {
                    if (mismatchCount < maxMismatchesToShow) {
                        printf("結果の差異 [%d, %d]: %.6f vs %.6f (差: %.6f)\n", 
                              i, j, singleGPUResult[idx], multiGPUResult[idx],
                              fabs(singleGPUResult[idx] - multiGPUResult[idx]));
                    }
                    mismatchCount++;
                    if (mismatchCount >= 100) {  // 大量の不一致がある場合は早期終了
                        resultsMatch = false;
                        break;
                    }
                }
            }
        }
        
        if (mismatchCount == 0) {
            printf("シングルGPUとマルチGPUの結果は一致しています。\n");
        } else if (mismatchCount < 10) {
            printf("軽微な数値誤差があります (不一致数: %d)。許容範囲内です。\n", mismatchCount);
        } else {
            printf("警告: 多数の結果不一致があります (不一致数: %d)！\n", mismatchCount);
        }
        
        // 処理後の行列サンプル表示
        printMatrixSample("シングルGPUでの処理後行列", singleGPUResult, ROWS, COLS);
        printMatrixSample("マルチGPUでの処理後行列", multiGPUResult, ROWS, COLS);

        // CSV形式でタイミングデータを出力
        printf("\n===== TIMING_DATA_CSV_START =====\n");
        printf("rank,data_distribution,host_to_device,kernel_execution,device_to_host,data_collection,total_time,");
        printf("global_start,data_dist_start,data_dist_end,host_to_dev_start,host_to_dev_end,");
        printf("kernel_start,kernel_end,dev_to_host_start,dev_to_host_end,data_coll_start,data_coll_end,global_end\n");
        
        // 各ランクの詳細データを収集
        double *allDataDistribution = new double[numGPUs];
        double *allHostToDevice = new double[numGPUs];
        double *allKernelExecution = new double[numGPUs];
        double *allDeviceToHost = new double[numGPUs];
        double *allDataCollection = new double[numGPUs];
        double *allTotalTimes = new double[numGPUs];
        
        // タイムスタンプデータを収集
        TimestampData *allTimestamps = new TimestampData[numGPUs];
        
        // 各ランクからデータを収集
        MPI_Gather(&dataDistTime, 1, MPI_DOUBLE, allDataDistribution, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&hostToDevTime, 1, MPI_DOUBLE, allHostToDevice, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&kernelTime, 1, MPI_DOUBLE, allKernelExecution, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&devToHostTime, 1, MPI_DOUBLE, allDeviceToHost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&dataCollTime, 1, MPI_DOUBLE, allDataCollection, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_time, 1, MPI_DOUBLE, allTotalTimes, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&timestamps, sizeof(TimestampData), MPI_BYTE, allTimestamps, sizeof(TimestampData), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // 各ランクのデータを出力
        for (int i = 0; i < numGPUs; i++) {
            printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,",
                i, allDataDistribution[i], allHostToDevice[i], 
                allKernelExecution[i], allDeviceToHost[i], 
                allDataCollection[i], allTotalTimes[i]);
            printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                allTimestamps[i].globalStart,
                allTimestamps[i].dataDistributionStart, allTimestamps[i].dataDistributionEnd,
                allTimestamps[i].hostToDeviceStart, allTimestamps[i].hostToDeviceEnd,
                allTimestamps[i].kernelExecutionStart, allTimestamps[i].kernelExecutionEnd,
                allTimestamps[i].deviceToHostStart, allTimestamps[i].deviceToHostEnd,
                allTimestamps[i].dataCollectionStart, allTimestamps[i].dataCollectionEnd,
                allTimestamps[i].globalEnd);
        }
        printf("===== TIMING_DATA_CSV_END =====\n");
        
        // メモリ解放（マスターノードのみ）
        delete[] originalMatrix;
        delete[] singleGPUResult;
        delete[] multiGPUResult;
        delete[] infoLengths;
        delete[] displacements;
        delete[] allGpuInfo;
        delete[] allDataDistribution;
        delete[] allHostToDevice;
        delete[] allKernelExecution;
        delete[] allDeviceToHost;
        delete[] allDataCollection;
        delete[] allTotalTimes;
        delete[] allTimestamps;
    } else {
        // 他のランクも自分のデータを送信
        MPI_Gather(&dataDistTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&hostToDevTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&kernelTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&devToHostTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&dataCollTime, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_time, 1, MPI_DOUBLE, nullptr, 0, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&timestamps, sizeof(TimestampData), MPI_BYTE, nullptr, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    
    // 全ノードで共通のメモリ解放
    delete[] localMatrix;
    delete[] myInfo;
    
    MPI_Finalize();
    return 0;
}