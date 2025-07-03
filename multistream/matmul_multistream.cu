#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>

// 計算集約的なカーネル（行列乗算）
__global__ void matmulKernel(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// より軽量な処理用カーネル
__global__ void vectorProcessKernel(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 複数の浮動小数点演算を含む処理
        float val = input[idx];
        for (int i = 0; i < 100; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        output[idx] = val;
    }
}

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
        return elapsed.count();
    }
};

class StreamBenchmark {
private:
    int N;  // 行列サイズ
    int numBatches;
    int numStreams;
    size_t matrixBytes;
    
    // ホストメモリ（Pinned）
    std::vector<float*> h_A, h_B, h_C;
    
    // デバイスメモリ
    std::vector<float*> d_A, d_B, d_C;
    
    // ストリーム
    std::vector<cudaStream_t> streams;
    
    // イベント（詳細な時間測定用）
    std::vector<cudaEvent_t> startEvents, stopEvents;
    std::vector<cudaEvent_t> h2dEvents, kernelEvents, d2hEvents;
    
public:
    StreamBenchmark(int matrixSize, int batches, int numStreams_) 
        : N(matrixSize), numBatches(batches), numStreams(numStreams_) {
        
        matrixBytes = N * N * sizeof(float);
        
        // ホストメモリ確保（Pinned Memory）
        h_A.resize(numBatches);
        h_B.resize(numBatches);
        h_C.resize(numBatches);
        
        for (int i = 0; i < numBatches; i++) {
            cudaMallocHost(&h_A[i], matrixBytes);
            cudaMallocHost(&h_B[i], matrixBytes);
            cudaMallocHost(&h_C[i], matrixBytes);
            
            // 初期化
            for (int j = 0; j < N * N; j++) {
                h_A[i][j] = static_cast<float>(rand()) / RAND_MAX;
                h_B[i][j] = static_cast<float>(rand()) / RAND_MAX;
                h_C[i][j] = 0.0f;
            }
        }
        
        // デバイスメモリ確保
        d_A.resize(numStreams);
        d_B.resize(numStreams);
        d_C.resize(numStreams);
        
        for (int i = 0; i < numStreams; i++) {
            cudaMalloc(&d_A[i], matrixBytes);
            cudaMalloc(&d_B[i], matrixBytes);
            cudaMalloc(&d_C[i], matrixBytes);
        }
        
        // ストリーム作成
        streams.resize(numStreams);
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // イベント作成
        startEvents.resize(numBatches);
        stopEvents.resize(numBatches);
        h2dEvents.resize(numBatches);
        kernelEvents.resize(numBatches);
        d2hEvents.resize(numBatches);
        
        for (int i = 0; i < numBatches; i++) {
            cudaEventCreate(&startEvents[i]);
            cudaEventCreate(&stopEvents[i]);
            cudaEventCreate(&h2dEvents[i]);
            cudaEventCreate(&kernelEvents[i]);
            cudaEventCreate(&d2hEvents[i]);
        }
    }
    
    ~StreamBenchmark() {
        // クリーンアップ
        for (int i = 0; i < numBatches; i++) {
            cudaFreeHost(h_A[i]);
            cudaFreeHost(h_B[i]);
            cudaFreeHost(h_C[i]);
        }
        
        for (int i = 0; i < numStreams; i++) {
            cudaFree(d_A[i]);
            cudaFree(d_B[i]);
            cudaFree(d_C[i]);
            cudaStreamDestroy(streams[i]);
        }
        
        for (int i = 0; i < numBatches; i++) {
            cudaEventDestroy(startEvents[i]);
            cudaEventDestroy(stopEvents[i]);
            cudaEventDestroy(h2dEvents[i]);
            cudaEventDestroy(kernelEvents[i]);
            cudaEventDestroy(d2hEvents[i]);
        }
    }
    
    // 単一ストリームでの逐次実行
    double benchmarkSequential() {
        Timer timer;
        timer.start();
        
        cudaStream_t stream = 0;  // デフォルトストリーム
        
        for (int batch = 0; batch < numBatches; batch++) {
            // H2D転送
            cudaMemcpy(d_A[0], h_A[batch], matrixBytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B[0], h_B[batch], matrixBytes, cudaMemcpyHostToDevice);
            
            // カーネル実行
            dim3 blockSize(16, 16);
            dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                         (N + blockSize.y - 1) / blockSize.y);
            matmulKernel<<<gridSize, blockSize, 0, stream>>>(d_A[0], d_B[0], d_C[0], N);
            
            // D2H転送
            cudaMemcpy(h_C[batch], d_C[0], matrixBytes, cudaMemcpyDeviceToHost);
        }
        
        cudaDeviceSynchronize();
        return timer.stop();
    }
    
    // 複数ストリームでの並列実行（パイプライン処理）
    double benchmarkPipelined() {
        Timer timer;
        timer.start();
        
        dim3 blockSize(16, 16);
        dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                     (N + blockSize.y - 1) / blockSize.y);
        
        // パイプライン処理
        for (int batch = 0; batch < numBatches; batch++) {
            int streamId = batch % numStreams;
            
            // 詳細な時間測定開始
            cudaEventRecord(startEvents[batch], streams[streamId]);
            
            // H2D転送
            cudaMemcpyAsync(d_A[streamId], h_A[batch], matrixBytes, 
                           cudaMemcpyHostToDevice, streams[streamId]);
            cudaMemcpyAsync(d_B[streamId], h_B[batch], matrixBytes, 
                           cudaMemcpyHostToDevice, streams[streamId]);
            cudaEventRecord(h2dEvents[batch], streams[streamId]);
            
            // カーネル実行
            matmulKernel<<<gridSize, blockSize, 0, streams[streamId]>>>(
                d_A[streamId], d_B[streamId], d_C[streamId], N);
            cudaEventRecord(kernelEvents[batch], streams[streamId]);
            
            // D2H転送
            cudaMemcpyAsync(h_C[batch], d_C[streamId], matrixBytes, 
                           cudaMemcpyDeviceToHost, streams[streamId]);
            cudaEventRecord(d2hEvents[batch], streams[streamId]);
            
            cudaEventRecord(stopEvents[batch], streams[streamId]);
        }
        
        cudaDeviceSynchronize();
        return timer.stop();
    }
    
    // 詳細な時間分析
    void analyzeDetailedTiming() {
        std::cout << "\n=== 詳細な時間分析 ===\n";
        std::cout << std::fixed << std::setprecision(3);
        
        float totalH2D = 0.0f, totalKernel = 0.0f, totalD2H = 0.0f;
        
        for (int batch = 0; batch < numBatches; batch++) {
            float h2dTime, kernelTime, d2hTime, totalTime;
            
            cudaEventElapsedTime(&h2dTime, startEvents[batch], h2dEvents[batch]);
            cudaEventElapsedTime(&kernelTime, h2dEvents[batch], kernelEvents[batch]);
            cudaEventElapsedTime(&d2hTime, kernelEvents[batch], d2hEvents[batch]);
            cudaEventElapsedTime(&totalTime, startEvents[batch], stopEvents[batch]);
            
            totalH2D += h2dTime;
            totalKernel += kernelTime;
            totalD2H += d2hTime;
            
            if (batch < 5) {  // 最初の5バッチのみ詳細表示
                std::cout << "Batch " << batch << " (Stream " << (batch % numStreams) << "): "
                          << "H2D=" << h2dTime << "ms, "
                          << "Kernel=" << kernelTime << "ms, "
                          << "D2H=" << d2hTime << "ms, "
                          << "Total=" << totalTime << "ms\n";
            }
        }
        
        std::cout << "\n平均時間 (全" << numBatches << "バッチ):\n";
        std::cout << "  H2D転送: " << totalH2D / numBatches << " ms\n";
        std::cout << "  カーネル実行: " << totalKernel / numBatches << " ms\n";
        std::cout << "  D2H転送: " << totalD2H / numBatches << " ms\n";
        std::cout << "  合計: " << (totalH2D + totalKernel + totalD2H) / numBatches << " ms\n";
    }
    
    // 並列度の測定
    void measureParallelism() {
        std::cout << "\n=== 並列度の分析 ===\n";
        
        // 各ストリームの実行時間の重複を分析
        std::vector<std::pair<float, float>> streamRanges(numBatches);
        
        for (int batch = 0; batch < numBatches; batch++) {
            float startTime, endTime;
            cudaEventElapsedTime(&startTime, startEvents[0], startEvents[batch]);
            cudaEventElapsedTime(&endTime, startEvents[0], stopEvents[batch]);
            streamRanges[batch] = {startTime, endTime};
        }
        
        // 重複度の計算
        float maxOverlap = 0.0f;
        for (int i = 0; i < numBatches - 1; i++) {
            for (int j = i + 1; j < numBatches; j++) {
                float overlap = std::min(streamRanges[i].second, streamRanges[j].second) - 
                               std::max(streamRanges[i].first, streamRanges[j].first);
                if (overlap > 0) {
                    maxOverlap = std::max(maxOverlap, overlap);
                }
            }
        }
        
        std::cout << "最大重複時間: " << maxOverlap << " ms\n";
        
        // 実際の並列度
        float totalSequentialTime = (streamRanges.back().second - streamRanges.front().first);
        float theoreticalSequentialTime = 0.0f;
        for (const auto& range : streamRanges) {
            theoreticalSequentialTime += (range.second - range.first);
        }
        
        float parallelEfficiency = theoreticalSequentialTime / totalSequentialTime;
        std::cout << "並列効率: " << parallelEfficiency << "x\n";
    }
    
    // タイムラインデータの出力
    void outputTimelineData() {
        std::cout << "\n=== TIMELINE_DATA_START ===\n";
        
        // 基準時刻を設定（最初のバッチの開始時刻）
        float baseTime = 0.0f;
        
        for (int batch = 0; batch < numBatches; batch++) {
            int streamId = batch % numStreams;
            
            float startTime, h2dTime, kernelTime, d2hTime, endTime;
            
            // 各イベントの時間を取得
            cudaEventElapsedTime(&startTime, startEvents[0], startEvents[batch]);
            cudaEventElapsedTime(&h2dTime, startEvents[0], h2dEvents[batch]);
            cudaEventElapsedTime(&kernelTime, startEvents[0], kernelEvents[batch]);
            cudaEventElapsedTime(&d2hTime, startEvents[0], d2hEvents[batch]);
            cudaEventElapsedTime(&endTime, startEvents[0], stopEvents[batch]);
            
            // CSV形式で出力
            std::cout << "BATCH," << batch << "," << streamId << ","
                      << startTime << "," << h2dTime << "," 
                      << kernelTime << "," << d2hTime << "," << endTime << "\n";
        }
        
        std::cout << "=== TIMELINE_DATA_END ===\n";
    }
};

int main(int argc, char* argv[]) {
    // パラメータ設定
    int matrixSize = (argc > 1) ? atoi(argv[1]) : 512;
    int numBatches = (argc > 2) ? atoi(argv[2]) : 8;
    int numStreams = (argc > 3) ? atoi(argv[3]) : 4;
    
    std::cout << "=== CUDA ストリーム並列化ベンチマーク ===\n";
    std::cout << "行列サイズ: " << matrixSize << "x" << matrixSize << "\n";
    std::cout << "バッチ数: " << numBatches << "\n";
    std::cout << "ストリーム数: " << numStreams << "\n";
    std::cout << "メモリ使用量: " << (matrixSize * matrixSize * sizeof(float) * 3 * numBatches) / (1024*1024) << " MB\n\n";
    
    StreamBenchmark benchmark(matrixSize, numBatches, numStreams);
    
    // ウォームアップ
    std::cout << "ウォームアップ中...\n";
    benchmark.benchmarkSequential();
    
    // ベンチマーク実行
    std::cout << "\n=== ベンチマーク実行 ===\n";
    
    double sequentialTime = benchmark.benchmarkSequential();
    std::cout << "単一ストリーム(逐次実行): " << sequentialTime << " ms\n";
    
    double pipelinedTime = benchmark.benchmarkPipelined();
    std::cout << "複数ストリーム(パイプライン): " << pipelinedTime << " ms\n";
    
    double speedup = sequentialTime / pipelinedTime;
    std::cout << "速度向上: " << speedup << "x\n";
    
    // 詳細分析
    benchmark.analyzeDetailedTiming();
    benchmark.measureParallelism();
    
    // タイムラインデータの出力を追加
    benchmark.outputTimelineData();
    
    return 0;
}