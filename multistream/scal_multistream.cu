#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <cstdlib>

// 行列要素を2倍にするカーネル
__global__ void scaleKernel(float *d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] = d[i] * 2.0f;
}

// 計測用関数
double measureTime(std::function<void()> func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

void printUsage(const char* programName) {
    printf("使用方法: %s [行列サイズ(2のべき乗)] [ストリーム数]\n", programName);
    printf("例: %s 24 4   # 2^24要素の行列を4つのストリームで処理\n", programName);
}

int main(int argc, char* argv[]) {
    // デフォルト値
    int powerOfTwo = 24;  // デフォルトは2^24要素
    int numStreams = 4;   // デフォルトは4ストリーム
    
    // コマンドライン引数の解析
    if (argc > 1) {
        powerOfTwo = atoi(argv[1]);
        if (powerOfTwo <= 0) {
            printf("エラー: 行列サイズは正の数を指定してください\n");
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (argc > 2) {
        numStreams = atoi(argv[2]);
        if (numStreams <= 0) {
            printf("エラー: ストリーム数は正の数を指定してください\n");
            printUsage(argv[0]);
            return 1;
        }
    }
    
    const int N = 1 << powerOfTwo; // 2のべき乗サイズ
    size_t bytes = N * sizeof(float);
    
    // ホストメモリ確保と初期化
    float *h_data = new float[N];
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    // デバイスメモリ確保
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // 基本パラメータ
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    
    printf("行列サイズ: 2^%d = %d 要素（%.2f MB）\n", powerOfTwo, N, bytes / (1024.0 * 1024.0));
    printf("グリッドサイズ: %d ブロック、ブロックサイズ: %d スレッド\n", gridSize.x, blockSize.x);
    printf("使用するストリーム数: %d\n", numStreams);
    
    // 方法1: 複数のstreamでGridを分散させて計算 (順次実行されるので意味ない)
    double time1 = measureTime([&]() {
        // ストリーム作成
        cudaStream_t* streams = new cudaStream_t[numStreams];
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // 各ストリームにデータをコピーして計算
        int streamSize = N / numStreams;
        for (int i = 0; i < numStreams; i++) {
            int offset = i * streamSize;
            int streamBytes = (i == numStreams - 1) ? (N - offset) * sizeof(float) : streamSize * sizeof(float);
            int streamElements = streamBytes / sizeof(float);
            
            cudaMemcpyAsync(&d_data[offset], &h_data[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
            
            dim3 streamGrid((streamElements + blockSize.x - 1) / blockSize.x);
            scaleKernel<<<streamGrid, blockSize, 0, streams[i]>>>(&d_data[offset], streamElements);
            
            cudaMemcpyAsync(&h_data[offset], &d_data[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        }
        
        // ストリームの破棄
        for (int i = 0; i < numStreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
        delete[] streams;
    });


    // 方法2: 複数のstreamでGridを分散させて計算（）
    double time2 = measureTime([&]() {
        // ストリーム作成
        cudaStream_t* streams = new cudaStream_t[numStreams];
        for (int i = 0; i < numStreams; i++) {
            cudaStreamCreate(&streams[i]);
        }
        
        // 各ストリームにデータをコピーして計算（並列実行）
        int streamSize = N / numStreams;
        
        // 全てのH2Dコピーを開始
        for (int i = 0; i < numStreams; i++) {
            int offset = i * streamSize;
            int streamBytes = (i == numStreams - 1) ? (N - offset) * sizeof(float) : streamSize * sizeof(float);
            cudaMemcpyAsync(&d_data[offset], &h_data[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
        }
        
        // 全てのカーネルを起動
        for (int i = 0; i < numStreams; i++) {
            int offset = i * streamSize;
            int streamElements = (i == numStreams - 1) ? (N - offset) : streamSize;
            dim3 streamGrid((streamElements + blockSize.x - 1) / blockSize.x);
            scaleKernel<<<streamGrid, blockSize, 0, streams[i]>>>(&d_data[offset], streamElements);
        }
        
        // 全てのD2Hコピーを開始
        for (int i = 0; i < numStreams; i++) {
            int offset = i * streamSize;
            int streamBytes = (i == numStreams - 1) ? (N - offset) * sizeof(float) : streamSize * sizeof(float);
            cudaMemcpyAsync(&h_data[offset], &d_data[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
        }
        
        // 全てのストリームの完了を待機
        for (int i = 0; i < numStreams; i++) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // ストリームの破棄
        for (int i = 0; i < numStreams; i++) {
            cudaStreamDestroy(streams[i]);
        }
        
        delete[] streams;
    });
    
    // 方法3: 単一のstreamにGridを載せて計算
    double time3 = measureTime([&]() {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice, stream);
        scaleKernel<<<gridSize, blockSize, 0, stream>>>(d_data, N);
        cudaMemcpyAsync(h_data, d_data, bytes, cudaMemcpyDeviceToHost, stream);
        
        cudaStreamDestroy(stream);
    });
    
    // 方法4: 一つのGridにまとめて計算（デフォルトストリーム）
    double time4 = measureTime([&]() {
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        scaleKernel<<<gridSize, blockSize>>>(d_data, N);
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    });
    
    // 結果出力
    printf("\n実行時間比較:\n");
    printf("1. 複数ストリームでGridを分散[順次実行] (%d ストリーム): %.3f ms\n", numStreams, time1);
    printf("2. 複数ストリームでGridを分散[並列実行] (%d ストリーム): %.3f ms\n", numStreams, time2);
    printf("3. 単一ストリームにGridを載せ: %.3f ms\n", time3);
    printf("4. 一つのGridにまとめて計算 : %.3f ms\n", time4);
    
    // メモリ解放
    delete[] h_data;
    cudaFree(d_data);
    
    return 0;
}