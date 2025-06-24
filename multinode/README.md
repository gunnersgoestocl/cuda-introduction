# Multi-Stream, Multi-Device, and Multi-Node GPU use

## Multi-Stream use template



## Multi-Device use template



## Multi-Node code (MPI-only) elements

複数のGPUノードを使用して計算するには、MPIを使用する必要があります。MPIは、Message Passing Interfaceの略で、分散メモリ環境でのプロセス間通信を行うための標準的なAPIです。以下にテンプレートを示す。

### ライブラリ
```c
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mpi.h>   // MPIヘッダー
#include <iomanip>  // 時間計測用, get_time
#include <time.h>
```

### データ処理カーネル
```c
/* 
 * 入力(1次元配列)のノード担当領域の先頭ポインタ
 * 出力(1次元配列)のノード担当領域の先頭ポインタ
 * 行数, 列数
 * それ以外の引数は、必要に応じて追加
 */
__global__ void kernel(float *input_head, float *output_head, int rows, int cols, ...) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < rows && idx < cols) {
        int index = idy * cols + idx;
        // データ処理
    }
}
```


### MPI初期化

MPIシステムを初期化し、プロセス間通信を可能にする
```c
int rank, size;
MPI_Init(&argc, &argv); // 引数: &argc, &argv - コマンドライン引数
```
なお、各プロセス自体は、`mpirun`コマンドで起動されており、起動時に各プロセスはノード情報を割り当てられている。

### プロセスのランク(ID)とサイズ(ノード数)を取得
```c
引数: &rank - ランクを格納する変数へのポインタ
/* 引数
 * - コミュニケータ 
 * - ランクを格納する変数へのポインタ
 */
MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // 引数: Mpi_comm, &rank
```
`MPI_COMM_WORLD`は、 **コミュニケータ** と呼ばれる、MPIのプロセスグループを表すオブジェクトの一種である。
各プロセスのIDは、コミュニケータ内で排他的に割り当てられ、0から始まる整数値で表される。
すなわち、異なるコミュニケータに属するプロセスは、同じIDを持つことがある。

```c
/* 引数
 * - コミュニケータ
 * - プロセス数を格納する変数へのポインタ
 */
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

コミュニケータは、ユーザが定義することも可能である。
```c
int world_rank, world_size;
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
/* 引数
 * - 分割元(グループ)のコミュニケータ
 * - 振り分け先(サブグループ)の番号 : int
 * - 振り分け先(サブグループ)におけるランクのソートキー : int
 * - 分割先のコミュニケータへのポインタ : MPI_Comm*
 */
int color = world_rank % 2;
int key = world_rank; // グローバルランクをキーにする
MPI_Comm new_comm;
MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &new_comm);
int new_rank, new_size;
MPI_Comm_rank(new_comm, &new_rank); // 新しいコミュニケータにおけるランクを取得
```
他にも、`MPI_Comm_group, MPI_Group_incl, MPI_Comm_create`を組み合わせて使用する方法もある。

### 各プロセスにGPUデバイス1つを割り当てる

プロセス ID (rank) は、全ノード共通の通し番号を割り振られているため、
ノードごとにデバイスIDを割り当て直す必要がある。

```c
int num_nodes = atoi(argv[1]);
int devices_per_node; cudaGetDeviceCount(&devices_per_node);
int igpu = rank % (size/num_nodes); // ノード内のGPUデバイスID
cudaSetDevice(igpu);                // 使用するデバイス数を確保
```

### マスタープロセスでのデータ初期化
```c
void init_data(float *data, int rows, int cols) {
    // ホストメモリ確保
    data = (float *)malloc(rows * cols * sizeof(float));
    // データ初期化
    initializeXXXX(data, rows, cols); // XXXXX: Matrix, Vector, etc.
}

float *inputData;   // ホストメモリ上の入力の先頭ポインタ
float *outputData;  // ホストメモリ上の出力の先頭ポインタ
if (rank == 0) {    // マスタープロセスのみ
    // データ初期化
    init_data(inputData, rows, cols);
    init_data(outputData, rows, cols);
}
```

### 各プロセスの担当領域の計算
```c
int dataSize = (argc > 1) ? (1 << atoi(argv[1])) : 1024;
// データを均等に分割する
int elementsPerProcess = dataSize / size;
int startIdx = rank * elementsPerProcess;
int numElements = (rank == size - 1) ? (dataSize - startIdx) : elementsPerProcess; // 最後のプロセスが余りを担当

// このプロセスが処理する部分データのメモリ確保
float *local_inputData = (float *)malloc(numElements * sizeof(float));
```

### データの分配

マスタープロセスから全プロセスへデータを分配
```c
if (rank == 0) {
    // マスタープロセスは自分の部分を保持
    memcpy(local_inputData, inputData + startIdx, numElements * sizeof(float));

    // 他のプロセスにそれぞれの部分を送信
    for (int i = 1; i < size; i++) {
        int processStartIdx = i * elementsPerProcess; // オフセット
        int processNumElements = (i == size - 1) ? (dataSize - nodeStartIdx) : elementsPerProcess; // プロセスが担当する要素数
        
        // データ送信
        /* 引数
         * - 送信データの先頭ポインタ
         * - 送信データの要素数
         * - データ型
         * - 送信先プロセスのランク
         * - メッセージタグ（識別用）
         * - コミュニケータ
         */
        MPI_Send(&inputData[processStartIdx], processNumElements, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
} else {
    // 他のプロセスはマスターから自分の部分を受信
    /* 引数
     * - 受信バッファ        
     * - 受信する要素数
     * - データ型
     * - 送信元プロセスのランク（マスター）
     * - メッセージタグ（識別用）
     * - 通信コンテキスト
     * - ステータス情報を無視
     */
    MPI_Recv(local_inputData, numElements, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
```
教育上の観点から、`MPI_Send`, `MPI_Recv`を使用しているが、`MPI_Scatter`ないしは`MPI_Scatterv`を使用することもできる。

### デバイスメモリの確保とデータ転送
```c
// GPUメモリ確保
float *d_inputData, *d_outputData; // デバイスメモリのポインタ
size_t bytes = numElements * sizeof(float); // デバイスメモリのサイズ
cudaMalloc((void**)&d_inputData, bytes);   // ポインタのアドレス, サイズ
cudaMalloc((void**)&d_outputData, bytes);

// データ転送（ホスト→デバイス）
cudaMemcpy(d_inputData, localData, bytes, cudaMemcpyHostToDevice); // 転送先のポインタ, 転送元のポインタ, サイズ, 転送方向

// カーネル設定と実行
int threadsPerBlock = 256;
int blocksPerGrid = (myElements + threadsPerBlock - 1) / threadsPerBlock;
computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, myElements, SCALAR);
cudaDeviceSynchronize();

// 結果をホストに戻す
cudaMemcpy(local_outputData, d_outputData, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_inputData);
cudaFree(d_outputData);
```

### 結果の収集

全プロセスからマスタープロセスに収集
```c
// 全ノードから結果をマスターノードに収集
if (rank == 0) {
    // マスターノードは自分の部分をresultDataに戻す
    memcpy(resultData, local_outputData, numElements * sizeof(float));
    
    // 他のノードからそれぞれの部分を受信
    for (int i = 1; i < size; i++) {
        int processStartIdx = i * elementsPerProcess;
        int processNumElements = (i == size - 1) ? (dataSize - processStartIdx) : elementsPerProcess;
            
        // データ受信
        /* 引数
         * - 受信データの格納先ポインタ
         * - 受信データの要素数
         * - データ型
         * - 送信元プロセスのランク
         * - メッセージタグ（識別用）
         * - コミュニケータ
         * - ステータス情報を無視
         */
        MPI_Recv(&resultData[processStartIdx], processNumElements, MPI_FLOAT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
} else {
    // 他のノードはマスターに結果を送信
    /* 引数:
     * - 送信データの先頭ポインタ
     * - 送信データの要素数
     * - データ型
     * - 送信先プロセスのランク（マスター）
     * - メッセージタグ（識別用）
     * - 通信コンテキスト
     */
    MPI_Send(local_outputData, numElements, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
}
```
教育上の観点から、`MPI_Recv`, `MPI_Send`を使用しているが、`MPI_Gather`ないしは`MPI_Gatherv`を使用することもできる。

### メモリ解放・MPI終了

```c
// マスタープロセスのメモリ解放
if (rank == 0) {
    free(inputData);
    free(outputData);
}
// 全ノードで共通のメモリ解放
free(local_inputData);

MPI_Finalize(); // MPI終了
```

## Multi-Node code overview
```cpp
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

// カーネル関数の定義
__global__ void computeKernel(float *data, int size, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * scalar;
    }
}

int main(int argc, char** argv) {
    int rank, size;
    const float SCALAR = 2.0f;
    
    // MPI初期化
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // データサイズ（コマンドラインから取得）
    int dataSize = (argc > 1) ? atoi(argv[1]) : 1024*1024;
    
    // ホスト名とローカルランクを決定するためのコミュニケータ作成
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int nameLen;
    MPI_Get_processor_name(hostname, &nameLen);
    
    // 同一ノード内のプロセスをグループ化（ノード内コミュニケータ作成）
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    
    // ノード内ランクと総プロセス数を取得
    int nodeRank, nodeSize;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    
    // このランクが担当するGPUを設定
    int gpuId = nodeRank;
    cudaSetDevice(gpuId);
    
    // 各プロセスの担当データ量を計算
    int elementsPerProcess = dataSize / size;
    int myStartElement = rank * elementsPerProcess;
    int myElements = (rank == size - 1) ? (dataSize - myStartElement) : elementsPerProcess;
    
    // ルートプロセスでデータを初期化
    float* globalData = nullptr;
    if (rank == 0) {
        globalData = new float[dataSize];
        // データ初期化
        for (int i = 0; i < dataSize; i++) {
            globalData[i] = static_cast<float>(i);
        }
    }
    
    // 各プロセスのデータ領域を確保
    float* localData = new float[myElements];
    
    // ルートプロセスから各プロセスにデータを分配
    MPI_Scatter(globalData, elementsPerProcess, MPI_FLOAT, 
                localData, myElements, MPI_FLOAT, 
                0, MPI_COMM_WORLD);
    
    // GPUメモリ確保とデータ転送
    size_t bytes = myElements * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, localData, bytes, cudaMemcpyHostToDevice);
    
    // カーネル設定と実行
    int threadsPerBlock = 256;
    int blocksPerGrid = (myElements + threadsPerBlock - 1) / threadsPerBlock;
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, myElements, SCALAR);
    cudaDeviceSynchronize();
    
    // 結果をホストに戻す
    cudaMemcpy(localData, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    
    // 結果をルートプロセスに集約
    MPI_Gather(localData, myElements, MPI_FLOAT, 
               globalData, elementsPerProcess, MPI_FLOAT, 
               0, MPI_COMM_WORLD);
    
    // 結果表示（ルートプロセスのみ）
    if (rank == 0) {
        std::cout << "計算完了！先頭5要素: ";
        for (int i = 0; i < 5 && i < dataSize; i++) {
            std::cout << globalData[i] << " ";
        }
        std::cout << std::endl;
        
        delete[] globalData;
    }
    
    // リソース解放
    delete[] localData;
    MPI_Comm_free(&nodeComm);
    MPI_Finalize();
    return 0;
}
```

## Multi-Node code (MPI+thread) overview
```cpp
#include <mpi.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <iostream>

// カーネル関数の定義
__global__ void computeKernel(float *data, int size, float scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * scalar;
    }
}

// 各GPUで実行するスレッド関数（ノード内並列処理）
void processOnDevice(int gpuId, float* localData, int startIdx, int elements, float scalar) {
    // GPUの設定
    cudaSetDevice(gpuId);
    
    // データサイズとメモリ確保
    size_t bytes = elements * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // データ転送（ホスト→デバイス）
    cudaMemcpy(d_data, localData + startIdx, bytes, cudaMemcpyHostToDevice);
    
    // カーネル設定と実行
    int threadsPerBlock = 256;
    int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, elements, scalar);
    cudaDeviceSynchronize();
    
    // 結果転送（デバイス→ホスト）
    cudaMemcpy(localData + startIdx, d_data, bytes, cudaMemcpyDeviceToHost);
    
    // メモリ解放
    cudaFree(d_data);
}

int main(int argc, char** argv) {
    int rank, size;
    const float SCALAR = 2.0f;
    
    // MPI初期化 - スレッドサポートレベルを指定
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "警告: MPIはMPI_THREAD_MULTIPLEをサポートしていません" << std::endl;
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // データサイズ（コマンドラインから取得）
    int dataSize = (argc > 1) ? atoi(argv[1]) : 1024*1024;
    
    // ノード内のGPU数を取得
    int numGpusPerNode;
    cudaGetDeviceCount(&numGpusPerNode);
    
    // ルートプロセスでデータを初期化
    float* globalData = nullptr;
    if (rank == 0) {
        globalData = new float[dataSize];
        // データ初期化
        for (int i = 0; i < dataSize; i++) {
            globalData[i] = static_cast<float>(i);
        }
    }
    
    // 各ノードが担当する要素数を計算
    int elementsPerNode = dataSize / size;
    int myStartElement = rank * elementsPerNode;
    int myElements = (rank == size - 1) ? (dataSize - myStartElement) : elementsPerNode;
    
    // 各ノードはローカルデータ領域を確保
    float* nodeData = new float[myElements];
    
    // ルートノードから各ノードにデータを分配
    MPI_Scatter(globalData, elementsPerNode, MPI_FLOAT, 
                nodeData, myElements, MPI_FLOAT, 
                0, MPI_COMM_WORLD);
    
    // ノード内での各GPUへのデータ分割
    int elementsPerGpu = myElements / numGpusPerNode;
    
    // 各GPUのスレッド作成と実行
    std::vector<std::thread> threads;
    for (int i = 0; i < numGpusPerNode; i++) {
        int gpuStartElement = i * elementsPerGpu;
        int gpuElements = (i == numGpusPerNode - 1) ? 
                          (myElements - gpuStartElement) : elementsPerGpu;
        
        threads.push_back(std::thread(
            processOnDevice, i, nodeData, gpuStartElement, gpuElements, SCALAR
        ));
    }
    
    // 全スレッドの終了を待機
    for (auto& t : threads) {
        t.join();
    }
    
    // 結果をルートプロセスに集約
    MPI_Gather(nodeData, myElements, MPI_FLOAT, 
               globalData, elementsPerNode, MPI_FLOAT, 
               0, MPI_COMM_WORLD);
    
    // 結果表示（ルートプロセスのみ）
    if (rank == 0) {
        std::cout << "計算完了！先頭5要素: ";
        for (int i = 0; i < 5 && i < dataSize; i++) {
            std::cout << globalData[i] << " ";
        }
        std::cout << std::endl;
        
        delete[] globalData;
    }
    
    delete[] nodeData;
    MPI_Finalize();
    return 0;
}
```