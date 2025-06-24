#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h> 
#include <cuda_runtime.h>   // 
#include <math.h>           // fabs

#define BLOCK_SIZE 16  // 共有メモリカーネル用のスレッドブロックサイズ

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

/*

行列積の場合

for (int i = 0; i < nx; ++i){
    for (int j = 0; j < ny; ++j){
        // iとjを用いた処理
    }
}

--------CUDA化--------

int i = threadIdx.y + blockDim.y * blockIdx.y; <- xかyかはiとjの使われ方による
int j = threadIdx.x + blockDim.x * blockIdx.x;
//iとjを用いた処理

インプレンス社: CUDA Cのp59とかにあった
*/

void initMatrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}


double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

__global__ void matMulGpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= dim || j >= dim) return;
    // 3重ループ可能では?
    for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
    }
}

// 共有メモリを使った最適化されたカーネル
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int M, int N, int K) {
    // M×N行列 A と N×K行列 B の乗算で、結果は M×K行列 C
    // この場合、M=N=K=dim

    // 共有メモリ (タイル領域)
    __shared__ double sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sB[BLOCK_SIZE][BLOCK_SIZE];

    // スレッドの行と列（グローバル座標）
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double value = 0.0;
    
    // タイルを順次読み込みながら計算を行う
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // タイルの一部を共有メモリに読み込む
        int tiledCol = t * BLOCK_SIZE + threadIdx.x;
        int tiledRow = t * BLOCK_SIZE + threadIdx.y;

        // Aの該当領域を読み込み
        if (row < M && tiledCol < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Bの該当領域を読み込み
        if (col < K && tiledRow < N) {
            sB[threadIdx.y][threadIdx.x] = B[tiledRow * K + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0;
        }

        // 全スレッドが読み込み終わるのを待つ
        __syncthreads();

        // タイルサイズ(BLOCK_SIZE)分だけ要素同士の積を加算
        for (int i = 0; i < BLOCK_SIZE; i++) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // 次のタイル計算まで同期
        __syncthreads();
    }

    // 計算結果をグローバルメモリに書き込む
    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

// CPU版の行列積計算関数を追加
void matMulCpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < dim; ++k) {
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
            }
        }
    }
}

// GPU結果とCPU結果を比較する関数
int compareResults(double *result1, double *result2, int dim, double tolerance, const char *name1, const char *name2) {
    int errors = 0;
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            if (fabs(result1[i*dim+j] - result2[i*dim+j]) > tolerance) {
                if (errors < 10) { // 最大10個のエラーまで表示
                    printf("結果が一致しません！位置 [%d, %d]: %s=%f, %s=%f, 差=%e\n", 
                        i, j, name1, result1[i*dim+j], name2, result2[i*dim+j], 
                        fabs(result1[i*dim+j] - result2[i*dim+j]));
                }
                errors++;
            }
        }
    }
    
    if (errors > 0) {
        printf("合計 %d 箇所で不一致が見つかりました\n", errors);
        return 0;  // 不一致
    }
    printf("%s と %s の結果は一致しています.\n", name1, name2);
    return 1;  // 一致
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

void debugMatrix(double *mat, int dim, int max_display){
    int show_dim = (dim < max_display) ? dim : max_display;
    for (int i = 0; i < show_dim; ++i){
        for (int j = 0; j < show_dim; ++j){
            printf("%f ", mat[i*dim+j]);
        }
        printf("\n");
    }
    if (dim > max_display) {
        printf("... 行列が大きすぎるため一部のみ表示 ...\n");
    }
}

int main(int argc, char **argv){
    if (argc < 2){
        terminate("Usage matmul_gpu dim_size");
    }

    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);

    // デバイス情報の表示
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    printf("使用するGPUデバイス: %s\n", prop.name);

    int n = atoi(argv[1]);
    struct timespec start_time, end_time;
    double *input_mat1 = (double *)malloc(sizeof(double)*n*n);
    double *input_mat2 = (double *)malloc(sizeof(double)*n*n);
    double *gpu_result = (double *)malloc(sizeof(double)*n*n);
    double *cpu_result = (double *)malloc(sizeof(double)*n*n);
    double *shared_gpu_result = (double *)malloc(sizeof(double)*n*n);

    // 行列の次元を出力
    printf("行列の次元: %d x %d\n", n, n);
    
    // 行列の初期化
    initMatrix(input_mat1, 3.0, n);
    initMatrix(input_mat2, 0.1, n);
    initMatrix(gpu_result, 0.0, n);
    initMatrix(cpu_result, 0.0, n);
    initMatrix(shared_gpu_result, 0.0, n);

    // デバイスメモリの確保
    double *d_input_mat1, *d_input_mat2, *d_output_mat, *d_shared_output_mat;
    cudaMalloc((void**) &d_input_mat1,sizeof(double)*n*n);
    cudaMalloc((void**) &d_input_mat2,sizeof(double)*n*n);
    cudaMalloc((void**) &d_output_mat,sizeof(double)*n*n);
    cudaMalloc((void**) &d_shared_output_mat, sizeof(double)*n*n);

    // ホスト→デバイスのデータ転送
    cudaMemcpy(d_input_mat1, input_mat1, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_mat2, input_mat2, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_mat, gpu_result, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shared_output_mat, shared_gpu_result, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    // 標準カーネル用の実行構成
    int std_threads = 32;   assert(std_threads * std_threads < 1024 + 1);
    dim3 std_block(std_threads, std_threads);
    dim3 std_grid((n+std_block.x-1)/std_block.x, (n+std_block.y-1)/std_block.y);

    // 共有メモリカーネル用の実行構成
    dim3 shared_block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 shared_grid((n + shared_block.x - 1) / shared_block.x, (n + shared_block.y - 1) / shared_block.y);
    
    // カーネル構成の出力
    printf("\n--- 標準カーネルの実行構成 ---\n");
    printf("ブロック次元: (%d, %d)\n", std_block.x, std_block.y);
    printf("グリッド次元: (%d, %d)\n", std_grid.x, std_grid.y);
    printf("総スレッド数: %d\n", std_grid.x * std_grid.y * std_block.x * std_block.y);
    
    printf("\n--- 共有メモリカーネルの実行構成 ---\n");
    printf("ブロック次元: (%d, %d)\n", shared_block.x, shared_block.y);
    printf("グリッド次元: (%d, %d)\n", shared_grid.x, shared_grid.y);
    printf("総スレッド数: %d\n", shared_grid.x * shared_grid.y * shared_block.x * shared_block.y);

    // ===== 標準GPUカーネルでの計算 =====
    printf("\n===== 1. 標準GPUカーネルの計算 =====\n");
    clock_gettime(CLOCK_REALTIME, &start_time);
    matMulGpu<<<std_grid, std_block>>>(d_input_mat1, d_input_mat2, d_output_mat, n);
    cudaDeviceSynchronize(); // Wait until GPU processing finishs.
    cudaCheckError();
    cudaMemcpy(gpu_result, d_output_mat, sizeof(double)*n*n, cudaMemcpyDeviceToHost); 
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("標準GPUカーネル計算時間: %f秒\n", calculateElapsedTime(start_time, end_time));

    // ===== 共有メモリを使ったGPUカーネルでの計算 =====
    printf("\n===== 2. 共有メモリGPUカーネルの計算 =====\n");
    clock_gettime(CLOCK_REALTIME, &start_time);
    matrixMulKernel<<<shared_grid, shared_block>>>(d_input_mat1, d_input_mat2, d_shared_output_mat, n, n, n);
    cudaDeviceSynchronize();
    cudaCheckError();
    cudaMemcpy(shared_gpu_result, d_shared_output_mat, sizeof(double)*n*n, cudaMemcpyDeviceToHost); 
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("共有メモリGPUカーネル計算時間: %f秒\n", calculateElapsedTime(start_time, end_time));

    // ===== CPUでの計算 =====
    printf("\n===== 3. CPU計算 =====\n");
    clock_gettime(CLOCK_REALTIME, &start_time);
    matMulCpu(input_mat1, input_mat2, cpu_result, n);
    clock_gettime(CLOCK_REALTIME, &end_time);
    printf("CPU計算時間: %f秒\n", calculateElapsedTime(start_time, end_time));

    // ===== 結果の比較 =====
    printf("\n===== 結果の比較 =====\n");
    double tolerance = 1e-10;  // 許容誤差

    int std_vs_cpu = compareResults(gpu_result, cpu_result, n, tolerance, "標準GPU", "CPU");
    int shared_vs_cpu = compareResults(shared_gpu_result, cpu_result, n, tolerance, "共有メモリGPU", "CPU");
    int std_vs_shared = compareResults(gpu_result, shared_gpu_result, n, tolerance, "標準GPU", "共有メモリGPU");
    
    printf("\n===== 比較結果まとめ =====\n");
    printf("標準GPU vs CPU: %s\n", std_vs_cpu ? "一致" : "不一致");
    printf("共有メモリGPU vs CPU: %s\n", shared_vs_cpu ? "一致" : "不一致");
    printf("標準GPU vs 共有メモリGPU: %s\n", std_vs_shared ? "一致" : "不一致");

    // デバイスメモリの解放
    cudaFree(d_input_mat1);
    cudaFree(d_input_mat2);
    cudaFree(d_output_mat);
    cudaFree(d_shared_output_mat);

    // デバッグ出力（小さな行列サイズのみ）
    if (n <= 10) {
        printf("\n===== 計算結果（一部） =====\n");
        printf("\n標準GPUカーネル計算結果:\n");
        debugMatrix(gpu_result, n, 10);
        printf("\n共有メモリGPUカーネル結果:\n");
        debugMatrix(shared_gpu_result, n, 10);
        printf("\nCPU計算結果:\n");
        debugMatrix(cpu_result, n, 10);
    }

    // メモリ解放
    free(input_mat1);
    free(input_mat2);
    free(gpu_result);
    free(cpu_result);
    free(shared_gpu_result);

    return 0;
}