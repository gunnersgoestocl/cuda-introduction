#include <stdio.h>
#include <cuda_runtime.h>

// CUDAエラーチェック用マクロ
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d in file %s\n", \
               cudaGetErrorString(err), __LINE__, __FILE__); \
        exit(1); \
    } \
} while(0)

__global__ void hello(){
    printf("Hello CUDA World !!\n");
}

int main() {
    hello<<< 2, 4 >>>();
    // カーネル起動エラーをチェック
    CUDA_CHECK(cudaGetLastError());
    // 同期エラーもチェック
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}