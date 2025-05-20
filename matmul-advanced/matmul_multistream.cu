#include <thread>
#include <cuda_runtime.h>

__global__ void myKernel(float *d, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) d[i] = d[i] * 2.0f;
}

int main() {
    // 
    const int N = 1<<20;
    float *h0 = new float[N], *h1 = new float[N];

    cudaSetDevice(0);                // ← GPU を切替
    float *d; size_t bytes = N*sizeof(float);
    cudaMalloc(&d, bytes);

    dim3 blk(256), grd((N+255)/256);

    cudaStream_t sCompute, sCopy;
    cudaStreamCreate(&sCompute);
    cudaStreamCreate(&sCopy);

    // 1) 非同期コピー（H→D）を stream sCopy
    cudaMemcpyAsync(d_in,  h_in,  bytes, cudaMemcpyHostToDevice, sCopy);

    // 2) カーネルを stream sCompute
    myKernel<<<grid, block, 0, sCompute>>>(d_in, d_out);

    // 3) 非同期コピー（D→H）を stream sCopy
    cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, sCopy);

    // 4) 任意の同期点
    cudaEvent_t done;  cudaEventCreate(&done);
    cudaEventRecord(done, sCopy);           // sCopy 完了後に立つ
    cudaStreamWaitEvent(sCompute, done, 0); // sCompute は done まで待つ
}