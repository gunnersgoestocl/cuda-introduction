/****************************************************************************
 *  gh200_vecadd_um_vs_explicit.cu
 *  Build:
 *    # Unified Memory version
 *    nvcc -O3 -arch=sm_90 -DUSE_MANAGED        vecadd_compare.cu -o um
 *    # Explicit copy version
 *    nvcc -O3 -arch=sm_90                      vecadd_compare.cu -o explicit
 *
 *  Run (default 120 GiB per vector → 360 GiB total):
 *    ./um        30           # 30 · 2^30 floats ≈ 120 GiB           (Unified)
 *    ./explicit  30           # same size, but chunked transfers     (Explicit)
 ***************************************************************************/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>

#define CHECK(call)                                                         \
  do {                                                                      \
    cudaError_t e = call;                                                   \
    if (e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(e));                                       \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

__global__ void vecAdd(const float *A, const float *B, float *C, uint64_t n) {
  uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) C[i] = A[i] + B[i];
}

static inline double sec_since(const std::chrono::high_resolution_clock::time_point &t0) {
  return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

int main(int argc, char **argv) {
  //-------------------------------------------------------------------- args
  const uint64_t Gi = (1ull << 30);
  uint64_t exp2 = (argc > 1 ? strtoull(argv[1], nullptr, 10) : 30);   // 2^exp2 floats
  uint64_t N    = (1ull << exp2);
  size_t   bytes= N * sizeof(float);
  int iters     = 20;                                                 // bigger → steadier
  printf("Elements: 2^%llu  (%.3f GiB per vector, %d iterations)\n",
         (unsigned long long)exp2, bytes / (double)Gi, iters);

  //------------------------------------------------------------------ alloc
  float *A, *B, *C;

#ifdef USE_MANAGED
  CHECK(cudaMallocManaged(&A, bytes));
  CHECK(cudaMallocManaged(&B, bytes));
  CHECK(cudaMallocManaged(&C, bytes));
  // Hint: keep everything resident in HBM before first kernel launch
  int gpu = 0;
  CHECK(cudaMemAdvise      (A, bytes, cudaMemAdviseSetPreferredLocation, gpu));
  CHECK(cudaMemAdvise      (B, bytes, cudaMemAdviseSetPreferredLocation, gpu));
  CHECK(cudaMemAdvise      (C, bytes, cudaMemAdviseSetPreferredLocation, gpu));
  CHECK(cudaMemPrefetchAsync(A, bytes, gpu));
  CHECK(cudaMemPrefetchAsync(B, bytes, gpu));
  CHECK(cudaMemPrefetchAsync(C, bytes, gpu));
  CHECK(cudaDeviceSynchronize());
#else
  // Host (pinned) and Device allocations
  float *hA, *hB, *hC;
  CHECK(cudaHostAlloc(&hA, bytes, cudaHostAllocPortable));
  CHECK(cudaHostAlloc(&hB, bytes, cudaHostAllocPortable));
  CHECK(cudaHostAlloc(&hC, bytes, cudaHostAllocPortable));
  // Fill host arrays
  for (uint64_t i = 0; i < N; ++i) { hA[i] = 1.f; hB[i] = 2.f; }

  float *dA, *dB, *dC;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC, bytes));

  // One-time H2D copy for A and B (they are read-only in kernel)
  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  A = dA;  B = dB;  C = dC;            // let kernel see device pointers
#endif

#ifndef USE_MANAGED
  // Init host C for reduction buffer
  for(uint64_t i=0;i<1024;++i) hC[i]=0.f;
#endif

#ifndef USE_MANAGED
  printf("Mode: Explicit copy\n");
#else
  printf("Mode: Unified Memory\n");
#endif

  //------------------------------------------------------------------- warm-up
  dim3 block(256);
  dim3 grid((N + block.x - 1) / block.x);
  vecAdd<<<grid, block>>>(A, B, C, N);
  CHECK(cudaDeviceSynchronize());

  //------------------------------------------------------------------- timing
  cudaEvent_t ev0, ev1;
  CHECK(cudaEventCreate(&ev0));
  CHECK(cudaEventCreate(&ev1));

  CHECK(cudaEventRecord(ev0));
  auto  wall0 = std::chrono::high_resolution_clock::now();

  double cpu_acc = 0.0;
  for (int t = 0; t < iters; ++t) {
    vecAdd<<<grid, block>>>(A, B, C, N);

#ifndef USE_MANAGED
    // copy just the 1k-element head back for CPU reduction
    CHECK(cudaMemcpy(hC, C, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
#endif
    CHECK(cudaDeviceSynchronize());

#ifdef USE_MANAGED
    // Managed: CPU touches the same memory (may trigger migration)
#endif
    float *head = 
#ifndef USE_MANAGED
      hC
#else
      C
#endif
    ;
    double local = 0.0;
    for (int k = 0; k < 1024; ++k) local += head[k];
    cpu_acc += local;                // fake usage
  }

  CHECK(cudaEventRecord(ev1));
  CHECK(cudaEventSynchronize(ev1));
  auto wall1 = std::chrono::high_resolution_clock::now();

  float gpu_ms = 0.f;
  CHECK(cudaEventElapsedTime(&gpu_ms, ev0, ev1));
  printf("Kernel + sync time (GPU timer): %.3f ms\n", gpu_ms);
  printf("Total wall-clock time         : %.3f s\n", sec_since(wall0));

#ifdef USE_MANAGED
  printf("Dummy CPU checksum            : %.1f\n", cpu_acc);
#else
  printf("Dummy CPU checksum            : %.1f (uses explicit copy)\n", cpu_acc);
#endif

  //------------------------------------------------------------------- teardown
#ifdef USE_MANAGED
  CHECK(cudaFree(A));  CHECK(cudaFree(B));  CHECK(cudaFree(C));
#else
  CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));
  CHECK(cudaFreeHost(hA)); CHECK(cudaFreeHost(hB)); CHECK(cudaFreeHost(hC));
#endif
  return 0;
}
