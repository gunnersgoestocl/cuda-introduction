# CUDA Programming Challenge

## “Brute‑Force k‑Nearest Neighbor (k‑NN) Kernel for a Vector Database”

---

## 1. Why this problem?

Modern vector databases (Pinecone, Milvus, Weaviate, FAISS, etc.) rely on extremely fast *vector similarity search* to retrieve the top‑K nearest embeddings out of millions or billions. Even highly‑optimized approximate indexes (IVF‑PQ, HNSW, ScaNN) fall back on an *exact* brute‑force pass for small candidate lists, re‑ranking, or recall evaluation. On GPUs, this exact pass is often the throughput bottleneck and dominates both **latency** and **energy** cost ([NVIDIA Developer][1]).

Designing a high‑performance CUDA kernel for brute‑force k‑NN is therefore a **bite‑sized yet high‑impact** exercise: it teaches memory‑bandwidth management, warp‑level primitives, register blocking, and GPU‑centric Top‑K selection, all of which translate directly to real‑world vector‑database speedups ([GitHub][2], [vincentfpgarcia.github.io][3]).

---

## 2. Learning goals

| Goal                                         | Why it matters for vector DBs                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Memory coalescing & shared‑memory tiling** | Distance computation is a pure memory‑bandwidth problem; poor layout wastes >50 % of GPU BW |
| **Warp‑level reduction & Top‑K selection**   | k‑NN requires a *partial* sort (k≪N); naïve global sorts are 100× slower                    |
| **Mixed precision & tensor cores**           | L2 / cosine distances tolerate FP16, doubling effective BW on Ampere+                       |
| **Roofline‑based performance analysis**      | Learn to attribute bottlenecks to compute vs. memory and guide optimizations                |

---

## 3. Problem statement

> **Implement and optimize a CUDA kernel that, given**
> – A database matrix **D** ∈ ℝ<sup>N×d</sup> (N ≤ 1 024 000, d = 128) stored row‑major in GPU global memory
> – A query matrix **Q** ∈ ℝ<sup>Q×d</sup> (Q ≤ 4096)
> **returns the indices and squared‑L2 distances of the k = 10 nearest neighbors for every query**, in *ascending* distance order.
>
> You must deliver:
>
> 1. A **baseline** CPU (single‑thread) reference implementation (for correctness & speedup measurement).
> 2. A **naïve** CUDA kernel (one thread = one distance) that is functionally correct.
> 3. At least **two incremental optimizations**, chosen from the menu below, that together achieve ≥ **250× speedup** over the CPU baseline on a recent NVIDIA RTX/Ampere or newer GPU.
> 4. A *one‑page* roofline analysis summarizing where each optimization moves the kernel on the roofline plot.

### Allowed optimization menu (pick at least two)

1. **Tiled shared‑memory loads** of query and database tiles (e.g., 32×128 floats) to turn scattered global reads into coalesced 128‑byte transactions.
2. **Vectorized I/O** (`float4`/`__half2`) and *mixed‑precision* accumulation using tensor‑core MMA or CUTLASS GEMM, exploiting the distance‑as‑GEMM identity ([NVIDIA Developer][1]).
3. **Warp‑level partial reduction**: use `__shfl_down_sync` or cooperative groups to keep a per‑warp Top‑K heap in registers, pushing only the final candidates to shared/global memory.
4. **Block‑tiling + register blocking**: each block computes a Q\_tile × D\_tile distance sub‑matrix; each thread accumulates four distances in registers to increase arithmetic intensity.
5. **Asynchronous memory copies** (`cp.async`) on Hopper/Ada architectures to overlap global‑to‑shared transfers with compute.

---

## 4. Reference interface (header)

```cpp
// distances_out: Q × k (row‑major),  float32
// indices_out  : Q × k (row‑major),  int32
void gpu_knn(const float* __restrict__ d_database,
             const float* __restrict__ d_queries,
             int32_t*    __restrict__ d_indices_out,
             float*      __restrict__ d_distances_out,
             int N, int Q, int d, int k /*=10*/);
```

Compile with `nvcc -arch=sm_80 -O3 -lineinfo`.

---

## 5. Evaluation protocol

1. **Correctness**: Mismatch ≤ 1 e‑4 with CPU reference per distance (allow FP16 rounding).
2. **Performance**: Report

   * *Throughput* = `(N·Q) / elapsed_time` distances · s⁻¹
   * *End‑to‑end latency* for Q = 1024, 4096
   * *SM occupancy* and achieved DRAM BW (from Nsight Compute)
3. **Scalability experiment**: Plot throughput vs. N on log‑scale up to 1 M. Verify O(N) behavior.
4. **Reflection**: ½‑page explaining which bottleneck you hit first and how each optimization attacked it.

---

## 6. Hints & starter ideas

* **Distance = GEMM** trick: ‖q−d‖² = ‖q‖²+‖d‖²−2 qᵀd → pre‑compute ‖d‖² once, reuse across queries; inner products become a Q×N GEMM that tensor cores can accelerate ([arXiv][4]).
* Use **`cub::WarpReduce`** or **`thrust::topk`‑style reduction** for efficient per‑warp Top‑K.
* Optimal tile sizes often align with **memory‑controller burst length** (128 B) and **warp size** (32). Start with 128 threads/block, 4 warps, 16×16 tile.
* Batch queries so that **Q\_tile · d** fits into shared memory (48–100 KiB per SM on Ampere).
* Study FAISS GPU kernels for proven parameter choices ([Engineering at Meta][5]).

---

## 7. Extension paths (optional, for extra credit)

| Path                     | Idea                                                                                                                                                   |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Indexing**             | Add a coarse *IVF* layer: cluster database into 1024 centroids (k‑means on CPU), search only the 4 nearest cells on GPU. Measure recall\@10 vs. speed. |
| **Precision vs. Recall** | Compare FP16 vs. FP32 distances on a real embedding set (e.g., GloVe vectors). Report recall degradation, if any.                                      |
| **Multi‑GPU sharding**   | Partition D across two GPUs; implement NCCL all‑gather of Top‑K, show near‑linear speedup ([GitHub][2]).                                               |
| **In‑GPU paging**        | Stream database chunks from host to device with `cudaMemcpyAsync` + compute/transfer overlap for N > 10 M.                                             |

---

## 8. Deliverables checklist

* [ ] `knn_cpu.cpp` + `knn_gpu.cu` source files
* [ ] `benchmark.py` or `run.sh` with reproducible timing
* [ ] Nsight Compute report (`.ncu-rep`) highlighting memory vs. compute utilization
* [ ] PDF write‑up (≤ 4 pages) with roofline plot and optimization narrative

Good luck, and enjoy squeezing every last GFLOP out of your GPU! 🎯

[1]: https://developer.nvidia.com/blog/accelerated-vector-search-approximating-with-nvidia-cuvs-ivf-flat/?utm_source=chatgpt.com "Accelerated Vector Search: Approximating with NVIDIA cuVS ..."
[2]: https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU?utm_source=chatgpt.com "Faiss on the GPU · facebookresearch/faiss Wiki - GitHub"
[3]: https://vincentfpgarcia.github.io/data/Garcia_2010_ICIP.pdf?utm_source=chatgpt.com "[PDF] k-nearest neighbor search: fast gpu-based implementations"
[4]: https://arxiv.org/pdf/1702.08734?utm_source=chatgpt.com "[PDF] Billion-scale similarity search with GPUs - arXiv"
[5]: https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/?utm_source=chatgpt.com "Faiss: A library for efficient similarity search - Engineering at Meta"
