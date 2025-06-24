# VECADD: Unified Memory vs Explicit Copy

* **Unified-Memory mode** (`-DUSE_MANAGED`)
  – allocations use `cudaMallocManaged` and pages are *prefetched* to HBM on GH200.
* **Explicit-Copy mode** (no macro)
  – host arrays live in pinned DDR, device arrays in HBM; every iteration performs the usual `cudaMemcpy`.

The program runs a simple but illustrative *CPU ↔ GPU* pipeline:

1. GPU kernel adds two very large vectors **every iteration**.
2. Immediately after the kernel, the CPU reads the first 1 k elements of the result to drive a reduction (think: convergence check).
   *In Unified-Memory mode the CPU just dereferences the same pointer; in Explicit-Copy mode those 1 k floats are copied back with `cudaMemcpy`.*

Because the loop mixes GPU-heavy work with frequent CPU touches, it showcases the **C2C-coherent page migration HW** in Grace Hopper.
With a problem size that *exceeds 96 GB HBM* the Unified-Memory version also demonstrates GPU-resident ↔ DDR migration that would be painful to script by hand.

---

## How & **what** to compare

| Metric                                        | Why it matters on GH200                                                       | How to capture                                                                             |
| --------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Total wall-clock time**                     | End-to-end productivity (kernel + CPU touches + migrations + copies).         | `std::chrono` (already in code).                                                           |
| **Kernel-only time**                          | Reveals if paging stalls the GPU.                                             | CUDA events (already in code).                                                             |
| **HBM ↔ LPDDR traffic**                       | Shows how well page-prefetch / explicit chunking keeps hot data in HBM.       | Nsight Systems → *Memory Throughput* counters `dram__read_bytes`, `mem_nvlink_read_bytes`. |
| **Page-migration count / bytes** *(UM only)*  | Direct indicator of C2C demand-paging overhead.                               | Nsight Compute: “`uvm_memcpy.bytes`”, “`uvm_migration.pages`”.                             |
| **Memcpy H2D / D2H volume** *(Explicit only)* | Overhead your manual code pays.                                               | Nsight Systems → memcpy timeline, or `nvprof --print-gpu-trace`.                           |
| **Peak HBM residency**                        | Whether UM oversubscription helped run >96 GB datasets without manual tiling. | Nsight Systems → *UM Page Residency* track.                                                |
| **CPU utilisation**                           | UM hides copies but may steal cycles handling faults.                         | `nsys stats --cpu-events`.                                                                 |

### Suggested workflow

1. **Compile both binaries** with the same `-O3 -arch=sm_89`.
2. Choose a size that exceeds 96 GB per vector (`./um 30` in the example ≈ 120 GB each; adjust downward if memory-limited).
3. Record:

   ```bash
   nsys profile --stats=true --trace=cuda,nvlink_c2c,osrt ./um        30
   nsys profile --stats=true --trace=cuda,memcpy,osrt  ./explicit 30
   ```
4. Compare:

   * ```
         wall-clock (`Time (ns)` in summary)  
     ```
   * ```
         `uvm_migration.pages` vs `Memcpy HtoD/DtoH`  
     ```
   * ```
         HBM traffic saturation – are you now bandwidth-bound?  
     ```

---

### Why this miniature benchmark is interesting on **Grace Hopper**

* **Oversubscription path** – allocating 3 × 120 GB uses DDR automatically; the GPU sees only the 2 MiB chunks it actually touches.
* **Frequent CPU reads** – forces dirty-page invalidations in C2C; see how HW coherence mitigates them compared with manual `cudaMemcpy`.
* **Prefetch hint** – shows how `cudaMemPrefetchAsync` can hide first-touch latency, a best-practice unique to Unified Memory.

---

> **Take-away-checklist**

* **Run once with small N** (fits 96 GB) – Unified vs Explicit should be practically identical.
* **Run once > 96 GB** – Explicit version must chunk or will OOM; Unified keeps code unchanged but you’ll see migration costs.
* Investigate trade-offs: if kernel time dwarfs migration, UM wins; if CPU accesses dominate, explicit fine-grained copies may still beat demand paging.
