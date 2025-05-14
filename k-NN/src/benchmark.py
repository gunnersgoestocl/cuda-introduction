#!/usr/bin/env python3
"""
benchmark.py - build & time the CPU baseline (and, if present, a GPU binary).

Usage:
    python benchmark.py --N 100000 --Q 1024 --d 128 --k 10
"""
import argparse, subprocess, re, time, pathlib, sys, os

def build_cpu():
    print("🛠️  Compiling knn_cpu.cpp …")
    cmd = ["g++", "-O3", "-std=c++17", "-DBUILD_STANDALONE",
           "knn_cpu.cpp", "-o", "knn_cpu"]
    subprocess.check_call(cmd)

def build_gpu() -> bool:
    cu = pathlib.Path("knn_gpu.cu")
    if not cu.exists():
        return False
    print("🛠️  Compiling knn_gpu.cu …")
    cmd = ["nvcc", "-O3", "-arch=sm_80", "knn_gpu.cu", "-o", "knn_gpu"]
    subprocess.check_call(cmd)
    return True

def run_binary(bin_path, N, Q, d, k):
    cmd = [f"./{bin_path}", str(N), str(Q), str(d), str(k)]
    t0   = time.time()
    out  = subprocess.check_output(cmd, text=True)
    dt   = (time.time() - t0) * 1e3  # backup wall‑time
    m = re.search(r"([0-9.]+)\s*ms", out)
    return float(m.group(1)) if m else dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100000)
    ap.add_argument("--Q", type=int, default=1024)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    build_cpu()
    cpu_ms = run_binary("knn_cpu", args.N, args.Q, args.d, args.k)
    print(f"🧮  CPU time : {cpu_ms:8.2f} ms")

    if build_gpu():
        gpu_ms = run_binary("knn_gpu", args.N, args.Q, args.d, args.k)
        print(f"🚀  GPU time : {gpu_ms:8.2f} ms   (speed-up *{cpu_ms / gpu_ms:,.1f})")
    else:
        print("⚠️   knn_gpu.cu not found - GPU benchmark skipped.")

if __name__ == "__main__":
    main()
