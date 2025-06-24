#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -j oe
#PBS -N gh200_um_bench
#PBS -V                       # 環境変数を引き継ぐ

#-------------------------------------------------------------
#  GH200 Unified-Memory benchmark
#  使い方:
#      qsub run.sh 35   # 2^35 floats  ≈ 136 GiB/vec (> 96 GiB HBM)
#      qsub run.sh      # 引数なしなら既定 EXP=35
#-------------------------------------------------------------
set -eu
module purge
module load cuda              # システムに合わせてバージョン指定可

cd "${PBS_O_WORKDIR}"

#-------- ビルド --------#
make clean
make all                     # um と explicit を生成

#-------- 問題サイズ設定 --------#
EXP=${1:-35}                 # N = 2^EXP 要素

echo "===== GH200 Unified-Memory vs Explicit-Copy ====="
echo "Problem size exponent : $EXP"
echo "Node                  : $(hostname)"
echo "CUDA version          : $(nvcc --version | grep release)"
echo "Start                 : $(date)"
echo

#-------- Unified Memory 実行 --------#
echo "--- Unified Memory (um) ---"
./um "$EXP"   | tee "um_${EXP}.log"

#-------- Explicit Copy 実行 --------#
echo
echo "--- Explicit Copy (explicit) ---"
./explicit "$EXP" | tee "explicit_${EXP}.log"

echo
echo "Finish: $(date)"
echo "Logs : um_${EXP}.log , explicit_${EXP}.log"
