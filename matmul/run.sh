#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

module purge
module load cuda

cd ${PBS_O_WORKDIR}

mkdir -p result

make clean
make

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定
RESULT_FILE="result_${TIMESTAMP}.txt"

echo "\n====EXPERIMENT 1: (64 x 64)===="
./matmul_compare 64

echo "\n====EXPERIMENT 2: (256 x 256)===="
./matmul_compare 256

echo "\n====EXPERIMENT 3: (1024 x 1024)===="
./matmul_compare 1024