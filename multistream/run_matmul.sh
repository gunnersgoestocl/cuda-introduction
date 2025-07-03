#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

MATRIX_SIZE=512
NUM_BATCHES=8
NUM_STREAMS=4

module purge
module load nvidia/24.9
module load cuda/12.6
module load ompi-cuda

cd ${PBS_O_WORKDIR}

mkdir -p result

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定
RESULT_FILE="result_matmul_multistream_${TIMESTAMP}.txt"

make clean
make

./matmul_multistream $MATRIX_SIZE $NUM_BATCHES $NUM_STREAMS

# ./matmul_multistream の実行が完了した後に以下を実行
if [ $? -eq 0 ]; then
    # move result
    mv latest_result.txt "result/${RESULT_FILE}"
else
    echo "matmul_multistream の実行中にエラーが発生しました。" >&2
    exit 1
fi