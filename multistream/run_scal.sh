#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

MATRIX_SIZE_POWER=24
NUM_GPUS=4
NUM_STREAMS=4
NUM_NODES=16

module purge
module load nvidia/24.9
module load cuda/12.6
module load ompi-cuda

cd ${PBS_O_WORKDIR}

mkdir -p result

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定
RESULT_FILE="result_scal_multistream_${TIMESTAMP}.txt"

make clean
make

./scal_multistream $MATRIX_SIZE_POWER $NUM_STREAMS

# ./scal_multistream の実行が完了した後に以下を実行
if [ $? -eq 0 ]; then
    # move result and attach symbolic link
    mv latest_result.txt "result/${RESULT_FILE}"
else
    echo "scal_multistream の実行中にエラーが発生しました。" >&2
    exit 1
fi