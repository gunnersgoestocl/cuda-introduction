#!/bin/bash
#PBS -q debug-g
#PBS -l select=16
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

MATRIX_SIZE_POWER=10
NUM_GPUS=4
NUM_STREAMS=4
NUM_NODES=16

module purge
module load nvidia/24.9
module load cuda/12.6
module load ompi-cuda

cd ${PBS_O_WORKDIR}

mkdir -p result src

make clean
make

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定
RESULT_FILE="result_multinode_${TIMESTAMP}.txt"
CSV_FILE="timing_data_${TIMESTAMP}.csv"

# ./matmul_multidevice $MATRIX_SIZE_POWER $NUM_GPUS
# ./matmul_multistream $MATRIX_SIZE_POWER $NUM_STREAMS
mpirun -n ${NUM_NODES} ./matmul_multinode ${MATRIX_SIZE_POWER}

if [ $? -eq 0 ]; then
    # CSVデータを抽出
    sed -n '/===== TIMING_DATA_CSV_START =====/,/===== TIMING_DATA_CSV_END =====/p' latest_result.txt | \
    grep -v "===== TIMING_DATA_CSV" > "result/${CSV_FILE}"

    # move result
	mv latest_result.txt "result/${RESULT_FILE}"
    cp matmul_multinode.cu "src/matmul_multinode_${TIMESTAMP}.cu"

    echo "結果ファイル: result/${RESULT_FILE}"
    echo "CSVファイル: result/${CSV_FILE}"
    
else
    echo "multinode の実行中にエラーが発生しました。" >&2
    exit 1
fi
