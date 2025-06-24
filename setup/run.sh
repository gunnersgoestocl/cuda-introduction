#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

module purge
module load cuda

cd ${PBS_O_WORKDIR}

make clean
make

mkdir -p result

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定
RESULT_FILE="result_${TIMESTAMP}.txt"

echo "\n==device detection=="
./device_query        # device detection

echo "\n==hello world with debug info=="
./hello_world_check   # with debug

echo "\n==hello world without debug info=="
./hello_world           # without debug

echo "\n==hello gpu=="
./hello_gpu

# move result and attach symbolic link
mv latest_result.txt "result/${RESULT_FILE}"
cd result
ln -sf ${RESULT_FILE} ../latest_result.txt
cd ..
echo "最新の結果は latest_result.txt でも確認できます"