#!/bin/bash
#PBS -q debug-g
#PBS -l select=1
#PBS -W group_list=gc64
#PBS -o latest_result.txt
#PBS -j oe

# ジョブディレクトリに移動
cd $PBS_O_WORKDIR

# 必要なディレクトリの作成
mkdir -p src ptx sass result

# タイムスタンプを生成 (YYYYMMDDHHMMSS形式)
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# 結果ファイル名を設定 (PBS -o オプションで使用するため)
RESULT_FILE="result_${TIMESTAMP}.txt"

# モジュールのロード
module purge
module load cuda

# コンパイラが利用可能か確認
which nvcc
if [ $? -ne 0 ]; then
    echo "Error: nvcc compiler not found. Check your module environment."
    exit 1
fi

# 実行対象ファイル
TARGET_FILE="tensor_core_nv.cu"

# コンパイル (make使用)
make clean
make

# 実行確認
if [ ! -x ./tensor_core_nv ]; then
    echo "Error: Executable 'tensor_core_nv' not found or not executable"
    exit 1
fi

# PTXファイルとSASSファイルの生成
PTX_FILE="tensor_core_nv_${TIMESTAMP}.ptx"
SASS_FILE="tensor_core_nv_${TIMESTAMP}.sass"

echo "Generating PTX and SASS files..."
cuobjdump -ptx tensor_core_nv > ${PTX_FILE}
cuobjdump --dump-sass tensor_core_nv > ${SASS_FILE}

echo "PTXファイル生成完了: ptx/${PTX_FILE}"
echo "SASSファイル生成完了: sass/${SASS_FILE}"

# TensorCore関連命令を検索してレポート
echo "TensorCore関連命令の検索結果:"
grep -E "wmma\." sass/${SASS_FILE} | head -n 5

# バックアップファイルをsrcディレクトリに移動
cp tensor_core_nv.cu src/tensor_core_nv_${TIMESTAMP}.cu

# 実行
echo -e "\n=== 実行結果 ==="
./tensor_core_nv

# プロファイリング実行（オプション）
if [ "$1" == "--profile" ]; then
    echo -e "\n=== プロファイリング実行 ==="
    PROFILE_OUTPUT="profile_${TIMESTAMP}"
    nsys profile -t cuda,nvtx,osrt -o "result/${PROFILE_OUTPUT}" ./tensor_core_nv
    echo "プロファイル結果は result/${PROFILE_OUTPUT} で確認できます"
fi

# 結果の要約
echo -e "\n=== 実験完了 ==="
echo "タイムスタンプ: ${TIMESTAMP}"
echo "ソースコード: src/tensor_core_nv_${TIMESTAMP}.cu"
echo "PTXコード: ptx/${PTX_FILE}"
echo "SASSコード: sass/${SASS_FILE}"
echo "実行結果: result/${RESULT_FILE}"

# 結果ファイルへのリンクを作成
if [ $? -eq 0 ]; then
    mv latest_result.txt "result/${RESULT_FILE}"
    # PTX/SASSファイルを適切なディレクトリに移動
    mv -v ${PTX_FILE} ptx/
    mv -v ${SASS_FILE} sass/
else
    echo "matmul_multistream の実行中にエラーが発生しました。" >&2
    exit 1
fi