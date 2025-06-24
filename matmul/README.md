# CUDAカーネルのPTX対応分析

ご提供いただいた matmul_compare.cu とその PTX コードの対応関係を解説します。特に2つのカーネル関数 `matMulGpu` と `matrixMulKernel` に着目して分析します。

## 1. matMulGpu カーネル

### C++コード
```cpp
__global__ void matMulGpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= dim || j >= dim) return;
    for (int k = 0; k < dim; ++k){
        output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
    }
}
```

### 対応するPTXコード（主要部分）

```plaintext
.visible .entry _Z9matMulGpuPdS_S_i(
	.param .u64 _Z9matMulGpuPdS_S_i_param_0,  // input_mat1
	.param .u64 _Z9matMulGpuPdS_S_i_param_1,  // input_mat2
	.param .u64 _Z9matMulGpuPdS_S_i_param_2,  // output_mat
	.param .u32 _Z9matMulGpuPdS_S_i_param_3   // dim
)
```

1. **インデックス計算部分**:
   ```
   mov.u32 	%r14, %ctaid.y;             // blockIdx.y を取得
   mov.u32 	%r15, %ntid.y;             // blockDim.y を取得
   mov.u32 	%r16, %tid.y;             // threadIdx.y を取得
   mad.lo.s32 	%r1, %r15, %r14, %r16;   // i = blockDim.y * blockIdx.y + threadIdx.y
   
   mov.u32 	%r17, %ctaid.x;             // blockIdx.x を取得
   mov.u32 	%r18, %ntid.x;             // blockDim.x を取得
   mov.u32 	%r19, %tid.x;             // threadIdx.x を取得
   mad.lo.s32 	%r2, %r18, %r17, %r19;   // j = blockDim.x * blockIdx.x + threadIdx.x
   ```

2. **境界チェック**:
   ```
   setp.ge.s32 	%p1, %r1, %r13;         // i >= dim ?
   setp.ge.s32 	%p2, %r2, %r13;         // j >= dim ?
   or.pred  	%p3, %p1, %p2;           // i >= dim || j >= dim
   setp.lt.s32 	%p4, %r13, 1;           // dim < 1 ?
   or.pred  	%p5, %p3, %p4;           // 条件判定
   @%p5 bra 	$L__BB0_7;                // 条件を満たす場合は関数終了へジャンプ
   ```

3. **ループ最適化部分**:
   ```
   // ループのアンロール最適化
   // 4つの繰り返しをまとめて処理
   $L__BB0_3:
       // input_mat1[i*dim+k] と input_mat2[k*dim+j] を読み込んで乗算
       ld.global.f64 	%fd7, [%rd30];       // input_mat2[k*dim+j]
       ld.global.f64 	%fd8, [%rd24];       // input_mat1[i*dim+k]
       fma.rn.f64 	%fd9, %fd8, %fd7, %fd20;  // output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j]
       st.global.f64 	[%rd3], %fd9;        // 結果を保存
   ```

## 2. matrixMulKernel カーネル（共有メモリ版）

### C++コード
```cpp
__global__ void matrixMulKernel(const double* A, const double* B, double* C, int M, int N, int K) {
    // 共有メモリ定義
    __shared__ double sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
```

### 対応するPTXコード（主要部分）

```plaintext
.visible .entry _Z15matrixMulKernelPKdS0_Pdiii(
	.param .u64 _Z15matrixMulKernelPKdS0_Pdiii_param_0,  // A
	.param .u64 _Z15matrixMulKernelPKdS0_Pdiii_param_1,  // B
	.param .u64 _Z15matrixMulKernelPKdS0_Pdiii_param_2,  // C
	.param .u32 _Z15matrixMulKernelPKdS0_Pdiii_param_3,  // M
	.param .u32 _Z15matrixMulKernelPKdS0_Pdiii_param_4,  // N
	.param .u32 _Z15matrixMulKernelPKdS0_Pdiii_param_5   // K
)
// 共有メモリの宣言
.shared .align 8 .b8 _ZZ15matrixMulKernelPKdS0_PdiiiE2sA[2048];  // sA配列
.shared .align 8 .b8 _ZZ15matrixMulKernelPKdS0_PdiiiE2sB[2048];  // sB配列
```

1. **インデックス計算**:
   ```
   mov.u32 	%r23, %ctaid.y;             // blockIdx.y を取得
   shl.b32 	%r24, %r23, 4;              // blockIdx.y << 4 (×16)
   mov.u32 	%r41, %tid.y;              // threadIdx.y を取得
   add.s32 	%r2, %r24, %r41;           // row = blockIdx.y * BLOCK_SIZE + threadIdx.y
   
   mov.u32 	%r25, %ctaid.x;             // blockIdx.x を取得
   shl.b32 	%r3, %r25, 4;              // blockIdx.x << 4 (×16)
   mov.u32 	%r40, %tid.x;              // threadIdx.x を取得
   add.s32 	%r5, %r3, %r40;            // col = blockIdx.x * BLOCK_SIZE + threadIdx.x
   ```

2. **共有メモリへのデータロード**:
   ```
   // 共有メモリのアドレス計算
   shl.b32 	%r27, %r41, 7;             // threadIdx.y * 128 (16要素×8バイト)
   mov.u32 	%r28, _ZZ15matrixMulKernelPKdS0_PdiiiE2sA;  // sAの開始アドレス
   add.s32 	%r8, %r28, %r27;           // sA行のベースアドレス
   
   // A行列から共有メモリへロード
   ld.global.f64 	%fd60, [%rd15];      // グローバルメモリから読み込み
   st.shared.f64 	[%r6], %fd60;        // 共有メモリに格納
   
   // B行列から共有メモリへロード
   ld.global.f64 	%fd61, [%rd11];      // グローバルメモリから読み込み
   st.shared.f64 	[%r7], %fd61;        // 共有メモリに格納
   
   // 同期バリア
   bar.sync 	0;                       // スレッド同期（全スレッドがデータロード完了するまで待機）
   ```

3. **積の計算とアンロールされたループ**:
   ```
   // アンロールされた16回の乗算加算ループ
   ld.shared.f64 	%fd12, [%r9];          // sB[0][threadIdx.x]
   ld.shared.f64 	%fd13, [%r8];          // sA[threadIdx.y][0]
   fma.rn.f64 	%fd14, %fd13, %fd12, %fd62;  // value += sA * sB
   
   // 以下同様の処理が15回繰り返される
   ```

4. **同期と次のタイル処理**:
   ```
   bar.sync 	0;                        // タイル処理完了同期
   // 次のタイルの準備
   shl.b32 	%r38, %r22, 4;              // K * 16
   add.s32 	%r42, %r42, %r38;           // 次のBタイルのインデックス更新
   ```

5. **結果の書き込み**:
   ```
   // 境界チェック後、結果をグローバルメモリに書き込み
   st.global.f64 	[%rd14], %fd62;       // C[row * K + col] = value
   ```

## 主な最適化の違い

1. **メモリアクセスパターン**:
   - `matMulGpu`: グローバルメモリから直接読み込み（レイテンシ大）
   - `matrixMulKernel`: 共有メモリへの一時格納で局所性を高め、メモリアクセスを削減

2. **ループ最適化**:
   - `matMulGpu`: 通常の4要素アンロール
   - `matrixMulKernel`: 16×16タイルを使った完全アンロール（16回の乗算加算）

3. **スレッド同期**:
   - `matMulGpu`: 同期なし（スレッド間協調なし）
   - `matrixMulKernel`: `bar.sync`でブロック内スレッド同期を実装

PTXコードからは、共有メモリ版の実装が、メモリアクセスの効率化とスレッド間の協調動作を意識した最適化が行われていることがわかります。