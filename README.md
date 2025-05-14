# cuda-introduction

以下では、まずCUDAプログラミングの概要を説明し、次に優れたCUDAプログラムの特徴を整理します。最後に、行列積演算を題材にした簡単なサンプルコードを示します。

---

## 1. CUDAプログラミングの概要

### GPUを用いた並列計算

* **CUDA (Compute Unified Device Architecture)** はNVIDIAが提供するGPU向けの並列プログラミング環境です。
* GPUは多数のコアを持ち、同時に多くの演算を実行できるため、行列演算や画像処理、ディープラーニングなどの大規模な並列計算に適しています。

### CUDAプログラミングの基本構造

1. **ホスト (CPU) コード** と **デバイス (GPU) コード** に分かれる

   * ホスト側(CPU)でデータの準備やGPUカーネル呼び出し、結果受け取りを実施する。
   * デバイス側(GPU)で行列積やベクトル加算といった並列タスクを実行する。
2. **カーネル関数**

   * `__global__`キーワードを使い、GPUで並列実行される関数を定義。
   * カーネルを呼び出す際には `<<<gridDim, blockDim>>>` のように、グリッド(blockの集合)とブロック(threadの集合)の次元を指定する。
3. **メモリ管理**

   * CUDAではホストメモリとデバイスメモリが分離しているため、`cudaMalloc()` や `cudaMemcpy()` を用いて転送が必要。
   * メモリのアクセスパターンや共有メモリの活用が性能に大きく影響する。

---

## 2. 優れたCUDAプログラムの特徴

1. **並列度の最大化**

   * 適切なグリッド・ブロック・スレッド構成 (thread block size) により、GPU上の多数のコアを無駄なく活用する。
2. **メモリアクセスの最適化 (メモリコアレス)**

   * グローバルメモリアクセスを連続したパターンにする (coalesced access) ことで帯域を有効に使い、転送コストを削減する。
3. **共有メモリの活用**

   * デバイスの高速なオンチップメモリ (shared memory) を使って、頻繁なグローバルメモリアクセスを減らす。
4. **レジスタやブロックあたりのリソースの効率的利用**

   * スレッドブロックあたりに割り当てられるレジスタ数や共有メモリ量には上限があるため、過不足なく利用し、高い“occupancy”(GPU全体の活性度)を目指す。
5. **分岐の最小化**

   * GPUでは同一warp内のスレッドが分岐すると性能低下につながるため、条件分岐を減らす工夫をする。
6. **非同期処理やストリームの活用**

   * データ転送と計算の重ね合わせ(オーバーラップ)などにより、ホストとデバイスを効率的に動かす。
7. **計算量とメモリアクセスのバランス**

   * 演算とメモリ転送のバランスを見極めてボトルネックを特定し、必要に応じてアルゴリズムを改良する。

---

## 3. 行列積を題材にしたサンプルコード

以下では、行列積 $C = A \times B$ を計算するカーネルの一例を示します。タイル(ブロック)ごとに共有メモリを利用してデータを取り込み、メモリアクセスの回数を削減する実装例です。学習や実験用の簡単な実装ですが、優れたCUDAプログラムでよく使われるテクニック（共有メモリによるタイルング）を取り入れています。

```cpp
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // スレッドブロックの次元(16x16)

__global__ void matrixMulKernel(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    // M×N行列 A と N×K行列 B の乗算で、結果は M×K行列 C
    // [M, N, K] はそれぞれ 行数×列数

    // 共有メモリ (タイル領域)
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    // スレッドの行と列（グローバル座標）
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float value = 0.0f;
    
    // タイルを順次読み込みながら計算を行う
    for (int t = 0; t < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // タイルの一部を共有メモリに読み込む
        int tiledCol = t * BLOCK_SIZE + threadIdx.x;
        int tiledRow = t * BLOCK_SIZE + threadIdx.y;

        // Aの該当領域を読み込み
        if (row < M && tiledCol < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Bの該当領域を読み込み
        if (col < K && tiledRow < N) {
            sB[threadIdx.y][threadIdx.x] = B[tiledRow * K + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 全スレッドが読み込み終わるのを待つ
        __syncthreads();

        // タイルサイズ(BLOCK_SIZE)分だけ要素同士の積を加算
        for (int i = 0; i < BLOCK_SIZE; i++) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        // 次のタイル計算まで同期
        __syncthreads();
    }

    // 計算結果をグローバルメモリに書き込む
    if (row < M && col < K) {
        C[row * K + col] = value;
    }
}

int main() {
    // 例として M=512, N=512, K=512 の行列サイズを仮定
    int M = 512, N = 512, K = 512;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // ホストメモリ確保
    std::vector<float> h_A(M*N, 1.0f);
    std::vector<float> h_B(N*K, 2.0f);
    std::vector<float> h_C(M*K, 0.0f);

    // デバイスメモリ確保
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // ホスト->デバイスにデータ転送
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    // グリッドとブロックの次元を決める
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // カーネル呼び出し
    matrixMulKernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    // デバイス->ホストへの結果受け取り
    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    // 結果の一部を確認 (例えば先頭要素)
    std::cout << "C[0] = " << h_C[0] << std::endl;

    // 後始末
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

### コードのポイント

1. **共有メモリ (sA, sB) を用いたタイルング**

   * グローバルメモリへのアクセスをブロック単位でまとめ、繰り返し利用することでメモリアクセスを削減している。
2. **コアレスメモリアクセス**

   * `(row * N + tiledCol)` や `(tiledRow * K + col)` のようにアクセス先を連続アドレスに配置し、可能な限りcoalesced accessを狙っている。
3. **条件分岐の最小化と同期**

   * 範囲外の要素アクセスには0をセットすることで計算を統一し、ブロック全体で同期を取りながらタイルごとに演算している。
4. **グリッドサイズ・ブロックサイズの設計**

   * `BLOCK_SIZE` は16や32などGPU特性に合わせて調整し、レジスタや共有メモリを過不足なく活用できるよう最適化する。

---

## 4. GPUの仕組みと cuda の提供する abstraction

### cuda プログラムのパラメータ

cuda プログラム中に登場するパラメータには以下のようなものがある。

- `gridDim (x, y)` : 1グリッドを構成するブロックの数(次元)
- `blockIdx (x, y)` : グリッド中のそのブロックの座標, `0≤blockIdx<gridDim` を満たす
- `blockDim (x, y)` : 1ブロックを構成するスレッドの数(次元), `blockDim.x < 1024, blockDim.y < 1024, blockDim.x * blockDim.y (* blockDim.z) ≤ 1024` を満たす必要がある
- `threadIdx (x, y)` : ブロック中のそのスレッド(を実行する計算コア)の座標, `0≤threadIdx<blockDim` を満たす

### `block`, `grid`, `thread` とハードウェア

同一`block`に属する`thread`は同一の **SM (stream multiprocessor)** で実行される。

> 現在のmiyabiに搭載されたGPUのアーキテクチャは、SM90である。 

また、同一`block`に属する`thread`どうしは、その`block`を担当する**SM**の **shared memory** を介して、データ転送や排他制御のためのバリア同期などを利用できる。
同一`grid`に属する`block`は同一の **GPU unit** 上のSMを利用して実行される。

つまり、`block`とは**SM**の abstraction であり、`thread`は**SM**内の各**CUDA core (processor)**の abstraction であり、`grid`は **GPU unit** の abstraction である。

### 3 questions to understand `cuda` 

ここで以下の疑問が生じる。
1. `block`が、割り当てられた**SM**の **processor(CUDA core)** 数を上回る`thread`を持つことは認められているのか、認められている場合どう処理しているのか
2. `grid`を構成する`block`数が、`grid`に対応する **GPU unit** を構成するSMの数を上回ることは認められているのか、認められている場合どう処理しているのか
3. 複数の`grid`に跨った実行はどのように行うのか

以上の疑問に対する答えを示せることは、`cuda` の実行モデルを理解し、ソフトウェアとしての `cuda` の本質を掴む上で極めて重要である。

まず、「同一`block`に属する`thread`は同一の SM で実行される」とは言ったが、これは「同時に実行する」ことを意味しない。また、「`block`に対してSMが割り当てられる」とも言ったが、これは「SMが一つの`block`しか担当できない」ことを意味しない。

SMで同時に実行されるのは、 **Warps** と呼ばれる同じ命令を実行するスレッド群(32の`thread`からなる)である。 **Warp** を構成する`thread`は全て同一の`block`に属する。

### Scheduling in NVIDIA GPU Architecture

**SM**は**ハードウェアスケジューラ(Warp Scheduler)**を有し、スケジューリングの単位は **Warp** である。このスケジューリングは、その**SM**に割り当てられた`block`の中で、`block`を跨いで行われる。例えば、

```txt
SM0: 
   residents:
      block0 (instruction0(1warp), instruction1(2warp))
      block1 (instruction2(4warp), instruction3(2warp))
      block2 (instruction4(1warp), instruction5(1warp))
      block3 (instruction6(1warp), instruction7(1warp))
   scheduler:
      instruction0(warp#1 of 1) -> instruction2(warp#1 of 4) -> instruction4(warp#1 of 1) -> instruction6(warp#1 of 1) ->
      instruction1(warp#2 of 2) -> instruction3(warp#1 of 2) -> instruction5(warp#1 of 1) -> instruction7(warp#1 of 1) ->
      instruction2(warp#2 of 4) -> instruction3(warp#2 of 2) -> 
      instruction1(warp#1 of 2)
```
といった具合に、`block`を跨いでスケジューリングされる。

また、各SMが持つ実行コア（CUDA Core）の数は、アーキテクチャによって異なり、32の倍数である。この場合、CUDA Coreの数 / 32 個の **Warp** を同時に実行可能である。

実行待ちの **Warp** は、スケジューリングキューに保管され、所定のスケジューリングポリシーに従って、実行可能な **Warp** が選択される。スケジューリングポリシーは、アーキテクチャによって異なるが、一般的には以下のようなものがある。

- **Round Robin**: スケジューリングキューの先頭から順に実行する。
- **Least Recently Used (LRU)**: 最も長い間実行待ちの **Warp** を優先的に実行する。
- **Greedy Then Oldest (GTO)**: 最も長い間実行待ちの **Warp** を優先的に実行するが、実行中の **Warp** がある場合は、実行中の **Warp** を優先的に実行する。

実際には、**GTO+ITS (Independent Thread Scheduling)**が最も挙動を近似していると考えられている。

### メモリアクセス

`cuda` には、メモリアクセスを **Warp** 単位で同時に行う(行いやすくする) abstraction が提供されている。

これを理解するために、非常に単純なGPUを用いた行列積のプログラムを以下に示す。
```c
__global__ void matMulGpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (i >= dim || j >= dim) return;
    // 3重ループ可能では?
    for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
    }
}
```
このプログラムにおけるメモリアクセスは、

- `input_mat1[i*dim+k]` : `input_mat1`の(i, k)成分をロードする
- `input_mat2[k*dim+j]` : `input_mat2`の(k, j)成分をロードする
- `output_mat[i*dim+j]` : `output_mat`の(i, j)成分をストアする

の3つである。

同じことを**CPU**で実行するプログラムは以下のようになる。
```c
void matMulCpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
            }
        }
    }
}
```
ただし、レジスタと変数の対応は以下の通りである。
- `sp+44`: `i`変数（一番外側のループのインデックス）
- `sp+40`: `j`変数（中間ループのインデックス）
- `sp+36`: `k`変数（一番内側のループのインデックス）
- `sp+24`: `input_mat1`のポインタ
- `sp+16`: `input_mat2`のポインタ
- `sp+8`: `output_mat`のポインタ
- `sp+4`: `dim`変数

これをコンパイルすると、以下のようなアセンブリコードが生成される。
```asm
.L10:
    // output_mat[i*dim+j] のアドレス計算と値読み込み
    ldr	w1, [sp, 44]        // w1 = i
    ldr	w0, [sp, 4]         // w0 = dim
    mul	w1, w1, w0          // w1 = i*dim
    ldr	w0, [sp, 40]        // w0 = j
    add	w0, w1, w0          // w0 = i*dim + j
    sxtw x0, w0             // x0に符号拡張
    lsl	x0, x0, 3           // x0 = 8*(i*dim + j) (doubleは8バイト)
    ldr	x1, [sp, 8]         // x1 = output_mat
    add	x0, x1, x0          // x0 = &output_mat[i*dim + j]
    ldr	d1, [x0]            // d1 = output_mat[i*dim + j]の現在値

    // input_mat1[i*dim+k] のアドレス計算と値読み込み
    ldr	w1, [sp, 44]        // w1 = i
    ldr	w0, [sp, 4]         // w0 = dim
    mul	w1, w1, w0          // w1 = i*dim
    ldr	w0, [sp, 36]        // w0 = k
    add	w0, w1, w0          // w0 = i*dim + k
    sxtw x0, w0             // x0に符号拡張
    lsl	x0, x0, 3           // x0 = 8*(i*dim + k)
    ldr	x1, [sp, 24]        // x1 = input_mat1 (input_mat1のアドレス)
    add	x0, x1, x0          // x0 = &input_mat1[i*dim + k]
    ldr	d2, [x0]            // d2 = input_mat1[i*dim + k]

    // input_mat2[k*dim+j] のアドレス計算と値読み込み
    ldr	w1, [sp, 36]        // w1 = k
    ldr	w0, [sp, 4]         // w0 = dim
    mul	w1, w1, w0          // w1 = k*dim
    ldr	w0, [sp, 40]        // w0 = j
    add	w0, w1, w0          // w0 = k*dim + j
    sxtw x0, w0             // x0に符号拡張
    lsl	x0, x0, 3           // x0 = 8*(k*dim + j)
    ldr	x1, [sp, 16]        // x1 = input_mat2
    add	x0, x1, x0          // x0 = &input_mat2[k*dim + j]
    ldr	d0, [x0]            // d0 = input_mat2[k*dim + j]

    // 乗算と加算
    fmul d0, d2, d0         // d0 = d2 * d0 = input_mat1[i*dim+k] * input_mat2[k*dim+j]

    // output_mat[i*dim+j]のアドレス再計算
    ldr	w1, [sp, 44]        // w1 = i
    ldr	w0, [sp, 4]         // w0 = dim
    mul	w1, w1, w0          // w1 = i*dim
    ldr	w0, [sp, 40]        // w0 = j
    add	w0, w1, w0          // w0 = i*dim + j
    sxtw x0, w0             // x0に符号拡張
    lsl	x0, x0, 3           // x0 = 8*(i*dim + j)
    ldr	x1, [sp, 8]         // x1 = output_mat
    add	x0, x1, x0          // x0 = &output_mat[i*dim + j]

    // 結果の更新
    fadd d0, d1, d0         // d0 = d1 + d0 = output_mat[i*dim+j] + 乗算結果
    str	d0, [x0]            // output_mat[i*dim+j] = d0 (更新された値を格納)

    // kループのインクリメント
    ldr	w0, [sp, 36]
    add	w0, w0, 1
    str	w0, [sp, 36]
```
これをSIMD(=複数のスレッドが異なるデータに対して同一の命令を実行)にするために唯一必要なことは、

**各スレッドが共通のレジスタ番地を用いて、異なる`i`, `j` を持つようにする**

ことである。
これを実現する機構が、スレッドごとに固有の`blockIdx`, `blockDim`, `threadIdx` を保持するレジスタである。

すなわち、各スレッドは共通のレジスタ番地(別のコアなので同じレジスタではない)に、`blockIdx`, `blockDim`, `threadIdx` を保持しており、これらを保持するレジスタ番地をオペランドに取って、`i`, `j` を同一の命令で計算し、その計算結果を共通のレジスタ番地に格納する。
最終的に、各スレッドが、そのレジスタ番地をオペランドに取って、`input_mat1`, `input_mat2`, `output_mat` のアドレスを計算し(この結果は当然スレッドごとに異なる)、メモリアクセスを行うことで
同一命令により異なるアドレスへのメモリアクセスを実現している。
```ptx
// インデックス計算部分
mov.u32 	%r14, %ctaid.y;               // blockIdx.y (ブロックごとに異なる) を取得
mov.u32 	%r15, %ntid.y;                // blockDim.y を取得
mov.u32 	%r16, %tid.y;                 // threadIdx.y (スレッドごとに異なる) を取得
mad.lo.s32 	%r1, %r15, %r14, %r16;   // i = blockDim.y * blockIdx.y + threadIdx.y

mov.u32 	%r17, %ctaid.x;               // blockIdx.x (ブロックごとに異なる) を取得
mov.u32 	%r18, %ntid.x;                // blockDim.x を取得
mov.u32 	%r19, %tid.x;                 // threadIdx.x (スレッドごとに異なる) を取得
mad.lo.s32 	%r2, %r18, %r17, %r19;   // j = blockDim.x * blockIdx.x + threadIdx.x

// ロード, 乗算, ストアのループ
// ループのアンロール最適化
// 4つの繰り返しをまとめて処理
$L__BB0_3:
    // input_mat1[i*dim+k] と input_mat2[k*dim+j] を読み込んで乗算
    ld.global.f64 	%fd7, [%rd30];       // input_mat2[k*dim+j]
    ld.global.f64 	%fd8, [%rd24];       // input_mat1[i*dim+k]
    fma.rn.f64 	%fd9, %fd8, %fd7, %fd20;  // output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j]
    st.global.f64 	[%rd3], %fd9;        // 結果を保存
```
