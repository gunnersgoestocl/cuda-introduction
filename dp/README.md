# CUDA × Dynamic Programming　課題仕様（決定版）

## 1. 目的

2 本の DNA 文字列 **S, T**（長さはそれぞれ 1 ≤ |S|, |T| ≤ 8 192）について、
Smith‑Waterman 局所アラインメントを **GPU で高速に** 計算せよ。
CPU 向けベースライン実装と比較して **10 倍以上の speed‑up** を達成すること。

[NVIDIA DEVELOPER Technical Blog](https://developer.nvidia.com/blog/boosting-dynamic-programming-performance-using-nvidia-hopper-gpu-dpx-instructions/) で紹介されている。

---

## 2.DNA 局所アラインメントスコアとは ― Smith‑Waterman の考え方

**局所アラインメント**は，「配列全体」ではなく **部分配列どうしが最も類似する領域** を探す手法です。
生物学では，遺伝子やタンパク質の断片的な保存領域（モチーフ）を検出する目的で広く使われます。1970 年の Needleman–Wunsch（全体整列）に対し，**Smith‑Waterman (1981)** は“スコアが負になったら 0 にリセットする”というアイデアで **局所**最適を保証します。([ウィキペディア][1])

---

### スコアリングモデル

1 文字ずつ比較して

* **Match** :+2 （同じ塩基）
* **Mismatch** : −1 （異なる塩基）
* **Gap**（挿入・欠失）:

  * 開始 −2（gap open）
  * 継続 −1/文字（gap extend）

得点行列 *H* のセル $H_{i,j}$ は「**S の i 文字目までと T の j 文字目までで終わる局所整列の最高得点**」を表します。

---

### 動的計画法（DP）の再帰式

$$
\boxed{
H_{i,j}= \max\!\bigl(
0,\; H_{i-1,j-1}+s_{i,j},\; E_{i,j},\; F_{i,j}
\bigr)
}
$$

* $s_{i,j}=+2$ か −1
* 縦ギャップ: $E_{i,j}= \max(H_{i-1,j}+G_o,\; E_{i-1,j}+G_e)$
* 横ギャップ: $F_{i,j}= \max(H_{i,j-1}+G_o,\; F_{i,j-1}+G_e)$

ここで **0 を含める**ことが局所整列の核心：途中で負スコアになる経路を切り捨て，新しい部分配列を再スタートできます。最終スコアは行列中の最大値 $\max_{i,j} H_{i,j}$。([Vlab][2])

---

### なぜ「動的計画法」なのか

| DP の 2 大要件                               | Smith‑Waterman での具体例                                      |
| ---------------------------------------- | --------------------------------------------------------- |
| **最適部分構造**<br>(optimal substructure)     | あるセルの最適値は「直前のセル＋その一手（match/ gap）」だけで決まる。                  |
| **重複部分問題**<br>(overlapping sub‑problems) | 長さ $n,m$ のすべての部分配列ペアは $(n+1)(m+1)$ 個のセルに整理でき，同じ計算を再利用できる。 |

もし全ての開始・終了位置を総当たりすれば指数時間ですが，DP は **1 回の表埋めで O(n m)** に削減します。

---

### 小さな例で表の動き（S = “ACG”, T = “AG”）

|   | 0 | A | G |
| - | - | - | - |
| 0 | 0 | 0 | 0 |
| A | 0 | 2 | 0 |
| C | 0 | 0 | 1 |
| G | 0 | 0 | 4 |

* `A` 対 `A` で +2 → 最高 2
* `G` 対 `G` で対角線 2+2 → 4 が最終最大値
* 列途中で負になりそうなら 0 にリセット（局所性）

---

### まとめ

* **局所アラインメントスコア**は「2 配列のどの部分がどれほど似ているか」を定量化する指標。
* Smith‑Waterman は **DP の典型例**：行列セルを小問題，更新式を帰納関係として解く。

  * 0 リセットが「部分配列」を自然に選択
  * アフィンギャップで生物学的実用性を高める（開くコスト＞伸ばすコスト） ([iwbbio.ugr.es][3])
* DP により **計算量は二次**だが，長配列では依然ボトルネック → GPU 波面並列化が研究テーマになる。

このように，**局所アラインメント**は生物情報学の基礎問題であり，同時に「依存関係が格子状」という DP らしい構造があるため，GPU の波面並列・タイル最適化を学ぶ題材として最適なのです。

[1]: https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm?utm_source=chatgpt.com "Smith–Waterman algorithm"
[2]: https://vlab.amrita.edu/?brch=274&cnt=1&sim=1433&sub=3&utm_source=chatgpt.com "Smith-Waterman Algorithm - Local Alignment of Sequences (Theory)"
[3]: https://iwbbio.ugr.es/2014/papers/IWBBIO_2014_paper_143.pdf?utm_source=chatgpt.com "[PDF] A Smith Waterman Sequence Alignment Algorithm with Affine Gap ..."


## 3. 入力仕様

| 項目                | 内容                                                                     |
| ----------------- | ---------------------------------------------------------------------- |
| ファイル数             | 1 個（`input.txt`）                                                       |
| フォーマット            | 先頭に整数 **N**（1 ≤ N ≤ 100）<br>以下 N 行に **S<sub>i</sub> 空白 T<sub>i</sub>** |
| アルファベット           | 大文字 `A C G T` のみ                                                       |
例
```txt
3
ACGTACGT CGTAC
GATTACA  ACTGAC
AAAAGGGG TTTTCCCC
```

---

## 4. 出力仕様  

| 項目 | 内容 |
|------|------|
| 行数 | N 行 |
| 各行 | **maxScore i j**（空白区切り）<br> - *maxScore* : 最大 SW スコア (int)<br> - *(i, j)* : そのスコアを取る行列セル座標 (1‑origin) |

```
7 4 6
5 6 5
0 0 0
```

> 行列 *H* で最高スコアが 0（類似性なし）の場合は `0 0 0` とする。

---

## 5. スコアパラメータ  

| パラメータ | 値 |
|------------|----|
| match      | +2 |
| mismatch   | −1 |
| gap_open   | −2 |
| gap_extend | −1 |

*Affine gap* モデル：  

```
H[i][j] = max(
0,
H[i-1][j-1] + (S[i]==T[j] ? +2 : -1),
max_k { H[i-k][j]   -2 -1*(k-1) },
max_k { H[i][j-k]   -2 -1*(k-1) }
)
```

---

## 6. 評価ルール  

| 項目 | 条件 |
|------|------|
| ハード | NVIDIA GPU（Turing 以上，例：RTX 3070）<br>Driver & CUDA 12.x |
| コンパイル | `nvcc -O3 -Xcompiler -O3 -arch=sm_86`（例） |
| 時間計測 | スコア計算カーネルのみ（`cudaEvent` で計測）。データ転送は除外 |
| ベンチ | 付属 `benchmark.py` で自動実行し<br> - **t_CPU** : 提供 CPU 実装<br> - **t_GPU** : あなたの実装<br> speed‑up `S = t_CPU / t_GPU` |
| 合格 | **S ≥ 10** かつ 全ペアで出力が一致 |

---

## 7. 納品ファイル

| ファイル | 役割 |
|----------|------|
| `sw_cpu.cpp` | 単純 2‑D DP（single‑thread） |
| `sw_gpu.cu`  | あなたの CUDA 解 |
| `run.sh` | ビルド & 実行（`./run.sh input.txt` で OK） |
| `benchmark.py` | 乱数入力生成 & 計測ツール |
| `README.md` | アルゴリズム概要・最適化ポイント・計測環境 |

---

## 8. 実装目安

| レベル | キー技法 | 期待 Speed‑up |
|--------|----------|---------------|
| ★☆☆ | 反対対角線ごとにループ、各セル 1 thread | 2–3× |
| ★★☆ | 32×32 タイル＋`__shared__` で境界共有 | 6–10× |
| ★★★ | Wavefront Alignment (WFA) など差分法 | 15× 以上 |

---


