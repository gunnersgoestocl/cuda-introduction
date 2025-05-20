# Parallelism in DNN Feedforward

DNN のフィードフォワードにおける並列化の手法の一つ `layer parallel` を実装してみる

DNN の構成は単純で、

```txt
input -> matmul -> activation(ReLU) -> matmul -> Softmax -> output
```

parallel に実行する layer の切り分けは、

```txt

```

とする。

入力の次元とバッチサイズがどのくらいになると layer parallel の効果が出てくるのかを調べる。

