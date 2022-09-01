## Transformer 知识总结

### 原理

Transformer 整个网络结构由 Attention 机制组成。在 RNN（包括 LSTM、GRU 等）中计算是顺序的，只能从左向右或者从右向左依次计算，这种机制带来的 2 个问题：

- 时间片 $t$ 的计算依赖 $t-1$ 时刻的计算结果，限制了模型的并行能力；
- 顺序计算的过程中信息会丢失。尽管 LSTM 使用门机制的结构来缓解长期依赖的问题，但是在特别长期时依旧表现不好；

Transformer 通过以下方式来解决上面的问题：

- 使用 Attention 机制，讲序列中的任意两个位置之间的距离缩小为一个常量；
- 因为不是类似 RNN 的顺序结构，因此具有更好的并行性。也更为符合现有的 GPU 框架；

### Encoder 和 Decoder 模块

Encoder 模块将 Backbone 输出的 feature map 转换成一维表征，然后结合 positional encoding 作为 Encoder 的输入。每个 Encoder 都由 Multi-Head Self-Attention 和 FFN 组成。和 Transformer Encoder 不同的是，因为 Encoder 具有位置不变性，DETR 将 positional encoding 添加到每一个 Multi-Head Self-Attention 中，来保证目标检测的位置敏感性。

Decoder 也具有位置不变性，Decoder 的 n 个 object query（可以理解为学习不同 object 的 positional embedding）必须是不同的，以便产生不同的结果，并且同时把它们添加到每一个 Multi-Head Attention 中。n 个 object queries 通过 Decoder 转换成一个 output embedding，然后 output embedding 通过 FFN 独立解码出 n 个预测结果，包含 box 和 class。对输入 embedding 同时使用 Self-Attention 和 Encoder-Decoder Attention，模型可以利用目标的相互关系来进行全局推理。

和 Transformer Decoder 不同的是，DETR 的每个 Decoder 并行输出 n 个对象，Transformer Decoder 使用的是自回归模型，串行输出 n 个对象，每次只能预测一个输出序列的一个元素。

### 多头注意力（Multi-Head Attention）

多头注意力的提出是为了对同一 key、value、query，希望抽取不同的信息，例如短距离和长距离，类似于 CV 中的感受野（field）。

## 参考

[1] [知乎专栏：计算机视觉面试题 - Transformer 相关问题总结，作者：爱者之贻](https://zhuanlan.zhihu.com/p/554814230)
