## 逐层归一化

逐层归一化（Layer-wise Normalization）是将传统机器学习中的数据归一化应用到深度神经网络中，对神经网络中隐藏层的输入进行归一化，从而使得网络更容易训练。

### 批量归一化

批量归一化（Batch Normalization, BN）中，如果 input batch 的 shape 为 (B, C, H, W)，统计出的 mean 和 variance 的 shape 为 (1, C, 1, 1). 

