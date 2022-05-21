## 大规模分段线性模型（LS-PLM）

早在 2012 年，大规模分段线性模型（Large Scale Piece-wise Linear Model）就是阿里巴巴的主流推荐模型，又被称为混合逻辑回归（Mixed Logistics Regression），可以看作在逻辑回归的基础上采用分而治之的思路，先对样本进行分片，再在样本分片中应用逻辑回归进行 CTR（Click Through Rate，点击率）预估。

## Embedding 技术

Embedding，中文译为「嵌入」，常被翻译为「向量化」或者「向量映射」。形式上讲，Embedding 就是用一个低维稠密的向量「表示」一个对象，可以是词、商品、电影。



---

**参考**

[1] 王喆，《深度学习推荐系统》2020