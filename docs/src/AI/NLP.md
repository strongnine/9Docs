多模态数据：文本、图像、音频、视频、结构化数据

自然语言处理发展的三个阶段：

- 1950 ～ 1970 年：基于经验、规则的阶段；
- 1970 ～ 2008 年：基于统计方法的阶段；
- 2008 年至今：基于深度学习技术的阶段；



**贝叶斯模型**

### 词袋模型

**词袋模型（Bag-of-words, BOW）**：假设词与词之间是上下文独立的，即不考虑词之间的上下文关系；

- 优点：
  - 简单易用速度快；
  - 在丢失一定预测精度的前提下，很好地通过词出现的频率来表征整个语句的信息；
- 缺点：仅考虑词在一个句子中是否出现，而不考虑词本身在句子中的重要性（使用 TF-IDF 可以考虑重要性）；

对于两个语句：

```python
"We have noticed a new sign in to your Zoho account."
"We have sent back permission."
```

构造语料字典：

```python
{
    'We': 2, 'have': 2, 'noticed': 1, 'a': 1,
    'new': 1, 'sign': 1, 'in': 1, 'to': 1,
    'your': 1, 'Zoho': 1, 'account': 1, 'sent': 1,
    'back': 1, 'permission.': 1
}
# 上面两个语句生成的 BOW 特征分别为：
[1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1]
[0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
```



**TF-IDF（Term Frequency-Inverse Document Frequency）**：使用 $\text{TF}\times \text{IDF}$ 对每一个出现的词进行加权：

$\text{TF-IDF}(t,d)=\text{TF}(t,d)\times \text{IDF}(t)$

其中 $\text{TF}(t,d)$ 为单词 $t$ 在文档 $d$ 中出现的频率，$\text{IDF}(t)$ 是逆文档频率，用来衡量单词 $t$ 对表达语义所起的重要性，表示为：

$\text{IDF}(t) = \log{\frac{\text{Num. of articles}}{\text{Num. of articles containing word }t+1}}$

直观解释为，如果一个单词在非常多的文章里面都出现，那么它可能是一个比较通用的词汇，对于区分谋篇文章特殊语义的贡献比较小，因此对权重做一定惩罚。

- 优点：简单易用速度快；
- 缺点：文本语料稀少、字典大小大于文本语料大小时，容易发生过拟合；



**N-Gram 语言模型**：假设有一个句子 $S(w_1,w_2,w_3,\cdots,w_n)$，其中 $w_i$ 代表句子中的词，那么这个句子的出现概率就是所以单词出现概率的乘积 $p(S)=p(w_1)\times p(w_2)\times p(w_3)\times\cdots\times p(w_n)$. 在此基础上加上马尔科夫假设，即当前词的出现之和前 $n$ 个词有关，则有：

$p(S) = p(w_1)\times p(w_2\mid w_1)\times\cdots \times p(w_n\mid w_{n-1}).$

N-Gram 模型可以与 BOW、TF-IDF 模型相结合，构建 Bi-Gram、Tri-Gram 等生成额外的稀疏特征向量，构建出来的特征比使用 Uni-Gram 的 BOW、TF-IDF 特征更具有表征能力。

词袋模型的问题：如果近义词出现在不同文本中，那么在计算这一类文本的相似度或者进行预测时，如果训练数据不含大量标注，就会出现无法识别拥有相似上下文语义词的情况；



### 词嵌入模型

**Word2Vec**：常用的模型训练方式为 CBOW 和 Skip-Gram 两种算法。

**glove**：

**fastText**：

> 针对中文词向量的预训练，有腾讯公开的 AI Lab 词向量。

### 深度学习

**TextCNN**：模型结构简单，训练和预测速度快，同时拥有比传统模型更高的精度。采用多尺度卷积来模拟 N-Gram 模型在文本上的操作，最终合并之后进行呢预测。适合短文本以及有明显端与结构的语料。

**DPCNN**：从 ResNet 结构中借鉴了残差块（residual block）的理念，模拟 CV 任务中对于图像特征进行逐层提取的操作。相比较于 TextCNN 能够在文本结构复杂、语义丰富或者上下文依赖性强的文本上有更好的表现。

**LSTM 类模型**：包括典型双向循环神经网络结构 LSTM、Bi-LSTM、Bi-GRU + Attention，LSTM 和 GRU 层具有非常好的时序拟合能力。Attention 机制对不同时间的状态值进行加权，能够进一步提升模型的预测能力，适合具有复杂语义上下文的文本。

**Attention 机制**：从原理上分析，是一种对词在句子中的不同状态进行加权的操作，从最原始的加权平均，逐步发展到 Self-Attention。通过使用词的相似度矩阵进行计算，调整词在句子中对应的权重，从而允许将词的加权求和作为输出或者下一层的输入。

现有的上下文相关的预训练模型包括：ELMo、GPT、BERT、BERT-wwm、ERNIE\_1.0、XLNet、ERNIE\_2.0、RoBERTa、ALBERT、ELECTRA

**ELMo**：是一个采用自回归语言模型方式训练的语言模型。自回归语言模型的本质是通过输入的文本序列，对下一个词进行预测，通过不断优化预测的准确率，使模型逐步学到上下文的语义关系。ELMo 的结构包括正向 LSTM 层和反向 LSTM 层，通过分别优化正向下一次词和反向下一个词达到更好的预测效果。

**GPT**：将 Multi-Head Attention 和 Transformer 结构应用到了语言模型的预训练上，采用正向 Transformer 结构，去除了其中的解码器，同 ELMo 模型一样，采用自回归语言模型的方式进行训练。

**BERT**：使用自编码器模式进行训练，模型结构中包含正向和反向 Transformer 结构。为了减少由双向 Transformer 结构和自编码器造成的信息溢出影响，BERT 在训练中引入了 MLM，防止 BERT 模型因双向 Self-Attention 而导致的过拟合。

**MLM**（Masked Language Model，遮蔽语言模型）：预训练中 15% 的词条（token）会被遮蔽，对于这 15% 的词条，有 80% 的概率会使用 [MASK] 替换，10% 的概率随机替换，10% 的概率保持原样，这个替换策略在模型训练中起到正则作用。

**RoBERTa**：是 Facebook 提出的模型，在 BERT 的基础上移除了 NSP（Next Sentence Prediction）机制，并且修改了 MLM 机制，调整了其参数。

**ERNIE**：是百度提出的模型，在 BERT 的基础上优化了对中文数据的预训练，增加了三个层次的预训练：Basic-Level Masking（第一层）、Phrase-Level Masking（第二层）、Entity-Level Masking（第三层），分别从字、短语、实体三个层次上加入先验知识，提高模型在中文语料上的预测能力。

**RoBERTa-wwm**：由哈工大讯飞联合实验室发布，并不是一个严格意义上的新模型，wwm（whole word mask）是一种训练策略。BERT 所用的 MLM 具有一定的随机性，会将原始词的 word pieces 遮蔽掉，而 wwm 策略中，相同词所属的 word pieces 被遮蔽之后，其同属其他部分也会一同被遮蔽，在保证词语完整性的同时又不影响其独立性。

卷积神经网络

循环神经网络



自注意力机制（Self-Attention ）

Transformer

