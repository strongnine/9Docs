## 关联规则挖掘

Apriori 算法是在「购物篮分析」中常用的关联规则挖掘算法，在实际工作中需要对数据集扫描多次。2000 年时提出的 FP-Growth 算法只需要扫描两次数据集即可以完成关联规则的挖掘。其主要贡献就是提出了 FP 树和项头表，通过 FP 树减少了频繁项集的存储以及计算时间。Apriori 的改进算法除了 FP-Growth 算法以外，还有 CBA 算法、GSP 算法。