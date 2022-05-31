## 生成对抗网络

2014 年，加拿大蒙特利尔大学的 Ian Goodfellow 和他的导师 Yoshua Bengio 提出生成对抗网络（Generative Adversarial Networks, GANs）。在 GANs 被提出来之后，发展迅速，出现了各种变种网络，包括 WGAN、InfoGAN、f-GANs、BiGAN、DCGAN、IRGAN 等。

对于 GANs 的理解，可以想象成假币者与警察间展开的一场猫捉老鼠游戏，造假币者试图造出以假乱真的假币，警察试图发现这些假币，对抗使得二者的水平都得到提高。

GANs 包括**生成器（Generator）**和**判别器（Discriminator）**两个部分。

（1）生成器的作用是合成「假」样本。它从先验分布中采样随机信号，通过神经网络得到模拟样本。

（2）判别器的作用是判断输入的样本是真实的还是合成的。它同时接收来自生成器的模拟样本和实际数据集的真实样本，并且判断当前接收的样本是「真」还是「假」。

GANs 实际上是一个二分类问题，判别器 $D$ 试图识别实际数据为真实样本，识别生成器生成的数据为模拟样本。它的损失函数写成**负对数似然 （Negative Log-Likelihood）**，也称为 Categorical Cross-Entropy Loss，即：

$\mathcal{L}(D) = -\int p(x) \left[ p(data \mid x) \log D(x) + p(g \mid x) \log(1-D(x))  \right]\,\text{d}x,\qquad \text{(1)}$

其中 $D(x)$ 表示判别器预测 $x$ 为真实样本的概率，$p(data \mid x)$ 和 $p(g \mid x)$ 表示 $x$ 分属真实数据集和生成器这两类的概率。即理解为，在给定样本 $x$ 的条件下，该样本来自真实数据集 $data$ 的概率和来自生成器的概率。

样本 $x$ 的来源应该各占实际数据集和生成器一半，即 $p_{\text{src}}(data)=p_{\text{src}}(g)= 0.5$。用 $p_{\text{data}}(x)\doteq p(x\mid data)$ 表示从实际数据集得到 $x$ 的概率，$p_{\text{g}}(x)\doteq p(x\mid g)$ 表示从生成器得到 $x$ 的概率，有 $x$ 的总概率：

$p(x) = p_{\text{src}}(data)p(x\mid data) + p_{\text{src}}(g)p(x\mid g).$

> 注：$\doteq$ 和 $\approx$ 是等价的，都是表达约等于的意思。一般写完等号之后，发现不是等于，而是约等于，所以就懒得涂抹写成 $\approx$，所以就添加一个点。

将损失函数 (1) 式中的 $p(x)p(data\mid x)$ 替换为 $p_{\text{src}}(data)p_{\text{data}}(x)$，以及将 $p(x)p(g\mid x)$ 替换为 $p_{\text{src}}(g)p_{\text{g}}(x)$，就可以得到最终的目标函数

$\mathcal{D}=-\frac{1}{2}\left( \mathbb{E}_{x\sim p_{\text{data}}(x)}\left[ \log D(x) \right] + \mathbb{E}_{x\sim p_{\text{g}}(x)}\left[ \log (1 - D(x)) \right]\right),$

在此基础上可以得到值函数

$V(G,D) = \mathbb{E}_{x\sim p_{\text{data}}(x)}\left[ \log D(x) \right] + \mathbb{E}_{x\sim p_{\text{g}}(x)}\left[ \log (1 - D(x)) \right].$

在训练的时候，判别器 $D$ 的目标就是最大化上述值函数，生成器 $G$ 的目标就是最小化它，因此整个 MinMax 问题可以表示为 $\underset{G}{\min}\underset{D}{\max} V(G,D)$。

### GANs 的训练方式

我们知道 GANs 的值函数为

$V(G,D) = \mathbb{E}_{x\sim p_{\text{data}}(x)}\left[ \log D(x) \right] + \mathbb{E}_{x\sim p_{\text{g}}(x)}\left[ \log (1 - D(x)) \right].$

在训练的时候，判别器 $D$ 的目标就是最大化上述值函数，生成器 $G$ 的目标就是最小化它，因此整个 MinMax 问题可以表示为 $\underset{G}{\min}\underset{D}{\max} V(G,D)$。

GANs 在训练的时候是采用生成器和判别器交替优化的方式进行的。

**判别器 $D$ 的训练**：

（1）先固定生成器 $G(\cdot)$；

（2）利用生成器随机模拟产生样本 $G(z)$ 作为负样本（$z$ 是一个随机向量），并从真实数据集中采样获得正样本 $X$；

（3）将正负样本输入到判别器 $D(\cdot)$ 中，根据判别器的输出 $D(X)$ 和 $D(G(z))$ 和样本标签来计算误差；

（4）最后利用误差反向传播算法来更新判别器 $D(\cdot)$ 的参数；

判别器的训练是这样的一个问题：给定生成器 $G$，寻找当前情况下的最优判别器 $D^*_G$ 。对于单个样本 $x$，最大化 $\underset{D}{\max} p_{\text{data}}(x)\log D(x) + p_{\text{g}}(x)\log(1-D(x))$ 的解为 $\hat{D}(x)=p_{\text{data}}(x)/[p_{\text{data}}(x)+p_{\text{g}}(x)]$，外面套上对 $x$ 的积分就得到 $\underset{D}{\max} V(G,D)$，解由单点变成一个函数解：

$D^*_G=\frac{p_{\text{data}}}{p_{\text{data}}+p_{\text{g}}}.$

此时 $\underset{G}{\min}V(G,D^*_G)=\underset{G}{\min}\left\{-\log 4 + 2\cdot \text{JSD}(p_{\text{data}}\| p_{\text{g}})\right\}$，其中 $\text{JSD}(\cdot)$ 是 JS 距离。

优化生成器 G 实际上是在最小化生成样本分布与真实样本分布的 JS 距离。最终达到的均衡点是 $\text{JSD}(p_{\text{data}}\| p_{\text{g}})$ 的最小值点，即 $p_{\text{g}}=p_{\text{data}}$ 时，$\text{JSD}(p_{\text{data}}\| p_{\text{g}})$ 取到零，最优解 $G^*(z)=x\sim p_{\text{data}}(x)$，$D^*(x)\equiv \frac{1}{2}$，值函数 $V(G^*.D^*)=-\log 4$。

**生成器 $G$ 的训练**：

（1）先固定判别器 $D(\cdot)$；

（2）然后利用当前生成器 $G(\cdot)$ 随机模拟产生样本 $G(z)$，输入到判别器 $G(\cdot)$ 中；

（3）根据判别器的输出 $D(G(z))$ 和样本标签来计算误差；

（4）最后利用误差反向传播算法来更新生成器 $G(\cdot)$ 的参数；

假设 $G^\prime$ 表示前一步的生成器，$D$ 是 $G^\prime$ 下的最优判别器 $D^*_{G^\prime}$。那么求解最优生成器 $G$ 的过程为：

$\underset{G}{\arg\min}\,V(G,D^*_{G^\prime})=\underset{G}{\arg\min}\,\text{KL}\left( p_{\text{g}} \| \frac{p_{\text{data}}+p_{\text{g}^\prime}}{2} \right) - \text{KL}(P_{\text{g}}\| P_{\text{g}^\prime}).$

由此可以知道（1）优化 $G$ 的过程是让 $G$ 远离前一步的 $G^\prime$，同时接近分布 $(p_{\text{data}}+p_{\text{g}^\prime})/2$；（2）达到均衡点时 $p_{\text{g}^\prime}=p_{\text{data}}$，有 $\underset{G}{\arg\min}\,V(G,D^*_{G^\prime})=\underset{G}{\arg\min}\,()$，如果用这时的判别器去训练一个全新的生成器 $G_\text{new}$，理论上可能啥也训练不出来。

---

**参考**：

[1] 诸葛越，葫芦娃，《百面机器学习》，中国工信出版集团，人民邮电出版社

[2] Goodfellow I. J., Pouget-Abadie J., Mirza M., et al. Generative adversarial networks[J]. Advances in Neural Information Processing Systems, 2014, 3: 2672-2680. 
