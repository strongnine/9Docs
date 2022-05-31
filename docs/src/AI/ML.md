## 验证方法

Holdout 检验

交叉检验

**自助法（Bootstrap）**：有放回地从 N 个样本中抽样 n 个样本。当样本规模比较小的时候，将样本集进行划分会让训练集进一步减小，这可能会影响模型训练效果。自助法是基于自助采样的检验方法。在 n 次采样过程中，有的样本会被重复采样，有的样本没有被抽出过，将这些没有被抽出的样本作为验证集，进行模型验证，这就是自助法的验证过程。

**交并比（Intersection over Union，IoU）**：交并比 IoU 衡量的是两个区域的重叠程度，是两个区域的交集比上并集。在目标检测任务重，如果模型输出的矩形框与人工标注的矩形框 IoU 值大于某个阈值（通常为 0.5）时，即认为模型输出正确。

**精准率与召回率（Precision & Recall）**：在目标检测中，假设有一组图片，Precision 代表我们模型检测出来的目标有多少是真正的目标物体，Recall 就是所有真实的目标有多少比例被模型检测出来了。目标检测中的真正例（True Positive）、真负例（True Negative）、假正例（False Positive）、假负例（False Positive）的定义如下：

|              | 实际为正 | 实际为负 |
| :----------: | :------: | :------: |
| **预测为正** |    TP    |    FP    |
| **预测为负** |    FN    |    TN    |

对于这四个指标可以这样去理解，后面的 Positive 和 Negative 以预测的结果为主，因为我们关注的是模型的预测，如果模型的预测与实际的标注不一样，那么这个预测就是「假的」，比如预测为负那么就称为 Negative，但是实际为正，与预测的不一样，那么就是「假的」False，因此这个预测就是 False Negative，这是一个「假的正例」是「错误的正例」。

精准率，就是在预测为正样本中实际为正样本的概率，也就是所有的 Positive 中 True Positive 的概率

$Precision = \frac{TP}{TP+FP},$

召回率，就是在实际为正样本中预测为正样本的概率，就是所有的实际标注为正样本的（TP + FN）预测为正样本的概率（TP）

$Recall = \frac{TP}{TP+FN},$

准确率，就是模型预测正确的（所有的 True：TP + TN）占全部的比例

$Accuracy = \frac{TP+TN}{TP+TN+FP+FN},$

**平均精度（Average precision，AP）**：是主流的目标检测模型评价指标，它的意思是不同召回率上的平均精度。我们希望训练好的模型 Precision 和 Recall 都越高越好，但是这两者之间有个矛盾，当 Recall 很小的时候 Precision 可能会很高，当 Recall 很大的时候，Precision 可能会很低。我们将不同 Recall 对应的 Precision 做一个曲线（PR 曲线），然后在这个曲线上计算 Precision 的均值。

**曲线下面积（Area Under Curve，AUC）**：

## 优化算法

### 损失函数总结

为了刻画模型输出与样本标签的匹配程度，定义损失函数 $L(\cdot,\cdot):Y\times Y\rightarrow \mathbb{R}_{\ge 0}$，$L(f(x_i,\theta),y_i)$ 越小，表明模型在该样本点匹配得越好。

> 为了具有更加简介的表达，将网络的输出表示为 $f$，而实际标签表达为 $y$。

**$0-1$ 损失函数**：最常用于二分类问题，$Y=\{1,-1\}$，我们希望 $\texttt{sign}\, f(x_i,\theta)=y_i$，所以 $0-1$ 损失函数为

$L_{0-1}(f,y) = 1_{fy\le 0},$

其中 $1_{P}$ 是指示函数（Indicator Function），当且仅当 $P$ 为真时取值为 1，否则取值为 0。$0-1$ 损失的优点是可以直观地刻画分类的错误率，缺点是由于其非凸、非光滑的特点，算法很难对该函数进行优化。

**Hinge 损失函数**：是 $0-1$ 损失函数相对紧的凸上界，且当 $fy\ge 1$ 时，函数不对其做任何惩罚。它在 $fy=1$ 处不可导，不能够用梯度下降法进行优化，而是用次梯度下降法（Subgradient Descent Method）。适用于 Maximum-Margin 分类，主要用于支持向量机（SVM）中，用来解间距最大化的问题。

$L_{\text{hinge}}(f,y)=\max\{0,1-fy\}.$

**感知损失函数（Perceptron Loss）**：是 Hinge 损失函数的一个变种。Hinge 对判定边界附近的点（正确端）惩罚力度很高，但是 Perceptron 只要样本的判定类别正确就行，不管其判定边界的距离。它比 Hinge 更加简单，不是 Max-margin Boundary，所以模型的泛化能力没有 Hinge 强。

$L_{\text{Perceptron}}=\max(0, -f).$

**Logistic 损失函数**：是 $0-1$ 损失函数的凸上界，该函数处处光滑，对所有的样本点都有所惩罚，因此对异常值相对更敏感一点。

$L_{\text{logistic}}(f,y)=\log_2(1+\exp(-fy)).$

**Log 对数损失函数**：即对数似然损失（Log-likelihood Loss），它的标准形式

$L_{\text{log}}(f(\boldsymbol{x};\theta),y)=-\log f_y(\boldsymbol{x};\theta),$

其中 $f_y(\boldsymbol{x};\theta)$ 可以看作真实类别 $y$ 的似然函数。

**交叉熵（Cross Entropy）损失函数**：对于两个概率分布，一般可以用交叉熵去衡量它们的差异。标签的真实分布 $\boldsymbol{y}$ 和模型预测分布 $f(\boldsymbol{x};\theta)$ 之间的交叉熵为

$\mathcal{L}(f(\boldsymbol{x};\theta),\boldsymbol{y})=-\boldsymbol{y}^\top\log f(\boldsymbol{x};\theta)=-\sum_{c=1}^Cy_c\log f_c(\boldsymbol{x};\theta).$

因为 $\boldsymbol{y}$ 为 one-hot 向量，因此交叉熵可以写为

$\mathcal{L}(f(\boldsymbol{x};\theta),\boldsymbol{y})=-\log f_y(\boldsymbol{x};\theta),$

其中 $f(\boldsymbol{x};\theta)$ 可以看作真实类别 $y$ 的似然函数。因此交叉熵损失函数也就是**负对数似然函数（Negative Log-Likelihood）**。

**平方损失（Mean Squared Error）函数**：在回归问题中最常用的损失函数。对于 $Y=\mathbb{R}$，我们希望 $f(x_i,\theta)\approx y_i$

$L_{\text{MSE}}(f,y)=(f-y)^2.$

**绝对损失（Mean Absolute Error）函数**：当预测值距离真实值较远的时候，平方损失函数的惩罚力度大，也就是说它对于异常点比较敏感。如果说平方损失函数是在做均值回归的话，那么绝对损失函数就是在做中值回归，对于异常点更加鲁棒一点。只不过绝对损失函数在 $f=y$ 处无法求导。

$L_{\text{MAE}}(f,y)=|f-y|.$

**Huber 损失函数**：也称为 Smooth L1 Loss， 综合考虑可导性和对异常点的鲁棒性。在 $|f-y|$ 较小的时候为平方损失，比较大的时候为线性损失

$L_{\text{Huber}}(f,y)=\begin{cases}(f-y)^2,\qquad |f-y|\le \delta\\ 2\delta|f-y|-\delta^2,\quad|f-y|>\delta\end{cases}$



### 随机梯度算法

随机梯度下降法本质上是采用迭代方式更新参数，每次迭代在当前位置的基础上，沿着某一方向迈一小步抵达下一位置，不断地重复这个步骤，它的更新公式为

$\theta_{t+1}=\theta_{t} - \eta g_t,$

其中 $\eta$ 是学习率。

**动量（Momentum）方法：**类比中学物理知识，当前梯度就好比当前时刻受力产生的加速度，前一次步长 $v_{t-1}$ 好比前一时刻的速度，当前步长 $v_t$ 好比当前加速度共同作用的结果。这就好比小球有了惯性，而刻画惯性的物理量是动量。模型参数的迭代公式为：

$v_t = \gamma v_{t-1} + \eta g_t,$

$\theta_{t+1} = \theta_t - v_t,$

在这里当前更新步长 $v_t$ 直接依赖于前一次步长 $v_{t-1}$ 和当前梯度 $g_t$，衰减系数 $\gamma$ 扮演了阻力的作用。

**AdaGrad 方法：**在应用中，我们希望更新频率低的参数可以拥有较大的更新步幅，而更新频率高的参数的步幅可以减小，AdaGrad 方法采用「历史梯度平方和」来衡量不同参数的梯度的稀疏性，取值越小表明越稀疏。AdaGrad 借鉴了 $\mathscr{l}_2$ 正则化的思想，每次迭代时自适应地调整每个参数的学习率。这样的方式保证了不同的参数有具有自适应学习率。具体的更新公式表示为：

在第 $t$ 次迭代时，先计算每个参数梯度平方的累计值

$G_t = \sum_{\tau=1}^t \boldsymbol{g}_\tau \odot \boldsymbol{g}_\tau,$

其中 $\odot$ 为按元素乘积，$\boldsymbol{g}_\tau\in \mathbb{R}^{|\theta|}$ 是第 $\tau$ 次迭代时的梯度。参数更新差值为

$\Delta\theta_t=-\frac{\eta}{\sqrt{G_t+\epsilon}}\odot\boldsymbol{g}_t,$

其中 $\alpha$ 是初始学习率，$\epsilon$ 是为了保持数值稳定性而设定的非常小的常数，一般取值为 $e^{-7}\sim e^{-10}$。分母中求和的形式实现了退火过程，意味着随着时间推移，学习速率 $\frac{\eta}{\sqrt{G_t+\epsilon}}$ 越来越小，保证算法的最终收敛。在 AdaGrad 算法中，如果某个参数的偏导数积累比较大，其学习率相对较小；相反如果其偏导数积累较小，其学习率相对较大，但整体是随着迭代次数的增加，学习率逐渐变小。

### Adam 算法

Adam 算法的全称是**自适应动量估计算法**（Adaptive Moment Estimation Algorithm），它将惯性保持和自适应两个优点结合，可以看作是动量法和 RMSprop 算法（或者 AdaGrad 算法）的结合。

它一方面记录梯度的一阶矩（First Moment）$M_t$，即过往梯度与当前梯度的平均，理解为「惯性」，是梯度 $\boldsymbol{g}_t$ 的指数加权平均。

另一方面记录梯度的二阶矩（Second Moment）$G_t$，即过往梯度平方与当前梯度平方的平均，理解为「自适应部分」，是梯度 $\boldsymbol{g}_t^2$ 的指数加权平均。

> 一阶矩可以理解为均值；二阶矩可以理解为未减去均值的方差

$M_t = \beta_1 M_{t-1} + (1 - \beta_1)\boldsymbol{g}_t,$

$G_t = \beta_2 G_{t-1} + (1 - \beta_2)\boldsymbol{g}_t\odot\boldsymbol{g}_t,$

其中 $\beta_1$ 和 $\beta_2$ 分别为两个移动平均的衰减率，通常取值为 $\beta_1 = 0.9$, $\beta_2 = 0.99$. 

Adam 算法考虑了 $M_t, G_t$ 在零初始情况下的偏置矫正。假设 $M_0=0, G_0=0$，那么在迭代初期 $M_t$ 和 $G_t$ 的值会比真实的均值和方差要小，特别是当 $\beta_1$ 和 $\beta_2$ 都接近于 1 时，偏差会很大。具体来说，Adam 算法的更新公式为：

$\hat{M}_t = \frac{M_t}{1 - \beta_1^t},$

$\hat{G}_t = \frac{G_t}{1 - \beta_2^t},$

$\Delta\theta_t = -\frac{\alpha}{\sqrt{\hat{G}_t + \epsilon}} \hat{M}_t,$

其中学习率 $\alpha$ 通常设为 0.001，并且也可以进行衰减，比如 $\alpha_t=\alpha_0/\sqrt{t}$. 

Adam 算法的物理意义：

> 《百面机器学习》163 页

### 逐层归一化

逐层归一化（Layer Normalization）是将传统机器学习中的数据归一化方法应用到深度神经网络中，对神经网络中隐藏的输入进行归一化，使得网络更容易训练。常用的逐层归一化方法有：批量归一化、层归一化、权重归一化和局部响应归一化。

**内部协变量偏移（Internal Covariate Shift）**：当使用随机梯度下降来训练网络时，每次参数更新都会导致该神经层的输入分布发生改变，越高的层，其输入分布会改变得越明显。从机器学习角度来看，如果一个神经层的输入分布发生了改变，那么其参数需要重新学习。

逐层归一化的能够提高训练效率的原因：

（1）**更好的尺度不变性**：把每个神经层的输入分布都归一化为标准正态分布，可以使得每个神经层对其输入具有更好的尺度不变性。不论低层的参数如何变化，高层的输入保持相对稳定。另外，尺度不变性可以使得我们更加高效地进行参数初始化以及超参选择。

（2）**更平滑的优化地形**：逐层归一化一方面可以使得大部分神经层的输入处于不饱和区域，从而让梯度变大，避免梯度消失问题；另一方面还可以使得神经网络的优化地形（Optimization Landscape）更加平滑，以及使梯度变得更加稳定，从而允许我们使用更大的学习率，并提高收敛速度。

**批量归一化（Batch Normalization，BN）方法** 是一种有效的逐层归一化方法，可以对神经网络中任意的中间层进行归一化操作。假设神经网络第 $l$ 层的净输入为 $\boldsymbol{z}^{(l)}$，神经元输出为 $\boldsymbol{a}^{(l)}$，即

$\boldsymbol{a}^{(l)} = f(\boldsymbol{z}^{(l)})=f\left( \boldsymbol{W}\boldsymbol{a}^{(l)} + \boldsymbol{b} \right),$

其中 $f(\cdot)$ 是激活函数，$\boldsymbol{W}, \boldsymbol{b}$ 是神经网络的参数。为了提高优化效率，就要使得净输入 $\boldsymbol{z}^{(l)}$ 的分布一致，比如都归一化到标准正态分布。归一化操作一般应用在仿射变换（Affine Transformation）$\boldsymbol{W}\boldsymbol{a}^{(l)}+\boldsymbol{b}$ 之后，激活函数之前。

为了提高归一化效率，一般使用标准化将净输入 $\boldsymbol{z}^{(l)}$ 的每一维都归一化到标准正态分布

$\hat{\boldsymbol{z}}^{(l)} = \frac{\boldsymbol{z}^{(l)}-\mathbb{E}[\boldsymbol{z}^{(l)}]}{\sqrt{\text{var}(\boldsymbol{z}^{(l)})+\epsilon}}，$

其中 $\mathbb{E}[\boldsymbol{z}^{(l)}]$ 和 $\text{var}(\boldsymbol{z}^{(l)})$ 是当前参数下 $\boldsymbol{z}^{(l)}$ 的每一维在整个训练集上的期望和方差。

## 集成学习

### Boosting 与 Bagging

机器学习问题的两种策略：一种是研发人员尝试各种模型，选择其中表现最好的模型，做重点调参优化；另一种是将多个分类器的结果统一成一个最终的决策，其中每个单独的分类器称为**基分类器**，使用这类策略的机器学习方法统称为**集成学习**。

集成学习分为 Boosting 和 Bagging 两种。**Boosting 方法**训练基分类器时采用串行方式，各个基分类器之间有依赖。它的基本思路是将基分类器层层叠加，每一层在训练的时候，对前一层基分类器分错的样本，给予更高的权重。测试时，根据各层分类器的结果的加权得到最终结果。**Bagging** 与 Boosting 的串行训练方式不同，Bagging 方法在训练过程中，各基分类器之间无强依赖，可以进行并行训练。最著名的算法之一就是基于决策树基分类器的**随机森林（Random Forest）**。Bagging 方法更像是一个集体决策的过程，每个个体都进行单独学习，在最终做决策时，每个个体单独做出判断，再通过投票的方式做出最后的集体决策。

**基分类器**，有时候又被称为弱分类器。基分类器的错误，是偏差和方差两种错误之和。偏差主要是由于分类器的表达能力有限导致的系统性错误，表现在训练误差不收敛，方差是由于分类器对于样本分布过于敏感，导致在训练样本数较少时，产生过拟合。而 Boosting 方法通过逐步聚焦于基分类器分错的样本，减小集成分类器的偏差。Bagging 方法则是采取分而治之的策略，通过对训练样本多次采样，并分别训练出多个不同模型，然后做综合，来减小集成分类器的方差。

最常用的基分类器是决策树：

- 决策树可以较为方便地将样本的权重整合到训练过程当中，而不需要使用过采样的方法来调整样本权重；
- 决策树的表达能力和泛化能力，可以通过调节树的层数来做折中；
- 数据样本的扰动对于决策树的影响较大，因此不同子样本集合生成的决策树基分类器随机性较大，这样的「不稳定学习期」更适合作为基分类器。（在这个点上，神经网络也因为不稳定性而适合作为基分类器，可以通过调节神经元数量、连接方式、网络层数、初始权值等方式引入随机性）；

**集成学习的基本步骤**。集成学习一般可以分为以下 3 个步骤：

（1）找到误差互相独立的基分类器；

（2）训练基分类器；

（3）合并基分类器的结果；

合并基分类器的方法有 voting 和 stacking 两种，前者对应 Bagging 方法，后者对应 Boosting 方法。

以 Adaboost 为例，其基分类器的训练和合并的基本步骤如下：

（1）确定基分类器：可以选择 ID3 决策树作为基分类器。虽然任何分类模型都可以作为基分类器，但树形模型由于结构简单且较为容易产生随机性所以比较常用。

（2）训练基分类器：假设训练集为 $\{x_i,y_i\},i=1,\dots,N$，其中 $y_i\in\{-1,1\}$，并且有 $T$ 个基分类器，则可以按照如下过程来训练基分类器：

- 初始化采样分布 $D_1(i)=1/N$；

- 令 $t=1,2,\dots,T$ 循环：

  - 从训练集中，按照 $D_t$ 分布，采样出子集 $S_i=\{x_i,y_i\},i=1,\dots,N$；

  - 用 $S_i$ 训练出基分类器 $h_t$；

  - 计算基分类器 $h_t$ 的错误率：

    $\varepsilon_t=\frac{\sum_{i=1}^{N_t}I[h_t(x_i)\neq y_i]D_i(x_i)}{N_t}$

    其中 $I[\cdot]$ 为判别函数；

  - 计算基分类器 $h_t$ 权重 $a_t=\log{\frac{(1-\varepsilon_t)}{\varepsilon_t}}$，这里可以看到错误率 $\varepsilon_t$ 越大，基分类器的权重 $a_t$ 就越小；

  - 设置下一次采样：
    
    $D_{t+1}=\begin{cases}D_t(i) \text{ or } \frac{D_t(i)(1-\varepsilon_t)}{\varepsilon_t}, \, h_t(x_i)\neq y_i;\\
    \frac{D_t(i)\varepsilon_t}{(1-\varepsilon_t)}, \, h_t(x_i)= y_i.\end{cases}$

（3）合并基分类器：给定一个未知样本 $z$，输出分类结果为加权投票的结果 $\text{Sign}(\sum_{t=1}^Th_t(z)a_t)$.

### 梯度提升决策树（GBDT）

### XGBoost

XGBoost 是陈天奇等人开发的一个开源机器学习项目，高效地实现了 GBDT 算法并进行了算法和工程上的许多改进，被广泛应用在 Kaggle 竞赛以及其他许多机器学习竞赛中。

XGBoost 本质上还是一个 GBDT（Gradient Boosting Decision Tree），只是把速度和效率发挥到极致，所以前面加上了 X（代表 Extreme）。原始的 GBDT 算法基于经验损失函数的负梯度来构造新的决策树，只是在决策树构建完成后再进行剪枝。XGBoost 在决策树构建阶段就加入了正则项，即

$L_t=\sum_i l\left(y_i,\, F_{t-1}(x_i)+f_t(x_i)\right)+\Omega(f_t),$

其中 $F_{t-1}(x_i)$ 表示现有的 $t-1$ 棵树最优解，树结构的正则项定义为

$\Omega(f_t)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^Tw^2_j,$

其中 $T$ 为叶子节点个数，$w_j$ 表示第 $j$ 个叶子节点的预测值。对该损失函数在 $F_{t-1}$ 处进行二阶泰勒展开可以推导出

$L_t\approx\overset{\sim}{L_t}=\sum_{j=1}^T\left[G_jw_j+\frac{1}{2}(H_j+\lambda)w^2_j\right]+\gamma T$

从所有的树结构中寻找最优的树结构是一个 NP-hard 问题，在实际中往往采用贪心法来构建出一个次优的树结构，基本思想是根据特定的准则选取最优的分裂。不同的决策树算法采用不同的准则，如 IC3 算法采用信息增益，C4.5 算法为了克服信息增益中容易偏向取值较多的特征而采用信息增益比，CART 算法使用基尼指数和平方误差，XGBoost 也有特定的准则来选取最优分裂。

XGBoost 与 GBDT 的区别和联系：

（1）GBDT 是机器学习算法，XGBoost 是该算法的工程实现；

（2）在使用 CART 作为基分类器时，XGBoost 显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力；

（3）GBDT 在模型训练时只使用了代价函数的一阶导数信息，XGBoost 对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数；

（4）传统的 GBDT 采用 CART 作为基分类器，XGBoost 支持多种类型的基分类器，比如线性分类器；

（5）传统的 GBDT 在每轮迭代时使用全部的数据，XGBoost 则采用了与随机森林相似的策略，支持对数据进行采样；

（6）传统的 GBDT 没有设计对缺失值进行处理，XGBoost 能够自动学习出缺失值的处理策略；

**XGBoost 的并行化：**boosting 是一种串行结构，它的并行不是在 tree 粒度上的，而是在特征粒度上的并行。决策树学习最耗时的一个步骤就是对特征的值进行排序（为了确定最佳分割点）。XGBoost 训练之前，预先对数据进行排序，保存为 block 结构，后面的迭代中重复地使用这个结构，大大减小计算量。

**XGBoost 的特点：**

- 传统的 GBDT 以 CART 作为基函数，而 XGBoost 相当于有 L1/L2 正则化项的分类或者回归
- 传统的 GBDT 在优化的时候只用到一阶导数，XGBoost 对代价函数进行了二阶泰勒展开，同时用到一阶和二阶导数。并且 XGBoost 工具支持自定义代价函数，只要函数可以一阶和二阶求导；
- XGBoost 在代价函数里加入了正则项，控制模型复杂度。正则项里包含了树的叶节点个数、每个叶子节点上输出 score 的 L2 模的平方和。从 Bias-variance tradeoff 角度来讲，正则项降低了模型 variance，使学习出来的模型更加简单，防止过拟合，这也是 XGBoost 优于传统 GBDT 的一个特性。 剪枝是都有的，叶子节点输出 L2 平滑是新增的；
- shrinkage 缩减和 column subsampling。shrinkage 缩减：类似于学习速率，在每一步 tree boosting 之后增加了一个参数 n（权重），通过这种方式来减小每棵树的影响力，给后面的树提供空间去优化模型。column subsampling：列（特征）抽样，随机森林那边学习来的，防止过拟合的效果比传统的行抽样还好（行抽样功能也有），并且有利于后面提到的并行化处理算法；
- split finding algorithms（划分点查找算法），树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法 greedy algorithm 枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以 XGBoost 还提出了一种可并行的近似直方图算法（Weighted Quantile Sketch），用于高效地生成候选的分割点；
- 对缺失值的处理。对于特征的值有缺失的样本，XGBoost 可以自动学习出它的分裂方向。 稀疏感知算法 Sparsity-aware Split Finding；
- 内置交叉验证（Built-in Cross-Validation），XGBoost 可以在 boosting 过程的每次迭代中运行交叉验证，因此很容易在一次运行中获得准确的最佳 boosting 迭代次数；
- XGBoost 支持并行，提高计算速度；



---

**参考**

[1] [GitHub 项目：ML-NLP](https://github.com/NLP-LOVE/ML-NLP)；

[2] [XGBoost 特点、调参、讨论](https://blog.csdn.net/niaolianjiulin/article/details/76574216)；

[3] 诸葛越，葫芦娃，《百面机器学习》，中国工信出版集团，人民邮电出版社

