[TOC]



## 数学基础

### 李群与李代数

在 SLAM 中位姿是未知的，需要解决「什么样的相机位姿最符合当前观测数据」的问题。典型方法：构建成一个优化问题，求解最优的 $R, t$，使得误差最小化。但旋转矩阵自身带有约束，例如正交且行列式为 1，做为优化变量时，会引入额外的约束，使得优化变得困难。

**通过李群 — 李代数间的转换关系，可以把位姿估计变成无约束的优化问题，简化求解方式。**

#### 李群

三维旋转矩阵构成了特殊正交群 $\text{SO}(3)$：
$$
\text{SO}(3) = \{R\in \mathbb{R}^{3\times 3} \mid RR^T=I,\text{det}(R)=1\}
$$
变换矩阵构成了特殊欧式群 $\text{SE}(3)$：
$$
\text{SE}(3)=\left\{ T = \begin{bmatrix} 
 R & t \\ 
 0^T & 1 \\
\end{bmatrix} \in \mathbb{R}^{4 \times 4} \mid R \in \text{SO}(3), t\in\mathbb{R}^3 \right\}
$$
旋转矩阵和变换矩阵对加法不封闭。具体来说，对于任意两个旋转矩阵，做加法之后不再是一个旋转矩阵。
$$
R_1 + R_2 \not\in \text{SO}(3), T_1 + T_2 \not\in \text{SE}(3)
$$
而对于乘法是封闭的：
$$
R_1R_2\in \text{SO(3)}, T_1T_2 \in \text{SE}(3)
$$
群：就是对于只有一个（良好的）运算的集合的称呼。是一种集合加上一种运算的代数结构。

李群：是指具有连续（光滑）性质的群。$\text{SO}(3)$ 和 $\text{SE}(3)$ 就是在实数空间上连续的。

#### 李代数

一个向量 $a$ 对应的反对称矩阵 $A$ 有如下关系：

$$
a^\land = A = \begin{bmatrix} 
  0  & -a_3 &  a_2 \\ 
 a_3 &   0  & -a_1 \\ 
-a_2 &  a_1 &   0  \\ 
\end{bmatrix}, 
A^\lor = a
$$

对于任意时刻下的旋转矩阵有：

$$
R(t){R(t)}^T = I
$$

两边对时间求导得到：

$$
\dot{R}(t) {R(t)}^T + R(t){\dot{R}(t)}^T = 0 \\
\dot{R}(t) {R(t)}^T = -\left( \dot{R}(t) {R(t)}^T \right)^T
$$

可以看出 $\dot{R}(t) {R(t)}^T$ 是一个反对称矩阵，因此可以找到一个三维向量 $\phi(t)\in\mathbb{R}^3$ 与之对应：

$$
\dot{R}(t) {R(t)}^T = \phi(t)^\land
$$

$$
\dot{R}(t) = \phi(t)^\land R(t) = \begin{bmatrix} 
    0   & -\phi_3 &  \phi_2 \\ 
 \phi_3 &     0   & -\phi_1 \\ 
-\phi_2 &  \phi_1 &     0   \\ 
\end{bmatrix} R(t)
$$

**因此，每对旋转矩阵求一次导数，只需要左乘一个 $\phi(t)^\land$ 即可。**设 $t_0 = 0$ 时，旋转矩阵为 $R(0) = I$。把 $R(t)$ 在 $t = 0$ 附近进行一阶泰勒展开：
$$
\begin{aligned}
R(t) & \approx R(t_0) + \dot{R}(t_0)(t - t_0) \\
     & = I + t \cdot \phi(t_0)^\land
\end{aligned}
$$
$\phi$ 反映了 $R$ 的导数性质，称它在 $\text{SO}(3)$ 原点附近的正切空间（Tangent Space）。设在 $t_0$ 附近，$\phi(t_0)=\phi_0$，
$$
\dot{R}(t) = \phi(t_0)^\land R(t) = \phi_0^\land R(t)
$$
这是关于 $R$ 的微分方程，初始值 $R(0)=I$，解得 $R(t)=\exp(\phi_0^\land t)$。

1. 给定某时刻的 $R$，就能求得一个 $\phi$，它描述了 $R$ 在局部的导数关系。$\phi$ 就是对应到 $\text{SO}(3)$ 上的李代数 $\mathfrak{so}(3)$。
2. 给定某个向量 $\phi$，计算矩阵指数 $\exp(\phi^\land)$，以及给定 $R$ 时，用相反的运算来计算 $\phi$，对应的就是李群与李代数的指数/对数映射。

（可以认为 $R = \exp{(\phi^\land)}$）。

$\text{SO}(3)$ 与 $\text{SE}(3)$ 的李代数分别为：
$$
\mathfrak{so}(3) = \left\{ \phi \in \mathbb{R}^3, \Phi=\phi^\land \in \mathbb{R}^{3\times 3} \right\}
$$

$$
\mathfrak{se}(3) = \left\{ 
\xi = 
\begin{bmatrix} 
\rho \\ \phi 
\end{bmatrix} \in \mathbb{R}^6, 
\rho\in\mathbb{R}^3,
\phi\in\mathfrak{so}(3),
\xi^\land=
\begin{bmatrix} 
\phi^\land & \rho \\ 
0^T    &  0  
\end{bmatrix} \in \mathbb{R}^{4 \times 4}
\right\}
$$

需要注意 $\mathfrak{se}(3)$ 中的 $\land$ 符号不再表示反对称，但可以理解为从向量到矩阵的转换。

#### 指数与对数映射

当前的问题是：如何计算 $\exp(\phi^\land)$？

$\phi$ 是三维向量，定义它的模长和方向分别为 $\theta$ 和 $a$，于是 $\phi=\theta a$，$a$ 为长度为 1 的方向向量，$\|a \| = 1$。对于 $a$ 有如下公式
$$
a^\land a^\land = \begin{bmatrix} 
-a^2_2-a_3^2  & a_1a_2 &  a_1a_3 \\ 
 a_1a_2 & -a_1^2-a_3^2  & a_2a_3 \\ 
a_1a_3 &  a_2a_3 & -a_1^2-a_2^2  \\ 
\end{bmatrix} = aa^T - I
$$

$$
a^\land a^\land a^\land = a^\land(aa^T - I) = -a^\land
$$

因此通过一系列的推导（比较复杂，直接记公式）可以得到：
$$
\begin{aligned}
\exp(\phi^\land) & = \exp(\theta a^\land) = \sum_{n=0}^\infty \frac{1}{n!}(\theta a^\land)^n \\
& = \cos{\theta}I + (1 - \cos{\theta})aa^T + \sin{\theta}a^\land
\end{aligned}
$$
这个公式与罗德里格斯公式如出一辙，表明 $\mathfrak{so}(3)$ 实际上就是由所谓的旋转向量组成的空间。

$\mathfrak{se}(3)$ 上的指数映射形式如下：
$$
\begin{aligned}
\exp(\xi^\land) &= \begin{bmatrix}
\sum_{n=0}^\infty \frac{1}{n!}(\phi^\land)^n & \sum_{n=0}^\infty \frac{1}{(n+1)!}(\phi^\land)^n\rho \\
0^T  & 1
\end{bmatrix} \\
&\triangleq \begin{bmatrix}
\exp{(\phi^\land)} & J\rho \\
0^T  & 1
\end{bmatrix}
\end{aligned}
$$
同样通过（复杂的）推导，可以求得：
$$
J = \frac{\sin{\theta}}{\theta}I + (1 - \frac{\sin{\theta}}{\theta})aa^T + \frac{1-\cos{\theta}}{\theta}a^\land
$$

#### 公式总结

对于李群 $\text{SO}(3)$，指数映射为：
$$
\exp(\phi^\land) = \exp(\theta a^\land) = \cos{\theta}I + (1 - \cos{\theta})aa^T + \sin{\theta}a^\land
$$
对于李代数 $\mathfrak{so}(3)$，对数映射为：
$$
\theta = \arccos{\frac{\tr(R) - 1}{2}}, Ra=a
$$
对于李群 $\text{SE}(3)$，指数映射为：
$$
\exp(\xi^\land) = \begin{bmatrix}
\exp{(\phi^\land)} & J\rho \\
0^T  & 1
\end{bmatrix}
$$
其中：
$$
J = \frac{\sin{\theta}}{\theta}I + (1 - \frac{\sin{\theta}}{\theta})aa^T + \frac{1-\cos{\theta}}{\theta}a^\land
$$
对于李代数 $\mathfrak{se}(3)$，对数映射为：
$$
\theta = \arccos{\frac{\tr(R) - 1}{2}}, Ra=a, t=J\rho
$$

#### BCH 公式与近似形式



#### 李群李代数的实际应用

在 SLAM 中，需要估计一个相机的位姿，该位姿由 $\text{SO}(3)$ 上的旋转矩阵，或者 $\text{SE}(3)$ 上的变幻矩阵描述。假设某个时刻相机位姿为 $T$，相机观察到一个世界坐标位于 $p$ 的点，考虑上随机噪声 $w$，就能够得到观测值 $z$ 为：
$$
z = Tp + w
$$
理想的观测与实际数据的误差为：
$$
e = z - Tp
$$
观测到 $N$ 个这样的路标点，就会产生 $N$ 个误差值，而想要估计位姿，就相当于寻找一个最优的 $T$，使得整体误差最小：
$$
\min_{T} J(T) = \sum_{i=1}^N \|z_i - Tp_i \|_2^2
$$
求解思路有两种：

1. 用李代数表示姿态，根据李代数加法对李代数求导。（比较复杂）
2. 对李群左乘或右乘微小扰动，然后对该扰动求导，也即左扰动和右扰动模型。（有更加简单的导数计算方式）

### 非线性优化

#### 状态估计

经典 SLAM 模型由一个运动方程和一个观测方程构成：
$$
\begin{cases}
x_k = f(x_{k-1}, u_k) + w_k \\
z_{k,j} = h(y_j, x_k) + v_{k, j}
\end{cases}
$$
噪声满足高斯分布 $w_k \sim \mathcal{N}(0, R_{k}), v_k \sim \mathcal{N}(0, Q_{k,j})$，均值为 0，$R_k, Q_{k,j}$ 为协方差矩阵。其中 $x_k$ 是相机位姿，可用 $\text{SE}(3)$ 来描述。假设在 $x_k$ 处对路标 $y_j$ 进行了一次观测，对应到图像上的像素位置为 $z_{k, j}$，那么观测方程可以表示为：
$$
sz_{k,j} = K(R_k y_j + t_k)
$$
其中 $K$ 为相机内参，$s$ 为像素点的距离，也是 $R_ky_j + t_k$ 的第三个分量。

希望通过带噪声的数据 $z$ 和 $u$ 推断位姿 $x$ 和地图 $y$，以及它们的概率分布，这构成了一个状态估计问题。有两种处理方法：

1. 持有一个当前时刻的估计状态，然后用新的数据来更新，称为增量/渐进（incremental）的方法，或者叫滤波器。仅关心当前时刻的状态估计 $x_k$，对之前的状态不多考虑。
2. 把数据合并起来处理，称为批量（batch）的方法。这种方法可以在更大的范围达到最优化，是当前视觉 SLAM 的主流方法。极端情况下，可以让机器人或无人机收集所有时刻的数据，再带回计算中心统一处理，这种方式是 SfM（Structure from Motion）的主流做法。

定义所有时刻的机器人位姿和路标点坐标：
$$
x = \{x_1, \dots, x_N \}, y = \{ y_1, \dots, y_M \}
$$
所有时刻的输入为 $u$，所有时刻的观测数据为 $z$。对机器人状态的估计，从概率学的观点来看，就是已知输入数据和观测数据，求状态 $x,y$ 的条件概率分布 $P(x,y\mid z,u)$。当不知道控制输入，只有一张张的图像时，相当于估计 $P(x,y\mid z)$ 的条件概率分布，此问题也称为 SfM，即如何从许多图像中重建三维空间结构。利用贝叶斯法则：

$$
\underbrace{P(x,y\mid z,u)}_{后验} = \frac{P(z,u \mid x,y)P(x,y)}{P(z,u)}
\propto \underbrace{P(z,u\mid x,y)}_{似然} \underbrace{P(x,y)}_{先验}
$$

直接求后验分布是困难的，但是求一个状态最优估计，使得在该状态下后验概率最大化，则是可行的：
$$
(x,y)^*_{\text{MAP}} = \arg \max P(x,y\mid z,u) 
= \arg\max P(z,u\mid x,y)P(x,y)
$$
求解最大后验概率等价于最大化似然（Likehood）和先验（Prior）的乘积。如果不知道机器人位姿或路标大概在什么地方，也即没有了先验，那么可以求解最大似然估计（Maximize Likehood estimation, MLE）：
$$
(x,y)^*_{\text{MLE}} = \arg\max P(z,u\mid x,y)
$$
**直观解释：似然是指：在现在的位姿下，可能产生怎样的观测数据。最大似然估计指：在什么样的状态下，最可能产生现在观测到的数据。**

如何求最大似然估计？对于某一次观测：
$$
z_{k,j} = h(y_j, x_k) + \underbrace{v_{k,j}}_{噪声}, \quad v_k \sim \mathcal{N}(0, Q_{k,j})
$$
观测数据的条件概率依然是一个高斯分布：
$$
P(z_{j,k}\mid x_k, y_j) = \mathcal{N}(h(y_j, x_k), Q_{k,j})
$$
任意高维高斯分布 $x\sim \mathcal{N}(\mu, \Sigma)$，其概率密度函数展开形式为：
$$
P(x) = \frac{1}{\sqrt{(2\pi)^N\det(\Sigma)}}\exp\left( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)
$$
对其取负对数，变为：
$$
-\ln(P(x)) = \underbrace{\frac{1}{2}\ln\left( (2\pi)^N\det(\Sigma)\right)}_{与 x 无关，可略去}
            +\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
$$
因此，求状态的最大似然估计，只需要最小化右侧的二次型项：
$$
\begin{aligned}
(x_k,y_j)^* &= \arg\max \mathcal{N}(h(y_j,x_k), Q_{k,j}) \\
 & = \arg\min \left( (z_{k,j}-h(x_k, y_j))^T Q^{-1}_{k,j} (z_{k,j}-h(x_k, y_j)) \right)
\end{aligned}
$$
该式等价于最小化噪声项（误差）的一个二次型，这个二次型称为马哈拉诺比斯距离（Mahalanobis distance），又叫马氏距离。可以看成由信息矩阵 $Q_{k,j}^{-1}$ 加权之后的欧氏距离。

假设各个时刻的输入和观测是相互独立的，可以对联合分布进行因式分解：
$$
P(z,u\mid x,y) = \prod_k P(u_k\mid x_{k-1}, x_k)\prod_{k,j}P(z_{k,j}\mid x_k, y_j)
$$
定义各次输入和观测数据与模型之间的误差：
$$
\begin{aligned}
e_{u,k} &= x_k - f(x_{k-1}, u_k) & = w_k & \sim \mathcal{N}(0, R_{k}) \\
e_{z,j,k} &= z_{k,j} - h(x_k, y_j) & = v_{k,j} & \sim \mathcal{N}(0, Q_{k,j}) 
\end{aligned}
$$
最小化所有时刻估计值与真实读数之间的马氏距离，等价于求最大似然估计：
$$
\min J(x,y) = \sum_k e_{u,k}^T R_k^{-1}e_{u,k} 
+ \sum_k\sum_j e_{z,j,k}^T Q_{k,j}^{-1}e_{z,k,j}
$$
据此得到一个最小二乘问题（Least Square Problem），其解等价于状态的最大似然估计。
