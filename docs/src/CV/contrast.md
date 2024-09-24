## 直方图统计

### 直方图均衡化

直方图均衡化（Histogram Equalization）可以增强图像对比度，主要的思想就是通过一个映射函数把一副图像的直方图分布变成近视均匀分布来提高图像的对比度，因此关键就是如何得到映射函数。

$s_k=\sum_{j=0}^k \frac{n_j}{n}\, k=0,1,2\cdots, L-1$

其中，$s_k$ 是当前灰度级经过累积分布函数映射之后的值，$n$ 是图像中像素的总和，$n_j$ 是当前灰度级的像素个数，$L$ 是图像中的灰度级别总数。

### 基于空间熵的全局对比度增强（SEGCE）

基于空间熵的灰度全局对比度增强（Spatial Entropy-Based Global Image Contrast Enhancement）通过计算图像的空间熵，得到一个可以拉伸像素值的映射函数，从而提高图像的对比度。

假设图像大小为 $H \times W$，每个像素值为 $x(i, j)$，SEGCE 的计算步骤如下：

首先将图像 $I$ 分成 $M \times N = K$ 个区域，对于每一个子区域 $I = \{ i_1, i_2, \dots, i_k \}$ 计算灰度空间直方图：

$h_k = \{ h_k( m, n ) \mid 1 \le m \le M, 1 \le n \le N \}$

其中 $h_k(m, n)$ 是 $\dots$

$N = \lfloor \left( \frac{K}{r} \right)^{\frac{1}{2}} \rfloor, M = \lfloor \left( {K}{r} \right)^{\frac{1}{2}} \rfloor$

计算空间熵：

$S_k = -\sum_{m-1}^{M}\sum_{n-1}^{N}h_k(m,n)\log_2(h_k(m,n))$

计算离散函数 $f_k$

$f_k = \frac{S_k}{\sum_{l=1,l\neq k}^{K} S_l}$

离散函数 $f_k$ 衡量灰度 $k$ 相比其它灰度级的重要性。计算累计分布函数之前对齐进行归一化

$f_k = \frac{f_k}{\sum_{l=1}^{K}f_l}$

计算累计分布函数 $F(k)$

$F_k=\sum_{l=1}^{k}f_l$

将映射函数拉伸到 $[0, 255]$，获取映射函数

$y_k = \lfloor F_k(y_u - y_d) + y_d \rfloor$

将原图像素利用映射函数得到新的对比度增强的图像。

### 应用

问题总结：

1. 对灰度图计算映射函数，然后使用同一个函数分别对 BGR 三个通道进行计算，效果可以，缺点是会使得画面偏暗；
2. 对 BGR 三个通道分别计算映射函数，缺点是会导致偏色；