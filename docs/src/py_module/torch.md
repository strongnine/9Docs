## 张量（Tensors）

官方文档：[Torch](https://pytorch.org/docs/1.12/torch.html#)

- 张量的构建、索引、切片、拼接、修改、随机采样
- 数学操作：逐点操作、规约操作、比较操作、频谱分析、其他操作、BLAS and LAPACK

### 张量的构建

[`tensor`](https://pytorch.org/docs/1.12/generated/torch.tensor.html#torch.tensor)：通过复制数据构造一个没有梯度历史的张量（也称为「叶张量」）。

> 叶张量：在自动微分机制（[Autograd Mechanics](https://pytorch.org/docs/1.12/notes/autograd.html)）一节可以了解 PyTorch 是如何实现自动微分的。

[`sparse_coo_tensor`](https://pytorch.org/docs/1.12/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor)：在给定索引处构造一个具有指定值的 COO (rdinate) 格式的稀疏张量。

[`asarray`](https://pytorch.org/docs/1.12/generated/torch.asarray.html#torch.asarray)：转化为张量。

[`as_tensor`](https://pytorch.org/docs/1.12/generated/torch.as_tensor.html#torch.as_tensor)：将数据转成张量，共享数据以及保留梯度历史。

[`as_strided`](https://pytorch.org/docs/1.12/generated/torch.as_strided.html#torch.as_strided)：创建具有指定大小、步幅和 `storage_offset` 的现有 torch.Tensor 输入的视图（view）。

[`from_numpy`](https://pytorch.org/docs/1.12/generated/torch.from_numpy.html#torch.from_numpy)：将 `numpy.ndarray` 转化为 `Tensor`

[`from_dlpack`](https://pytorch.org/docs/1.12/generated/torch.from_dlpack.html#torch.from_dlpack)：将外部库中的张量转换为 `torch.Tensor`

[`frombuffer`](https://pytorch.org/docs/1.12/generated/torch.frombuffer.html#torch.frombuffer)：从实现 Python 缓冲区协议的对象创建一维张量

[`zeros`](https://pytorch.org/docs/1.12/generated/torch.zeros.html#torch.zeros)、[`ones`](https://pytorch.org/docs/1.12/generated/torch.ones.html#torch.ones)：构建给定 `size` 一样的全 0 张量或者全 1 张量

[`zeros_like`](https://pytorch.org/docs/1.12/generated/torch.zeros_like.html#torch.zeros_like)、[`ones_like`](https://pytorch.org/docs/1.12/generated/torch.ones_like.html#torch.ones_like)：构建与给定张量一样 `size` 的全 0 张量或者全 1 张量

[`arange`](https://pytorch.org/docs/1.12/generated/torch.arange.html#torch.arange)：在区间 `[start, end)` 中以固定步长 `step` 构建一维张量，大小为 $\lceil (\text{end} - \text{start})/\text{step} \rceil$

[`range`](https://pytorch.org/docs/1.12/generated/torch.range.html#torch.range)：从 `start` 到 `end` 以步长 `step` 构建一维张量，大小为 $\lfloor (\text{end} - \text{start})/\text{step} \rfloor + 1$

[`linspace`](https://pytorch.org/docs/1.12/generated/torch.linspace.html#torch.linspace)：在  `start` 到 `end`  均匀分布的一维张量，大小为 `step`（包含 `end` 在内）

[`logspace`](https://pytorch.org/docs/1.12/generated/torch.logspace.html#torch.logspace)：在以 `base` 为底的对数尺度中 $\text{base}^\text{start}$ 到 $\text{base}^\text{end}$ 均匀分布的一维向量（包含 $\text{base}^\text{end}$ 在内）

[`eye`](https://pytorch.org/docs/1.12/generated/torch.eye.html#torch.eye)：单位二维向量

[`empty`](https://pytorch.org/docs/1.12/generated/torch.empty.html#torch.empty)：构建还未初始化的张量

[`empty_like`](https://pytorch.org/docs/1.12/generated/torch.empty_like.html#torch.empty_like)：构建与给定张量一样 `size` 的未初始化张量

[`empty_strided`](https://pytorch.org/docs/1.12/generated/torch.empty_strided.html#torch.empty_strided)：创建一个具有指定大小和步幅并填充未定义数据的张量

[`full`](https://pytorch.org/docs/1.12/generated/torch.full.html#torch.full)、[`full_like`](https://pytorch.org/docs/1.12/generated/torch.full_like.html#torch.full_like)：构建全为给定值 `fill_value` 的张量

[`quantize_per_tensor`](https://pytorch.org/docs/1.12/generated/torch.quantize_per_tensor.html#torch.quantize_per_tensor)：将浮点张量转换为具有给定比例和零点的量化张量

[`quantize_per_channel`](https://pytorch.org/docs/1.12/generated/torch.quantize_per_channel.html#torch.quantize_per_channel)：将浮点张量转换为具有给定比例和零点的每通道量化张量

[`dequantize`](https://pytorch.org/docs/1.12/generated/torch.dequantize.html#torch.dequantize)：通过反量化量化张量返回 `float32` 张量

[`complex`](https://pytorch.org/docs/1.12/generated/torch.complex.html#torch.complex)：构造一个复数张量，其实部等于 `real`，虚部等于 `imag`

[`polar`](https://pytorch.org/docs/1.12/generated/torch.polar.html#torch.polar)：构造一个复数张量，其元素是笛卡尔坐标，对应于绝对值 `abs` 和角度 `angle` 的极坐标

[`heaviside`](https://pytorch.org/docs/1.12/generated/torch.heaviside.html#torch.heaviside)：计算输入中每个元素的 `Heaviside` 阶跃函数

### 索引、切片、拼接、修改

[`adjoint`](https://pytorch.org/docs/1.12/generated/torch.adjoint.html#torch.adjoint)：返回张量共轭和最后两个维度转置的引用视图

[`argwhere`](https://pytorch.org/docs/1.12/generated/torch.argwhere.html#torch.argwhere)：返回输入的所有非零元素的索引张量

[`cat`](https://pytorch.org/docs/1.12/generated/torch.cat.html#torch.cat)、[`concat`](https://pytorch.org/docs/1.12/generated/torch.concat.html#torch.concat)：连接给定维度中给定的 `seq` 张量序列

[`conj`](https://pytorch.org/docs/1.12/generated/torch.conj.html#torch.conj)：返回输入张量的翻转共轭位的引用视图

[`chunk`](https://pytorch.org/docs/1.12/generated/torch.chunk.html#torch.chunk)：尝试将张量拆分为指定数量的块

[`split`](https://pytorch.org/docs/1.12/generated/torch.split.html#torch.split)：将张量拆分为块

[`tensor_split`](https://pytorch.org/docs/1.12/generated/torch.tensor_split.html#torch.tensor_split)：根据索引或由 `indices_or_sections` 指定的部分数量，将张量拆分为多个子张量，所有这些子张量都是输入的视图

[`vsplit`](https://pytorch.org/docs/1.12/generated/torch.vsplit.html#torch.vsplit)、[`hsplit`](https://pytorch.org/docs/1.12/generated/torch.hsplit.html#torch.hsplit)、[`dsplit`](https://pytorch.org/docs/1.12/generated/torch.dsplit.html#torch.dsplit)：根据 `indices_or_sections` 将输入张量（具有三个或更多维度的张量）沿着垂直、水、深度方向拆分为多个张量

[`row_stack`](https://pytorch.org/docs/1.12/generated/torch.row_stack.html#torch.row_stack)：通过在张量中垂直堆叠张量来创建一个新张量

[`column_stack`](https://pytorch.org/docs/1.12/generated/torch.column_stack.html#torch.column_stack)：通过在张量中水平堆叠张量来创建一个新张量

[`vstack`](https://pytorch.org/docs/1.12/generated/torch.vstack.html#torch.vstack)、[`hstack`](https://pytorch.org/docs/1.12/generated/torch.hstack.html#torch.hstack)、[`dstack`](https://pytorch.org/docs/1.12/generated/torch.dstack.html#torch.dstack)：沿着垂直、水平、深度方向堆叠张量

> 垂直、水平、深度三个方向对应着第一、二、三个维度

[`stack`](https://pytorch.org/docs/1.12/generated/torch.stack.html#torch.stack)：沿着给定的维度拼接张量序列

[`gather`](https://pytorch.org/docs/1.12/generated/torch.gather.html#torch.gather)：沿由 `dim` 指定的轴收集值

[`index_add`](https://pytorch.org/docs/1.12/generated/torch.index_add.html#torch.index_add)、[`index_copy`](https://pytorch.org/docs/1.12/generated/torch.index_copy.html#torch.index_copy)、[`index_reduce`](https://pytorch.org/docs/1.12/generated/torch.index_reduce.html#torch.index_reduce)：

[`index_select`](https://pytorch.org/docs/1.12/generated/torch.index_select.html#torch.index_select)：返回一个新张量，该张量使用索引中的条目沿维度 `dim` 索引输入张量，该条目是 `LongTensor`

[`masked_select`](https://pytorch.org/docs/1.12/generated/torch.masked_select.html#torch.masked_select)：返回一个新的一维张量，它根据布尔掩码掩码索引输入张量，该掩码掩码是 `BoolTensor`

[`movedim`](https://pytorch.org/docs/1.12/generated/torch.movedim.html#torch.movedim)、[`moveaxis`](https://pytorch.org/docs/1.12/generated/torch.moveaxis.html#torch.moveaxis)：将 `source` 中位置的输入维度移动到目标位置

[`narrow`](https://pytorch.org/docs/1.12/generated/torch.narrow.html#torch.narrow)：返回一个新的张量，它是输入张量的缩小版本

[`nonzero`](https://pytorch.org/docs/1.12/generated/torch.nonzero.html#torch.nonzero)：返回非零元素的索引

[`permute`](https://pytorch.org/docs/1.12/generated/torch.permute.html#torch.permute)：返回原始张量 `input` 的置换之后的视图

[`reshape`](https://pytorch.org/docs/1.12/generated/torch.reshape.html#torch.reshape)：重新定义输入张量的 `size`

[`select`](https://pytorch.org/docs/1.12/generated/torch.select.html#torch.select)：在给定索引处沿选定维度对输入张量进行切片

[`scatter`](https://pytorch.org/docs/1.12/generated/torch.scatter.html#torch.scatter)、[`torch.Tensor.scatter_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_)：

[`diagonal_scatter`](https://pytorch.org/docs/1.12/generated/torch.diagonal_scatter.html#torch.diagonal_scatter)：将 `src` 张量的值相对于 `dim1` 和 `dim2` 沿输入的对角线元素嵌入到输入张量中

[`select_scatter`](https://pytorch.org/docs/1.12/generated/torch.select_scatter.html#torch.select_scatter)：将 `src` 张量的值嵌入到给定索引处的输入中

[`slice_scatter`](https://pytorch.org/docs/1.12/generated/torch.slice_scatter.html#torch.slice_scatter)：将 `src` 张量的值嵌入到给定维度的输入中

[`scatter_add`](https://pytorch.org/docs/1.12/generated/torch.scatter_add.html#torch.scatter_add)：

[`scatter_reduce`](https://pytorch.org/docs/1.12/generated/torch.scatter_reduce.html#torch.scatter_reduce)：

[`tile`](https://pytorch.org/docs/1.12/generated/torch.tile.html#torch.tile)：通过重复输入的元素构造一个张量

[`squeeze`](https://pytorch.org/docs/1.12/generated/torch.squeeze.html#torch.squeeze)：将所有大小为 1 的维度都去掉

[`unsqueeze`](https://pytorch.org/docs/1.12/generated/torch.unsqueeze.html#torch.unsqueeze)：返回一个插入到指定位置的尺寸为 1 的新张量

[`unbind`](https://pytorch.org/docs/1.12/generated/torch.unbind.html#torch.unbind)：去除指定维度（返回的是一个保存对这个维度进行分割之后所有矩阵的元组）

[`swapaxes`](https://pytorch.org/docs/1.12/generated/torch.swapaxes.html#torch.swapaxes)、[`swapdims`](https://pytorch.org/docs/1.12/generated/torch.swapdims.html#torch.swapdims)、[`torch.transpose()`](https://pytorch.org/docs/1.12/generated/torch.transpose.html#torch.transpose)：交换 `dim0` 和 `dim1`

[`t`](https://pytorch.org/docs/1.12/generated/torch.t.html#torch.t)：要求 `input` 张量的维度必须小于等于 2，转换维度 0 和维度 1

[`take`](https://pytorch.org/docs/1.12/generated/torch.take.html#torch.take)：返回一个新的张量，其中输入的元素在给定的索引处

[`take_along_dim`](https://pytorch.org/docs/1.12/generated/torch.take_along_dim.html#torch.take_along_dim)：从沿给定暗淡的索引中选择一维索引处的输入值

[`where`](https://pytorch.org/docs/1.12/generated/torch.where.html#torch.where)：根据条件返回从 `x` 或 `y` 中选择的元素的张量

### 随机采样

[`seed`](https://pytorch.org/docs/1.12/generated/torch.seed.html#torch.seed)：随机设置一个随机种子，不需要给定参数（会返回一个 64 比特的数字代表设置的种子）

[`manual_seed	`](https://pytorch.org/docs/1.12/generated/torch.manual_seed.html#torch.manual_seed)：设定随机种子为给定的 `seed`，会返回一个 `torch.Generator`

[`initial_seed`](https://pytorch.org/docs/1.12/generated/torch.initial_seed.html#torch.initial_seed)：将生成随机数的初始种子以  Python `long` 数据类型返回

[`get_rng_state`](https://pytorch.org/docs/1.12/generated/torch.get_rng_state.html#torch.get_rng_state)：以 `torch.ByteTensor` 的数据类型返回随机数生成器

[`set_rng_state`](https://pytorch.org/docs/1.12/generated/torch.set_rng_state.html#torch.set_rng_state)：设置随机数生成器状态

**随机数生成器**

[`bernoulli`](https://pytorch.org/docs/1.12/generated/torch.bernoulli.html#torch.bernoulli)：伯努利分布，生成 0 或者 1

[`multinomial`](https://pytorch.org/docs/1.12/generated/torch.multinomial.html#torch.multinomial)：返回一个张量，其中每行包含从位于张量输入的相应行中的多项概率分布中采样的 `num_samples` 个索引

[`normal`](https://pytorch.org/docs/1.12/generated/torch.normal.html#torch.normal)：返回从给出均值和标准差的独立正态分布中抽取的随机数张量

[`poisson`](https://pytorch.org/docs/1.12/generated/torch.poisson.html#torch.poisson)：返回与输入相同大小的张量，每个元素从泊松分布中采样，速率参数由输入中的相应元素给出

[`rand`](https://pytorch.org/docs/1.12/generated/torch.rand.html#torch.rand)、[`rand_like`](https://pytorch.org/docs/1.12/generated/torch.rand_like.html#torch.rand_like)：单位均匀分布 $[0, 1)$

[`randint`](https://pytorch.org/docs/1.12/generated/torch.randint.html#torch.randint)、[`randint_like`](https://pytorch.org/docs/1.12/generated/torch.randint_like.html#torch.randint_like)：`[low, high)` 之间的随机整数

[`randn`](https://pytorch.org/docs/1.12/generated/torch.randn.html#torch.randn)、[`randn_like`](https://pytorch.org/docs/1.12/generated/torch.randn_like.html#torch.randn_like)：标准正太分布采样

[`randperm`](https://pytorch.org/docs/1.12/generated/torch.randperm.html#torch.randperm)：返回从 0 到 n - 1 的整数的随机排列

一些 In-place 的随机采样：

- [`torch.Tensor.bernoulli_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.bernoulli_.html#torch.Tensor.bernoulli_) - in-place version of [`torch.bernoulli()`](https://pytorch.org/docs/1.12/generated/torch.bernoulli.html#torch.bernoulli)
- [`torch.Tensor.cauchy_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.cauchy_.html#torch.Tensor.cauchy_) - numbers drawn from the Cauchy distribution
- [`torch.Tensor.exponential_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.exponential_.html#torch.Tensor.exponential_) - numbers drawn from the exponential distribution
- [`torch.Tensor.geometric_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.geometric_.html#torch.Tensor.geometric_) - elements drawn from the geometric distribution
- [`torch.Tensor.log_normal_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.log_normal_.html#torch.Tensor.log_normal_) - samples from the log-normal distribution
- [`torch.Tensor.normal_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.normal_.html#torch.Tensor.normal_) - in-place version of [`torch.normal()`](https://pytorch.org/docs/1.12/generated/torch.normal.html#torch.normal)
- [`torch.Tensor.random_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.random_.html#torch.Tensor.random_) - numbers sampled from the discrete uniform distribution
- [`torch.Tensor.uniform_()`](https://pytorch.org/docs/1.12/generated/torch.Tensor.uniform_.html#torch.Tensor.uniform_) - numbers sampled from the continuous uniform distribution

### 逐点操作

[`abs`](https://pytorch.org/docs/1.12/generated/torch.abs.html#torch.abs)、[`absolute`](https://pytorch.org/docs/1.12/generated/torch.absolute.html#torch.absolute)：计算输入张量 `input` 每个元素的绝对值

[`cos`](https://pytorch.org/docs/1.12/generated/torch.cos.html#torch.cos) ([`cosh`](https://pytorch.org/docs/1.12/generated/torch.cosh.html#torch.cosh))：计算每个元素的正弦、余弦、正切

[`asin`](https://pytorch.org/docs/1.12/generated/torch.asin.html#torch.asin) ([`arcsin`](https://pytorch.org/docs/1.12/generated/torch.arcsin.html#torch.arcsin))、[`acos`](https://pytorch.org/docs/1.12/generated/torch.acos.html#torch.acos) ([`arccos`](https://pytorch.org/docs/1.12/generated/torch.arccos.html#torch.arccos))、[`atan`](https://pytorch.org/docs/1.12/generated/torch.atan.html#torch.atan) ([`arctan`](https://pytorch.org/docs/1.12/generated/torch.arctan.html#torch.arctan))：计算每个元素的反正弦、反余弦、反正切

[`asinh`](https://pytorch.org/docs/1.12/generated/torch.asinh.html#torch.asinh) ([`arcsinh`](https://pytorch.org/docs/1.12/generated/torch.arcsinh.html#torch.arcsinh))、[`acosh`](https://pytorch.org/docs/1.12/generated/torch.acosh.html#torch.acosh) ([`arccosh`](https://pytorch.org/docs/1.12/generated/torch.arccosh.html#torch.arccosh))、[`atanh`](https://pytorch.org/docs/1.12/generated/torch.atanh.html#torch.atanh) ([`arctanh`](https://pytorch.org/docs/1.12/generated/torch.arctanh.html#torch.arctanh))：计算每个元素的反双曲正弦、反双曲余弦、反双曲正切

[`atan2`](https://pytorch.org/docs/1.12/generated/torch.atan2.html#torch.atan2) ([`arctan2`](https://pytorch.org/docs/1.12/generated/torch.arctan2.html#torch.arctan2))：有两个输入 `input` 和 `other`，考虑象限的 $\text{input}_i/\text{other}_i$ 的逐元素反正切

[`add`](https://pytorch.org/docs/1.12/generated/torch.add.html#torch.add)：将输入 `input` 点乘以 `alpha` 再加上 `other`

[`div`](https://pytorch.org/docs/1.12/generated/torch.div.html#torch.div) ([`divide`](https://pytorch.org/docs/1.12/generated/torch.divide.html#torch.divide))：将输入 `input` 的每个元素除以 `other` 的对应元素

[`frac`](https://pytorch.org/docs/1.12/generated/torch.frac.html#torch.frac)：计算输入中每个元素的小数部分

[`floor_divide`](https://pytorch.org/docs/1.12/generated/torch.floor_divide.html#torch.floor_divide)：按元素计算 `input` 除以 `other`，并将每个商向零的方向舍入

[`addcdiv`](https://pytorch.org/docs/1.12/generated/torch.addcdiv.html#torch.addcdiv)：将 `tensor1` 元素点除 `tensor2` 的结果乘以标量 `value` 再加到 `input` 上

[`addcmul`](https://pytorch.org/docs/1.12/generated/torch.addcmul.html#torch.addcmul)：将 `tensor1` 元素点乘 `tensor2` 的结果乘以标量 `value` 再加到 `input` 上

[`exp`](https://pytorch.org/docs/1.12/generated/torch.exp.html#torch.exp)：计算每个元素的指数

[`frexp`](https://pytorch.org/docs/1.12/generated/torch.frexp.html#torch.frexp)：将输入分解为尾数（mantissa）和指数张量，即 $\text{input}=\text{mantissa}\times 2^{\text{exponent}}$



[`exp2`](https://pytorch.org/docs/1.12/generated/torch.exp2.html#torch.exp2) ([`special.exp2()`](https://pytorch.org/docs/1.12/special.html#torch.special.exp2))

[`expm1`](https://pytorch.org/docs/1.12/generated/torch.expm1.html#torch.expm1) ([`special.expm1()`](https://pytorch.org/docs/1.12/special.html#torch.special.expm1))

[`float_power`](https://pytorch.org/docs/1.12/generated/torch.float_power.html#torch.float_power)：计算每个元素的 `exponent` 次幂

[`angle`](https://pytorch.org/docs/1.12/generated/torch.angle.html#torch.angle)：计算每个元素的角度（以弧度为单位）

[`deg2rad`](https://pytorch.org/docs/1.12/generated/torch.deg2rad.html#torch.deg2rad)：返回一个新的张量，输入的每个元素都从角度转换为弧度

**位计算**

[`bitwise_not`](https://pytorch.org/docs/1.12/generated/torch.bitwise_not.html#torch.bitwise_not)：计算 `input` 张量的按位非

[`bitwise_and`](https://pytorch.org/docs/1.12/generated/torch.bitwise_and.html#torch.bitwise_and)：计算 `input` 和 `other` 的按位与

[`bitwise_or`](https://pytorch.org/docs/1.12/generated/torch.bitwise_or.html#torch.bitwise_or)：计算 `input` 和 `other` 的按位或

[`bitwise_xor`](https://pytorch.org/docs/1.12/generated/torch.bitwise_xor.html#torch.bitwise_xor)：计算 `input` 和 `other` 的按位异或

[`bitwise_left_shift`](https://pytorch.org/docs/1.12/generated/torch.bitwise_left_shift.html#torch.bitwise_left_shift)：计算 `input` 左移 `other` 位的结果

[`bitwise_right_shift`](https://pytorch.org/docs/1.12/generated/torch.bitwise_right_shift.html#torch.bitwise_right_shift)：计算 `input` 右移 `other` 位的结果



[`ceil`](https://pytorch.org/docs/1.12/generated/torch.ceil.html#torch.ceil)：返回一个新的张量，其中输入元素的 `ceil` 是大于或等于每个元素的最小整数

[`floor`](https://pytorch.org/docs/1.12/generated/torch.floor.html#torch.floor)：返回一个新的张量，其中输入元素的 `floor`，最大整数小于或等于每个元素

[`clamp`](https://pytorch.org/docs/1.12/generated/torch.clamp.html#torch.clamp) ([`clip`](https://pytorch.org/docs/1.12/generated/torch.clip.html#torch.clip))：将 `input` 中的所有元素限制在 `[min, max]` 范围内

[`conj_physical`](https://pytorch.org/docs/1.12/generated/torch.conj_physical.html#torch.conj_physical)：计算给定输入张量的元素共轭

[`copysign`](https://pytorch.org/docs/1.12/generated/torch.copysign.html#torch.copysign)：创建一个新的浮点张量，元素为 `input` 的大小和 `other` 元素的符号



[`digamma`](https://pytorch.org/docs/1.12/generated/torch.digamma.html#torch.digamma) ([`special.digamma()`](https://pytorch.org/docs/1.12/special.html#torch.special.digamma))：

[`erf`](https://pytorch.org/docs/1.12/generated/torch.erf.html#torch.erf) ([`special.erf()`](https://pytorch.org/docs/1.12/special.html#torch.special.erf))：

[`erfc`](https://pytorch.org/docs/1.12/generated/torch.erfc.html#torch.erfc) ([`special.erfc()`](https://pytorch.org/docs/1.12/special.html#torch.special.erfc))：

[`erfinv`](https://pytorch.org/docs/1.12/generated/torch.erfinv.html#torch.erfinv) ([`special.erfinv()`](https://pytorch.org/docs/1.12/special.html#torch.special.erfinv))：

[`fake_quantize_per_channel_affine`](https://pytorch.org/docs/1.12/generated/torch.fake_quantize_per_channel_affine.html#torch.fake_quantize_per_channel_affine)：返回一个新张量，其中输入假数据中的数据使用 `scale`、`zero_point`、`quant_min` 和 `quant_max` 在指定 `axis` 的通道上按通道进行量化

[`fake_quantize_per_tensor_affine`](https://pytorch.org/docs/1.12/generated/torch.fake_quantize_per_tensor_affine.html#torch.fake_quantize_per_tensor_affine)：返回一个新张量，其中输入假数据中使用 `scale`、`zero_point`、`quant_min` 和 `quant_max` 量化

[`fix`](https://pytorch.org/docs/1.12/generated/torch.fix.html#torch.fix) ([`trunc()`](https://pytorch.org/docs/1.12/generated/torch.trunc.html#torch.trunc))：

[`fmod`](https://pytorch.org/docs/1.12/generated/torch.fmod.html#torch.fmod)：Applies C++’s [std::fmod](https://en.cppreference.com/w/cpp/numeric/math/fmod) entrywise.

[`gradient`](https://pytorch.org/docs/1.12/generated/torch.gradient.html#torch.gradient)：使用二阶精确中心差法（ [second-order accurate central differences method](https://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf)）估计函数 $g:\mathbb{R}^n\rightarrow \mathbb{R}$ 在一个或多个维度上的梯度

[`imag`](https://pytorch.org/docs/1.12/generated/torch.imag.html#torch.imag)：将张量自身虚数部分作为新张量进行返回



## 自动微分机制

[Autograd Mechanics](https://pytorch.org/docs/1.12/notes/autograd.html)

## 问题

❓`item()` 和 `detach()` 的区别是什么？

- `item()` 返回的是 Tensor 中的值，且只能返回单个值（即标量），不能返回向量；
- `detach()` 是阻断反向传播，返回值仍然为 Tensor；

