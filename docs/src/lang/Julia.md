查看 dict 里是否有某个 key：`haskey(dick, key)`

## Distributions.jl

离散分布（Discrete Distributions）的类型

按照给定分布进行采样

## CurveFit.jl

[CurveFit.jl](https://github.com/pjabardo/CurveFit.jl) 是 Julia 中实现曲线拟合的包。

### 线性最小二乘法

线性最小二乘法（Linear Least Square）常用于寻找离散数据集的近似值。给定点集 `x[i]` 和 `y[i]` 以及一系列函数 `f_i(x)`，最小二乘法通过最小化与 `y[i]` 相关的误差平方来找到对应的系数 `a[i]`，例如 `a[1]*f_1(x) + a[2]*f_2(x) + ... + a[n]*f_n(x)`.

基础功能用 QR 分解来实现：`A \ y`：`coefs = A \ y`，其中 `A[:. i] = f_i(x)`. 通常 `x` 是单个变量，如果需要多个自变量，可以使用相同的过程，类似于：`A[:, i] = f_i(x1, x2, ..., xn) `.

不同的拟合方式：

- `linear_fit(x, y)` finds coefficients `a` and `b` for `y[i] = a + b*x[i]`
- `power_fit(x, y)` finds coefficients `a` and `b` for `y[i] = a *x[i]^b`
- `log_fit(x, y)` finds coefficients `a` and `b` for `y[i] = a + b*log(x[i])`
- `exp_fit(x, y)` finds coefficients `a` and `b` for `y[i] = a*exp(b*x[i])`
- `expsum_fit(x, y, 2, withconst = true)` finds coefficients `k`, `p`, and `λ` for `y[i] = k + p[1]*exp(λ[1]*x[i]) + p[2]*exp(λ[2]*x[i])`
- `poly_fit(x, y, n)` finds coefficients `a[k]` for `y[i] = a[1] + a[2]*x[i] + a[3]*x[i]^2 + a[n+1]*x[i]^n`
- `linear_king_fit(E, U)`, find coefficients `a` and `b` for `E[i]^2 = a + b*U^0.5`
- `linear_rational_fit(x, y, p, q)` finds the coefficients for rational polynomials: `y[i] = (a[1] + a[2]*x[i] + ... + a[p+1]*x[i]^p) / (1 + a[p+1+1]*x[i] + ... + a[p+1+q]*x[i]^q)`

### 非线性最小二乘法

有时拟合函数相对于拟合系数是非线性的。 在这种情况下，给定系数的近似值，拟合函数围绕该近似值线性化，并且线性最小二乘法用于计算对近似系数的校正。 重复此迭代直到达到收敛。 拟合函数具有以下形式：`f(x_1, x_2, x_3, ..., x_n, a_1, a_2, ..., a_p) = 0`，其中 `xi` 是已知的数据点，`ai` 是要拟合的系数。

当模型公式在拟合系数上不是线性时，非线性算法是必要的。 这个库实现了一个不明确需要导数的牛顿型算法。 这是在函数中实现的：`coefs, converged, iter = nonlinear_fit(x, fun, a0, eps=1e-7, maxiter=200)`

在这个函数中，`x` 是一个数组，其中每一列代表数据集的一个不同变量，`fun` 是可调用的（callable），它返回拟合误差，要求可以使用以下签方式调用：`residual = fun(x, a)`，其中 `x` 是一个表示一行参数数组 `x` 的向量，`a` 是拟合系数的估计值，这些系数都不能为零（以提供比例）。`eps`  和 `maxiter` 是收敛参数。

`nonlinear_fit` 函数用于实现一下拟合函数：

- `king_fit(E, U)` find coefficients `a`, `b` and `n` for `E[i]^2 = a + b*U^n`
- `rational_fit` Just like `linear_rational_fit` but tries to improve the results using nonlinear least squares (`nonlinear_fit`)

### 通用接口

CurveFit.jl 开发了方便使用不同曲线拟合的通用接口：`fit = curve_fit(::Type{T}, x, y...)`，其中 `T` 是 curve fitting type，The following cases are implemented:

- `curve_fit(LinearFit, x, y)`
- `curve_fit(LogFit, x, y)`
- `curve_fit(PowerFit, x, y)`
- `curve_fit(ExpFit, x, y)`
- `curve_fit(Polynomial, x, y, n=1)`
- `curve_fit(LinearKingFit, E, U)`
- `curve_fit(KingFit, E, U)`
- `curve_fit(RationalPoly, x, y, p, q)`

`curve_fit` 通用函数返回一个对象，该对象可用于使用 `apply_fit` 计算模型的估计值。 `call` 被重载，以便对象可以用作函数。

### 用例

```julia
using PyPlot
using CurveFit

x = 0.0:0.02:2.0
y0 = @. 1 + x + x*x + randn()/10
fit = curve_fit(Polynomial, x, y0, 2)
y0b = fit.(x) 
plot(x, y0, "o", x, y0b, "r-", linewidth=3)
```



