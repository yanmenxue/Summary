从下列动力学方程

$h^{\prime}(t) =\mathbf{A} h(t)+\mathbf{B} x(t)$

$y(t)  =\mathbf{C} h(t)+\mathbf{D} x(t)$

要得到离散时间系统：
$$
\begin{aligned}
h_t &= \overline{A}h_{t-1} + \overline{B}x_t \quad (3) \\
y_t &= Ch_t + Dx_t \quad (4)
\end{aligned}
$$
## 推导过程

### 步骤 1：解连续时间微分方程

方程 (1) 是一个一阶线性微分方程，其标准解为：
$
h(t) = e^{A(t - t_0)}h(t_0) + \int_{t_0}^{t} e^{A(t - \tau)}Bx(\tau)d\tau
$

### 步骤 2：应用零阶保持（ZOH）假设

在离散化中，我们假设：
1. 输入 $x(t)$ 在时间区间 $[t_{k-1}, t_k]$ 内保持恒定，即 $x(t) = x_k$
2. 采样时间间隔为 $\Delta$，即 $t_k - t_{k-1} = \Delta$

### 步骤 3：计算离散时间解

令 $t_0 = (k-1)\Delta$，$t = k\Delta$，代入解中：
$
h(k\Delta) = e^{\mathbf{A}\Delta}h((k-1)\Delta) + \int_{(k-1)\Delta}^{k\Delta} e^{\mathbf{A}(k\Delta - \tau)}\mathbf{B}x_k d\tau
$

令 $s = k\Delta - \tau$，则 $d\tau = -ds$，积分限变为：
- 当 $\tau = (k-1)\Delta$ 时，$s = \Delta$
- 当 $\tau = k\Delta$ 时，$s = 0$

代入得：
$
h(k\Delta) = e^{\mathbf{A}\Delta}h((k-1)\Delta) + \left[\int_{\Delta}^{0} e^{\mathbf{A}s}\mathbf{B}x_k (-ds)\right]
$
$
= e^{\mathbf{A}\Delta}h((k-1)\Delta) + \left[\int_{0}^{\Delta} e^{\mathbf{A}s}ds\right]\mathbf{B}x_k
$

### 步骤 4：处理积分项

计算积分 $\int_{0}^{\Delta} e^{\mathbf{A}s}ds$：

如果 $\mathbf{A}$ 可逆：
$
\int_{0}^{\Delta} e^{\mathbf{A}s}ds = \mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})
$

如果 $\mathbf{A}$ 不可逆，可以使用矩阵指数函数的级数展开。

### 步骤 5：得到最终离散化公式

代入积分结果：
$
h(k\Delta) = e^{\mathbf{A}\Delta}h((k-1)\Delta) + \mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}x_k
$

定义：
$
\begin{aligned}
\overline{\mathbf{A}} &= e^{\mathbf{A}\Delta} \\
\overline{\mathbf{B}} &= \mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}
\end{aligned}
$

则离散时间系统为：
$
h_k = \overline{\mathbf{A}}h_{k-1} + \overline{\mathbf{B}}x_k \\
y_k = \mathbf{C}h_k + \mathbf{D}x_k
$

---

## 简化版本（常用形式）

在实际应用中，经常使用以下近似形式
$
\overline{\mathbf{A}} &= e^{\mathbf{A}\Delta} \\
\overline{\mathbf{B}} &= (\mathbf{A}\Delta)^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}\Delta \approx (\mathbf{A}\Delta)^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}\Delta
$

或者当 $\mathbf{A}\Delta$ 很小时，使用泰勒展开近似：
$
\overline{\mathbf{A}} &\approx \mathbf{I} + \mathbf{A}\Delta \\
\overline{\mathbf{B}} &\approx \mathbf{B}\Delta
$

---

## 关键要点

1. **离散化方法**：使用零阶保持（ZOH）假设，即在采样间隔内输入保持恒定
2. **矩阵指数**：$e^{\mathbf{A}\Delta}$ 是离散化的核心，将连续时间动态转换为离散时间动态
3. **参数关系**：
   - $\overline{\mathbf{A}} = e^{\mathbf{A}\Delta}$（状态转移矩阵）
   - $\overline{\mathbf{B}} = \mathbf{A}^{-1}(e^{\mathbf{A}\Delta} - \mathbf{I})\mathbf{B}$（输入矩阵）
   - $\mathbf{C}$ 和 $\mathbf{D}$ 保持不变

这就是从连续时间状态空间模型推导出离散时间递归方程的完整数学过程！


